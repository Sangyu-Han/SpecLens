from __future__ import annotations

import argparse
import html
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autolabel_eval.config import EvalConfig
from autolabel_eval.feature_bank import build_feature_bank, load_feature_bank
from autolabel_eval.metrics import average_precision_binary, roc_auc_binary


SCRIPTS_DIR = ROOT / "scripts"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    # Avoid MKL/libgomp crashes in long-running batch subprocesses.
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    return env


def _format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    if abs(float(value)) >= 1.0:
        return f"{float(value):.6f}"
    return f"{float(value):.6g}"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearman_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    ys = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if yt.size == 0 or ys.size == 0 or yt.size != ys.size:
        return float("nan")
    yt_rank = _average_ranks(yt)
    ys_rank = _average_ranks(ys)
    yt_std = float(np.std(yt_rank))
    ys_std = float(np.std(ys_rank))
    if yt_std <= 0.0 or ys_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(yt_rank, ys_rank)[0, 1])


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    overrides: dict[str, Any] = {
        "workspace_root": Path(args.workspace_root).resolve(),
        "model_name": str(args.vision_model_name),
        "blocks": tuple(int(v) for v in args.blocks),
        "features_per_block": int(args.master_features_per_block),
        "train_examples_per_feature": int(args.train_examples_per_feature),
        "holdout_examples_per_feature": int(args.holdout_examples_per_feature),
        "deciles_root_override": Path(args.deciles_root).resolve(),
        "checkpoints_root_override": Path(args.checkpoints_root).resolve(),
        "checkpoint_relpath_template": str(args.checkpoint_pattern),
        "dataset_root_override": Path(args.dataset_root).resolve(),
        "erf_recovery_threshold": float(args.erf_threshold),
        "shuffle_feature_candidates": True,
        "random_seed": int(args.selection_seed),
    }
    config = replace(EvalConfig(), **overrides)
    config.ensure_dirs()
    return config


def _config_matches_feature_bank(payload: dict[str, Any], config: EvalConfig) -> bool:
    recorded = dict(payload.get("config") or {})
    expected = {
        "model_name": str(config.model_name),
        "blocks": [int(v) for v in config.blocks],
        "features_per_block": int(config.features_per_block),
        "train_examples_per_feature": int(config.train_examples_per_feature),
        "holdout_examples_per_feature": int(config.holdout_examples_per_feature),
        "deciles_root": str(config.deciles_root),
        "checkpoints_root": str(config.checkpoints_root),
        "checkpoint_relpath_template": str(config.checkpoint_relpath_template),
        "dataset_root_override": str(config.dataset_root_override) if config.dataset_root_override is not None else None,
        "shuffle_feature_candidates": True,
        "random_seed": int(config.random_seed),
    }
    for key, value in expected.items():
        if recorded.get(key) != value:
            return False
    return True


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    command_log: list[dict[str, Any]],
) -> dict[str, Any]:
    print(f"[start] {label}", flush=True)
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    elapsed = time.time() - started
    record = {
        "label": label,
        "cmd": list(cmd),
        "cwd": str(cwd),
        "elapsed_sec": float(elapsed),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    command_log.append(record)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed with code {proc.returncode}\n"
            f"CMD: {shlex.join(cmd)}\n"
            f"STDOUT tail:\n{proc.stdout[-4000:]}\n"
            f"STDERR tail:\n{proc.stderr[-4000:]}"
        )
    print(f"[done] {label} ({elapsed:.1f}s)", flush=True)
    return record


def _ensure_feature_bank(config: EvalConfig, *, rebuild: bool) -> dict[str, Any]:
    need_build = bool(rebuild) or not config.feature_bank_json.exists()
    if not need_build:
        existing = load_feature_bank(config)
        if not _config_matches_feature_bank(existing, config):
            need_build = True
        else:
            for block_idx in config.blocks:
                n_features = len(list((existing["blocks"] or {})[str(int(block_idx))]["features"]))
                if n_features < int(config.features_per_block):
                    need_build = True
                    break
            if not need_build:
                return existing
    return build_feature_bank(config)


def _build_master_manifest(
    feature_bank: dict[str, Any],
    *,
    blocks: tuple[int, ...],
    master_features_per_block: int,
    session_name: str,
    feature_bank_json: Path,
    out_path: Path,
    selection_seed: int,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    by_block: dict[str, list[dict[str, Any]]] = {}
    for block_idx in blocks:
        block_features = list((feature_bank["blocks"] or {})[str(int(block_idx))]["features"])
        chosen = block_features[: int(master_features_per_block)]
        if len(chosen) < int(master_features_per_block):
            raise SystemExit(
                f"Requested {master_features_per_block} features for block {block_idx}, "
                f"but feature bank only has {len(chosen)}"
            )
        block_rows = []
        for feature in chosen:
            row = {
                "feature_key": str(feature["feature_key"]),
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
            }
            selected.append(row)
            block_rows.append(row)
        by_block[str(block_idx)] = block_rows
    payload = {
        "session_name": str(session_name),
        "source": "feature_bank_master_selection_random_patch_only",
        "feature_bank_json": str(feature_bank_json),
        "blocks": [int(v) for v in blocks],
        "features_per_block": int(master_features_per_block),
        "selection_seed": int(selection_seed),
        "selection_mode": "random_feature_bank_order",
        "exclude_special_token_features": True,
        "selected_feature_keys": [str(row["feature_key"]) for row in selected],
        "selection": selected,
        "features": selected,
        "features_by_block": by_block,
    }
    _write_json(out_path, payload)
    return payload


def _build_repeat_manifest(
    master_manifest: dict[str, Any],
    *,
    repeat_index: int,
    repeat_size_per_block: int,
    out_path: Path,
) -> dict[str, Any]:
    blocks = [int(v) for v in master_manifest["blocks"]]
    per_block = dict(master_manifest["features_by_block"])
    selected: list[dict[str, Any]] = []
    for block_idx in blocks:
        rows = list(per_block[str(block_idx)])
        start = int(repeat_index) * int(repeat_size_per_block)
        end = start + int(repeat_size_per_block)
        chunk = rows[start:end]
        if len(chunk) != int(repeat_size_per_block):
            raise SystemExit(
                f"repeat {repeat_index + 1} for block {block_idx} expected {repeat_size_per_block} features, "
                f"got {len(chunk)}"
            )
        selected.extend(chunk)
    payload = {
        "session_name": f"{master_manifest['session_name']}_r{repeat_index + 1:02d}",
        "source": "repeat_suite_chunk",
        "repeat_index": int(repeat_index + 1),
        "repeat_size_per_block": int(repeat_size_per_block),
        "blocks": blocks,
        "selected_feature_keys": [str(row["feature_key"]) for row in selected],
        "selection": selected,
        "features": selected,
    }
    _write_json(out_path, payload)
    return payload


def _group_by_block(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(int(row["block_idx"])), []).append(row)
    return out


def _aggregate_axis(repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    axis_data: dict[str, dict[str, list[dict[str, Any]]]] = {"axis1": {}, "axis2": {}}
    per_repeat: list[dict[str, Any]] = []
    for row in repeat_rows:
        summary = _read_json(Path(row["axis_summary_json"]))
        per_repeat.append(
            {
                "repeat_index": int(row["repeat_index"]),
                "session_name": str(summary["session_name"]),
                "axis1": {
                    variant_id: dict(summary["axis1"][variant_id]["overall"])
                    for variant_id in summary["axis1"]
                },
                "axis2": {
                    variant_id: dict(summary["axis2"][variant_id]["overall"])
                    for variant_id in summary["axis2"]
                },
            }
        )
        for axis_name in ("axis1", "axis2"):
            for variant_id, variant_payload in dict(summary[axis_name]).items():
                axis_data[axis_name].setdefault(variant_id, []).extend(list(variant_payload["per_item"]))

    out: dict[str, Any] = {"per_repeat": per_repeat}
    for axis_name, by_variant in axis_data.items():
        axis_payload: dict[str, Any] = {}
        for variant_id, items in by_variant.items():
            if axis_name == "axis1":
                grouped = _group_by_block(items)
                axis_payload[variant_id] = {
                    "n_items": int(len(items)),
                    "overall": {
                        "top1_accuracy": _mean([float(row["correct"]) for row in items]),
                        "mean_confidence": _mean([float(row.get("confidence", float("nan"))) for row in items]),
                        "mean_elapsed_sec": _mean([float(row.get("elapsed_sec", float("nan"))) for row in items]),
                    },
                    "per_block": {
                        block_key: {
                            "n_items": int(len(block_rows)),
                            "top1_accuracy": _mean([float(row["correct"]) for row in block_rows]),
                        }
                        for block_key, block_rows in grouped.items()
                    },
                }
            else:
                grouped = _group_by_block(items)
                axis_payload[variant_id] = {
                    "n_items": int(len(items)),
                    "overall": {
                        "top1_accuracy": _mean([float(row["top1_correct"]) for row in items]),
                        "mrr": _mean([float(row["reciprocal_rank"]) for row in items]),
                        "nDCG@16": _mean([float(row["ndcg"]) for row in items]),
                        "Recall@3": _mean([float(row["recall_at_3"]) for row in items]),
                        "Recall@5": _mean([float(row["recall_at_5"]) for row in items]),
                        "mean_confidence": _mean([float(row.get("confidence", float("nan"))) for row in items]),
                        "mean_elapsed_sec": _mean([float(row.get("elapsed_sec", float("nan"))) for row in items]),
                    },
                    "per_block": {
                        block_key: {
                            "n_items": int(len(block_rows)),
                            "top1_accuracy": _mean([float(row["top1_correct"]) for row in block_rows]),
                            "mrr": _mean([float(row["reciprocal_rank"]) for row in block_rows]),
                            "nDCG@16": _mean([float(row["ndcg"]) for row in block_rows]),
                            "Recall@3": _mean([float(row["recall_at_3"]) for row in block_rows]),
                            "Recall@5": _mean([float(row["recall_at_5"]) for row in block_rows]),
                        }
                        for block_key, block_rows in grouped.items()
                    },
                }
        out[axis_name] = axis_payload
    return out


def _aggregate_supp(repeat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_data: dict[str, dict[str, list[dict[str, Any]]]] = {"supp_valid": {}, "supp_test": {}}
    per_repeat: list[dict[str, Any]] = []
    for row in repeat_rows:
        summary = _read_json(Path(row["supp_summary_json"]))
        per_repeat.append(
            {
                "repeat_index": int(row["repeat_index"]),
                "session_name": str(summary["session_name"]),
                "supp_valid": {
                    variant_id: dict(summary["supp_valid"][variant_id]["overall"])
                    for variant_id in summary["supp_valid"]
                },
                "supp_test": {
                    variant_id: dict(summary["supp_test"][variant_id]["overall"])
                    for variant_id in summary["supp_test"]
                },
            }
        )
        for split_name in ("supp_valid", "supp_test"):
            for variant_id, variant_payload in dict(summary[split_name]).items():
                split_data[split_name].setdefault(variant_id, []).extend(list(variant_payload["per_item"]))

    out: dict[str, Any] = {"per_repeat": per_repeat}
    for split_name, by_variant in split_data.items():
        split_payload: dict[str, Any] = {}
        for variant_id, items in by_variant.items():
            grouped = _group_by_block(items)

            def _compute_metrics(item_rows: list[dict[str, Any]]) -> dict[str, Any]:
                y_true: list[int] = []
                y_true_cont: list[float] = []
                y_score: list[float] = []
                for item in item_rows:
                    for record in list(item.get("records", [])):
                        y_true.append(int(record["binary_label"]))
                        y_true_cont.append(float(record["normalized_activation"]))
                        y_score.append(float(record["predicted_score_norm"]))
                y_true_np = np.asarray(y_true, dtype=np.int64)
                y_true_cont_np = np.asarray(y_true_cont, dtype=np.float64)
                y_score_np = np.asarray(y_score, dtype=np.float64)
                return {
                    "n_items": int(len(item_rows)),
                    "records_per_item": 4,
                    "overall": {
                        "spearman_rho": _spearman_score(y_true_cont_np, y_score_np),
                        "auroc": roc_auc_binary(y_true_np, y_score_np),
                        "average_precision": average_precision_binary(y_true_np, y_score_np),
                        "mae": float(np.mean(np.abs(y_true_cont_np - y_score_np))) if y_score_np.size else float("nan"),
                        "mean_confidence": _mean([float(row.get("overall_confidence", float("nan"))) for row in item_rows]),
                        "mean_elapsed_sec": _mean([float(row.get("elapsed_sec", float("nan"))) for row in item_rows]),
                    },
                }

            variant_summary = _compute_metrics(items)
            variant_summary["per_block"] = {
                block_key: _compute_metrics(block_rows)["overall"]
                for block_key, block_rows in grouped.items()
            }
            split_payload[variant_id] = variant_summary
        out[split_name] = split_payload
    return out


def _build_suite_html(out_path: Path, payload: dict[str, Any]) -> None:
    repeat_rows = list(payload["repeats"])
    repeat_cards = []
    for row in repeat_rows:
        axis = dict(row.get("axis_metrics") or {})
        supp = dict(row.get("supp_metrics") or {})
        repeat_cards.append(
            f"""
            <tr>
              <td>r{int(row['repeat_index']):02d}</td>
              <td>{int(row['feature_count'])}</td>
              <td><a href="{html.escape(str(row['repeat_manifest_rel']))}">manifest</a></td>
              <td><a href="{html.escape(str(row['compare_html_rel']))}">compare</a></td>
              <td><a href="{html.escape(str(row['erf_review_html_rel']))}">ERF review</a></td>
              <td><a href="{html.escape(str(row['sae_review_html_rel']))}">SAE review</a></td>
              <td>{_format_float(float(axis.get('axis1_erf_top1', float('nan'))))} / {_format_float(float(axis.get('axis1_sae_top1', float('nan'))))}</td>
              <td>{_format_float(float(axis.get('axis2_erf_top1', float('nan'))))} / {_format_float(float(axis.get('axis2_sae_top1', float('nan'))))}</td>
              <td>{_format_float(float(supp.get('supp_test_erf_rho', float('nan'))))} / {_format_float(float(supp.get('supp_test_sae_rho', float('nan'))))}</td>
            </tr>
            """
        )

    axis = dict(payload["aggregate"]["axis"])
    supp = dict(payload["aggregate"]["supp"])
    consistency = dict(payload["consistency"])

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cyan-Cross Shortdesc Repeat Suite</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    .hero, .card, .table-wrap {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px; }}
    .meta {{ color:#6b645d; margin-top:6px; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e7dfd4; font-size:14px; }}
    th {{ color:#5f574f; font-weight:600; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Cyan-Cross Shortdesc Repeat Suite</h1>
      <div class="meta">session_prefix={html.escape(str(payload['session_prefix']))} | repeat_count={int(payload['repeat_count'])} | total_features={int(payload['master_feature_count'])}</div>
      <div class="meta">master selection: <a href="{html.escape(str(payload['master_manifest_rel']))}">manifest</a> | consistency: <a href="{html.escape(str(consistency['html_rel']))}">html</a> / <a href="{html.escape(str(consistency['summary_rel']))}">json</a></div>
      <div class="meta">consistency mean |ledger-forward|={_format_float(float(consistency['overall_abs_err_mean']))} | max={_format_float(float(consistency['overall_abs_err_max']))}</div>
    </section>
    <div class="grid">
      <section class="card">
        <h3>Axis 1</h3>
        <div class="meta">ERF top1={_format_float(float(axis['axis1']['erf_cyan_cross']['overall']['top1_accuracy']))}</div>
        <div class="meta">SAE top1={_format_float(float(axis['axis1']['sae_only']['overall']['top1_accuracy']))}</div>
      </section>
      <section class="card">
        <h3>Axis 2</h3>
        <div class="meta">ERF top1 / MRR / R@5 = {_format_float(float(axis['axis2']['erf_cyan_cross']['overall']['top1_accuracy']))} / {_format_float(float(axis['axis2']['erf_cyan_cross']['overall']['mrr']))} / {_format_float(float(axis['axis2']['erf_cyan_cross']['overall']['Recall@5']))}</div>
        <div class="meta">SAE top1 / MRR / R@5 = {_format_float(float(axis['axis2']['sae_only']['overall']['top1_accuracy']))} / {_format_float(float(axis['axis2']['sae_only']['overall']['mrr']))} / {_format_float(float(axis['axis2']['sae_only']['overall']['Recall@5']))}</div>
      </section>
      <section class="card">
        <h3>Supp Test</h3>
        <div class="meta">ERF rho / AUROC / AP = {_format_float(float(supp['supp_test']['erf_cyan_cross']['overall']['spearman_rho']))} / {_format_float(float(supp['supp_test']['erf_cyan_cross']['overall']['auroc']))} / {_format_float(float(supp['supp_test']['erf_cyan_cross']['overall']['average_precision']))}</div>
        <div class="meta">SAE rho / AUROC / AP = {_format_float(float(supp['supp_test']['sae_only']['overall']['spearman_rho']))} / {_format_float(float(supp['supp_test']['sae_only']['overall']['auroc']))} / {_format_float(float(supp['supp_test']['sae_only']['overall']['average_precision']))}</div>
      </section>
    </div>
    <section class="table-wrap">
      <h2>Repeats</h2>
      <table>
        <thead>
          <tr>
            <th>Repeat</th>
            <th>Features</th>
            <th>Manifest</th>
            <th>Compare</th>
            <th>ERF</th>
            <th>SAE</th>
            <th>Axis1 Top1 E/S</th>
            <th>Axis2 Top1 E/S</th>
            <th>Supp Test Rho E/S</th>
          </tr>
        </thead>
        <tbody>
          {''.join(repeat_cards)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    out_path.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 10 x (20/20/20) cyan-cross shortdesc suites and aggregate metrics.")
    parser.add_argument("--session-prefix", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 6, 10])
    parser.add_argument("--master-features-per-block", type=int, default=200)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--repeat-size-per-block", type=int, default=20)
    parser.add_argument("--train-examples-per-feature", type=int, default=5)
    parser.add_argument("--holdout-examples-per-feature", type=int, default=2)
    parser.add_argument("--label-model", default="gpt-5.4")
    parser.add_argument("--label-reasoning-effort", default="xhigh")
    parser.add_argument("--label-prompt-style", default="label_shortdesc_where_v1")
    parser.add_argument("--jobs-label", type=int, default=6)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--judge-reasoning-effort", default="xhigh")
    parser.add_argument("--jobs-axis", type=int, default=4)
    parser.add_argument("--jobs-supp", type=int, default=4)
    parser.add_argument("--axis2-candidate-count", type=int, default=16)
    parser.add_argument("--vision-model-name", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    parser.add_argument(
        "--deciles-root",
        default="/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_index/deciles",
    )
    parser.add_argument(
        "--checkpoints-root",
        default="/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_sae",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="model.blocks.{block_idx}/step_0050000_tokens_204800000.pt",
    )
    parser.add_argument("--dataset-root", default="/data/datasets/imagenet/val")
    parser.add_argument("--erf-threshold", type=float, default=0.90)
    parser.add_argument("--erf-support-min-attribution", type=float, default=0.10)
    parser.add_argument("--selection-seed", type=int, default=20260424)
    parser.add_argument("--rebuild-feature-bank", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    suite_root = workspace_root / "outputs" / "repeat_suites" / str(args.session_prefix)
    suite_root.mkdir(parents=True, exist_ok=True)
    manifests_root = workspace_root / "outputs" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    config = _build_config_from_args(args)
    command_log: list[dict[str, Any]] = []
    master_manifest_path = manifests_root / f"{args.session_prefix}_master_selection.json"
    if bool(args.resume) and master_manifest_path.exists():
        master_manifest = _read_json(master_manifest_path)
    else:
        feature_bank = _ensure_feature_bank(config, rebuild=bool(args.rebuild_feature_bank))
        master_manifest = _build_master_manifest(
            feature_bank,
            blocks=tuple(int(v) for v in args.blocks),
            master_features_per_block=int(args.master_features_per_block),
            session_name=f"{args.session_prefix}_master_selection",
            feature_bank_json=config.feature_bank_json,
            out_path=master_manifest_path,
            selection_seed=int(args.selection_seed),
        )

    consistency_session_name = f"{args.session_prefix}_ledger_consistency"
    consistency_summary_path = workspace_root / "outputs" / "consistency_checks" / consistency_session_name / "summary.json"
    if not bool(args.resume) or not consistency_summary_path.exists():
        consistency_cmd = [
            str(args.python_bin),
            str(SCRIPTS_DIR / "check_ledger_activation_consistency.py"),
            "--workspace-root",
            str(workspace_root),
            "--session-name",
            consistency_session_name,
            "--feature-manifest-json",
            str(master_manifest_path),
            "--max-features-per-block",
            str(int(args.master_features_per_block)),
            "--features-per-block",
            str(int(args.master_features_per_block)),
            "--train-examples-per-feature",
            str(int(args.train_examples_per_feature)),
            "--holdout-examples-per-feature",
            str(int(args.holdout_examples_per_feature)),
            "--vision-model-name",
            str(args.vision_model_name),
            "--deciles-root",
            str(args.deciles_root),
            "--checkpoints-root",
            str(args.checkpoints_root),
            "--checkpoint-pattern",
            str(args.checkpoint_pattern),
            "--dataset-root",
            str(args.dataset_root),
            "--erf-threshold",
            str(float(args.erf_threshold)),
        ]
        _run_command(consistency_cmd, cwd=ROOT, label="ledger consistency check", command_log=command_log)
    consistency_summary = _read_json(consistency_summary_path)

    repeat_rows: list[dict[str, Any]] = []
    for repeat_index in range(int(args.repeat_count)):
        repeat_manifest_path = manifests_root / f"{args.session_prefix}_r{repeat_index + 1:02d}_selection.json"
        repeat_manifest = _build_repeat_manifest(
            master_manifest,
            repeat_index=repeat_index,
            repeat_size_per_block=int(args.repeat_size_per_block),
            out_path=repeat_manifest_path,
        )
        repeat_prefix = f"{args.session_prefix}_r{repeat_index + 1:02d}"
        pipeline_manifest_path = workspace_root / "outputs" / "pipeline_runs" / repeat_prefix / "pipeline_manifest.json"
        if not bool(args.resume) or not pipeline_manifest_path.exists():
            pipeline_cmd = [
                str(args.python_bin),
                str(SCRIPTS_DIR / "run_cyancross_shortdesc_pipeline.py"),
                "--session-prefix",
                repeat_prefix,
                "--workspace-root",
                str(workspace_root),
                "--feature-manifest-json",
                str(repeat_manifest_path),
                "--label-model",
                str(args.label_model),
                "--label-reasoning-effort",
                str(args.label_reasoning_effort),
                "--label-prompt-style",
                str(args.label_prompt_style),
                "--jobs-label",
                str(int(args.jobs_label)),
                "--judge-model",
                str(args.judge_model),
                "--judge-reasoning-effort",
                str(args.judge_reasoning_effort),
                "--jobs-axis",
                str(int(args.jobs_axis)),
                "--jobs-supp",
                str(int(args.jobs_supp)),
                "--axis2-candidate-count",
                str(int(args.axis2_candidate_count)),
                "--vision-model-name",
                str(args.vision_model_name),
                "--deciles-root",
                str(args.deciles_root),
                "--checkpoints-root",
                str(args.checkpoints_root),
                "--checkpoint-pattern",
                str(args.checkpoint_pattern),
                "--dataset-root",
                str(args.dataset_root),
                "--erf-threshold",
                str(float(args.erf_threshold)),
                "--erf-support-min-attribution",
                str(float(args.erf_support_min_attribution)),
            ]
            if bool(args.resume):
                pipeline_cmd.append("--resume")
            _run_command(
                pipeline_cmd,
                cwd=ROOT,
                label=f"repeat r{repeat_index + 1:02d} pipeline",
                command_log=command_log,
            )
        pipeline_manifest = _read_json(pipeline_manifest_path)
        axis_summary_json = str(pipeline_manifest["paths"]["axis_summary_json"])
        supp_summary_json = str(pipeline_manifest["paths"]["supp_summary_json"])
        axis_summary = _read_json(Path(axis_summary_json))
        supp_summary = _read_json(Path(supp_summary_json))
        repeat_rows.append(
            {
                "repeat_index": int(repeat_index + 1),
                "repeat_manifest_json": str(repeat_manifest_path),
                "feature_count": int(len(repeat_manifest["selected_feature_keys"])),
                "compare_html": str(pipeline_manifest["paths"]["compare_html"]),
                "erf_review_html": str(pipeline_manifest["paths"]["erf_review_html"]),
                "sae_review_html": str(pipeline_manifest["paths"]["sae_review_html"]),
                "axis_summary_json": axis_summary_json,
                "supp_summary_json": supp_summary_json,
                "axis_metrics": {
                    "axis1_erf_top1": float(axis_summary["axis1"]["erf_cyan_cross"]["overall"]["top1_accuracy"]),
                    "axis1_sae_top1": float(axis_summary["axis1"]["sae_only"]["overall"]["top1_accuracy"]),
                    "axis2_erf_top1": float(axis_summary["axis2"]["erf_cyan_cross"]["overall"]["top1_accuracy"]),
                    "axis2_sae_top1": float(axis_summary["axis2"]["sae_only"]["overall"]["top1_accuracy"]),
                },
                "supp_metrics": {
                    "supp_test_erf_rho": float(supp_summary["supp_test"]["erf_cyan_cross"]["overall"]["spearman_rho"]),
                    "supp_test_sae_rho": float(supp_summary["supp_test"]["sae_only"]["overall"]["spearman_rho"]),
                },
            }
        )

    axis_aggregate = _aggregate_axis(repeat_rows)
    supp_aggregate = _aggregate_supp(repeat_rows)
    aggregate = {"axis": axis_aggregate, "supp": supp_aggregate}

    consistency_html_path = consistency_summary_path.parent / "index.html"
    suite_rows_with_rel: list[dict[str, Any]] = []
    for row in repeat_rows:
        suite_rows_with_rel.append(
            {
                **row,
                "repeat_manifest_rel": os.path.relpath(str(row["repeat_manifest_json"]), str(suite_root)),
                "compare_html_rel": os.path.relpath(str(row["compare_html"]), str(suite_root)),
                "erf_review_html_rel": os.path.relpath(str(row["erf_review_html"]), str(suite_root)),
                "sae_review_html_rel": os.path.relpath(str(row["sae_review_html"]), str(suite_root)),
            }
        )

    payload = {
        "session_prefix": str(args.session_prefix),
        "workspace_root": str(workspace_root),
        "repeat_count": int(args.repeat_count),
        "repeat_size_per_block": int(args.repeat_size_per_block),
        "master_feature_count": int(len(master_manifest["selected_feature_keys"])),
        "master_manifest_json": str(master_manifest_path),
        "master_manifest_rel": os.path.relpath(str(master_manifest_path), str(suite_root)),
        "consistency": {
            "summary_json": str(consistency_summary_path),
            "summary_rel": os.path.relpath(str(consistency_summary_path), str(suite_root)),
            "html": str(consistency_html_path),
            "html_rel": os.path.relpath(str(consistency_html_path), str(suite_root)),
            "overall_abs_err_mean": float(consistency_summary["overall"]["abs_score_err"]["mean"]),
            "overall_abs_err_max": float(consistency_summary["overall"]["abs_score_err"]["max"]),
        },
        "repeats": suite_rows_with_rel,
        "aggregate": aggregate,
        "commands": command_log,
    }

    _write_json(suite_root / "suite_manifest.json", payload)
    _write_json(suite_root / "aggregate_summary.json", aggregate)

    report_lines = [
        "# Cyan-Cross Shortdesc Repeat Suite",
        "",
        f"- session_prefix: `{args.session_prefix}`",
        f"- repeat_count: `{args.repeat_count}`",
        f"- repeat_size_per_block: `{args.repeat_size_per_block}`",
        f"- master_manifest: `{master_manifest_path}`",
        f"- consistency summary: `{consistency_summary_path}`",
        f"- aggregate summary: `{suite_root / 'aggregate_summary.json'}`",
        f"- suite html: `{suite_root / 'index.html'}`",
        "",
        "## Aggregate Axis 1",
        f"- ERF top1: `{_format_float(float(axis_aggregate['axis1']['erf_cyan_cross']['overall']['top1_accuracy']))}`",
        f"- SAE top1: `{_format_float(float(axis_aggregate['axis1']['sae_only']['overall']['top1_accuracy']))}`",
        "",
        "## Aggregate Axis 2",
        f"- ERF top1 / MRR / R@5: `{_format_float(float(axis_aggregate['axis2']['erf_cyan_cross']['overall']['top1_accuracy']))}` / `{_format_float(float(axis_aggregate['axis2']['erf_cyan_cross']['overall']['mrr']))}` / `{_format_float(float(axis_aggregate['axis2']['erf_cyan_cross']['overall']['Recall@5']))}`",
        f"- SAE top1 / MRR / R@5: `{_format_float(float(axis_aggregate['axis2']['sae_only']['overall']['top1_accuracy']))}` / `{_format_float(float(axis_aggregate['axis2']['sae_only']['overall']['mrr']))}` / `{_format_float(float(axis_aggregate['axis2']['sae_only']['overall']['Recall@5']))}`",
        "",
        "## Aggregate Supplementary Test",
        f"- ERF rho / AUROC / AP / MAE: `{_format_float(float(supp_aggregate['supp_test']['erf_cyan_cross']['overall']['spearman_rho']))}` / `{_format_float(float(supp_aggregate['supp_test']['erf_cyan_cross']['overall']['auroc']))}` / `{_format_float(float(supp_aggregate['supp_test']['erf_cyan_cross']['overall']['average_precision']))}` / `{_format_float(float(supp_aggregate['supp_test']['erf_cyan_cross']['overall']['mae']))}`",
        f"- SAE rho / AUROC / AP / MAE: `{_format_float(float(supp_aggregate['supp_test']['sae_only']['overall']['spearman_rho']))}` / `{_format_float(float(supp_aggregate['supp_test']['sae_only']['overall']['auroc']))}` / `{_format_float(float(supp_aggregate['supp_test']['sae_only']['overall']['average_precision']))}` / `{_format_float(float(supp_aggregate['supp_test']['sae_only']['overall']['mae']))}`",
    ]
    _write_text(suite_root / "report.md", "\n".join(report_lines) + "\n")

    _build_suite_html(suite_root / "index.html", payload)

    commands_text = "\n".join(
        f"# {row['label']}\n{shlex.join(list(row['cmd']))}\n"
        for row in command_log
    )
    _write_text(suite_root / "commands.sh", commands_text)

    print(suite_root / "suite_manifest.json")


if __name__ == "__main__":
    main()
