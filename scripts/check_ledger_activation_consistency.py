from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

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
from autolabel_eval.legacy import LegacyRuntime


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    if abs(float(value)) >= 1.0:
        return f"{float(value):.6f}"
    return f"{float(value):.6g}"


def _build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    overrides: dict[str, Any] = {
        "workspace_root": Path(args.workspace_root).resolve(),
        "model_name": str(args.vision_model_name),
        "blocks": tuple(int(v) for v in args.blocks),
        "features_per_block": int(args.features_per_block),
        "train_examples_per_feature": int(args.train_examples_per_feature),
        "holdout_examples_per_feature": int(args.holdout_examples_per_feature),
        "deciles_root_override": Path(args.deciles_root).resolve(),
        "checkpoints_root_override": Path(args.checkpoints_root).resolve(),
        "checkpoint_relpath_template": str(args.checkpoint_pattern),
        "dataset_root_override": Path(args.dataset_root).resolve(),
        "erf_recovery_threshold": float(args.erf_threshold),
    }
    config = replace(EvalConfig(), **overrides)
    config.ensure_dirs()
    return config


def _feature_keys_from_manifest(path: Path) -> list[str]:
    payload = _read_json(path)
    out: list[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("selected_feature_keys"), list):
            out.extend(str(v).strip() for v in payload["selected_feature_keys"] if str(v).strip())
        elif isinstance(payload.get("selection"), list):
            for row in payload["selection"]:
                key = str((row or {}).get("feature_key") or "").strip()
                if key:
                    out.append(key)
        elif isinstance(payload.get("features"), list):
            for row in payload["features"]:
                key = str((row or {}).get("feature_key") or "").strip()
                if key:
                    out.append(key)
    if not out:
        raise ValueError(f"No feature keys found in {path}")
    return out


def _select_feature_keys(feature_bank: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if str(args.feature_manifest_json).strip():
        wanted = _feature_keys_from_manifest(Path(args.feature_manifest_json).resolve())
        deduped: list[str] = []
        seen: set[str] = set()
        for key in wanted:
            if key not in seen:
                deduped.append(key)
                seen.add(key)
        return deduped

    per_block = int(args.max_features_per_block)
    selected: list[str] = []
    for block_idx in tuple(int(v) for v in args.blocks):
        block_features = list((feature_bank["blocks"] or {})[str(block_idx)]["features"])
        chosen = block_features[:per_block]
        if len(chosen) < per_block:
            raise SystemExit(
                f"Requested {per_block} features for block {block_idx}, "
                f"but only found {len(chosen)} in feature_bank.json"
            )
        selected.extend(str(row["feature_key"]) for row in chosen)
    return selected


def _iter_rows(feature: dict[str, Any], split: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if split in ("all", "train"):
        out.extend(("train", row) for row in list(feature.get("train", [])))
    if split in ("all", "holdout"):
        out.extend(("holdout", row) for row in list(feature.get("holdout", [])))
    return out


def _summary_stats(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "mean": float(statistics.mean(finite)),
        "median": float(statistics.median(finite)),
        "max": float(max(finite)),
    }


def _build_html(out_path: Path, payload: dict[str, Any]) -> None:
    worst_rows = list(payload.get("worst_rows", []))
    row_html = []
    for row in worst_rows:
        row_html.append(
            "<tr>"
            f"<td>{html.escape(str(row['feature_key']))}</td>"
            f"<td>{int(row['sample_id'])}</td>"
            f"<td>{int(row['token_idx'])}</td>"
            f"<td>{html.escape(str(row['split']))}</td>"
            f"<td>{_format_float(float(row['ledger_score']))}</td>"
            f"<td>{_format_float(float(row['current_act_at_target']))}</td>"
            f"<td>{_format_float(float(row['current_abs_score_err']))}</td>"
            f"<td>{_format_float(float(row['saved_vs_current_abs_diff']))}</td>"
            "</tr>"
        )

    block_cards = []
    for block_key, summary in payload.get("per_block", {}).items():
        block_cards.append(
            f"""
            <section class="card">
              <h3>Block {html.escape(str(block_key))}</h3>
              <div class="meta">rows={int(summary['n_rows'])} | failures={int(summary['n_validation_failed'])} | over_tol={int(summary['n_err_gt_tolerance'])}</div>
              <div class="meta">mean_abs_err={_format_float(float(summary['abs_score_err']['mean']))} | max_abs_err={_format_float(float(summary['abs_score_err']['max']))}</div>
              <div class="meta">mean_saved_diff={_format_float(float(summary['saved_vs_current_abs_diff']['mean']))} | max_saved_diff={_format_float(float(summary['saved_vs_current_abs_diff']['max']))}</div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ledger Activation Consistency</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    .hero, .card, .table-wrap {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; }}
    .meta {{ color:#6b645d; margin-top:6px; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e7dfd4; font-size:14px; }}
    th {{ color:#5f574f; font-weight:600; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Ledger Activation Consistency</h1>
      <div class="meta">workspace={html.escape(str(payload['workspace_root']))}</div>
      <div class="meta">checked_features={int(payload['n_features'])} | checked_rows={int(payload['n_rows'])} | failures={int(payload['n_validation_failed'])}</div>
      <div class="meta">abs_err mean={_format_float(float(payload['overall']['abs_score_err']['mean']))} | max={_format_float(float(payload['overall']['abs_score_err']['max']))}</div>
      <div class="meta">saved_vs_current mean={_format_float(float(payload['overall']['saved_vs_current_abs_diff']['mean']))} | max={_format_float(float(payload['overall']['saved_vs_current_abs_diff']['max']))}</div>
      <div class="meta">tolerance={_format_float(float(payload['tolerance']))} | all_within_tolerance={bool(payload['all_within_tolerance'])}</div>
    </section>
    <div class="grid">
      {''.join(block_cards)}
    </div>
    <section class="table-wrap">
      <h2>Worst Rows</h2>
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>Sample</th>
            <th>Token</th>
            <th>Split</th>
            <th>Ledger</th>
            <th>Forward</th>
            <th>|ledger-forward|</th>
            <th>|saved-forward|</th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    out_path.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-forward selected ledger rows and compare against saved ledger activations.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--feature-manifest-json", default="")
    parser.add_argument("--max-features-per-block", type=int, default=200)
    parser.add_argument("--features-per-block", type=int, default=200)
    parser.add_argument("--train-examples-per-feature", type=int, default=5)
    parser.add_argument("--holdout-examples-per-feature", type=int, default=2)
    parser.add_argument("--split", choices=("all", "train", "holdout"), default="all")
    parser.add_argument("--build-feature-bank-if-missing", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 6, 10])
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
    args = parser.parse_args()

    config = _build_config_from_args(args)
    if not config.feature_bank_json.exists():
        if not bool(args.build_feature_bank_if_missing):
            raise SystemExit(
                f"feature_bank.json missing at {config.feature_bank_json}; "
                "pass --build-feature-bank-if-missing to create it first"
            )
        build_feature_bank(config)
    feature_bank = load_feature_bank(config)
    selected_feature_keys = _select_feature_keys(feature_bank, args)
    feature_lookup = {
        str(feature["feature_key"]): feature
        for block_key in feature_bank["blocks"]
        for feature in list(feature_bank["blocks"][block_key]["features"])
    }

    out_dir = config.workspace_root / "outputs" / "consistency_checks" / str(args.session_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = LegacyRuntime(config)

    rows: list[dict[str, Any]] = []
    try:
        for feature_key in selected_feature_keys:
            feature = feature_lookup.get(str(feature_key))
            if feature is None:
                raise SystemExit(f"feature {feature_key} is absent from feature_bank.json")
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            for split_name, row in _iter_rows(feature, str(args.split)):
                saved_validation = dict(row.get("validation") or {})
                current = runtime.validate_feature_token(
                    str(row["image_path"]),
                    block_idx,
                    feature_id,
                    int(row["target_patch_idx"]),
                    float(row["ledger_score"]),
                )
                current_act = float("nan")
                current_err = float("nan")
                saved_vs_current = float("nan")
                target_ratio = float("nan")
                max_act = float("nan")
                argmax_tok = -1
                validation_failed = current is None
                if current is not None:
                    current_act = float(current["act_at_target"])
                    current_err = float(current["abs_score_err"])
                    target_ratio = float(current["target_to_max_ratio"])
                    max_act = float(current["max_act"])
                    argmax_tok = int(current["argmax_tok"])
                    if "act_at_target" in saved_validation:
                        saved_vs_current = abs(current_act - float(saved_validation["act_at_target"]))
                    elif math.isfinite(current_act):
                        saved_vs_current = 0.0
                rows.append(
                    {
                        "feature_key": str(feature_key),
                        "block_idx": block_idx,
                        "feature_id": feature_id,
                        "split": split_name,
                        "sample_id": int(row["sample_id"]),
                        "token_idx": int(row["target_patch_idx"]),
                        "token_uid": str(row["token_uid"]),
                        "image_path": str(row["image_path"]),
                        "ledger_score": float(row["ledger_score"]),
                        "saved_validation": saved_validation,
                        "current_validation": current,
                        "current_act_at_target": current_act,
                        "current_abs_score_err": current_err,
                        "saved_vs_current_abs_diff": saved_vs_current,
                        "target_to_max_ratio": target_ratio,
                        "max_act": max_act,
                        "argmax_tok": argmax_tok,
                        "validation_failed": bool(validation_failed),
                    }
                )
    finally:
        runtime.close()

    tolerance = float(args.tolerance)
    per_block: dict[str, Any] = {}
    for block_idx in tuple(int(v) for v in args.blocks):
        block_rows = [row for row in rows if int(row["block_idx"]) == int(block_idx)]
        per_block[str(block_idx)] = {
            "n_rows": int(len(block_rows)),
            "n_validation_failed": int(sum(1 for row in block_rows if bool(row["validation_failed"]))),
            "n_err_gt_tolerance": int(
                sum(
                    1
                    for row in block_rows
                    if math.isfinite(float(row["current_abs_score_err"]))
                    and float(row["current_abs_score_err"]) > tolerance
                )
            ),
            "abs_score_err": _summary_stats([float(row["current_abs_score_err"]) for row in block_rows]),
            "saved_vs_current_abs_diff": _summary_stats(
                [float(row["saved_vs_current_abs_diff"]) for row in block_rows]
            ),
        }

    worst_rows = sorted(
        rows,
        key=lambda row: (
            -1.0 if not math.isfinite(float(row["current_abs_score_err"])) else float(row["current_abs_score_err"]),
            -1.0
            if not math.isfinite(float(row["saved_vs_current_abs_diff"]))
            else float(row["saved_vs_current_abs_diff"]),
        ),
        reverse=True,
    )[:50]
    overall = {
        "abs_score_err": _summary_stats([float(row["current_abs_score_err"]) for row in rows]),
        "saved_vs_current_abs_diff": _summary_stats([float(row["saved_vs_current_abs_diff"]) for row in rows]),
    }
    mismatch_count = int(
        sum(
            1
            for row in rows
            if math.isfinite(float(row["current_abs_score_err"]))
            and float(row["current_abs_score_err"]) > tolerance
        )
    )
    payload = {
        "session_name": str(args.session_name),
        "workspace_root": str(config.workspace_root),
        "feature_manifest_json": str(Path(args.feature_manifest_json).resolve()) if str(args.feature_manifest_json).strip() else "",
        "feature_bank_json": str(config.feature_bank_json),
        "n_features": int(len(selected_feature_keys)),
        "n_rows": int(len(rows)),
        "n_validation_failed": int(sum(1 for row in rows if bool(row["validation_failed"]))),
        "tolerance": tolerance,
        "all_within_tolerance": bool(mismatch_count == 0),
        "overall": overall,
        "per_block": per_block,
        "worst_rows": worst_rows,
        "rows": rows,
    }

    _write_json(out_dir / "summary.json", payload)
    report_lines = [
        "# Ledger Activation Consistency",
        "",
        f"- session_name: `{args.session_name}`",
        f"- n_features: `{len(selected_feature_keys)}`",
        f"- n_rows: `{len(rows)}`",
        f"- n_validation_failed: `{payload['n_validation_failed']}`",
        f"- tolerance: `{_format_float(tolerance)}`",
        f"- all_within_tolerance: `{payload['all_within_tolerance']}`",
        f"- overall mean |ledger-forward|: `{_format_float(float(overall['abs_score_err']['mean']))}`",
        f"- overall max |ledger-forward|: `{_format_float(float(overall['abs_score_err']['max']))}`",
        f"- overall mean |saved-forward|: `{_format_float(float(overall['saved_vs_current_abs_diff']['mean']))}`",
        f"- overall max |saved-forward|: `{_format_float(float(overall['saved_vs_current_abs_diff']['max']))}`",
    ]
    _write_text(out_dir / "report.md", "\n".join(report_lines) + "\n")
    _build_html(out_dir / "index.html", payload)
    print(out_dir / "summary.json")

    if bool(args.fail_on_mismatch) and (not bool(payload["all_within_tolerance"]) or int(payload["n_validation_failed"]) > 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
