from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import sys
import time
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

import duckdb
import numpy as np
from PIL import Image

from autolabel_eval.config import EvalConfig
from autolabel_eval.isolated_codex import run_isolated_codex_exec
from autolabel_eval.legacy import LegacyRuntime
from autolabel_eval.metrics import ndcg_at_k, recall_at_k
from autolabel_eval.utils import write_json


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate_words(text: str, max_words: int = 18, max_chars: int = 140) -> str:
    words = text.replace("\n", " ").split()
    out = " ".join(words[:max_words]).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _derive_description(output: dict[str, Any]) -> str:
    description = _norm_text(output.get("description"))
    if description:
        return description
    support_summary = _norm_text(output.get("support_summary"))
    if support_summary:
        return _truncate_words(support_summary, max_words=20, max_chars=240)
    detailed = _norm_text(output.get("detailed_description"))
    if detailed:
        return _truncate_words(detailed, max_words=20, max_chars=240)
    rationale = _norm_text(output.get("rationale"))
    if rationale:
        return _truncate_words(rationale, max_words=20, max_chars=240)
    return ""


def _build_config_from_args(args: Any) -> EvalConfig:
    config = EvalConfig()
    overrides: dict[str, Any] = {}
    if getattr(args, "workspace_root", None):
        overrides["workspace_root"] = Path(args.workspace_root)
    if getattr(args, "vision_model_name", None):
        overrides["model_name"] = str(args.vision_model_name)
    if getattr(args, "deciles_root", None):
        overrides["deciles_root_override"] = Path(args.deciles_root)
    if getattr(args, "offline_meta_root", None):
        offline_meta = str(args.offline_meta_root).strip()
        if offline_meta:
            overrides["offline_meta_root_override"] = Path(offline_meta)
    if getattr(args, "checkpoints_root", None):
        overrides["checkpoints_root_override"] = Path(args.checkpoints_root)
    if getattr(args, "checkpoint_pattern", None):
        overrides["checkpoint_relpath_template"] = str(args.checkpoint_pattern)
    if getattr(args, "dataset_root", None):
        overrides["dataset_root_override"] = Path(args.dataset_root)
    if overrides:
        config = replace(config, **overrides)
        config.ensure_dirs()
    return config


def _parquet_glob(config: EvalConfig, block_idx: int) -> str:
    return str((config.deciles_root / f"layer_part=model.blocks.{int(block_idx)}" / "**/*.parquet").as_posix())


def _save_original_image(image_path: str, out_path: Path, *, size: int = 224) -> None:
    image = Image.open(image_path).convert("RGB").resize((int(size), int(size)), Image.BICUBIC)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _load_variant_labels_from_cls_raw(raw_path: Path) -> dict[str, dict[str, dict[str, str]]]:
    payload = _read_json(raw_path)
    variants = {"erf": {}, "sae": {}}
    for row in payload["features"]:
        feature_key = str(row["feature_key"])
        erf_label = dict(row.get("erf_label") or {})
        sae_label = dict(row.get("sae_label") or {})
        variants["erf"][feature_key] = {
            "canonical_label": _norm_text(erf_label.get("canonical_label")),
            "description": _derive_description(erf_label),
        }
        variants["sae"][feature_key] = {
            "canonical_label": _norm_text(sae_label.get("canonical_label")),
            "description": _derive_description(sae_label),
        }
    return variants


def _find_unseen_cls_holdout(
    conn: duckdb.DuckDBPyConnection,
    runtime: LegacyRuntime,
    config: EvalConfig,
    *,
    block_idx: int,
    feature_id: int,
    used_sample_ids: set[int],
    max_scan_rows: int,
    max_score_error: float,
) -> dict[str, Any] | None:
    rows = conn.execute(
        f"""
        select sample_id, frame_idx, score
        from read_parquet('{_parquet_glob(config, block_idx)}')
        where unit = {int(feature_id)} and y = -1 and x = 0
        order by score desc, sample_id asc
        """
    ).fetchall()
    if not rows:
        return None
    sample_ids = [int(sample_id) for sample_id, _frame_idx, _score in rows[: max(1, int(max_scan_rows))]]
    path_map = runtime.lookup_paths(sample_ids)
    for sample_id, frame_idx, score in rows[: max(1, int(max_scan_rows))]:
        sample_id = int(sample_id)
        if sample_id in used_sample_ids:
            continue
        image_path = path_map.get(sample_id)
        if not image_path:
            continue
        validation = runtime.validate_feature_special_token(
            str(image_path),
            int(block_idx),
            int(feature_id),
            0,
            float(score),
        )
        if validation is None:
            continue
        if abs(float(validation.get("abs_score_err", float("inf")))) > float(max_score_error):
            continue
        return {
            "sample_id": int(sample_id),
            "frame_idx": int(frame_idx),
            "ledger_score": float(score),
            "image_path": str(image_path),
            "validation": dict(validation),
            "heldout_mode": "unseen_cls_row",
        }
    return None


def _axis2_prompt(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "You are evaluating how discriminative CLS-token feature labels are for one held-out image.",
        "",
        "You will receive one original image.",
        "The relevant token is the CLS token, which summarizes global image evidence and has no visible patch marker.",
        "",
        "Candidate labels:",
    ]
    for row in candidates:
        description = _norm_text(row.get("description"))
        if description:
            lines.append(f"- {row['candidate_code']}: {row['canonical_label']} | {description}")
        else:
            lines.append(f"- {row['candidate_code']}: {row['canonical_label']}")
    lines.extend(
        [
            "",
            "Task:",
            "- Pick the single best matching candidate label for the CLS-token feature expressed by this image.",
            "- Also return a full ranking of all candidate codes from best to worst.",
            "- Prefer the label that best explains the global visual pattern or global image-level cue, not a tiny local detail.",
            "- Ignore rendering artifacts and panel styling.",
            "",
            "Return only JSON.",
        ]
    )
    return "\n".join(lines)


def _axis2_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "best_candidate": {"type": "string", "maxLength": 8},
            "ranked_candidates": {
                "type": "array",
                "items": {"type": "string", "maxLength": 8},
                "maxItems": 32,
            },
            "confidence": {"type": "number"},
            "brief_reason": {"type": "string", "maxLength": 240},
        },
        "required": ["best_candidate", "ranked_candidates", "confidence", "brief_reason"],
    }


def _normalize_ranking(best_candidate: str, ranked_candidates: list[str], valid_codes: list[str]) -> list[str]:
    valid = [str(code).lower() for code in valid_codes]
    seen: set[str] = set()
    ordered: list[str] = []
    for code in [str(best_candidate).lower(), *[str(v).lower() for v in ranked_candidates]]:
        if code in valid and code not in seen:
            ordered.append(code)
            seen.add(code)
    for code in valid:
        if code not in seen:
            ordered.append(code)
            seen.add(code)
    return ordered


def _reciprocal_rank(gold_code: str, ranked_codes: list[str]) -> float:
    gold = str(gold_code).lower()
    for idx, code in enumerate(ranked_codes, start=1):
        if str(code).lower() == gold:
            return 1.0 / float(idx)
    return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def _build_review_html(out_path: Path, *, items: list[dict[str, Any]], results: dict[str, Any]) -> None:
    item_by_key = {(str(item["feature_key"]), int(item["sample_id"])): item for item in items}
    erf_rows = {(str(row["feature_key"]), int(row["sample_id"])): row for row in results["erf"]["per_item"]}
    sae_rows = {(str(row["feature_key"]), int(row["sample_id"])): row for row in results["sae"]["per_item"]}
    blocks: list[str] = []
    for item in items:
        key = (str(item["feature_key"]), int(item["sample_id"]))
        erf = erf_rows[key]
        sae = sae_rows[key]
        blocks.append(
            f"""
    <section class="feature">
      <h2>{item['feature_key']}</h2>
      <div class="meta">block={int(item['block_idx'])} | sample_id={int(item['sample_id'])} | heldout_mode={item['heldout_mode']} | candidates={len(item['candidate_codes'])}</div>
      <div class="image"><img src="{item['input_image']}" alt="heldout image"></div>
      <div class="cards">
        <div class="card">
          <h3>ERF labels</h3>
          <div><b>gold code</b>: {item['gold_code']}</div>
          <div><b>best</b>: {erf['best_candidate']} ({'correct' if int(erf['top1_correct']) else 'wrong'})</div>
          <div><b>reason</b>: {erf['output'].get('brief_reason','')}</div>
        </div>
        <div class="card">
          <h3>SAE labels</h3>
          <div><b>gold code</b>: {item['gold_code']}</div>
          <div><b>best</b>: {sae['best_candidate']} ({'correct' if int(sae['top1_correct']) else 'wrong'})</div>
          <div><b>reason</b>: {sae['output'].get('brief_reason','')}</div>
        </div>
      </div>
    </section>
"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLS Axis 2 Review</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta {{ color:#6b645d; margin-bottom:10px; }}
    .image img {{ display:block; width:224px; height:224px; border-radius:12px; border:1px solid #ddd2c4; }}
    .cards {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:12px; }}
    .card {{ background:#f6efe6; border:1px solid #e0d5c8; border-radius:12px; padding:12px; }}
    h1, h2, h3 {{ margin:0 0 8px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>CLS Axis 2 Review</h1>
      <p>Held-out whole-image evidence for CLS-token feature label discrimination.</p>
    </section>
    {''.join(blocks)}
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLS-token Axis 2 pilot using whole-image held-out evidence.")
    parser.add_argument("--session-name", default="clip50k_cls_axis2_20260424")
    parser.add_argument("--label-session-name", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--vision-model-name", default="vit_base_patch16_clip_224")
    parser.add_argument("--deciles-root", required=True)
    parser.add_argument("--offline-meta-root", default="")
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-scan-rows", type=int, default=256)
    parser.add_argument("--max-score-error", type=float, default=1e-3)
    args = parser.parse_args()

    config = _build_config_from_args(args)
    label_session_dir = config.workspace_root / "outputs" / "review_sessions" / str(args.label_session_name)
    selection_manifest_path = label_session_dir / "selection_manifest.json"
    raw_predictions_path = label_session_dir / "raw_predictions.json"
    if not selection_manifest_path.exists():
        raise SystemExit(f"Missing selection manifest: {selection_manifest_path}")
    if not raw_predictions_path.exists():
        raise SystemExit(f"Missing raw predictions: {raw_predictions_path}")

    session_dir = config.workspace_root / "outputs" / "axis_pilot_sessions" / str(args.session_name)
    session_dir.mkdir(parents=True, exist_ok=True)

    selection_manifest = _read_json(selection_manifest_path)
    variant_labels = _load_variant_labels_from_cls_raw(raw_predictions_path)
    selected_features = list(selection_manifest["features"])
    selected_by_block: dict[int, list[dict[str, Any]]] = {}
    for feature in selected_features:
        selected_by_block.setdefault(int(feature["block_idx"]), []).append(feature)
    for block_idx in list(selected_by_block):
        selected_by_block[int(block_idx)] = sorted(selected_by_block[int(block_idx)], key=lambda row: int(row["feature_id"]))

    conn = duckdb.connect()
    runtime = LegacyRuntime(config)
    try:
        axis2_items: list[dict[str, Any]] = []
        for feature in selected_features:
            feature_key = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            used_sample_ids = {int(row["sample_id"]) for row in feature["label_examples"]}
            heldout = _find_unseen_cls_holdout(
                conn,
                runtime,
                config,
                block_idx=block_idx,
                feature_id=feature_id,
                used_sample_ids=used_sample_ids,
                max_scan_rows=int(args.max_scan_rows),
                max_score_error=float(args.max_score_error),
            )
            if heldout is None:
                fallback = dict(feature["label_examples"][0])
                heldout = {
                    "sample_id": int(fallback["sample_id"]),
                    "frame_idx": 0,
                    "ledger_score": float(fallback["ledger_score"]),
                    "image_path": str(fallback["image_path"]),
                    "validation": dict(fallback.get("validation") or {}),
                    "heldout_mode": "reused_label_example",
                }
            image_rel = f"heldout_assets/{_slug(feature_key)}__sample_{int(heldout['sample_id'])}.png"
            image_abs = session_dir / image_rel
            _save_original_image(str(heldout["image_path"]), image_abs)

            candidates = [
                {
                    "feature_key": str(row["feature_key"]),
                    "is_gold": str(row["feature_key"]) == feature_key,
                }
                for row in selected_by_block[int(block_idx)]
            ]
            shuffle_rng = random.Random(4000 + block_idx * 100000 + int(heldout["sample_id"]) * 101 + feature_id)
            shuffle_rng.shuffle(candidates)
            for idx, row in enumerate(candidates, start=1):
                row["candidate_code"] = f"c{idx:02d}"
            gold_code = next(str(row["candidate_code"]) for row in candidates if bool(row["is_gold"]))
            axis2_items.append(
                {
                    "feature_key": feature_key,
                    "block_idx": int(block_idx),
                    "feature_id": int(feature_id),
                    "sample_id": int(heldout["sample_id"]),
                    "heldout_mode": str(heldout["heldout_mode"]),
                    "input_image": str(image_rel),
                    "candidate_codes": [str(row["candidate_code"]) for row in candidates],
                    "gold_code": str(gold_code),
                    "candidates": candidates,
                }
            )
    finally:
        runtime.close()
        conn.close()

    write_json(
        session_dir / "pilot_manifest.json",
        {
            "session_name": str(args.session_name),
            "label_session_name": str(args.label_session_name),
            "n_items": len(axis2_items),
            "axis2_items": axis2_items,
        },
    )

    axis2_schema_path = session_dir / "axis2_output_schema.json"
    write_json(axis2_schema_path, _axis2_schema())

    results_root = session_dir / "variant_results"
    results_root.mkdir(parents=True, exist_ok=True)

    def run_variant(variant_id: str) -> dict[str, Any]:
        variant_dir = results_root / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        tasks: list[dict[str, Any]] = []
        for item in axis2_items:
            candidates = []
            for row in item["candidates"]:
                label = variant_labels[variant_id][str(row["feature_key"])]
                candidates.append(
                    {
                        "candidate_code": str(row["candidate_code"]),
                        "feature_key": str(row["feature_key"]),
                        "canonical_label": label["canonical_label"],
                        "description": _truncate_words(label["description"], max_words=28, max_chars=220),
                    }
                )
            tasks.append(
                {
                    "item": item,
                    "candidates": candidates,
                    "prompt_text": _axis2_prompt(candidates),
                    "out_json": variant_dir / f"{_slug(item['feature_key'])}__sample_{item['sample_id']}.json",
                    "image_path": session_dir / str(item["input_image"]),
                }
            )

        def worker(task: dict[str, Any]) -> dict[str, Any]:
            result = run_isolated_codex_exec(
                artifact_dir=task["out_json"].parent,
                artifact_stem=task["out_json"].stem,
                prompt_text=task["prompt_text"],
                schema=_read_json(axis2_schema_path),
                images=[Path(task["image_path"])],
                model=str(args.model),
                reasoning_effort=str(args.reasoning_effort),
                temp_prefix="cls_axis2_",
            )
            item = task["item"]
            output = dict(result["output"])
            ranked = _normalize_ranking(
                best_candidate=str(output.get("best_candidate", "")),
                ranked_candidates=list(output.get("ranked_candidates") or []),
                valid_codes=list(item["candidate_codes"]),
            )
            gold_code = str(item["gold_code"]).lower()
            score_map = {code: float(len(ranked) - idx) for idx, code in enumerate(ranked)}
            y_true = np.asarray([1 if str(code).lower() == gold_code else 0 for code in item["candidate_codes"]], dtype=np.int64)
            y_score = np.asarray([score_map.get(str(code).lower(), 0.0) for code in item["candidate_codes"]], dtype=np.float32)
            return {
                "feature_key": str(item["feature_key"]),
                "block_idx": int(item["block_idx"]),
                "feature_id": int(item["feature_id"]),
                "sample_id": int(item["sample_id"]),
                "heldout_mode": str(item["heldout_mode"]),
                "gold_code": gold_code,
                "candidate_count": int(len(item["candidate_codes"])),
                "best_candidate": str(output.get("best_candidate", "")).lower(),
                "ranked_candidates": ranked,
                "top1_correct": int(ranked[0] == gold_code),
                "reciprocal_rank": _reciprocal_rank(gold_code, ranked),
                "ndcg": ndcg_at_k(y_true, y_score, k=len(item["candidate_codes"])),
                "recall_at_3": recall_at_k(y_true, y_score, 3),
                "recall_at_5": recall_at_k(y_true, y_score, 5),
                "confidence": float(output.get("confidence", 0.0) or 0.0),
                "elapsed_sec": float(result["elapsed_sec"]),
                "returncode": int(result["returncode"]),
                "forbidden_trace_hits": list(result.get("forbidden_trace_hits") or []),
                "output": output,
            }

        rows: list[dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for idx, future in enumerate(cf.as_completed(futures), start=1):
                rows.append(future.result())
                if idx % 12 == 0 or idx == len(tasks):
                    print(f"[cls-axis2 {variant_id} {idx:03d}/{len(tasks):03d}]", flush=True)

        row_by_key = {(str(row["feature_key"]), int(row["sample_id"])): row for row in rows}
        ordered_rows = [row_by_key[(str(item["feature_key"]), int(item["sample_id"]))] for item in axis2_items]
        block_metrics: dict[str, Any] = {}
        for block_idx, features in selected_by_block.items():
            block_rows = [row for row in ordered_rows if int(row["block_idx"]) == int(block_idx)]
            candidate_count = len(features)
            block_metrics[str(block_idx)] = {
                "n_items": len(block_rows),
                "candidate_count": int(candidate_count),
                "chance_accuracy": (1.0 / float(candidate_count)) if candidate_count > 0 else float("nan"),
                "top1_accuracy": _mean([float(row["top1_correct"]) for row in block_rows]),
                "mrr": _mean([float(row["reciprocal_rank"]) for row in block_rows]),
                f"nDCG@{int(candidate_count)}": _mean([float(row["ndcg"]) for row in block_rows]),
                "Recall@3": _mean([float(row["recall_at_3"]) for row in block_rows]),
                "Recall@5": _mean([float(row["recall_at_5"]) for row in block_rows]),
            }
        summary = {
            "variant_id": str(variant_id),
            "axis": "cls_axis2",
            "n_items": len(ordered_rows),
            "overall": {
                "top1_accuracy": _mean([float(row["top1_correct"]) for row in ordered_rows]),
                "mrr": _mean([float(row["reciprocal_rank"]) for row in ordered_rows]),
                "mean_ndcg": _mean([float(row["ndcg"]) for row in ordered_rows]),
                "Recall@3": _mean([float(row["recall_at_3"]) for row in ordered_rows]),
                "Recall@5": _mean([float(row["recall_at_5"]) for row in ordered_rows]),
                "mean_confidence": _mean([float(row["confidence"]) for row in ordered_rows]),
                "mean_elapsed_sec": _mean([float(row["elapsed_sec"]) for row in ordered_rows]),
                "mean_candidate_count": _mean([float(row["candidate_count"]) for row in ordered_rows]),
                "mean_chance_accuracy": _mean([1.0 / float(row["candidate_count"]) for row in ordered_rows]),
            },
            "per_block": block_metrics,
            "per_item": ordered_rows,
        }
        write_json(variant_dir / "results.json", summary)
        return summary

    variant_order = ["erf", "sae"]
    results = {variant_id: run_variant(variant_id) for variant_id in variant_order}

    report_lines = [
        "# CLS Axis 2 Pilot",
        "",
        f"- Session: `{args.session_name}`",
        f"- Label session: `{args.label_session_name}`",
        f"- Judge model: `{args.model}`",
        f"- Reasoning effort: `{args.reasoning_effort}`",
        f"- Items: `{len(axis2_items)}` held-out CLS examples",
        "- Evidence shown to the judge: one held-out original image only",
        "- Objective token: CLS token with no visible patch marker",
        "- Candidate set: all selected CLS features from the same block",
        "",
        "## Overall",
        "",
        "| metric | erf | sae |",
        "|---|---:|---:|",
    ]
    for metric in ("top1_accuracy", "mrr", "mean_ndcg", "Recall@3", "Recall@5", "mean_confidence", "mean_elapsed_sec", "mean_candidate_count", "mean_chance_accuracy"):
        report_lines.append(
            f"| {metric} | {results['erf']['overall'][metric]:.3f} | {results['sae']['overall'][metric]:.3f} |"
        )
    report_lines.extend(["", "## Per Block", ""])
    for block_idx in sorted(selected_by_block):
        candidate_count = len(selected_by_block[int(block_idx)])
        report_lines.extend(
            [
                f"### Block {int(block_idx)}",
                "",
                f"- Candidate count: `{int(candidate_count)}`",
                f"- Chance accuracy: `{1.0 / float(candidate_count):.3f}`" if candidate_count > 0 else "- Chance accuracy: `nan`",
                f"- ERF top1 / MRR: `{results['erf']['per_block'][str(block_idx)]['top1_accuracy']:.3f}` / `{results['erf']['per_block'][str(block_idx)]['mrr']:.3f}`",
                f"- SAE top1 / MRR: `{results['sae']['per_block'][str(block_idx)]['top1_accuracy']:.3f}` / `{results['sae']['per_block'][str(block_idx)]['mrr']:.3f}`",
                "",
            ]
        )
    (session_dir / "report.md").write_text("\n".join(report_lines))
    _build_review_html(session_dir / "review.html", items=axis2_items, results=results)
    write_json(
        session_dir / "session_manifest.json",
        {
            "pilot_manifest_json": str(session_dir / "pilot_manifest.json"),
            "report_md": str(session_dir / "report.md"),
            "review_html": str(session_dir / "review.html"),
            "variant_results": {variant: str(results_root / variant / "results.json") for variant in variant_order},
        },
    )
    print(session_dir)


if __name__ == "__main__":
    main()
