from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import random
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
from PIL import Image

from autolabel_eval.config import EvalConfig
from autolabel_eval.isolated_codex import run_isolated_codex_exec
from autolabel_eval.legacy import LegacyRuntime
from autolabel_eval.rendering import (
    CLIP_ZERO_RGB,
    save_feature_actmap_masked_image,
    save_support_mask_image,
)


PROMPT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical_label": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["canonical_label", "description"],
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _feature_key(block_idx: int, feature_id: int) -> str:
    return f"block_{int(block_idx)}/feature_{int(feature_id)}"


def _build_config_from_args(args: argparse.Namespace) -> EvalConfig:
    offline_meta_override = None
    offline_meta_arg = str(getattr(args, "offline_meta_root", "") or "").strip()
    if offline_meta_arg:
        offline_meta_override = Path(offline_meta_arg)
    else:
        deciles_root = Path(args.deciles_root)
        sibling_offline_meta = deciles_root.parent / "offline_meta"
        if sibling_offline_meta.exists():
            offline_meta_override = sibling_offline_meta
    config = replace(
        EvalConfig(),
        workspace_root=Path(args.workspace_root),
        model_name=str(args.vision_model_name),
        deciles_root_override=Path(args.deciles_root),
        offline_meta_root_override=offline_meta_override,
        checkpoints_root_override=Path(args.checkpoints_root),
        checkpoint_relpath_template=str(args.checkpoint_pattern),
        dataset_root_override=Path(args.dataset_root),
        erf_recovery_threshold=float(args.erf_threshold),
        erf_support_min_normalized_attribution=float(args.erf_support_min_attribution),
    )
    config.ensure_dirs()
    return config


def _role_text(token_x: int, prefix_count: int) -> str:
    token_x = int(token_x)
    prefix_count = int(prefix_count)
    if token_x == 0 or prefix_count <= 1:
        return "CLS token"
    return f"register token {token_x}"


def _parquet_glob(config: EvalConfig, block_idx: int) -> str:
    return str((config.deciles_root / f"layer_part=model.blocks.{int(block_idx)}" / "**/*.parquet").as_posix())


def _infer_prefix_count(conn: duckdb.DuckDBPyConnection, config: EvalConfig, block_idx: int) -> int:
    max_x = conn.execute(
        f"""
        select max(x)
        from read_parquet('{_parquet_glob(config, block_idx)}')
        where y = -1
        """
    ).fetchone()[0]
    if max_x is None:
        raise RuntimeError(f"block {block_idx}: no token rows found in ledger")
    prefix_count = int(max_x) + 1 - int(config.n_patches)
    if prefix_count <= 0:
        raise RuntimeError(
            f"block {block_idx}: invalid inferred prefix count {prefix_count} "
            f"(max_x={max_x}, n_patches={config.n_patches})"
        )
    return int(prefix_count)


def _candidate_units_for_block(
    conn: duckdb.DuckDBPyConnection,
    config: EvalConfig,
    *,
    block_idx: int,
    top_k: int,
) -> list[dict[str, Any]]:
    prefix_count = _infer_prefix_count(conn, config, block_idx)
    rows = conn.execute(
        f"""
        with ranked as (
          select unit, score, sample_id, frame_idx, y, x,
                 row_number() over (partition by unit order by score desc, sample_id asc, x asc) as rn
          from read_parquet('{_parquet_glob(config, block_idx)}')
        )
        select unit, score, sample_id, frame_idx, y, x, rn
        from ranked
        where rn <= {int(top_k)}
        order by unit asc, rn asc
        """
    ).fetchall()

    by_unit: dict[int, list[dict[str, Any]]] = {}
    for unit, score, sample_id, frame_idx, y, x, rn in rows:
        by_unit.setdefault(int(unit), []).append(
            {
                "unit": int(unit),
                "score": float(score),
                "sample_id": int(sample_id),
                "frame_idx": int(frame_idx),
                "y": int(y),
                "x": int(x),
                "rank": int(rn),
            }
        )

    candidates: list[dict[str, Any]] = []
    for unit, unit_rows in by_unit.items():
        if len(unit_rows) < int(top_k):
            continue
        unit_rows = sorted(unit_rows, key=lambda row: int(row["rank"]))
        top_rows = unit_rows[: int(top_k)]
        if any(int(row["y"]) != -1 for row in top_rows):
            continue
        token_values = [int(row["x"]) for row in top_rows]
        if any(token_x < 0 or token_x >= prefix_count for token_x in token_values):
            continue
        if len(set(token_values)) != 1:
            continue
        candidates.append(
            {
                "feature_key": _feature_key(block_idx, unit),
                "block_idx": int(block_idx),
                "feature_id": int(unit),
                "prefix_count": int(prefix_count),
                "objective_token_x": int(token_values[0]),
                "objective_token_role": _role_text(int(token_values[0]), prefix_count),
                "top_rows": top_rows,
                "top5_mean_score": float(sum(float(row["score"]) for row in top_rows) / max(len(top_rows), 1)),
            }
        )
    return candidates


def _load_manifest_candidates_for_block(
    manifest_path: Path,
    *,
    block_idx: int,
    prefix_count: int,
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    block_payload = dict((payload.get("blocks") or {}).get(str(int(block_idx))) or {})
    accepted = list(block_payload.get("accepted_features") or [])
    candidates: list[dict[str, Any]] = []
    for row in accepted:
        feature_id = int(row["feature_id"])
        seed_rows = list(row.get("seed_top5_rows") or [])
        candidates.append(
            {
                "feature_key": _feature_key(block_idx, feature_id),
                "block_idx": int(block_idx),
                "feature_id": int(feature_id),
                "prefix_count": int(prefix_count),
                "objective_token_x": 0,
                "objective_token_role": _role_text(0, prefix_count),
                "top_rows": seed_rows,
                "top5_mean_score": float(
                    sum(float(r.get("score", 0.0)) for r in seed_rows) / max(len(seed_rows), 1)
                ),
                "candidate_source": "accepted_cls_feature_manifest",
                "seed_sample_ids": list(row.get("seed_sample_ids") or []),
                "cls_activation_stats": dict(row.get("cls_activation_stats") or {}),
            }
        )
    candidates.sort(
        key=lambda item: (
            -float(item.get("cls_activation_stats", {}).get("mean", 0.0)),
            -float(item.get("top5_mean_score", 0.0)),
            int(item["feature_id"]),
        )
    )
    return candidates


def _collect_examples_for_feature(
    conn: duckdb.DuckDBPyConnection,
    runtime: LegacyRuntime,
    *,
    config: EvalConfig,
    block_idx: int,
    feature_id: int,
    token_x: int,
    top_k: int,
    max_scan_rows: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        select sample_id, frame_idx, score, y, x
        from read_parquet('{_parquet_glob(config, block_idx)}')
        where unit = {int(feature_id)} and y = -1 and x = {int(token_x)}
        order by score desc, sample_id asc
        """
    ).fetchall()
    if not rows:
        return []

    sample_ids = [int(sample_id) for sample_id, _frame_idx, _score, _y, _x in rows]
    sid_to_path = runtime.lookup_paths(sample_ids)
    accepted: list[dict[str, Any]] = []
    seen_sample_ids: set[int] = set()

    for sample_id, frame_idx, score, y, x in rows[: max(1, int(max_scan_rows))]:
        sample_id = int(sample_id)
        if sample_id in seen_sample_ids:
            continue
        image_path = sid_to_path.get(sample_id, "")
        if not image_path:
            continue
        accepted.append(
            {
                "sample_id": int(sample_id),
                "frame_idx": int(frame_idx),
                "ledger_score": float(score),
                "objective_token_x": int(token_x),
                "image_path": str(image_path),
                "validation": None,
            }
        )
        seen_sample_ids.add(sample_id)
        if len(accepted) >= int(max_scan_rows):
            break
    return accepted


def _render_feature_examples(
    runtime: LegacyRuntime,
    *,
    session_dir: Path,
    feature: dict[str, Any],
    examples: list[dict[str, Any]],
    top_k: int,
    max_score_error: float,
) -> dict[str, Any] | None:
    feature_key = str(feature["feature_key"])
    block_idx = int(feature["block_idx"])
    feature_id = int(feature["feature_id"])
    token_x = int(feature["objective_token_x"])
    feature_dir = session_dir / "assets" / _slug(feature_key)
    validated_rows: list[dict[str, Any]] = []

    for row in examples:
        image_path = str(row["image_path"])
        validation = row.get("validation")
        if validation is None:
            validation = runtime.validate_feature_special_token(
                image_path,
                int(block_idx),
                int(feature_id),
                int(token_x),
                float(row["ledger_score"]),
            )
        if validation is None:
            continue
        if abs(float(validation.get("abs_score_err", float("inf")))) > float(max_score_error):
            continue
        row_payload = dict(row)
        row_payload["validation"] = dict(validation)
        validated_rows.append(row_payload)
        if len(validated_rows) >= int(top_k):
            break

    if len(validated_rows) < int(top_k):
        return None

    rendered_examples: list[dict[str, Any]] = []

    for rank, row in enumerate(validated_rows):
        image_path = str(row["image_path"])
        validation = dict(row["validation"])
        sae_fire_path = feature_dir / f"example_{rank:02d}_sae_fire.png"
        erf_path = feature_dir / f"example_{rank:02d}_feature_erf_special_token.png"
        erf_json_path = feature_dir / f"example_{rank:02d}_feature_erf_special_token.json"

        actmap = runtime.feature_activation_map_visible_patches(image_path, block_idx, feature_id)
        save_feature_actmap_masked_image(
            image_path,
            actmap,
            sae_fire_path,
            token_idx=None,
            background_color=CLIP_ZERO_RGB,
        )

        erf_payload = runtime.cautious_feature_erf_special_token(
            image_path,
            block_idx,
            token_x,
            feature_id,
        )
        save_support_mask_image(
            image_path,
            erf_payload["support_indices"],
            erf_path,
            token_idx=0,
            mode="masked_black",
            background_color=CLIP_ZERO_RGB,
            mask_resample=Image.NEAREST,
            include_token_box=False,
        )
        _write_json(erf_json_path, erf_payload)

        rendered_examples.append(
            {
                "rank": int(rank),
                "sample_id": int(row["sample_id"]),
                "objective_token_x": int(token_x),
                "objective_token_role": str(feature["objective_token_role"]),
                "image_path": str(image_path),
                "ledger_score": float(row["ledger_score"]),
                "validation": dict(validation),
                "validated_feature_activation": float(validation["act_at_target"]),
                "baseline_feature_activation": float(erf_payload.get("baseline_feature_activation", 0.0)),
                "effective_feature_activation_delta": float(erf_payload.get("effective_feature_activation_delta", 0.0)),
                "support_recovery": float(erf_payload.get("support_recovery", 0.0)),
                "support_size": int(erf_payload.get("support_size", 0)),
                "recovery_max": float(erf_payload.get("recovery_max", 0.0)),
                "recovery_threshold_reached": bool(erf_payload.get("recovery_threshold_reached", False)),
                "erf_absent_reason": str(erf_payload.get("erf_absent_reason", "") or ""),
                "sae_fire": str(sae_fire_path.relative_to(session_dir)),
                "feature_erf_special_token": str(erf_path.relative_to(session_dir)),
                "feature_erf_json": str(erf_json_path.relative_to(session_dir)),
            }
        )

    return {
        "feature_key": feature_key,
        "block_idx": int(block_idx),
        "feature_id": int(feature_id),
        "prefix_count": int(feature["prefix_count"]),
        "objective_token_x": int(token_x),
        "objective_token_role": str(feature["objective_token_role"]),
        "label_examples": rendered_examples,
    }


def _build_erf_prompt(token_role: str, *, empty_examples: int, total_examples: int) -> str:
    empty_hint = ""
    if int(empty_examples) > 0:
        empty_hint = (
            f"- {int(empty_examples)} of the {int(total_examples)} ERF examples are blank because no patch-dependent support "
            "was found for those samples. Use the visible evidence in the non-blank examples to label the feature. "
            "Only call it a bias feature if all examples are blank.\n"
        )
    return f"""You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

Each provided image shows only the pixels inside the ERF support. Gray areas are hidden and provide no image information.
The objective token in these examples is the {token_role}. This token has no visible image patch marker.

Task:
Produce the final human-readable labeler output for this feature.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these ERF images.
- Keep that canonical label fixed.
- Prefer the strongest reusable shared concept that is directly supported across examples.
{empty_hint}- If every ERF image is entirely gray with no visible evidence, treat the feature as a `bias feature` rather than forcing a visual concept label.
- Then write `description` in 1 to 3 short sentences.
- The description should help a human reviewer understand the shared visual pattern across the examples.
- Mention any recurring appearance, context, relation, relative placement, or shared objective-token position only if it is visibly part of the common pattern across examples.
- If supported by the examples, mention any clue that this behaves like a {token_role} feature, such as diffuse or global evidence, non-local context, or inconsistent local support.
- Do not let the description rewrite, narrow, or replace the canonical label.
- Focus only on the shared visible evidence exposed by the ERF support.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.

Output only a single JSON object with keys:
- canonical_label
- description
"""


def _build_sae_prompt(token_role: str) -> str:
    return f"""You are auditing one visual SAE feature using only isolated SAE firing evidence.

Each provided image shows only pixels in strongly firing visible patches for the feature. Gray hidden regions provide no image information.
The objective token in these examples is the {token_role}. This token has no visible image patch marker.

Task:
Produce the final human-readable labeler output for this feature.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these images.
- Keep that canonical label fixed.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- Then write `description` in 1 to 3 short sentences.
- The description should help a human reviewer understand the shared visual pattern across the examples.
- Mention any recurring appearance, context, relation, relative placement, or shared objective-token position only if it is visibly part of the common pattern across examples.
- If supported by the examples, mention any clue that this behaves like a {token_role} feature, such as diffuse or global evidence, non-local context, or inconsistent local support.
- Do not let the description rewrite, narrow, or replace the canonical label.
- Focus only on the shared visible evidence in the visible firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.

Output only a single JSON object with keys:
- canonical_label
- description
"""


def _label_one_feature(
    *,
    session_dir: Path,
    feature: dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    feature_key = str(feature["feature_key"])
    artifact_dir = session_dir / "artifacts" / _slug(feature_key)
    token_role = str(feature["objective_token_role"])
    label_examples = list(feature["label_examples"])
    empty_erf_examples = sum(1 for example in label_examples if int(example.get("support_size", 0)) <= 0)
    total_erf_examples = len(label_examples)
    erf_images = [session_dir / str(example["feature_erf_special_token"]) for example in feature["label_examples"]]
    sae_images = [session_dir / str(example["sae_fire"]) for example in feature["label_examples"]]
    if total_erf_examples > 0 and empty_erf_examples == total_erf_examples:
        erf_result = {
            "output": {
                "canonical_label": "bias feature",
                "description": (
                    f"All ERF examples are blank, indicating no patch-dependent support was found for this {token_role.lower()} feature."
                ),
            },
            "returncode": 0,
            "elapsed_sec": 0.0,
            "forbidden_trace_hits": [],
            "label_mode": "deterministic_all_empty_bias_feature",
        }
    else:
        erf_result = run_isolated_codex_exec(
            artifact_dir=artifact_dir,
            artifact_stem="erf_label",
            prompt_text=_build_erf_prompt(
                token_role,
                empty_examples=int(empty_erf_examples),
                total_examples=int(total_erf_examples),
            ),
            schema=PROMPT_SCHEMA,
            images=erf_images,
            model=model,
            reasoning_effort=reasoning_effort,
        )
    sae_result = run_isolated_codex_exec(
        artifact_dir=artifact_dir,
        artifact_stem="sae_label",
        prompt_text=_build_sae_prompt(token_role),
        schema=PROMPT_SCHEMA,
        images=sae_images,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    return {
        "feature_key": feature_key,
        "block_idx": int(feature["block_idx"]),
        "feature_id": int(feature["feature_id"]),
        "objective_token_x": int(feature["objective_token_x"]),
        "objective_token_role": token_role,
        "label_examples": label_examples,
        "erf_label": dict(erf_result.get("output") or {}),
        "sae_label": dict(sae_result.get("output") or {}),
        "erf_exec": {
            "returncode": int(erf_result["returncode"]),
            "elapsed_sec": float(erf_result["elapsed_sec"]),
            "forbidden_trace_hits": list(erf_result.get("forbidden_trace_hits") or []),
            "label_mode": str(erf_result.get("label_mode") or "isolated_codex_exec"),
        },
        "sae_exec": {
            "returncode": int(sae_result["returncode"]),
            "elapsed_sec": float(sae_result["elapsed_sec"]),
            "forbidden_trace_hits": list(sae_result.get("forbidden_trace_hits") or []),
        },
    }


def _build_source_review_html(out_path: Path, payload: dict[str, Any]) -> None:
    feature_blocks: list[str] = []
    for feature in payload["features"]:
        panels_html: list[str] = []
        for example in feature["label_examples"]:
            absent_reason = str(example.get("erf_absent_reason") or "").strip()
            absent_meta = f" | absent_reason={html.escape(absent_reason)}" if absent_reason else ""
            panels_html.append(
                f"""
      <div class="panel">
        <div class="meta">Example {int(example["rank"]) + 1} | sample_id={int(example["sample_id"])} | objective={html.escape(str(example["objective_token_role"]))} | ledger_act={float(example["ledger_score"]):.3f} | validated_act={float(example["validated_feature_activation"]):.3f} | baseline_act={float(example["baseline_feature_activation"]):.3f} | delta={float(example["effective_feature_activation_delta"]):.3f} | max_recovery={float(example["recovery_max"]):.3f} | support_recovery={float(example["support_recovery"]):.3f} | threshold_reached={str(bool(example["recovery_threshold_reached"])).lower()}{absent_meta}</div>
        <div class="pair">
          <figure>
            <img src="{html.escape(str(example["feature_erf_special_token"]))}" alt="ERF panel">
            <figcaption>ERF evidence</figcaption>
          </figure>
          <figure>
            <img src="{html.escape(str(example["sae_fire"]))}" alt="SAE panel">
            <figcaption>SAE firing patches</figcaption>
          </figure>
        </div>
      </div>
"""
            )
        feature_blocks.append(
            f"""
    <section class="feature">
      <h2>{html.escape(str(feature["feature_key"]))}</h2>
      <div class="meta">objective token: {html.escape(str(feature["objective_token_role"]))} (x={int(feature["objective_token_x"])})</div>
      <div class="stack">
        {''.join(panels_html)}
      </div>
    </section>
"""
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLS Token Source Review</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta {{ color:#6b645d; }}
    .pair {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:10px; }}
    figure {{ margin:0; background:#f3ede5; border-radius:12px; overflow:hidden; border:1px solid #ddd2c4; }}
    figure img {{ display:block; width:100%; height:auto; }}
    figcaption {{ padding:8px 10px; font-size:13px; color:#544c43; }}
    .stack {{ display:flex; flex-direction:column; gap:14px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>CLS / Special Token Label Source</h1>
      <p>Feature panels for special-token labeling. ERF panels show patch evidence only; SAE panels show visible firing patches only.</p>
    </section>
    {''.join(feature_blocks)}
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text)


def _build_compare_html(out_path: Path, payload: dict[str, Any]) -> None:
    feature_blocks: list[str] = []
    for feature in payload["features"]:
        label_rows: list[str] = []
        for key, title in (("erf_label", "ERF"), ("sae_label", "SAE")):
            pred = dict(feature.get(key) or {})
            label_rows.append(
                f"""
      <div class="labelcard">
        <h3>{title}</h3>
        <div class="label">{html.escape(str(pred.get("canonical_label", ""))) or "&nbsp;"}</div>
        <div class="desc">{html.escape(str(pred.get("description", ""))) or "&nbsp;"}</div>
      </div>
"""
            )
        panels_html: list[str] = []
        for example in feature["label_examples"]:
            absent_reason = str(example.get("erf_absent_reason") or "").strip()
            absent_meta = f" | absent_reason={html.escape(absent_reason)}" if absent_reason else ""
            panels_html.append(
                f"""
      <div class="panel">
        <div class="meta">Example {int(example["rank"]) + 1} | sample_id={int(example["sample_id"])} | objective={html.escape(str(example["objective_token_role"]))} | ledger_act={float(example["ledger_score"]):.3f} | validated_act={float(example["validated_feature_activation"]):.3f} | baseline_act={float(example["baseline_feature_activation"]):.3f} | delta={float(example["effective_feature_activation_delta"]):.3f} | max_recovery={float(example["recovery_max"]):.3f} | support_recovery={float(example["support_recovery"]):.3f} | threshold_reached={str(bool(example["recovery_threshold_reached"])).lower()}{absent_meta}</div>
        <div class="pair">
          <figure>
            <img src="{html.escape(str(example["feature_erf_special_token"]))}" alt="ERF panel">
            <figcaption>ERF evidence</figcaption>
          </figure>
          <figure>
            <img src="{html.escape(str(example["sae_fire"]))}" alt="SAE panel">
            <figcaption>SAE firing patches</figcaption>
          </figure>
        </div>
      </div>
"""
            )
        feature_blocks.append(
            f"""
    <section class="feature">
      <h2>{html.escape(str(feature["feature_key"]))}</h2>
      <div class="meta">objective token: {html.escape(str(feature["objective_token_role"]))} (x={int(feature["objective_token_x"])})</div>
      <div class="labels">{''.join(label_rows)}</div>
      <div class="stack">
        {''.join(panels_html)}
      </div>
    </section>
"""
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CLS Token Label Compare</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1520px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta {{ color:#6b645d; }}
    .labels {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin:12px 0 16px; }}
    .labelcard {{ background:#f6efe6; border:1px solid #e0d5c8; border-radius:12px; padding:12px; }}
    .labelcard h3 {{ margin:0 0 6px; font-size:14px; text-transform:uppercase; letter-spacing:0.04em; color:#7b5641; }}
    .label {{ font-weight:700; margin-bottom:6px; }}
    .desc {{ color:#544c43; line-height:1.45; white-space:pre-wrap; }}
    .pair {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; margin-top:10px; }}
    figure {{ margin:0; background:#f3ede5; border-radius:12px; overflow:hidden; border:1px solid #ddd2c4; }}
    figure img {{ display:block; width:100%; height:auto; }}
    figcaption {{ padding:8px 10px; font-size:13px; color:#544c43; }}
    .stack {{ display:flex; flex-direction:column; gap:14px; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>CLS / Special Token Label Compare</h1>
      <p>Each feature compares ERF evidence versus SAE firing patches for a special-token objective with no visible patch marker.</p>
    </section>
    {''.join(feature_blocks)}
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLS/special-token ERF vs SAE labeling pilot.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--session-name", default="clip50k_cls_token_label_pilot_20260423")
    parser.add_argument("--vision-model-name", default="vit_base_patch16_clip_224")
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 6, 10])
    parser.add_argument("--features-per-block", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--deciles-root", required=True)
    parser.add_argument("--offline-meta-root", default="")
    parser.add_argument("--checkpoints-root", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--label-model", default="gpt-5.4")
    parser.add_argument("--label-reasoning-effort", default="xhigh")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--erf-threshold", type=float, default=0.80)
    parser.add_argument("--erf-support-min-attribution", type=float, default=0.10)
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--max-scan-rows", type=int, default=128)
    parser.add_argument("--max-score-error", type=float, default=1e-3)
    parser.add_argument("--cls-feature-manifest", default="")
    args = parser.parse_args()

    config = _build_config_from_args(args)
    session_dir = config.workspace_root / "outputs" / "review_sessions" / str(args.session_name)
    session_dir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    runtime = LegacyRuntime(config)
    try:
        selection: list[dict[str, Any]] = []
        candidate_summary: dict[str, Any] = {"blocks": {}}
        for block_idx in [int(v) for v in args.blocks]:
            prefix_count = _infer_prefix_count(conn, config, block_idx)
            manifest_arg = str(args.cls_feature_manifest or "").strip()
            if manifest_arg:
                candidates = _load_manifest_candidates_for_block(
                    Path(manifest_arg),
                    block_idx=block_idx,
                    prefix_count=int(prefix_count),
                )
            else:
                candidates = _candidate_units_for_block(conn, config, block_idx=block_idx, top_k=int(args.top_k))
            rng = random.Random(int(args.random_seed) + int(block_idx))
            rng.shuffle(candidates)
            selected_block: list[dict[str, Any]] = []
            for candidate in candidates:
                examples = _collect_examples_for_feature(
                    conn,
                    runtime,
                    config=config,
                    block_idx=int(block_idx),
                    feature_id=int(candidate["feature_id"]),
                    token_x=int(candidate["objective_token_x"]),
                    top_k=int(args.top_k),
                    max_scan_rows=int(args.max_scan_rows),
                )
                if len(examples) < int(args.top_k):
                    continue
                rendered = _render_feature_examples(
                    runtime,
                    session_dir=session_dir,
                    feature=candidate,
                    examples=examples,
                    top_k=int(args.top_k),
                    max_score_error=float(args.max_score_error),
                )
                if rendered is None:
                    continue
                selection.append(rendered)
                selected_block.append(
                    {
                        "feature_key": str(candidate["feature_key"]),
                        "feature_id": int(candidate["feature_id"]),
                        "objective_token_x": int(candidate["objective_token_x"]),
                        "objective_token_role": str(candidate["objective_token_role"]),
                    }
                )
                print(
                    f"[block {block_idx}] selected {candidate['feature_key']} "
                    f"with {len(rendered['label_examples'])}/{int(args.top_k)} CLS examples",
                    flush=True,
                )
                if len(selected_block) >= int(args.features_per_block):
                    break
            candidate_summary["blocks"][str(block_idx)] = {
                "candidate_count": int(len(candidates)),
                "selected_count": int(len(selected_block)),
                "requested_count": int(args.features_per_block),
                "shortfall_allowed": bool(args.allow_shortfall),
                "candidate_source": "accepted_cls_feature_manifest" if manifest_arg else "strict_topk_cls_rows",
                "selected": selected_block,
            }
            if len(selected_block) < int(args.features_per_block):
                if not bool(args.allow_shortfall):
                    raise RuntimeError(
                        f"block {block_idx}: only selected {len(selected_block)} special-token features; "
                        f"requested {args.features_per_block}"
                    )
                print(
                    f"[block {block_idx}] shortfall: selected {len(selected_block)} / "
                    f"{int(args.features_per_block)} features",
                    flush=True,
                )

        manifest = {
            "session_name": str(args.session_name),
            "vision_model_name": str(args.vision_model_name),
            "blocks": [int(v) for v in args.blocks],
            "features_per_block": int(args.features_per_block),
            "top_k": int(args.top_k),
            "random_seed": int(args.random_seed),
            "candidate_summary": candidate_summary,
            "features": selection,
        }
        _write_json(session_dir / "selection_manifest.json", manifest)
        _build_source_review_html(session_dir / "source_review.html", manifest)

        started = time.time()
        label_results: list[dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = [
                pool.submit(
                    _label_one_feature,
                    session_dir=session_dir,
                    feature=feature,
                    model=str(args.label_model),
                    reasoning_effort=str(args.label_reasoning_effort),
                )
                for feature in selection
            ]
            for future in cf.as_completed(futures):
                label_results.append(future.result())
        label_results.sort(key=lambda row: (int(row["block_idx"]), int(row["feature_id"])))

        raw_payload = {
            "session_name": str(args.session_name),
            "vision_model_name": str(args.vision_model_name),
            "generation_mode": "isolated_codex_exec_per_feature_special_token",
            "features": label_results,
            "label_model": str(args.label_model),
            "label_reasoning_effort": str(args.label_reasoning_effort),
            "elapsed_sec": float(time.time() - started),
        }
        _write_json(session_dir / "raw_predictions.json", raw_payload)
        _build_compare_html(session_dir / "label_compare.html", raw_payload)
        _write_json(
            session_dir / "session_manifest.json",
            {
                "selection_manifest_json": str(session_dir / "selection_manifest.json"),
                "source_review_html": str(session_dir / "source_review.html"),
                "raw_predictions_json": str(session_dir / "raw_predictions.json"),
                "label_compare_html": str(session_dir / "label_compare.html"),
            },
        )
        print(session_dir)
    finally:
        runtime.close()
        conn.close()


if __name__ == "__main__":
    main()
