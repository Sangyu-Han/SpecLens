from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import os
import shutil
import statistics
import subprocess
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

try:
    import torch

    try:
        import torchvision.ops  # noqa: F401
    except Exception:
        for _name in list(sys.modules):
            if _name == "torchvision" or _name.startswith("torchvision."):
                sys.modules.pop(_name, None)
        try:
            _tv_lib = torch.library.Library("torchvision", "DEF")
            _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass
except Exception:
    torch = None  # type: ignore[assignment]

import numpy as np
from PIL import ImageFont

from autolabel_eval.config import EvalConfig
from autolabel_eval.feature_bank import load_feature_bank
from autolabel_eval.isolated_codex import run_isolated_codex_exec
from autolabel_eval.legacy import LegacyRuntime
from autolabel_eval.metrics import average_precision_binary, roc_auc_binary
from autolabel_eval.rendering import save_original_with_token_box
from autolabel_eval.utils import token_uid, write_json


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _build_config_from_args(args: Any) -> EvalConfig:
    config = EvalConfig()
    overrides: dict[str, Any] = {}
    if getattr(args, "workspace_root", None):
        overrides["workspace_root"] = Path(args.workspace_root)
    if getattr(args, "vision_model_name", None):
        overrides["model_name"] = str(args.vision_model_name)
    if getattr(args, "train_per_feature", None):
        overrides["train_examples_per_feature"] = int(args.train_per_feature)
    if getattr(args, "holdout_per_feature", None):
        overrides["holdout_examples_per_feature"] = int(args.holdout_per_feature)
    if getattr(args, "deciles_root", None):
        overrides["deciles_root_override"] = Path(args.deciles_root)
    if getattr(args, "offline_meta_root", None):
        overrides["offline_meta_root_override"] = Path(args.offline_meta_root)
    if getattr(args, "checkpoints_root", None):
        overrides["checkpoints_root_override"] = Path(args.checkpoints_root)
    if getattr(args, "checkpoint_pattern", None):
        overrides["checkpoint_relpath_template"] = str(args.checkpoint_pattern)
    if getattr(args, "dataset_root", None):
        overrides["dataset_root_override"] = Path(args.dataset_root)
    if getattr(args, "erf_threshold", None) is not None:
        overrides["erf_recovery_threshold"] = float(args.erf_threshold)
    if overrides:
        config = replace(config, **overrides)
        config.ensure_dirs()
    return config


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
    for key in ("notes", "adjacent_context", "primary_locus"):
        text = _norm_text(output.get(key))
        if text:
            return text
    return ""


def _load_variant_labels(raw_path: Path) -> dict[str, dict[str, str]]:
    payload = _read_json(raw_path)
    out: dict[str, dict[str, str]] = {}
    for row in payload["features"]:
        output = dict(row.get("output") or {})
        out[str(row["feature_key"])] = {
            "canonical_label": _norm_text(output.get("canonical_label")),
            "description": _derive_description(output),
        }
    return out


def _build_variant_manifest(config: EvalConfig, session_name: str, raw_name: str) -> tuple[Path, Path]:
    session_dir = config.workspace_root / "outputs" / "review_sessions" / session_name
    return session_dir / "selection_manifest.json", session_dir / raw_name


def _feature_target_activation_scale(feature: dict[str, Any]) -> float:
    acts: list[float] = []
    for row in list(feature.get("train", [])) + list(feature.get("holdout", [])):
        validation = row.get("validation") or {}
        if "act_at_target" in validation:
            acts.append(float(validation["act_at_target"]))
    if not acts:
        return 1.0
    return max(1e-6, float(np.median(np.asarray(acts, dtype=np.float32))))


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


def _normalize_score_0_4(value: Any) -> float:
    try:
        return float(max(0.0, min(4.0, float(value))) / 4.0)
    except Exception:
        return float("nan")


def _mean_confuser_score(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def _load_selection_feature_keys(manifest_path: Path, *, limit: int) -> list[str]:
    payload = _read_json(manifest_path)
    feature_rows = list(payload["features"])
    if limit > 0:
        feature_rows = feature_rows[:limit]
    return [str(row["feature_key"]) for row in feature_rows]


def _ensure_original_token_box(
    *,
    session_dir: Path,
    image_path: str,
    block_idx: int,
    sample_id: int,
    token_idx: int,
    token_cache: dict[str, dict[str, str]],
) -> dict[str, str]:
    uid = token_uid(block_idx, sample_id, token_idx)
    cached = token_cache.get(uid)
    if cached is not None:
        return cached
    token_dir = session_dir / "record_assets" / _slug(uid)
    token_dir.mkdir(parents=True, exist_ok=True)
    original_path = token_dir / "original_token_box.png"
    save_original_with_token_box(image_path, original_path, token_idx, marker_style="cross")
    payload = {
        "token_uid": uid,
        "original_with_token_box": str(original_path),
    }
    token_cache[uid] = payload
    return payload


def _axis1_negatives(actmap: np.ndarray, cosine_map: np.ndarray, target_idx: int, *, count: int) -> list[int]:
    target_act = float(actmap[int(target_idx)])
    candidate_indices = [idx for idx in range(int(actmap.shape[0])) if idx != int(target_idx)]
    low_activation_max = max(1e-6, 0.10 * max(target_act, 1e-6))
    confident_pool = [idx for idx in candidate_indices if float(actmap[idx]) <= low_activation_max]
    if len(confident_pool) < count:
        candidate_indices_sorted = sorted(candidate_indices, key=lambda idx: (float(actmap[idx]), idx))
        confident_pool = candidate_indices_sorted[: max(count * 4, count)]
    hard_sorted = sorted(confident_pool, key=lambda idx: (-float(cosine_map[idx]), float(actmap[idx]), int(idx)))
    return [int(idx) for idx in hard_sorted[:count]]


def _supp_prompt(label: str, description: str, record_codes: list[str]) -> str:
    codes = ", ".join(record_codes)
    description_line = f"Feature description: {description}\n" if description else ""
    order_lines = [f"- {code}: one original image with a cyan-cross-marked token" for code in record_codes]
    return (
        "You are evaluating whether a feature explanation predicts held-out token activations.\n\n"
        "You will receive four individual images, not a composite sheet. Each image shows only the original image with one cyan cross marking the token.\n"
        "Images are provided in this exact record-code order:\n"
        + "\n".join(order_lines)
        + "\n\n"
        f"Feature label: {label}\n"
        f"{description_line}"
        "Task:\n"
        "- For each record code, predict how strongly this feature should activate on that cyan-cross-marked token.\n"
        "- Use integer scores 0, 1, 2, 3, or 4.\n"
        "- 0 means the explanation does not fit the token at all.\n"
        "- 4 means the explanation strongly fits the token.\n"
        "- Judge only the cyan-cross-marked token and its immediately visible local content.\n"
        "- Ignore panel styling, record codes, and visualization layout.\n\n"
        f"Valid record codes: {codes}\n"
        "Return only JSON."
    )


def _supp_schema() -> dict[str, Any]:
    score_field = {"type": "integer", "minimum": 0, "maximum": 4}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "record_scores": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "r01": score_field,
                    "r02": score_field,
                    "r03": score_field,
                    "r04": score_field,
                },
                "required": ["r01", "r02", "r03", "r04"],
            },
            "overall_confidence": {"type": "number"},
            "brief_reason": {"type": "string", "maxLength": 240},
        },
        "required": ["record_scores", "overall_confidence", "brief_reason"],
    }


def _run_codex_eval(
    *,
    schema_path: Path,
    out_json: Path,
    prompt_text: str,
    images: list[Path],
    model: str,
    reasoning_effort: str,
) -> tuple[int, str, str, float, dict[str, Any], list[str]]:
    result = run_isolated_codex_exec(
        artifact_dir=out_json.parent,
        artifact_stem=out_json.stem,
        prompt_text=prompt_text,
        schema=_read_json(schema_path),
        images=images,
        model=model,
        reasoning_effort=reasoning_effort,
        temp_prefix="supp_eval_",
    )
    return (
        int(result["returncode"]),
        str(result["stdout_tail"]),
        str(result["stderr_tail"]),
        float(result["elapsed_sec"]),
        dict(result["output"]),
        list(result["forbidden_trace_hits"]),
    )

def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3f}"


def _rel(base: Path, path: Path) -> str:
    return os.path.relpath(path, start=base)


def _build_variant_html(
    *,
    out_path: Path,
    variant_id: str,
    split_name: str,
    session_name: str,
    items: list[dict[str, Any]],
) -> None:
    def render_images(images: list[dict[str, str]]) -> str:
        cards = []
        for row in images:
            cards.append(
                f"""
                <div class="panel">
                  <div class="meta">{html.escape(str(row['record_code']))}</div>
                  <img src="{html.escape(str(row['image_rel']))}" alt="{html.escape(str(row['record_code']))}">
                </div>
                """
            )
        return f"<div class='panel-grid'>{''.join(cards)}</div>"

    cards = []
    for item in items:
        description_html = ""
        description = _norm_text(item.get("description"))
        if description:
            description_html = f"<div><strong>description:</strong> {html.escape(description)}</div>"
        cards.append(
            f"""
            <div class="feature">
              <div class="meta">{html.escape(str(item['feature_key']))} | {html.escape(split_name)}</div>
              <div class="label">{html.escape(str(item['canonical_label']))}</div>
              {description_html}
              {render_images(item["images"])}
            </div>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenAI-Style Supplementary Pilot</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1500px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta {{ color:#6b645d; margin-bottom:8px; }}
    .label {{ font-size:22px; font-weight:700; margin-bottom:8px; }}
    .panel-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .panel img {{ width:100%; border-radius:12px; border:1px solid #ddd3c6; background:#f0ebe2; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>OpenAI-Style Supplementary Pilot</h1>
      <div class="meta">Session: {html.escape(session_name)} | Variant: {html.escape(variant_id)} | Split: {html.escape(split_name)}</div>
      <div class="meta">Judge evidence mirrors the machine evaluation exactly: label text plus original image with cyan-cross-marked token only.</div>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    out_path.write_text(html_text)


def _build_human_eval_pages(
    *,
    config: EvalConfig,
    session_dir: Path,
    manifest: dict[str, Any],
    variant_payloads: dict[str, dict[str, Any]],
) -> None:
    review_root = config.workspace_root / "outputs" / "review_sessions"
    served_root = review_root / "supplementary_pilot_sessions"
    served_session_dir = served_root / manifest["session_name"]

    if served_root.is_symlink() or served_root.is_file():
        served_root.unlink()
    served_root.mkdir(parents=True, exist_ok=True)
    if served_session_dir.is_symlink() or served_session_dir.is_file():
        served_session_dir.unlink()
    if served_session_dir.exists():
        shutil.rmtree(served_session_dir)
    served_session_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("pilot_manifest.json", "summary.json", "report.md", "file_index.json"):
        src_file = session_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, served_session_dir / filename)

    human_root = served_session_dir / "human_eval"
    human_root.mkdir(parents=True, exist_ok=True)

    index_cards: list[str] = []
    for variant_id in manifest["variant_order"]:
        variant_dir = human_root / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("supp_valid", "supp_test"):
            page_name = f"{split_name}.html"
            items = []
            for item in variant_payloads[variant_id][split_name]["per_item"]:
                items.append(
                    {
                        "feature_key": item["feature_key"],
                        "canonical_label": item["canonical_label"],
                        "description": item["description"],
                        "images": [
                            {
                                "record_code": str(record["record_code"]),
                                "image_rel": _rel(variant_dir, Path(str(record["original_with_token_box"]))),
                            }
                            for record in item["records"]
                        ],
                    }
                )
            _build_variant_html(
                out_path=variant_dir / page_name,
                variant_id=variant_id,
                split_name=split_name,
                session_name=manifest["session_name"],
                items=items,
            )
        index_cards.append(
            f"""
            <div class="card">
              <h2>{html.escape(variant_id)}</h2>
              <div><a href="{html.escape(variant_id)}/supp_valid.html">supp_valid</a></div>
              <div><a href="{html.escape(variant_id)}/supp_test.html">supp_test</a></div>
            </div>
            """
        )

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Supplementary Human Audit</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .hero, .card {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    a {{ color:#2457c5; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>OpenAI-Style Supplementary Human Audit</h1>
      <div>Session: {html.escape(manifest['session_name'])}</div>
    </div>
    {''.join(index_cards)}
  </div>
</body>
</html>
"""
    (human_root / "index.html").write_text(index_html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-name", default="openai_style_supplementary_pilot_20260420")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--features-limit", type=int, default=0)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--variant-a-id", default="correct_erf")
    parser.add_argument("--variant-a-session", default="random20_gpt54_medium_erfablation_carrierfirst_20260420_correct_erf")
    parser.add_argument("--variant-b-id", default="no_erf")
    parser.add_argument("--variant-b-session", default="random20_gpt54_medium_erfablation_carrierfirst_20260420_no_erf")
    parser.add_argument("--variant-c-id", default="")
    parser.add_argument("--variant-c-session", default="")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--vision-model-name", default="")
    parser.add_argument("--train-per-feature", type=int, default=0)
    parser.add_argument("--holdout-per-feature", type=int, default=0)
    parser.add_argument("--deciles-root", default="")
    parser.add_argument("--offline-meta-root", default="")
    parser.add_argument("--checkpoints-root", default="")
    parser.add_argument("--checkpoint-pattern", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--erf-threshold", type=float, default=0.90)
    args = parser.parse_args()

    config = _build_config_from_args(args)
    session_dir = config.workspace_root / "outputs" / "supplementary_pilot_sessions" / args.session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    if args.variant:
        variant_specs = []
        for spec in list(args.variant):
            if "=" not in str(spec):
                raise SystemExit(f"Invalid --variant value {spec!r}; expected VARIANT_ID=SESSION_NAME")
            variant_id, session_name = str(spec).split("=", 1)
            variant_specs.append((variant_id.strip(), session_name.strip()))
    else:
        variant_specs = []
        for variant_id, session_name in (
            (str(args.variant_a_id), str(args.variant_a_session)),
            (str(args.variant_b_id), str(args.variant_b_session)),
            (str(args.variant_c_id), str(args.variant_c_session)),
        ):
            if variant_id.strip() and session_name.strip():
                variant_specs.append((variant_id.strip(), session_name.strip()))
    if len(variant_specs) < 2:
        raise SystemExit("Need at least 2 variants")
    variant_ids = [variant_id for variant_id, _ in variant_specs]
    if len(set(variant_ids)) != len(variant_ids):
        raise SystemExit(f"Variant IDs must be unique: {variant_ids}")
    variant_manifests = {
        variant_id: _build_variant_manifest(config, session_name, "raw_predictions.json")
        for variant_id, session_name in variant_specs
    }

    selected_feature_keys = _load_selection_feature_keys(
        variant_manifests[variant_ids[0]][0],
        limit=int(args.features_limit),
    )
    feature_bank = load_feature_bank(config)
    feature_lookup = {
        str(feature["feature_key"]): feature
        for block_payload in feature_bank["blocks"].values()
        for feature in block_payload["features"]
    }
    selected_features = [feature_lookup[key] for key in selected_feature_keys]
    selected_by_block: dict[int, list[dict[str, Any]]] = {}
    full_by_block: dict[int, list[dict[str, Any]]] = {}
    for block_idx in config.blocks:
        rows = [feature for feature in selected_features if int(feature["block_idx"]) == int(block_idx)]
        selected_by_block[int(block_idx)] = sorted(rows, key=lambda row: int(row["feature_id"]))
        all_rows = [feature for feature in feature_lookup.values() if int(feature["block_idx"]) == int(block_idx)]
        full_by_block[int(block_idx)] = sorted(all_rows, key=lambda row: int(row["feature_id"]))

    thresholds: dict[str, float] = {}
    for feature in feature_lookup.values():
        scale = _feature_target_activation_scale(feature)
        thresholds[str(feature["feature_key"])] = max(1e-6, float(config.axis2_positive_relative_threshold) * scale)

    variant_labels = {
        variant_id: _load_variant_labels(raw_path)
        for variant_id, (_, raw_path) in variant_manifests.items()
    }

    runtime = LegacyRuntime(config)
    token_cache: dict[str, dict[str, str]] = {}
    try:
        confuser_rankings: dict[str, list[dict[str, Any]]] = {}
        for feature in selected_features:
            feature_key = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            block_features = full_by_block[block_idx]
            feature_ids = [int(row["feature_id"]) for row in block_features]
            feature_keys = [str(row["feature_key"]) for row in block_features]
            accum: dict[str, list[float]] = {key: [] for key in feature_keys if key != feature_key}
            for holdout_row in feature["holdout"]:
                image_path = str(holdout_row["image_path"])
                token_idx = int(holdout_row["target_patch_idx"])
                values = runtime.feature_vector_at_token(image_path, block_idx, token_idx, feature_ids)
                for idx, candidate_key in enumerate(feature_keys):
                    if candidate_key == feature_key:
                        continue
                    threshold = thresholds[candidate_key]
                    accum[candidate_key].append(float(values[idx]) / max(threshold, 1e-6))
            ranked = []
            for candidate_key, vals in accum.items():
                ranked.append(
                    {
                        "feature_key": candidate_key,
                        "mean_norm_activation": _mean_confuser_score(vals),
                    }
                )
            confuser_rankings[feature_key] = sorted(
                ranked,
                key=lambda row: (-float(row["mean_norm_activation"]), str(row["feature_key"])),
            )

        supp_valid_items: list[dict[str, Any]] = []
        supp_test_items: list[dict[str, Any]] = []

        for feature in selected_features:
            feature_key = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            scale = _feature_target_activation_scale(feature)
            holdout_rows = sorted(
                list(feature["holdout"]),
                key=lambda row: (int(row["sample_id"]), int(row["target_patch_idx"])),
            )
            if len(holdout_rows) < 2:
                raise RuntimeError(f"Expected at least 2 holdout rows for {feature_key}, got {len(holdout_rows)}")

            for split_idx, split_name in enumerate(("supp_valid", "supp_test")):
                holdout_row = holdout_rows[split_idx]
                sample_id = int(holdout_row["sample_id"])
                target_idx = int(holdout_row["target_patch_idx"])
                image_path = str(holdout_row["image_path"])

                positive_box = _ensure_original_token_box(
                    session_dir=session_dir,
                    image_path=image_path,
                    block_idx=block_idx,
                    sample_id=sample_id,
                    token_idx=target_idx,
                    token_cache=token_cache,
                )
                positive_raw = float((holdout_row.get("validation") or {}).get("act_at_target", holdout_row.get("ledger_score", 0.0)))

                actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
                cosine_map = runtime.token_cosine_map(image_path, block_idx, target_idx)
                same_image_neg_idx = _axis1_negatives(
                    np.asarray(actmap, dtype=np.float32),
                    np.asarray(cosine_map, dtype=np.float32),
                    target_idx,
                    count=1,
                )[0]
                same_image_box = _ensure_original_token_box(
                    session_dir=session_dir,
                    image_path=image_path,
                    block_idx=block_idx,
                    sample_id=sample_id,
                    token_idx=int(same_image_neg_idx),
                    token_cache=token_cache,
                )
                same_image_raw = float(actmap[int(same_image_neg_idx)])

                confuser_records: list[dict[str, Any]] = []
                for ranked in confuser_rankings[feature_key]:
                    confuser_key = str(ranked["feature_key"])
                    confuser_feature = feature_lookup[confuser_key]
                    confuser_holdout_rows = sorted(
                        list(confuser_feature["holdout"]),
                        key=lambda row: (int(row["sample_id"]), int(row["target_patch_idx"])),
                    )
                    if split_idx >= len(confuser_holdout_rows):
                        continue
                    confuser_row = confuser_holdout_rows[split_idx]
                    conf_sample_id = int(confuser_row["sample_id"])
                    conf_token_idx = int(confuser_row["target_patch_idx"])
                    conf_image_path = str(confuser_row["image_path"])
                    current_feature_act = float(
                        runtime.feature_vector_at_token(conf_image_path, block_idx, conf_token_idx, [feature_id])[0]
                    )
                    if current_feature_act >= thresholds[feature_key]:
                        continue
                    conf_box = _ensure_original_token_box(
                        session_dir=session_dir,
                        image_path=conf_image_path,
                        block_idx=block_idx,
                        sample_id=conf_sample_id,
                        token_idx=conf_token_idx,
                        token_cache=token_cache,
                    )
                    confuser_records.append(
                        {
                            "role": "confuser_negative",
                            "source_feature_key": confuser_key,
                            "sample_id": conf_sample_id,
                            "token_idx": conf_token_idx,
                            "raw_activation": current_feature_act,
                            **conf_box,
                        }
                    )
                    if len(confuser_records) >= 2:
                        break
                if len(confuser_records) != 2:
                    raise RuntimeError(f"Could not build 2 confuser negatives for {feature_key} {split_name}")

                records = [
                    {
                        "role": "positive",
                        "sample_id": sample_id,
                        "token_idx": target_idx,
                        "raw_activation": positive_raw,
                        **positive_box,
                    },
                    {
                        "role": "same_image_negative",
                        "sample_id": sample_id,
                        "token_idx": int(same_image_neg_idx),
                        "raw_activation": same_image_raw,
                        **same_image_box,
                    },
                    *confuser_records,
                ]

                for rec_idx, record in enumerate(records, start=1):
                    record["record_code"] = f"r{rec_idx:02d}"
                    record["binary_label"] = 1 if record["role"] == "positive" else 0
                    record["normalized_activation"] = float(
                        max(0.0, min(1.0, float(record["raw_activation"]) / max(scale, 1e-6)))
                    )

                item = {
                    "feature_key": feature_key,
                    "block_idx": block_idx,
                    "feature_id": feature_id,
                    "split": split_name,
                    "scale": scale,
                    "sample_id": sample_id,
                    "token_idx": target_idx,
                    "record_codes": [str(record["record_code"]) for record in records],
                    "input_images": [
                        {
                            "record_code": str(record["record_code"]),
                            "image_path": str(record["original_with_token_box"]),
                        }
                        for record in records
                    ],
                    "records": records,
                }
                if split_name == "supp_valid":
                    supp_valid_items.append(item)
                else:
                    supp_test_items.append(item)
    finally:
        runtime.close()

    pilot_manifest = {
        "session_name": args.session_name,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "selected_feature_keys": selected_feature_keys,
        "supp_valid_items": supp_valid_items,
        "supp_test_items": supp_test_items,
        "variant_sessions": {variant_id: session_name for variant_id, session_name in variant_specs},
        "variant_order": list(variant_ids),
        "thresholds": thresholds,
    }
    write_json(session_dir / "pilot_manifest.json", pilot_manifest)

    schema_path = session_dir / "supp_output_schema.json"
    write_json(schema_path, _supp_schema())

    results_root = session_dir / "variant_results"
    results_root.mkdir(parents=True, exist_ok=True)

    def run_variant(variant_id: str, split_name: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        variant_dir = results_root / variant_id / split_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for item in items:
            label = variant_labels[variant_id][item["feature_key"]]
            prompt_text = _supp_prompt(
                label=label["canonical_label"],
                description=label["description"],
                record_codes=list(item["record_codes"]),
            )
            out_json = variant_dir / f"{_slug(item['feature_key'])}__sample_{item['sample_id']}.json"
            tasks.append(
                {
                    "item": item,
                    "prompt_text": prompt_text,
                    "out_json": out_json,
                    "image_paths": [Path(row["image_path"]) for row in item["input_images"]],
                    "canonical_label": label["canonical_label"],
                    "description": label["description"],
                }
            )

        def worker(task: dict[str, Any]) -> dict[str, Any]:
            returncode, stdout_tail, stderr_tail, elapsed, output, forbidden_trace_hits = _run_codex_eval(
                schema_path=schema_path,
                out_json=task["out_json"],
                prompt_text=task["prompt_text"],
                images=list(task["image_paths"]),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            item = task["item"]
            raw_scores = dict(output.get("record_scores") or {})
            ordered_pred = []
            ordered_true = []
            binary_true = []
            record_rows: list[dict[str, Any]] = []
            for record in item["records"]:
                code = str(record["record_code"]).lower()
                pred_norm = _normalize_score_0_4(raw_scores.get(code) if code in raw_scores else raw_scores.get(code.upper()))
                true_norm = float(record["normalized_activation"])
                binary = int(record["binary_label"])
                ordered_pred.append(pred_norm)
                ordered_true.append(true_norm)
                binary_true.append(binary)
                record_rows.append(
                    {
                        "record_code": code,
                        "role": str(record["role"]),
                        "binary_label": binary,
                        "raw_activation": float(record["raw_activation"]),
                        "normalized_activation": true_norm,
                        "predicted_score_0_4": int(raw_scores.get(code, raw_scores.get(code.upper(), 0))),
                        "predicted_score_norm": pred_norm,
                        "original_with_token_box": str(record["original_with_token_box"]),
                    }
                )
            y_true = np.asarray(binary_true, dtype=np.int64)
            y_score = np.asarray(ordered_pred, dtype=np.float64)
            y_true_cont = np.asarray(ordered_true, dtype=np.float64)
            return {
                "feature_key": item["feature_key"],
                "block_idx": int(item["block_idx"]),
                "feature_id": int(item["feature_id"]),
                "split": split_name,
                "sample_id": int(item["sample_id"]),
                "canonical_label": task["canonical_label"],
                "description": task["description"],
                "spearman_rho": _spearman_score(y_true_cont, y_score),
                "auroc": roc_auc_binary(y_true, y_score),
                "ap": average_precision_binary(y_true, y_score),
                "mae": float(np.mean(np.abs(y_true_cont - y_score))),
                "overall_confidence": float(output.get("overall_confidence", 0.0) or 0.0),
                "elapsed_sec": elapsed,
                "returncode": returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "forbidden_trace_hits": forbidden_trace_hits,
                "output": output,
                "records": record_rows,
                "input_images": list(item["input_images"]),
            }

        rows: list[dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for idx, future in enumerate(cf.as_completed(futures), start=1):
                rows.append(future.result())
                if idx % 12 == 0 or idx == len(tasks):
                    print(f"[supp {split_name} {variant_id} {idx:03d}/{len(tasks):03d}]", flush=True)

        row_by_key = {(row["feature_key"], int(row["sample_id"])): row for row in rows}
        ordered_rows = [row_by_key[(item["feature_key"], int(item["sample_id"]))] for item in items]
        block_metrics: dict[str, Any] = {}
        for block_idx in config.blocks:
            block_rows = [row for row in ordered_rows if int(row["block_idx"]) == int(block_idx)]
            block_metrics[str(block_idx)] = {
                "n_items": len(block_rows),
                "spearman_rho": float(np.nanmean([row["spearman_rho"] for row in block_rows])) if block_rows else float("nan"),
                "auroc": float(np.nanmean([row["auroc"] for row in block_rows])) if block_rows else float("nan"),
                "ap": float(np.nanmean([row["ap"] for row in block_rows])) if block_rows else float("nan"),
                "mae": float(np.nanmean([row["mae"] for row in block_rows])) if block_rows else float("nan"),
            }
        summary = {
            "variant_id": variant_id,
            "split": split_name,
            "n_items": len(ordered_rows),
            "records_per_item": 4,
            "overall": {
                "spearman_rho": float(np.nanmean([row["spearman_rho"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "auroc": float(np.nanmean([row["auroc"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "average_precision": float(np.nanmean([row["ap"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mae": float(np.nanmean([row["mae"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mean_confidence": float(np.nanmean([row["overall_confidence"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mean_elapsed_sec": float(np.nanmean([row["elapsed_sec"] for row in ordered_rows])) if ordered_rows else float("nan"),
            },
            "per_block": block_metrics,
            "per_item": ordered_rows,
        }
        write_json(variant_dir / "results.json", summary)
        return summary

    variant_results: dict[str, dict[str, Any]] = {}
    for variant_id in variant_ids:
        variant_results[variant_id] = {
            "supp_valid": run_variant(variant_id, "supp_valid", supp_valid_items),
            "supp_test": run_variant(variant_id, "supp_test", supp_test_items),
        }

    report_lines = [
        "# OpenAI-Style Supplementary Pilot",
        "",
        f"- Session: `{args.session_name}`",
        f"- Model: `{args.model}`",
        f"- Reasoning effort: `{args.reasoning_effort}`",
        f"- Feature universe: `{len(selected_feature_keys)}` shared features",
        "- Evidence shown to the judge: original image + cyan-cross-marked token only",
        "- Explanation shown to the judge: canonical_label + description",
        "- `supp_valid` is for smoke/debug only",
        "- Main supplementary headline metrics are from `supp_test`",
        "",
        "## supp_test Headline Metrics",
        "",
        "| metric | " + " | ".join(variant_ids) + " |",
        "|---:|" + "|".join(["---:"] * len(variant_ids)) + "|",
    ]
    for metric in ("spearman_rho", "auroc", "average_precision", "mae", "mean_confidence", "mean_elapsed_sec"):
        vals = [
            _format_float(variant_results[variant_id]["supp_test"]["overall"][metric])
            for variant_id in variant_ids
        ]
        report_lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    report_lines.extend(
        [
            "",
            "## supp_valid Metrics",
            "",
            "| metric | " + " | ".join(variant_ids) + " |",
            "|---:|" + "|".join(["---:"] * len(variant_ids)) + "|",
        ]
    )
    for metric in ("spearman_rho", "auroc", "average_precision", "mae", "mean_confidence", "mean_elapsed_sec"):
        vals = [
            _format_float(variant_results[variant_id]["supp_valid"]["overall"][metric])
            for variant_id in variant_ids
        ]
        report_lines.append(f"| {metric} | " + " | ".join(vals) + " |")

    report_lines.extend(
        [
            "",
            "## Focus Features (supp_test Spearman / AUROC)",
            "",
            "| feature | " + " | ".join(f"{variant_id}" for variant_id in variant_ids) + " |",
            "|---|" + "|".join(["---"] * len(variant_ids)) + "|",
        ]
    )
    focus_keys = [
        "block_2/feature_21767",
        "block_2/feature_6741",
        "block_2/feature_11999",
        "block_6/feature_15095",
        "block_6/feature_16384",
        "block_6/feature_7180",
        "block_6/feature_2322",
        "block_10/feature_13682",
        "block_10/feature_6900",
        "block_10/feature_19572",
        "block_10/feature_24103",
        "block_10/feature_8323",
        "block_10/feature_9816",
    ]
    by_variant_feature = {
        variant_id: {str(row["feature_key"]): row for row in variant_results[variant_id]["supp_test"]["per_item"]}
        for variant_id in variant_ids
    }
    for key in focus_keys:
        if key not in by_variant_feature[variant_ids[0]]:
            continue
        values = []
        for variant_id in variant_ids:
            row = by_variant_feature[variant_id][key]
            values.append(f"{_format_float(float(row['spearman_rho']))} / {_format_float(float(row['auroc']))}")
        report_lines.append(f"| `{key}` | " + " | ".join(values) + " |")

    summary_payload = {
        "session_name": args.session_name,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "selected_feature_keys": selected_feature_keys,
        "variant_order": list(variant_ids),
        "supp_valid": {variant_id: variant_results[variant_id]["supp_valid"] for variant_id in variant_ids},
        "supp_test": {variant_id: variant_results[variant_id]["supp_test"] for variant_id in variant_ids},
    }
    write_json(session_dir / "summary.json", summary_payload)
    (session_dir / "report.md").write_text("\n".join(report_lines) + "\n")

    file_index = {
        "session_name": args.session_name,
        "canonical_session_dir": str(session_dir),
        "summary_json": str(session_dir / "summary.json"),
        "report_md": str(session_dir / "report.md"),
        "pilot_manifest_json": str(session_dir / "pilot_manifest.json"),
        "supp_output_schema_json": str(schema_path),
        "record_assets_dir": str(session_dir / "record_assets"),
        "variant_results": {
            variant_id: {
                "supp_valid_results_json": str(results_root / variant_id / "supp_valid" / "results.json"),
                "supp_test_results_json": str(results_root / variant_id / "supp_test" / "results.json"),
            }
            for variant_id in variant_ids
        },
        "source_sessions": {variant_id: session_name for variant_id, session_name in variant_specs},
        "scripts": {
            "runner": str(Path(__file__).resolve()),
        },
    }
    write_json(session_dir / "file_index.json", file_index)

    _build_human_eval_pages(
        config=config,
        session_dir=session_dir,
        manifest=pilot_manifest,
        variant_payloads=variant_results,
    )


if __name__ == "__main__":
    main()
