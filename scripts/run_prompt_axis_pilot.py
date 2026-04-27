from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
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
from autolabel_eval.metrics import ndcg_at_k, recall_at_k
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


def _feature_target_activation_scale(feature: dict[str, Any]) -> float:
    acts: list[float] = []
    for row in list(feature["train"]) + list(feature["holdout"]):
        validation = row.get("validation") or {}
        if "act_at_target" in validation:
            acts.append(float(validation["act_at_target"]))
    if not acts:
        return 0.0
    return float(np.median(np.asarray(acts, dtype=np.float32)))


def _axis1_prompt(label: str, description: str, candidate_codes: list[str]) -> str:
    codes = ", ".join(candidate_codes)
    order_lines = []
    for code in candidate_codes:
        order_lines.append(f"- {code}: original image with one cyan-cross-marked token")
    description_line = f"Feature description: {description}\n\n" if description else ""
    return (
        "You are evaluating whether a feature label can identify the correct token in one image.\n\n"
        "You will receive individual images, not a contact sheet.\n"
        "For each candidate token code, one image is provided in candidate-code order.\n"
        "Each image shows the original image with one cyan cross marking the candidate token.\n\n"
        "The image groups appear in this exact order:\n"
        + "\n".join(order_lines)
        + "\n\n"
        f"Feature label: {label}\n"
        f"{description_line}"
        "Task:\n"
        "- Choose the single candidate code whose cyan-cross-marked token best matches the feature label.\n"
        "- Use token-local evidence, not the whole scene.\n"
        "- Ignore rendering artifacts and panel styling.\n\n"
        f"Valid candidate codes: {codes}\n"
        "Return only JSON."
    )


def _axis2_prompt(candidates: list[dict[str, Any]]) -> str:
    lines = [
        "You are evaluating how discriminative feature labels are for one token.",
        "",
        "You will receive one individual image containing token evidence.",
        "The image shows the original image with a cyan cross marking the target token.",
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
            "- Pick the single best matching candidate label.",
            "- Also return a full ranking of all candidate codes from best to worst.",
            "- Prefer the label that best explains the token-local cue, not just the broad scene.",
            "- Ignore rendering artifacts and panel styling.",
            "",
            "Return only JSON.",
        ]
    )
    return "\n".join(lines)


def _axis1_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_candidate": {"type": "string", "maxLength": 8},
            "confidence": {"type": "number"},
            "brief_reason": {"type": "string", "maxLength": 240},
        },
        "required": ["selected_candidate", "confidence", "brief_reason"],
    }


def _axis2_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "best_candidate": {"type": "string", "maxLength": 8},
            "ranked_candidates": {
                "type": "array",
                "items": {"type": "string", "maxLength": 8},
                "maxItems": 20,
            },
            "confidence": {"type": "number"},
            "brief_reason": {"type": "string", "maxLength": 240},
        },
        "required": ["best_candidate", "ranked_candidates", "confidence", "brief_reason"],
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
        temp_prefix="axis_eval_",
    )
    return (
        int(result["returncode"]),
        str(result["stdout_tail"]),
        str(result["stderr_tail"]),
        float(result["elapsed_sec"]),
        dict(result["output"]),
        list(result["forbidden_trace_hits"]),
    )


def _mean_confuser_score(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def _ensure_token_evidence(
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
    token_dir = session_dir / "token_assets" / _slug(uid)
    token_dir.mkdir(parents=True, exist_ok=True)
    original_path = token_dir / "original_token_box.png"
    save_original_with_token_box(image_path, original_path, token_idx, marker_style="cross")
    payload = {
        "token_uid": uid,
        "original_with_token_box": str(original_path),
    }
    token_cache[uid] = payload
    return payload


def _axis1_hard_negatives(actmap: np.ndarray, cosine_map: np.ndarray, target_idx: int, *, count: int) -> list[int]:
    target_act = float(actmap[int(target_idx)])
    candidate_indices = [idx for idx in range(int(actmap.shape[0])) if idx != int(target_idx)]
    low_activation_max = max(1e-6, 0.10 * max(target_act, 1e-6))
    confident_pool = [idx for idx in candidate_indices if float(actmap[idx]) <= low_activation_max]
    if len(confident_pool) < count:
        candidate_indices_sorted = sorted(candidate_indices, key=lambda idx: (float(actmap[idx]), idx))
        confident_pool = candidate_indices_sorted[: max(count * 4, count)]
    hard_sorted = sorted(confident_pool, key=lambda idx: (-float(cosine_map[idx]), float(actmap[idx]), int(idx)))
    return [int(idx) for idx in hard_sorted[:count]]


def _axis1_random_negatives(num_tokens: int, target_idx: int, *, count: int, rng: random.Random) -> list[int]:
    candidate_indices = [idx for idx in range(int(num_tokens)) if idx != int(target_idx)]
    if len(candidate_indices) <= count:
        return [int(idx) for idx in candidate_indices]
    return [int(idx) for idx in rng.sample(candidate_indices, count)]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-name", default="prompt_axis_pilot_20260420")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--features-limit", type=int, default=0)
    parser.add_argument("--axis2-candidate-count", type=int, default=16)
    parser.add_argument("--axis1-negative-mode", choices=("hard", "random"), default="random")
    parser.add_argument("--skip-axis2", action="store_true")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--variant-a-id", default="carrier_first")
    parser.add_argument("--variant-a-session", default="random20_gpt54_medium_carrierfirst_cyan_rerender_20260420")
    parser.add_argument("--variant-b-id", default="short_hardneg")
    parser.add_argument("--variant-b-session", default="random20_gpt54_medium_shortprompt_hardneg_cyan_rerender_20260420")
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
    if int(args.axis2_candidate_count) < 2:
        raise SystemExit("--axis2-candidate-count must be at least 2")

    config = _build_config_from_args(args)
    session_dir = config.workspace_root / "outputs" / "axis_pilot_sessions" / args.session_name
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

    selection_manifest = _read_json(variant_manifests[variant_ids[0]][0])
    selected_feature_keys = [str(row["feature_key"]) for row in selection_manifest["features"]]
    if args.features_limit and int(args.features_limit) > 0:
        selected_feature_keys = selected_feature_keys[: int(args.features_limit)]

    feature_bank = load_feature_bank(config)
    feature_lookup = {
        str(feature["feature_key"]): feature
        for block_payload in feature_bank["blocks"].values()
        for feature in block_payload["features"]
    }
    selected_features = [feature_lookup[key] for key in selected_feature_keys]
    selected_by_block: dict[int, list[dict[str, Any]]] = {}
    for block_idx in config.blocks:
        rows = [feature for feature in selected_features if int(feature["block_idx"]) == int(block_idx)]
        selected_by_block[int(block_idx)] = sorted(rows, key=lambda row: int(row["feature_id"]))

    thresholds: dict[str, float] = {}
    for feature in selected_features:
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
            block_features = selected_by_block[block_idx]
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

        axis1_items: list[dict[str, Any]] = []
        axis2_items: list[dict[str, Any]] = []
        for feature in selected_features:
            feature_key = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            holdout_row = sorted(
                list(feature["holdout"]),
                key=lambda row: (int(row["sample_id"]), int(row["target_patch_idx"])),
            )[0]
            image_path = str(holdout_row["image_path"])
            sample_id = int(holdout_row["sample_id"])
            target_idx = int(holdout_row["target_patch_idx"])

            positive_evidence = _ensure_token_evidence(
                session_dir=session_dir,
                image_path=image_path,
                block_idx=block_idx,
                sample_id=sample_id,
                token_idx=target_idx,
                token_cache=token_cache,
            )
            actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
            cosine_map = runtime.token_cosine_map(image_path, block_idx, target_idx)
            axis1_rng = random.Random(500 + block_idx * 100000 + sample_id * 101 + feature_id)
            if str(args.axis1_negative_mode) == "hard":
                neg_indices = _axis1_hard_negatives(
                    np.asarray(actmap, dtype=np.float32),
                    np.asarray(cosine_map, dtype=np.float32),
                    target_idx,
                    count=3,
                )
            else:
                neg_indices = _axis1_random_negatives(int(actmap.shape[0]), target_idx, count=3, rng=axis1_rng)
            axis1_candidates: list[dict[str, Any]] = [
                {
                    "token_idx": target_idx,
                    "is_positive": True,
                    **positive_evidence,
                }
            ]
            for neg_idx in neg_indices:
                neg_evidence = _ensure_token_evidence(
                    session_dir=session_dir,
                    image_path=image_path,
                    block_idx=block_idx,
                    sample_id=sample_id,
                    token_idx=int(neg_idx),
                    token_cache=token_cache,
                )
                axis1_candidates.append(
                    {
                        "token_idx": int(neg_idx),
                        "is_positive": False,
                        **neg_evidence,
                    }
                )
            shuffle_rng = random.Random(1000 + block_idx * 100000 + sample_id * 101 + feature_id)
            shuffle_rng.shuffle(axis1_candidates)
            for idx, row in enumerate(axis1_candidates, start=1):
                row["candidate_code"] = f"c{idx:02d}"
            gold_axis1 = next(str(row["candidate_code"]) for row in axis1_candidates if bool(row["is_positive"]))
            axis1_input_images = []
            for row in axis1_candidates:
                axis1_input_images.append(
                    {
                        "candidate_code": str(row["candidate_code"]),
                        "kind": "original_with_token_box",
                        "image_path": str(row["original_with_token_box"]),
                    }
                )
            axis1_items.append(
                {
                    "feature_key": feature_key,
                    "block_idx": block_idx,
                    "feature_id": feature_id,
                    "sample_id": sample_id,
                    "token_idx": target_idx,
                    "gold_code": gold_axis1,
                    "candidate_codes": [str(row["candidate_code"]) for row in axis1_candidates],
                    "input_images": axis1_input_images,
                    "candidates": axis1_candidates,
                }
            )

            block_features = selected_by_block[block_idx]
            block_feature_ids = [int(row["feature_id"]) for row in block_features]
            block_feature_keys = [str(row["feature_key"]) for row in block_features]
            act_values = runtime.feature_vector_at_token(image_path, block_idx, target_idx, block_feature_ids)
            act_by_key = {key: float(act_values[idx]) for idx, key in enumerate(block_feature_keys)}
            target_candidate_count = int(args.axis2_candidate_count)
            candidate_keys = [feature_key]
            for row in confuser_rankings[feature_key]:
                key = str(row["feature_key"])
                if act_by_key[key] < thresholds[key]:
                    candidate_keys.append(key)
                if len(candidate_keys) >= target_candidate_count:
                    break
            if len(candidate_keys) < target_candidate_count:
                for key in block_feature_keys:
                    if key != feature_key and key not in candidate_keys and act_by_key[key] < thresholds[key]:
                        candidate_keys.append(key)
                    if len(candidate_keys) >= target_candidate_count:
                        break
            if len(candidate_keys) < target_candidate_count:
                raise RuntimeError(f"Could not construct {target_candidate_count}-way Axis 2 candidate set for {feature_key}")
            axis2_candidates = [
                {
                    "feature_key": key,
                    "canonical_label": "",
                    "description": "",
                    "is_gold": key == feature_key,
                    "confuser_score": 0.0 if key == feature_key else next(
                        (
                            float(row["mean_norm_activation"])
                            for row in confuser_rankings[feature_key]
                            if str(row["feature_key"]) == key
                        ),
                        float("nan"),
                    ),
                }
                for key in candidate_keys[:target_candidate_count]
            ]
            shuffle_rng = random.Random(2000 + block_idx * 100000 + sample_id * 101 + feature_id)
            shuffle_rng.shuffle(axis2_candidates)
            for idx, row in enumerate(axis2_candidates, start=1):
                row["candidate_code"] = f"c{idx:02d}"
            gold_axis2 = next(str(row["candidate_code"]) for row in axis2_candidates if bool(row["is_gold"]))
            axis2_items.append(
                {
                    "feature_key": feature_key,
                    "block_idx": block_idx,
                    "feature_id": feature_id,
                    "sample_id": sample_id,
                    "token_idx": target_idx,
                    "gold_code": gold_axis2,
                    "candidate_codes": [str(row["candidate_code"]) for row in axis2_candidates],
                    "input_images": [
                        {"kind": "original_with_token_box", "image_path": str(positive_evidence["original_with_token_box"])},
                    ],
                    "candidates": axis2_candidates,
                }
            )
    finally:
        runtime.close()

    write_json(
        session_dir / "pilot_manifest.json",
        {
            "session_name": args.session_name,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "selected_feature_keys": selected_feature_keys,
            "axis1_items": axis1_items,
            "axis2_items": axis2_items,
            "variant_sessions": {variant_id: session_name for variant_id, session_name in variant_specs},
            "thresholds": thresholds,
        },
    )

    axis1_schema_path = session_dir / "axis1_output_schema.json"
    axis2_schema_path = session_dir / "axis2_output_schema.json"
    write_json(axis1_schema_path, _axis1_schema())
    write_json(axis2_schema_path, _axis2_schema())

    results_root = session_dir / "variant_results"
    results_root.mkdir(parents=True, exist_ok=True)

    def run_axis1_variant(variant_id: str) -> dict[str, Any]:
        variant_dir = results_root / variant_id / "axis1"
        variant_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for item in axis1_items:
            label = variant_labels[variant_id][item["feature_key"]]
            prompt_text = _axis1_prompt(
                label=label["canonical_label"],
                description=label["description"],
                candidate_codes=list(item["candidate_codes"]),
            )
            out_json = variant_dir / f"{_slug(item['feature_key'])}__sample_{item['sample_id']}.json"
            tasks.append(
                {
                    "item": item,
                    "prompt_text": prompt_text,
                    "out_json": out_json,
                    "image_paths": [Path(row["image_path"]) for row in item["input_images"]],
                }
            )

        def worker(task: dict[str, Any]) -> dict[str, Any]:
            returncode, stdout_tail, stderr_tail, elapsed, output, forbidden_trace_hits = _run_codex_eval(
                schema_path=axis1_schema_path,
                out_json=task["out_json"],
                prompt_text=task["prompt_text"],
                images=list(task["image_paths"]),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            item = task["item"]
            selected = str(output.get("selected_candidate", "")).lower()
            correct = int(selected == str(item["gold_code"]).lower())
            return {
                "feature_key": item["feature_key"],
                "block_idx": int(item["block_idx"]),
                "feature_id": int(item["feature_id"]),
                "sample_id": int(item["sample_id"]),
                "gold_code": str(item["gold_code"]).lower(),
                "selected_candidate": selected,
                "correct": correct,
                "confidence": float(output.get("confidence", 0.0) or 0.0),
                "elapsed_sec": elapsed,
                "returncode": returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "forbidden_trace_hits": forbidden_trace_hits,
                "output": output,
            }

        rows: list[dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for idx, future in enumerate(cf.as_completed(futures), start=1):
                rows.append(future.result())
                if idx % 12 == 0 or idx == len(tasks):
                    print(f"[axis1 {variant_id} {idx:03d}/{len(tasks):03d}]", flush=True)

        row_by_key = {(row["feature_key"], int(row["sample_id"])): row for row in rows}
        ordered_rows = [row_by_key[(item["feature_key"], int(item["sample_id"]))] for item in axis1_items]
        top1_values = [int(row["correct"]) for row in ordered_rows]
        block_metrics: dict[str, Any] = {}
        for block_idx in config.blocks:
            block_rows = [row for row in ordered_rows if int(row["block_idx"]) == int(block_idx)]
            block_metrics[str(block_idx)] = {
                "n_items": len(block_rows),
                "top1_accuracy": float(np.mean([row["correct"] for row in block_rows])) if block_rows else float("nan"),
            }
        summary = {
            "variant_id": variant_id,
            "axis": "axis1",
            "n_items": len(ordered_rows),
            "candidate_count": 4,
            "chance_accuracy": 0.25,
            "overall": {
                "top1_accuracy": float(np.mean(top1_values)) if top1_values else float("nan"),
                "mean_confidence": float(np.mean([row["confidence"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mean_elapsed_sec": float(np.mean([row["elapsed_sec"] for row in ordered_rows])) if ordered_rows else float("nan"),
            },
            "per_block": block_metrics,
            "per_item": ordered_rows,
        }
        write_json(variant_dir / "results.json", summary)
        return summary

    def run_axis2_variant(variant_id: str) -> dict[str, Any]:
        variant_dir = results_root / variant_id / "axis2"
        variant_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for item in axis2_items:
            candidates = []
            for row in item["candidates"]:
                label = variant_labels[variant_id][row["feature_key"]]
                candidates.append(
                    {
                        "candidate_code": row["candidate_code"],
                        "feature_key": row["feature_key"],
                        "canonical_label": label["canonical_label"],
                        "description": label["description"],
                    }
                )
            prompt_text = _axis2_prompt(candidates)
            out_json = variant_dir / f"{_slug(item['feature_key'])}__sample_{item['sample_id']}.json"
            tasks.append(
                {
                    "item": item,
                    "candidates": candidates,
                    "prompt_text": prompt_text,
                    "out_json": out_json,
                    "image_paths": [Path(row["image_path"]) for row in item["input_images"]],
                }
            )

        def worker(task: dict[str, Any]) -> dict[str, Any]:
            returncode, stdout_tail, stderr_tail, elapsed, output, forbidden_trace_hits = _run_codex_eval(
                schema_path=axis2_schema_path,
                out_json=task["out_json"],
                prompt_text=task["prompt_text"],
                images=list(task["image_paths"]),
                model=args.model,
                reasoning_effort=args.reasoning_effort,
            )
            item = task["item"]
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
                "feature_key": item["feature_key"],
                "block_idx": int(item["block_idx"]),
                "feature_id": int(item["feature_id"]),
                "sample_id": int(item["sample_id"]),
                "gold_code": gold_code,
                "best_candidate": str(output.get("best_candidate", "")).lower(),
                "ranked_candidates": ranked,
                "top1_correct": int(ranked[0] == gold_code),
                "reciprocal_rank": _reciprocal_rank(gold_code, ranked),
                "ndcg": ndcg_at_k(y_true, y_score, k=len(item["candidate_codes"])),
                "recall_at_3": recall_at_k(y_true, y_score, 3),
                "recall_at_5": recall_at_k(y_true, y_score, 5),
                "confidence": float(output.get("confidence", 0.0) or 0.0),
                "elapsed_sec": elapsed,
                "returncode": returncode,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "forbidden_trace_hits": forbidden_trace_hits,
                "output": output,
            }

        rows: list[dict[str, Any]] = []
        with cf.ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for idx, future in enumerate(cf.as_completed(futures), start=1):
                rows.append(future.result())
                if idx % 12 == 0 or idx == len(tasks):
                    print(f"[axis2 {variant_id} {idx:03d}/{len(tasks):03d}]", flush=True)

        row_by_key = {(row["feature_key"], int(row["sample_id"])): row for row in rows}
        ordered_rows = [row_by_key[(item["feature_key"], int(item["sample_id"]))] for item in axis2_items]
        block_metrics: dict[str, Any] = {}
        for block_idx in config.blocks:
            block_rows = [row for row in ordered_rows if int(row["block_idx"]) == int(block_idx)]
            block_metrics[str(block_idx)] = {
                "n_items": len(block_rows),
                "top1_accuracy": float(np.mean([row["top1_correct"] for row in block_rows])) if block_rows else float("nan"),
                "mrr": float(np.mean([row["reciprocal_rank"] for row in block_rows])) if block_rows else float("nan"),
                f"nDCG@{int(args.axis2_candidate_count)}": float(np.mean([row["ndcg"] for row in block_rows])) if block_rows else float("nan"),
            }
        summary = {
            "variant_id": variant_id,
            "axis": "axis2",
            "n_items": len(ordered_rows),
            "candidate_count": int(args.axis2_candidate_count),
            "chance_accuracy": 1.0 / float(args.axis2_candidate_count),
            "overall": {
                "top1_accuracy": float(np.mean([row["top1_correct"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mrr": float(np.mean([row["reciprocal_rank"] for row in ordered_rows])) if ordered_rows else float("nan"),
                f"nDCG@{int(args.axis2_candidate_count)}": float(np.mean([row["ndcg"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "Recall@3": float(np.mean([row["recall_at_3"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "Recall@5": float(np.mean([row["recall_at_5"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mean_confidence": float(np.mean([row["confidence"] for row in ordered_rows])) if ordered_rows else float("nan"),
                "mean_elapsed_sec": float(np.mean([row["elapsed_sec"] for row in ordered_rows])) if ordered_rows else float("nan"),
            },
            "per_block": block_metrics,
            "per_item": ordered_rows,
        }
        write_json(variant_dir / "results.json", summary)
        return summary

    variant_order = list(variant_ids)
    axis1_results = {variant_id: run_axis1_variant(variant_id) for variant_id in variant_order}
    axis2_results = {} if args.skip_axis2 else {variant_id: run_axis2_variant(variant_id) for variant_id in variant_order}

    def metric_line(axis_name: str, metric_key: str) -> list[str]:
        values = [f"{axis1_results[v]['overall'][metric_key]:.3f}" if axis_name == "axis1" else f"{axis2_results[v]['overall'][metric_key]:.3f}" for v in variant_order]
        return values

    report_lines = [
        "# Label Variant Axis Pilot",
        "",
        f"- Session: `{args.session_name}`",
        f"- Model: `{args.model}`",
        f"- Reasoning effort: `{args.reasoning_effort}`",
        f"- Feature universe: `{len(selected_feature_keys)}` shared features (`20 per block`) from the rerendered review set",
        (
            "- Axis 1 design: 4-way same-image token choice (`1 positive + 3 random negatives`)"
            if str(args.axis1_negative_mode) == "random"
            else "- Axis 1 design: 4-way same-image token choice (`1 positive + 3 high-cos low-act negatives`)"
        ),
        (
            f"- Axis 2 design: {int(args.axis2_candidate_count)}-way same-block label choice with prompt-independent activation confusers"
            if not args.skip_axis2
            else "- Axis 2 design: skipped"
        ),
        "- Evidence shown to the judge: original image + cyan-cross-marked token only",
        "- One held-out token per feature was used for this pilot",
        "",
        "## Overall Metrics",
        "",
        "| axis | metric | " + " | ".join(variant_order) + " |",
        "|---|---:|" + "|".join(["---:"] * len(variant_order)) + "|",
    ]
    for metric in ("top1_accuracy", "mean_confidence", "mean_elapsed_sec"):
        values = metric_line("axis1", metric)
        report_lines.append(f"| Axis 1 | {metric} | " + " | ".join(values) + " |")
    if not args.skip_axis2:
        for metric in ("top1_accuracy", "mrr", f"nDCG@{int(args.axis2_candidate_count)}", "Recall@3", "Recall@5", "mean_confidence", "mean_elapsed_sec"):
            values = metric_line("axis2", metric)
            report_lines.append(f"| Axis 2 | {metric} | " + " | ".join(values) + " |")

    if not args.skip_axis2:
        report_lines.extend(
            [
                "",
                "## Focus Features",
                "",
                "| feature | " + " | ".join(f"{variant_id} axis2 top1" for variant_id in variant_order) + " |",
                "|---|" + "|".join(["---:"] * len(variant_order)) + "|",
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
        axis2_by_variant_feature = {
            variant: {str(row["feature_key"]): row for row in axis2_results[variant]["per_item"]} for variant in variant_order
        }
        for key in focus_keys:
            if key not in axis2_by_variant_feature[variant_order[0]]:
                continue
            values = [str(axis2_by_variant_feature[variant][key]["top1_correct"]) for variant in variant_order]
            report_lines.append("| `{}` | {} |".format(key, " | ".join(values)))

    write_json(
        session_dir / "summary.json",
        {
            "session_name": args.session_name,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "axis1": axis1_results,
            "axis2": axis2_results,
            "selected_feature_keys": selected_feature_keys,
        },
    )
    (session_dir / "report.md").write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
