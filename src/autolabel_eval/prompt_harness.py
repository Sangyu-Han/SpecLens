from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .autolabel_loop import _default_prompt_config, _render_student_prompt
from .config import EvalConfig
from .utils import write_json


PROMPT_TEMPLATES_DIR = Path(__file__).resolve().parent / "prompt_templates"


def build_real_clip_harness_config(
    *,
    harness_root: Path,
    train_examples_per_feature: int = 5,
    holdout_examples_per_feature: int = 5,
    features_per_block: int = 64,
) -> EvalConfig:
    harness_root = Path(harness_root)
    config = replace(
        EvalConfig(),
        workspace_root=harness_root,
        model_name="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        features_per_block=int(features_per_block),
        train_examples_per_feature=int(train_examples_per_feature),
        holdout_examples_per_feature=int(holdout_examples_per_feature),
        shuffle_feature_candidates=False,
        deciles_root_override=Path("/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_index/deciles"),
        checkpoints_root_override=Path("/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_sae"),
        checkpoint_relpath_template="model.blocks.{block_idx}/step_0050000_tokens_204800000.pt",
        dataset_root_override=Path("/data/datasets/imagenet/val"),
    )
    config.ensure_dirs()
    return config


def default_pge_student_prompt() -> str:
    base = _render_student_prompt(_default_prompt_config())
    preamble = """You are the generator/student labeler in a planner-generator-evaluator harness.

Rules:
- Stay general: do not rely on feature IDs, block IDs, or one-off memorized examples.
- Treat the feature act map as localization evidence only; cyan/teal is an overlay, never semantic image content.
- Use original image plus act map to identify the broad recurring carrier first.
- Use feature-conditioned ERF only to refine that carrier, not to replace it with a tiny patch unless the broader carrier clearly fails.
- If ERF is omitted in this condition, finalize from the broad carrier and visible local evidence.
- Keep the canonical label short and reusable.
- Do not mention ERF, heatmap, activation, positives, or examples in canonical_label or support_summary.

In addition to the normal label fields, expose three short visible rationale fields:
- carrier_draft: the broad recurring carrier hypothesis before ERF refinement
- erf_refinement: how ERF changed or constrained the draft
- why_not_broader: why a broader host/scene/object label was rejected

Return only JSON matching the schema.

"""
    return preamble + base + "\n"


def default_openai_vision_noerf_prompt() -> str:
    return (PROMPT_TEMPLATES_DIR / "openai_vision_noerf_v0.md").read_text()


def builtin_harness_prompt_text(prompt_family: str) -> str:
    key = str(prompt_family).strip().lower()
    if key in ("carrier_first_pge_v0", "carrier_first", "default"):
        return default_pge_student_prompt()
    if key == "openai_vision_noerf_v0":
        return default_openai_vision_noerf_prompt()
    raise KeyError(f"Unknown harness prompt family: {prompt_family}")


def pge_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primary_locus": {"type": "string", "maxLength": 220},
            "adjacent_context": {"type": "string", "maxLength": 220},
            "canonical_label": {"type": "string", "maxLength": 80},
            "support_summary": {"type": "string", "maxLength": 140},
            "description": {"type": "string", "maxLength": 260},
            "notes": {"type": "string", "maxLength": 500},
            "carrier_draft": {"type": "string", "maxLength": 180},
            "erf_refinement": {"type": "string", "maxLength": 220},
            "why_not_broader": {"type": "string", "maxLength": 220},
            "confidence": {"type": "number"},
        },
        "required": [
            "primary_locus",
            "adjacent_context",
            "canonical_label",
            "support_summary",
            "description",
            "notes",
            "carrier_draft",
            "erf_refinement",
            "why_not_broader",
            "confidence",
        ],
    }


def build_label_examples_from_feature(feature: dict[str, Any]) -> list[dict[str, Any]]:
    train_rows = list(feature.get("train", []))
    examples: list[dict[str, Any]] = []
    for idx, row in enumerate(train_rows, start=1):
        examples.append(
            {
                "rank": int(idx),
                "sample_id": int(row["sample_id"]),
                "token_idx": int(row["target_patch_idx"]),
            }
        )
    return examples


def write_loop_feature_manifest(path: Path, features: list[dict[str, Any]], *, tag: str) -> None:
    payload = {
        "tag": str(tag),
        "features": [
            {
                "feature_key": str(feature["feature_key"]),
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "selection_stats": dict(feature.get("selection_stats") or {}),
                "label_examples": build_label_examples_from_feature(feature),
            }
            for feature in features
        ],
    }
    write_json(path, payload)
