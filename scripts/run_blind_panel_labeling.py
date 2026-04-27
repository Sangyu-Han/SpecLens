from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autolabel_eval.isolated_codex import run_isolated_codex_exec


PANEL_KINDS = (
    "blind_erf_only_v1",
    "blind_erf_plain_v1",
    "blind_erf_cyan_dot_v1",
    "blind_erf_cyan_cross_v1",
    "blind_erf_cyan_dashed_box_v1",
    "blind_erf_locator_grid_v1",
    "blind_erf_patch_box_v1",
    "blind_erf_overview_zoom_v1",
    "blind_sae_only_v1",
)
PROMPT_STYLES = (
    "commonality_v2",
    "contextlite_v1",
    "isolated_erf_discrete_v1",
    "isolated_erf_stepdown_v1",
    "isolated_sae_stepdown_v1",
    "label_first_handoff_v1",
    "label_metric_where_v1",
    "label_shortdesc_where_v1",
    "label_shortdesc_where_rationale_v1",
    "label_then_handoff_v2",
    "two_stage_v1",
)
FORBIDDEN_CONTACT_SHEET_KEYS = (
    "contact_sheet",
    "positive_contact_sheet",
    "negative_contact_sheet",
    "sheet_path",
)


def _is_erf_panel_kind(panel_kind: str) -> bool:
    return str(panel_kind).strip().lower() in {
        "blind_erf_only_v1",
        "blind_erf_plain_v1",
        "blind_erf_cyan_dot_v1",
        "blind_erf_cyan_cross_v1",
        "blind_erf_cyan_dashed_box_v1",
        "blind_erf_locator_grid_v1",
        "blind_erf_patch_box_v1",
        "blind_erf_overview_zoom_v1",
    }


def _erf_visualization_description(panel_kind: str) -> str:
    panel_key = str(panel_kind).strip().lower()
    if panel_key == "blind_erf_only_v1":
        return (
            "Each provided image shows only the pixels inside the ERF support. "
            "Gray areas are hidden and provide no image information. "
            "The small green dot marks the objective patch whose evidence is being explained."
        )
    if panel_key == "blind_erf_plain_v1":
        return (
            "Each provided image shows only the pixels inside the ERF support. "
            "Gray areas are hidden and provide no image information. "
            "There is no separate objective-patch marker in these images."
        )
    if panel_key == "blind_erf_cyan_dot_v1":
        return (
            "Each provided image shows only the pixels inside the ERF support. "
            "Gray areas are hidden and provide no image information. "
            "A slightly translucent cyan dot marks the objective patch and is not semantic image content."
        )
    if panel_key == "blind_erf_cyan_cross_v1":
        return (
            "Each provided image shows only the pixels inside the ERF support. "
            "Gray areas are hidden and provide no image information. "
            "A small cyan cross marks the objective patch center and is not semantic image content."
        )
    if panel_key == "blind_erf_cyan_dashed_box_v1":
        return (
            "Each provided image shows only the pixels inside the ERF support. "
            "Gray areas are hidden and provide no image information. "
            "A thin translucent cyan dashed box marks the objective patch boundary and is not semantic image content."
        )
    if panel_key == "blind_erf_locator_grid_v1":
        return (
            "Each provided image is a composite. The large view shows only the pixels inside the ERF support; gray areas are hidden and provide no image information. "
            "A small abstract 14x14 locator grid shows the ERF support layout and the objective patch position. "
            "The locator grid contains no semantic image content."
        )
    if panel_key == "blind_erf_patch_box_v1":
        return (
            "Each provided image shows only the pixels inside the ERF support. "
            "Gray areas are hidden and provide no image information. "
            "A thin red box marks the objective patch boundary and is not semantic image content."
        )
    if panel_key == "blind_erf_overview_zoom_v1":
        return (
            "Each provided image is a composite with a full ERF-support view and an enlarged local zoom around the objective patch. "
            "Thin red boxes mark the same objective patch in the full view and the zoom. "
            "The boxes and the multi-view layout are not semantic image content."
        )
    raise KeyError(f"Unknown ERF panel kind: {panel_kind}")


def _erf_focus_clause(panel_kind: str) -> str:
    panel_key = str(panel_kind).strip().lower()
    if panel_key == "blind_erf_only_v1":
        return "Focus only on the shared visible evidence around the green-dot patch."
    if panel_key == "blind_erf_plain_v1":
        return (
            "Focus only on the shared visible evidence exposed by the ERF support. "
            "Do not invent a hidden objective location beyond what the visible support itself justifies."
        )
    if panel_key == "blind_erf_cyan_dot_v1":
        return "Focus on the shared visible evidence inside or immediately around the cyan-dot-marked patch."
    if panel_key == "blind_erf_cyan_cross_v1":
        return "Focus on the shared visible evidence inside or immediately around the cyan-cross-marked patch."
    if panel_key == "blind_erf_cyan_dashed_box_v1":
        return "Focus on the shared visible evidence inside or immediately around the cyan dashed box."
    if panel_key == "blind_erf_locator_grid_v1":
        return (
            "Use the large ERF view for semantic evidence. "
            "Use the locator grid only to understand where the objective patch sits within the support. "
            "Focus on the shared visible evidence near the implied objective location in the large ERF view."
        )
    if panel_key == "blind_erf_patch_box_v1":
        return "Focus on the shared visible evidence inside or immediately around the red-boxed patch."
    if panel_key == "blind_erf_overview_zoom_v1":
        return (
            "Focus on the shared visible evidence inside or immediately around the red-boxed patch. "
            "Use the zoom only to inspect local detail; do not let the zoom override the broader reusable concept if the overview repeatedly supports it."
        )
    raise KeyError(f"Unknown ERF panel kind: {panel_kind}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _assert_panel_only_examples(feature_key: str, examples: list[dict[str, Any]]) -> None:
    for example in examples:
        forbidden = sorted(set(example.keys()) & set(FORBIDDEN_CONTACT_SHEET_KEYS))
        if forbidden:
            raise ValueError(
                f"{feature_key}: contact-sheet fields are forbidden in blind panel labeling: {forbidden}"
            )


def _build_schema(prompt_style: str) -> dict[str, Any]:
    prompt_key = str(prompt_style).strip().lower()
    if prompt_key in ("isolated_erf_stepdown_v1", "isolated_erf_discrete_v1", "isolated_sae_stepdown_v1"):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "feature_key": {"type": "string"},
                "condition": {"type": "string"},
                "canonical_label": {"type": "string"},
                "support_summary": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "feature_key",
                "condition",
                "canonical_label",
                "support_summary",
                "rationale",
                "confidence",
            ],
        }
    if prompt_key == "label_first_handoff_v1":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "canonical_label": {"type": "string"},
                "rationale": {"type": "string"},
                "support_summary": {"type": "string"},
                "target_cue": {"type": "string"},
                "adjacent_context": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "canonical_label",
                "rationale",
                "confidence",
            ],
        }
    if prompt_key == "label_metric_where_v1":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "canonical_label": {"type": "string"},
                "description": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "canonical_label",
                "description",
                "rationale",
                "confidence",
            ],
        }
    if prompt_key == "label_shortdesc_where_v1":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "canonical_label": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": [
                "canonical_label",
                "description",
            ],
        }
    if prompt_key == "label_shortdesc_where_rationale_v1":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "canonical_label": {"type": "string"},
                "description": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": [
                "canonical_label",
                "description",
                "rationale",
            ],
        }
    if prompt_key == "label_then_handoff_v2":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "canonical_label": {"type": "string"},
                "rationale": {"type": "string"},
                "support_summary": {"type": "string"},
                "target_cue": {"type": "string"},
                "adjacent_context": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "canonical_label",
                "rationale",
                "confidence",
            ],
        }
    if prompt_key == "two_stage_v1":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "canonical_label": {"type": "string"},
                "rationale": {"type": "string"},
                "support_summary": {"type": "string"},
                "target_cue": {"type": "string"},
                "adjacent_context": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "canonical_label",
                "rationale",
                "confidence",
            ],
        }
    if prompt_key == "contextlite_v1":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "primary_locus": {"type": "string"},
                "adjacent_context": {"type": "string"},
                "support_summary": {"type": "string"},
                "canonical_label": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": [
                "primary_locus",
                "adjacent_context",
                "support_summary",
                "canonical_label",
                "rationale",
                "confidence",
            ],
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "detailed_description": {"type": "string"},
            "canonical_label": {"type": "string"},
            "rationale": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "detailed_description",
            "canonical_label",
            "rationale",
            "confidence",
        ],
    }


def _panel_spec(panel_kind: str) -> dict[str, str]:
    panel_key = str(panel_kind).strip().lower()
    if panel_key not in PANEL_KINDS:
        raise KeyError(f"Unknown panel kind: {panel_kind}")
    if panel_key == "blind_erf_only_v1":
        return {
            "asset_key": "feature_erf_on_original",
            "title_suffix": "Blind ERF-only",
            "caption": "ERF evidence only (support visible; gray hidden; green dot = objective token)",
        }
    if panel_key == "blind_erf_plain_v1":
        return {
            "asset_key": "feature_erf_plain",
            "title_suffix": "Blind ERF Plain",
            "caption": "ERF evidence only (support visible; gray hidden; no objective-patch marker)",
        }
    if panel_key == "blind_erf_cyan_dot_v1":
        return {
            "asset_key": "feature_erf_cyan_dot",
            "title_suffix": "Blind ERF + Cyan Dot",
            "caption": "ERF evidence only (support visible; gray hidden; translucent cyan dot marks objective patch)",
        }
    if panel_key == "blind_erf_cyan_cross_v1":
        return {
            "asset_key": "feature_erf_cyan_cross",
            "title_suffix": "Blind ERF + Cyan Cross",
            "caption": "ERF evidence only (support visible; gray hidden; cyan cross marks objective patch center)",
        }
    if panel_key == "blind_erf_cyan_dashed_box_v1":
        return {
            "asset_key": "feature_erf_cyan_dashed_box",
            "title_suffix": "Blind ERF + Cyan Dashed Box",
            "caption": "ERF evidence only (support visible; gray hidden; translucent cyan dashed box marks objective patch)",
        }
    if panel_key == "blind_erf_locator_grid_v1":
        return {
            "asset_key": "feature_erf_locator_grid",
            "title_suffix": "Blind ERF + Locator Grid",
            "caption": "ERF evidence only (support visible; gray hidden; side locator grid marks objective patch and support layout)",
        }
    if panel_key == "blind_erf_patch_box_v1":
        return {
            "asset_key": "feature_erf_patch_box",
            "title_suffix": "Blind ERF + Patch Box",
            "caption": "ERF evidence only (support visible; gray hidden; thin red box marks objective patch)",
        }
    if panel_key == "blind_erf_overview_zoom_v1":
        return {
            "asset_key": "feature_erf_overview_zoom",
            "title_suffix": "Blind ERF + Overview Zoom",
            "caption": "ERF evidence only (full support view plus local objective-patch zoom; red box marks objective patch)",
        }
    return {
        "asset_key": "sae_fire",
        "title_suffix": "Blind SAE-fire-only",
        "caption": "SAE firing evidence only (firing pixels visible; gray hidden)",
    }


def _build_prompt_text(panel_kind: str, prompt_style: str) -> str:
    panel_key = str(panel_kind).strip().lower()
    prompt_key = str(prompt_style).strip().lower()
    if prompt_key == "isolated_erf_stepdown_v1":
        if panel_key != "blind_erf_only_v1":
            raise KeyError(f"{prompt_style} only supports blind_erf_only_v1")
        return """You are labeling exactly one visual SAE feature from one isolated ERF-only image set. The feature id carries no semantics. Use only the visible pixels in the five images. Gray hidden regions are unknown and provide no evidence. Each image shows only pixels inside the feature-conditioned ERF support. The small green dot marks the objective token patch and is not semantic image content.

Task:
Choose the strongest reusable visual concept directly supported across the five images.

Rules:
- Prefer the highest-level category visibly justified across several images.
- If full object identity is not directly visible, step down to an object part, text/logo, material, texture, contour, or geometric structure.
- Do not use generic labels like patch, region, bright spot, dark area, object part, or rectangular patch unless nothing more specific is justified.
- Ignore masking boundaries, gray fill, borders, and the green dot.
- If the five images do not support one reusable concept, return `uninterpretable`.

Return JSON only:
{
  "feature_key": "<FEATURE_KEY>",
  "condition": "erf",
  "canonical_label": "string",
  "support_summary": "string",
  "rationale": "string",
  "confidence": 0.0
}
"""
    if prompt_key == "isolated_erf_discrete_v1":
        if panel_key != "blind_erf_only_v1":
            raise KeyError(f"{prompt_style} only supports blind_erf_only_v1")
        return """You are labeling exactly one visual SAE feature from one isolated ERF-only image set. The feature id carries no semantics. Use only the visible pixels in the five images. Gray hidden regions are unknown and provide no evidence. Each image shows only pixels inside the feature-conditioned ERF support. The small green dot marks the objective token patch and is not semantic image content.

Task:
Choose the strongest reusable visual concept directly supported across the five images.

Rules:
- Prefer the highest-level category visibly justified across several images.
- Across the five images, identify the most distinctive visual element that recurs across the set, even if it appears only as partial views in different images.
- When a repeated discrete mark, symbol, text fragment, logo-like graphic, or semantic part is visibly supported across multiple images, prefer labeling that recurring element rather than the broader carrier object or surface material.
- Use material, texture, contour, or surface-property labels when no repeated discrete element is visibly supported across the set.
- Do not use generic labels like patch, region, bright spot, dark area, object part, or rectangular patch unless nothing more specific is justified.
- Ignore masking boundaries, gray fill, borders, and the green dot.
- If the five images do not support one reusable concept, return `uninterpretable`.

Return JSON only:
{
  "feature_key": "<FEATURE_KEY>",
  "condition": "erf",
  "canonical_label": "string",
  "support_summary": "string",
  "rationale": "string",
  "confidence": 0.0
}
"""
    if prompt_key == "isolated_sae_stepdown_v1":
        if panel_key != "blind_sae_only_v1":
            raise KeyError(f"{prompt_style} only supports blind_sae_only_v1")
        return """You are labeling exactly one visual SAE feature from one isolated SAE-only image set. The feature id carries no semantics. Use only the visible pixels in the five images. Gray hidden regions are unknown and provide no evidence. Each image shows only pixels in strongly firing SAE patches.

Task:
Choose the strongest reusable visual concept directly supported across the five images.

Rules:
- Prefer the highest-level category visibly justified across several images.
- If full object identity is not directly visible, step down to an object part, text/logo, material, texture, contour, or geometric structure.
- Do not use generic labels like patch, region, bright spot, dark area, object part, or rectangular patch unless nothing more specific is justified.
- Ignore masking boundaries, gray fill, and borders.
- If the five images do not support one reusable concept, return `uninterpretable`.

Return JSON only:
{
  "feature_key": "<FEATURE_KEY>",
  "condition": "sae",
  "canonical_label": "string",
  "support_summary": "string",
  "rationale": "string",
  "confidence": 0.0
}
"""
    if prompt_key == "two_stage_v1":
        return (
            "This session uses a two-stage labeling process.\n\n"
            "Stage 1:\n"
            "- choose the best reusable canonical label from the images\n"
            "- explain briefly why that label is right\n\n"
            "Stage 2:\n"
            "- keep the canonical label fixed\n"
            "- generate optional axis handoff notes such as support_summary, target_cue, and adjacent_context\n"
            "- do not rewrite the canonical label during stage 2\n"
        )
    if panel_key == "blind_erf_only_v1" and prompt_key == "label_then_handoff_v2":
        return """You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

Each provided image shows only the pixels inside the ERF support. Gray areas are hidden and provide no image information. The small green dot marks the objective patch whose evidence is being explained.

Task:
Choose the best reusable feature label first. Only after the label is chosen should you produce optional token-level handoff notes for later evaluation.

Follow these stages in order:
Stage 1: Decide the canonical label.
- Pick the strongest reusable shared concept that is directly supported across the examples.
- Prefer a reusable object/material/text/structure category when it is genuinely repeated.
- Do not collapse the label to a tiny local cue if a broader reusable concept is repeatedly visible.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.

Stage 2: Explain why that label is right.
- Write `rationale` to justify the chosen canonical label.

Stage 3: Produce optional axis handoff notes.
- Without changing the canonical label, optionally fill any of:
  - `support_summary`: one short sentence about what is consistently visible
  - `target_cue`: the most useful local cue for token-level matching
  - `adjacent_context`: nearby context that repeatedly helps disambiguate the cue
- If an optional field is not clearly supported, leave it empty.
- These optional fields may be narrower than the canonical label, but they must not replace or rewrite it.

Additional rules:
- Focus only on the shared visible evidence around the green-dot patch.
- Ignore anything outside the visible ERF evidence.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- support_summary
- target_cue
- adjacent_context
- confidence
"""

    if panel_key == "blind_sae_only_v1" and prompt_key == "label_then_handoff_v2":
        return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Choose the best reusable feature label first. Only after the label is chosen should you produce optional token-level handoff notes for later evaluation.

Follow these stages in order:
Stage 1: Decide the canonical label.
- Pick the strongest reusable shared concept that is directly supported across the examples.
- Prefer a reusable object/material/text/structure category when it is genuinely repeated.
- Do not collapse the label to a tiny local cue if a broader reusable concept is repeatedly visible.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.

Stage 2: Explain why that label is right.
- Write `rationale` to justify the chosen canonical label.

Stage 3: Produce optional axis handoff notes.
- Without changing the canonical label, optionally fill any of:
  - `support_summary`: one short sentence about what is consistently visible
  - `target_cue`: the most useful local cue for token-level matching
  - `adjacent_context`: nearby context that repeatedly helps disambiguate the cue
- If an optional field is not clearly supported, leave it empty.
- These optional fields may be narrower than the canonical label, but they must not replace or rewrite it.

Additional rules:
- Focus only on the shared visible evidence exposed by the firing patches.
- Ignore anything outside the visible SAE evidence.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- support_summary
- target_cue
- adjacent_context
- confidence
"""

    if _is_erf_panel_kind(panel_key) and prompt_key == "label_first_handoff_v1":
        return f"""You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

{_erf_visualization_description(panel_key)}

Task:
Find the best reusable feature label first, then optionally produce short token-level handoff notes for later evaluation.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these ERF images.
- Keep that canonical label fixed. Do not narrow or rewrite it just to satisfy token-local explanation fields.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- After choosing the canonical label, explain briefly why that label is the best shared interpretation in `rationale`.
- Then optionally add any of the following only if they are genuinely useful and supported:
  - `support_summary`: one short sentence about what is consistently visible across examples
  - `target_cue`: the most useful local cue for token-level matching
  - `adjacent_context`: nearby context that repeatedly helps disambiguate the cue
- If one of those optional fields is not clearly supported, leave it empty rather than forcing it.
- {_erf_focus_clause(panel_key)}
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible ERF evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- support_summary
- target_cue
- adjacent_context
- confidence
"""

    if _is_erf_panel_kind(panel_key) and prompt_key == "label_metric_where_v1":
        return f"""You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

{_erf_visualization_description(panel_key)}

Task:
Choose the best reusable feature label first, then write a short metric description that summarizes the shared evidence and where this feature tends to activate.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these ERF images.
- Keep that canonical label fixed. Do not narrow or rewrite it while writing the description.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny local cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Then write `description` in 2 to 4 sentences.
- The description must summarize the common visible evidence across the five images.
- The description must also say where the feature tends to activate when that is supported by the images. This may be relative to an object or part, a boundary, interior versus exterior, background versus foreground, center versus edge, corner of frame, gap between paired objects, or another stable positional tendency.
- If the positional tendency is weak, inconsistent, or unavailable from the visible evidence, say that briefly rather than inventing it.
- The description may mention narrower local cues or activation loci, but it must not replace or rewrite the canonical label.
- Write `rationale` to explain briefly why the canonical label is still the best reusable label.
- {_erf_focus_clause(panel_key)}
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible ERF evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- description
- rationale
- confidence
"""

    if panel_key == "blind_sae_only_v1" and prompt_key == "label_first_handoff_v1":
        return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Find the best reusable feature label first, then optionally produce short token-level handoff notes for later evaluation.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these SAE-fire images.
- Keep that canonical label fixed. Do not narrow or rewrite it just to satisfy token-local explanation fields.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- After choosing the canonical label, explain briefly why that label is the best shared interpretation in `rationale`.
- Then optionally add any of the following only if they are genuinely useful and supported:
  - `support_summary`: one short sentence about what is consistently visible across examples
  - `target_cue`: the most useful local cue for token-level matching
  - `adjacent_context`: nearby context that repeatedly helps disambiguate the cue
- If one of those optional fields is not clearly supported, leave it empty rather than forcing it.
- Focus only on the shared visible evidence exposed by the firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible SAE evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- support_summary
- target_cue
- adjacent_context
- confidence
"""

    if panel_key == "blind_sae_only_v1" and prompt_key == "label_metric_where_v1":
        return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Choose the best reusable feature label first, then write a short metric description that summarizes the shared evidence and where this feature tends to activate.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these SAE-fire images.
- Keep that canonical label fixed. Do not narrow or rewrite it while writing the description.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny local cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Then write `description` in 2 to 4 sentences.
- The description must summarize the common visible evidence across the five images.
- The description must also say where the feature tends to activate when that is supported by the images. This may be relative to an object or part, a boundary, interior versus exterior, background versus foreground, center versus edge, corner of frame, gap between paired objects, or another stable positional tendency visible from the firing patches.
- If the positional tendency is weak, inconsistent, or unavailable from the visible evidence, say that briefly rather than inventing it.
- The description may mention narrower local cues or activation loci, but it must not replace or rewrite the canonical label.
- Write `rationale` to explain briefly why the canonical label is still the best reusable label.
- Focus only on the shared visible evidence exposed by the firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible SAE evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- description
- rationale
- confidence
"""

    if _is_erf_panel_kind(panel_key) and prompt_key == "label_shortdesc_where_v1":
        return f"""You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

{_erf_visualization_description(panel_key)}

Task:
Produce the final human-readable labeler output for this feature.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these ERF images.
- Keep that canonical label fixed.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Then write `description` in 1 to 3 short sentences.
- The description should help a human reviewer understand the shared visual pattern across the examples.
- Mention any recurring appearance, context, relation, relative placement, or shared objective-token position only if it is visibly part of the common pattern across examples.
- Keep the description short, natural, and readable by a human reviewer.
- Do not let the description rewrite, narrow, or replace the canonical label.
- {_erf_focus_clause(panel_key)}
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible ERF evidence.

Output only a single JSON object with keys:
- canonical_label
- description
"""

    if panel_key == "blind_sae_only_v1" and prompt_key == "label_shortdesc_where_v1":
        return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Produce the final human-readable labeler output for this feature.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these SAE-fire images.
- Keep that canonical label fixed.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Then write `description` in 1 to 3 short sentences.
- The description should help a human reviewer understand the shared visual pattern across the examples.
- Mention any recurring appearance, context, relation, relative placement, or shared objective-token position only if it is visibly part of the common pattern across examples.
- Keep the description short, natural, and readable by a human reviewer.
- Do not let the description rewrite, narrow, or replace the canonical label.
- Focus only on the shared visible evidence exposed by the firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible SAE evidence.

Output only a single JSON object with keys:
- canonical_label
- description
"""

    if _is_erf_panel_kind(panel_key) and prompt_key == "label_shortdesc_where_rationale_v1":
        return f"""You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

{_erf_visualization_description(panel_key)}

Task:
Produce the final human-readable labeler output for this feature.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these ERF images.
- Keep that canonical label fixed.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Then write `description` as exactly one sentence.
- The description should add one short piece of useful information that helps a person understand the feature.
- When supported, include where the feature tends to activate: for example on a part, on a boundary, inside a gap, near the background, at a frame edge, or another stable positional tendency.
- If no stable positional tendency is visible, omit location rather than inventing it.
- Keep the description short, natural, and readable by a human reviewer.
- Do not let the description rewrite, narrow, or replace the canonical label.
- Then write a short `rationale` of one or two sentences explaining why the canonical label is the best reusable label.
- The rationale may mention uncertainty or why nearby alternatives are worse, but it must not rewrite the label.
- {_erf_focus_clause(panel_key)}
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible ERF evidence.

Output only a single JSON object with keys:
- canonical_label
- description
- rationale
"""

    if panel_key == "blind_sae_only_v1" and prompt_key == "label_shortdesc_where_rationale_v1":
        return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Produce the final human-readable labeler output for this feature.

Instructions:
- First decide the best reusable `canonical_label` for the shared concept across these SAE-fire images.
- Keep that canonical label fixed.
- Prefer the strongest reusable shared concept that is directly supported across examples.
- If a broader concept is genuinely repeated and visually supported, keep it. Do not collapse it into an overly tiny cue unless the broader concept is not actually stable.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Then write `description` as exactly one sentence.
- The description should add one short piece of useful information that helps a person understand the feature.
- When supported, include where the feature tends to activate: for example on a part, on a boundary, inside a gap, near the background, at a frame edge, or another stable positional tendency visible from the firing patches.
- If no stable positional tendency is visible, omit location rather than inventing it.
- Keep the description short, natural, and readable by a human reviewer.
- Do not let the description rewrite, narrow, or replace the canonical label.
- Then write a short `rationale` of one or two sentences explaining why the canonical label is the best reusable label.
- The rationale may mention uncertainty or why nearby alternatives are worse, but it must not rewrite the label.
- Focus only on the shared visible evidence exposed by the firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible SAE evidence.

Output only a single JSON object with keys:
- canonical_label
- description
- rationale
"""

    if panel_key == "blind_erf_only_v1" and prompt_key == "contextlite_v1":
        return """You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

Each provided image shows only the pixels inside the ERF support. Gray areas are hidden and provide no image information. The small green dot marks the objective patch whose evidence is being explained.

Task:
Across these ERF images, identify the most concrete recurring visual commonality without letting the label collapse into an over-specific hallucinated object.

Instructions:
- First, identify the smallest recurring visible thing centered on or immediately around the green-dot patch. Put that in `primary_locus`.
- Then note only the adjacent context that repeatedly seems necessary to disambiguate the locus. Put that in `adjacent_context`.
- Then write a one-sentence `support_summary` describing what is consistently visible across examples.
- Only after that, choose the best short `canonical_label`.
- Keep the canonical label conservative and reusable. If a lower-level structure, material, contour, boundary, or texture is more stable than a whole object identity, prefer the lower-level label.
- Do not upgrade a partial tip, spot, contour, or fragment into a specific object category unless that object identity is directly visible in multiple examples.
- Use `primary_locus` and `adjacent_context` to hold extra detail instead of overloading the canonical label.
- Focus only on the shared visible evidence around the green-dot patch.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible ERF evidence.
- If the evidence is weak or inconsistent, say so in the rationale.

Output only a single JSON object with keys:
- primary_locus
- adjacent_context
- support_summary
- canonical_label
- rationale
- confidence
"""

    if panel_key == "blind_sae_only_v1" and prompt_key == "contextlite_v1":
        return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Across these SAE-fire images, identify the most concrete recurring visual commonality without letting the label collapse into an over-specific hallucinated object.

Instructions:
- First, identify the smallest recurring visible thing exposed by the firing patches. Put that in `primary_locus`.
- Then note only the adjacent context that repeatedly seems necessary to disambiguate the locus. Put that in `adjacent_context`.
- Then write a one-sentence `support_summary` describing what is consistently visible across examples.
- Only after that, choose the best short `canonical_label`.
- Keep the canonical label conservative and reusable. If a lower-level structure, material, contour, boundary, or texture is more stable than a whole object identity, prefer the lower-level label.
- Do not upgrade a partial tip, spot, contour, or fragment into a specific object category unless that object identity is directly visible in multiple examples.
- Use `primary_locus` and `adjacent_context` to hold extra detail instead of overloading the canonical label.
- Focus only on the shared visible evidence exposed by the firing patches.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- Ignore anything outside the visible SAE evidence.
- If the evidence is weak or inconsistent, say so in the rationale.

Output only a single JSON object with keys:
- primary_locus
- adjacent_context
- support_summary
- canonical_label
- rationale
- confidence
"""

    if _is_erf_panel_kind(panel_key):
        return f"""You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

{_erf_visualization_description(panel_key)}

Task:
Across these ERF images, identify the most concrete recurring visual commonality.

Instructions:
- First, form a detailed explanation of what tends to be visible around the objective patch across examples: local structure, material/part, typical surrounding context, and what kinds of activations seem included or excluded.
- Then compress that explanation into the best short canonical label.
- {_erf_focus_clause(panel_key)}
- Name the repeated visual concept in your own words.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- If the ERF repeatedly exposes a more specific object, material, texture, or structure, prefer that.
- Ignore anything outside the visible ERF evidence.
- If the evidence is weak or inconsistent, say so in the rationale.

Output only a single JSON object with keys:
- detailed_description
- canonical_label
- rationale
- confidence
"""

    return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Task:
Across these SAE-fire images, identify the most concrete recurring visual commonality.

Instructions:
- First, form a detailed explanation of what tends to be visible inside the firing patches across examples: local structure, material/part, typical surrounding context, and what kinds of activations seem included or excluded.
- Then compress that explanation into the best short canonical label.
- Focus only on the shared visible evidence exposed by the firing patches.
- Name the repeated visual concept in your own words.
- Do not use generic labels like patch, region, area, object part, bright spot, or rectangular patch unless that is truly all that is shared.
- If the firing patches repeatedly expose a more specific object, material, texture, or structure, prefer that.
- Ignore anything outside the visible SAE evidence.
- If the evidence is weak or inconsistent, say so in the rationale.

Output only a single JSON object with keys:
- detailed_description
- canonical_label
- rationale
- confidence
"""


def _build_two_stage_stage1_prompt(panel_kind: str) -> str:
    panel_key = str(panel_kind).strip().lower()
    if panel_key == "blind_erf_only_v1":
        return """You are auditing one visual SAE feature using only feature-conditioned ERF evidence.

Each provided image shows only the pixels inside the ERF support. Gray areas are hidden and provide no image information. The small green dot marks the objective patch whose evidence is being explained.

Stage 1 task:
Choose the best reusable canonical label for the shared concept across these ERF images.

Instructions:
- Pick the strongest reusable shared concept that is directly supported across the examples.
- Prefer a reusable object/material/text/structure category when it is genuinely repeated.
- Do not collapse the label to a tiny local cue if a broader reusable concept is repeatedly visible.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Focus only on the shared visible evidence around the green-dot patch.
- Ignore anything outside the visible ERF evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- confidence
"""

    return """You are auditing one visual SAE feature using only SAE activation evidence.

Each provided image shows only the pixels in strongly firing feature patches. Gray areas are hidden and provide no image information.

Stage 1 task:
Choose the best reusable canonical label for the shared concept across these SAE-fire images.

Instructions:
- Pick the strongest reusable shared concept that is directly supported across the examples.
- Prefer a reusable object/material/text/structure category when it is genuinely repeated.
- Do not collapse the label to a tiny local cue if a broader reusable concept is repeatedly visible.
- Do not hallucinate a specific object identity from a partial fragment unless that identity is directly visible in multiple examples.
- Focus only on the shared visible evidence exposed by the firing patches.
- Ignore anything outside the visible SAE evidence.
- If the evidence is weak or inconsistent, say so in the rationale and lower confidence.

Output only a single JSON object with keys:
- canonical_label
- rationale
- confidence
"""


def _build_two_stage_stage2_prompt(panel_kind: str, canonical_label: str, rationale: str) -> str:
    panel_key = str(panel_kind).strip().lower()
    evidence_desc = (
        "feature-conditioned ERF evidence around the green-dot patch"
        if panel_key == "blind_erf_only_v1"
        else "SAE activation evidence exposed by the firing patches"
    )
    return f"""You are auditing one visual SAE feature using only {evidence_desc}.

The canonical label has already been chosen and is frozen.

Frozen canonical label: {canonical_label}
Stage-1 rationale: {rationale}

Stage 2 task:
Generate optional token-level handoff notes for later evaluation without changing the canonical label.

Instructions:
- Keep the canonical label fixed. Do not revise, narrow, or rewrite it.
- Optionally fill any of:
  - support_summary: one short sentence about what is consistently visible
  - target_cue: the most useful local cue for token-level matching
  - adjacent_context: nearby context that repeatedly helps disambiguate the cue
- If an optional field is not clearly supported, leave it as an empty string.
- These optional fields may be narrower than the canonical label, but they must not replace or redefine it.
- Focus only on the visible evidence in the provided images.

Output only a single JSON object with keys:
- support_summary
- target_cue
- adjacent_context
"""


def _build_review_html(out_path: Path, raw_payload: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind Panel Review</title>
  <style>
    body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:#f6f2ea; color:#231f1a; }}
    .page {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero, .feature {{ background:#fffdf9; border:1px solid #e1d8cb; border-radius:16px; padding:18px; margin-bottom:18px; box-shadow:0 8px 24px rgba(0,0,0,0.04); }}
    .meta {{ color:#6b645d; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; margin-top:14px; }}
    .panel {{ border:1px solid #ddd3c6; border-radius:12px; background:#fff; padding:10px; }}
    .panel img {{ width:100%; border-radius:10px; border:1px solid #ddd3c6; background:#f0ebe2; }}
    .pred {{ margin-top:16px; border:1px solid #ece3d6; border-radius:12px; padding:14px; background:#fff; }}
    .label {{ font-size:22px; font-weight:700; }}
    .k {{ font-weight:700; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Blind Panel Label Review</h1>
      <div class="meta">Session: {raw_payload['session_name']} | Prompt: {raw_payload['panel_kind']} | Model: {raw_payload['model']} | Reasoning: {raw_payload['reasoning_effort']}</div>
    </div>
"""
    for row in raw_payload["features"]:
        out = row.get("output") or {}
        panels_html = "".join(
            f"""
      <div class="panel">
        <div class="meta">Example {int(example["rank"]) + 1}</div>
        <img src="{example['review_image']}" alt="panel">
      </div>
"""
            for example in row.get("label_examples", [])
        )
        field_order = (
            "description",
            "primary_locus",
            "adjacent_context",
            "support_summary",
            "target_cue",
            "detailed_description",
            "rationale",
            "confidence",
        )
        field_html = "".join(
            f'<div><span class="k">{key}:</span> {out.get(key, "")}</div>'
            for key in field_order
            if key in out
        )
        html += f"""
    <div class="feature">
      <div class="meta">{row['feature_key']}</div>
      <div class="grid">{panels_html}</div>
      <div class="pred">
        <div class="label">{out.get('canonical_label', '')}</div>
        {field_html}
        <div class="meta">elapsed: {row['elapsed_sec']:.1f}s</div>
      </div>
    </div>
"""
    html += """
  </div>
</body>
</html>
"""
    out_path.write_text(html)


def _run_condition(
    *,
    workspace_root: Path,
    source_workspace_root: Path,
    source_session: str,
    session_name: str,
    panel_kind: str,
    model: str,
    reasoning_effort: str,
    prompt_label: str,
    prompt_style: str,
    jobs: int,
    limit: int,
    feature_keys: list[str],
) -> Path:
    review_root = workspace_root / "outputs" / "review_sessions"
    source_review_root = source_workspace_root / "outputs" / "review_sessions"
    source_session_dir = source_review_root / source_session
    source_manifest = json.loads((source_session_dir / "selection_manifest.json").read_text())
    features = list(source_manifest["features"])
    if feature_keys:
        wanted = {str(key).strip() for key in feature_keys if str(key).strip()}
        features = [feature for feature in features if str(feature["feature_key"]) in wanted]
    if limit > 0:
        features = features[:limit]

    panel_spec = _panel_spec(panel_kind)
    asset_key = str(panel_spec["asset_key"])
    prompt_text = _build_prompt_text(panel_kind, prompt_style)

    new_session_dir = review_root / session_name
    new_session_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = new_session_dir / f"{prompt_label}.md"
    schema_path = new_session_dir / "output_schema.json"
    prompt_path.write_text(prompt_text)
    _write_json(schema_path, _build_schema(prompt_style))

    rendered_features: list[dict[str, Any]] = []
    for feature in features:
        feature_key = str(feature["feature_key"])
        source_examples = list(feature["label_examples"])
        _assert_panel_only_examples(feature_key, source_examples)
        rendered_examples: list[dict[str, Any]] = []
        for example in source_examples:
            rel_path = str(example[asset_key])
            rendered_examples.append(
                {
                    "rank": int(example["rank"]),
                    "sample_id": int(example["sample_id"]),
                    "token_idx": int(example["token_idx"]),
                    asset_key: rel_path,
                    "source_image": str((source_session_dir / rel_path).resolve()),
                    "review_image": os.path.relpath(source_session_dir / rel_path, start=new_session_dir),
                    "staged_name": f"example_{int(example['rank']):02d}{Path(rel_path).suffix.lower() or '.png'}",
                }
            )
        rendered_features.append(
            {
                "feature_key": feature_key,
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "selection_stats": dict(feature.get("selection_stats", {})),
                "panel_caption": panel_spec["caption"],
                "panel_title_suffix": panel_spec["title_suffix"],
                "label_examples": rendered_examples,
            }
        )

    selection_manifest = {
        "session_name": session_name,
        "panel_kind": panel_kind,
        "prompt": {
            "prompt_path": str(prompt_path),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_label": prompt_label,
            "prompt_style": prompt_style,
            "generation_mode": "isolated_codex_exec_per_feature",
            "isolation_mode": "repo_external_temp_workspace",
        },
        "source_session": source_session,
        "source_workspace_root": str(source_workspace_root),
        "features": rendered_features,
    }
    _write_json(new_session_dir / "selection_manifest.json", selection_manifest)

    predictions_dir = new_session_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    batch_start = time.time()

    def _label_feature(feature: dict[str, Any]) -> dict[str, Any]:
        feature_key = str(feature["feature_key"])
        feature_stem = _slug(feature_key)
        _assert_panel_only_examples(feature_key, list(feature["label_examples"]))
        images = [Path(example["source_image"]) for example in sorted(feature["label_examples"], key=lambda row: int(row["rank"]))]
        prompt_style_key = str(prompt_style).strip().lower()
        feature_prompt_text = str(prompt_text).replace("<FEATURE_KEY>", feature_key)
        if prompt_style_key == "two_stage_v1":
            stage1 = run_isolated_codex_exec(
                artifact_dir=predictions_dir,
                artifact_stem=f"{feature_stem}__stage1",
                prompt_text=_build_two_stage_stage1_prompt(panel_kind),
                schema=None,
                images=images,
                model=model,
                reasoning_effort=reasoning_effort,
                temp_prefix="blind_label_stage1_",
            )
            stage1_output = dict(stage1.get("output") or {})
            stage2 = run_isolated_codex_exec(
                artifact_dir=predictions_dir,
                artifact_stem=f"{feature_stem}__stage2",
                prompt_text=_build_two_stage_stage2_prompt(
                    panel_kind,
                    str(stage1_output.get("canonical_label", "")).strip(),
                    str(stage1_output.get("rationale", "")).strip(),
                ),
                schema=None,
                images=images,
                model=model,
                reasoning_effort=reasoning_effort,
                temp_prefix="blind_label_stage2_",
            )
            stage2_output = dict(stage2.get("output") or {})
            combined_output = {
                "canonical_label": str(stage1_output.get("canonical_label", "")).strip(),
                "rationale": str(stage1_output.get("rationale", "")).strip(),
                "support_summary": str(stage2_output.get("support_summary", "")).strip(),
                "target_cue": str(stage2_output.get("target_cue", "")).strip(),
                "adjacent_context": str(stage2_output.get("adjacent_context", "")).strip(),
                "confidence": stage1_output.get("confidence", ""),
            }
            return {
                "feature_key": feature_key,
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "label_examples": [
                    {
                        "rank": int(example["rank"]),
                        "sample_id": int(example["sample_id"]),
                        "token_idx": int(example["token_idx"]),
                        "review_image": str(example["review_image"]),
                        "staged_name": str(example["staged_name"]),
                    }
                    for example in sorted(feature["label_examples"], key=lambda row: int(row["rank"]))
                ],
                "elapsed_sec": float(stage1["elapsed_sec"]) + float(stage2["elapsed_sec"]),
                "returncode": max(int(stage1["returncode"]), int(stage2["returncode"])),
                "stdout_tail": str(stage2["stdout_tail"]),
                "stderr_tail": str(stage2["stderr_tail"]),
                "staged_inputs": list(stage1["staged_inputs"]),
                "forbidden_trace_hits": sorted(
                    set(list(stage1["forbidden_trace_hits"]) + list(stage2["forbidden_trace_hits"]))
                ),
                "stage1_output": stage1_output,
                "stage2_output": stage2_output,
                "stage1_elapsed_sec": float(stage1["elapsed_sec"]),
                "stage2_elapsed_sec": float(stage2["elapsed_sec"]),
                "stage1_stdout_tail": str(stage1["stdout_tail"]),
                "stage1_stderr_tail": str(stage1["stderr_tail"]),
                "stage2_stdout_tail": str(stage2["stdout_tail"]),
                "stage2_stderr_tail": str(stage2["stderr_tail"]),
                "output": combined_output,
            }

        result = run_isolated_codex_exec(
            artifact_dir=predictions_dir,
            artifact_stem=feature_stem,
            prompt_text=feature_prompt_text,
            schema=None,
            images=images,
            model=model,
            reasoning_effort=reasoning_effort,
            temp_prefix="blind_label_",
        )
        return {
            "feature_key": feature_key,
            "block_idx": int(feature["block_idx"]),
            "feature_id": int(feature["feature_id"]),
            "label_examples": [
                {
                    "rank": int(example["rank"]),
                    "sample_id": int(example["sample_id"]),
                    "token_idx": int(example["token_idx"]),
                    "review_image": str(example["review_image"]),
                    "staged_name": str(example["staged_name"]),
                }
                for example in sorted(feature["label_examples"], key=lambda row: int(row["rank"]))
            ],
            "elapsed_sec": float(result["elapsed_sec"]),
            "returncode": int(result["returncode"]),
            "stdout_tail": str(result["stdout_tail"]),
            "stderr_tail": str(result["stderr_tail"]),
            "staged_inputs": list(result["staged_inputs"]),
            "forbidden_trace_hits": list(result["forbidden_trace_hits"]),
            "output": dict(result["output"]),
        }

    raw_features: list[dict[str, Any]] = []
    jobs = max(1, int(jobs))
    if jobs == 1:
        for idx, feature in enumerate(rendered_features, start=1):
            raw_features.append(_label_feature(feature))
            if idx % 10 == 0 or idx == len(rendered_features):
                print(f"[{panel_kind} {idx:03d}/{len(rendered_features):03d}]", flush=True)
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_label_feature, feature) for feature in rendered_features]
            for idx, future in enumerate(cf.as_completed(futures), start=1):
                raw_features.append(future.result())
                if idx % 10 == 0 or idx == len(rendered_features):
                    print(f"[{panel_kind} {idx:03d}/{len(rendered_features):03d}]", flush=True)
        raw_features.sort(key=lambda row: str(row["feature_key"]))

    raw_payload = {
        "session_name": session_name,
        "panel_kind": panel_kind,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_label": prompt_label,
        "prompt_style": prompt_style,
        "prompt_path": str(prompt_path),
        "source_session": source_session,
        "features": raw_features,
        "batch_elapsed_sec": time.time() - batch_start,
    }
    raw_path = new_session_dir / "raw_predictions.json"
    _write_json(raw_path, raw_payload)
    _build_review_html(new_session_dir / "label_team_review.html", raw_payload)
    _write_json(
        new_session_dir / "review_summary.json",
        {
            "session_name": session_name,
            "panel_kind": panel_kind,
            "selection_manifest_json": str(new_session_dir / "selection_manifest.json"),
            "raw_predictions_json": str(raw_path),
            "review_html": str(new_session_dir / "label_team_review.html"),
            "prompt_path": str(prompt_path),
            "prompt_style": prompt_style,
            "source_session": source_session,
        },
    )
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--source-workspace-root", default="")
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--panel-kind", required=True, choices=PANEL_KINDS)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--prompt-label", default="blind_panel_prompt")
    parser.add_argument("--prompt-style", default="commonality_v2", choices=PROMPT_STYLES)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--feature-key", action="append", default=[])
    args = parser.parse_args()

    _run_condition(
        workspace_root=Path(args.workspace_root),
        source_workspace_root=Path(args.source_workspace_root) if str(args.source_workspace_root).strip() else Path(args.workspace_root),
        source_session=str(args.source_session),
        session_name=str(args.session_name),
        panel_kind=str(args.panel_kind),
        model=str(args.model),
        reasoning_effort=str(args.reasoning_effort),
        prompt_label=str(args.prompt_label),
        prompt_style=str(args.prompt_style),
        jobs=int(args.jobs),
        limit=int(args.limit),
        feature_keys=[str(v) for v in list(args.feature_key)],
    )


if __name__ == "__main__":
    main()
