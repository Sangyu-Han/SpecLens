from __future__ import annotations

import html
import json
import os
import random
import shutil
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import EvalConfig
from .feature_bank import load_feature_bank
from .legacy import LegacyRuntime
from .rendering import (
    save_erf_heatmap_image,
    save_feature_actmap_overlay,
    save_original_with_token_box,
    save_support_detail_crop_image,
    save_support_mask_image,
    save_support_outline_crop_image,
)
from .study_protocol import (
    _collect_label_examples,
    _feature_lookup,
    _norm_text,
    _safe_float,
    _select_features_for_label_session,
)
from .utils import read_json, read_jsonl, write_json, write_jsonl


META_TAGS: tuple[str, ...] = (
    "cls_feature",
    "massive_activation",
    "high_frequency_texture",
    "positional_or_border_bias",
    "global_diffuse",
    "likely_uninterpretable",
)

PHASE_PROMPT_STABILIZATION = "prompt_stabilization"
PHASE_MODEL_SLIMMING = "model_slimming"
PHASE_AXIS_GUIDED = "axis_guided_prompt_improvement"
PHASE_GATE_PROMOTION = "axis_guided_prompt_update_promotion"

PHASE_ORDER: tuple[str, ...] = (
    PHASE_PROMPT_STABILIZATION,
    PHASE_MODEL_SLIMMING,
    PHASE_AXIS_GUIDED,
)

PHASE_LABELS: dict[str, str] = {
    PHASE_PROMPT_STABILIZATION: "Phase 1: Prompt Stabilization",
    PHASE_MODEL_SLIMMING: "Phase 2: Student Model Slimming",
    PHASE_AXIS_GUIDED: "Phase 3: Axis 1/2-Guided Prompt Improvement",
    PHASE_GATE_PROMOTION: "Axis-Guided Prompt Promotion",
}

PHASE_CANDIDATE_NEXT: dict[str, str] = {
    PHASE_PROMPT_STABILIZATION: PHASE_MODEL_SLIMMING,
    PHASE_MODEL_SLIMMING: PHASE_AXIS_GUIDED,
    PHASE_AXIS_GUIDED: PHASE_GATE_PROMOTION,
}

PHASE_GATE_DECISIONS: dict[str, tuple[str, ...]] = {
    PHASE_PROMPT_STABILIZATION: ("advance_to_phase_2", "stay_in_phase_1"),
    PHASE_MODEL_SLIMMING: (
        "accept_lighter_model_and_advance_to_phase_3",
        "keep_stronger_model_and_advance_to_phase_3",
        "stay_in_phase_2",
    ),
    PHASE_AXIS_GUIDED: ("accept_axis_guided_prompt_update", "reject_and_continue_axis_phase"),
}

PLATEAU_ACCEPT_DELTA_MAX = 0.05
DOMINANT_FAILURE_SHARE_MAX = 0.30
MODEL_SMALL_DROP_MAX = 0.05
MODEL_FAILURE_TAG_MULTIPLIER_MAX = 2.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _autolabel_session_dir(config: EvalConfig, session_name: str) -> Path:
    return config.autolabel_root / session_name


def _round_dir(config: EvalConfig, session_name: str, round_index: int) -> Path:
    return _autolabel_session_dir(config, session_name) / "rounds" / f"round_{int(round_index):03d}"


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _relpath(base_dir: Path, path: Path) -> str:
    return os.path.relpath(path, start=base_dir)


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return read_json(path)


def _load_session_manifest(config: EvalConfig, session_name: str) -> dict[str, Any]:
    path = _autolabel_session_dir(config, session_name) / "session_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing autolabel session manifest at {path}")
    return _with_session_defaults(read_json(path))


def _write_session_manifest(config: EvalConfig, session_name: str, manifest: dict[str, Any]) -> None:
    write_json(_autolabel_session_dir(config, session_name) / "session_manifest.json", _with_session_defaults(manifest))


def _load_feature_pool(config: EvalConfig, session_name: str) -> dict[str, Any]:
    path = _autolabel_session_dir(config, session_name) / "feature_pool.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature pool at {path}")
    return read_json(path)


def _load_current_prompt_config(config: EvalConfig, session_name: str) -> dict[str, Any]:
    path = _autolabel_session_dir(config, session_name) / "current_prompt_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing current prompt config at {path}")
    return read_json(path)


def _write_current_prompt_config(config: EvalConfig, session_name: str, payload: dict[str, Any]) -> None:
    write_json(_autolabel_session_dir(config, session_name) / "current_prompt_config.json", payload)


def _load_prompt_history(config: EvalConfig, session_name: str) -> list[dict[str, Any]]:
    path = _autolabel_session_dir(config, session_name) / "prompt_history.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def _write_prompt_history(config: EvalConfig, session_name: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(_autolabel_session_dir(config, session_name) / "prompt_history.jsonl", rows)


def _load_feature_state(config: EvalConfig, session_name: str) -> list[dict[str, Any]]:
    path = _autolabel_session_dir(config, session_name) / "feature_state.jsonl"
    if not path.exists():
        return []
    return [_normalize_feature_state_row(row) for row in read_jsonl(path)]


def _write_feature_state(config: EvalConfig, session_name: str, rows: list[dict[str, Any]]) -> None:
    write_jsonl(
        _autolabel_session_dir(config, session_name) / "feature_state.jsonl",
        [_normalize_feature_state_row(row) for row in rows],
    )


def _feature_state_map(config: EvalConfig, session_name: str) -> dict[str, dict[str, Any]]:
    return {str(row["feature_key"]): row for row in _load_feature_state(config, session_name)}


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _session_track_defaults(session_name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    selection = dict(manifest.get("selection_diagnostics") or {})
    focus_block = selection.get("focus_block")
    focus_blocks: list[int] = []
    if focus_block is not None:
        focus_blocks.append(int(focus_block))
    protocol = _norm_text(selection.get("protocol")) or "session"
    label = _norm_text(selection.get("track_label"))
    if not label:
        label = f"block_{int(focus_block)}_track" if focus_blocks else session_name
    return {
        "track_id": _norm_text(selection.get("track_id")) or session_name,
        "label": label,
        "focus_blocks": focus_blocks,
        "protocol": protocol,
    }


def _phase_state_defaults(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_phase": PHASE_PROMPT_STABILIZATION,
        "active_phase_gate": None,
        "phase_gate_counter": 0,
        "history": [],
        "phase_baselines": {},
        "student_model": {
            "model_id": "external_unspecified_student",
            "source": "session_default",
            "frozen": False,
        },
    }


def _with_session_defaults(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest or {})
    session_name = str(payload.get("session_name", ""))
    track = dict(payload.get("track") or {})
    defaults = _session_track_defaults(session_name, payload)
    for key, value in defaults.items():
        track.setdefault(key, value)
    payload["track"] = track
    phase_state = dict(payload.get("phase_state") or {})
    for key, value in _phase_state_defaults(payload).items():
        phase_state.setdefault(key, deepcopy(value))
    payload["phase_state"] = phase_state
    return payload


def _normalize_feature_state_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.setdefault("review_count", 0)
    payload.setdefault("revision_count", 0)
    payload.setdefault("first_agent_round", None)
    payload.setdefault("last_source_type", None)
    return payload


def _accept_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = _status_counts(rows)
    accepted = int(counts.get("accepted", 0))
    terminal = accepted + int(counts.get("uninterpretable", 0)) + int(counts.get("skipped", 0))
    return {
        "accepted": accepted,
        "terminal_reviewed": terminal,
        "cumulative_accept_ratio": (float(accepted) / float(terminal)) if terminal else None,
    }


def _default_prompt_config() -> dict[str, Any]:
    return {
        "version": 2,
        "updated_at": _now_iso(),
        "student_prompt_style": "carrier_first_v1",
        "prompt": {
            "emphasize_activation_geometry": True,
            "emphasize_erf_content": True,
            "allow_uninterpretable": True,
            "ask_positional_or_border_bias": True,
            "prefer_concise_canonical_label": True,
            "allow_richer_notes": True,
            "student_two_stage_locality": True,
            "student_blind_to_prior_semantic_feedback": True,
            "teacher_grounding_auditor_mode": True,
            "reviewer_scope_tag_mode": True,
            "default_to_geometry_when_erf_tiny": True,
            "distinguish_locus_from_subtype": True,
            "adjacent_context_requires_erf_support": True,
            "avoid_overprecision_from_mixed_erfs": True,
            "confidence_tracks_grounding_quality": True,
            "relation_requires_two_sided_erf_support": True,
            "uninterpretable_if_actmap_nonpredictive": True,
            "require_local_appearance_check": True,
            "require_relative_direction_check": True,
            "require_orientation_check": True,
            "prioritize_zoomed_erf_for_micro_motifs": True,
            "require_majority_support_check": True,
            "escalate_to_uninterpretable_when_shared_motif_is_weak": True,
            "require_explicit_positional_bias_resolution": True,
            "require_patch_offset_check": True,
            "require_joint_appearance_geometry_resolution": True,
            "require_heatmap_zoom_reconciliation": True,
            "require_actmap_hypothesis_testing": True,
            "allow_majority_rule_with_partial_exceptions": True,
            "retain_relation_when_actmap_and_heatmap_agree": True,
            "require_neighbor_relative_candidate_generation": True,
            "require_hypothesis_falsification_against_precise_support": True,
            "use_zoom_relative_hint_when_available": False,
        },
        "visualization": {
            "include_actmap": True,
            "include_erf_support": False,
            "include_erf_zoom": True,
            "include_erf_zoom_detail_sidecar": False,
            "include_erf_heatmap_sidecar": False,
            "late_layer_heatmap_enabled": True,
        },
    }


def _feature_prompt_context(prompt_config: dict[str, Any]) -> list[str]:
    prompt_flags = dict(prompt_config.get("prompt", {}))
    viz_config = dict(prompt_config.get("visualization", {}))
    lines: list[str] = []
    if prompt_flags.get("emphasize_activation_geometry"):
        lines.append("Start by identifying the recurring activation geometry across samples before naming any semantic concept.")
        lines.append("Use the feature activation map to reason about where the feature tends to fire, including repeated row/column or upper/lower placement patterns.")
    if prompt_flags.get("require_actmap_hypothesis_testing"):
        lines.append("Use the actmap across the whole batch to generate a working hypothesis before naming the feature: ask what local rule would explain why these particular token positions light up, such as being above, below, or adjacent to a recurring local cue.")
        lines.append("Do not evaluate each sample independently. Form a small number of candidate local rules from the batch-level actmap pattern, then test those rules against the ERF zoom and heatmap across many examples before choosing a label.")
    if prompt_flags.get("require_neighbor_relative_candidate_generation"):
        lines.append("Before settling on a token-intrinsic explanation, generate at least one candidate hypothesis where the token activates because of a cue in a nearby patch rather than in the token patch itself.")
        lines.append("Explicitly compare three candidate families whenever plausible: a token-patch cue, a neighboring-offset cue, and a broader scene or object-part cue.")
    if prompt_flags.get("emphasize_erf_content"):
        lines.append("Use the feature-conditioned ERF evidence to reason about what local visual support/content is sufficient for the feature activation.")
        lines.append("Do not rely on whole-image scene semantics if the ERF evidence does not support them.")
        lines.append("Do not stop at the most obvious object part; ask what makes these positive examples selective relative to many broader images that also contain that part.")
        lines.append("Explicitly compare a broad object-part hypothesis against a narrower relational/configurational hypothesis.")
        lines.append("If the activation map follows one structure but the feature-conditioned ERF repeatedly includes an adjacent interacting object or contact region, prefer the narrower relational/configurational label unless the evidence clearly supports the part alone.")
        lines.append("Ask whether many plausible images containing the broad part would likely not activate this feature, and identify the extra condition, nearby object, or contact relation that makes the positive set more selective.")
        lines.append("Do not claim that an adjacent object is unnecessary unless the ERF evidence consistently excludes it.")
    if prompt_flags.get("student_two_stage_locality"):
        lines.append("Before choosing a canonical label, first write the smallest recurring primary locus and then separately list any adjacent context.")
        lines.append("Prefer locus over scene. If a label names a whole object or event, rewrite it as the smallest recurring causal region supported by the actmap and ERF.")
    if prompt_flags.get("default_to_geometry_when_erf_tiny"):
        lines.append("If ERFs repeatedly collapse to slivers, tiny patches, or very small fragments, default to geometric, appearance-based, or texture-based labels instead of functional object-part labels unless the same part identity is repeatedly clear.")
    if prompt_flags.get("distinguish_locus_from_subtype"):
        lines.append("Separate a stable local locus from a stable semantic subtype: keep the canonical label at the strongest level actually supported, and mention weaker subtype guesses only as uncertainty in the notes.")
    if prompt_flags.get("adjacent_context_requires_erf_support"):
        lines.append("Include adjacent context only when it repeatedly falls inside the ERF support; if it is merely nearby or mixed across examples, mark it as optional or uncertain rather than essential.")
    if prompt_flags.get("avoid_overprecision_from_mixed_erfs"):
        lines.append("If the recurring local pattern is real but the exact attachment, top-part extent, or subpart geometry varies across examples, prefer a slightly looser local structural label over an overprecise one.")
    if prompt_flags.get("confidence_tracks_grounding_quality"):
        lines.append("Calibrate confidence to grounding quality: lower confidence when the local recurrence is real but ERFs are heterogeneous, tiny, or mixed with nearby context.")
    if prompt_flags.get("relation_requires_two_sided_erf_support"):
        lines.append("Prefer a relational or configurational label only when both sides of the relation are consistently part of the ERF; otherwise stay on the single-sided local pattern.")
    if prompt_flags.get("uninterpretable_if_actmap_nonpredictive"):
        lines.append("If the actmap does not provide a stable, predictive locus across examples and the ERFs collapse to unrelated tiny patches, prefer `uninterpretable` over a catch-all local label.")
    if prompt_flags.get("require_local_appearance_check"):
        lines.append("When the feature appears appearance-driven, explicitly test whether a stable color family, brightness polarity, or repeated local line pattern explains the positives better than a vague object-part story.")
        lines.append("Do not collapse a specific recurring appearance cue into a loose color family if a narrower color-plus-structure explanation fits the batch better.")
        lines.append("Do not downgrade a stable dark/light/color/texture cue into a generic patch-offset story when the appearance cue itself is more consistent across examples.")
        lines.append("If the recurring cue is texture-like, stripe-like, or material-like, preserve that appearance in the canonical label instead of collapsing it to `patch`.")
    if prompt_flags.get("require_relative_direction_check"):
        lines.append("Explicitly check patch-relative direction: whether the stable cue lies above, below, left, or right of the target patch, and keep that directional relation when it is part of the recurrence.")
        lines.append("Treat a relative-direction hypothesis as secondary if a token-local appearance or texture cue explains more examples more precisely.")
    if prompt_flags.get("require_patch_offset_check"):
        lines.append("When a cue is direction-relative to the target patch, also decide whether it is immediately adjacent or offset by roughly one to three patch-widths, and preserve that offset when it is stable.")
    if prompt_flags.get("require_orientation_check"):
        lines.append("Explicitly check local orientation such as horizontal, vertical, diagonal, or clustered parallel strokes; do not reduce a stable directional structure to a generic patch label.")
    if prompt_flags.get("require_joint_appearance_geometry_resolution"):
        lines.append("For line-like, bar-like, or stroke-like motifs, resolve both appearance and geometry together: keep stable darkness/lightness or color information along with horizontal, vertical, or diagonal structure rather than collapsing to only one of those axes.")
        lines.append("For border or frame cues, preserve the dominant orientation when it is stable, such as a horizontal border band instead of a generic frame label.")
    if prompt_flags.get("prioritize_zoomed_erf_for_micro_motifs"):
        lines.append("For tiny or high-frequency local motifs, inspect the ERF zoom carefully and prioritize repeated micro-structure visible there, such as short line clusters, claw-like marks, or thin stroke bundles.")
        if viz_config.get("include_erf_zoom_detail_sidecar"):
            lines.append("If an ERF zoom detail view is provided, use it as a contrast-enhanced aid for tiny dark/light or pixel-level motifs, but verify any claimed pattern against the raw ERF zoom rather than trusting the detail view alone.")
    if prompt_flags.get("require_heatmap_zoom_reconciliation"):
        lines.append("When the ERF heatmap and ERF zoom do not highlight exactly the same content, treat the heatmap as coarse localization and the zoom as local confirmation. Promote a cue to the primary locus only if it recurs in the zoom; if it appears mainly in the heatmap, keep it as broader or uncertain context rather than the canonical label.")
    if prompt_flags.get("use_zoom_relative_hint_when_available"):
        lines.append("If a zoom-relative hint is provided, use it as a geometric aid for counting patch offsets and support direction relative to the token, but verify it against the actmap and images rather than trusting it blindly.")
    if prompt_flags.get("retain_relation_when_actmap_and_heatmap_agree"):
        lines.append("If the actmap and heatmap repeatedly support the same directional or relational cue but some zoom crops underresolve it, keep that relation as a tentative but real majority-supported hypothesis rather than dropping it entirely.")
    if prompt_flags.get("require_majority_support_check"):
        lines.append("Before promoting a motif into the canonical label, check that it is clearly supported by a majority of examples rather than only a salient subset.")
    if prompt_flags.get("allow_majority_rule_with_partial_exceptions"):
        lines.append("Allow a rule to survive partial failures: if a narrow local relation explains most examples and the remaining misses look like weak crops, borderline ERF failures, or noisy exceptions, keep the rule and note the exceptions instead of defaulting to `uninterpretable`.")
    if prompt_flags.get("require_hypothesis_falsification_against_precise_support"):
        lines.append("After choosing a candidate hypothesis, try to falsify it against the exact actmap hotspot and ERF zoom locus. Reject any broader scene story that does not match the precise local support, even if it feels semantically coherent.")
    if prompt_flags.get("escalate_to_uninterpretable_when_shared_motif_is_weak"):
        lines.append("If a proposed shared motif stays weak after comparing examples, especially when ERFs are tiny, fragmented, or loosely related, prefer `uninterpretable` over an overfit local story.")
    if prompt_flags.get("require_explicit_positional_bias_resolution"):
        lines.append("Explicitly decide whether repeated left-right, top-bottom, or border placement explains the feature better than local semantics, and mention that resolution in the notes.")
    if prompt_flags.get("ask_positional_or_border_bias"):
        lines.append("Explicitly check for positional bias, border bias, or frame-like behavior.")
    if prompt_flags.get("allow_uninterpretable"):
        lines.append("If the feature looks uninterpretable, say so directly instead of forcing a label.")
    if prompt_flags.get("prefer_concise_canonical_label"):
        lines.append("Keep the canonical label short and reusable.")
    if prompt_flags.get("allow_richer_notes"):
        lines.append("Use description/notes to capture ambiguity, scope, and failure modes.")
        lines.append("In notes, explicitly mention what actmap pattern and what ERF evidence supported your label.")
    return lines


def _render_student_prompt(prompt_config: dict[str, Any]) -> str:
    if str(prompt_config.get("student_prompt_style") or "").strip().lower() == "carrier_first_v1":
        return """You are the student labeler. Use only the student-visible evidence:
- original image with boxed token
- feature act map
- feature-conditioned ERF zoom
- feature-conditioned ERF heatmap

Interpret the evidence as:
- `feature act map`: where the feature tends to fire. The cyan/teal overlay is an activation heatmap visualization only, not scene color or object content.
- `feature-conditioned ERF zoom`: the smallest local support sufficient at the target token
- `feature-conditioned ERF heatmap`: the coarse support extent and any surrounding context that is actually required

For each feature, output:
- `primary_locus`
- `adjacent_context`
- `canonical_label`
- `support_summary`
- `description`
- `notes`
- `confidence`

Use this decision order:

1. Fix the supported span before naming the concept.
- Use the act map and heatmap together to decide how large the real support is.
- Do not treat cyan/teal overlay color, token boxes, or any panel-rendering artifact as evidence about the image content.
- If the heatmap covers more than the token-centered detail, do not collapse to the most salient local part.
- If the heatmap stays small, do not broaden to the whole object or scene.
- A correct category with the wrong supported extent is still wrong.

2. Treat the obvious host as a hypothesis, not the default answer.
- Before using a whole object, body region, or scene label, rule out these smaller support types:
  - tiny local motif or texture
  - interface or boundary
  - surrounding field or negative space
  - object-field arrangement
  - nearby offset element that transfers better than the token-centered patch
- Only use a whole-host label if no smaller transferable support explains the examples better.

3. Decide the support type.
- Ask whether the support is mainly:
  - whole host
  - subpart
  - interface/boundary
  - surrounding field
  - object-field arrangement
  - material/surface
  - tiny local motif
- If support survives after mentally removing the most salient object, part, or inner content, demote that object, part, or content.

4. Recover the recurring carrier before using generic fallback language.
- Before using a generic field, boundary, edge, or patch label, check whether a more specific recurring carrier survives across examples.
- A recurring carrier may be:
  - a material or texture family
  - a low-saturation color-state
  - a repeated thin motif or mark family
  - symbolic or printed content
  - a sparse articulated structure
  - a distinctive apparatus or object subpart
- If the feature would stop transferring after mentally removing that recurring carrier, make the carrier primary and treat surrounding field/boundary as secondary context.
- Host identity may vary while the recurring carrier remains stable. Do not demote the carrier just because the carrier appears on different hosts.

5. Use background, field, and boundary labels only as a last resort.
- Plain field, background, edge, or generic patch labels are allowed only if the feature still transfers after mentally removing the recurring carrier.
- If removing the carrier breaks transfer, the carrier is primary and the field/boundary belongs in `adjacent_context` or `description`.
- Do not let a safe-but-broad field reading override a narrower recurring supported cue.

6. Treat weak evidence as real evidence.
- Blank, dark, pale, muted, low-texture, and low-contrast regions can be primary support.
- Muted color-states such as cream, beige, tan, off-white, pale gray, buff, dusk glow, or other low-saturation light/dark states count as real evidence when they recur.
- If the transferable cue is weak but stable, do not replace it with a more salient nearby object or scene explanation.

7. For tiny support, let the zoom outrank the scene.
- When support is tiny, classify the zoomed pattern before using scene or object semantics.
- Distinguish primitive pattern types such as patch, line, cluster, grid/check, boundary, or other small repeated motif.
- Keep minimal micro-structure qualifiers like orientation, repetition, clusteredness, crossing structure, or mark-family behavior when they are needed for transfer.

8. Keep only necessary qualifiers.
- Drop incidental color, lighting, material, time-of-day, or scene cues if the feature still transfers without them.
- Keep a simple qualifier only when removing it would overgeneralize: position/offset, interface, state, orientation, material/surface, muted color-state, or recurring carrier type.

9. Write the description before finalizing the label.
- The `description` must have exactly 3 sentences in this order:
  1. where the activation is centered
  2. what extra ERF/heatmap-supported context is required for transfer
  3. what nearby content is frequent but not primary
- Sentence 1 must be centered on the act-map / heatmap-supported locus, not the most salient host.
- Sentence 2 must use the heatmap to decide whether broader context is actually required; do not let the zoom erase a broader required relation or field.
- Sentence 3 must explicitly demote frequent but non-primary host/context content.

10. Make the canonical label a compression of the structured description.
- The `canonical_label` must describe sentence 1 plus only the minimal necessary part of sentence 2.
- Do not include sentence-3 context in the label.
- If the description supports a compact locus plus a required relation, prefer that compressed relation label over a broad host label.

11. Distill an evaluator-friendly support summary.
- `support_summary` must be a single short sentence or sentence fragment.
- It must summarize the minimal transferable cue in plain semantic terms.
- Do not mention `ERF`, `heatmap`, `activation`, `examples`, `support`, or any other analysis process words.
- Do not include frequent-but-non-primary context.
- Think of it as the shortest human-readable explanation of what has to be present for the label to apply.

12. Match specificity to the supported evidence.
- After support span and support type are correct, choose the narrowest label that still transfers across examples.
- Tighten one step at a time and stop before the label becomes brittle or instance-specific.

Output concise, evidence-grounded labels. Prefer the smallest transferable supported explanation over the nearest familiar prototype."""
    prompt_flags = dict(prompt_config.get("prompt", {}))
    student_fields = [
        "- canonical_label",
        "- description",
        "- notes",
        "- confidence",
    ]
    if prompt_flags.get("student_two_stage_locality"):
        student_fields = [
            "- primary_locus",
            "- adjacent_context",
            "- canonical_label",
            "- description",
            "- notes",
            "- confidence",
        ]
    lines = [
        "You are the student labeler.",
        "For each feature, inspect the provided evidence examples and produce:",
        *student_fields,
    ]
    lines.extend(f"- {line}" for line in _feature_prompt_context(prompt_config))
    lines.append("Do not hallucinate certainty. Use notes to capture ambiguity.")
    return "\n".join(lines)


def _render_teacher_prompt(prompt_config: dict[str, Any]) -> str:
    prompt_flags = dict(prompt_config.get("prompt", {}))
    lines = [
        "You are the teacher.",
        "Review the student's label, description, and notes.",
        "Return a verdict and field-level critique.",
        "If GT is provided for the feature, use it carefully as optional supervision.",
        "Check whether the student actually grounded the label in the activation-map geometry and feature-conditioned ERF evidence.",
        "Call out labels that seem driven by whole-image semantics rather than the actmap/ERF evidence.",
        "Explicitly check whether the student stopped too early at a broad object-part label when the evidence supports a narrower relational or configurational label.",
        "If the ERF repeatedly includes an adjacent interacting object or contact region, say whether the student underweighted that evidence.",
    ]
    if prompt_flags.get("teacher_grounding_auditor_mode"):
        lines.append("Act as a grounding auditor, not a semantic coach.")
        lines.append("Do not propose substitute canonical labels or answer phrases unless GT is explicitly provided.")
        lines.append("Focus on scope, locality, ERF grounding, and whether adjacent context was underweighted or overstated.")
    if any(
        prompt_flags.get(flag)
        for flag in (
            "require_local_appearance_check",
            "require_relative_direction_check",
            "require_patch_offset_check",
            "require_orientation_check",
            "require_joint_appearance_geometry_resolution",
            "prioritize_zoomed_erf_for_micro_motifs",
            "require_heatmap_zoom_reconciliation",
            "require_explicit_positional_bias_resolution",
            "require_actmap_hypothesis_testing",
            "allow_majority_rule_with_partial_exceptions",
            "retain_relation_when_actmap_and_heatmap_agree",
            "require_neighbor_relative_candidate_generation",
            "require_hypothesis_falsification_against_precise_support",
        )
    ):
        lines.append("Explicitly audit whether the student failed to generate or test the right actmap-level hypothesis, missed a stable appearance cue, patch-relative direction or patch-offset relation, local orientation, joint appearance-plus-geometry structure, ERF-zoom micro-motif, heatmap-versus-zoom disagreement, or positional-bias explanation that is visible in the evidence.")
    if prompt_flags.get("allow_uninterpretable"):
        lines.append("Do not force interpretability if the evidence does not support it.")
    return "\n".join(lines)


def _render_reviewer_prompt(prompt_config: dict[str, Any]) -> str:
    prompt_flags = dict(prompt_config.get("prompt", {}))
    lines = [
        "You are the reviewer.",
        "Read the student and teacher outputs for the batch.",
        "For each feature, diagnose failure modes.",
        "Explicitly note when the labeler appears to have ignored the activation map or the feature-conditioned ERF evidence.",
        "Explicitly diagnose broad-object-part versus narrow-relational/configurational confusion when it appears.",
        "At the batch level, propose prompt changes and visualization changes.",
        "Do not directly mutate the prompts; only propose changes with rationale.",
    ]
    if prompt_flags.get("reviewer_scope_tag_mode"):
        lines.append("Prefer abstract failure tags such as `scope_too_broad_part`, `scope_too_broad_scene`, `adjacent_context_underweighted`, or `erf_grounding_overclaim` over direct semantic answer suggestions.")
    if any(
        prompt_flags.get(flag)
        for flag in (
            "require_local_appearance_check",
            "require_relative_direction_check",
            "require_patch_offset_check",
            "require_orientation_check",
            "require_joint_appearance_geometry_resolution",
            "prioritize_zoomed_erf_for_micro_motifs",
            "require_heatmap_zoom_reconciliation",
            "require_explicit_positional_bias_resolution",
            "require_actmap_hypothesis_testing",
            "allow_majority_rule_with_partial_exceptions",
            "retain_relation_when_actmap_and_heatmap_agree",
            "require_neighbor_relative_candidate_generation",
            "require_hypothesis_falsification_against_precise_support",
        )
    ):
        lines.append("Diagnose missed actmap-hypothesis-testing, appearance-cue, relative-direction or patch-offset, orientation, joint appearance-plus-geometry, ERF-zoom micro-motif, heatmap-versus-zoom reconciliation, and positional-bias failures when they seem to explain the student error.")
    lines.append("Only propose generalizable student-prompt improvements that could help unseen features; do not smuggle a feature-specific answer into a prompt suggestion.")
    if dict(prompt_config.get("visualization", {})).get("include_erf_heatmap_sidecar"):
        lines.append("Comment on whether the ERF heatmap sidecar is helping or hurting the loop, especially when the heatmap and zoom emphasize different structures.")
    return "\n".join(lines)


def _set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = mapping
    parts = [part for part in str(dotted_path).split(".") if part]
    if not parts:
        raise ValueError("Empty proposal path")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _proposal_id(scope: str, index: int) -> str:
    return f"{scope}_{int(index):02d}"


def _normalize_proposals(scope: str, rows: list[Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows or [], start=1):
        if isinstance(row, str):
            text = _norm_text(row)
            normalized.append(
                {
                    "proposal_id": _proposal_id(scope, idx),
                    "scope": scope,
                    "path": "",
                    "proposed_value": None,
                    "rationale": text,
                    "label": text or _proposal_id(scope, idx),
                }
            )
            continue
        normalized.append(
            {
                "proposal_id": _proposal_id(scope, idx),
                "scope": scope,
                "path": str(row.get("path", "")),
                "proposed_value": row.get("proposed_value"),
                "rationale": _norm_text(row.get("rationale")),
                "label": _norm_text(row.get("label")) or str(row.get("path", "")),
            }
        )
    return normalized


def _student_visible_label_examples(
    label_examples: list[dict[str, Any]],
    prompt_config: dict[str, Any],
) -> list[dict[str, Any]]:
    viz_config = dict(prompt_config.get("visualization", {}))
    prompt_flags = dict(prompt_config.get("prompt", {}))
    rows: list[dict[str, Any]] = []
    for example in label_examples:
        filtered = dict(example)
        if not viz_config.get("include_actmap"):
            filtered.pop("feature_actmap", None)
            filtered.pop("feature_actmap_abs", None)
        if not viz_config.get("include_erf_support"):
            filtered.pop("feature_erf_support", None)
            filtered.pop("feature_erf_support_abs", None)
        if not viz_config.get("include_erf_zoom"):
            filtered.pop("feature_erf_zoom", None)
            filtered.pop("feature_erf_zoom_abs", None)
        if not viz_config.get("include_erf_zoom_detail_sidecar"):
            filtered.pop("feature_erf_zoom_detail", None)
            filtered.pop("feature_erf_zoom_detail_abs", None)
        if not viz_config.get("include_erf_heatmap_sidecar"):
            filtered.pop("feature_erf_heatmap", None)
            filtered.pop("feature_erf_heatmap_abs", None)
        if not prompt_flags.get("use_zoom_relative_hint_when_available"):
            filtered.pop("feature_erf_zoom_meta", None)
        rows.append(filtered)
    return rows


def _zoom_relative_hint(example: dict[str, Any]) -> dict[str, Any] | None:
    meta = dict(example.get("feature_erf_zoom_meta") or {})
    offset = meta.get("support_centroid_offset_from_token")
    vertical = meta.get("support_vertical_relation")
    horizontal = meta.get("support_horizontal_relation")
    if not offset and not vertical and not horizontal:
        return None
    hint: dict[str, Any] = {}
    if vertical:
        hint["vertical_relation"] = vertical
    if horizontal:
        hint["horizontal_relation"] = horizontal
    if isinstance(offset, dict):
        hint["delta_row"] = round(float(offset.get("delta_row", 0.0)), 2)
        hint["delta_col"] = round(float(offset.get("delta_col", 0.0)), 2)
    return hint or None


def _render_agent_packet_markdown(packet: dict[str, Any], round_dir: Path) -> str:
    viz_config = dict(packet.get("prompt_config", {}).get("visualization", {}))
    prompt_flags = dict(packet.get("prompt_config", {}).get("prompt", {}))
    lines: list[str] = [
        f"# Autolabel Agent Packet",
        "",
        f"- session_name: `{packet['session_name']}`",
        f"- round_index: `{packet['round_index']}`",
        f"- response_path: `{round_dir / 'agent_response.json'}`",
        "",
        "## Roles",
        "",
        "### Student",
        "```text",
        str(packet["rendered_prompts"]["student"]),
        "```",
        "",
        "### Teacher",
        "```text",
        str(packet["rendered_prompts"]["teacher"]),
        "```",
        "",
        "### Reviewer",
        "```text",
        str(packet["rendered_prompts"]["reviewer"]),
        "```",
        "",
        "## Response contract",
        "",
        "Write `agent_response.json` with:",
        "- `features`: one object per feature",
        "- `reviewer_batch`: prompt/visualization proposals for the next round",
        "",
        "Each feature object should contain `feature_key`, `student`, `teacher`, and `reviewer_feature`.",
        "",
        "## Feature evidence",
    ]
    for feature in packet["features"]:
        lines.append("")
        lines.append(f"### {feature['feature_key']}")
        lines.append(f"- block: `{feature['block_idx']}`")
        lines.append(f"- feature_id: `{feature['feature_id']}`")
        if feature.get("provided_gt"):
            lines.append(f"- provided_gt: `{json.dumps(feature['provided_gt'], ensure_ascii=False)}`")
        if feature.get("prior_review_metadata"):
            lines.append(
                f"- prior_review_metadata: `{json.dumps(feature['prior_review_metadata'], ensure_ascii=False)}`"
            )
        if feature.get("prior_failure_tags"):
            lines.append(f"- prior_failure_tags: `{json.dumps(feature['prior_failure_tags'], ensure_ascii=False)}`")
        for example in feature["label_examples"]:
            parts = [f"original=`{example['original_with_token_box_abs']}`"]
            if viz_config.get("include_actmap") and example.get("feature_actmap_abs"):
                parts.append(f"actmap=`{example['feature_actmap_abs']}`")
            if viz_config.get("include_erf_support") and example.get("feature_erf_support_abs"):
                parts.append(f"feature_erf_support=`{example['feature_erf_support_abs']}`")
            if viz_config.get("include_erf_zoom") and example.get("feature_erf_zoom_abs"):
                parts.append(f"feature_erf_zoom=`{example['feature_erf_zoom_abs']}`")
                zoom_hint = _zoom_relative_hint(example)
                if prompt_flags.get("use_zoom_relative_hint_when_available") and zoom_hint:
                    parts.append(f"feature_erf_zoom_hint=`{json.dumps(zoom_hint, ensure_ascii=False)}`")
            if viz_config.get("include_erf_zoom_detail_sidecar") and example.get("feature_erf_zoom_detail_abs"):
                parts.append(f"feature_erf_zoom_detail=`{example['feature_erf_zoom_detail_abs']}`")
            if viz_config.get("include_erf_heatmap_sidecar") and example.get("feature_erf_heatmap_abs"):
                parts.append(f"feature_erf_heatmap=`{example['feature_erf_heatmap_abs']}`")
            lines.append(f"- sample {example['rank'] + 1}: " + " ".join(parts))
    return "\n".join(lines) + "\n"


def _initial_feature_state_row(feature: dict[str, Any], prompt_config_version: int) -> dict[str, Any]:
    return {
        "feature_key": str(feature["feature_key"]),
        "block_idx": int(feature["block_idx"]),
        "feature_id": int(feature["feature_id"]),
        "status": "awaiting_agent",
        "terminal": False,
        "created_at": _now_iso(),
        "last_updated_at": _now_iso(),
        "current_prompt_config_version": int(prompt_config_version),
        "last_agent_round": None,
        "last_human_round": None,
        "terminal_round": None,
        "latest_student": {},
        "latest_teacher": {},
        "latest_reviewer_feature": {},
        "latest_human_feedback": {},
        "provided_gt": {},
        "meta_tags": [],
        "meta_notes": "",
        "final_label": {},
        "final_human_decision": None,
        "review_count": 0,
        "revision_count": 0,
        "first_agent_round": None,
        "last_source_type": None,
    }


def _agent_visible_human_feedback(row: dict[str, Any]) -> dict[str, Any]:
    latest = dict(row.get("latest_human_feedback") or {})
    if not latest:
        return {}
    human_decision = _norm_text(latest.get("human_decision")).lower()
    visible = {
        "human_decision": human_decision,
        "meta_tags": list(latest.get("meta_tags") or []),
        "meta_notes": _norm_text(latest.get("meta_notes")),
        "round_index": latest.get("round_index"),
        "reviewed_at": latest.get("reviewed_at"),
        "has_private_human_feedback": bool(_norm_text(latest.get("human_feedback"))),
    }
    gt = {
        "canonical_label": _norm_text(latest.get("gt_canonical_label")),
        "description": _norm_text(latest.get("gt_description")),
        "notes": _norm_text(latest.get("gt_notes")),
    }
    if human_decision == "provide_gt" or any(gt.values()):
        visible["provided_gt_guidance"] = gt
    return {
        key: value
        for key, value in visible.items()
        if value not in ("", None, [], {})
    }


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = str(text).lower()
    return any(needle in lowered for needle in needles)


def _prior_failure_tags(row: dict[str, Any]) -> list[str]:
    latest_teacher = dict(row.get("latest_teacher") or {})
    latest_reviewer = dict(row.get("latest_reviewer_feature") or {})
    latest_human = dict(row.get("latest_human_feedback") or {})
    texts = [
        _norm_text(latest_teacher.get("canonical_label_feedback")),
        _norm_text(latest_teacher.get("description_feedback")),
        _norm_text(latest_teacher.get("notes_feedback")),
        _norm_text(latest_teacher.get("overall_feedback")),
        _norm_text(latest_reviewer.get("diagnosis")),
        *[_norm_text(item) for item in list(latest_reviewer.get("failure_modes") or [])],
        _norm_text(latest_human.get("human_feedback")),
    ]
    tags: list[str] = []
    if _norm_text(latest_human.get("human_decision")).lower() == "revise":
        tags.append("prior_revision")
    if any(_contains_any(text, ("too generic", "too broad", "broad ", "generic ", "underfit")) for text in texts if text):
        tags.append("scope_too_broad_part")
    if any(_contains_any(text, ("scene-level", "whole scene", "full scene", "whole animal", "full animal", "event", "scene label")) for text in texts if text):
        tags.append("scope_too_broad_scene")
    if any(_contains_any(text, ("adjacent", "nearby animal", "contact region", "junction", "support relation", "supporting")) for text in texts if text):
        tags.append("adjacent_context_underweighted")
    if any(_contains_any(text, ("overstate", "overclaim", "underweighted the erf", "erf evidence was partially underused", "not well supported")) for text in texts if text):
        tags.append("erf_grounding_issue")
    if any(_contains_any(text, ("border bias", "frame-like", "positional bias")) for text in texts if text):
        tags.append("possible_positional_bias")
    return list(dict.fromkeys(tag for tag in tags if tag))


def _student_visible_review_metadata(row: dict[str, Any], prompt_config: dict[str, Any]) -> dict[str, Any]:
    prompt_flags = dict(prompt_config.get("prompt", {}))
    latest = _agent_visible_human_feedback(row)
    if not latest:
        return {}
    if prompt_flags.get("student_blind_to_prior_semantic_feedback"):
        filtered = {
            "human_decision": latest.get("human_decision"),
            "round_index": latest.get("round_index"),
            "reviewed_at": latest.get("reviewed_at"),
            "has_private_human_feedback": latest.get("has_private_human_feedback"),
            "meta_tags": latest.get("meta_tags"),
        }
        return {
            key: value
            for key, value in filtered.items()
            if value not in ("", None, [], {})
        }
    return latest


def _autolabel_label_example_target(config: EvalConfig, block_idx: int) -> int:
    mapping = dict(getattr(config, "autolabel_label_examples_by_block", {}) or {})
    return int(mapping.get(int(block_idx), config.study_label_examples_per_feature))


def _render_autolabel_feature_pool_row(
    *,
    config: EvalConfig,
    session_dir: Path,
    runtime: LegacyRuntime,
    frame_cache: dict[int, Any],
    sid_to_path_cache: dict[int, str],
    feature: dict[str, Any],
) -> dict[str, Any]:
    feature_key_value = str(feature["feature_key"])
    feature_dir = session_dir / "assets" / _slug(feature_key_value)
    if feature_dir.exists():
        shutil.rmtree(feature_dir)
    label_examples = _collect_label_examples(
        config=config,
        runtime=runtime,
        feature=feature,
        target_count=_autolabel_label_example_target(config, int(feature["block_idx"])),
        frame_cache=frame_cache,
        sid_to_path_cache=sid_to_path_cache,
    )
    rendered_examples: list[dict[str, Any]] = []
    for rank, row in enumerate(label_examples):
        image_path = str(row["image_path"])
        block_idx = int(feature["block_idx"])
        feature_id = int(feature["feature_id"])
        token_idx = int(row["target_patch_idx"])
        original_path = feature_dir / f"example_{rank:02d}_original_token.png"
        actmap_path = feature_dir / f"example_{rank:02d}_feature_actmap.png"
        erf_support_path = feature_dir / f"example_{rank:02d}_feature_erf_support.png"
        erf_zoom_path = feature_dir / f"example_{rank:02d}_feature_erf_zoom.png"
        erf_zoom_detail_path = feature_dir / f"example_{rank:02d}_feature_erf_zoom_detail.png"
        erf_heatmap_path = feature_dir / f"example_{rank:02d}_feature_erf_heatmap.png"
        erf_json_path = feature_dir / f"example_{rank:02d}_feature_erf.json"

        save_original_with_token_box(image_path, original_path, token_idx)
        actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
        save_feature_actmap_overlay(image_path, actmap, actmap_path, token_idx=token_idx)
        erf = runtime.cautious_feature_erf(image_path, block_idx, token_idx, feature_id)
        save_support_mask_image(image_path, erf["support_indices"], erf_support_path, token_idx=token_idx)
        zoom_meta = save_support_outline_crop_image(
            image_path,
            erf["support_indices"],
            erf_zoom_path,
            token_idx=token_idx,
            score_map=erf["prob_scores"],
        )
        save_support_detail_crop_image(
            image_path,
            erf["support_indices"],
            erf_zoom_detail_path,
            token_idx=token_idx,
        )
        save_erf_heatmap_image(
            image_path,
            erf["prob_scores"],
            erf_heatmap_path,
            token_idx=token_idx,
        )
        write_json(erf_json_path, erf)
        rendered_examples.append(
            {
                "rank": rank,
                "sample_id": int(row["sample_id"]),
                "token_idx": token_idx,
                "label_example_source": str(row.get("label_example_source", "train")),
                "original_with_token_box": _relpath(session_dir, original_path),
                "feature_actmap": _relpath(session_dir, actmap_path),
                "feature_erf_support": _relpath(session_dir, erf_support_path),
                "feature_erf_zoom": _relpath(session_dir, erf_zoom_path),
                "feature_erf_zoom_detail": _relpath(session_dir, erf_zoom_detail_path),
                "feature_erf_heatmap": _relpath(session_dir, erf_heatmap_path),
                "feature_erf_json": _relpath(session_dir, erf_json_path),
                "original_with_token_box_abs": str(original_path.resolve()),
                "feature_actmap_abs": str(actmap_path.resolve()),
                "feature_erf_support_abs": str(erf_support_path.resolve()),
                "feature_erf_zoom_abs": str(erf_zoom_path.resolve()),
                "feature_erf_zoom_detail_abs": str(erf_zoom_detail_path.resolve()),
                "feature_erf_heatmap_abs": str(erf_heatmap_path.resolve()),
                "feature_erf_json_abs": str(erf_json_path.resolve()),
                "feature_erf_zoom_meta": dict(zoom_meta),
            }
        )
    return {
        "feature_key": feature_key_value,
        "block_idx": int(feature["block_idx"]),
        "feature_id": int(feature["feature_id"]),
        "selection_stats": feature.get("selection_stats", {}),
        "label_examples": rendered_examples,
        "holdout_rows": [dict(row) for row in feature["holdout"]],
    }


def refresh_autolabel_session_assets(config: EvalConfig, session_name: str) -> dict[str, Any]:
    session_dir = _autolabel_session_dir(config, session_name)
    round_policy = _load_round_policy_if_exists(config, session_name)
    feature_pool_path = session_dir / "feature_pool.json"
    if not feature_pool_path.exists():
        raise FileNotFoundError(f"Missing feature pool at {feature_pool_path}")
    feature_pool = read_json(feature_pool_path)
    feature_bank = load_feature_bank(config)
    features_by_key = _feature_lookup(feature_bank)
    runtime = LegacyRuntime(config)
    frame_cache: dict[int, Any] = {}
    sid_to_path_cache: dict[int, str] = {}
    refreshed_rows: list[dict[str, Any]] = []
    try:
        for entry in feature_pool.get("features", []):
            feature_key_value = str(entry["feature_key"])
            feature = features_by_key.get(feature_key_value)
            if feature is None:
                raise KeyError(f"Feature {feature_key_value} not found in feature bank")
            refreshed_rows.append(
                _render_autolabel_feature_pool_row(
                    config=config,
                    session_dir=session_dir,
                    runtime=runtime,
                    frame_cache=frame_cache,
                    sid_to_path_cache=sid_to_path_cache,
                    feature=feature,
                )
            )
    finally:
        runtime.close()
    feature_pool["features"] = refreshed_rows
    feature_pool["refreshed_at"] = _now_iso()
    write_json(feature_pool_path, feature_pool)
    return {
        "session_dir": str(session_dir),
        "feature_pool_json": str(feature_pool_path),
        "n_features": len(refreshed_rows),
        "examples_per_feature": {
            str(row["feature_key"]): len(list(row.get("label_examples", [])))
            for row in refreshed_rows
        },
    }


def _top_up_awaiting_agent_features(
    config: EvalConfig,
    *,
    session_name: str,
    rows: list[dict[str, Any]],
    feature_pool: dict[str, Any],
    manifest: dict[str, Any],
    prompt_config_version: int,
    seed: int,
) -> list[str]:
    target_per_block = int(manifest.get("features_per_block") or config.autolabel_session_default_features_per_block)
    awaiting_per_block: dict[int, int] = {}
    existing_feature_keys = {str(row["feature_key"]) for row in rows}
    track = dict(manifest.get("track") or {})
    focus_blocks = [int(block_idx) for block_idx in list(track.get("focus_blocks") or []) if str(block_idx).strip()]
    top_up_blocks = focus_blocks or [int(block_idx) for block_idx in config.blocks]
    for row in rows:
        if str(row.get("status", "")) != "awaiting_agent":
            continue
        block_idx = int(row["block_idx"])
        awaiting_per_block[block_idx] = awaiting_per_block.get(block_idx, 0) + 1

    feature_bank = load_feature_bank(config)
    session_dir = _autolabel_session_dir(config, session_name)
    runtime = LegacyRuntime(config)
    frame_cache: dict[int, Any] = {}
    sid_to_path_cache: dict[int, str] = {}
    rng = random.Random(int(seed))
    added: list[str] = []
    try:
        for block_idx in top_up_blocks:
            need = max(0, target_per_block - int(awaiting_per_block.get(int(block_idx), 0)))
            if need <= 0:
                continue
            candidates = [
                feature
                for feature in feature_bank["blocks"][str(block_idx)]["features"]
                if str(feature["feature_key"]) not in existing_feature_keys
            ]
            rng.shuffle(candidates)
            for feature in candidates[:need]:
                feature_pool_row = _render_autolabel_feature_pool_row(
                    config=config,
                    session_dir=session_dir,
                    runtime=runtime,
                    frame_cache=frame_cache,
                    sid_to_path_cache=sid_to_path_cache,
                    feature=feature,
                )
                feature_pool.setdefault("features", []).append(feature_pool_row)
                rows.append(_initial_feature_state_row(feature, prompt_config_version=prompt_config_version))
                existing_feature_keys.add(str(feature["feature_key"]))
                awaiting_per_block[int(block_idx)] = awaiting_per_block.get(int(block_idx), 0) + 1
                added.append(str(feature["feature_key"]))
    finally:
        runtime.close()
    return added


def _round_status_payload(
    *,
    session_name: str,
    round_index: int,
    prompt_config: dict[str, Any],
    feature_keys: list[str],
    status: str,
    reviewer_batch: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "session_name": session_name,
        "round_index": int(round_index),
        "status": status,
        "created_at": _now_iso(),
        "prompt_config_version": int(prompt_config["version"]),
        "prompt_config": prompt_config,
        "feature_keys": list(feature_keys),
        "reviewer_batch": reviewer_batch or {},
    }
    if extra:
        payload.update(extra)
    return payload


def _feature_examples_cards(feature: dict[str, Any], prompt_config: dict[str, Any]) -> str:
    viz_config = dict(prompt_config.get("visualization", {}))
    cards: list[str] = []
    for example in feature["label_examples"]:
        support_html = ""
        if viz_config.get("include_erf_support"):
            support_html = f"""
            <div class="evidence-slot">
              <img src="{html.escape(str(example['feature_erf_support']))}" alt="feature-erf-support">
              <div class="caption">Feature-conditioned ERF support</div>
            </div>
            """
        heatmap_html = ""
        if viz_config.get("include_erf_heatmap_sidecar"):
            heatmap_html = f"""
            <div class="evidence-slot">
              <img src="{html.escape(str(example['feature_erf_heatmap']))}" alt="feature-erf-heatmap">
              <div class="caption">Feature-conditioned ERF quantitative heatmap</div>
            </div>
            """
        actmap_html = ""
        if viz_config.get("include_actmap"):
            actmap_html = f"""
            <div class="evidence-slot">
              <img src="{html.escape(str(example['feature_actmap']))}" alt="actmap">
              <div class="caption">Feature SAE activation localization panel (cyan = activation overlay only)</div>
            </div>
            """
        erf_zoom_html = ""
        if viz_config.get("include_erf_zoom"):
            erf_zoom_html = f"""
            <div class="evidence-slot">
              <img src="{html.escape(str(example['feature_erf_zoom']))}" alt="feature-erf-zoom">
              <div class="caption">Feature-conditioned ERF zoom (black outside support)</div>
            </div>
            """
        erf_zoom_detail_html = ""
        if viz_config.get("include_erf_zoom_detail_sidecar"):
            erf_zoom_detail_html = f"""
            <div class="evidence-slot">
              <img src="{html.escape(str(example['feature_erf_zoom_detail']))}" alt="feature-erf-zoom-detail">
              <div class="caption">ERF zoom detail</div>
            </div>
            """
        cards.append(
            f"""
            <div class="example-card">
              <div class="example-head">Sample {int(example['rank']) + 1} | sample_id={int(example['sample_id'])} tok={int(example['token_idx'])}</div>
              <div class="evidence-grid">
                <div class="evidence-slot">
                  <img src="{html.escape(str(example['original_with_token_box']))}" alt="original">
                  <div class="caption">Original + token</div>
                </div>
                {actmap_html}
                {support_html}
                {erf_zoom_html}
                {erf_zoom_detail_html}
                {heatmap_html}
              </div>
            </div>
            """
        )
    return "".join(cards)


def _batch_review_default_state(round_summary: dict[str, Any]) -> dict[str, Any]:
    reviewer_batch = dict(round_summary.get("reviewer_batch", {}))
    prompt_decisions = {
        str(row["proposal_id"]): ""
        for row in reviewer_batch.get("prompt_change_proposals", [])
    }
    viz_decisions = {
        str(row["proposal_id"]): ""
        for row in reviewer_batch.get("visualization_change_proposals", [])
    }
    return {
        "prompt_proposals": prompt_decisions,
        "visualization_proposals": viz_decisions,
        "batch_notes": "",
    }


def _feature_review_default_state(feature_response: dict[str, Any]) -> dict[str, Any]:
    return {
        "human_decision": "",
        "human_feedback": "",
        "canonical_label": "",
        "support_summary": "",
        "description": "",
        "notes": "",
        "gt_canonical_label": "",
        "gt_description": "",
        "gt_notes": "",
        "meta_tags": {tag: False for tag in META_TAGS},
        "meta_notes": "",
    }


def _build_human_review_initial_state(round_summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "session_name": str(round_summary["session_name"]),
        "round_index": int(round_summary["round_index"]),
        "saved_at": _now_iso(),
        "batch_review": _batch_review_default_state(round_summary),
        "items": {
            str(feature["feature_key"]): _feature_review_default_state(feature)
            for feature in round_summary["features"]
        },
    }
    for item in payload["items"].values():
        item["human_decision"] = ""
        item["human_feedback"] = ""
        item["canonical_label"] = ""
        item["support_summary"] = ""
        item["description"] = ""
        item["notes"] = ""
        item["gt_canonical_label"] = ""
        item["gt_description"] = ""
        item["gt_notes"] = ""
        item["meta_tags"] = {tag: False for tag in META_TAGS}
        item["meta_notes"] = ""
    return payload


def _reviewer_batch_panel(round_summary: dict[str, Any]) -> str:
    reviewer_batch = dict(round_summary.get("reviewer_batch", {}))

    def render_rows(scope_key: str, title: str) -> str:
        rows = reviewer_batch.get(scope_key, [])
        if not rows:
            return f"<div class='proposal-empty'>{html.escape(title)}: no proposals</div>"
        html_rows: list[str] = []
        decision_key = "prompt_proposals" if scope_key == "prompt_change_proposals" else "visualization_proposals"
        for row in rows:
            proposal_id = html.escape(str(row["proposal_id"]))
            html_rows.append(
                f"""
                <tr>
                  <td>{html.escape(str(row['label']))}</td>
                  <td><code>{html.escape(str(row['path']))}</code></td>
                  <td><code>{html.escape(json.dumps(row['proposed_value'], ensure_ascii=False))}</code></td>
                  <td>{html.escape(_norm_text(row.get('rationale')))}</td>
                  <td>
                    <select data-batch-proposal="true" data-decision-key="{decision_key}" data-proposal-id="{proposal_id}">
                      <option value="">pending</option>
                      <option value="approve">approve</option>
                      <option value="reject">reject</option>
                    </select>
                  </td>
                </tr>
                """
            )
        return f"""
        <div class="proposal-block">
          <h3>{html.escape(title)}</h3>
          <table class="proposal-table">
            <thead>
              <tr>
                <th>Label</th>
                <th>Path</th>
                <th>Value</th>
                <th>Rationale</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody>
              {''.join(html_rows)}
            </tbody>
          </table>
        </div>
        """

    return f"""
    <section class="batch-panel">
      <h2>Batch-level reviewer proposals</h2>
      <p>{html.escape(_norm_text(reviewer_batch.get('summary_rationale')) or 'No reviewer batch rationale provided.')}</p>
      <p class="muted">`Prompt changes` means reviewer suggestions about how the student / teacher / reviewer instructions should change next round. `Visualization changes` means reviewer suggestions about what evidence to show next round, such as ERF heatmap sidecar on/off or emphasis changes. These do not apply automatically; you approve or reject them here.</p>
      {render_rows('prompt_change_proposals', 'Prompt changes')}
      {render_rows('visualization_change_proposals', 'Visualization changes')}
      <label class="field">
        <div class="field-label">Batch notes</div>
        <textarea rows="4" data-batch-field="batch_notes" placeholder="Why are you approving or rejecting these proposals? Korean is okay."></textarea>
      </label>
    </section>
    """


def _agent_cards(feature: dict[str, Any]) -> str:
    student = dict(feature.get("student", {}))
    teacher = dict(feature.get("teacher", {}))
    reviewer_feature = dict(feature.get("reviewer_feature", {}))
    gt_html = ""
    if feature.get("provided_gt"):
        gt_html = f"<div class='agent-extra'><strong>Provided GT:</strong> {html.escape(json.dumps(feature['provided_gt'], ensure_ascii=False))}</div>"
    student_extra = ""
    if _norm_text(student.get("primary_locus")):
        student_extra += f"<div><strong>primary_locus:</strong> {html.escape(_norm_text(student.get('primary_locus')))}</div>"
    if _norm_text(student.get("adjacent_context")):
        student_extra += f"<div><strong>adjacent_context:</strong> {html.escape(_norm_text(student.get('adjacent_context')))}</div>"
    if _norm_text(student.get("support_summary")):
        student_extra += f"<div><strong>support_summary:</strong> {html.escape(_norm_text(student.get('support_summary')))}</div>"
    teacher_extra = ""
    if _norm_text(teacher.get("scope_assessment")):
        teacher_extra += f"<div><strong>scope_assessment:</strong> {html.escape(_norm_text(teacher.get('scope_assessment')))}</div>"
    if _norm_text(teacher.get("grounding_assessment")):
        teacher_extra += f"<div><strong>grounding_assessment:</strong> {html.escape(_norm_text(teacher.get('grounding_assessment')))}</div>"
    if _norm_text(teacher.get("adjacent_context_assessment")):
        teacher_extra += f"<div><strong>adjacent_context_assessment:</strong> {html.escape(_norm_text(teacher.get('adjacent_context_assessment')))}</div>"
    reviewer_extra = ""
    failure_tags = list(reviewer_feature.get("failure_tags") or [])
    if failure_tags:
        reviewer_extra += f"<div><strong>failure_tags:</strong> {html.escape(', '.join(str(tag) for tag in failure_tags))}</div>"
    teacher_feedback_html = ""
    for key in ("canonical_label_feedback", "description_feedback", "notes_feedback", "overall_feedback"):
        if _norm_text(teacher.get(key)):
            teacher_feedback_html += f"<div><strong>{html.escape(key)}:</strong> {html.escape(_norm_text(teacher.get(key)))}</div>"
    if not teacher_feedback_html:
        teacher_feedback_html = (
            f"<div><strong>feedback:</strong> {html.escape(_norm_text(teacher.get('feedback')))}</div>"
            f"<div><strong>field critique:</strong> {html.escape(_norm_text(teacher.get('field_critique')))}</div>"
        )
    reviewer_suggestion_html = ""
    prompt_suggestions = reviewer_feature.get("prompt_suggestions")
    visualization_suggestions = reviewer_feature.get("visualization_suggestions")
    if isinstance(prompt_suggestions, list) and prompt_suggestions:
        reviewer_suggestion_html += f"<div><strong>prompt_suggestions:</strong> {html.escape(' | '.join(str(item) for item in prompt_suggestions))}</div>"
    if isinstance(visualization_suggestions, list) and visualization_suggestions:
        reviewer_suggestion_html += f"<div><strong>visualization_suggestions:</strong> {html.escape(' | '.join(str(item) for item in visualization_suggestions))}</div>"
    if not reviewer_suggestion_html:
        reviewer_suggestion_html = (
            f"<div><strong>prompt suggestion:</strong> {html.escape(_norm_text(reviewer_feature.get('prompt_suggestion')))}</div>"
            f"<div><strong>visual suggestion:</strong> {html.escape(_norm_text(reviewer_feature.get('visualization_suggestion')))}</div>"
        )
    return f"""
    <div class="agent-grid">
      <section class="agent-card">
        <h3>Student</h3>
        {student_extra}
        <div><strong>canonical_label:</strong> {html.escape(_norm_text(student.get('canonical_label')))}</div>
        <div><strong>description:</strong> {html.escape(_norm_text(student.get('description')))}</div>
        <div><strong>notes:</strong> {html.escape(_norm_text(student.get('notes')))}</div>
        <div><strong>confidence:</strong> {html.escape(str(student.get('confidence', '')))}</div>
      </section>
      <section class="agent-card">
        <h3>Teacher</h3>
        <div><strong>verdict:</strong> {html.escape(_norm_text(teacher.get('verdict')))}</div>
        {teacher_extra}
        {teacher_feedback_html}
        {gt_html}
      </section>
      <section class="agent-card">
        <h3>Reviewer</h3>
        {reviewer_extra}
        <div><strong>diagnosis:</strong> {html.escape(_norm_text(reviewer_feature.get('diagnosis')))}</div>
        {reviewer_suggestion_html}
      </section>
    </div>
    """


def _human_review_controls(feature_key_value: str) -> str:
    decision_buttons = "".join(
        f'<button type="button" class="choice-btn" data-choice="true" data-target-kind="field" data-feature-key="{html.escape(feature_key_value)}" data-field="human_decision" data-choice-value="{value}">{html.escape(value)}</button>'
        for value in ("accept", "revise", "uninterpretable", "provide_gt", "skip")
    )
    meta_boxes = "".join(
        f"""
        <label class="checkbox">
          <input type="checkbox" data-meta-tag="true" data-feature-key="{html.escape(feature_key_value)}" data-tag="{html.escape(tag)}">
          <span>{html.escape(tag)}</span>
        </label>
        """
        for tag in META_TAGS
    )
    return f"""
    <section class="human-card">
      <h3>Human review</h3>
      <p class="muted">Main input should be free-form human feedback. Use English for <code>canonical_label</code> only if you choose to manually override it. Korean is okay for feedback, description, notes, GT fields, meta notes, and batch notes.</p>
      <label class="field">
        <div class="field-label">Decision</div>
        <input type="hidden" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="human_decision">
        <div class="choice-group">{decision_buttons}</div>
      </label>
      <label class="field">
        <div class="field-label">Human feedback</div>
        <textarea rows="5" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="human_feedback" placeholder="What is wrong or right about the current agent output? What should student / teacher / reviewer do differently next round? Korean is okay."></textarea>
      </label>
      <details class="optional-edit-block">
        <summary>Optional manual edits / GT</summary>
        <p class="muted">Leave this collapsed if you only want to give natural-language feedback. Open it only when you want to directly override label fields or provide explicit GT-like corrections.</p>
      <label class="field">
        <div class="field-label">Canonical label</div>
        <input type="text" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="canonical_label" placeholder="Short English semantic label">
      </label>
      <label class="field">
        <div class="field-label">Description</div>
        <textarea rows="4" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="description" placeholder="Semantic description. Korean is okay. Avoid saying 'ERF shows ...' here."></textarea>
      </label>
      <label class="field">
        <div class="field-label">Notes</div>
        <textarea rows="4" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="notes" placeholder="Ambiguity, ERF/actmap evidence, failure modes. Korean is okay."></textarea>
      </label>
      <div class="gt-grid">
        <label class="field">
          <div class="field-label">GT canonical label</div>
          <input type="text" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="gt_canonical_label" placeholder="Short English semantic label">
        </label>
        <label class="field">
          <div class="field-label">GT description</div>
          <textarea rows="3" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="gt_description" placeholder="Korean is okay."></textarea>
        </label>
        <label class="field">
          <div class="field-label">GT notes</div>
          <textarea rows="3" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="gt_notes" placeholder="Korean is okay."></textarea>
        </label>
      </div>
      <div class="meta-box">
        <div class="field-label">Meta tags</div>
        <div class="checkbox-grid">{meta_boxes}</div>
      </div>
      <label class="field">
        <div class="field-label">Meta notes</div>
        <textarea rows="3" data-feature-field="true" data-feature-key="{html.escape(feature_key_value)}" data-field="meta_notes" placeholder="Korean is okay."></textarea>
      </label>
      </details>
    </section>
    """


def _feature_review_sections(round_summary: dict[str, Any]) -> str:
    prompt_config = dict(round_summary["prompt_config"])
    sections: list[str] = []
    for feature in round_summary["features"]:
        feature_key_value = str(feature["feature_key"])
        chips = " ".join(
            f"<span class='chip'>{html.escape(str(key))}: {html.escape(str(value))}</span>"
            for key, value in {
                "block": feature["block_idx"],
                "feature_id": feature["feature_id"],
                "status": feature.get("previous_status", "awaiting_human"),
            }.items()
        )
        sections.append(
            f"""
            <article class="feature-item" data-feature-item="{html.escape(feature_key_value)}">
              <header class="feature-head">
                <h2>{html.escape(feature_key_value)}</h2>
                <div class="chip-row">{chips}</div>
              </header>
              {_agent_cards(feature)}
              <section class="evidence-panel">
                {_feature_examples_cards(feature, prompt_config)}
              </section>
              {_human_review_controls(feature_key_value)}
            </article>
            """
        )
    return "".join(sections)


def _build_human_review_html(
    *,
    round_summary: dict[str, Any],
    initial_state: dict[str, Any],
) -> str:
    payload = {
        "session_name": str(round_summary["session_name"]),
        "round_index": int(round_summary["round_index"]),
        "prompt_config": round_summary["prompt_config"],
        "initial_state": initial_state,
    }
    body = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autolabel Human Review</title>
  <style>
    :root {{
      --bg: #f3efe8;
      --panel: #fffdf8;
      --line: #d8d0c5;
      --ink: #181818;
      --muted: #6b645d;
      --accent: #b75a1f;
      --accent-soft: #f6e5d8;
      --danger: #8d2f2f;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f7f1e8 0%, #f0ebe2 100%);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .page {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .batch-panel, .feature-item, .toolbar {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.04);
    }}
    .hero, .batch-panel, .toolbar {{
      padding: 18px 20px;
      margin-bottom: 18px;
    }}
    .feature-item {{
      padding: 18px;
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px;
    }}
    .muted {{
      color: var(--muted);
      line-height: 1.5;
    }}
    .toolbar-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 10px;
    }}
    button, .toolbar .file-picker span, select {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 12px;
      cursor: pointer;
      font: inherit;
    }}
    .primary-btn {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
    }}
    .file-picker input {{
      display: none;
    }}
    .toolbar textarea, .human-card textarea, .human-card input[type="text"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fff;
      font: inherit;
      color: var(--ink);
    }}
    .toolbar textarea {{
      min-height: 84px;
      resize: vertical;
    }}
    .proposal-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }}
    .proposal-table th, .proposal-table td {{
      text-align: left;
      padding: 8px;
      border-bottom: 1px solid #ece7df;
      vertical-align: top;
      font-size: 14px;
    }}
    .proposal-empty {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 8px;
    }}
    .feature-head {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }}
    .chip-row {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    .chip {{
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #efc7ae;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
    }}
    .agent-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0 18px;
    }}
    .agent-card, .human-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 14px;
    }}
    .agent-card div {{
      margin-bottom: 8px;
      line-height: 1.45;
    }}
    .agent-extra {{
      font-size: 13px;
      color: var(--muted);
    }}
    .example-card {{
      border: 1px solid #e8e0d6;
      border-radius: 14px;
      background: #fff;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .example-head {{
      font-weight: 600;
      margin-bottom: 10px;
      font-size: 14px;
    }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .evidence-slot img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #111;
      border: 1px solid #ccc;
      border-radius: 10px;
    }}
    .caption {{
      margin-top: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    .field {{
      display: block;
      margin-bottom: 12px;
    }}
    .field-label {{
      display: block;
      margin-bottom: 4px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .choice-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .choice-btn {{
      border-radius: 999px;
      min-width: 56px;
      text-align: center;
    }}
    .choice-btn.is-active {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .gt-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .checkbox-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }}
    .checkbox {{
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 14px;
      color: var(--ink);
    }}
    .optional-edit-block {{
      border-top: 1px solid #ece7df;
      margin-top: 10px;
      padding-top: 10px;
    }}
    .optional-edit-block summary {{
      cursor: pointer;
      font-weight: 600;
      margin-bottom: 10px;
    }}
    @media (max-width: 1100px) {{
      .agent-grid, .gt-grid, .checkbox-grid, .evidence-grid {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 760px) {{
      .agent-grid, .gt-grid, .checkbox-grid, .evidence-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Autolabel Human Review</h1>
      <p class="muted">Review the agent outputs, give free-form human feedback, and decide whether each feature should be accepted, revised, marked uninterpretable, or skipped.</p>
      <p class="muted">Human feedback may be written in Korean. Only <code>canonical_label</code> and <code>gt_canonical_label</code> should stay short English semantic labels when you choose to edit them directly.</p>
      <p class="muted">This page tries server autosave when served over localhost and keeps export/import as fallback.</p>
    </section>
    <section class="toolbar">
      <div class="toolbar-row">
        <button type="button" id="btn-load-server">Load Server State</button>
        <button type="button" id="btn-load-local">Load Local</button>
        <button type="button" id="btn-reset">Reset</button>
        <button type="button" id="btn-export">Export JSON</button>
        <button type="button" id="btn-copy">Copy JSON</button>
        <button type="button" id="btn-submit" class="primary-btn">Submit to Server</button>
        <label class="file-picker">
          <input type="file" id="file-import" accept="application/json">
          <span>Import file</span>
        </label>
      </div>
      <textarea id="json-import" spellcheck="false" placeholder="Paste JSON here to import"></textarea>
      <div class="toolbar-row">
        <button type="button" id="btn-import">Import JSON</button>
        <span id="storage-status" class="status">Idle</span>
      </div>
    </section>
    {_reviewer_batch_panel(round_summary)}
    {_feature_review_sections(round_summary)}
  </div>
  <script>
    window.__AUTO_LABEL_SPEC__ = {json.dumps(payload, ensure_ascii=False)};
  </script>
  <script>
    (() => {{
      const SPEC = window.__AUTO_LABEL_SPEC__;
      const STORAGE_KEY = `autolabel.review.${{SPEC.session_name}}.round_${{SPEC.round_index}}.v2`;
      const DEFAULT_STATE = structuredClone(SPEC.initial_state);
      const STATUS = document.getElementById("storage-status");
      const IMPORT_BOX = document.getElementById("json-import");
      const IMPORT_FILE = document.getElementById("file-import");
      const IS_HTTP = window.location.protocol === "http:" || window.location.protocol === "https:";
      const SERVER_STATE_URL = IS_HTTP ? `${{window.location.origin}}/__state__` : "";
      const AUTOSAVE_URL = IS_HTTP ? `${{window.location.origin}}/__autosave__` : "";
      const SUBMIT_URL = IS_HTTP ? `${{window.location.origin}}/__submit__` : "";

      let state = structuredClone(DEFAULT_STATE);

      function setStatus(text) {{
        if (STATUS) STATUS.textContent = text;
      }}

      function ensureFeature(featureKey) {{
        if (!state.items) state.items = {{}};
        if (!state.items[featureKey]) state.items[featureKey] = {{
          human_decision: "",
          human_feedback: "",
          canonical_label: "",
          description: "",
          notes: "",
          gt_canonical_label: "",
          gt_description: "",
          gt_notes: "",
          meta_tags: {{}},
          meta_notes: "",
        }};
        if (!state.items[featureKey].meta_tags) state.items[featureKey].meta_tags = {{}};
        return state.items[featureKey];
      }}

      function ensureBatch() {{
        if (!state.batch_review) state.batch_review = {{prompt_proposals: {{}}, visualization_proposals: {{}}, batch_notes: ""}};
        if (!state.batch_review.prompt_proposals) state.batch_review.prompt_proposals = {{}};
        if (!state.batch_review.visualization_proposals) state.batch_review.visualization_proposals = {{}};
        return state.batch_review;
      }}

      function syncDomFromState() {{
        document.querySelectorAll("[data-feature-field='true']").forEach((el) => {{
          const featureKey = el.dataset.featureKey;
          const field = el.dataset.field;
          const value = ensureFeature(featureKey)[field];
          if (el.type === "hidden" || el.tagName === "TEXTAREA" || el.tagName === "INPUT") {{
            el.value = value ?? "";
          }}
        }});
        document.querySelectorAll("[data-meta-tag='true']").forEach((el) => {{
          const featureKey = el.dataset.featureKey;
          const tag = el.dataset.tag;
          el.checked = !!ensureFeature(featureKey).meta_tags?.[tag];
        }});
        document.querySelectorAll("[data-batch-field]").forEach((el) => {{
          const field = el.dataset.batchField;
          el.value = ensureBatch()[field] ?? "";
        }});
        document.querySelectorAll("[data-batch-proposal='true']").forEach((el) => {{
          const decisionKey = el.dataset.decisionKey;
          const proposalId = el.dataset.proposalId;
          const batch = ensureBatch();
          el.value = batch[decisionKey]?.[proposalId] ?? "";
        }});
        refreshChoiceGroups();
      }}

      function refreshChoiceGroups() {{
        document.querySelectorAll("[data-choice='true']").forEach((button) => {{
          const featureKey = button.dataset.featureKey;
          const field = button.dataset.field;
          const value = button.dataset.choiceValue ?? "";
          const current = ensureFeature(featureKey)[field] ?? "";
          button.classList.toggle("is-active", current === value);
        }});
      }}

      function exportJSON() {{
        return JSON.stringify({{
          session_name: SPEC.session_name,
          round_index: SPEC.round_index,
          exported_at: new Date().toISOString(),
          state,
        }}, null, 2);
      }}

      function persistLocal() {{
        localStorage.setItem(STORAGE_KEY, exportJSON());
      }}

      async function persistRemote(submit=false) {{
        if (!IS_HTTP) return false;
        const url = submit ? SUBMIT_URL : AUTOSAVE_URL;
        try {{
          const response = await fetch(url, {{
            method: "POST",
            headers: {{"Content-Type": "application/json"}},
            body: exportJSON(),
          }});
          if (!response.ok) {{
            throw new Error(`HTTP ${{response.status}}`);
          }}
          return true;
        }} catch (err) {{
          setStatus(`Server save failed: ${{err}}`);
          return false;
        }}
      }}

      async function saveNow(submit=false) {{
        persistLocal();
        const ok = await persistRemote(submit);
        if (submit) {{
          setStatus(ok ? "Submitted to server" : "Saved locally; server submit failed");
        }} else {{
          setStatus(ok ? "Saved locally + server" : "Saved locally");
        }}
      }}

      async function loadServerState() {{
        if (!IS_HTTP) {{
          setStatus("Not running over HTTP");
          return false;
        }}
        try {{
          const response = await fetch(SERVER_STATE_URL, {{method: "GET"}});
          if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
          const payload = await response.json();
          state = payload.state || payload;
          syncDomFromState();
          setStatus("Loaded server state");
          return true;
        }} catch (err) {{
          setStatus(`Server load failed: ${{err}}`);
          return false;
        }}
      }}

      function loadLocalState() {{
        try {{
          const raw = localStorage.getItem(STORAGE_KEY);
          if (!raw) {{
            setStatus("No local saved state");
            return false;
          }}
          const payload = JSON.parse(raw);
          state = payload.state || payload;
          syncDomFromState();
          setStatus("Loaded local state");
          return true;
        }} catch (err) {{
          setStatus(`Local load failed: ${{err}}`);
          return false;
        }}
      }}

      function importJSON(text) {{
        const payload = JSON.parse(text);
        state = payload.state || payload;
        syncDomFromState();
        saveNow(false);
      }}

      function download(filename, text) {{
        const blob = new Blob([text], {{type: "application/json"}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      }}

      document.addEventListener("input", (ev) => {{
        const el = ev.target;
        if (!(el instanceof HTMLElement)) return;
        if (el.matches("[data-feature-field='true']")) {{
          ensureFeature(el.dataset.featureKey)[el.dataset.field] = el.value;
          saveNow(false);
          return;
        }}
        if (el.matches("[data-batch-field]")) {{
          ensureBatch()[el.dataset.batchField] = el.value;
          saveNow(false);
        }}
      }});

      document.addEventListener("change", (ev) => {{
        const el = ev.target;
        if (!(el instanceof HTMLElement)) return;
        if (el.matches("[data-meta-tag='true']")) {{
          ensureFeature(el.dataset.featureKey).meta_tags[el.dataset.tag] = !!el.checked;
          saveNow(false);
          return;
        }}
        if (el.matches("[data-batch-proposal='true']")) {{
          const batch = ensureBatch();
          if (!batch[el.dataset.decisionKey]) batch[el.dataset.decisionKey] = {{}};
          batch[el.dataset.decisionKey][el.dataset.proposalId] = el.value;
          saveNow(false);
          return;
        }}
        if (el.matches("[data-feature-field='true']")) {{
          ensureFeature(el.dataset.featureKey)[el.dataset.field] = el.value;
          saveNow(false);
        }}
      }});

      document.addEventListener("click", (ev) => {{
        const el = ev.target;
        if (!(el instanceof HTMLElement)) return;
        if (!el.matches("[data-choice='true']")) return;
        ensureFeature(el.dataset.featureKey)[el.dataset.field] = el.dataset.choiceValue ?? "";
        refreshChoiceGroups();
        saveNow(false);
      }});

      document.getElementById("btn-load-server")?.addEventListener("click", () => loadServerState());
      document.getElementById("btn-load-local")?.addEventListener("click", () => loadLocalState());
      document.getElementById("btn-reset")?.addEventListener("click", () => {{
        state = structuredClone(DEFAULT_STATE);
        syncDomFromState();
        saveNow(false);
      }});
      document.getElementById("btn-export")?.addEventListener("click", () => {{
        download(`human_review_state.json`, exportJSON());
      }});
      document.getElementById("btn-copy")?.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(exportJSON());
          setStatus("Copied JSON");
        }} catch (err) {{
          setStatus(`Copy failed: ${{err}}`);
        }}
      }});
      document.getElementById("btn-import")?.addEventListener("click", () => {{
        try {{
          importJSON(IMPORT_BOX.value.trim());
        }} catch (err) {{
          setStatus(`Import failed: ${{err}}`);
        }}
      }});
      document.getElementById("btn-submit")?.addEventListener("click", async () => {{
        state.submitted_at = new Date().toISOString();
        await saveNow(true);
      }});
      if (IMPORT_FILE) IMPORT_FILE.addEventListener("change", async () => {{
        const file = IMPORT_FILE.files && IMPORT_FILE.files[0];
        if (!file) return;
        try {{
          const text = await file.text();
          importJSON(text);
        }} catch (err) {{
          setStatus(`File import failed: ${{err}}`);
        }}
      }});

      syncDomFromState();
      if (IS_HTTP) {{
        loadServerState().then((loaded) => {{
          if (!loaded && !loadLocalState()) {{
            saveNow(false);
          }}
        }});
      }} else if (!loadLocalState()) {{
        saveNow(false);
      }}
    }})();
  </script>
</body>
</html>
"""
    return body


def _session_html_path(config: EvalConfig, session_name: str, round_index: int) -> Path:
    return _round_dir(config, session_name, round_index) / "human_review.html"


def _session_state_path(config: EvalConfig, session_name: str, round_index: int) -> Path:
    return _round_dir(config, session_name, round_index) / "human_review_state.json"


def _session_round_policy_path(config: EvalConfig, session_name: str) -> Path:
    return _autolabel_session_dir(config, session_name) / "round_policy.json"


def _load_round_policy_if_exists(config: EvalConfig, session_name: str) -> dict[str, Any] | None:
    path = _session_round_policy_path(config, session_name)
    if not path.exists():
        return None
    return read_json(path)


def _phase_gate_root(config: EvalConfig, session_name: str) -> Path:
    return _autolabel_session_dir(config, session_name) / "phase_gates"


def _phase_gate_dir(config: EvalConfig, session_name: str, gate_id: str) -> Path:
    return _phase_gate_root(config, session_name) / str(gate_id)


def _phase_gate_manifest_path(config: EvalConfig, session_name: str, gate_id: str) -> Path:
    return _phase_gate_dir(config, session_name, gate_id) / "phase_gate_manifest.json"


def _phase_gate_summary_path(config: EvalConfig, session_name: str, gate_id: str) -> Path:
    return _phase_gate_dir(config, session_name, gate_id) / "phase_gate_summary.json"


def _phase_gate_state_path(config: EvalConfig, session_name: str, gate_id: str) -> Path:
    return _phase_gate_dir(config, session_name, gate_id) / "phase_gate_state.json"


def _phase_gate_html_path(config: EvalConfig, session_name: str, gate_id: str) -> Path:
    return _phase_gate_dir(config, session_name, gate_id) / "phase_gate.html"


def _phase_gate_submit_path(config: EvalConfig, session_name: str, gate_id: str) -> Path:
    return _phase_gate_dir(config, session_name, gate_id) / "phase_gate_submit.json"


def _active_phase_gate(manifest: dict[str, Any]) -> dict[str, Any] | None:
    return dict(manifest.get("phase_state", {}).get("active_phase_gate") or {}) or None


def _feature_source_type_for_agent(row: dict[str, Any]) -> str:
    if int(row.get("review_count") or 0) > 0 or row.get("last_human_round") is not None:
        return "carry_over"
    return "fresh"


def _infer_round_feature_source_type(feature: dict[str, Any], round_summary: dict[str, Any] | None = None) -> str:
    source_type = _norm_text(feature.get("source_type")).lower()
    if source_type in {"fresh", "carry_over"}:
        return source_type
    feature_key_value = str(feature.get("feature_key", ""))
    policy = dict((round_summary or {}).get("policy") or {})
    carry_over = {str(item) for item in list(policy.get("carry_over_features") or [])}
    active = {str(item) for item in list(policy.get("round_001_active_features") or [])}
    if feature_key_value and feature_key_value in carry_over:
        return "carry_over"
    if feature_key_value and feature_key_value in active:
        return "fresh"
    return "unknown"


def _decision_counts_template() -> dict[str, int]:
    return {
        "accept": 0,
        "revise": 0,
        "uninterpretable": 0,
        "provide_gt": 0,
        "skip": 0,
        "unknown": 0,
    }


def _decision_rate(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return float(count) / float(total)


def _backlog_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    fresh = 0
    carry_over = 0
    for row in rows:
        if str(row.get("status", "")) != "awaiting_agent":
            continue
        source_type = _feature_source_type_for_agent(row)
        if source_type == "carry_over":
            carry_over += 1
        else:
            fresh += 1
    return {
        "fresh_backlog_size": fresh,
        "carry_over_backlog_size": carry_over,
        "total_backlog_size": fresh + carry_over,
    }


def _extract_round_feature_records(
    *,
    config: EvalConfig,
    session_name: str,
    round_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    features = [dict(feature) for feature in list(round_summary.get("features") or [])]
    if all(_norm_text(feature.get("human_decision")) for feature in features):
        return features
    try:
        review_state = _load_human_review_state(config, session_name, int(round_summary["round_index"]))
    except FileNotFoundError:
        return features
    items = dict(review_state.get("items") or {})
    for feature in features:
        feature_key_value = str(feature.get("feature_key", ""))
        item_state = dict(items.get(feature_key_value) or {})
        if item_state:
            feature["human_decision"] = _norm_text(item_state.get("human_decision")).lower()
            feature["human_feedback"] = _norm_text(item_state.get("human_feedback"))
            feature["meta_tags"] = [tag for tag in META_TAGS if bool(item_state.get("meta_tags", {}).get(tag))]
            feature["meta_notes"] = _norm_text(item_state.get("meta_notes"))
        feature.setdefault("source_type", _infer_round_feature_source_type(feature, round_summary))
    return features


def _round_metrics_from_summary(
    *,
    config: EvalConfig,
    session_name: str,
    round_summary: dict[str, Any],
) -> dict[str, Any]:
    feature_records = _extract_round_feature_records(config=config, session_name=session_name, round_summary=round_summary)
    counts = _decision_counts_template()
    by_source = {
        "fresh": _decision_counts_template(),
        "carry_over": _decision_counts_template(),
        "unknown": _decision_counts_template(),
    }
    failure_hist = Counter()
    accepted_samples: list[dict[str, Any]] = []
    revised_samples: list[dict[str, Any]] = []
    for feature in feature_records:
        decision = _norm_text(feature.get("human_decision")).lower() or "unknown"
        if decision not in counts:
            decision = "unknown"
        source_type = _infer_round_feature_source_type(feature, round_summary)
        counts[decision] += 1
        by_source.setdefault(source_type, _decision_counts_template())
        by_source[source_type][decision] += 1
        if decision != "accept":
            for tag in list(feature.get("reviewer_feature", {}).get("failure_tags") or []):
                failure_hist[str(tag)] += 1
        sample_row = {
            "feature_key": str(feature.get("feature_key", "")),
            "source_type": source_type,
            "human_decision": decision,
            "student": dict(feature.get("student") or {}),
            "teacher": dict(feature.get("teacher") or {}),
            "reviewer_feature": dict(feature.get("reviewer_feature") or {}),
            "label_examples": [dict(example) for example in list(feature.get("label_examples") or [])],
        }
        if decision == "accept" and len(accepted_samples) < 3:
            accepted_samples.append(sample_row)
        if decision in {"revise", "uninterpretable"} and len(revised_samples) < 3:
            revised_samples.append(sample_row)
    fresh_total = sum(by_source["fresh"].values()) - by_source["fresh"]["unknown"]
    carry_over_total = sum(by_source["carry_over"].values()) - by_source["carry_over"]["unknown"]
    non_accept_total = counts["revise"] + counts["uninterpretable"]
    top_failure_share = None
    if non_accept_total and failure_hist:
        top_failure_share = max(int(value) for value in failure_hist.values()) / float(non_accept_total)
    return {
        "decision_counts": counts,
        "by_source": by_source,
        "fresh_accept_rate": _decision_rate(by_source["fresh"]["accept"], fresh_total),
        "fresh_revise_rate": _decision_rate(by_source["fresh"]["revise"], fresh_total),
        "fresh_uninterpretable_rate": _decision_rate(by_source["fresh"]["uninterpretable"], fresh_total),
        "carry_over_resolve_rate": _decision_rate(
            by_source["carry_over"]["accept"] + by_source["carry_over"]["uninterpretable"] + by_source["carry_over"]["skip"],
            carry_over_total,
        ),
        "carry_over_total": carry_over_total,
        "fresh_total": fresh_total,
        "failure_tag_histogram": dict(sorted(failure_hist.items(), key=lambda item: (-item[1], item[0]))),
        "top_failure_share": top_failure_share,
        "representative_accepts": accepted_samples,
        "representative_non_accepts": revised_samples,
    }


def _phase_gate_initial_state(manifest: dict[str, Any], gate_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_name": str(manifest["session_name"]),
        "track_id": str(manifest.get("track", {}).get("track_id") or manifest["session_name"]),
        "gate_id": str(gate_manifest["gate_id"]),
        "current_phase": str(gate_manifest["current_phase"]),
        "candidate_next_phase": str(gate_manifest["candidate_next_phase"]),
        "decision": "",
        "notes": "",
        "saved_at": _now_iso(),
    }


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _reviewed_round_summaries(config: EvalConfig, session_name: str) -> list[dict[str, Any]]:
    rounds_root = _autolabel_session_dir(config, session_name) / "rounds"
    if not rounds_root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for path in sorted(rounds_root.glob("round_*/round_summary.json")):
        summary = read_json(path)
        if str(summary.get("status", "")) != "human_review_completed":
            continue
        summaries.append(summary)
    return summaries


def _feature_card_image_relpath(gate_dir: Path, example: dict[str, Any], field: str) -> str:
    abs_field = f"{field}_abs"
    if _norm_text(example.get(abs_field)):
        return _relpath(gate_dir, Path(str(example[abs_field])))
    return str(example.get(field, ""))


def _representative_feature_card(gate_dir: Path, feature: dict[str, Any]) -> str:
    student = dict(feature.get("student") or {})
    reviewer = dict(feature.get("reviewer_feature") or {})
    examples = list(feature.get("label_examples") or [])
    example_html = ""
    if examples:
        example = dict(examples[0])
        image_src = _feature_card_image_relpath(gate_dir, example, "original_with_token_box")
        actmap_src = _feature_card_image_relpath(gate_dir, example, "feature_actmap")
        erf_src = _feature_card_image_relpath(gate_dir, example, "feature_erf_zoom")
        example_html = f"""
        <div class="gate-example-grid">
          <div class="gate-evidence-slot">
            <img src="{html.escape(image_src)}" alt="original">
            <div class="caption">Original + token</div>
          </div>
          <div class="gate-evidence-slot">
            <img src="{html.escape(actmap_src)}" alt="actmap">
            <div class="caption">Actmap (cyan = activation overlay only)</div>
          </div>
          <div class="gate-evidence-slot">
            <img src="{html.escape(erf_src)}" alt="erf-zoom">
            <div class="caption">ERF zoom (black outside support)</div>
          </div>
        </div>
        """
    return f"""
    <article class="gate-feature-card">
      <div class="chip-row">
        <span class="chip">{html.escape(str(feature.get('feature_key', '')))}</span>
        <span class="chip">{html.escape(str(feature.get('source_type', 'unknown')))}</span>
        <span class="chip">{html.escape(str(feature.get('human_decision', 'unknown')))}</span>
      </div>
      <div><strong>Label:</strong> {html.escape(_norm_text(student.get('canonical_label')))}</div>
      <div><strong>Notes:</strong> {html.escape(_norm_text(student.get('notes')))}</div>
      <div><strong>Reviewer tags:</strong> {html.escape(', '.join(str(tag) for tag in list(reviewer.get('failure_tags') or [])) or 'none')}</div>
      {example_html}
    </article>
    """


def _phase1_gate_summary(config: EvalConfig, session_name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = _load_feature_state(config, session_name)
    prompt_history = _load_prompt_history(config, session_name)
    reviewed_rounds = _reviewed_round_summaries(config, session_name)
    recent_rounds = reviewed_rounds[-5:]
    recent_metrics: list[dict[str, Any]] = []
    failure_hist = Counter()
    accepted_examples: list[dict[str, Any]] = []
    non_accept_examples: list[dict[str, Any]] = []
    for summary in recent_rounds:
        metrics = _round_metrics_from_summary(config=config, session_name=session_name, round_summary=summary)
        backlog = {
            "carry_over_backlog_size_after_round": summary.get("carry_over_backlog_size_after_round"),
            "fresh_backlog_size_after_round": summary.get("fresh_backlog_size_after_round"),
            "total_backlog_size_after_round": summary.get("total_backlog_size_after_round"),
        }
        recent_metrics.append(
            {
                "round_index": int(summary["round_index"]),
                "prompt_config_version": int(summary.get("prompt_config_version") or 0),
                "decision_counts": metrics["decision_counts"],
                "by_source": metrics["by_source"],
                "fresh_accept_rate": metrics["fresh_accept_rate"],
                "fresh_revise_rate": metrics["fresh_revise_rate"],
                "fresh_uninterpretable_rate": metrics["fresh_uninterpretable_rate"],
                "carry_over_resolve_rate": metrics["carry_over_resolve_rate"],
                **backlog,
            }
        )
        failure_hist.update(metrics["failure_tag_histogram"])
        for feature in metrics["representative_accepts"]:
            if len(accepted_examples) < 3:
                accepted_examples.append(feature)
        for feature in metrics["representative_non_accepts"]:
            if len(non_accept_examples) < 3:
                non_accept_examples.append(feature)
    plateau_ready = False
    plateau_reasons: list[str] = []
    if len(recent_metrics) >= 2:
        prev_round = recent_metrics[-2]
        last_round = recent_metrics[-1]
        fresh_accept_delta = None
        if prev_round["fresh_accept_rate"] is not None and last_round["fresh_accept_rate"] is not None:
            fresh_accept_delta = abs(float(last_round["fresh_accept_rate"]) - float(prev_round["fresh_accept_rate"]))
        carry_over_backlog_ok = False
        prev_backlog = prev_round.get("carry_over_backlog_size_after_round")
        last_backlog = last_round.get("carry_over_backlog_size_after_round")
        if isinstance(prev_backlog, int) and isinstance(last_backlog, int):
            carry_over_backlog_ok = last_backlog <= prev_backlog + 1
        current_non_accept_total = int(last_round["decision_counts"]["revise"]) + int(last_round["decision_counts"]["uninterpretable"])
        top_failure_share = None
        if current_non_accept_total and failure_hist:
            top_failure_share = max(int(value) for value in failure_hist.values()) / float(current_non_accept_total)
        plateau_ready = (
            fresh_accept_delta is not None
            and fresh_accept_delta <= PLATEAU_ACCEPT_DELTA_MAX
            and carry_over_backlog_ok
            and (top_failure_share is None or top_failure_share <= DOMINANT_FAILURE_SHARE_MAX)
        )
        plateau_reasons.extend(
            [
                f"fresh_accept_delta={_format_rate(fresh_accept_delta) if fresh_accept_delta is not None else 'n/a'}",
                f"carry_over_backlog_ok={carry_over_backlog_ok}",
                f"top_failure_share={_format_rate(top_failure_share)}",
            ]
        )
    else:
        plateau_reasons.append("Need at least two completed reviewed rounds to check plateau readiness.")
    backlog_metrics = _backlog_metrics(rows)
    return {
        "kind": "phase1_gate_summary",
        "current_phase": PHASE_PROMPT_STABILIZATION,
        "candidate_next_phase": PHASE_MODEL_SLIMMING,
        "track": dict(manifest.get("track") or {}),
        "prompt_version_history": [
            {
                "version": int(row.get("version") or 0),
                "created_at": _norm_text(row.get("created_at")),
                "reason": _norm_text(row.get("reason")),
            }
            for row in prompt_history
        ],
        "recent_round_metrics": recent_metrics,
        "top_failure_tags": dict(sorted(failure_hist.items(), key=lambda item: (-item[1], item[0]))[:10]),
        "representative_accepts": accepted_examples,
        "representative_non_accepts": non_accept_examples,
        "current_backlog": backlog_metrics,
        "plateau_readiness": {
            "ready": plateau_ready,
            "reasons": plateau_reasons,
        },
    }


def _normalize_phase2_gate_summary(manifest: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    baseline_model = _norm_text(payload.get("baseline_model_id")) or _norm_text(
        manifest.get("phase_state", {}).get("student_model", {}).get("model_id")
    )
    candidate_model = _norm_text(payload.get("candidate_model_id"))
    frozen_eval = dict(payload.get("frozen_eval") or {})
    baseline_metrics = dict(frozen_eval.get("baseline") or {})
    candidate_metrics = dict(frozen_eval.get("candidate") or {})
    baseline_accept = _safe_float(baseline_metrics.get("accept_rate"))
    candidate_accept = _safe_float(candidate_metrics.get("accept_rate"))
    baseline_non_accept = _safe_float(baseline_metrics.get("non_accept_rate"))
    if baseline_non_accept is None:
        baseline_non_accept = (_safe_float(baseline_metrics.get("revise_rate")) or 0.0) + (_safe_float(baseline_metrics.get("uninterpretable_rate")) or 0.0)
    candidate_non_accept = _safe_float(candidate_metrics.get("non_accept_rate"))
    if candidate_non_accept is None:
        candidate_non_accept = (_safe_float(candidate_metrics.get("revise_rate")) or 0.0) + (_safe_float(candidate_metrics.get("uninterpretable_rate")) or 0.0)
    if isinstance(payload.get("small_drop_pass"), bool):
        small_drop_pass = bool(payload.get("small_drop_pass"))
    else:
        small_drop_pass = (
            baseline_accept is not None
            and candidate_accept is not None
            and baseline_non_accept is not None
            and candidate_non_accept is not None
            and (baseline_accept - candidate_accept) <= MODEL_SMALL_DROP_MAX
            and (candidate_non_accept - baseline_non_accept) <= MODEL_SMALL_DROP_MAX
        )
    return {
        "kind": "phase2_gate_summary",
        "current_phase": PHASE_MODEL_SLIMMING,
        "candidate_next_phase": PHASE_AXIS_GUIDED,
        "baseline_model_id": baseline_model,
        "candidate_model_id": candidate_model,
        "frozen_eval": frozen_eval,
        "major_failure_tag_deltas": dict(payload.get("major_failure_tag_deltas") or {}),
        "representative_regressions": list(payload.get("representative_regressions") or []),
        "representative_preserved_wins": list(payload.get("representative_preserved_wins") or []),
        "small_drop_pass": bool(small_drop_pass),
        "source_payload_path": _norm_text(payload.get("source_payload_path")),
    }


def _normalize_phase3_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    promotion_ready = payload.get("promotion_ready")
    if not isinstance(promotion_ready, bool):
        promotion_ready = bool(payload.get("ordinary_label_quality", {}).get("non_regressive")) and bool(
            payload.get("axis_diagnostics", {}).get("recurring_tags")
        )
    return {
        "kind": "phase3_gate_summary",
        "current_phase": PHASE_AXIS_GUIDED,
        "candidate_next_phase": PHASE_GATE_PROMOTION,
        "axis_diagnostics": dict(payload.get("axis_diagnostics") or {}),
        "derived_prompt_changes": list(payload.get("derived_prompt_changes") or []),
        "ordinary_label_quality": dict(payload.get("ordinary_label_quality") or {}),
        "representative_cases": list(payload.get("representative_cases") or []),
        "promotion_ready": bool(promotion_ready),
        "source_payload_path": _norm_text(payload.get("source_payload_path")),
    }


def _phase_gate_metric_cards(summary: dict[str, Any]) -> str:
    if summary.get("kind") == "phase1_gate_summary":
        backlog = dict(summary.get("current_backlog") or {})
        plateau = dict(summary.get("plateau_readiness") or {})
        rounds = list(summary.get("recent_round_metrics") or [])
        last_round = rounds[-1] if rounds else {}
        cards = [
            ("Prompt version", str((summary.get("prompt_version_history") or [{}])[-1].get("version", "n/a"))),
            ("Fresh accept", _format_rate(last_round.get("fresh_accept_rate"))),
            ("Fresh revise", _format_rate(last_round.get("fresh_revise_rate"))),
            ("Carry-over resolve", _format_rate(last_round.get("carry_over_resolve_rate"))),
            ("Carry-over backlog", str(backlog.get("carry_over_backlog_size", "n/a"))),
            ("Plateau ready", "yes" if plateau.get("ready") else "no"),
        ]
    elif summary.get("kind") == "phase2_gate_summary":
        frozen_eval = dict(summary.get("frozen_eval") or {})
        baseline = dict(frozen_eval.get("baseline") or {})
        candidate = dict(frozen_eval.get("candidate") or {})
        cards = [
            ("Baseline model", str(summary.get("baseline_model_id") or "n/a")),
            ("Candidate model", str(summary.get("candidate_model_id") or "n/a")),
            ("Baseline accept", _format_rate(_safe_float(baseline.get("accept_rate")))),
            ("Candidate accept", _format_rate(_safe_float(candidate.get("accept_rate")))),
            ("Baseline non-accept", _format_rate(_safe_float(baseline.get("non_accept_rate")))),
            ("Small-drop pass", "yes" if summary.get("small_drop_pass") else "no"),
        ]
    else:
        ordinary = dict(summary.get("ordinary_label_quality") or {})
        axis_diag = dict(summary.get("axis_diagnostics") or {})
        cards = [
            ("Promotion ready", "yes" if summary.get("promotion_ready") else "no"),
            ("Axis recurring tags", str(len(axis_diag.get("recurring_tags") or []))),
            ("Derived prompt changes", str(len(summary.get("derived_prompt_changes") or []))),
            ("Ordinary label quality", json.dumps(ordinary, ensure_ascii=False)),
        ]
    return "".join(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in cards
    )


def _phase1_summary_sections(summary: dict[str, Any], gate_dir: Path) -> str:
    prompt_rows = "".join(
        f"""
        <tr>
          <td>{int(row.get('version') or 0)}</td>
          <td>{html.escape(_norm_text(row.get('reason')))}</td>
          <td>{html.escape(_norm_text(row.get('created_at')))}</td>
        </tr>
        """
        for row in list(summary.get("prompt_version_history") or [])
    )
    round_rows = "".join(
        f"""
        <tr>
          <td>{int(row.get('round_index') or 0)}</td>
          <td>{int(row.get('prompt_config_version') or 0)}</td>
          <td>{html.escape(json.dumps(row.get('decision_counts') or {}, ensure_ascii=False))}</td>
          <td>{_format_rate(row.get('fresh_accept_rate'))}</td>
          <td>{_format_rate(row.get('fresh_revise_rate'))}</td>
          <td>{_format_rate(row.get('carry_over_resolve_rate'))}</td>
          <td>{html.escape(str(row.get('carry_over_backlog_size_after_round', 'n/a')))}</td>
        </tr>
        """
        for row in list(summary.get("recent_round_metrics") or [])
    )
    failure_rows = "".join(
        f"<tr><td>{html.escape(str(tag))}</td><td>{int(count)}</td></tr>"
        for tag, count in dict(summary.get("top_failure_tags") or {}).items()
    ) or "<tr><td colspan='2'>No failure tags recorded.</td></tr>"
    plateau = dict(summary.get("plateau_readiness") or {})
    accept_cards = "".join(_representative_feature_card(gate_dir, row) for row in list(summary.get("representative_accepts") or []))
    non_accept_cards = "".join(_representative_feature_card(gate_dir, row) for row in list(summary.get("representative_non_accepts") or []))
    return f"""
    <section class="summary-panel">
      <h2>Prompt Version History</h2>
      <table class="summary-table">
        <thead><tr><th>Version</th><th>Reason</th><th>Created</th></tr></thead>
        <tbody>{prompt_rows}</tbody>
      </table>
    </section>
    <section class="summary-panel">
      <h2>Recent Round Metrics</h2>
      <table class="summary-table">
        <thead><tr><th>Round</th><th>Prompt</th><th>Decision counts</th><th>Fresh accept</th><th>Fresh revise</th><th>Carry-over resolve</th><th>Carry-over backlog</th></tr></thead>
        <tbody>{round_rows}</tbody>
      </table>
      <p class="muted">Plateau readiness: <strong>{'ready' if plateau.get('ready') else 'not ready'}</strong> | {html.escape(' | '.join(str(item) for item in list(plateau.get('reasons') or [])))}</p>
    </section>
    <section class="summary-panel">
      <h2>Top Failure Tags</h2>
      <table class="summary-table">
        <thead><tr><th>Tag</th><th>Count</th></tr></thead>
        <tbody>{failure_rows}</tbody>
      </table>
    </section>
    <section class="summary-panel">
      <h2>Representative Accepted Features</h2>
      <div class="feature-card-grid">{accept_cards or "<div class='muted'>No accepted representative features yet.</div>"}</div>
    </section>
    <section class="summary-panel">
      <h2>Representative Non-Accept Features</h2>
      <div class="feature-card-grid">{non_accept_cards or "<div class='muted'>No revised or uninterpretable representative features yet.</div>"}</div>
    </section>
    """


def _phase2_summary_sections(summary: dict[str, Any]) -> str:
    frozen_eval = dict(summary.get("frozen_eval") or {})
    baseline = dict(frozen_eval.get("baseline") or {})
    candidate = dict(frozen_eval.get("candidate") or {})
    delta_rows = "".join(
        f"<tr><td>{html.escape(str(tag))}</td><td>{html.escape(json.dumps(delta, ensure_ascii=False))}</td></tr>"
        for tag, delta in dict(summary.get("major_failure_tag_deltas") or {}).items()
    ) or "<tr><td colspan='2'>No failure-tag deltas supplied.</td></tr>"
    regressions = "".join(
        f"<li><strong>{html.escape(str(row.get('feature_key', '')))}</strong>: {html.escape(_norm_text(row.get('summary')))}</li>"
        for row in list(summary.get("representative_regressions") or [])
    ) or "<li>No representative regressions supplied.</li>"
    preserved = "".join(
        f"<li><strong>{html.escape(str(row.get('feature_key', '')))}</strong>: {html.escape(_norm_text(row.get('summary')))}</li>"
        for row in list(summary.get("representative_preserved_wins") or [])
    ) or "<li>No representative preserved wins supplied.</li>"
    return f"""
    <section class="summary-panel">
      <h2>Frozen Evaluation Comparison</h2>
      <table class="summary-table">
        <thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th></tr></thead>
        <tbody>
          <tr><td>accept_rate</td><td>{_format_rate(_safe_float(baseline.get('accept_rate')))}</td><td>{_format_rate(_safe_float(candidate.get('accept_rate')))}</td></tr>
          <tr><td>non_accept_rate</td><td>{_format_rate(_safe_float(baseline.get('non_accept_rate')))}</td><td>{_format_rate(_safe_float(candidate.get('non_accept_rate')))}</td></tr>
          <tr><td>revise_rate</td><td>{_format_rate(_safe_float(baseline.get('revise_rate')))}</td><td>{_format_rate(_safe_float(candidate.get('revise_rate')))}</td></tr>
          <tr><td>uninterpretable_rate</td><td>{_format_rate(_safe_float(baseline.get('uninterpretable_rate')))}</td><td>{_format_rate(_safe_float(candidate.get('uninterpretable_rate')))}</td></tr>
        </tbody>
      </table>
      <p class="muted">Small-drop pass: <strong>{'yes' if summary.get('small_drop_pass') else 'no'}</strong></p>
    </section>
    <section class="summary-panel">
      <h2>Failure Tag Deltas</h2>
      <table class="summary-table">
        <thead><tr><th>Tag</th><th>Delta</th></tr></thead>
        <tbody>{delta_rows}</tbody>
      </table>
    </section>
    <section class="summary-panel">
      <h2>Representative Regressions</h2>
      <ul>{regressions}</ul>
    </section>
    <section class="summary-panel">
      <h2>Representative Preserved Wins</h2>
      <ul>{preserved}</ul>
    </section>
    """


def _phase3_summary_sections(summary: dict[str, Any]) -> str:
    axis_diag = dict(summary.get("axis_diagnostics") or {})
    recurring_tags = "".join(
        f"<li><strong>{html.escape(str(item.get('tag', '')))}</strong>: {html.escape(_norm_text(item.get('summary')))}</li>"
        for item in list(axis_diag.get("recurring_tags") or [])
    ) or "<li>No recurring tags supplied.</li>"
    prompt_changes = "".join(
        f"<li><code>{html.escape(_norm_text(change.get('path')))}</code>: {html.escape(_norm_text(change.get('rationale')))}</li>"
        for change in list(summary.get("derived_prompt_changes") or [])
    ) or "<li>No derived prompt changes supplied.</li>"
    representative = "".join(
        f"<li><strong>{html.escape(str(row.get('feature_key', '')))}</strong>: {html.escape(_norm_text(row.get('summary')))}</li>"
        for row in list(summary.get("representative_cases") or [])
    ) or "<li>No representative cases supplied.</li>"
    return f"""
    <section class="summary-panel">
      <h2>Axis Diagnostics</h2>
      <ul>{recurring_tags}</ul>
    </section>
    <section class="summary-panel">
      <h2>Derived Prompt Changes</h2>
      <ul>{prompt_changes}</ul>
    </section>
    <section class="summary-panel">
      <h2>Ordinary Label Quality</h2>
      <pre>{html.escape(json.dumps(summary.get('ordinary_label_quality') or {}, ensure_ascii=False, indent=2))}</pre>
    </section>
    <section class="summary-panel">
      <h2>Representative Cases</h2>
      <ul>{representative}</ul>
    </section>
    """


def _phase_gate_decision_controls(current_phase: str) -> str:
    buttons = "".join(
        f'<button type="button" class="choice-btn" data-choice="true" data-choice-value="{html.escape(value)}">{html.escape(value)}</button>'
        for value in PHASE_GATE_DECISIONS.get(current_phase, ())
    )
    return f"""
    <section class="summary-panel">
      <h2>Human Gate Decision</h2>
      <label class="field">
        <div class="field-label">Decision</div>
        <input type="hidden" id="gate-decision" value="">
        <div class="choice-group">{buttons}</div>
      </label>
      <label class="field">
        <div class="field-label">Notes</div>
        <textarea id="gate-notes" rows="5" placeholder="Why are you advancing or holding this track? Korean is okay."></textarea>
      </label>
    </section>
    """


def _build_phase_gate_html(*, gate_manifest: dict[str, Any], gate_summary: dict[str, Any], initial_state: dict[str, Any], gate_dir: Path) -> str:
    if gate_summary.get("kind") == "phase1_gate_summary":
        summary_sections = _phase1_summary_sections(gate_summary, gate_dir)
    elif gate_summary.get("kind") == "phase2_gate_summary":
        summary_sections = _phase2_summary_sections(gate_summary)
    else:
        summary_sections = _phase3_summary_sections(gate_summary)
    payload = {
        "gate_manifest": gate_manifest,
        "gate_summary": gate_summary,
        "initial_state": initial_state,
    }
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Autolabel Phase Gate</title>
  <style>
    :root {{
      --bg: #f4f0ea;
      --panel: #fffdf8;
      --line: #d8d0c5;
      --ink: #181818;
      --muted: #6b645d;
      --accent: #245c73;
      --accent-soft: #dfeff5;
    }}
    body {{ margin: 0; background: linear-gradient(180deg, #f7f3ec 0%, #f0ebe2 100%); color: var(--ink); font-family: ui-sans-serif, system-ui, sans-serif; }}
    .page {{ max-width: 1420px; margin: 0 auto; padding: 24px; }}
    .hero, .toolbar, .summary-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }}
    .metric-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: #fff; }}
    .metric-label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric-value {{ margin-top: 6px; font-size: 18px; font-weight: 700; }}
    .summary-table {{ width: 100%; border-collapse: collapse; }}
    .summary-table th, .summary-table td {{ text-align: left; border-bottom: 1px solid #ece7df; padding: 8px; vertical-align: top; font-size: 14px; }}
    .toolbar-row {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }}
    button {{ border: 1px solid var(--line); background: #fff; color: var(--ink); border-radius: 10px; padding: 8px 12px; cursor: pointer; font: inherit; }}
    .primary-btn {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .choice-group {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .choice-btn.is-active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .field textarea, .toolbar textarea {{ width: 100%; box-sizing: border-box; border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; background: #fff; font: inherit; color: var(--ink); }}
    .toolbar textarea {{ min-height: 84px; resize: vertical; }}
    .field-label {{ display: block; margin-bottom: 4px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .status, .muted {{ color: var(--muted); }}
    .feature-card-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .gate-feature-card {{ border: 1px solid var(--line); border-radius: 14px; background: #fff; padding: 12px; }}
    .gate-example-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    .gate-evidence-slot img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #111; border: 1px solid #ccc; border-radius: 10px; }}
    .chip-row {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }}
    .chip {{ background: var(--accent-soft); color: var(--accent); border: 1px solid #bfd8e2; border-radius: 999px; padding: 4px 10px; font-size: 12px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
    @media (max-width: 1100px) {{ .metric-grid, .feature-card-grid, .gate-example-grid {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 760px) {{ .metric-grid, .feature-card-grid, .gate-example-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Autolabel Phase Gate</h1>
      <p class="muted">Track <strong>{html.escape(str(gate_manifest.get('track_id', '')))}</strong> | {html.escape(PHASE_LABELS.get(str(gate_manifest.get('current_phase')), str(gate_manifest.get('current_phase'))))} -> {html.escape(PHASE_LABELS.get(str(gate_manifest.get('candidate_next_phase')), str(gate_manifest.get('candidate_next_phase'))))}</p>
      <p class="muted">Use this page to decide whether this track is ready to move to the next phase. This is summary-first and separate from per-feature round review.</p>
      <div class="metric-grid">{_phase_gate_metric_cards(gate_summary)}</div>
    </section>
    <section class="toolbar">
      <div class="toolbar-row">
        <button type="button" id="btn-load-server">Load Server State</button>
        <button type="button" id="btn-load-local">Load Local</button>
        <button type="button" id="btn-reset">Reset</button>
        <button type="button" id="btn-export">Export JSON</button>
        <button type="button" id="btn-copy">Copy JSON</button>
        <button type="button" id="btn-submit" class="primary-btn">Submit to Server</button>
      </div>
      <textarea id="json-import" spellcheck="false" placeholder="Paste JSON here to import"></textarea>
      <div class="toolbar-row">
        <button type="button" id="btn-import">Import JSON</button>
        <span id="storage-status" class="status">Idle</span>
      </div>
    </section>
    {_phase_gate_decision_controls(str(gate_manifest.get("current_phase", "")))}
    {summary_sections}
  </div>
  <script>
    window.__AUTO_LABEL_PHASE_GATE__ = {json.dumps(payload, ensure_ascii=False)};
  </script>
  <script>
    (() => {{
      const SPEC = window.__AUTO_LABEL_PHASE_GATE__;
      const STORAGE_KEY = `autolabel.phase_gate.${{SPEC.gate_manifest.session_name}}.${{SPEC.gate_manifest.gate_id}}`;
      const DEFAULT_STATE = structuredClone(SPEC.initial_state);
      const STATUS = document.getElementById("storage-status");
      const IMPORT_BOX = document.getElementById("json-import");
      const IS_HTTP = window.location.protocol === "http:" || window.location.protocol === "https:";
      const SERVER_STATE_URL = IS_HTTP ? `${{window.location.origin}}/__state__` : "";
      const AUTOSAVE_URL = IS_HTTP ? `${{window.location.origin}}/__autosave__` : "";
      const SUBMIT_URL = IS_HTTP ? `${{window.location.origin}}/__submit__` : "";
      let state = structuredClone(DEFAULT_STATE);
      function setStatus(text) {{ if (STATUS) STATUS.textContent = text; }}
      function syncDom() {{
        document.getElementById("gate-decision").value = state.decision || "";
        document.getElementById("gate-notes").value = state.notes || "";
        document.querySelectorAll("[data-choice='true']").forEach((button) => {{
          button.classList.toggle("is-active", (state.decision || "") === (button.dataset.choiceValue || ""));
        }});
      }}
      function exportJSON() {{
        return JSON.stringify({{
          session_name: SPEC.gate_manifest.session_name,
          gate_id: SPEC.gate_manifest.gate_id,
          exported_at: new Date().toISOString(),
          state,
        }}, null, 2);
      }}
      function persistLocal() {{ localStorage.setItem(STORAGE_KEY, exportJSON()); }}
      async function persistRemote(submit=false) {{
        if (!IS_HTTP) return false;
        const url = submit ? SUBMIT_URL : AUTOSAVE_URL;
        try {{
          const response = await fetch(url, {{ method: "POST", headers: {{"Content-Type": "application/json"}}, body: exportJSON() }});
          if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
          return true;
        }} catch (err) {{
          setStatus(`Server save failed: ${{err}}`);
          return false;
        }}
      }}
      async function saveNow(submit=false) {{
        persistLocal();
        const ok = await persistRemote(submit);
        setStatus(submit ? (ok ? "Submitted to server" : "Saved locally; server submit failed") : (ok ? "Saved locally + server" : "Saved locally"));
      }}
      function loadLocal() {{
        try {{
          const raw = localStorage.getItem(STORAGE_KEY);
          if (!raw) return false;
          const payload = JSON.parse(raw);
          state = payload.state || payload;
          syncDom();
          setStatus("Loaded local state");
          return true;
        }} catch (err) {{
          setStatus(`Local load failed: ${{err}}`);
          return false;
        }}
      }}
      async function loadServer() {{
        if (!IS_HTTP) return false;
        try {{
          const response = await fetch(SERVER_STATE_URL, {{method: "GET"}});
          if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
          const payload = await response.json();
          state = payload.state || payload;
          syncDom();
          setStatus("Loaded server state");
          return true;
        }} catch (err) {{
          setStatus(`Server load failed: ${{err}}`);
          return false;
        }}
      }}
      function importJSON(text) {{
        const payload = JSON.parse(text);
        state = payload.state || payload;
        syncDom();
        saveNow(false);
      }}
      document.getElementById("gate-notes")?.addEventListener("input", (ev) => {{
        state.notes = ev.target.value;
        saveNow(false);
      }});
      document.addEventListener("click", (ev) => {{
        const el = ev.target;
        if (!(el instanceof HTMLElement)) return;
        if (el.matches("[data-choice='true']")) {{
          state.decision = el.dataset.choiceValue || "";
          syncDom();
          saveNow(false);
        }}
      }});
      document.getElementById("btn-load-server")?.addEventListener("click", () => loadServer());
      document.getElementById("btn-load-local")?.addEventListener("click", () => loadLocal());
      document.getElementById("btn-reset")?.addEventListener("click", () => {{
        state = structuredClone(DEFAULT_STATE);
        syncDom();
        saveNow(false);
      }});
      document.getElementById("btn-export")?.addEventListener("click", () => {{
        const blob = new Blob([exportJSON()], {{type: "application/json"}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "phase_gate_state.json";
        a.click();
        URL.revokeObjectURL(url);
      }});
      document.getElementById("btn-copy")?.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(exportJSON());
          setStatus("Copied JSON");
        }} catch (err) {{
          setStatus(`Copy failed: ${{err}}`);
        }}
      }});
      document.getElementById("btn-import")?.addEventListener("click", () => {{
        try {{
          importJSON(IMPORT_BOX.value.trim());
        }} catch (err) {{
          setStatus(`Import failed: ${{err}}`);
        }}
      }});
      document.getElementById("btn-submit")?.addEventListener("click", async () => {{
        state.submitted_at = new Date().toISOString();
        await saveNow(true);
      }});
      syncDom();
      if (IS_HTTP) {{
        loadServer().then((loaded) => {{
          if (!loaded && !loadLocal()) saveNow(false);
        }});
      }} else if (!loadLocal()) {{
        saveNow(false);
      }}
    }})();
  </script>
</body>
</html>
"""


def build_autolabel_phase_gate(
    config: EvalConfig,
    *,
    session_name: str,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    phase_state = dict(manifest.get("phase_state") or {})
    active_gate = _active_phase_gate(manifest)
    if active_gate and str(active_gate.get("status", "")) == "awaiting_human":
        gate_id = str(active_gate["gate_id"])
        return {
            "session_name": session_name,
            "gate_id": gate_id,
            "phase_gate_html": str(_phase_gate_html_path(config, session_name, gate_id)),
            "phase_gate_state_json": str(_phase_gate_state_path(config, session_name, gate_id)),
            "phase_gate_manifest_json": str(_phase_gate_manifest_path(config, session_name, gate_id)),
            "phase_gate_summary_json": str(_phase_gate_summary_path(config, session_name, gate_id)),
        }
    current_phase = str(phase_state.get("current_phase") or PHASE_PROMPT_STABILIZATION)
    gate_counter = int(phase_state.get("phase_gate_counter") or 0) + 1
    candidate_next_phase = PHASE_CANDIDATE_NEXT[current_phase]
    gate_id = f"gate_{gate_counter:03d}_{current_phase}__to__{candidate_next_phase}"
    gate_dir = _phase_gate_dir(config, session_name, gate_id)
    gate_dir.mkdir(parents=True, exist_ok=True)

    if current_phase == PHASE_PROMPT_STABILIZATION:
        gate_summary = _phase1_gate_summary(config, session_name, manifest)
    else:
        if summary_json is None:
            raise FileNotFoundError("Phase gate for this phase requires --response-json with the evaluation summary payload.")
        payload = read_json(Path(summary_json))
        payload["source_payload_path"] = str(Path(summary_json))
        if current_phase == PHASE_MODEL_SLIMMING:
            gate_summary = _normalize_phase2_gate_summary(manifest, payload)
        else:
            gate_summary = _normalize_phase3_gate_summary(payload)

    gate_manifest = {
        "kind": "autolabel_phase_gate",
        "gate_id": gate_id,
        "session_name": session_name,
        "track_id": str(manifest.get("track", {}).get("track_id") or session_name),
        "created_at": _now_iso(),
        "status": "awaiting_human",
        "current_phase": current_phase,
        "candidate_next_phase": candidate_next_phase,
        "prompt_config_version": int(manifest.get("current_prompt_config_version") or 0),
        "student_model_id": _norm_text(phase_state.get("student_model", {}).get("model_id")),
        "supporting_rounds": [int(row.get("round_index") or 0) for row in _reviewed_round_summaries(config, session_name)],
        "frozen_eval_references": list(gate_summary.get("frozen_eval", {}).get("artifacts") or []),
    }
    initial_state = _phase_gate_initial_state(manifest, gate_manifest)
    html_payload = _build_phase_gate_html(
        gate_manifest=gate_manifest,
        gate_summary=gate_summary,
        initial_state=initial_state,
        gate_dir=gate_dir,
    )
    write_json(_phase_gate_manifest_path(config, session_name, gate_id), gate_manifest)
    write_json(_phase_gate_summary_path(config, session_name, gate_id), gate_summary)
    write_json(_phase_gate_state_path(config, session_name, gate_id), initial_state)
    _phase_gate_html_path(config, session_name, gate_id).write_text(html_payload)

    phase_state["phase_gate_counter"] = gate_counter
    phase_state["active_phase_gate"] = {
        "gate_id": gate_id,
        "status": "awaiting_human",
        "current_phase": current_phase,
        "candidate_next_phase": candidate_next_phase,
        "created_at": gate_manifest["created_at"],
    }
    manifest["phase_state"] = phase_state
    _write_session_manifest(config, session_name, manifest)
    return {
        "session_name": session_name,
        "gate_id": gate_id,
        "phase_gate_html": str(_phase_gate_html_path(config, session_name, gate_id)),
        "phase_gate_state_json": str(_phase_gate_state_path(config, session_name, gate_id)),
        "phase_gate_manifest_json": str(_phase_gate_manifest_path(config, session_name, gate_id)),
        "phase_gate_summary_json": str(_phase_gate_summary_path(config, session_name, gate_id)),
    }


def _load_phase_gate_state(config: EvalConfig, session_name: str, gate_id: str) -> dict[str, Any]:
    path = _phase_gate_state_path(config, session_name, gate_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing phase gate state at {path}")
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        return dict(payload["state"])
    return payload


def apply_autolabel_phase_gate(
    config: EvalConfig,
    *,
    session_name: str,
    gate_id: str | None = None,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    active_gate = _active_phase_gate(manifest)
    if gate_id is None:
        if not active_gate:
            raise RuntimeError(f"No active phase gate for session {session_name}")
        gate_id = str(active_gate["gate_id"])
    gate_manifest = read_json(_phase_gate_manifest_path(config, session_name, gate_id))
    gate_summary = read_json(_phase_gate_summary_path(config, session_name, gate_id))
    gate_state = _load_phase_gate_state(config, session_name, gate_id)
    decision = _norm_text(gate_state.get("decision")).lower()
    allowed = set(PHASE_GATE_DECISIONS.get(str(gate_manifest.get("current_phase")), ()))
    if decision not in allowed:
        raise RuntimeError(f"Phase gate decision must be one of: {', '.join(sorted(allowed))}")

    phase_state = dict(manifest.get("phase_state") or {})
    current_phase = str(gate_manifest["current_phase"])
    next_phase = str(gate_manifest["candidate_next_phase"])
    student_model = dict(phase_state.get("student_model") or {})
    if current_phase == PHASE_PROMPT_STABILIZATION and decision == "advance_to_phase_2":
        if not bool(gate_summary.get("plateau_readiness", {}).get("ready")):
            raise RuntimeError("Phase 1 gate cannot advance because plateau readiness is not satisfied.")
        phase_state["current_phase"] = PHASE_MODEL_SLIMMING
        phase_state.setdefault("phase_baselines", {})
        phase_state["phase_baselines"][PHASE_PROMPT_STABILIZATION] = {
            "approved_prompt_config_version": int(manifest.get("current_prompt_config_version") or 0),
            "approved_at": _now_iso(),
            "gate_id": gate_id,
        }
    elif current_phase == PHASE_MODEL_SLIMMING:
        if decision == "accept_lighter_model_and_advance_to_phase_3":
            if not bool(gate_summary.get("small_drop_pass")):
                raise RuntimeError("Phase 2 gate cannot accept the lighter model because the small-drop rule did not pass.")
            student_model["model_id"] = _norm_text(gate_summary.get("candidate_model_id")) or student_model.get("model_id")
            phase_state["current_phase"] = PHASE_AXIS_GUIDED
        elif decision == "keep_stronger_model_and_advance_to_phase_3":
            phase_state["current_phase"] = PHASE_AXIS_GUIDED
        phase_state.setdefault("phase_baselines", {})
        phase_state["phase_baselines"][PHASE_MODEL_SLIMMING] = {
            "baseline_model_id": _norm_text(gate_summary.get("baseline_model_id")),
            "candidate_model_id": _norm_text(gate_summary.get("candidate_model_id")),
            "selected_model_id": student_model.get("model_id"),
            "approved_at": _now_iso(),
            "gate_id": gate_id,
            "decision": decision,
        }
    elif current_phase == PHASE_AXIS_GUIDED and decision == "accept_axis_guided_prompt_update":
        if not bool(gate_summary.get("promotion_ready")):
            raise RuntimeError("Phase 3 gate cannot accept the axis-guided prompt update because promotion readiness is false.")
        phase_state.setdefault("phase_baselines", {})
        phase_state["phase_baselines"][PHASE_AXIS_GUIDED] = {
            "accepted_prompt_config_version": int(manifest.get("current_prompt_config_version") or 0),
            "approved_at": _now_iso(),
            "gate_id": gate_id,
        }

    phase_state["student_model"] = student_model
    phase_state["history"] = list(phase_state.get("history") or [])
    phase_state["history"].append(
        {
            "gate_id": gate_id,
            "reviewed_at": _now_iso(),
            "current_phase": current_phase,
            "candidate_next_phase": next_phase,
            "decision": decision,
            "notes": _norm_text(gate_state.get("notes")),
            "prompt_config_version": int(manifest.get("current_prompt_config_version") or 0),
            "student_model_id": _norm_text(student_model.get("model_id")),
        }
    )
    phase_state["active_phase_gate"] = None
    manifest["phase_state"] = phase_state
    _write_session_manifest(config, session_name, manifest)

    gate_manifest["status"] = "completed"
    gate_manifest["decision"] = decision
    gate_manifest["notes"] = _norm_text(gate_state.get("notes"))
    gate_manifest["reviewed_at"] = _now_iso()
    write_json(_phase_gate_manifest_path(config, session_name, gate_id), gate_manifest)
    return {
        "session_name": session_name,
        "gate_id": gate_id,
        "decision": decision,
        "current_phase": phase_state.get("current_phase"),
        "student_model_id": student_model.get("model_id"),
    }


class _AutolabelReviewHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        directory: str,
        state_path: Path,
        submit_path: Path,
        **kwargs,
    ) -> None:
        self._state_path = Path(state_path)
        self._submit_path = Path(submit_path)
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self.path = "/human_review.html"
            return super().do_GET()
        if parsed.path == "/__state__":
            payload = _read_json_if_exists(self._state_path, {})
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return
        if parsed.path == "/__autosave__":
            write_json(self._state_path, payload)
            self._json_ok({"status": "saved", "state_path": str(self._state_path)})
            return
        if parsed.path == "/__submit__":
            write_json(self._state_path, payload)
            write_json(self._submit_path, {"submitted_at": _now_iso(), "payload": payload})
            self._json_ok({"status": "submitted", "submit_path": str(self._submit_path)})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _json_ok(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


class _AutolabelPhaseGateHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        directory: str,
        state_path: Path,
        submit_path: Path,
        **kwargs,
    ) -> None:
        self._state_path = Path(state_path)
        self._submit_path = Path(submit_path)
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"", "/"}:
            self.path = "/phase_gate.html"
            return super().do_GET()
        if parsed.path == "/__state__":
            payload = _read_json_if_exists(self._state_path, {})
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return
        if parsed.path == "/__autosave__":
            write_json(self._state_path, payload)
            self._json_ok({"status": "saved", "state_path": str(self._state_path)})
            return
        if parsed.path == "/__submit__":
            write_json(self._state_path, payload)
            write_json(self._submit_path, {"submitted_at": _now_iso(), "payload": payload})
            self._json_ok({"status": "submitted", "submit_path": str(self._submit_path)})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _json_ok(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def start_autolabel_review_server(
    config: EvalConfig,
    *,
    session_name: str,
    round_index: int,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[ThreadingHTTPServer, str]:
    round_dir = _round_dir(config, session_name, round_index)
    html_path = round_dir / "human_review.html"
    if not html_path.exists():
        raise FileNotFoundError(f"Missing human review page at {html_path}. Ingest the round first.")
    session_dir = _autolabel_session_dir(config, session_name)
    handler = partial(
        _AutolabelReviewHandler,
        directory=str(session_dir),
        state_path=_session_state_path(config, session_name, round_index),
        submit_path=round_dir / "human_review_submit.json",
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/rounds/round_{int(round_index):03d}/human_review.html"
    return server, url


def serve_autolabel_session(
    config: EvalConfig,
    *,
    session_name: str,
    round_index: int | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    if round_index is None:
        round_index = int(manifest.get("latest_human_round") or manifest.get("latest_agent_round") or manifest["next_round_index"])
    server, url = start_autolabel_review_server(
        config,
        session_name=session_name,
        round_index=int(round_index),
        host=host,
        port=port,
    )
    try:
        print(f"[autolabel-server] serving {url}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
    return {"url": url, "host": host, "port": port, "round_index": int(round_index)}


def start_autolabel_phase_gate_server(
    config: EvalConfig,
    *,
    session_name: str,
    gate_id: str,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> tuple[ThreadingHTTPServer, str]:
    html_path = _phase_gate_html_path(config, session_name, gate_id)
    if not html_path.exists():
        raise FileNotFoundError(f"Missing phase gate page at {html_path}. Build the phase gate first.")
    session_dir = _autolabel_session_dir(config, session_name)
    handler = partial(
        _AutolabelPhaseGateHandler,
        directory=str(session_dir),
        state_path=_phase_gate_state_path(config, session_name, gate_id),
        submit_path=_phase_gate_submit_path(config, session_name, gate_id),
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/phase_gates/{gate_id}/phase_gate.html"
    return server, url


def serve_autolabel_phase_gate(
    config: EvalConfig,
    *,
    session_name: str,
    gate_id: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    active_gate = _active_phase_gate(manifest)
    if gate_id is None:
        if not active_gate:
            raise RuntimeError(f"No active phase gate for session {session_name}")
        gate_id = str(active_gate["gate_id"])
    server, url = start_autolabel_phase_gate_server(
        config,
        session_name=session_name,
        gate_id=str(gate_id),
        host=host,
        port=port,
    )
    try:
        print(f"[autolabel-phase-gate] serving {url}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
    return {"url": url, "host": host, "port": port, "gate_id": str(gate_id)}


def build_autolabel_session(
    config: EvalConfig,
    *,
    session_name: str,
    features_per_block: int | None = None,
    seed: int | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    session_dir = _autolabel_session_dir(config, session_name)
    manifest_path = session_dir / "session_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Autolabel session already exists at {manifest_path}")
    session_dir.mkdir(parents=True, exist_ok=True)

    feature_bank = load_feature_bank(config)
    features_per_block = int(features_per_block or config.autolabel_session_default_features_per_block)
    seed = int(seed if seed is not None else config.autolabel_session_default_seed)
    if feature_specs:
        features_by_key = {
            (int(feature["block_idx"]), int(feature["feature_id"])): feature
            for block_payload in feature_bank["blocks"].values()
            for feature in block_payload["features"]
        }
        selected_features: list[dict[str, Any]] = []
        missing: list[str] = []
        for block_idx, feature_id in feature_specs:
            feature = features_by_key.get((int(block_idx), int(feature_id)))
            if feature is None:
                missing.append(f"{int(block_idx)}:{int(feature_id)}")
                continue
            selected_features.append(feature)
        if missing:
            raise ValueError(f"Unknown feature specs for autolabel session: {', '.join(missing)}")
        selection_diagnostics = {
            "explicit_feature_specs": [f"{int(block_idx)}:{int(feature_id)}" for block_idx, feature_id in feature_specs],
            "selected": len(selected_features),
        }
    else:
        selected_features, selection_diagnostics = _select_features_for_label_session(
            config,
            feature_bank,
            features_per_block=features_per_block,
            seed=seed,
        )
    prompt_config = _default_prompt_config()

    runtime = LegacyRuntime(config)
    frame_cache: dict[int, Any] = {}
    sid_to_path_cache: dict[int, str] = {}
    feature_pool_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    try:
        total = len(selected_features)
        for idx, feature in enumerate(selected_features, start=1):
            feature_pool_rows.append(
                _render_autolabel_feature_pool_row(
                    config=config,
                    session_dir=session_dir,
                    runtime=runtime,
                    frame_cache=frame_cache,
                    sid_to_path_cache=sid_to_path_cache,
                    feature=feature,
                )
            )
            state_rows.append(_initial_feature_state_row(feature, prompt_config_version=int(prompt_config["version"])))
            print(
                f"[autolabel-session {idx:03d}/{total:03d}] block={int(feature['block_idx'])} feature={int(feature['feature_id'])}",
                flush=True,
            )
    finally:
        runtime.close()

    feature_pool_payload = {
        "session_name": session_name,
        "created_at": _now_iso(),
        "features": feature_pool_rows,
    }
    write_json(session_dir / "feature_pool.json", feature_pool_payload)
    _write_current_prompt_config(config, session_name, prompt_config)
    _write_prompt_history(
        config,
        session_name,
        [
            {
                "version": int(prompt_config["version"]),
                "created_at": _now_iso(),
                "reason": "session_init",
                "prompt_config": prompt_config,
                "approved_prompt_proposals": [],
                "approved_visualization_proposals": [],
            }
        ],
    )
    _write_feature_state(config, session_name, state_rows)
    manifest = {
        "kind": "autolabel_session",
        "session_name": session_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "seed": seed,
        "features_per_block": features_per_block,
        "label_examples_per_feature": int(config.study_label_examples_per_feature),
        "label_examples_per_feature_by_block": dict(config.autolabel_label_examples_by_block),
        "selection_diagnostics": selection_diagnostics,
        "feature_count": len(feature_pool_rows),
        "status_counts": _status_counts(state_rows),
        "current_prompt_config_version": int(prompt_config["version"]),
        "next_round_index": 1,
        "latest_agent_round": None,
        "latest_human_round": None,
    }
    manifest = _with_session_defaults(manifest)
    _write_session_manifest(config, session_name, manifest)
    return {
        **manifest,
        "session_dir": str(session_dir),
        "feature_pool_json": str(session_dir / "feature_pool.json"),
        "current_prompt_config_json": str(session_dir / "current_prompt_config.json"),
        "feature_state_jsonl": str(session_dir / "feature_state.jsonl"),
        "prompt_history_jsonl": str(session_dir / "prompt_history.jsonl"),
    }


def build_autolabel_round_packet(
    config: EvalConfig,
    *,
    session_name: str,
    round_index: int | None = None,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    session_dir = _autolabel_session_dir(config, session_name)
    feature_pool = _load_feature_pool(config, session_name)
    prompt_config = _load_current_prompt_config(config, session_name)
    round_policy = _load_round_policy_if_exists(config, session_name)
    state_map = _feature_state_map(config, session_name)
    prompt_history = _load_prompt_history(config, session_name)
    if round_index is None:
        round_index = int(manifest["next_round_index"])
    round_dir = _round_dir(config, session_name, int(round_index))
    round_dir.mkdir(parents=True, exist_ok=True)

    feature_lookup = {str(row["feature_key"]): row for row in feature_pool["features"]}
    pending_rows = [row for row in state_map.values() if str(row.get("status", "")) == "awaiting_agent"]
    if not pending_rows:
        raise RuntimeError(f"No awaiting_agent features remain in autolabel session {session_name}")
    prompt_flags = dict(prompt_config.get("prompt", {}))
    features_payload: list[dict[str, Any]] = []
    for row in pending_rows:
        source_type = _feature_source_type_for_agent(row)
        if row.get("first_agent_round") is None:
            row["first_agent_round"] = int(round_index)
        row["last_source_type"] = source_type
        feature_key_value = str(row["feature_key"])
        pool_row = feature_lookup[feature_key_value]
        feature_payload = {
            "feature_key": feature_key_value,
            "block_idx": int(pool_row["block_idx"]),
            "feature_id": int(pool_row["feature_id"]),
            "source_type": source_type,
            "label_examples": _student_visible_label_examples(list(pool_row["label_examples"]), prompt_config),
            "prior_review_metadata": _student_visible_review_metadata(row, prompt_config),
            "prior_failure_tags": _prior_failure_tags(row),
            "provided_gt": dict(row.get("provided_gt") or {}),
            "meta_tags": list(row.get("meta_tags") or []),
            "meta_notes": _norm_text(row.get("meta_notes")),
        }
        if not prompt_flags.get("student_blind_to_prior_semantic_feedback"):
            feature_payload["latest_teacher"] = dict(row.get("latest_teacher") or {})
            feature_payload["latest_reviewer_feature"] = dict(row.get("latest_reviewer_feature") or {})
        features_payload.append(feature_payload)
    _write_feature_state(config, session_name, list(state_map.values()))

    packet = {
        "kind": "autolabel_agent_packet",
        "session_name": session_name,
        "round_index": int(round_index),
        "built_at": _now_iso(),
        "prompt_config": prompt_config,
        "rendered_prompts": {
            "student": _render_student_prompt(prompt_config),
            "teacher": _render_teacher_prompt(prompt_config),
            "reviewer": _render_reviewer_prompt(prompt_config),
        },
        "reviewer_history": prompt_history,
        "features": features_payload,
        "expected_response_schema": {
            "session_name": session_name,
            "round_index": int(round_index),
            "features": [
                {
                    "feature_key": "<feature_key>",
                    "student": {
                        "primary_locus": "...",
                        "adjacent_context": "...",
                        "canonical_label": "...",
                        "support_summary": "...",
                        "description": "...",
                        "notes": "...",
                        "confidence": 0.5,
                    },
                    "teacher": {
                        "verdict": "accept | revise | reject | uninterpretable",
                        "scope_assessment": "localized | too_broad_part | too_broad_scene | unclear",
                        "grounding_assessment": "grounded | partially_grounded | weakly_grounded",
                        "adjacent_context_assessment": "appropriate | underweighted | overstated | not_applicable",
                        "canonical_label_feedback": "...",
                        "description_feedback": "...",
                        "notes_feedback": "...",
                        "overall_feedback": "...",
                    },
                    "reviewer_feature": {
                        "diagnosis": "...",
                        "failure_tags": ["scope_too_broad_part"],
                        "prompt_suggestions": ["..."],
                        "visualization_suggestions": ["..."],
                    },
                }
            ],
            "reviewer_batch": {
                "summary_rationale": "...",
                "prompt_change_proposals": [
                    {
                        "path": "prompt.emphasize_activation_geometry",
                        "proposed_value": True,
                        "rationale": "...",
                        "label": "optional short label",
                    }
                ],
                "visualization_change_proposals": [
                    {
                        "path": "visualization.include_erf_heatmap_sidecar",
                        "proposed_value": True,
                        "rationale": "...",
                        "label": "optional short label",
                    }
                ],
            },
        },
    }
    write_json(round_dir / "agent_packet.json", packet)
    (round_dir / "agent_packet.md").write_text(_render_agent_packet_markdown(packet, round_dir))
    round_summary = _round_status_payload(
        session_name=session_name,
        round_index=int(round_index),
        prompt_config=prompt_config,
        feature_keys=[str(row["feature_key"]) for row in pending_rows],
        status="packet_built",
        extra={
            "track": dict(manifest.get("track") or {}),
            "current_phase": str(manifest.get("phase_state", {}).get("current_phase") or PHASE_PROMPT_STABILIZATION),
            "policy": round_policy,
            "html_path": None,
            "default_state_json": str(_session_state_path(config, session_name, int(round_index))),
            "features": [],
        },
    )
    write_json(round_dir / "round_summary.json", round_summary)
    manifest["latest_agent_round"] = int(round_index)
    _write_session_manifest(config, session_name, manifest)
    return {
        "session_name": session_name,
        "round_index": int(round_index),
        "round_dir": str(round_dir),
        "agent_packet_json": str(round_dir / "agent_packet.json"),
        "agent_packet_md": str(round_dir / "agent_packet.md"),
        "n_features": len(features_payload),
    }


def _normalize_agent_feature_response(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_key": str(row["feature_key"]),
        "student": dict(row.get("student") or {}),
        "teacher": dict(row.get("teacher") or {}),
        "reviewer_feature": dict(row.get("reviewer_feature") or {}),
    }


def ingest_autolabel_round(
    config: EvalConfig,
    *,
    session_name: str,
    round_index: int | None = None,
    response_json: Path | None = None,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    session_dir = _autolabel_session_dir(config, session_name)
    if round_index is None:
        round_index = int(manifest["latest_agent_round"] or manifest["next_round_index"])
    round_dir = _round_dir(config, session_name, int(round_index))
    packet_path = round_dir / "agent_packet.json"
    if not packet_path.exists():
        raise FileNotFoundError(f"Missing agent packet at {packet_path}. Run build-autolabel-round-packet first.")
    packet = read_json(packet_path)
    response_path = Path(response_json) if response_json is not None else round_dir / "agent_response.json"
    if not response_path.exists():
        raise FileNotFoundError(f"Missing agent response at {response_path}")
    response_payload = read_json(response_path)
    round_policy = _load_round_policy_if_exists(config, session_name)
    if response_path.resolve() != (round_dir / "agent_response.json").resolve():
        shutil.copyfile(response_path, round_dir / "agent_response.json")
    prompt_config = dict(packet["prompt_config"])
    state_rows = _load_feature_state(config, session_name)
    state_map = {str(row["feature_key"]): row for row in state_rows}
    feature_pool = _load_feature_pool(config, session_name)
    pool_map = {str(row["feature_key"]): row for row in feature_pool["features"]}

    reviewer_batch_raw = dict(response_payload.get("reviewer_batch") or {})
    reviewer_batch = {
        "summary_rationale": _norm_text(reviewer_batch_raw.get("summary_rationale")),
        "prompt_change_proposals": _normalize_proposals(
            "prompt",
            reviewer_batch_raw.get("prompt_change_proposals") or [],
        ),
        "visualization_change_proposals": _normalize_proposals(
            "visualization",
            reviewer_batch_raw.get("visualization_change_proposals") or [],
        ),
    }

    ingested_features: list[dict[str, Any]] = []
    for feature_row in response_payload.get("features") or []:
        normalized = _normalize_agent_feature_response(feature_row)
        feature_key_value = str(normalized["feature_key"])
        if feature_key_value not in state_map:
            raise KeyError(f"Unknown feature_key in agent response: {feature_key_value}")
        state = state_map[feature_key_value]
        previous_status = str(state.get("status", "awaiting_agent"))
        state["latest_student"] = normalized["student"]
        state["latest_teacher"] = normalized["teacher"]
        state["latest_reviewer_feature"] = normalized["reviewer_feature"]
        state["status"] = "awaiting_human"
        state["last_agent_round"] = int(round_index)
        state["last_updated_at"] = _now_iso()
        pool_row = pool_map[feature_key_value]
        label_examples: list[dict[str, Any]] = []
        for example in pool_row["label_examples"]:
            rewritten = dict(example)
            for field in (
                "original_with_token_box",
                "feature_actmap",
                "feature_erf_support",
                "feature_erf_zoom",
                "feature_erf_zoom_detail",
                "feature_erf_heatmap",
                "feature_erf_json",
            ):
                if field in rewritten:
                    rewritten[field] = _relpath(round_dir, session_dir / str(example[field]))
            label_examples.append(rewritten)
        ingested_features.append(
            {
                "feature_key": feature_key_value,
                "block_idx": int(pool_row["block_idx"]),
                "feature_id": int(pool_row["feature_id"]),
                "previous_status": previous_status,
                "source_type": _norm_text(feature_row.get("source_type")) or _feature_source_type_for_agent(state),
                "student": normalized["student"],
                "teacher": normalized["teacher"],
                "reviewer_feature": normalized["reviewer_feature"],
                "provided_gt": dict(state.get("provided_gt") or {}),
                "label_examples": label_examples,
            }
        )
    _write_feature_state(config, session_name, list(state_map.values()))

    round_summary = _round_status_payload(
        session_name=session_name,
        round_index=int(round_index),
        prompt_config=prompt_config,
        feature_keys=[str(row["feature_key"]) for row in ingested_features],
        status="awaiting_human_review",
        reviewer_batch=reviewer_batch,
        extra={
            "track": dict(manifest.get("track") or {}),
            "current_phase": str(manifest.get("phase_state", {}).get("current_phase") or PHASE_PROMPT_STABILIZATION),
            "policy": round_policy,
            "agent_packet_json": str(packet_path),
            "agent_response_json": str(round_dir / "agent_response.json"),
            "features": ingested_features,
        },
    )
    initial_state = _build_human_review_initial_state(round_summary)
    html_payload = _build_human_review_html(round_summary=round_summary, initial_state=initial_state)
    (_session_html_path(config, session_name, int(round_index))).write_text(html_payload)
    write_json(_session_state_path(config, session_name, int(round_index)), initial_state)
    round_summary["html_path"] = str(_session_html_path(config, session_name, int(round_index)))
    round_summary["default_state_json"] = str(_session_state_path(config, session_name, int(round_index)))
    write_json(round_dir / "round_summary.json", round_summary)

    manifest["latest_agent_round"] = int(round_index)
    manifest["latest_human_round"] = int(round_index)
    manifest["status_counts"] = _status_counts(list(state_map.values()))
    _write_session_manifest(config, session_name, manifest)
    return {
        "session_name": session_name,
        "round_index": int(round_index),
        "round_dir": str(round_dir),
        "n_features": len(ingested_features),
        "human_review_html": str(_session_html_path(config, session_name, int(round_index))),
        "human_review_state_json": str(_session_state_path(config, session_name, int(round_index))),
    }


def _load_human_review_state(config: EvalConfig, session_name: str, round_index: int) -> dict[str, Any]:
    path = _session_state_path(config, session_name, round_index)
    if not path.exists():
        raise FileNotFoundError(f"Missing human review state at {path}")
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        return dict(payload["state"])
    return payload


def _approved_proposals(review_state: dict[str, Any], round_summary: dict[str, Any], *, proposal_key: str) -> list[dict[str, Any]]:
    batch_review = dict(review_state.get("batch_review") or {})
    decisions = dict(batch_review.get(proposal_key) or {})
    summary_key = "prompt_change_proposals" if proposal_key == "prompt_proposals" else "visualization_change_proposals"
    approved: list[dict[str, Any]] = []
    for row in round_summary.get("reviewer_batch", {}).get(summary_key, []):
        if str(decisions.get(str(row["proposal_id"]), "")) == "approve":
            approved.append(dict(row))
    return approved


def advance_autolabel_round(
    config: EvalConfig,
    *,
    session_name: str,
    round_index: int | None = None,
) -> dict[str, Any]:
    manifest = _load_session_manifest(config, session_name)
    feature_pool = _load_feature_pool(config, session_name)
    if round_index is None:
        round_index = int(manifest.get("latest_human_round") or manifest.get("latest_agent_round") or manifest["next_round_index"])
    round_dir = _round_dir(config, session_name, int(round_index))
    round_summary_path = round_dir / "round_summary.json"
    if not round_summary_path.exists():
        raise FileNotFoundError(f"Missing round summary at {round_summary_path}")
    round_summary = read_json(round_summary_path)
    review_state = _load_human_review_state(config, session_name, int(round_index))
    state_rows = _load_feature_state(config, session_name)
    state_map = {str(row["feature_key"]): row for row in state_rows}
    current_prompt_config = _load_current_prompt_config(config, session_name)
    prompt_history = _load_prompt_history(config, session_name)
    current_phase = str(manifest.get("phase_state", {}).get("current_phase") or PHASE_PROMPT_STABILIZATION)

    incomplete: list[str] = []
    for feature in round_summary["features"]:
        feature_key_value = str(feature["feature_key"])
        item_state = dict(review_state.get("items", {}).get(feature_key_value) or {})
        human_decision = _norm_text(item_state.get("human_decision")).lower()
        if human_decision not in {"accept", "revise", "uninterpretable", "provide_gt", "skip"}:
            incomplete.append(feature_key_value)
    if incomplete:
        raise RuntimeError(f"Human review is incomplete for features: {', '.join(incomplete)}")

    for feature in round_summary["features"]:
        feature_key_value = str(feature["feature_key"])
        row = state_map[feature_key_value]
        item_state = dict(review_state["items"][feature_key_value])
        human_decision = _norm_text(item_state.get("human_decision")).lower()
        source_type = _infer_round_feature_source_type(feature, round_summary)
        meta_tags = [tag for tag in META_TAGS if bool(item_state.get("meta_tags", {}).get(tag))]
        latest_human_feedback = {
            "human_decision": human_decision,
            "human_feedback": _norm_text(item_state.get("human_feedback")),
            "canonical_label": _norm_text(item_state.get("canonical_label")),
            "support_summary": _norm_text(item_state.get("support_summary")),
            "description": _norm_text(item_state.get("description")),
            "notes": _norm_text(item_state.get("notes")),
            "meta_tags": meta_tags,
            "meta_notes": _norm_text(item_state.get("meta_notes")),
            "gt_canonical_label": _norm_text(item_state.get("gt_canonical_label")),
            "gt_description": _norm_text(item_state.get("gt_description")),
            "gt_notes": _norm_text(item_state.get("gt_notes")),
            "reviewed_at": _now_iso(),
            "round_index": int(round_index),
        }
        row["latest_human_feedback"] = latest_human_feedback
        row["meta_tags"] = meta_tags
        row["meta_notes"] = _norm_text(item_state.get("meta_notes"))
        row["last_human_round"] = int(round_index)
        row["last_updated_at"] = _now_iso()
        row["review_count"] = int(row.get("review_count") or 0) + 1
        row["last_source_type"] = source_type
        provided_gt = {
            "canonical_label": _norm_text(item_state.get("gt_canonical_label")),
            "description": _norm_text(item_state.get("gt_description")),
            "notes": _norm_text(item_state.get("gt_notes")),
        }
        if any(_norm_text(value) for value in provided_gt.values()):
            row["provided_gt"] = provided_gt
        if human_decision == "accept":
            row["status"] = "accepted"
            row["terminal"] = True
            row["terminal_round"] = int(round_index)
            row["final_human_decision"] = human_decision
            row["final_label"] = {
                "canonical_label": latest_human_feedback["canonical_label"] or _norm_text(row.get("latest_student", {}).get("canonical_label")),
                "support_summary": latest_human_feedback["support_summary"] or _norm_text(row.get("latest_student", {}).get("support_summary")),
                "description": latest_human_feedback["description"] or _norm_text(row.get("latest_student", {}).get("description")),
                "notes": latest_human_feedback["notes"] or _norm_text(row.get("latest_student", {}).get("notes")),
                "prompt_config_version": int(current_prompt_config["version"]),
            }
        elif human_decision == "uninterpretable":
            row["status"] = "uninterpretable"
            row["terminal"] = True
            row["terminal_round"] = int(round_index)
            row["final_human_decision"] = human_decision
            row["final_label"] = {}
        elif human_decision == "skip":
            row["status"] = "skipped"
            row["terminal"] = True
            row["terminal_round"] = int(round_index)
            row["final_human_decision"] = human_decision
            row["final_label"] = {}
        else:
            row["status"] = "awaiting_agent"
            row["terminal"] = False
            row["terminal_round"] = None
            row["final_human_decision"] = None
            row["final_label"] = {}
            if human_decision == "revise":
                row["revision_count"] = int(row.get("revision_count") or 0) + 1
        feature["human_decision"] = human_decision
        feature["human_feedback"] = latest_human_feedback["human_feedback"]
        feature["meta_tags"] = meta_tags
        feature["meta_notes"] = _norm_text(item_state.get("meta_notes"))
        feature["source_type"] = source_type

    approved_prompt = _approved_proposals(review_state, round_summary, proposal_key="prompt_proposals")
    approved_viz = _approved_proposals(review_state, round_summary, proposal_key="visualization_proposals")
    next_prompt_config = deepcopy(current_prompt_config)
    if approved_prompt or approved_viz:
        for row in approved_prompt + approved_viz:
            _set_nested(next_prompt_config, str(row["path"]), row.get("proposed_value"))
        next_prompt_config["version"] = int(current_prompt_config["version"]) + 1
        next_prompt_config["updated_at"] = _now_iso()
        prompt_history.append(
            {
                "version": int(next_prompt_config["version"]),
                "created_at": _now_iso(),
                "reason": f"round_{int(round_index):03d}_approved_proposals",
                "prompt_config": next_prompt_config,
                "approved_prompt_proposals": approved_prompt,
                "approved_visualization_proposals": approved_viz,
            }
        )
        _write_prompt_history(config, session_name, prompt_history)
        _write_current_prompt_config(config, session_name, next_prompt_config)

    rows = list(state_map.values())
    round_features = list(round_summary["features"])
    round_accepts = sum(
        1
        for feature in round_features
        if _norm_text(review_state["items"][str(feature["feature_key"])].get("human_decision")).lower() == "accept"
    )
    round_accept_ratio = float(round_accepts) / float(len(round_features)) if round_features else None

    added_feature_keys: list[str] = []
    if bool(manifest.get("top_up_new_features", True)):
        added_feature_keys = _top_up_awaiting_agent_features(
            config,
            session_name=session_name,
            rows=rows,
            feature_pool=feature_pool,
            manifest=manifest,
            prompt_config_version=int(next_prompt_config["version"]),
            seed=int(manifest.get("seed", 0)) + int(round_index) + int(next_prompt_config["version"]),
        )

    write_json(_autolabel_session_dir(config, session_name) / "feature_pool.json", feature_pool)
    _write_feature_state(config, session_name, rows)

    remaining = [row for row in rows if str(row["status"]) == "awaiting_agent"]
    backlog_metrics = _backlog_metrics(rows)
    if remaining:
        manifest["next_round_index"] = int(round_index) + 1
    manifest["latest_human_round"] = int(round_index)
    manifest["current_prompt_config_version"] = int(next_prompt_config["version"])
    manifest["status_counts"] = _status_counts(rows)
    manifest["accept_metrics"] = _accept_metrics(rows)
    manifest["feature_count"] = len(feature_pool.get("features", []))
    _write_session_manifest(config, session_name, manifest)

    round_metrics = _round_metrics_from_summary(
        config=config,
        session_name=session_name,
        round_summary=round_summary,
    )

    round_summary["status"] = "human_review_completed"
    round_summary["track"] = dict(manifest.get("track") or {})
    round_summary["current_phase"] = current_phase
    round_summary["applied_prompt_change_proposals"] = approved_prompt
    round_summary["applied_visualization_change_proposals"] = approved_viz
    round_summary["batch_review"] = dict(review_state.get("batch_review") or {})
    round_summary["advanced_at"] = _now_iso()
    round_summary["next_round_index"] = manifest["next_round_index"] if remaining else None
    round_summary["round_accept_ratio"] = round_accept_ratio
    round_summary["added_feature_keys_for_next_round"] = added_feature_keys
    round_summary["cumulative_accept_metrics"] = manifest["accept_metrics"]
    round_summary["round_decision_counts"] = round_metrics["decision_counts"]
    round_summary["source_decision_counts"] = round_metrics["by_source"]
    round_summary["failure_tag_histogram"] = round_metrics["failure_tag_histogram"]
    round_summary["fresh_accept_rate"] = round_metrics["fresh_accept_rate"]
    round_summary["fresh_revise_rate"] = round_metrics["fresh_revise_rate"]
    round_summary["fresh_uninterpretable_rate"] = round_metrics["fresh_uninterpretable_rate"]
    round_summary["carry_over_resolve_rate"] = round_metrics["carry_over_resolve_rate"]
    round_summary["carry_over_total"] = round_metrics["carry_over_total"]
    round_summary["fresh_total"] = round_metrics["fresh_total"]
    round_summary["carry_over_backlog_size_after_round"] = backlog_metrics["carry_over_backlog_size"]
    round_summary["fresh_backlog_size_after_round"] = backlog_metrics["fresh_backlog_size"]
    round_summary["total_backlog_size_after_round"] = backlog_metrics["total_backlog_size"]
    write_json(round_summary_path, round_summary)

    if remaining:
        _round_dir(config, session_name, int(manifest["next_round_index"])).mkdir(parents=True, exist_ok=True)

    return {
        "session_name": session_name,
        "round_index": int(round_index),
        "remaining_features": len(remaining),
        "next_round_index": manifest.get("next_round_index") if remaining else None,
        "current_prompt_config_version": int(next_prompt_config["version"]),
        "status_counts": manifest["status_counts"],
        "round_accept_ratio": round_accept_ratio,
        "cumulative_accept_metrics": manifest["accept_metrics"],
        "added_feature_keys_for_next_round": added_feature_keys,
    }


def promote_autolabel_labels(
    config: EvalConfig,
    *,
    session_name: str,
) -> dict[str, Any]:
    state_rows = _load_feature_state(config, session_name)
    existing = read_jsonl(config.label_registry_jsonl) if config.label_registry_jsonl.exists() else []
    existing_ids = {str(row.get("record_id", "")) for row in existing}
    new_rows: list[dict[str, Any]] = []
    for row in state_rows:
        if str(row.get("status", "")) != "accepted":
            continue
        final_label = dict(row.get("final_label") or {})
        if not _norm_text(final_label.get("canonical_label")):
            continue
        record_id = f"autolabel:{session_name}:{row['feature_key']}:round_{int(row['terminal_round'])}"
        if record_id in existing_ids:
            continue
        new_rows.append(
            {
                "record_id": record_id,
                "feature_key": str(row["feature_key"]),
                "block_idx": int(row["block_idx"]),
                "feature_id": int(row["feature_id"]),
                "session_name": session_name,
                "session_kind": "autolabel_loop_v1",
                "provider_type": "autolabel_human_approved",
                "provider_id": f"autolabel:{session_name}",
                "exported_at": _now_iso(),
                "ingested_at": _now_iso(),
                "status": "accepted",
                "canonical_label": _norm_text(final_label.get("canonical_label")),
                "support_summary": _norm_text(final_label.get("support_summary")),
                "description": _norm_text(final_label.get("description")),
                "notes": _norm_text(final_label.get("notes")) or _norm_text(row.get("meta_notes")),
                "confidence": _safe_float(row.get("latest_student", {}).get("confidence")),
                "source_round": int(row["terminal_round"]),
                "final_human_decision": str(row.get("final_human_decision", "")),
                "meta_tags": list(row.get("meta_tags") or []),
                "meta_notes": _norm_text(row.get("meta_notes")),
                "gt_provided": bool(any(_norm_text(v) for v in dict(row.get("provided_gt") or {}).values())),
                "provided_gt": dict(row.get("provided_gt") or {}),
                "prompt_config_version_used": int(final_label.get("prompt_config_version") or row.get("current_prompt_config_version") or 0),
            }
        )
    if new_rows:
        write_jsonl(config.label_registry_jsonl, [*existing, *new_rows])
    return {
        "session_name": session_name,
        "label_registry_jsonl": str(config.label_registry_jsonl),
        "n_promoted": len(new_rows),
    }
