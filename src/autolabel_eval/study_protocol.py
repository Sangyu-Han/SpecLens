from __future__ import annotations

import html
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .feature_bank import load_feature_bank
from .legacy import LegacyRuntime, token_record_from_row
from .metrics import (
    average_precision_binary,
    f1_accuracy_at_threshold,
    ndcg_at_k,
    recall_at_k,
    roc_auc_binary,
)
from .rendering import (
    save_cosine_overlay_image,
    save_feature_actmap_overlay,
    save_original_with_token_box,
    save_support_detail_crop_image,
    save_support_mask_image,
    save_support_outline_crop_image,
)
from .study_html import (
    RoleSpec,
    StudyItem,
    axis1_team_session,
    choice_field,
    number_field,
    select_field,
    session_manifest as build_study_manifest,
    textarea_field,
    text_field,
    write_study_page,
)
from .utils import feature_key, read_jsonl, token_uid, write_json, write_jsonl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_dir(config: EvalConfig, session_name: str) -> Path:
    return config.study_root / session_name


def _resolve_session_dir(config: EvalConfig, session_name: str, session_dir: Path | None = None) -> Path:
    return Path(session_dir) if session_dir is not None else _session_dir(config, session_name)


def _slug(value: str) -> str:
    return str(value).replace("/", "__")


def _relpath(base_dir: Path, path: Path) -> str:
    return str(path.relative_to(base_dir))


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _read_session_response(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "state" in payload and isinstance(payload["state"], dict):
        return payload
    if isinstance(payload, dict) and "items" in payload:
        return {"state": payload}
    raise ValueError(f"Unsupported study response payload shape at {path}")


def load_label_registry(config: EvalConfig) -> list[dict[str, Any]]:
    if not config.label_registry_jsonl.exists():
        return []
    return read_jsonl(config.label_registry_jsonl)


def current_label_map(config: EvalConfig) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for row in load_label_registry(config):
        current[str(row["feature_key"])] = row
    return current


def accepted_label_map(config: EvalConfig) -> dict[str, dict[str, Any]]:
    current = current_label_map(config)
    return {
        key: row
        for key, row in current.items()
        if str(row.get("status", "")) == "accepted" and _norm_text(row.get("canonical_label"))
    }


def _feature_lookup(feature_bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for block_payload in feature_bank["blocks"].values():
        for feature in block_payload["features"]:
            out[str(feature["feature_key"])] = feature
    return out


def _select_feature_specs(
    feature_bank: dict[str, Any],
    feature_specs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    lookup = _feature_lookup(feature_bank)
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for block_idx, feature_id in feature_specs:
        key = feature_key(int(block_idx), int(feature_id))
        feature = lookup.get(key)
        if feature is None:
            missing.append(key)
            continue
        selected.append(feature)
    if missing:
        raise KeyError(f"Unknown feature_specs: {missing}")
    return selected


def _default_evidence_profile() -> dict[str, Any]:
    return {
        "variant_id": "raw_only",
        "label": "Raw ERF zoom only",
        "include_erf_zoom_detail": False,
    }


def _normalize_evidence_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(_default_evidence_profile())
    if profile:
        normalized.update(profile)
    normalized["include_erf_zoom_detail"] = bool(normalized.get("include_erf_zoom_detail"))
    normalized["variant_id"] = _norm_text(normalized.get("variant_id")) or "raw_only"
    normalized["label"] = _norm_text(normalized.get("label")) or normalized["variant_id"]
    return normalized


def _select_features_for_label_session(
    config: EvalConfig,
    feature_bank: dict[str, Any],
    *,
    features_per_block: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current = current_label_map(config)
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"blocks": {}}
    for block_idx in config.blocks:
        features = list(feature_bank["blocks"][str(block_idx)]["features"])
        pending = [
            feature
            for feature in features
            if str(current.get(str(feature["feature_key"]), {}).get("status", "")) != "accepted"
        ]
        rng.shuffle(pending)
        chosen = pending[: int(features_per_block)]
        diagnostics["blocks"][str(block_idx)] = {
            "available_pending": len(pending),
            "selected": len(chosen),
            "requested": int(features_per_block),
        }
        selected.extend(chosen)
    return selected, diagnostics


def _label_item_evidence_html(entry: dict[str, Any]) -> str:
    cards: list[str] = []
    label_examples = list(entry.get("label_examples", entry.get("train_examples", [])))
    for sample in label_examples:
        detail_html = ""
        if _norm_text(sample.get("token_erf_zoom_detail")):
            detail_html = f"""
                <div>
                  <img src="{html.escape(sample['token_erf_zoom_detail'])}" alt="erf-zoom-detail" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom detail</div>
                </div>
            """
        cards.append(
            f"""
            <div style="border:1px solid #ddd;border-radius:12px;padding:10px;background:#fff;">
              <div style="font-weight:600;margin-bottom:4px;">Sample {int(sample['rank']) + 1}</div>
              <div style="font-size:12px;color:#666;margin-bottom:8px;">sample_id={sample['sample_id']} tok={sample['token_idx']}</div>
              <div style="display:grid;grid-template-columns:repeat({4 if detail_html else 3},minmax(0,1fr));gap:8px;align-items:start;">
                <div>
                  <img src="{html.escape(sample['original_with_token_box'])}" alt="original" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">Original + token</div>
                </div>
                <div>
                  <img src="{html.escape(sample['feature_actmap'])}" alt="actmap" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">Feature activation map</div>
                </div>
                <div>
                  <img src="{html.escape(sample['token_erf_zoom'])}" alt="erf-zoom" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom crop + checkerboard outside support + outer contour</div>
                </div>
                {detail_html}
              </div>
            </div>
            """
        )
    return (
        "<div>"
        "<p style='margin:0 0 12px;color:#666;'>"
        f"Inspect the {len(label_examples)} label examples and produce a reusable label. "
        "Use a short canonical phrase plus an optional short description."
        "</p>"
        f"<div style='display:grid;gap:12px;'>{''.join(cards)}</div>"
        "</div>"
    )


def _axis1_eval_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            role_id="judge",
            title="Judge",
            instructions=(
                "Choose the single token that best matches the fixed feature label. "
                "Then add one confidence score for the choice."
            ),
            fields=(
                choice_field("selected_candidate", "Selected candidate", ("c01", "c02")),
                choice_field(
                    "score_0_10",
                    "Confidence 0-10",
                    tuple(str(v) for v in range(11)),
                    help_text="0 means very weak confidence. 10 means very strong confidence.",
                ),
                textarea_field(
                    "brief_reason",
                    "Brief reason",
                    rows=3,
                    placeholder="Optional note about why one token fit the label better. Korean is okay.",
                ),
            ),
        ),
    )


def _axis2_eval_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            role_id="judge",
            title="Judge",
            instructions=(
                "Pick the best matching feature label first. Optionally add a short ranking. "
                "Korean reasons are okay."
            ),
            fields=(
                choice_field("best_candidate", "Best candidate", ("c01", "c02", "c03", "c04")),
                textarea_field(
                    "ranked_candidates",
                    "Ranked candidate codes",
                    rows=4,
                    placeholder="c01, c04, c02",
                    help_text="Optional. Use the candidate codes from the table below the evidence.",
                ),
                textarea_field("brief_reason", "Brief reason", rows=3, placeholder="Why this ranking? Korean is okay."),
            ),
        ),
    )


def _human_label_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            role_id="labeler",
            title="Labeler",
            instructions=(
                "Write the final label directly. Korean notes are okay. "
                "Prefer a short English canonical label if possible for downstream evaluation."
            ),
            fields=(
                text_field("canonical_label", "Canonical label", placeholder="short English phrase"),
                textarea_field(
                    "description",
                    "Description",
                    rows=4,
                    placeholder="Short general description. Korean is okay.",
                ),
                textarea_field(
                    "notes",
                    "Notes",
                    rows=3,
                    placeholder="Uncertainty, ambiguity, or extra comments. Korean is okay.",
                ),
                number_field("confidence", "Confidence", default=0.5, step="0.05"),
                choice_field("status", "Status", ("accept", "uncertain", "polysemantic", "skip"), default="uncertain"),
            ),
        ),
    )


def _axis1_item_evidence_html(item: dict[str, Any]) -> str:
    desc_html = ""
    if _norm_text(item.get("description")):
        desc_html = f"<div style='font-size:13px;color:#666;margin-top:4px;'>{html.escape(str(item['description']))}</div>"
    candidate_cards: list[str] = []
    for candidate in item["candidates"]:
        code = html.escape(str(candidate["candidate_code"]))
        detail_html = ""
        if _norm_text(candidate.get("token_erf_zoom_detail")):
            detail_html = f"""
                <div>
                  <img src="{html.escape(str(candidate['token_erf_zoom_detail']))}" alt="erf-zoom-detail" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom detail</div>
                </div>
            """
        candidate_cards.append(
            f"""
            <div style="border:1px solid #ddd;border-radius:12px;padding:10px;background:#fff;">
              <div style="font-weight:700;margin-bottom:8px;">{code}</div>
              <div style="display:grid;grid-template-columns:repeat({4 if detail_html else 3},minmax(0,1fr));gap:10px;align-items:start;">
                <div>
                  <img src="{html.escape(str(candidate['original_with_token_box']))}" alt="original" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">Original + boxed token</div>
                </div>
                <div>
                  <img src="{html.escape(str(candidate['token_neighbor_cosine']))}" alt="cosine" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">Token neighbor cosine map</div>
                </div>
                <div>
                  <img src="{html.escape(str(candidate['token_erf_zoom']))}" alt="erf-zoom" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom crop + checkerboard outside support + outer contour</div>
                </div>
                {detail_html}
              </div>
            </div>
            """
        )
    return f"""
    <div>
      <div style="margin-bottom:12px;padding:12px;border:1px solid #ddd;border-radius:12px;background:#fff;">
        <div style="font-size:12px;color:#666;text-transform:uppercase;letter-spacing:0.04em;">Fixed feature label</div>
        <div style="font-size:22px;font-weight:700;">{html.escape(str(item['canonical_label']))}</div>
        {desc_html}
      </div>
      <div style="margin-bottom:12px;padding:10px 12px;border:1px solid #ddd;border-radius:12px;background:#fff7ec;color:#73431f;font-size:13px;">
        Exactly one candidate token is intended to match the feature label in this image.
      </div>
      <div style="display:grid;gap:12px;">{''.join(candidate_cards)}</div>
    </div>
    """


def _axis2_item_evidence_html(item: dict[str, Any]) -> str:
    detail_html = ""
    if _norm_text(item.get("token_erf_zoom_detail")):
        detail_html = f"""
        <div>
          <img src="{html.escape(str(item['token_erf_zoom_detail']))}" alt="erf-zoom-detail" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
          <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom detail</div>
        </div>
        """
    rows = []
    for candidate in item["candidates"]:
        desc = _norm_text(candidate.get("description"))
        desc_html = f"<div style='font-size:12px;color:#666;'>{html.escape(desc)}</div>" if desc else ""
        rows.append(
            f"""
            <tr>
              <td style="padding:6px 8px;border-bottom:1px solid #eee;font-family:monospace;">{html.escape(candidate['candidate_code'])}</td>
              <td style="padding:6px 8px;border-bottom:1px solid #eee;">{html.escape(candidate['canonical_label'])}{desc_html}</td>
              <td style="padding:6px 8px;border-bottom:1px solid #eee;font-size:12px;color:#666;font-family:monospace;">{html.escape(candidate['feature_key'])}</td>
            </tr>
            """
        )
    return f"""
    <div>
      <div style="display:grid;grid-template-columns:repeat({4 if detail_html else 3},minmax(0,1fr));gap:10px;margin-bottom:12px;align-items:start;">
        <div>
          <img src="{html.escape(str(item['original_with_token_box']))}" alt="original" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
          <div style="font-size:12px;color:#666;margin-top:4px;">Original + boxed token</div>
        </div>
        <div>
          <img src="{html.escape(str(item['token_neighbor_cosine']))}" alt="cosine" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
          <div style="font-size:12px;color:#666;margin-top:4px;">Token neighbor cosine map</div>
        </div>
        <div>
          <img src="{html.escape(str(item['token_erf_zoom']))}" alt="erf-zoom" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
          <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom crop + checkerboard outside support + outer contour</div>
        </div>
        {detail_html}
      </div>
      <div style="border:1px solid #ddd;border-radius:12px;background:#fff;overflow:hidden;">
        <div style="padding:10px 12px;border-bottom:1px solid #eee;font-weight:700;">Candidate labels</div>
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr style="background:#faf7f2;">
              <th style="text-align:left;padding:8px;">Code</th>
              <th style="text-align:left;padding:8px;">Label</th>
              <th style="text-align:left;padding:8px;">Feature</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </div>
    """


def _feature_target_activation_scale(feature: dict[str, Any]) -> float:
    acts: list[float] = []
    for row in list(feature["train"]) + list(feature["holdout"]):
        validation = row.get("validation") or {}
        if "act_at_target" in validation:
            acts.append(float(validation["act_at_target"]))
    if not acts:
        return 0.0
    return float(np.median(np.asarray(acts, dtype=np.float32)))


def _axis1_item_id(feature_key_value: str, sample_id: int, token_idx: int) -> str:
    return f"{feature_key_value}::sample_{int(sample_id)}::tok_{int(token_idx)}"


def _axis1_episode_id(feature_key_value: str, sample_id: int) -> str:
    return f"{feature_key_value}::sample_{int(sample_id)}"


def _axis2_item_id(block_idx: int, sample_id: int, token_idx: int) -> str:
    return f"axis2::block_{int(block_idx)}::sample_{int(sample_id)}::tok_{int(token_idx)}"


def _shuffle_in_unison(items: list[StudyItem], payloads: list[dict[str, Any]], *, seed: int) -> None:
    if len(items) != len(payloads):
        raise ValueError("items and payloads must have the same length for shuffling")
    rng = random.Random(int(seed))
    order = list(range(len(items)))
    rng.shuffle(order)
    shuffled_items = [items[idx] for idx in order]
    shuffled_payloads = [payloads[idx] for idx in order]
    items[:] = shuffled_items
    payloads[:] = shuffled_payloads


def _select_mixed_axis1_negatives(
    *,
    feature_key_value: str,
    sample_id: int,
    target_idx: int,
    actmap: np.ndarray,
    cosine_map: np.ndarray,
    hard_count: int,
    easy_count: int,
) -> list[dict[str, Any]]:
    target_act = float(actmap[int(target_idx)])
    candidate_indices = [idx for idx in range(int(actmap.shape[0])) if idx != int(target_idx)]
    if not candidate_indices:
        return []

    n_negatives = int(hard_count) + int(easy_count)
    if n_negatives <= 0:
        return []

    low_activation_max = max(1e-6, 0.10 * max(target_act, 1e-6))
    confident_pool = [idx for idx in candidate_indices if float(actmap[idx]) <= low_activation_max]
    if len(confident_pool) < int(n_negatives):
        fallback_count = min(len(candidate_indices), max(int(n_negatives) * 3, int(n_negatives)))
        candidate_indices_sorted = sorted(candidate_indices, key=lambda idx: (float(actmap[idx]), idx))
        confident_pool = candidate_indices_sorted[:fallback_count]

    rng = random.Random(
        int(sample_id) * 131 + int(target_idx) * 17 + sum(ord(ch) for ch in str(feature_key_value))
    )
    hard_count = max(0, int(hard_count))
    easy_count = max(0, int(easy_count))

    hard_sorted = sorted(
        confident_pool,
        key=lambda idx: (-float(cosine_map[idx]), float(actmap[idx]), int(idx)),
    )
    hard_indices = hard_sorted[:hard_count]

    remaining = [idx for idx in confident_pool if idx not in set(hard_indices)]
    easy_candidates = list(remaining)
    rng.shuffle(easy_candidates)
    easy_indices = easy_candidates[:easy_count]

    chosen = hard_indices + easy_indices
    if len(chosen) < int(n_negatives):
        fallback = [idx for idx in candidate_indices if idx not in set(chosen)]
        rng.shuffle(fallback)
        chosen.extend(fallback[: int(n_negatives) - len(chosen)])

    payload: list[dict[str, Any]] = []
    hard_set = set(hard_indices)
    for idx in chosen[: int(n_negatives)]:
        payload.append(
            {
                "token_idx": int(idx),
                "negative_kind": "hard_context" if int(idx) in hard_set else "easy_random",
                "feature_activation": float(actmap[int(idx)]),
                "target_cosine": float(cosine_map[int(idx)]),
                "target_activation": float(target_act),
                "low_activation_max": float(low_activation_max),
            }
        )
    return payload


def _axis1_response_complete(item_state: dict[str, Any]) -> bool:
    judge = dict(item_state.get("judge", {}))
    candidate_code = _norm_text(judge.get("selected_candidate")).lower()
    score_0_10 = _safe_float(judge.get("score_0_10"))
    return candidate_code in {"c01", "c02"} and score_0_10 is not None


def _final_axis1_selected_candidate(item_state: dict[str, Any]) -> tuple[str, float]:
    judge = dict(item_state.get("judge", {}))
    candidate_code = _norm_text(judge.get("selected_candidate")).lower()
    score_0_10 = _safe_float(judge.get("score_0_10"))
    if candidate_code not in {"c01", "c02"} or score_0_10 is None:
        raise ValueError("Missing selected candidate or confidence score")
    return candidate_code, float(max(0.0, min(100.0, 10.0 * float(score_0_10))))


def _binary_summary(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float) -> dict[str, float]:
    return {
        "auprc": average_precision_binary(y_true, y_score),
        "auroc": roc_auc_binary(y_true, y_score),
        **f1_accuracy_at_threshold(y_true, y_score, threshold=threshold),
    }


def _extract_role_state(payload: dict[str, Any], item_id: str, role_id: str) -> dict[str, Any]:
    return dict(payload.get("state", {}).get("items", {}).get(item_id, {}).get(role_id, {}))


def _collect_label_examples(
    *,
    config: EvalConfig,
    runtime: LegacyRuntime,
    feature: dict[str, Any],
    target_count: int,
    frame_cache: dict[int, Any],
    sid_to_path_cache: dict[int, str],
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    examples: list[dict[str, Any]] = []
    seen_token_uids: set[str] = set()
    seen_sample_ids: set[int] = set()
    holdout_sample_ids = {int(row["sample_id"]) for row in feature["holdout"]}

    for row in feature["train"]:
        copied = dict(row)
        copied["label_example_source"] = "train"
        examples.append(copied)
        seen_token_uids.add(str(row["token_uid"]))
        seen_sample_ids.add(int(row["sample_id"]))
        if len(examples) >= target_count:
            return examples[:target_count]

    block_idx = int(feature["block_idx"])
    feature_id = int(feature["feature_id"])
    frame = frame_cache.setdefault(block_idx, runtime.load_decile_frame(block_idx))
    feature_rows = frame[frame["unit"] == feature_id].sort_values("score", ascending=False)
    sample_ids = [int(sid) for sid in feature_rows["sample_id"].astype(int).unique().tolist()]
    missing = [sid for sid in sample_ids if sid not in sid_to_path_cache]
    if missing:
        sid_to_path_cache.update(runtime.lookup_paths(missing))

    for row in feature_rows.itertuples(index=False):
        if len(examples) >= target_count:
            break
        sample_id = int(row.sample_id)
        tok_idx = runtime.row_x_to_token_idx(int(row.x))
        if tok_idx < 0 or tok_idx >= config.n_patches:
            continue
        if sample_id in holdout_sample_ids:
            continue
        if sample_id in seen_sample_ids:
            continue
        image_path = sid_to_path_cache.get(sample_id, "")
        if not image_path:
            continue
        candidate_uid = f"block_{block_idx}/sample_{sample_id}/tok_{tok_idx}"
        if candidate_uid in seen_token_uids:
            continue
        validation = runtime.validate_feature_token(
            image_path,
            block_idx,
            feature_id,
            tok_idx,
            float(row.score),
        )
        if validation is None:
            continue
        accepted = token_record_from_row(
            block_idx,
            feature_id,
            row,
            image_path,
            validation,
            token_idx=tok_idx,
        )
        accepted["label_example_source"] = "extra_train"
        examples.append(accepted)
        seen_token_uids.add(candidate_uid)
        seen_sample_ids.add(sample_id)
    if len(examples) < target_count:
        raise RuntimeError(
            f"{feature['feature_key']} only yielded {len(examples)} non-holdout validated label examples; "
            f"required {target_count}"
        )
    return examples[:target_count]


def _final_axis1_decision(item_state: dict[str, Any]) -> tuple[str, float]:
    judge = dict(item_state.get("judge", {}))
    decision = _norm_text(judge.get("decision")).lower()
    score = _safe_float(judge.get("score_0_100"))
    if score is None:
        score_0_10 = _safe_float(judge.get("score_0_10"))
        if score_0_10 is not None:
            score = 10.0 * float(score_0_10)
    if decision in {"yes", "no"} and score is not None:
        return decision, float(max(0.0, min(100.0, score)))

    evaluator = dict(item_state.get("evaluator", {}))
    generator = dict(item_state.get("generator", {}))
    decision = _norm_text(evaluator.get("final_decision")).lower()
    if decision not in {"yes", "no"}:
        decision = _norm_text(generator.get("decision")).lower()
    score = _safe_float(evaluator.get("final_score_0_100"))
    if score is None:
        score = _safe_float(generator.get("score_0_100"))
    if score is None:
        score = 100.0 if decision == "yes" else 0.0
    return decision, float(max(0.0, min(100.0, score)))


def _parse_ranked_codes(text: str, valid_codes: set[str]) -> list[str]:
    tokens = re.split(r"[\s,;>\n\r\t]+", str(text or "").strip())
    ranked: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = token.strip().lower()
        if not value or value not in valid_codes or value in seen:
            continue
        ranked.append(value)
        seen.add(value)
    return ranked


def _final_axis2_ranking(item_state: dict[str, Any], valid_codes: set[str]) -> list[str]:
    judge = dict(item_state.get("judge", {}))
    best_candidate = _norm_text(judge.get("best_candidate")).lower()
    ranked = _parse_ranked_codes(_norm_text(judge.get("ranked_candidates")), valid_codes)
    if best_candidate and best_candidate in valid_codes:
        if best_candidate in ranked:
            ranked = [best_candidate] + [code for code in ranked if code != best_candidate]
        else:
            ranked = [best_candidate, *ranked]
    if ranked:
        return ranked

    evaluator = dict(item_state.get("evaluator", {}))
    generator = dict(item_state.get("generator", {}))
    ranked = _parse_ranked_codes(_norm_text(evaluator.get("final_ranked_candidates")), valid_codes)
    if ranked:
        return ranked
    return _parse_ranked_codes(_norm_text(generator.get("ranked_candidates")), valid_codes)


def _extract_axis_logging_fields(item_state: dict[str, Any], *, evidence_profile: dict[str, Any]) -> dict[str, Any]:
    judge = dict(item_state.get("judge", {}))
    evaluator = dict(item_state.get("evaluator", {}))
    generator = dict(item_state.get("generator", {}))
    structured_rationale = (
        judge.get("structured_rationale")
        or evaluator.get("structured_rationale")
        or generator.get("structured_rationale")
        or {}
    )
    free_text_rationale = (
        _norm_text(judge.get("free_text_rationale"))
        or _norm_text(judge.get("brief_reason"))
        or _norm_text(evaluator.get("free_text_rationale"))
        or _norm_text(evaluator.get("feedback"))
        or _norm_text(generator.get("free_text_rationale"))
        or _norm_text(generator.get("brief_reason"))
    )
    failure_tags = judge.get("failure_tags")
    if failure_tags is None:
        failure_tags = evaluator.get("failure_tags")
    if failure_tags is None:
        failure_tags = generator.get("failure_tags")
    if not isinstance(failure_tags, list):
        failure_tags = []
    confidence = _safe_float(judge.get("confidence"))
    if confidence is None:
        confidence = _safe_float(evaluator.get("confidence"))
    if confidence is None:
        confidence = _safe_float(generator.get("confidence"))
    return {
        "structured_rationale": structured_rationale if structured_rationale is not None else {},
        "free_text_rationale": free_text_rationale,
        "failure_tags": [str(tag) for tag in failure_tags],
        "confidence": confidence,
        "used_sidecar_variant": _norm_text(evidence_profile.get("variant_id")),
    }


def build_study_label_session(
    config: EvalConfig,
    *,
    session_name: str,
    features_per_block: int | None = None,
    seed: int | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
    session_dir: Path | None = None,
    label_prompt_version: str | None = None,
    label_variant: str | None = None,
    generated_for_experiment: str | None = None,
    evidence_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_bank = load_feature_bank(config)
    features_per_block = int(features_per_block or config.study_session_default_features_per_block)
    seed = int(seed if seed is not None else config.study_session_default_seed)
    session_dir = _resolve_session_dir(config, session_name, session_dir=session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    evidence_profile = _normalize_evidence_profile(evidence_profile)

    if feature_specs:
        selected_features = _select_feature_specs(feature_bank, feature_specs)
        selection_diagnostics = {
            "mode": "explicit_feature_specs",
            "requested_feature_specs": [[int(block_idx), int(feature_id)] for block_idx, feature_id in feature_specs],
            "selected": len(selected_features),
        }
    else:
        selected_features, selection_diagnostics = _select_features_for_label_session(
            config,
            feature_bank,
            features_per_block=features_per_block,
            seed=seed,
        )
    runtime = LegacyRuntime(config)
    items: list[StudyItem] = []
    manifest_features: list[dict[str, Any]] = []
    label_example_target = int(config.study_label_examples_per_feature)
    frame_cache: dict[int, Any] = {}
    sid_to_path_cache: dict[int, str] = {}
    try:
        total = len(selected_features)
        for idx, feature in enumerate(selected_features, start=1):
            feature_key_value = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            feature_dir = session_dir / "label_team_assets" / _slug(feature_key_value)
            label_examples = _collect_label_examples(
                config=config,
                runtime=runtime,
                feature=feature,
                target_count=label_example_target,
                frame_cache=frame_cache,
                sid_to_path_cache=sid_to_path_cache,
            )
            rendered_examples: list[dict[str, Any]] = []
            for rank, row in enumerate(label_examples):
                image_path = str(row["image_path"])
                token_idx = int(row["target_patch_idx"])
                original_path = feature_dir / f"train_{rank:02d}_original_token.png"
                actmap_path = feature_dir / f"train_{rank:02d}_feature_actmap.png"
                erf_path = feature_dir / f"train_{rank:02d}_token_erf_support.png"
                erf_zoom_path = feature_dir / f"train_{rank:02d}_token_erf_zoom.png"
                erf_zoom_detail_path = feature_dir / f"train_{rank:02d}_token_erf_zoom_detail.png"
                erf_json_path = feature_dir / f"train_{rank:02d}_token_erf.json"
                save_original_with_token_box(image_path, original_path, token_idx)
                actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
                save_feature_actmap_overlay(image_path, actmap, actmap_path, token_idx=token_idx)
                erf = runtime.cautious_token_erf(image_path, block_idx, token_idx)
                save_support_mask_image(image_path, erf["support_indices"], erf_path, token_idx=token_idx)
                save_support_outline_crop_image(
                    image_path,
                    erf["support_indices"],
                    erf_zoom_path,
                    score_map=erf["prob_scores"],
                )
                if evidence_profile.get("include_erf_zoom_detail"):
                    save_support_detail_crop_image(
                        image_path,
                        erf["support_indices"],
                        erf_zoom_detail_path,
                        token_idx=token_idx,
                    )
                write_json(erf_json_path, erf)
                rendered_examples.append(
                    {
                        "rank": rank,
                        "sample_id": int(row["sample_id"]),
                        "token_idx": token_idx,
                        "image_path": image_path,
                        "label_example_source": str(row.get("label_example_source", "train")),
                        "original_with_token_box": _relpath(session_dir, original_path),
                        "feature_actmap": _relpath(session_dir, actmap_path),
                        "token_erf": _relpath(session_dir, erf_path),
                        "token_erf_zoom": _relpath(session_dir, erf_zoom_path),
                        "token_erf_zoom_detail": _relpath(session_dir, erf_zoom_detail_path)
                        if evidence_profile.get("include_erf_zoom_detail")
                        else "",
                        "token_erf_json": _relpath(session_dir, erf_json_path),
                    }
                )
            entry = {
                "item_id": feature_key_value,
                "feature_key": feature_key_value,
                "block_idx": block_idx,
                "feature_id": feature_id,
                "train_examples": [dict(row) for row in feature["train"]],
                "label_examples": rendered_examples,
                "holdout_rows": feature["holdout"],
                "selection_stats": feature.get("selection_stats", {}),
            }
            manifest_features.append(entry)
            items.append(
                StudyItem(
                    item_id=feature_key_value,
                    title=f"{feature_key_value}",
                    evidence_html=_label_item_evidence_html(entry),
                    metadata={"block": block_idx, "feature_id": feature_id},
                )
            )
            print(
                f"[study-label {idx:03d}/{total:03d}] block={block_idx} feature={feature_id}",
                flush=True,
            )
    finally:
        runtime.close()

    session = axis1_team_session(
        session_id=f"{session_name}__label_team",
        title=f"Axis 1 Label Team Session: {session_name}",
        items=items,
        roles=_human_label_roles(),
        storage_key=f"autolabel.study.{session_name}.label_team.v1",
        export_filename="label_team_response.json",
        intro_html=(
            f"Each feature item asks for one final human label. "
            f"You now see up to {label_example_target} non-holdout validated examples per feature. "
            "Write a short reusable canonical label plus a short description or notes."
        ),
        footer_html=(
            "This page autosaves in the browser, but you still need to export the JSON and save it as "
            "<code>label_team_response.json</code> in this session directory, "
            "then run the ingest command."
        ),
    )
    write_study_page(session_dir / "label_team.html", session)
    manifest = {
        "kind": "label_team",
        "session_name": session_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "seed": seed,
        "features_per_block": features_per_block,
        "label_examples_per_feature": label_example_target,
        "label_prompt_version": _norm_text(label_prompt_version),
        "label_variant": _norm_text(label_variant),
        "generated_for_experiment": _norm_text(generated_for_experiment),
        "evidence_profile": evidence_profile,
        "selection_diagnostics": selection_diagnostics,
        "selected_features": manifest_features,
        "study_session_manifest": build_study_manifest(session),
        "html_path": str(session_dir / "label_team.html"),
        "default_response_json": str(session_dir / "label_team_response.json"),
    }
    write_json(session_dir / "label_team_manifest.json", manifest)
    return manifest


def ingest_study_label_session(
    config: EvalConfig,
    *,
    session_name: str,
    response_json: Path | None = None,
    provider_type: str = "human",
    session_dir: Path | None = None,
) -> dict[str, Any]:
    session_dir = _resolve_session_dir(config, session_name, session_dir=session_dir)
    manifest_path = session_dir / "label_team_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing label team manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    response_path = Path(response_json) if response_json is not None else session_dir / "label_team_response.json"
    if not response_path.exists():
        raise FileNotFoundError(
            f"Missing label team response at {response_path}. Export JSON from label_team.html first."
        )
    response_payload = _read_session_response(response_path)
    existing = load_label_registry(config)
    existing_ids = {str(row.get("record_id", "")) for row in existing}
    new_rows: list[dict[str, Any]] = []
    accepted = 0
    for feature in manifest["selected_features"]:
        item_id = str(feature["item_id"])
        labeler = _extract_role_state(response_payload, item_id, "labeler")
        planner = _extract_role_state(response_payload, item_id, "planner")
        generator = _extract_role_state(response_payload, item_id, "generator")
        evaluator = _extract_role_state(response_payload, item_id, "evaluator")
        canonical_label = _norm_text(labeler.get("canonical_label")) or _norm_text(generator.get("canonical_label"))
        description = _norm_text(labeler.get("description")) or _norm_text(generator.get("description"))
        notes = _norm_text(labeler.get("notes"))
        structured_rationale = labeler.get("structured_rationale")
        if not structured_rationale:
            structured_rationale = generator.get("structured_rationale")
        free_text_rationale = _norm_text(labeler.get("free_text_rationale")) or _norm_text(generator.get("free_text_rationale"))
        label_status = _norm_text(labeler.get("status")).lower()
        evaluator_decision = _norm_text(evaluator.get("decision")).lower()
        if label_status == "accept" and canonical_label:
            status = "accepted"
            accepted += 1
        elif label_status == "polysemantic":
            status = "polysemantic"
        elif label_status == "skip":
            status = "skipped"
        elif label_status == "uncertain":
            status = "uncertain"
        elif evaluator_decision == "reject":
            status = "rejected"
        elif evaluator_decision == "accept" and canonical_label:
            status = "accepted"
            accepted += 1
        else:
            status = "uncertain"
        record_id = f"{session_name}:{item_id}:{response_payload.get('exported_at', '')}"
        if record_id in existing_ids:
            continue
        new_rows.append(
            {
                "record_id": record_id,
                "feature_key": str(feature["feature_key"]),
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "session_name": session_name,
                "session_kind": "label_team",
                "provider_type": str(provider_type),
                "provider_id": f"{provider_type}:{session_name}",
                "exported_at": response_payload.get("exported_at"),
                "ingested_at": _now_iso(),
                "status": status,
                "canonical_label": canonical_label,
                "description": description,
                "notes": notes,
                "labeler": labeler,
                "planner": planner,
                "generator": generator,
                "evaluator": evaluator,
                "label_prompt_version": _norm_text(manifest.get("label_prompt_version")),
                "label_variant": _norm_text(manifest.get("label_variant")),
                "generated_for_experiment": _norm_text(manifest.get("generated_for_experiment")),
                "evidence_profile": dict(manifest.get("evidence_profile") or {}),
                "structured_rationale": structured_rationale if structured_rationale is not None else {},
                "free_text_rationale": free_text_rationale,
                "confidence": _safe_float(labeler.get("confidence")) if _safe_float(labeler.get("confidence")) is not None else _safe_float(generator.get("confidence")),
                "review_score": _safe_float(evaluator.get("score")),
            }
        )
    if new_rows:
        write_jsonl(config.label_registry_jsonl, [*existing, *new_rows])
    summary = {
        "session_name": session_name,
        "response_json": str(response_path),
        "provider_type": provider_type,
        "n_new_records": len(new_rows),
        "n_accepted_in_response": accepted,
        "label_registry_jsonl": str(config.label_registry_jsonl),
    }
    write_json(session_dir / "label_team_ingest_summary.json", summary)
    return summary


def _selected_or_fallback_labeled_features(
    config: EvalConfig,
    *,
    session_name: str,
    features_per_block: int | None = None,
    seed: int | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
    accepted_rows: dict[str, dict[str, Any]] | None = None,
    session_dir: Path | None = None,
) -> list[dict[str, Any]]:
    feature_bank = load_feature_bank(config)
    features_by_key = _feature_lookup(feature_bank)
    accepted = accepted_rows or accepted_label_map(config)
    if feature_specs:
        explicit = _select_feature_specs(feature_bank, feature_specs)
        return [feature for feature in explicit if str(feature["feature_key"]) in accepted]
    requested = int(features_per_block or config.study_session_default_features_per_block)
    rng = random.Random(int(seed if seed is not None else config.study_session_default_seed) + 17)

    label_manifest_path = _resolve_session_dir(config, session_name, session_dir=session_dir) / "label_team_manifest.json"
    prioritized_keys: list[str] = []
    if label_manifest_path.exists():
        label_manifest = json.loads(label_manifest_path.read_text())
        prioritized_keys = [
            str(entry["feature_key"])
            for entry in label_manifest.get("selected_features", [])
            if str(entry["feature_key"]) in accepted
        ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in prioritized_keys:
        if key in features_by_key and key not in seen:
            selected.append(features_by_key[key])
            seen.add(key)

    for block_idx in config.blocks:
        current_block = [row for row in selected if int(row["block_idx"]) == int(block_idx)]
        if len(current_block) >= requested:
            continue
        candidates = [
            features_by_key[key]
            for key, label_row in accepted.items()
            if key in features_by_key and int(features_by_key[key]["block_idx"]) == int(block_idx) and key not in seen
        ]
        rng.shuffle(candidates)
        need = max(0, requested - len(current_block))
        for feature in candidates[:need]:
            selected.append(feature)
            seen.add(str(feature["feature_key"]))
    return selected


def build_study_axis1_session(
    config: EvalConfig,
    *,
    session_name: str,
    features_per_block: int | None = None,
    seed: int | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
    accepted_rows: dict[str, dict[str, Any]] | None = None,
    session_dir: Path | None = None,
    evidence_profile: dict[str, Any] | None = None,
    evaluator_prompt_version: str | None = None,
) -> dict[str, Any]:
    accepted = accepted_rows or accepted_label_map(config)
    seed = int(seed if seed is not None else config.study_session_default_seed)
    evidence_profile = _normalize_evidence_profile(evidence_profile)
    selected_features = _selected_or_fallback_labeled_features(
        config,
        session_name=session_name,
        features_per_block=features_per_block,
        seed=seed,
        feature_specs=feature_specs,
        accepted_rows=accepted,
        session_dir=session_dir,
    )
    if not selected_features:
        raise RuntimeError("No accepted labels available. Run label team + ingest first.")
    session_dir = _resolve_session_dir(config, session_name, session_dir=session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    runtime = LegacyRuntime(config)
    items: list[StudyItem] = []
    manifest_items: list[dict[str, Any]] = []
    token_cache: dict[str, dict[str, str]] = {}
    hard_negatives_per_holdout = 1
    easy_negatives_per_holdout = 0
    negatives_per_holdout = 1
    try:
        n_total = 0
        for feature in selected_features:
            n_total += len(feature["holdout"])
        item_counter = 0
        for feature in selected_features:
            feature_key_value = str(feature["feature_key"])
            label_row = accepted[feature_key_value]
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            canonical_label = str(label_row["canonical_label"])
            description = _norm_text(label_row.get("description"))
            train_sample_ids = {int(row["sample_id"]) for row in feature["train"]}
            holdout_sample_ids = {int(row["sample_id"]) for row in feature["holdout"]}
            overlap_sample_ids = sorted(train_sample_ids & holdout_sample_ids)
            if overlap_sample_ids:
                raise RuntimeError(
                    f"Label/eval leakage detected for {feature_key_value}: overlapping sample_ids {overlap_sample_ids}"
                )
            for holdout_row in feature["holdout"]:
                image_path = str(holdout_row["image_path"])
                sample_id = int(holdout_row["sample_id"])
                target_idx = int(holdout_row["target_patch_idx"])
                actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
                cosine_map_target = runtime.token_cosine_map(image_path, block_idx, target_idx)
                negatives = _select_mixed_axis1_negatives(
                    feature_key_value=feature_key_value,
                    sample_id=sample_id,
                    target_idx=target_idx,
                    actmap=np.asarray(actmap, dtype=np.float32),
                    cosine_map=np.asarray(cosine_map_target, dtype=np.float32),
                    hard_count=hard_negatives_per_holdout,
                    easy_count=easy_negatives_per_holdout,
                )
                chosen_rows = [
                    {
                        "token_idx": int(target_idx),
                        "label": 1,
                        "negative_kind": None,
                        "feature_activation": float(actmap[target_idx]),
                        "target_cosine": 1.0,
                        "target_activation": float(actmap[target_idx]),
                        "low_activation_max": None,
                    }
                ] + [
                    {
                        "token_idx": int(row["token_idx"]),
                        "label": 0,
                        "negative_kind": str(row["negative_kind"]),
                        "feature_activation": float(row["feature_activation"]),
                        "target_cosine": float(row["target_cosine"]),
                        "target_activation": float(row["target_activation"]),
                        "low_activation_max": row["low_activation_max"],
                    }
                    for row in negatives
                ]
                if len(chosen_rows) != 1 + negatives_per_holdout:
                    raise RuntimeError(
                        f"Expected exactly {1 + negatives_per_holdout} candidates for {feature_key_value} sample {sample_id}, "
                        f"got {len(chosen_rows)}"
                    )
                candidate_payloads: list[dict[str, Any]] = []
                for chosen_row in chosen_rows:
                    token_idx = int(chosen_row["token_idx"])
                    uid = token_uid(block_idx, sample_id, token_idx)
                    if uid not in token_cache:
                        token_dir = session_dir / "axis1_assets" / _slug(uid)
                        original_path = token_dir / "original_token_box.png"
                        erf_path = token_dir / "token_erf_support.png"
                        erf_zoom_path = token_dir / "token_erf_zoom.png"
                        erf_zoom_detail_path = token_dir / "token_erf_zoom_detail.png"
                        cosine_path = token_dir / "token_neighbor_cosine.png"
                        erf_json_path = token_dir / "token_erf.json"
                        save_original_with_token_box(image_path, original_path, token_idx)
                        erf = runtime.cautious_token_erf(image_path, block_idx, token_idx)
                        save_support_mask_image(image_path, erf["support_indices"], erf_path, token_idx=token_idx)
                        save_support_outline_crop_image(
                            image_path,
                            erf["support_indices"],
                            erf_zoom_path,
                            score_map=erf["prob_scores"],
                        )
                        if evidence_profile.get("include_erf_zoom_detail"):
                            save_support_detail_crop_image(
                                image_path,
                                erf["support_indices"],
                                erf_zoom_detail_path,
                                token_idx=token_idx,
                            )
                        cosine = runtime.token_cosine_map(image_path, block_idx, token_idx)
                        save_cosine_overlay_image(image_path, cosine, cosine_path, token_idx=token_idx)
                        write_json(erf_json_path, erf)
                        token_cache[uid] = {
                            "original_with_token_box": _relpath(session_dir, original_path),
                            "token_erf": _relpath(session_dir, erf_path),
                            "token_erf_zoom": _relpath(session_dir, erf_zoom_path),
                            "token_erf_zoom_detail": _relpath(session_dir, erf_zoom_detail_path)
                            if evidence_profile.get("include_erf_zoom_detail")
                            else "",
                            "token_neighbor_cosine": _relpath(session_dir, cosine_path),
                            "token_erf_json": _relpath(session_dir, erf_json_path),
                        }
                    candidate_payloads.append(
                        {
                            "token_idx": int(token_idx),
                            "token_uid": uid,
                            "label": int(chosen_row["label"]),
                            "negative_kind": chosen_row["negative_kind"],
                            "feature_activation": float(chosen_row["feature_activation"]),
                            "target_cosine": float(chosen_row["target_cosine"]),
                            "target_activation": float(chosen_row["target_activation"]),
                            "low_activation_max": chosen_row["low_activation_max"],
                            **token_cache[uid],
                        }
                    )
                local_rng = random.Random(seed + block_idx * 100000 + sample_id * 101 + feature_id)
                local_rng.shuffle(candidate_payloads)
                for idx, row in enumerate(candidate_payloads, start=1):
                    row["candidate_code"] = f"c{idx:02d}"
                item_id = _axis1_episode_id(feature_key_value, sample_id)
                item_payload = {
                    "axis1_item_id": item_id,
                    "feature_key": feature_key_value,
                    "block_idx": block_idx,
                    "feature_id": feature_id,
                    "sample_id": sample_id,
                    "canonical_label": canonical_label,
                    "description": description,
                    "candidates": candidate_payloads,
                }
                manifest_items.append(item_payload)
                items.append(
                    StudyItem(
                        item_id=item_id,
                        title=f"{feature_key_value} | sample {sample_id}",
                        evidence_html=_axis1_item_evidence_html(item_payload),
                        metadata={"block": block_idx, "feature_id": feature_id, "sample_id": sample_id},
                    )
                )
                item_counter += 1
                if item_counter % 8 == 0 or item_counter == n_total:
                    print(f"[study-axis1 {item_counter:04d}/{n_total:04d}]", flush=True)
    finally:
        runtime.close()

    _shuffle_in_unison(items, manifest_items, seed=seed + 101)

    session = axis1_team_session(
        session_id=f"{session_name}__axis1_eval",
        title=f"Axis 1 Eval Session: {session_name}",
        items=items,
        roles=_axis1_eval_roles(),
        storage_key=f"autolabel.study.{session_name}.axis1_pairwise.v3",
        export_filename="axis1_eval_response.json",
        intro_html=(
            "Each item fixes one feature label and shows two candidate tokens from the same image. "
            "Choose the single token that best matches the label."
        ),
        footer_html=(
            "Export the JSON and save it as <code>axis1_eval_response.json</code> in this session directory, "
            "then run the score command."
        ),
    )
    write_study_page(session_dir / "axis1_eval.html", session)
    manifest = {
        "kind": "axis1_eval",
        "session_name": session_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "evidence_profile": evidence_profile,
        "evaluator_prompt_version": _norm_text(evaluator_prompt_version),
        "selected_feature_keys": [str(feature["feature_key"]) for feature in selected_features],
        "design": {
            "positives_per_holdout": 1,
            "hard_negatives_per_holdout": hard_negatives_per_holdout,
            "easy_negatives_per_holdout": easy_negatives_per_holdout,
            "order_randomized": True,
            "choice_task": "pairwise_hard_choice",
            "main_metric": "top1_accuracy",
            "chance_accuracy": 0.5,
        },
        "items": manifest_items,
        "study_session_manifest": build_study_manifest(session),
        "html_path": str(session_dir / "axis1_eval.html"),
        "default_response_json": str(session_dir / "axis1_eval_response.json"),
    }
    write_json(session_dir / "axis1_eval_manifest.json", manifest)
    return manifest


def score_study_axis1_session(
    config: EvalConfig,
    *,
    session_name: str,
    response_json: Path | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    session_dir = _resolve_session_dir(config, session_name, session_dir=session_dir)
    manifest_path = session_dir / "axis1_eval_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Axis 1 manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    response_path = Path(response_json) if response_json is not None else session_dir / "axis1_eval_response.json"
    if not response_path.exists():
        raise FileNotFoundError(
            f"Missing Axis 1 response at {response_path}. Export JSON from axis1_eval.html first."
        )
    response_payload = _read_session_response(response_path)
    evidence_profile = _normalize_evidence_profile(manifest.get("evidence_profile"))

    per_feature_meta: dict[str, dict[str, Any]] = {}
    score_rows: list[dict[str, Any]] = []
    incomplete_item_ids: list[str] = []
    per_feature_correct: dict[str, list[int]] = {}
    confidence_values: list[float] = []
    for item in manifest["items"]:
        item_state = dict(response_payload["state"]["items"].get(str(item["axis1_item_id"]), {}))
        if not _axis1_response_complete(item_state):
            incomplete_item_ids.append(str(item["axis1_item_id"]))
            continue
        feature_key_value = str(item["feature_key"])
        per_feature_meta[feature_key_value] = {
            "block_idx": int(item["block_idx"]),
            "feature_id": int(item["feature_id"]),
            "canonical_label": str(item["canonical_label"]),
        }
        selected_candidate, score_0_100 = _final_axis1_selected_candidate(item_state)
        positive_candidate = next((c for c in item["candidates"] if int(c["label"]) == 1), None)
        if positive_candidate is None:
            raise RuntimeError(f"Axis1 item {item['axis1_item_id']} has no positive candidate")
        correct = int(str(positive_candidate["candidate_code"]).lower() == selected_candidate)
        per_feature_correct.setdefault(feature_key_value, []).append(correct)
        confidence_values.append(float(score_0_100) / 100.0)
        logging_fields = _extract_axis_logging_fields(item_state, evidence_profile=evidence_profile)
        score_rows.append(
            {
                "axis1_item_id": str(item["axis1_item_id"]),
                "feature_key": feature_key_value,
                "block_idx": int(item["block_idx"]),
                "feature_id": int(item["feature_id"]),
                "sample_id": int(item["sample_id"]),
                "selected_candidate": selected_candidate,
                "positive_candidate": str(positive_candidate["candidate_code"]).lower(),
                "selected_token_uid": next(
                    str(c["token_uid"]) for c in item["candidates"] if str(c["candidate_code"]).lower() == selected_candidate
                ),
                "correct": correct,
                "confidence_0_100": float(score_0_100),
                "provider": "human_interactive_html",
                "prompt_version": _norm_text(manifest.get("evaluator_prompt_version")) or "study_axis1_human_v3_pairwise",
                **logging_fields,
            }
        )

    if incomplete_item_ids:
        preview = ", ".join(incomplete_item_ids[:8])
        suffix = "" if len(incomplete_item_ids) <= 8 else f" ... (+{len(incomplete_item_ids) - 8} more)"
        raise ValueError(
            "Axis 1 response is incomplete. Fill every item with explicit decision and score before scoring: "
            f"{preview}{suffix}"
        )

    per_feature: dict[str, Any] = {}
    feature_accuracies: list[float] = []
    for feature_key_value in sorted(per_feature_correct):
        values = np.asarray(per_feature_correct[feature_key_value], dtype=np.float32)
        acc = float(values.mean()) if values.size else float("nan")
        per_feature[feature_key_value] = {
            **per_feature_meta[feature_key_value],
            "n_items": int(values.size),
            "n_correct": int(values.sum()),
            "accuracy": acc,
            "chance_accuracy": 0.5,
        }
        feature_accuracies.append(acc)
    overall_accuracy = float(np.mean([row["correct"] for row in score_rows])) if score_rows else float("nan")
    summary = {
        "session_name": session_name,
        "response_json": str(response_path),
        "evidence_profile": evidence_profile,
        "evaluator_prompt_version": _norm_text(manifest.get("evaluator_prompt_version")),
        "per_feature": per_feature,
        "overall": {
            "top1_accuracy": overall_accuracy,
            "macro_accuracy": float(np.nanmean(np.asarray(feature_accuracies, dtype=np.float64))) if feature_accuracies else float("nan"),
            "chance_accuracy": 0.5,
            "mean_confidence": float(np.mean(confidence_values)) if confidence_values else float("nan"),
            "n_items": int(len(score_rows)),
        },
    }
    write_jsonl(session_dir / "axis1_eval_scores.jsonl", score_rows)
    write_json(session_dir / "axis1_eval_results.json", summary)
    return summary


def build_study_axis2_session(
    config: EvalConfig,
    *,
    session_name: str,
    features_per_block: int | None = None,
    seed: int | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
    accepted_rows: dict[str, dict[str, Any]] | None = None,
    session_dir: Path | None = None,
    evidence_profile: dict[str, Any] | None = None,
    evaluator_prompt_version: str | None = None,
) -> dict[str, Any]:
    feature_bank = load_feature_bank(config)
    features_by_key = _feature_lookup(feature_bank)
    accepted = accepted_rows or accepted_label_map(config)
    seed = int(seed if seed is not None else config.study_session_default_seed)
    evidence_profile = _normalize_evidence_profile(evidence_profile)
    selected_features = _selected_or_fallback_labeled_features(
        config,
        session_name=session_name,
        features_per_block=features_per_block,
        seed=seed,
        feature_specs=feature_specs,
        accepted_rows=accepted,
        session_dir=session_dir,
    )
    if not selected_features:
        raise RuntimeError("No accepted labels available. Run label team + ingest first.")
    session_dir = _resolve_session_dir(config, session_name, session_dir=session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    accepted_by_block: dict[int, list[dict[str, Any]]] = {}
    for block_idx in config.blocks:
        block_features = []
        for key, label_row in accepted.items():
            feature = features_by_key.get(key)
            if feature is not None and int(feature["block_idx"]) == int(block_idx):
                block_features.append(
                    {
                        "feature": feature,
                        "label_row": label_row,
                        "threshold": float(config.axis2_positive_relative_threshold) * _feature_target_activation_scale(feature),
                    }
                )
        accepted_by_block[int(block_idx)] = sorted(
            block_features,
            key=lambda row: (int(row["feature"]["feature_id"]), str(row["feature"]["feature_key"])),
        )

    runtime = LegacyRuntime(config)
    items: list[StudyItem] = []
    manifest_items: list[dict[str, Any]] = []
    token_cache: dict[str, dict[str, str]] = {}
    try:
        total = sum(len(feature["holdout"]) for feature in selected_features)
        done = 0
        candidates_per_item = max(2, int(config.study_axis2_candidates_per_item))
        for feature in selected_features:
            block_idx = int(feature["block_idx"])
            train_sample_ids = {int(row["sample_id"]) for row in feature["train"]}
            holdout_sample_ids = {int(row["sample_id"]) for row in feature["holdout"]}
            overlap_sample_ids = sorted(train_sample_ids & holdout_sample_ids)
            if overlap_sample_ids:
                raise RuntimeError(
                    f"Label/eval leakage detected for {feature['feature_key']}: overlapping sample_ids {overlap_sample_ids}"
                )
            block_candidates = accepted_by_block.get(block_idx, [])
            if len(block_candidates) < 2:
                continue
            source_key = str(feature["feature_key"])
            source_row = next((row for row in block_candidates if str(row["feature"]["feature_key"]) == source_key), None)
            if source_row is None:
                continue
            negative_pool = [
                row for row in block_candidates if str(row["feature"]["feature_key"]) != source_key
            ]
            if not negative_pool:
                continue
            for holdout_row in feature["holdout"]:
                image_path = str(holdout_row["image_path"])
                sample_id = int(holdout_row["sample_id"])
                token_idx = int(holdout_row["target_patch_idx"])
                uid = token_uid(block_idx, sample_id, token_idx)
                if uid not in token_cache:
                    token_dir = session_dir / "axis2_assets" / _slug(uid)
                    original_path = token_dir / "original_token_box.png"
                    erf_path = token_dir / "token_erf_support.png"
                    erf_zoom_path = token_dir / "token_erf_zoom.png"
                    erf_zoom_detail_path = token_dir / "token_erf_zoom_detail.png"
                    cosine_path = token_dir / "token_neighbor_cosine.png"
                    erf_json_path = token_dir / "token_erf.json"
                    save_original_with_token_box(image_path, original_path, token_idx)
                    erf = runtime.cautious_token_erf(image_path, block_idx, token_idx)
                    save_support_mask_image(image_path, erf["support_indices"], erf_path, token_idx=token_idx)
                    save_support_outline_crop_image(
                        image_path,
                        erf["support_indices"],
                        erf_zoom_path,
                        score_map=erf["prob_scores"],
                    )
                    if evidence_profile.get("include_erf_zoom_detail"):
                        save_support_detail_crop_image(
                            image_path,
                            erf["support_indices"],
                            erf_zoom_detail_path,
                            token_idx=token_idx,
                        )
                    cosine = runtime.token_cosine_map(image_path, block_idx, token_idx)
                    save_cosine_overlay_image(image_path, cosine, cosine_path, token_idx=token_idx)
                    write_json(erf_json_path, erf)
                    token_cache[uid] = {
                        "original_with_token_box": _relpath(session_dir, original_path),
                        "token_erf": _relpath(session_dir, erf_path),
                        "token_erf_zoom": _relpath(session_dir, erf_zoom_path),
                        "token_erf_zoom_detail": _relpath(session_dir, erf_zoom_detail_path)
                        if evidence_profile.get("include_erf_zoom_detail")
                        else "",
                        "token_neighbor_cosine": _relpath(session_dir, cosine_path),
                        "token_erf_json": _relpath(session_dir, erf_json_path),
                    }
                local_rng = random.Random(seed + int(block_idx) * 100000 + int(sample_id) * 101 + int(token_idx))
                shuffled_negatives = list(negative_pool)
                local_rng.shuffle(shuffled_negatives)
                chosen_rows = [source_row, *shuffled_negatives[: max(0, candidates_per_item - 1)]]
                candidate_rows = [
                    {
                        "feature_key": str(row["feature"]["feature_key"]),
                        "feature_id": int(row["feature"]["feature_id"]),
                        "canonical_label": str(row["label_row"]["canonical_label"]),
                        "description": _norm_text(row["label_row"].get("description")),
                        "threshold": float(row["threshold"]),
                    }
                    for row in chosen_rows
                ]
                shuffled_candidates = list(candidate_rows)
                local_rng.shuffle(shuffled_candidates)
                for idx, row in enumerate(shuffled_candidates, start=1):
                    row["candidate_code"] = f"c{idx:02d}"
                shuffled_feature_ids = [int(row["feature_id"]) for row in shuffled_candidates]
                shuffled_thresholds = np.asarray([float(row["threshold"]) for row in shuffled_candidates], dtype=np.float32)
                activations = runtime.feature_vector_at_token(image_path, block_idx, token_idx, shuffled_feature_ids)
                positives = (activations >= shuffled_thresholds).astype(np.int64)
                item_id = _axis2_item_id(block_idx, sample_id, token_idx)
                payload = {
                    "axis2_item_id": item_id,
                    "block_idx": block_idx,
                    "sample_id": sample_id,
                    "token_idx": token_idx,
                    "token_uid": uid,
                    "source_feature_key": str(feature["feature_key"]),
                    "candidates": shuffled_candidates,
                    "candidate_feature_ids": [int(v) for v in shuffled_feature_ids],
                    "candidate_codes": [str(row["candidate_code"]) for row in shuffled_candidates],
                    "ground_truth": positives.astype(int).tolist(),
                    "activation_values": activations.astype(float).tolist(),
                    "token_erf_zoom_detail": token_cache[uid].get("token_erf_zoom_detail", ""),
                    **token_cache[uid],
                }
                manifest_items.append(payload)
                items.append(
                    StudyItem(
                        item_id=item_id,
                        title=f"block {block_idx} | sample {sample_id} | tok {token_idx}",
                        evidence_html=_axis2_item_evidence_html(payload),
                        metadata={"block": block_idx, "sample_id": sample_id, "source": str(feature["feature_key"])},
                    )
                )
                done += 1
                if done % 8 == 0 or done == total:
                    print(f"[study-axis2 {done:03d}/{total:03d}]", flush=True)
    finally:
        runtime.close()

    if not manifest_items:
        raise RuntimeError(
            "Axis 2 needs at least two accepted labels in the same block. "
            "Build and ingest more label-team outputs first."
        )

    _shuffle_in_unison(items, manifest_items, seed=seed + 211)

    session = axis1_team_session(
        session_id=f"{session_name}__axis2_eval",
        title=f"Axis 2 Eval Session: {session_name}",
        items=items,
        roles=_axis2_eval_roles(),
        storage_key=f"autolabel.study.{session_name}.axis2.v1",
        export_filename="axis2_eval_response.json",
        intro_html=(
            "Each item fixes one token and asks you to rank the candidate feature labels for that token."
        ),
        footer_html=(
            "Export the JSON and save it as <code>axis2_eval_response.json</code> in this session directory, "
            "then run the score command."
        ),
    )
    write_study_page(session_dir / "axis2_eval.html", session)
    manifest = {
        "kind": "axis2_eval",
        "session_name": session_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "evidence_profile": evidence_profile,
        "evaluator_prompt_version": _norm_text(evaluator_prompt_version),
        "selected_feature_keys": [str(feature["feature_key"]) for feature in selected_features],
        "design": {
            "candidates_per_item_max": candidates_per_item,
            "task": "small_bank_feature_discrimination",
        },
        "items": manifest_items,
        "study_session_manifest": build_study_manifest(session),
        "html_path": str(session_dir / "axis2_eval.html"),
        "default_response_json": str(session_dir / "axis2_eval_response.json"),
    }
    write_json(session_dir / "axis2_eval_manifest.json", manifest)
    return manifest


def score_study_axis2_session(
    config: EvalConfig,
    *,
    session_name: str,
    response_json: Path | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    session_dir = _resolve_session_dir(config, session_name, session_dir=session_dir)
    manifest_path = session_dir / "axis2_eval_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Axis 2 manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    response_path = Path(response_json) if response_json is not None else session_dir / "axis2_eval_response.json"
    if not response_path.exists():
        raise FileNotFoundError(
            f"Missing Axis 2 response at {response_path}. Export JSON from axis2_eval.html first."
        )
    response_payload = _read_session_response(response_path)
    evidence_profile = _normalize_evidence_profile(manifest.get("evidence_profile"))

    per_item: list[dict[str, Any]] = []
    matrix_true: list[np.ndarray] = []
    matrix_score: list[np.ndarray] = []
    candidate_columns: dict[str, list[tuple[str, int]]] = {}
    for item in manifest["items"]:
        item_state = dict(response_payload["state"]["items"].get(str(item["axis2_item_id"]), {}))
        valid_codes = {str(code).lower() for code in item["candidate_codes"]}
        ranked = _final_axis2_ranking(item_state, valid_codes)
        score_map = {code: float(len(ranked) - idx) for idx, code in enumerate(ranked)}
        y_score = np.asarray([score_map.get(str(code).lower(), 0.0) for code in item["candidate_codes"]], dtype=np.float32)
        y_true = np.asarray(item["ground_truth"], dtype=np.int64)
        logging_fields = _extract_axis_logging_fields(item_state, evidence_profile=evidence_profile)
        matrix_true.append(y_true)
        matrix_score.append(y_score)
        candidate_columns[str(item["axis2_item_id"])] = [
            (str(candidate["feature_key"]), int(candidate["feature_id"])) for candidate in item["candidates"]
        ]
        per_item.append(
            {
                "axis2_item_id": str(item["axis2_item_id"]),
                "block_idx": int(item["block_idx"]),
                "sample_id": int(item["sample_id"]),
                "token_uid": str(item["token_uid"]),
                "n_candidates": int(len(item["candidate_codes"])),
                "n_positive": int(y_true.sum()),
                "ranked_candidates": ranked,
                "ap": average_precision_binary(y_true, y_score),
                "ndcg": ndcg_at_k(y_true, y_score, k=len(item["candidate_codes"])),
                **{f"recall_at_{k}": recall_at_k(y_true, y_score, int(k)) for k in config.axis2_metric_ks},
                **logging_fields,
            }
        )

    if not matrix_true:
        raise RuntimeError("No Axis 2 items were available to score.")

    y_true_matrix = np.stack(matrix_true, axis=0)
    y_score_matrix = np.stack(matrix_score, axis=0)
    overall = {
        "mAP": float(np.nanmean([row["ap"] for row in per_item])),
        f"nDCG@{y_true_matrix.shape[1]}": float(np.nanmean([row["ndcg"] for row in per_item])),
    }
    for k in config.axis2_metric_ks:
        overall[f"Recall@{int(k)}"] = float(np.nanmean([row[f"recall_at_{int(k)}"] for row in per_item]))

    feature_truth: dict[str, list[int]] = {}
    feature_score: dict[str, list[float]] = {}
    for row, item in enumerate(manifest["items"]):
        for col, candidate in enumerate(item["candidates"]):
            key = str(candidate["feature_key"])
            feature_truth.setdefault(key, []).append(int(y_true_matrix[row, col]))
            feature_score.setdefault(key, []).append(float(y_score_matrix[row, col]))
    per_feature_one_vs_rest: dict[str, Any] = {}
    for key in sorted(feature_truth):
        yt = np.asarray(feature_truth[key], dtype=np.int64)
        ys = np.asarray(feature_score[key], dtype=np.float32)
        per_feature_one_vs_rest[key] = {
            "n_items": int(yt.size),
            "n_positive": int(yt.sum()),
            "auprc": average_precision_binary(yt, ys),
            **f1_accuracy_at_threshold(yt, ys, threshold=float(config.decision_threshold)),
        }

    summary = {
        "session_name": session_name,
        "response_json": str(response_path),
        "evidence_profile": evidence_profile,
        "evaluator_prompt_version": _norm_text(manifest.get("evaluator_prompt_version")),
        "overall": overall,
        "per_item": per_item,
        "per_feature_one_vs_rest": per_feature_one_vs_rest,
    }
    write_json(session_dir / "axis2_eval_results.json", summary)
    return summary
