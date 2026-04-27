from __future__ import annotations

import html
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .axis_experiments import build_axis_feature_subsets, summarize_axis_experiment
from .config import EvalConfig
from .feature_bank import load_feature_bank
from .legacy import LegacyRuntime
from .rendering import (
    save_feature_actmap_overlay,
    save_original_with_token_box,
    save_support_detail_crop_image,
    save_support_outline_crop_image,
)
from .study_html import (
    RoleSpec,
    StudyItem,
    axis1_team_session,
    select_field,
    session_manifest as build_study_manifest,
    textarea_field,
    write_study_page,
)
from .study_protocol import (
    _collect_label_examples,
    _axis1_item_evidence_html,
    _axis2_item_evidence_html,
    _feature_lookup,
    _norm_text,
    _normalize_evidence_profile,
    _safe_float,
    accepted_label_map,
    build_study_axis1_session,
    build_study_axis2_session,
    build_study_label_session,
    ingest_study_label_session,
    score_study_axis1_session,
    score_study_axis2_session,
)
from .utils import feature_key, read_json, write_json, write_jsonl


DEFAULT_LABEL_PROMPT_VERSION = "study_label_team_v1"
DEFAULT_EVALUATOR_PROMPT_VERSION = "study_axis_evaluator_v1"
DEFAULT_AUDIT_CONFIDENCE_THRESHOLD = 0.55


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_feature_spec(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value)
    if ":" in text:
        left, right = text.split(":", 1)
        return int(left), int(right)
    if text.startswith("block_") and "/feature_" in text:
        block_part, feature_part = text.split("/feature_", 1)
        return int(block_part.replace("block_", "")), int(feature_part)
    raise ValueError(f"Unsupported feature spec: {value!r}")


def _normalize_feature_specs(specs: list[Any] | None) -> list[tuple[int, int]]:
    return [_parse_feature_spec(spec) for spec in list(specs or [])]


def _experiment_root(config: EvalConfig, experiment_name: str) -> Path:
    return config.axis_runs_root / experiment_name


def _demand_path(config: EvalConfig, experiment_name: str) -> Path:
    return config.axis_label_demand_root / f"{experiment_name}.json"


def _label_session_dir(config: EvalConfig, experiment_name: str) -> Path:
    return config.axis_label_sessions_root / experiment_name


def _audit_dir(config: EvalConfig, experiment_name: str, variant_id: str) -> Path:
    return config.axis_audits_root / experiment_name / variant_id


def _rebase_relative_paths(value: Any, *, source_dir: Path, target_dir: Path) -> Any:
    if isinstance(value, dict):
        return {str(key): _rebase_relative_paths(subvalue, source_dir=source_dir, target_dir=target_dir) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_rebase_relative_paths(subvalue, source_dir=source_dir, target_dir=target_dir) for subvalue in value]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.startswith(("http://", "https://", "data:")):
            return value
        path = Path(text)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".json"}:
            return value
        resolved = path.resolve() if path.is_absolute() else (source_dir / path).resolve()
        return os.path.relpath(resolved, start=target_dir.resolve())
    return value


def _load_or_default_spec(
    *,
    config: EvalConfig,
    experiment_name: str,
    spec_json: Path | None,
    feature_specs: list[tuple[int, int]] | None,
    features_per_block: int | None,
    seed: int | None,
) -> dict[str, Any]:
    if spec_json is not None:
        payload = read_json(spec_json)
    else:
        payload = {}
    payload.setdefault("experiment_name", experiment_name)
    if feature_specs:
        payload.setdefault("feature_specs", [[int(block_idx), int(feature_id)] for block_idx, feature_id in feature_specs])
    if features_per_block is not None:
        payload.setdefault("features_per_block", int(features_per_block))
    if seed is not None:
        payload.setdefault("seed", int(seed))
    payload.setdefault("variants", [{"variant_id": "raw_only", "label": "Raw ERF zoom only", "include_erf_zoom_detail": False}])
    payload.setdefault("label_prompt_version", DEFAULT_LABEL_PROMPT_VERSION)
    payload.setdefault("evaluator_prompt_version", DEFAULT_EVALUATOR_PROMPT_VERSION)
    return payload


def _select_experiment_features(
    config: EvalConfig,
    *,
    feature_specs: list[tuple[int, int]] | None,
    features_per_block: int | None,
    seed: int | None,
) -> list[dict[str, Any]]:
    feature_bank = load_feature_bank(config)
    features_by_key = _feature_lookup(feature_bank)
    if feature_specs:
        selected: list[dict[str, Any]] = []
        for block_idx, feature_id in feature_specs:
            key = feature_key(int(block_idx), int(feature_id))
            feature = features_by_key.get(key)
            if feature is None:
                raise KeyError(f"Unknown feature {key}")
            selected.append(feature)
        return selected

    requested = int(features_per_block or config.study_session_default_features_per_block)
    rng = random.Random(int(seed if seed is not None else config.study_session_default_seed))
    selected = []
    for block_idx in config.blocks:
        candidates = list(feature_bank["blocks"][str(block_idx)]["features"])
        rng.shuffle(candidates)
        selected.extend(candidates[:requested])
    return selected


def _load_prior_feature_signals(config: EvalConfig, experiment_name: str) -> dict[str, dict[str, int]]:
    signals: dict[str, dict[str, int]] = defaultdict(lambda: {"audit_hits": 0, "low_metric_hits": 0})
    experiment_root = _experiment_root(config, experiment_name)
    if experiment_root.exists():
        for results_path in experiment_root.glob("*/axis1_eval_results.json"):
            payload = read_json(results_path)
            for feature_key_value, row in dict(payload.get("per_feature") or {}).items():
                if (_safe_float(row.get("accuracy")) or 0.0) < 0.75:
                    signals[str(feature_key_value)]["low_metric_hits"] += 1
        for results_path in experiment_root.glob("*/axis2_eval_results.json"):
            payload = read_json(results_path)
            for feature_key_value, row in dict(payload.get("per_feature_one_vs_rest") or {}).items():
                if (_safe_float(row.get("auprc")) or 0.0) < 0.50:
                    signals[str(feature_key_value)]["low_metric_hits"] += 1
    audits_root = config.axis_audits_root / experiment_name
    if audits_root.exists():
        for audit_path in audits_root.glob("*/audit_results.json"):
            payload = read_json(audit_path)
            for row in list(payload.get("items") or []):
                for feature_key_value in list(row.get("related_feature_keys") or []):
                    signals[str(feature_key_value)]["audit_hits"] += 1
    return signals


def build_axis_label_demand(
    config: EvalConfig,
    *,
    experiment_name: str,
    spec_json: Path | None = None,
    feature_specs: list[tuple[int, int]] | None = None,
    features_per_block: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    spec = _load_or_default_spec(
        config=config,
        experiment_name=experiment_name,
        spec_json=spec_json,
        feature_specs=feature_specs,
        features_per_block=features_per_block,
        seed=seed,
    )
    selected = _select_experiment_features(
        config,
        feature_specs=_normalize_feature_specs(spec.get("feature_specs")),
        features_per_block=spec.get("features_per_block"),
        seed=spec.get("seed"),
    )
    accepted = accepted_label_map(config)
    prior_signals = _load_prior_feature_signals(config, experiment_name)
    rows: list[dict[str, Any]] = []
    for feature in selected:
        key = str(feature["feature_key"])
        label_row = dict(accepted.get(key) or {})
        signal = prior_signals.get(key, {})
        rows.append(
            {
                "feature_key": key,
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "appears_in_axis1": True,
                "appears_in_axis2": True,
                "is_missing_label": not bool(label_row),
                "reason_for_demand": "selected_for_frozen_axis_eval",
                "label_status": _norm_text(label_row.get("status")) or ("missing" if not label_row else "accepted"),
                "cached_label": {
                    "canonical_label": _norm_text(label_row.get("canonical_label")),
                    "description": _norm_text(label_row.get("description")),
                }
                if label_row
                else {},
                "prior_audit_hits": int(signal.get("audit_hits") or 0),
                "prior_low_metric_hits": int(signal.get("low_metric_hits") or 0),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["is_missing_label"] else 1,
            -int(row["prior_audit_hits"]),
            -int(row["prior_low_metric_hits"]),
            int(row["block_idx"]),
            int(row["feature_id"]),
        )
    )
    for idx, row in enumerate(rows, start=1):
        row["priority_rank"] = idx

    payload = {
        "experiment_name": experiment_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "spec": spec,
        "selected_feature_keys": [str(feature["feature_key"]) for feature in selected],
        "selected_feature_specs": [[int(feature["block_idx"]), int(feature["feature_id"])] for feature in selected],
        "rows": rows,
        "summary": {
            "n_selected": len(rows),
            "n_missing_label": sum(int(row["is_missing_label"]) for row in rows),
            "n_cached_label": sum(int(not row["is_missing_label"]) for row in rows),
        },
    }
    target_path = _demand_path(config, experiment_name)
    write_json(target_path, payload)
    payload["output_json"] = str(target_path)
    return payload


def run_dynamic_label_generation(
    config: EvalConfig,
    *,
    experiment_name: str,
    response_json: Path | None = None,
    provider_type: str = "axis_labeler",
) -> dict[str, Any]:
    demand_path = _demand_path(config, experiment_name)
    if not demand_path.exists():
        raise FileNotFoundError(f"Missing axis label demand at {demand_path}")
    demand = read_json(demand_path)
    missing_specs = [
        (int(row["block_idx"]), int(row["feature_id"]))
        for row in list(demand.get("rows") or [])
        if bool(row.get("is_missing_label"))
    ]
    session_dir = _label_session_dir(config, experiment_name)
    if not missing_specs:
        summary = {
            "experiment_name": experiment_name,
            "session_dir": str(session_dir),
            "built": False,
            "ingested": False,
            "missing_feature_count": 0,
            "remaining_missing_feature_count": 0,
        }
        write_json(session_dir / "label_generation_summary.json", summary)
        return summary

    spec = dict(demand.get("spec") or {})
    session_name = f"{experiment_name}__axis_label_demand"
    label_prompt_version = _norm_text(spec.get("label_prompt_version")) or DEFAULT_LABEL_PROMPT_VERSION
    label_variant = _norm_text(spec.get("label_variant")) or "axis_demand"
    label_session = build_study_label_session(
        config,
        session_name=session_name,
        feature_specs=missing_specs,
        session_dir=session_dir,
        label_prompt_version=label_prompt_version,
        label_variant=label_variant,
        generated_for_experiment=experiment_name,
        evidence_profile={"variant_id": "raw_only", "label": "Raw ERF zoom only", "include_erf_zoom_detail": False},
    )
    ingested = False
    if response_json is not None or (session_dir / "label_team_response.json").exists():
        ingest_study_label_session(
            config,
            session_name=session_name,
            session_dir=session_dir,
            response_json=response_json,
            provider_type=provider_type,
        )
        ingested = True
        demand = build_axis_label_demand(
            config,
            experiment_name=experiment_name,
            spec_json=None,
            feature_specs=_normalize_feature_specs(spec.get("feature_specs")),
            features_per_block=spec.get("features_per_block"),
            seed=spec.get("seed"),
        )
    summary = {
        "experiment_name": experiment_name,
        "session_dir": str(session_dir),
        "built": True,
        "ingested": ingested,
        "missing_feature_count": len(missing_specs),
        "remaining_missing_feature_count": int(dict(demand.get("summary") or {}).get("n_missing_label") or 0),
        "label_session_html": str(session_dir / "label_team.html"),
        "label_session_response_json": str(session_dir / "label_team_response.json"),
        "label_prompt_version": label_prompt_version,
    }
    write_json(session_dir / "label_generation_summary.json", summary)
    return summary


def _accepted_snapshot(config: EvalConfig, feature_keys: list[str]) -> dict[str, dict[str, Any]]:
    accepted = accepted_label_map(config)
    missing = [key for key in feature_keys if key not in accepted]
    if missing:
        raise RuntimeError(f"Cannot freeze axis experiment; missing accepted labels for {missing}")
    return {key: dict(accepted[key]) for key in feature_keys}


def build_axis_experiment(
    config: EvalConfig,
    *,
    experiment_name: str,
    spec_json: Path | None = None,
) -> dict[str, Any]:
    demand_path = _demand_path(config, experiment_name)
    if not demand_path.exists():
        raise FileNotFoundError(f"Missing axis label demand at {demand_path}")
    demand = read_json(demand_path)
    demand_spec = dict(demand.get("spec") or {})
    spec = dict(demand_spec)
    spec.setdefault("experiment_name", experiment_name)
    if spec_json is not None:
        override_spec = dict(read_json(spec_json))
        merged_variants = override_spec.get("variants", spec.get("variants"))
        spec.update(override_spec)
        if merged_variants is not None:
            spec["variants"] = merged_variants
    spec.setdefault("feature_specs", demand.get("selected_feature_specs") or demand_spec.get("feature_specs") or [])
    if "features_per_block" not in spec and demand_spec.get("features_per_block") is not None:
        spec["features_per_block"] = demand_spec.get("features_per_block")
    if "seed" not in spec and demand_spec.get("seed") is not None:
        spec["seed"] = demand_spec.get("seed")
    spec.setdefault("variants", [{"variant_id": "raw_only", "label": "Raw ERF zoom only", "include_erf_zoom_detail": False}])
    spec.setdefault("label_prompt_version", DEFAULT_LABEL_PROMPT_VERSION)
    spec.setdefault("evaluator_prompt_version", DEFAULT_EVALUATOR_PROMPT_VERSION)
    feature_specs = _normalize_feature_specs(demand.get("selected_feature_specs"))
    feature_keys = [str(row["feature_key"]) for row in list(demand.get("rows") or [])]
    frozen_labels = _accepted_snapshot(config, feature_keys)
    experiment_root = _experiment_root(config, experiment_name)
    experiment_root.mkdir(parents=True, exist_ok=True)
    frozen_label_path = experiment_root / "frozen_labels.json"
    write_json(frozen_label_path, frozen_labels)

    variant_payloads: list[dict[str, Any]] = []
    evaluator_prompt_version = _norm_text(spec.get("evaluator_prompt_version")) or DEFAULT_EVALUATOR_PROMPT_VERSION
    for raw_variant in list(spec.get("variants") or []):
        evidence_profile = _normalize_evidence_profile(raw_variant)
        variant_id = str(evidence_profile["variant_id"])
        variant_dir = experiment_root / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        session_name = f"{experiment_name}__{variant_id}"
        axis1_manifest = build_study_axis1_session(
            config,
            session_name=session_name,
            feature_specs=feature_specs,
            accepted_rows=frozen_labels,
            session_dir=variant_dir,
            evidence_profile=evidence_profile,
            evaluator_prompt_version=evaluator_prompt_version,
        )
        axis2_manifest = build_study_axis2_session(
            config,
            session_name=session_name,
            feature_specs=feature_specs,
            accepted_rows=frozen_labels,
            session_dir=variant_dir,
            evidence_profile=evidence_profile,
            evaluator_prompt_version=evaluator_prompt_version,
        )
        variant_payloads.append(
            {
                "variant_id": variant_id,
                "label": evidence_profile["label"],
                "evidence_profile": evidence_profile,
                "variant_dir": str(variant_dir),
                "axis1_manifest_json": str(variant_dir / "axis1_eval_manifest.json"),
                "axis1_html": axis1_manifest["html_path"],
                "axis1_response_json": axis1_manifest["default_response_json"],
                "axis2_manifest_json": str(variant_dir / "axis2_eval_manifest.json"),
                "axis2_html": axis2_manifest["html_path"],
                "axis2_response_json": axis2_manifest["default_response_json"],
            }
        )

    manifest = {
        "experiment_name": experiment_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "spec": spec,
        "selected_feature_keys": feature_keys,
        "selected_feature_specs": [[int(block_idx), int(feature_id)] for block_idx, feature_id in feature_specs],
        "frozen_labels_json": str(frozen_label_path),
        "variants": variant_payloads,
    }
    write_json(experiment_root / "experiment_manifest.json", manifest)
    return manifest


def run_axis_experiment(
    config: EvalConfig,
    *,
    experiment_name: str,
) -> dict[str, Any]:
    experiment_root = _experiment_root(config, experiment_name)
    manifest_path = experiment_root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing experiment manifest at {manifest_path}")
    manifest = read_json(manifest_path)
    variant_summaries: list[dict[str, Any]] = []
    incomplete_variants: list[str] = []
    for variant in list(manifest.get("variants") or []):
        variant_dir = Path(variant["variant_dir"])
        axis1_response = Path(variant["axis1_response_json"])
        axis2_response = Path(variant["axis2_response_json"])
        if not axis1_response.exists() or not axis2_response.exists():
            incomplete_variants.append(str(variant["variant_id"]))
            continue
        axis1_summary = score_study_axis1_session(
            config,
            session_name=f"{experiment_name}__{variant['variant_id']}",
            session_dir=variant_dir,
        )
        axis2_summary = score_study_axis2_session(
            config,
            session_name=f"{experiment_name}__{variant['variant_id']}",
            session_dir=variant_dir,
        )
        variant_summaries.append(
            {
                "variant_id": str(variant["variant_id"]),
                "label": str(variant["label"]),
                "axis1_metrics_json": str(variant_dir / "axis1_eval_results.json"),
                "axis2_results_json": str(variant_dir / "axis2_eval_results.json"),
                "axis1_overall": axis1_summary["overall"],
                "axis2_overall": axis2_summary["overall"],
            }
        )
    if not variant_summaries:
        raise RuntimeError("No complete variants were available to score.")
    if not config.axis_feature_subsets_json.exists():
        build_axis_feature_subsets(config)
    summary_spec = {
        "experiment_name": experiment_name,
        "baseline_variant": str((manifest.get("spec") or {}).get("baseline_variant") or variant_summaries[0]["variant_id"]),
        "subsets_json": str(config.axis_feature_subsets_json),
        "variants": [
            {
                "variant_id": row["variant_id"],
                "label": row["label"],
                "axis1_metrics_json": row["axis1_metrics_json"],
                "axis2_results_json": row["axis2_results_json"],
            }
            for row in variant_summaries
        ],
    }
    summary_spec_path = experiment_root / "summary_spec.json"
    write_json(summary_spec_path, summary_spec)
    summary = summarize_axis_experiment(config, spec_json=summary_spec_path)
    run_summary = {
        "experiment_name": experiment_name,
        "completed_variants": [row["variant_id"] for row in variant_summaries],
        "incomplete_variants": incomplete_variants,
        "summary_json": summary["output_json"],
        "summary_html": summary["output_html"],
    }
    write_json(experiment_root / "run_summary.json", run_summary)
    return run_summary


def _axis_failure_audit_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(
            role_id="auditor",
            title="Auditor",
            instructions=(
                "First answer the same task the agent saw using the same evidence. "
                "Then judge whether the label was good enough and whether the model used the evidence correctly. "
                "Korean notes are okay."
            ),
            fields=(
                select_field("selected_candidate", "Axis1 selected candidate", ("c01", "c02", "c03", "c04")),
                select_field(
                    "score_0_10",
                    "Axis1 confidence 0-10",
                    tuple(str(v) for v in range(11)),
                    help_text="Use for Axis1 items only.",
                ),
                select_field("best_candidate", "Axis2 best candidate", ("c01", "c02", "c03", "c04")),
                textarea_field(
                    "ranked_candidates",
                    "Axis2 ranked candidates",
                    rows=3,
                    placeholder="c01, c04, c02",
                    help_text="Use for Axis2 items only.",
                ),
                textarea_field(
                    "brief_reason",
                    "Your task reason",
                    rows=3,
                    placeholder="Why did you choose this answer? Korean is okay.",
                ),
                select_field("audit_decision", "Audit decision", ("accept", "revise", "unclear")),
                select_field("label_quality_judgment", "Label quality", ("good", "needs_revision", "unclear")),
                select_field("axis_answer_quality_judgment", "Axis answer quality", ("good", "bad", "unclear")),
                textarea_field("audit_feedback", "Audit feedback", rows=5, placeholder="What information was missing or misused?"),
            ),
        ),
    )


def _collect_feature_support_pack(
    config: EvalConfig,
    *,
    feature: dict[str, Any],
    out_dir: Path,
    include_erf_zoom_detail: bool,
    max_examples: int = 10,
) -> list[dict[str, Any]]:
    runtime = LegacyRuntime(config)
    frame_cache: dict[int, Any] = {}
    sid_to_path_cache: dict[int, str] = {}
    try:
        examples = _collect_label_examples(
            config=config,
            runtime=runtime,
            feature=feature,
            target_count=max_examples,
            frame_cache=frame_cache,
            sid_to_path_cache=sid_to_path_cache,
        )
        rendered: list[dict[str, Any]] = []
        for rank, row in enumerate(examples):
            image_path = str(row["image_path"])
            token_idx = int(row["target_patch_idx"])
            original_path = out_dir / f"support_{rank:02d}_original.png"
            actmap_path = out_dir / f"support_{rank:02d}_actmap.png"
            erf_zoom_path = out_dir / f"support_{rank:02d}_erf_zoom.png"
            erf_zoom_detail_path = out_dir / f"support_{rank:02d}_erf_zoom_detail.png"
            save_original_with_token_box(image_path, original_path, token_idx)
            actmap = runtime.feature_activation_map(image_path, int(feature["block_idx"]), int(feature["feature_id"]))
            save_feature_actmap_overlay(image_path, actmap, actmap_path, token_idx=token_idx)
            erf = runtime.cautious_token_erf(image_path, int(feature["block_idx"]), token_idx)
            save_support_outline_crop_image(image_path, erf["support_indices"], erf_zoom_path, token_idx=token_idx, score_map=erf["prob_scores"])
            if include_erf_zoom_detail:
                save_support_detail_crop_image(image_path, erf["support_indices"], erf_zoom_detail_path, token_idx=token_idx)
            rendered.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "token_idx": token_idx,
                    "original_with_token_box": str(original_path),
                    "feature_actmap": str(actmap_path),
                    "token_erf_zoom": str(erf_zoom_path),
                    "token_erf_zoom_detail": str(erf_zoom_detail_path) if include_erf_zoom_detail else "",
                }
            )
        return rendered
    finally:
        runtime.close()


def _audit_item_evidence_html(item: dict[str, Any]) -> str:
    task_item = dict(item.get("task_item") or {})
    if str(item.get("axis_kind")) == "axis1":
        task_html = _axis1_item_evidence_html(task_item)
        target_html = f"<div><strong>Ground truth candidate:</strong> {html.escape(_norm_text(item.get('ground_truth_answer')))}</div>"
        agent_html = f"<div><strong>Agent selected:</strong> {html.escape(_norm_text(item.get('agent_answer')))}</div>"
    else:
        task_html = _axis2_item_evidence_html(task_item)
        target_html = f"<div><strong>Ground truth positives:</strong> {html.escape(', '.join(list(item.get('ground_truth_answers') or [])))}</div>"
        agent_html = f"<div><strong>Agent best candidate:</strong> {html.escape(_norm_text(item.get('agent_answer')))}</div>"
    support_sections = []
    for pack in list(item.get("feature_support_packs") or []):
        sample_cards = []
        for sample in list(pack.get("samples") or []):
            detail_html = ""
            if _norm_text(sample.get("token_erf_zoom_detail")):
                detail_html = f"""
                <div>
                  <img src="{html.escape(str(sample['token_erf_zoom_detail']))}" alt="feature-erf-zoom-detail" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;">
                  <div style="font-size:12px;color:#666;margin-top:4px;">ERF zoom detail</div>
                </div>
                """
            sample_cards.append(
                f"""
                <div style="border:1px solid #ddd;border-radius:12px;padding:10px;background:#fff;">
                  <div style="font-size:12px;color:#666;margin-bottom:6px;">sample {int(sample['sample_id'])} tok {int(sample['token_idx'])}</div>
                  <div style="display:grid;grid-template-columns:repeat({4 if detail_html else 3},minmax(0,1fr));gap:8px;">
                    <div><img src="{html.escape(str(sample['original_with_token_box']))}" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;"><div style="font-size:12px;color:#666;margin-top:4px;">Original + token</div></div>
                    <div><img src="{html.escape(str(sample['feature_actmap']))}" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;"><div style="font-size:12px;color:#666;margin-top:4px;">Feature actmap</div></div>
                    <div><img src="{html.escape(str(sample['token_erf_zoom']))}" style="width:100%;aspect-ratio:1/1;object-fit:contain;background:#111;border:1px solid #ccc;border-radius:8px;"><div style="font-size:12px;color:#666;margin-top:4px;">Feature ERF zoom</div></div>
                    {detail_html}
                  </div>
                </div>
                """
            )
        support_sections.append(
            f"""
            <section style="margin-top:16px;">
              <h4 style="margin:0 0 8px;">Support pack: {html.escape(str(pack['feature_key']))}</h4>
              <div style="font-size:13px;color:#666;margin-bottom:8px;">{html.escape(_norm_text(pack.get('canonical_label')))}</div>
              <div style="display:grid;gap:10px;">{''.join(sample_cards)}</div>
            </section>
            """
        )

    return f"""
    <div>
      <div style="border:1px solid #ddd;border-radius:12px;padding:12px;background:#fff;margin-bottom:12px;">
        <div style="font-weight:700;margin-bottom:6px;">Task + Agent result</div>
        <div><strong>Axis kind:</strong> {html.escape(str(item['axis_kind']))}</div>
        <div><strong>Correct:</strong> {html.escape(str(item['correct']))}</div>
        <div><strong>Confidence:</strong> {html.escape(str(item.get('confidence', '')))}</div>
        {agent_html}
        {target_html}
        <div><strong>Answer summary:</strong> {html.escape(_norm_text(item.get('answer_summary')))}</div>
        <div><strong>Agent rationale:</strong> {html.escape(_norm_text(item.get('free_text_rationale')))}</div>
      </div>
      {task_html}
      {''.join(support_sections)}
    </div>
    """


def _load_axis_variant_payload(variant_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    axis1_manifest = read_json(variant_dir / "axis1_eval_manifest.json")
    axis1_results = read_json(variant_dir / "axis1_eval_results.json")
    axis2_manifest = read_json(variant_dir / "axis2_eval_manifest.json")
    axis2_results = read_json(variant_dir / "axis2_eval_results.json")
    return axis1_manifest, axis1_results, axis2_manifest, axis2_results


def build_axis_failure_audit(
    config: EvalConfig,
    *,
    experiment_name: str,
    variant_id: str,
    confidence_threshold: float = DEFAULT_AUDIT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    experiment_root = _experiment_root(config, experiment_name)
    variant_dir = experiment_root / variant_id
    if not variant_dir.exists():
        raise FileNotFoundError(f"Missing variant directory at {variant_dir}")
    axis1_manifest, axis1_results, axis2_manifest, axis2_results = _load_axis_variant_payload(variant_dir)
    evidence_profile = _normalize_evidence_profile(axis1_manifest.get("evidence_profile"))
    feature_bank = load_feature_bank(config)
    features_by_key = _feature_lookup(feature_bank)
    audit_root = _audit_dir(config, experiment_name, variant_id)
    audit_root.mkdir(parents=True, exist_ok=True)

    audit_items: list[dict[str, Any]] = []
    axis1_items_by_id = {str(item["axis1_item_id"]): item for item in list(axis1_manifest.get("items") or [])}
    for row in list(axis1_results.get("per_item", []) or []):
        confidence = _safe_float(row.get("confidence")) or (_safe_float(row.get("confidence_0_100")) or 0.0) / 100.0
        is_failure = int(row.get("correct", 0)) == 0 or confidence < float(confidence_threshold) or bool(row.get("failure_tags"))
        if not is_failure:
            continue
        manifest_item = axis1_items_by_id.get(str(row["axis1_item_id"]))
        if manifest_item is None:
            continue
        related_feature_keys = [str(manifest_item["feature_key"])]
        ground_truth_answer = next(
            (
                str(candidate["candidate_code"])
                for candidate in manifest_item["candidates"]
                if int(candidate.get("label", 0)) == 1
            ),
            _norm_text(row.get("positive_candidate")),
        )
        audit_items.append(
            {
                "audit_item_id": f"axis1::{row['axis1_item_id']}",
                "axis_kind": "axis1",
                "related_feature_keys": related_feature_keys,
                "correct": bool(row.get("correct")),
                "confidence": confidence,
                "agent_answer": _norm_text(row.get("selected_candidate")),
                "ground_truth_answer": ground_truth_answer,
                "answer_summary": f"selected={row.get('selected_candidate')} positive={row.get('positive_candidate')}",
                "free_text_rationale": _norm_text(row.get("free_text_rationale")),
                "task_item": _rebase_relative_paths(manifest_item, source_dir=variant_dir, target_dir=audit_root),
            }
        )

    axis2_items_by_id = {str(item["axis2_item_id"]): item for item in list(axis2_manifest.get("items") or [])}
    for row in list(axis2_results.get("per_item", []) or []):
        ranked = list(row.get("ranked_candidates") or [])
        top_code = str(ranked[0]).lower() if ranked else ""
        manifest_item = axis2_items_by_id.get(str(row["axis2_item_id"]))
        if manifest_item is None:
            continue
        top_correct = False
        if top_code:
            for idx, code in enumerate(manifest_item["candidate_codes"]):
                if str(code).lower() == top_code:
                    top_correct = bool(manifest_item["ground_truth"][idx])
                    break
        confidence = _safe_float(row.get("confidence"))
        is_failure = (not top_correct) or (confidence is not None and confidence < float(confidence_threshold)) or bool(row.get("failure_tags"))
        if not is_failure:
            continue
        related_feature_keys = [str(candidate["feature_key"]) for candidate, is_positive in zip(manifest_item["candidates"], manifest_item["ground_truth"]) if int(is_positive) == 1]
        audit_items.append(
            {
                "audit_item_id": f"axis2::{row['axis2_item_id']}",
                "axis_kind": "axis2",
                "related_feature_keys": related_feature_keys,
                "correct": top_correct,
                "confidence": confidence,
                "agent_answer": top_code,
                "ground_truth_answers": [
                    str(code)
                    for code, is_positive in zip(manifest_item["candidate_codes"], manifest_item["ground_truth"])
                    if int(is_positive) == 1
                ],
                "answer_summary": f"top_ranked={','.join(ranked[:3])}",
                "free_text_rationale": _norm_text(row.get("free_text_rationale")),
                "task_item": _rebase_relative_paths(manifest_item, source_dir=variant_dir, target_dir=audit_root),
            }
        )

    if not audit_items:
        summary = {
            "experiment_name": experiment_name,
            "variant_id": variant_id,
            "html_path": "",
            "default_response_json": str(audit_root / "axis_failure_audit_response.json"),
            "n_items": 0,
            "message": "No failed or uncertain Axis items met the audit threshold.",
        }
        write_json(audit_root / "axis_failure_audit_manifest.json", summary)
        return summary

    current_accepted = accepted_label_map(config)
    for item in audit_items:
        support_packs = []
        for related_feature_key in item["related_feature_keys"]:
            feature = features_by_key.get(str(related_feature_key))
            if feature is None:
                continue
            pack_dir = audit_root / "assets" / str(item["audit_item_id"]).replace("/", "__") / str(related_feature_key).replace("/", "__")
            samples = _collect_feature_support_pack(
                config,
                feature=feature,
                out_dir=pack_dir,
                include_erf_zoom_detail=bool(evidence_profile.get("include_erf_zoom_detail")),
                max_examples=10,
            )
            samples = _rebase_relative_paths(samples, source_dir=audit_root, target_dir=audit_root)
            support_packs.append(
                {
                    "feature_key": str(related_feature_key),
                    "canonical_label": _norm_text(current_accepted.get(str(related_feature_key), {}).get("canonical_label")),
                    "samples": samples,
                }
            )
        item["feature_support_packs"] = support_packs

    study_items = [
        StudyItem(
            item_id=str(item["audit_item_id"]),
            title=f"{item['axis_kind']} | {item['audit_item_id']}",
            evidence_html=_audit_item_evidence_html(item),
            metadata={"axis_kind": item["axis_kind"]},
        )
        for item in audit_items
    ]
    session = axis1_team_session(
        session_id=f"{experiment_name}__{variant_id}__axis_audit",
        title=f"Axis Failure Audit: {experiment_name} / {variant_id}",
        items=study_items,
        roles=_axis_failure_audit_roles(),
        storage_key=f"autolabel.axis_audit.{experiment_name}.{variant_id}.v1",
        export_filename="axis_failure_audit_response.json",
        intro_html="Review only the failed or uncertain Axis items. Decide whether the label was sufficient and whether the model used the right evidence.",
        footer_html="Export the JSON and save it as <code>axis_failure_audit_response.json</code> in this audit directory, then ingest it.",
    )
    write_study_page(audit_root / "axis_failure_audit.html", session)
    manifest = {
        "kind": "axis_failure_audit",
        "experiment_name": experiment_name,
        "variant_id": variant_id,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "evidence_profile": evidence_profile,
        "items": audit_items,
        "study_session_manifest": build_study_manifest(session),
        "html_path": str(audit_root / "axis_failure_audit.html"),
        "default_response_json": str(audit_root / "axis_failure_audit_response.json"),
    }
    write_json(audit_root / "axis_failure_audit_manifest.json", manifest)
    return manifest


def ingest_axis_failure_audit(
    config: EvalConfig,
    *,
    experiment_name: str,
    variant_id: str,
    response_json: Path | None = None,
) -> dict[str, Any]:
    audit_root = _audit_dir(config, experiment_name, variant_id)
    manifest_path = audit_root / "axis_failure_audit_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing audit manifest at {manifest_path}")
    manifest = read_json(manifest_path)
    response_path = Path(response_json) if response_json is not None else audit_root / "axis_failure_audit_response.json"
    if not response_path.exists():
        raise FileNotFoundError(f"Missing audit response at {response_path}")
    payload = read_json(response_path)
    if "state" not in payload:
        raise ValueError(f"Unsupported audit response payload at {response_path}")
    items = []
    for item in list(manifest.get("items") or []):
        state = dict(payload["state"]["items"].get(str(item["audit_item_id"]), {}))
        auditor = dict(state.get("auditor", {}))
        axis_kind = str(item["axis_kind"])
        human_selected_candidate = _norm_text(auditor.get("selected_candidate"))
        human_best_candidate = _norm_text(auditor.get("best_candidate"))
        human_ranked_candidates = _norm_text(auditor.get("ranked_candidates"))
        human_brief_reason = _norm_text(auditor.get("brief_reason"))
        human_confidence_0_10 = _safe_float(auditor.get("score_0_10"))
        agent_answer = _norm_text(item.get("agent_answer"))
        human_answer = human_selected_candidate if axis_kind == "axis1" else human_best_candidate
        if axis_kind == "axis1":
            ground_truth = _norm_text(item.get("ground_truth_answer"))
            human_axis_answer_correct = human_answer.lower() == ground_truth.lower() if human_answer and ground_truth else None
        else:
            positives = [str(code).lower() for code in list(item.get("ground_truth_answers") or [])]
            human_axis_answer_correct = human_answer.lower() in positives if human_answer and positives else None
        items.append(
            {
                "audit_item_id": str(item["audit_item_id"]),
                "axis_kind": axis_kind,
                "related_feature_keys": list(item.get("related_feature_keys") or []),
                "human_selected_candidate": human_selected_candidate,
                "human_best_candidate": human_best_candidate,
                "human_ranked_candidates": human_ranked_candidates,
                "human_brief_reason": human_brief_reason,
                "human_confidence_0_10": human_confidence_0_10,
                "human_matches_agent": human_answer.lower() == agent_answer.lower() if human_answer and agent_answer else None,
                "human_axis_answer_correct": human_axis_answer_correct,
                "audit_decision": _norm_text(auditor.get("audit_decision")),
                "audit_feedback": _norm_text(auditor.get("audit_feedback")),
                "label_quality_judgment": _norm_text(auditor.get("label_quality_judgment")),
                "axis_answer_quality_judgment": _norm_text(auditor.get("axis_answer_quality_judgment")),
            }
        )
    summary = {
        "experiment_name": experiment_name,
        "variant_id": variant_id,
        "response_json": str(response_path),
        "items": items,
        "decision_counts": dict(Counter(_norm_text(item["audit_decision"]) for item in items if _norm_text(item["audit_decision"]))),
    }
    write_json(audit_root / "audit_results.json", summary)
    return summary


def review_axis_experiment(
    config: EvalConfig,
    *,
    experiment_name: str,
    response_json: Path | None = None,
) -> dict[str, Any]:
    experiment_root = _experiment_root(config, experiment_name)
    summary_path = config.axis_experiments_root / experiment_name / "summary.json"
    if not summary_path.exists():
        summary_path = experiment_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing axis experiment summary for {experiment_name}")
    summary = read_json(summary_path)
    audits_root = config.axis_audits_root / experiment_name
    audit_payloads = []
    if audits_root.exists():
        for audit_path in sorted(audits_root.glob("*/audit_results.json")):
            audit_payloads.append(read_json(audit_path))
    failure_hist = Counter()
    for variant in list(summary.get("variants") or []):
        variant_dir = experiment_root / str(variant["variant_id"])
        for result_name in ("axis1_eval_results.json", "axis2_eval_results.json"):
            result_path = variant_dir / result_name
            if not result_path.exists():
                continue
            payload = read_json(result_path)
            for row in list(payload.get("per_item") or []):
                for tag in list(row.get("failure_tags") or []):
                    failure_hist[str(tag)] += 1
    review_packet = {
        "experiment_name": experiment_name,
        "created_at": _now_iso(),
        "summary_json": str(summary_path),
        "baseline_variant": summary.get("baseline_variant"),
        "variants": summary.get("variants"),
        "audit_results": audit_payloads,
        "failure_tag_histogram": dict(failure_hist),
        "instruction": "Review metric deltas, per-item failure tags, and optional human audits. Propose prompt patches for the labeler prompt only; do not revise the evaluator prompt.",
    }
    write_json(experiment_root / "review_packet.json", review_packet)
    markdown = [
        f"# Axis Experiment Review: {experiment_name}",
        "",
        f"- Baseline variant: `{summary.get('baseline_variant')}`",
        f"- Failure tags: `{dict(failure_hist)}`",
        "",
        "Use this packet to generate labeler-prompt-only patch proposals from metric failures and audits.",
    ]
    (experiment_root / "review_packet.md").write_text("\n".join(markdown), encoding="utf-8")
    output = {
        "experiment_name": experiment_name,
        "review_packet_json": str(experiment_root / "review_packet.json"),
        "review_packet_md": str(experiment_root / "review_packet.md"),
    }
    if response_json is not None:
        reviewer_response = read_json(response_json)
        write_json(experiment_root / "review_response.json", reviewer_response)
        output["review_response_json"] = str(experiment_root / "review_response.json")
    return output
