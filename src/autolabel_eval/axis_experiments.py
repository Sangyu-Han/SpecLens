from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .utils import read_json, read_jsonl, write_json


SUBSET_DESCRIPTIONS: dict[str, str] = {
    "microtexture_stroke": "Tiny texture, stripe, stroke, check, mosaic, or mesh-like local motifs.",
    "color_relative_position": "Features whose interpretation depends on both appearance/color and a local relative position cue.",
    "boundary_frame": "Features driven by borders, frames, seams, rims, side edges, or explicit boundary contact.",
}

_MICROTEXTURE_PATTERNS = (
    "texture",
    "stripe",
    "stroke",
    "slat",
    "mesh",
    "check",
    "checker",
    "mosaic",
    "pattern",
    "tick",
    "ribbed",
    "hatch",
    "grid",
    "crosshatch",
    "sliver",
)
_COLOR_PATTERNS = (
    "red",
    "pink",
    "orange",
    "yellow",
    "brown",
    "black",
    "white",
    "dark",
    "bright",
    "pale",
    "light",
    "color",
    "reddish",
    "pinkish",
    "yellowish",
    "brownish",
    "blackish",
    "whitish",
)
_RELATIVE_PATTERNS = (
    "below",
    "above",
    "left",
    "right",
    "under",
    "over",
    "lower",
    "upper",
    "adjacent",
    "neighbor",
    "offset",
    "horizontal",
    "vertical",
    "diagonal",
)
_BOUNDARY_PATTERNS = (
    "border",
    "boundary",
    "frame",
    "rim",
    "edge",
    "side",
    "seam",
    "rail",
    "outline",
    "contour",
    "contact",
)
_AXIS1_FEATURE_METRICS = (
    "auprc",
    "auroc",
    "f1_at_0.5",
    "accuracy_at_0.5",
    "precision_at_0.5",
    "recall_at_0.5",
    "accuracy",
    "chance_accuracy",
)
_AXIS1_OVERALL_METRICS = (
    "macro_auprc",
    "macro_auroc",
    "macro_f1_at_0.5",
    "micro_auprc",
    "micro_auroc",
    "f1_at_0.5",
    "accuracy_at_0.5",
    "precision_at_0.5",
    "recall_at_0.5",
    "top1_accuracy",
    "macro_accuracy",
    "chance_accuracy",
    "mean_confidence",
    "n_items",
)
_AXIS2_FEATURE_METRICS = (
    "auprc",
    "f1_at_0.5",
    "accuracy_at_0.5",
    "precision_at_0.5",
    "recall_at_0.5",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(pattern in lower for pattern in patterns)


def _derive_subset_tags(text: str, meta_tags: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    if _contains_any(text, _MICROTEXTURE_PATTERNS) or bool(meta_tags.get("high_frequency_texture")):
        tags.add("microtexture_stroke")
    color_hit = _contains_any(text, _COLOR_PATTERNS)
    relative_hit = _contains_any(text, _RELATIVE_PATTERNS)
    if (color_hit and relative_hit) or bool(meta_tags.get("appearance_color_relative_position")):
        tags.add("color_relative_position")
    if _contains_any(text, _BOUNDARY_PATTERNS) or bool(meta_tags.get("positional_or_border_bias")):
        tags.add("boundary_frame")
    return sorted(tags)


def _iter_feature_state_paths(config: EvalConfig) -> list[Path]:
    if not config.autolabel_root.exists():
        return []
    return sorted(path for path in config.autolabel_root.glob("*/feature_state.jsonl") if path.exists())


def _feature_catalog_from_sources(config: EvalConfig) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}

    def ensure_row(feature_key: str, *, block_idx: int | None = None, feature_id: int | None = None) -> dict[str, Any]:
        row = catalog.setdefault(
            feature_key,
            {
                "feature_key": feature_key,
                "block_idx": int(block_idx) if block_idx is not None else None,
                "feature_id": int(feature_id) if feature_id is not None else None,
                "text_fields": [],
                "meta_tags": {},
                "sources": [],
            },
        )
        if row.get("block_idx") is None and block_idx is not None:
            row["block_idx"] = int(block_idx)
        if row.get("feature_id") is None and feature_id is not None:
            row["feature_id"] = int(feature_id)
        return row

    if config.label_registry_jsonl.exists():
        for row in read_jsonl(config.label_registry_jsonl):
            status = _norm_text(row.get("status")).lower()
            if status not in {"accept", "accepted"}:
                continue
            feature_key = _norm_text(row.get("feature_key"))
            if not feature_key:
                continue
            entry = ensure_row(
                feature_key,
                block_idx=_safe_float(row.get("block_idx")),
                feature_id=_safe_float(row.get("feature_id")),
            )
            for key in ("canonical_label", "description", "notes"):
                value = _norm_text(row.get(key))
                if value:
                    entry["text_fields"].append(value)
            labeler = dict(row.get("labeler") or {})
            for key in ("canonical_label", "description", "notes"):
                value = _norm_text(labeler.get(key))
                if value:
                    entry["text_fields"].append(value)
            entry["sources"].append(
                {
                    "source_type": "label_registry",
                    "session_name": _norm_text(row.get("session_name")),
                    "status": status,
                }
            )

    for path in _iter_feature_state_paths(config):
        session_name = path.parent.name
        for row in read_jsonl(path):
            feature_key = _norm_text(row.get("feature_key"))
            if not feature_key:
                continue
            final_decision = _norm_text(row.get("final_human_decision")).lower()
            status = _norm_text(row.get("status")).lower()
            if final_decision not in {"accept", "accepted"} and status != "accepted":
                continue
            entry = ensure_row(
                feature_key,
                block_idx=_safe_float(row.get("block_idx")),
                feature_id=_safe_float(row.get("feature_id")),
            )
            final_label = dict(row.get("final_label") or {})
            latest_student = dict(row.get("latest_student") or {})
            latest_human_feedback = dict(row.get("latest_human_feedback") or {})
            for payload in (final_label, latest_student, latest_human_feedback):
                for key in ("canonical_label", "description", "notes"):
                    value = _norm_text(payload.get(key))
                    if value:
                        entry["text_fields"].append(value)
            meta_tags = row.get("meta_tags") or {}
            if isinstance(meta_tags, dict):
                entry["meta_tags"].update(meta_tags)
            human_meta_tags = latest_human_feedback.get("meta_tags") or {}
            if isinstance(human_meta_tags, dict):
                entry["meta_tags"].update(human_meta_tags)
            entry["sources"].append(
                {
                    "source_type": "autolabel_session",
                    "session_name": session_name,
                    "status": status,
                    "final_human_decision": final_decision,
                }
            )

    return catalog


def build_axis_feature_subsets(
    config: EvalConfig,
    *,
    output_json: Path | None = None,
) -> dict[str, Any]:
    catalog = _feature_catalog_from_sources(config)
    feature_subsets: dict[str, Any] = {}
    subsets: dict[str, list[str]] = {name: [] for name in SUBSET_DESCRIPTIONS}
    for feature_key in sorted(catalog):
        row = catalog[feature_key]
        text_blob = "\n".join(row["text_fields"])
        tags = _derive_subset_tags(text_blob, row["meta_tags"])
        feature_subsets[feature_key] = {
            "feature_key": feature_key,
            "block_idx": row["block_idx"],
            "feature_id": row["feature_id"],
            "subsets": tags,
            "text_excerpt": text_blob[:400],
            "meta_tags": row["meta_tags"],
            "sources": row["sources"],
        }
        for tag in tags:
            subsets.setdefault(tag, []).append(feature_key)
    payload = {
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "subset_descriptions": SUBSET_DESCRIPTIONS,
        "feature_subsets": feature_subsets,
        "subsets": {key: sorted(values) for key, values in subsets.items()},
    }
    target_path = output_json or config.axis_feature_subsets_json
    write_json(target_path, payload)
    payload["output_json"] = str(target_path)
    return payload


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [value for value in (_safe_float(row.get(metric)) for row in rows) if value is not None and np.isfinite(value)]
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _feature_subset_summary(
    per_feature: dict[str, Any],
    feature_keys: list[str],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    rows = [dict(per_feature[key]) for key in feature_keys if key in per_feature]
    return {
        "n_features": len(rows),
        **{metric: _mean_metric(rows, metric) for metric in metrics},
    }


def _overall_metric_summary(payload: dict[str, Any], metric_keys: tuple[str, ...]) -> dict[str, Any]:
    overall = dict(payload.get("overall") or {})
    summary = {
        key: _safe_float(overall.get(key))
        for key in metric_keys
        if key in overall and _safe_float(overall.get(key)) is not None
    }
    if summary:
        return summary
    return {
        key: _safe_float(value)
        for key, value in overall.items()
        if _safe_float(value) is not None
    }


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(current.keys()) | set(baseline.keys()))
    delta: dict[str, Any] = {}
    for key in keys:
        current_value = _safe_float(current.get(key))
        baseline_value = _safe_float(baseline.get(key))
        if current_value is None or baseline_value is None:
            continue
        delta[key] = float(current_value - baseline_value)
    return delta


def _render_metric_table(title: str, metrics: dict[str, Any], deltas: dict[str, Any] | None = None) -> str:
    rows = []
    for key, value in metrics.items():
        if key == "n_features" or value is None:
            continue
        delta_html = ""
        if deltas and key in deltas:
            delta = float(deltas[key])
            sign = "+" if delta >= 0 else ""
            delta_html = f"<div class='delta'>{sign}{delta:.4f}</div>"
        rows.append(
            f"""
            <div class="metric-card">
              <div class="metric-label">{html.escape(str(key))}</div>
              <div class="metric-value">{value:.4f}</div>
              {delta_html}
            </div>
            """
        )
    if not rows:
        rows.append("<div class='muted'>No metrics available.</div>")
    return f"<section><h3>{html.escape(title)}</h3><div class='metric-grid'>{''.join(rows)}</div></section>"


def _render_axis_experiment_html(summary: dict[str, Any], output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    variant_sections: list[str] = []
    for variant in summary["variants"]:
        subset_sections: list[str] = []
        for subset_name, subset_payload in variant.get("subsets", {}).items():
            axis1_metrics = dict(subset_payload.get("axis1") or {})
            axis2_metrics = dict(subset_payload.get("axis2") or {})
            axis1_delta = dict((variant.get("delta_vs_baseline") or {}).get("subsets", {}).get(subset_name, {}).get("axis1") or {})
            axis2_delta = dict((variant.get("delta_vs_baseline") or {}).get("subsets", {}).get(subset_name, {}).get("axis2") or {})
            subset_sections.append(
                f"""
                <div class="subset-card">
                  <h4>{html.escape(subset_name)}</h4>
                  <div class="muted">{html.escape(summary.get('subset_descriptions', {}).get(subset_name, ''))}</div>
                  {_render_metric_table('Axis 1 subset', axis1_metrics, axis1_delta)}
                  {_render_metric_table('Axis 2 subset', axis2_metrics, axis2_delta)}
                </div>
                """
            )
        variant_sections.append(
            f"""
            <section class="variant-card">
              <h2>{html.escape(str(variant['variant_id']))}</h2>
              <div class="muted">{html.escape(str(variant.get('label') or ''))}</div>
              {_render_metric_table('Axis 1 overall', dict(variant.get('axis1', {}).get('overall') or {}), dict((variant.get('delta_vs_baseline') or {}).get('axis1_overall') or {}))}
              {_render_metric_table('Axis 2 overall', dict(variant.get('axis2', {}).get('overall') or {}), dict((variant.get('delta_vs_baseline') or {}).get('axis2_overall') or {}))}
              <div class="subset-grid">{''.join(subset_sections) if subset_sections else "<div class='muted'>No subset metrics.</div>"}</div>
            </section>
            """
        )
    output_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(summary['experiment_name'])}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; background: #f7f4ee; color: #1f2937; }}
    h1, h2, h3, h4 {{ margin: 0 0 8px; }}
    .muted {{ color: #6b7280; margin-bottom: 12px; }}
    .variant-card, .subset-card {{ background: white; border: 1px solid #ddd6cc; border-radius: 16px; padding: 16px; margin-bottom: 20px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 10px 0 16px; }}
    .metric-card {{ border: 1px solid #ece7df; border-radius: 12px; padding: 10px 12px; background: #fff; }}
    .metric-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
    .metric-value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .delta {{ font-size: 12px; color: #7c3aed; margin-top: 4px; }}
    .subset-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
  <h1>{html.escape(summary['experiment_name'])}</h1>
  <div class="muted">Baseline variant: {html.escape(summary['baseline_variant'])}</div>
  {''.join(variant_sections)}
</body>
</html>""",
        encoding="utf-8",
    )


def summarize_axis_experiment(
    config: EvalConfig,
    *,
    spec_json: Path,
    subsets_json: Path | None = None,
    output_json: Path | None = None,
    output_html: Path | None = None,
) -> dict[str, Any]:
    spec = read_json(spec_json)
    experiment_name = _norm_text(spec.get("experiment_name")) or spec_json.stem
    target_root = config.axis_experiments_root / experiment_name
    target_root.mkdir(parents=True, exist_ok=True)

    subset_payload: dict[str, Any] = {}
    subset_path = subsets_json or (Path(spec["subsets_json"]) if spec.get("subsets_json") else None)
    if subset_path is None and config.axis_feature_subsets_json.exists():
        subset_path = config.axis_feature_subsets_json
    if subset_path is not None and subset_path.exists():
        subset_payload = read_json(subset_path)
    subset_descriptions = dict(subset_payload.get("subset_descriptions") or SUBSET_DESCRIPTIONS)
    subset_map = {
        subset_name: list(feature_keys)
        for subset_name, feature_keys in dict(subset_payload.get("subsets") or {}).items()
    }

    variants = list(spec.get("variants") or [])
    if not variants:
        raise ValueError("Axis experiment spec must include at least one variant.")
    baseline_variant = _norm_text(spec.get("baseline_variant")) or _norm_text(variants[0].get("variant_id"))

    variant_summaries: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = _norm_text(variant.get("variant_id"))
        if not variant_id:
            raise ValueError("Each axis experiment variant must include variant_id.")
        axis1_payload = read_json(Path(variant["axis1_metrics_json"])) if variant.get("axis1_metrics_json") else {}
        axis2_payload = read_json(Path(variant["axis2_results_json"])) if variant.get("axis2_results_json") else {}
        axis1_overall = _overall_metric_summary(axis1_payload, _AXIS1_OVERALL_METRICS)
        axis2_overall = _overall_metric_summary(axis2_payload, tuple())
        subsets_summary: dict[str, Any] = {}
        for subset_name, feature_keys in subset_map.items():
            subsets_summary[subset_name] = {
                "axis1": _feature_subset_summary(dict(axis1_payload.get("per_feature") or {}), feature_keys, _AXIS1_FEATURE_METRICS),
                "axis2": _feature_subset_summary(dict(axis2_payload.get("per_feature_one_vs_rest") or {}), feature_keys, _AXIS2_FEATURE_METRICS),
            }
        variant_summaries.append(
            {
                "variant_id": variant_id,
                "label": _norm_text(variant.get("label")) or variant_id,
                "artifacts": {
                    "axis1_metrics_json": str(variant.get("axis1_metrics_json") or ""),
                    "axis2_results_json": str(variant.get("axis2_results_json") or ""),
                },
                "axis1": {"overall": axis1_overall},
                "axis2": {"overall": axis2_overall},
                "subsets": subsets_summary,
            }
        )

    baseline = next((variant for variant in variant_summaries if variant["variant_id"] == baseline_variant), None)
    if baseline is None:
        raise ValueError(f"Baseline variant {baseline_variant!r} not found in experiment spec.")
    for variant in variant_summaries:
        variant["delta_vs_baseline"] = {
            "axis1_overall": _metric_delta(dict(variant.get("axis1", {}).get("overall") or {}), dict(baseline.get("axis1", {}).get("overall") or {})),
            "axis2_overall": _metric_delta(dict(variant.get("axis2", {}).get("overall") or {}), dict(baseline.get("axis2", {}).get("overall") or {})),
            "subsets": {
                subset_name: {
                    "axis1": _metric_delta(
                        dict(variant.get("subsets", {}).get(subset_name, {}).get("axis1") or {}),
                        dict(baseline.get("subsets", {}).get(subset_name, {}).get("axis1") or {}),
                    ),
                    "axis2": _metric_delta(
                        dict(variant.get("subsets", {}).get(subset_name, {}).get("axis2") or {}),
                        dict(baseline.get("subsets", {}).get(subset_name, {}).get("axis2") or {}),
                    ),
                }
                for subset_name in subset_map
            },
        }

    summary = {
        "experiment_name": experiment_name,
        "created_at": _now_iso(),
        "config": config.to_dict(),
        "baseline_variant": baseline_variant,
        "spec_json": str(spec_json),
        "subsets_json": str(subset_path) if subset_path is not None else None,
        "subset_descriptions": subset_descriptions,
        "variants": variant_summaries,
    }
    summary_path = output_json or (target_root / "summary.json")
    html_path = output_html or (target_root / "summary.html")
    write_json(summary_path, summary)
    _render_axis_experiment_html(summary, html_path)
    summary["output_json"] = str(summary_path)
    summary["output_html"] = str(html_path)
    return summary
