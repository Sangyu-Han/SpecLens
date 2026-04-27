from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .config import EvalConfig
from .legacy import LegacyRuntime, token_record_from_row
from .rendering import (
    save_feature_actmap_overlay,
    save_original_with_token_box,
    save_support_mask_image,
    save_support_outline_crop_image,
)
from .utils import feature_key, write_json


def _html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f5f5; color: #111; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .feature {{ background: white; border: 1px solid #ddd; padding: 16px; margin-bottom: 20px; }}
    .sample {{ background: #fafafa; border: 1px solid #e5e5e5; padding: 12px; margin-top: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(6, minmax(180px, 1fr)); gap: 12px; align-items: start; }}
    .panel {{ background: #fff; border: 1px solid #e5e5e5; padding: 8px; }}
    img {{ width: 100%; max-width: 280px; border: 1px solid #ccc; display: block; }}
    .meta {{ font-size: 12px; color: #555; margin-top: 4px; }}
    code {{ background: #eee; padding: 1px 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _item_relpath(base_dir: Path, path: Path) -> str:
    return str(path.relative_to(base_dir))


def _collect_feature_rows(
    runtime: LegacyRuntime,
    config: EvalConfig,
    *,
    block_idx: int,
    feature_id: int,
    required_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = runtime.load_decile_frame(block_idx)
    feature_rows = frame[frame["unit"] == int(feature_id)].sort_values("score", ascending=False)
    if feature_rows.empty:
        raise ValueError(f"block {block_idx} feature {feature_id} is absent from decile ledger")

    sample_ids = [int(sid) for sid in feature_rows["sample_id"].astype(int).unique().tolist()]
    sid_to_path = runtime.lookup_paths(sample_ids)
    accepted: list[dict[str, Any]] = []
    for row in feature_rows.itertuples(index=False):
        tok_idx = runtime.row_x_to_token_idx(int(row.x))
        if tok_idx < 0 or tok_idx >= config.n_patches:
            continue
        img_path = sid_to_path.get(int(row.sample_id), "")
        if not img_path:
            continue
        validation = runtime.validate_feature_token(
            img_path,
            int(block_idx),
            int(feature_id),
            tok_idx,
            float(row.score),
        )
        if validation is None:
            continue
        accepted.append(
            token_record_from_row(
                block_idx,
                int(feature_id),
                row,
                img_path,
                validation,
                token_idx=tok_idx,
            )
        )
        if len(accepted) >= int(required_count):
            break

    stats = {
        "candidate_rows": int(len(feature_rows)),
        "required_examples": int(required_count),
        "accepted_examples": int(len(accepted)),
    }
    if len(accepted) < int(required_count):
        raise RuntimeError(
            f"block {block_idx} feature {feature_id} only yielded {len(accepted)} validated examples; "
            f"required {required_count}"
        )
    return accepted, stats


def _objective_label(objective_mode: str) -> str:
    if objective_mode == "cosine":
        return "Cosine"
    if objective_mode == "unit_ref_dot":
        return "Unit-ref dot"
    return objective_mode


def _support_from_trace(erf_payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    ranking = [int(v) for v in erf_payload["ranking"]]
    trace = list(erf_payload["recovery_trace"])
    if not trace:
        raise ValueError("Empty ERF recovery_trace")

    support_size = len(ranking)
    support_recovery = float(trace[-1].get("score", trace[-1].get("cosine", 0.0)))
    for entry in trace:
        recovery = float(entry.get("score", entry.get("cosine", 0.0)))
        if recovery >= float(threshold):
            support_size = int(entry["budget"])
            support_recovery = recovery
            break
    support_indices = ranking[:support_size]
    return {
        "threshold": float(threshold),
        "support_size": int(support_size),
        "support_recovery": float(support_recovery),
        "support_indices": support_indices,
    }


def _support_from_cumulative_mass(erf_payload: dict[str, Any], mass_threshold: float) -> dict[str, Any]:
    ranking = [int(v) for v in erf_payload["ranking"]]
    scores = [max(float(v), 0.0) for v in erf_payload["prob_scores"]]
    if not ranking or not scores:
        raise ValueError("Empty ERF payload")
    total_mass = float(sum(scores))
    if total_mass <= 0.0:
        support_size = 1
        support_indices = ranking[:support_size]
        support_mass = float(sum(scores[idx] for idx in support_indices))
        return {
            "mass_threshold": float(mass_threshold),
            "support_size": int(support_size),
            "support_mass": float(support_mass),
            "support_mass_fraction": 0.0,
            "support_indices": support_indices,
        }

    support_size = len(ranking)
    running = 0.0
    target = max(0.0, min(1.0, float(mass_threshold))) * total_mass
    for idx, patch_idx in enumerate(ranking, start=1):
        running += float(scores[int(patch_idx)])
        if running >= target:
            support_size = idx
            break

    support_indices = ranking[:support_size]
    support_mass = float(sum(scores[idx] for idx in support_indices))
    return {
        "mass_threshold": float(mass_threshold),
        "support_size": int(support_size),
        "support_mass": float(support_mass),
        "support_mass_fraction": float(support_mass / total_mass),
        "support_indices": support_indices,
    }


def build_feature_binary_diagnostics(
    config: EvalConfig,
    *,
    session_name: str,
    feature_specs: Sequence[tuple[int, int]],
    thresholds: Sequence[float] = (0.75, 0.80, 0.85, 0.90),
    examples_per_feature: int = 5,
    objective_modes: Sequence[str] = ("cosine",),
) -> dict[str, Any]:
    session_dir = config.human_study_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(float(v) for v in thresholds)
    objective_modes = tuple(str(v) for v in objective_modes)
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    if any(v <= 0.0 or v > 1.0 for v in thresholds):
        raise ValueError(f"Invalid thresholds: {thresholds}")
    if not objective_modes:
        raise ValueError("objective_modes must be non-empty")

    erf_config = replace(config, erf_recovery_threshold=max(thresholds))
    runtime = LegacyRuntime(erf_config)
    feature_entries: list[dict[str, Any]] = []
    try:
        total = len(feature_specs)
        for feature_idx, (block_idx, feature_id) in enumerate(feature_specs, start=1):
            rows, selection_stats = _collect_feature_rows(
                runtime,
                config,
                block_idx=int(block_idx),
                feature_id=int(feature_id),
                required_count=int(examples_per_feature),
            )
            feature_dir = session_dir / "assets" / feature_key(block_idx, feature_id).replace("/", "__")
            sample_entries: list[dict[str, Any]] = []
            objective_summaries: dict[str, dict[str, list[int]]] = {
                objective_mode: {f"{threshold:.2f}": [] for threshold in thresholds}
                for objective_mode in objective_modes
            }
            for rank, row in enumerate(rows):
                image_path = str(row["image_path"])
                token_idx = int(row["target_patch_idx"])
                original_path = feature_dir / f"sample_{rank:02d}_original.png"
                actmap_path = feature_dir / f"sample_{rank:02d}_feature_actmap.png"
                save_original_with_token_box(image_path, original_path, token_idx)
                actmap = runtime.feature_activation_map(image_path, int(block_idx), int(feature_id))
                save_feature_actmap_overlay(image_path, actmap, actmap_path, token_idx=token_idx)
                objective_entries: list[dict[str, Any]] = []
                for objective_mode in objective_modes:
                    erf_json_path = feature_dir / f"sample_{rank:02d}_{objective_mode}_token_erf.json"
                    erf_payload = runtime.cautious_token_erf(
                        image_path,
                        int(block_idx),
                        token_idx,
                        objective_mode=objective_mode,
                    )
                    write_json(erf_json_path, erf_payload)
                    threshold_entries: list[dict[str, Any]] = []
                    for threshold in thresholds:
                        support = _support_from_trace(erf_payload, float(threshold))
                        objective_summaries[objective_mode][f"{threshold:.2f}"].append(int(support["support_size"]))
                        support_path = (
                            feature_dir
                            / f"sample_{rank:02d}_{objective_mode}_support_t{int(round(threshold * 100)):02d}.png"
                        )
                        contour_path = (
                            feature_dir
                            / f"sample_{rank:02d}_{objective_mode}_contour_t{int(round(threshold * 100)):02d}.png"
                        )
                        save_support_mask_image(
                            image_path,
                            support["support_indices"],
                            support_path,
                            token_idx=token_idx,
                        )
                        contour_meta = save_support_outline_crop_image(
                            image_path,
                            support["support_indices"],
                            contour_path,
                            score_map=erf_payload["prob_scores"],
                        )
                        threshold_entries.append(
                            {
                                "threshold": float(threshold),
                                "support_size": int(support["support_size"]),
                                "support_recovery": float(support["support_recovery"]),
                                "support_image": _item_relpath(session_dir, support_path),
                                "contour_image": _item_relpath(session_dir, contour_path),
                                "contour_meta": contour_meta,
                            }
                        )
                    objective_entries.append(
                        {
                            "objective_mode": objective_mode,
                            "objective_label": _objective_label(objective_mode),
                            "objective_metric_name": str(erf_payload["objective_metric_name"]),
                            "token_erf_json": _item_relpath(session_dir, erf_json_path),
                            "thresholds": threshold_entries,
                        }
                    )
                sample_entries.append(
                    {
                        "rank": int(rank),
                        "sample_id": int(row["sample_id"]),
                        "token_idx": token_idx,
                        "image_path": image_path,
                        "token_uid": str(row["token_uid"]),
                        "validation": dict(row["validation"]),
                        "original_with_token_box": _item_relpath(session_dir, original_path),
                        "feature_actmap": _item_relpath(session_dir, actmap_path),
                        "objectives": objective_entries,
                    }
                )

            feature_entries.append(
                {
                    "feature_key": feature_key(block_idx, feature_id),
                    "block_idx": int(block_idx),
                    "feature_id": int(feature_id),
                    "selection_stats": selection_stats,
                    "objective_summary": {
                        objective_mode: {
                            key: {
                                "min_support_size": min(values),
                                "max_support_size": max(values),
                                "mean_support_size": float(sum(values) / max(len(values), 1)),
                            }
                            for key, values in threshold_stats.items()
                        }
                        for objective_mode, threshold_stats in objective_summaries.items()
                    },
                    "samples": sample_entries,
                }
            )
            print(
                f"[feature-diagnostics {feature_idx:02d}/{total:02d}] "
                f"block={block_idx} feature={feature_id} samples={len(sample_entries)}",
                flush=True,
            )
    finally:
        runtime.close()

    manifest = {
        "config": config.to_dict(),
        "session_name": session_name,
        "thresholds": list(thresholds),
        "objective_modes": list(objective_modes),
        "feature_specs": [{"block_idx": int(b), "feature_id": int(f)} for b, f in feature_specs],
        "features": feature_entries,
    }
    write_json(session_dir / "diagnostics_manifest.json", manifest)

    sections = [
        "<h1>Feature Binary ERF Diagnostics</h1>",
        "<p>"
        "Each sample shows the original target token, the feature activation map, and binary ERF contour views "
        "defined as the smallest ranked prefix whose token-recovery objective reaches threshold "
        f"<code>{', '.join(f'{t:.2f}' for t in thresholds)}</code>."
        "</p>",
    ]
    for feature in feature_entries:
        summary_rows = []
        for objective_mode in objective_modes:
            stats_by_threshold = feature["objective_summary"][objective_mode]
            summary_rows.append(f"<div class='meta'><strong>{_objective_label(objective_mode)}</strong></div>")
            summary_rows.extend(
                f"<div class='meta'>tau={threshold}: "
                f"mean size={stats['mean_support_size']:.1f}, "
                f"range=[{stats['min_support_size']}, {stats['max_support_size']}]</div>"
                for threshold, stats in stats_by_threshold.items()
            )
        sample_html = []
        for sample in feature["samples"]:
            threshold_panels = []
            for objective_entry in sample["objectives"]:
                threshold_panels.append(
                    f"""
                    <div class="panel">
                      <div class="meta"><strong>{objective_entry['objective_label']}</strong></div>
                    </div>
                    """
                )
                threshold_panels.extend(
                    f"""
                    <div class="panel">
                      <img src="{entry['contour_image']}" alt="contour">
                      <div class="meta">{objective_entry['objective_label']} | tau={entry['threshold']:.2f} | size={entry['support_size']} | score={entry['support_recovery']:.3f}</div>
                      <div class="meta">checkerboard outside support + outer contour</div>
                    </div>
                    """
                    for entry in objective_entry["thresholds"]
                )
            sample_html.append(
                f"""
                <div class="sample">
                  <div class="meta">sample_id={sample['sample_id']} token={sample['token_idx']} uid={sample['token_uid']}</div>
                  <div class="meta">
                    act={sample['validation']['act_at_target']:.3f},
                    max={sample['validation']['max_act']:.3f},
                    ratio={sample['validation']['target_to_max_ratio']:.3f}
                  </div>
                  <div class="grid">
                    <div class="panel">
                      <img src="{sample['original_with_token_box']}" alt="original">
                      <div class="meta">Original + target token</div>
                    </div>
                    <div class="panel">
                      <img src="{sample['feature_actmap']}" alt="actmap">
                      <div class="meta">Feature activation map</div>
                    </div>
                    {''.join(threshold_panels)}
                  </div>
                </div>
                """
            )
        sections.append(
            f"""
            <div class="feature">
              <h2>{feature['feature_key']}</h2>
              <div class="meta">candidate_rows={feature['selection_stats']['candidate_rows']} validated={feature['selection_stats']['accepted_examples']}</div>
              {''.join(summary_rows)}
              {''.join(sample_html)}
            </div>
            """
        )

    (session_dir / "index.html").write_text(_html_page("Feature Binary ERF Diagnostics", "\n".join(sections)))
    return manifest


def build_feature_mass_diagnostics(
    config: EvalConfig,
    *,
    session_name: str,
    feature_specs: Sequence[tuple[int, int]],
    mass_thresholds: Sequence[float] = (0.90, 0.95, 0.99),
    examples_per_feature: int = 5,
    objective_mode: str | None = None,
) -> dict[str, Any]:
    session_dir = config.human_study_root / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    mass_thresholds = tuple(float(v) for v in mass_thresholds)
    if not mass_thresholds:
        raise ValueError("mass_thresholds must be non-empty")
    if any(v <= 0.0 or v > 1.0 for v in mass_thresholds):
        raise ValueError(f"Invalid mass_thresholds: {mass_thresholds}")

    objective_mode = str(objective_mode or config.erf_objective_mode)
    runtime = LegacyRuntime(config)
    feature_entries: list[dict[str, Any]] = []
    try:
        total = len(feature_specs)
        for feature_idx, (block_idx, feature_id) in enumerate(feature_specs, start=1):
            rows, selection_stats = _collect_feature_rows(
                runtime,
                config,
                block_idx=int(block_idx),
                feature_id=int(feature_id),
                required_count=int(examples_per_feature),
            )
            feature_dir = session_dir / "assets" / feature_key(block_idx, feature_id).replace("/", "__")
            sample_entries: list[dict[str, Any]] = []
            summary: dict[str, list[int]] = {f"{threshold:.2f}": [] for threshold in mass_thresholds}
            for rank, row in enumerate(rows):
                image_path = str(row["image_path"])
                token_idx = int(row["target_patch_idx"])
                original_path = feature_dir / f"sample_{rank:02d}_original.png"
                actmap_path = feature_dir / f"sample_{rank:02d}_feature_actmap.png"
                save_original_with_token_box(image_path, original_path, token_idx)
                actmap = runtime.feature_activation_map(image_path, int(block_idx), int(feature_id))
                save_feature_actmap_overlay(image_path, actmap, actmap_path, token_idx=token_idx)
                erf_json_path = feature_dir / f"sample_{rank:02d}_{objective_mode}_token_erf.json"
                erf_payload = runtime.cautious_token_erf(
                    image_path,
                    int(block_idx),
                    token_idx,
                    objective_mode=objective_mode,
                )
                write_json(erf_json_path, erf_payload)
                threshold_entries: list[dict[str, Any]] = []
                for mass_threshold in mass_thresholds:
                    support = _support_from_cumulative_mass(erf_payload, float(mass_threshold))
                    summary[f"{mass_threshold:.2f}"].append(int(support["support_size"]))
                    support_path = (
                        feature_dir
                        / f"sample_{rank:02d}_{objective_mode}_mass_q{int(round(mass_threshold * 100)):02d}.png"
                    )
                    save_support_mask_image(
                        image_path,
                        support["support_indices"],
                        support_path,
                        token_idx=token_idx,
                    )
                    threshold_entries.append(
                        {
                            "mass_threshold": float(mass_threshold),
                            "support_size": int(support["support_size"]),
                            "support_mass_fraction": float(support["support_mass_fraction"]),
                            "support_image": _item_relpath(session_dir, support_path),
                        }
                    )
                sample_entries.append(
                    {
                        "rank": int(rank),
                        "sample_id": int(row["sample_id"]),
                        "token_idx": token_idx,
                        "token_uid": str(row["token_uid"]),
                        "validation": dict(row["validation"]),
                        "original_with_token_box": _item_relpath(session_dir, original_path),
                        "feature_actmap": _item_relpath(session_dir, actmap_path),
                        "token_erf_json": _item_relpath(session_dir, erf_json_path),
                        "mass_thresholds": threshold_entries,
                    }
                )

            feature_entries.append(
                {
                    "feature_key": feature_key(block_idx, feature_id),
                    "block_idx": int(block_idx),
                    "feature_id": int(feature_id),
                    "selection_stats": selection_stats,
                    "objective_mode": objective_mode,
                    "objective_label": _objective_label(objective_mode),
                    "mass_summary": {
                        key: {
                            "min_support_size": min(values),
                            "max_support_size": max(values),
                            "mean_support_size": float(sum(values) / max(len(values), 1)),
                        }
                        for key, values in summary.items()
                    },
                    "samples": sample_entries,
                }
            )
            print(
                f"[feature-mass-diagnostics {feature_idx:02d}/{total:02d}] "
                f"block={block_idx} feature={feature_id} samples={len(sample_entries)}",
                flush=True,
            )
    finally:
        runtime.close()

    manifest = {
        "config": config.to_dict(),
        "session_name": session_name,
        "mass_thresholds": list(mass_thresholds),
        "objective_mode": objective_mode,
        "feature_specs": [{"block_idx": int(b), "feature_id": int(f)} for b, f in feature_specs],
        "features": feature_entries,
    }
    write_json(session_dir / "diagnostics_manifest.json", manifest)

    sections = [
        "<h1>Feature Cumulative-Mass ERF Diagnostics</h1>",
        "<p>"
        "Each sample shows binary masks defined by the smallest ranked prefix whose normalized ERF mass reaches "
        f"<code>{', '.join(f'{t:.2f}' for t in mass_thresholds)}</code>. "
        "This uses the learned ERF score ranking directly and does not run insertion recovery prefixes."
        "</p>",
    ]
    for feature in feature_entries:
        summary_rows = [f"<div class='meta'><strong>{feature['objective_label']}</strong></div>"]
        summary_rows.extend(
            f"<div class='meta'>q={threshold}: "
            f"mean size={stats['mean_support_size']:.1f}, "
            f"range=[{stats['min_support_size']}, {stats['max_support_size']}]</div>"
            for threshold, stats in feature["mass_summary"].items()
        )
        sample_html = []
        for sample in feature["samples"]:
            panels = []
            panels.extend(
                f"""
                <div class="panel">
                  <img src="{entry['support_image']}" alt="support">
                  <div class="meta">q={entry['mass_threshold']:.2f} | size={entry['support_size']} | covered mass={entry['support_mass_fraction']:.3f}</div>
                </div>
                """
                for entry in sample["mass_thresholds"]
            )
            sample_html.append(
                f"""
                <div class="sample">
                  <div class="meta">sample_id={sample['sample_id']} token={sample['token_idx']} uid={sample['token_uid']}</div>
                  <div class="meta">
                    act={sample['validation']['act_at_target']:.3f},
                    max={sample['validation']['max_act']:.3f},
                    ratio={sample['validation']['target_to_max_ratio']:.3f}
                  </div>
                  <div class="grid">
                    <div class="panel">
                      <img src="{sample['original_with_token_box']}" alt="original">
                      <div class="meta">Original + target token</div>
                    </div>
                    <div class="panel">
                      <img src="{sample['feature_actmap']}" alt="actmap">
                      <div class="meta">Feature activation map</div>
                    </div>
                    {''.join(panels)}
                  </div>
                </div>
                """
            )
        sections.append(
            f"""
            <div class="feature">
              <h2>{feature['feature_key']}</h2>
              <div class="meta">candidate_rows={feature['selection_stats']['candidate_rows']} validated={feature['selection_stats']['accepted_examples']}</div>
              {''.join(summary_rows)}
              {''.join(sample_html)}
            </div>
            """
        )
    (session_dir / "index.html").write_text(_html_page("Feature Cumulative-Mass ERF Diagnostics", "\n".join(sections)))
    return manifest
