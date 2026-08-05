#!/usr/bin/env python3
"""Compact Perceiver ERFs and estimate group-level internal importance."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codex_research_sae_autolabel_rebuttal.build_perceiver_multifeature_decile_fri_gallery import (  # noqa: E402
    GRID_SIZE,
    MEAN,
    N_PATCHES,
    FeatureEvaluator,
    load_runtime,
    save_rgb,
    unnormalize,
)


METHOD_LABELS = {
    "fri64_final_plus_grad": "FRI-64",
    "ig_abs_sum": "IG-32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("render", "assemble"), required=True)
    parser.add_argument("--input-results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=int, nargs="+")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--prune-passes", type=int, default=3)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--alpha-floor", type=float, default=0.08)
    parser.add_argument("--part-name", default="part")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for render")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_records(path: Path, features: list[int] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = None if features is None else {int(feature) for feature in features}
    records = [
        dict(record)
        for record in payload["records"]
        if requested is None or int(record["feature"]) in requested
    ]
    if not records:
        raise ValueError("No records match --features")
    return payload, records


def winner_for_record(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    candidates = []
    for name in METHOD_LABELS:
        metric = record["metrics"][name]
        candidates.append(
            (
                int(metric["support_size"]),
                -float(metric["auc"]),
                name,
                metric,
            )
        )
    _size, _negative_auc, name, metric = min(candidates, key=lambda item: item[:3])
    return name, dict(metric)


def masks_to_pixels(pixels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    grid = masks.reshape(-1, 1, GRID_SIZE, GRID_SIZE).to(pixels.device, pixels.dtype)
    expanded = F.interpolate(grid, size=pixels.shape[-2:], mode="nearest")
    return pixels.expand(grid.shape[0], -1, -1, -1) * expanded


def support_mask(support: list[int]) -> np.ndarray:
    mask = np.zeros(N_PATCHES, dtype=np.float32)
    mask[np.asarray(support, dtype=np.int64)] = 1.0
    return mask


def evaluate_numpy_masks(
    masks: np.ndarray,
    recovery_for_masks: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(masks), int(batch_size)):
            batch = torch.from_numpy(
                masks[start : start + int(batch_size)].astype(np.float32, copy=False)
            ).to(device)
            output.append(recovery_for_masks(batch).detach().float().cpu())
    return torch.cat(output).numpy()


def recovery_for_support(
    support: list[int],
    recovery_for_masks: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
) -> float:
    mask = torch.from_numpy(support_mask(support)).to(device)
    with torch.inference_mode():
        return float(recovery_for_masks(mask.reshape(1, -1))[0])


def backward_prune(
    support: list[int],
    recovery_for_masks: Callable[[torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    threshold: float,
    max_passes: int,
) -> tuple[list[int], float, int]:
    current = list(support)
    evaluations = 0
    for _pass in range(int(max_passes)):
        removed = 0
        for patch in list(reversed(current)):
            candidate = [item for item in current if item != patch]
            recovery = recovery_for_support(
                candidate,
                recovery_for_masks,
                device=device,
            )
            evaluations += 1
            if recovery >= float(threshold):
                current = candidate
                removed += 1
        if removed == 0:
            break
    final_recovery = recovery_for_support(
        current,
        recovery_for_masks,
        device=device,
    )
    evaluations += 1
    return current, final_recovery, evaluations


def rank_bands(support: list[int], requested_groups: int) -> list[list[int]]:
    group_count = min(max(int(requested_groups), 1), len(support))
    arrays = np.array_split(np.asarray(support, dtype=np.int64), group_count)
    return [array.astype(int).tolist() for array in arrays if len(array)]


def coalition_masks(groups: list[list[int]]) -> np.ndarray:
    output = np.zeros((1 << len(groups), N_PATCHES), dtype=np.float32)
    for coalition in range(1 << len(groups)):
        for group_index, group in enumerate(groups):
            if coalition & (1 << group_index):
                output[coalition, np.asarray(group, dtype=np.int64)] = 1.0
    return output


@dataclass(frozen=True)
class GroupImportance:
    shapley: list[float]
    insertion: list[float]
    removal: list[float]
    coalition_values: list[float]
    efficiency_error: float


def exact_group_importance(values: np.ndarray, group_count: int) -> GroupImportance:
    if len(values) != 1 << int(group_count):
        raise ValueError("Coalition table has the wrong length")
    factorial = [math.factorial(index) for index in range(group_count + 1)]
    denominator = float(factorial[group_count])
    shapley = np.zeros(group_count, dtype=np.float64)
    full = (1 << group_count) - 1
    for group_index in range(group_count):
        bit = 1 << group_index
        for coalition in range(1 << group_count):
            if coalition & bit:
                continue
            size = coalition.bit_count()
            weight = factorial[size] * factorial[group_count - size - 1] / denominator
            shapley[group_index] += weight * (
                float(values[coalition | bit]) - float(values[coalition])
            )
    baseline = float(values[0])
    full_value = float(values[full])
    insertion = [float(values[1 << index]) - baseline for index in range(group_count)]
    removal = [
        full_value - float(values[full ^ (1 << index)]) for index in range(group_count)
    ]
    efficiency_error = float(shapley.sum() - (full_value - baseline))
    return GroupImportance(
        shapley=shapley.tolist(),
        insertion=insertion,
        removal=removal,
        coalition_values=np.asarray(values, dtype=np.float32).tolist(),
        efficiency_error=efficiency_error,
    )


def hard_support_image(pixels: torch.Tensor, support: list[int]) -> np.ndarray:
    mask = torch.from_numpy(support_mask(support)).to(pixels.device, pixels.dtype)
    expanded = F.interpolate(
        mask.reshape(1, 1, GRID_SIZE, GRID_SIZE),
        size=pixels.shape[-2:],
        mode="nearest",
    )[0, 0]
    original = unnormalize(pixels[0])
    background = MEAN.numpy().reshape(1, 1, 3)
    return original * expanded.cpu().numpy()[..., None] + background * (
        1.0 - expanded.cpu().numpy()[..., None]
    )


def alpha_support_image(
    pixels: torch.Tensor,
    groups: list[list[int]],
    shapley: list[float],
    *,
    alpha_floor: float,
) -> tuple[np.ndarray, list[float]]:
    positive = np.maximum(np.asarray(shapley, dtype=np.float32), 0.0)
    if float(positive.max(initial=0.0)) > 1e-8:
        normalized = positive / float(positive.max())
    else:
        normalized = np.ones_like(positive)
    alphas = float(alpha_floor) + (1.0 - float(alpha_floor)) * normalized
    patch_alpha = np.zeros(N_PATCHES, dtype=np.float32)
    selected = np.zeros(N_PATCHES, dtype=np.float32)
    for group, alpha in zip(groups, alphas, strict=True):
        indices = np.asarray(group, dtype=np.int64)
        patch_alpha[indices] = float(alpha)
        selected[indices] = 1.0
    alpha_grid = torch.from_numpy(patch_alpha).reshape(1, 1, GRID_SIZE, GRID_SIZE)
    support_grid = torch.from_numpy(selected).reshape(1, 1, GRID_SIZE, GRID_SIZE)
    alpha_map = F.interpolate(alpha_grid, size=(224, 224), mode="nearest")[0, 0].numpy()
    support_map = F.interpolate(support_grid, size=(224, 224), mode="nearest")[0, 0].numpy()
    original = unnormalize(pixels[0])
    background = MEAN.numpy().reshape(1, 1, 3)
    selected_image = original * alpha_map[..., None]
    rendered = selected_image * support_map[..., None] + background * (
        1.0 - support_map[..., None]
    )
    return rendered, alphas.astype(float).tolist()


def run_render(args: argparse.Namespace) -> None:
    input_path = require_path(args.input_results, "--input-results")
    dataset_root = require_path(args.dataset_root, "--dataset-root")
    _payload, records = load_records(input_path, args.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    device = torch.device(args.device)
    sae, model, transform = load_runtime(args, device)
    evaluator = FeatureEvaluator(model, sae)
    output_rows: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(records, start=1):
            feature = int(record["feature"])
            sample_id = int(record["sample_id"])
            latent = int(record["latent_index"])
            source_path = dataset_root / str(record["relative_path"])
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                pixels = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                _full_pre, full_hard = evaluator.preactivation(
                    pixels, latent=latent, feature=feature
                )
                _baseline_pre, baseline_hard = evaluator.preactivation(
                    torch.zeros_like(pixels), latent=latent, feature=feature
                )
            denominator = float(full_hard[0] - baseline_hard[0])
            if denominator <= 1e-8:
                raise RuntimeError(
                    f"Non-positive denominator for feature={feature}, sample={sample_id}"
                )

            def recovery_for_masks(mask_batch: torch.Tensor) -> torch.Tensor:
                _pre, hard = evaluator.preactivation(
                    masks_to_pixels(pixels, mask_batch),
                    latent=latent,
                    feature=feature,
                )
                return (hard - baseline_hard[0]) / denominator

            winner_name, winner_metric = winner_for_record(record)
            winner_support = [int(item) for item in winner_metric["support_indices"]]
            threshold = float(record.get("threshold", 0.80))
            pruned, pruned_recovery, prune_evaluations = backward_prune(
                winner_support,
                recovery_for_masks,
                device=device,
                threshold=threshold,
                max_passes=int(args.prune_passes),
            )
            groups = rank_bands(pruned, int(args.groups))
            masks = coalition_masks(groups)
            coalition_recoveries = evaluate_numpy_masks(
                masks,
                recovery_for_masks,
                device=device,
                batch_size=int(args.eval_batch_size),
            )
            importance = exact_group_importance(coalition_recoveries, len(groups))

            stem = (
                f"f{feature:04d}_d{int(record['decile'])}_"
                f"r{int(record['within_decile_rank'])}_s{sample_id:06d}"
            )
            paths = {
                "original": assets / f"{stem}_original.jpg",
                "winner_erf": assets / f"{stem}_winner_erf80.jpg",
                "pruned_erf": assets / f"{stem}_pruned_erf80.jpg",
                "group_shapley_erf": assets / f"{stem}_group_shapley_erf80.jpg",
            }
            save_rgb(unnormalize(pixels[0]), paths["original"])
            save_rgb(hard_support_image(pixels, winner_support), paths["winner_erf"])
            save_rgb(hard_support_image(pixels, pruned), paths["pruned_erf"])
            alpha_image, group_alphas = alpha_support_image(
                pixels,
                groups,
                importance.shapley,
                alpha_floor=float(args.alpha_floor),
            )
            save_rgb(alpha_image, paths["group_shapley_erf"])
            output_rows.append(
                {
                    **record,
                    "winner": {
                        "method": winner_name,
                        "method_label": METHOD_LABELS[winner_name],
                        "support_size": len(winner_support),
                        "support_indices": winner_support,
                        "support_recovery": float(winner_metric["support_recovery"]),
                    },
                    "pruned": {
                        "support_size": len(pruned),
                        "support_indices": pruned,
                        "support_recovery": float(pruned_recovery),
                        "evaluations": int(prune_evaluations),
                    },
                    "groups": [
                        {
                            "group": group_index,
                            "rank_start": int(sum(len(item) for item in groups[:group_index])),
                            "rank_end": int(
                                sum(len(item) for item in groups[: group_index + 1]) - 1
                            ),
                            "size": len(group),
                            "patch_indices": group,
                            "shapley": float(importance.shapley[group_index]),
                            "insertion": float(importance.insertion[group_index]),
                            "removal": float(importance.removal[group_index]),
                            "display_alpha": float(group_alphas[group_index]),
                        }
                        for group_index, group in enumerate(groups)
                    ],
                    "group_importance": {
                        "coalition_count": len(importance.coalition_values),
                        "coalition_recoveries": importance.coalition_values,
                        "efficiency_error": importance.efficiency_error,
                    },
                    "assets": {
                        name: str(path.relative_to(args.output_dir))
                        for name, path in paths.items()
                    },
                }
            )
            print(
                f"[{index}/{len(records)}] f={feature} D{record['decile']} "
                f"r={record['within_decile_rank']} {METHOD_LABELS[winner_name]} "
                f"k={len(winner_support)}->{len(pruned)} "
                f"groups={len(groups)} coalitions={len(masks)}",
                flush=True,
            )
    finally:
        evaluator.close()
    output = {
        "source_results": str(input_path),
        "threshold": 0.80,
        "requested_groups": int(args.groups),
        "prune_passes": int(args.prune_passes),
        "alpha_floor": float(args.alpha_floor),
        "part_name": args.part_name,
        "records": output_rows,
    }
    path = args.output_dir / f"results_{args.part_name}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def aggregate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["feature"]), int(record["decile"]))
        groups.setdefault(key, []).append(record)
    output = []
    for (feature, decile), rows in sorted(groups.items()):
        winner = np.asarray([row["winner"]["support_size"] for row in rows])
        pruned = np.asarray([row["pruned"]["support_size"] for row in rows])
        output.append(
            {
                "feature": feature,
                "decile": decile,
                "n": len(rows),
                "winner_mean": float(winner.mean()),
                "pruned_mean": float(pruned.mean()),
                "mean_reduction": float((winner - pruned).mean()),
                "relative_reduction": float(
                    1.0 - pruned.sum() / max(float(winner.sum()), 1.0)
                ),
                "fri_winners": sum(
                    row["winner"]["method"] == "fri64_final_plus_grad" for row in rows
                ),
                "ig_winners": sum(row["winner"]["method"] == "ig_abs_sum" for row in rows),
            }
        )
    return output


def event_card(record: dict[str, Any]) -> str:
    figures = [
        (record["assets"]["original"], "Original"),
        (
            record["assets"]["winner_erf"],
            f"Best prefix: {record['winner']['method_label']} · k={record['winner']['support_size']}",
        ),
        (
            record["assets"]["pruned_erf"],
            f"Backward-pruned ERF80 · k={record['pruned']['support_size']}",
        ),
        (
            record["assets"]["group_shapley_erf"],
            f"{len(record['groups'])}-group exact Shapley alpha",
        ),
    ]
    figure_html = "".join(
        f'<figure><img src="{html.escape(path)}" loading="lazy">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for path, caption in figures
    )
    top_groups = sorted(record["groups"], key=lambda item: -float(item["shapley"]))[:3]
    group_text = ", ".join(
        f"G{item['group']} φ={item['shapley']:.3f}"
        for item in top_groups
    )
    return f"""
    <article class="event-card">
      <div class="event-head"><b>#{int(record['within_decile_rank']) + 1}</b>
      {html.escape(str(record['label_name']))}</div>
      <div class="quad">{figure_html}</div>
      <p>activation {float(record['activation']):.3f} · recovery {float(record['pruned']['support_recovery']):.3f}<br>{html.escape(group_text)}</p>
    </article>
    """


def run_assemble(args: argparse.Namespace) -> None:
    parts = [
        path
        for path in sorted(args.output_dir.glob("results_*.json"))
        if path.name != "results.json"
    ]
    if not parts:
        raise FileNotFoundError(f"No results_*.json in {args.output_dir}")
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for path in parts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload if metadata is None else metadata
        records.extend(payload["records"])
    keys = [
        (
            int(record["feature"]),
            int(record["decile"]),
            int(record["within_decile_rank"]),
        )
        for record in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate feature/decile/rank records")
    records.sort(
        key=lambda record: (
            int(record["feature"]),
            int(record["decile"]),
            int(record["within_decile_rank"]),
        )
    )
    threshold = float((metadata or {}).get("threshold", 0.80))
    summary = aggregate_rows(records)
    total_winner = sum(int(record["winner"]["support_size"]) for record in records)
    total_pruned = sum(int(record["pruned"]["support_size"]) for record in records)
    overall = {
        "n": len(records),
        "winner_mean": total_winner / len(records),
        "pruned_mean": total_pruned / len(records),
        "relative_reduction": 1.0 - total_pruned / max(float(total_winner), 1.0),
        "fri_winners": sum(
            record["winner"]["method"] == "fri64_final_plus_grad" for record in records
        ),
        "ig_winners": sum(
            record["winner"]["method"] == "ig_abs_sum" for record in records
        ),
        "winner_gt96": sum(
            int(record["winner"]["support_size"]) > 96 for record in records
        ),
        "pruned_gt96": sum(
            int(record["pruned"]["support_size"]) > 96 for record in records
        ),
        "min_pruned_recovery": min(
            float(record["pruned"]["support_recovery"]) for record in records
        ),
        "below_threshold": sum(
            float(record["pruned"]["support_recovery"]) < threshold
            for record in records
        ),
        "group_count": sum(len(record["groups"]) for record in records),
        "negative_shapley_groups": sum(
            float(group["shapley"]) < 0.0
            for record in records
            for group in record["groups"]
        ),
        "interaction_groups": sum(
            float(group["shapley"]) > 0.05
            and float(group["insertion"]) < 0.05
            for record in records
            for group in record["groups"]
        ),
        "max_shapley_efficiency_error": max(
            abs(float(record["group_importance"]["efficiency_error"]))
            for record in records
        ),
    }
    output = {
        "metadata": metadata,
        "overall": overall,
        "summary": summary,
        "records": records,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    features = sorted({int(record["feature"]) for record in records})
    nav = " ".join(f"<a href='#feature-{feature}'>F{feature}</a>" for feature in features)
    sections = []
    for feature in features:
        decile_sections = []
        deciles = sorted(
            {
                int(record["decile"])
                for record in records
                if int(record["feature"]) == feature
            }
        )
        for decile in deciles:
            rows = [
                record
                for record in records
                if int(record["feature"]) == feature
                and int(record["decile"]) == decile
            ]
            cards = "".join(event_card(record) for record in rows)
            decile_sections.append(
                f"<section><h3>D{decile} · top five within decile</h3>"
                f'<div class="event-grid">{cards}</div></section>'
            )
        sections.append(
            f'<section class="feature" id="feature-{feature}">'
            f"<h2>Feature {feature}</h2>{''.join(decile_sections)}</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perceiver ERF compaction and group Shapley</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f4f2;color:#171918;font-family:Inter,system-ui,sans-serif}}
main{{max-width:1920px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}
.intro{{max-width:1180px;color:#4c5550;line-height:1.5}}nav{{position:sticky;top:0;z-index:3;padding:10px 0;background:#f3f4f2;border-bottom:1px solid #c9ceca}}
nav a{{display:inline-block;margin:2px 8px 2px 0;padding:5px 8px;background:#fff;border:1px solid #c4cac5;border-radius:4px;color:#174f3d;text-decoration:none}}
.feature{{padding:28px 0;border-top:2px solid #4d5550}}h2{{font-size:24px;letter-spacing:0}}h3{{font-size:17px;letter-spacing:0;color:#3d4641}}
.event-grid{{display:grid;grid-template-columns:repeat(5,minmax(340px,1fr));gap:12px;overflow-x:auto;padding-bottom:8px}}
.event-card{{min-width:340px;background:#fff;border:1px solid #cbd0cc;border-radius:6px;padding:10px}}
.event-head{{min-height:38px;font-size:13px;line-height:1.35}}.quad{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}}
figure{{margin:0;border:1px solid #dde1de;background:#eceeec}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain}}
figcaption{{padding:5px 6px;min-height:30px;background:#fff;font-size:11px;line-height:1.3}}.event-card p{{margin:8px 1px 0;font-size:11px;color:#59615c;line-height:1.4}}
@media(max-width:900px){{main{{padding:14px}}.event-grid{{grid-template-columns:repeat(5,minmax(300px,1fr))}}.event-card{{min-width:300px}}}}
</style></head><body><main><h1>Perceiver ERF compaction and internal group importance</h1>
<p class="intro">For each event, choose the smaller exact ERF80 prefix from FRI-64 and IG-32, backward-prune while preserving recovery ≥ 0.80, then divide the retained ranking into up to eight equal-count bands. The final panel evaluates all group coalitions and uses exact group-Shapley values only to darken selected evidence. Grey remains the hard-mask baseline; the alpha panel is an explanatory rendering, not the recovery input.</p>
<p class="intro"><b>Overall:</b> mean k {overall['winner_mean']:.1f} → {overall['pruned_mean']:.1f}; reduction {100 * overall['relative_reduction']:.1f}%; FRI/IG winners {overall['fri_winners']}/{overall['ig_winners']}; k&gt;96 {overall['winner_gt96']} → {overall['pruned_gt96']}.</p>
<nav>{nav}</nav>{''.join(sections)}</main></body></html>"""
    (args.output_dir / "gallery.html").write_text(page, encoding="utf-8")

    lines = [
        "# Perceiver ERF compaction and group-Shapley summary",
        "",
        "## Protocol",
        "",
        "1. For each event, choose the smaller exact ERF80 prefix from the existing "
        "FRI-64 and IG-32 rankings (breaking size ties by insertion AUC).",
        "2. Backward-prune individual selected patches while retaining normalized "
        "hard-mask recovery >= 0.80.",
        "3. Split the retained ranking into up to eight equal-count bands and evaluate "
        "all 2^G coalitions. Eight bands cap this at 256 coalitions per event; ten "
        "bands would require 1,024.",
        "4. Compute exact group-Shapley values. The gallery keeps unselected patches "
        "at the grey hard-mask baseline and darkens retained patches according to "
        "positive Shapley importance. This alpha panel is explanatory rendering only, "
        "not an input used to establish recovery.",
        "",
        "## Checks and headline results",
        "",
        f"Overall N={overall['n']}: mean k80 {overall['winner_mean']:.2f} -> {overall['pruned_mean']:.2f} "
        f"({100 * overall['relative_reduction']:.1f}% reduction).",
        f"FRI/IG supplied the smaller exact prefix for "
        f"{overall['fri_winners']}/{overall['ig_winners']} events; supports larger "
        f"than 96 patches fell from {overall['winner_gt96']} to {overall['pruned_gt96']}.",
        f"All pruned supports retain the threshold: minimum recovery "
        f"{overall['min_pruned_recovery']:.6f}; below-threshold events "
        f"{overall['below_threshold']}.",
        f"Exact Shapley efficiency holds to numerical precision "
        f"(maximum absolute error {overall['max_shapley_efficiency_error']:.3e}).",
        f"Among {overall['group_count']} groups, "
        f"{overall['interaction_groups']} have Shapley contribution > .05 despite "
        f"group-only insertion < .05, indicating that much of the retained evidence "
        f"acts jointly rather than as independently sufficient regions. "
        f"{overall['negative_shapley_groups']} groups have negative Shapley values "
        f"and are rendered at the alpha floor.",
        "",
        "The main result is that a fixed ranked prefix substantially overstates the "
        "required support when recovery is non-monotonic or redundant. Backward "
        "pruning should therefore follow threshold crossing. The group-Shapley panel "
        "adds a conditional decomposition inside that compact hard support; it does "
        "not replace the hard sufficiency test and remains conditional on the chosen "
        "support and rank-band grouping.",
        "",
        "## Feature/decile breakdown",
        "",
        "| Feature | Decile | N | Best-prefix k | Pruned k | Reduction | FRI/IG winners |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['feature']} | {row['decile']} | {row['n']} | "
            f"{row['winner_mean']:.1f} | {row['pruned_mean']:.1f} | "
            f"{100 * row['relative_reduction']:.1f}% | "
            f"{row['fri_winners']}/{row['ig_winners']} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'gallery.html'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "render":
        run_render(args)
    else:
        run_assemble(args)


if __name__ == "__main__":
    main()
