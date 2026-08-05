#!/usr/bin/env python3
"""Benchmark native-pixel IxG and FRI on Perceiver latent SAE features."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageOps
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from codex_research_sae_autolabel_rebuttal.build_perceiver_multifeature_decile_fri_gallery import (  # noqa: E402
    FeatureEvaluator,
    MEAN,
    load_runtime,
    save_rgb,
    unnormalize,
)
from src.core.attribution.fri.solver import FRIConfig, run_fri  # noqa: E402


IMAGE_SIZE = 224
N_PIXELS = IMAGE_SIZE * IMAGE_SIZE
OLD_GROUP_COUNT = 14 * 14
REGULARIZER_SCALE = OLD_GROUP_COUNT / N_PIXELS
METHOD_LABELS = {
    "native_ixg": "Native IxG",
    "native_fri16": "Native FRI-16",
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
    parser.add_argument("--decile", type=int, default=0)
    parser.add_argument("--events-per-feature", type=int, default=1)
    parser.add_argument("--fri-steps", type=int, default=16)
    parser.add_argument("--tv-multiplier", type=float, default=1.0)
    parser.add_argument("--irrelevance-multiplier", type=float, default=1.0)
    parser.add_argument("--rank-bucket-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--part-name", default="part")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for render")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def select_records(
    path: Path,
    *,
    features: list[int] | None,
    decile: int,
    events_per_feature: int,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requested = None if features is None else {int(feature) for feature in features}
    selected: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for source in payload["records"]:
        feature = int(source["feature"])
        if requested is not None and feature not in requested:
            continue
        if int(source["decile"]) != int(decile):
            continue
        if counts.get(feature, 0) >= int(events_per_feature):
            continue
        selected.append(dict(source))
        counts[feature] = counts.get(feature, 0) + 1
    if not selected:
        raise ValueError("No records matched the requested feature/decile panel")
    return selected


def native_mask_to_pixels(pixels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    mask_images = masks.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).to(
        device=pixels.device, dtype=pixels.dtype
    )
    return pixels.expand(mask_images.shape[0], -1, -1, -1) * mask_images


def evaluate_masks(
    masks: np.ndarray,
    recovery_for_masks,
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


def native_gate_ixg(objective_for_mask, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    mask = torch.ones(N_PIXELS, device=device, dtype=torch.float32, requires_grad=True)
    value = objective_for_mask(mask)
    gradient = torch.autograd.grad(value, mask, retain_graph=False)[0]
    scores = gradient.detach().abs()
    if not torch.isfinite(scores).all() or float(scores.max()) <= 1e-8:
        scores = torch.ones_like(scores)
    inverse = 1.0 / (scores + 1e-8)
    irrelevance = (inverse / inverse.max().clamp(min=1e-8)).detach()
    return scores.cpu().numpy().astype(np.float32), irrelevance


def rank_bucket_masks(scores: np.ndarray, bucket_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="mergesort")
    bucket_size = max(1, int(bucket_size))
    budgets = np.arange(0, N_PIXELS + 1, bucket_size, dtype=np.int32)
    if budgets[-1] != N_PIXELS:
        budgets = np.concatenate([budgets, np.asarray([N_PIXELS], dtype=np.int32)])
    masks = np.zeros((len(budgets), N_PIXELS), dtype=np.uint8)
    current = np.zeros(N_PIXELS, dtype=np.uint8)
    cursor = 0
    for row, budget in enumerate(budgets):
        if int(budget) > cursor:
            current[order[cursor : int(budget)]] = 1
            cursor = int(budget)
        masks[row] = current
    return masks, order, budgets


def curve_metrics(
    recoveries: np.ndarray,
    order: np.ndarray,
    budgets: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    crossing = np.flatnonzero(recoveries >= float(threshold))
    row = int(crossing[0]) if crossing.size else len(budgets) - 1
    support_size = int(budgets[row])
    return {
        "support_size": support_size,
        "support_fraction": support_size / N_PIXELS,
        "support_recovery": float(recoveries[row]),
        "support_indices": order[:support_size].astype(int).tolist(),
        "auc": float(np.trapz(recoveries, budgets.astype(np.float32) / N_PIXELS)),
        "budgets": budgets.astype(int).tolist(),
        "recoveries": recoveries.astype(float).tolist(),
    }


def score_heatmap(original: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0.0).reshape(
        IMAGE_SIZE, IMAGE_SIZE
    )
    positive = values[values > 0]
    scale = float(np.quantile(positive, 0.995)) if positive.size else 1.0
    normalized = np.clip(values / max(scale, 1e-8), 0.0, 1.0)
    color = colormaps["inferno"](normalized)[..., :3].astype(np.float32)
    strength = (0.18 + 0.72 * normalized)[..., None]
    return original * (1.0 - strength) + color * strength


def hard_pixel_support(original: np.ndarray, support: list[int]) -> np.ndarray:
    keep = np.zeros(N_PIXELS, dtype=np.float32)
    keep[np.asarray(support, dtype=np.int64)] = 1.0
    keep = keep.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    background = MEAN.numpy().reshape(1, 1, 3)
    return background + keep * (original - background)


def mean_alpha_support(
    original: np.ndarray,
    scores: np.ndarray,
    support: list[int],
    *,
    alpha_floor: float = 0.08,
) -> np.ndarray:
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0.0)
    selected = np.zeros(N_PIXELS, dtype=np.float32)
    indices = np.asarray(support, dtype=np.int64)
    selected[indices] = 1.0
    selected_values = values[indices]
    scale = float(np.quantile(selected_values, 0.98)) if selected_values.size else 1.0
    alpha = np.zeros(N_PIXELS, dtype=np.float32)
    alpha[indices] = float(alpha_floor) + (1.0 - float(alpha_floor)) * np.clip(
        selected_values / max(scale, 1e-8), 0.0, 1.0
    )
    alpha = alpha.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    background = MEAN.numpy().reshape(1, 1, 3)
    return background + alpha * (original - background)


def copy_source_pruned(record: dict[str, Any], input_root: Path, output_path: Path) -> None:
    source = input_root / str(record["assets"]["pruned_erf"])
    with Image.open(source) as image:
        image.convert("RGB").save(output_path, quality=91, optimize=True)


def run_render(args: argparse.Namespace) -> None:
    input_path = require_path(args.input_results, "--input-results")
    dataset_root = require_path(args.dataset_root, "--dataset-root")
    records = select_records(
        input_path,
        features=args.features,
        decile=int(args.decile),
        events_per_feature=int(args.events_per_feature),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    device = torch.device(args.device)
    sae, model, transform = load_runtime(args, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    evaluator = FeatureEvaluator(model, sae)
    output_rows: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(records, start=1):
            feature = int(record["feature"])
            latent = int(record["latent_index"])
            sample_id = int(record["sample_id"])
            source_path = dataset_root / str(record["relative_path"])
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                pixels = transform(image).unsqueeze(0).to(device)

            with torch.inference_mode():
                full_pre, full_hard = evaluator.preactivation(
                    pixels, latent=latent, feature=feature
                )
                baseline_pre, baseline_hard = evaluator.preactivation(
                    torch.zeros_like(pixels), latent=latent, feature=feature
                )
            hard_denominator = float(full_hard[0] - baseline_hard[0])
            if hard_denominator <= 1e-8:
                raise RuntimeError(
                    f"Non-positive hard denominator for feature={feature}, sample={sample_id}"
                )

            def objective_for_mask(mask: torch.Tensor) -> torch.Tensor:
                pre, _hard = evaluator.preactivation(
                    native_mask_to_pixels(pixels, mask.reshape(1, -1)),
                    latent=latent,
                    feature=feature,
                )
                return pre[0]

            def recovery_for_masks(mask_batch: torch.Tensor) -> torch.Tensor:
                _pre, hard = evaluator.preactivation(
                    native_mask_to_pixels(pixels, mask_batch),
                    latent=latent,
                    feature=feature,
                )
                return (hard - baseline_hard[0]) / hard_denominator

            ixg_scores, irrelevance = native_gate_ixg(objective_for_mask, device)
            effective_seed = int(args.seed) + feature * 100_000 + sample_id
            fri_result = run_fri(
                n_patches=N_PIXELS,
                grid_size=IMAGE_SIZE,
                objective_for_mask=objective_for_mask,
                full_objective=full_pre[0].detach(),
                baseline_objective=baseline_pre[0].detach(),
                irrelevance=irrelevance,
                config=FRIConfig(
                    steps=int(args.fri_steps),
                    lr=0.45,
                    lr_end=0.01,
                    init_scores=ixg_scores,
                    init_score_prob_floor=0.05,
                    init_score_prob_ceiling=0.95,
                    reg_warmup_frac=0.20,
                    tv_weight=(
                        0.01 * REGULARIZER_SCALE * float(args.tv_multiplier)
                    ),
                    irrelevance_weight=(
                        0.05
                        * REGULARIZER_SCALE
                        * float(args.irrelevance_multiplier)
                    ),
                    objective_mode="random_budget_softins",
                    random_budget_distribution="low",
                    score_mode="final_plus_grad",
                    seed=effective_seed,
                ),
                device=device,
                dtype=pixels.dtype,
            )
            fri_scores = np.asarray(fri_result.scores, dtype=np.float32)
            methods = {
                "native_ixg": ixg_scores,
                "native_fri16": fri_scores,
            }
            masks_orders_budgets = {
                name: rank_bucket_masks(scores, int(args.rank_bucket_size))
                for name, scores in methods.items()
            }
            metrics: dict[str, dict[str, Any]] = {}
            for name, (masks, order, budgets) in masks_orders_budgets.items():
                recoveries = evaluate_masks(
                    masks,
                    recovery_for_masks,
                    device=device,
                    batch_size=int(args.eval_batch_size),
                )
                metrics[name] = curve_metrics(
                    recoveries,
                    order,
                    budgets,
                    float(args.threshold),
                )

            original = unnormalize(pixels[0])
            stem = (
                f"f{feature:04d}_d{int(record['decile'])}_"
                f"r{int(record['within_decile_rank'])}_s{sample_id:06d}"
            )
            paths = {
                "original": assets / f"{stem}_original.jpg",
                "grouped_pruned": assets / f"{stem}_grouped_pruned_erf80.jpg",
                "ixg_heatmap": assets / f"{stem}_native_ixg_heatmap.jpg",
                "fri_heatmap": assets / f"{stem}_native_fri16_heatmap.jpg",
                "ixg_erf": assets / f"{stem}_native_ixg_erf80.jpg",
                "fri_erf": assets / f"{stem}_native_fri16_erf80.jpg",
                "fri_mean_alpha": assets / f"{stem}_native_fri16_mean_alpha.jpg",
            }
            save_rgb(original, paths["original"])
            copy_source_pruned(record, input_path.parent, paths["grouped_pruned"])
            save_rgb(score_heatmap(original, ixg_scores), paths["ixg_heatmap"])
            save_rgb(score_heatmap(original, fri_scores), paths["fri_heatmap"])
            save_rgb(
                hard_pixel_support(original, metrics["native_ixg"]["support_indices"]),
                paths["ixg_erf"],
            )
            save_rgb(
                hard_pixel_support(original, metrics["native_fri16"]["support_indices"]),
                paths["fri_erf"],
            )
            save_rgb(
                mean_alpha_support(
                    original,
                    fri_scores,
                    metrics["native_fri16"]["support_indices"],
                ),
                paths["fri_mean_alpha"],
            )
            output_rows.append(
                {
                    **record,
                    "effective_seed": effective_seed,
                    "native_pixel_count": N_PIXELS,
                    "rank_bucket_size": int(args.rank_bucket_size),
                    "metrics": metrics,
                    "assets": {
                        name: str(path.relative_to(args.output_dir))
                        for name, path in paths.items()
                    },
                }
            )
            print(
                f"[{index}/{len(records)}] f={feature} sample={sample_id} "
                f"IxG k80={metrics['native_ixg']['support_size']} "
                f"FRI k80={metrics['native_fri16']['support_size']} "
                f"AUC={metrics['native_ixg']['auc']:.3f}/{metrics['native_fri16']['auc']:.3f}",
                flush=True,
            )
    finally:
        evaluator.close()
    output = {
        "source_results": str(input_path),
        "part_name": args.part_name,
        "fri_steps": int(args.fri_steps),
        "tv_multiplier": float(args.tv_multiplier),
        "irrelevance_multiplier": float(args.irrelevance_multiplier),
        "rank_bucket_size": int(args.rank_bucket_size),
        "threshold": float(args.threshold),
        "records": output_rows,
    }
    path = args.output_dir / f"results_{args.part_name}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def event_card(record: dict[str, Any]) -> str:
    ixg = record["metrics"]["native_ixg"]
    fri = record["metrics"]["native_fri16"]
    figures = [
        (record["assets"]["original"], "Original"),
        (record["assets"]["grouped_pruned"], "Previous grouped, pruned ERF80"),
        (record["assets"]["ixg_heatmap"], "Native IxG heatmap"),
        (record["assets"]["fri_heatmap"], "Native FRI-16 heatmap"),
        (
            record["assets"]["ixg_erf"],
            f"IxG hard pixel ERF80 · k≤{ixg['support_size']}",
        ),
        (
            record["assets"]["fri_erf"],
            f"FRI hard pixel ERF80 · k≤{fri['support_size']}",
        ),
        (record["assets"]["fri_mean_alpha"], "FRI support · mean-alpha importance"),
    ]
    figure_html = "".join(
        f'<figure><img src="{html.escape(path)}" loading="lazy">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for path, caption in figures
    )
    return f"""
    <article class="event-card">
      <h2>Feature {int(record['feature'])} · {html.escape(str(record['label_name']))}</h2>
      <p>latent {int(record['latent_index'])} · activation {float(record['activation']):.3f}</p>
      <div class="panels">{figure_html}</div>
      <table><tr><th>Ranking</th><th>k80 upper bound</th><th>fraction</th><th>recovery</th><th>AUC</th></tr>
      <tr><td>Native IxG</td><td>{int(ixg['support_size'])}</td><td>{100*float(ixg['support_fraction']):.1f}%</td><td>{float(ixg['support_recovery']):.3f}</td><td>{float(ixg['auc']):.3f}</td></tr>
      <tr><td>Native FRI-16</td><td>{int(fri['support_size'])}</td><td>{100*float(fri['support_fraction']):.1f}%</td><td>{float(fri['support_recovery']):.3f}</td><td>{float(fri['auc']):.3f}</td></tr></table>
    </article>
    """


def run_assemble(args: argparse.Namespace) -> None:
    parts = sorted(args.output_dir.glob("results_*.json"))
    if not parts:
        raise FileNotFoundError(f"No results_*.json in {args.output_dir}")
    records: list[dict[str, Any]] = []
    for path in parts:
        records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    keys = [
        (int(row["feature"]), int(row["decile"]), int(row["within_decile_rank"]))
        for row in records
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate event records")
    records.sort(key=lambda row: (int(row["feature"]), int(row["within_decile_rank"])))
    ixg_auc = np.asarray([row["metrics"]["native_ixg"]["auc"] for row in records])
    fri_auc = np.asarray([row["metrics"]["native_fri16"]["auc"] for row in records])
    ixg_k = np.asarray([row["metrics"]["native_ixg"]["support_size"] for row in records])
    fri_k = np.asarray([row["metrics"]["native_fri16"]["support_size"] for row in records])
    summary = {
        "n": len(records),
        "ixg_auc_mean": float(ixg_auc.mean()),
        "fri_auc_mean": float(fri_auc.mean()),
        "fri_auc_wins": int((fri_auc > ixg_auc).sum()),
        "ixg_k80_mean": float(ixg_k.mean()),
        "fri_k80_mean": float(fri_k.mean()),
        "fri_k80_wins": int((fri_k < ixg_k).sum()),
    }
    output = {"summary": summary, "records": records}
    (args.output_dir / "results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    cards = "".join(event_card(record) for record in records)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perceiver native-pixel FRI</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f3f1;color:#171918;font-family:Inter,system-ui,sans-serif}}
main{{max-width:1900px;margin:auto;padding:28px}}h1{{font-size:30px;letter-spacing:0}}.intro{{max-width:1200px;line-height:1.5;color:#4b544f}}
.event-card{{padding:24px 0;border-top:2px solid #4f5752}}.event-card h2{{font-size:22px;letter-spacing:0}}
.panels{{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:10px}}figure{{margin:0;background:#fff;border:1px solid #cbd0cc}}
img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;image-rendering:auto}}figcaption{{padding:7px;min-height:34px;font-size:12px}}
table{{margin-top:12px;border-collapse:collapse;background:#fff}}th,td{{padding:7px 10px;border:1px solid #ccd1cd;text-align:right}}th:first-child,td:first-child{{text-align:left}}
@media(max-width:1000px){{.panels{{grid-template-columns:repeat(2,minmax(0,1fr))}}main{{padding:14px}}}}
</style></head><body><main><h1>Perceiver native-pixel IxG vs FRI-16</h1>
<p class="intro">The attribution variables are all 50,176 native image pixels. Pixels are ranked individually; only hard-curve evaluation groups adjacent ranks into buckets of {int(records[0]['rank_bucket_size'])}. Mean replacement preserves the model's Fourier coordinate scaffold. The previous 16×16 grouped ERF is shown only as a qualitative reference.</p>
<p class="intro"><b>N={summary['n']}:</b> mean insertion AUC IxG/FRI {summary['ixg_auc_mean']:.3f}/{summary['fri_auc_mean']:.3f}; FRI AUC wins {summary['fri_auc_wins']}/{summary['n']}; mean k80 IxG/FRI {summary['ixg_k80_mean']:.0f}/{summary['fri_k80_mean']:.0f}; FRI smaller-k wins {summary['fri_k80_wins']}/{summary['n']}.</p>
{cards}</main></body></html>"""
    (args.output_dir / "gallery.html").write_text(page, encoding="utf-8")
    report = f"""# Perceiver native-pixel FRI smoke benchmark

- N: {summary['n']}
- Mean insertion AUC, IxG / FRI-16: {summary['ixg_auc_mean']:.4f} / {summary['fri_auc_mean']:.4f}
- FRI AUC wins: {summary['fri_auc_wins']} / {summary['n']}
- Mean pixel-k80 upper bound, IxG / FRI-16: {summary['ixg_k80_mean']:.1f} / {summary['fri_k80_mean']:.1f}
- FRI smaller-k wins: {summary['fri_k80_wins']} / {summary['n']}

The k80 values are upper bounds at the configured rank-bucket granularity.
"""
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {args.output_dir / 'gallery.html'}", flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "render":
        run_render(args)
    else:
        run_assemble(args)


if __name__ == "__main__":
    main()
