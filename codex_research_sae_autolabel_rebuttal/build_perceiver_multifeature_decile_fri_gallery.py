#!/usr/bin/env python3
"""Build a multi-feature Perceiver activation-decile FRI/IG gallery."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.attribution.fri.solver import FRIConfig, run_fri  # noqa: E402
from src.core.sae.registry import create_sae  # noqa: E402
from src.packs.clip.dataset.builders import build_clip_transform  # noqa: E402
from src.packs.perceiver.models.model_loaders import load_perceiver_model  # noqa: E402


DEFAULT_FEATURES = [2048, 625, 2860, 577, 1387, 1904, 3950, 1431, 1994, 2758]
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])
GRID_SIZE = 14
N_PATCHES = GRID_SIZE * GRID_SIZE
LAYER = "model.perceiver.encoder.depth_4_visual_residual_tap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("census", "render", "assemble"), required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features", type=int, nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--deciles", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--examples-per-decile", type=int, default=5)
    parser.add_argument("--census-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--fri-steps", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--part-name", default="part")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for this stage")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_sae(path: Path, device: torch.device):
    package = torch.load(path, map_location="cpu")
    config = dict(package["sae_config"])
    config["device"] = str(device)
    sae = create_sae(config.get("sae_type", "batch-topk"), config)
    sae.load_state_dict(package["sae_state"], strict=True)
    return sae.eval().to(device)


def load_runtime(args: argparse.Namespace, device: torch.device):
    checkpoint = require_path(args.checkpoint, "--checkpoint")
    hf_cache = require_path(args.hf_cache, "--hf-cache")
    sae = load_sae(checkpoint, device)
    model = load_perceiver_model(
        {
            "hf_model": "deepmind/vision-perceiver-fourier",
            "cache_dir": str(hf_cache),
            "local_files_only": True,
            "torch_dtype": "float32",
            "image_size": 224,
            "tap_after_block": 4,
            "tap_mode": "visual_residual",
        },
        device=device,
        full_config={"sae": {"training": {"use_amp": False}}},
    )
    transform = build_clip_transform(
        {
            "image_size": 224,
            "resize_size": 256,
            "interpolation": "bicubic",
            "mean": MEAN.tolist(),
            "std": STD.tolist(),
        },
        is_train=False,
    )
    return sae, model, transform


class FeatureEvaluator:
    def __init__(self, model: Any, sae: Any):
        self.model = model
        self.sae = sae
        self.captured: list[torch.Tensor] = []
        tap = model.perceiver.encoder.depth_4_visual_residual_tap
        self.handle = tap.register_forward_hook(
            lambda _module, _inputs, output: self.captured.append(output)
        )

    def close(self) -> None:
        self.handle.remove()

    def hidden(self, pixels: torch.Tensor) -> torch.Tensor:
        self.captured.clear()
        self.model(inputs=pixels)
        if len(self.captured) != 1:
            raise RuntimeError(f"Expected one tap output, got {len(self.captured)}")
        return self.captured[0].float()

    def preactivation(
        self, pixels: torch.Tensor, *, latent: int, feature: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(pixels)[:, int(latent)]
        processed, _mean, _std = self.sae.preprocess_input(hidden)
        if self.sae.config.get("input_global_center_norm", False):
            pre = processed @ self.sae.W_enc[:, int(feature)] + self.sae.b_enc[int(feature)]
        else:
            pre = (
                (processed - self.sae.b_dec) @ self.sae.W_enc[:, int(feature)]
                + self.sae.b_enc[int(feature)]
            )
        hard = self.sae.encode(hidden)[:, int(feature)]
        return pre, hard


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], root: Path, transform: Any):
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.root / str(row["relative_path"])
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            pixels = self.transform(image)
        return pixels, index


def read_source_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["records"] if isinstance(payload, dict) else payload
    return [dict(row) for row in rows]


def run_census(args: argparse.Namespace) -> None:
    source_path = require_path(args.source_manifest, "--source-manifest")
    dataset_root = require_path(args.dataset_root, "--dataset-root")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    source_rows = read_source_manifest(source_path)
    sae, model, transform = load_runtime(args, device)
    evaluator = FeatureEvaluator(model, sae)
    dataset = ManifestDataset(source_rows, dataset_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=int(args.census_batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    features = [int(feature) for feature in args.features]
    scores = np.zeros((len(source_rows), len(features)), dtype=np.float32)
    latents = np.zeros((len(source_rows), len(features)), dtype=np.int16)
    try:
        for step, (pixels, indices) in enumerate(loader):
            pixels = pixels.to(device, non_blocking=True)
            with torch.inference_mode():
                hidden = evaluator.hidden(pixels)
                activations = sae.encode(hidden)[..., features]
                batch_scores, batch_latents = activations.max(dim=1)
            idx = indices.numpy()
            scores[idx] = batch_scores.float().cpu().numpy()
            latents[idx] = batch_latents.to(torch.int16).cpu().numpy()
            if step % 100 == 0:
                completed = min((step + 1) * len(pixels), len(dataset))
                print(f"census {completed}/{len(dataset)}", flush=True)
    finally:
        evaluator.close()

    selected: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    requested_deciles = sorted({int(value) for value in args.deciles})
    for feature_index, feature in enumerate(features):
        positive = np.flatnonzero(scores[:, feature_index] > 0)
        order = positive[
            np.argsort(-scores[positive, feature_index], kind="mergesort")
        ]
        positive_ranks = {int(source_index): rank for rank, source_index in enumerate(order)}
        bins = np.array_split(order, 10)
        feature_summary = {
            "feature": feature,
            "positive_count": int(len(order)),
            "positive_frequency": float(len(order) / max(len(source_rows), 1)),
            "deciles": [],
        }
        for decile, indices in enumerate(bins):
            values = scores[indices, feature_index]
            feature_summary["deciles"].append(
                {
                    "decile": decile,
                    "count": int(len(indices)),
                    "maximum": float(values.max()) if len(values) else None,
                    "minimum": float(values.min()) if len(values) else None,
                }
            )
            if decile not in requested_deciles:
                continue
            if len(indices) < int(args.examples_per_decile):
                raise RuntimeError(
                    f"Feature {feature} D{decile} has only {len(indices)} positive events"
                )
            for within_decile_rank, source_index in enumerate(
                indices[: int(args.examples_per_decile)]
            ):
                source = source_rows[int(source_index)]
                selected.append(
                    {
                        **source,
                        "feature": feature,
                        "activation": float(scores[source_index, feature_index]),
                        "latent_index": int(latents[source_index, feature_index]),
                        "decile": decile,
                        "within_decile_rank": within_decile_rank,
                        "global_positive_rank": int(positive_ranks[int(source_index)]),
                    }
                )
        summaries.append(feature_summary)
    output = {
        "target": LAYER,
        "source_manifest": str(source_path),
        "dataset_root": str(dataset_root),
        "features": features,
        "deciles": requested_deciles,
        "examples_per_decile": int(args.examples_per_decile),
        "population_size": len(source_rows),
        "feature_summaries": summaries,
        "records": selected,
    }
    path = args.output_dir / "selection.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {path} with {len(selected)} selected events", flush=True)


def masks_to_pixels(pixels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masks = masks.reshape(-1, 1, GRID_SIZE, GRID_SIZE).to(pixels.device, pixels.dtype)
    expanded = F.interpolate(masks, size=pixels.shape[-2:], mode="nearest")
    return pixels.expand(masks.shape[0], -1, -1, -1) * expanded


def inverse_gradient_irrelevance(
    objective: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones(N_PATCHES, device=device, dtype=dtype, requires_grad=True)
    value = objective(mask)
    gradient = torch.autograd.grad(value, mask)[0].abs()
    if not torch.isfinite(gradient).all() or float(gradient.max()) <= 1e-8:
        return torch.ones_like(mask)
    inverse = 1.0 / (gradient + 1e-8)
    return (inverse / inverse.max().clamp(min=1e-8)).detach()


def ig_scores(
    evaluator: FeatureEvaluator,
    pixels: torch.Tensor,
    *,
    latent: int,
    feature: int,
    steps: int,
) -> np.ndarray:
    gradient_sum = torch.zeros_like(pixels, dtype=torch.float32)
    for step in range(int(steps)):
        alpha = float(step + 1) / float(steps)
        interpolated = (pixels * alpha).detach().requires_grad_(True)
        pre, _hard = evaluator.preactivation(
            interpolated, latent=latent, feature=feature
        )
        gradient = torch.autograd.grad(pre[0], interpolated)[0]
        gradient_sum.add_(gradient.detach().float())
    contribution = pixels.detach().float() * (gradient_sum / float(steps))
    patches = contribution.reshape(1, 3, GRID_SIZE, 16, GRID_SIZE, 16)
    scores = patches.permute(0, 2, 4, 1, 3, 5).abs().sum((3, 4, 5))[0]
    return scores.detach().cpu().numpy().reshape(-1)


def dense_masks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="mergesort")
    masks = np.zeros((N_PATCHES + 1, N_PATCHES), dtype=np.uint8)
    for budget, patch_index in enumerate(order, start=1):
        masks[budget] = masks[budget - 1]
        masks[budget, int(patch_index)] = 1
    return masks, order


def evaluate_recoveries(
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


def curve_metrics(
    recoveries: np.ndarray, order: np.ndarray, threshold: float
) -> dict[str, Any]:
    crossing = np.flatnonzero(recoveries >= float(threshold))
    support_size = int(crossing[0]) if crossing.size else N_PATCHES
    return {
        "support_size": support_size,
        "support_recovery": float(recoveries[support_size]),
        "support_indices": order[:support_size].tolist(),
        "auc": float(np.trapz(recoveries, np.arange(N_PATCHES + 1) / N_PATCHES)),
        "recoveries": recoveries.tolist(),
    }


def unnormalize(pixels: torch.Tensor) -> np.ndarray:
    image = pixels.detach().float().cpu() * STD[:, None, None] + MEAN[:, None, None]
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def save_rgb(array: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))
    image.save(path, quality=91, optimize=True)


def heatmap(original: np.ndarray, scores: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(scores, dtype=np.float32), 0.0)
    if np.any(values > 0):
        scale = max(float(np.quantile(values[values > 0], 0.98)), 1e-8)
    else:
        scale = 1.0
    grid = torch.from_numpy(
        np.clip(values / scale, 0, 1).reshape(1, 1, GRID_SIZE, GRID_SIZE)
    )
    resized = F.interpolate(
        grid, size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    color = colormaps["inferno"](resized)[..., :3]
    return np.clip(original * 0.40 + color * 0.60, 0, 1)


def hard_support_image(pixels: torch.Tensor, support: list[int]) -> np.ndarray:
    mask = torch.zeros(N_PATCHES, device=pixels.device, dtype=pixels.dtype)
    mask[torch.as_tensor(support, device=pixels.device)] = 1.0
    expanded = F.interpolate(
        mask.reshape(1, 1, GRID_SIZE, GRID_SIZE),
        size=pixels.shape[-2:],
        mode="nearest",
    )
    return unnormalize(pixels[0] * expanded[0])


def run_render(args: argparse.Namespace) -> None:
    selection_path = require_path(args.selection, "--selection")
    dataset_root = require_path(args.dataset_root, "--dataset-root")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    requested = {int(feature) for feature in args.features}
    records = [
        dict(record)
        for record in selection["records"]
        if int(record["feature"]) in requested
    ]
    if not records:
        raise ValueError("No selected records match --features")
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

            def values_for_masks(
                mask_batch: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return evaluator.preactivation(
                    masks_to_pixels(pixels, mask_batch),
                    latent=latent,
                    feature=feature,
                )

            def raw_pre_for_mask(mask: torch.Tensor) -> torch.Tensor:
                pre, _hard = values_for_masks(mask.reshape(1, -1))
                return pre[0]

            def recovery_for_masks(mask_batch: torch.Tensor) -> torch.Tensor:
                _pre, hard = values_for_masks(mask_batch)
                return (hard - baseline_hard[0]) / hard_denominator

            ig = ig_scores(
                evaluator,
                pixels,
                latent=latent,
                feature=feature,
                steps=int(args.ig_steps),
            )
            irrelevance = inverse_gradient_irrelevance(
                raw_pre_for_mask, device, pixels.dtype
            )
            effective_seed = int(args.seed) + feature * 100_000 + sample_id
            fri_result = run_fri(
                n_patches=N_PATCHES,
                grid_size=GRID_SIZE,
                objective_for_mask=raw_pre_for_mask,
                full_objective=full_pre[0].detach(),
                baseline_objective=baseline_pre[0].detach(),
                irrelevance=irrelevance,
                config=FRIConfig(
                    steps=int(args.fri_steps),
                    lr=0.45,
                    lr_end=0.01,
                    init_prob=0.50,
                    seed=effective_seed,
                    optimizer_mode="cautious_adam_cosine",
                    objective_mode="random_budget_softins",
                    tv_weight=0.01,
                    irrelevance_weight=0.05,
                    score_mode="final_plus_grad",
                ),
                device=device,
                dtype=pixels.dtype,
            )
            fri = np.asarray(fri_result.scores, dtype=np.float32)
            method_scores = {"fri64_final_plus_grad": fri, "ig_abs_sum": ig}
            method_names = list(method_scores)
            masks_and_orders = [dense_masks(method_scores[name]) for name in method_names]
            all_masks = np.concatenate([item[0] for item in masks_and_orders])
            all_recoveries = evaluate_recoveries(
                all_masks,
                recovery_for_masks,
                device=device,
                batch_size=int(args.eval_batch_size),
            ).reshape(len(method_names), N_PATCHES + 1)
            metrics = {
                name: curve_metrics(
                    all_recoveries[method_index],
                    masks_and_orders[method_index][1],
                    float(args.threshold),
                )
                for method_index, name in enumerate(method_names)
            }
            original = unnormalize(pixels[0])
            stem = (
                f"f{feature:04d}_d{int(record['decile'])}_"
                f"r{int(record['within_decile_rank'])}_s{sample_id:06d}"
            )
            paths = {
                "original": assets / f"{stem}_original.jpg",
                "ig_heatmap": assets / f"{stem}_ig_heatmap.jpg",
                "fri_heatmap": assets / f"{stem}_fri_heatmap.jpg",
                "fri_erf80": assets / f"{stem}_fri_erf80.jpg",
            }
            save_rgb(original, paths["original"])
            save_rgb(heatmap(original, ig), paths["ig_heatmap"])
            save_rgb(heatmap(original, fri), paths["fri_heatmap"])
            save_rgb(
                hard_support_image(
                    pixels,
                    metrics["fri64_final_plus_grad"]["support_indices"],
                ),
                paths["fri_erf80"],
            )
            output_rows.append(
                {
                    **record,
                    "source_path": str(source_path),
                    "effective_seed": effective_seed,
                    "full_hard_activation": float(full_hard[0]),
                    "baseline_hard_activation": float(baseline_hard[0]),
                    "metrics": metrics,
                    "assets": {
                        name: str(path.relative_to(args.output_dir))
                        for name, path in paths.items()
                    },
                }
            )
            print(
                f"[{index}/{len(records)}] f={feature} D{record['decile']} "
                f"r={record['within_decile_rank']} "
                f"k80(FRI/IG)={metrics['fri64_final_plus_grad']['support_size']}/"
                f"{metrics['ig_abs_sum']['support_size']}",
                flush=True,
            )
    finally:
        evaluator.close()
    output = {
        "target": LAYER,
        "part_name": args.part_name,
        "features": sorted(requested),
        "threshold": float(args.threshold),
        "ig_steps": int(args.ig_steps),
        "fri_steps": int(args.fri_steps),
        "records": output_rows,
    }
    path = args.output_dir / f"results_{args.part_name}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def run_assemble(args: argparse.Namespace) -> None:
    selection_path = require_path(args.selection, "--selection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result_paths = sorted(args.output_dir.glob("results_*.json"))
    if not result_paths:
        raise FileNotFoundError(f"No results_*.json in {args.output_dir}")
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    rows.sort(
        key=lambda row: (
            int(row["feature"]),
            int(row["decile"]),
            int(row["within_decile_rank"]),
        )
    )
    expected = {
        (int(row["feature"]), int(row["decile"]), int(row["within_decile_rank"]))
        for row in selection["records"]
    }
    observed = {
        (int(row["feature"]), int(row["decile"]), int(row["within_decile_rank"]))
        for row in rows
    }
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} rendered cells: {missing[:10]}")

    feature_sections: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    for feature in sorted({int(row["feature"]) for row in rows}):
        feature_rows = [row for row in rows if int(row["feature"]) == feature]
        decile_sections: list[str] = []
        for decile in sorted({int(row["decile"]) for row in feature_rows}):
            decile_rows = [
                row for row in feature_rows if int(row["decile"]) == decile
            ]
            cards: list[str] = []
            for row in decile_rows:
                metrics = row["metrics"]
                fri = metrics["fri64_final_plus_grad"]
                ig = metrics["ig_abs_sum"]
                assets = row["assets"]
                label = html.escape(str(row.get("label_name", row.get("label", ""))))
                cards.append(
                    f"""
                    <article class="event-card">
                      <div class="event-head"><b>#{int(row['within_decile_rank']) + 1}</b> {label}</div>
                      <div class="quad">
                        <figure><img src="{html.escape(assets['original'])}" loading="lazy"><figcaption>Original</figcaption></figure>
                        <figure><img src="{html.escape(assets['ig_heatmap'])}" loading="lazy"><figcaption>IG heatmap · k80 {int(ig['support_size'])}</figcaption></figure>
                        <figure><img src="{html.escape(assets['fri_heatmap'])}" loading="lazy"><figcaption>FRI-64 ranking</figcaption></figure>
                        <figure><img src="{html.escape(assets['fri_erf80'])}" loading="lazy"><figcaption>FRI ERF80 · k={int(fri['support_size'])}</figcaption></figure>
                      </div>
                      <p>activation {float(row['activation']):.3f} · latent {int(row['latent_index'])} · global positive rank {int(row['global_positive_rank']) + 1}</p>
                    </article>
                    """
                )
            decile_sections.append(
                f"<section class='decile'><h3>D{decile} · top five within decile</h3>"
                f"<div class='event-grid'>{''.join(cards)}</div></section>"
            )
            summary_rows.append(
                {
                    "feature": feature,
                    "decile": decile,
                    "events": len(decile_rows),
                    "fri_mean_k80": mean(
                        [
                            float(row["metrics"]["fri64_final_plus_grad"]["support_size"])
                            for row in decile_rows
                        ]
                    ),
                    "ig_mean_k80": mean(
                        [
                            float(row["metrics"]["ig_abs_sum"]["support_size"])
                            for row in decile_rows
                        ]
                    ),
                }
            )
        feature_sections.append(
            f"<section class='feature' id='feature-{feature}'><h2>Feature {feature}</h2>"
            f"{''.join(decile_sections)}</section>"
        )

    features = sorted({int(row["feature"]) for row in rows})
    nav = " ".join(
        f"<a href='#feature-{feature}'>F{feature}</a>" for feature in features
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perceiver FRI activation deciles</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f4f2;color:#171918;font-family:Inter,system-ui,sans-serif}}
main{{max-width:1920px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:30px;letter-spacing:0}}
.intro{{max-width:1100px;color:#4c5550;line-height:1.5}}nav{{position:sticky;top:0;z-index:3;padding:10px 0;background:#f3f4f2;border-bottom:1px solid #c9ceca}}
nav a{{display:inline-block;margin:2px 8px 2px 0;padding:5px 8px;background:#fff;border:1px solid #c4cac5;border-radius:4px;color:#174f3d;text-decoration:none}}
.feature{{padding:28px 0;border-top:2px solid #4d5550}}h2{{font-size:24px;letter-spacing:0}}h3{{font-size:17px;letter-spacing:0;color:#3d4641}}
.event-grid{{display:grid;grid-template-columns:repeat(5,minmax(340px,1fr));gap:12px;overflow-x:auto;padding-bottom:8px}}
.event-card{{min-width:340px;background:#fff;border:1px solid #cbd0cc;border-radius:6px;padding:10px}}
.event-head{{min-height:38px;font-size:13px;line-height:1.35}}.quad{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}}
figure{{margin:0;border:1px solid #dde1de;background:#eceeec}}img{{display:block;width:100%;aspect-ratio:1;object-fit:contain}}
figcaption{{padding:5px 6px;min-height:30px;background:#fff;font-size:11px;line-height:1.3}}.event-card p{{margin:8px 1px 0;font-size:11px;color:#59615c;line-height:1.4}}
@media(max-width:900px){{main{{padding:14px}}.event-grid{{grid-template-columns:repeat(5,minmax(300px,1fr))}}.event-card{{min-width:300px}}}}
</style></head><body><main><h1>Perceiver visual-residual SAE: FRI across activation deciles</h1>
<p class="intro">Ten SAE features over a frozen 10,000-image census. D0 is the highest positive-activation decile. Each row contains the five highest-ranked events within that decile. FRI ERF80 is selected by exact hard-prefix insertion; the learned Perceiver latent index is not an input-patch coordinate.</p>
<nav>{nav}</nav>{''.join(feature_sections)}</main></body></html>"""
    (args.output_dir / "gallery.html").write_text(page, encoding="utf-8")
    summary = {
        "target": LAYER,
        "selection": str(selection_path),
        "records": rows,
        "summary": summary_rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Perceiver multi-feature activation-decile FRI gallery",
        "",
        "| Feature | Decile | N | FRI mean k80 | IG mean k80 |",
        "|---:|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {row['feature']} | {row['decile']} | {row['events']} | "
        f"{row['fri_mean_k80']:.1f} | {row['ig_mean_k80']:.1f} |"
        for row in summary_rows
    )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'gallery.html'}", flush=True)


def main() -> None:
    args = parse_args()
    if not 0 < float(args.threshold) <= 1:
        raise ValueError("--threshold must be in (0, 1]")
    if any(decile < 0 or decile > 9 for decile in args.deciles):
        raise ValueError("--deciles must be in [0, 9]")
    if args.stage == "census":
        run_census(args)
    elif args.stage == "render":
        run_render(args)
    else:
        run_assemble(args)


if __name__ == "__main__":
    main()
