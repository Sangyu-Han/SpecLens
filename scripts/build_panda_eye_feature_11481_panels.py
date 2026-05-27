#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import argparse
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autolabel_eval.config import EvalConfig
from autolabel_eval.legacy import LegacyRuntime
from autolabel_eval.rendering import (
    CLIP_ZERO_RGB,
    _normalize_positive_values,
    save_feature_actmap_masked_image,
    save_model_input_image,
    save_support_mask_image,
)


@dataclass(frozen=True)
class FeatureSpec:
    block_idx: int
    feature_id: int
    rank: int
    sample_id: int
    token_idx: int
    image_path: Path
    output_slug: str

    @property
    def feature_key(self) -> str:
        return f"block_{self.block_idx}/feature_{self.feature_id}"

    @property
    def feature_dir_name(self) -> str:
        return f"block_{self.block_idx}__feature_{self.feature_id}"


FEATURE_SPECS = {
    11481: FeatureSpec(
        block_idx=10,
        feature_id=11481,
        rank=0,
        sample_id=19406,
        token_idx=61,
        image_path=Path("/data/datasets/imagenet/val/n02510455/ILSVRC2012_val_00007107.JPEG"),
        output_slug="panda_eye_feature_11481_top1",
    ),
    11389: FeatureSpec(
        block_idx=10,
        feature_id=11389,
        rank=0,
        sample_id=17596,
        token_idx=56,
        image_path=Path("/data/datasets/imagenet/val/n02422106/ILSVRC2012_val_00047424.JPEG"),
        output_slug="feature_11389_top1",
    ),
}
FEATURE_RANK_SPECS = {
    (11389, 1): FeatureSpec(
        block_idx=10,
        feature_id=11389,
        rank=1,
        sample_id=17569,
        token_idx=73,
        image_path=Path("/data/datasets/imagenet/val/n02422106/ILSVRC2012_val_00017389.JPEG"),
        output_slug="feature_11389_top2",
    ),
}

SOURCE_SESSION_DIR = (
    ROOT
    / "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions"
    / "clip50k_block10_clsnever_strictsynset4_sparse8_56_renderfix_20260429_source_panel5_t90_attr10"
)
ERFCYAN_SESSION_DIR = (
    ROOT
    / "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions"
    / "clip50k_block10_clsnever_strictsynset4_sparse8_56_renderfix_20260429_erfcyan_source"
)
OUT_ROOT = ROOT / "outputs/autolabel_metric_runs/figures"
GRID_CROP_ROW = 3
GRID_CROP_COL = 0
GRID_CROP_SIZE = 8


def _normalize_max_one(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    max_value = float(arr.max())
    if max_value <= 0.0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / max_value).astype(np.float32)


def _save_heatmap_grid(values: np.ndarray, out_path: Path, *, image_size: int = 224, grid_size: int = 14) -> None:
    grid = _normalize_max_one(np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size))
    _save_heatmap_grid_array(grid, out_path, image_size=image_size, grid_size=grid_size)


def _save_heatmap_grid_raw_clipped(
    values: np.ndarray,
    out_path: Path,
    *,
    image_size: int = 224,
    grid_size: int = 14,
) -> None:
    # Deliberately do not max-normalize. Matplotlib colormaps clip values
    # outside [0, 1], so this shows raw activations on the unit firing scale.
    grid = np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    grid = np.clip(grid, 0.0, None)
    _save_heatmap_grid_array(grid, out_path, image_size=image_size, grid_size=grid_size)


def _save_thresholded_activation_grid(
    values: np.ndarray,
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
    activation_threshold: float = 0.24,
) -> np.ndarray:
    values_grid = np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size)
    scaled = _normalize_positive_values(values_grid, lower_percentile=58.0, upper_percentile=99.5, gamma=0.9)
    mask_grid = (scaled >= float(activation_threshold)).astype(np.float32)
    token_row, token_col = divmod(int(token_idx), int(grid_size))
    if 0 <= token_row < grid_size and 0 <= token_col < grid_size:
        mask_grid[token_row, token_col] = 1.0
    if int(mask_grid.sum()) <= 0 and float(np.max(values_grid)) > 0.0:
        row, col = divmod(int(np.argmax(values_grid)), grid_size)
        mask_grid[row, col] = 1.0
    thresholded = scaled * mask_grid
    _save_heatmap_grid_array(thresholded, out_path, image_size=image_size, grid_size=grid_size)
    return thresholded.astype(np.float32)


def _save_thresholded_attribution_grid(
    values: np.ndarray,
    out_path: Path,
    *,
    min_normalized_attribution: float,
    image_size: int = 224,
    grid_size: int = 14,
) -> np.ndarray:
    grid = _normalize_max_one(np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size))
    thresholded = np.where(grid >= float(min_normalized_attribution), grid, 0.0).astype(np.float32)
    _save_heatmap_grid_array(thresholded, out_path, image_size=image_size, grid_size=grid_size)
    return thresholded


def _save_heatmap_grid_array(grid: np.ndarray, out_path: Path, *, image_size: int = 224, grid_size: int = 14) -> None:
    grid = np.asarray(grid, dtype=np.float32).reshape(grid_size, grid_size)
    cmap_values = 0.5 + 0.5 * np.clip(grid, 0.0, 1.0)
    rgba_small = colormaps["seismic"](cmap_values, bytes=True)
    rgb_small = rgba_small[..., :3]
    rgb_small[grid <= 0.0] = 255
    cell = image_size // grid_size
    heatmap = Image.fromarray(rgb_small, mode="RGB").resize((image_size, image_size), Image.NEAREST)
    draw = ImageDraw.Draw(heatmap, "RGBA")
    line_color = (68, 48, 44, 150)
    for i in range(grid_size + 1):
        pos = i * cell
        draw.line((pos, 0, pos, image_size), fill=line_color, width=1)
        draw.line((0, pos, image_size, pos), fill=line_color, width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap.save(out_path)


def _load_erf_payload(erf_json: Path) -> dict[str, Any]:
    if not erf_json.exists():
        raise FileNotFoundError(f"Missing ERF payload: {erf_json}")
    return json.loads(erf_json.read_text())


def _compute_activation_map(spec: FeatureSpec) -> np.ndarray:
    runtime = _build_runtime()
    return _compute_activation_map_with_runtime(runtime, spec)


def _build_runtime() -> LegacyRuntime:
    config = replace(
        EvalConfig(),
        workspace_root=ROOT / "outputs/autolabel_metric_runs/clip50k_full_workspace",
        model_name="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        image_size=224,
        resize_size=256,
        grid_size=14,
        n_patches=196,
        erf_recovery_threshold=0.9,
        erf_support_min_normalized_attribution=0.1,
    )
    return LegacyRuntime(config)


def _compute_activation_map_with_runtime(runtime: LegacyRuntime, spec: FeatureSpec) -> np.ndarray:
    return runtime.feature_activation_map(str(spec.image_path), spec.block_idx, spec.feature_id)


def _compute_prefix_recoveries(
    runtime: LegacyRuntime,
    spec: FeatureSpec,
    prefix_order: list[int],
) -> list[float]:
    import torch

    artifacts = runtime.forward_block(str(spec.image_path), spec.block_idx)
    capture = artifacts.capture
    sae = runtime.load_sae(spec.block_idx)
    with torch.no_grad():
        full_acts = sae(artifacts.patch_out).get("feature_acts")
        full_activation = full_acts[0, int(spec.token_idx), int(spec.feature_id)].detach()
    do_forward_masked, get_block_out = runtime.adapter.make_masked_forward(
        artifacts.x,
        capture,
        block_idx=int(spec.block_idx),
    )
    dtype = capture.patch_tokens.dtype
    dev = capture.patch_tokens.device
    with torch.no_grad():
        do_forward_masked(torch.zeros(runtime.config.n_patches, device=dev, dtype=dtype))
        baseline_block_out = get_block_out()
        prefix = runtime.adapter.prefix_count()
        baseline_patch_out = baseline_block_out[:, prefix:, :]
        baseline_acts = sae(baseline_patch_out).get("feature_acts")
        baseline_activation = baseline_acts[0, int(spec.token_idx), int(spec.feature_id)].detach()

        scores: list[float] = []
        for prefix_size in range(1, len(prefix_order) + 1):
            hard_mask = torch.zeros(runtime.config.n_patches, device=dev, dtype=dtype)
            hard_mask[torch.as_tensor(prefix_order[:prefix_size], device=dev)] = 1.0
            do_forward_masked(hard_mask)
            block_out = get_block_out()
            prefix_count = runtime.adapter.prefix_count()
            patch_out = block_out[:, prefix_count:, :]
            recovery = runtime._feature_activation_recovery_objective(
                patch_out,
                sae=sae,
                token_idx=int(spec.token_idx),
                feature_id=int(spec.feature_id),
                full_activation=full_activation,
                baseline_activation=baseline_activation,
            )
            scores.append(float(recovery.detach().cpu()))
    return scores


def _contact_sheet(paths: list[tuple[str, Path]], out_path: Path) -> None:
    label_h = 22
    gap = 8
    tile = 224
    width = len(paths) * tile + (len(paths) - 1) * gap
    height = tile + label_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        sheet.paste(image, (x, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (tile - text_w) / 2, tile + 4), label, fill=(20, 20, 20), font=font)
        x += tile + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _grid_crop_path(path: Path, *, row: int = GRID_CROP_ROW, col: int = GRID_CROP_COL, size: int = GRID_CROP_SIZE) -> Path:
    return path.with_name(f"{path.stem}_grid_r{row:02d}_c{col:02d}_{size}x{size}{path.suffix}")


def _save_grid_crop(
    in_path: Path,
    *,
    row: int = GRID_CROP_ROW,
    col: int = GRID_CROP_COL,
    size: int = GRID_CROP_SIZE,
    image_size: int = 224,
    grid_size: int = 14,
) -> Path:
    patch = image_size // grid_size
    x0 = int(col) * patch
    y0 = int(row) * patch
    side = int(size) * patch
    out_path = _grid_crop_path(in_path, row=row, col=col, size=size)
    image = Image.open(in_path).convert("RGB")
    crop = image.crop((x0, y0, x0 + side, y0 + side)).resize((image_size, image_size), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def _safe_score_text(score: float) -> str:
    return f"{float(score):.4f}".replace("-", "m").replace(".", "p")


def _export_prefix_insertions(
    *,
    spec: FeatureSpec,
    out_dir: Path,
    prefix_order: list[int],
    recoveries: list[float],
) -> dict[str, Any]:
    prefix_dir = out_dir / f"prefix_insertions_grid_r{GRID_CROP_ROW:02d}_c{GRID_CROP_COL:02d}_{GRID_CROP_SIZE}x{GRID_CROP_SIZE}"
    tmp_dir = prefix_dir / "_full_tmp"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for idx, (patch_index, score) in enumerate(zip(prefix_order, recoveries, strict=True), start=1):
        inserted = prefix_order[:idx]
        score_text = _safe_score_text(score)
        full_path = tmp_dir / f"prefix_k{idx:03d}_recovery_{score_text}_full.png"
        save_support_mask_image(
            str(spec.image_path),
            inserted,
            full_path,
            token_idx=spec.token_idx,
            image_size=224,
            grid_size=14,
            resize_size=256,
            mode="masked_black",
            background_color=CLIP_ZERO_RGB,
            include_token_box=True,
            token_marker_style="cross",
            token_marker_color=(0, 220, 255),
            mask_resample=Image.NEAREST,
        )
        crop_path = prefix_dir / f"prefix_k{idx:03d}_recovery_{score_text}.png"
        crop_tmp = _save_grid_crop(full_path)
        Image.open(crop_tmp).convert("RGB").save(crop_path)
        crop_tmp.unlink(missing_ok=True)
        records.append(
            {
                "prefix_size": idx,
                "new_patch_index": int(patch_index),
                "inserted_patch_indices": [int(v) for v in inserted],
                "recovery": float(score),
                "image": str(crop_path),
            }
        )
    for full_path in tmp_dir.glob("*.png"):
        full_path.unlink()
    tmp_dir.rmdir()
    sheet_path = prefix_dir / "prefix_insertions_contact_sheet.png"
    tile = 112
    label_h = 20
    cols = 8
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
    for item_idx, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB").resize((tile, tile), Image.NEAREST)
        x = (item_idx % cols) * tile
        y = (item_idx // cols) * (tile + label_h)
        sheet.paste(image, (x, y))
        label = f"k={int(record['prefix_size'])} rec={float(record['recovery']):.3f}"
        draw.text((x + 3, y + tile + 3), label, fill=(20, 20, 20), font=font)
    sheet.save(sheet_path)
    manifest = {
        "feature_key": spec.feature_key,
        "rank": spec.rank,
        "sample_id": spec.sample_id,
        "token_idx": spec.token_idx,
        "grid_crop": {
            "row": GRID_CROP_ROW,
            "col": GRID_CROP_COL,
            "size": GRID_CROP_SIZE,
            "pixel_box_before_resize": [
                GRID_CROP_COL * (224 // 14),
                GRID_CROP_ROW * (224 // 14),
                (GRID_CROP_COL + GRID_CROP_SIZE) * (224 // 14),
                (GRID_CROP_ROW + GRID_CROP_SIZE) * (224 // 14),
            ],
            "resized_to": [224, 224],
        },
        "records": records,
    }
    manifest_path = prefix_dir / "prefix_recovery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {
        "prefix_insertions_dir": str(prefix_dir),
        "prefix_insertions_manifest": str(manifest_path),
        "prefix_insertions_contact_sheet": str(sheet_path),
        "prefix_insertions_count": len(records),
        "prefix_final_recovery": float(recoveries[-1]) if recoveries else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-id", type=int, default=11481, choices=sorted(FEATURE_SPECS))
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--export-prefix-insertions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feature_id = int(args.feature_id)
    rank = int(args.rank)
    if rank == 0:
        spec = FEATURE_SPECS[feature_id]
    else:
        key = (feature_id, rank)
        if key not in FEATURE_RANK_SPECS:
            raise ValueError(f"No built-in spec for feature_id={feature_id} rank={rank}")
        spec = FEATURE_RANK_SPECS[key]
    source_dir = SOURCE_SESSION_DIR / "assets" / spec.feature_dir_name
    erfcyan_dir = ERFCYAN_SESSION_DIR / "assets" / spec.feature_dir_name
    erf_json = erfcyan_dir / f"example_{spec.rank:02d}_feature_erf.json"
    out_dir = OUT_ROOT / spec.output_slug

    if not spec.image_path.exists():
        raise FileNotFoundError(f"Missing ImageNet source image: {spec.image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    erf_payload = _load_erf_payload(erf_json)
    support_indices = [int(v) for v in erf_payload.get("support_indices", [])]
    if not support_indices:
        raise ValueError(f"ERF payload has no support_indices: {erf_json}")

    runtime = _build_runtime()
    activation_map = _compute_activation_map_with_runtime(runtime, spec)
    attribution_map = np.asarray(erf_payload["normalized_attribution"], dtype=np.float32)

    cropped_path = out_dir / "cropped_image.png"
    sae_masked_path = out_dir / "sae_only_masked.png"
    erf_masked_path = out_dir / "erf_masked_cyan_cross.png"
    activation_grid_path = out_dir / "sae_activation_grid_seismic.png"
    activation_grid_thresholded_path = out_dir / "sae_activation_grid_seismic_thresholded.png"
    activation_grid_raw_path = out_dir / "sae_activation_grid_seismic_raw_clipped.png"
    attribution_grid_path = out_dir / "attribution_grid_seismic.png"
    attribution_grid_thresholded_path = out_dir / "attribution_grid_seismic_thresholded.png"
    top_label = f"top{spec.rank + 1}"
    contact_sheet_path = out_dir / f"feature_{spec.feature_id}_{top_label}_panels.png"
    contact_sheet_raw_path = out_dir / f"feature_{spec.feature_id}_{top_label}_panels_raw_al_act.png"

    save_model_input_image(str(spec.image_path), cropped_path, image_size=224, resize_size=256)
    save_feature_actmap_masked_image(
        str(spec.image_path),
        activation_map,
        sae_masked_path,
        token_idx=spec.token_idx,
        image_size=224,
        grid_size=14,
        resize_size=256,
        background_color=CLIP_ZERO_RGB,
    )
    existing_erf_cross = erfcyan_dir / f"example_{spec.rank:02d}_feature_erf_cyan_cross.png"
    if existing_erf_cross.exists():
        Image.open(existing_erf_cross).convert("RGB").save(erf_masked_path)
    else:
        save_support_mask_image(
            str(spec.image_path),
            support_indices,
            erf_masked_path,
            token_idx=spec.token_idx,
            image_size=224,
            grid_size=14,
            resize_size=256,
            mode="masked_black",
            background_color=CLIP_ZERO_RGB,
            include_token_box=True,
            token_marker_style="cross",
            token_marker_color=(0, 220, 255),
            mask_resample=Image.NEAREST,
        )
    _save_heatmap_grid(activation_map, activation_grid_path)
    thresholded_activation_grid = _save_thresholded_activation_grid(
        activation_map,
        activation_grid_thresholded_path,
        token_idx=spec.token_idx,
    )
    _save_heatmap_grid_raw_clipped(activation_map, activation_grid_raw_path)
    _save_heatmap_grid(attribution_map, attribution_grid_path)
    min_attr = float(erf_payload.get("support_min_normalized_attribution", 0.1))
    thresholded_attribution_grid = _save_thresholded_attribution_grid(
        attribution_map,
        attribution_grid_thresholded_path,
        min_normalized_attribution=min_attr,
    )

    np.save(out_dir / "sae_activation_grid_raw_values.npy", np.asarray(activation_map, dtype=np.float32).reshape(14, 14))
    np.save(out_dir / "sae_activation_grid_values.npy", _normalize_max_one(activation_map).reshape(14, 14))
    np.save(out_dir / "sae_activation_grid_thresholded_values.npy", thresholded_activation_grid)
    np.save(out_dir / "attribution_grid_values.npy", _normalize_max_one(attribution_map).reshape(14, 14))
    np.save(out_dir / "attribution_grid_thresholded_values.npy", thresholded_attribution_grid)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature_key": spec.feature_key,
                "block_idx": spec.block_idx,
                "feature_id": spec.feature_id,
                "rank": spec.rank,
                "sample_id": spec.sample_id,
                "token_idx": spec.token_idx,
                "image_path": str(spec.image_path),
                "model_name": "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
                "source_dir": str(source_dir),
                "erf_json": str(erf_json),
                "support_size": len(support_indices),
                "support_threshold": float(erf_payload.get("support_threshold", 0.9)),
                "support_recovery": float(erf_payload.get("support_recovery", 0.0)),
                "support_min_normalized_attribution": min_attr,
                "activation_max": float(np.asarray(activation_map, dtype=np.float32).max()),
                "attribution_max_before_normalization": float(np.asarray(attribution_map, dtype=np.float32).max()),
                "outputs": {
                    "cropped_image": str(cropped_path),
                    "sae_only_masked": str(sae_masked_path),
                    "erf_masked": str(erf_masked_path),
                    "sae_activation_grid_seismic": str(activation_grid_path),
                    "sae_activation_grid_seismic_thresholded": str(activation_grid_thresholded_path),
                    "sae_activation_grid_seismic_raw_clipped": str(activation_grid_raw_path),
                    "attribution_grid_seismic": str(attribution_grid_path),
                    "attribution_grid_seismic_thresholded": str(attribution_grid_thresholded_path),
                    "contact_sheet": str(contact_sheet_path),
                    "contact_sheet_raw_al_act": str(contact_sheet_raw_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    _contact_sheet(
        [
            ("crop", cropped_path),
            ("AL mask", sae_masked_path),
            ("ERF mask", erf_masked_path),
            ("AL act.", activation_grid_thresholded_path),
            ("ERF attr.", attribution_grid_thresholded_path),
        ],
        contact_sheet_path,
    )
    _contact_sheet(
        [
            ("crop", cropped_path),
            ("AL mask", sae_masked_path),
            ("ERF mask", erf_masked_path),
            ("AL act. raw", activation_grid_raw_path),
            ("ERF attr.", attribution_grid_thresholded_path),
        ],
        contact_sheet_raw_path,
    )
    grid_crop_paths = {
        "cropped_image_grid_crop": _save_grid_crop(cropped_path),
        "sae_only_masked_grid_crop": _save_grid_crop(sae_masked_path),
        "erf_masked_grid_crop": _save_grid_crop(erf_masked_path),
        "sae_activation_grid_thresholded_grid_crop": _save_grid_crop(activation_grid_thresholded_path),
        "attribution_grid_thresholded_grid_crop": _save_grid_crop(attribution_grid_thresholded_path),
    }
    contact_sheet_grid_crop_path = out_dir / f"feature_{spec.feature_id}_{top_label}_panels_grid_r{GRID_CROP_ROW:02d}_c{GRID_CROP_COL:02d}_{GRID_CROP_SIZE}x{GRID_CROP_SIZE}.png"
    _contact_sheet(
        [
            ("crop", grid_crop_paths["cropped_image_grid_crop"]),
            ("AL mask", grid_crop_paths["sae_only_masked_grid_crop"]),
            ("ERF mask", grid_crop_paths["erf_masked_grid_crop"]),
            ("AL act.", grid_crop_paths["sae_activation_grid_thresholded_grid_crop"]),
            ("ERF attr.", grid_crop_paths["attribution_grid_thresholded_grid_crop"]),
        ],
        contact_sheet_grid_crop_path,
    )
    metadata_path = out_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["grid_crop"] = {
        "row": GRID_CROP_ROW,
        "col": GRID_CROP_COL,
        "size": GRID_CROP_SIZE,
        "pixel_box_before_resize": [
            GRID_CROP_COL * (224 // 14),
            GRID_CROP_ROW * (224 // 14),
            (GRID_CROP_COL + GRID_CROP_SIZE) * (224 // 14),
            (GRID_CROP_ROW + GRID_CROP_SIZE) * (224 // 14),
        ],
        "resized_to": [224, 224],
    }
    metadata["outputs"].update({key: str(value) for key, value in grid_crop_paths.items()})
    metadata["outputs"]["contact_sheet_grid_crop"] = str(contact_sheet_grid_crop_path)
    if bool(args.export_prefix_insertions):
        prefix_order = [int(v) for v in support_indices]
        recoveries = _compute_prefix_recoveries(runtime, spec, prefix_order)
        prefix_outputs = _export_prefix_insertions(
            spec=spec,
            out_dir=out_dir,
            prefix_order=prefix_order,
            recoveries=recoveries,
        )
        metadata["outputs"].update(prefix_outputs)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(json.loads((out_dir / "metadata.json").read_text())["outputs"], indent=2))


if __name__ == "__main__":
    main()
