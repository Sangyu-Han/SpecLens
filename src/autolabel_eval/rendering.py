from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLIP_ZERO_RGB = (
    int(round(0.48145466 * 255.0)),
    int(round(0.4578275 * 255.0)),
    int(round(0.40821073 * 255.0)),
)


def _resize_image(image: Image.Image, size: int = 224) -> Image.Image:
    return image.resize((size, size), Image.BICUBIC)


def _patch_box(token_idx: int, grid_size: int = 14, image_size: int = 224) -> tuple[int, int, int, int]:
    patch = image_size // grid_size
    row, col = divmod(int(token_idx), int(grid_size))
    x0 = col * patch
    y0 = row * patch
    return x0, y0, x0 + patch, y0 + patch


def _patch_center(token_idx: int, grid_size: int = 14, image_size: int = 224) -> tuple[float, float]:
    x0, y0, x1, y1 = _patch_box(token_idx, grid_size=grid_size, image_size=image_size)
    return (float(x0 + x1) / 2.0, float(y0 + y1) / 2.0)


def _draw_patch_dot(
    image: Image.Image,
    token_idx: int,
    *,
    grid_size: int = 14,
    image_size: int = 224,
    fill: tuple[int, int, int] = (40, 220, 80),
    outline: tuple[int, int, int] = (255, 255, 255),
    radius_ratio: float = 0.12,
) -> None:
    patch = image_size / max(grid_size, 1)
    radius = max(2.0, patch * float(radius_ratio))
    cx, cy = _patch_center(token_idx, grid_size=grid_size, image_size=image_size)
    draw = ImageDraw.Draw(image)
    outer = (cx - radius - 1.0, cy - radius - 1.0, cx + radius + 1.0, cy + radius + 1.0)
    inner = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(outer, fill=outline)
    draw.ellipse(inner, fill=fill)


def _draw_patch_box(
    image: Image.Image,
    token_idx: int,
    *,
    grid_size: int = 14,
    image_size: int = 224,
    outline: tuple[int, int, int] = (255, 64, 64),
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        _patch_box(token_idx, grid_size=grid_size, image_size=image_size),
        outline=outline,
        width=max(1, int(width)),
    )


def _rgba(color: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (int(color[0]), int(color[1]), int(color[2]), max(0, min(255, int(alpha))))


def _alpha_composite_overlay(image: Image.Image, painter: callable) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    painter(draw)
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    image.paste(composed)


def _draw_patch_dot_translucent(
    image: Image.Image,
    token_idx: int,
    *,
    grid_size: int = 14,
    image_size: int = 224,
    fill: tuple[int, int, int] = (0, 220, 255),
    outline: tuple[int, int, int] = (180, 250, 255),
    fill_alpha: int = 116,
    outline_alpha: int = 168,
    radius_ratio: float = 0.11,
) -> None:
    patch = image_size / max(grid_size, 1)
    radius = max(2.0, patch * float(radius_ratio))
    cx, cy = _patch_center(token_idx, grid_size=grid_size, image_size=image_size)

    def _paint(draw: ImageDraw.ImageDraw) -> None:
        outer = (cx - radius - 1.0, cy - radius - 1.0, cx + radius + 1.0, cy + radius + 1.0)
        inner = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(outer, fill=_rgba(outline, outline_alpha))
        draw.ellipse(inner, fill=_rgba(fill, fill_alpha))

    _alpha_composite_overlay(image, _paint)


def _draw_patch_cross(
    image: Image.Image,
    token_idx: int,
    *,
    grid_size: int = 14,
    image_size: int = 224,
    color: tuple[int, int, int] = (0, 220, 255),
    alpha: int = 178,
    width: int = 2,
    arm_ratio: float = 0.36,
) -> None:
    patch = image_size / max(grid_size, 1)
    arm = max(4.0, patch * float(arm_ratio))
    cx, cy = _patch_center(token_idx, grid_size=grid_size, image_size=image_size)

    def _paint(draw: ImageDraw.ImageDraw) -> None:
        rgba = _rgba(color, alpha)
        draw.line((cx - arm, cy, cx + arm, cy), fill=rgba, width=max(1, int(width)))
        draw.line((cx, cy - arm, cx, cy + arm), fill=rgba, width=max(1, int(width)))

    _alpha_composite_overlay(image, _paint)


def _draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    *,
    horizontal: bool,
    fixed: float,
    start: float,
    end: float,
    color: tuple[int, int, int, int],
    width: int,
    dash: float,
    gap: float,
) -> None:
    pos = float(start)
    while pos < float(end):
        seg_end = min(float(end), pos + float(dash))
        if horizontal:
            draw.line((pos, fixed, seg_end, fixed), fill=color, width=max(1, int(width)))
        else:
            draw.line((fixed, pos, fixed, seg_end), fill=color, width=max(1, int(width)))
        pos = seg_end + float(gap)


def _draw_patch_dashed_box(
    image: Image.Image,
    token_idx: int,
    *,
    grid_size: int = 14,
    image_size: int = 224,
    color: tuple[int, int, int] = (0, 220, 255),
    alpha: int = 168,
    width: int = 2,
    inset: float = 1.0,
    dash_ratio: float = 0.20,
    gap_ratio: float = 0.12,
) -> None:
    x0, y0, x1, y1 = _patch_box(token_idx, grid_size=grid_size, image_size=image_size)
    patch = image_size / max(grid_size, 1)
    dash = max(3.0, patch * float(dash_ratio))
    gap = max(2.0, patch * float(gap_ratio))

    def _paint(draw: ImageDraw.ImageDraw) -> None:
        rgba = _rgba(color, alpha)
        lx0 = x0 + inset
        ly0 = y0 + inset
        lx1 = x1 - inset
        ly1 = y1 - inset
        _draw_dashed_segment(draw, horizontal=True, fixed=ly0, start=lx0, end=lx1, color=rgba, width=width, dash=dash, gap=gap)
        _draw_dashed_segment(draw, horizontal=True, fixed=ly1, start=lx0, end=lx1, color=rgba, width=width, dash=dash, gap=gap)
        _draw_dashed_segment(draw, horizontal=False, fixed=lx0, start=ly0, end=ly1, color=rgba, width=width, dash=dash, gap=gap)
        _draw_dashed_segment(draw, horizontal=False, fixed=lx1, start=ly0, end=ly1, color=rgba, width=width, dash=dash, gap=gap)

    _alpha_composite_overlay(image, _paint)


def _support_grid(support_indices: Iterable[int], grid_size: int) -> np.ndarray:
    support = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for index in support_indices:
        row, col = divmod(int(index), grid_size)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            support[row, col] = 1
    return support


def _checkerboard_array(
    height: int,
    width: int,
    *,
    tile: int = 8,
    light: tuple[int, int, int] = (200, 200, 200),
    dark: tuple[int, int, int] = (160, 160, 160),
) -> np.ndarray:
    tile = max(1, int(tile))
    rr = (np.arange(height) // tile) % 2
    cc = (np.arange(width) // tile) % 2
    parity = (rr[:, None] + cc[None, :]) % 2
    out = np.empty((height, width, 3), dtype=np.float32)
    out[parity == 0] = np.asarray(light, dtype=np.float32)
    out[parity == 1] = np.asarray(dark, dtype=np.float32)
    return out


def _checkerboard_overlay(
    height: int,
    width: int,
    *,
    tile_px: int = 4,
    light_rgb: tuple[int, int, int] = (242, 242, 242),
    dark_rgb: tuple[int, int, int] = (124, 124, 124),
) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.int32)
    tiles = ((yy // max(1, int(tile_px))) + (xx // max(1, int(tile_px)))) % 2
    overlay = np.empty((height, width, 3), dtype=np.float32)
    overlay[tiles == 0] = np.asarray(light_rgb, dtype=np.float32)
    overlay[tiles == 1] = np.asarray(dark_rgb, dtype=np.float32)
    return overlay


def _normalize_positive_values(
    values: np.ndarray,
    *,
    lower_percentile: float,
    upper_percentile: float,
    gamma: float = 1.0,
) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, None)
    positive = values[values > 0.0]
    if positive.size:
        lo = float(np.percentile(positive, float(lower_percentile)))
        hi = float(np.percentile(positive, float(upper_percentile)))
        if hi <= lo:
            lo = float(positive.min())
            hi = float(positive.max())
    else:
        lo = 0.0
        hi = max(float(values.max()), 1e-6)
    scaled = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        scaled = np.power(scaled, float(gamma))
    return scaled


def _interpolate_rgb(
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
    weight: np.ndarray,
) -> np.ndarray:
    start = np.asarray(start_rgb, dtype=np.float32) / 255.0
    end = np.asarray(end_rgb, dtype=np.float32) / 255.0
    return start + (end - start) * weight[..., None]


def _cyan_sequential_colormap(scaled: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.asarray(scaled, dtype=np.float32), 0.0, 1.0)
    dark_rgb = (7, 19, 27)
    mid_rgb = (17, 157, 181)
    high_rgb = (218, 255, 255)
    split = 0.72
    low_weight = np.clip(scaled / split, 0.0, 1.0)
    high_weight = np.clip((scaled - split) / max(1.0 - split, 1e-6), 0.0, 1.0)
    low_branch = _interpolate_rgb(dark_rgb, mid_rgb, low_weight)
    high_branch = _interpolate_rgb(mid_rgb, high_rgb, high_weight)
    return np.where((scaled < split)[..., None], low_branch, high_branch)


def _core_grid_from_scores(
    support: np.ndarray,
    score_map: Sequence[float] | None,
    *,
    grid_size: int,
    mass_fraction: float,
) -> np.ndarray:
    core = np.zeros_like(support, dtype=np.uint8)
    if score_map is None:
        return core
    scores = np.asarray(score_map, dtype=np.float32).reshape(grid_size, grid_size)
    scores = np.maximum(scores, 0.0) * support.astype(np.float32)
    coords = np.argwhere(support > 0)
    if coords.size == 0:
        return core
    values = scores[support > 0]
    total = float(values.sum())
    if total <= 0.0:
        return core
    target = max(0.0, min(1.0, float(mass_fraction))) * total
    order = np.argsort(-values, kind="mergesort")
    running = 0.0
    for idx in order:
        row, col = coords[int(idx)]
        core[int(row), int(col)] = 1
        running += float(values[int(idx)])
        if running >= target:
            break
    return core


def _draw_patch_contours(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    grid_size: int,
    offset_x: int,
    offset_y: int,
    cell_w: float,
    cell_h: float,
    color: tuple[int, int, int],
    width: int,
) -> None:
    for row in range(row_start, row_stop):
        for col in range(col_start, col_stop):
            if not mask[row, col]:
                continue
            local_row = row - row_start
            local_col = col - col_start
            x0 = offset_x + local_col * cell_w
            x1 = offset_x + (local_col + 1) * cell_w
            y0 = offset_y + local_row * cell_h
            y1 = offset_y + (local_row + 1) * cell_h

            if row == 0 or not mask[row - 1, col]:
                draw.line((x0, y0, x1, y0), fill=color, width=width)
            if row == grid_size - 1 or not mask[row + 1, col]:
                draw.line((x0, y1, x1, y1), fill=color, width=width)
            if col == 0 or not mask[row, col - 1]:
                draw.line((x0, y0, x0, y1), fill=color, width=width)
            if col == grid_size - 1 or not mask[row, col + 1]:
                draw.line((x1, y0, x1, y1), fill=color, width=width)


def _expand_bounds(start: int, stop: int, *, margin: int, min_size: int, limit: int) -> tuple[int, int]:
    start = max(0, int(start) - int(margin))
    stop = min(int(limit), int(stop) + int(margin))
    size = stop - start
    if size >= int(min_size):
        return start, stop

    need = int(min_size) - size
    grow_lo = need // 2
    grow_hi = need - grow_lo
    start = max(0, start - grow_lo)
    stop = min(int(limit), stop + grow_hi)
    size = stop - start
    if size >= int(min_size):
        return start, stop
    if start == 0:
        stop = min(int(limit), int(min_size))
    elif stop == int(limit):
        start = max(0, int(limit) - int(min_size))
    return start, stop


def save_original_with_token_box(
    image_path: str,
    out_path: Path,
    token_idx: int,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    marker_style: str = "box",
) -> None:
    image = _resize_image(Image.open(image_path).convert("RGB"))
    style = str(marker_style).strip().lower()
    if style == "cross":
        _draw_patch_cross(image, token_idx, color=color, alpha=210, width=3)
    else:
        draw = ImageDraw.Draw(image)
        draw.rectangle(_patch_box(token_idx), outline=color, width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def save_activation_region_image(
    image_path: str,
    activation_map: Sequence[float],
    out_path: Path,
    *,
    image_size: int = 224,
    grid_size: int = 14,
) -> None:
    image = _resize_image(Image.open(image_path).convert("RGB"), size=image_size)
    acts = np.asarray(activation_map, dtype=np.float32).reshape(grid_size, grid_size)
    mask_small = (acts < 1e-5).astype(np.uint8) * 224
    mask = Image.fromarray(mask_small, mode="L").resize((image_size, image_size), Image.NEAREST)
    background = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    composite = Image.composite(background, image, mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)


def save_feature_actmap_masked_image(
    image_path: str,
    activation_map: Sequence[float],
    out_path: Path,
    *,
    token_idx: int | None = None,
    image_size: int = 224,
    grid_size: int = 14,
    activation_threshold: float = 0.24,
    include_token_box: bool = False,
    box_color: tuple[int, int, int] = (0, 255, 255),
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, Any]:
    image = _resize_image(Image.open(image_path).convert("RGB"), size=image_size)
    values = np.asarray(activation_map, dtype=np.float32).reshape(grid_size, grid_size)
    scaled = _normalize_positive_values(values, lower_percentile=58.0, upper_percentile=99.5, gamma=0.9)
    mask_grid = (scaled >= float(activation_threshold)).astype(np.uint8)
    if token_idx is not None:
        token_row, token_col = divmod(int(token_idx), grid_size)
        if 0 <= token_row < grid_size and 0 <= token_col < grid_size:
            mask_grid[token_row, token_col] = 1
    all_zero_map = bool(float(np.max(values)) <= 0.0)
    fallback_to_argmax = False
    if int(mask_grid.sum()) <= 0 and not all_zero_map:
        row, col = divmod(int(np.argmax(values)), grid_size)
        mask_grid[row, col] = 1
        fallback_to_argmax = True

    mask = Image.fromarray((mask_grid * 255).astype(np.uint8), mode="L").resize((image_size, image_size), Image.NEAREST)
    background = Image.new("RGB", (image_size, image_size), background_color)
    composite = Image.composite(image, background, mask)
    if include_token_box and token_idx is not None:
        draw = ImageDraw.Draw(composite)
        draw.rectangle(_patch_box(token_idx, grid_size=grid_size, image_size=image_size), outline=box_color, width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)
    return {
        "visible_patch_count": int(mask_grid.sum()),
        "activation_threshold": float(activation_threshold),
        "overlay_mode": "masked_visible_patches_only",
        "all_zero_activation_map": bool(all_zero_map),
        "fallback_to_argmax": bool(fallback_to_argmax),
    }


def save_feature_actmap_overlay(
    image_path: str,
    activation_map: Sequence[float],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
) -> None:
    image = np.asarray(_resize_image(Image.open(image_path).convert("RGB"), size=image_size), dtype=np.float32) / 255.0
    values = np.asarray(activation_map, dtype=np.float32).reshape(grid_size, grid_size)
    scaled = _normalize_positive_values(values, lower_percentile=58.0, upper_percentile=99.5, gamma=0.9)
    focus = np.clip((scaled - 0.10) / 0.90, 0.0, 1.0)
    alpha_small = np.power(focus, 0.85) * 0.72
    alpha_img = Image.fromarray(np.clip(alpha_small * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
        (image_size, image_size),
        Image.NEAREST,
    )
    alpha = (np.asarray(alpha_img, dtype=np.float32) / 255.0)[..., None]
    overlay_rgb = np.asarray((64, 224, 255), dtype=np.float32) / 255.0
    blended = np.clip(image * (1.0 - alpha) + overlay_rgb * alpha, 0.0, 1.0)
    result = Image.fromarray((blended * 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    draw.rectangle(_patch_box(token_idx, grid_size=grid_size, image_size=image_size), outline=(0, 255, 255), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)


def save_sae_fire_on_original(
    image_path: str,
    activation_map: Sequence[float],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
    lower_percentile: float = 58.0,
    upper_percentile: float = 99.5,
    gamma: float = 0.9,
    threshold: float = 0.10,
) -> None:
    image = np.asarray(_resize_image(Image.open(image_path).convert("RGB"), size=image_size), dtype=np.float32)
    values = np.asarray(activation_map, dtype=np.float32).reshape(grid_size, grid_size)
    scaled = _normalize_positive_values(values, lower_percentile=lower_percentile, upper_percentile=upper_percentile, gamma=gamma)
    focus = np.clip((scaled - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    alpha_img = Image.fromarray(np.clip(focus * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
        (image_size, image_size),
        Image.BILINEAR,
    )
    alpha = (np.asarray(alpha_img, dtype=np.float32) / 255.0)[..., None]
    blended = image * alpha
    result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    draw.rectangle(_patch_box(token_idx, grid_size=grid_size, image_size=image_size), outline=(0, 255, 255), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)


def save_support_mask_image(
    image_path: str,
    support_indices: Iterable[int],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
    dim_factor: float = 0.15,
    mode: str = "dimmed",
    background_color: tuple[int, int, int] = (0, 0, 0),
    include_token_box: bool = True,
    mask_resample: int = Image.NEAREST,
    token_marker_style: str = "box",
    token_marker_color: tuple[int, int, int] = (40, 220, 80),
    token_marker_alpha: int = 168,
) -> Image.Image:
    image = np.asarray(_resize_image(Image.open(image_path).convert("RGB"), size=image_size), dtype=np.float32)
    support = np.zeros((grid_size, grid_size), dtype=np.float32)
    for index in support_indices:
        row, col = divmod(int(index), grid_size)
        support[row, col] = 1.0
    support_mask = Image.fromarray((support * 255).astype(np.uint8), mode="L").resize((image_size, image_size), mask_resample)
    support_arr = np.asarray(support_mask, dtype=np.float32) / 255.0
    if str(mode).strip().lower() == "masked_black":
        hidden = np.zeros_like(image)
        hidden[...] = np.asarray(background_color, dtype=np.float32)
        blended = hidden + (image - hidden) * support_arr[..., None]
    else:
        dark = image * float(dim_factor)
        blended = dark + (image - dark) * support_arr[..., None]
    result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    if include_token_box:
        marker_style = str(token_marker_style).strip().lower()
        if marker_style == "dot":
            _draw_patch_dot(
                result,
                token_idx,
                grid_size=grid_size,
                image_size=image_size,
                fill=token_marker_color,
            )
        elif marker_style == "dot_translucent":
            _draw_patch_dot_translucent(
                result,
                token_idx,
                grid_size=grid_size,
                image_size=image_size,
                fill=token_marker_color,
                fill_alpha=token_marker_alpha,
                outline_alpha=min(255, int(token_marker_alpha) + 36),
            )
        elif marker_style == "cross":
            _draw_patch_cross(
                result,
                token_idx,
                grid_size=grid_size,
                image_size=image_size,
                color=token_marker_color,
                alpha=token_marker_alpha,
            )
        elif marker_style == "dashed_box":
            _draw_patch_dashed_box(
                result,
                token_idx,
                grid_size=grid_size,
                image_size=image_size,
                color=token_marker_color,
                alpha=token_marker_alpha,
            )
        else:
            _draw_patch_box(
                result,
                token_idx,
                grid_size=grid_size,
                image_size=image_size,
                outline=token_marker_color,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return result


def _build_locator_grid_image(
    support_indices: Iterable[int],
    *,
    token_idx: int,
    grid_size: int = 14,
    size: int = 112,
    padding: int = 8,
    background: tuple[int, int, int] = (248, 245, 239),
    base_fill: tuple[int, int, int] = (232, 227, 220),
    support_fill: tuple[int, int, int] = (183, 177, 170),
    support_outline: tuple[int, int, int] = (153, 147, 140),
    target_outline: tuple[int, int, int] = (255, 64, 64),
    target_fill: tuple[int, int, int] = (255, 239, 239),
) -> Image.Image:
    size = max(56, int(size))
    padding = max(4, int(padding))
    canvas = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(canvas)
    usable = max(1, size - 2 * padding)
    cell = usable / max(grid_size, 1)
    support_set = {int(index) for index in support_indices}
    target = int(token_idx)
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            x0 = padding + col * cell
            y0 = padding + row * cell
            x1 = padding + (col + 1) * cell
            y1 = padding + (row + 1) * cell
            fill = support_fill if idx in support_set else base_fill
            outline = support_outline if idx in support_set else (210, 205, 198)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=1)
            if idx == target:
                inset = max(1.0, cell * 0.12)
                draw.rectangle(
                    (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
                    fill=target_fill,
                    outline=target_outline,
                    width=max(2, int(round(cell * 0.15))),
                )
    border = ImageDraw.Draw(canvas)
    border.rectangle((0, 0, size - 1, size - 1), outline=(190, 184, 176), width=1)
    return canvas


def save_support_locator_grid_image(
    image_path: str,
    support_indices: Iterable[int],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
    dim_factor: float = 0.15,
    background_color: tuple[int, int, int] = (0, 0, 0),
    mask_resample: int = Image.NEAREST,
    locator_size: int = 112,
    gutter: int = 16,
) -> None:
    main = save_support_mask_image(
        image_path,
        support_indices,
        out_path,
        token_idx=token_idx,
        image_size=image_size,
        grid_size=grid_size,
        dim_factor=dim_factor,
        mode="masked_black",
        background_color=background_color,
        include_token_box=False,
        mask_resample=mask_resample,
    )
    locator = _build_locator_grid_image(
        support_indices,
        token_idx=token_idx,
        grid_size=grid_size,
        size=locator_size,
    )
    canvas_w = int(image_size + gutter + locator_size)
    canvas_h = int(max(image_size, locator_size))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 241, 234))
    canvas.paste(main, (0, (canvas_h - int(image_size)) // 2))
    canvas.paste(locator, (int(image_size + gutter), (canvas_h - int(locator_size)) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_support_overview_zoom_image(
    image_path: str,
    support_indices: Iterable[int],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
    dim_factor: float = 0.15,
    background_color: tuple[int, int, int] = (0, 0, 0),
    mask_resample: int = Image.NEAREST,
    token_marker_color: tuple[int, int, int] = (255, 64, 64),
    crop_size: int = 96,
    zoom_size: int = 176,
    gutter: int = 16,
) -> None:
    overview = save_support_mask_image(
        image_path,
        support_indices,
        out_path,
        token_idx=token_idx,
        image_size=image_size,
        grid_size=grid_size,
        dim_factor=dim_factor,
        mode="masked_black",
        background_color=background_color,
        include_token_box=True,
        mask_resample=mask_resample,
        token_marker_style="box",
        token_marker_color=token_marker_color,
    )
    x0, y0, x1, y1 = _patch_box(token_idx, grid_size=grid_size, image_size=image_size)
    cx = int(round((x0 + x1) / 2.0))
    cy = int(round((y0 + y1) / 2.0))
    crop_size = max(int(crop_size), x1 - x0)
    left = max(0, min(image_size - crop_size, cx - crop_size // 2))
    top = max(0, min(image_size - crop_size, cy - crop_size // 2))
    crop = overview.crop((left, top, left + crop_size, top + crop_size)).resize((zoom_size, zoom_size), Image.NEAREST)
    zx0 = int(round((x0 - left) * zoom_size / max(crop_size, 1)))
    zy0 = int(round((y0 - top) * zoom_size / max(crop_size, 1)))
    zx1 = int(round((x1 - left) * zoom_size / max(crop_size, 1)))
    zy1 = int(round((y1 - top) * zoom_size / max(crop_size, 1)))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((zx0, zy0, zx1, zy1), outline=token_marker_color, width=3)
    draw.rectangle((0, 0, zoom_size - 1, zoom_size - 1), outline=(190, 184, 176), width=1)
    canvas_w = int(image_size + gutter + zoom_size)
    canvas_h = int(max(image_size, zoom_size))
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 241, 234))
    canvas.paste(overview, (0, (canvas_h - int(image_size)) // 2))
    canvas.paste(crop, (int(image_size + gutter), (canvas_h - int(zoom_size)) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_support_outline_crop_image(
    image_path: str,
    support_indices: Iterable[int],
    out_path: Path,
    *,
    token_idx: int | None = None,
    score_map: Sequence[float] | None = None,
    image_size: int = 224,
    grid_size: int = 14,
    margin_patches: int = 0,
    min_crop_patches: int = 4,
    max_recommended_area_ratio: float = 0.70,
    outline_color: tuple[int, int, int] = (255, 180, 0),
    background_mode: str = "checkerboard",
    background_color: tuple[int, int, int] = (18, 18, 18),
    checkerboard_tile: int = 8,
    checkerboard_light: tuple[int, int, int] = (200, 200, 200),
    checkerboard_dark: tuple[int, int, int] = (160, 160, 160),
) -> dict[str, Any]:
    support = _support_grid(support_indices, grid_size=grid_size)
    rows, cols = np.nonzero(support)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Empty support_indices")
    token_row = token_col = None
    if token_idx is not None:
        token_row, token_col = divmod(int(token_idx), grid_size)
        rows = np.concatenate([rows, np.asarray([token_row], dtype=rows.dtype)])
        cols = np.concatenate([cols, np.asarray([token_col], dtype=cols.dtype)])

    row_start, row_stop = _expand_bounds(
        int(rows.min()),
        int(rows.max()) + 1,
        margin=margin_patches,
        min_size=min_crop_patches,
        limit=grid_size,
    )
    col_start, col_stop = _expand_bounds(
        int(cols.min()),
        int(cols.max()) + 1,
        margin=margin_patches,
        min_size=min_crop_patches,
        limit=grid_size,
    )

    crop_h_patches = row_stop - row_start
    crop_w_patches = col_stop - col_start
    crop_area_ratio = float(crop_h_patches * crop_w_patches) / float(grid_size * grid_size)
    is_recommended = crop_area_ratio <= float(max_recommended_area_ratio) and (
        crop_h_patches < grid_size or crop_w_patches < grid_size
    )

    image = _resize_image(Image.open(image_path).convert("RGB"), size=image_size)
    patch = image_size // grid_size
    crop_box = (
        col_start * patch,
        row_start * patch,
        col_stop * patch,
        row_stop * patch,
    )
    crop = image.crop(crop_box)
    crop_w = max(1, crop_box[2] - crop_box[0])
    crop_h = max(1, crop_box[3] - crop_box[1])
    scale = min(float(image_size) / float(crop_w), float(image_size) / float(crop_h))
    resize_w = max(1, int(round(crop_w * scale)))
    resize_h = max(1, int(round(crop_h * scale)))
    resized = crop.resize((resize_w, resize_h), Image.BICUBIC)
    crop_support = support[row_start:row_stop, col_start:col_stop]
    support_mask_small = Image.fromarray((crop_support * 255).astype(np.uint8), mode="L")
    support_mask = support_mask_small.resize((resize_w, resize_h), Image.BILINEAR)
    support_mask_arr = (np.asarray(support_mask, dtype=np.float32) / 255.0)[..., None]
    resized_arr = np.asarray(resized, dtype=np.float32)
    if str(background_mode).lower() == "checkerboard":
        outside_arr = _checkerboard_array(
            resize_h,
            resize_w,
            tile=checkerboard_tile,
            light=checkerboard_light,
            dark=checkerboard_dark,
        )
        canvas_bg = _checkerboard_array(
            image_size,
            image_size,
            tile=checkerboard_tile,
            light=checkerboard_light,
            dark=checkerboard_dark,
        )
        canvas = Image.fromarray(np.clip(canvas_bg, 0.0, 255.0).astype(np.uint8))
    else:
        outside_arr = np.zeros_like(resized_arr)
        outside_arr[...] = np.asarray(background_color, dtype=np.float32)
        canvas = Image.new("RGB", (image_size, image_size), background_color)
    spotlight_arr = resized_arr * support_mask_arr + outside_arr * (1.0 - support_mask_arr)
    spotlight = Image.fromarray(np.clip(spotlight_arr, 0.0, 255.0).astype(np.uint8))
    offset_x = (image_size - resize_w) // 2
    offset_y = (image_size - resize_h) // 2
    canvas.paste(spotlight, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)

    cell_w = float(resize_w) / float(crop_w_patches)
    cell_h = float(resize_h) / float(crop_h_patches)
    outline_width = max(2, int(round(min(cell_w, cell_h) * 0.12)))

    _draw_patch_contours(
        draw,
        support,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        grid_size=grid_size,
        offset_x=offset_x,
        offset_y=offset_y,
        cell_w=cell_w,
        cell_h=cell_h,
        color=outline_color,
        width=outline_width,
    )

    token_visible = False
    token_center_patch = None
    support_rows, support_cols = np.nonzero(crop_support)
    support_centroid = None
    if support_rows.size and support_cols.size:
        support_centroid = (
            float(np.mean(support_rows.astype(np.float32) + 0.5)),
            float(np.mean(support_cols.astype(np.float32) + 0.5)),
        )
    if token_row is not None and token_col is not None:
        if row_start <= token_row < row_stop and col_start <= token_col < col_stop:
            token_visible = True
            local_row = token_row - row_start
            local_col = token_col - col_start
            token_center_patch = (float(local_row) + 0.5, float(local_col) + 0.5)

    support_centroid_offset = None
    support_vertical_relation = None
    support_horizontal_relation = None
    if support_centroid is not None and token_center_patch is not None:
        delta_row = float(support_centroid[0] - token_center_patch[0])
        delta_col = float(support_centroid[1] - token_center_patch[1])
        support_centroid_offset = {
            "delta_row": delta_row,
            "delta_col": delta_col,
        }
        if delta_row > 0.35:
            support_vertical_relation = "below"
        elif delta_row < -0.35:
            support_vertical_relation = "above"
        else:
            support_vertical_relation = "aligned"
        if delta_col > 0.35:
            support_horizontal_relation = "right"
        elif delta_col < -0.35:
            support_horizontal_relation = "left"
        else:
            support_horizontal_relation = "aligned"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "crop_patch_bounds": {
            "row_start": int(row_start),
            "row_stop": int(row_stop),
            "col_start": int(col_start),
            "col_stop": int(col_stop),
        },
        "crop_patch_size": {
            "height": int(crop_h_patches),
            "width": int(crop_w_patches),
        },
        "crop_area_ratio": float(crop_area_ratio),
        "is_recommended": bool(is_recommended),
        "overlay_mode": "solid_background_outside_support_with_outer_contour",
        "support_patch_count": int(support.sum()),
        "token_visible_in_crop": bool(token_visible),
        "token_idx": int(token_idx) if token_idx is not None else None,
        "support_centroid_patch": {
            "row": float(support_centroid[0]),
            "col": float(support_centroid[1]),
        }
        if support_centroid is not None
        else None,
        "support_centroid_offset_from_token": support_centroid_offset,
        "support_vertical_relation": support_vertical_relation,
        "support_horizontal_relation": support_horizontal_relation,
        "core_patch_count": 0,
        "has_distinct_core": False,
    }


def save_support_detail_crop_image(
    image_path: str,
    support_indices: Iterable[int],
    out_path: Path,
    *,
    token_idx: int | None = None,
    image_size: int = 224,
    grid_size: int = 14,
    margin_patches: int = 0,
    min_crop_patches: int = 4,
    background_gray: int = 18,
) -> None:
    support = _support_grid(support_indices, grid_size=grid_size)
    rows, cols = np.nonzero(support)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Empty support_indices")
    if token_idx is not None:
        token_row, token_col = divmod(int(token_idx), grid_size)
        rows = np.concatenate([rows, np.asarray([token_row], dtype=rows.dtype)])
        cols = np.concatenate([cols, np.asarray([token_col], dtype=cols.dtype)])

    row_start, row_stop = _expand_bounds(
        int(rows.min()),
        int(rows.max()) + 1,
        margin=margin_patches,
        min_size=min_crop_patches,
        limit=grid_size,
    )
    col_start, col_stop = _expand_bounds(
        int(cols.min()),
        int(cols.max()) + 1,
        margin=margin_patches,
        min_size=min_crop_patches,
        limit=grid_size,
    )

    image = _resize_image(Image.open(image_path).convert("RGB"), size=image_size)
    patch = image_size // grid_size
    crop_box = (
        col_start * patch,
        row_start * patch,
        col_stop * patch,
        row_stop * patch,
    )
    crop = image.crop(crop_box).convert("L")
    crop_w = max(1, crop_box[2] - crop_box[0])
    crop_h = max(1, crop_box[3] - crop_box[1])
    scale = min(float(image_size) / float(crop_w), float(image_size) / float(crop_h))
    resize_w = max(1, int(round(crop_w * scale)))
    resize_h = max(1, int(round(crop_h * scale)))
    resized = crop.resize((resize_w, resize_h), Image.BICUBIC)

    crop_support = support[row_start:row_stop, col_start:col_stop]
    support_mask_small = Image.fromarray((crop_support * 255).astype(np.uint8), mode="L")
    support_mask = support_mask_small.resize((resize_w, resize_h), Image.BILINEAR)
    support_arr = np.asarray(support_mask, dtype=np.float32) / 255.0

    gray = np.asarray(resized, dtype=np.float32)
    focus = gray[support_arr > 0.5]
    if focus.size < 8:
        focus = gray.reshape(-1)
    lo = float(np.percentile(focus, 2.0))
    hi = float(np.percentile(focus, 98.0))
    if hi <= lo:
        lo = float(focus.min())
        hi = float(focus.max()) if focus.size else 255.0
    scaled = np.clip((gray - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    detail = Image.fromarray(np.clip(scaled * 255.0, 0, 255).astype(np.uint8), mode="L")
    detail = detail.filter(ImageFilter.UnsharpMask(radius=1.0, percent=180, threshold=2))
    detail_arr = np.asarray(detail, dtype=np.float32)
    outside = np.full_like(detail_arr, float(background_gray))
    blended = detail_arr * support_arr + outside * (1.0 - support_arr)
    detail_rgb = np.repeat(np.clip(blended[..., None], 0.0, 255.0), 3, axis=-1).astype(np.uint8)
    canvas = Image.new("RGB", (image_size, image_size), (background_gray, background_gray, background_gray))
    offset_x = (image_size - resize_w) // 2
    offset_y = (image_size - resize_h) // 2
    canvas.paste(Image.fromarray(detail_rgb), (offset_x, offset_y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_erf_heatmap_image(
    image_path: str,
    score_map: Sequence[float],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
) -> None:
    values = np.asarray(score_map, dtype=np.float32).reshape(grid_size, grid_size)
    scaled = _normalize_positive_values(values, lower_percentile=10.0, upper_percentile=99.0, gamma=0.8)

    panel = Image.new("RGB", (image_size, image_size), (10, 16, 22))
    draw = ImageDraw.Draw(panel)
    pad_x = 12
    pad_y = 12
    colorbar_gap = 10
    colorbar_w = 14
    label_gap = 8
    usable_w = image_size - 2 * pad_x - colorbar_gap - colorbar_w - label_gap
    cell_px = max(8, usable_w // grid_size)
    heat_px = cell_px * grid_size
    grid_x = max(pad_x, (image_size - (heat_px + colorbar_gap + colorbar_w + label_gap)) // 2)
    grid_y = max(pad_y, (image_size - heat_px) // 2)
    color_small = np.clip(_cyan_sequential_colormap(scaled) * 255.0, 0, 255).astype(np.uint8)
    heatmap = Image.fromarray(color_small, mode="RGB").resize((heat_px, heat_px), Image.NEAREST)
    panel.paste(heatmap, (grid_x, grid_y))

    grid_line = (36, 64, 74)
    for step in range(grid_size + 1):
        coord = grid_x + step * cell_px
        draw.line((coord, grid_y, coord, grid_y + heat_px), fill=grid_line, width=1)
        coord = grid_y + step * cell_px
        draw.line((grid_x, coord, grid_x + heat_px, coord), fill=grid_line, width=1)

    box = _patch_box(token_idx, grid_size=grid_size, image_size=heat_px)
    token_box = (grid_x + box[0], grid_y + box[1], grid_x + box[2], grid_y + box[3])
    draw.rectangle(token_box, outline=(245, 252, 255), width=3)

    bar_x = grid_x + heat_px + colorbar_gap
    bar_y = grid_y
    bar = np.linspace(1.0, 0.0, num=heat_px, dtype=np.float32)[:, None]
    bar = np.repeat(bar, colorbar_w, axis=1)
    bar_rgb = np.clip(_cyan_sequential_colormap(bar) * 255.0, 0, 255).astype(np.uint8)
    panel.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, bar_y))
    draw.rectangle((bar_x, bar_y, bar_x + colorbar_w, bar_y + heat_px), outline=(92, 128, 140), width=1)
    label_x = bar_x + colorbar_w + 4
    draw.text((label_x, bar_y - 2), "1.0", fill=(210, 235, 242))
    draw.text((label_x, bar_y + heat_px - 10), "0.0", fill=(210, 235, 242))

    result = panel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)


def save_cosine_overlay_image(
    image_path: str,
    cosine_map: Sequence[float],
    out_path: Path,
    *,
    token_idx: int,
    image_size: int = 224,
    grid_size: int = 14,
) -> None:
    image = np.asarray(_resize_image(Image.open(image_path).convert("RGB"), size=image_size), dtype=np.float32) / 255.0
    values = np.asarray(cosine_map, dtype=np.float32).reshape(grid_size, grid_size)
    if values.size == 0:
        raise ValueError("Empty cosine map")
    max_abs = max(float(np.abs(values).max()), 1e-6)
    scaled = np.clip(values / max_abs, -1.0, 1.0)
    red = np.where(scaled >= 0.0, 1.0, 1.0 + scaled)
    green = 1.0 - np.abs(scaled)
    blue = np.where(scaled <= 0.0, 1.0, 1.0 - scaled)
    heat = np.stack([red, green, blue], axis=-1)
    heat_img = Image.fromarray(np.clip(heat * 255.0, 0, 255).astype(np.uint8)).resize(
        (image_size, image_size),
        Image.BILINEAR,
    )
    heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0
    blended = np.clip(0.60 * heat_arr + 0.40 * image, 0.0, 1.0)
    result = Image.fromarray((blended * 255.0).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    draw.rectangle(_patch_box(token_idx, grid_size=grid_size, image_size=image_size), outline=(0, 255, 255), width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
