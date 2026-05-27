#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from autolabel_eval.config import EvalConfig
from autolabel_eval.legacy import LegacyRuntime
from autolabel_eval.rendering import CLIP_ZERO_RGB, save_model_input_image, save_support_mask_image


@dataclass(frozen=True)
class ExampleSpec:
    block_idx: int
    feature_id: int
    rank: int
    sample_id: int
    token_idx: int
    image_path: Path
    slug: str

    @property
    def feature_key(self) -> str:
        return f"block_{self.block_idx}/feature_{self.feature_id}"


EXAMPLES = (
    ExampleSpec(
        block_idx=10,
        feature_id=11481,
        rank=0,
        sample_id=19406,
        token_idx=61,
        image_path=Path("/data/datasets/imagenet/val/n02510455/ILSVRC2012_val_00007107.JPEG"),
        slug="clip_b10_f11481_top1_panda_eye",
    ),
    ExampleSpec(
        block_idx=10,
        feature_id=11389,
        rank=0,
        sample_id=17596,
        token_idx=56,
        image_path=Path("/data/datasets/imagenet/val/n02422106/ILSVRC2012_val_00047424.JPEG"),
        slug="clip_b10_f11389_top1",
    ),
    ExampleSpec(
        block_idx=10,
        feature_id=11389,
        rank=1,
        sample_id=17569,
        token_idx=73,
        image_path=Path("/data/datasets/imagenet/val/n02422106/ILSVRC2012_val_00017389.JPEG"),
        slug="clip_b10_f11389_top2",
    ),
)

QUANTILES = (50, 60, 70, 75, 80)
OUT_DIR = ROOT / "outputs/autolabel_metric_runs/figures/al_threshold_sweep_20260507"


def _build_runtime() -> LegacyRuntime:
    config = replace(
        EvalConfig(),
        workspace_root=ROOT / "outputs/autolabel_metric_runs/clip50k_full_workspace",
        model_name="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        image_size=224,
        resize_size=256,
        grid_size=14,
        n_patches=196,
        device="cuda:0",
    )
    return LegacyRuntime(config)


def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def _feature_top_scores(runtime: LegacyRuntime, spec: ExampleSpec, *, n: int = 100) -> np.ndarray:
    frame = runtime.load_decile_frame(spec.block_idx)
    rows = frame[frame["unit"].astype(int) == int(spec.feature_id)].sort_values("score", ascending=False)
    scores: list[float] = []
    for row in rows.itertuples(index=False):
        token_idx = runtime.row_x_to_token_idx(int(row.x))
        if 0 <= token_idx < int(runtime.config.n_patches):
            score = float(row.score)
            if score > 0:
                scores.append(score)
        if len(scores) >= int(n):
            break
    if len(scores) < 1:
        raise RuntimeError(f"No positive patch-token scores for {spec.feature_key}")
    return np.asarray(scores, dtype=np.float32)


def _support_from_threshold(values: np.ndarray, tau: float) -> list[int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return [int(idx) for idx in np.flatnonzero(values >= float(tau))]


def _label_panel(panel: Image.Image, title: str, subtitle: str) -> Image.Image:
    header_h = 46
    out = Image.new("RGB", (panel.width, panel.height + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    title_font = _load_font(13, bold=True)
    sub_font = _load_font(10, bold=False)
    draw.text((8, 5), title, fill=(28, 28, 28), font=title_font)
    draw.text((8, 25), subtitle, fill=(90, 90, 90), font=sub_font)
    out.paste(panel, (0, header_h))
    return out


def _make_contact_sheet(panels: list[Image.Image], out_path: Path, *, gap: int = 6) -> None:
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _make_all_contact_sheet(example_sheets: list[Image.Image], out_path: Path, *, gap: int = 10) -> None:
    width = max(sheet.width for sheet in example_sheets)
    height = sum(sheet.height for sheet in example_sheets) + gap * (len(example_sheets) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for sheet in example_sheets:
        canvas.paste(sheet, (0, y))
        y += sheet.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime = _build_runtime()
    manifest: dict[str, Any] = {
        "output_dir": str(OUT_DIR),
        "quantiles": list(QUANTILES),
        "examples": [],
    }
    sheets: list[Image.Image] = []
    try:
        for spec in EXAMPLES:
            example_dir = OUT_DIR / spec.slug
            example_dir.mkdir(parents=True, exist_ok=True)
            values = runtime.feature_activation_map(str(spec.image_path), spec.block_idx, spec.feature_id)
            scores = _feature_top_scores(runtime, spec, n=100)

            panel_paths: list[dict[str, Any]] = []
            original_path = example_dir / "original_model_input.png"
            save_model_input_image(
                str(spec.image_path),
                original_path,
                image_size=224,
                resize_size=256,
            )
            original = _label_panel(Image.open(original_path).convert("RGB"), "Original", spec.feature_key)
            panels = [original]

            conditions: list[tuple[str, float]] = [("SAE > 0", 1e-12)]
            for q in QUANTILES:
                conditions.append((f"Q{q}", float(np.percentile(scores, float(q)))))

            for label, tau in conditions:
                support = _support_from_threshold(values, tau)
                out_path = example_dir / f"{label.lower().replace(' ', '_').replace('>', 'gt')}.png"
                save_support_mask_image(
                    str(spec.image_path),
                    support,
                    out_path,
                    token_idx=spec.token_idx,
                    image_size=224,
                    grid_size=14,
                    resize_size=256,
                    mode="masked_black",
                    background_color=CLIP_ZERO_RGB,
                    include_token_box=False,
                )
                panel = _label_panel(
                    Image.open(out_path).convert("RGB"),
                    label,
                    f"{len(support)} patches | tau={tau:.3g}",
                )
                panels.append(panel)
                panel_paths.append(
                    {
                        "condition": label,
                        "tau": float(tau),
                        "visible_patch_count": int(len(support)),
                        "path": str(out_path),
                    }
                )

            sheet_path = example_dir / "threshold_sweep_contact_sheet.png"
            _make_contact_sheet(panels, sheet_path)
            sheets.append(Image.open(sheet_path).convert("RGB"))
            manifest["examples"].append(
                {
                    "feature_key": spec.feature_key,
                    "rank": int(spec.rank),
                    "sample_id": int(spec.sample_id),
                    "token_idx": int(spec.token_idx),
                    "image_path": str(spec.image_path),
                    "top100_score_min": float(np.min(scores)),
                    "top100_score_max": float(np.max(scores)),
                    "top100_score_count": int(scores.size),
                    "original": str(original_path),
                    "contact_sheet": str(sheet_path),
                    "panels": panel_paths,
                }
            )
            print(sheet_path, flush=True)
    finally:
        runtime.close()

    combined_path = OUT_DIR / "al_threshold_sweep_contact_sheet.png"
    _make_all_contact_sheet(sheets, combined_path)
    manifest["combined_contact_sheet"] = str(combined_path)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(combined_path, flush=True)


if __name__ == "__main__":
    main()
