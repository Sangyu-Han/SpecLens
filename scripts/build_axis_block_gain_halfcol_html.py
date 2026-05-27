#!/usr/bin/env python3
"""Build a half-column HTML/SVG gain figure for Task B Top-1."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs/autolabel_metric_runs/figures/axis_block_taskB_gain_halfcol_html_20260506.html"
BLOCKS = [2, 6, 10]
MODELS = ["CLIP-B/16", "SigLIP-B/16", "DINOv3-S/16"]
SHORT_MODEL = {"CLIP-B/16": "CLIP", "SigLIP-B/16": "SigLIP", "DINOv3-S/16": "DINOv3"}
AXIS_SUMMARY_PATTERNS = {
    "CLIP-B/16": (
        "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/axis_pilot_sessions/"
        "clip50k_metric200pb_randompatch_renderfix_20260429_r{r:02d}_axis_shortdesc/summary.json"
    ),
    "SigLIP-B/16": (
        "outputs/autolabel_metric_runs/siglip_full_workspace/outputs/axis_pilot_sessions/"
        "siglip_metric200pb_randompatch_renderfix_20260429_r{r:02d}_axis_shortdesc/summary.json"
    ),
    "DINOv3-S/16": (
        "outputs/autolabel_metric_runs/dinov3_full_workspace/outputs/axis_pilot_sessions/"
        "dinov3_metric200pb_randompatch_20260429_renderfix292_r{r:02d}_axis_shortdesc/summary.json"
    ),
}


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var) / math.sqrt(len(values))


def load_taskb_top1_gain() -> dict[str, dict[int, tuple[float, float]]]:
    gains: dict[str, dict[int, tuple[float, float]]] = {}
    for model, pattern in AXIS_SUMMARY_PATTERNS.items():
        gains[model] = {}
        for block in BLOCKS:
            repeat_deltas: list[float] = []
            for r in range(1, 11):
                path = REPO / pattern.format(r=r)
                with path.open() as f:
                    summary = json.load(f)
                block_key = str(block)
                erf = summary["axis2"]["erf_cyan_cross"]["per_block"][block_key]["top1_accuracy"]
                al = summary["axis2"]["sae_only"]["per_block"][block_key]["top1_accuracy"]
                repeat_deltas.append(100.0 * (erf - al))
            gains[model][block] = mean_se(repeat_deltas)
    return gains


def render_svg(gains: dict[str, dict[int, tuple[float, float]]]) -> str:
    width = 340
    height = 225
    left = 34
    right = 8
    top = 26
    bottom = 36
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymin = -8.0
    ymax = 24.0
    group_w = plot_w / len(MODELS)
    bar_w = 18
    offsets = {2: -21, 6: 0, 10: 21}
    colors = {2: "#e3b5ae", 6: "#c9625a", 10: "#8f2428"}

    def y_for(value: float) -> float:
        return top + plot_h * (1.0 - (value - ymin) / (ymax - ymin))

    zero_y = y_for(0.0)
    chunks: list[str] = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Task B Top-1 ERF gain">']
    for tick in [-5, 0, 10, 20]:
        y = y_for(tick)
        chunks.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        chunks.append(f'<text x="{left-6}" y="{y+3:.2f}" text-anchor="end" class="tick">{tick}</text>')
    chunks.append(f'<line x1="{left}" x2="{width-right}" y1="{zero_y:.2f}" y2="{zero_y:.2f}" class="zero"/>')
    chunks.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h}" class="axis"/>')

    for model_idx, model in enumerate(MODELS):
        center = left + group_w * (model_idx + 0.5)
        chunks.append(f'<text x="{center:.2f}" y="{height-16}" text-anchor="middle" class="model">{esc(SHORT_MODEL[model])}</text>')
        for block in BLOCKS:
            mean, se = gains[model][block]
            x = center + offsets[block] - bar_w / 2
            if mean >= 0:
                y = y_for(mean)
                h = zero_y - y
            else:
                y = zero_y
                h = y_for(mean) - zero_y
            err_top = y_for(mean + se)
            err_bottom = y_for(mean - se)
            chunks.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" '
                f'rx="1.7" class="bar b{block}"/>'
            )
            chunks.append(f'<line x1="{x+bar_w/2:.2f}" x2="{x+bar_w/2:.2f}" y1="{err_top:.2f}" y2="{err_bottom:.2f}" class="err"/>')
            chunks.append(f'<line x1="{x+bar_w/2-4:.2f}" x2="{x+bar_w/2+4:.2f}" y1="{err_top:.2f}" y2="{err_top:.2f}" class="err"/>')
            chunks.append(f'<line x1="{x+bar_w/2-4:.2f}" x2="{x+bar_w/2+4:.2f}" y1="{err_bottom:.2f}" y2="{err_bottom:.2f}" class="err"/>')
            if block == 10:
                label_y = y_for(mean + se) - 5 if mean >= 0 else y_for(mean - se) + 11
                chunks.append(
                    f'<text x="{x+bar_w/2:.2f}" y="{label_y:.2f}" text-anchor="middle" class="value">'
                    f'+{mean:.1f}</text>'
                )

    chunks.append("</svg>")
    return "\n".join(chunks)


def render_html(gains: dict[str, dict[int, tuple[float, float]]]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Task B Top-1 gain</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; background: #fff; color: #17202b; }}
    body {{
      width: 3.42in;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ width: 3.42in; padding: 0; margin: 0; }}
    .head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      padding-bottom: 4px;
      border-bottom: 1px solid #d7dee7;
      margin-bottom: 2px;
    }}
    h1 {{
      margin: 0;
      font-size: 9.7px;
      line-height: 1.05;
      font-weight: 840;
      color: #111b27;
    }}
    .legend {{
      display: flex;
      gap: 6px;
      align-items: center;
      color: #566272;
      font-size: 7.1px;
      white-space: nowrap;
    }}
    .legend span {{ display: inline-flex; align-items: center; gap: 2px; }}
    .swatch {{ width: 7px; height: 7px; border-radius: 1.5px; display: inline-block; }}
    .b2 {{ fill: #e3b5ae; background: #e3b5ae; }}
    .b6 {{ fill: #c9625a; background: #c9625a; }}
    .b10 {{ fill: #8f2428; background: #8f2428; }}
    .chart {{ display: block; width: 100%; height: auto; overflow: visible; }}
    .grid {{ stroke: #d8dee6; stroke-width: 0.8; opacity: 0.74; }}
    .axis {{ stroke: #9aa5b1; stroke-width: 0.9; }}
    .zero {{ stroke: #28313c; stroke-width: 1; opacity: 0.72; }}
    .tick {{ font-size: 7px; fill: #667384; }}
    .model {{ font-size: 8.2px; fill: #222d3a; font-weight: 760; }}
    .err {{ stroke: #27313e; stroke-width: 0.95; stroke-linecap: round; opacity: 0.78; }}
    .value {{ font-size: 7.1px; fill: #7f1d22; font-weight: 850; }}
    .ylabel {{
      position: absolute;
      left: -2px;
      top: 102px;
      transform: rotate(-90deg);
      transform-origin: left top;
      font-size: 7.2px;
      color: #3c4653;
      font-weight: 680;
    }}
    .wrap {{ position: relative; }}
    .foot {{
      margin: -2px 0 0;
      text-align: right;
      color: #6b7280;
      font-size: 5.9px;
      line-height: 1.15;
    }}
  </style>
</head>
<body>
  <main>
    <header class="head">
      <h1>Task B Top-1 gain (ERF - AL)</h1>
      <div class="legend">
        <span><i class="swatch b2"></i>B2</span>
        <span><i class="swatch b6"></i>B6</span>
        <span><i class="swatch b10"></i>B10</span>
      </div>
    </header>
    <div class="wrap">
      <div class="ylabel">percentage points</div>
      {render_svg(gains)}
    </div>
    <p class="foot">Error bars: paired SE across 10 repeats.</p>
  </main>
</body>
</html>
"""


def crop_with_margin(path: Path) -> None:
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return
    out = ImageOps.expand(im.crop(bbox), border=10, fill=(255, 255, 255))
    out.save(path)


def export_png(html_path: Path) -> Path:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome is None:
        raise RuntimeError("Could not find google-chrome/chromium for PNG export")
    png_path = html_path.with_suffix(".png")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--force-device-scale-factor=3",
        "--window-size=390,285",
        f"--screenshot={png_path}",
        "file://" + str(html_path.resolve()),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    crop_with_margin(png_path)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    gains = load_taskb_top1_gain()
    out.write_text(render_html(gains), encoding="utf-8")
    print(out.relative_to(REPO))
    if not args.no_export:
        png = export_png(out)
        print(png.relative_to(REPO))


if __name__ == "__main__":
    main()
