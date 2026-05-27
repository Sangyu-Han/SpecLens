#!/usr/bin/env python3
"""Build an HTML/SVG absolute Task B Top-1 bar figure from the block table."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "outputs/autolabel_metric_runs/experiments_section_axis_block_raw_data_20260506.md"
DEFAULT_OUT = REPO / "outputs/autolabel_metric_runs/figures/axis_block_taskB_absolute_bar_html_20260506.html"
BLOCKS = [2, 6, 10]
METRIC_NAMES = ["Task A Top-1", "Task B Top-1", "Task B MRR", "Task B R@5"]
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


def parse_value(cell: str) -> tuple[float, float]:
    match = re.search(r"([0-9.]+)\s*\u00b1\s*([0-9.]+)", cell)
    if not match:
        raise ValueError(f"Could not parse mean/error cell: {cell!r}")
    return float(match.group(1)), float(match.group(2))


def parse_markdown(path: Path) -> dict[str, dict[int, dict[str, dict[str, tuple[float, float]]]]]:
    data: dict[str, dict[int, dict[str, dict[str, tuple[float, float]]]]] = {}
    model: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## ") and not line.startswith("## Source"):
            model = line.removeprefix("## ").strip()
            data[model] = {}
            continue
        if not model or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0] in {"Block", "---"}:
            continue
        try:
            block = int(cells[0])
        except ValueError:
            continue
        variant = cells[1]
        data[model].setdefault(block, {})[variant] = {
            metric: parse_value(cell) for metric, cell in zip(METRIC_NAMES, cells[2:], strict=True)
        }
    return data


def load_paired_delta_se() -> dict[str, dict[int, float]]:
    se: dict[str, dict[int, float]] = {}
    for model, pattern in AXIS_SUMMARY_PATTERNS.items():
        se[model] = {}
        for block in BLOCKS:
            diffs: list[float] = []
            for r in range(1, 11):
                path = REPO / pattern.format(r=r)
                with path.open() as f:
                    summary = json.load(f)
                block_key = str(block)
                erf = summary["axis2"]["erf_cyan_cross"]["per_block"][block_key]["top1_accuracy"]
                al = summary["axis2"]["sae_only"]["per_block"][block_key]["top1_accuracy"]
                diffs.append(erf - al)
            mean = sum(diffs) / len(diffs)
            var = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
            se[model][block] = math.sqrt(var) / math.sqrt(len(diffs))
    return se


def panel_svg(model: str, model_data: dict[int, dict[str, dict[str, tuple[float, float]]]], paired_se: dict[int, float]) -> str:
    width = 310
    height = 250
    left = 36
    right = 12
    top = 24
    bottom = 38
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = 0.78
    group_w = plot_w / len(BLOCKS)
    bar_w = 25
    colors = {"AL-based": "#8b96a5", "ERF-based": "#b73535"}

    def x_for(i: int, variant: str) -> float:
        center = left + group_w * (i + 0.5)
        return center - bar_w - 2 if variant == "AL-based" else center + 2

    def y_for(value: float) -> float:
        return top + plot_h * (1.0 - value / ymax)

    chunks: list[str] = [
        f'<svg class="panel-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(model)} Task B Top-1 bar chart">',
        f'<text x="{left}" y="13" class="panel-title">{esc(model)}</text>',
    ]
    for tick in [0.0, 0.2, 0.4, 0.6]:
        y = y_for(tick)
        chunks.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        chunks.append(f'<text x="{left-7}" y="{y+3:.2f}" class="tick" text-anchor="end">{tick:.1f}</text>')
    chunks.append(f'<line x1="{left}" x2="{width-right}" y1="{top+plot_h:.2f}" y2="{top+plot_h:.2f}" class="axis"/>')
    chunks.append(f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+plot_h:.2f}" class="axis"/>')

    for i, block in enumerate(BLOCKS):
        center = left + group_w * (i + 0.5)
        chunks.append(f'<text x="{center:.2f}" y="{height-16}" class="block-label" text-anchor="middle">block {block}</text>')
        al_mean, al_se = model_data[block]["AL-based"]["Task B Top-1"]
        erf_mean, erf_se = model_data[block]["ERF-based"]["Task B Top-1"]
        for variant, mean, se in [("AL-based", al_mean, al_se), ("ERF-based", erf_mean, erf_se)]:
            x = x_for(i, variant)
            y = y_for(mean)
            bar_h = top + plot_h - y
            err_top = y_for(mean + se)
            err_bottom = y_for(max(0.0, mean - se))
            chunks.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{bar_h:.2f}" '
                f'rx="2" class="bar {esc(variant.lower().split("-")[0])}"/>'
            )
            chunks.append(f'<line x1="{x+bar_w/2:.2f}" x2="{x+bar_w/2:.2f}" y1="{err_top:.2f}" y2="{err_bottom:.2f}" class="err"/>')
            chunks.append(f'<line x1="{x+bar_w/2-5:.2f}" x2="{x+bar_w/2+5:.2f}" y1="{err_top:.2f}" y2="{err_top:.2f}" class="err"/>')
            chunks.append(f'<line x1="{x+bar_w/2-5:.2f}" x2="{x+bar_w/2+5:.2f}" y1="{err_bottom:.2f}" y2="{err_bottom:.2f}" class="err"/>')
            chunks.append(f'<text x="{x+bar_w/2:.2f}" y="{y-5:.2f}" class="value" text-anchor="middle">{mean:.2f}</text>')
        delta = erf_mean - al_mean
        delta_y = y_for(max(al_mean, erf_mean) + max(al_se, erf_se) + 0.055)
        delta_x = center
        chunks.append(
            f'<text x="{delta_x:.2f}" y="{delta_y:.2f}" class="delta" text-anchor="middle">'
            f'{delta*100:+.1f} pp</text>'
        )
        # Paired SE is not drawn directly on bars, but expose it in the SVG title for provenance.
        chunks.append(f'<title>block {block} paired delta SE: {paired_se[block]*100:.2f} pp</title>')

    chunks.append("</svg>")
    return "\n".join(chunks)


def render_html(data: dict[str, dict[int, dict[str, dict[str, tuple[float, float]]]]], paired_se: dict[str, dict[int, float]]) -> str:
    panels = "\n".join(
        f'<section class="panel">{panel_svg(model, model_data, paired_se[model])}</section>'
        for model, model_data in data.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Task B Top-1 by block</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; background: #fff; color: #1f2d3a; }}
    body {{
      width: 7.1in;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ width: 7.1in; padding: 0; margin: 0; }}
    .figure-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
      margin: 0 0 7px;
      border-bottom: 1px solid #d5dde6;
      padding-bottom: 5px;
    }}
    h1 {{
      margin: 0;
      font-size: 13px;
      line-height: 1.05;
      font-weight: 820;
      color: #102033;
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 11px;
      color: #526170;
      font-size: 9px;
      white-space: nowrap;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
      display: inline-block;
    }}
    .swatch.al {{ background: #8b96a5; }}
    .swatch.erf {{ background: #b73535; }}
    .panels {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 5px;
    }}
    .panel {{ min-width: 0; }}
    .panel-svg {{
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }}
    .panel-title {{
      font-size: 12px;
      font-weight: 820;
      fill: #102033;
    }}
    .grid {{ stroke: #d8dee6; stroke-width: 1; opacity: 0.78; }}
    .axis {{ stroke: #8c98a5; stroke-width: 1.05; }}
    .tick {{ font-size: 9px; fill: #667384; }}
    .block-label {{ font-size: 9.5px; fill: #303b4a; font-weight: 720; }}
    .bar.al {{ fill: #8b96a5; }}
    .bar.erf {{ fill: #b73535; }}
    .err {{ stroke: #27313e; stroke-width: 1.15; stroke-linecap: round; opacity: 0.75; }}
    .value {{ font-size: 8.4px; fill: #2f3a48; font-weight: 700; }}
    .delta {{ font-size: 8.6px; fill: #8d1f24; font-weight: 850; }}
    .foot {{
      margin-top: 3px;
      color: #6b7280;
      font-size: 7.8px;
      line-height: 1.2;
      text-align: right;
    }}
  </style>
</head>
<body>
  <main>
    <header class="figure-head">
      <h1>Task B Top-1 accuracy by block</h1>
      <div class="legend">
        <span><i class="swatch al"></i> AL-based</span>
        <span><i class="swatch erf"></i> ERF-based</span>
      </div>
    </header>
    <section class="panels">
      {panels}
    </section>
    <p class="foot">Bars show mean over 10 repeats. Error bars show reported SE for each condition. Labels above block groups show ERF-based minus AL-based gain.</p>
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
    out = ImageOps.expand(im.crop(bbox), border=14, fill=(255, 255, 255))
    out.save(path)


def export_png(html_path: Path, width: int, height: int) -> Path:
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
        f"--window-size={width},{height}",
        f"--screenshot={png_path}",
        "file://" + str(html_path.resolve()),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    crop_with_margin(png_path)
    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else REPO / args.source
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    data = parse_markdown(source)
    paired_se = load_paired_delta_se()
    out.write_text(render_html(data, paired_se), encoding="utf-8")
    print(out.relative_to(REPO))
    if not args.no_export:
        print(export_png(out, width=820, height=330).relative_to(REPO))


if __name__ == "__main__":
    main()
