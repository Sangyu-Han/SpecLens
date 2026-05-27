#!/usr/bin/env python3
"""Build appendix qualitative HTML/PNG pages from autolabel review artifacts.

The generated pages reuse the compact two-column evidence design used for the
main qualitative figure. Each row is one feature; the left block is activation
based autolabeling and the right block is ERF-based autolabeling.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageChops, ImageOps


REPO = Path(__file__).resolve().parents[1]
FIGURE_ROOT = REPO / "outputs/autolabel_metric_runs/figures/appendix_qualitative"
CSS_WIDTH_PX = 672
DEVICE_SCALE = 3
SAFETY_MARGIN_PX = 14


@dataclass(frozen=True)
class CategoryConfig:
    slug: str
    display_name: str
    row_title: str
    model_label: str
    sae_raw: Path
    erf_raw: Path
    block_filter: int | None = 10


@dataclass(frozen=True)
class LabelResult:
    label: str
    description: str


@dataclass(frozen=True)
class FigureRow:
    title: str
    key: str
    sae_label: LabelResult
    erf_label: LabelResult
    sae_images: list[Path]
    erf_images: list[Path]


def p(rel: str) -> Path:
    return REPO / rel


RANDOM_PATCH_SOURCES = {
    "clip": {
        "model_label": "CLIP-B/16",
        "display_label": "CLIP-B/16",
        "sae_raw": p(
            "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions/"
            "clip50k_metric200pb_randompatch_renderfix_20260429_r01_sae_shortdesc/raw_predictions.json"
        ),
        "erf_raw": p(
            "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions/"
            "clip50k_metric200pb_randompatch_renderfix_20260429_r01_erfcross_shortdesc/raw_predictions.json"
        ),
    },
    "siglip": {
        "model_label": "SigLIP-B/16",
        "display_label": "SigLIP-B/16",
        "sae_raw": p(
            "outputs/autolabel_metric_runs/siglip_full_workspace/outputs/review_sessions/"
            "siglip_metric200pb_randompatch_renderfix_20260429_r01_sae_shortdesc/raw_predictions.json"
        ),
        "erf_raw": p(
            "outputs/autolabel_metric_runs/siglip_full_workspace/outputs/review_sessions/"
            "siglip_metric200pb_randompatch_renderfix_20260429_r01_erfcross_shortdesc/raw_predictions.json"
        ),
    },
    "dinov3": {
        "model_label": "DINOv3-S/16",
        "display_label": "DINOv3-S/16",
        "sae_raw": p(
            "outputs/autolabel_metric_runs/dinov3_full_workspace/outputs/review_sessions/"
            "dinov3_metric200pb_randompatch_20260429_renderfix292_r01_sae_shortdesc/raw_predictions.json"
        ),
        "erf_raw": p(
            "outputs/autolabel_metric_runs/dinov3_full_workspace/outputs/review_sessions/"
            "dinov3_metric200pb_randompatch_20260429_renderfix292_r01_erfcross_shortdesc/raw_predictions.json"
        ),
    },
}


def _make_random_patch_categories() -> dict[str, CategoryConfig]:
    categories: dict[str, CategoryConfig] = {}
    for pack in ("clip", "siglip", "dinov3"):
        source = RANDOM_PATCH_SOURCES[pack]
        for block in (2, 6, 10):
            slug = f"{pack}_block{block}_random_patch"
            categories[slug] = CategoryConfig(
                slug=slug,
                display_name=f"{source['display_label']} block {block} random sparse patch features",
                row_title="Random sparse patch feature",
                model_label=str(source["model_label"]),
                sae_raw=source["sae_raw"],
                erf_raw=source["erf_raw"],
                block_filter=block,
            )
    return categories


CATEGORIES: dict[str, CategoryConfig] = {
    **_make_random_patch_categories(),
    "clip_cls_token": CategoryConfig(
        slug="clip_cls_token",
        display_name="CLIP CLS-token features",
        row_title="CLS-token feature",
        model_label="CLIP-B/16",
        sae_raw=p(
            "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions/"
            "clip50k_cls_token_label_block10_100_renderfix_20260501_v2/"
            "random20_seed42_sae_nonlocal_style/raw_predictions.json"
        ),
        erf_raw=p(
            "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions/"
            "clip50k_cls_token_label_block10_100_renderfix_20260501_v2/"
            "random20_seed42_erf_nonlocal_style/raw_predictions.json"
        ),
    ),
    "clip_nonlocal_sparse": CategoryConfig(
        slug="clip_nonlocal_sparse",
        display_name="CLIP non-local sparse-firing features",
        row_title="Non-local sparse-firing feature",
        model_label="CLIP-B/16",
        sae_raw=p(
            "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions/"
            "clip50k_block10_nonlocal_random20_seed42_sae_shortdesc/raw_predictions.json"
        ),
        erf_raw=p(
            "outputs/autolabel_metric_runs/clip50k_full_workspace/outputs/review_sessions/"
            "clip50k_block10_nonlocal_random20_seed42_erfcross_shortdesc/raw_predictions.json"
        ),
    ),
}


CURATED_PAPER_FEATURES: dict[str, list[str]] = {
    "clip_block6_random_patch": [
        "block_6/feature_11347",
        "block_6/feature_11453",
        "block_6/feature_11692",
        "block_6/feature_11818",
        "block_6/feature_349",
    ],
    "siglip_block2_random_patch": [
        "block_2/feature_10120",
        "block_2/feature_10308",
        "block_2/feature_11509",
        "block_2/feature_1388",
        "block_2/feature_1922",
    ],
    "siglip_block6_random_patch": [
        "block_6/feature_10027",
        "block_6/feature_11164",
        "block_6/feature_11309",
        "block_6/feature_1803",
        "block_6/feature_2340",
    ],
    "siglip_block10_random_patch": [
        "block_10/feature_11137",
        "block_10/feature_11352",
        "block_10/feature_11756",
        "block_10/feature_336",
        "block_10/feature_3315",
    ],
    "dinov3_block6_random_patch": [
        "block_6/feature_1383",
        "block_6/feature_1614",
        "block_6/feature_1673",
        "block_6/feature_1826",
        "block_6/feature_2405",
    ],
}


def load_features(raw_path: Path) -> list[dict]:
    with raw_path.open() as f:
        data = json.load(f)
    features = data.get("features", data)
    if not isinstance(features, list):
        raise ValueError(f"{raw_path} does not contain a feature list")
    return features


def parse_label(feature: dict) -> LabelResult:
    output = feature.get("output") or feature.get("sae_label") or feature.get("erf_label") or {}
    return LabelResult(
        label=str(output.get("canonical_label", "")).strip() or "(missing label)",
        description=str(output.get("description", "")).strip() or "(missing description)",
    )


def resolve_review_images(raw_path: Path, feature: dict) -> list[Path]:
    images: list[Path] = []
    for example in feature.get("label_examples", [])[:5]:
        raw_image = example.get("review_image")
        if not raw_image:
            continue
        image_path = Path(raw_image)
        if not image_path.is_absolute():
            image_path = raw_path.parent / image_path
        images.append(image_path.resolve())
    return images


def block_idx(feature_key: str) -> int | None:
    prefix = "block_"
    if not feature_key.startswith(prefix):
        return None
    try:
        return int(feature_key[len(prefix) :].split("/", 1)[0])
    except ValueError:
        return None


def compact_display_name(config: CategoryConfig) -> str:
    if config.slug.endswith("_random_patch") and config.block_filter is not None:
        return f"block {config.block_filter} SAE features"
    if config.slug == "clip_cls_token":
        return "CLS-token SAE features"
    if config.slug == "clip_nonlocal_sparse":
        return "non-local sparse-firing SAE features"
    return config.display_name


def index_display_name(config: CategoryConfig, compact: bool) -> str:
    if not compact:
        return config.display_name
    if config.slug.endswith("_random_patch") and config.block_filter is not None:
        return f"{config.model_label} block {config.block_filter} SAE features"
    if config.slug == "clip_cls_token":
        return "CLIP-B/16 CLS-token SAE features"
    if config.slug == "clip_nonlocal_sparse":
        return "CLIP-B/16 non-local sparse-firing SAE features"
    return config.display_name


def build_rows(config: CategoryConfig, limit: int, curated_keys: list[str] | None = None) -> list[FigureRow]:
    sae_features = load_features(config.sae_raw)
    erf_features = {f["feature_key"]: f for f in load_features(config.erf_raw)}
    rows: list[FigureRow] = []
    for sae_feature in sae_features:
        feature_key = sae_feature["feature_key"]
        if config.block_filter is not None and block_idx(feature_key) != config.block_filter:
            continue
        erf_feature = erf_features.get(feature_key)
        if erf_feature is None:
            continue
        sae_images = resolve_review_images(config.sae_raw, sae_feature)
        erf_images = resolve_review_images(config.erf_raw, erf_feature)
        if len(sae_images) < 5 or len(erf_images) < 5:
            continue
        missing = [path for path in sae_images[:5] + erf_images[:5] if not path.exists()]
        if missing:
            print(f"warning: skipping {feature_key}; missing {missing[0]}")
            continue
        rows.append(
            FigureRow(
                title=config.row_title,
                key=f"{config.model_label} {feature_key}",
                sae_label=parse_label(sae_feature),
                erf_label=parse_label(erf_feature),
                sae_images=sae_images[:5],
                erf_images=erf_images[:5],
            )
        )
        if curated_keys is None and len(rows) >= limit:
            break

    if curated_keys:
        by_key = {row.key.split(" ", 1)[1]: row for row in rows}
        selected: list[FigureRow] = []
        missing: list[str] = []
        for feature_key in curated_keys:
            row = by_key.get(feature_key)
            if row is None:
                missing.append(feature_key)
                continue
            selected.append(row)
        if missing:
            print(f"warning: {config.slug} missing curated features: {', '.join(missing)}")
        selected_keys = {row.key for row in selected}
        for row in rows:
            if len(selected) >= limit:
                break
            if row.key not in selected_keys:
                selected.append(row)
        rows = selected

    return rows


def rel_image(path: Path, html_path: Path) -> str:
    return os.path.relpath(path, html_path.parent)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_tile_set(images: Iterable[Path], html_path: Path, alt_prefix: str) -> str:
    chunks: list[str] = []
    for idx, image_path in enumerate(images, start=1):
        src = esc(rel_image(image_path, html_path))
        chunks.append(
            f'<div class="tile"><span class="idx">{idx}</span>'
            f'<img src="{src}" alt="{esc(alt_prefix)} {idx}"></div>'
        )
    return "\n".join(chunks)


def render_block(
    *,
    mode_class: str,
    heading: str,
    count: int,
    images: list[Path],
    label: LabelResult,
    html_path: Path,
    alt_prefix: str,
) -> str:
    return f"""
        <section class="evidence-block {mode_class}">
          <div class="block-top">
            <h4>{esc(heading)}</h4>
            <span>{count} images</span>
          </div>
          <div class="strip">
            {render_tile_set(images, html_path, alt_prefix)}
          </div>
          <div class="label-box">
            <div class="label-line"><span>Label</span> <strong>{esc(label.label)}</strong></div>
            <p>{esc(label.description)}</p>
          </div>
        </section>"""


def render_row(row: FigureRow, html_path: Path, show_row_title: bool) -> str:
    if show_row_title:
        feature_head = f"""
        <div class="feature-head">
          <h3 class="feature-title">{esc(row.title)}</h3>
          <div class="feature-key">{esc(row.key)}</div>
        </div>"""
    else:
        feature_head = f"""
        <div class="feature-head compact">
          <div class="feature-key only-key">{esc(row.key)}</div>
        </div>"""
    return f"""
      <section class="feature">
        {feature_head}
        <div class="blocks">
          {render_block(
              mode_class="sae",
              heading="AL-based autolabeling",
              count=len(row.sae_images),
              images=row.sae_images,
              label=row.sae_label,
              html_path=html_path,
              alt_prefix=f"{row.key} AL-based autolabeling",
          )}
          {render_block(
              mode_class="erf",
              heading="ERF-based autolabeling",
              count=len(row.erf_images),
              images=row.erf_images,
              label=row.erf_label,
              html_path=html_path,
              alt_prefix=f"{row.key} ERF-based autolabeling",
          )}
        </div>
      </section>"""


def render_html(
    config: CategoryConfig,
    rows: list[FigureRow],
    html_path: Path,
    page_idx: int,
    page_count: int,
    *,
    show_page_count: bool,
    show_row_title: bool,
    compact_title: bool,
) -> str:
    display_name = compact_display_name(config) if compact_title else config.display_name
    row_html = "\n".join(render_row(row, html_path, show_row_title) for row in rows)
    page_suffix = f" page {page_idx}" if show_page_count else ""
    page_count_html = f"<span>page {page_idx}/{page_count}</span>" if show_page_count else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(display_name)}{esc(page_suffix)}</title>
  <style>
    :root {{
      --ink: #1f2d3a;
      --muted: #607086;
      --line: #d1d9e2;
      --sae: #93651d;
      --erf: #058299;
      --sae-bg: #fff9ee;
      --erf-bg: #effdff;
      --mask: #7c796e;
    }}

    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; background: #fff; color: var(--ink); }}
    body {{
      width: 7in;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .page {{ width: 7in; margin: 0; padding: 0; background: #fff; }}
    .figure {{ width: 7in; margin: 0; padding: 0; background: #fff; }}
    .category-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      padding: 0 0 5px;
      border-bottom: 1px solid var(--line);
      margin-bottom: 5px;
    }}
    .category-head h2 {{
      margin: 0;
      font-size: 10.5px;
      line-height: 1.1;
      font-weight: 780;
      color: var(--ink);
    }}
    .category-head span {{
      color: var(--muted);
      font-size: 7.7px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: nowrap;
    }}
    .feature {{
      padding: 0 0 6px;
      margin: 0 0 6px;
      border-bottom: 1px solid var(--line);
    }}
    .feature:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: 0; }}
    .feature-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 4px;
    }}
    .feature-head.compact {{
      justify-content: flex-end;
      margin-bottom: 2px;
    }}
    .feature-title {{
      margin: 0;
      font-size: 10px;
      font-weight: 780;
      color: var(--ink);
    }}
    .feature-key {{
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 8.2px;
      white-space: nowrap;
    }}
    .feature-key.only-key {{
      margin-left: auto;
    }}
    .blocks {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 5px;
    }}
    .evidence-block {{
      min-width: 0;
      border: 1px solid var(--line);
      border-top-width: 3px;
      background: #fff;
    }}
    .evidence-block.sae {{ border-top-color: var(--sae); }}
    .evidence-block.erf {{ border-top-color: var(--erf); }}
    .block-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      padding: 5px 6px 4px;
      background: #fff;
    }}
    .block-top h4 {{
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0;
      font-size: 9.8px;
      line-height: 1.05;
      font-weight: 820;
    }}
    .sae .block-top h4 {{ color: var(--sae); }}
    .erf .block-top h4 {{ color: var(--erf); }}
    .block-top span {{
      color: var(--muted);
      font-size: 8px;
      white-space: nowrap;
    }}
    .strip {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 0;
      padding: 0 3px 3px;
      background: #fff;
    }}
    .tile {{
      position: relative;
      height: 65px;
      overflow: hidden;
      background: var(--mask);
    }}
    .tile img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}
    .idx {{
      position: absolute;
      top: 1px;
      left: 1px;
      z-index: 2;
      min-width: 11px;
      height: 11px;
      padding: 0 3px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.93);
      color: var(--ink);
      font-size: 7px;
      line-height: 11px;
      text-align: center;
      font-weight: 800;
    }}
    .label-box {{
      border-top: 1px solid var(--line);
      padding: 4px 6px 4px;
      min-height: 66px;
    }}
    .sae .label-box {{ background: var(--sae-bg); }}
    .erf .label-box {{ background: var(--erf-bg); }}
    .label-line {{
      display: flex;
      align-items: baseline;
      gap: 5px;
      margin: 0 0 3px;
      min-width: 0;
    }}
    .label-line span {{
      color: var(--muted);
      font-size: 7.7px;
      text-transform: uppercase;
      font-weight: 800;
      letter-spacing: 0;
    }}
    .label-line strong {{
      color: #102033;
      font-size: 10.2px;
      line-height: 1.05;
      font-weight: 850;
    }}
    .label-box p {{
      margin: 0;
      color: #2b3a4a;
      font-size: 8.7px;
      line-height: 1.08;
    }}
  </style>
</head>
<body>
  <main class="page">
    <figure class="figure">
      <header class="category-head">
        <h2>{esc(display_name)}</h2>
        {page_count_html}
      </header>
      {row_html}
    </figure>
  </main>
</body>
</html>
"""


def write_pages(
    config: CategoryConfig,
    rows: list[FigureRow],
    rows_per_page: int,
    out_dir: Path,
    *,
    single_page: bool,
    show_page_count: bool,
    show_row_title: bool,
    compact_title: bool,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_paths: list[Path] = []
    pages = [rows] if single_page else [rows[i : i + rows_per_page] for i in range(0, len(rows), rows_per_page)]
    for idx, page_rows in enumerate(pages, start=1):
        html_path = out_dir / (f"{config.slug}.html" if single_page else f"{config.slug}_page{idx:02d}.html")
        html_path.write_text(
            render_html(
                config,
                page_rows,
                html_path,
                idx,
                len(pages),
                show_page_count=show_page_count,
                show_row_title=show_row_title,
                compact_title=compact_title,
            ),
            encoding="utf-8",
        )
        html_paths.append(html_path)
    return html_paths


def crop_with_margin(path: Path) -> None:
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return
    cropped = im.crop(bbox)
    out = ImageOps.expand(cropped, border=SAFETY_MARGIN_PX, fill=(255, 255, 255))
    out.save(path)


def export_pngs(html_paths: list[Path], window_height: int) -> list[Path]:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome is None:
        raise RuntimeError("Could not find google-chrome/chromium for PNG export")
    png_paths: list[Path] = []
    for html_path in html_paths:
        png_path = html_path.with_suffix(".png")
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--force-device-scale-factor={DEVICE_SCALE}",
            f"--window-size={CSS_WIDTH_PX},{window_height}",
            f"--screenshot={png_path}",
            "file://" + str(html_path.resolve()),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        crop_with_margin(png_path)
        png_paths.append(png_path)
    return png_paths


def write_index(manifest: dict[str, dict], out_dir: Path, index_name: str) -> Path:
    rows: list[str] = []
    for slug, info in manifest.items():
        html_links = []
        for idx, rel_path in enumerate(info["html"], start=1):
            target = os.path.relpath(REPO / rel_path, out_dir)
            html_links.append(f'<a href="{esc(target)}">page {idx}</a>')
        png_links = []
        for idx, rel_path in enumerate(info["png"], start=1):
            target = os.path.relpath(REPO / rel_path, out_dir)
            png_links.append(f'<a href="{esc(target)}">png {idx}</a>')
        features = info["features"]
        rows.append(
            f"""
      <section class="category">
        <div class="category-main">
          <h2>{esc(info["display_name"])}</h2>
          <p>{len(features)} features, {len(info["html"])} HTML pages</p>
          <p class="feature-list">{esc(", ".join(features[:3]))}{' ...' if len(features) > 3 else ''}</p>
        </div>
        <div class="links">
          <div>{' '.join(html_links)}</div>
          <div class="png-links">{' '.join(png_links)}</div>
        </div>
      </section>"""
        )

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Appendix qualitative labeling panels</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px;
      background: #f5f7fa;
      color: #1f2d3a;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; line-height: 1.1; }}
    .subtitle {{ margin: 0 0 20px; color: #607086; font-size: 14px; }}
    .category {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(260px, 0.58fr);
      gap: 18px;
      align-items: start;
      padding: 14px 16px;
      margin: 0 0 10px;
      background: #fff;
      border: 1px solid #d1d9e2;
      border-radius: 8px;
    }}
    .category h2 {{ margin: 0 0 4px; font-size: 17px; line-height: 1.15; }}
    .category p {{ margin: 0; color: #607086; font-size: 13px; line-height: 1.35; }}
    .feature-list {{
      margin-top: 6px !important;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px !important;
    }}
    .links {{
      display: grid;
      gap: 7px;
      font-size: 13px;
      line-height: 1.6;
    }}
    a {{
      display: inline-block;
      margin: 0 5px 5px 0;
      padding: 2px 7px;
      border: 1px solid #c6d2de;
      border-radius: 999px;
      color: #0a7184;
      text-decoration: none;
      background: #f4feff;
    }}
    .png-links:empty {{ display: none; }}
    @media (max-width: 760px) {{
      body {{ padding: 14px; }}
      .category {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Appendix qualitative labeling panels</h1>
    <p class="subtitle">Each page compares AL-based and ERF-based autolabeling for five evidence images per feature.</p>
    {''.join(rows)}
  </main>
</body>
</html>
"""
    index_path = out_dir / index_name
    index_path.write_text(index_html, encoding="utf-8")
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        action="append",
        choices=sorted(CATEGORIES),
        help="Category slug to build. Can be repeated. Defaults to all categories.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of features per category.")
    parser.add_argument("--rows-per-page", type=int, default=2, help="Feature rows per PNG/HTML page.")
    parser.add_argument("--out-dir", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--window-height", type=int, default=1600, help="Chrome screenshot viewport height in CSS pixels.")
    parser.add_argument("--no-export", action="store_true", help="Only write HTML pages; skip PNG export.")
    parser.add_argument("--index-name", default="index.html", help="HTML index filename written under --out-dir.")
    parser.add_argument("--single-page", action="store_true", help="Write one continuous HTML file per category.")
    parser.add_argument("--hide-page-count", action="store_true", help="Do not render page x/y in the category header.")
    parser.add_argument("--hide-row-title", action="store_true", help="Do not render repeated per-feature type headings.")
    parser.add_argument("--compact-title", action="store_true", help="Use compact visual titles such as 'block 2 SAE features'.")
    parser.add_argument(
        "--curated-paper-samples",
        action="store_true",
        help="Use hand-picked appendix feature selections for categories with curated overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    args.out_dir = args.out_dir.resolve()
    categories = args.category or list(CATEGORIES)
    manifest: dict[str, dict] = {}
    for slug in categories:
        config = CATEGORIES[slug]
        curated_keys = CURATED_PAPER_FEATURES.get(slug) if args.curated_paper_samples else None
        rows = build_rows(config, args.limit, curated_keys=curated_keys)
        if not rows:
            print(f"{slug}: no rows")
            continue
        out_dir = args.out_dir / slug
        html_paths = write_pages(
            config,
            rows,
            args.rows_per_page,
            out_dir,
            single_page=args.single_page,
            show_page_count=not args.hide_page_count,
            show_row_title=not args.hide_row_title,
            compact_title=args.compact_title,
        )
        png_paths = [] if args.no_export else export_pngs(html_paths, args.window_height)
        display_name = index_display_name(config, args.compact_title)
        manifest[slug] = {
            "display_name": display_name,
            "features": [row.key for row in rows],
            "html": [str(path.relative_to(REPO)) for path in html_paths],
            "png": [str(path.relative_to(REPO)) for path in png_paths],
        }
        print(f"{slug}: {len(rows)} features, {len(html_paths)} pages")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(REPO)}")
    index_path = write_index(manifest, args.out_dir, args.index_name)
    print(f"wrote {index_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
