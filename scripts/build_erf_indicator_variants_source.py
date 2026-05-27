from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autolabel_eval.rendering import (  # noqa: E402
    CLIP_ZERO_RGB,
    save_support_locator_grid_image,
    save_support_mask_image,
    save_support_overview_zoom_image,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_preview_html(session_dir: Path, features: list[dict[str, Any]]) -> None:
    parts = [
        "<!doctype html>",
        "<html><body style='font-family:sans-serif;background:#f6f2ea;color:#231f1a'>",
        "<h1>ERF Indicator Variant Source Preview</h1>",
    ]
    for feature in features[:8]:
        feature_key = str(feature["feature_key"])
        parts.append(f"<section style='background:#fff;border:1px solid #ddd;padding:16px;margin:16px 0;border-radius:12px'>")
        parts.append(f"<h2 style='font-family:monospace'>{feature_key}</h2>")
        for example in sorted(feature.get("label_examples", []), key=lambda row: int(row["rank"])):
            rank = int(example["rank"])
            token_idx = int(example["token_idx"])
            cards = []
            for title, key in (
                ("plain", "feature_erf_plain"),
                ("locator", "feature_erf_locator_grid"),
                ("cyan dot", "feature_erf_cyan_dot"),
                ("cyan cross", "feature_erf_cyan_cross"),
                ("cyan dashed box", "feature_erf_cyan_dashed_box"),
                ("box", "feature_erf_patch_box"),
                ("zoom", "feature_erf_overview_zoom"),
            ):
                rel = html_escape(str(example[key]))
                cards.append(
                    f"<div><div>{title}</div><img src='{rel}' width='320' style='border:1px solid #ccc;background:#eee'></div>"
                )
            parts.append(
                f"<div style='margin-bottom:16px'><div>example_{rank:02d} | tok={token_idx}</div>"
                f"<div style='display:flex;gap:16px;flex-wrap:wrap'>{''.join(cards)}</div></div>"
            )
        parts.append("</section>")
    parts.append("</body></html>")
    (session_dir / "variant_source_preview.html").write_text("\n".join(parts))


def html_escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _infer_session_geometry(source_manifest: dict[str, Any], source_session_dir: Path) -> tuple[int, int, int]:
    config = dict(source_manifest.get("config") or {})
    image_size = int(config.get("image_size") or 0)
    grid_size = int(config.get("grid_size") or 0)
    resize_size = int(config.get("resize_size") or 0)
    if image_size > 0 and grid_size > 0:
        if resize_size <= 0:
            resize_size = int(image_size * 256 / 224)
        return image_size, grid_size, resize_size

    session_hint = f"{source_manifest.get('session_name') or ''} {source_session_dir}".lower()
    if "dinov3" in session_hint:
        return 256, 16, int(256 * 256 / 224)
    if "siglip" in session_hint or "clip" in session_hint:
        return 224, 14, 256

    max_idx = -1
    for feature in list(source_manifest.get("features") or []):
        for example in list(feature.get("label_examples") or []):
            max_idx = max(max_idx, int(example.get("token_idx", -1)))
    if max_idx < 0:
        raise ValueError("Could not infer session geometry from source manifest")
    n_patches = int(max_idx) + 1
    grid_size = int(round(n_patches ** 0.5))
    if grid_size * grid_size != n_patches:
        raise ValueError(f"Could not infer square patch grid from max token idx {max_idx}")
    image_size = int(grid_size * 16)
    resize_size = int(image_size * 256 / 224)
    return image_size, grid_size, resize_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-session-dir", required=True)
    parser.add_argument("--out-session-dir", required=True)
    args = parser.parse_args()

    source_session_dir = Path(args.source_session_dir).resolve()
    out_session_dir = Path(args.out_session_dir).resolve()
    out_session_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((source_session_dir / "selection_manifest.json").read_text())
    image_size, grid_size, resize_size = _infer_session_geometry(source_manifest, source_session_dir)
    out_features: list[dict[str, Any]] = []

    for feature in source_manifest["features"]:
        feature_key = str(feature["feature_key"])
        feature_slug = feature_key.replace("/", "__")
        out_feature_dir = out_session_dir / "assets" / feature_slug
        out_feature_dir.mkdir(parents=True, exist_ok=True)
        out_examples: list[dict[str, Any]] = []
        for example in sorted(feature.get("label_examples", []), key=lambda row: int(row["rank"])):
            rank = int(example["rank"])
            image_path = str(example["image_path"])
            token_idx = int(example["token_idx"])
            erf_json_src = source_session_dir / str(example["feature_erf_json"])
            erf_payload = json.loads(erf_json_src.read_text())
            if not list(erf_payload.get("support_indices") or []) and not bool(
                erf_payload.get("recovery_threshold_reached", False)
            ):
                raise RuntimeError(
                    f"Empty ERF support in source session for {feature_key} rank={rank} "
                    f"sample_id={int(example['sample_id'])} token_idx={token_idx}"
                )

            locator_path = out_feature_dir / f"example_{rank:02d}_feature_erf_locator_grid.png"
            plain_path = out_feature_dir / f"example_{rank:02d}_feature_erf_plain.png"
            cyan_dot_path = out_feature_dir / f"example_{rank:02d}_feature_erf_cyan_dot.png"
            cyan_cross_path = out_feature_dir / f"example_{rank:02d}_feature_erf_cyan_cross.png"
            cyan_dashed_box_path = out_feature_dir / f"example_{rank:02d}_feature_erf_cyan_dashed_box.png"
            box_path = out_feature_dir / f"example_{rank:02d}_feature_erf_patch_box.png"
            zoom_path = out_feature_dir / f"example_{rank:02d}_feature_erf_overview_zoom.png"
            erf_json_path = out_feature_dir / f"example_{rank:02d}_feature_erf.json"

            save_support_mask_image(
                image_path,
                erf_payload["support_indices"],
                plain_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                mode="masked_black",
                background_color=CLIP_ZERO_RGB,
                include_token_box=False,
                mask_resample=Image.NEAREST,
            )
            save_support_locator_grid_image(
                image_path,
                erf_payload["support_indices"],
                locator_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                background_color=CLIP_ZERO_RGB,
                mask_resample=Image.NEAREST,
            )
            save_support_mask_image(
                image_path,
                erf_payload["support_indices"],
                cyan_dot_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                mode="masked_black",
                background_color=CLIP_ZERO_RGB,
                include_token_box=True,
                mask_resample=Image.NEAREST,
                token_marker_style="dot_translucent",
                token_marker_color=(0, 220, 255),
                token_marker_alpha=112,
            )
            save_support_mask_image(
                image_path,
                erf_payload["support_indices"],
                cyan_cross_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                mode="masked_black",
                background_color=CLIP_ZERO_RGB,
                include_token_box=True,
                mask_resample=Image.NEAREST,
                token_marker_style="cross",
                token_marker_color=(0, 220, 255),
                token_marker_alpha=176,
            )
            save_support_mask_image(
                image_path,
                erf_payload["support_indices"],
                cyan_dashed_box_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                mode="masked_black",
                background_color=CLIP_ZERO_RGB,
                include_token_box=True,
                mask_resample=Image.NEAREST,
                token_marker_style="dashed_box",
                token_marker_color=(0, 220, 255),
                token_marker_alpha=168,
            )
            save_support_mask_image(
                image_path,
                erf_payload["support_indices"],
                box_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                mode="masked_black",
                background_color=CLIP_ZERO_RGB,
                include_token_box=True,
                mask_resample=Image.NEAREST,
                token_marker_style="box",
                token_marker_color=(255, 64, 64),
            )
            save_support_overview_zoom_image(
                image_path,
                erf_payload["support_indices"],
                zoom_path,
                token_idx=token_idx,
                image_size=image_size,
                grid_size=grid_size,
                resize_size=resize_size,
                background_color=CLIP_ZERO_RGB,
                mask_resample=Image.NEAREST,
                token_marker_color=(255, 64, 64),
            )
            _write_json(erf_json_path, erf_payload)

            out_examples.append(
                {
                    **example,
                    "feature_erf_json": str(erf_json_path.relative_to(out_session_dir)),
                    "feature_erf_plain": str(plain_path.relative_to(out_session_dir)),
                    "feature_erf_locator_grid": str(locator_path.relative_to(out_session_dir)),
                    "feature_erf_cyan_dot": str(cyan_dot_path.relative_to(out_session_dir)),
                    "feature_erf_cyan_cross": str(cyan_cross_path.relative_to(out_session_dir)),
                    "feature_erf_cyan_dashed_box": str(cyan_dashed_box_path.relative_to(out_session_dir)),
                    "feature_erf_patch_box": str(box_path.relative_to(out_session_dir)),
                    "feature_erf_overview_zoom": str(zoom_path.relative_to(out_session_dir)),
                }
            )

        out_features.append(
            {
                "feature_key": feature_key,
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "selection_stats": dict(feature.get("selection_stats", {})),
                "label_examples": out_examples,
            }
        )

    out_manifest = {
        "session_name": out_session_dir.name,
        "source_session": str(source_manifest.get("session_name") or source_session_dir.name),
        "source_session_dir": str(source_session_dir),
        "config": {
            "image_size": int(image_size),
            "resize_size": int(resize_size),
            "grid_size": int(grid_size),
            "render_spatial_preprocess": "resize_shorter_edge_then_center_crop",
        },
        "variant_methods": [
            "feature_erf_plain",
            "feature_erf_locator_grid",
            "feature_erf_cyan_dot",
            "feature_erf_cyan_cross",
            "feature_erf_cyan_dashed_box",
            "feature_erf_patch_box",
            "feature_erf_overview_zoom",
        ],
        "features": out_features,
    }
    _write_json(out_session_dir / "selection_manifest.json", out_manifest)
    _build_preview_html(out_session_dir, out_features)
    print(out_session_dir)


if __name__ == "__main__":
    main()
