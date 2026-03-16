#!/usr/bin/env python3
"""
Visualize SAE feature contribution summaries.

Given a manifest produced by scripts/sae_global_contribution.py, this script
renders a single figure showing:
  - Left: top-k activating images for the chosen feature.
  - Right: for each attribution method, the top-k classes (by |normalized mean
    contribution|) for that feature, using the per-feature max-abs normalization
    from the global contribution outputs.

Example:
  python scripts/sae_global_contribution_viz.py \\
    --manifest outputs/sae_global_contributions/manifest.json \\
    --layer blocks.10.mlp \\
    --unit 123 \\
    --methods activation_patch,input_x_grad,logitprism \\
    --topk-images 5 --topk-classes 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Ensure repo root on sys.path and set CWD for src/ imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    os.chdir(REPO_ROOT)
except Exception:
    pass

from src.core.indexing.registry_utils import sanitize_layer_name  # noqa: E402


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_figsize(raw: Optional[str], *, fallback_height: float) -> Tuple[float, float]:
    if not raw:
        return (11.0, fallback_height)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        return (11.0, fallback_height)
    try:
        return (float(parts[0]), float(parts[1]))
    except Exception:
        return (11.0, fallback_height)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base: Path, raw: str | Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p.expanduser()


def _find_feature_id(features: Sequence[Dict[str, Any]], *, layer: str, unit: int) -> Optional[int]:
    for entry in features:
        if str(entry.get("layer")) == layer and int(entry.get("unit", -1)) == int(unit):
            return int(entry.get("id", -1))
    return None


def _load_top_images(meta_path: Path, *, topk: int) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    if not meta_path.exists():
        print(f"[warn] feature meta not found: {meta_path}")
        return []
    try:
        meta = _load_json(meta_path)
    except Exception as exc:
        print(f"[warn] failed to load meta {meta_path}: {exc}")
        return []
    samples = meta.get("samples", []) or []
    samples = sorted(samples, key=lambda s: float(s.get("score", 0.0)), reverse=True)
    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    for sample in samples:
        img_path = Path(sample.get("path", "")).expanduser()
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as img:
                arr = np.array(img.convert("RGB"))
        except Exception:
            continue
        out.append((arr, sample))
        if len(out) >= topk:
            break
    return out


def _topk_classes(vec: np.ndarray, *, k: int) -> List[Tuple[int, float]]:
    if vec.ndim != 1 or vec.size == 0:
        return []
    k = max(1, min(k, vec.shape[0]))
    idx = np.argpartition(np.abs(vec), -k)[-k:]
    idx = idx[np.argsort(-np.abs(vec[idx]))]
    return [(int(i), float(vec[i])) for i in idx]


def _maybe_load_class_names(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return {}
    try:
        data = _load_json(path)
    except Exception as exc:
        print(f"[warn] failed to load class names {path}: {exc}")
        return {}
    if isinstance(data, list):
        return {int(i): str(name) for i, name in enumerate(data)}
    if isinstance(data, dict):
        out: Dict[int, str] = {}
        for k, v in data.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def _build_figure(
    *,
    title: str,
    images: List[Tuple[np.ndarray, Dict[str, Any]]],
    method_rows: List[Dict[str, Any]],
    topk_images: int,
    class_names: Dict[int, str],
    figsize: Tuple[float, float],
) -> plt.Figure:
    num_methods = max(1, len(method_rows))
    img_rows = max(1, topk_images if images else 1)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.25)
    left_grid = outer[0].subgridspec(img_rows, 1, hspace=0.05)
    right_grid = outer[1].subgridspec(num_methods, 1, hspace=0.45)

    for i in range(img_rows):
        ax = fig.add_subplot(left_grid[i])
        if i < len(images):
            img, meta = images[i]
            ax.imshow(img)
            score = meta.get("score", None)
            decile = meta.get("decile", None)
            rank = meta.get("rank_in_decile", None)
            title_bits = []
            if score is not None:
                try:
                    title_bits.append(f"score={float(score):.3f}")
                except Exception:
                    pass
            if decile is not None:
                title_bits.append(f"decile={decile}")
            if rank is not None:
                title_bits.append(f"rank={rank}")
            ax.set_title(", ".join(title_bits) if title_bits else f"sample {i+1}", fontsize=9)
        ax.axis("off")

    for row_idx, row in enumerate(method_rows):
        ax = fig.add_subplot(right_grid[row_idx])
        vals = [v for _, v in row["classes"]]
        cls_ids = [c for c, _ in row["classes"]]
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
        y_pos = np.arange(len(cls_ids))
        ax.barh(y_pos, vals, color=colors)
        ax.axvline(0.0, color="#444444", linewidth=1)
        ax.set_xlim(-1.05, 1.05)
        labels = [class_names.get(cid, str(cid)) for cid in cls_ids]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{row['method']} (top {len(cls_ids)})", fontsize=10)
        if row_idx == num_methods - 1:
            ax.set_xlabel("Normalized mean contribution (mean / max_abs)", fontsize=9)
        for y, v in zip(y_pos, vals):
            ax.text(
                v + (0.02 if v >= 0 else -0.02),
                y,
                f"{v:.3f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

    fig.suptitle(title, fontsize=12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize global SAE contributions for a single feature.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json from sae_global_contribution.")
    parser.add_argument("--layer", type=str, help="Layer name of the feature (required unless --feature-id).")
    parser.add_argument("--unit", type=int, help="Unit index of the feature (required unless --feature-id).")
    parser.add_argument("--feature-id", type=int, default=None, help="Feature id (index in features.json).")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods to include (default: all).")
    parser.add_argument(
        "--image-method",
        type=str,
        default=None,
        help="Method to use for sample thumbnails (default: first selected).",
    )
    parser.add_argument("--topk-images", type=int, default=5)
    parser.add_argument("--topk-classes", type=int, default=5)
    parser.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="Optional JSON (list or {id:name}) for class labels in plots.",
    )
    parser.add_argument("--figsize", type=str, default=None, help="Override figsize as 'W,H'.")
    parser.add_argument("--out", type=Path, default=None, help="Output image path (png).")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser()
    manifest_dir = manifest_path.parent
    manifest = _load_json(manifest_path)

    available_methods = list((manifest.get("methods") or {}).keys())
    if not available_methods:
        raise RuntimeError("No methods found in manifest.")

    selected_methods = _parse_csv_list(args.methods) if args.methods else available_methods
    selected_methods = [m for m in selected_methods if m in available_methods]
    if not selected_methods:
        raise RuntimeError("No matching methods found in manifest for requested --methods.")

    class_names = _maybe_load_class_names(args.class_names)

    method_rows: List[Dict[str, Any]] = []
    feature_layer: Optional[str] = args.layer
    feature_unit: Optional[int] = args.unit

    for method in selected_methods:
        info = manifest["methods"].get(method, {})
        root = _resolve_path(manifest_dir, info.get("root", ""))
        features_path = _resolve_path(manifest_dir, info.get("features", root / "features.json"))
        mean_path = _resolve_path(manifest_dir, info.get("mean", root / "mean_contributions.npy"))
        max_abs_path = root / "max_abs.npy"

        if not features_path.exists() or not mean_path.exists():
            print(f"[warn] missing artifacts for method={method} (features={features_path}, mean={mean_path}); skipping")
            continue

        features_meta = _load_json(features_path)
        features = features_meta.get("features", []) or []

        if args.feature_id is not None:
            fid = int(args.feature_id)
        else:
            if feature_layer is None or feature_unit is None:
                raise ValueError("Either --feature-id or (--layer and --unit) must be provided.")
            found = _find_feature_id(features, layer=feature_layer, unit=int(feature_unit))
            if found is None:
                print(f"[warn] feature not found for method={method} layer={feature_layer} unit={feature_unit}; skipping")
                continue
            fid = int(found)

        if fid < 0 or fid >= len(features):
            print(f"[warn] feature id {fid} out of range for method={method}; skipping")
            continue

        feat_entry = features[fid]
        feature_layer = feature_layer or feat_entry.get("layer")
        feature_unit = feature_unit if feature_unit is not None else int(feat_entry.get("unit", -1))

        mean_matrix = np.load(mean_path)
        if mean_matrix.ndim != 2 or fid >= mean_matrix.shape[0]:
            print(f"[warn] mean matrix shape {mean_matrix.shape} too small for feature {fid} (method={method}); skipping")
            continue
        mean_vec = np.asarray(mean_matrix[fid], dtype=np.float32)

        max_abs_val = None
        if max_abs_path.exists():
            try:
                max_abs_arr = np.load(max_abs_path)
                if fid < max_abs_arr.shape[0]:
                    max_abs_val = float(max_abs_arr[fid])
            except Exception:
                max_abs_val = None
        if max_abs_val is None:
            try:
                max_abs_val = float(feat_entry.get("max_abs", 0.0))
            except Exception:
                max_abs_val = float(np.max(np.abs(mean_vec))) if mean_vec.size > 0 else 0.0
        denom = max(max_abs_val, np.finfo(np.float32).eps)
        norm_vec = mean_vec / denom

        top_classes = _topk_classes(norm_vec, k=int(args.topk_classes))
        method_rows.append(
            {
                "method": method,
                "classes": top_classes,
                "root": root,
                "layer": feat_entry.get("layer"),
                "unit": int(feat_entry.get("unit", -1)),
            }
        )

    if not method_rows:
        raise RuntimeError("No valid methods to plot after filtering/loading artifacts.")

    # Choose method for thumbnails.
    image_method = args.image_method or method_rows[0]["method"]
    image_row = next((row for row in method_rows if row["method"] == image_method), method_rows[0])
    img_layer = feature_layer or image_row.get("layer")
    img_unit = feature_unit if feature_unit is not None else image_row.get("unit")
    per_feature_meta = None
    if img_layer is not None and img_unit is not None:
        per_feature_meta = (
            image_row["root"]
            / "per_feature"
            / sanitize_layer_name(str(img_layer))
            / f"unit_{int(img_unit)}.json"
        )
    images = _load_top_images(per_feature_meta, topk=int(args.topk_images)) if per_feature_meta else []

    height_guess = max(len(images) or 1, len(method_rows)) * 1.6
    figsize = _parse_figsize(args.figsize, fallback_height=height_guess)
    title_layer = img_layer or feature_layer or "layer"
    title_unit = img_unit if img_unit is not None else feature_unit
    figure_title = f"Layer={title_layer}, unit={title_unit}, methods={', '.join([r['method'] for r in method_rows])}"
    fig = _build_figure(
        title=figure_title,
        images=images,
        method_rows=method_rows,
        topk_images=int(args.topk_images),
        class_names=class_names,
        figsize=figsize,
    )

    out_path = args.out
    if out_path is None:
        safe_layer = sanitize_layer_name(str(title_layer))
        out_dir = manifest_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_layer}_unit{title_unit}_contrib.png"
    else:
        out_path = out_path.expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    print(f"[ok] wrote figure to {out_path}")


if __name__ == "__main__":
    main()
