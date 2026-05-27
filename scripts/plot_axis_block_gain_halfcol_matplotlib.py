#!/usr/bin/env python3
"""Create a compact matplotlib bar chart for Task B Top-1 gains."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs/autolabel_metric_runs/figures/axis_block_taskB_gain_halfcol_matplotlib_20260506"
MODELS = ["CLIP-B/16", "SigLIP-B/16", "DINOv3-S/16"]
MODEL_LABELS = ["CLIP-B/16", "SigLIP-B/16", "DINOv3-S/16"]
BLOCKS = [2, 6, 10]
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


def mean_se(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var) / math.sqrt(len(values))


def load_taskb_gains() -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((len(MODELS), len(BLOCKS)), dtype=float)
    ses = np.zeros_like(means)
    for model_idx, model in enumerate(MODELS):
        pattern = AXIS_SUMMARY_PATTERNS[model]
        for block_idx, block in enumerate(BLOCKS):
            repeat_deltas: list[float] = []
            for r in range(1, 11):
                path = REPO / pattern.format(r=r)
                with path.open() as f:
                    summary = json.load(f)
                block_key = str(block)
                erf = summary["axis2"]["erf_cyan_cross"]["per_block"][block_key]["top1_accuracy"]
                al = summary["axis2"]["sae_only"]["per_block"][block_key]["top1_accuracy"]
                repeat_deltas.append(erf - al)
            means[model_idx, block_idx], ses[model_idx, block_idx] = mean_se(repeat_deltas)
    return means, ses


def plot(out_prefix: Path) -> list[Path]:
    means, ses = load_taskb_gains()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Carlito", "Calibri", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )

    fig, ax = plt.subplots(figsize=(2.45, 1.45))
    x = np.arange(len(MODELS)) * 0.52
    width = 0.105
    offsets = np.array([-width, 0.0, width])

    # Muted, print-friendly depth palette with increasing visual weight.
    colors = ["#c7c7c7", "#8da0cb", "#b2182b"]

    for block_idx, block in enumerate(BLOCKS):
        xpos = x + offsets[block_idx]
        ax.bar(
            xpos,
            means[:, block_idx],
            width=width * 0.94,
            color=colors[block_idx],
            edgecolor="#30343b",
            linewidth=0.45,
            label=f"Block {block}",
            zorder=3,
        )

    ax.axhline(0, color="#222222", linewidth=0.75, zorder=2)
    ax.set_ylabel("Top-1 gain")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_xlim(x[0] - 0.23, x[-1] + 0.23)
    ax.set_ylim(-0.10, 0.30)
    ax.set_yticks([-0.10, 0.00, 0.10, 0.20, 0.30])
    ax.set_yticklabels(["-0.10", "0.00", "0.10", "0.20", "0.30"])
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.42, alpha=0.70, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.tick_params(axis="both", length=2.5, width=0.55, color="#555555", pad=2.0)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.995),
        ncol=3,
        frameon=False,
        handlelength=1.05,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    for patch in legend.get_patches():
        patch.set_height(5.0)
        patch.set_y(-1.0)

    fig.subplots_adjust(left=0.18, right=0.995, top=0.995, bottom=0.15)
    paths = [out_prefix.with_suffix(".png"), out_prefix.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_prefix = args.out_prefix if args.out_prefix.is_absolute() else REPO / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in plot(out_prefix):
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
