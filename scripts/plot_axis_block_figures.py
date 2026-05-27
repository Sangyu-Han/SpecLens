#!/usr/bin/env python3
"""Plot per-block LMM decoding figures from the markdown raw-data table."""

from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "outputs/autolabel_metric_runs/experiments_section_axis_block_raw_data_20260506.md"
DEFAULT_OUT_DIR = REPO / "outputs/autolabel_metric_runs/figures"
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


def series(
    data: dict[str, dict[int, dict[str, dict[str, tuple[float, float]]]]],
    model: str,
    variant: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[float] = []
    errs: list[float] = []
    for block in BLOCKS:
        mean, err = data[model][block][variant][metric]
        means.append(mean)
        errs.append(err)
    return np.array(means), np.array(errs)


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty value list")
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.stdev(values) / np.sqrt(len(values))


def load_paired_top1_deltas() -> dict[str, dict[str, dict[int, tuple[float, float]]]]:
    """Return ERF-minus-AL paired deltas by model/task/block.

    Each delta is computed per repeat and per block before taking the mean/SE.
    This preserves the pairing between AL-based and ERF-based evaluations.
    """

    deltas: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
    for model, pattern in AXIS_SUMMARY_PATTERNS.items():
        deltas[model] = {"Task A": {}, "Task B": {}}
        by_task_block: dict[tuple[str, int], list[float]] = {
            (task, block): [] for task in ("Task A", "Task B") for block in BLOCKS
        }
        for r in range(1, 11):
            path = REPO / pattern.format(r=r)
            with path.open() as f:
                summary = json.load(f)
            for task, axis in (("Task A", "axis1"), ("Task B", "axis2")):
                for block in BLOCKS:
                    block_key = str(block)
                    erf = summary[axis]["erf_cyan_cross"]["per_block"][block_key]["top1_accuracy"]
                    al = summary[axis]["sae_only"]["per_block"][block_key]["top1_accuracy"]
                    by_task_block[(task, block)].append(erf - al)
        for task in ("Task A", "Task B"):
            for block in BLOCKS:
                deltas[model][task][block] = mean_se(by_task_block[(task, block)])
    return deltas


def style_axes(ax: plt.Axes, *, ylim: tuple[float, float]) -> None:
    ax.set_xticks(BLOCKS)
    ax.set_xlim(1.6, 10.4)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#d8dee6", linewidth=0.7, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")
    ax.tick_params(axis="both", labelsize=8.5, color="#6b7280")


def draw_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str,
    alpha: float,
    band_alpha: float,
    zorder: int,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        marker="o",
        markersize=4.2,
        markeredgewidth=0.9,
        label=label,
        alpha=alpha,
        zorder=zorder,
    )
    ax.fill_between(x, y - err, y + err, color=color, alpha=band_alpha, linewidth=0, zorder=zorder - 1)


def plot_top1(data: dict[str, dict[int, dict[str, dict[str, tuple[float, float]]]]], out_dir: Path) -> list[Path]:
    models = list(data)
    x = np.array(BLOCKS)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15), sharey=True)
    task_a_color = "#55799e"
    task_b_color = "#b43a38"
    variant_style = {"AL-based": "--", "ERF-based": "-"}
    legend_handles = None

    for ax, model in zip(axes, models, strict=True):
        for variant in ("AL-based", "ERF-based"):
            y, err = series(data, model, variant, "Task A Top-1")
            draw_line(
                ax,
                x,
                y,
                err,
                label=f"{variant.split('-')[0]} Task A",
                color=task_a_color,
                linestyle=variant_style[variant],
                alpha=0.48,
                band_alpha=0.055,
                zorder=2,
            )
            y, err = series(data, model, variant, "Task B Top-1")
            draw_line(
                ax,
                x,
                y,
                err,
                label=f"{variant.split('-')[0]} Task B",
                color=task_b_color,
                linestyle=variant_style[variant],
                alpha=1.0,
                band_alpha=0.12,
                zorder=5,
            )
        ax.set_title(model, fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xlabel("Block", fontsize=9.5)
        style_axes(ax, ylim=(0.0, 0.78))
        ax.axhline(0.25, color=task_a_color, linewidth=0.8, alpha=0.20, linestyle=":")
        ax.axhline(0.0625, color=task_b_color, linewidth=0.8, alpha=0.20, linestyle=":")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Top-1 accuracy", fontsize=9.5)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        fontsize=8.3,
        handlelength=2.7,
        columnspacing=1.15,
    )
    fig.text(0.995, 0.015, "Bands show reported SE.", ha="right", va="bottom", fontsize=7.2, color="#6b7280")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), w_pad=1.1)

    paths = [
        out_dir / "axis_block_top1_taskA_taskB_main_20260506.png",
        out_dir / "axis_block_top1_taskA_taskB_main_20260506.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_taskb_appendix(
    data: dict[str, dict[int, dict[str, dict[str, tuple[float, float]]]]], out_dir: Path
) -> list[Path]:
    models = list(data)
    x = np.array(BLOCKS)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.25), sharex=True)
    metrics = [("Task B MRR", "MRR", (0.25, 0.88)), ("Task B R@5", "Recall@5", (0.35, 1.02))]
    colors = {"AL-based": "#7b8492", "ERF-based": "#b43a38"}
    styles = {"AL-based": "--", "ERF-based": "-"}
    legend_handles = None

    for row_idx, (metric, ylabel, ylim) in enumerate(metrics):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            for variant in ("AL-based", "ERF-based"):
                y, err = series(data, model, variant, metric)
                draw_line(
                    ax,
                    x,
                    y,
                    err,
                    label=variant.replace("-based", ""),
                    color=colors[variant],
                    linestyle=styles[variant],
                    alpha=1.0,
                    band_alpha=0.11 if variant == "ERF-based" else 0.085,
                    zorder=4 if variant == "ERF-based" else 3,
                )
            if row_idx == 0:
                ax.set_title(model, fontsize=10.5, fontweight="bold", pad=7)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Block", fontsize=9.5)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9.5)
            style_axes(ax, ylim=ylim)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=2.7,
        columnspacing=1.4,
    )
    fig.text(0.995, 0.012, "Bands show reported SE.", ha="right", va="bottom", fontsize=7.2, color="#6b7280")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=1.25, w_pad=1.1)

    paths = [
        out_dir / "axis_block_taskB_mrr_r5_appendix_20260506.png",
        out_dir / "axis_block_taskB_mrr_r5_appendix_20260506.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_top1_delta_3panel(out_dir: Path) -> list[Path]:
    deltas = load_paired_top1_deltas()
    models = list(deltas)
    x = np.array(BLOCKS)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.75), sharey=True)
    colors = {"Task A": "#6f7f90", "Task B": "#b43a38"}
    styles = {"Task A": "--", "Task B": "-"}
    labels = {"Task A": "Task A: token localization", "Task B": "Task B: label discrimination"}
    legend_handles = None

    for ax, model in zip(axes, models, strict=True):
        for task in ("Task A", "Task B"):
            y = np.array([100.0 * deltas[model][task][block][0] for block in BLOCKS])
            err = np.array([100.0 * deltas[model][task][block][1] for block in BLOCKS])
            alpha = 0.58 if task == "Task A" else 1.0
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=colors[task],
                linestyle=styles[task],
                linewidth=1.8 if task == "Task A" else 2.45,
                marker="o",
                markersize=4.5 if task == "Task A" else 5.0,
                capsize=2.6,
                capthick=0.9,
                elinewidth=0.9,
                alpha=alpha,
                label=labels[task],
                zorder=3 if task == "Task B" else 2,
            )
        block10_gain = 100.0 * deltas[model]["Task B"][10][0]
        ax.annotate(
            f"+{block10_gain:.1f} pp",
            xy=(10, block10_gain),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=colors["Task B"],
        )
        ax.set_title(model, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlabel("Block", fontsize=9.3)
        ax.axhline(0, color="#2f3742", linewidth=0.9, alpha=0.65)
        ax.set_xticks(BLOCKS)
        ax.set_xlim(1.5, 10.5)
        ax.set_ylim(-16, 25)
        ax.grid(axis="y", color="#d8dee6", linewidth=0.7, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9aa5b1")
        ax.spines["bottom"].set_color("#9aa5b1")
        ax.tick_params(axis="both", labelsize=8.4, color="#6b7280")
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("ERF-based gain over AL-based (pp)", fontsize=9.3)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
        fontsize=8.4,
        handlelength=2.8,
        columnspacing=1.6,
    )
    fig.text(0.995, 0.008, "Error bars show paired SE over 10 repeats.", ha="right", va="bottom", fontsize=7.1, color="#6b7280")
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=1.1)

    paths = [
        out_dir / "axis_block_top1_delta_3panel_main_20260506.png",
        out_dir / "axis_block_top1_delta_3panel_main_20260506.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_taskb_delta_grouped_bar(out_dir: Path) -> list[Path]:
    deltas = load_paired_top1_deltas()
    models = list(deltas)
    x = np.arange(len(models))
    width = 0.22
    offsets = {2: -width, 6: 0.0, 10: width}
    colors = {2: "#d8aaa5", 6: "#c9695f", 10: "#8f2428"}

    fig, ax = plt.subplots(figsize=(6.8, 3.05))
    for block in BLOCKS:
        means = np.array([100.0 * deltas[model]["Task B"][block][0] for model in models])
        errs = np.array([100.0 * deltas[model]["Task B"][block][1] for model in models])
        bars = ax.bar(
            x + offsets[block],
            means,
            width,
            yerr=errs,
            capsize=2.8,
            color=colors[block],
            edgecolor="#ffffff",
            linewidth=0.8,
            label=f"block {block}",
            zorder=3,
        )
        for bar, mean in zip(bars, means, strict=True):
            va = "bottom" if mean >= 0 else "top"
            offset = 0.8 if mean >= 0 else -1.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + offset,
                f"{mean:+.1f}",
                ha="center",
                va=va,
                fontsize=7.3,
                color="#303845",
            )

    ax.axhline(0, color="#2f3742", linewidth=0.9, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylabel("Task B Top-1 gain (pp)", fontsize=9.4)
    ax.set_ylim(-8.5, 25)
    ax.grid(axis="y", color="#d8dee6", linewidth=0.7, alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")
    ax.tick_params(axis="both", labelsize=8.5, color="#6b7280")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=3, frameon=False, fontsize=8.2, handlelength=1.0)
    fig.text(0.995, 0.010, "Error bars show paired SE over 10 repeats.", ha="right", va="bottom", fontsize=7.1, color="#6b7280")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    paths = [
        out_dir / "axis_block_taskB_delta_grouped_bar_20260506.png",
        out_dir / "axis_block_taskB_delta_grouped_bar_20260506.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_top1_delta_singlecol(out_dir: Path) -> list[Path]:
    deltas = load_paired_top1_deltas()
    models = list(deltas)
    x = np.array(BLOCKS)
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 4.85), sharex=True, sharey=True)
    colors = {"Task A": "#6f7f90", "Task B": "#b43a38"}
    styles = {"Task A": "--", "Task B": "-"}
    labels = {"Task A": "Task A", "Task B": "Task B"}
    legend_handles = None

    for ax, model in zip(axes, models, strict=True):
        for task in ("Task A", "Task B"):
            y = np.array([100.0 * deltas[model][task][block][0] for block in BLOCKS])
            err = np.array([100.0 * deltas[model][task][block][1] for block in BLOCKS])
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=colors[task],
                linestyle=styles[task],
                linewidth=1.45 if task == "Task A" else 2.0,
                marker="o",
                markersize=3.8,
                capsize=2.2,
                capthick=0.8,
                elinewidth=0.8,
                alpha=0.56 if task == "Task A" else 1.0,
                label=labels[task],
                zorder=3 if task == "Task B" else 2,
            )
        block10_gain = 100.0 * deltas[model]["Task B"][10][0]
        ax.text(
            10.15,
            block10_gain,
            f"+{block10_gain:.1f}",
            ha="left",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=colors["Task B"],
        )
        ax.set_title(model, fontsize=8.8, fontweight="bold", loc="left", pad=2)
        ax.axhline(0, color="#2f3742", linewidth=0.8, alpha=0.65)
        ax.set_xticks(BLOCKS)
        ax.set_xlim(1.5, 11.55)
        ax.set_ylim(-16, 25)
        ax.grid(axis="y", color="#d8dee6", linewidth=0.6, alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#9aa5b1")
        ax.spines["bottom"].set_color("#9aa5b1")
        ax.tick_params(axis="both", labelsize=7.4, color="#6b7280", pad=1.5)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel("Block", fontsize=8.0)
    fig.supylabel("ERF gain over AL (pp)", x=0.02, fontsize=8.0)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 1.01),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=2.2,
        columnspacing=0.9,
    )
    fig.text(0.995, 0.006, "Error bars: paired SE over 10 repeats.", ha="right", va="bottom", fontsize=6.1, color="#6b7280")
    fig.tight_layout(rect=(0.04, 0.035, 1, 0.965), h_pad=0.58)

    paths = [
        out_dir / "axis_block_top1_delta_3row_singlecol_20260506.png",
        out_dir / "axis_block_top1_delta_3row_singlecol_20260506.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def plot_taskb_delta_grouped_bar_singlecol(out_dir: Path) -> list[Path]:
    deltas = load_paired_top1_deltas()
    models = list(deltas)
    x = np.arange(len(models))
    width = 0.22
    offsets = {2: -width, 6: 0.0, 10: width}
    colors = {2: "#d8aaa5", 6: "#c9695f", 10: "#8f2428"}

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    for block in BLOCKS:
        means = np.array([100.0 * deltas[model]["Task B"][block][0] for model in models])
        errs = np.array([100.0 * deltas[model]["Task B"][block][1] for model in models])
        ax.bar(
            x + offsets[block],
            means,
            width,
            yerr=errs,
            capsize=2.1,
            color=colors[block],
            edgecolor="#ffffff",
            linewidth=0.6,
            label=f"B{block}",
            zorder=3,
        )
        for xpos, mean in zip(x + offsets[block], means, strict=True):
            if block == 10:
                ax.text(xpos, mean + 0.9, f"{mean:.1f}", ha="center", va="bottom", fontsize=6.5, color="#303845")

    ax.axhline(0, color="#2f3742", linewidth=0.8, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(["CLIP", "SigLIP", "DINOv3"], fontsize=7.2)
    ax.set_ylabel("Task B Top-1 gain (pp)", fontsize=7.8)
    ax.set_ylim(-8.5, 25)
    ax.grid(axis="y", color="#d8dee6", linewidth=0.6, alpha=0.65, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa5b1")
    ax.spines["bottom"].set_color("#9aa5b1")
    ax.tick_params(axis="both", labelsize=7.2, color="#6b7280", pad=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=3, frameon=False, fontsize=6.9, handlelength=0.9, columnspacing=0.8)
    fig.text(0.995, 0.006, "Error bars: paired SE.", ha="right", va="bottom", fontsize=6.0, color="#6b7280")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))

    paths = [
        out_dir / "axis_block_taskB_delta_grouped_bar_singlecol_20260506.png",
        out_dir / "axis_block_taskB_delta_grouped_bar_singlecol_20260506.pdf",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else REPO / args.source
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data = parse_markdown(source)
    paths = plot_top1(data, out_dir)
    paths += plot_taskb_appendix(data, out_dir)
    paths += plot_top1_delta_3panel(out_dir)
    paths += plot_taskb_delta_grouped_bar(out_dir)
    paths += plot_top1_delta_singlecol(out_dir)
    paths += plot_taskb_delta_grouped_bar_singlecol(out_dir)
    for path in paths:
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
