#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence


REPO = Path(os.environ.get("SPECLENS_REPO", str(Path(__file__).resolve().parents[1]))).expanduser()
DEFAULT_IN_JSON = REPO / "outputs" / "paper_feature_erf" / "results_feature_erf_multimodel_main.json"
DEFAULT_OUT_MD = REPO / "outputs" / "paper_feature_erf" / "table_feature_erf_multimodel_main.md"
METHODS = (
    "plain_ixg",
    "ig_block0_abs",
    "libragrad_ig50",
    "naive_attn_rollout",
    "attnlrp_ixg",
    "inflow_erf",
    "cautious_cos",
)
METRICS = ("stoch_ins_delta", "insertion_auc", "mas_ins_auc", "rep_idsds")
PACKS = ("clip", "siglip", "dinov3")
BLOCKS = ("2", "6", "10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paper-ready markdown tables from feature ERF benchmark JSON.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_IN_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def _fmt_metric(stats: Dict[str, Any] | None) -> str:
    if not stats:
        return "---"
    return f"{float(stats['mean']):.3f} ± {float(stats['std']):.3f}"


def _metric_cell(method_metrics: Dict[str, Any], metric: str) -> str:
    return _fmt_metric(method_metrics.get(metric))


def _group_metrics(group: Dict[str, Any], method: str) -> Dict[str, Any]:
    return group.get(method, {}) if isinstance(group, dict) else {}


def render_main_table(payload: Dict[str, Any]) -> str:
    overall = payload.get("aggregate_overall", {})
    by_pack = payload.get("aggregate_by_pack", {})

    headers = ["Method"]
    for group_name in ("Overall", "CLIP", "SigLIP", "DINOv3"):
        for metric in METRICS:
            headers.append(f"{group_name}:{metric}")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for method in METHODS:
        row = [method]
        for group in (overall, by_pack.get("clip", {}), by_pack.get("siglip", {}), by_pack.get("dinov3", {})):
            method_metrics = _group_metrics(group, method)
            for metric in METRICS:
                row.append(_metric_cell(method_metrics, metric))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_appendix_tables(payload: Dict[str, Any]) -> str:
    by_pack_block = payload.get("aggregate_by_pack_block", {})
    sections = []
    for pack in PACKS:
        sections.append(f"## {pack.upper()}")
        pack_block = by_pack_block.get(pack, {})
        for block in BLOCKS:
            sections.append(f"### Block {block}")
            lines = [
                "| Method | stoch_ins_delta | insertion_auc | mas_ins_auc | rep_idsds | stoch_seed_means |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            block_group = pack_block.get(block, {})
            for method in METHODS:
                method_metrics = block_group.get(method, {})
                seed_stats = method_metrics.get("stoch_ins_delta_by_seed", {})
                seed_render = ", ".join(
                    f"{seed}:{float(stats['mean']):.3f}" for seed, stats in sorted(seed_stats.items())
                ) or "---"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            method,
                            _metric_cell(method_metrics, "stoch_ins_delta"),
                            _metric_cell(method_metrics, "insertion_auc"),
                            _metric_cell(method_metrics, "mas_ins_auc"),
                            _metric_cell(method_metrics, "rep_idsds"),
                            seed_render,
                        ]
                    )
                    + " |"
                )
            sections.append("\n".join(lines))
            sections.append("")
    return "\n".join(sections).strip()


def render_summary(payload: Dict[str, Any]) -> str:
    meta = payload.get("_meta", {})
    packs = ", ".join(sorted(meta.get("packs", {}).keys()))
    blocks = ", ".join(str(x) for x in meta.get("blocks", []))
    methods = ", ".join(meta.get("methods", []))
    return "\n".join(
        [
            "# Feature ERF Paper Tables",
            "",
            "## Run Metadata",
            f"- Packs: `{packs}`",
            f"- Blocks: `{blocks}`",
            f"- Methods: `{methods}`",
            f"- n_features per (model, block): `{meta.get('n_features')}`",
            f"- n_images per feature: `{meta.get('n_images')}`",
            f"- stochastic evaluator seeds: `{meta.get('stoch_seeds')}`",
            f"- stochastic steps / samples: `{meta.get('stoch_steps')}` / `{meta.get('stoch_samples')}`",
            "",
            "## Main Unified Table",
            render_main_table(payload),
            "",
            "## Appendix: Model × Block Tables",
            render_appendix_tables(payload),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text())
    rendered = render_summary(payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(rendered)
    print(args.output_md)


if __name__ == "__main__":
    main()
