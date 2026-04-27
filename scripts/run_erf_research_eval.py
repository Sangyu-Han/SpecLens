from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO = Path("/home/sangyu/Desktop/Master/SpecLens")
CODEX_DIR = Path("/home/sangyu/Desktop/Master/codex_research_softins")
CLAUDE_DIR = Path("/home/sangyu/Desktop/Master/claude_research_softins")
DEFAULT_OUT_DIR = REPO / "outputs" / "erf_runner"

MAIN_METHODS = ("plain_ixg", "inflow_erf", "ac_s32_codex", "cautious_cos_codex")

@dataclass(frozen=True)
class ExperimentSpec:
    script: Path
    compare_metrics: Tuple[str, ...]
    default_compare: Path | None = None
    default_output_name: str | None = None


EXPERIMENTS: Dict[str, ExperimentSpec] = {
    "wide_cautious": ExperimentSpec(
        script=CODEX_DIR / "eval_r31_wide_cautious_codex.py",
        compare_metrics=("soft_ins_delta", "insertion_auc", "mas_ins_auc", "rep_idsds"),
        default_compare=CODEX_DIR / "outputs" / "results_r31_wide_cautious_codex_merged.json",
        default_output_name="results_wide_cautious.json",
    ),
    "stochastic_wide": ExperimentSpec(
        script=CODEX_DIR / "eval_r32_stochastic_wide.py",
        compare_metrics=("stoch_ins_auc", "stoch_ins_delta", "stoch_del_auc", "stoch_auc_gap"),
        default_compare=CODEX_DIR / "outputs" / "results_r32_stochastic_wide_merged.json",
        default_output_name="results_stochastic_wide.json",
    ),
    "r18": ExperimentSpec(
        script=CODEX_DIR / "eval_r18.py",
        compare_metrics=("soft_ins_delta", "insertion_auc", "mas_ins_auc", "rep_idsds"),
        default_compare=CODEX_DIR / "outputs" / "results_r18.json",
        default_output_name="results_r18.json",
    ),
    "multimodel_erf": ExperimentSpec(
        script=CODEX_DIR / "eval_multimodel_erf.py",
        compare_metrics=("soft_ins_delta", "insertion_auc", "mas_ins_auc", "rep_idsds"),
        default_compare=None,
        default_output_name="results_multimodel_erf.json",
    ),
}


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    argv = sys.argv[1:]
    forwarded: List[str] = []
    if "--" in argv:
        idx = argv.index("--")
        forwarded = argv[idx + 1 :]
        argv = argv[:idx]

    parser = argparse.ArgumentParser(
        description=(
            "Run canonical ERF research evaluators from SpecLens and compare "
            "the produced JSON against a reference result."
        )
    )
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--compare-to", type=Path, default=None)
    parser.add_argument("--max-abs-diff", type=float, default=1e-6)
    parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args(argv), forwarded


def _default_output_path(spec: ExperimentSpec) -> Path:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if spec.default_output_name is None:
        raise ValueError("Experiment is missing default output name")
    return DEFAULT_OUT_DIR / spec.default_output_name


def _row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        int(row.get("block_idx", -1)),
        int(row.get("feature_id", -1)),
        int(row.get("target_patch_idx", row.get("ledger_patch_idx", row.get("tok_max", -1)))),
        str(row.get("image_path", "")),
    )


def _flatten_numeric_sections_for_metrics(
    payload: Dict[str, Any],
    metrics_to_compare: Sequence[str],
) -> Iterable[Tuple[str, float]]:
    for blk, methods in payload.get("aggregate_by_block", {}).items():
        for method, metrics in methods.items():
            for metric in metrics_to_compare:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    yield (f"aggregate_by_block.{blk}.{method}.{metric}", float(value))
    for method, metrics in payload.get("aggregate_overall", {}).items():
        for metric in metrics_to_compare:
            value = metrics.get(metric)
            if isinstance(value, (int, float)):
                yield (f"aggregate_overall.{method}.{metric}", float(value))


def _compare_payloads(
    candidate: Dict[str, Any],
    reference: Dict[str, Any],
    *,
    metrics_to_compare: Sequence[str],
    atol: float,
) -> Dict[str, Any]:
    cand_map = dict(_flatten_numeric_sections_for_metrics(candidate, metrics_to_compare))
    ref_map = dict(_flatten_numeric_sections_for_metrics(reference, metrics_to_compare))
    shared_paths = sorted(set(cand_map) & set(ref_map))

    agg_diffs: List[Tuple[str, float, float, float]] = []
    for path in shared_paths:
        c = cand_map[path]
        r = ref_map[path]
        agg_diffs.append((path, c, r, abs(c - r)))

    row_diffs: List[Tuple[Tuple[Any, ...], str, str, float, float, float]] = []
    cand_rows = {_row_key(row): row for row in candidate.get("per_triple", [])}
    ref_rows = {_row_key(row): row for row in reference.get("per_triple", [])}
    for key in sorted(set(cand_rows) & set(ref_rows)):
        crow = cand_rows[key]
        rrow = ref_rows[key]
        c_methods = crow.get("methods", {})
        r_methods = rrow.get("methods", {})
        for method in MAIN_METHODS:
            if method not in c_methods or method not in r_methods:
                continue
            c_md = c_methods[method]
            r_md = r_methods[method]
            if "error" in c_md or "error" in r_md:
                continue
            for metric in metrics_to_compare:
                c_val = c_md.get(metric)
                r_val = r_md.get(metric)
                if not isinstance(c_val, (int, float)) or not isinstance(r_val, (int, float)):
                    continue
                row_diffs.append((key, method, metric, float(c_val), float(r_val), abs(float(c_val) - float(r_val))))

    max_agg = max((item[3] for item in agg_diffs), default=0.0)
    max_row = max((item[5] for item in row_diffs), default=0.0)
    offenders = [item for item in agg_diffs if item[3] > atol]
    offenders_rows = [item for item in row_diffs if item[5] > atol]
    return {
        "shared_aggregate_points": len(agg_diffs),
        "shared_per_triple_points": len(row_diffs),
        "max_aggregate_abs_diff": max_agg,
        "max_per_triple_abs_diff": max_row,
        "aggregate_offenders": offenders[:20],
        "per_triple_offenders": offenders_rows[:20],
        "passed": max(max_agg, max_row) <= atol,
    }


def main() -> None:
    args, forwarded = parse_args()
    spec = EXPERIMENTS[args.experiment]

    output_json = args.output_json or _default_output_path(spec)
    compare_to = None if args.skip_compare else (args.compare_to or spec.default_compare)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if "--output-json" not in forwarded:
        forwarded = [*forwarded, "--output-json", str(output_json)]

    cmd = [sys.executable, str(spec.script), *forwarded]
    print("=== ERF Runner ===")
    print("repo:", REPO)
    print("experiment:", args.experiment)
    print("script:", spec.script)
    print("output:", output_json)
    if compare_to is not None:
        print("compare_to:", compare_to)
    print("command:", " ".join(shlex.quote(part) for part in cmd))
    print()
    sys.stdout.flush()

    completed = subprocess.run(cmd, cwd=str(REPO))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    if compare_to is None:
        print("comparison: skipped")
        return

    candidate = json.loads(output_json.read_text())
    reference = json.loads(compare_to.read_text())
    diff = _compare_payloads(
        candidate,
        reference,
        metrics_to_compare=spec.compare_metrics,
        atol=float(args.max_abs_diff),
    )

    print("\n=== Comparison ===")
    print(
        f"aggregate_points={diff['shared_aggregate_points']} "
        f"per_triple_points={diff['shared_per_triple_points']}"
    )
    print(f"max_aggregate_abs_diff={diff['max_aggregate_abs_diff']:.6g}")
    print(f"max_per_triple_abs_diff={diff['max_per_triple_abs_diff']:.6g}")

    if diff["aggregate_offenders"]:
        print("\nTop aggregate offenders:")
        for path, cand, ref, adiff in diff["aggregate_offenders"]:
            print(f"  {path}: cand={cand:.6g} ref={ref:.6g} diff={adiff:.6g}")

    if diff["per_triple_offenders"]:
        print("\nTop per-triple offenders:")
        for key, method, metric, cand, ref, adiff in diff["per_triple_offenders"]:
            print(
                f"  key={key} {method}.{metric}: cand={cand:.6g} ref={ref:.6g} diff={adiff:.6g}"
            )

    if not diff["passed"]:
        raise SystemExit(2)

    print("\ncomparison: passed")


if __name__ == "__main__":
    main()
