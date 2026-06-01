#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_benchmark_module():
    path = REPO / "scripts" / "run_feature_erf_paper_benchmark.py"
    spec = importlib.util.spec_from_file_location("speclens_run_feature_erf_paper_benchmark", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare preserved legacy cautious_cos against core FRI on real benchmark cases."
    )
    parser.add_argument("--pack", default="clip", choices=list(bench.mm.PACK_NAMES) + ["all"])
    parser.add_argument("--blocks", nargs="*", type=int, default=[6])
    parser.add_argument("--n-features", type=int, default=1)
    parser.add_argument("--n-images", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imagenet-val", type=Path, default=bench.mm.IMAGENET_VAL)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=REPO / "outputs/fri_compare/core_vs_legacy.json")
    parser.add_argument(
        "--baseline-mode",
        choices=("default", "zero", "mean_image", "global_mean_h", "global_mean_h_plus_pos"),
        default="global_mean_h_plus_pos",
    )
    parser.add_argument("--index-root", action="append", default=[])
    parser.add_argument("--sae-root", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cautious-steps", type=int, default=32)
    parser.add_argument("--cautious-lr", type=float, default=0.45)
    parser.add_argument("--cautious-lr-end", type=float, default=0.01)
    parser.add_argument("--cautious-tv-weight", type=float, default=0.01)
    parser.add_argument("--cautious-irr-weight", type=float, default=0.05)
    parser.add_argument("--cautious-init-prob", type=float, default=0.5)
    parser.add_argument("--cautious-init-mode", choices=("uniform", "plain_ixg"), default="uniform")
    parser.add_argument("--cautious-reg-warmup-frac", type=float, default=0.0)
    parser.add_argument("--cautious-restarts", type=int, default=1)
    parser.add_argument("--cautious-budget-samples", type=int, default=1)
    parser.add_argument("--cautious-select-best", action="store_true")
    parser.add_argument(
        "--cautious-objective-mode",
        choices=("random_budget_softins", "fixed_budget_softins", "direct_recovery"),
        default="random_budget_softins",
    )
    parser.add_argument(
        "--cautious-optimizer-mode",
        choices=("cautious_adam_cosine", "adam_cosine"),
        default="cautious_adam_cosine",
    )
    parser.add_argument("--cautious-fixed-budget-frac", type=float, default=0.10)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def _load_manifest_rows(path: Path | None) -> List[Dict[str, Any]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("triples") or payload.get("rows") or payload.get("per_triple") or []
    else:
        raise TypeError(f"Unsupported manifest payload in {path}")
    return [dict(row) for row in rows]


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> int:
    k = min(int(k), int(a.size), int(b.size))
    if k <= 0:
        return 0
    return len(set(np.argsort(-a, kind="mergesort")[:k]) & set(np.argsort(-b, kind="mergesort")[:k]))


def _compare_scores(legacy: np.ndarray, core: np.ndarray) -> Dict[str, Any]:
    diff = np.asarray(core, dtype=np.float32) - np.asarray(legacy, dtype=np.float32)
    abs_diff = np.abs(diff)
    finite_spearman = spearmanr(legacy, core).statistic
    if finite_spearman is None or (isinstance(finite_spearman, float) and math.isnan(finite_spearman)):
        finite_spearman = None
    return {
        "array_equal": bool(np.array_equal(legacy, core)),
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "num_nonzero_diff": int(np.count_nonzero(diff)),
        "spearman": None if finite_spearman is None else float(finite_spearman),
        "top1_same": bool(np.argmax(legacy) == np.argmax(core)) if legacy.size and core.size else True,
        "top5_overlap": _topk_overlap(legacy, core, 5),
        "top10_overlap": _topk_overlap(legacy, core, 10),
        "legacy_top10": np.argsort(-legacy, kind="mergesort")[:10].astype(int).tolist(),
        "core_top10": np.argsort(-core, kind="mergesort")[:10].astype(int).tolist(),
    }


def _build_cautious_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "lr": float(args.cautious_lr),
        "lr_end": float(args.cautious_lr_end),
        "tv_weight": float(args.cautious_tv_weight),
        "irr_weight": float(args.cautious_irr_weight),
        "init_prob": float(args.cautious_init_prob),
        "init_mode": str(args.cautious_init_mode),
        "reg_warmup_frac": float(args.cautious_reg_warmup_frac),
        "restarts": int(args.cautious_restarts),
        "budget_samples": int(args.cautious_budget_samples),
        "select_best": bool(args.cautious_select_best),
        "objective_mode": str(args.cautious_objective_mode),
        "optimizer_mode": str(args.cautious_optimizer_mode),
        "fixed_budget_frac": float(args.cautious_fixed_budget_frac),
        "seed": int(args.seed),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level)),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("compare_fri")

    for pack, root in bench._parse_root_overrides(args.index_root).items():
        bench.mm.INDEX_ROOTS[pack] = root
        log.info("index root override: %s=%s", pack, root)
    for pack, root in bench._parse_root_overrides(args.sae_root).items():
        bench.mm.SAE_ROOTS[pack] = root
        log.info("sae root override: %s=%s", pack, root)

    packs = list(bench.mm.PACK_NAMES) if args.pack == "all" else [str(args.pack)]
    blocks = [int(b) for b in args.blocks]
    manifest_rows = _load_manifest_rows(args.manifest_json)
    cautious_kwargs = _build_cautious_kwargs(args)
    rows: List[Dict[str, Any]] = []
    total_cases = 0
    failed_cases = 0

    log.info("repo=%s", REPO)
    log.info("legacy_module=%s", bench.LEGACY_MM_PATH)
    log.info("device=%s baseline=%s packs=%s blocks=%s", args.device, args.baseline_mode, packs, blocks)
    log.info("cautious_kwargs=%s", cautious_kwargs)

    for pack in packs:
        pack_spec = bench.mm.build_pack_spec(pack)
        log.info("loading model pack=%s", pack)
        model = bench.mm.load_model(pack_spec, args.device)
        model.eval()
        transform = bench.mm.load_transform(pack_spec)
        n_prefix = bench.mm.infer_prefix_count(model)
        log.info("pack=%s prefix_count=%d", pack, n_prefix)

        sae_cache: Dict[int, Any] = {}

        def get_sae(block_idx: int):
            if block_idx not in sae_cache:
                log.info("loading SAE pack=%s block=%d", pack, block_idx)
                sae_cache[block_idx] = bench.mm.load_sae(pack, block_idx, args.device)
                sae_cache[block_idx].eval()
            return sae_cache[block_idx]

        for block_idx in blocks:
            if manifest_rows is None:
                log.info(
                    "sampling triples pack=%s block=%d n_features=%d n_images=%d seed=%d",
                    pack,
                    block_idx,
                    args.n_features,
                    args.n_images,
                    args.seed,
                )
                triples = bench.mm.sample_triples_from_index(
                    pack,
                    block_idx,
                    Path(args.imagenet_val),
                    n_prefix=n_prefix,
                    n_features=int(args.n_features),
                    n_images=int(args.n_images),
                    seed=int(args.seed),
                )
            else:
                triples = [
                    row
                    for row in manifest_rows
                    if str(row.get("pack")) == pack and int(row.get("block_idx", -1)) == int(block_idx)
                ]
            if args.limit is not None:
                triples = triples[: int(args.limit)]
            log.info("pack=%s block=%d cases=%d", pack, block_idx, len(triples))

            sae = get_sae(block_idx)
            for idx, triple in enumerate(triples):
                total_cases += 1
                case_id = {
                    "pack": pack,
                    "block_idx": int(block_idx),
                    "feature_id": int(triple["feature_id"]),
                    "sample_id": int(triple["sample_id"]),
                    "tok_max": int(triple["tok_max"]),
                    "image_path": str(triple.get("image_path", "")),
                }
                log.info("case %d start: %s", total_cases, case_id)
                try:
                    state = bench._build_state_with_baseline(
                        model,
                        sae,
                        triple,
                        transform,
                        str(args.device),
                        n_prefix,
                        str(args.baseline_mode),
                    )
                    n_patches = int(state.h_b0_patches.shape[1])
                    grid = int(round(math.sqrt(n_patches)))
                    init_scores = None
                    if str(args.cautious_init_mode) == "plain_ixg":
                        log.debug("case %d computing plain_ixg init", total_cases)
                        init_scores = np.asarray(bench.mm.run_plain_ixg(state), dtype=np.float32)
                    cfg = bench._fri_config_from_kwargs(
                        cautious_steps=int(args.cautious_steps),
                        cautious_kwargs=cautious_kwargs,
                        init_scores=init_scores,
                    )
                    log.debug("case %d running legacy cautious_cos", total_cases)
                    legacy = bench._run_legacy_cautious_cos(state, n_patches, grid, cfg)
                    log.debug("case %d running core fri", total_cases)
                    core = bench._run_core_fri(state, n_patches, grid, cfg)
                    cmp = _compare_scores(legacy, core)
                    passed = bool(cmp["max_abs_diff"] <= float(args.atol))
                    failed_cases += 0 if passed else 1
                    log.info(
                        "case %d done: passed=%s exact=%s max_abs=%.9g nonzero=%d top1_same=%s spearman=%s",
                        total_cases,
                        passed,
                        cmp["array_equal"],
                        cmp["max_abs_diff"],
                        cmp["num_nonzero_diff"],
                        cmp["top1_same"],
                        cmp["spearman"],
                    )
                    rows.append({**case_id, "passed": passed, **cmp})
                except Exception as exc:
                    failed_cases += 1
                    log.exception("case %d failed: %s", total_cases, case_id)
                    rows.append({**case_id, "passed": False, "error": repr(exc)})

    summary = {
        "total_cases": int(total_cases),
        "failed_cases": int(failed_cases),
        "passed": bool(failed_cases == 0),
        "atol": float(args.atol),
        "config": {
            "pack": str(args.pack),
            "blocks": blocks,
            "n_features": int(args.n_features),
            "n_images": int(args.n_images),
            "baseline_mode": str(args.baseline_mode),
            "device": str(args.device),
            "seed": int(args.seed),
            "cautious_kwargs": cautious_kwargs,
        },
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    log.info("wrote %s", args.output_json)
    log.info("summary: total=%d failed=%d passed=%s", total_cases, failed_cases, failed_cases == 0)
    if failed_cases:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
