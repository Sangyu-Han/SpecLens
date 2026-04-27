#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import scripts.run_feature_erf_paper_benchmark as bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose cautious baseline sensitivity on a tiny multimodel slice.")
    p.add_argument("--pack", default="all", choices=list(bench.mm.PACK_NAMES) + ["all"])
    p.add_argument("--blocks", nargs="*", type=int, default=list(bench.mm.BLOCK_IDXS))
    p.add_argument("--n-features", type=int, default=1)
    p.add_argument("--n-images", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imagenet-val", type=Path, default=bench.mm.IMAGENET_VAL)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--index-root", action="append", default=[])
    p.add_argument("--sae-root", action="append", default=[])
    p.add_argument("--cautious-steps", type=int, default=32)
    p.add_argument("--cautious-lr", type=float, default=0.45)
    p.add_argument("--cautious-lr-end", type=float, default=0.01)
    p.add_argument("--cautious-tv-weight", type=float, default=0.01)
    p.add_argument("--cautious-irr-weight", type=float, default=0.05)
    p.add_argument("--cautious-init-prob", type=float, default=0.5)
    p.add_argument("--stoch-steps", type=int, default=20)
    p.add_argument("--stoch-samples", type=int, default=5)
    p.add_argument("--stoch-seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument(
        "--baseline-a",
        choices=("default", "zero", "mean_image", "global_mean_h", "global_mean_h_plus_pos"),
        default="default",
    )
    p.add_argument(
        "--baseline-b",
        choices=("default", "zero", "mean_image", "global_mean_h", "global_mean_h_plus_pos"),
        default="global_mean_h_plus_pos",
    )
    return p.parse_args()


def _entropy01(scores: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = np.maximum(s, 0.0)
    total = float(s.sum())
    if total <= 1e-12:
        return 1.0
    p = s / total
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent / math.log(len(p)))


def _score_stats(scores: np.ndarray) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    return {
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        "score_min": float(np.min(s)),
        "score_max": float(np.max(s)),
        "score_entropy01": _entropy01(s),
    }


def _feature_obj(state: bench.mm.SAEState, w: torch.Tensor) -> float:
    with torch.no_grad():
        state.do_forward_masked(w)
        return float(state.objective_getter_single().item())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("diagnose_cautious_baseline")
    args = parse_args()

    packs = list(bench.mm.PACK_NAMES) if args.pack == "all" else [args.pack]
    blocks = list(args.blocks or bench.mm.BLOCK_IDXS)
    val_root = Path(args.imagenet_val)

    for pack, root in bench._parse_root_overrides(args.index_root).items():
        bench.mm.INDEX_ROOTS[pack] = root
    for pack, root in bench._parse_root_overrides(args.sae_root).items():
        bench.mm.SAE_ROOTS[pack] = root

    out_rows: List[Dict[str, Any]] = []

    for pack in packs:
        spec = bench.mm.build_pack_spec(pack)
        model = bench.mm.load_model(spec, args.device)
        transform = bench.mm.load_transform(spec)
        n_prefix = bench.mm.infer_prefix_count(model)
        log.info("[%s] prefix_count=%d", pack, n_prefix)
        sae_cache: Dict[int, Any] = {}

        def get_sae(block_idx: int):
            if block_idx not in sae_cache:
                sae_cache[block_idx] = bench.mm.load_sae(pack, block_idx, args.device)
            return sae_cache[block_idx]

        for block_idx in blocks:
            triples = bench.mm.sample_triples_from_index(
                pack,
                block_idx,
                val_root,
                n_prefix=n_prefix,
                n_features=int(args.n_features),
                n_images=int(args.n_images),
                seed=int(args.seed),
            )
            if not triples:
                continue
            sae = get_sae(block_idx)
            for triple in triples:
                row: Dict[str, Any] = {
                    "pack": pack,
                    "block_idx": int(block_idx),
                    "feature_id": int(triple["feature_id"]),
                    "sample_id": int(triple["sample_id"]),
                    "tok_max": int(triple["tok_max"]),
                }
                per_baseline: Dict[str, Any] = {}
                baselines = (str(args.baseline_a), str(args.baseline_b))
                for baseline_mode in baselines:
                    state = bench._build_state_with_baseline(
                        model, sae, triple, transform, args.device, n_prefix, baseline_mode
                    )
                    n_patches = int(state.h_b0_patches.shape[1])
                    grid = int(round(math.sqrt(n_patches)))
                    ones = torch.ones(n_patches, device=state.x.device, dtype=state.h_b0_patches.dtype)
                    zeros = torch.zeros_like(ones)
                    half = torch.full_like(ones, 0.5)

                    act_full = _feature_obj(state, ones)
                    act_base = _feature_obj(state, zeros)
                    act_half = _feature_obj(state, half)
                    full_safe = max(abs(act_full), 1e-8)

                    scores = bench.mm.run_cautious_cos(
                        state,
                        n_patches,
                        grid,
                        steps=int(args.cautious_steps),
                        lr=float(args.cautious_lr),
                        lr_end=float(args.cautious_lr_end),
                        tv_weight=float(args.cautious_tv_weight),
                        irr_weight=float(args.cautious_irr_weight),
                        init_prob=float(args.cautious_init_prob),
                        seed=int(args.seed),
                    )
                    scores = np.asarray(scores, dtype=np.float32)
                    metrics = bench._evaluate_scores(
                        state,
                        scores,
                        stoch_steps=int(args.stoch_steps),
                        stoch_samples=int(args.stoch_samples),
                        stoch_seeds=[int(s) for s in args.stoch_seeds],
                    )

                    per_baseline[baseline_mode] = {
                        "act_full": act_full,
                        "act_baseline": act_base,
                        "act_uniform05": act_half,
                        "baseline_ratio": float(act_base / full_safe),
                        "uniform05_ratio": float(act_half / full_safe),
                        **_score_stats(scores),
                        "metrics": {
                            "stoch_ins_delta": float(metrics["stoch_ins_delta"]),
                            "insertion_auc": float(metrics["insertion_auc"]),
                            "mas_ins_auc": float(metrics["mas_ins_auc"]),
                            "rep_idsds": float(metrics["rep_idsds"]),
                        },
                    }

                a, bname = baselines
                b = per_baseline[a]
                d = per_baseline[bname]
                row[a] = b
                row[bname] = d
                row["compare"] = {
                    "baseline_a": a,
                    "baseline_b": bname,
                    "baseline_ratio": float(d["baseline_ratio"] - b["baseline_ratio"]),
                    "uniform05_ratio": float(d["uniform05_ratio"] - b["uniform05_ratio"]),
                    "score_std": float(d["score_std"] - b["score_std"]),
                    "score_entropy01": float(d["score_entropy01"] - b["score_entropy01"]),
                    "stoch_ins_delta": float(d["metrics"]["stoch_ins_delta"] - b["metrics"]["stoch_ins_delta"]),
                    "insertion_auc": float(d["metrics"]["insertion_auc"] - b["metrics"]["insertion_auc"]),
                    "mas_ins_auc": float(d["metrics"]["mas_ins_auc"] - b["metrics"]["mas_ins_auc"]),
                    "rep_idsds": float(d["metrics"]["rep_idsds"] - b["metrics"]["rep_idsds"]),
                }
                out_rows.append(row)
                log.info(
                    "[%s blk=%d feat=%d sid=%d] default ins=%.4f mean ins=%.4f default std=%.4f mean std=%.4f",
                    pack,
                    block_idx,
                    row["feature_id"],
                    row["sample_id"],
                    b["metrics"]["insertion_auc"],
                    d["metrics"]["insertion_auc"],
                    b["score_std"],
                    d["score_std"],
                )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": out_rows}, indent=2))


if __name__ == "__main__":
    main()
