#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import scripts.run_feature_erf_paper_benchmark as bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect one ERF case under two baseline contracts.")
    p.add_argument("--pack", required=True, choices=list(bench.mm.PACK_NAMES))
    p.add_argument("--block", type=int, required=True)
    p.add_argument("--feature-id", type=int, required=True)
    p.add_argument("--sample-id", type=int, required=True)
    p.add_argument("--tok-max", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imagenet-val", type=Path, default=bench.mm.IMAGENET_VAL)
    p.add_argument("--index-root", action="append", default=[])
    p.add_argument("--sae-root", action="append", default=[])
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--cautious-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
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


def main() -> None:
    args = parse_args()
    for pack, root in bench._parse_root_overrides(args.index_root).items():
        bench.mm.INDEX_ROOTS[pack] = root
    for pack, root in bench._parse_root_overrides(args.sae_root).items():
        bench.mm.SAE_ROOTS[pack] = root

    spec = bench.mm.build_pack_spec(args.pack)
    model = bench.mm.load_model(spec, args.device)
    transform = bench.mm.load_transform(spec)
    sae = bench.mm.load_sae(args.pack, int(args.block), args.device)
    n_prefix = bench.mm.infer_prefix_count(model)
    val_root = Path(args.imagenet_val)

    triples = bench.mm.sample_triples_from_index(
        args.pack,
        int(args.block),
        val_root,
        n_prefix=n_prefix,
        n_features=150,
        n_images=2,
        seed=int(args.seed),
    )
    triple = None
    for t in triples:
        if (
            int(t["feature_id"]) == int(args.feature_id)
            and int(t["sample_id"]) == int(args.sample_id)
            and int(t["tok_max"]) == int(args.tok_max)
        ):
            triple = t
            break
    if triple is None:
        raise RuntimeError("Target triple not found in sampled manifest")

    budgets = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    out = {"triple": {k: triple[k] for k in ("pack", "block_idx", "feature_id", "sample_id", "tok_max", "image_path")}}
    cached_scores = {}
    target_patch = int(triple["tok_max"])
    out["triple"]["target_patch"] = target_patch

    baselines = (str(args.baseline_a), str(args.baseline_b))
    for baseline in baselines:
        state = bench._build_state_with_baseline(model, sae, triple, transform, args.device, n_prefix, baseline)
        n_patches = int(state.h_b0_patches.shape[1])
        grid = int(round(math.sqrt(n_patches)))
        cautious = np.asarray(
            bench.mm.run_cautious_cos(
                state,
                n_patches,
                grid,
                steps=int(args.cautious_steps),
                lr=0.45,
                lr_end=0.01,
                tv_weight=0.01,
                irr_weight=0.05,
                init_prob=0.5,
                seed=int(args.seed),
            ),
            dtype=np.float32,
        )
        plain = np.asarray(bench.mm.run_plain_ixg(state), dtype=np.float32)
        cached_scores[baseline] = cautious.copy()
        cautious_norm = bench.mm.normalize_scores(cautious)
        plain_norm = bench.mm.normalize_scores(plain)
        cautious_order = np.argsort(-cautious_norm)
        plain_order = np.argsort(-plain_norm)

        def _rank_of(order: np.ndarray, idx: int) -> int:
            match = np.where(order == idx)[0]
            if len(match) == 0:
                return -1
            return int(match[0]) + 1

        def curve(scores: np.ndarray):
            order = np.argsort(-bench.mm.normalize_scores(scores))
            full = bench._feature_response(
                state,
                bench.mm._run_injected(state.model, state.x, state.prefix_tokens, state.h_b0_patches, state.block_idx),
            )
            full = max(abs(full), 1e-8)
            vals = []
            for frac in budgets:
                k = max(1, min(n_patches, int(round(frac * n_patches))))
                idx = order[:k]
                h_cur = state.baseline.clone()
                h_cur[0, idx] = state.h_b0_patches[0, idx]
                block = bench.mm._run_injected(state.model, state.x, state.prefix_tokens, h_cur, state.block_idx)
                vals.append(float(bench._feature_response(state, block) / full))
            return vals

        metrics = bench._evaluate_scores(state, cautious, stoch_steps=20, stoch_samples=5, stoch_seeds=[0, 1, 2])
        out[baseline] = {
            "cautious_stats": {
                "mean": float(cautious.mean()),
                "std": float(cautious.std()),
                "min": float(cautious.min()),
                "max": float(cautious.max()),
            },
            "plain_stats": {
                "mean": float(plain.mean()),
                "std": float(plain.std()),
                "min": float(plain.min()),
                "max": float(plain.max()),
            },
            "cautious_top10": cautious_order[:10].tolist(),
            "plain_top10": plain_order[:10].tolist(),
            "cautious_target_patch_rank": _rank_of(cautious_order, target_patch),
            "plain_target_patch_rank": _rank_of(plain_order, target_patch),
            "cautious_target_patch_score": float(cautious_norm[target_patch]),
            "plain_target_patch_score": float(plain_norm[target_patch]),
            "cautious_curve": curve(cautious),
            "plain_curve": curve(plain),
            "cautious_metrics": {
                "stoch_ins_delta": float(metrics["stoch_ins_delta"]),
                "insertion_auc": float(metrics["insertion_auc"]),
                "mas_ins_auc": float(metrics["mas_ins_auc"]),
                "rep_idsds": float(metrics["rep_idsds"]),
            },
        }

    a, b = baselines
    out["compare"] = {
        "baseline_a": a,
        "baseline_b": b,
        "cautious_top10_overlap": int(len(set(out[a]["cautious_top10"]) & set(out[b]["cautious_top10"]))),
        "plain_top10_overlap": int(len(set(out[a]["plain_top10"]) & set(out[b]["plain_top10"]))),
        "cautious_score_spearman": float(spearmanr(cached_scores[a], cached_scores[b]).statistic),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
