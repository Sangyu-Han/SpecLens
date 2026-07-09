#!/usr/bin/env python3
"""CHEAPER NECESSITY via ADAPTIVE-granularity conditional deletion. Full greedy resolves redundancy
but costs O(N^2) (~19k fwd). Fixed-chunk conditional is cheap (~500) but coarse (granularity gap to
greedy). KEY: the deletion AUC is dominated by the EARLY removals (steep drop at small-k), so spend
precision (per=1, greedy) on the first fine_k removals and go COARSE afterwards. Combined with the
cooperative-Banzhaf prior (captures redundant supporters that single-occ misses). Track ACTUAL
forwards. Vision (CLIP), greedy del=gold. Question: does adaptive approach greedy at << O(N^2)?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, banzhaf_pred
from vision_metric_utils import raw_auc_from_hard_curves

REPO = Path(__file__).resolve().parents[1]


def single_occ_prior(runner):
    occ = np.ones((N, N), np.float32); occ[np.arange(N), np.arange(N)] = 0.0
    full = float(runner.prob_curve(np.ones((1, N), np.float32))[0])
    return full - np.asarray(runner.prob_curve(occ, CHUNK), np.float32), N


def greedy_full(runner):
    km = np.ones(N, np.float32); order = []; rem = list(range(N)); nf = 0
    while rem:
        masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
        probs = np.asarray(runner.prob_curve(masks, CHUNK), np.float32); nf += len(rem)
        j = int(np.argmin(probs)); km[rem[j]] = 0.0; order.append(rem[j]); rem.pop(j)
    return _rank(order), nf


def cond_prior(runner, prior, M=100, fine_k=0, coarse_per=12):
    """conditional greedy on top-M candidates; per=1 for the first fine_k removals, then coarse_per."""
    cand = [int(i) for i in np.argsort(-prior)][:M]
    km = np.ones(N, np.float32); order = []; rem = list(cand); nf = 0
    while rem:
        per = 1 if len(order) < fine_k else coarse_per
        masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
        probs = np.asarray(runner.prob_curve(masks, CHUNK), np.float32); nf += len(rem)
        idx = list(np.argsort(probs)[:per])
        for j in idx:
            km[rem[j]] = 0.0; order.append(rem[j])
        for j in sorted(idx, reverse=True):
            rem.pop(j)
    order += [int(i) for i in np.argsort(-prior) if int(i) not in set(order)]
    return _rank(order), nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=8)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--Mbz", type=int, default=512)
    args = ap.parse_args()
    syms = F.load_patch_repo()
    import timm
    print(f"[model] {args.model}", flush=True)
    model = timm.create_model(args.model, pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    cases = [("zebra", str(REPO / "multi_object_zebra_elephant.jpg"), 340)]
    for i in range(args.n_img):
        r = df.iloc[i * 7]; cases.append((str(r["desc"])[:12], str(r["path"]), int(r["target"])))
    # (method, prior, fine_k, coarse_per)
    CFG = [("greedy", None, None, None),
           ("chunk_bz", "bz", 0, 12),
           ("adapt_bz_lite", "bz", 8, 12),
           ("adapt_bz", "bz", 20, 10),
           ("adapt_occ", "occ", 20, 10)]
    rows = []
    for name, path, target in cases:
        x, _ = syms["load_image"](Path(path)); x = x.to(args.device)
        h_b0, cls_tok, baseline = syms["get_b0"](model, x)
        runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
        pr_occ, c_occ = single_occ_prior(runner)
        bz = banzhaf_pred(runner, M=args.Mbz)
        res = {"case": name}; cost = {}
        for m, prior, fk, cp in CFG:
            if m == "greedy":
                rank, nf = greedy_full(runner); cost[m] = nf
            else:
                prior_arr = bz if prior == "bz" else pr_occ
                prior_cost = args.Mbz if prior == "bz" else c_occ
                rank, nf = cond_prior(runner, prior_arr, args.M, fk, cp); cost[m] = nf + prior_cost
            _, dele = raw_auc_from_hard_curves(F, runner, rank, n_patches=N, chunk=CHUNK)
            res[m] = round(float(dele), 4)
        rows.append(res)
        print(f"{name:12s} | " + " ".join(f"{m}:d{res[m]:.3f}" for m, _, _, _ in CFG), flush=True)
    print("\n=== MEAN raw-prob deletion AUC (LOWER=better) + COST (actual forwards) ===")
    for m, _, _, _ in CFG:
        d = float(np.mean([r[m] for r in rows]))
        print(f"  {m:14s} del={d:.3f}  cost~={cost[m]:6d}")
    print("(KEY: adapt_bz between chunk_bz and greedy in quality, at << greedy cost? does fine-early close the gap?)")
    Path(REPO / "outputs/adaptive_necessity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
