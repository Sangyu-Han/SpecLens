#!/usr/bin/env python3
"""CHEAP NECESSITY via SURROGATE-GREEDY (reuse the cooperative-Banzhaf coalitions; LLM-derived).
Full greedy is O(N^2) because it re-evaluates conditionally each round. But the Banzhaf already samples
M coalitions (varied keep-masks + recoveries). Fit a 2nd-order surrogate R ≈ a0 + a·k + kᵀBk over the
top-K candidates, then run greedy DELETION ON THE SURROGATE (predicted R-drop, conditional via Bk) with
ZERO extra forwards. Cost = M (the Banzhaf samples) only — no chunked rounds. Does surrogate-greedy del
≈ true greedy at ~M cost (cheaper than chunk_bz ~980, far cheaper than greedy ~19k)? RAW-prob del [0,1]."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, greedy_pred
from research_vision_cheap_necessity import chunked_with_prior
from research_vision_raw_metric import raw_auc

REPO = Path(__file__).resolve().parents[1]


def coalitions(runner, M, seed=0):
    gen = np.random.default_rng(seed)
    pf = gen.random((M, 1)); Z = (gen.random((M, N)) < pf).astype(np.float32)
    R = np.asarray(runner.prob_curve(Z, CHUNK), np.float32)
    n1 = Z.sum(0)
    marg = (Z * R[:, None]).sum(0) / np.clip(n1, 1, None) - ((1 - Z) * R[:, None]).sum(0) / np.clip(M - n1, 1, None)
    return Z, R, marg


def surrogate_order(Z, R, cand, lam=2.0):
    """fit R ≈ a0 + a·k + kᵀBk on candidate keep-cols; greedy-remove by predicted conditional R-drop."""
    Zc = Z[:, cand]; K = len(cand); iu = np.triu_indices(K, 1)
    pair = Zc[:, iu[0]] * Zc[:, iu[1]]
    X = np.concatenate([np.ones((len(R), 1)), Zc, pair], 1)
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ R)
    a = w[1:1 + K]; B = np.zeros((K, K)); B[iu] = w[1 + K:]; B = B + B.T
    k = np.ones(K); order = []; rem = set(range(K))
    for _ in range(K):
        drop = a + B @ k                                    # predicted R-drop of removing each i (conditional via Bk)
        d = drop.copy()
        for i in range(K):
            if i not in rem:
                d[i] = -1e18
        i = int(np.argmax(d)); k[i] = 0.0; order.append(int(cand[i])); rem.discard(i)
    return order


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=5); ap.add_argument("--M", type=int, default=512); ap.add_argument("--K", type=int, default=28)
    args = ap.parse_args()
    import timm
    syms = F.load_patch_repo()
    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    MODELS = ["vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", "vit_base_patch16_224.augreg2_in21k_ft_in1k"]
    allr = {}
    for mp in MODELS:
        model = timm.create_model(mp, pretrained=True).eval().to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        rows = []
        for i in range(args.n_img):
            r = df.iloc[i * 7]; path, target = str(r["path"]), int(r["target"])
            try:
                x, _ = syms["load_image"](Path(path)); x = x.to(args.device)
            except Exception:
                continue
            h_b0, cls_tok, baseline = syms["get_b0"](model, x)
            runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
            Z, R, marg = coalitions(runner, args.M)
            cand = [int(j) for j in np.argsort(-marg)][:args.K]
            sur_order = surrogate_order(Z, R, cand)
            sur_full = _rank(sur_order + [int(j) for j in np.argsort(-marg) if int(j) not in set(sur_order)])
            g_rank, _ = greedy_pred(runner)
            chunk = chunked_with_prior(runner, marg, M=100, R=8)
            try:
                inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
            except Exception:
                inflow = np.zeros(N, np.float32)
            d = {}
            for m, s in [("surrogate", sur_full), ("greedy", g_rank), ("chunk_bz", chunk), ("banzhaf", marg), ("inflow", inflow)]:
                _, dele = raw_auc(runner, s); d[m] = round(dele, 4)
            rows.append(d)
            print(f"  {mp.split('.')[0][-10:]} {i} | surrogate {d['surrogate']:.3f} greedy {d['greedy']:.3f} "
                  f"chunk_bz {d['chunk_bz']:.3f} banzhaf {d['banzhaf']:.3f} INFLOW {d['inflow']:.3f}", flush=True)
        allr[mp] = rows
    print("\n=== SURROGATE-GREEDY cheap necessity, RAW del [0,1] (LOWER=better) ===")
    print(f"   cost: surrogate ~={args.M} fwd (reused) | chunk_bz ~={args.M+470} | greedy ~={N*(N+1)//2} | inflow 1")
    for mp, rows in allr.items():
        if not rows:
            continue
        a = lambda k: float(np.mean([r[k] for r in rows]))
        sg = 100.0 * np.mean([r["surrogate"] <= r["chunk_bz"] for r in rows])
        si = 100.0 * np.mean([r["surrogate"] < r["inflow"] for r in rows])
        print(f"  {mp.split('/')[-1][:24]:24s} n={len(rows)} | surrogate {a('surrogate'):.3f} greedy {a('greedy'):.3f} "
              f"chunk_bz {a('chunk_bz'):.3f} banzhaf {a('banzhaf'):.3f} INFLOW {a('inflow'):.3f} | sur≤chunk_bz {sg:.0f}% sur<inflow {si:.0f}%")
    print("(KEY: surrogate del ≈ greedy/chunk_bz at ~512 fwd & ZERO extra rounds? = cheap necessity from reused coalitions)")
    json.dump(allr, open(REPO / "outputs/vision_surrogate_necessity.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
