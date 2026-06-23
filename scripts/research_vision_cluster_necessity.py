#!/usr/bin/env python3
"""CHEAP NECESSITY by exploiting 'the model already knows necessity' — its hidden representation
encodes which patches are REDUNDANT (similar patches that substitute for each other = same cluster).
Redundancy is exactly what makes necessity expensive (must remove ALL copies of a redundant group);
if the model's hidden states reveal the groups, we remove them GROUP-WISE -> resolve redundancy at
~G^2 cost (G clusters) instead of patch-level O(N^2). cluster-greedy: hidden (1 fwd) -> k-means G
clusters -> greedy remove whole clusters; within-cluster order by hidden value-norm. RAW del [0,1].
Compare vs patch-greedy (gold), chunk_bz (~982), banzhaf, inflow."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, banzhaf_pred, greedy_pred
from research_vision_cheap_necessity import chunked_with_prior
from research_vision_raw_metric import raw_auc

REPO = Path(__file__).resolve().parents[1]


def kmeans(X, G, iters=25, seed=0):
    gen = np.random.default_rng(seed); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    c = X[gen.choice(len(X), G, replace=False)].copy()
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None] - c[None]) ** 2).sum(-1); lab = d.argmin(1)
        for g in range(G):
            if (lab == g).any():
                c[g] = X[lab == g].mean(0)
    return lab


def cluster_greedy(runner, patch_H, G=20):
    """k-means clusters of hidden patch reps -> greedy remove WHOLE clusters -> patch order.
    cost = 1 (hidden, external) + ~G^2/2 forwards."""
    lab = kmeans(patch_H, G)
    norm = np.linalg.norm(patch_H, axis=1)                       # within-cluster order by value-norm
    clusters = [np.where(lab == g)[0] for g in range(G) if (lab == g).any()]
    km = np.ones(N, np.float32); order = []; rem = list(range(len(clusters)))
    while rem:
        masks = []
        for ci in rem:
            m = km.copy(); m[clusters[ci]] = 0.0; masks.append(m)
        probs = np.asarray(runner.prob_curve(np.stack(masks), CHUNK), np.float32)
        j = int(np.argmin(probs)); ci = rem[j]
        pats = sorted(clusters[ci].tolist(), key=lambda p: -norm[p])   # high-norm first within cluster
        for p in pats:
            km[p] = 0.0; order.append(int(p))
        rem.pop(j)
    return _rank(order)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=5); ap.add_argument("--G", type=int, default=20)
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
            feats = {}
            hh = model.blocks[-1].register_forward_hook(lambda m, inp, o: feats.__setitem__("h", o.detach()))
            with torch.no_grad():
                model(x)
            hh.remove()
            H = feats["h"][0]; npref = H.shape[0] - N; patch_H = H[npref:].float().cpu().numpy()  # [196,D]
            h_b0, cls_tok, baseline = syms["get_b0"](model, x)
            runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
            clg = cluster_greedy(runner, patch_H, args.G)
            g_rank, _ = greedy_pred(runner)
            bz = banzhaf_pred(runner, M=512); chunk = chunked_with_prior(runner, bz, M=100, R=8)
            try:
                inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
            except Exception:
                inflow = np.zeros(N, np.float32)
            d = {}
            for m, s in [("cluster", clg), ("greedy", g_rank), ("chunk_bz", chunk), ("banzhaf", bz), ("inflow", inflow)]:
                _, dele = raw_auc(runner, s); d[m] = round(dele, 4)
            rows.append(d)
            print(f"  {mp.split('.')[0][-10:]} {i} | cluster {d['cluster']:.3f} greedy {d['greedy']:.3f} "
                  f"chunk_bz {d['chunk_bz']:.3f} banzhaf {d['banzhaf']:.3f} INFLOW {d['inflow']:.3f}", flush=True)
        allr[mp] = rows
    print(f"\n=== CLUSTER-GREEDY necessity (model-hidden redundancy groups), RAW del [0,1] ===")
    print(f"   cost: cluster ~={args.G*(args.G+1)//2}+1fwd | chunk_bz ~=982 | greedy ~={N*(N+1)//2} | inflow 1")
    for mp, rows in allr.items():
        if not rows:
            continue
        a = lambda k: float(np.mean([r[k] for r in rows]))
        cg = 100.0 * np.mean([r["cluster"] <= r["chunk_bz"] for r in rows]); ci = 100.0 * np.mean([r["cluster"] < r["inflow"] for r in rows])
        print(f"  {mp.split('/')[-1][:24]:24s} n={len(rows)} | cluster {a('cluster'):.3f} greedy {a('greedy'):.3f} "
              f"chunk_bz {a('chunk_bz'):.3f} banzhaf {a('banzhaf'):.3f} INFLOW {a('inflow'):.3f} | clust≤chunk_bz {cg:.0f}% clust<inflow {ci:.0f}%")
    print("(KEY: cluster del ≈ greedy at ~210 fwd? = model's hidden redundancy-groups give cheap necessity)")
    json.dump(allr, open(REPO / "outputs/vision_cluster_necessity.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
