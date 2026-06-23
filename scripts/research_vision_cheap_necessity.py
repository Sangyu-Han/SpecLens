#!/usr/bin/env python3
"""RESOLVING REDUNDANCY CHEAPLY (vision necessity), using the LLM-derived cooperative-Banzhaf idea.
Full greedy resolves redundancy but costs O(N^2) (~19k fwd). The cheap chunked-conditional uses a
SINGLE-OCC prior, which MISSES redundant patches (removing one backup does not hurt -> single-occ~0).
NEW: use the cooperative-Banzhaf marginal as the prior (it averages over coalitions, so a redundant
patch shows positive marginal in the coalitions where its primary is absent) -> the candidate set
COVERS the redundant supporters -> chunked-greedy on those candidates resolves redundancy at ~20x
lower cost. Compare deletion AUC (LOWER=better necessity) + cost: greedy / chunked-singleocc /
chunked-banzhaf / raw-banzhaf, on CLIP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, banzhaf_pred, greedy_pred

REPO = Path(__file__).resolve().parents[1]


def single_occ_prior(runner):
    occ = np.ones((N, N), np.float32); occ[np.arange(N), np.arange(N)] = 0.0
    full = float(runner.prob_curve(np.ones((1, N), np.float32))[0])
    return full - np.asarray(runner.prob_curve(occ, CHUNK), np.float32)        # cost N


def chunked_with_prior(runner, prior, M=100, R=8):
    """chunked-conditional greedy restricted to the top-M candidates by `prior`. cost ~ M*(R+1)/2."""
    cand = [int(i) for i in np.argsort(-prior)][:M]
    km = np.ones(N, np.float32); order = []; rem = list(cand); per = max(1, M // R)
    while rem:
        masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
        probs = np.asarray(runner.prob_curve(masks, CHUNK), np.float32)
        idx = list(np.argsort(probs)[:per])
        for j in idx:
            km[rem[j]] = 0.0; order.append(rem[j])
        for j in sorted(idx, reverse=True):
            rem.pop(j)
    order += [int(i) for i in np.argsort(-prior) if int(i) not in set(order)]
    return _rank(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=8)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--Mbz", type=int, default=512)
    ap.add_argument("--multi-target", type=int, default=340)      # 340=zebra, 386=African elephant
    ap.add_argument("--multi-name", default="zebra")
    args = ap.parse_args()
    syms = F.load_patch_repo()
    import timm
    print(f"[model] {args.model}", flush=True)
    model = timm.create_model(args.model, pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    cases = [(args.multi_name, str(REPO / "multi_object_zebra_elephant.jpg"), args.multi_target)]
    for i in range(args.n_img):
        r = df.iloc[i * 7]; cases.append((str(r["desc"])[:12], str(r["path"]), int(r["target"])))
    rows = []
    for name, path, target in cases:
        x, _ = syms["load_image"](Path(path)); x = x.to(args.device)
        h_b0, cls_tok, baseline = syms["get_b0"](model, x)
        runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
        g_rank, k = greedy_pred(runner)
        prior_occ = single_occ_prior(runner)
        bz = banzhaf_pred(runner, M=args.Mbz)
        sc = {"greedy": g_rank,
              "chunked_occ": chunked_with_prior(runner, prior_occ, args.M, args.R),
              "chunked_bz": chunked_with_prior(runner, bz, args.M, args.R),
              "banzhaf": bz}
        cost = {"greedy": N * (N + 1) // 2, "chunked_occ": N + args.M * (args.R + 1) // 2,
                "chunked_bz": args.Mbz + args.M * (args.R + 1) // 2, "banzhaf": args.Mbz}
        res = {"case": name, "k": int(k)}
        for m, s in sc.items():
            _, dele, _, _ = F.hard_curves(runner, s); res[m + "_del"] = round(float(dele), 4)
        rows.append(res)
        print(f"{name:12s} k={k:3d} | greedy:d{res['greedy_del']:.3f} chunked_occ:d{res['chunked_occ_del']:.3f} "
              f"chunked_bz:d{res['chunked_bz_del']:.3f} banzhaf:d{res['banzhaf_del']:.3f}", flush=True)
    print("\n=== MEAN deletion AUC (LOWER=better necessity) + COST (forwards) ===")
    for m in ["greedy", "chunked_occ", "chunked_bz", "banzhaf"]:
        d = float(np.mean([r[m + "_del"] for r in rows]))
        print(f"  {m:12s} del={d:.3f}  cost~={cost[m]:6d}")
    print("(KEY: does chunked_bz < chunked_occ and approach greedy, at ~20x lower cost than greedy?)")
    Path(REPO / "outputs/vision_cheap_necessity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
