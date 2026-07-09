#!/usr/bin/env python3
"""IG-LEVEL (~32-64 fwd/bwd) vs INFLOW, BOTH faces, with an APPROPRIATE PRIOR.
- SUFFICIENCY (ins, HIGHER=better): does Banzhaf M=32/64 beat inflow insertion (0.936)? (FRI's home.)
- NECESSITY (del, LOWER=better): the gradcond failed because |grad| candidates MISS the necessary
  patches. KEY IDEA: use INFLOW itself as the prior (1 fwd, already ranks necessary patches well) ->
  top-K candidates -> short conditional greedy that REFINES the early order (which dominates del AUC).
  Cost ~ 1 (inflow) + ~30 (greedy). Does inflow-prior conditional beat plain inflow cheaply?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, banzhaf_pred, grad_pred, ig_pred
from vision_metric_utils import raw_auc_from_hard_curves

REPO = Path(__file__).resolve().parents[1]


def cond_with_prior(runner, prior, n_fwd, K):
    """top-K by `prior` -> per=1 greedy within n_fwd forward budget; rest by `prior`."""
    cand = [int(i) for i in np.argsort(-prior)][:K]
    km = np.ones(N, np.float32); order = []; rem = list(cand); used = 0
    while rem and used + len(rem) <= n_fwd:
        masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
        probs = np.asarray(runner.prob_curve(masks, CHUNK), np.float32); used += len(rem)
        j = int(np.argmin(probs)); km[rem[j]] = 0.0; order.append(rem[j]); rem.pop(j)
    order += [int(i) for i in np.argsort(-prior) if int(i) not in set(order)]
    return _rank(order), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=8)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
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
    rows = []
    for name, path, target in cases:
        x, _ = syms["load_image"](Path(path)); x = x.to(args.device)
        h_b0, cls_tok, baseline = syms["get_b0"](model, x)
        runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
        try:
            inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
        except Exception:
            inflow = np.zeros(N, np.float32)
        gc32, _ = cond_with_prior(runner, grad_pred(runner), 31, K=8)         # |grad| prior (1 bwd)
        ic32, u_i32 = cond_with_prior(runner, inflow, 31, K=8)                # INFLOW prior
        ic64, u_i64 = cond_with_prior(runner, inflow, 60, K=12)              # INFLOW prior, bigger budget
        sc = {"inflow": (inflow, 1), "bz32": (banzhaf_pred(runner, M=32), 32),
              "bz64": (banzhaf_pred(runner, M=64), 64), "gradcond32": (gc32, 32),
              "inflowcond32": (ic32, u_i32 + 1), "inflowcond64": (ic64, u_i64 + 1)}
        res = {"case": name}; cost = {}
        for m, (s, c) in sc.items():
            ins, dele = raw_auc_from_hard_curves(F, runner, s, n_patches=N, chunk=CHUNK)
            res[m + "_del"] = round(float(dele), 4); res[m + "_ins"] = round(float(ins), 4); cost[m] = c
        rows.append(res)
        print(f"{name:12s} | DEL inflow {res['inflow_del']:.3f} inflowcond32 {res['inflowcond32_del']:.3f} "
              f"inflowcond64 {res['inflowcond64_del']:.3f} bz64 {res['bz64_del']:.3f} | INS inflow {res['inflow_ins']:.3f} "
              f"bz32 {res['bz32_ins']:.3f} bz64 {res['bz64_ins']:.3f}", flush=True)
    print("\n=== IG-LEVEL vs INFLOW RAW-PROB AUC (DEL ↓ necessity / INS ↑ sufficiency; ~32-64 budget) ===")
    for m in ["inflow", "bz32", "bz64", "gradcond32", "inflowcond32", "inflowcond64"]:
        d = float(np.mean([r[m + "_del"] for r in rows])); i = float(np.mean([r[m + "_ins"] for r in rows]))
        wd = "" if m == "inflow" else f" delwin {100.0*np.mean([r[m+'_del'] < r['inflow_del'] for r in rows]):.0f}%"
        wi = "" if m == "inflow" else f" inswin {100.0*np.mean([r[m+'_ins'] > r['inflow_ins'] for r in rows]):.0f}%"
        print(f"  {m:13s} del={d:.3f} ins={i:.3f} cost~={cost[m]:3d}{wd}{wi}")
    print("(inflow del=0.314 ins=0.936 @1fwd — sufficiency: bz beats ins? necessity: inflow-prior conditional beats del cheaply?)")
    Path(REPO / "outputs/vision_iglevel_necessity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
