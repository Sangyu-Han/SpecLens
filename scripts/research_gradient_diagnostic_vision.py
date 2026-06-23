#!/usr/bin/env python3
"""WHY does gradient/FRI work on VISION but not LLM? Run the SAME gradient diagnostic on CLIP that
gave corr(grad,occlusion)=+0.002 on the LLM. For each uniform patch-mask level a (m=a*ones, patches
interpolated toward the mean-patch baseline): gradient g_i = d(prob)/d(m_i), correlated with the
ACTUAL patch-occlusion effect occ_i = full_prob - prob(patch i -> baseline). If vision's corr is
HIGH (unlike the LLM's ~0) -> the vision gradient is genuinely informative (local ~ occlusion =
the model is more locally-linear / patch contribution additive), explaining why FRI finds the
sufficiency set on vision. Also report recovery at each a (is the masked image in-distribution?)."""
from __future__ import annotations

import argparse
import importlib.util as _ilu
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
FPATH = REPO.parent / "_fri_research_archive_20260610/scripts/research_fri_frontier.py"
_s = _ilu.spec_from_file_location("frontier", FPATH); F = _ilu.module_from_spec(_s); _s.loader.exec_module(F)
N = 196
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
AS = [1.0, 0.95, 0.8, 0.5, 0.3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=4)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    args = ap.parse_args()
    syms = F.load_patch_repo()
    import timm
    model = timm.create_model(args.model, pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    cases = [("zebra", str(REPO / "multi_object_zebra_elephant.jpg"), 340)]
    for i in range(args.n_img):
        r = df.iloc[i * 7]; cases.append((str(r["desc"])[:12], str(r["path"]), int(r["target"])))

    results = []
    for name, path, target in cases:
        try:
            x, _ = syms["load_image"](Path(path)); x = x.to(dev)
            h_b0, cls_tok, baseline = syms["get_b0"](model, x)
            runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
            dtype = runner.dtype
            full = float(runner.prob_curve(np.ones((1, N), np.float32))[0])
            base = float(runner.prob_curve(np.zeros((1, N), np.float32))[0])
            den = max(full - base, 1e-6)
            # occlusion ground truth: occ_i = full - prob(patch i -> baseline)
            occm = np.ones((N, N), np.float32); occm[np.arange(N), np.arange(N)] = 0.0
            occ = full - runner.prob_curve(occm, 64)

            def ins_auc(scores):
                order = np.argsort(-scores)
                aucs = []
                for f in FR:
                    k = int(round(f * N)); m = np.zeros((1, N), np.float32)
                    if k:
                        m[0, order[:k]] = 1.0
                    aucs.append((float(runner.prob_curve(m)[0]) - base) / den)
                return float(np.trapz(aucs, FR) / FR[-1])

            rows = {}
            for a in AS:
                m = torch.full((N,), float(a), device=dev, dtype=dtype, requires_grad=True)
                prob = runner.probs_for_masks(m.unsqueeze(0))[0]
                g = torch.autograd.grad(prob, m)[0].detach().cpu().numpy()
                rec_a = (float(runner.prob_curve(np.full((1, N), float(a), np.float32))[0]) - base) / den
                rows[a] = dict(rec=rec_a, corr=float(spearmanr(g, occ).correlation),
                               gmag=float(np.abs(g).mean()), auc=ins_auc(g))
            results.append(dict(name=name, occ_auc=ins_auc(occ), rows=rows))
            print(f"  {name:12s} full={full:.2f} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} failed: {type(e).__name__}: {e}", flush=True)

    print("\n=== VISION gradient @ uniform patch-mask a: corr with occlusion, recovery, insAUC ===")
    print(f"{'a':>5s} | {'recovery':>8s} | {'corr(grad,occ)':>14s} | {'|grad|':>9s} | {'grad insAUC':>11s}")
    for a in AS:
        rec = np.mean([r["rows"][a]["rec"] for r in results]); corr = np.nanmean([r["rows"][a]["corr"] for r in results])
        gm = np.mean([r["rows"][a]["gmag"] for r in results]); auc = np.mean([r["rows"][a]["auc"] for r in results])
        print(f"{a:5.2f} | {rec:8.3f} | {corr:+14.3f} | {gm:9.2e} | {auc:11.3f}")
    print(f"\nocclusion-order insAUC (ceiling) = {np.mean([r['occ_auc'] for r in results]):.3f}")
    print(f"=== COMPARE to LLM: LLM corr(grad@1.0, occ) = +0.002 (uninformative). VISION corr@1.0 = "
          f"{np.nanmean([r['rows'][1.0]['corr'] for r in results]):+.3f} ===")
    print("(if VISION corr >> LLM +0.002 -> the vision gradient IS informative = why FRI finds the sufficiency set on vision)")


if __name__ == "__main__":
    main()
