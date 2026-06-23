#!/usr/bin/env python3
"""Does the SIGN of the gradient identify distractors (unrelated regions) on VISION? The directional
gradient sd_i = d(prob)/d(m_i) is SIGNED: sd_i>0 = patch supports the class, sd_i<0 = removing it
HELPS (distractor). |grad| discards this. Check: (1) corr(sd, signed-occlusion); (2) do the
NEGATIVE-sd patches have negative occlusion (removing them raises the prob = distractor)?;
(3) does the SIGNED ranking beat the MAGNITUDE |sd| on insertion (= the sign helps)?"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=5)
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

    res = []
    for name, path, target in cases:
        try:
            x, _ = syms["load_image"](Path(path)); x = x.to(dev)
            h_b0, cls_tok, baseline = syms["get_b0"](model, x)
            runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
            dtype = runner.dtype
            full = float(runner.prob_curve(np.ones((1, N), np.float32))[0])
            base = float(runner.prob_curve(np.zeros((1, N), np.float32))[0]); den = max(full - base, 1e-6)
            m = torch.ones(N, device=dev, dtype=dtype, requires_grad=True)
            prob = runner.probs_for_masks(m.unsqueeze(0))[0]
            sd = torch.autograd.grad(prob, m)[0].detach().cpu().numpy()      # signed directional
            occm = np.ones((N, N), np.float32); occm[np.arange(N), np.arange(N)] = 0.0
            occ = full - runner.prob_curve(occm, 64)                          # signed occlusion

            def ins_auc(sc):
                order = np.argsort(-sc); aucs = []
                for f in FR:
                    k = int(round(f * N)); mm = np.zeros((1, N), np.float32)
                    if k:
                        mm[0, order[:k]] = 1.0
                    aucs.append((float(runner.prob_curve(mm)[0]) - base) / den)
                return float(np.trapz(aucs, FR) / FR[-1])

            neg = sd < 0
            # keep only the supporters (sd>=0), remove the distractors (sd<0): does prob stay high?
            keep_pos = np.zeros((1, N), np.float32); keep_pos[0, ~neg] = 1.0
            prob_drop_distractors = float(runner.prob_curve(keep_pos)[0])
            res.append(dict(name=name, full=full, corr=float(spearmanr(sd, occ).correlation),
                            n_neg=int(neg.sum()), neg_occ=float(occ[neg].mean()) if neg.any() else float("nan"),
                            pos_occ=float(occ[~neg].mean()),
                            auc_signed=ins_auc(sd), auc_mag=ins_auc(np.abs(sd)), auc_occ=ins_auc(occ),
                            drop_distractors_prob=prob_drop_distractors))
            print(f"  {name:12s} full={full:.2f} done", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} failed: {type(e).__name__}: {e}", flush=True)

    print("\n=== VISION: does the gradient SIGN identify distractors? ===")
    print(f"corr(signed-grad, signed-occlusion) = {np.nanmean([r['corr'] for r in res]):+.3f}")
    print(f"negative-sd patches: count={np.mean([r['n_neg'] for r in res]):.0f}/196 | "
          f"their mean occlusion = {np.nanmean([r['neg_occ'] for r in res]):+.4f} "
          f"(NEGATIVE => removing them RAISES prob = distractor); positive-sd mean occ = {np.mean([r['pos_occ'] for r in res]):+.4f}")
    print(f"drop all negative-sd (distractor) patches: prob {np.mean([r['drop_distractors_prob'] for r in res]):.3f} "
          f"vs full {np.mean([r['full'] for r in res]):.3f}  (>= full => distractors were unrelated/harmful)")
    print("\n=== does the SIGN help insertion on vision? (signed vs magnitude) ===")
    print(f"  signed sd  insAUC = {np.mean([r['auc_signed'] for r in res]):.3f}")
    print(f"  |sd| mag   insAUC = {np.mean([r['auc_mag'] for r in res]):.3f}")
    print(f"  signed-occ insAUC = {np.mean([r['auc_occ'] for r in res]):.3f} (ceiling)")
    print("(if signed > magnitude AND negative-sd occ<0 -> the gradient sign correctly flags distractors on vision)")


if __name__ == "__main__":
    main()
