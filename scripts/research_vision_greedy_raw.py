#!/usr/bin/env python3
"""Does the GREEDY necessity gold beat inflow with the CORRECTED raw-prob metric? (chunk_bz/banzhaf
are only competitive with inflow on the raw metric; greedy is the real necessity weapon but O(N^2).)
RAW-prob del AUC [0,1]; 2 model types (CLIP-laion, supervised-augreg2), small n (greedy expensive)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import F, N, banzhaf_pred, greedy_pred
from research_vision_cheap_necessity import chunked_with_prior
from research_vision_raw_metric import raw_auc

REPO = Path(__file__).resolve().parents[1]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n-img", type=int, default=6)
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
            g_rank, _ = greedy_pred(runner)
            bz = banzhaf_pred(runner, M=512); cb = chunked_with_prior(runner, bz, M=100, R=8)
            try:
                inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
            except Exception:
                inflow = np.zeros(N, np.float32)
            d = {}
            for m, s in [("greedy", g_rank), ("chunk_bz", cb), ("bz512", bz), ("inflow", inflow)]:
                _, dele = raw_auc(runner, s); d[m] = round(dele, 4)
            rows.append(d)
            print(f"  {mp.split('.')[0][-12:]} {i} | greedy {d['greedy']:.3f} chunk_bz {d['chunk_bz']:.3f} "
                  f"bz512 {d['bz512']:.3f} INFLOW {d['inflow']:.3f}", flush=True)
        allr[mp] = rows
    print("\n=== GREEDY necessity vs INFLOW, RAW-prob del AUC [0,1] (LOWER=better) ===")
    for mp, rows in allr.items():
        if not rows:
            continue
        a = lambda k: float(np.mean([r[k] for r in rows]))
        gw = 100.0 * np.mean([r["greedy"] < r["inflow"] for r in rows])
        cw = 100.0 * np.mean([r["chunk_bz"] < r["inflow"] for r in rows])
        print(f"  {mp.split('/')[-1][:26]:26s} n={len(rows)} | greedy {a('greedy'):.3f} chunk_bz {a('chunk_bz'):.3f} "
              f"bz512 {a('bz512'):.3f} INFLOW {a('inflow'):.3f} | greedy<inflow {gw:.0f}% chunk_bz<inflow {cw:.0f}%")
    json.dump(allr, open(REPO / "outputs/vision_greedy_raw.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
