#!/usr/bin/env python3
"""Are SUFFICIENCY and NECESSITY orthogonal? Test the user's idea: build a HYBRID order = necessity
set FIRST (greedy), then sufficiency (fri_alt-32) for the rest. If the hybrid's INSERTION >= the pure
sufficiency-specialized attribution (fri_alt), then one ordering improves BOTH faces (not orthogonal).
Also the FAIR sufficiency baseline is fri_alt-32 (proper FRI), not inflow (attention, not sufficiency-
specialized). RAW-prob AUC [0,1]. Methods: fri_alt(suff,32step), greedy(nec), inflow, banzhaf, hybrid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import F, N, _rank, banzhaf_pred, greedy_pred
from research_vision_raw_metric import raw_auc

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--n-img", type=int, default=6)
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
            g_rank, k_nec = greedy_pred(runner)                       # necessity order + set size
            g_order = [int(j) for j in np.argsort(-g_rank)]
            fri = np.asarray(F.fri_alt_prob(runner, steps=32)["final"], np.float32)  # sufficiency (32-step FRI)
            fri_order = [int(j) for j in np.argsort(-fri)]
            bz = banzhaf_pred(runner, M=512)
            try:
                inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
            except Exception:
                inflow = np.zeros(N, np.float32)
            nec_set = set(g_order[:k_nec])
            hybrid_order = g_order[:k_nec] + [p for p in fri_order if p not in nec_set]   # necessity-first ++ sufficiency
            hybrid = _rank(hybrid_order)
            d = {"k_nec": int(k_nec)}
            for m, s in [("fri_alt", fri), ("greedy", g_rank), ("inflow", inflow), ("banzhaf", bz), ("hybrid", hybrid)]:
                ins, dele = raw_auc(runner, s); d[m + "_ins"] = round(ins, 4); d[m + "_del"] = round(dele, 4)
            rows.append(d)
            print(f"  {mp.split('.')[0][-10:]} {i} k={d['k_nec']:3d} | INS fri_alt {d['fri_alt_ins']:.3f} hybrid {d['hybrid_ins']:.3f} "
                  f"greedy {d['greedy_ins']:.3f} inflow {d['inflow_ins']:.3f} | DEL greedy {d['greedy_del']:.3f} "
                  f"hybrid {d['hybrid_del']:.3f} fri_alt {d['fri_alt_del']:.3f}", flush=True)
        allr[mp] = rows
    print("\n=== ORTHOGONALITY: necessity-first hybrid vs pure sufficiency (RAW AUC; ins↑ del↓) ===")
    for mp, rows in allr.items():
        if not rows:
            continue
        a = lambda k: float(np.mean([r[k] for r in rows]))
        hyb_vs_fri = 100.0 * np.mean([r["hybrid_ins"] >= r["fri_alt_ins"] for r in rows])
        fri_vs_inf = 100.0 * np.mean([r["fri_alt_ins"] > r["inflow_ins"] for r in rows])
        print(f"  {mp.split('/')[-1][:24]:24s} n={len(rows)}", flush=True)
        print(f"     INS(suff↑): fri_alt {a('fri_alt_ins'):.3f} hybrid {a('hybrid_ins'):.3f} greedy {a('greedy_ins'):.3f} "
              f"banzhaf {a('banzhaf_ins'):.3f} inflow {a('inflow_ins'):.3f} | hybrid≥fri_alt {hyb_vs_fri:.0f}% fri_alt>inflow {fri_vs_inf:.0f}%")
        print(f"     DEL(nec↓):  greedy {a('greedy_del'):.3f} hybrid {a('hybrid_del'):.3f} fri_alt {a('fri_alt_del'):.3f} "
              f"inflow {a('inflow_del'):.3f}")
    print("(KEY: hybrid_ins >= fri_alt_ins? -> necessity-first ALSO good for insertion = NOT orthogonal, one order does both)")
    json.dump(allr, open(REPO / "outputs/vision_orthogonality.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
