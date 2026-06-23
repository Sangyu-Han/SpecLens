#!/usr/bin/env python3
"""CORRECTED METRIC (literature RISE-style RAW-PROB AUC, in [0,1]) — the F.hard_curves AUC divides by
p_full so it exceeds 1 when a subset beats the full image (distractor removal); that is the bug. Here
ins/del = trapz of the RAW class probability over the full insertion/deletion curve (p_ins/p_del are
already returned by hard_curves). Re-evaluate sufficiency (Banzhaf vs inflow, ins HIGHER=better) and
necessity (chunk_bz vs inflow, del LOWER=better) across MULTIPLE timm ViTs (CLIP-laion, CLIP-openai,
supervised augreg2) at n↑, to see if the FRI insertion gain over inflow is real/meaningful."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import F, N, banzhaf_pred, grad_pred
from research_vision_cheap_necessity import chunked_with_prior, single_occ_prior

REPO = Path(__file__).resolve().parents[1]
XS = np.linspace(0, 1, N + 1)


def raw_auc(runner, score):
    """RAW-prob insertion/deletion AUC in [0,1] (NOT divided by p_full)."""
    r = F.hard_curves(runner, np.asarray(score, np.float32).reshape(-1))
    p_ins, p_del = np.asarray(r[2], np.float64), np.asarray(r[3], np.float64)
    return float(np.trapz(p_ins, XS)), float(np.trapz(p_del, XS))


def run_model(model_path, n_img, Mbz, dev):
    import timm
    syms = F.load_patch_repo()
    print(f"  [model] {model_path}", flush=True)
    model = timm.create_model(model_path, pretrained=True).eval().to(dev)
    for p in model.parameters():
        p.requires_grad = False
    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    cases = []
    for i in range(n_img):
        r = df.iloc[i * 5]; cases.append((str(r["path"]), int(r["target"])))
    rows = []
    for path, target in cases:
        try:
            x, _ = syms["load_image"](Path(path)); x = x.to(dev)
        except Exception:
            continue
        h_b0, cls_tok, baseline = syms["get_b0"](model, x)
        runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
        bz = banzhaf_pred(runner, M=Mbz); bz64 = banzhaf_pred(runner, M=64)
        occ = single_occ_prior(runner); g = grad_pred(runner)
        chunk_bz = np.zeros(N, np.float32)
        co = chunked_with_prior(runner, bz, M=100, R=8); chunk_bz = co       # _rank order score
        try:
            inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
        except Exception:
            inflow = np.zeros(N, np.float32)
        sc = {"bz512": bz, "bz64": bz64, "chunk_bz": chunk_bz, "single_occ": occ, "grad": g, "inflow": inflow}
        res = {}
        for m, s in sc.items():
            ins, dele = raw_auc(runner, s); res[m + "_ins"] = round(ins, 4); res[m + "_del"] = round(dele, 4)
        rows.append(res)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=12)
    ap.add_argument("--Mbz", type=int, default=512)
    args = ap.parse_args()
    MODELS = ["vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
              "vit_base_patch16_clip_224.openai",
              "vit_base_patch16_224.augreg2_in21k_ft_in1k"]
    allr = {}
    for mp in MODELS:
        try:
            allr[mp] = run_model(mp, args.n_img, args.Mbz, args.device)
        except Exception as e:  # noqa: BLE001
            import traceback; print(f"  FAIL {mp}: {e}\n{traceback.format_exc()[-400:]}", flush=True)
    print("\n=== RAW-PROB metric (ins∈[0,1] HIGHER=better sufficiency; del LOWER=better necessity) ===")
    for mp, rows in allr.items():
        if not rows:
            continue
        nm = mp.split("/")[-1][:28]
        a = lambda k: float(np.mean([r[k] for r in rows]))
        # sufficiency: banzhaf ins vs inflow ; necessity: chunk_bz del vs inflow
        si64 = 100.0 * np.mean([r["bz64_ins"] > r["inflow_ins"] for r in rows])
        si512 = 100.0 * np.mean([r["bz512_ins"] > r["inflow_ins"] for r in rows])
        nd = 100.0 * np.mean([r["chunk_bz_del"] < r["inflow_del"] for r in rows])
        print(f"  {nm:28s} n={len(rows)}", flush=True)
        print(f"     INS(suff↑): bz64 {a('bz64_ins'):.3f} bz512 {a('bz512_ins'):.3f} grad {a('grad_ins'):.3f} INFLOW {a('inflow_ins'):.3f}"
              f"  | bz64>inflow {si64:.0f}% bz512>inflow {si512:.0f}%")
        print(f"     DEL(nec↓):  chunk_bz {a('chunk_bz_del'):.3f} single_occ {a('single_occ_del'):.3f} bz512 {a('bz512_del'):.3f} INFLOW {a('inflow_del'):.3f}"
              f"  | chunk_bz<inflow {nd:.0f}%")
    json.dump(allr, open(REPO / "outputs/vision_raw_metric.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
