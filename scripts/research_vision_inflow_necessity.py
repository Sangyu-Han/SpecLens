#!/usr/bin/env python3
"""NECESSITY (and sufficiency) vs INFLOW — the strong vision attention-flow baseline (the correct
vision comparison, NOT AttnLRP). Methods: greedy-conditional, chunk_bz (Banzhaf-prior conditional),
banzhaf-raw, single_occ, INFLOW. Both insertion (ins, HIGHER=better) and deletion (del, LOWER=better)
via F.hard_curves. Question: can our necessity (greedy/chunk_bz) beat inflow on deletion? sufficiency
(banzhaf) on insertion? CLIP (timm), n images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, banzhaf_pred, greedy_pred
from research_vision_cheap_necessity import chunked_with_prior, single_occ_prior

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=8)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--Mbz", type=int, default=512)
    ap.add_argument("--R", type=int, default=8)
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
        g_rank, k = greedy_pred(runner)
        bz = banzhaf_pred(runner, M=args.Mbz)
        pr_occ = single_occ_prior(runner)
        try:
            inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
        except Exception as e:  # noqa: BLE001
            print(f"  [inflow fail {type(e).__name__}] -> zeros", flush=True); inflow = np.zeros(N, np.float32)
        sc = {"greedy": g_rank, "chunk_bz": chunked_with_prior(runner, bz, args.M, args.R),
              "banzhaf": bz, "single_occ": pr_occ, "inflow": inflow}
        res = {"case": name}
        for m, s in sc.items():
            ins, dele, _, _ = F.hard_curves(runner, s)
            res[m + "_del"] = round(float(dele), 4); res[m + "_ins"] = round(float(ins), 4)
        rows.append(res)
        print(f"{name:12s} | DEL greedy {res['greedy_del']:.3f} chunk_bz {res['chunk_bz_del']:.3f} "
              f"banzhaf {res['banzhaf_del']:.3f} INFLOW {res['inflow_del']:.3f} | INS banzhaf {res['banzhaf_ins']:.3f} "
              f"INFLOW {res['inflow_ins']:.3f}", flush=True)
    print("\n=== vs INFLOW (DEL ↓ necessity, INS ↑ sufficiency) ===")
    for m in ["greedy", "chunk_bz", "banzhaf", "single_occ", "inflow"]:
        d = float(np.mean([r[m + "_del"] for r in rows])); i = float(np.mean([r[m + "_ins"] for r in rows]))
        print(f"  {m:11s} del={d:.3f}  ins={i:.3f}")
    inf_d = np.mean([r["inflow_del"] for r in rows]); inf_i = np.mean([r["inflow_ins"] for r in rows])
    for m in ["greedy", "chunk_bz", "banzhaf"]:
        wd = 100.0 * np.mean([r[m + "_del"] < r["inflow_del"] for r in rows])
        wi = 100.0 * np.mean([r[m + "_ins"] > r["inflow_ins"] for r in rows])
        print(f"  {m:11s} beats inflow: DEL {wd:.0f}%  INS {wi:.0f}%")
    print(f"(inflow del={inf_d:.3f} ins={inf_i:.3f} — can necessity beat it on DEL? sufficiency on INS?)")
    Path(REPO / "outputs/vision_inflow_necessity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
