#!/usr/bin/env python3
"""CORE vs CONTEXT orthogonality on the MULTI-OBJECT zebra-elephant image. The single-object
orthogonality test had core=context (object IS the scene) -> not orthogonal. Here for the ELEPHANT
target, raising p(elephant) wants the elephant CORE *plus* zebra/savanna CONTEXT (insertion is
context-inclusive), but the NECESSITY set is just the elephant core (removing it breaks elephant;
removing context does not). So necessity-first hybrid insertion should be WORSE than pure sufficiency
= orthogonality appears. Test both targets (zebra=salient top-1, elephant=top-2 needs context).
Reports raw-prob insertion CURVE + AUC for fri_alt(suff) / greedy(nec) / hybrid / banzhaf / inflow."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_pred_attribution_bench import CHUNK, F, N, _rank, banzhaf_pred, greedy_pred

REPO = Path(__file__).resolve().parents[1]
XS = np.linspace(0, 1, N + 1)
FRP = [0.0, .02, .05, .1, .15, .2, .3, .5]                       # curve readout points


def raw_curves(runner, score):
    r = F.hard_curves(runner, np.asarray(score, np.float32).reshape(-1))
    p_ins, p_del = np.asarray(r[2], np.float64), np.asarray(r[3], np.float64)
    return p_ins, p_del


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    args = ap.parse_args()
    import timm
    syms = F.load_patch_repo()
    model = timm.create_model(args.model, pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    x, _ = syms["load_image"](Path(REPO / "multi_object_zebra_elephant.jpg")); x = x.to(args.device)
    out = {}
    for tname, target in [("zebra", 340), ("elephant", 386)]:
        h_b0, cls_tok, baseline = syms["get_b0"](model, x)
        runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
        g_rank, k_nec = greedy_pred(runner); g_order = [int(j) for j in np.argsort(-g_rank)]
        fri = np.asarray(F.fri_alt_prob(runner, steps=32)["final"], np.float32); fri_order = [int(j) for j in np.argsort(-fri)]
        bz = banzhaf_pred(runner, M=512)
        try:
            inflow = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
        except Exception:
            inflow = np.zeros(N, np.float32)
        nec_set = set(g_order[:k_nec])
        hybrid = _rank(g_order[:k_nec] + [p for p in fri_order if p not in nec_set])
        res = {"k_nec": int(k_nec)}; curves = {}
        for m, s in [("fri_alt", fri), ("greedy", g_rank), ("hybrid", hybrid), ("banzhaf", bz), ("inflow", inflow)]:
            p_ins, p_del = raw_curves(runner, s)
            res[m + "_ins"] = round(float(np.trapz(p_ins, XS)), 4); res[m + "_del"] = round(float(np.trapz(p_del, XS)), 4)
            curves[m] = [round(float(p_ins[int(round(f * N))]), 3) for f in FRP]      # insertion prob curve
        out[tname] = {"auc": res, "ins_curve": curves}
        print(f"\n### {tname} (target {target}) k_nec={k_nec} ###", flush=True)
        print(f"  INS AUC: fri_alt {res['fri_alt_ins']:.3f} hybrid {res['hybrid_ins']:.3f} greedy {res['greedy_ins']:.3f} "
              f"banzhaf {res['banzhaf_ins']:.3f} inflow {res['inflow_ins']:.3f}  | hybrid-fri_alt {res['hybrid_ins']-res['fri_alt_ins']:+.3f}", flush=True)
        print(f"  DEL AUC: greedy {res['greedy_del']:.3f} hybrid {res['hybrid_del']:.3f} fri_alt {res['fri_alt_del']:.3f}", flush=True)
        print(f"  INS curve @FR{FRP}:", flush=True)
        for m in ["fri_alt", "hybrid", "greedy", "inflow"]:
            print(f"    {m:9s} {curves[m]}", flush=True)
    Path(REPO / "outputs/zebra_elephant_orth.json").write_text(json.dumps(out, indent=1))
    print("\n(KEY: for ELEPHANT, is hybrid_ins < fri_alt_ins? = necessity-first MISSES context = orthogonal here)")


if __name__ == "__main__":
    main()
