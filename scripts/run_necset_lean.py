#!/usr/bin/env python3
"""Lean budget tiers of NSD-FRI on diag8 (or n100 subset).

Tiers (extra forward probes on top of the HG-FRI solve):
  lean32 : backbone = sc (no selection). Singleton restoration scan of the
           backbone top-32 at the dead(32) state (33 probes). Guard: keep
           reorder only if the scanned dead-state is actually dead
           (p0 < 0.5 p_full), else keep backbone. ~33 probes.
  lean64 : channel selection over {sc, ksm, bl, clu} at ks=(8,24,56) (12
           probes) -> backbone; singleton scan top-48 at dead(48) (49
           probes); est-guard final vs backbone reusing the selection grid
           (3 probes). ~64 probes.
Reference: full pipeline numbers come from research_necset_stitch runs.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.core.attribution.fri import HGFRIConfig, inverse_grad_irrelevance, run_hgfri

_spec = _ilu.spec_from_file_location("ns", REPO / "scripts/research_necset_stitch.py")
ns = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(ns)
q = ns.q
N = ns.N


def lean_case(model, syms, path, target, seed, device, tier="lean32"):
    x, _ = syms["load_image"](Path(path))
    x = x.to(device)
    h_b0, cls_tok, base = syms["get_b0"](model, x)
    runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)

    def ofp(h_var, ct=cls_tok, xx=x, t=target):
        model.zero_grad(set_to_none=True)
        inj = [torch.cat([ct, h_var], 1)]
        hk = model.blocks[0].register_forward_pre_hook(lambda m, a: (inj[0],))
        try:
            out = model(xx)
        finally:
            hk.remove()
        return torch.softmax(out[0], 0)[t]

    irr = inverse_grad_irrelevance(input_patches=h_b0, objective_from_patches=ofp)
    out = run_hgfri(q._Oracle(runner), irrelevance=irr, config=HGFRIConfig(seed=seed))
    sc = out["scores"]

    with torch.no_grad():
        p_full = float(runner.probs_for_masks(
            torch.ones(1, N, device=runner.dev, dtype=runner.dtype))[0])
        p_base = float(runner.probs_for_masks(
            torch.zeros(1, N, device=runner.dev, dtype=runner.dtype))[0])

    pc = ns.ProbeCounter(runner)
    if tier == "lean128":
        # select(4ch x 5ks = 20) + dead(64) singleton@48 (49) + m96 (33)
        # + guard (5+handful) ~= 112 probes
        K = ns.content_kernel(runner.h_diff)
        ksm = 0.5 * ns._n01(sc) + 0.5 * (K @ ns._n01(sc))
        fin, hgd = out["final"], out["hgd"]
        bl = ns._n01(fin) + ns._n01(hgd)
        prior = ns._n01(sc) + 0.5 * ns._n01(hgd)
        clu, _ = ns.cluster_order_scores(runner.h_diff, prior, k=10, seed=0)
        channels = {"sc": sc, "ksm": ksm, "bl": bl, "clu": clu}
        ks_sel = (4, 12, 32, 64, 120)
        best, est = ns.select_prefix(pc, channels, ks=ks_sel,
                                     p_full=p_full, p_base=p_base)
        backbone = channels[best]
        o = ns.order_of(backbone)
        m0 = np.ones(N, np.float32)
        m0[np.asarray(o[:64], int)] = 0.0
        masks = [m0]
        for i in o[:48]:
            m = m0.copy()
            m[int(i)] = 1.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        p0 = float(vals[0])
        if p0 < 0.5 * p_full:
            jump = {int(i): float(v) - p0 for i, v in zip(o[:48], vals[1:])}
            res = max(2e-4 * p_full, 1e-9)
            rk = {int(pch): r for r, pch in enumerate(o)}
            tol = 0.002 * p_full
            pos = sorted([i for i in o[:48] if jump[int(i)] >= -tol],
                         key=lambda i: (-round(jump[int(i)] / res), rk[int(i)]))
            anti = sorted([i for i in o[:48] if jump[int(i)] < -tol],
                          key=lambda i: (-round(jump[int(i)] / res), rk[int(i)]))
            o1 = np.asarray(pos + [i for i in o[48:64]] + [i for i in o[64:]] + anti, int)
            # m96 segment scan at dead(96)
            mid_idx = [int(i) for i in o1[64:96]]
            m0m = np.ones(N, np.float32)
            m0m[o1[:96]] = 0.0
            masks_m = [m0m]
            for i in mid_idx:
                m = m0m.copy()
                m[int(i)] = 1.0
                masks_m.append(m)
            vals_m = pc.probs(np.stack(masks_m))
            p0m = float(vals_m[0])
            jm = {int(i): float(v) - p0m for i, v in zip(mid_idx, vals_m[1:])}
            mid_sorted = sorted(mid_idx, key=lambda i: -jm[int(i)])
            v1 = ns.rank_scores_from_order(
                np.asarray(list(o1[:64]) + mid_sorted + list(o1[96:]), int))
            _, e2 = ns.select_prefix(pc, {"v1": v1}, ks=ks_sel,
                                     p_full=p_full, p_base=p_base)
            if e2["v1"] <= est[best]:
                final, pick = v1, f"lobo1m96@d64({best})"
            else:
                final, pick = backbone, f"select:{best}"
        else:
            final, pick = backbone, f"select:{best}(p0 alive)"
    elif tier == "lean96":
        # sc backbone; full dead(64) singleton scan (65) + m96 segment (33)
        backbone = sc
        v1, dg1 = ns.lobo_singleton(pc, backbone, top=64, p_full=p_full)
        if dg1["p0"] < 0.5 * p_full:
            o1 = ns.order_of(v1)
            mid_idx = list(o1[64:96])
            m0m = np.ones(N, np.float32)
            m0m[np.asarray(o1[:96], int)] = 0.0
            masks_m = [m0m]
            for i in mid_idx:
                m = m0m.copy()
                m[int(i)] = 1.0
                masks_m.append(m)
            vals_m = pc.probs(np.stack(masks_m))
            p0m = float(vals_m[0])
            jm = {int(i): float(v) - p0m for i, v in zip(mid_idx, vals_m[1:])}
            mid_sorted = sorted(mid_idx, key=lambda i: -jm[int(i)])
            final = ns.rank_scores_from_order(
                np.asarray(list(o1[:64]) + mid_sorted + list(o1[96:]), int))
            pick = "lobo1m96(sc)"
        else:
            final, pick = backbone, "sc(p0 alive)"
    elif tier == "lean32":
        # dead(64) baseline, singleton-scan the top-32 only: 33 probes
        backbone = sc
        o = ns.order_of(backbone)
        m0 = np.ones(N, np.float32)
        m0[np.asarray(o[:64], int)] = 0.0
        masks = [m0]
        for i in o[:32]:
            m = m0.copy()
            m[int(i)] = 1.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        p0 = float(vals[0])
        if p0 < 0.5 * p_full:
            jump = {int(i): float(v) - p0 for i, v in zip(o[:32], vals[1:])}
            res = max(2e-4 * p_full, 1e-9)
            rk = {int(pch): r for r, pch in enumerate(o)}
            tol = 0.002 * p_full
            pos = sorted([i for i in o[:32] if jump[int(i)] >= -tol],
                         key=lambda i: (-round(jump[int(i)] / res), rk[int(i)]))
            anti = sorted([i for i in o[:32] if jump[int(i)] < -tol],
                          key=lambda i: (-round(jump[int(i)] / res), rk[int(i)]))
            seq = pos + [i for i in o[32:]] + anti
            final = ns.rank_scores_from_order(np.asarray(seq, int))
            pick = "lobo1@32/dead64"
        else:
            final, pick = backbone, "sc(p0 alive)"
    else:  # lean64
        K = ns.content_kernel(runner.h_diff)
        ksm = 0.5 * ns._n01(sc) + 0.5 * (K @ ns._n01(sc))
        fin, hgd = out["final"], out["hgd"]
        bl = ns._n01(fin) + ns._n01(hgd)
        prior = ns._n01(sc) + 0.5 * ns._n01(hgd)
        clu, _ = ns.cluster_order_scores(runner.h_diff, prior, k=10, seed=0)
        channels = {"sc": sc, "ksm": ksm, "bl": bl, "clu": clu}
        ks_sel = (8, 24, 56)
        best, est = ns.select_prefix(pc, channels, ks=ks_sel,
                                     p_full=p_full, p_base=p_base)
        backbone = channels[best]
        o = ns.order_of(backbone)
        m0 = np.ones(N, np.float32)
        m0[np.asarray(o[:64], int)] = 0.0
        masks = [m0]
        for i in o[:48]:
            m = m0.copy()
            m[int(i)] = 1.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        p0 = float(vals[0])
        if p0 < 0.5 * p_full:
            jump = {int(i): float(v) - p0 for i, v in zip(o[:48], vals[1:])}
            res = max(2e-4 * p_full, 1e-9)
            rk = {int(pch): r for r, pch in enumerate(o)}
            tol = 0.002 * p_full
            pos = sorted([i for i in o[:48] if jump[int(i)] >= -tol],
                         key=lambda i: (-round(jump[int(i)] / res), rk[int(i)]))
            anti = sorted([i for i in o[:48] if jump[int(i)] < -tol],
                          key=lambda i: (-round(jump[int(i)] / res), rk[int(i)]))
            seq = pos + [i for i in o[48:]] + anti
            v1 = ns.rank_scores_from_order(np.asarray(seq, int))
            _, e2 = ns.select_prefix(pc, {"v1": v1}, ks=ks_sel,
                                     p_full=p_full, p_base=p_base)
            final = v1 if e2["v1"] <= est[best] else backbone
            pick = f"lobo1@48d64({best})" if e2["v1"] <= est[best] else f"select:{best}"
        else:
            final, pick = backbone, f"select:{best}(p0 alive)"

    hins, hdel, _, _ = q.hard_curves(runner, final, chunk=128)
    return {"hdel": float(hdel), "hins": float(hins), "pick": pick,
            "probes": pc.n, "p_full": p_full}, np.asarray(final, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tier", choices=("lean32", "lean64", "lean96", "lean128"), default="lean32")
    ap.add_argument("--n100", action="store_true",
                    help="run on the n100 CSV (vs the diag8 cases)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    if args.n100:
        import pandas as pd
        df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
        cases = [(Path(r["path"]).stem, r["path"], int(r["target"]),
                  float(r["inflow_hdel"]), float(r["inflow_hins"]))
                 for _, r in df.iterrows()]
    else:
        cases = [(n, p, t, None, None) for n, p, t in ns.CASES]

    agg = []
    rows = {}
    scores_npz = {}
    t0 = time.time()
    for name, path, target, infl_del_csv, infl_ins_csv in cases:
        if infl_del_csv is None:
            infl = syms["inflow"](model, x := syms["load_image"](Path(path))[0].to(args.device),
                                  target_class=target).astype(np.float32)
            h_b0, cls_tok, base = syms["get_b0"](model, x)
            runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)
            _, infl_del, _, _ = q.hard_curves(runner, infl, chunk=128)
        else:
            infl_del = infl_del_csv
        r, final_scores = lean_case(model, syms, path, target, int(args.seed), args.device,
                                    tier=args.tier)
        agg.append((name, r, infl_del))
        rows[name] = {**r, "inflow_hdel": float(infl_del)}
        scores_npz[f"{name}_final"] = final_scores
        print(f"[{name:<28}] {args.tier}={r['hdel']:.3f}/{r['hins']:.2f} "
              f"inflow={infl_del:.3f} pick={r['pick']} pr={r['probes']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    hd = np.mean([r["hdel"] for _, r, _ in agg])
    hmed = np.median([r["hdel"] for _, r, _ in agg])
    hi = np.mean([r["hins"] for _, r, _ in agg])
    wins = sum(r["hdel"] < i for _, r, i in agg)
    print(f"\n{args.tier}: hdel {hd:.4f} (med {hmed:.4f})  hins {hi:.4f}  "
          f"win_del {wins}/{len(agg)}")
    if args.out:
        args.out.write_text(json.dumps(
            {"tier": args.tier, "hdel_mean": float(hd), "hins_mean": float(hi),
             "win_del": int(wins), "n": len(agg), "rows": rows}, indent=1))
        np.savez_compressed(args.out.with_suffix(".scores.npz"), **scores_npz)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
