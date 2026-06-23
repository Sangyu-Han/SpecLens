#!/usr/bin/env python3
"""n100 validation of the necessary-set probe pipeline (NSD-FRI).

Pipeline per case (exploration form; per-variant metrics logged so budget
tiers can be ablated offline):
  1. HG-FRI solve (existing FRI budget) -> free channels
  2. prefix-probe channel selection (ks grid)
  3. stitch + gsc constructors, same-grid verification -> backbone W
  4. dead-end singleton restoration reorders (lobo1 family) on W (+select)
  5. est-guard final pick

Saves per-case hdel/hins for every variant + final scores arrays (npz) for
offline MAS/Stoch evaluation.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

CSV = REPO / "outputs/class_fri/softplus_failure_scan_n100.csv"


def run_case(model, syms, path, target, seed, device, *, return_all_scores=False):
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

    sc, fin, hgd = out["scores"], out["final"], out["hgd"]
    K = ns.content_kernel(runner.h_diff)
    ksm = 0.5 * ns._n01(sc) + 0.5 * (K @ ns._n01(sc))
    bl = ns._n01(fin) + ns._n01(hgd)
    prior = ns._n01(sc) + 0.5 * ns._n01(hgd)
    clu, _ = ns.cluster_order_scores(runner.h_diff, prior, k=10, seed=0)
    gnorm = ns._n01(1.0 / (irr.detach().cpu().numpy().astype(np.float64) + 1e-6))
    fg, _, _ = ns.front_greedy(runner, ns.order_of(sc), F=48)
    channels = {"sc": sc, "fin": fin, "hgd": hgd, "bl": bl, "ksm": ksm,
                "clu": clu, "gnorm": gnorm, "fg": fg}

    with torch.no_grad():
        p_full = float(runner.probs_for_masks(
            torch.ones(1, N, device=runner.dev, dtype=runner.dtype))[0])
        p_base = float(runner.probs_for_masks(
            torch.zeros(1, N, device=runner.dev, dtype=runner.dtype))[0])

    ks_sel = (2, 4, 8, 12, 20, 32, 48, 72, 120)
    pc = ns.ProbeCounter(runner)
    best, est = ns.select_prefix(pc, channels, ks=ks_sel, p_full=p_full, p_base=p_base)
    sel_scores = channels[best]

    st_scores, _, _ = ns.stitch_greedy(pc, channels, backbone=best, p_full=p_full)
    gsc_scores, _, _ = ns.front_greedy(runner, ns.order_of(sel_scores), F=48, warm=4)

    _, ver_est = ns.select_prefix(pc, {"st": st_scores, "gsc": gsc_scores},
                                  ks=ks_sel, p_full=p_full, p_base=p_base)
    cands = {"select": (est[best], sel_scores),
             "stitch": (ver_est["st"], st_scores),
             "gsc": (ver_est["gsc"], gsc_scores)}
    ver_pick = min(cands, key=lambda k: cands[k][0])
    ver_scores = cands[ver_pick][1]

    # lobo1 family on the verify winner
    lobo_variants = {}
    cblocks = ns.content_blocks_of_top(ver_scores, runner.h_diff, top=64, kclust=10)
    vC, dgC = ns.lobo_reorder(pc, ver_scores, top=64, blocks=cblocks,
                              ratio=True, p_full=p_full)
    lobo_variants["loboC"] = vC
    v1, dg1 = ns.lobo_singleton(pc, ver_scores, top=64, p_full=p_full)
    lobo_variants["lobo1"] = v1
    v1c, _ = ns.lobo_singleton(pc, ver_scores, top=64, p_full=p_full,
                               within_only=True, blocks=cblocks,
                               block_jumps=dgC["jumps"],
                               precomputed=(dg1["p0"], dg1["jump_map"]))
    lobo_variants["lobo1Ci"] = v1c
    if ver_pick != "select":
        v1s, _ = ns.lobo_singleton(pc, sel_scores, top=64, p_full=p_full)
        lobo_variants["lobo1S"] = v1s
    # m96 scan on lobo1's order
    o1 = ns.order_of(v1)
    mid_idx = list(o1[64:96])
    m0m = np.ones(N, np.float32)
    m0m[np.asarray(o1[:96], int)] = 0.0
    masks_m = [m0m] + []
    for i in mid_idx:
        m = m0m.copy()
        m[int(i)] = 1.0
        masks_m.append(m)
    vals_m = pc.probs(np.stack(masks_m))
    p0m = float(vals_m[0])
    jm = {int(i): float(v) - p0m for i, v in zip(mid_idx, vals_m[1:])}
    mid_sorted = sorted(mid_idx, key=lambda i: -jm[int(i)])
    lobo_variants["lobo1m96"] = ns.rank_scores_from_order(
        np.asarray(list(o1[:64]) + mid_sorted + list(o1[96:]), int))

    _, lobo_est = ns.select_prefix(pc, lobo_variants, ks=ks_sel,
                                   p_full=p_full, p_base=p_base)
    lobo_best = min(lobo_est, key=lobo_est.get)
    if lobo_est[lobo_best] <= cands[ver_pick][0]:
        final_scores, final_pick = lobo_variants[lobo_best], f"{lobo_best}({ver_pick})"
        final_est = lobo_est[lobo_best]
    else:
        final_scores, final_pick = ver_scores, ver_pick
        final_est = cands[ver_pick][0]

    # second-level refiner: synergy segments ON the current best ordering
    # (functional segmentation; emergent continuity, no 2D prior)
    segs, Js, seg_J, _sd = ns.synergy_segments(
        pc, runner, final_scores, top=40, p_full=p_full, seed=seed)
    sy_scores = ns.segment_order_scores(
        segs, Js, final_scores, p_full=p_full, seg_J=seg_J)
    lobo_variants["lobo1Sy"] = sy_scores
    _, sy_est = ns.select_prefix(pc, {"sy": sy_scores}, ks=ks_sel,
                                 p_full=p_full, p_base=p_base)
    if sy_est["sy"] <= final_est:
        final_scores, final_pick = sy_scores, f"sy({final_pick})"

    evals = {**{k: v for k, v in channels.items() if k in ("sc", "ksm")},
             "select": sel_scores, "verify": ver_scores,
             **lobo_variants, "final": final_scores}
    metrics = {}
    for nm, s in evals.items():
        hins, hdel, _, _ = q.hard_curves(runner, s, chunk=128)
        metrics[nm] = {"hins": float(hins), "hdel": float(hdel)}
    diag = {
        "selected": best, "ver_pick": ver_pick, "final_pick": final_pick,
        "probes": pc.n, "p_full": p_full, "p_base": p_base,
        "lobo1_p0": dg1["p0"],
    }
    if return_all_scores:
        return metrics, final_scores, diag, evals
    return metrics, final_scores, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/necset_n100.json")
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    df = pd.read_csv(CSV)
    if args.limit:
        df = df.iloc[: args.limit]
    rows = {}
    scores_npz = {}
    t0 = time.time()
    for n_done, (_, r) in enumerate(df.iterrows(), start=1):
        key = Path(r["path"]).stem
        try:
            metrics, final_scores, diag = run_case(
                model, syms, r["path"], int(r["target"]), int(args.seed), args.device)
        except Exception as e:
            print(f"[{n_done:03d}] {key} ERROR {e}", flush=True)
            continue
        rows[key] = {
            "csv": {"inflow_hdel": float(r["inflow_hdel"]),
                    "inflow_hins": float(r["inflow_hins"]),
                    "soft_hdel": float(r["soft_hdel"]),
                    "desc": str(r["desc"]), "prob": float(r["prob"])},
            "metrics": metrics,
            "diag": diag,
        }
        scores_npz[f"{key}_final"] = np.asarray(final_scores, np.float32)
        f = metrics["final"]
        print(f"[{n_done:03d}/{len(df)}] {key} {str(r['desc'])[:18]:<18} "
              f"final={f['hdel']:.3f}/{f['hins']:.2f} "
              f"inflow={r['inflow_hdel']:.3f} pick={diag['final_pick']} "
              f"pr={diag['probes']} ({time.time()-t0:.0f}s)", flush=True)

    # summary
    nms = sorted({m for k in rows for m in rows[k]["metrics"]})
    summary = {}
    infl = np.array([rows[k]["csv"]["inflow_hdel"] for k in rows])
    infl_i = np.array([rows[k]["csv"]["inflow_hins"] for k in rows])
    for nm in nms:
        sub = [k for k in rows if nm in rows[k]["metrics"]]
        if len(sub) < len(rows):
            continue
        d = np.array([rows[k]["metrics"][nm]["hdel"] for k in rows])
        hi = np.array([rows[k]["metrics"][nm]["hins"] for k in rows])
        summary[nm] = {
            "hdel_mean": float(d.mean()), "hdel_median": float(np.median(d)),
            "hins_mean": float(hi.mean()),
            "win_del": int((d < infl).sum()),
            "win_both": int(((d < infl) & (hi > infl_i)).sum()),
            "n": int(len(d)),
        }
        print(f"{nm:10} hdel {d.mean():.4f} (med {np.median(d):.4f}) "
              f"hins {hi.mean():.4f} win_del {(d < infl).sum()} "
              f"win_both {((d < infl) & (hi > infl_i)).sum()}")
    print(f"inflow     hdel {infl.mean():.4f} (med {np.median(infl):.4f}) "
          f"hins {infl_i.mean():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    np.savez_compressed(args.out.with_suffix(".scores.npz"), **scores_npz)
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
