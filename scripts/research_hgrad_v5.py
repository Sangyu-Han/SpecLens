#!/usr/bin/env python3
"""V5: arm-A chassis + gate switch + full-state pooling + mid-ins harvest.

Membership pools (per-state L1-normalized fields, equal vote):
  hgd   : del/dense harvest states (dens>=0.7)
  hgd_f : + full-state field (free from the irr backward), vote weight 1
  hgd_m : + mid-density ins states (0.4<=dens<0.7), vote weight 0.5
Gates on sd = Σ ixg over harvest states:
  none | gmad3 (sd<-3MAD -> x0.2) | gsw (apply z<-1 -> x0.3 ONLY if
  min(sd) < -4*MAD: competitor-present switch) | gsw2 (switch -4MAD, demote
  sd<-2MAD x0.3)
Blends: bn λ{1,2} with final head.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FRONTIER_PATH = REPO / "scripts/research_fri_frontier.py"
if not FRONTIER_PATH.exists():
    FRONTIER_PATH = REPO.parent / "_fri_research_archive_20260610/scripts/research_fri_frontier.py"
_spec = _ilu.spec_from_file_location("research_fri_frontier", FRONTIER_PATH)
_frontier = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_frontier)
Block0Runner = _frontier.Block0Runner
hard_curves = _frontier.hard_curves
load_patch_repo = _frontier.load_patch_repo

N, GRID = 196, 14
CSV = REPO / "outputs/class_fri/softplus_failure_scan_n100.csv"
ZEBRA = REPO / "multi_object_zebra_elephant.jpg"


def _logit(p):
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _n01(v):
    v = np.maximum(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return v / (v.max() + 1e-12)


def state_forward(runner, state_w):
    h_probe = torch.zeros_like(runner.h_diff, requires_grad=True)
    h_mix = runner.base + state_w.view(1, N, 1) * runner.h_diff + h_probe
    h_inj = torch.cat([runner.cls, h_mix], dim=1)
    holder = [h_inj]
    hook = runner.model.blocks[0].register_forward_pre_hook(lambda m, a: (holder[0],))
    try:
        out = runner.model(runner.x)
    finally:
        hook.remove()
    return out[0], h_probe


def make_irr_and_hgfull(runner):
    h_var = runner.h_b0.detach().clone().requires_grad_(True)
    h_inj = torch.cat([runner.cls, h_var], dim=1)
    holder = [h_inj]
    hook = runner.model.blocks[0].register_forward_pre_hook(lambda m, a: (holder[0],))
    try:
        out = runner.model(runner.x)
    finally:
        hook.remove()
    torch.softmax(out[0], dim=0)[runner.target].backward()
    grad = h_var.grad[0]
    gnorm = grad.norm(dim=-1)
    ixg_full = (grad * runner.h_diff[0]).sum(-1).detach().cpu().numpy()
    inv = 1.0 / (gnorm + 1e-8)
    irr = (inv / inv.max().clamp(min=1e-8)).detach().reshape(-1)
    return irr, gnorm.detach().cpu().numpy(), ixg_full


def solve_collect(runner, *, steps=32, seed=42, irr=None,
                  lr=0.45, lr_end=0.01, irr_weight=0.05,
                  l1_weight=0.003, deletion_weight=0.5, budget_mode="iid",
                  kmat=None, ktv_weight=0.0):
    dev, dtype = runner.dev, runner.dtype
    with torch.no_grad():
        full_obj = runner.probs_for_masks(torch.ones(1, N, device=dev, dtype=dtype))[0]
        base_obj = runner.probs_for_masks(torch.zeros(1, N, device=dev, dtype=dtype))[0]
    scale = float((full_obj - base_obj).abs().clamp(min=1e-8).item())

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    la = torch.full((N,), _logit(0.5), device=dev, dtype=dtype)
    m_v = torch.zeros(N, device=dev, dtype=dtype)
    v_v = torch.zeros(N, device=dev, dtype=dtype)
    recs = {"hg": [], "ixg": [], "dens": [], "phase": []}

    strat_budgets = None
    if budget_mode == "stratified":
        rng_np = np.random.default_rng(seed)
        n_ins = (steps + 1) // 2
        n_del = steps - n_ins
        ins_b = rng_np.permutation(np.linspace(0.03, 0.97, n_ins)) * N
        del_b = rng_np.permutation(np.linspace(0.03, 0.97, max(n_del, 1))) * N
        strat_budgets = {"ins": list(ins_b), "del": list(del_b)}

    def _rec(v):
        denom = (full_obj - base_obj)
        denom = denom if denom.abs() >= 1e-8 else torch.full_like(denom, 1e-8)
        return (v - base_obj) / denom

    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        probs = torch.sigmoid(la_req)
        p = probs / (probs.sum() + 1e-8)
        is_del = step % 2 == 1
        if strat_budgets is not None:
            budget = float(strat_budgets["del" if is_del else "ins"].pop())
        else:
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
        w = (p * budget).clamp(max=1.0)
        state_w = (1.0 - w) if is_del else w
        logits, h_probe = state_forward(runner, state_w)
        act = torch.softmax(logits, dim=0)[runner.target]
        rec_term = deletion_weight * _rec(act) if is_del else (1.0 - _rec(act))
        loss = rec_term + irr_weight * (probs * irr).sum() + l1_weight * probs.sum()
        if kmat is not None and ktv_weight > 0:
            diff = (probs.view(-1, 1) - probs.view(1, -1)).abs()
            loss = loss + ktv_weight * (kmat * diff).sum() / 2.0
        loss.backward()
        # analytic reg-grad subtraction -> rec_g without a second backward
        # (ktv grad contaminates rec_g slightly; acceptable for the ds channel)
        g_total = la_req.grad.detach()
        sig = probs.detach()
        reg_g = (irr_weight * irr + l1_weight) * sig * (1.0 - sig)
        rec_g = g_total - reg_g
        if is_del:
            recs.setdefault("ds", torch.zeros(N, device=dev, dtype=dtype))
            recs["ds"] += (-rec_g).clamp(min=0.0)
        if h_probe.grad is not None:
            sgn = (1.0 / max(deletion_weight, 1e-8)) if is_del else -1.0
            g = h_probe.grad.detach()[0] * sgn * scale
            recs["hg"].append(g.norm(dim=-1).cpu().numpy())
            recs["ixg"].append((g * runner.h_diff[0]).sum(-1).cpu().numpy())
            recs["dens"].append(float(state_w.detach().mean().item()))
            recs["phase"].append("del" if is_del else "ins")
        g = la_req.grad.detach()
        t = step + 1
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g * g
        adam_dir = (m_v / (1 - beta1 ** t)) / ((v_v / (1 - beta2 ** t)).sqrt() + eps)
        cmask = (adam_dir * g > 0).to(dtype)
        cmask = cmask * (N / cmask.sum().clamp(min=1.0))
        la = la - cur_lr * adam_dir * cmask

    ds = recs.get("ds")
    return {
        "final": torch.sigmoid(la).detach().cpu().numpy(),
        "la": la.detach().cpu().numpy(),
        "ds": (ds.cpu().numpy() if ds is not None else np.zeros(N, np.float32)),
        "hg": np.stack(recs["hg"]), "ixg": np.stack(recs["ixg"]),
        "dens": np.asarray(recs["dens"]), "phase": np.asarray(recs["phase"]),
    }


def build_readouts(tr, hg_full, ixg_full):
    final = tr["final"]
    hg, ixg, dens, phase = tr["hg"], tr["ixg"], tr["dens"], tr["phase"]
    dense_sel = dens >= 0.7
    mid_ins = (dens >= 0.4) & (dens < 0.7) & (phase == "ins")

    def pool(rows, weights):
        out = np.zeros(N)
        for r, w in zip(rows, weights):
            out += w * (r / (r.sum() + 1e-12))
        return out

    hgd = pool(hg[dense_sel], np.ones(dense_sel.sum()))
    hgd_f = hgd + 1.0 * (hg_full / (hg_full.sum() + 1e-12))
    hgd_m = hgd + pool(hg[mid_ins], 0.5 * np.ones(mid_ins.sum())) if mid_ins.any() else hgd
    hgd_fm = hgd_f + (hgd_m - hgd)

    sd = ixg[dense_sel].sum(0)
    med = np.median(sd)
    mad = np.median(np.abs(sd - med)) + 1e-12
    z = (sd - sd.mean()) / (sd.std() + 1e-12)
    zmin = float(z.min())
    sd_min_mads = float((sd.min() - 0.0) / mad)
    gate_defs = {
        "": np.ones(N),
        "_gmad3": np.where(sd < -3.0 * mad, 0.2, 1.0),
        "_gsw": np.where(z < -1.0, 0.3, 1.0) if sd.min() < -4.0 * mad else np.ones(N),
        "_gsw2": np.where(sd < -2.0 * mad, 0.3, 1.0) if sd.min() < -4.0 * mad else np.ones(N),
    }
    out = {"final": final}
    for mem_name, mem in (("hgd", hgd), ("hgdf", hgd_f), ("hgdm", hgd_m), ("hgdfm", hgd_fm)):
        for gn, gv in gate_defs.items():
            mg = mem * gv
            for lam, lt in ((1.0, "1"), (2.0, "2")):
                out[f"bn_{mem_name}_{lt}{gn}"] = lam * _n01(final) + _n01(mg)
    stats = {"zmin": zmin, "sdmin_mads": sd_min_mads,
             "n_below_2mad": int((sd < -2 * mad).sum()),
             "n_below_3mad": int((sd < -3 * mad).sum()),
             "n_dense": int(dense_sel.sum()), "n_midins": int(mid_ins.sum())}
    return out, stats


def load_cases(spec):
    df = pd.read_csv(CSV)
    losers = [16, 38, 94, 88, 18, 93, 40, 49]
    controls = [76, 21, 59, 79]
    cases = []
    if spec in ("subset", "losers"):
        sel = losers if spec == "losers" else losers + controls
        for idx in sel:
            row = df[df["idx"] == idx].iloc[0]
            cases.append({
                "name": f"{'L' if idx in losers else 'C'}{idx}_{row['desc'].split(',')[0][:12]}",
                "path": row["path"], "target": int(row["target"]),
                "csv": {"inflow_hdel": float(row["inflow_hdel"])}})
    if spec in ("subset", "zebra"):
        cases.append({"name": "zebra_elephant386", "path": str(ZEBRA), "target": 386, "csv": {}})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cases", default="subset")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/peel_research/hgrad_v5.json")
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()
    import timm
    syms = load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    results = {}
    t0 = time.time()
    for case in load_cases(args.cases):
        name = case["name"]
        x, _ = syms["load_image"](Path(case["path"]))
        x = x.to(args.device)
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = Block0Runner(model, x, h_b0, cls_tok, base, case["target"])
        irr, hg_full, ixg_full = make_irr_and_hgfull(runner)
        tr = solve_collect(runner, steps=32, seed=42, irr=irr)
        reads, stats = build_readouts(tr, hg_full, ixg_full)
        entry = {"csv": case["csv"], "stats": stats, "methods": {}}
        for rname, sc in reads.items():
            hins, hdel, _, _ = hard_curves(runner, sc, chunk=args.chunk)
            entry["methods"][rname] = {"hins": hins, "hdel": hdel}
        results[name] = entry
        print(f"[{name}] final={entry['methods']['final']['hdel']:.3f} "
              f"bn_hgdf_1={entry['methods']['bn_hgdf_1']['hdel']:.3f} "
              f"bn_hgdf_1_gsw={entry['methods']['bn_hgdf_1_gsw']['hdel']:.3f} "
              f"zmin={stats['zmin']:.2f} sdminMAD={stats['sdmin_mads']:.1f} "
              f"(inflow {case['csv'].get('inflow_hdel', float('nan')):.3f})", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    methods = list(next(iter(results.values()))["methods"].keys())
    groups = {}
    for grp, pred in (("losers", lambda n: n.startswith("L")),
                      ("controls", lambda n: n.startswith("C")),
                      ("zebra", lambda n: n.startswith("zebra"))):
        sel = [r for n, r in results.items() if pred(n)]
        if sel:
            groups[grp] = sel
    print("\n=== mean hdel / hins ===")
    print(f"{'method':<20}" + "".join(f" | {g:<13}" for g in groups))
    for m in methods:
        line = f"{m:<20}"
        for g, sel in groups.items():
            hd = np.mean([s["methods"][m]["hdel"] for s in sel])
            hi = np.mean([s["methods"][m]["hins"] for s in sel])
            line += f" | {hd:.3f}/{hi:.3f}"
        print(line)
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
