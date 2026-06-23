#!/usr/bin/env python3
"""Robust input SET-perturbation necessity (fully MODEL-AGNOSTIC).

The hunt: a necessity method as model-agnostic as FRI (input-only, no hidden, no
architecture-specific routing). Prior input-level attempts (RBN smooth mask,
budget-Shapley) FAILED -> only MATCH inflow, never beat it. Fresh angle from
finding #7 (redundancy = OR over substitutable disjoint supports):

  A single FRI finds ONE minimal sufficient set = ONE OR branch. Deletion in
  FRI order leaves the substitutable branches intact -> target survives -> bad
  hdel (the documented insertion-good/deletion-bad asymmetry). FIX = PEEL the
  branches: find a sufficient set, BAN its core, find the NEXT (forced-different)
  sufficient branch, ... -> a deletion order that covers ALL branches,
  strongest-first. Input-only (FRI + input masking); nothing model-specific.

Methods (input-deletion hdel, mean-baseline, LOWER=better necessity):
  fri          single FRI sufficiency map        (ref: one branch -> bad del)
  fri_peel     sequential sufficient-set peeling  (NEW, the set-perturbation)
  fri_cover    soft coverage Sum_k p_k over the K diverse peeled solves (NEW)
  occlusion    single-patch input deletion        (agnostic baseline to BEAT)
  grad         input x grad of the class logit    (agnostic baseline)
  hidden_ablate ablate last-hidden position       (white-box reference)
  random
A model-agnostic necessity must be the most ROBUST across architectures: beat
occlusion/grad on BOTH resnet50 (CNN) and the ViTs.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = _ilu.spec_from_file_location(name, REPO / rel)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_thn = _load("thn", "scripts/test_hidden_ablation_necessity.py")
ClassArch, curves = _thn.ClassArch, _thn.curves


def fri_solve_restricted(arch, x, c, keep, seed=0, steps=32, lr=0.4, lr_end=0.01, l1=0.003):
    """FRI sufficiency solve restricted to the non-banned (keep==1) positions."""
    N = arch.gh * arch.gw
    dev = arch.device
    keep = keep.to(dev).float()
    with torch.no_grad():
        base = arch.masked_logit(x, torch.zeros(N, device=dev), c)
        full = arch.masked_logit(x, torch.ones(N, device=dev), c)
    den = (full - base).abs().clamp(min=1e-4)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    la = torch.zeros(N, device=dev)
    mv = torch.zeros(N, device=dev)
    vv = torch.zeros(N, device=dev)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        p = torch.sigmoid(la_req) * keep                 # forbid banned positions
        budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
        state = (p / (p.sum() + 1e-8) * budget).clamp(max=1.0)
        with torch.enable_grad():
            r = (arch.masked_logit(x, state, c) - base) / den
            loss = (1 - r) + l1 * p.sum()
            g = torch.autograd.grad(loss, la_req)[0].detach()
        t = step + 1
        mv = b1 * mv + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
        cm = (adam * g > 0).float()
        cm = cm * (N / cm.sum().clamp(min=1.0))
        la = la - cur * adam * cm
    return (torch.sigmoid(la) * keep).detach().float().cpu().numpy()


def _batched_keep_logits(arch, x, keep_masks, c, chunk=64):
    """keep_masks [M,N] (1=keep,0=mask -> mean) -> class-c logits [M] (chunked)."""
    ph, pw = 224 // arch.gh, 224 // arch.gw
    outs = []
    for s in range(0, keep_masks.shape[0], chunk):
        km = keep_masks[s:s + chunk]
        M = km.shape[0]
        pm = km.view(M, arch.gh, arch.gw).repeat_interleave(ph, 1).repeat_interleave(pw, 2)
        xb = x * pm.view(M, 1, 224, 224)
        with torch.no_grad():
            logits = arch.model(xb)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
        outs.append(logits[:, c])
    return torch.cat(outs)


def greedy_cond_nec(arch, x, c):
    """Greedy CONDITIONAL deletion = the model-agnostic deletion ORACLE: at each
    step remove the patch whose removal (given already-removed) drops the target
    most. Directly handles redundancy (once copies are gone the load-bearing
    patch's conditional marginal rises). O(N^2) forwards, batched per step.
    Returns the deletion order (most-necessary first) = upper bound on hdel."""
    N = arch.gh * arch.gw
    dev = arch.device
    removed = torch.zeros(N, device=dev)
    order: list[int] = []
    remaining = list(range(N))
    for _ in range(N):
        keep_base = 1.0 - removed
        idx = torch.tensor(remaining, device=dev)
        masks = keep_base.unsqueeze(0).repeat(len(remaining), 1)
        masks[torch.arange(len(remaining), device=dev), idx] = 0.0
        vals = _batched_keep_logits(arch, x, masks, c)       # [M] target after removing each
        j = int(torch.argmin(vals).item())                   # min target = max drop
        pick = remaining[j]
        order.append(pick)
        removed[pick] = 1.0
        remaining.pop(j)
    return np.array(order, dtype=int)


def fri_setpert(arch, x, c, K=8, p_thresh=0.4, max_core_frac=0.4):
    """Peel sufficient sets: round r solves FRI among non-banned patches, bans its
    core (p>thresh), continues. Returns (peel_order, cover_order, round0_p).
    peel deletion order = branch cores strongest-first, then leftovers by coverage."""
    N = arch.gh * arch.gw
    dev = arch.device
    keep = torch.ones(N, device=dev)
    banned = np.zeros(N, dtype=bool)
    covers = np.zeros(N, dtype=np.float32)
    order: list[int] = []
    first_p = None
    cap = max(1, int(max_core_frac * N))
    for r in range(K):
        p = fri_solve_restricted(arch, x, c, keep, seed=r)
        if first_p is None:
            first_p = p.copy()
        covers += p * (~banned)
        cand = [int(i) for i in np.argsort(-p) if (not banned[i]) and p[i] > p_thresh]
        cand = cand[:cap]
        if not cand:
            break
        order.extend(cand)                               # within-round by suff weight
        banned[cand] = True
        keep = torch.tensor((~banned).astype(np.float32), device=dev)
        if banned.all():
            break
    oset = set(order)
    rest = [int(i) for i in np.argsort(-covers) if int(i) not in oset]
    peel_order = np.array(order + rest, dtype=int)
    cover_order = np.argsort(-covers)
    return peel_order, cover_order, first_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["resnet50", "clip"])
    ap.add_argument("--nimg", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/setpert_necessity.json")
    args = ap.parse_args()

    import torchvision.transforms as T
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths)
    imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    # greedy_cond = ORACLE of INPUT-PATCH necessity ("what the model already
    # knows", by definition of the deletion game) -- a diagnostic ceiling, NOT a
    # proposed method. Everything else is a cheap candidate that tries to REVEAL it.
    methods = ["greedy_cond", "fri", "fri_peel", "fri_cover", "occlusion", "grad",
               "hidden_ablate", "random"]
    report = {}
    for mk in args.models:
        arch = ClassArch(mk, args.device)
        dev = args.device
        N = arch.gh * arch.gw
        topn = max(1, round(0.25 * N))
        agg = {m: {"hdel": [], "hins": [], "ovl": []} for m in methods}
        rng = np.random.default_rng(0)
        for ip in imgs:
            img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(dev)
            x = (img - arch.mean) / arch.std
            c = arch.class_of(x)
            obj, full, base = arch.obj_fn(x, c)
            if abs(full - base) < 1e-3:
                continue
            peel_o, cover_o, fri_p = fri_setpert(arch, x, c, K=args.K)
            orders = {
                "greedy_cond": greedy_cond_nec(arch, x, c),
                "fri": np.argsort(-fri_p),
                "fri_peel": peel_o,
                "fri_cover": cover_o,
                "occlusion": np.argsort(-arch.occlusion_nec(x, c)),
                "grad": np.argsort(-arch.grad_nec(x, c)),
                "hidden_ablate": np.argsort(-arch.hidden_ablate_nec(x, c)),
                "random": rng.permutation(N),
            }
            oracle_top = set(int(i) for i in orders["greedy_cond"][:topn])
            for m, o in orders.items():
                hd, hi = curves(obj, o, N)
                agg[m]["hdel"].append(hd)
                agg[m]["hins"].append(hi)
                agg[m]["ovl"].append(len(set(int(i) for i in o[:topn]) & oracle_top) / topn)
            torch.cuda.empty_cache()
        rep = {m: {"hdel": float(np.mean(v["hdel"])), "hins": float(np.mean(v["hins"])),
                   "ovl_oracle": float(np.mean(v["ovl"])), "n": len(v["hdel"])}
               for m, v in agg.items() if v["hdel"]}
        report[mk] = {"kind": arch.kind, "grid": N, "topn": topn, **rep}
        n = rep["grad"]["n"]
        print(f"\n=== {mk} ({arch.kind}, grid {arch.gh}x{arch.gw}, n={n}) "
              f"hdel (LOWER=better) / hins / top-{topn} overlap-with-oracle ===", flush=True)
        for m in methods:
            if m in rep:
                tag = {"greedy_cond": "  <== ORACLE (input-patch necessity ceiling)",
                       "fri_peel": "  <== set-perturbation candidate",
                       "occlusion": "  <== cheap agnostic baseline"}.get(m, "")
                print(f"   {m:14s} hdel={rep[m]['hdel']:.3f}  hins={rep[m]['hins']:.3f}  "
                      f"ovl={rep[m]['ovl_oracle']:.2f}{tag}", flush=True)
        for ref in ("occlusion", "grad", "fri"):
            a = np.array(agg["fri_peel"]["hdel"])
            b = np.array(agg[ref]["hdel"])
            if len(a) and len(a) == len(b):
                wins = int((a < b).sum())
                print(f"     fri_peel vs {ref:9s}: peel better on {wins}/{len(a)} "
                      f"(mean {a.mean():.3f} vs {b.mean():.3f})", flush=True)
        del arch.model
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
