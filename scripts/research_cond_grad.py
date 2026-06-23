#!/usr/bin/env python3
"""Cheap conditional necessity: can a CONDITIONAL GRADIENT (1 backward per round at
the current removed state) recover the greedy-conditional oracle nearly as well as
the occlusion marginal (N forwards per round), at ~N x lower cost?

Single-pass gradient fails (saturation-blind, like occlusion). But the missing
ingredient is CONDITIONING: re-computing the gradient AFTER each commit de-saturates
it (the removed copies stop hiding the rest). Chunked, R rounds:
  - cond_r{R}      : occlusion marginal  (~R*N forwards)   [established best]
  - cond_grad_r{R} : gradient at removed state (R backwards) [cheap candidate]
Eval = class-logit deletion hdel + %oracle-gap recovered. resnet50(N=49)+clip(N=196).
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = _ilu.spec_from_file_location(name, REPO / rel)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


_thn = _load("thn", "scripts/test_hidden_ablation_necessity.py")
_sp = _load("sp", "scripts/research_setpert_necessity.py")
_cr = _load("cr", "scripts/research_cond_reveal.py")
ClassArch, curves = _thn.ClassArch, _thn.curves
greedy_cond_nec = _sp.greedy_cond_nec
cond_chunked_nec = _cr.cond_chunked_nec


def cond_chunked_grad(arch, x, c, rounds):
    """Chunked greedy-conditional deletion ranked by the gradient of the class logit
    wrt the keep-mask AT THE CURRENT REMOVED STATE (1 backward per round)."""
    N = arch.gh * arch.gw; dev = arch.device
    removed = torch.zeros(N, device=dev)
    order: list[int] = []; remaining = list(range(N))
    chunk = max(1, math.ceil(N / rounds)); back = 0
    while remaining:
        keep = (1.0 - removed).clone().requires_grad_(True)
        pix = arch.pixel_mask(keep).view(1, 1, 224, 224)
        lg = arch.logits(x * pix)[c]
        g = torch.autograd.grad(lg, keep)[0].detach()      # ∂logit/∂keep_i : high = necessary
        back += 1
        g = g.clone(); g[removed.bool()] = -1e9
        k = min(chunk, len(remaining))
        sel = torch.topk(g, k).indices.tolist()
        order.extend(sel)
        for p in sel:
            removed[p] = 1.0
        remaining = [r for r in remaining if r not in set(sel)]
    return np.array(order, dtype=int), back


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["resnet50", "clip"])
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--rounds", nargs="+", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/cond_grad.json")
    args = ap.parse_args()
    import torchvision.transforms as T
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths); imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    rounds = args.rounds
    keys = ["oracle"] + [f"cond_r{r}" for r in rounds] + [f"cond_grad_r{r}" for r in rounds] + ["grad1", "random"]
    report = {}
    for mk in args.models:
        arch = ClassArch(mk, args.device); dev = args.device; N = arch.gh * arch.gw
        topn = max(1, round(0.25 * N))
        agg = {m: {"hdel": [], "ovl": []} for m in keys}
        rng = np.random.default_rng(0)
        for ip in imgs:
            img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(dev)
            x = (img - arch.mean) / arch.std
            c = arch.class_of(x)
            obj, full, base = arch.obj_fn(x, c)
            if abs(full - base) < 1e-3:
                continue
            orders = {"oracle": greedy_cond_nec(arch, x, c)}
            for r in rounds:
                orders[f"cond_r{r}"] = cond_chunked_nec(arch, x, c, r)[0]
                orders[f"cond_grad_r{r}"] = cond_chunked_grad(arch, x, c, r)[0]
            orders["grad1"] = cond_chunked_grad(arch, x, c, 1)[0]      # 1-pass gradient (R=1)
            orders["random"] = rng.permutation(N)
            otop = set(int(i) for i in orders["oracle"][:topn])
            for m, o in orders.items():
                hd, _ = curves(obj, o, N)
                agg[m]["hdel"].append(hd)
                agg[m]["ovl"].append(len(set(int(i) for i in o[:topn]) & otop) / topn)
            torch.cuda.empty_cache()
        rep = {m: {"hdel": float(np.mean(v["hdel"])), "ovl": float(np.mean(v["ovl"])), "n": len(v["hdel"])}
               for m, v in agg.items() if v["hdel"]}
        report[mk] = rep
        occ = rep.get("cond_r1", rep["grad1"])
        ora_h = rep["oracle"]["hdel"]
        # reference for %gain = the R=1 unconditional (use grad1 as the cheap floor)
        base_h = rep["grad1"]["hdel"]
        span = max(base_h - ora_h, 1e-6)
        print(f"\n=== {mk} (N={N}, n={rep['oracle']['n']}) conditional-GRADIENT (R bwd) vs occlusion-marginal (R*N fwd) ===")
        print(f"   {'method':16s} {'hdel↓':>7s} {'ovl':>5s} {'%gain':>6s}  {'cost':>14s}")
        for m in keys:
            if m in rep:
                gain = 100.0 * (base_h - rep[m]["hdel"]) / span
                cost = ("oracle ~N²/2 fwd" if m == "oracle" else
                        f"{m.split('_r')[-1]} bwd" if "grad_r" in m else
                        f"~{m.split('_r')[-1]}*N fwd" if "cond_r" in m else
                        "1 bwd" if m == "grad1" else "0")
                print(f"   {m:16s} {rep[m]['hdel']:>7.3f} {rep[m]['ovl']:>5.2f} {gain:>5.0f}%  {cost:>14s}", flush=True)
        del arch.model; torch.cuda.empty_cache()
    args.out.write_text(__import__("json").dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
