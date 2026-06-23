#!/usr/bin/env python3
"""Does a RICHER conditional marginal (KL of the full output distribution) recover
the greedy-conditional necessity ORACLE in FEWER rounds than the scalar-logit
marginal? rich-KL was the best 1-pass necessity signal (beats scalar occlusion +
grad on input-deletion hdel across resnet/clip/deit3/augreg); test whether it also
makes the CONDITIONAL rounds more efficient (cut R = cut cost).

Chunked greedy-conditional deletion, R rounds, commit top ceil(N/R) per round.
  - cond_r{R}    : commit by scalar class-logit marginal      (existing, baseline)
  - cond_kl_r{R} : commit by KL(p_current || p_remove-i)       (rich marginal)
Eval = class-logit deletion hdel (the actual necessity metric) + %oracle-gap
recovered + top-25% overlap with the O(N^2) oracle. resnet50(N=49)+clip(N=196).
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


def _batched_full_logits(arch, x, keep_masks, chunk=48):
    """keep_masks [M,N] -> full logits [M,C] (pixel mean-mask, chunked)."""
    ph, pw = 224 // arch.gh, 224 // arch.gw
    outs = []
    for s in range(0, keep_masks.shape[0], chunk):
        km = keep_masks[s:s + chunk]; M = km.shape[0]
        pm = km.view(M, arch.gh, arch.gw).repeat_interleave(ph, 1).repeat_interleave(pw, 2)
        with torch.no_grad():
            lg = arch.model(x * pm.view(M, 1, 224, 224))
            if isinstance(lg, (tuple, list)):
                lg = lg[0]
        outs.append(lg)
    return torch.cat(outs)


def cond_chunked_kl(arch, x, c, rounds):
    """Chunked greedy-conditional deletion ranked by KL(p_current || p_remove-i)."""
    N = arch.gh * arch.gw; dev = arch.device
    removed = torch.zeros(N, device=dev)
    order: list[int] = []; remaining = list(range(N))
    chunk = max(1, math.ceil(N / rounds)); calls = 0
    while remaining:
        keep_base = 1.0 - removed
        lp_cur = torch.log_softmax(_batched_full_logits(arch, x, keep_base.unsqueeze(0))[0], 0)
        p_cur = lp_cur.exp()
        idx = torch.tensor(remaining, device=dev)
        masks = keep_base.unsqueeze(0).repeat(len(remaining), 1)
        masks[torch.arange(len(remaining), device=dev), idx] = 0.0
        lg = _batched_full_logits(arch, x, masks)              # [M,C]
        calls += 1
        lq = torch.log_softmax(lg, 1)
        kl = (p_cur.unsqueeze(0) * (lp_cur.unsqueeze(0) - lq)).sum(1)   # KL [M], higher=more necessary
        k = min(chunk, len(remaining))
        sel = torch.topk(kl, k).indices.tolist()
        picks = [remaining[j] for j in sel]
        order.extend(picks)
        for p in picks:
            removed[p] = 1.0
        pickset = set(picks); remaining = [r for r in remaining if r not in pickset]
    return np.array(order, dtype=int), calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["resnet50", "clip"])
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--rounds", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/cond_reveal_kl.json")
    args = ap.parse_args()
    import torchvision.transforms as T
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths); imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    rounds = args.rounds
    keys = ["oracle"] + [f"cond_r{r}" for r in rounds] + [f"cond_kl_r{r}" for r in rounds] + ["occlusion", "random"]
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
                orders[f"cond_kl_r{r}"] = cond_chunked_kl(arch, x, c, r)[0]
            orders["occlusion"] = np.argsort(-arch.occlusion_nec(x, c)) if hasattr(arch, "occlusion_nec") else cond_chunked_nec(arch, x, c, 1)[0]
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
        occ_h, ora_h = rep["occlusion"]["hdel"], rep["oracle"]["hdel"]
        span = max(occ_h - ora_h, 1e-6)
        print(f"\n=== {mk} (N={N}, n={rep['oracle']['n']}) conditional necessity: KL-marginal vs logit-marginal ===")
        print(f"   {'method':14s} {'hdel↓':>7s} {'ovl':>5s} {'%oracle-gain':>13s}")
        for m in keys:
            if m in rep:
                gain = 100.0 * (occ_h - rep[m]["hdel"]) / span
                print(f"   {m:14s} {rep[m]['hdel']:>7.3f} {rep[m]['ovl']:>5.2f} {gain:>12.0f}%", flush=True)
        del arch.model; torch.cuda.empty_cache()
    args.out.write_text(__import__("json").dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
