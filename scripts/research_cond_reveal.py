#!/usr/bin/env python3
"""How cheaply can we REVEAL the sharp input-patch necessity set the model knows?

Established (setpert_necessity, n=12): the greedy-CONDITIONAL deletion oracle has
hdel ~2-2.4x better than every cheap signal (occlusion/grad/hidden/FRI). The
missing ingredient is CONDITIONING: occlusion = unconditional single-patch
marginal (saturation-blind to redundancy); the oracle re-measures each patch's
marginal GIVEN what is already removed -> handles the OR-over-substitutable
structure. Conditioning is the whole gap.

This maps the COST->QUALITY frontier of model-agnostic conditional necessity:
chunked conditional deletion in R rounds. Each round = 1 batched forward over the
remaining patches (rank by conditional marginal), commit the top ceil(N/R) at
once. R=1 == occlusion (commit all by one unconditional pass); R=N == the oracle
(commit one at a time). Question: how few rounds recover most of the oracle gain?

Pure input perturbation -> fully MODEL-AGNOSTIC, no hidden / no patch==token.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import random
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
_sp = _load("sp", "scripts/research_setpert_necessity.py")
ClassArch, curves = _thn.ClassArch, _thn.curves
_batched_keep_logits, greedy_cond_nec = _sp._batched_keep_logits, _sp.greedy_cond_nec


def cond_chunked_nec(arch, x, c, rounds):
    """Chunked greedy-conditional deletion. Returns (order, n_forward_calls)."""
    N = arch.gh * arch.gw
    dev = arch.device
    removed = torch.zeros(N, device=dev)
    order: list[int] = []
    remaining = list(range(N))
    chunk = max(1, math.ceil(N / rounds))
    calls = 0
    while remaining:
        keep_base = 1.0 - removed
        idx = torch.tensor(remaining, device=dev)
        masks = keep_base.unsqueeze(0).repeat(len(remaining), 1)
        masks[torch.arange(len(remaining), device=dev), idx] = 0.0
        vals = _batched_keep_logits(arch, x, masks, c)          # target after removing each
        calls += 1
        k = min(chunk, len(remaining))
        sel = torch.topk(-vals, k).indices.tolist()             # k most necessary (lowest target)
        picks = [remaining[j] for j in sel]
        order.extend(picks)
        for p in picks:
            removed[p] = 1.0
        pickset = set(picks)
        remaining = [r for r in remaining if r not in pickset]
    return np.array(order, dtype=int), calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["resnet50", "clip"])
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/cond_reveal.json")
    args = ap.parse_args()

    import torchvision.transforms as T
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths)
    imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    rounds = args.rounds
    cond_keys = [f"cond_r{r}" for r in rounds]
    methods = ["oracle"] + cond_keys + ["occlusion", "random"]
    report = {}
    for mk in args.models:
        arch = ClassArch(mk, args.device)
        dev = args.device
        N = arch.gh * arch.gw
        topn = max(1, round(0.25 * N))
        agg = {m: {"hdel": [], "ovl": [], "calls": []} for m in methods}
        rng = np.random.default_rng(0)
        for ip in imgs:
            img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(dev)
            x = (img - arch.mean) / arch.std
            c = arch.class_of(x)
            obj, full, base = arch.obj_fn(x, c)
            if abs(full - base) < 1e-3:
                continue
            orders = {}
            orders["oracle"], oc_calls = greedy_cond_nec(arch, x, c), N
            for r, key in zip(rounds, cond_keys):
                o, calls = cond_chunked_nec(arch, x, c, r)
                orders[key] = o
                agg[key]["calls"].append(calls)
            orders["occlusion"] = np.argsort(-arch.occlusion_nec(x, c))
            orders["random"] = rng.permutation(N)
            agg["oracle"]["calls"].append(oc_calls)
            agg["occlusion"]["calls"].append(1)
            agg["random"]["calls"].append(0)
            oracle_top = set(int(i) for i in orders["oracle"][:topn])
            for m, o in orders.items():
                hd, _ = curves(obj, o, N)
                agg[m]["hdel"].append(hd)
                agg[m]["ovl"].append(len(set(int(i) for i in o[:topn]) & oracle_top) / topn)
            torch.cuda.empty_cache()
        rep = {m: {"hdel": float(np.mean(v["hdel"])), "ovl_oracle": float(np.mean(v["ovl"])),
                   "calls": float(np.mean(v["calls"])) if v["calls"] else 0.0,
                   "n": len(v["hdel"])} for m, v in agg.items() if v["hdel"]}
        report[mk] = {"kind": arch.kind, "grid": N, "topn": topn, **rep}
        n = rep["occlusion"]["n"]
        # fraction of oracle gain recovered: (occ - method) / (occ - oracle)
        occ_h, ora_h = rep["occlusion"]["hdel"], rep["oracle"]["hdel"]
        span = max(occ_h - ora_h, 1e-6)
        print(f"\n=== {mk} ({arch.kind}, grid {arch.gh}x{arch.gw}, N={N}, n={n}) "
              f"conditional-necessity cost->quality ===", flush=True)
        print(f"   {'method':10s} {'hdel↓':>7s} {'ovl':>5s} {'calls':>6s} {'%gain':>6s}", flush=True)
        for m in methods:
            if m in rep:
                gain = 100.0 * (occ_h - rep[m]["hdel"]) / span
                print(f"   {m:10s} {rep[m]['hdel']:>7.3f} {rep[m]['ovl_oracle']:>5.2f} "
                      f"{rep[m]['calls']:>6.0f} {gain:>5.0f}%", flush=True)
        del arch.model
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
