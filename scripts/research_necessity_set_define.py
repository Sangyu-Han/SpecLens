#!/usr/bin/env python3
"""Define the NECESSITY SET like FRI defined the ERF: not the raw score/order, but
a THRESHOLD support + a filtering step.

FRI ERF = minimal patches whose INSERTION recovers 90% of the target.
Necessity set (this) = minimal patches whose REMOVAL DESTROYS 90% of the target
(recovery <= 0.1), then FILTERED so every member is truly load-bearing.

  1. greedy-conditional deletion, STOP at 90% destruction -> candidate set S.
  2. leave-one-out FILTER: drop p from S if deleting S\\{p} still destroys (>=90%)
     -> p was not necessary (a substitute in S covers it). Repeat to a fixpoint.
  3. validate: deleting S_filt destroys; deleting S_filt\\{p} does NOT (every p
     load-bearing) -> S_filt is a MINIMAL necessary set.

Reports |S_raw|, |S_filt|, frac pruned, validation, vs the 90%-suff FRI-style set
size. resnet50(N=49)+clip(N=196), class-logit target, pixel mean-baseline.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
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
ClassArch = _thn.ClassArch
_batched_keep_logits = _sp._batched_keep_logits


def greedy_necessity_set(arch, x, c, full, base, thresh, max_size):
    """greedy-conditional deletion, STOP when recovery <= thresh (90% destroyed)."""
    N = arch.gh * arch.gw; dev = arch.device
    den = abs(full - base) + 1e-6
    removed = torch.zeros(N, device=dev)
    S = []
    remaining = list(range(N))
    rec = 1.0
    while remaining and rec > thresh and len(S) < max_size:
        keep_base = 1.0 - removed
        idx = torch.tensor(remaining, device=dev)
        masks = keep_base.unsqueeze(0).repeat(len(remaining), 1)
        masks[torch.arange(len(remaining), device=dev), idx] = 0.0
        vals = _batched_keep_logits(arch, x, masks, c)
        j = int(torch.argmin(vals).item())                  # most destructive next patch
        p = remaining[j]
        removed[p] = 1.0; S.append(p); remaining.remove(p)
        rec = (float(vals[j]) - base) / den
    return S, rec


def _recovery_del(arch, x, c, del_set, full, base):
    """recovery when del_set is removed (rest kept)."""
    N = arch.gh * arch.gw; dev = arch.device
    m = torch.ones(N, device=dev)
    for q in del_set:
        m[q] = 0.0
    v = float(_batched_keep_logits(arch, x, m.unsqueeze(0), c)[0])
    return (v - base) / (abs(full - base) + 1e-6)


def filter_necessary(arch, x, c, S, full, base, thresh):
    """drop p if deleting S\\{p} still destroys (p redundant). greedy, batched LOO."""
    N = arch.gh * arch.gw; dev = arch.device
    den = abs(full - base) + 1e-6
    filt = list(S)
    while len(filt) > 1:
        masks = torch.ones(len(filt), N, device=dev)
        for r, _ in enumerate(filt):
            for q in filt:
                masks[r, q] = 0.0
            masks[r, filt[r]] = 1.0                       # keep p (= delete filt\{p})
        recs = (_batched_keep_logits(arch, x, masks, c) - base) / den
        redundant = recs <= thresh                        # still destroyed without deleting p -> p redundant
        if not bool(redundant.any()):
            break
        cand = torch.where(redundant)[0]
        j = int(cand[torch.argmin(recs[cand])].item())    # prune the most-redundant first
        filt.pop(j)
    return filt


def fri_suff_set(arch, x, c, full, base, thresh_keep, max_size):
    """FRI-analog sufficiency set: greedy-conditional INSERTION, stop when recovery
    >= thresh_keep (90% recovered). Returns the inserted set size."""
    N = arch.gh * arch.gw; dev = arch.device
    den = abs(full - base) + 1e-6
    kept = torch.zeros(N, device=dev)
    S = []; remaining = list(range(N)); rec = 0.0
    while remaining and rec < thresh_keep and len(S) < max_size:
        idx = torch.tensor(remaining, device=dev)
        masks = kept.unsqueeze(0).repeat(len(remaining), 1)
        masks[torch.arange(len(remaining), device=dev), idx] = 1.0     # insert each
        vals = _batched_keep_logits(arch, x, masks, c)
        j = int(torch.argmax(vals).item())                            # most recovering
        p = remaining[j]; kept[p] = 1.0; S.append(p); remaining.remove(p)
        rec = (float(vals[j]) - base) / den
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["resnet50", "clip"])
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--destroy", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/necessity_set_define.json")
    args = ap.parse_args()
    import torchvision.transforms as T
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths); imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    thresh = 1.0 - args.destroy
    report = {}
    for mk in args.models:
        arch = ClassArch(mk, args.device); dev = args.device; N = arch.gh * arch.gw
        rows = []
        for ip in imgs:
            img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(dev)
            x = (img - arch.mean) / arch.std
            c = arch.class_of(x)
            _, full, base = arch.obj_fn(x, c)
            if abs(full - base) < 1e-3:
                continue
            S_raw, rec_raw = greedy_necessity_set(arch, x, c, full, base, thresh, max_size=N)
            S_filt = filter_necessary(arch, x, c, S_raw, full, base, thresh)
            # validation
            destroyed = _recovery_del(arch, x, c, S_filt, full, base) <= thresh
            loadbearing = sum(1 for p in S_filt
                              if _recovery_del(arch, x, c, [q for q in S_filt if q != p], full, base) > thresh)
            suff = fri_suff_set(arch, x, c, full, base, args.destroy, max_size=N)
            rows.append({"n_raw": len(S_raw), "n_filt": len(S_filt), "destroyed": bool(destroyed),
                         "loadbearing": loadbearing, "n_suff": len(suff),
                         "overlap_nec_suff": len(set(S_filt) & set(suff)) / max(len(S_filt), 1)})
        a = lambda k: float(np.mean([r[k] for r in rows]))
        rep = {"N": N, "n": len(rows), "n_raw": a("n_raw"), "n_filt": a("n_filt"),
               "frac_pruned": 1 - a("n_filt") / max(a("n_raw"), 1e-6),
               "frac_destroyed": a("destroyed"), "frac_loadbearing": a("loadbearing") / max(a("n_filt"), 1e-6),
               "n_suff": a("n_suff"), "overlap_nec_suff": a("overlap_nec_suff")}
        report[mk] = rep
        print(f"\n=== {mk} (N={N}, n={rep['n']}) NECESSITY SET (90% destruction) ===")
        print(f"   raw greedy prefix:  {rep['n_raw']:.1f} patches")
        print(f"   after LOO filter:   {rep['n_filt']:.1f} patches  ({100*rep['frac_pruned']:.0f}% pruned as non-necessary)")
        print(f"   validation: destroys 90% in {100*rep['frac_destroyed']:.0f}% imgs; "
              f"every member load-bearing in {100*rep['frac_loadbearing']:.0f}%")
        print(f"   vs FRI-style SUFFICIENCY set: {rep['n_suff']:.1f} patches; "
              f"nec∩suff overlap {rep['overlap_nec_suff']:.2f}", flush=True)
        del arch.model; torch.cuda.empty_cache()
    args.out.write_text(__import__("json").dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
