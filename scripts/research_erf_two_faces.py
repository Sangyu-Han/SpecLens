#!/usr/bin/env python3
"""The TWO FACES of the per-token ERF, layer-resolved.

For a hidden token j at block L, its input ERF has two faces:
  necessary-ERF : which input i, when DELETED, most reduces token_j
                  (||h_L[j](full) - h_L[j](del i)||)
  sufficient-ERF: which input i, when INSERTED ALONE, best recovers token_j
                  (cos(h_L[j](only i), h_L[j](full)))   <- FRI's family

User claim to test: in EARLY layers the necessary top-1 is the SELF-patch
(i=j, the diagonal), but the sufficiency map (FRI-style) does NOT rank the
self-patch top-1.  => explains FRI insertion-good / deletion-bad.

Per layer we report, over patch tokens j:
  self-rank under necessity / sufficiency (1 = self-patch is top, lower better)
  top-1 hit rate (fraction with self-patch rank == 1)
  diagonal dominance (self score / mean off-diagonal)
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec = _ilu.spec_from_file_location("xm", REPO / "scripts/research_xmodel_mechanism.py")
xm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(xm)
from src.utils.image import load_image_clip
MODELS = xm.MODELS


def acts_all_layers(r, masks_np, chunk=24):
    """masks [B,N] -> dict L -> [B,N,C] patch-token acts at block L output."""
    m = r.model
    out_acc = None
    for s in range(0, len(masks_np), chunk):
        mb = torch.as_tensor(masks_np[s:s + chunk], device=r.dev, dtype=r.dtype)
        B = mb.shape[0]
        h_mix = r.base + mb.unsqueeze(-1) * r.h_diff
        h_inj = torch.cat([r.prefix.expand(B, -1, -1), h_mix], dim=1)
        outs = {}
        hooks = [m.blocks[i].register_forward_hook(
            lambda mod, a, o, i=i: outs.__setitem__(i, (o if torch.is_tensor(o) else o[0]).detach()))
            for i in range(len(m.blocks))]
        holder = [h_inj]
        pre = m.blocks[0].register_forward_pre_hook(lambda mod, a: (holder[0],))
        try:
            with torch.no_grad():
                m(r.x.expand(B, -1, -1, -1))
        finally:
            pre.remove()
            for h in hooks:
                h.remove()
        cur = {L: outs[L][:, r.npref:].float().cpu() for L in outs}
        if out_acc is None:
            out_acc = {L: [cur[L]] for L in cur}
        else:
            for L in cur:
                out_acc[L].append(cur[L])
    return {L: torch.cat(out_acc[L], 0) for L in out_acc}


def self_rank_stats(M, higher_is_top=True):
    """M [N_input, N_token]; for each token j rank of self input i=j.
    Returns (median_self_rank, top1_rate, median_diag_dominance)."""
    N = M.shape[0]
    ranks, diag_dom = [], []
    sign = -1.0 if higher_is_top else 1.0
    for j in range(N):
        col = M[:, j]
        order = np.argsort(sign * col)          # best first
        rank = int(np.where(order == j)[0][0]) + 1
        ranks.append(rank)
        self_s = col[j]
        off = np.delete(col, j)
        dd = float(self_s / (np.abs(off).mean() + 1e-8))
        diag_dom.append(dd)
    ranks = np.array(ranks)
    return float(np.median(ranks)), float((ranks == 1).mean()), float(np.median(diag_dom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["clip", "dinov2"])
    ap.add_argument("--nimg", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/erf_two_faces.json")
    args = ap.parse_args()
    import timm

    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(123).shuffle(paths)
    imgs = paths[: args.nimg]
    report = {}
    for mk in args.models:
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in MODELS[mk] else {}
        model = timm.create_model(MODELS[mk], pretrained=True, **kw).eval().to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        nb = len(model.blocks)
        # accumulate per-layer stats over images
        nec_rank = {L: [] for L in range(nb)}; nec_top1 = {L: [] for L in range(nb)}; nec_dd = {L: [] for L in range(nb)}
        suf_rank = {L: [] for L in range(nb)}; suf_top1 = {L: [] for L in range(nb)}; suf_dd = {L: [] for L in range(nb)}
        for ip in imgs:
            x, _ = load_image_clip(ip); x = x.to(args.device)
            r = xm.XRunner(model, x, args.device)
            N = r.N
            full = acts_all_layers(r, np.ones((1, N), np.float32))      # {L:[1,N,C]}
            # necessity: delete each input i
            delm = np.ones((N, N), np.float32); delm[np.arange(N), np.arange(N)] = 0.0
            adel = acts_all_layers(r, delm)                             # {L:[N,N,C]}
            # sufficiency: insert input i alone
            insm = np.zeros((N, N), np.float32); insm[np.arange(N), np.arange(N)] = 1.0
            ains = acts_all_layers(r, insm)
            for L in range(nb):
                f = full[L][0]                                          # [N,C]
                # necessity D[i,j] = ||f[j] - adel[L][i,j]||
                D = (f.unsqueeze(0) - adel[L]).norm(dim=-1).numpy()      # [i_input, j_token]
                # sufficiency S[i,j] = cos(ains[L][i,j], f[j])
                an = ains[L] / ains[L].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                fn = f / f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                S = (an * fn.unsqueeze(0)).sum(-1).numpy()              # [i_input, j_token]
                mr, t1, dd = self_rank_stats(D, higher_is_top=True)
                nec_rank[L].append(mr); nec_top1[L].append(t1); nec_dd[L].append(dd)
                mr, t1, dd = self_rank_stats(S, higher_is_top=True)
                suf_rank[L].append(mr); suf_top1[L].append(t1); suf_dd[L].append(dd)
            del full, adel, ains
            torch.cuda.empty_cache()
        rep = {"model": MODELS[mk], "N": r.N, "nblocks": nb, "n_img": len(imgs), "by_layer": {}}
        for L in range(nb):
            rep["by_layer"][L] = {
                "nec_self_rank": float(np.mean(nec_rank[L])), "nec_top1": float(np.mean(nec_top1[L])),
                "nec_diag_dom": float(np.mean(nec_dd[L])),
                "suf_self_rank": float(np.mean(suf_rank[L])), "suf_top1": float(np.mean(suf_top1[L])),
                "suf_diag_dom": float(np.mean(suf_dd[L]))}
        report[mk] = rep
        print(f"\n=== {mk} ({MODELS[mk]}) N={r.N} blocks={nb} ===")
        print(f"{'L':>3} | {'NEC self-rank':>13} {'top1':>6} {'diagdom':>8} | {'SUF self-rank':>13} {'top1':>6} {'diagdom':>8}")
        for L in range(nb):
            d = rep["by_layer"][L]
            print(f"{L:>3} | {d['nec_self_rank']:>13.1f} {d['nec_top1']:>6.2f} {d['nec_diag_dom']:>8.1f} | "
                  f"{d['suf_self_rank']:>13.1f} {d['suf_top1']:>6.2f} {d['suf_diag_dom']:>8.1f}", flush=True)
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
