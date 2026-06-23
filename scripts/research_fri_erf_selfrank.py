#!/usr/bin/env python3
"""Does the ACTUAL FRI-ERF solve rank the self-patch top-1? (layer-resolved)

The user's exact claim: FRI (optimized sparse sufficient set) often fails to
rank the self-patch (i=j) top-1, so it is a poor NECESSARY-ERF. Here we run a
real per-token FRI insertion solve targeting recovery of hidden token j at
block L (cos(h_L[j](mask), h_L[j](full))), and compare the self-patch rank in
the FRI map vs in the necessity map (delete each input -> token j drop).

For sampled tokens j and layers L: FRI-ERF self-rank vs necessity self-rank.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec = _ilu.spec_from_file_location("xm", REPO / "scripts/research_xmodel_mechanism.py")
xm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(xm)
from src.utils.image import load_image_clip
MODELS = xm.MODELS


def token_at(r, mask, L, j, grad=False):
    """token j at block L output for a single mask [N] (grad-capable)."""
    m = r.model
    h_mix = r.base + mask.unsqueeze(0).unsqueeze(-1) * r.h_diff
    h_inj = torch.cat([r.prefix, h_mix], dim=1)
    store = {}
    hk = m.blocks[L].register_forward_hook(
        lambda mod, a, o: store.__setitem__("h", (o if torch.is_tensor(o) else o[0])))
    holder = [h_inj]
    pre = m.blocks[0].register_forward_pre_hook(lambda mod, a: (holder[0],))
    try:
        m(r.x)
    finally:
        pre.remove(); hk.remove()
    return store["h"][0, r.npref + j]


def fri_erf_token(r, L, j, steps=32, lr=0.45, lr_end=0.01, l1_weight=0.003,
                  irr_weight=0.05, init_prob=0.5, seed=42):
    """Pure-ins FRI solve to recover hidden token j at block L. Returns soft mask."""
    dev, dtype, N = r.dev, r.dtype, r.N
    with torch.no_grad():
        full_tok = token_at(r, torch.ones(N, device=dev, dtype=dtype), L, j).detach()
        ft = full_tok / full_tok.norm().clamp(min=1e-8)
        base_tok = token_at(r, torch.zeros(N, device=dev, dtype=dtype), L, j).detach()
        bt = base_tok / base_tok.norm().clamp(min=1e-8)
        base_obj = (bt @ ft)
    den = (1.0 - base_obj); den = den if den.abs() >= 1e-8 else torch.full_like(den, 1e-8)
    # irrelevance := inverse mask-sensitivity of the token-recovery target
    m1 = torch.ones(N, device=dev, dtype=dtype, requires_grad=True)
    t = token_at(r, m1, L, j); tn = t / t.norm().clamp(min=1e-8)
    g0 = torch.autograd.grad((tn @ ft), m1)[0].abs()
    inv = 1.0 / (g0 + 1e-8); irr = (inv / inv.max().clamp(min=1e-8)).detach()

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    la = torch.full((N,), math.log(init_prob / (1 - init_prob)), device=dev, dtype=dtype)
    m_v = torch.zeros(N, device=dev, dtype=dtype); v_v = torch.zeros(N, device=dev, dtype=dtype)
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        probs = torch.sigmoid(la_req)
        p = probs / (probs.sum() + 1e-8)
        budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
        state = (p * budget).clamp(max=1.0)
        t = token_at(r, state, L, j); tn = t / t.norm().clamp(min=1e-8)
        rec = ((tn @ ft) - base_obj) / den
        loss = (1.0 - rec) + irr_weight * (probs * irr).sum() + l1_weight * probs.sum()
        g = torch.autograd.grad(loss, la_req)[0].detach()
        tt = step + 1
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g * g
        adam_dir = (m_v / (1 - beta1 ** tt)) / ((v_v / (1 - beta2 ** tt)).sqrt() + eps)
        cmask = (adam_dir * g > 0).to(dtype); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
        la = la - cur_lr * adam_dir * cmask
    return torch.sigmoid(la).detach().float().cpu().numpy()


def nec_selfrank_all(r, L):
    """necessity D[i,j] = ||token_j(full) - token_j(del i)|| ; return per-token self rank."""
    N = r.N
    with torch.no_grad():
        full = token_block_all(r, np.ones((1, N), np.float32), L)[0]   # [N,C]
        delm = np.ones((N, N), np.float32); delm[np.arange(N), np.arange(N)] = 0.0
        adel = token_block_all(r, delm, L)                            # [N_input,N_tok,C]
    D = (full[None] - adel).norm(dim=-1).numpy()                       # [i,j]
    ranks = np.array([int(np.where(np.argsort(-D[:, j]) == j)[0][0]) + 1 for j in range(N)])
    return D, ranks


def token_block_all(r, masks_np, L, chunk=24):
    m = r.model; acc = []
    for s in range(0, len(masks_np), chunk):
        mb = torch.as_tensor(masks_np[s:s + chunk], device=r.dev, dtype=r.dtype)
        B = mb.shape[0]
        h_mix = r.base + mb.unsqueeze(-1) * r.h_diff
        h_inj = torch.cat([r.prefix.expand(B, -1, -1), h_mix], dim=1)
        store = {}
        hk = m.blocks[L].register_forward_hook(
            lambda mod, a, o: store.__setitem__("h", (o if torch.is_tensor(o) else o[0]).detach()))
        holder = [h_inj]
        pre = m.blocks[0].register_forward_pre_hook(lambda mod, a: (holder[0],))
        try:
            with torch.no_grad():
                m(r.x.expand(B, -1, -1, -1))
        finally:
            pre.remove(); hk.remove()
        acc.append(store["h"][:, r.npref:].float().cpu())
    return torch.cat(acc, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["clip"])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11])
    ap.add_argument("--nimg", type=int, default=2)
    ap.add_argument("--ntok", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/fri_erf_selfrank.json")
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
        per_layer = {L: {"fri": [], "nec": []} for L in args.layers}
        for ip in imgs:
            x, _ = load_image_clip(ip); x = x.to(args.device)
            r = xm.XRunner(model, x, args.device)
            N = r.N
            # sample tokens by last-hidden norm (informative patches)
            lh = r.last_hidden_patches(); norms = np.linalg.norm(lh, axis=1)
            toks = np.argsort(-norms)[: args.ntok]
            for L in args.layers:
                _, nec_ranks = nec_selfrank_all(r, L)
                for j in toks:
                    fri = fri_erf_token(r, L, int(j))
                    fri_self_rank = int(np.where(np.argsort(-fri) == j)[0][0]) + 1
                    per_layer[L]["fri"].append(fri_self_rank)
                    per_layer[L]["nec"].append(int(nec_ranks[j]))
            torch.cuda.empty_cache()
        rep = {"model": MODELS[mk], "N": r.N, "layers": {}}
        print(f"\n=== {mk} FRI-ERF vs necessity self-patch rank (median; 1=self is top) ===")
        print(f"{'L':>3} | {'FRI self-rank':>13} {'FRI top1':>9} | {'NEC self-rank':>13} {'NEC top1':>9}")
        for L in args.layers:
            fr = np.array(per_layer[L]["fri"]); nr = np.array(per_layer[L]["nec"])
            rep["layers"][L] = {"fri_self_rank_med": float(np.median(fr)), "fri_top1": float((fr == 1).mean()),
                                "nec_self_rank_med": float(np.median(nr)), "nec_top1": float((nr == 1).mean()),
                                "n": int(len(fr))}
            d = rep["layers"][L]
            print(f"{L:>3} | {d['fri_self_rank_med']:>13.1f} {d['fri_top1']:>9.2f} | "
                  f"{d['nec_self_rank_med']:>13.1f} {d['nec_top1']:>9.2f}", flush=True)
        report[mk] = rep
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
