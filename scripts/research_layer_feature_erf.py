#!/usr/bin/env python3
"""Does MODEL-AGNOSTIC FRI insertion/deletion work for INTERMEDIATE-layer
features (the original FRI vision: any feature, any layer)?

Target = a token's representation at block L (a concrete intermediate feature).
Model-agnostic: only needs the layer-L activation + autograd (no attention, no
position-matching, no hidden-cluster). For sampled tokens at layers {2,5,8,11}:
  hins = insertion AUC of cos(token_j(keep top-k), full)   [sufficiency]
  hdel = deletion  AUC of cos(token_j(del  top-k), full)   [necessity]
via a pure-insertion FRI solve (the model-agnostic tool) vs random order.
Hypothesis: early layers (local, low-redundancy) -> both work; late layers ->
deletion degrades (redundancy needs modules).
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


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, REPO / path)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


xm = _load("xm", "scripts/research_xmodel_mechanism.py")
MODELS = xm.MODELS


def tokensL(r, masks_np, L, chunk=24):
    """token reps at block L for a batch of masks -> [B,N,C]."""
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
    return torch.cat(acc, 0).numpy()


def fri_solve_token(r, L, j, full_tok, steps=32, lr=0.45, lr_end=0.01, l1=0.003, init=0.5, seed=42):
    """model-agnostic pure-insertion FRI solve: maximize cos(token_j@L, full_tok)."""
    dev, dtype, N = r.dev, r.dtype, r.N
    m = r.model

    def tok(state):
        h_mix = r.base + state.unsqueeze(0).unsqueeze(-1) * r.h_diff
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
        t = store["h"][0, r.npref + j]
        return t / t.norm().clamp(min=1e-8)
    ft = torch.as_tensor(full_tok, device=dev, dtype=dtype); ft = ft / ft.norm().clamp(min=1e-8)
    with torch.no_grad():
        base_obj = float((tok(torch.zeros(N, device=dev, dtype=dtype)) @ ft))
    den = max(1.0 - base_obj, 1e-6)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    la = torch.full((N,), math.log(init / (1 - init)), device=dev, dtype=dtype)
    b1, b2, eps = 0.9, 0.999, 1e-8
    mv = torch.zeros(N, device=dev, dtype=dtype); vv = torch.zeros(N, device=dev, dtype=dtype)
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        p = torch.sigmoid(la_req)
        budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
        state = (p / (p.sum() + 1e-8) * budget).clamp(max=1.0)
        rec = ((tok(state) @ ft) - base_obj) / den
        loss = (1 - rec) + l1 * p.sum()
        g = torch.autograd.grad(loss, la_req)[0].detach()
        t = step + 1
        mv = b1 * mv + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
        adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
        cmask = (adam * g > 0).to(dtype); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
        la = la - cur * adam * cmask
    return torch.sigmoid(la).detach().float().cpu().numpy()


def grad_feature_erf(r, L, j, full_tok):
    """inflow-ERF analog for a feature: |d cos(token_j@L, full) / d mask| per patch
    (input×grad sensitivity of the feature to each patch's presence). 1 backward."""
    dev, dtype, N = r.dev, r.dtype, r.N
    m = r.model
    mask = torch.ones(N, device=dev, dtype=dtype, requires_grad=True)
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
    t = store["h"][0, r.npref + j]
    ft = torch.as_tensor(full_tok, device=dev, dtype=dtype); ft = ft / ft.norm().clamp(min=1e-8)
    obj = (t / t.norm().clamp(min=1e-8)) @ ft
    g = torch.autograd.grad(obj, mask)[0].abs()
    return g.float().cpu().numpy()


def curves(r, L, j, full_tok, score, ks):
    N = r.N
    order = np.argsort(-score)
    ins = np.zeros((len(ks), N), np.float32); de = np.ones((len(ks), N), np.float32)
    for a, k in enumerate(ks):
        ins[a, order[:k]] = 1.0; de[a, order[:k]] = 0.0
    ft = full_tok / (np.linalg.norm(full_tok) + 1e-8)
    ti = tokensL(r, ins, L)[:, j]; td = tokensL(r, de, L)[:, j]
    ci = (ti / (np.linalg.norm(ti, axis=1, keepdims=True) + 1e-8)) @ ft
    cd = (td / (np.linalg.norm(td, axis=1, keepdims=True) + 1e-8)) @ ft
    base = float(cd[-1]); den = max(1.0 - base, 1e-6)
    xs = np.array(ks) / N
    hins = float(np.trapz(np.clip((ci - base) / den, 0, 1), xs))
    hdel = float(np.trapz(np.clip((cd - base) / den, 0, 1), xs))
    return hins, hdel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--models", nargs="+", default=["clip"])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 5, 8, 11])
    ap.add_argument("--nimg", type=int, default=3)
    ap.add_argument("--ntok", type=int, default=8)
    ap.add_argument("--del-baseline", default="mean")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/layer_feature_erf.json")
    args = ap.parse_args()
    import timm
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(123).shuffle(paths); imgs = paths[: args.nimg]
    report = {}
    for mk in args.models:
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in MODELS[mk] else {}
        model = timm.create_model(MODELS[mk], pretrained=True, **kw).eval().to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        ks = [4, 8, 16, 32, 48, 72, 120, 196]
        agg = {L: {"fri_hins": [], "fri_hdel": [], "grad_hins": [], "grad_hdel": [],
                   "rnd_hins": [], "rnd_hdel": []} for L in args.layers}
        rng = np.random.default_rng(0)
        for ip in imgs:
            x, _ = xm.load_image(ip, norm=xm.MODEL_NORM.get(mk, "clip")); x = x.to(args.device)
            r = xm.XRunner(model, x, args.device, baseline=args.del_baseline)
            N = r.N
            for L in args.layers:
                full = tokensL(r, np.ones((1, N), np.float32), L)[0]   # [N,C]
                norms = np.linalg.norm(full, axis=1); toks = np.argsort(-norms)[: args.ntok]
                for j in toks:
                    fri = fri_solve_token(r, L, int(j), full[j])
                    hi, hd = curves(r, L, int(j), full[j], fri, ks)
                    agg[L]["fri_hins"].append(hi); agg[L]["fri_hdel"].append(hd)
                    gr = grad_feature_erf(r, L, int(j), full[j])
                    hig, hdg = curves(r, L, int(j), full[j], gr, ks)
                    agg[L]["grad_hins"].append(hig); agg[L]["grad_hdel"].append(hdg)
                    rs = rng.random(N)
                    hi2, hd2 = curves(r, L, int(j), full[j], rs, ks)
                    agg[L]["rnd_hins"].append(hi2); agg[L]["rnd_hdel"].append(hd2)
            torch.cuda.empty_cache()
        rep = {L: {k: float(np.mean(v)) for k, v in agg[L].items()} for L in args.layers}
        report[mk] = rep
        print(f"\n=== {mk} feature ERF by layer: FRI vs GRAD(inflow-ERF analog) vs random ===")
        print(f"{'layer':>5} | {'FRI hins':>8} {'grad hins':>9} {'rnd hins':>8} | {'FRI hdel':>8} {'grad hdel':>9} {'rnd hdel':>8}  (hins↑ hdel↓ better)")
        for L in args.layers:
            d = rep[L]
            print(f"{L:>5} | {d['fri_hins']:>8.3f} {d['grad_hins']:>9.3f} {d['rnd_hins']:>8.3f} | "
                  f"{d['fri_hdel']:>8.3f} {d['grad_hdel']:>9.3f} {d['rnd_hdel']:>8.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
