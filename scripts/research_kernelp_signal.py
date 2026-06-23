#!/usr/bin/env python3
"""What signal lets kernelP-MS separate target/competitor/background?

Zebra/elephant case (region masks from ZTRACE). One big field-state
collection (masks, p_target, logit_target, margin, p_competitor), then:

  A. side decomposition  : suff-DIM vs nec-DIM region means (ELE/ZEB/BG)
  B. density binning     : where (in coalition density) the signal lives
  C. geometry ablation   : iid vs kernelP at matched state counts -> is
                           coherence required for the SIGNAL or only for
                           variance reduction?
  D. objective ablation  : DIM on prob vs logit vs margin -> softmax-
                           coupling test (competitor negativity should die
                           on raw logit if it comes from renormalization)
  E. few-state extraction: kernel-ridge regression  p ~ M beta with
                           pos-kernel smoothness prior, at n=64/128/256
                           states -> the FRI-budget extraction candidate.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = _ilu.spec_from_file_location("cw", REPO / "scripts/research_corefri_rw.py")
cw = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cw)
ns, q = cw.ns, cw.q
N, GRID = 196, 14
ZTRACE = "/home/sangyu/Desktop/Master/_fri_research_archive_20260610/outputs/research_step_signals/zebra_elephant_trace.npz"


def collect_states(runner, mode, n_states, seed=42):
    """Sample dual-sided field/iid states; record mask, p, logit, margin,
    p_comp for each."""
    rng = np.random.default_rng(seed)
    dev, dtype = runner.dev, runner.dtype
    chols = None
    if mode == "kernelP_ms":
        with torch.no_grad():
            h = runner.base[0].detach().float()
            hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            S = (hn @ hn.T)
        chols = [torch.linalg.cholesky(
            torch.exp((S - 1.0) / t) + 1e-3 * torch.eye(N, device=S.device)
        ).to(dtype) for t in (0.6, 0.25, 0.1)]

    def gen(n, density):
        k = int(round(density * N))
        k = min(max(k, 1), N - 1)
        out = np.zeros((n, N), np.float32)
        for i in range(n):
            if mode == "kernelP_ms":
                si = int(rng.integers(3))
                z = torch.randn(N, device=dev, dtype=dtype)
                f = (chols[si] @ z).cpu().numpy()
            else:
                f = rng.standard_normal(N)
            out[i, np.argpartition(-f, k)[:k]] = 1.0
        return out

    half = n_states // 2
    dens_s = np.exp(rng.uniform(np.log(0.05), np.log(0.5), half))
    dens_d = rng.uniform(0.1, 0.5, n_states - half)
    masks, sides, dens = [], [], []
    for d in dens_s:
        masks.append(gen(1, float(d))[0])
        sides.append("ins")
        dens.append(float(d))
    for d in dens_d:
        masks.append(1.0 - gen(1, float(d))[0])
        sides.append("del")
        dens.append(float(d))
    masks = np.stack(masks)

    # competitor = top non-target of the full image
    with torch.no_grad():
        full_logits = runner.logits_for_masks(
            torch.ones(1, N, device=dev, dtype=dtype))[0]
        comp_cls = int(torch.argsort(-full_logits)[1].item()) \
            if int(torch.argmax(full_logits)) == runner.target \
            else int(torch.argmax(full_logits))
        vals = {"p": [], "logit": [], "margin": [], "p_comp": []}
        for i in range(0, len(masks), 96):
            mt = torch.as_tensor(masks[i:i + 96], device=dev, dtype=dtype)
            logits = runner.logits_for_masks(mt)
            sm = torch.softmax(logits, -1)
            vals["p"].append(sm[:, runner.target].cpu().numpy())
            vals["p_comp"].append(sm[:, comp_cls].cpu().numpy())
            vals["logit"].append(logits[:, runner.target].cpu().numpy())
            comp = logits.clone()
            comp[:, runner.target] = -torch.inf
            vals["margin"].append(
                (logits[:, runner.target]
                 - torch.logsumexp(comp.topk(20, dim=1).values, dim=1)).cpu().numpy())
    return (masks, np.asarray(sides), np.asarray(dens),
            {k: np.concatenate(v) for k, v in vals.items()}, comp_cls)


def dim_scores(masks, vals, rows=None):
    if rows is None:
        rows = np.ones(len(masks), bool)
    M, v = masks[rows], vals[rows]
    out = np.zeros(N)
    for i in range(N):
        a, b = v[M[:, i] > 0.5], v[M[:, i] <= 0.5]
        if len(a) >= 2 and len(b) >= 2:
            out[i] = a.mean() - b.mean()
    return out


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    name, path, target = ns.CASES[0]  # zebra/elephant, target 386
    x, _ = syms["load_image"](Path(path))
    x = x.to(args.device)
    h_b0, cls_tok, base = syms["get_b0"](model, x)
    runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)

    zt = np.load(ZTRACE)
    ELE = np.asarray(zt["region_tgt"]).ravel().astype(int)
    ZEB = np.asarray(zt["region_dis"]).ravel().astype(int)
    BG = np.asarray(zt["region_bg"]).ravel().astype(int)

    def reg(v):
        return f"ELE={v[ELE].mean():+.2f} ZEB={v[ZEB].mean():+.2f} BG={v[BG].mean():+.2f}"

    t0 = time.time()
    masks, sides, dens, vals, comp_cls = collect_states(
        runner, "kernelP_ms", 4096, seed=42)
    print(f"[collect] kernelP-MS 4096 states ({time.time()-t0:.0f}s) "
          f"competitor_cls={comp_cls}")

    # A. side decomposition (z-scored DIM per side)
    ins_rows, del_rows = sides == "ins", sides == "del"
    suff = zscore(dim_scores(masks, vals["p"], ins_rows))
    nec = zscore(dim_scores(masks, vals["p"], del_rows))
    print("\nA. side decomposition (DIM on p_target):")
    print(f"   suff(ins side): {reg(suff)}")
    print(f"   nec (del side): {reg(nec)}")

    # B. density binning (ins side; del side analog)
    print("\nB. density bins:")
    for side, rows0 in (("ins", ins_rows), ("del", del_rows)):
        for lo, hi in ((0.05, 0.15), (0.15, 0.3), (0.3, 0.5)):
            rows = rows0 & (dens >= lo) & (dens < hi)
            if rows.sum() < 40:
                continue
            d = zscore(dim_scores(masks, vals["p"], rows))
            print(f"   {side} d=[{lo:.2f},{hi:.2f}) n={rows.sum():4d}  {reg(d)}")

    # C. geometry ablation: iid at matched counts
    print("\nC. geometry ablation (z(suff)+z(nec) region means):")
    for mode in ("kernelP_ms", "iid"):
        for n_st in (4096, 512, 128):
            m2, s2, _, v2, _ = collect_states(runner, mode, n_st, seed=7)
            sc = zscore(dim_scores(m2, v2["p"], s2 == "ins")) + \
                zscore(dim_scores(m2, v2["p"], s2 == "del"))
            print(f"   {mode:10} @{n_st:4d}: {reg(sc)}")

    # D. objective ablation (full 4k, both sides z-sum)
    print("\nD. objective ablation:")
    for key in ("p", "logit", "margin", "p_comp"):
        sc = zscore(dim_scores(masks, vals[key], ins_rows)) + \
            zscore(dim_scores(masks, vals[key], del_rows))
        print(f"   {key:7}: {reg(sc)}")

    # E. kernel-ridge extraction at few states
    print("\nE. kernel-ridge extraction (p ~ M beta, pos-kernel prior):")
    with torch.no_grad():
        h = runner.base[0].detach().float()
        hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = (hn @ hn.T).cpu().numpy().astype(np.float64)
    Kp = np.exp((S - 1.0) / 0.25) + 1e-3 * np.eye(N)
    Kinv = np.linalg.inv(Kp)
    rng = np.random.default_rng(0)
    for n_st in (64, 128, 256):
        for mode_tag, msel, vsel in (("field", masks, vals["p"]),):
            idx = rng.choice(len(msel), n_st, replace=False)
            M = msel[idx].astype(np.float64)
            v = vsel[idx].astype(np.float64)
            Mc = M - M.mean(0)
            vc = v - v.mean()
            for lam_tag, A in (("ridge", np.eye(N)), ("Kprior", Kinv)):
                lam = 1e-3 * n_st
                beta = np.linalg.solve(Mc.T @ Mc + lam * A, Mc.T @ vc)
                b = zscore(beta)
                print(f"   n={n_st:3d} {lam_tag:6}: {reg(b)}")
    # E2. ridge on the SOLVE'S OWN visited states (zero extra forwards)
    print("\nE2. solve-state kernel-ridge (per-side fit, margin target):")
    import math as _math

    def solve_record(seed=42, steps=32):
        dev, dtype = runner.dev, runner.dtype
        with torch.no_grad():
            full_obj = runner.probs_for_masks(torch.ones(1, N, device=dev, dtype=dtype))[0]
            base_obj = runner.probs_for_masks(torch.zeros(1, N, device=dev, dtype=dtype))[0]
        def rec_of(v):
            den = full_obj - base_obj
            den = den if den.abs() >= 1e-8 else torch.full_like(den, 1e-8)
            return (v - base_obj) / den
        h_var = runner.h_b0.detach().clone().requires_grad_(True)
        h_inj = torch.cat([runner.cls, h_var], dim=1)
        holder = [h_inj]
        hook = runner.model.blocks[0].register_forward_pre_hook(lambda m, a: (holder[0],))
        try:
            out_ = runner.model(runner.x)
        finally:
            hook.remove()
        torch.softmax(out_[0], dim=0)[runner.target].backward()
        gnorm = h_var.grad[0].norm(dim=-1)
        inv = 1.0 / (gnorm + 1e-8)
        irr = (inv / inv.max().clamp(min=1e-8)).detach().reshape(-1)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        la = torch.full((N,), 0.0, device=dev, dtype=dtype)
        m_v = torch.zeros(N, device=dev, dtype=dtype)
        v_v = torch.zeros(N, device=dev, dtype=dtype)
        b1, b2, eps_ = 0.9, 0.999, 1e-8
        recs = {"ins": ([], []), "del": ([], [])}
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur_lr = 0.01 + 0.5 * (0.45 - 0.01) * (1 + _math.cos(_math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            probs = torch.sigmoid(la_req)
            p_ = probs / (probs.sum() + 1e-8)
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            w = (p_ * budget).clamp(max=1.0)
            state = w if step % 2 == 0 else 1.0 - w
            logits = runner.logits_for_masks(state.unsqueeze(0))[0]
            act = torch.softmax(logits, 0)[runner.target]
            comp = logits.detach().clone()
            comp[runner.target] = -torch.inf
            marg = float((logits[runner.target]
                          - torch.logsumexp(comp.topk(20).values, 0)).detach())
            side = "ins" if step % 2 == 0 else "del"
            recs[side][0].append(state.detach().cpu().numpy())
            recs[side][1].append(marg)
            rec_term = (1.0 - rec_of(act)) if side == "ins" else 0.5 * rec_of(act)
            loss = rec_term + 0.05 * (probs * irr).sum() + 0.003 * probs.sum()
            g_ = torch.autograd.grad(loss, la_req)[0].detach()
            t_ = step + 1
            m_v = b1 * m_v + (1 - b1) * g_
            v_v = b2 * v_v + (1 - b2) * g_ * g_
            ad = (m_v / (1 - b1**t_)) / ((v_v / (1 - b2**t_)).sqrt() + eps_)
            cm = (ad * g_ > 0).to(dtype)
            cm = cm * (N / cm.sum().clamp(min=1.0))
            la = la - cur_lr * ad * cm
        return recs

    def kr_fit(M, v, lam_scale=1e-2):
        M = np.asarray(M, np.float64)
        v = np.asarray(v, np.float64)
        Mc = M - M.mean(0)
        vc = v - v.mean()
        lam = lam_scale * max(len(v), 1)
        return np.linalg.solve(Mc.T @ Mc + lam * Kinv, Mc.T @ vc)

    for seeds in ([42], [42, 43]):
        bs = []
        for sd in seeds:
            recs = solve_record(seed=sd)
            b_ins = kr_fit(*recs["ins"])
            b_del = kr_fit(*recs["del"])
            bs.append(zscore(b_ins) + zscore(b_del))
        b = np.mean(bs, axis=0)
        n_states = 32 * len(seeds)
        print(f"   solve-states n={n_states}: {reg(zscore(b))}")

    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
