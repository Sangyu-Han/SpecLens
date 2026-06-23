#!/usr/bin/env python3
"""Shared machinery for the FRI-G cost-reduction session (S/P sweeps).

Builds on research_fri_halfstep (hs) but:
  - per-case cholesky factors are built locally (avoids the id()-keyed
    _chol_cache collision hazard across many runners)
  - nested probe sampler: one 64-state collection, per-side prefix
    subsampling -> all P share states (clean P-curves, 1 batched call)
  - generalized ridge (side/joint fit, rank-target, lam/ktemp knobs)
  - fri_alt_prob replica with hyper overrides + beta-mixed irrelevance +
    optional (state, margin) recording for mixed-design ridge
"""
from __future__ import annotations

import importlib.util as _ilu
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = _ilu.spec_from_file_location("hs", REPO / "scripts/research_fri_halfstep.py")
hs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(hs)
ks, ns, q = hs.ks, hs.ns, hs.q
N, GRID = 196, 14
CSV = REPO / "outputs/class_fri/softplus_failure_scan_n100.csv"
OUTDIR = REPO / "outputs/class_fri/research_frontier"


def dev_cases():
    import pandas as pd

    df = pd.read_csv(CSV)
    cases = list(ns.CASES)
    for idx in (1, 12, 36, 39):
        row = df[df["idx"] == idx].iloc[0]
        cases.append((f"ctrl:{row['desc'].split(',')[0][:9]}", row["path"], int(row["target"])))
    return cases


import os as _os

# cross-model: override the timm model via FRI_MODEL env var (must be a
# 196-patch / 1-CLS / IN1k-head ViT for the existing N=196 pipeline:
# clip [default], deit3_base_patch16_224.fb_in22k_ft_in1k,
# vit_base_patch16_224.augreg2_in21k_ft_in1k, beit*).
_MODEL_ALIASES = {
    "clip": "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
    "deit3": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "augreg": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "beit": "beit_base_patch16_224.in22k_ft_in22k_in1k",
}


def load_model(device):
    import timm

    syms = q.load_patch_repo()
    name = _os.environ.get("FRI_MODEL", "clip")
    name = _MODEL_ALIASES.get(name, name)
    model = timm.create_model(name, pretrained=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, syms


def make_runner(model, syms, path, target, device):
    x, _ = syms["load_image"](Path(path))
    x = x.to(device)
    h_b0, cls_tok, base = syms["get_b0"](model, x)
    if target < 0:
        with torch.no_grad():
            hold = [torch.cat([cls_tok, h_b0], dim=1)]
            hk = model.blocks[0].register_forward_pre_hook(lambda m, a: (hold[0],))
            try:
                target = int(model(x)[0].argmax().item())
            finally:
                hk.remove()
    return q.Block0Runner(model, x, h_b0, cls_tok, base, int(target)), int(target)


def pos_sim(runner):
    with torch.no_grad():
        h = runner.base[0].detach().float()
        hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return hn @ hn.T


def kinv_of(runner, ktemp):
    S = pos_sim(runner).cpu().numpy().astype(np.float64)
    return np.linalg.inv(np.exp((S - 1.0) / ktemp) + 1e-3 * np.eye(N))


def nested_field_states(runner, n_max=64, seed=42, temps=(0.6, 0.25, 0.1)):
    """Dual-sided seeded pos-field probe states; margins in 1 batched pass.
    State i<half: ins side; i>=half: del side (same dists as
    hs.seeded_field_states). Prefix-per-side subsampling keeps nesting."""
    S = pos_sim(runner)
    chols = [torch.linalg.cholesky(
        torch.exp((S - 1.0) / t) + 1e-3 * torch.eye(N, device=S.device)
    ).to(runner.dtype) for t in temps]
    gen = torch.Generator(device=runner.dev)
    gen.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    half = n_max // 2
    masks, sides = [], []
    for i in range(n_max):
        side = "ins" if i < half else "del"
        d = float(np.exp(rng.uniform(np.log(0.05), np.log(0.5)))) if side == "ins" \
            else float(rng.uniform(0.1, 0.5))
        si = int(rng.integers(3))
        z = torch.randn(N, generator=gen, device=runner.dev, dtype=runner.dtype)
        f = (chols[si] @ z).cpu().numpy()
        k = int(np.clip(round(d * N), 1, N - 1))
        m = np.zeros(N, np.float32)
        m[np.argpartition(-f, k)[:k]] = 1.0
        masks.append(m if side == "ins" else 1.0 - m)
        sides.append(side)
    masks = np.stack(masks)
    sides = np.asarray(sides)
    margins = []
    with torch.no_grad():
        for i in range(0, len(masks), 96):
            mt = torch.as_tensor(masks[i:i + 96], device=runner.dev, dtype=runner.dtype)
            logits = runner.logits_for_masks(mt)
            comp = logits.clone()
            comp[:, runner.target] = -torch.inf
            margins.append((logits[:, runner.target]
                            - torch.logsumexp(comp.topk(20, dim=1).values, dim=1)
                            ).cpu().numpy())
    return masks, sides, np.concatenate(margins)


def subsample(masks, sides, margins, P):
    """First P//2 states of each side (prefix-nested)."""
    ins_idx = np.where(sides == "ins")[0][: P // 2]
    del_idx = np.where(sides == "del")[0][: P // 2]
    idx = np.concatenate([ins_idx, del_idx])
    return masks[idx], sides[idx], margins[idx]


def _rank_z(v):
    from scipy.stats import rankdata

    r = rankdata(v)
    return (r - r.mean()) / (r.std() + 1e-12)


def ridge_beta_gen(masks, sides, margins, Kinv, *, mode="side", rank_t=False,
                   lam_scale=1e-3):
    """Generalized signed-beta ridge. mode: side (per-side fit, z-sum) |
    joint (per-side centering, single stacked fit)."""

    def prep(rows):
        M = masks[rows].astype(np.float64)
        v = margins[rows].astype(np.float64)
        if rank_t:
            v = _rank_z(v)
        return M - M.mean(0), v - v.mean()

    if mode == "side":
        betas = []
        for side in ("ins", "del"):
            Mc, vc = prep(sides == side)
            lam = lam_scale * max(len(vc), 1)
            betas.append(ks.zscore(np.linalg.solve(Mc.T @ Mc + lam * Kinv, Mc.T @ vc)))
        return ks.zscore(betas[0] + betas[1])
    Mc_i, vc_i = prep(sides == "ins")
    Mc_d, vc_d = prep(sides == "del")
    Mc = np.vstack([Mc_i, Mc_d])
    vc = np.concatenate([vc_i, vc_d])
    lam = lam_scale * max(len(vc), 1)
    return ks.zscore(np.linalg.solve(Mc.T @ Mc + lam * Kinv, Mc.T @ vc))


def pair_states(runner, P, seed, temps=(0.6, 0.25, 0.1)):
    """P//2 antithetic complement pairs (m, 1-m); log-spaced densities."""
    S = pos_sim(runner)
    chols = [torch.linalg.cholesky(
        torch.exp((S - 1.0) / t) + 1e-3 * torch.eye(N, device=S.device)
    ).to(runner.dtype) for t in temps]
    gen = torch.Generator(device=runner.dev)
    gen.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    npairs = P // 2
    dens = np.exp(np.linspace(np.log(0.12), np.log(0.5), npairs))
    masks = []
    for d in dens:
        si = int(rng.integers(len(chols)))
        z = torch.randn(N, generator=gen, device=runner.dev, dtype=runner.dtype)
        f = (chols[si] @ z).cpu().numpy()
        k = int(np.clip(round(d * N), 1, N - 1))
        m = np.zeros(N, np.float32)
        m[np.argpartition(-f, k)[:k]] = 1.0
        masks.append(m)
        masks.append(1.0 - m)
    masks = np.stack(masks)
    margins = []
    with torch.no_grad():
        for i in range(0, len(masks), 96):
            mt = torch.as_tensor(masks[i:i + 96], device=runner.dev, dtype=runner.dtype)
            logits = runner.logits_for_masks(mt)
            comp = logits.clone()
            comp[:, runner.target] = -torch.inf
            margins.append((logits[:, runner.target]
                            - torch.logsumexp(comp.topk(20, dim=1).values, dim=1)
                            ).cpu().numpy())
    return masks, np.concatenate(margins)


def beta_pair(masks, margins, Kinv, lam_scale=1e-3, rank_t=False):
    """Paired-difference ridge: y = margin(m)-margin(1-m), X = 2m-1."""
    m = masks[0::2].astype(np.float64)
    y = (margins[0::2] - margins[1::2]).astype(np.float64)
    if rank_t:
        y = _rank_z(y)
    X = 2.0 * m - 1.0
    Xc = X - X.mean(0)
    yc = y - y.mean()
    lam = lam_scale * max(len(yc), 1)
    return ks.zscore(np.linalg.solve(Xc.T @ Xc + lam * Kinv, Xc.T @ yc))


def gate_score(fri, beta, floor=0.1, temp=0.5):
    return ns._n01(fri) * (floor + (1.0 - floor) / (1.0 + np.exp(-beta / temp)))


def eval_score(runner, score):
    _, _, pi, pd_ = q.hard_curves(runner, score, chunk=128)
    ci, cd = hs.clip_aucs(pi, pd_)
    return {"hdel": cd, "hins": ci}


def fri_solve(runner, *, steps=32, lr=0.45, lr_end=0.01, tv_weight=0.01,
              irr_weight=0.05, l1_weight=0.003, deletion_weight=0.5,
              init_prob=0.5, seed=42, irr_beta=None, irr_beta_w=0.0,
              record_states=False):
    """q.fri_alt_prob replica. Extras:
      irr_beta/irr_beta_w : irrelevance := (1-w)*inv_gradnorm + w*n01(-beta)
      record_states       : returns visited (soft state, margin, side) lists
    With irr_beta_w=0, record_states=False -> bit-identical to q.fri_alt_prob."""
    dev, dtype = runner.dev, runner.dtype
    with torch.no_grad():
        full_obj = runner.probs_for_masks(torch.ones(1, N, device=dev, dtype=dtype))[0]
        base_obj = runner.probs_for_masks(torch.zeros(1, N, device=dev, dtype=dtype))[0]

    h_var = runner.h_b0.detach().clone().requires_grad_(True)
    h_inj = torch.cat([runner.cls, h_var], dim=1)
    holder = [h_inj]
    hook = runner.model.blocks[0].register_forward_pre_hook(lambda m, a: (holder[0],))
    try:
        out = runner.model(runner.x)
    finally:
        hook.remove()
    torch.softmax(out[0], dim=0)[runner.target].backward()
    gnorm = h_var.grad[0].norm(dim=-1)
    inv = 1.0 / (gnorm + 1e-8)
    irr = (inv / inv.max().clamp(min=1e-8)).detach().reshape(-1)
    if irr_beta is not None and irr_beta_w > 0.0:
        nb = ns._n01(np.maximum(-np.asarray(irr_beta, np.float64), 0.0))
        irr_b = torch.as_tensor(nb, device=dev, dtype=dtype)
        irr = (1.0 - irr_beta_w) * irr + irr_beta_w * irr_b

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    la = torch.full((N,), math.log(init_prob / (1 - init_prob)), device=dev, dtype=dtype)
    m_v = torch.zeros(N, device=dev, dtype=dtype)
    v_v = torch.zeros(N, device=dev, dtype=dtype)
    del_support = torch.zeros(N, device=dev, dtype=dtype)
    rec_states = []

    def _tv(z):
        g = z.view(GRID, GRID)
        return (g[:, :-1] - g[:, 1:]).abs().sum() + (g[:-1, :] - g[1:, :]).abs().sum()

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
        budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
        w = (p * budget).clamp(max=1.0)
        state = w if step % 2 == 0 else 1.0 - w
        logits = runner.logits_for_masks(state.unsqueeze(0))[0]
        act = torch.softmax(logits, dim=0)[runner.target]
        if step % 2 == 0:
            rec_term = 1.0 - _rec(act)
        else:
            rec_term = deletion_weight * _rec(act)
        if record_states:
            with torch.no_grad():
                comp = logits.detach().clone()
                comp[runner.target] = -torch.inf
                marg = float(logits[runner.target].detach()
                             - torch.logsumexp(comp.topk(20).values, 0))
            rec_states.append((state.detach().cpu().numpy(), marg,
                               "ins" if step % 2 == 0 else "del"))
        loss = rec_term + irr_weight * (probs * irr).sum() + l1_weight * probs.sum() \
            + tv_weight * _tv(probs)
        rec_g = torch.autograd.grad(rec_term, la_req, retain_graph=True)[0].detach()
        loss.backward()
        g = la_req.grad.detach()
        if step % 2 == 1:
            del_support += (-rec_g).clamp(min=0.0)
        t = step + 1
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g * g
        adam_dir = (m_v / (1 - beta1 ** t)) / ((v_v / (1 - beta2 ** t)).sqrt() + eps)
        cmask = (adam_dir * g > 0).to(dtype)
        cmask = cmask * (N / cmask.sum().clamp(min=1.0))
        la = la - cur_lr * adam_dir * cmask

    final = torch.sigmoid(la).detach()
    sp = torch.nn.functional.softplus(la).detach()

    def _norm(x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        return x / x.max().clamp(min=1e-8)

    out = {
        "final": final.cpu().numpy(),
        "softplus_x_del": torch.sqrt(_norm(sp) * _norm(del_support)).cpu().numpy(),
        "log_alphas": la.detach().cpu().numpy(),
    }
    if record_states:
        out["states"] = rec_states
    return out


def inflow_score(model, syms, x, target):
    return syms["inflow"](model, x, target_class=target).astype(np.float32)
