#!/usr/bin/env python3
"""Necessary-set ordering via prefix-deletion probes over free channels.

Reframe (from NOTES.md): hdel = SET IDENTIFICATION of the evidence union; the
per-case best ordering varies across channels/readouts (3-seed oracle 0.288 vs
single 0.32-0.37). Instead of one global readout, spend ~32 extra FORWARD
probes on the deletion objective itself:

  v0 select : estimate each candidate ordering's deletion curve at J prefix
              points; pick the per-case argmin (channel selection).
  v1 stitch : greedy block-coalition cover — at each stage, each channel
              proposes its next-b undeleted patches; evaluate p(full \ D∪P);
              take the best block (necessary-set greedy with channel
              proposals; myopia softened by stage-0 backbone from v0).

Channels are all FREE from one HG-FRI solve (existing FRI budget) + generic
token ops (content k-means / content-knn smoothing — modality-general, no 2D
prior, no radius):
  sc   integrated HG-FRI score   fin  solve mask        hgd  grad-field
  bl   n01(fin)+n01(hgd)         ksm  content-knn smoothed sc
  clu  content-cluster group ordering (clusters ranked by top member prior)

References per case: inflow, soft FRI (frontier replica), raw channels.
Metrics: batched hard ins/del AUC (frontier convention p/p_full).
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from src.core.attribution.fri import HGFRIConfig, inverse_grad_irrelevance, run_hgfri

_spec = _ilu.spec_from_file_location("q", REPO / "scripts/qual_hgfri_heatmaps.py")
q = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(q)

N, GRID = 196, 14

CASES = [
    ("zebra/eleph", str(REPO / "multi_object_zebra_elephant.jpg"), 386),
    ("banana", "/media/sangyu/Dataset/imagenet/val/n07753592/ILSVRC2012_val_00032327.JPEG", 954),
    ("lemur", "/media/sangyu/Dataset/imagenet/val/n02497673/ILSVRC2012_val_00011144.JPEG", 383),
    ("eft", "/media/sangyu/Dataset/imagenet/val/n01631663/ILSVRC2012_val_00028601.JPEG", 27),
    ("altar", "/media/sangyu/Dataset/imagenet/val/n02699494/ILSVRC2012_val_00021990.JPEG", 406),
    ("doormat", "/media/sangyu/Dataset/imagenet/val/n03223299/ILSVRC2012_val_00030383.JPEG", 539),
    ("nematode", "/media/sangyu/Dataset/imagenet/val/n01930112/ILSVRC2012_val_00029702.JPEG", 111),
    ("hoopskirt", "/media/sangyu/Dataset/imagenet/val/n03534580/ILSVRC2012_val_00021106.JPEG", 601),
]


def _n01(v):
    v = np.maximum(np.nan_to_num(np.asarray(v, np.float64).reshape(-1)), 0.0)
    return v / (v.max() + 1e-12)


def _kmeans(X, k, seed=0, iters=25):
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    c = Xn[rng.choice(len(Xn), k, replace=False)]
    lab = np.zeros(len(Xn), np.int32)
    for _ in range(iters):
        d = Xn @ c.T
        lab_new = d.argmax(1).astype(np.int32)
        if (lab_new == lab).all():
            break
        lab = lab_new
        for j in range(k):
            m = lab == j
            if m.any():
                cj = Xn[m].mean(0)
                c[j] = cj / (np.linalg.norm(cj) + 1e-12)
    return lab


def content_kernel(h_diff, knn=8, temp=0.3):
    with torch.no_grad():
        h = h_diff[0].detach().float()
        hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = (hn @ hn.T)
        K = torch.exp((S - 1.0) / temp)
        K.fill_diagonal_(0.0)
        vals, idx = torch.topk(K, k=knn, dim=1)
        Ks = torch.zeros_like(K)
        Ks.scatter_(1, idx, vals)
        Ks = 0.5 * (Ks + Ks.T)
        Ks = Ks / Ks.sum(1, keepdim=True).clamp(min=1e-8)
    return Ks.cpu().numpy()


def cluster_order_scores(h_diff, prior, k=10, seed=0):
    """Group ordering: content clusters ranked by evidence strength (mean of
    top-3 member priors); within cluster by prior. No spatial info."""
    X = h_diff[0].detach().float().cpu().numpy()
    lab = _kmeans(X, k, seed=seed)
    p = _n01(prior)
    cl_rank = {}
    for j in range(k):
        m = np.where(lab == j)[0]
        if len(m) == 0:
            cl_rank[j] = -1.0
            continue
        top = np.sort(p[m])[::-1][:3]
        cl_rank[j] = float(top.mean())
    order = sorted(range(N), key=lambda i: (-cl_rank[lab[i]], -p[i]))
    sc = np.zeros(N, np.float64)
    sc[np.asarray(order)] = np.linspace(1.0, 0.0, N, endpoint=False)
    return sc, lab


def dense_ixg(runner, T=8, seed=42, del_lo=0.05, del_hi=0.20):
    """Signed deletion-marginal ensemble at dense perturbed states.

    T fwd+bwd at random ~90% density states; mask-gradient g_i at a state
    where i is present ~ first-order drop if i is deleted. Positive part is
    an evidence-membership channel (notes Exp2: dense-state ixg sign
    separates evidence/competitor/bg). Model-agnostic, no spatial prior.
    """
    dev, dtype = runner.dev, runner.dtype
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    acc = torch.zeros(N, dtype=torch.float64)
    cnt = torch.zeros(N, dtype=torch.float64)
    for t in range(T):
        frac = del_lo + (del_hi - del_lo) * torch.rand(1, generator=gen).item()
        k = max(1, int(round(frac * N)))
        perm = torch.randperm(N, generator=gen)[:k]
        m = torch.ones(N, device=dev, dtype=dtype)
        m[perm.to(dev)] = 0.0
        m_req = m.clone().requires_grad_(True)
        p = runner.probs_for_masks(m_req.unsqueeze(0))[0]
        g = torch.autograd.grad(p, m_req)[0].detach().cpu().double()
        present = m.detach().cpu().double()
        acc += g * present
        cnt += present
    s = (acc / cnt.clamp(min=1.0)).numpy()
    return np.maximum(s, 0.0)


def _front_state_grad(runner, present, signal="margin"):
    """One fwd+bwd at the deletion front. Returns (p_target, grad np[N])."""
    dev, dtype = runner.dev, runner.dtype
    m = torch.zeros(N, device=dev, dtype=dtype)
    m[np.where(present)[0]] = 1.0
    m_req = m.clone().requires_grad_(True)
    logits = runner.logits_for_masks(m_req.unsqueeze(0))[0]
    p = float(torch.softmax(logits.detach(), 0)[runner.target])
    if signal == "margin":
        comp = logits.detach().clone()
        comp[runner.target] = -torch.inf
        obj = logits[runner.target] - torch.logsumexp(comp.topk(20).values, 0)
    else:
        obj = torch.softmax(logits, 0)[runner.target]
    g = torch.autograd.grad(obj, m_req)[0].detach().cpu().numpy().astype(np.float64)
    return p, g


def front_greedy(runner, base_order, F=48, plan=((16, 2), (32, 4)),
                 signal="margin", warm=0):
    """Gradient-front greedy deletion chaining.

    Chains margin-gradient marginals at the moving deletion front:
    re-linearize, delete top-R positive-marginal patches, repeat. `plan` is
    ((upto_k, R), ...) so the early curve gets finer steps. `warm` deletes
    base_order's top-`warm` first (unsaturates the front before chaining).
    Returns ordering scores, the probed running curve [(k, p)], #fallbacks.
    """
    present = np.ones(N, bool)
    seq: list[int] = []
    curve: list[tuple[int, float]] = []
    n_fallback = 0
    for i in [j for j in base_order[:warm]]:
        present[i] = False
        seq.append(int(i))
    done = len(seq)
    while done < F:
        R = next((r for upto, r in plan if done < upto), plan[-1][1])
        p_front, g = _front_state_grad(runner, present, signal)
        if seq:
            curve.append((done, p_front))
        g = np.where(present, g, -np.inf)
        top = [int(i) for i in np.argsort(-g)[:R]
               if np.isfinite(g[i]) and g[i] > 0.0]
        if not top:
            n_fallback += 1
            top = [i for i in base_order if present[i]][:R]
        for i in top:
            present[i] = False
            seq.append(int(i))
        done = len(seq)
    seq.extend([i for i in base_order if present[i]])
    return rank_scores_from_order(np.asarray(seq, int)), curve, n_fallback


def order_of(scores):
    return np.argsort(-np.asarray(scores, np.float64).reshape(-1), kind="mergesort")


def rank_scores_from_order(order):
    sc = np.zeros(N, np.float64)
    sc[np.asarray(order)] = np.linspace(1.0, 0.0, N, endpoint=False)
    return sc.astype(np.float32)


class ProbeCounter:
    def __init__(self, runner, chunk=128):
        self.runner = runner
        self.chunk = chunk
        self.n = 0

    def probs(self, masks_np):
        self.n += len(masks_np)
        return self.runner.prob_curve(np.asarray(masks_np, np.float32), self.chunk)


def margin_probe(runner, masks_np, chunk=128, topk=20):
    """Batched target-vs-topk-competitor margin for masks."""
    outs = []
    t = runner.target
    with torch.no_grad():
        for i in range(0, len(masks_np), chunk):
            m = torch.as_tensor(masks_np[i:i + chunk], device=runner.dev, dtype=runner.dtype)
            logits = runner.logits_for_masks(m)
            comp = logits.clone()
            comp[:, t] = -torch.inf
            mg = logits[:, t] - torch.logsumexp(comp.topk(topk, dim=1).values, dim=1)
            outs.append(mg.cpu().numpy())
    return np.concatenate(outs)


def select_prefix(pc, channels, ks=(8, 20, 40, 72, 120), p_full=None, p_base=None):
    """Per-channel deletion prefix probes -> AUC estimate -> argmin channel."""
    names = list(channels)
    masks, meta = [], []
    for nm in names:
        o = order_of(channels[nm])
        for k in ks:
            m = np.ones(N, np.float32)
            m[o[:k]] = 0.0
            masks.append(m)
            meta.append((nm, k))
    vals = pc.probs(np.stack(masks))
    est = {}
    xs = np.array([0.0, *[k / N for k in ks], 1.0])
    for nm in names:
        ys = [p_full] + [float(v) for (mn, k), v in zip(meta, vals) if mn == nm] + [p_base]
        est[nm] = float(np.trapz(np.array(ys) / max(p_full, 1e-8), xs))
    best = min(est, key=est.get)
    return best, est


def stitch_greedy(pc, channels, blocks=(4, 4, 8, 8, 12, 16, 24, 32, 40, 48),
                  backbone=None, sticky_eps=0.003, sat_spread=0.02, p_full=1.0):
    """Greedy block-coalition cover. Each stage: every channel proposes its
    next-b undeleted patches; evaluate deletion value; take min. `backbone`
    breaks near-ties; under saturation (all proposals flat) fall back to the
    backbone proposal instead of trusting noise. Also returns the stitched
    curve samples (cumulative deleted count, value) for final verification."""
    names = list(channels)
    orders = {nm: order_of(channels[nm]) for nm in names}
    deleted = np.zeros(N, bool)
    seq = []
    prev_pick = backbone
    curve = []
    for b in blocks:
        b = min(b, N - int(deleted.sum()))
        if b <= 0:
            break
        props, masks = [], []
        for nm in names:
            o = orders[nm]
            cand = [i for i in o if not deleted[i]][:b]
            m = np.ones(N, np.float32)
            m[deleted] = 0.0
            m[np.asarray(cand, int)] = 0.0
            props.append((nm, cand))
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        i_best = int(np.argmin(vals))
        spread = float(vals.max() - vals.min())
        if spread < sat_spread * max(p_full, 1e-8) and prev_pick is not None:
            i_best = names.index(prev_pick)  # saturated: probe is noise
        elif prev_pick is not None:
            i_prev = names.index(prev_pick)
            if vals[i_prev] <= vals[i_best] + sticky_eps:
                i_best = i_prev
        nm, cand = props[i_best]
        deleted[np.asarray(cand, int)] = True
        seq.extend(cand)
        curve.append((int(deleted.sum()), float(vals[i_best])))
        prev_pick = nm
    seq.extend([i for i in range(N) if not deleted[i]])
    return rank_scores_from_order(np.asarray(seq, int)), prev_pick, curve


def est_auc_from_curve(curve, p_full, p_base):
    ks = [0] + [k for k, _ in curve] + [N]
    ys = [p_full] + [v for _, v in curve] + [p_base]
    xs = np.asarray(ks, np.float64) / N
    return float(np.trapz(np.asarray(ys) / max(p_full, 1e-8), xs))


def lobo_reorder(pc, base_scores, *, top=64, band=8, p_full=1.0,
                 anti_tol=0.01, blocks=None, ratio=False,
                 interleave=False, interleave_floor=0.3, precomputed=None,
                 within_scores=None):
    """Leave-one-band-in restoration reorder of the backbone top-`top`.

    Deletion saturation makes delete-from-full probes noise; probe the
    UNSATURATED end instead: state_b = (full \\ top) + band_b. The jump
    p(state_b) - p(full \\ top) measures whether the band re-fires the
    target given remaining context. Every re-firing band must precede the
    rest in a deletion order (hitting-set requirement); redundant copies all
    re-fire (banana), anti-evidence bands restore BELOW baseline -> global
    tail (nematode bounce). Cost: #blocks + 1 probes.

    `blocks`: optional explicit partition of the top set (e.g. content
    clusters). `ratio`: order by jump/size (AUC-greedy) instead of raw jump.
    """
    base_order = order_of(base_scores)
    top_idx = list(base_order[:top])
    if blocks is None:
        bands = [top_idx[i:i + band] for i in range(0, len(top_idx), band)]
    else:
        bands = [list(b) for b in blocks if len(b)]
    if within_scores is not None:
        ws = np.asarray(within_scores, np.float64).reshape(-1)
        bands = [sorted(bd, key=lambda i: -ws[i]) for bd in bands]
    if precomputed is not None:
        p0, jumps = precomputed["p0"], list(precomputed["jumps"])
    else:
        m0 = np.ones(N, np.float32)
        m0[np.asarray(top_idx, int)] = 0.0
        masks = [m0]
        for bd in bands:
            m = m0.copy()
            m[np.asarray(bd, int)] = 1.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        p0 = float(vals[0])
        jumps = [float(v) - p0 for v in vals[1:]]
    tol = anti_tol * max(p_full, 1e-8)
    key = (lambda t: -(t[0] / max(len(bands[t[1]]), 1))) if ratio else (lambda t: -t[0])
    keep = [(j, bi) for bi, j in enumerate(jumps) if j >= -tol]
    tail_b = [bi for bi, j in enumerate(jumps) if j < -tol]
    keep.sort(key=key)
    if interleave:
        # Redundant copies are EACH sufficient: completing one copy at a
        # time cannot kill the target early; the optimal early order hits
        # every strong copy's core first. Sainte-Laguë apportionment:
        # deletion slot t goes to band argmax jump/(2*taken+1).
        floor = interleave_floor * max(p_full, 1e-8)
        strong = [(j, bi) for j, bi in keep if j >= floor]
        weak = [(j, bi) for j, bi in keep if j < floor]
        ptr = {bi: 0 for _, bi in strong}
        taken = {bi: 0 for _, bi in strong}
        seq = []
        n_strong = sum(len(bands[bi]) for _, bi in strong)
        jmap = dict((bi, j) for j, bi in strong)
        while len(seq) < n_strong:
            avail = [bi for _, bi in strong if ptr[bi] < len(bands[bi])]
            bi = max(avail, key=lambda b: jmap[b] / (2 * taken[b] + 1))
            seq.append(bands[bi][ptr[bi]])
            ptr[bi] += 1
            taken[bi] += 1
        seq.extend(i for _, bi in weak for i in bands[bi])
    else:
        seq = [i for _, bi in keep for i in bands[bi]]
    mid = [i for i in base_order[top:]]
    tail = [i for bi in tail_b for i in bands[bi]]
    return (rank_scores_from_order(np.asarray(seq + mid + tail, int)),
            {"p0": p0, "jumps": jumps, "n_tail_bands": len(tail_b)})


def lobo_grow(pc, base_scores, *, top=64, band=4, p_full=1.0,
              add_tol_frac=0.15, probe_m=3, dead_cap_frac=0.5):
    """Dead-set growth: build the maximal NON-firing retained set greedily.

    Deletion AUC is minimized when, walking the order backwards, the
    retained set stays dead as long as possible. Static LOBO jumps can't see
    pair-sufficiency (weak bands that fire only together). Growth fixes it:
    start from context-only (top deleted), repeatedly re-probe the weakest
    few candidate bands CONDITIONED on the current retained set and add the
    one with the smallest conditional jump; stop when everything left
    re-fires. Never-added bands = hitting set = deletion front (ordered by
    conditional jump desc). Anti-evidence (negative jump) joins the dead set
    first -> deleted last. Cost: (top/band + 1) static + ~probe_m per round.
    """
    base_order = order_of(base_scores)
    top_idx = list(base_order[:top])
    bands = [top_idx[i:i + band] for i in range(0, len(top_idx), band)]
    nb = len(bands)
    m0 = np.ones(N, np.float32)
    m0[np.asarray(top_idx, int)] = 0.0
    masks = [m0] + []
    for bd in bands:
        m = m0.copy()
        m[np.asarray(bd, int)] = 1.0
        masks.append(m)
    vals = pc.probs(np.stack(masks))
    p0 = float(vals[0])
    static = {b: float(vals[1 + b]) - p0 for b in range(nb)}
    known = dict(static)
    S = m0.copy()
    p_S = p0
    add_tol = add_tol_frac * max(p_full, 1e-8)
    dead_cap = dead_cap_frac * max(p_full, 1e-8)
    # competitor-immune addability: the STATIC jump (measured before any
    # competitor mass returns) must also be small, else renormalization
    # suppression can eat real evidence into the dead set (zebra failure).
    cand = sorted([b for b in range(nb) if static[b] <= add_tol],
                  key=lambda b: known[b])
    fixed_front = [b for b in range(nb) if static[b] > add_tol]
    dead_seq: list[int] = []
    rounds = 0
    while cand and rounds < nb:
        rounds += 1
        probe_bs = sorted(cand, key=lambda b: known[b])[:probe_m]
        masks = []
        for b in probe_bs:
            m = S.copy()
            m[np.asarray(bands[b], int)] = 1.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        for b, v in zip(probe_bs, vals):
            known[b] = float(v) - p_S
        b_min = min(probe_bs, key=lambda b: known[b])
        if known[b_min] > add_tol or p_S + known[b_min] > dead_cap:
            break
        S[np.asarray(bands[b_min], int)] = 1.0
        p_S = p_S + known[b_min]
        dead_seq.append(b_min)
        cand.remove(b_min)
    front = sorted(fixed_front + cand, key=lambda b: -known[b])
    seq = [i for b in front for i in bands[b]]
    mid = [i for i in base_order[top:]]
    tail = [i for b in reversed(dead_seq) for i in bands[b]]
    return (rank_scores_from_order(np.asarray(seq + mid + tail, int)),
            {"p0": p0, "n_dead": len(dead_seq), "n_front": len(front),
             "p_S": p_S, "jumps": [known[b] for b in range(nb)]})


def knee_field(runner, base_scores, ks=(8, 16, 24, 32, 48, 64), signal="prob"):
    """Gradient fields along the backbone deletion path (the curve knee).

    The full-state gradient is saturation-blind and the single-restoration
    jump is too coarse; but at mid-density states on the metric's own path
    every surviving copy is partially pivotal. One fwd+bwd per state gives
    ALL patch marginals (present: drop-if-deleted; absent: gain-if-
    restored; both signed the same way). Per-state max-normalized, averaged.
    Cost: len(ks) fwd+bwd. Black-box, no spatial prior.
    """
    base_order = order_of(base_scores)
    dev, dtype = runner.dev, runner.dtype
    acc = np.zeros(N, np.float64)
    curve = []
    for k in ks:
        present = np.ones(N, bool)
        present[base_order[:k]] = False
        m = torch.zeros(N, device=dev, dtype=dtype)
        m[np.where(present)[0]] = 1.0
        m_req = m.clone().requires_grad_(True)
        logits = runner.logits_for_masks(m_req.unsqueeze(0))[0]
        if signal == "margin":
            comp = logits.detach().clone()
            comp[runner.target] = -torch.inf
            obj = logits[runner.target] - torch.logsumexp(comp.topk(20).values, 0)
        else:
            obj = torch.softmax(logits, 0)[runner.target]
        curve.append((int(k), float(torch.softmax(logits.detach(), 0)[runner.target])))
        g = torch.autograd.grad(obj, m_req)[0].detach().cpu().numpy().astype(np.float64)
        mx = np.abs(g).max()
        if mx > 1e-12:
            acc += g / mx
    return acc / max(len(ks), 1), curve


def lobo_grad_reorder(runner, base_scores, blocks, *, p_full=1.0,
                      anti_tol=0.01, interleave=True, interleave_floor=0.3,
                      signal="prob"):
    """LOBO over content blocks with WITHIN-block restoration gradients.

    Each block's restoration probe (context + block alone) is made fwd+bwd:
    the backward at that minimal single-copy state gives unsaturated
    within-block criticality (no cross-copy suppression, no competitor
    renormalization). Cross-block order: restoration jump (interleaved via
    Sainte-Laguë or sequential); within-block order: own-state gradient.
    Cost: #blocks+1 fwd, #blocks bwd.
    """
    dev, dtype = runner.dev, runner.dtype
    base_order = order_of(base_scores)
    top_idx = [i for b in blocks for i in b]
    m0 = np.ones(N, np.float32)
    m0[np.asarray(top_idx, int)] = 0.0
    with torch.no_grad():
        p0 = float(runner.probs_for_masks(
            torch.as_tensor(m0, device=dev, dtype=dtype).unsqueeze(0))[0])
    jumps, wgrads = [], []
    for bd in blocks:
        m = m0.copy()
        m[np.asarray(bd, int)] = 1.0
        m_req = torch.as_tensor(m, device=dev, dtype=dtype).clone().requires_grad_(True)
        logits = runner.logits_for_masks(m_req.unsqueeze(0))[0]
        if signal == "margin":
            comp = logits.detach().clone()
            comp[runner.target] = -torch.inf
            obj = logits[runner.target] - torch.logsumexp(comp.topk(20).values, 0)
        else:
            obj = torch.softmax(logits, 0)[runner.target]
        p_b = float(torch.softmax(logits.detach(), 0)[runner.target])
        g = torch.autograd.grad(obj, m_req)[0].detach().cpu().numpy().astype(np.float64)
        jumps.append(p_b - p0)
        wgrads.append(g)
    tol = anti_tol * max(p_full, 1e-8)
    rank_pos = {int(p): r for r, p in enumerate(base_order)}
    ordered_blocks = []
    for bi, bd in enumerate(blocks):
        g = wgrads[bi]
        members = sorted(bd, key=lambda i: (-g[i], rank_pos[i]))
        ordered_blocks.append(members)
    keep = [(j, bi) for bi, j in enumerate(jumps) if j >= -tol]
    tail_b = [bi for bi, j in enumerate(jumps) if j < -tol]
    keep.sort(key=lambda t: -(t[0] / max(len(blocks[t[1]]), 1)))
    if interleave:
        floor = interleave_floor * max(p_full, 1e-8)
        strong = [(j, bi) for j, bi in keep if j >= floor]
        weak = [(j, bi) for j, bi in keep if j < floor]
        ptr = {bi: 0 for _, bi in strong}
        taken = {bi: 0 for _, bi in strong}
        jmap = dict((bi, j) for j, bi in strong)
        seq = []
        n_strong = sum(len(ordered_blocks[bi]) for _, bi in strong)
        while len(seq) < n_strong:
            avail = [bi for _, bi in strong if ptr[bi] < len(ordered_blocks[bi])]
            bi = max(avail, key=lambda b: jmap[b] / (2 * taken[b] + 1))
            seq.append(ordered_blocks[bi][ptr[bi]])
            ptr[bi] += 1
            taken[bi] += 1
        seq.extend(i for _, bi in weak for i in ordered_blocks[bi])
    else:
        seq = [i for _, bi in keep for i in ordered_blocks[bi]]
    mid = [i for i in base_order if i not in set(top_idx)]
    tail = [i for bi in tail_b for i in ordered_blocks[bi]]
    return (rank_scores_from_order(np.asarray(seq + mid + tail, int)),
            {"p0": p0, "jumps": jumps, "n_tail_bands": len(tail_b)})


def cluster_completion_reorder(base_scores, h_diff, cblocks, lobo_diag, *,
                               p_full=1.0, interleave=True,
                               interleave_floor=0.3, anti_tol=0.01,
                               sim_tau=0.55, cap_per=8):
    """Cluster completion: strong-jump clusters pull in content-similar
    patches from OUTSIDE the top set (kill-set members stranded at rank
    64-96). Zero extra probes — reuses LOBO jumps + content space."""
    base_order = order_of(base_scores)
    top_set = set(int(i) for b in cblocks for i in b)
    X = h_diff[0].detach().float().cpu().numpy()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    jumps = list(lobo_diag["jumps"])
    tol = anti_tol * max(p_full, 1e-8)
    floor = interleave_floor * max(p_full, 1e-8)
    rank_pos = {int(p): r for r, p in enumerate(base_order)}
    used = set(top_set)
    ext_blocks = []
    for bi, bd in enumerate(cblocks):
        members = sorted(bd, key=lambda i: rank_pos[i])
        if jumps[bi] >= floor:
            mu = Xn[np.asarray(bd, int)].mean(0)
            mu = mu / (np.linalg.norm(mu) + 1e-12)
            sims = Xn @ mu
            cand = [(float(sims[i]), int(i)) for i in range(N)
                    if i not in used and sims[i] > sim_tau]
            cand.sort(key=lambda t: -t[0])
            extra = [i for _, i in cand[:cap_per]]
            used.update(extra)
            members = members + extra
        ext_blocks.append(members)
    keep = [(j, bi) for bi, j in enumerate(jumps) if j >= -tol]
    tail_b = [bi for bi, j in enumerate(jumps) if j < -tol]
    keep.sort(key=lambda t: -(t[0] / max(len(cblocks[t[1]]), 1)))
    if interleave:
        strong = [(j, bi) for j, bi in keep if j >= floor]
        weak = [(j, bi) for j, bi in keep if j < floor]
        ptr = {bi: 0 for _, bi in strong}
        taken = {bi: 0 for _, bi in strong}
        jmap = dict((bi, j) for j, bi in strong)
        seq = []
        n_strong = sum(len(ext_blocks[bi]) for _, bi in strong)
        while len(seq) < n_strong:
            avail = [bi for _, bi in strong if ptr[bi] < len(ext_blocks[bi])]
            bi = max(avail, key=lambda b: jmap[b] / (2 * taken[b] + 1))
            seq.append(ext_blocks[bi][ptr[bi]])
            ptr[bi] += 1
            taken[bi] += 1
        seq.extend(i for _, bi in weak for i in ext_blocks[bi])
    else:
        seq = [i for _, bi in keep for i in ext_blocks[bi]]
    seq_set = set(seq)
    tail = [i for bi in tail_b for i in ext_blocks[bi] if i not in seq_set]
    mid = [i for i in base_order if i not in seq_set and i not in set(tail)]
    return rank_scores_from_order(np.asarray(seq + mid + tail, int))


def lobo_singleton(pc, base_scores, *, top=64, p_full=1.0, anti_tol=0.002,
                   within_only=False, blocks=None, interleave_floor=0.3,
                   block_jumps=None, precomputed=None, quant_frac=2e-4):
    """Patch-level restoration at the dead end (LOBO band=1).

    Restore each top-`top` patch ALONE into the dead context (top deleted).
    Absolute jumps are tiny but their ranking measures per-patch
    evidence-with-context, free of multi-copy saturation — the output-level
    analogue of white-box localization. Cost: top+1 probes.

    within_only: keep cluster `blocks` order (by `block_jumps`, interleaved)
    and use singleton jumps only as the within-block sorter.
    """
    base_order = order_of(base_scores)
    top_idx = list(base_order[:top])
    if precomputed is not None:
        p0, jump = precomputed
    else:
        m0 = np.ones(N, np.float32)
        m0[np.asarray(top_idx, int)] = 0.0
        masks = [m0]
        for i in top_idx:
            m = m0.copy()
            m[int(i)] = 1.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        p0 = float(vals[0])
        jump = {int(i): float(v) - p0 for i, v in zip(top_idx, vals[1:])}
    tol = anti_tol * max(p_full, 1e-8)
    if within_only and blocks is not None:
        floor = interleave_floor * max(p_full, 1e-8)
        ordered_blocks = [sorted(bd, key=lambda i: -jump.get(int(i), 0.0))
                          for bd in blocks]
        keep = [(j, bi) for bi, j in enumerate(block_jumps) if j >= -0.01 * p_full]
        tail_b = [bi for bi, j in enumerate(block_jumps) if j < -0.01 * p_full]
        keep.sort(key=lambda t: -(t[0] / max(len(blocks[t[1]]), 1)))
        strong = [(j, bi) for j, bi in keep if j >= floor]
        weak = [(j, bi) for j, bi in keep if j < floor]
        ptr = {bi: 0 for _, bi in strong}
        taken = {bi: 0 for _, bi in strong}
        jmap = dict((bi, j) for j, bi in strong)
        seq = []
        n_strong = sum(len(ordered_blocks[bi]) for _, bi in strong)
        while len(seq) < n_strong:
            avail = [bi for _, bi in strong if ptr[bi] < len(ordered_blocks[bi])]
            bi = max(avail, key=lambda b: jmap[b] / (2 * taken[b] + 1))
            seq.append(ordered_blocks[bi][ptr[bi]])
            ptr[bi] += 1
            taken[bi] += 1
        seq.extend(i for _, bi in weak for i in ordered_blocks[bi])
        tail = [i for bi in tail_b for i in ordered_blocks[bi]]
    else:
        # quantize jumps to the probe noise floor; ties fall back to the
        # backbone rank (stabilizes the zero-jump tail ordering)
        res = max(quant_frac * max(p_full, 1e-8), 1e-9)
        rank_pos = {int(p): r for r, p in enumerate(base_order)}

        def keyf(i):
            return (-round(jump[int(i)] / res), rank_pos[int(i)])

        pos = [i for i in top_idx if jump[int(i)] >= -tol]
        tail = [i for i in top_idx if jump[int(i)] < -tol]
        seq = sorted(pos, key=keyf)
        tail = sorted(tail, key=keyf)
    mid = [i for i in base_order[top:]]
    return (rank_scores_from_order(np.asarray(list(seq) + mid + list(tail), int)),
            {"p0": p0, "jumps": [jump[int(i)] for i in top_idx],
             "jump_map": {int(k): float(v) for k, v in jump.items()},
             "n_tail_bands": len(tail)})


def content_blocks_of_top(base_scores, h_diff, *, top=64, kclust=10):
    """Partition the backbone top-`top` into content k-means blocks
    (semantic copies; modality-general, no spatial info)."""
    base_order = order_of(base_scores)
    top_idx = base_order[:top]
    X = h_diff[0].detach().float().cpu().numpy()[top_idx]
    lab = _kmeans(X, min(kclust, len(top_idx)), seed=0)
    rank_pos = {int(p): r for r, p in enumerate(base_order)}
    blocks = []
    for j in range(lab.max() + 1):
        members = [int(top_idx[i]) for i in range(len(top_idx)) if lab[i] == j]
        if members:
            members.sort(key=lambda i: rank_pos[i])
            blocks.append(members)
    return blocks


def conditional_block_reorder(pc, runner, base_scores, h_diff, *, top=64,
                              kclust=10, p_full=1.0, anti_tol=0.004,
                              probe_topm=None):
    """Conditional cluster-block reorder of the backbone's top-`top`.

    Diagnosis: the necessary set lives inside the backbone top-64 but is
    mis-ordered (redundant copies interleaved with context/anti-evidence).
    Partition top-`top` into content clusters (k-means on h_diff, no spatial
    info), then greedily order blocks by CONDITIONAL deletion value: each
    round probes remaining blocks given everything deleted so far and takes
    the best drop-per-patch. Blocks whose conditional deletion RAISES the
    objective (anti-evidence) are demoted to the global tail.
    """
    base_order = order_of(base_scores)
    top_idx = base_order[:top]
    X = h_diff[0].detach().float().cpu().numpy()[top_idx]
    lab = _kmeans(X, min(kclust, len(top_idx)), seed=0)
    blocks = []
    rank_pos = {int(p): r for r, p in enumerate(base_order)}
    for j in range(lab.max() + 1):
        members = [int(top_idx[i]) for i in range(len(top_idx)) if lab[i] == j]
        if members:
            members.sort(key=lambda i: rank_pos[i])
            blocks.append(members)

    deleted = np.zeros(N, bool)
    seq: list[int] = []
    tail_blocks: list[list[int]] = []
    remaining = list(range(len(blocks)))
    p_cur = p_full
    while remaining:
        masks = []
        for b in remaining:
            m = np.ones(N, np.float32)
            m[deleted] = 0.0
            m[np.asarray(blocks[b], int)] = 0.0
            masks.append(m)
        vals = pc.probs(np.stack(masks))
        drops = {b: (p_cur - float(v)) for b, v in zip(remaining, vals)}
        vmap = {b: float(v) for b, v in zip(remaining, vals)}
        ratio = {b: drops[b] / max(len(blocks[b]), 1) for b in remaining}
        b_best = max(remaining, key=lambda b: ratio[b])
        if drops[b_best] < -anti_tol * max(p_full, 1e-8):
            # every remaining block is anti-evidence -> all to tail
            tail_blocks.extend(blocks[b] for b in remaining)
            break
        seq.extend(blocks[b_best])
        deleted[np.asarray(blocks[b_best], int)] = True
        p_cur = vmap[b_best]
        remaining.remove(b_best)
        # demote blocks that look anti-evidence under current conditioning
        for b in list(remaining):
            if drops[b] < -anti_tol * max(p_full, 1e-8):
                tail_blocks.append(blocks[b])
                remaining.remove(b)
    mid = [i for i in base_order if not deleted[i]
           and not any(i in tb for tb in tail_blocks)]
    tail = [i for tb in tail_blocks for i in tb]
    full_seq = seq + mid + tail
    return rank_scores_from_order(np.asarray(full_seq, int)), len(tail_blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/necset_stitch_diag8.json")
    ap.add_argument("--skip-soft", action="store_true")
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    rows = {}
    agg = {}
    t0 = time.time()
    for name, path, target in CASES:
        t_case = time.time()
        x, _ = syms["load_image"](Path(path))
        x = x.to(args.device)
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)

        def ofp(h_var, ct=cls_tok, xx=x, t=target):
            model.zero_grad(set_to_none=True)
            inj = [torch.cat([ct, h_var], 1)]
            hk = model.blocks[0].register_forward_pre_hook(lambda m, a: (inj[0],))
            try:
                out = model(xx)
            finally:
                hk.remove()
            return torch.softmax(out[0], 0)[t]

        irr = inverse_grad_irrelevance(input_patches=h_b0, objective_from_patches=ofp)
        out = run_hgfri(q._Oracle(runner), irrelevance=irr,
                        config=HGFRIConfig(seed=int(args.seed)))

        sc, fin, hgd = out["scores"], out["final"], out["hgd"]
        K = content_kernel(runner.h_diff)
        ksm = 0.5 * _n01(sc) + 0.5 * (K @ _n01(sc))
        bl = _n01(fin) + _n01(hgd)
        prior = _n01(sc) + 0.5 * _n01(hgd)
        clu, lab = cluster_order_scores(runner.h_diff, prior, k=10, seed=0)
        gnorm = _n01(1.0 / (irr.detach().cpu().numpy().astype(np.float64) + 1e-6))
        fg, _, fg_fallbacks = front_greedy(runner, order_of(sc), F=48)

        channels = {"sc": sc, "fin": fin, "hgd": hgd, "bl": bl, "ksm": ksm,
                    "clu": clu, "gnorm": gnorm, "fg": fg}

        with torch.no_grad():
            p_full = float(runner.probs_for_masks(
                torch.ones(1, N, device=runner.dev, dtype=runner.dtype))[0])
            p_base = float(runner.probs_for_masks(
                torch.zeros(1, N, device=runner.dev, dtype=runner.dtype))[0])

        ks_sel = (2, 4, 8, 12, 20, 32, 48, 72, 120)
        pc_sel = ProbeCounter(runner)
        best, est = select_prefix(pc_sel, channels, ks=ks_sel,
                                  p_full=p_full, p_base=p_base)
        sel_scores = channels[best]

        pc_st = ProbeCounter(runner)
        st_scores, last_pick, st_curve = stitch_greedy(
            pc_st, channels, backbone=best, p_full=p_full)

        # guided sequential cover: front-greedy chain warm-started on backbone
        gsc_scores, gsc_curve, gsc_fb = front_greedy(
            runner, order_of(sel_scores), F=48, warm=4)

        # final verification on the SAME ks grid (avoids curve-grid bias)
        pc_ver = ProbeCounter(runner)
        _, ver_est = select_prefix(
            pc_ver, {"st": st_scores, "gsc": gsc_scores}, ks=ks_sel,
            p_full=p_full, p_base=p_base)
        cands = {"select": (est[best], sel_scores),
                 "stitch": (ver_est["st"], st_scores),
                 "gsc": (ver_est["gsc"], gsc_scores)}
        ver_pick = min(cands, key=lambda k: cands[k][0])
        ver_scores = cands[ver_pick][1]
        st_est = ver_est["st"]

        # LOBO restoration reorders of the verify winner's top set
        pc_cbr = ProbeCounter(runner)
        lobo_variants = {}
        lobo_diags = {}
        v, dg = lobo_reorder(pc_cbr, ver_scores, top=64, band=4, p_full=p_full)
        lobo_variants["lobo4"], lobo_diags["lobo4"] = v, dg
        v, dgi = lobo_reorder(pc_cbr, ver_scores, top=64, band=4, p_full=p_full,
                              interleave=True, precomputed=dg)
        lobo_variants["lobo4i"], lobo_diags["lobo4i"] = v, dgi
        cblocks = content_blocks_of_top(ver_scores, runner.h_diff, top=64, kclust=10)
        v, dg = lobo_reorder(pc_cbr, ver_scores, top=64, blocks=cblocks,
                             ratio=True, p_full=p_full)
        lobo_variants["loboC"], lobo_diags["loboC"] = v, dg
        v, dgi = lobo_reorder(pc_cbr, ver_scores, top=64, blocks=cblocks,
                              ratio=True, p_full=p_full, interleave=True,
                              precomputed=dg)
        lobo_variants["loboCi"], lobo_diags["loboCi"] = v, dgi
        v, dg = lobo_reorder(pc_cbr, ver_scores, top=96, band=6, p_full=p_full)
        lobo_variants["lobo96"], lobo_diags["lobo96"] = v, dg
        v, dgi = lobo_reorder(pc_cbr, ver_scores, top=96, band=6, p_full=p_full,
                              interleave=True, precomputed=dg)
        lobo_variants["lobo96i"], lobo_diags["lobo96i"] = v, dgi
        dgC = lobo_diags["loboC"]
        v, dg1 = lobo_singleton(pc_cbr, ver_scores, top=64, p_full=p_full)
        lobo_variants["lobo1"], lobo_diags["lobo1"] = v, dg1
        v, _ = lobo_singleton(pc_cbr, ver_scores, top=64, p_full=p_full,
                              within_only=True, blocks=cblocks,
                              block_jumps=dgC["jumps"],
                              precomputed=(dg1["p0"], dg1["jump_map"]))
        lobo_variants["lobo1Ci"], lobo_diags["lobo1Ci"] = v, dgC
        v, dg96 = lobo_singleton(pc_cbr, ver_scores, top=96, p_full=p_full)
        lobo_variants["lobo1_96"], lobo_diags["lobo1_96"] = v, dg96
        # lobo1 on the select backbone (different top-64 membership)
        v, dgS = lobo_singleton(pc_cbr, sel_scores, top=64, p_full=p_full)
        lobo_variants["lobo1S"], lobo_diags["lobo1S"] = v, dgS
        # mid-segment singleton scan (ranks 64-96) on top of lobo1's order:
        # reorders only the mid band, keeps lobo1's first-64 intact
        lobo1_scores = lobo_variants["lobo1"]
        o1 = order_of(lobo1_scores)
        mid_idx = list(o1[64:96])
        m0m = np.ones(N, np.float32)
        m0m[np.asarray(o1[:96], int)] = 0.0
        masks_m = [m0m]
        for i in mid_idx:
            m = m0m.copy()
            m[int(i)] = 1.0
            masks_m.append(m)
        vals_m = pc_cbr.probs(np.stack(masks_m))
        p0m = float(vals_m[0])
        jm = {int(i): float(v) - p0m for i, v in zip(mid_idx, vals_m[1:])}
        mid_sorted = sorted(mid_idx, key=lambda i: -jm[int(i)])
        seq_m = list(o1[:64]) + mid_sorted + list(o1[96:])
        lobo_variants["lobo1m96"] = rank_scores_from_order(np.asarray(seq_m, int))
        lobo_diags["lobo1m96"] = {"p0": p0m, "jumps": [jm[int(i)] for i in mid_idx],
                                  "n_tail_bands": 0}
        # A-rescan: when dead(64) was partially alive, re-rank lobo1's first
        # 64 (set fixed) by singleton jumps at the deeper dead(96) baseline
        if dg1["p0"] > 0.05 * p_full:
            a_idx = list(o1[:64])
            masks_a = []
            for i in a_idx:
                m = m0m.copy()
                m[int(i)] = 1.0
                masks_a.append(m)
            vals_a = pc_cbr.probs(np.stack(masks_a))
            ja = {int(i): float(v) - p0m for i, v in zip(a_idx, vals_a)}
            res_q = max(2e-4 * p_full, 1e-9)
            rkp = {int(pch): r for r, pch in enumerate(o1)}
            a_sorted = sorted(a_idx, key=lambda i: (-round(ja[int(i)] / res_q), rkp[int(i)]))
            seq_a = a_sorted + mid_sorted + list(o1[96:])
            lobo_variants["lobo1A96"] = rank_scores_from_order(np.asarray(seq_a, int))
            lobo_diags["lobo1A96"] = {"p0": p0m, "jumps": [ja[int(i)] for i in a_idx],
                                      "n_tail_bands": 0}

        # margin-objective singleton variant (alive-baseline cases)
        o_w = order_of(ver_scores)
        top_idx_m = list(o_w[:64])
        m0w = np.ones(N, np.float32)
        m0w[np.asarray(top_idx_m, int)] = 0.0
        masks_w = [m0w]
        for i in top_idx_m:
            m = m0w.copy()
            m[int(i)] = 1.0
            masks_w.append(m)
        mvals = margin_probe(runner, np.stack(masks_w))
        pc_cbr.n += len(masks_w)
        mg0 = float(mvals[0])
        jmg = {int(i): float(v) - mg0 for i, v in zip(top_idx_m, mvals[1:])}
        rkw = {int(pch): r for r, pch in enumerate(o_w)}
        mres = max(1e-3, 1e-3 * abs(mg0))
        pos_m = sorted([i for i in top_idx_m if jmg[int(i)] >= -mres],
                       key=lambda i: (-round(jmg[int(i)] / mres), rkw[int(i)]))
        tail_m = sorted([i for i in top_idx_m if jmg[int(i)] < -mres],
                        key=lambda i: (-round(jmg[int(i)] / mres), rkw[int(i)]))
        seq_mg = pos_m + [i for i in o_w[64:]] + tail_m
        lobo_variants["lobo1M"] = rank_scores_from_order(np.asarray(seq_mg, int))
        lobo_diags["lobo1M"] = {"p0": mg0, "jumps": [jmg[int(i)] for i in top_idx_m],
                                "n_tail_bands": len(tail_m)}

        _, lobo_est = select_prefix(pc_cbr, lobo_variants, ks=ks_sel,
                                    p_full=p_full, p_base=p_base)
        lobo_best = min(lobo_est, key=lobo_est.get)
        cbr_scores = lobo_variants[lobo_best]
        cbr_tails = lobo_diags[lobo_best].get(
            "n_tail_bands", lobo_diags[lobo_best].get("n_dead", 0))
        if lobo_est[lobo_best] <= cands[ver_pick][0]:
            final_scores, final_pick = cbr_scores, f"{lobo_best}({ver_pick})"
        else:
            final_scores, final_pick = ver_scores, ver_pick

        infl = syms["inflow"](model, x, target_class=target).astype(np.float32)
        evals = {"inflow": infl, **channels, "select": sel_scores,
                 "stitch": st_scores, "gsc": gsc_scores, "verify": ver_scores,
                 **lobo_variants, "final": final_scores}
        if not args.skip_soft:
            soft = q.fri_alt_prob(runner, seed=int(args.seed))["softplus_x_del"]
            evals["soft"] = soft

        res = {}
        for nm, s in evals.items():
            hins, hdel, _, _ = q.hard_curves(runner, s, chunk=128)
            res[nm] = {"hins": hins, "hdel": hdel}
            agg.setdefault(nm, []).append((hdel, hins))

        rows[name] = {
            "target": target,
            "p_full": p_full,
            "selected": best,
            "select_est": est,
            "stitch_last_pick": last_pick,
            "verify_pick": ver_pick,
            "stitch_est": st_est,
            "probes_select": pc_sel.n,
            "probes_stitch": pc_st.n,
            "metrics": res,
        }
        rows[name]["final_pick"] = final_pick
        rows[name]["cbr_tails"] = cbr_tails
        rows[name]["lobo_diags"] = {k: {kk: vv for kk, vv in d.items()}
                                    for k, d in lobo_diags.items()}
        line = f"[{name:<11}]"
        for nm in ("inflow", "verify", "lobo1", "lobo1Ci", "lobo1S", "lobo1m96", "lobo1M", "final"):
            if nm in res:
                line += f" {nm}={res[nm]['hdel']:.3f}"
        line += (f" | sel={best} ver={ver_pick} fin={final_pick}"
                 f" pr={pc_sel.n}+{pc_st.n}+{pc_ver.n}+{pc_cbr.n} ({time.time()-t_case:.0f}s)")
        print(line, flush=True)

    print(f"\n=== means (hdel / hins), n={len(rows)} ===")
    summary = {}
    for nm, v in agg.items():
        hd = float(np.mean([a for a, _ in v]))
        hi = float(np.mean([b for _, b in v]))
        summary[nm] = {"hdel": hd, "hins": hi}
        print(f"  {nm:<8} {hd:.4f} / {hi:.4f}")
    infl_d = np.array([rows[n]["metrics"]["inflow"]["hdel"] for n in rows])
    all_nms = sorted({m for n in rows for m in rows[n]["metrics"]})
    for nm in all_nms:
        sub = [n for n in rows if nm in rows[n]["metrics"]]
        if nm == "inflow" or len(sub) < len(rows):
            continue
        d = np.array([rows[n]["metrics"][nm]["hdel"] for n in sub])
        idl = np.array([rows[n]["metrics"]["inflow"]["hdel"] for n in sub])
        loss_cases = [n for n in sub if rows[n]["metrics"][nm]["hdel"] >= rows[n]["metrics"]["inflow"]["hdel"]]
        print(f"  win_del {nm} vs inflow: {(d < idl).sum()}/{len(d)}  losses: {loss_cases}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()


def synergy_segments(pc, runner, base_scores, *, top=40, p_full=1.0,
                     pairs_per=3, knn_frac=0.5, seed=0):
    """Functional segments from pair restoration at the dead end.

    syn(i,j) = J({i,j}) - J(i) - J(j): >0 complements (same evidence
    module), <0 substitutes (redundant copies). Positive-synergy
    agglomeration; whole-segment restoration probes give segment mass even
    when singletons are saturated. Modality-general (forward probes only).
    """
    base_order = order_of(base_scores)
    top_idx = [int(i) for i in base_order[:top]]
    m0 = np.ones(N, np.float32)
    m0[np.asarray(top_idx, int)] = 0.0
    masks = [m0]
    for i in top_idx:
        m = m0.copy()
        m[i] = 1.0
        masks.append(m)
    vals = pc.probs(np.stack(masks))
    p0 = float(vals[0])
    J = {i: float(v) - p0 for i, v in zip(top_idx, vals[1:])}
    X = runner.h_diff[0].detach().float().cpu().numpy()[np.asarray(top_idx, int)]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    jump_rank = sorted(range(len(top_idx)), key=lambda a: -J[top_idx[a]])
    pair_set = set()
    for a in range(len(top_idx)):
        sims = np.argsort(-S[a])
        n_knn = max(1, int(round(pairs_per * knn_frac)))
        cands = [b for b in sims[1:1 + n_knn]]
        n_top = pairs_per - n_knn
        cands += [b for b in jump_rank[:n_top + 2] if b != a][:n_top]
        for b in cands:
            pair_set.add((min(a, b), max(a, b)))
    pairs = sorted(pair_set)
    masks = []
    for a, b in pairs:
        m = m0.copy()
        m[top_idx[a]] = 1.0
        m[top_idx[b]] = 1.0
        masks.append(m)
    vals = pc.probs(np.stack(masks))
    syn = {}
    for (a, b), v in zip(pairs, vals):
        syn[(a, b)] = (float(v) - p0) - J[top_idx[a]] - J[top_idx[b]]
    parent = list(range(len(top_idx)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    syn_scale = max(1e-3 * p_full, 1e-6)
    edges = sorted(((s_, a, b) for (a, b), s_ in syn.items() if s_ > syn_scale),
                   key=lambda t: -t[0])
    for s_, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for a in range(len(top_idx)):
        groups.setdefault(find(a), []).append(top_idx[a])
    segments = sorted(groups.values(), key=lambda g: -max(J[i] for i in g))
    masks = []
    for seg in segments:
        m = m0.copy()
        m[np.asarray(seg, int)] = 1.0
        masks.append(m)
    vals = pc.probs(np.stack(masks))
    seg_J = [float(v) - p0 for v in vals]
    return segments, J, seg_J, {
        "p0": p0, "n_pairs": len(pairs), "n_segments": len(segments),
        "seg_sizes": [len(s_) for s_ in segments]}


def segment_order_scores(segments, J, base_scores, *, p_full=1.0,
                         interleave_floor=0.25, anti_tol=0.002, seg_J=None):
    base_order = order_of(base_scores)
    rank_pos = {int(p): r for r, p in enumerate(base_order)}
    tol = anti_tol * max(p_full, 1e-8)
    floor = interleave_floor * max(p_full, 1e-8)
    seg_mass = list(seg_J) if seg_J is not None else [
        max(J[i] for i in seg) for seg in segments]
    res = max(2e-4 * max(p_full, 1e-8), 1e-9)
    ordered = [sorted(seg, key=lambda i: (-round(J[i] / res), rank_pos[i]))
               for seg in segments]
    strong = [(m, si) for si, m in enumerate(seg_mass) if m >= floor]
    weak = [(m, si) for si, m in enumerate(seg_mass) if -tol <= m < floor]
    anti_items = [i for si, m in enumerate(seg_mass) if m < -tol
                  for i in ordered[si]]
    strong.sort(key=lambda t: -t[0])
    weak.sort(key=lambda t: -t[0])
    ptr = {si: 0 for _, si in strong}
    taken = {si: 0 for _, si in strong}
    mmap = dict((si, m) for m, si in strong)
    seq = []
    n_strong = sum(len(ordered[si]) for _, si in strong)
    while len(seq) < n_strong:
        avail = [si for _, si in strong if ptr[si] < len(ordered[si])]
        si = max(avail, key=lambda s_: mmap[s_] / (2 * taken[s_] + 1))
        seq.append(ordered[si][ptr[si]])
        ptr[si] += 1
        taken[si] += 1
    seq.extend(i for _, si in weak for i in ordered[si])
    in_top = set(seq) | set(anti_items)
    mid = [i for i in base_order if int(i) not in in_top]
    return rank_scores_from_order(np.asarray(seq + mid + anti_items, int))
