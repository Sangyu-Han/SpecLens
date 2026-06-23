#!/usr/bin/env python3
"""CoreFRI coalition geometry study: pointillism vs coherence vs OOD.

User hypotheses tested:
  (1) pointillist masks win hard-del partly by pushing states OOD ("breaking
      the model"), not by removing evidence -> gate DIM rounds whose states
      semantically derail (top-1 leaves the full image's top-K).
  (2) blob quality = CONNECTEDNESS, not 2D: random-walk coalitions on the
      h_b0 token-similarity knn graph (h_b0 = content + model's own pos
      code; no grid coordinates anywhere) should match blob quality.

Modes: blob (archive, vision prior) | kernel (archive, similarity fields) |
rw (knn random walks) | rw_gate (rw + OOD gate). Same DIM machinery
(adaptive density window, dual-sided suff/nec, z-sum readout, ~4.2k states).
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO.parent / "_fri_research_archive_20260610"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = _ilu.spec_from_file_location("nsr", REPO / "scripts/run_necset_n100.py")
nsr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nsr)
ns = nsr.ns
q = nsr.q

_cspec = _ilu.spec_from_file_location("cf", ARCHIVE / "scripts/research_corefri.py")
cf = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(cf)

N, GRID = 196, 14


def knn_graph(tokens, k=8, blend=None, alpha=0.5):
    """knn over token reps. `tokens`=h_b0 -> content-dominant; `tokens`=base
    (the model's own positional code) -> locality geometry, no grid coords.
    blend: optional second token set; sim = sim_a * sim_b**alpha."""
    with torch.no_grad():
        h = tokens[0].detach().float()
        hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = (hn @ hn.T).cpu().numpy()
        if blend is not None:
            g = blend[0].detach().float()
            gn = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            S2 = (gn @ gn.T).cpu().numpy()
            S = S * np.clip(S2, 0.0, None) ** alpha
    np.fill_diagonal(S, -np.inf)
    nbrs = np.argsort(-S, axis=1)[:, :k]
    return nbrs


def rw_masks(rng, n, density, nbrs, walk_len=10):
    """Connected coalitions: unions of random walks on the token knn graph.
    Connectedness without coordinates -> blob-like in vision, span-like in
    text, correlated-group-like in tabular."""
    k = int(round(density * N))
    k = min(max(k, 1), N - 1)
    out = np.zeros((n, N), np.float32)
    for i in range(n):
        sel = set()
        while len(sel) < k:
            cur = int(rng.integers(N))
            sel.add(cur)
            steps = int(min(walk_len, k - len(sel) + 1))
            for _ in range(steps):
                cur = int(nbrs[cur][rng.integers(nbrs.shape[1])])
                sel.add(cur)
                if len(sel) >= k:
                    break
        out[i, list(sel)] = 1.0
    return out


def corefri_rw(runner, *, mask_mode="rw", rounds_uniform=40, rounds_biased=24,
               batch=64, dense_frac=0.25, bias_w=1.0, seed=42, chunk=80,
               ood_gate=False, gate_topk=20, knn_k=8, walk_len=10):
    """Archive corefri_scores logic with rw mask mode + optional OOD gate."""
    rng = np.random.default_rng(seed)
    dev, dtype = runner.dev, runner.dtype
    with torch.no_grad():
        full_logits = runner.logits_for_masks(
            torch.ones(1, N, device=dev, dtype=dtype))[0]
        p_full = float(torch.softmax(full_logits, 0)[runner.target])
        top_set = set(torch.topk(full_logits, 5).indices.cpu().tolist())
        top_set.add(int(runner.target))

    if mask_mode == "rwP":
        nbrs = knn_graph(runner.base, k=knn_k)          # positional geometry
    elif mask_mode == "rwPC":
        nbrs = knn_graph(runner.base, k=knn_k, blend=runner.h_diff, alpha=0.5)
    else:
        nbrs = knn_graph(runner.h_b0, k=knn_k)
    chol = None
    chols = None
    if mask_mode in ("kernel", "kernelP", "kernelPC", "kernelP_ms"):
        with torch.no_grad():
            if mask_mode == "kernel":
                h = runner.h_b0[0]
            else:
                h = runner.base[0]            # model's own positional code
            hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            S = (hn @ hn.T).float().cpu().numpy()
            if mask_mode == "kernelPC":
                g = runner.h_diff[0].detach().float()
                gn = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                S2 = (gn @ gn.T).cpu().numpy()
                S = S * np.clip(S2, 0.0, None) ** 0.5
        if mask_mode == "kernelP_ms":
            # multi-scale temps = the blob DEFAULT_SCALES analog, prior-free
            chols = []
            for temp in (0.6, 0.25, 0.1):
                K = np.exp((S - 1.0) / temp) + 1e-3 * np.eye(N)
                chols.append(np.linalg.cholesky(K))
        else:
            K = np.exp((S - 1.0) / 0.15) + 1e-3 * np.eye(N)
            chol = np.linalg.cholesky(K)

    def gen_masks(n, density, bias=None, bw=0.0):
        if mask_mode == "blob":
            return cf.blob_masks(rng, n, density, cf.DEFAULT_SCALES,
                                 bias=bias, bias_w=bw)
        if mask_mode in ("kernel", "kernelP", "kernelPC"):
            return cf.kernel_masks(rng, n, density, chol, bias=bias, bias_w=bw)
        if mask_mode == "kernelP_ms":
            # per-mask random scale (long/mid/short range fields)
            parts = []
            counts = rng.multinomial(n, [1/3, 1/3, 1/3])
            for c_, ch_ in zip(counts, chols):
                if c_ > 0:
                    parts.append(cf.kernel_masks(rng, int(c_), density, ch_,
                                                 bias=bias, bias_w=bw))
            return np.concatenate(parts) if parts else np.zeros((0, N), np.float32)
        if mask_mode in ("rw", "rwP", "rwPC"):
            m = rw_masks(rng, n, density, nbrs, walk_len=walk_len)
            if bias is not None and bw > 0:
                # bias via seeded walks from high-bias patches
                m2 = rw_masks(rng, n, density, nbrs, walk_len=walk_len)
                pick = rng.random(n) < 0.5
                m[pick] = m2[pick]
            return m
        raise ValueError(mask_mode)

    def probs_and_gate(masks):
        outs, gates = [], []
        with torch.no_grad():
            for i in range(0, len(masks), chunk):
                mt = torch.as_tensor(masks[i:i + chunk], device=dev, dtype=dtype)
                logits = runner.logits_for_masks(mt)
                sm = torch.softmax(logits, -1)
                pr = sm[:, runner.target]
                outs.append(pr.cpu().numpy())
                msp = sm.max(-1).values.cpu().numpy()
                top1 = logits.argmax(-1).cpu().numpy()
                in_top = np.array([int(t) in top_set for t in top1])
                gates.append(in_top | (msp >= 0.4))
        return np.concatenate(outs), np.concatenate(gates)

    probe_d = (0.02, 0.04, 0.07, 0.12, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)
    probe_masks = np.concatenate([gen_masks(12, d) for d in probe_d])
    probe_p, _ = probs_and_gate(probe_masks)
    if ood_gate:
        # diagnostic: MSP under deletion states, mode vs iid, same density
        for dd in (0.2, 0.4):
            md_mode = 1.0 - gen_masks(24, dd)
            md_iid = 1.0 - cf.iid_masks(rng, 24, dd)
            _, _g1 = probs_and_gate(md_mode)
            _, _g2 = probs_and_gate(md_iid)
            print(f"    [oodprobe d={dd}] in-dist: mode={_g1.mean():.2f} iid={_g2.mean():.2f}",
                  flush=True)
    probe = {d: float(probe_p[i * 12:(i + 1) * 12].mean()) for i, d in enumerate(probe_d)}
    recs = {d: probe[d] / max(p_full, 1e-8) for d in probe}
    inside = [d for d in probe_d if 0.05 <= recs[d] <= 0.75]
    if not inside:
        diffs = [(recs[b] - recs[a], a, b) for a, b in zip(probe_d, probe_d[1:])]
        _, a, b = max(diffs)
        inside = [a, b]
    s_lo, s_hi = max(min(inside) * 0.7, 0.02), min(max(inside) * 1.3, 0.95)
    inside_k = [d for d in probe_d if 0.25 <= recs[d] <= 0.95]
    if inside_k:
        d_lo, d_hi = max(1.0 - max(inside_k), 0.03), min(1.0 - min(inside_k), 0.7)
        if d_hi <= d_lo:
            d_lo, d_hi = 0.1, 0.5
    else:
        d_lo, d_hi = 0.1, 0.5

    suff, nec = cf._DIM(), cf._DIM()
    n_dense = max(4, int(batch * dense_frac))
    n_sparse = batch - n_dense
    n_gated = [0, 0]

    def run_phase(rounds, bias):
        for _ in range(rounds):
            d_s = float(np.exp(rng.uniform(np.log(s_lo), np.log(s_hi))))
            d_d = float(np.exp(rng.uniform(np.log(d_lo), np.log(d_hi))))
            ms = gen_masks(n_sparse, d_s, bias=bias, bw=bias_w)
            md = gen_masks(n_dense, d_d, bias=bias, bw=bias_w)
            probs, gate = probs_and_gate(np.concatenate([ms, 1.0 - md]))
            ps, pd_ = probs[:n_sparse], probs[n_sparse:]
            gs, gd = gate[:n_sparse], gate[n_sparse:]
            if ood_gate:
                # deletion side: a state whose top-1 leaves the full top-K is
                # OOD-broken -> exclude from necessity estimation
                keepd = gd
                n_gated[0] += int((~keepd).sum())
                n_gated[1] += len(keepd)
                if keepd.sum() >= 4:
                    nec.add((1.0 - md)[keepd], pd_[keepd],
                            max(float(pd_[keepd].std()), 1e-6))
            else:
                nec.add(1.0 - md, pd_, max(float(pd_.std()), 1e-6))
            suff.add(ms, ps, max(float(ps.std()), 1e-6))

    run_phase(rounds_uniform, None)
    interim = suff.dim()
    bias = (interim - interim.mean()) / (interim.std() + 1e-12)
    run_phase(rounds_biased, bias)
    s_arr, n_arr = suff.dim(), nec.dim()

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-12)

    return {"scores": (z(s_arr) + z(n_arr)).astype(np.float32),
            "gated_frac": (n_gated[0] / max(n_gated[1], 1)) if ood_gate else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/qual_corefri_rw.png")
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    mean = torch.tensor(model.pretrained_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"]).view(3, 1, 1)

    def up(m):
        return np.kron(np.asarray(m, np.float64).reshape(GRID, GRID),
                       np.ones((16, 16)))

    def draw_signed(ax, img, sc, title):
        ax.imshow(img)
        m = np.asarray(sc, np.float64)
        vmax = max(np.abs(m).max(), 1e-8)
        ax.imshow(up(m), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)

    rows = {}
    modes = [("blob", False), ("kernelP", False), ("kernelP_ms", False)]
    labels = ["CoreFRI-blob", "kernelP (single)", "kernelP-MS (multi-scale)"]
    ncol = 1 + len(modes)
    fig, axes = plt.subplots(len(ns.CASES), ncol,
                             figsize=(3.1 * ncol, 3.3 * len(ns.CASES)))
    t0 = time.time()
    for ri, (name, path, target) in enumerate(ns.CASES):
        x, _ = syms["load_image"](Path(path))
        x = x.to(args.device)
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)
        img = (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        ax = axes[ri, 0]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=8)
        res = {}
        line = f"[{name:<11}]"
        for ci, ((mode, gate), lab) in enumerate(zip(modes, labels), start=1):
            out = corefri_rw(runner, mask_mode=mode, ood_gate=gate,
                             seed=int(args.seed))
            sc_ = out["scores"]
            hins, hdel, _, _ = q.hard_curves(runner, sc_, chunk=128)
            res[lab] = {"hdel": float(hdel), "hins": float(hins),
                        "gated_frac": out["gated_frac"]}
            gtxt = f" g={out['gated_frac']:.2f}" if gate else ""
            draw_signed(axes[ri, ci], img, sc_, f"{lab} {hdel:.3f}{gtxt}")
            line += f" {lab.split()[0]}={hdel:.3f}"
        rows[name] = res
        print(line, flush=True)

    print()
    for lab in labels:
        d = np.array([rows[k][lab]["hdel"] for k in rows])
        print(f"{lab:16} mean {d.mean():.4f}")
    plt.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
