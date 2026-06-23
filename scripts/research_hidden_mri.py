#!/usr/bin/env python3
"""Hidden-layer MRI: which LAST-layer signal carries NECESSITY (not just
localization)? User hint: inflow localizes well because it exploits the
last hidden layer -- but localization != necessity (measured). Find the
last-hidden signal aligned with the single-del oracle / good deletion.

Per case (full state, 1 fwd + 1 bwd):
  a_j   : last-block CLS->patch attention (head-mean)
  a3_j  : last-3-block CLS->patch attention
  ag_j  : gradcam on CLS attention  a_j * relu(d z_t / d a_j)
  hg_j  : last-hidden input x grad  relu(h_L[j] . d z_t/d h_L[j])
  lev_j : ATTENTION-RENORM LEVERAGE  a_j * (c_j - cbar), c_j = hg_j,
          cbar = attention-weighted mean of c (the contribution that
          re-normalization would redistribute) -> models deletion drop
  lev_g : same with c_j = ag_j
Compares each to the single-del oracle (spearman) and as an hdel score.
Also tests FUNCTIONAL clustering: kmeans on last-hidden value vectors vs
content (h_diff) -- does deleting whole last-hidden clusters drop p more
(super-additive evidence modules)?  Archetypes + dev sample.
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

_spec = _ilu.spec_from_file_location("gc", REPO / "scripts/research_fri_g_common.py")
gc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(gc)
q = gc.q
N = 196


def spear(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def hidden_signals(runner):
    """One fwd+bwd at full state; capture last-block attention + last
    hidden, return candidate per-patch necessity signals."""
    model = runner.model
    dev, dtype = runner.dev, runner.dtype
    for blk in model.blocks:
        blk.attn.fused_attn = False
    nb = len(model.blocks)
    attn_store = {}      # block idx -> attn weights [1,H,N,N] (graph)
    hL_store = {}        # block idx -> output [1,N,D] (graph)
    hooks = []
    for bi, blk in enumerate(model.blocks):
        hooks.append(blk.attn.attn_drop.register_forward_hook(
            lambda m, a, o, i=bi: attn_store.__setitem__(i, o)))
        hooks.append(blk.register_forward_hook(
            lambda m, a, o, i=bi: hL_store.__setitem__(
                i, (o if torch.is_tensor(o) else o[0]))))
    h_mix = (runner.base + runner.h_diff).detach().clone().requires_grad_(True)
    h_inj = torch.cat([runner.cls, h_mix], dim=1)
    holder = [h_inj]
    pre = model.blocks[0].register_forward_pre_hook(lambda m, a: (holder[0],))
    try:
        out = model(runner.x)
        z_t = out[0, runner.target]
        A_last = attn_store[nb - 1]            # [1,H,N,N]
        hL = hL_store[nb - 1]                  # [1,N,D] last block output
        # grads
        gA = torch.autograd.grad(z_t, A_last, retain_graph=True)[0]   # [1,H,N,N]
        ghL = torch.autograd.grad(z_t, hL, retain_graph=False)[0]     # [1,N,D]
    finally:
        pre.remove()
        for h in hooks:
            h.remove()

    A = A_last.detach()[0]                       # [H,N,N]
    a = A[:, 0, 1:].mean(0).cpu().numpy()        # CLS->patch, head-mean [196]
    a3 = np.mean([attn_store[i].detach()[0][:, 0, 1:].mean(0).cpu().numpy()
                  for i in range(nb - 3, nb)], axis=0)
    gA0 = gA.detach()[0][:, 0, 1:].mean(0).cpu().numpy()
    ag = a * np.clip(gA0, 0, None)
    hLd = hL.detach()[0, 1:]                      # [196,D]
    ghLd = ghL.detach()[0, 1:]                    # [196,D]
    hg = np.clip((hLd * ghLd).sum(-1).cpu().numpy(), 0, None)
    # attention-renorm leverage: a_j*(c_j - cbar), cbar = attn-wgt mean of c
    aw = a / (a.sum() + 1e-8)

    def leverage(c):
        cbar = float((aw * c).sum())
        return a * (c - cbar)

    lev = leverage(hg)
    lev_g = leverage(ag)
    return {"a": a, "a3": a3, "ag": ag, "hg": hg, "lev": lev, "lev_g": lev_g,
            "hL": hLd.float().cpu().numpy()}


def single_del_oracle(runner):
    masks = np.ones((N, N), np.float32)
    masks[np.arange(N), np.arange(N)] = 0.0
    p_del = runner.prob_curve(masks, 128)
    p_full = float(runner.prob_curve(np.ones((1, N), np.float32))[0])
    return (p_full - p_del).astype(np.float32), p_full


def cluster_superadd(runner, vecs, o_del, p_full, k=12, seed=0, tag=""):
    from sklearn.cluster import KMeans
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(vn).labels_
    masks, sums = [], []
    for c in range(k):
        m = np.ones(N, np.float32); m[lab == c] = 0.0
        masks.append(m); sums.append(o_del[lab == c].sum())
    grp = p_full - runner.prob_curve(np.stack(masks), 128)
    sums = np.array(sums)
    # super-additivity where the cluster has real evidence mass
    sel = sums > 0.02 * p_full
    if sel.sum() == 0:
        return 0.0, lab
    return float(np.mean((grp[sel] - sums[sel]) / (sums[sel] + 1e-6))), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=gc.OUTDIR / "hidden_mri.json")
    args = ap.parse_args()

    model, syms = gc.load_model(args.device)
    cases = gc.dev_cases()[: args.n]
    sigs = ["a", "a3", "ag", "hg", "lev", "lev_g"]
    rows = {}
    t0 = time.time()
    for name, path, target in cases:
        runner, target = gc.make_runner(model, syms, path, target, args.device)
        o_del, p_full = single_del_oracle(runner)
        infl = gc.inflow_score(model, syms, runner.x, target)
        S = hidden_signals(runner)
        rec = {"inflow": {"hdel": gc.eval_score(runner, infl)["hdel"],
                          "rho_odel": spear(infl, o_del)}}
        for s in sigs:
            rec[s] = {"hdel": gc.eval_score(runner, S[s])["hdel"],
                      "rho_odel": spear(S[s], o_del)}
        # functional vs content clustering super-additivity
        sa_fn, _ = cluster_superadd(runner, S["hL"], o_del, p_full, tag="hL")
        sa_ct, _ = cluster_superadd(runner, runner.h_diff[0].float().cpu().numpy(),
                                    o_del, p_full, tag="content")
        rec["superadd_lasthidden"] = sa_fn
        rec["superadd_content"] = sa_ct
        rows[name] = rec
        print(f"[{name:<13}] infl hd={rec['inflow']['hdel']:.3f}/rho={rec['inflow']['rho_odel']:+.2f} "
              f"lev hd={rec['lev']['hdel']:.3f}/rho={rec['lev']['rho_odel']:+.2f} "
              f"hg rho={rec['hg']['rho_odel']:+.2f} a rho={rec['a']['rho_odel']:+.2f} "
              f"SA hL/ct={sa_fn:+.2f}/{sa_ct:+.2f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== hidden MRI (dev{len(rows)}) ===")
    print(f"{'sig':8} {'hdel':>7} {'rho_odel':>9}")
    summ = {}
    for s in ["inflow"] + sigs:
        hd = np.mean([rows[k][s]["hdel"] for k in rows])
        rho = np.mean([rows[k][s]["rho_odel"] for k in rows])
        summ[s] = {"hdel": float(hd), "rho_odel": float(rho)}
        print(f"{s:8} {hd:7.4f} {rho:>+9.3f}")
    sahl = np.mean([rows[k]["superadd_lasthidden"] for k in rows])
    sact = np.mean([rows[k]["superadd_content"] for k in rows])
    print(f"\ncluster super-additivity (deleting whole cluster vs sum-of-singles):")
    print(f"  last-hidden value clusters: {sahl:+.3f}")
    print(f"  content (h_diff) clusters : {sact:+.3f}")
    summ["superadd"] = {"last_hidden": float(sahl), "content": float(sact)}
    args.out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
