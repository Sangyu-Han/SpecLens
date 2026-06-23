#!/usr/bin/env python3
"""Random-Budget Necessity (RBN) — the deletion-value mirror of FRI.

Self-contained (input->output only; NO hidden self-patch, NO external saliency).
Mechanism: soft necessity mask m; each step sample random PARTIAL backgrounds
b~Bernoulli(rho) INDEPENDENT of m, and push m toward patches whose removal from
b drops the target (escapes full-background saturation -> hits the MRI
'informative band').  loss = E_b[ tgt(b*(1-m)) ] + l1*||m||.

Runs the solve AND records MRI-style diagnostics so failure is informative:
 - rho-resolved drop of the final top set (where is the signal? informative band)
 - corr(m, single-del oracle), corr(m, inflow)  (is it just saliency / per-patch?)
 - module coherence: top-k overlap with a single last-hidden cluster (DIAGNOSTIC
   ONLY, not used by the method) -> does RBN find a redundant module?
 - saturation of m, eval hins/hdel vs inflow.
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
nec = _load("nec", "scripts/research_xmodel_necessity.py")
from src.utils.image import load_image_clip
from src.baselines.inflow import inflow_attribution
MODELS = xm.MODELS


def tgt_fn(r):
    def tgt(states):  # states [B,N] differentiable -> [B]
        feat, logit = r._fwd(states)
        if r.has_head:
            return torch.softmax(logit, -1)[:, r.target]
        fn = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (fn @ r.full_feat.squeeze())
    return tgt


def rbn_solve(r, steps=48, n_bg=24, l1=0.02, lr=0.4, lr_end=0.02, seed=42,
              rho_min=0.0, rho_max=1.0):
    dev, dtype, N = r.dev, r.dtype, r.N
    tgt = tgt_fn(r)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    la = torch.zeros(N, device=dev, dtype=dtype)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_v = torch.zeros(N, device=dev, dtype=dtype); v_v = torch.zeros(N, device=dev, dtype=dtype)
    hist = {"loss": [], "tdel": [], "msum": []}
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        m = torch.sigmoid(la_req)
        rho = rho_min + (rho_max - rho_min) * torch.rand(n_bg, 1, generator=gen, device=dev)
        b = (torch.rand(n_bg, N, generator=gen, device=dev) < rho).to(dtype)
        state = b * (1.0 - m.unsqueeze(0))                 # remove candidates from bg
        t_del = tgt(state)
        loss = t_del.mean() + l1 * m.sum()
        g = torch.autograd.grad(loss, la_req)[0].detach()
        t = step + 1
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g * g
        adam_dir = (m_v / (1 - beta1 ** t)) / ((v_v / (1 - beta2 ** t)).sqrt() + eps)
        la = la - cur_lr * adam_dir
        hist["loss"].append(float(loss)); hist["tdel"].append(float(t_del.mean())); hist["msum"].append(float(m.sum()))
    return torch.sigmoid(la).detach().float().cpu().numpy(), la.detach().float().cpu().numpy(), hist


def rho_signal(r, score, ks=32, n_bg=64, seed=7):
    """drop of removing the top-ks (by score) from random backgrounds at each rho."""
    dev, dtype, N = r.dev, r.dtype, r.N
    tgt = tgt_fn(r)
    order = np.argsort(-score)[:ks]
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    out = {}
    with torch.no_grad():
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            b = (torch.rand(n_bg, N, generator=gen, device=dev) < rho).to(dtype)
            b_on = b.clone(); b_on[:, order] = 1.0
            b_off = b_on.clone(); b_off[:, order] = 0.0
            drop = (tgt(b_on) - tgt(b_off)).mean()
            out[rho] = float(drop)
    return out


def module_coherence(score, hL, ks=32, k=16):
    """DIAGNOSTIC ONLY: of the top-ks patches, what frac fall in the single most
    common last-hidden cluster? high => RBN found a coherent redundant module."""
    labs = nec.kml(hL, k)
    top = np.argsort(-score)[:ks]
    counts = np.bincount(labs[top], minlength=k)
    return float(counts.max() / ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["clip"])
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--n-bg", type=int, default=24)
    ap.add_argument("--l1", type=float, default=0.02)
    ap.add_argument("--rho-min", type=float, default=0.0)
    ap.add_argument("--rho-max", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/rbn_diag.json")
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
        rows = []
        for ip in imgs:
            try:
                x, _ = load_image_clip(ip); x = x.to(args.device)
                r = xm.XRunner(model, x, args.device)
                m, theta, hist = rbn_solve(r, steps=args.steps, n_bg=args.n_bg, l1=args.l1,
                                           rho_min=args.rho_min, rho_max=args.rho_max)
                hi_r, hd_r = nec.hard_curves(r, theta)          # rank by theta (finer than saturated m)
                # baselines
                if r.has_head:
                    infl = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                else:
                    infl = nec.last_attn_localization(r).astype(np.float64)
                hi_i, hd_i = nec.hard_curves(r, infl)
                # diagnostics
                o_del, full = xm.single_del_oracle(r)
                hL = r.last_hidden_patches()
                rsig = rho_signal(r, theta)
                rsig_inf = rho_signal(r, infl)
                rows.append({
                    "img": ip.name,
                    "hdel_rbn": hd_r, "hins_rbn": hi_r, "hdel_inflow": hd_i, "hins_inflow": hi_i,
                    "corr_m_singledel": float(np.corrcoef(theta, o_del)[0, 1]),
                    "corr_m_inflow": float(np.corrcoef(theta, infl)[0, 1]),
                    "mod_coh_rbn": module_coherence(theta, hL),
                    "mod_coh_inflow": module_coherence(infl, hL),
                    "msat": float((m > 0.9).mean()), "mzero": float((m < 0.1).mean()),
                    "rho_signal_rbn": rsig, "rho_signal_inflow": rsig_inf,
                    "tdel_final": hist["tdel"][-1], "tdel_init": hist["tdel"][0],
                })
                print(f"[{mk} {ip.name}] hdel rbn {hd_r:.3f} vs inflow {hd_i:.3f} | "
                      f"corr(m,inflow) {rows[-1]['corr_m_inflow']:.2f} modcoh {rows[-1]['mod_coh_rbn']:.2f} "
                      f"msat {rows[-1]['msat']:.2f}", flush=True)
            except Exception as e:
                print(f"[{mk} {ip.name}] ERR {type(e).__name__}: {e}", flush=True)
            finally:
                model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if not rows:
            continue
        def mean(k):
            return float(np.mean([row[k] for row in rows]))
        agg = {k: mean(k) for k in ["hdel_rbn", "hins_rbn", "hdel_inflow", "hins_inflow",
                                    "corr_m_singledel", "corr_m_inflow", "mod_coh_rbn",
                                    "mod_coh_inflow", "msat", "mzero"]}
        from scipy.stats import wilcoxon
        dr = np.array([row["hdel_rbn"] for row in rows]); di = np.array([row["hdel_inflow"] for row in rows])
        try:
            agg["p_hdel_rbn_better"] = float(wilcoxon(di, dr, alternative="greater").pvalue)
        except ValueError:
            agg["p_hdel_rbn_better"] = float("nan")
        agg["win_rbn"] = int((dr < di).sum()); agg["n"] = len(rows)
        # mean rho-signal
        rs = {str(rho): float(np.mean([row["rho_signal_rbn"][rho] for row in rows])) for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)}
        rs_i = {str(rho): float(np.mean([row["rho_signal_inflow"][rho] for row in rows])) for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)}
        agg["rho_signal_rbn"] = rs; agg["rho_signal_inflow"] = rs_i
        report[mk] = {"agg": agg, "rows": rows}
        print(f"\n=== {mk}: RBN vs inflow (n={len(rows)}) ===")
        print(f"  hdel  RBN {agg['hdel_rbn']:.3f}  inflow {agg['hdel_inflow']:.3f}  "
              f"(win {agg['win_rbn']}/{agg['n']}, p {agg['p_hdel_rbn_better']:.2g})")
        print(f"  hins  RBN {agg['hins_rbn']:.3f}  inflow {agg['hins_inflow']:.3f}")
        print(f"  corr(m,single-del) {agg['corr_m_singledel']:.2f}  corr(m,inflow) {agg['corr_m_inflow']:.2f}")
        print(f"  module-coherence RBN {agg['mod_coh_rbn']:.2f}  inflow {agg['mod_coh_inflow']:.2f}  (1=single module)")
        print(f"  m saturation>0.9 {agg['msat']:.2f}  m~0 {agg['mzero']:.2f}")
        print(f"  rho-signal (drop of top32 vs background density):")
        print(f"    RBN    " + "  ".join(f"r{rho}:{rs[str(rho)]:.3f}" for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)))
        print(f"    inflow " + "  ".join(f"r{rho}:{rs_i[str(rho)]:.3f}" for rho in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)))
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
