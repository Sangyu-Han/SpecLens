#!/usr/bin/env python3
"""Test the complement-necessity / alpha-unified soft-mask objective (Bharti-Yi-
Sulam 2024, eq 13) as an FRI extension, AND whether FRI's RANDOM-BUDGET trick
makes the necessity solve budget-robust (-> better hard-deletion AUC).

Definitions (mean-baseline f_0):
  suf(m) = (p_full - f(keep m)) / (p_full - p_base)      # keep m -> full
  nec(m) = (f(keep 1-m) - p_base) / (p_full - p_base)     # keep complement -> base
  loss   = a*suf + (1-a)*nec + l1*|m|
Necessity score = m (delete high-m first). hard-deletion AUC, vs inflow.

Variants: nec-plain (a=0, no random budget = paper-style fixed solve),
nec-rb (a=0 + random budget), uni-rb (a=0.5 + random budget).
RANDOM BUDGET: each step sample B~U(0,N), w=softtopB(m); suf keeps w, nec keeps
1-w -> the objective must hold at ALL budget levels -> budget-robust ranking.
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
from src.baselines.inflow import inflow_attribution
MODELS = xm.MODELS


def solve(r, steps=64, alpha=0.0, l1=0.01, lr=0.4, lr_end=0.02, random_budget=False, seed=42):
    dev, dtype, N = r.dev, r.dtype, r.N

    def tgt(state):                                   # [N] -> scalar (differentiable)
        feat, logit = r._fwd(state.unsqueeze(0))
        if r.has_head:
            return torch.softmax(logit, -1)[0, r.target]
        if getattr(r, "use_probe", False):
            return torch.softmax(r._probe_logits(feat), -1)[0, r.target]
        fn = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (fn @ r.full_feat.squeeze())

    with torch.no_grad():
        p_full = float(tgt(torch.ones(N, device=dev, dtype=dtype)))
        p_base = float(tgt(torch.zeros(N, device=dev, dtype=dtype)))
    den = max(p_full - p_base, 1e-6)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    la = torch.zeros(N, device=dev, dtype=dtype)
    b1, b2, eps = 0.9, 0.999, 1e-8
    mv = torch.zeros(N, device=dev, dtype=dtype); vv = torch.zeros(N, device=dev, dtype=dtype)
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        m = torch.sigmoid(la_req)
        if random_budget:
            p = m / (m.sum() + 1e-8)
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            w = (p * budget).clamp(max=1.0)
            keep_suf, keep_nec = w, 1.0 - w
        else:
            keep_suf, keep_nec = m, 1.0 - m
        loss = l1 * m.sum()
        if alpha > 0:
            loss = loss + alpha * (p_full - tgt(keep_suf)) / den
        if alpha < 1:
            loss = loss + (1 - alpha) * (tgt(keep_nec) - p_base) / den
        g = torch.autograd.grad(loss, la_req)[0].detach()
        t = step + 1
        mv = b1 * mv + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
        adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
        la = la - cur_lr * adam
    return torch.sigmoid(la).detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["clip"])
    ap.add_argument("--nimg", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--del-baseline", default="mean")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/necessity_complement.json")
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
        acc = {"inflow": [], "nec_plain": [], "nec_rb": [], "uni_rb": [], "uni_rb_hins": []}
        for ip in imgs:
            try:
                x, _ = xm.load_image(ip, norm=xm.MODEL_NORM.get(mk, "clip")); x = x.to(args.device)
                r = xm.XRunner(model, x, args.device, baseline=args.del_baseline)
                if r.has_head:
                    infl = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                else:
                    infl = nec.last_attn_localization(r).astype(np.float64)
                acc["inflow"].append(nec.hard_curves(r, infl)[1])
                acc["nec_plain"].append(nec.hard_curves(r, solve(r, args.steps, 0.0, random_budget=False))[1])
                acc["nec_rb"].append(nec.hard_curves(r, solve(r, args.steps, 0.0, random_budget=True))[1])
                s_uni = solve(r, args.steps, 0.5, random_budget=True)
                hi, hd = nec.hard_curves(r, s_uni)
                acc["uni_rb"].append(hd); acc["uni_rb_hins"].append(hi)
                print(f"[{mk} {ip.name}] inflow {acc['inflow'][-1]:.3f} | nec_plain {acc['nec_plain'][-1]:.3f} "
                      f"| nec_rb {acc['nec_rb'][-1]:.3f} | uni_rb {acc['uni_rb'][-1]:.3f} (hins {hi:.3f})", flush=True)
            except Exception as e:
                print(f"[{mk} {ip.name}] ERR {type(e).__name__}: {e}", flush=True)
            finally:
                model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        from scipy.stats import wilcoxon
        a = {k: np.array(v) for k, v in acc.items()}
        def pw(x, y):
            try:
                return float(wilcoxon(x, y, alternative="greater").pvalue)
            except ValueError:
                return float("nan")
        rep = {"n": len(a["inflow"]), **{f"{k}_hdel": float(a[k].mean()) for k in ("inflow", "nec_plain", "nec_rb", "uni_rb")},
               "uni_rb_hins": float(a["uni_rb_hins"].mean()),
               "nec_plain<inflow_p": pw(a["inflow"], a["nec_plain"]),
               "nec_rb<inflow_p": pw(a["inflow"], a["nec_rb"]),
               "nec_rb<nec_plain_p": pw(a["nec_plain"], a["nec_rb"]),
               "uni_rb<inflow_p": pw(a["inflow"], a["uni_rb"])}
        report[mk] = rep
        print(f"\n=== {mk} (n={rep['n']}, {args.del_baseline}-baseline) complement-necessity ===")
        print(f"  inflow {rep['inflow_hdel']:.3f} | nec_plain {rep['nec_plain_hdel']:.3f} "
              f"| nec_rb {rep['nec_rb_hdel']:.3f} | uni_rb {rep['uni_rb_hdel']:.3f} (hins {rep['uni_rb_hins']:.3f})")
        print(f"  nec_plain<inflow p {rep['nec_plain<inflow_p']:.2g} | nec_rb<inflow p {rep['nec_rb<inflow_p']:.2g} "
              f"| RANDOM-BUDGET helps? nec_rb<nec_plain p {rep['nec_rb<nec_plain_p']:.2g} | uni_rb<inflow p {rep['uni_rb<inflow_p']:.2g}", flush=True)
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
