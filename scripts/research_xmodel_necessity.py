#!/usr/bin/env python3
"""Cross-model necessity transfer (self-contained on XRunner; no CLIP-locked
repo plumbing, no inflow).

Tests the core claim across architectures: does reordering a localization
map into LAST-HIDDEN MODULES (whole-module deletion + est-guard) improve
hard-DELETION over the localization alone?

  localization L = last-layer attention to the pooling token (CLS row for
    prefix>0 models; column-mean attention received, for GAP/no-CLS).
    Universal, cheap, model-general.
  necessity = guarded module reorder of L using last-hidden value clusters.
  metric = hdel/hins via target (class prob if IN1k head; else feature-
    cosine drift for headless SigLIP/DINOv2).

Reports per model: hdel(L) vs hdel(necessity), win rate, + super-add and
massive-activation profile. Models incl. DINOv2 reg4 (registers) & SigLIP
(no CLS) — the token-pathology stress cases.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec = _ilu.spec_from_file_location("xm", REPO / "scripts/research_xmodel_mechanism.py")
xm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(xm)
from src.utils.image import load_image_clip
from src.baselines.inflow import inflow_attribution
MODELS = xm.MODELS
KS_GUARD = (4, 8, 16, 32, 48, 72, 120)


def kml(vecs, k, seed=0):
    from sklearn.cluster import KMeans
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit(vn).labels_


def hard_curves(r, scores):
    N = r.N
    order = np.argsort(-np.asarray(scores, np.float64).reshape(-1))
    ins = np.zeros((N + 1, N), np.float32); dele = np.ones((N + 1, N), np.float32)
    for k in range(1, N + 1):
        ins[k] = ins[k - 1]; ins[k, order[k - 1]] = 1.0
        dele[k] = dele[k - 1]; dele[k, order[k - 1]] = 0.0
    pi = r.target_curve(ins); pd_ = r.target_curve(dele)
    base, full = float(pi[0]), float(pi[-1])
    den = max(full - base, 1e-6)
    xs = np.linspace(0, 1, N + 1)
    reci = np.clip((pi - base) / den, 0, 1); recd = np.clip((pd_ - base) / den, 0, 1)
    return float(np.trapz(reci, xs)), float(np.trapz(recd, xs))   # hins, hdel


def clip_auc(r, order, p_full, p_base):
    N = r.N
    masks = []
    for k in KS_GUARD:
        m = np.ones(N, np.float32); m[order[:k]] = 0.0
        masks.append(m)
    vals = r.target_curve(np.stack(masks))
    xs = np.array([0.0, *[k / N for k in KS_GUARD], 1.0]); den = max(p_full - p_base, 1e-6)
    ys = np.array([p_full, *vals.tolist(), p_base])
    return float(np.trapz(np.clip((ys - p_base) / den, 0, 1), xs))


def last_attn_localization(r):
    """last-block attention to the pool. CLS row (prefix>0) or received-mean."""
    m = r.model
    for b in m.blocks:
        if hasattr(b.attn, "fused_attn"):
            b.attn.fused_attn = False
    store = {}
    hk = m.blocks[-1].attn.attn_drop.register_forward_hook(
        lambda mod, a, o: store.__setitem__("A", o.detach()))
    holder = [torch.cat([r.prefix, r.h_b0], dim=1)]
    pre = m.blocks[0].register_forward_pre_hook(lambda mod, a: (holder[0],))
    try:
        with torch.no_grad():
            m(r.x)
    finally:
        pre.remove(); hk.remove()
    A = store["A"][0]                                   # [H,T,T]
    if r.npref > 0:
        loc = A[:, 0, r.npref:].mean(0)                 # CLS -> patch
    else:
        loc = A[:, :, :].mean(0).mean(0)[r.npref:]      # received attention
    return loc.float().cpu().numpy()


def fri_localization(r, steps=32, seed=42, lr=0.45, lr_end=0.01,
                     l1_weight=0.003, irr_weight=0.05, init_prob=0.5):
    """Pure-insertion FRI solve (method A) used as a model-general localization
    / implicit ERF. Target = class prob (head) or feature-cosine recovery
    (headless). No 2D TV prior (FRI spirit: model-agnostic, RoPE/tabular-safe).
    irrelevance := inverse mask-sensitivity (ERF-like). Returns soft mask [N]
    (higher = more sufficient)."""
    import math
    dev, dtype, N = r.dev, r.dtype, r.N

    def tgt(feat, logit):
        if r.has_head:
            return torch.softmax(logit[0], 0)[r.target]
        fn = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (fn @ r.full_feat.squeeze()).squeeze()

    with torch.no_grad():
        f1, l1_ = r._fwd(torch.ones(1, N, device=dev, dtype=dtype))
        f0, l0_ = r._fwd(torch.zeros(1, N, device=dev, dtype=dtype))
        full_obj = tgt(f1, l1_).detach(); base_obj = tgt(f0, l0_).detach()
    den = (full_obj - base_obj)
    den = den if den.abs() >= 1e-8 else torch.full_like(den, 1e-8)
    # irrelevance := inverse mask-sensitivity at full image (ERF-like)
    m1 = torch.ones(1, N, device=dev, dtype=dtype, requires_grad=True)
    feat, logit = r._fwd(m1)
    g0 = torch.autograd.grad(tgt(feat, logit), m1)[0][0].abs()
    inv = 1.0 / (g0 + 1e-8); irr = (inv / inv.max().clamp(min=1e-8)).detach().reshape(-1)

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
        state = (p * budget).clamp(max=1.0)            # pure insertion (no del phase)
        feat, logit = r._fwd(state.unsqueeze(0))
        rec = (tgt(feat, logit) - base_obj) / den
        loss = (1.0 - rec) + irr_weight * (probs * irr).sum() + l1_weight * probs.sum()
        g = torch.autograd.grad(loss, la_req)[0].detach()
        t = step + 1
        m_v = beta1 * m_v + (1 - beta1) * g
        v_v = beta2 * v_v + (1 - beta2) * g * g
        adam_dir = (m_v / (1 - beta1 ** t)) / ((v_v / (1 - beta2 ** t)).sqrt() + eps)
        cmask = (adam_dir * g > 0).to(dtype); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
        la = la - cur_lr * adam_dir * cmask
    return torch.sigmoid(la).detach().float().cpu().numpy()


def module_order(labels, loc, crank):
    order = []
    for c in np.argsort(-crank):
        mem = np.where(labels == c)[0]
        order.extend(mem[np.argsort(-loc[mem])].tolist())
    return np.asarray(order, int)


def rankscore(order, N):
    s = np.zeros(N); s[order] = np.arange(N, 0, -1); return s


def hidden_ablate_ranks(r, clusters_by_k, p_full):
    """w_c per cluster = drop in target when cluster's tokens are neutralized
    at the LAST-BLOCK INPUT (representation-necessity ranking). One dict per k."""
    m = r.model; nb = len(m.blocks); N = r.N
    cap = {}
    hk = m.blocks[nb - 1].register_forward_pre_hook(lambda mod, a: cap.__setitem__("h", a[0].detach()))
    holder0 = [torch.cat([r.prefix, r.h_b0], dim=1)]
    pre0 = m.blocks[0].register_forward_pre_hook(lambda mod, a: (holder0[0],))
    try:
        with torch.no_grad():
            m(r.x)
    finally:
        pre0.remove(); hk.remove()
    H = cap["h"]; npref = H.shape[1] - N; pmean = H[0, npref:].mean(0)

    def tgt(Hm):
        holder = [Hm]
        pre = m.blocks[nb - 1].register_forward_pre_hook(lambda mod, a: (holder[0],))
        store = {}
        hk2 = m.blocks[-1].register_forward_hook(lambda mod, a, o: store.__setitem__("h", (o if torch.is_tensor(o) else o[0])))
        try:
            with torch.no_grad():
                out = m(r.x)
        finally:
            pre.remove(); hk2.remove()
        if r.has_head:
            return float(torch.softmax(out[0], 0)[r.target])
        lh2 = store["h"]; nrm = (m.norm(lh2) if hasattr(m, "norm") else lh2)
        if getattr(r, "use_probe", False):
            pooled = nrm[:, 0] if r.npref > 0 else nrm[:, npref:].mean(1)   # match XRunner pooling
            return float(torch.softmax(r._probe_logits(pooled), -1)[0, r.target])
        pooled = nrm[:, npref:].mean(1)
        pn = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return float((pn @ r.full_feat.squeeze()).item())

    out = {}
    for k, clusters in clusters_by_k.items():
        w = np.zeros(len(clusters))
        for c, mem in enumerate(clusters):
            Hm = H.clone(); Hm[0, np.asarray(mem, int) + npref] = pmean
            w[c] = max(p_full - tgt(Hm), 0.0)
        out[k] = w
    return out


def guarded_necessity(r, loc, hL, p_full, p_base, ks=(8, 12, 16, 24), hidden=False):
    N = r.N
    cands = {"loc": rankscore(np.argsort(-loc), N)}
    labs = {k: kml(hL, k) for k in ks}
    for k in ks:
        lab = labs[k]
        crank = np.array([loc[lab == c].sum() for c in range(k)])
        cands[f"mod{k}"] = rankscore(module_order(lab, loc, crank), N)
    if hidden:
        clusters_by_k = {k: [np.where(labs[k] == c)[0] for c in range(k)] for k in ks}
        wbyk = hidden_ablate_ranks(r, clusters_by_k, p_full)
        for k in ks:
            cands[f"hid{k}"] = rankscore(module_order(labs[k], loc, wbyk[k]), N)
    best, bauc = None, 1e9
    for nm, sc in cands.items():
        auc = clip_auc(r, np.argsort(-sc), p_full, p_base)
        if auc < bauc:
            bauc, best = auc, nm
    return cands[best], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--nimg", type=int, default=10)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ks-mod", type=int, nargs="+", default=[8, 12, 16, 24])
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--superadd", action="store_true", help="also compute super-add (slow: N fwd/img)")
    ap.add_argument("--baseline", choices=("attn", "inflow"), default="attn",
                    help="localization baseline: attn (model-general) or inflow (head models only)")
    ap.add_argument("--localization", choices=("attn", "inflow", "fri"), default=None,
                    help="localization source; overrides --baseline if set (fri = pure-ins FRI solve)")
    ap.add_argument("--fri-steps", type=int, default=32)
    ap.add_argument("--del-baseline", choices=("pos", "mean"), default="pos",
                    help="deletion operator baseline (mean fixes SSL/headless OOD)")
    ap.add_argument("--probe-dir", type=Path, default=None,
                    help="dir with probe_{model}.pt -> sharp class-prob target for headless")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/xmodel_necessity.json")
    args = ap.parse_args()
    import timm

    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(123).shuffle(paths)
    imgs = paths[: args.nimg]
    report = {}
    for mk in args.models:
        try:
            kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in MODELS[mk] else {}
            model = timm.create_model(MODELS[mk], pretrained=True, **kw).eval().to(args.device)
            for p in model.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[{mk}] LOAD FAIL {e}", flush=True); continue
        probe = None
        if args.probe_dir is not None:
            pp = args.probe_dir / f"probe_{mk}.pt"
            if pp.exists():
                probe = torch.load(pp, map_location=args.device)
                print(f"[{mk}] loaded probe head (acc {probe.get('acc', float('nan')):.3f})", flush=True)
        L_hdel, N_hdel, L_hins, N_hins, picks, saL, saC, mtimes = [], [], [], [], [], [], [], []
        t0 = time.time()
        nb = len(model.blocks)
        # method forward-cost (masks) — analytic, deterministic given ks/k
        locsrc = args.localization or args.baseline
        ncand = 1 + len(args.ks_mod) + (len(args.ks_mod) if args.hidden else 0)
        cost = {"localization_fwd": (args.fri_steps + 1 if locsrc == "fri" else 1),
                "last_hidden_fwd": 1,
                "guard_fwd_batched": ncand * len(KS_GUARD),
                "hidden_ablation_fwd_seq": (sum(args.ks_mod) if args.hidden else 0),
                "backward": (args.fri_steps + 1 if locsrc == "fri" else 0)}
        for ip in imgs:
            try:
                x, _ = xm.load_image(ip, norm=xm.MODEL_NORM.get(mk, "clip")); x = x.to(args.device)
                r = xm.XRunner(model, x, args.device, baseline=args.del_baseline, probe=probe)
                with torch.no_grad():
                    p_full = float(r.target_curve(np.ones((1, r.N), np.float32))[0])
                    p_base = float(r.target_curve(np.zeros((1, r.N), np.float32))[0])
                tm0 = time.time()                         # METHOD wall-clock
                if locsrc == "fri":
                    loc = fri_localization(r, steps=args.fri_steps).astype(np.float64)
                elif locsrc == "inflow" and r.has_head:
                    loc = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                else:
                    loc = last_attn_localization(r)
                hL = r.last_hidden_patches()
                nec, pick = guarded_necessity(r, loc, hL, p_full, p_base,
                                              ks=args.ks_mod, hidden=args.hidden)
                mtimes.append(time.time() - tm0)
                hi_l, hd_l = hard_curves(r, loc)          # EVAL (metric), not method
                hi_n, hd_n = hard_curves(r, nec)
                L_hdel.append(hd_l); N_hdel.append(hd_n); L_hins.append(hi_l); N_hins.append(hi_n)
                picks.append(pick)
                if args.superadd:
                    o_del, full = xm.single_del_oracle(r)
                    saL.append(xm.superadd(r, kml(hL, args.k), o_del, full, args.k))
                    saC.append(xm.superadd(r, kml(r.h_diff[0].float().cpu().numpy(), args.k), o_del, full, args.k))
            except Exception as e:
                print(f"[{mk}] {ip.name} ERR {type(e).__name__}: {e}", flush=True); continue
            finally:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
        if not N_hdel:
            del model; torch.cuda.empty_cache(); continue
        from scipy.stats import wilcoxon
        Ld, Nd = np.array(L_hdel), np.array(N_hdel)
        win = int((Nd < Ld).sum())
        try:
            p_hd = float(wilcoxon(Ld, Nd, alternative="greater").pvalue)
        except ValueError:
            p_hd = float("nan")
        def md(v):
            v = [x for x in v if np.isfinite(x)]; return float(np.median(v)) if v else float("nan")
        method_fwd = sum(cost.values())
        report[mk] = {"model": MODELS[mk], "npref": r.npref, "N": r.N, "has_head": bool(r.has_head),
                      "loc_hdel": float(Ld.mean()), "nec_hdel": float(Nd.mean()), "p_hdel": p_hd,
                      "loc_hins": float(np.mean(L_hins)), "nec_hins": float(np.mean(N_hins)),
                      "win_del": win, "n": len(Nd),
                      "loc_hdel_arr": Ld.tolist(), "nec_hdel_arr": Nd.tolist(),
                      "localization": locsrc,
                      "superadd_lasthid": md(saL), "superadd_content": md(saC),
                      "mod_pick_frac": float(np.mean([p != "loc" for p in picks])),
                      "cost_method_fwd": method_fwd, "cost_breakdown": cost,
                      "method_walltime_s_per_img": float(np.mean(mtimes))}
        print(f"[{mk:11}] N={r.N} pref={r.npref} head={r.has_head} | "
              f"hdel loc {Ld.mean():.3f} -> nec {Nd.mean():.3f} (win {win}/{len(Nd)}, p {p_hd:.2g}) | "
              f"cost {method_fwd}fwd ({cost['hidden_ablation_fwd_seq']}seq+{cost['guard_fwd_batched']}batch) "
              f"{np.mean(mtimes):.2f}s/img ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    print("\n=== cross-model necessity: PERFORMANCE (hdel, lower=better) ===")
    print(f"{'model':12} {'N':>4} {'head':>5} {'loc_hdel':>9} {'nec_hdel':>9} {'win':>8} {'p_hdel':>8} {'nec_hins':>9}")
    for mk, rr in report.items():
        print(f"{mk:12} {rr['N']:>4} {str(rr['has_head']):>5} "
              f"{rr['loc_hdel']:>9.3f} {rr['nec_hdel']:>9.3f} {str(rr['win_del'])+'/'+str(rr['n']):>8} "
              f"{rr['p_hdel']:>8.2g} {rr['nec_hins']:>9.3f}")
    print("\n=== COST (method only; eval/hard_curves excluded) ===")
    print(f"{'model':12} {'fwd_total':>9} {'seq(hidabl)':>11} {'batched':>8} {'s/img':>7}")
    for mk, rr in report.items():
        cb = rr["cost_breakdown"]
        print(f"{mk:12} {rr['cost_method_fwd']:>9} {cb['hidden_ablation_fwd_seq']:>11} "
              f"{cb['guard_fwd_batched']+cb['localization_fwd']+cb['last_hidden_fwd']:>8} "
              f"{rr['method_walltime_s_per_img']:>7.2f}")
    args.out.write_text(json.dumps(report, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
