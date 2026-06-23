#!/usr/bin/env python3
"""Cheap SELF-CONTAINED necessity-ERF: can input->output (or input-gradient)
fingerprints recover the redundant MODULES that last-hidden clustering finds,
WITHOUT using the hidden layer (so it works when tokens don't 1-1 map to inputs,
e.g. SAM v2) and WITHOUT any external saliency?

Per-patch fingerprints (cluster each -> modules; delete whole modules):
  hidden   (REF, known-good)  : last-hidden patch token            [uses hidden]
  content  (REF, known-bad)   : input embedding h_diff             [pixels]
  gradtgt  (self-contained)   : d target / d patch-embedding (1 bwd)
  delfeat  (self-contained)   : feat(full) - feat(del i)  (N fwd)
  delpart  (self-contained)   : avg over random PARTIAL bg of
                                 feat(b) - feat(b\\i)  (depletes redundancy)

For each fingerprint: super-additivity (module signature), and hard-deletion
(module order ranked by WHOLE-GROUP deletion drop = self-contained necessity,
within-cluster by single-deletion effect). Compare to inflow. Module agreement
with the hidden clustering (does the self-contained one find the same modules?).
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
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


def fp_gradtgt(r):
    """d target / d patch-embedding, per patch -> [N,C]. 1 backward."""
    m = r.model
    h_var = r.h_b0.detach().clone().requires_grad_(True)
    h_inj = torch.cat([r.prefix, h_var], dim=1)
    store = {}
    hk = m.blocks[-1].register_forward_hook(
        lambda mod, a, o: store.__setitem__("h", (o if torch.is_tensor(o) else o[0])))
    holder = [h_inj]
    pre = m.blocks[0].register_forward_pre_hook(lambda mod, a: (holder[0],))
    try:
        out = m(r.x)
    finally:
        pre.remove(); hk.remove()
    if r.has_head:
        obj = torch.softmax(out[0], 0)[r.target]
    elif getattr(r, "use_probe", False):
        lh = store["h"]; nrm = (m.norm(lh) if hasattr(m, "norm") else lh)
        pooled = nrm[:, 0] if r.npref > 0 else nrm[:, r.npref:].mean(1)   # match XRunner pooling
        obj = torch.softmax(r._probe_logits(pooled), -1)[0, r.target]
    else:
        lh = store["h"]; pooled = (m.norm(lh) if hasattr(m, "norm") else lh)[:, r.npref:].mean(1)
        pn = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        obj = (pn @ r.full_feat.squeeze()).squeeze()
    obj.backward()
    return h_var.grad[0].float().cpu().numpy()


def feat_for_masks(r, masks_np, chunk=64):
    """pooled feature per mask -> [B,C] (no grad)."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(masks_np), chunk):
            mb = torch.as_tensor(masks_np[s:s + chunk], device=r.dev, dtype=r.dtype)
            feat, _ = r._fwd(mb)
            outs.append(feat.float().cpu())
    return torch.cat(outs, 0).numpy()


def fp_delfeat(r):
    """feat(full) - feat(del i) per patch -> [N,C]. N fwd."""
    N = r.N
    full = feat_for_masks(r, np.ones((1, N), np.float32))[0]
    delm = np.ones((N, N), np.float32); delm[np.arange(N), np.arange(N)] = 0.0
    df = full[None] - feat_for_masks(r, delm)
    return df


def fp_delpart(r, n_bg=4, rho=0.5, seed=3):
    """avg over random partial bg of feat(b) - feat(b without i) -> [N,C]."""
    N = r.N
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, r.full_feat.shape[-1]), np.float32)
    for _ in range(n_bg):
        b = (rng.random(N) < rho).astype(np.float32)
        b[:] = np.maximum(b, 0.0)
        base = feat_for_masks(r, b[None])[0]
        masks = np.repeat(b[None], N, 0)
        masks[np.arange(N), np.arange(N)] = 0.0          # remove patch i from bg
        df = base[None] - feat_for_masks(r, masks)        # [N,C]
        # only meaningful where i was in bg; zero else
        df = df * b[:, None]
        acc += df
    return acc / n_bg


def group_del_crank(r, labels, k, full):
    """crank[c] = drop when whole cluster c deleted. k fwd. Self-contained."""
    N = r.N
    masks = []
    for c in range(k):
        mm = np.ones(N, np.float32); mm[labels == c] = 0.0
        masks.append(mm)
    vals = r.target_curve(np.stack(masks))
    return np.maximum(full - vals, 0.0)


def module_hdel(r, fp, single_del, full, k=16):
    """cluster fp -> modules; order by group-deletion drop, within by single-del."""
    labels = nec.kml(fp, k)
    crank = group_del_crank(r, labels, k, full)
    order = nec.module_order(labels, single_del, crank)
    score = nec.rankscore(order, r.N)
    hi, hd = nec.hard_curves(r, score)
    return hd, hi, labels


def cluster_agreement(la, lb, k=16):
    """fraction of patch-pairs whose same/diff-cluster status agrees."""
    N = len(la)
    sa = (la[:, None] == la[None, :]); sb = (lb[:, None] == lb[None, :])
    return float((sa == sb).mean())


def guarded_sc(r, fp, single_del, full, p_full, p_base, ks=(8, 12, 16, 24)):
    """SELF-CONTAINED guarded necessity: cluster fp at multiple k, order each by
    group-deletion drop (crank) + single-del within, plus a single-del-only
    candidate; guard picks best by probed deletion curve. NO hidden, NO saliency."""
    N = r.N
    cands = {"single": nec.rankscore(np.argsort(-single_del), N)}
    for k in ks:
        lab = nec.kml(fp, k)
        crank = group_del_crank(r, lab, k, full)
        cands[f"mod{k}"] = nec.rankscore(nec.module_order(lab, single_del, crank), N)
    best, bauc = None, 1e9
    for nm, sc in cands.items():
        auc = nec.clip_auc(r, np.argsort(-sc), p_full, p_base)
        if auc < bauc:
            bauc, best = auc, nm
    return cands[best], best


def guarded_cheap(r, gt, p_full, p_base, ks=(8, 12, 16, 24)):
    """ULTRA-CHEAP self-contained: gradtgt ONLY (1 bwd). within-rank = gradient
    magnitude (NO N single-deletions). crank = group-deletion drop. + guard.
    Cost = 1 bwd + Sk crank probes + guard (no per-patch single-del)."""
    N = r.N
    within = np.linalg.norm(gt, axis=1)
    cands = {"within": nec.rankscore(np.argsort(-within), N)}
    for k in ks:
        lab = nec.kml(gt, k)
        crank = group_del_crank(r, lab, k, p_full)
        cands[f"mod{k}"] = nec.rankscore(nec.module_order(lab, within, crank), N)
    best, bauc = None, 1e9
    for nm, sc in cands.items():
        auc = nec.clip_auc(r, np.argsort(-sc), p_full, p_base)
        if auc < bauc:
            bauc, best = auc, nm
    return cands[best], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["clip"])
    ap.add_argument("--nimg", type=int, default=8)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--guarded", action="store_true", help="self-contained GUARDED necessity (delfeat/gradtgt + guard) vs inflow")
    ap.add_argument("--cheap", action="store_true", help="ULTRA-cheap gradtgt-only (1 bwd, no N single-del); requires --guarded")
    ap.add_argument("--baseline", choices=("pos", "mean"), default="pos", help="deletion baseline (mean fixes dinov2 OOD)")
    ap.add_argument("--probe-dir", type=Path, default=None, help="dir with probe_{model}.pt (sharp class-prob target for headless)")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/necessity_erf.json")
    args = ap.parse_args()
    import timm

    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(123).shuffle(paths)
    imgs = paths[: args.nimg]
    FPS = ["hidden", "content", "gradtgt", "delfeat", "delpart"]
    report = {}
    for mk in args.models:
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in MODELS[mk] else {}
        model = timm.create_model(MODELS[mk], pretrained=True, **kw).eval().to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        probe = None
        if args.probe_dir is not None:
            pp = args.probe_dir / f"probe_{mk}.pt"
            if pp.exists():
                probe = torch.load(pp, map_location=args.device)
                print(f"[{mk}] loaded probe head (acc {probe.get('acc', float('nan')):.3f})", flush=True)
        acc = {f: {"hdel": [], "superadd": [], "agree_hidden": []} for f in FPS}
        accg = {"delfeat_g": [], "gradtgt_g": [], "picks": []}
        infl_hdel = []
        for ip in imgs:
            try:
                x, _ = xm.load_image(ip, norm=xm.MODEL_NORM.get(mk, "clip")); x = x.to(args.device)
                r = xm.XRunner(model, x, args.device, baseline=args.baseline, probe=probe)
                N = r.N
                if args.guarded and args.cheap:
                    p_full = float(r.target_curve(np.ones((1, N), np.float32))[0])
                    p_base = float(r.target_curve(np.zeros((1, N), np.float32))[0])
                    gt = fp_gradtgt(r)
                    sc, pk = guarded_cheap(r, gt, p_full, p_base)
                    hd = nec.hard_curves(r, sc)[1]
                    accg["gradtgt_g"].append(hd); accg["delfeat_g"].append(hd); accg["picks"].append((pk, pk))
                    if r.has_head:
                        infl = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                    else:
                        infl = nec.last_attn_localization(r).astype(np.float64)
                    infl_hdel.append(nec.hard_curves(r, infl)[1])
                    print(f"[{mk} {ip.name}] cheap_g:{hd:.3f} inflow:{infl_hdel[-1]:.3f} pick={pk}", flush=True)
                    model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    continue
                o_del, full = xm.single_del_oracle(r)
                if args.guarded:
                    p_full = float(r.target_curve(np.ones((1, N), np.float32))[0])
                    p_base = float(r.target_curve(np.zeros((1, N), np.float32))[0])
                    df = fp_delfeat(r); gt = fp_gradtgt(r)
                    sc_df, pk_df = guarded_sc(r, df, o_del, full, p_full, p_base)
                    sc_gt, pk_gt = guarded_sc(r, gt, o_del, full, p_full, p_base)
                    accg["delfeat_g"].append(nec.hard_curves(r, sc_df)[1])
                    accg["gradtgt_g"].append(nec.hard_curves(r, sc_gt)[1])
                    accg["picks"].append((pk_df, pk_gt))
                    if r.has_head:
                        infl = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                    else:
                        infl = nec.last_attn_localization(r).astype(np.float64)
                    infl_hdel.append(nec.hard_curves(r, infl)[1])
                    print(f"[{mk} {ip.name}] delfeat_g:{accg['delfeat_g'][-1]:.3f} "
                          f"gradtgt_g:{accg['gradtgt_g'][-1]:.3f} inflow:{infl_hdel[-1]:.3f} "
                          f"picks={pk_df},{pk_gt}", flush=True)
                    model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    continue
                fps = {
                    "hidden": r.last_hidden_patches(),
                    "content": r.h_diff[0].float().cpu().numpy(),
                    "gradtgt": fp_gradtgt(r),
                    "delfeat": fp_delfeat(r),
                    "delpart": fp_delpart(r),
                }
                lab_hidden = nec.kml(fps["hidden"], args.k)
                for f in FPS:
                    hd, hi, lab = module_hdel(r, fps[f], o_del, full, k=args.k)
                    acc[f]["hdel"].append(hd)
                    acc[f]["superadd"].append(xm.superadd(r, lab, o_del, full, args.k))
                    acc[f]["agree_hidden"].append(cluster_agreement(lab, lab_hidden))
                if r.has_head:
                    infl = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                else:
                    infl = nec.last_attn_localization(r).astype(np.float64)
                infl_hdel.append(nec.hard_curves(r, infl)[1])
                print(f"[{mk} {ip.name}] " + " ".join(
                    f"{f}:{acc[f]['hdel'][-1]:.3f}" for f in FPS) + f" inflow:{infl_hdel[-1]:.3f}", flush=True)
            except Exception as e:
                print(f"[{mk} {ip.name}] ERR {type(e).__name__}: {e}", flush=True)
            finally:
                model.zero_grad(set_to_none=True); torch.cuda.empty_cache()

        if args.guarded:
            from scipy.stats import wilcoxon as _wx
            dg = np.array(accg["delfeat_g"]); gg = np.array(accg["gradtgt_g"]); ia = np.array(infl_hdel)
            def pw(a):
                try:
                    return float(_wx(ia, a, alternative="greater").pvalue)
                except ValueError:
                    return float("nan")
            report[mk] = {"n": len(ia), "inflow_hdel": float(ia.mean()),
                          "delfeat_g_hdel": float(dg.mean()), "gradtgt_g_hdel": float(gg.mean()),
                          "delfeat_g_p": pw(dg), "gradtgt_g_p": pw(gg),
                          "delfeat_g_win": int((dg < ia).sum()), "gradtgt_g_win": int((gg < ia).sum()),
                          "delfeat_g_arr": dg.tolist(), "gradtgt_g_arr": gg.tolist(), "inflow_arr": ia.tolist()}
            r2 = report[mk]
            print(f"\n=== {mk}: SELF-CONTAINED GUARDED necessity vs inflow (n={r2['n']}) ===")
            print(f"  delfeat_g {r2['delfeat_g_hdel']:.3f} (win {r2['delfeat_g_win']}/{r2['n']}, p<inflow {r2['delfeat_g_p']:.2g})")
            print(f"  gradtgt_g {r2['gradtgt_g_hdel']:.3f} (win {r2['gradtgt_g_win']}/{r2['n']}, p<inflow {r2['gradtgt_g_p']:.2g})")
            print(f"  inflow    {r2['inflow_hdel']:.3f}")
            del model; torch.cuda.empty_cache(); continue

        def md(v):
            v = [x for x in v if np.isfinite(x)]; return float(np.median(v)) if v else float("nan")
        from scipy.stats import wilcoxon
        rep = {"inflow_hdel": float(np.mean(infl_hdel)), "n": len(infl_hdel), "fps": {}}
        ia = np.array(infl_hdel)
        for f in FPS:
            fa = np.array(acc[f]["hdel"])
            try:
                p_inf = float(wilcoxon(ia, fa, alternative="greater").pvalue)  # fp < inflow?
            except ValueError:
                p_inf = float("nan")
            rep["fps"][f] = {"hdel": float(np.mean(fa)),
                             "superadd": md(acc[f]["superadd"]),
                             "agree_hidden": float(np.mean(acc[f]["agree_hidden"])),
                             "p_beats_inflow": p_inf, "win_vs_inflow": int((fa < ia).sum()),
                             "hdel_arr": fa.tolist()}
        report[mk] = rep
        print(f"\n=== {mk}: self-contained necessity-ERF (n={rep['n']}) ===")
        print(f"{'fingerprint':10} {'hdel':>7} {'super-add':>10} {'agree_h':>8} {'p<inflow':>9} {'win':>6}  (lower hdel better)")
        for f in FPS:
            d = rep["fps"][f]
            print(f"{f:10} {d['hdel']:>7.3f} {d['superadd']:>+10.2f} {d['agree_hidden']:>8.2f} "
                  f"{d['p_beats_inflow']:>9.2g} {str(d['win_vs_inflow'])+'/'+str(rep['n']):>6}")
        print(f"{'inflow':10} {rep['inflow_hdel']:>7.3f}")
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
