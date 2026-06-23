#!/usr/bin/env python3
"""HG-FRI n100 validation on softplus_failure_scan_n100.csv cases.

Method (68 states/img): 30-step tv-free fri_alt_prob solve (60) + irr (2) +
full/base (2) + dense-state h-grad harvest (free) + group value-tests (<=4).
Readouts: bn1_vg / bn15_vgp / bn2_vgp (+ final, bn1 ablations).
Compares per-case vs CSV inflow/soft references.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec5 = _ilu.spec_from_file_location("hg5", REPO / "scripts/research_hgrad_v5.py")
hg5 = _ilu.module_from_spec(_spec5)
_spec5.loader.exec_module(hg5)
_spec6 = _ilu.spec_from_file_location("hg6", REPO / "scripts/research_hgrad_v6.py")
hg6 = _ilu.module_from_spec(_spec6)
_spec6.loader.exec_module(hg6)

N = hg5.N
_n01 = hg5._n01
CSV = REPO / "outputs/class_fri/softplus_failure_scan_n100.csv"
ZEBRA = REPO / "multi_object_zebra_elephant.jpg"


def make_smoother(runner, knn=8, temp=0.15, alpha=0.5):
    with torch.no_grad():
        h = runner.h_b0[0]
        hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = (hn @ hn.T).float()
        K = torch.exp((S - 1.0) / temp)
        K.fill_diagonal_(0.0)
        vals, idx = torch.topk(K, k=knn, dim=1)
        Ks = torch.zeros_like(K)
        Ks.scatter_(1, idx, vals)
        Ks = 0.5 * (Ks + Ks.T)
        Ks = Ks / Ks.sum(1, keepdim=True).clamp(min=1e-8)
        Knp = Ks.cpu().numpy()

    def smooth(v):
        v = _n01(np.asarray(v, np.float64))
        return (1 - alpha) * v + alpha * (Knp @ v)

    return smooth


def make_kmat(runner, src="pos", temp=0.05, knn=8, content_temp=0.3):
    with torch.no_grad():
        def cos_sim(emb):
            hn = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return (hn @ hn.T).float()

        if src == "pos":
            K = torch.exp((cos_sim(runner.base[0]) - 1.0) / temp)
        elif src == "h":
            K = torch.exp((cos_sim(runner.h_b0[0]) - 1.0) / temp)
        elif src == "bilateral":
            # near in position AND similar in content (edge-preserving)
            K = torch.exp((cos_sim(runner.base[0]) - 1.0) / temp) \
                * torch.exp((cos_sim(runner.h_diff[0]) - 1.0) / content_temp)
        else:
            raise ValueError(src)
        K.fill_diagonal_(0.0)
        vals, idx = torch.topk(K, k=knn, dim=1)
        Ks = torch.zeros_like(K)
        Ks.scatter_(1, idx, vals)
        Ks = 0.5 * (Ks + Ks.T)
        Ks = Ks * (364.0 / Ks.sum().clamp(min=1e-8))
    return Ks.to(runner.dtype)


def hgfri_scores(runner, *, solve_steps=29, lift_k=32, eps=0.6, seed=42,
                 budget_mode="iid", ktv="none", ktv_weight=0.01, irr_weight=0.05,
                 n_solves=1):
    """Returns dict of readout scores + diagnostics. <=68 states total."""
    if irr_weight > 0:
        irr, hg_full, ixg_full = hg5.make_irr_and_hgfull(runner)      # 2 states
    else:
        irr = torch.zeros(hg5.N, device=runner.dev, dtype=runner.dtype)  # 0 states
    kmat = make_kmat(runner, src=ktv) if ktv in ("pos", "h", "bilateral") else None
    if n_solves <= 1:
        tr = hg5.solve_collect(runner, steps=solve_steps, seed=seed, irr=irr,
                               irr_weight=irr_weight,
                               budget_mode=budget_mode, kmat=kmat,
                               ktv_weight=(ktv_weight if kmat is not None else 0.0))
    else:
        # ensemble of short solves: averaged mask, pooled harvest/ds
        steps_each = solve_steps // n_solves
        trs = [hg5.solve_collect(runner, steps=steps_each, seed=seed + 1000 * i,
                                 irr=irr, irr_weight=irr_weight,
                                 budget_mode=budget_mode, kmat=kmat,
                                 ktv_weight=(ktv_weight if kmat is not None else 0.0))
               for i in range(n_solves)]
        tr = {
            "final": np.mean([t["final"] for t in trs], axis=0),
            "la": np.mean([t["la"] for t in trs], axis=0),
            "ds": np.sum([t["ds"] for t in trs], axis=0),
            "hg": np.concatenate([t["hg"] for t in trs], axis=0),
            "ixg": np.concatenate([t["ixg"] for t in trs], axis=0),
            "dens": np.concatenate([t["dens"] for t in trs], axis=0),
            "phase": np.concatenate([t["phase"] for t in trs], axis=0),
        }
    final = tr["final"]
    dense_sel = tr["dens"] >= 0.7
    hg = tr["hg"][dense_sel]
    hgd = np.zeros(N)
    for r_ in hg:
        hgd += r_ / (r_.sum() + 1e-12)

    lift = _n01(hgd) * (1.0 - _n01(final))
    order = np.argsort(-lift)
    G = order[:lift_k]
    dev, dtype = runner.dev, runner.dtype
    with torch.no_grad():
        p_full = float(runner.probs_for_masks(
            torch.ones(1, N, device=dev, dtype=dtype))[0].item())
        m = np.ones(N, np.float32)
        m[G] = 0.0
        p_del = float(runner.probs_for_masks(
            torch.as_tensor(m, device=dev, dtype=dtype).unsqueeze(0))[0].item())
    r0 = p_del / max(p_full, 1e-8)
    demote = np.zeros(N, bool)
    glog = {"tests": [{"n": len(G), "ratio": r0}], "branch": "none"}
    if r0 > 1.0 + eps:  # strong rise: competitor -> demote + refine + iter2
        glog["branch"] = "competitor"
        demote[G] = True
        d2, glog2 = hg6.group_sign_test(
            runner, np.where(demote, 0.0, lift), k=12, eps=eps, max_tests=1)
        demote = demote | d2
        glog["tests"] += glog2["tests"]
    elif r0 > 1.15:  # ambiguous band: suff test disambiguates anti-evidence
        m2 = np.zeros(N, np.float32)
        m2[G] = 1.0
        with torch.no_grad():
            p_alone = float(runner.probs_for_masks(
                torch.as_tensor(m2, device=dev, dtype=dtype).unsqueeze(0))[0].item())
        suff = p_alone / max(p_full, 1e-8)
        glog["tests"].append({"n": len(G), "suff": suff})
        if suff < 0.15:
            glog["branch"] = "anti-evidence"
            demote[G] = True
        else:
            glog["branch"] = "evidence-kept"
    hgd_gated = hgd * np.where(demote, 0.05, 1.0)
    pen = demote.astype(float)

    # guided frontier probe (post-gate): backward at full-minus-(gated lift top-32)
    lift_g = _n01(hgd_gated) * (1.0 - _n01(final))
    Gf = np.argsort(-lift_g)[:lift_k]
    mf = np.ones(N, np.float32)
    mf[Gf] = 0.0
    state_w = torch.as_tensor(mf, device=dev, dtype=dtype)
    h_probe = torch.zeros_like(runner.h_diff, requires_grad=True)
    h_mix = runner.base + state_w.view(1, N, 1) * runner.h_diff + h_probe
    h_inj = torch.cat([runner.cls, h_mix], dim=1)
    holder = [h_inj]
    hook = runner.model.blocks[0].register_forward_pre_hook(lambda m, a: (holder[0],))
    try:
        out = runner.model(runner.x)
    finally:
        hook.remove()
    act = torch.softmax(out[0], dim=0)[runner.target]
    act.backward()
    gfr = h_probe.grad.detach()[0].norm(dim=-1).cpu().numpy()
    hgd_fr = hgd_gated + 2.0 * (gfr / (gfr.sum() + 1e-12)) * np.where(demote, 0.05, 1.0)

    # adaptive head weight from the redundancy probe:
    #   r0 ~ 1   -> complement of lift recovers fully -> redundant -> hg matters
    #   r0 < 0.9 -> lift removal broke prob -> mask already has the core
    lam_ad = 0.7 if 0.92 <= r0 <= 1.08 else (1.5 if r0 < 0.92 else 1.0)
    # robust median pooling variant
    if hg.shape[0]:
        hg_nrm_rows = hg / (hg.sum(1, keepdims=True) + 1e-12)
        hgd_med = np.median(hg_nrm_rows, axis=0) * hg.shape[0]
    else:
        hgd_med = hgd
    hgd_med_gated = hgd_med * np.where(demote, 0.05, 1.0)
    ds = tr["ds"]
    ds_gated = _n01(ds) * np.where(demote, 0.05, 1.0)
    reads = {
        "final": final,
        "bn_ad_fr": lam_ad * _n01(final) + _n01(hgd_fr) - lam_ad * pen,
        "bn_ad_fr_ds03": lam_ad * _n01(final) + _n01(hgd_fr) + 0.3 * ds_gated - (lam_ad + 0.3) * pen,
        "bn_ad_fr_ds05": lam_ad * _n01(final) + _n01(hgd_fr) + 0.5 * ds_gated - (lam_ad + 0.5) * pen,
        "bn_ad_fr_ds08": lam_ad * _n01(final) + _n01(hgd_fr) + 0.8 * ds_gated - (lam_ad + 0.8) * pen,
        "bn1_fr_ds05": _n01(final) + _n01(hgd_fr) + 0.5 * ds_gated - 1.5 * pen,
    }
    return reads, {"glog": glog, "n_demoted": int(demote.sum()),
                   "n_dense": int(dense_sel.sum()), "r0": r0, "lam_ad": lam_ad,
                   "branch": glog.get("branch")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/peel_research/hgfri_n100.json")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-mode", default="iid")
    ap.add_argument("--ktv", default="none", choices=("none", "pos", "h", "bilateral"))
    ap.add_argument("--ktv-weight", type=float, default=0.01)
    ap.add_argument("--irr-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-solves", type=int, default=1)
    ap.add_argument("--solve-steps", type=int, default=29)
    args = ap.parse_args()
    import timm
    syms = hg5.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    df = pd.read_csv(CSV)
    if args.limit:
        df = df.head(args.limit)
    results = {}
    run_means = {k: [] for k in ("final", "bn_ad_fr", "bn_ad_fr_ds03", "bn_ad_fr_ds05", "bn_ad_fr_ds08", "bn1_fr_ds05")}
    run_means_ins = {k: [] for k in run_means}
    t0 = time.time()
    scores_dump = {}
    for i, row in df.iterrows():
        name = row["image"].replace(".JPEG", "")
        x, _ = syms["load_image"](Path(row["path"]))
        x = x.to(args.device)
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = hg5.Block0Runner(model, x, h_b0, cls_tok, base, int(row["target"]))
        reads, diag = hgfri_scores(runner, budget_mode=str(args.budget_mode),
                                   ktv=str(args.ktv), ktv_weight=float(args.ktv_weight),
                                   irr_weight=float(args.irr_weight), seed=int(args.seed),
                                   n_solves=int(args.n_solves),
                                   solve_steps=int(args.solve_steps))
        entry = {"csv": {"inflow_hdel": float(row["inflow_hdel"]),
                         "inflow_hins": float(row["inflow_hins"]),
                         "soft_hdel": float(row["soft_hdel"]),
                         "soft_hins": float(row["soft_hins"]),
                         "desc": str(row["desc"]), "prob": float(row["prob"])},
                 "diag": {"n_demoted": diag["n_demoted"], "r0": diag["r0"],
                          "branch": diag.get("branch"),
                          "ratios": [t.get("ratio", t.get("suff")) for t in diag["glog"]["tests"]]},
                 "methods": {}}
        for rname, sc in reads.items():
            hins, hdel, _, _ = hg5.hard_curves(runner, sc, chunk=args.chunk)
            entry["methods"][rname] = {"hins": hins, "hdel": hdel}
            run_means[rname].append(hdel)
            run_means_ins[rname].append(hins)
        scores_dump[name] = {k: np.asarray(v, np.float32) for k, v in reads.items()}
        results[name] = entry
        if (len(run_means["final"]) % 10) == 0 or len(run_means["final"]) == len(df):
            n = len(run_means["final"])
            infl = df["inflow_hdel"].head(n).mean()
            line = f"[{n:3d}/{len(df)}] inflow_hdel={infl:.4f}"
            for k in ("bn_ad_fr", "bn_ad_fr_ds05", "bn1_fr_ds05"):
                line += f" | {k} {np.mean(run_means[k]):.4f}/{np.mean(run_means_ins[k]):.4f}"
            print(line + f" ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    np.savez_compressed(args.out.with_suffix(".scores.npz"),
                        **{f"{n}_{k}": v for n, d in scores_dump.items() for k, v in d.items()})

    print("\n=== n100 summary (hdel / hins, wins vs inflow) ===")
    infl_hdel = df["inflow_hdel"].values[: len(results)]
    infl_hins = df["inflow_hins"].values[: len(results)]
    soft_hdel = df["soft_hdel"].values[: len(results)]
    for k in run_means:
        hd = np.asarray(run_means[k])
        hi = np.asarray(run_means_ins[k])
        wins_d = int((hd < infl_hdel).sum())
        wins_i = int((hi > infl_hins).sum())
        wins_b = int(((hd < infl_hdel) & (hi > infl_hins)).sum())
        wins_s = int((hd < soft_hdel).sum())
        print(f"  {k:<10} hdel={hd.mean():.4f} hins={hi.mean():.4f} "
              f"| win_del={wins_d} win_ins={wins_i} win_both={wins_b} win_vs_soft={wins_s}")
    print(f"  inflow     hdel={infl_hdel.mean():.4f} hins={infl_hins.mean():.4f}")
    print(f"  soft(CSV)  hdel={soft_hdel.mean():.4f} hins={df['soft_hins'].values[:len(results)].mean():.4f}")
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
