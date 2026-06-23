#!/usr/bin/env python3
"""Test the user's hypothesis: for REDUNDANT (part) features, the feature's own
ACTIVATION map is the NECESSITY set -> deleting patches in activation order kills
the feature FASTER than deleting in FRI (sufficiency) order. For CONCENTRATED
(whole-object / low-level / bias) features, activation ~= FRI (coincide).

Per (feature, top-firing token), fair same-objective comparison using the
established autolabel_eval feature-recovery machinery (mean-baseline =
global_mean_h_plus_pos). Orders compared:
  - activation : feature_activation_map (where the feature fires)  [necessity?]
  - fri        : cautious_feature_erf prob_scores                  [sufficiency]
  - grad       : input_x_grad_feature_erf                          [baseline]
  - random
Metrics: hdel (deletion AUC, LOWER=faster kill=better necessity), hins (insertion
AUC, HIGHER=better sufficiency). Prediction: redundant -> hdel_act << hdel_fri;
concentrated -> hdel_act ~= hdel_fri. The GAP (hdel_fri - hdel_act) = "part-ness".
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from autolabel_eval.config import EvalConfig
from autolabel_eval.legacy import LegacyRuntime

VAL_ROOT = Path("/media/sangyu/Dataset/imagenet/val")
SRC_MANIFEST = ROOT / "outputs/class_fri/research_frontier/redundant_feature_panels/manifest.json"
OUT_DIR = ROOT / "outputs/class_fri/research_frontier"
BLOCK = 10
FRACS = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0], dtype=np.float32)


def build_runtime(device: str) -> LegacyRuntime:
    return LegacyRuntime(replace(
        EvalConfig(), device=device,
        model_name="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
        image_size=224, resize_size=256, grid_size=14, n_patches=196,
        erf_recovery_threshold=0.9, erf_support_min_normalized_attribution=0.1,
    ))


def make_objective(runtime: LegacyRuntime, image: str, token: int, feat: int):
    """Returns obj(mask[N], 1=keep) -> feature-activation recovery in ~[0,1]
    (mean-baseline), plus the feature's per-patch activation map [N]."""
    art = runtime.forward_block(image, BLOCK)
    sae = runtime.load_sae(BLOCK)
    prefix = runtime.adapter.prefix_count()
    dev = art.capture.patch_tokens.device
    dtype = art.capture.patch_tokens.dtype
    N = int(runtime.config.n_patches)
    with torch.no_grad():
        full_acts = sae(art.patch_out).get("feature_acts")
        full_activation = full_acts[0, int(token), int(feat)].detach()
        act_map = full_acts[0, :, int(feat)].detach().float().cpu().numpy()
    do_fwd, get_out = runtime.adapter.make_masked_forward(art.x, art.capture, block_idx=BLOCK)
    with torch.no_grad():
        do_fwd(torch.zeros(N, device=dev, dtype=dtype))
        base_patch = get_out()[:, prefix:, :]
        base_activation = sae(base_patch).get("feature_acts")[0, int(token), int(feat)].detach()

    def obj(mask_np: np.ndarray) -> float:
        with torch.no_grad():
            mask = torch.as_tensor(mask_np, device=dev, dtype=dtype)
            do_fwd(mask)
            patch = get_out()[:, prefix:, :]
            rec = runtime._feature_activation_recovery_objective(
                patch, sae=sae, token_idx=int(token), feature_id=int(feat),
                full_activation=full_activation, baseline_activation=base_activation)
            return float(rec.item())
    return obj, act_map, N


def ring_indices(token: int, R: int, G: int = 14) -> set:
    """self-patch + (2R+1)x(2R+1) neighborhood on the GxG grid. R<0 -> empty."""
    if R < 0:
        return set()
    r0, c0 = divmod(int(token), G)
    out = set()
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < G and 0 <= c < G:
                out.add(r * G + c)
    return out


def deletion_curve(obj, order: np.ndarray, N: int, keep: set = frozenset()) -> tuple[np.ndarray, float]:
    order = [i for i in order if int(i) not in keep]
    base = np.ones(N, dtype=np.float32)  # forced-keep stay 1
    rec = []
    for fr in FRACS:
        k = int(round(fr * len(order)))
        mask = base.copy()
        if k > 0:
            mask[order[:k]] = 0.0
        rec.append(obj(mask))
    rec = np.array(rec, dtype=np.float32)
    hdel = float(np.trapz(rec, FRACS))   # lower = faster kill = better necessity order
    return rec, hdel


def insertion_curve(obj, order: np.ndarray, N: int, keep: set = frozenset()) -> tuple[np.ndarray, float]:
    order = [i for i in order if int(i) not in keep]
    base = np.zeros(N, dtype=np.float32)
    for i in keep:
        base[int(i)] = 1.0  # forced-keep always inserted
    rec = []
    for fr in FRACS:
        k = int(round(fr * len(order)))
        mask = base.copy()
        if k > 0:
            mask[order[:k]] = 1.0
        rec.append(obj(mask))
    rec = np.array(rec, dtype=np.float32)
    hins = float(np.trapz(rec, FRACS))   # higher = better sufficiency order
    return rec, hins


def _img_path(row: dict) -> Path:
    return VAL_ROOT / row["class_dir"] / row["image"]


def orders_for(runtime, image, token, feat, act_map, N, rng):
    out = {"activation": np.argsort(-act_map, kind="mergesort")}
    try:
        erf = runtime.cautious_feature_erf(image, BLOCK, token, feat)
        out["fri"] = np.argsort(-np.asarray(erf["prob_scores"], dtype=np.float32), kind="mergesort")
    except Exception as e:
        print(f"   fri failed: {e}", flush=True)
    try:
        g = runtime.input_x_grad_feature_erf(image, BLOCK, token, feat)
        if isinstance(g, dict):
            g = g.get("prob_scores", g.get("scores"))
        out["grad"] = np.argsort(-np.asarray(g, dtype=np.float32), kind="mergesort")
    except Exception as e:
        print(f"   grad failed: {e}", flush=True)
    out["random"] = rng.permutation(N)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nimg", type=int, default=3)
    ap.add_argument("--max-feats", type=int, default=0)
    ap.add_argument("--viz-feats", type=int, nargs="*", default=[5148, 6455, 6530, 8373, 10001, 1183])
    ap.add_argument("--keep-self-ring", type=int, default=-1,
                    help="-1=delete everything (default); 0=forbid deleting self-patch; 1=forbid self+8 neighbors")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "activation_necessity_deletion.json")
    args = ap.parse_args()
    if args.keep_self_ring >= 0 and "activation_necessity_deletion.json" in str(args.out):
        args.out = OUT_DIR / f"activation_necessity_deletion_keepself{args.keep_self_ring}.json"
    man = json.loads(SRC_MANIFEST.read_text())
    runtime = build_runtime(args.device)
    rng = np.random.default_rng(0)
    viz = {}
    results = {"redundant": [], "concentrated": []}
    methods = ["activation", "fri", "grad", "random"]
    for group in ("redundant", "concentrated"):
        feats = man[group]
        if args.max_feats > 0:
            feats = feats[: args.max_feats]
        for fr_i, feat_rec in enumerate(feats):
            feat = int(feat_rec["feature_id"])
            per_img = {m: {"hdel": [], "hins": []} for m in methods}
            curves_for_viz = None
            for row in feat_rec["rows"][: args.nimg]:
                ip = _img_path(row)
                if not ip.exists():
                    continue
                token = int(row.get("argmax_patch", 0))
                try:
                    obj, act_map, N = make_objective(runtime, str(ip), token, feat)
                except Exception as e:
                    print(f"[{group}] feat {feat} obj failed: {e}", flush=True)
                    continue
                ords = orders_for(runtime, str(ip), token, feat, act_map, N, rng)
                keep = ring_indices(token, args.keep_self_ring)
                row_curves = {}
                for m in methods:
                    if m not in ords:
                        continue
                    drec, hdel = deletion_curve(obj, ords[m], N, keep)
                    irec, hins = insertion_curve(obj, ords[m], N, keep)
                    per_img[m]["hdel"].append(hdel)
                    per_img[m]["hins"].append(hins)
                    row_curves[m] = {"del": drec.tolist(), "ins": irec.tolist(), "hdel": hdel, "hins": hins}
                if curves_for_viz is None and feat in args.viz_feats:
                    curves_for_viz = {"image": str(ip), "token": token, "act_map": act_map.tolist(),
                                      "orders": {m: ords[m][:60].tolist() for m in ords}, "curves": row_curves}
            agg = {m: {"hdel": float(np.mean(per_img[m]["hdel"])) if per_img[m]["hdel"] else float("nan"),
                       "hins": float(np.mean(per_img[m]["hins"])) if per_img[m]["hins"] else float("nan")}
                   for m in methods}
            rec = {"feature_id": feat, "group": group, "oracle_sat": feat_rec.get("oracle_sat"),
                   "act_max": feat_rec.get("act_max"), "spatial_frac": feat_rec.get("spatial_frac"),
                   "agg": agg, "gap_fri_minus_act": agg["fri"]["hdel"] - agg["activation"]["hdel"]}
            results[group].append(rec)
            if curves_for_viz is not None:
                viz[feat] = {**curves_for_viz, "rec": rec}
            print(f"[{group} {fr_i+1}/{len(feats)}] feat {feat} sat={feat_rec.get('oracle_sat'):.3f} | "
                  f"hdel act={agg['activation']['hdel']:.3f} fri={agg['fri']['hdel']:.3f} "
                  f"grad={agg.get('grad',{}).get('hdel',float('nan')):.3f} rnd={agg['random']['hdel']:.3f} | "
                  f"gap(fri-act)={rec['gap_fri_minus_act']:+.3f}", flush=True)
    args.out.write_text(json.dumps({"results": results, "viz": viz, "fracs": FRACS.tolist()}, indent=1))
    # summary
    print("\n=== SUMMARY: mean hdel (lower=better deletion) / hins (higher=better sufficiency) ===")
    for group in ("redundant", "concentrated"):
        rs = results[group]
        if not rs:
            continue
        print(f"\n-- {group} (n={len(rs)}) --")
        for m in methods:
            hd = np.nanmean([r["agg"][m]["hdel"] for r in rs])
            hi = np.nanmean([r["agg"][m]["hins"] for r in rs])
            print(f"   {m:11s} hdel={hd:.3f}  hins={hi:.3f}")
        gaps = np.array([r["gap_fri_minus_act"] for r in rs])
        wins = int(np.sum(gaps > 0))
        print(f"   gap(fri_hdel - act_hdel): mean {gaps.mean():+.3f}  median {np.median(gaps):+.3f}  "
              f"act-better-deletion in {wins}/{len(rs)}")
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
