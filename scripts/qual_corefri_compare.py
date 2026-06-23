#!/usr/bin/env python3
"""Side-by-side: archive CoreFRI (blob & kernel modes) vs NSD/SegCoreFRI.

Cols: image | inflow | CoreFRI-blob (vision prior, ~4.2k states) |
CoreFRI-kernel (token-similarity fields, prior-free, ~4.2k states) |
SegCoreFRI (segments, ~190 probes) | NSD chosen (guarded final).
Signed maps in RdBu (red=evidence, blue=anti); hdel in titles.
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

_spec = _ilu.spec_from_file_location("sg", REPO / "scripts/research_segcorefri.py")
sg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(sg)
nsr, ns, q = sg.nsr, sg.ns, sg.q

_cspec = _ilu.spec_from_file_location("cf", ARCHIVE / "scripts/research_corefri.py")
cf = _ilu.module_from_spec(_cspec)
_cspec.loader.exec_module(cf)

N, GRID = 196, 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/qual_corefri_compare.png")
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

    def draw_jet(ax, img, sc, title, gamma=1.0):
        ax.imshow(img)
        ax.imshow(up(ns._n01(sc) ** gamma), cmap="jet", alpha=0.55, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)

    rows = {}
    ncol = 6
    fig, axes = plt.subplots(len(ns.CASES), ncol,
                             figsize=(3.1 * ncol, 3.3 * len(ns.CASES)))
    t0 = time.time()
    for ri, (name, path, target) in enumerate(ns.CASES):
        metrics, final_scores, diag = nsr.run_case(
            model, syms, path, target, int(args.seed), args.device)
        x, _ = syms["load_image"](Path(path))
        x = x.to(args.device)
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)
        img = (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        p_full, p_base = diag["p_full"], diag["p_base"]

        cb = cf.corefri_scores(runner, seed=int(args.seed), mask_mode="blob")["scores"]
        ck = cf.corefri_scores(runner, seed=int(args.seed), mask_mode="kernel")["scores"]

        pc = ns.ProbeCounter(runner)
        segs, J, seg_J, sd = ns.synergy_segments(
            pc, runner, final_scores, top=64, p_full=p_full, seed=int(args.seed))
        segs = sg.cap_merge_segments(segs, runner.h_diff, cap=12)
        suff_z, nec_z = sg.segcore_dim(pc, segs, p_full=p_full, rounds=64,
                                       seed=int(args.seed))
        sgsc = sg.segcore_scores(segs, J, suff_z, nec_z, final_scores)
        cands = {"v2final": final_scores, "segcore": sgsc}
        best, _ = ns.select_prefix(pc, cands, ks=(2, 4, 8, 12, 20, 32, 48, 72, 120),
                                   p_full=p_full, p_base=p_base)
        chosen = cands[best]

        infl = syms["inflow"](model, x, target_class=target).astype(np.float32)
        res = {}
        for nm, s_ in {"inflow": infl, "cf_blob": cb, "cf_kernel": ck,
                       "segcore": sgsc, "chosen": chosen}.items():
            hins, hdel, _, _ = q.hard_curves(runner, s_, chunk=128)
            res[nm] = {"hdel": float(hdel), "hins": float(hins)}
        rows[name] = {"metrics": res, "pick": best}

        ax = axes[ri, 0]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=8)
        draw_jet(axes[ri, 1], img, infl, f"inflow {res['inflow']['hdel']:.3f}")
        draw_signed(axes[ri, 2], img, cb,
                    f"CoreFRI-blob (4.2k st) {res['cf_blob']['hdel']:.3f}")
        draw_signed(axes[ri, 3], img, ck,
                    f"CoreFRI-kernel (4.2k st) {res['cf_kernel']['hdel']:.3f}")
        draw_signed(axes[ri, 4], img, sgsc,
                    f"SegCoreFRI (~190 pr) {res['segcore']['hdel']:.3f}")
        draw_jet(axes[ri, 5], img, chosen,
                 f"NSD chosen[{best}] {res['chosen']['hdel']:.3f}")
        print(f"[{name:<11}] inflow={res['inflow']['hdel']:.3f} "
              f"cfB={res['cf_blob']['hdel']:.3f} cfK={res['cf_kernel']['hdel']:.3f} "
              f"segcore={res['segcore']['hdel']:.3f} chosen={res['chosen']['hdel']:.3f}",
              flush=True)

    for nm in ("inflow", "cf_blob", "cf_kernel", "segcore", "chosen"):
        d = np.array([rows[k]["metrics"][nm]["hdel"] for k in rows])
        print(f"{nm:9} mean {d.mean():.4f}")
    plt.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
