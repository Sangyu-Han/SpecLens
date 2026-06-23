#!/usr/bin/env python3
"""Qualitative GREEDY-ORACLE necessary set (the lowest-hdel method, ~0.11) on the
zebra-elephant. greedy-conditional INPUT deletion (remove the patch with max
conditional marginal each round) -> necessity ORDER + 90%-destroy SET. Per target:
original | necessity-order heatmap (removed-first=most necessary) | NECESSARY set.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = _ilu.spec_from_file_location(name, REPO / rel)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


hg5 = _load("hg5", "scripts/research_hgrad_v5.py")
VIZ = REPO / "outputs/class_fri/research_frontier/greedy_oracle_viz"


def greedy_full(runner, N, dev):
    removed = np.zeros(N, np.float32); order = []; remaining = list(range(N)); probc = []
    while remaining:
        masks = np.tile(1.0 - removed, (len(remaining), 1)).astype(np.float32)
        for r, i in enumerate(remaining):
            masks[r, i] = 0.0
        with torch.no_grad():
            vals = runner.probs_for_masks(torch.as_tensor(masks, device=dev))
        vals = vals.detach().cpu().numpy() if torch.is_tensor(vals) else np.asarray(vals)
        j = int(np.argmin(vals)); p = remaining[j]
        order.append(p); removed[p] = 1.0; remaining.remove(p); probc.append(float(vals[j]))
    return order, np.array(probc, np.float32)


def overlay(ax, img, idx, g, title, color=(1, 0.2, 0.2)):
    m = np.zeros((g, g))
    for i in idx:
        m[i // g, i % g] = 1.0
    big = np.kron(m, np.ones((224 // g, 224 // g)))
    ax.imshow((img * np.where(big[..., None] > 0, 1.0, 0.35)).astype(np.uint8))
    ax.imshow(np.dstack([np.full_like(big, color[0]), np.full_like(big, color[1]), np.full_like(big, color[2]), big * 0.25]))
    ax.set_title(title, fontsize=10); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image", default="multi_object_zebra_elephant.jpg")
    ap.add_argument("--target-class", type=int, nargs="+", default=[386, 340])
    ap.add_argument("--destroy", type=float, default=0.9)
    args = ap.parse_args()
    import timm
    syms = hg5.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device=args.device).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=args.device).view(1, 3, 1, 1)
    x, _ = syms["load_image"](REPO / args.image); x = x.to(args.device)
    img = ((x * std + mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    N = x.shape[-1] // 16 * (x.shape[-1] // 16); g = x.shape[-1] // 16
    thresh = 1.0 - args.destroy
    VIZ.mkdir(parents=True, exist_ok=True)
    for c in args.target_class:
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = hg5.Block0Runner(model, x, h_b0, cls_tok, base, c)
        with torch.no_grad():
            full = float(runner.probs_for_masks(torch.ones(1, N, device=args.device)))
            base_p = float(runner.probs_for_masks(torch.zeros(1, N, device=args.device)))
        den = abs(full - base_p) + 1e-6
        order, probc = greedy_full(runner, N, args.device)
        rec = (probc - base_p) / den
        kstop = int(np.argmax(rec <= thresh)) + 1 if (rec <= thresh).any() else N
        S = order[:kstop]
        # MARGINAL (prob drop per removal) -> sharp; the rank-tail is arbitrary once destroyed
        marg = np.zeros(N); prev = full
        for k, p in enumerate(order):
            marg[p] = max(prev - float(probc[k]), 0.0); prev = float(probc[k])
        nec = marg / (marg.max() + 1e-9)
        curve = np.concatenate([[full], probc])
        fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
        ax[0].imshow(img); ax[0].set_title(f"original | GREEDY ORACLE\nclass {c}, full prob {full:.2f}", fontsize=9); ax[0].axis("off")
        ax[1].imshow(img); ax[1].imshow(np.kron(nec.reshape(g, g), np.ones((224 // g, 224 // g))), cmap="hot", alpha=0.6)
        ax[1].set_title("necessity MARGINAL (prob drop / removal)\nsharp: only the set is hot", fontsize=9); ax[1].axis("off")
        overlay(ax[2], img, S, g, f"NECESSARY set ({len(S)}/{N}={100*len(S)//N}%)\nremove -> {int(100*args.destroy)}% destroyed", (1, 0.2, 0.2))
        ax[3].plot(range(len(curve)), curve, lw=1.2)
        ax[3].axvline(kstop, color="r", ls="--", lw=1, label=f"{kstop}패치=90% destroy")
        ax[3].axhline(base_p + thresh * den, color="gray", ls=":", lw=1, label="destroy thresh")
        ax[3].set_xlim(0, min(55, N)); ax[3].set_xlabel("patches removed (greedy order)"); ax[3].set_ylabel("target prob")
        ax[3].set_title("deletion profile: hits FLOOR after the set\n-> tail order is arbitrary (noise)", fontsize=9)
        ax[3].legend(fontsize=8); ax[3].grid(alpha=0.3)
        out = VIZ / f"greedy_class{c}.png"
        fig.tight_layout(); fig.savefig(out, dpi=95); plt.close(fig)
        kf = min(40, N - 1)
        print(f"class {c}: necessary {len(S)}/{N} ({100*len(S)//N}%) | prob full {full:.3f} -> after{kstop} {float(probc[kstop-1]):.3f} -> after40 {float(probc[kf]):.3f} (floor) -> {out.name}", flush=True)
    print(f"[done] -> {VIZ}")


if __name__ == "__main__":
    main()
