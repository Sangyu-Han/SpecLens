#!/usr/bin/env python3
"""Qualitative hidden_fri vs inflow on n images. Per image: original | hidden_fri
NECESSARY set | inflow set, both at the SAME k = hidden_fri's 90%-destroy size (genuine
input deletion), so you compare WHERE each method puts the necessity. Pages of 10."""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import random
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
HiddenNec = _load("hv", "scripts/viz_necessity_hidden_vit.py").HiddenNec


def overlay(ax, img, idx, g, title, color):
    m = np.zeros((g, g))
    for i in idx:
        m[i // g, i % g] = 1.0
    big = np.kron(m, np.ones((224 // g, 224 // g)))
    ax.imshow((img * np.where(big[..., None] > 0, 1.0, 0.3)).astype(np.uint8))
    ax.imshow(np.dstack([np.full_like(big, color[0]), np.full_like(big, color[1]),
                         np.full_like(big, color[2]), big * 0.30]))
    ax.set_title(title, fontsize=8); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nimg", type=int, default=30)
    ap.add_argument("--per-page", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/hfri_vs_inflow")
    args = ap.parse_args()
    import timm
    syms = hg5.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = np.array(cfg["mean"]); std = np.array(cfg["std"])
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths); imgs = paths[: args.nimg]
    rows = []
    for n, ip in enumerate(imgs, 1):
        x, _ = syms["load_image"](ip); x = x.to(args.device)
        with torch.no_grad():
            c = int(model(x)[0].argmax())
        hn = HiddenNec(model, x, args.device, target="prob"); N, g = hn.N, hn.g
        fri = hn.fri_necessity(c)[0]
        inflow = np.asarray(syms["inflow"](model, x, target_class=c), dtype=np.float32)
        full = hn.input_prob(torch.ones(N), c); base = hn.input_prob(torch.zeros(N), c)
        den = abs(full - base) + 1e-6
        order = np.argsort(-fri); removed = torch.zeros(N); k = N
        for kk, p in enumerate(order):
            removed[int(p)] = 1.0
            if (hn.input_prob(1 - removed, c) - base) / den <= 0.1:
                k = kk + 1; break
        k = int(np.clip(k, 5, 80))
        img = ((x[0].permute(1, 2, 0).cpu().numpy() * std + mean).clip(0, 1) * 255).astype(np.uint8)
        rows.append((img, np.argsort(-fri)[:k], np.argsort(-inflow)[:k], k, c, g, ip.stem))
        print(f"[{n}/{len(imgs)}] {ip.stem} class {c} k={k}", flush=True)
        torch.cuda.empty_cache()
    args.out.mkdir(parents=True, exist_ok=True)
    pp = args.per_page
    for pg in range((len(rows) + pp - 1) // pp):
        chunk = rows[pg * pp:(pg + 1) * pp]
        fig, ax = plt.subplots(len(chunk), 3, figsize=(7.5, 2.5 * len(chunk)))
        if len(chunk) == 1:
            ax = ax[None, :]
        for i, (img, fset, iset, k, c, g, stem) in enumerate(chunk):
            ax[i, 0].imshow(img); ax[i, 0].set_title(f"class {c} | k={k}", fontsize=8); ax[i, 0].axis("off")
            overlay(ax[i, 1], img, fset, g, "hidden_fri", (1, 0.15, 0.15))
            overlay(ax[i, 2], img, iset, g, "inflow", (0.15, 0.4, 1))
        fig.suptitle(f"hidden_fri vs inflow necessary set (page {pg+1})", fontsize=11)
        fig.tight_layout(); fig.savefig(args.out / f"page{pg+1}.png", dpi=98); plt.close(fig)
        print(f"  saved page{pg+1}.png ({len(chunk)} imgs)", flush=True)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
