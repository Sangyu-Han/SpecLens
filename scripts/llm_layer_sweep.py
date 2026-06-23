#!/usr/bin/env python3
"""LLM self-patch breakdown: per layer L, occlude each position's hidden state (->mean)
and measure the answer-logit drop = per-position necessity at L. overlap(L)=Spearman(
imp_L, imp_0) where L=0 is the genuine INPUT necessity. If self-patch holds, imp_L~imp_0
(same positions matter); it breaks when info moves to summary/last positions in late
layers. Outputs overlap(L) curve + a position x layer importance heatmap (Eiffel prompt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
VIZ = REPO / "outputs/class_fri/research_frontier/llm_selfpatch"
PROMPTS = [
    ("The Eiffel Tower is located in the city of", "Paris"),
    ("The capital city of Japan is", "Tokyo"),
    ("The chemical symbol for gold is", "Au"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The first president of the United States was George", "Washington"),
    ("The Great Wall is located in the country of", "China"),
    ("The tallest mountain in the world is Mount", "Everest"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    layers = model.model.layers; nl = len(layers)
    state = {"drop": None}

    def make_hook():
        def hook(mod, a, kw):
            h = a[0]
            if state["drop"] is not None:
                mean = h.mean(dim=1, keepdim=True)
                h = h.clone()
                for r, pos in enumerate(state["drop"]):
                    h[r, pos] = mean[r, 0]
            return (h,) + a[1:], kw
        return hook

    def imp_at_layer(ids, tgt, full, L):
        """occlude each position at layer L (batched), return per-position logit drop."""
        T = ids.shape[1]
        batch = ids.repeat(T, 1)
        hd = layers[L].register_forward_pre_hook(make_hook(), with_kwargs=True)
        state["drop"] = list(range(T))
        with torch.no_grad():
            lg = model(batch).logits[:, -1, tgt].float().cpu().numpy()
        state["drop"] = None; hd.remove()
        return full - lg

    sweep = list(range(nl))
    overlaps = {L: [] for L in sweep}
    heat = None; heat_tokens = None
    for p, ans in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            full_lg = model(ids).logits[0, -1]
        tgt = int(full_lg.argmax()); full = float(full_lg[tgt])
        imps = {L: imp_at_layer(ids, tgt, full, L) for L in sweep}
        imp0 = imps[0]
        for L in sweep:
            r = spearmanr(imps[L], imp0).correlation if len(imp0) > 2 else np.nan
            overlaps[L].append(float(r) if r == r else 0.0)
        if p.startswith("The Eiffel"):
            heat = np.stack([imps[L] for L in sweep], axis=1)  # [pos, layer]
            heat_tokens = [tok.decode(t) for t in ids[0].tolist()]
    ov = np.array([np.mean(overlaps[L]) for L in sweep])
    VIZ.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    ax[0].plot(sweep, ov, "-o", ms=3)
    ax[0].axhline(0, color="gray", lw=0.7); ax[0].set_xlabel("layer L (hidden masked)")
    ax[0].set_ylabel("Spearman(imp_L, imp_0=input)"); ax[0].set_ylim(-0.5, 1.05); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"self-patch validity vs layer ({args.model.split('/')[-1]}, n={len(PROMPTS)})\nhigh=hidden necessity matches input; drop=self-patch breaks")
    hn = heat / (np.abs(heat).max(0, keepdims=True) + 1e-9)
    im = ax[1].imshow(hn, aspect="auto", cmap="hot")
    ax[1].set_yticks(range(len(heat_tokens))); ax[1].set_yticklabels(heat_tokens, fontsize=7)
    ax[1].set_xlabel("layer L"); ax[1].set_title("necessity per position x layer (Eiffel->Paris)\ninfo moves: content tokens -> last/summary")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(VIZ / "selfpatch_breakdown.png", dpi=110); plt.close(fig)
    print("=== overlap(L) = Spearman(necessity_L, necessity_input) ===")
    for L in sweep[::2]:
        print(f"  L{L:2d}: {ov[L]:+.3f}")
    print(f"early(L0-4) mean {ov[:5].mean():.3f} | mid(L12-16) {ov[12:17].mean():.3f} | late(L24+) {ov[24:].mean():.3f}")
    print(f"[done] -> {VIZ}/selfpatch_breakdown.png")


if __name__ == "__main__":
    main()
