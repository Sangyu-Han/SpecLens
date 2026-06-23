#!/usr/bin/env python3
"""Necessary set in the HIDDEN domain (user's idea): mask the last-block-input PATCH
TOKENS by INTERPOLATING toward the hidden MEAN vector (m*token + (1-m)*mean), instead
of masking input pixels. Each token already encodes "I am class c" (patching showed
this), so removing it removes that evidence directly — no input-pixel OOD / diagonal
confound. Also CHEAP: only the last block + head re-run per mask (no full forward).

greedy-conditional over the N patch tokens -> 90% destroy -> necessary set, mapped
to input by the self-patch (token i <-> input patch i on a ViT). 4-panel viz vs the
input-domain result.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
VIZ = REPO / "outputs/class_fri/research_frontier/necessity_hidden_viz"


class HiddenNec:
    def __init__(self, model, x, device, target="logit"):
        self.model = model; self.dev = device; self.target = target
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))
        cap = {}
        h = model.blocks[-1].register_forward_pre_hook(lambda m, a: cap.__setitem__("H", a[0].detach()))
        with torch.no_grad():
            model(x)
        h.remove()
        self.H = cap["H"]                                  # [1, T, D] input to last block
        self.N = self.H.shape[1] - self.prefix
        self.patches = self.H[:, self.prefix:, :]          # [1, N, D]
        self.mean = self.patches.mean(dim=1, keepdim=True)  # [1,1,D] hidden mean vector
        self.x = x                                          # for INPUT-domain removal
        self.g = int(round(self.N ** 0.5))

    def input_prob(self, keep, c):
        """class prob with INPUT patches removed (genuine deletion, pixel mean-baseline)."""
        keep = torch.as_tensor(keep, device=self.dev, dtype=torch.float32)
        s = 224 // self.g
        pm = keep.view(self.g, self.g).repeat_interleave(s, 0).repeat_interleave(s, 1).view(1, 1, 224, 224)
        with torch.no_grad():
            lg = self.model(self.x * pm)[0]                # masked input -> 0 = normalized mean
        return float(torch.softmax(lg, -1)[c]) if self.target == "prob" else float(lg[c])

    def masked_logits(self, keep, chunk=64):
        """keep [M,N] (1=keep, 0->hidden mean) -> class logits [M, C]; only last block+head."""
        M = keep.shape[0]; outs = []
        for s in range(0, M, chunk):
            kb = keep[s:s + chunk].to(self.dev)
            mb = kb.shape[0]
            Hb = self.H.repeat(mb, 1, 1).clone()
            p = kb.view(mb, self.N, 1)
            Hb[:, self.prefix:, :] = p * self.patches + (1 - p) * self.mean   # interpolate to mean
            with torch.no_grad():
                o = self.model.blocks[-1](Hb)
                o = self.model.norm(o)
                lg = self.model.forward_head(o)
            outs.append(lg)
        o = torch.cat(outs)
        return torch.softmax(o, -1) if self.target == "prob" else o   # logit: stable for low-prob classes

    def greedy(self, c, thresh):
        N = self.N; dev = self.dev
        full = float(self.masked_logits(torch.ones(1, N))[0, c])
        base = float(self.masked_logits(torch.zeros(1, N))[0, c])
        den = abs(full - base) + 1e-6
        removed = torch.zeros(N); order = []; remaining = list(range(N)); rec = 1.0; nfwd = 0
        while remaining and rec > thresh:
            keep = (1 - removed).unsqueeze(0).repeat(len(remaining), 1)
            idx = torch.tensor(remaining)
            keep[torch.arange(len(remaining)), idx] = 0.0
            vals = self.masked_logits(keep)[:, c]; nfwd += len(remaining)
            j = int(torch.argmin(vals)); p = remaining[j]
            order.append(p); removed[p] = 1.0; remaining.remove(p)
            rec = (float(vals[j]) - base) / den
        return sorted([int(i) for i, v in enumerate(removed) if v > 0]), order, nfwd, full, base

    def _fwd_grad(self, keep, c):
        """differentiable class logit for a soft keep-mask [N] (interpolate to mean)."""
        p = keep.view(1, self.N, 1)
        Hp = p * self.patches + (1 - p) * self.mean
        Hb = torch.cat([self.H[:, :self.prefix, :], Hp], dim=1)
        o = self.model.blocks[-1](Hb); o = self.model.norm(o)
        h = self.model.forward_head(o)[0]
        return torch.softmax(h, -1)[c] if self.target == "prob" else h[c]

    def fri_necessity(self, c, steps=64, lr=0.4, lr_end=0.01, l1=0.004, seed=0):
        """FRI-style necessity solve: soft REMOVAL mask, random-budget, MINIMIZE
        recovery (destroy) + sparse removal. Returns per-token removal-necessity."""
        import math
        N = self.N; dev = self.dev
        with torch.no_grad():
            full = float(self.masked_logits(torch.ones(1, N))[0, c])
            base = float(self.masked_logits(torch.zeros(1, N))[0, c])
        den = abs(full - base) + 1e-6
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            p = torch.sigmoid(la_req)                                  # removal prob
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            r_state = (p / (p.sum() + eps) * budget).clamp(max=1)       # remove ~budget (soft top)
            keep = 1.0 - r_state
            rec = (self._fwd_grad(keep, c) - base) / den
            loss = rec + l1 * p.sum()                                   # minimize recovery(=destroy)+sparse
            gg = torch.autograd.grad(loss, la_req)[0].detach()
            t = step + 1
            mv = b1 * mv + (1 - b1) * gg; vv = b2 * vv + (1 - b2) * gg * gg
            adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
            cmask = (adam * gg > 0).float(); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
            la = la - cur * adam * cmask
        return torch.sigmoid(la).detach().float().cpu().numpy(), full, base

    def fri_alt(self, c, steps=64, lr=0.4, lr_end=0.01, l1=0.004, del_w=0.7, seed=0):
        """Canonical FRI: ALTERNATE insertion (keep top -> maximize recovery) and
        deletion (remove top -> minimize recovery). Shared probs = suf AND nec set."""
        import math
        N = self.N; dev = self.dev
        with torch.no_grad():
            full = float(self.masked_logits(torch.ones(1, N))[0, c])
            base = float(self.masked_logits(torch.zeros(1, N))[0, c])
        den = abs(full - base) + 1e-6
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(steps):
            is_del = (step % 2 == 1)
            frac = step / max(steps - 1, 1)
            cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            p = torch.sigmoid(la_req)
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            w = (p / (p.sum() + eps) * budget).clamp(max=1)
            state = w if not is_del else (1.0 - w)                      # keep-top (ins) / remove-top (del)
            rec = (self._fwd_grad(state, c) - base) / den
            rec_term = (1.0 - rec) if not is_del else del_w * rec       # ins: maximize rec / del: minimize rec
            loss = rec_term + l1 * p.sum()
            gg = torch.autograd.grad(loss, la_req)[0].detach()
            t = step + 1
            mv = b1 * mv + (1 - b1) * gg; vv = b2 * vv + (1 - b2) * gg * gg
            adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
            cmask = (adam * gg > 0).float(); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
            la = la - cur * adam * cmask
        return torch.sigmoid(la).detach().float().cpu().numpy(), full, base

    def fri_set(self, c, thresh, solver="alt"):
        """FRI scores from the HIDDEN solve, but the 90%-destroy SET decided by GENUINE
        INPUT removal (delete the actual input patches) — fits the deletion philosophy."""
        scores, _, _ = (self.fri_alt(c) if solver == "alt" else self.fri_necessity(c))
        full = self.input_prob(torch.ones(self.N), c)              # INPUT-domain full/base
        base = self.input_prob(torch.zeros(self.N), c)
        den = abs(full - base) + 1e-6
        order = list(np.argsort(-scores)); removed = torch.zeros(self.N); S = []; nfwd = 0
        for p in order:
            removed[p] = 1.0; S.append(int(p))
            rec = (self.input_prob(1 - removed, c) - base) / den   # GENUINE input deletion
            nfwd += 1
            if rec <= thresh:
                break
        return sorted(S), order, nfwd, scores

    def suff(self, c, thresh_keep):
        """FRI-analog sufficiency in hidden domain: greedy insertion (keep i from mean)."""
        N = self.N
        full = float(self.masked_logits(torch.ones(1, N))[0, c])
        base = float(self.masked_logits(torch.zeros(1, N))[0, c]); den = abs(full - base) + 1e-6
        kept = torch.zeros(N); S = []; remaining = list(range(N)); rec = 0.0
        while remaining and rec < thresh_keep:
            keep = kept.unsqueeze(0).repeat(len(remaining), 1)
            idx = torch.tensor(remaining); keep[torch.arange(len(remaining)), idx] = 1.0
            vals = self.masked_logits(keep)[:, c]
            j = int(torch.argmax(vals)); p = remaining[j]
            kept[p] = 1.0; S.append(p); remaining.remove(p); rec = (float(vals[j]) - base) / den
        return S


def overlay(ax, img, idx, g, title, color):
    m = np.zeros((g, g))
    for i in idx:
        m[i // g, i % g] = 1.0
    big = np.kron(m, np.ones((224 // g, 224 // g)))
    ax.imshow((img * np.where(big[..., None] > 0, 1.0, 0.35)).astype(np.uint8))
    ax.imshow(np.dstack([np.full_like(big, color[0]), np.full_like(big, color[1]),
                         np.full_like(big, color[2]), big * 0.25]))
    ax.set_title(title, fontsize=10); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--image", default="multi_object_zebra_elephant.jpg")
    ap.add_argument("--target-class", type=int, nargs="+", default=[386, 340])
    ap.add_argument("--destroy", type=float, default=0.9)
    ap.add_argument("--method", default="fri", choices=["greedy", "fri"])
    ap.add_argument("--solver", default="delonly", choices=["alt", "delonly"], help="alt=insertion+deletion (canonical); delonly=deletion-only")
    ap.add_argument("--target", default="logit", choices=["logit", "prob"], help="destruction criterion (logit stable for low-prob classes)")
    args = ap.parse_args()
    import timm
    import torchvision.transforms as T
    from PIL import Image
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=args.device).view(1, 3, 1, 1)
    tf_img = T.Compose([T.Resize(256), T.CenterCrop(224)])
    pil = tf_img(Image.open(args.image).convert("RGB")); img = np.asarray(pil, np.uint8)
    x = (T.ToTensor()(pil).unsqueeze(0).to(args.device) - mean) / std
    hn = HiddenNec(model, x, args.device, target=args.target); g = int(round(hn.N ** 0.5)); N = hn.N
    thresh = 1.0 - args.destroy
    VIZ.mkdir(parents=True, exist_ok=True)
    for c in args.target_class:
        if args.method == "fri":
            S, order, nfwd, scores = hn.fri_set(c, thresh, solver=args.solver)
            tag = f"FRI {args.solver} (ins+del)" if args.solver == "alt" else f"FRI {args.solver}"
            nec_rank = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)   # FRI saliency map
        else:
            S, order, nfwd, full, base = hn.greedy(c, thresh)
            tag = "greedy O(N^2)"
            nec_rank = np.zeros(N)
            for jj, p in enumerate(order):
                nec_rank[p] = len(order) - jj
            if nec_rank.max() > 0:
                nec_rank /= nec_rank.max()
        suff = hn.suff(c, args.destroy)
        fig, ax = plt.subplots(1, 4, figsize=(15, 4))
        ax[0].imshow(img); ax[0].set_title(f"original | HIDDEN-domain necessity (class {c})\n{tag}", fontsize=10); ax[0].axis("off")
        ax[1].imshow(img); ax[1].imshow(np.kron(nec_rank.reshape(g, g), np.ones((224 // g, 224 // g))), cmap="hot", alpha=0.6)
        ax[1].set_title("necessity score (hidden)\nhot = most necessary", fontsize=10); ax[1].axis("off")
        overlay(ax[2], img, S, g, f"NECESSARY set ({len(S)}/{N}={100*len(S)//N}%)\nhidden-token removal destroys", (1, 0.2, 0.2))
        overlay(ax[3], img, suff, g, f"SUFFICIENCY set ({len(suff)}/{N})", (0.2, 0.5, 1))
        out = VIZ / f"hidden_{args.method}_class{c}.png"
        fig.tight_layout(); fig.savefig(out, dpi=95); plt.close(fig)
        print(f"class {c} [{args.method}]: necessary {len(S)}/{N} ({100*len(S)//N}%), {nfwd} eval-fwd, suff {len(suff)} -> {out.name}", flush=True)
    print(f"[done] -> {VIZ}")


if __name__ == "__main__":
    main()
