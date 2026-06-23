#!/usr/bin/env python3
"""NER lead #3 — LAYER SWEEP on a self-patch ViT (CLIP).

Generalize the last-block hidden-FRI solve (viz_necessity_hidden_vit.HiddenNec) to
solve at ANY block i: mask block-i input PATCH tokens toward the layer-i hidden MEAN,
forward blocks[i:] + norm + head (FRI delonly, prob target), read out per-token
removal-necessity, map to input by self-patch (token i <-> input patch i), score[196].

Purpose: on CLIP self-patch holds at every depth, so the deletion-AUC should stay ~flat
across i if the rich-hidden solve is depth-robust. WHERE it starts to drift = how far a
necessity solve can sit from the readout before the representation stops being a faithful
self-patch proxy. This is the empirical floor for how deep NER can chain (the assumption
SAMv2 stresses), and the tool we reuse on non-self-patch models next.

Metric: hard_ins/hard_del AUC via the established Block0Runner harness (same eval inflow
and hidden-FRI are scored by). Lower hard_del = better. Compared to inflow each image.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import timm  # noqa: F401 — pin timm 1.0.7 in sys.modules BEFORE the repo path insert below
            # (loading the hg5 harness otherwise shadows it with a 0.4.12 that lacks the CLIP model)

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name, rel):
    spec = _ilu.spec_from_file_location(name, REPO / rel)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hg5 = _load("hg5", "scripts/research_hgrad_v5.py")


class LayerNec:
    """FRI-delonly necessity solve at an arbitrary block i (self-patch mapping)."""

    def __init__(self, model, x, device, target="prob"):
        self.model = model
        self.dev = device
        self.target = target
        self.x = x
        self.prefix = int(getattr(model, "num_prefix_tokens", 1))
        self.L = len(model.blocks)
        # one forward, capture the INPUT to every block
        cap = {}
        hooks = []
        for i, b in enumerate(model.blocks):
            hooks.append(b.register_forward_pre_hook(
                lambda m, a, i=i: cap.__setitem__(i, a[0].detach())))
        with torch.no_grad():
            clean = model(x)[0]
        for h in hooks:
            h.remove()
        self.H = cap                                   # {i: [1,T,D] input to block i}
        self.N = self.H[0].shape[1] - self.prefix
        self.g = int(round(self.N ** 0.5))
        self.clean_prob = torch.softmax(clean, -1).detach()

    def _tail(self, Hb, i):
        """forward blocks[i:] + norm + head -> logits[1,C] (differentiable)."""
        o = Hb
        for b in self.model.blocks[i:]:
            o = b(o)
        o = self.model.norm(o)
        return self.model.forward_head(o)

    def _fwd_grad(self, keep, c, i):
        H_i = self.H[i]
        patches = H_i[:, self.prefix:, :]
        mean = patches.mean(dim=1, keepdim=True)
        p = keep.view(1, self.N, 1)
        Hp = p * patches + (1 - p) * mean
        Hb = torch.cat([H_i[:, :self.prefix, :], Hp], dim=1)
        lg = self._tail(Hb, i)[0]
        return torch.softmax(lg, -1)[c] if self.target == "prob" else lg[c]

    @torch.no_grad()
    def _recon_prob(self, i, c):
        return float(self._fwd_grad(torch.ones(self.N, device=self.dev), c, i))

    def fri_necessity(self, c, i, steps=64, lr=0.4, lr_end=0.01, l1=0.004, seed=0):
        N, dev = self.N, self.dev
        with torch.no_grad():
            full = float(self._fwd_grad(torch.ones(N, device=dev), c, i))
            base = float(self._fwd_grad(torch.zeros(N, device=dev), c, i))
        den = abs(full - base) + 1e-6
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            p = torch.sigmoid(la_req)                              # removal prob
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            r_state = (p / (p.sum() + eps) * budget).clamp(max=1)
            keep = 1.0 - r_state
            rec = (self._fwd_grad(keep, c, i) - base) / den
            loss = rec + l1 * p.sum()
            gg = torch.autograd.grad(loss, la_req)[0].detach()
            t = step + 1
            mv = b1 * mv + (1 - b1) * gg; vv = b2 * vv + (1 - b2) * gg * gg
            adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
            cmask = (adam * gg > 0).float(); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
            la = la - cur * adam * cmask
        return torch.sigmoid(la).detach().float().cpu().numpy(), full, base


PAV = "/home/sangyu/Desktop/Master/patch-attribution-vit"  # multi-metric evaluator lives here


def greedy_oracle_score(runner, N, dev):
    """gold necessity: greedy-conditional INPUT(block0-inject) removal -> marginal prob-drop / patch."""
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
    full = float(runner.probs_for_masks(torch.ones(1, N, device=dev)))
    marg = np.zeros(N); prev = full
    for k, p in enumerate(order):
        marg[p] = max(prev - float(probc[k]), 0.0); prev = float(probc[k])
    return marg


def input_gradient_score(model, x, c, g):
    """|grad| of prob_c wrt input, pooled per patch (the 'gradient fails' reference)."""
    x2 = x.clone().requires_grad_(True)
    prob = torch.softmax(model(x2)[0], -1)[c]
    grad = torch.autograd.grad(prob, x2)[0][0]            # [3,224,224]
    s = 224 // g
    gmap = grad.abs().sum(0).view(g, s, g, s).sum(dim=(1, 3))
    return gmap.flatten().detach().float().cpu().numpy()


def run_rigorous(model, syms, imgs, args):
    """{inflow, gradient, fri@layers, oracle} x {hard/mas/stoch}_del on the SAME images."""
    sys.path.insert(0, PAV)
    from src.eval.metrics_block0 import evaluate_method_block0
    dev = args.device
    methods = ["inflow", "gradient"] + [f"fri@L{L}" for L in args.layers] + ["oracle"]
    acc = {m: {"hard": [], "mas": [], "stoch": []} for m in methods}
    t0 = time.time()
    for ii, path in enumerate(imgs):
        try:
            x, _ = syms["load_image"](Path(path)); x = x.to(dev)
        except Exception as e:
            print(f"  skip {Path(path).name}: {e}", flush=True); continue
        with torch.no_grad():
            c = int(model(x)[0].argmax())
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = hg5.Block0Runner(model, x, h_b0, cls_tok, base, c)
        ln = LayerNec(model, x, dev, target=args.target); g = ln.g; N = ln.N
        scores = {"inflow": np.asarray(syms["inflow"](model, x, c), np.float32),
                  "gradient": input_gradient_score(model, x, c, g)}
        for L in args.layers:
            scores[f"fri@L{L}"] = ln.fri_necessity(c, L, steps=args.steps)[0]
        scores["oracle"] = greedy_oracle_score(runner, N, dev)
        line = [f"[{ii+1}/{len(imgs)}] {Path(path).name} c={c}"]
        for m in methods:
            r = evaluate_method_block0(model, x, h_b0, cls_tok, base, c, scores[m],
                                       stoch_n_steps=10, stoch_n_samples=3)
            acc[m]["hard"].append(r["hard_del_auc"]); acc[m]["mas"].append(r["mas_del_auc"]); acc[m]["stoch"].append(r["stoch_del_auc"])
            line.append(f"{m}:{r['hard_del_auc']:.2f}")
        print("  " + " | ".join(line), flush=True)
        del ln, runner; torch.cuda.empty_cache()
    print(f"\n=== RIGOROUS MEAN over {len(acc['oracle']['hard'])} imgs (del AUC, LOWER=better) ===", flush=True)
    o = {k: np.mean(acc["oracle"][k]) for k in ("hard", "mas", "stoch")}
    inf = {k: np.mean(acc["inflow"][k]) for k in ("hard", "mas", "stoch")}
    print(f"  {'method':<10} {'hard_del':>9} {'mas_del':>9} {'stoch_del':>9}   gap-to-oracle closed (hard)", flush=True)
    for m in methods:
        h = np.mean(acc[m]["hard"]); ma = np.mean(acc[m]["mas"]); st = np.mean(acc[m]["stoch"])
        closed = (inf["hard"] - h) / (inf["hard"] - o["hard"] + 1e-9) * 100 if m not in ("inflow", "oracle") else float("nan")
        win = np.mean(np.array(acc[m]["hard"]) < np.array(acc["inflow"]["hard"])) * 100
        tag = f"  closed {closed:4.0f}% of inflow->oracle | beats inflow {win:.0f}%" if m not in ("inflow", "oracle") else ("  <-- GOLD" if m == "oracle" else "  <-- baseline")
        print(f"  {m:<10} {h:>9.3f} {ma:>9.3f} {st:>9.3f}{tag}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


def collect_images(val_dir, n, extra):
    """One image per DISTINCT class, spread across the label space (avoids all-tench)."""
    paths = list(extra)
    vd = Path(val_dir)
    if vd.is_dir() and n > 0:
        subdirs = sorted([d for d in vd.iterdir() if d.is_dir()])
        if subdirs:
            step = max(1, len(subdirs) // n)
            for d in subdirs[::step][:n]:
                f = sorted(d.glob("*.JPEG"))
                if f:
                    paths.append(f[0])
        else:
            files = sorted(vd.glob("*.JPEG"))
            step = max(1, len(files) // n)
            paths += files[::step][:n]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 10, 11])
    ap.add_argument("--nimg", type=int, default=4, help="ImageNet val images (+ multi-object)")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--val-dir", default="/media/sangyu/Dataset/imagenet/val")
    ap.add_argument("--target", default="prob", choices=["prob", "logit"])
    ap.add_argument("--mode", default="sweep", choices=["sweep", "rigorous"],
                    help="sweep=hdel vs inflow across layers; rigorous=multi-metric vs oracle gold")
    args = ap.parse_args()

    import timm
    dev = args.device
    syms = hg5.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
                              pretrained=True).eval().to(dev)
    for p in model.parameters():
        p.requires_grad = False

    imgs = collect_images(args.val_dir, args.nimg, [REPO / "multi_object_zebra_elephant.jpg"])
    print(f"[setup] model=clip vit-b/16  mode={args.mode}  layers={args.layers}  n={len(imgs)}  steps={args.steps}  target={args.target}", flush=True)

    if args.mode == "rigorous":
        run_rigorous(model, syms, imgs, args)
        return

    rows = {i: [] for i in args.layers}
    inflow_row = []
    t0 = time.time()
    for ii, path in enumerate(imgs):
        try:
            x, _ = syms["load_image"](Path(path)); x = x.to(dev)
        except Exception as e:
            print(f"  skip {Path(path).name}: {e}", flush=True); continue
        with torch.no_grad():
            c = int(model(x)[0].argmax())
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = hg5.Block0Runner(model, x, h_b0, cls_tok, base, c)
        # inflow baseline
        infl = syms["inflow"](model, x, c)
        _, infl_del, _, _ = hg5.hard_curves(runner, np.asarray(infl, np.float32))
        inflow_row.append(infl_del)
        ln = LayerNec(model, x, dev, target=args.target)
        msg = [f"[{ii+1}/{len(imgs)}] {Path(path).name} c={c} p={float(ln.clean_prob[c]):.2f} | inflow hdel {infl_del:.3f}"]
        for i in args.layers:
            recon = ln._recon_prob(i, c)
            sc, full, baseval = ln.fri_necessity(c, i, steps=args.steps)
            _, hdel, _, _ = hg5.hard_curves(runner, sc)
            rows[i].append(hdel)
            msg.append(f"L{i}:hdel {hdel:.3f}(recon {recon:.2f})")
        print("  " + " | ".join(msg), flush=True)
        del ln, runner
        torch.cuda.empty_cache()

    print(f"\n=== MEAN over {len(inflow_row)} imgs  (hard_del: LOWER=better) ===", flush=True)
    print(f"  inflow      hdel {np.mean(inflow_row):.4f}", flush=True)
    for i in args.layers:
        if rows[i]:
            d = np.array(rows[i])
            win = np.mean(d < np.array(inflow_row))
            print(f"  layer {i:2d}    hdel {d.mean():.4f}  (beats inflow {win*100:.0f}% of imgs)", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
