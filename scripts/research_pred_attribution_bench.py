#!/usr/bin/env python3
"""CLIP prediction-attribution benchmark: our NECESSITY (greedy/chunked/soft-mask) + FRI
(sufficiency) vs standard baselines (inflow, gradient, integrated-gradients) on ImageNet
classification. Metrics: deletion AUC (↓ = necessity) + insertion AUC (↑ = sufficiency) via
hard_curves. Shows the two-pillar on STANDARD prediction attribution + whether conditional
necessity beats the baselines at deletion."""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
FPATH = REPO.parent / "_fri_research_archive_20260610/scripts/research_fri_frontier.py"
_s = _ilu.spec_from_file_location("frontier", FPATH); F = _ilu.module_from_spec(_s); _s.loader.exec_module(F)
N = 196
CHUNK = 64


def _rank(order):
    sc = np.zeros(N, np.float32); sc[np.asarray(order, int)] = np.arange(len(order), 0, -1); return sc


def greedy_pred(runner, stop=0.1):
    km = np.ones(N, np.float32); order = []; rem = list(range(N))
    full = float(runner.prob_curve(np.ones((1, N), np.float32))[0]); k = None
    while rem:
        masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
        probs = runner.prob_curve(masks, CHUNK)
        j = int(np.argmin(probs)); km[rem[j]] = 0.0; order.append(rem[j]); bv = float(probs[j]); rem.pop(j)
        if k is None and bv <= stop * full:
            k = len(order)
    return _rank(order), (k or len(order))


def chunked_pred(runner, M=100, R=8):
    occ = np.ones((N, N), np.float32); occ[np.arange(N), np.arange(N)] = 0.0
    full = float(runner.prob_curve(np.ones((1, N), np.float32))[0])
    prior = full - runner.prob_curve(occ, CHUNK)               # single-occ necessity
    cand = [int(i) for i in np.argsort(-prior)][:M]
    km = np.ones(N, np.float32); order = []; rem = list(cand); per = max(1, M // R)
    while rem:
        masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
        probs = runner.prob_curve(masks, CHUNK)
        idx = list(np.argsort(probs)[:per])
        for j in idx:
            km[rem[j]] = 0.0; order.append(rem[j])
        for j in sorted(idx, reverse=True):
            rem.pop(j)
    order += [int(i) for i in np.argsort(-prior) if int(i) not in set(order)]
    return _rank(order)


def softmask_pred(runner, steps=64, l1=0.002, seed=0):
    dev, dtype = runner.dev, runner.dtype
    with torch.no_grad():
        full = runner.probs_for_masks(torch.ones(1, N, device=dev, dtype=dtype))[0]
        base = runner.probs_for_masks(torch.zeros(1, N, device=dev, dtype=dtype))[0]
    den = (full - base).clamp(min=1e-6)
    la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    for step in range(steps):
        lr = 0.01 + 0.5 * (0.4 - 0.01) * (1 + math.cos(math.pi * step / (steps - 1)))
        lar = la.clone().requires_grad_(True); p = torch.sigmoid(lar); pn = p / (p.sum() + 1e-6)
        b = float(torch.rand(1, generator=gen, device=dev).item()) * N
        r = (pn * b).clamp(max=1.0)
        prob = runner.probs_for_masks((1.0 - r).unsqueeze(0))[0]
        loss = (prob - base) / den + l1 * p.sum()              # minimize recovery = necessity
        g = torch.autograd.grad(loss, lar)[0].detach(); t = step + 1
        mv = 0.9 * mv + 0.1 * g; vv = 0.999 * vv + 0.001 * g * g
        adam = (mv / (1 - 0.9 ** t)) / ((vv / (1 - 0.999 ** t)).sqrt() + 1e-8)
        cm = (adam * g > 0).float(); cm = cm * (N / cm.sum().clamp(min=1.0))
        la = la - lr * adam * cm
    return torch.sigmoid(la).detach().cpu().numpy()


def grad_pred(runner):
    m = torch.ones(N, device=runner.dev, dtype=runner.dtype, requires_grad=True)
    prob = runner.probs_for_masks(m.unsqueeze(0))[0]
    return torch.autograd.grad(prob, m)[0].abs().detach().cpu().numpy()


def banzhaf_pred(runner, M=512, seed=0):
    """Cooperative restricted-range Banzhaf (the LLM-unification signal) on vision patches: does the
    SAME ranking that finds the sufficiency set ALSO do necessity (deletion)? marg high = important."""
    gen = np.random.default_rng(seed)
    pf = gen.random((M, 1)); Z = (gen.random((M, N)) < pf).astype(np.float32)
    R = np.asarray(runner.prob_curve(Z, CHUNK), np.float32)
    n1 = Z.sum(0)
    return ((Z * R[:, None]).sum(0) / np.clip(n1, 1, None) - ((1 - Z) * R[:, None]).sum(0) / np.clip(M - n1, 1, None)).astype(np.float32)


def ig_pred(runner, steps=20):
    tot = torch.zeros(N, device=runner.dev, dtype=runner.dtype)
    for a in np.linspace(1.0 / steps, 1.0, steps):
        m = torch.full((N,), float(a), device=runner.dev, dtype=runner.dtype, requires_grad=True)
        prob = runner.probs_for_masks(m.unsqueeze(0))[0]
        tot = tot + torch.autograd.grad(prob, m)[0].detach()
    return (tot / steps).abs().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=3)
    ap.add_argument("--model", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
                    help="timm ViT with an in1k classification head (CLIP / supervised / DeiT)")
    ap.add_argument("--out", default="outputs/pred_attribution_bench.json")
    ap.add_argument("--chunk", type=int, default=64, help="batch size for prob_curve (lower=less GPU peak)")
    args = ap.parse_args()
    global CHUNK
    CHUNK = int(args.chunk)
    syms = F.load_patch_repo()
    import timm
    print(f"[model] {args.model}", flush=True)
    model = timm.create_model(args.model, pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    cases = [("zebra", str(REPO / "multi_object_zebra_elephant.jpg"), 340)]
    for i in range(args.n_img):
        r = df.iloc[i * 7]; cases.append((str(r["desc"])[:12], str(r["path"]), int(r["target"])))
    methods = ["greedy", "chunked", "softmask", "fri", "banzhaf", "inflow", "gradient", "ig"]
    rows = []
    for name, path, target in cases:
        x, _ = syms["load_image"](Path(path)); x = x.to(args.device)
        h_b0, cls_tok, baseline = syms["get_b0"](model, x)
        runner = F.Block0Runner(model, x, h_b0, cls_tok, baseline, target)
        sc = {}
        sc["greedy"], k = greedy_pred(runner)
        sc["chunked"] = chunked_pred(runner)
        sc["softmask"] = softmask_pred(runner)
        sc["fri"] = np.asarray(F.fri_alt_prob(runner, steps=32)["final"], np.float32)
        try:
            sc["inflow"] = np.asarray(syms["inflow"](model, x, target_class=target), np.float32).reshape(-1)
        except Exception as e:
            if name == cases[0][0]:
                print(f"[inflow unavailable for {args.model}: {type(e).__name__}] -> skipped (zeros)", flush=True)
            sc["inflow"] = np.zeros(N, np.float32)
        sc["gradient"] = grad_pred(runner)
        sc["banzhaf"] = banzhaf_pred(runner)
        sc["ig"] = ig_pred(runner)
        res = {"case": name, "target": target, "k": int(k)}
        for m in methods:
            ins, dele, _, _ = F.hard_curves(runner, sc[m]); res[m + "_del"] = round(dele, 4); res[m + "_ins"] = round(ins, 4)
        rows.append(res)
        print(f"{name:12s} k={k:3d} | " + " ".join(f"{m}:d{res[m+'_del']:.2f}/i{res[m+'_ins']:.2f}" for m in methods), flush=True)
    print("\n=== MEAN (del ↓ necessity, ins ↑ sufficiency) ===")
    print(f"{'method':10s} {'del_auc':>8s} {'ins_auc':>8s}")
    agg = {}
    for m in methods:
        d = float(np.mean([r[m + "_del"] for r in rows])); i = float(np.mean([r[m + "_ins"] for r in rows]))
        agg[m] = {"del": round(d, 4), "ins": round(i, 4)}
        print(f"{m:10s} {d:8.3f} {i:8.3f}")
    Path(args.out).write_text(json.dumps({"rows": rows, "mean": agg}, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
