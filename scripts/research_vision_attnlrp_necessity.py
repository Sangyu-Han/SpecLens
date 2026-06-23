#!/usr/bin/env python3
"""NECESSITY vs AttnLRP on VISION (deletion AUC, LOWER=better). Uses torchvision vit_b_16 (lxt
supports vit_torch) so AttnLRP (monkey-patched input x grad) is a REAL baseline. Patch-deletion:
mask 16x16 blocks -> class prob. Methods: AttnLRP, greedy-conditional, chunk_bz (Banzhaf-prior
conditional), banzhaf-raw, single_occ. Question: does the perturbation necessity beat AttnLRP on vision?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
G, P, N = 14, 16, 196
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-img", type=int, default=6)
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--Mbz", type=int, default=512)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    import torchvision.models.vision_transformer as tvit
    from lxt.efficient import monkey_patch
    monkey_patch(tvit, verbose=False)                       # AttnLRP backward rules (forward unchanged)
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    w = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=w).eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    tf = w.transforms()

    def load(path):
        img = Image.open(path).convert("RGB")
        return tf(img).unsqueeze(0).to(dev)                  # [1,3,224,224], normalized (0 = channel mean)

    def patch_mask_img(x, keep):                             # keep[N] -> x with dropped patches zeroed
        m = torch.as_tensor(keep, device=dev, dtype=x.dtype).reshape(G, G)
        mimg = m.repeat_interleave(P, 0).repeat_interleave(P, 1)
        return x * mimg[None, None]

    df = pd.read_csv(REPO / "outputs/class_fri/softplus_failure_scan_n100.csv")
    cands = [(str(REPO / "multi_object_zebra_elephant.jpg"), 386)]
    for i in range(args.n_img * 3):
        r = df.iloc[i * 5]; cands.append((str(r["path"]), int(r["target"])))
    cases = []
    for path, tgt in cands:
        try:
            x = load(path)
        except Exception:
            continue
        with torch.no_grad():
            pred = int(model(x)[0].argmax())
        if pred == tgt:                                      # torchvision-correct only
            cases.append((Path(path).stem[:12], x, tgt))
        if len(cases) >= args.n_img:
            break
    print(f"cases: {len(cases)}", flush=True)

    def run(name, x, target):
        with torch.no_grad():
            full = float(F.softmax(model(x)[0], -1)[target])
            base = float(F.softmax(model(patch_mask_img(x, np.zeros(N)))[0], -1)[target])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def prob_curve(masks):                               # masks [B,N] -> recovery [B]
            out = []
            for s in range(0, len(masks), args.bs):
                mb = masks[s:s + args.bs]
                xb = torch.cat([patch_mask_img(x, m) for m in mb], 0)
                out.append(F.softmax(model(xb), -1)[:, target])
            return ((torch.cat(out) - base) / den).cpu().numpy()

        @torch.no_grad()
        def del_auc(order):
            order = list(order); aucs = []
            for f in FR:
                k = int(round(f * N)); m = np.ones(N, np.float32)
                if k:
                    m[np.asarray(order[:k], int)] = 0.0
                aucs.append(prob_curve(m[None])[0])
            return float(np.trapz(aucs, FR) / FR[-1])

        # AttnLRP relevance (monkey-patched input x grad), aggregate per patch
        xg = x.clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(xg)[0, target].backward()
        rel = (xg.grad * xg)[0].sum(0)                       # [224,224]
        attn = rel.reshape(G, P, G, P).sum((1, 3)).flatten().detach().cpu().numpy()  # [196]

        @torch.no_grad()
        def single_occ():
            occ = np.zeros(N, np.float32)
            masks = np.ones((N, N), np.float32); masks[np.arange(N), np.arange(N)] = 0.0
            rr = prob_curve(masks)
            return 1.0 - rr                                  # high = necessary

        @torch.no_grad()
        def banzhaf():
            gen = np.random.default_rng(0); pf = gen.random((args.Mbz, 1))
            Z = (gen.random((args.Mbz, N)) < pf).astype(np.float32); R = prob_curve(Z); n1 = Z.sum(0)
            return (Z * R[:, None]).sum(0) / np.clip(n1, 1, None) - ((1 - Z) * R[:, None]).sum(0) / np.clip(args.Mbz - n1, 1, None)

        @torch.no_grad()
        def greedy():
            km = np.ones(N, np.float32); order = []; rem = list(range(N))
            while rem:
                masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
                probs = prob_curve(masks); j = int(np.argmin(probs)); km[rem[j]] = 0.0; order.append(rem[j]); rem.pop(j)
            return order

        @torch.no_grad()
        def chunked(prior):
            cand = [int(i) for i in np.argsort(-prior)][:args.M]
            km = np.ones(N, np.float32); order = []; rem = list(cand); per = max(1, args.M // args.R)
            while rem:
                masks = np.tile(km, (len(rem), 1)); masks[np.arange(len(rem)), rem] = 0.0
                probs = prob_curve(masks); idx = list(np.argsort(probs)[:per])
                for j in idx:
                    km[rem[j]] = 0.0; order.append(rem[j])
                for j in sorted(idx, reverse=True):
                    rem.pop(j)
            order += [int(i) for i in np.argsort(-prior) if int(i) not in set(order)]
            return order

        occ = single_occ(); bz = banzhaf()
        return dict(attn=del_auc(np.argsort(-attn)), greedy=del_auc(greedy()),
                    chunk_bz=del_auc(chunked(bz)), banzhaf=del_auc(np.argsort(-bz)),
                    single_occ=del_auc(np.argsort(-occ)))

    rows = []
    for name, x, tgt in cases:
        try:
            r = run(name, x, tgt); r["case"] = name; rows.append(r)
            print(f"  {name:12s} | attn={r['attn']:.3f} greedy={r['greedy']:.3f} chunk_bz={r['chunk_bz']:.3f} "
                  f"banzhaf={r['banzhaf']:.3f} single_occ={r['single_occ']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {name} OOM", flush=True)

    print("\n=== VISION NECESSITY vs AttnLRP (deletion AUC, LOWER=better) — torchvision vit_b_16 ===")
    for m in ["attn", "greedy", "chunk_bz", "banzhaf", "single_occ"]:
        print(f"  {m:11s} del={np.mean([r[m] for r in rows]):.3f}")
    bz_win = 100.0 * np.mean([r["chunk_bz"] < r["attn"] for r in rows])
    g_win = 100.0 * np.mean([r["greedy"] < r["attn"] for r in rows])
    print(f"(chunk_bz<attn {bz_win:.0f}%, greedy<attn {g_win:.0f}% — does perturbation necessity beat AttnLRP on vision?)")
    Path(REPO / "outputs/vision_attnlrp_necessity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
