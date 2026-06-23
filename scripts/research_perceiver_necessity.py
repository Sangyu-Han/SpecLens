#!/usr/bin/env python3
"""Necessity on a POSITION-DECOUPLED latent-bottleneck transformer (Perceiver).

Perceiver = 512 learned LATENTS cross-attend to 50176 input pixels. The latents
have NO positional correspondence to inputs -> position-matching is IMPOSSIBLE.
So the hidden->input necessity mapping MUST be position-free. This is exactly
where our method (hidden-ablation finds necessary latents; map to input via
cross-attention) is the ONLY option, competing vs cross-attn-rollout / gradient.

Native ImageNet head -> sharp class-prob target (no probe needed).
Masking = MEAN-baseline (deleted patch -> mean pixel; mask ratio up -> mean image,
stays in-distribution). Input units = 14x14=196 patches over the 224x224 pixels.

Compares input-deletion AUC (lower=better):
  random | xattn-rollout (mean latent->patch) | gradient | NECESSITY (ours)
ours = sum_c w_c * (cluster c latents' cross-attn to patches), w_c = hidden-
ablation class-prob drop of latent cluster c (necessary latents).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def kml(vecs, k, seed=0):
    from sklearn.cluster import KMeans
    vn = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit(vn).labels_


class PerceiverRunner:
    def __init__(self, model, pixel_values, device, grid=14, patch=16):
        self.model = model; self.dev = device
        self.pv = pixel_values.to(device)              # [1,3,224,224]
        self.grid = grid; self.patch = patch; self.N = grid * grid
        self.mean_pix = self.pv.mean(dim=(2, 3), keepdim=True)   # [1,3,1,1] mean baseline
        with torch.no_grad():
            out = model(pixel_values=self.pv, output_hidden_states=True, output_attentions=True)
        self.cls = int(out.logits.argmax())
        self.p_full = float(torch.softmax(out.logits[0], 0)[self.cls])
        self.full_lat = out.hidden_states[-1][0].float().cpu().numpy()    # [512,1024]
        xa = out.cross_attentions[0][0].mean(0)        # [512, 50176] (avg heads)
        side = self.grid * self.patch
        xa = xa.reshape(xa.shape[0], side, side)        # [512,224,224]
        xa = xa.reshape(xa.shape[0], grid, patch, grid, patch).mean(dim=(2, 4))  # [512,14,14]
        self.xattn = xa.reshape(xa.shape[0], self.N).float().cpu().numpy()  # [512,196]
        self.nlat = self.full_lat.shape[0]

    def _mask_pv(self, patch_masks):                    # [B,N] -> [B,3,224,224]
        B = patch_masks.shape[0]
        keep = patch_masks.reshape(B, self.grid, self.grid)
        keep = keep.repeat_interleave(self.patch, 1).repeat_interleave(self.patch, 2)[:, None]
        return self.pv * keep + self.mean_pix * (1 - keep)

    def target_curve(self, masks_np, chunk=32):
        outs = []
        with torch.no_grad():
            for s in range(0, len(masks_np), chunk):
                m = torch.as_tensor(masks_np[s:s + chunk], device=self.dev, dtype=self.pv.dtype)
                lo = self.model(pixel_values=self._mask_pv(m)).logits
                outs.append(torch.softmax(lo, -1)[:, self.cls].cpu().numpy())
        return np.concatenate(outs)

    def hidden_ablate(self, clusters):
        """w_c = class-prob drop when latent cluster c is neutralized to mean latent
        at the last self-attention block output. len(clusters) forwards."""
        enc = self.model.perceiver.encoder
        w = np.zeros(len(clusters))
        for c, mem in enumerate(clusters):
            idx = torch.as_tensor(np.asarray(mem, int), device=self.dev)

            def hook(mod, inp, out, idx=idx):
                h = out[0] if isinstance(out, tuple) else out
                h2 = h.clone(); h2[:, idx, :] = h.mean(dim=1, keepdim=True)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            hk = enc.self_attends[-1].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    lo = self.model(pixel_values=self.pv).logits
            finally:
                hk.remove()
            w[c] = max(self.p_full - float(torch.softmax(lo[0], 0)[self.cls]), 0.0)
        return w

    def grad_saliency(self):
        pv = self.pv.clone().requires_grad_(True)
        lo = self.model(pixel_values=pv).logits
        torch.softmax(lo[0], 0)[self.cls].backward()
        g = pv.grad[0].abs().sum(0)                      # [224,224]
        g = g.reshape(self.grid, self.patch, self.grid, self.patch).mean(dim=(1, 3))
        return g.reshape(self.N).float().cpu().numpy()


def hdel(r, score):
    N = r.N
    order = np.argsort(-np.asarray(score, np.float64).reshape(-1))
    masks = np.ones((N + 1, N), np.float32)
    for k in range(1, N + 1):
        masks[k] = masks[k - 1]; masks[k, order[k - 1]] = 0.0
    pd_ = r.target_curve(masks)
    base, full = float(pd_[-1]), float(pd_[0]); den = max(full - base, 1e-6)
    xs = np.linspace(0, 1, N + 1)
    return float(np.trapz(np.clip((pd_ - base) / den, 0, 1), xs))


def necessity_score(r, ks=(8, 12, 16, 24)):
    """position-FREE: cluster latents -> hidden-ablation necessary weight -> map to
    patches via cross-attention. guard over k by probed deletion curve."""
    best, bauc = None, 1e9
    for k in ks:
        lab = kml(r.full_lat, k)
        clusters = [np.where(lab == c)[0] for c in range(k)]
        w = r.hidden_ablate(clusters)                   # [k] necessity per latent cluster
        xn = r.xattn / (r.xattn.max(1, keepdims=True) + 1e-8)
        score = np.zeros(r.N)
        for c in range(k):
            score += w[c] * xn[clusters[c]].sum(0)
        # quick guard: AUC over a few cut points
        order = np.argsort(-score)
        masks = np.stack([np.where(np.arange(r.N) < kk, 0.0, 1.0)[np.argsort(order)] for kk in (16, 32, 48, 72)])
        auc = r.target_curve(masks).mean()
        if auc < bauc:
            bauc, best = auc, score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nimg", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/perceiver_necessity.json")
    args = ap.parse_args()
    from transformers import PerceiverForImageClassificationLearned, PerceiverImageProcessor
    import random
    proc = PerceiverImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
    model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned").eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(123).shuffle(paths); imgs = paths[: args.nimg]
    rng = np.random.default_rng(0)
    acc = {"random": [], "xattn": [], "grad": [], "necessity": []}
    for ip in imgs:
        try:
            pv = proc(Image.open(ip).convert("RGB"), return_tensors="pt")["pixel_values"]
            r = PerceiverRunner(model, pv, args.device)
            acc["random"].append(np.mean([hdel(r, rng.random(r.N)) for _ in range(3)]))
            acc["xattn"].append(hdel(r, r.xattn.mean(0)))         # cross-attn rollout baseline
            acc["grad"].append(hdel(r, r.grad_saliency()))
            acc["necessity"].append(hdel(r, necessity_score(r)))
            print(f"[{ip.name}] rand {acc['random'][-1]:.3f} xattn {acc['xattn'][-1]:.3f} "
                  f"grad {acc['grad'][-1]:.3f} NEC {acc['necessity'][-1]:.3f}", flush=True)
        except Exception as e:
            print(f"[{ip.name}] ERR {type(e).__name__}: {e}", flush=True)
        finally:
            torch.cuda.empty_cache()
    from scipy.stats import wilcoxon
    a = {k: np.array(v) for k, v in acc.items()}
    def pw(x, y):
        try:
            return float(wilcoxon(x, y, alternative="greater").pvalue)
        except ValueError:
            return float("nan")
    rep = {"n": len(a["random"]), **{f"{k}_hdel": float(a[k].mean()) for k in acc}}
    rep["nec_beats_xattn_p"] = pw(a["xattn"], a["necessity"])
    rep["nec_beats_grad_p"] = pw(a["grad"], a["necessity"])
    rep["nec_genuine_p(rand)"] = pw(a["random"], a["necessity"])
    print(f"\n=== Perceiver necessity (n={rep['n']}, position-DECOUPLED, mean-baseline) ===")
    print(f"  random {rep['random_hdel']:.3f} | xattn-rollout {rep['xattn_hdel']:.3f} | "
          f"grad {rep['grad_hdel']:.3f} | NECESSITY {rep['necessity_hdel']:.3f}")
    print(f"  nec<xattn p {rep['nec_beats_xattn_p']:.2g} | nec<grad p {rep['nec_beats_grad_p']:.2g} | "
          f"nec<random p {rep['nec_genuine_p(rand)']:.2g}")
    args.out.write_text(json.dumps(rep, indent=1))
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
