#!/usr/bin/env python3
"""Cross-architecture two-faces test: is the necessity/sufficiency DIVERGENCE
caused by ATTENTION? CNN (ResNet, no attention) should COLLAPSE the two faces;
ViT (self-attn) should DIVERGE.

MASKING (mean-vector, OOD-safe). Two modes, chosen per the user's rule -- compare
RANDOM-masking collapse, pick the more on-manifold (higher random hdel):
  - "pixel" : mask input pixels -> normalized 0 (= ImageNet mean colour).
  - "token" : treat the STEM as a tokenizer and mask its OUTPUT to the per-image
              MEAN token vector. ViT: patch_embed output -> mean patch-embedding
              (== global_mean_h spirit). ResNet: conv1 output -> mean conv1 vector.
              This avoids pixel-level edge artifacts in conv filters.

Target = a DEEP unit (channel at its argmax location: ViT last-block token-dim,
ResNet layer4 channel). Per target: FRI insertion ERF (sufficiency), occlusion +
gradient (necessity), curves -> hins/hdel. Cross-arch metric: SUF<->NEC overlap
(spearman of FRI vs occlusion / gradient) + hdel(FRI order).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    "resnet50": ("resnet50.a1_in1k", "cnn"),
    "clip": ("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", "vit"),
    "dinov2_reg": ("vit_base_patch14_reg4_dinov2.lvd142m", "vit"),
    "siglip": ("vit_base_patch16_siglip_gap_224.webli", "vit"),
}
FRACS = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0], dtype=np.float32)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


class Arch:
    def __init__(self, key, device, mask_mode="token"):
        import timm
        name, kind = MODELS[key]
        self.key, self.kind, self.device, self.mask_mode = key, kind, device, mask_mode
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in name else {}
        self.model = timm.create_model(name, pretrained=True, **kw).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        if kind == "cnn":
            self.target_mod = self.model.layer4
            self.tok_mod = self.model.conv1
            self.gh = self.gw = 14
            self.tok_spatial = True
            self.prefix = 0
        else:
            self.target_mod = self.model.blocks[-1]
            self.tok_mod = self.model.patch_embed
            self.prefix = int(getattr(self.model, "num_prefix_tokens", 1))
            ps = 14 if "patch14" in name else 16
            self.gh = self.gw = 224 // ps
            self.tok_spatial = False
        self._act = {}
        self._cur_mask = None
        self.target_mod.register_forward_hook(
            lambda m, i, o: self._act.__setitem__("o", o if torch.is_tensor(o) else o[0]))
        self.tok_mod.register_forward_hook(self._tok_hook)

    def _tok_hook(self, module, inp, out):
        if self._cur_mask is None or self.mask_mode != "token":
            return out
        m = self._cur_mask
        if self.tok_spatial:                                   # [1,C,H,W]
            mean_vec = out.mean(dim=(2, 3), keepdim=True).detach()
            H, W = out.shape[2], out.shape[3]
            g = m.view(self.gh, self.gw)
            g = g.repeat_interleave(H // self.gh, 0).repeat_interleave(W // self.gw, 1).view(1, 1, H, W)
            return out * g + mean_vec * (1 - g)
        elif out.dim() == 4:                                   # [1,H,W,D] patch tokens
            mean_vec = out.mean(dim=(1, 2), keepdim=True).detach()
            g = m.view(1, self.gh, self.gw, 1)
            return out * g + mean_vec * (1 - g)
        else:                                                  # [1,N,D] patch tokens
            mean_vec = out.mean(dim=1, keepdim=True).detach()
            g = m.view(1, -1, 1)
            return out * g + mean_vec * (1 - g)

    def pixel_mask(self, mask_grid):
        pm = torch.as_tensor(mask_grid, device=self.device, dtype=torch.float32).view(self.gh, self.gw)
        ph, pw = 224 // self.gh, 224 // self.gw
        return pm.repeat_interleave(ph, 0).repeat_interleave(pw, 1)

    def _feat_tensor(self):
        o = self._act["o"]
        if self.kind == "cnn":
            return o[0].reshape(o.shape[1], -1).t()      # [H*W, C]
        return o[0, self.prefix:, :]                      # [P, D]

    def full_feat(self, x):
        self._cur_mask = None
        with torch.no_grad():
            self.model(x)
        return self._feat_tensor()

    def unit(self, x, mask_grid, chan, loc):
        """differentiable target activation under the mask (respects mask_mode)."""
        mg = mask_grid if torch.is_tensor(mask_grid) else torch.as_tensor(
            mask_grid, device=self.device, dtype=torch.float32)
        if self.mask_mode == "pixel":
            self._cur_mask = None
            pix = self.pixel_mask(mg).view(1, 1, 224, 224)
            self.model(x * pix)
        else:
            self._cur_mask = mg
            try:
                self.model(x)
            finally:
                self._cur_mask = None
        return self._feat_tensor()[loc, chan]


def make_obj(arch, x, chan, loc):
    N = arch.gh * arch.gw; dev = arch.device
    with torch.no_grad():
        full = arch.unit(x, torch.ones(N, device=dev), chan, loc)
        base = arch.unit(x, torch.zeros(N, device=dev), chan, loc)
    den = (full - base).abs().clamp(min=1e-6)

    def obj(mask):
        with torch.no_grad():
            return float(((arch.unit(x, mask, chan, loc) - base) / den).clamp(-0.2, 1.2).item())
    return obj, float(full), float(base)


def fri_solve(arch, x, chan, loc, steps=32, lr=0.4, lr_end=0.01, l1=0.003, seed=42):
    N = arch.gh * arch.gw; dev = arch.device
    with torch.no_grad():
        base = arch.unit(x, torch.zeros(N, device=dev), chan, loc)
        full = arch.unit(x, torch.ones(N, device=dev), chan, loc)
    den = (full - base).abs().clamp(min=1e-6)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    la = torch.zeros(N, device=dev)
    mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for step in range(steps):
        frac = step / max(steps - 1, 1)
        cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
        la_req = la.clone().requires_grad_(True)
        p = torch.sigmoid(la_req)
        budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
        state = (p / (p.sum() + 1e-8) * budget).clamp(max=1.0)
        with torch.enable_grad():
            r = (arch.unit(x, state, chan, loc) - base) / den
            loss = (1 - r) + l1 * p.sum()
            g = torch.autograd.grad(loss, la_req)[0].detach()
        t = step + 1
        mv = b1 * mv + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
        adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
        cmask = (adam * g > 0).float(); cmask = cmask * (N / cmask.sum().clamp(min=1.0))
        la = la - cur * adam * cmask
    return torch.sigmoid(la).detach().float().cpu().numpy()


def occlusion(arch, x, chan, loc):
    N = arch.gh * arch.gw; dev = arch.device
    with torch.no_grad():
        full = arch.unit(x, torch.ones(N, device=dev), chan, loc).item()
        drops = np.zeros(N, np.float32)
        ones = torch.ones(N, device=dev)
        for i in range(N):
            m = ones.clone(); m[i] = 0.0
            drops[i] = full - arch.unit(x, m, chan, loc).item()
    return drops


def gradinput(arch, x, chan, loc):
    N = arch.gh * arch.gw; dev = arch.device
    mask = torch.ones(N, device=dev, requires_grad=True)
    v = arch.unit(x, mask, chan, loc)
    g = torch.autograd.grad(v, mask)[0].abs()
    return g.detach().float().cpu().numpy()


def curves(obj, order, N):
    order = np.asarray(order)
    drec, irec = [], []
    for fr in FRACS:
        k = int(round(fr * N))
        dm = np.ones(N, np.float32); im = np.zeros(N, np.float32)
        if k > 0:
            dm[order[:k]] = 0.0; im[order[:k]] = 1.0
        drec.append(obj(dm)); irec.append(obj(im))
    return float(np.trapz(np.array(drec), FRACS)), float(np.trapz(np.array(irec), FRACS))


def run_model(arch, imgs, tf, ntarget, rng):
    dev = arch.device; N = arch.gh * arch.gw
    agg = {m: {"hdel": [], "hins": []} for m in ("fri", "occlusion", "grad", "random")}
    ov_occ, ov_grad = [], []
    from PIL import Image
    for ip in imgs:
        img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(dev)
        x = (img - arch.mean) / arch.std
        feat = arch.full_feat(x)
        top_ch = torch.argsort(-feat.var(0))[:ntarget].tolist()
        for chan in top_ch:
            loc = int(torch.argmax(feat[:, chan]).item())
            obj, full, base = make_obj(arch, x, chan, loc)
            if abs(full - base) < 1e-4:
                continue
            fri = fri_solve(arch, x, chan, loc)
            occ = occlusion(arch, x, chan, loc)
            grd = gradinput(arch, x, chan, loc)
            orders = {"fri": np.argsort(-fri), "occlusion": np.argsort(-occ),
                      "grad": np.argsort(-grd), "random": rng.permutation(N)}
            for m, o in orders.items():
                hd, hi = curves(obj, o, N)
                agg[m]["hdel"].append(hd); agg[m]["hins"].append(hi)
            ov_occ.append(spearman(fri, occ)); ov_grad.append(spearman(fri, grd))
        torch.cuda.empty_cache()
    rep = {m: {"hdel": float(np.mean(v["hdel"])), "hins": float(np.mean(v["hins"]))}
           for m, v in agg.items() if v["hdel"]}
    rep["overlap_fri_occ"] = float(np.mean(ov_occ)) if ov_occ else float("nan")
    rep["overlap_fri_grad"] = float(np.mean(ov_grad)) if ov_grad else float("nan")
    rep["n_targets"] = len(ov_occ); rep["kind"] = arch.kind
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--mask-modes", nargs="+", default=["token", "pixel"])
    ap.add_argument("--nimg", type=int, default=4)
    ap.add_argument("--ntarget", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/twofaces_crossarch.json")
    args = ap.parse_args()
    import random
    import torchvision.transforms as T
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(7).shuffle(paths); imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    report = {}
    for mk in args.models:
        for mode in args.mask_modes:
            arch = Arch(mk, args.device, mask_mode=mode)
            rng = np.random.default_rng(0)
            rep = run_model(arch, imgs, tf, args.ntarget, rng)
            report[f"{mk}::{mode}"] = rep
            print(f"\n=== {mk} [{mode}] ({rep['kind']}, n={rep['n_targets']}) ===")
            for m in ("fri", "occlusion", "grad", "random"):
                if m in rep:
                    print(f"   {m:11s} hdel={rep[m]['hdel']:.3f}  hins={rep[m]['hins']:.3f}")
            print(f"   overlap FRI<->occ={rep['overlap_fri_occ']:+.3f}  FRI<->grad={rep['overlap_fri_grad']:+.3f}", flush=True)
            del arch.model
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print("\n=== ON-MANIFOLD CHECK (random hdel; HIGHER = baseline more on-manifold) + TWO-FACES ===")
    print(f"{'model::mode':22s} {'kind':4s} {'random_hdel':>11s} {'fri_hdel':>9s} {'occ_hdel':>9s} {'fri_hins':>9s} {'ovlp_occ':>9s} {'ovlp_grad':>9s}")
    for k, r in report.items():
        if "fri" in r:
            print(f"{k:22s} {r['kind']:4s} {r['random']['hdel']:>11.3f} {r['fri']['hdel']:>9.3f} "
                  f"{r['occlusion']['hdel']:>9.3f} {r['fri']['hins']:>9.3f} {r['overlap_fri_occ']:>+9.3f} {r['overlap_fri_grad']:>+9.3f}")
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
