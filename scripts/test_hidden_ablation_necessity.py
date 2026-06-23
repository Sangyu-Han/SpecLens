#!/usr/bin/env python3
"""Is there a MODEL-AGNOSTIC necessity method, better than grad/occlusion, that
reads the model's OWN compression by ABLATING the hidden layer?

Idea (user): the model already computed necessity (it routes inputs into its
representation). Input occlusion fails on redundancy (single patch substitutable);
but ABLATING the hidden position right before the readout removes the contribution
the model AGGREGATED there -> genuine necessity. Works for any spatially-aligned
hidden rep (ViT tokens AND CNN feature maps).

Target = the predicted CLASS logit. Necessity orders compared:
  - hidden_ablate : ablate each spatial position of the last-hidden (ViT: input to
                    last block; CNN: layer4 output) -> mean -> class-logit drop.
  - grad          : input x grad of the class logit (the inflow/grad baseline).
  - occlusion     : mask each input cell -> class-logit drop.
  - fri           : insertion FRI to the class logit (sufficiency, reference).
  - random
Eval: INPUT-deletion hdel (mean-baseline pixel mask) + insertion hins of the class.
Lower hdel = better necessity. Prediction: hidden_ablate < grad/occlusion.
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
    "deit3": ("deit3_base_patch16_224.fb_in22k_ft_in1k", "vit"),
    "augreg": ("vit_base_patch16_224.augreg2_in21k_ft_in1k", "vit"),
    "dinov2_reg": ("vit_base_patch14_reg4_dinov2.lvd142m", "vit"),
    "siglip": ("vit_base_patch16_siglip_gap_224.webli", "vit"),
}
FRACS = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0], dtype=np.float32)


class ClassArch:
    def __init__(self, key, device):
        import timm
        name, kind = MODELS[key]
        self.key, self.kind, self.device = key, kind, device
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in name else {}
        self.model = timm.create_model(name, pretrained=True, **kw).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        if kind == "cnn":
            self.gh = self.gw = 7                      # layer4 spatial grid
            self.abl_mod = self.model.layer4
            self.prefix = 0
        else:
            ps = 14 if "patch14" in name else 16
            self.gh = self.gw = 224 // ps
            self.abl_mod = self.model.blocks[-1]       # ablate at INPUT to last block
            self.prefix = int(getattr(self.model, "num_prefix_tokens", 1))
        self._abl = None  # (positions_mask[N], mean_vec) for hidden ablation
        if kind == "cnn":
            self.abl_mod.register_forward_hook(self._abl_hook_cnn)
        else:
            self.abl_mod.register_forward_pre_hook(self._abl_hook_vit)

    # ---- hidden ablation hooks (replace masked positions with mean vector) ----
    def _abl_hook_cnn(self, m, inp, out):
        if self._abl is None:
            return out
        keep, mean = self._abl                      # keep[N] 1=keep, mean[C]
        H, W = out.shape[2], out.shape[3]
        g = keep.view(1, 1, H, W)
        return out * g + mean.view(1, -1, 1, 1) * (1 - g)

    def _abl_hook_vit(self, m, args):
        if self._abl is None:
            return None
        x = args[0]
        keep, mean = self._abl                      # keep[N] over patch tokens
        g = torch.ones(x.shape[1], device=x.device)
        g[self.prefix:] = keep
        g = g.view(1, -1, 1)
        x = x * g + mean.view(1, 1, -1) * (1 - g)
        return (x,) + args[1:]

    def logits(self, x):
        return self.model(x)[0]

    def pixel_mask(self, mask_grid):
        pm = torch.as_tensor(mask_grid, device=self.device, dtype=torch.float32).view(self.gh, self.gw)
        ph, pw = 224 // self.gh, 224 // self.gw
        return pm.repeat_interleave(ph, 0).repeat_interleave(pw, 1)

    def class_of(self, x):
        with torch.no_grad():
            return int(self.logits(x).argmax().item())

    def masked_logit(self, x, mask_grid, c):
        pix = self.pixel_mask(mask_grid).view(1, 1, 224, 224)
        return self.logits(x * pix)[c]

    def obj_fn(self, x, c):
        N = self.gh * self.gw; dev = self.device
        with torch.no_grad():
            full = self.masked_logit(x, torch.ones(N, device=dev), c)
            base = self.masked_logit(x, torch.zeros(N, device=dev), c)
        den = (full - base).abs().clamp(min=1e-4)

        def obj(mask):
            with torch.no_grad():
                return float(((self.masked_logit(x, mask, c) - base) / den).clamp(-0.2, 1.2).item())
        return obj, float(full), float(base)

    # ---- necessity methods ----
    def hidden_ablate_nec(self, x, c):
        """ablate each hidden position -> class-logit drop (the model's own necessity)."""
        N = self.gh * self.gw; dev = self.device
        with torch.no_grad():
            # capture mean vector at the ablation site
            cap = {}
            if self.kind == "cnn":
                h = self.abl_mod.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o.detach()))
                self.model(x); h.remove()
                mean = cap["v"][0].mean(dim=(1, 2))               # [C]
            else:
                h = self.abl_mod.register_forward_pre_hook(lambda m, a: cap.__setitem__("v", a[0].detach()))
                self.model(x); h.remove()
                mean = cap["v"][0, self.prefix:, :].mean(dim=0)   # [D]
            full = self.logits(x)[c].item()
            nec = np.zeros(N, np.float32)
            for i in range(N):
                keep = torch.ones(N, device=dev); keep[i] = 0.0
                self._abl = (keep, mean)
                nec[i] = full - self.logits(x)[c].item()
            self._abl = None
        return nec

    def grad_nec(self, x, c):
        N = self.gh * self.gw; dev = self.device
        mask = torch.ones(N, device=dev, requires_grad=True)
        pix = self.pixel_mask(mask).view(1, 1, 224, 224)
        v = self.logits(x * pix)[c]
        g = torch.autograd.grad(v, mask)[0].abs()
        return g.detach().float().cpu().numpy()

    def occlusion_nec(self, x, c):
        N = self.gh * self.gw; dev = self.device
        with torch.no_grad():
            full = self.masked_logit(x, torch.ones(N, device=dev), c).item()
            nec = np.zeros(N, np.float32)
            ones = torch.ones(N, device=dev)
            for i in range(N):
                m = ones.clone(); m[i] = 0.0
                nec[i] = full - self.masked_logit(x, m, c).item()
        return nec

    def fri_solve(self, x, c, steps=32, lr=0.4, lr_end=0.01, l1=0.003, seed=42):
        N = self.gh * self.gw; dev = self.device
        with torch.no_grad():
            base = self.masked_logit(x, torch.zeros(N, device=dev), c)
            full = self.masked_logit(x, torch.ones(N, device=dev), c)
        den = (full - base).abs().clamp(min=1e-4)
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        la = torch.zeros(N, device=dev); mv = torch.zeros(N, device=dev); vv = torch.zeros(N, device=dev)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * frac))
            la_req = la.clone().requires_grad_(True)
            p = torch.sigmoid(la_req)
            budget = float(torch.rand(1, generator=gen, device=dev).item() * N)
            state = (p / (p.sum() + 1e-8) * budget).clamp(max=1.0)
            with torch.enable_grad():
                r = (self.masked_logit(x, state, c) - base) / den
                loss = (1 - r) + l1 * p.sum()
                g = torch.autograd.grad(loss, la_req)[0].detach()
            t = step + 1
            mv = b1 * mv + (1 - b1) * g; vv = b2 * vv + (1 - b2) * g * g
            adam = (mv / (1 - b1 ** t)) / ((vv / (1 - b2 ** t)).sqrt() + eps)
            cm = (adam * g > 0).float(); cm = cm * (N / cm.sum().clamp(min=1.0))
            la = la - cur * adam * cm
        return torch.sigmoid(la).detach().float().cpu().numpy()


def curves(obj, order, N):
    order = np.asarray(order); drec, irec = [], []
    for fr in FRACS:
        k = int(round(fr * N))
        dm = np.ones(N, np.float32); im = np.zeros(N, np.float32)
        if k > 0:
            dm[order[:k]] = 0.0; im[order[:k]] = 1.0
        drec.append(obj(dm)); irec.append(obj(im))
    return float(np.trapz(np.array(drec), FRACS)), float(np.trapz(np.array(irec), FRACS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["resnet50", "clip", "dinov2_reg", "siglip"])
    ap.add_argument("--nimg", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/hidden_ablation_necessity.json")
    args = ap.parse_args()
    import random
    import torchvision.transforms as T
    from PIL import Image
    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(11).shuffle(paths); imgs = paths[: args.nimg]
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    methods = ["hidden_ablate", "grad", "occlusion", "fri", "random"]
    report = {}
    for mk in args.models:
        arch = ClassArch(mk, args.device); dev = args.device; N = arch.gh * arch.gw
        agg = {m: {"hdel": [], "hins": []} for m in methods}
        rng = np.random.default_rng(0)
        for ip in imgs:
            img = tf(Image.open(ip).convert("RGB")).unsqueeze(0).to(dev)
            x = (img - arch.mean) / arch.std
            c = arch.class_of(x)
            obj, full, base = arch.obj_fn(x, c)
            if abs(full - base) < 1e-3:
                continue
            orders = {
                "hidden_ablate": np.argsort(-arch.hidden_ablate_nec(x, c)),
                "grad": np.argsort(-arch.grad_nec(x, c)),
                "occlusion": np.argsort(-arch.occlusion_nec(x, c)),
                "fri": np.argsort(-arch.fri_solve(x, c)),
                "random": rng.permutation(N),
            }
            for m, o in orders.items():
                hd, hi = curves(obj, o, N)
                agg[m]["hdel"].append(hd); agg[m]["hins"].append(hi)
            torch.cuda.empty_cache()
        rep = {m: {"hdel": float(np.mean(v["hdel"])), "hins": float(np.mean(v["hins"])), "n": len(v["hdel"])}
               for m, v in agg.items() if v["hdel"]}
        report[mk] = {"kind": arch.kind, "grid": N, **rep}
        print(f"\n=== {mk} ({arch.kind}, grid {arch.gh}x{arch.gw}, n={rep['grad']['n']}) "
              f"input-deletion hdel (LOWER=better necessity) / insertion hins ===")
        for m in methods:
            if m in rep:
                star = "  <== model-agnostic necessity" if m == "hidden_ablate" else ""
                print(f"   {m:14s} hdel={rep[m]['hdel']:.3f}  hins={rep[m]['hins']:.3f}{star}", flush=True)
        del arch.model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print("\n=== HIDDEN-ABLATION vs grad/occlusion (input-deletion hdel, lower=better) ===")
    print(f"{'model':12s} {'kind':4s} {'hidden_abl':>10s} {'grad':>7s} {'occ':>7s} {'fri':>7s} {'random':>7s}")
    for mk, r in report.items():
        if "grad" in r:
            print(f"{mk:12s} {r['kind']:4s} {r['hidden_ablate']['hdel']:>10.3f} {r['grad']['hdel']:>7.3f} "
                  f"{r['occlusion']['hdel']:>7.3f} {r['fri']['hdel']:>7.3f} {r['random']['hdel']:>7.3f}")
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
