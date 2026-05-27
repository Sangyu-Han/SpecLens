"""Recon metrics for the paper's per-layer batch-topk SAEs on ImageNet val.

Self-contained: only depends on torch + timm + PIL/torchvision. No SpecLens src.
SAE module is reimplemented inline to match the BatchTopK eval-mode forward.

Outputs JSON with per-(backbone, block) means.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------- #
# SAE module — minimal reimplementation matching SpecLens BatchTopKSAE
#   * preprocess_input with input_unit_norm = True (per-token mean/std)
#   * encode: pre = (x - b_dec) @ W_enc + b_enc; eval-mode hard threshold gate
#   * decode: acts @ W_dec + b_dec, then unscale
# ----------------------------------------------------------------------- #
class BatchTopKSAEEval(nn.Module):
    def __init__(self, act_size: int, dict_size: int, input_unit_norm: bool = True):
        super().__init__()
        self.act_size = act_size
        self.dict_size = dict_size
        self.input_unit_norm = input_unit_norm
        self.W_enc = nn.Parameter(torch.zeros(act_size, dict_size))
        self.W_dec = nn.Parameter(torch.zeros(dict_size, act_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.b_dec = nn.Parameter(torch.zeros(act_size))
        self.register_buffer("threshold", torch.tensor(0.0))

    def _preprocess(self, x):
        if not self.input_unit_norm:
            return x, None, None
        x_mean = x.mean(dim=-1, keepdim=True)
        xc = x - x_mean
        x_std = xc.std(dim=-1, keepdim=True)
        xn = xc / (x_std + 1e-5)
        return xn, x_mean, x_std

    def _postprocess(self, x_hat, x_mean, x_std):
        if not self.input_unit_norm:
            return x_hat
        return x_hat * x_std + x_mean

    @torch.no_grad()
    def forward(self, x):
        xn, mu, sd = self._preprocess(x)
        pre = (xn - self.b_dec) @ self.W_enc + self.b_enc
        # eval-mode hard threshold gate (BatchTopK's expected sparsity at inference)
        relu_pre = F.relu(pre)
        gate = (pre > self.threshold).to(pre.dtype)
        acts = relu_pre * gate
        x_hat_n = acts @ self.W_dec + self.b_dec
        x_hat = self._postprocess(x_hat_n, mu, sd)
        # return both standardized-space pair (xn, x_hat_n) and original-space pair (x, x_hat)
        return x_hat, acts, xn, x_hat_n


def load_sae_from_ckpt(ckpt_path: Path, device: str):
    pkg = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = pkg["sae_config"]
    sae = BatchTopKSAEEval(
        act_size=int(cfg["act_size"]),
        dict_size=int(cfg["dict_size"]),
        input_unit_norm=bool(cfg.get("input_unit_norm", False)),
    )
    state = pkg["sae_state"]
    keep = {k: v for k, v in state.items() if k in {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"}}
    sae.load_state_dict(keep, strict=False)
    sae.to(device).eval()
    return sae, pkg


# ----------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------- #
class ValImagesDataset(Dataset):
    def __init__(self, root: Path, image_size: int, mean, std, limit: int):
        from torchvision import transforms
        from PIL import Image

        self.Image = Image
        resize = int(round(image_size * 256 / 224))
        self.tf = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        files = []
        for cls in sorted(os.listdir(root)):
            cls_dir = root / cls
            if not cls_dir.is_dir():
                continue
            for fn in sorted(os.listdir(cls_dir)):
                if fn.endswith((".JPEG", ".jpeg", ".jpg", ".png")):
                    files.append(cls_dir / fn)
                    if len(files) >= limit:
                        break
            if len(files) >= limit:
                break
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.Image.open(self.files[i]).convert("RGB")
        return self.tf(img)


# ----------------------------------------------------------------------- #
# Backbone specs
# ----------------------------------------------------------------------- #
BACKBONES = {
    "clip": dict(
        timm_name="vit_base_patch16_clip_224",
        image_size=224,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        sae_root="/media/mipal/1TB/sangyu/SpecLens_outputs/clip_50k_sae",
    ),
    "siglip": dict(
        timm_name="vit_base_patch16_siglip_224",
        image_size=224,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        sae_root="/media/mipal/1TB/sangyu/SpecLens_outputs/siglip_50k_sae",
    ),
    "dinov3": dict(
        timm_name="vit_small_patch16_dinov3",
        image_size=256,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        sae_root="/media/mipal/1TB/sangyu/SpecLens_outputs/dinov3_50k_sae",
    ),
}
BLOCKS = [2, 6, 10]


def build_timm_model(name: str, device: str):
    import timm
    model = timm.create_model(name, pretrained=True)
    model.eval().to(device)
    # Disable any internal output post-processing so we just need the blocks to fire.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def hook_block(model, block_idx: int):
    """Register hook on model.blocks[block_idx]; return (handle, holder list)."""
    holder = []

    def fn(_m, _i, out):
        holder.clear()
        holder.append(out.detach())

    handle = model.blocks[block_idx].register_forward_hook(fn)
    return handle, holder


@torch.no_grad()
def evaluate_backbone(name: str, spec: dict, num_images: int, batch_size: int, device: str):
    print(f"\n=== {name} ({spec['timm_name']}) ===", flush=True)
    model = build_timm_model(spec["timm_name"], device)

    hooks = {}
    holders = {}
    for b in BLOCKS:
        h, hold = hook_block(model, b)
        hooks[b] = h
        holders[b] = hold

    saes = {}
    for b in BLOCKS:
        ckpt = sorted(Path(spec["sae_root"], f"model.blocks.{b}").glob("step_*.pt"))[-1]
        sae, pkg = load_sae_from_ckpt(ckpt, device)
        saes[b] = (sae, pkg["step"], ckpt.name)
        print(f"  block {b}: {ckpt.name} d={sae.act_size} dict={sae.dict_size} thr={float(sae.threshold):.4f}", flush=True)

    val_root = Path(os.path.expanduser("~/data/ILSVRC2012/val"))
    ds = ValImagesDataset(val_root, image_size=spec["image_size"], mean=spec["mean"], std=spec["std"], limit=num_images)
    print(f"  val images: {len(ds)}", flush=True)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=False, drop_last=False)

    # Per-block accumulators (collect both original-space and standardized-space stats)
    accum = {}
    for b in BLOCKS:
        d = saes[b][0].act_size
        m = saes[b][0].dict_size
        accum[b] = dict(
            n_tokens=0,
            # original-space
            sum_sq_err=0.0,
            sum_x=torch.zeros(d, device=device, dtype=torch.float64),
            sum_xx=0.0,
            sum_cos=0.0,
            sum_rel_l2=0.0,    # SAELens-style per-token ||x-xhat||^2 / ||x||^2
            # standardized-space (matches train-time EV)
            sum_sq_err_n=0.0,
            sum_xn=torch.zeros(d, device=device, dtype=torch.float64),
            sum_xxn=0.0,
            # other
            sum_l0=0.0,
            active_mask=torch.zeros(m, dtype=torch.bool, device=device),
        )

    t0 = time.time()
    for bi, imgs in enumerate(dl):
        imgs = imgs.to(device, non_blocking=True)
        _ = model(imgs)
        for b in BLOCKS:
            x = holders[b][0]                         # (B, T, D)
            x = x.reshape(-1, x.shape[-1]).float()
            sae, _, _ = saes[b]
            x_hat, acts, xn, x_hat_n = sae(x)
            # original space
            err = (x - x_hat).pow(2).sum().item()
            xx = x.pow(2).sum().item()
            cos = F.cosine_similarity(x, x_hat, dim=-1).sum().item()
            per_tok_resid_sq = (x - x_hat).pow(2).sum(-1)
            per_tok_x_sq = x.pow(2).sum(-1)
            rel_l2 = (per_tok_resid_sq / (per_tok_x_sq + 1e-8)).sum().item()
            # standardized space (train-time EV uses these)
            err_n = (xn - x_hat_n).pow(2).sum().item()
            xxn = xn.pow(2).sum().item()
            accum[b]["n_tokens"] += x.shape[0]
            accum[b]["sum_sq_err"] += err
            accum[b]["sum_xx"] += xx
            accum[b]["sum_x"] += x.sum(dim=0).double()
            accum[b]["sum_cos"] += cos
            accum[b]["sum_rel_l2"] += rel_l2
            accum[b]["sum_sq_err_n"] += err_n
            accum[b]["sum_xxn"] += xxn
            accum[b]["sum_xn"] += xn.sum(dim=0).double()
            accum[b]["active_mask"] |= (acts != 0).any(dim=0)
            accum[b]["sum_l0"] += (acts != 0).float().sum(dim=-1).sum().item()
        if (bi + 1) % 20 == 0:
            print(f"    batch {bi+1}/{len(dl)} elapsed={time.time()-t0:.1f}s", flush=True)

    for h in hooks.values():
        h.remove()

    out = {}
    for b in BLOCKS:
        a = accum[b]
        n = a["n_tokens"]
        d = saes[b][0].act_size
        m = saes[b][0].dict_size
        # Original-space EV (over all tokens, total variance)
        mean_x = a["sum_x"] / n
        var_x_orig = a["sum_xx"] / n - mean_x.pow(2).sum().item()
        mse_token = a["sum_sq_err"] / n
        fvu_orig = mse_token / var_x_orig if var_x_orig > 0 else float("nan")
        ev_orig = 1.0 - fvu_orig
        # Standardized-space EV (matches train-time logging convention)
        mean_xn = a["sum_xn"] / n
        var_xn = a["sum_xxn"] / n - mean_xn.pow(2).sum().item()
        mse_token_n = a["sum_sq_err_n"] / n
        fvu_std = mse_token_n / var_xn if var_xn > 0 else float("nan")
        ev_std = 1.0 - fvu_std
        # Other
        cos = a["sum_cos"] / n
        rel_l2 = a["sum_rel_l2"] / n
        dead = float((~a["active_mask"]).sum().item()) / m
        l0 = a["sum_l0"] / n
        out[b] = dict(
            n_tokens=int(n),
            act_dim=int(d),
            dict_size=int(m),
            step=int(saes[b][1]),
            mse_token=float(mse_token),
            mse_per_coord=float(mse_token / d),
            explained_var_orig=float(ev_orig),
            fvu_orig=float(fvu_orig),
            explained_var_std=float(ev_std),
            fvu_std=float(fvu_std),
            relative_l2=float(rel_l2),
            cosine=float(cos),
            l0=float(l0),
            dead_frac=float(dead),
            ckpt=str(saes[b][2]),
        )
        print(
            f"  -> block {b}: ev_std={ev_std:.4f} ev_orig={ev_orig:.4f} rel_l2={rel_l2:.4f} cos={cos:.4f} L0={l0:.1f} dead={dead*100:.2f}% (n={n})",
            flush=True,
        )

    del model, saes
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-images", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=str, default="paper_sae_recon_eval.json")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES.keys()))
    args = ap.parse_args()

    results = {}
    for bb in args.backbones:
        results[bb] = evaluate_backbone(
            bb, BACKBONES[bb],
            num_images=args.num_images,
            batch_size=args.batch_size,
            device=args.device,
        )

    payload = dict(args=vars(args), results=results)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
