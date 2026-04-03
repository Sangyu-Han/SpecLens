"""
Verify that scores in the Parquet ledger match actual SAE activations.

Usage:
    python scripts/verify_index.py \
        --index_dir outputs/clip_imagenet_topk_index \
        --sae_dir outputs/clip_imagenet_topk \
        --config configs/clip_imagenet_topk_index.yaml \
        --n_samples 50
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.sae.registry import create_sae


def build_transform(cfg):
    """Match the exact transform used in build_indexing_dataset (is_train=False)."""
    image_size = cfg["dataset"].get("image_size", 224)
    mean = cfg["dataset"].get("mean", [0.48145466, 0.4578275, 0.40821073])
    std  = cfg["dataset"].get("std",  [0.26862954, 0.26130258, 0.27577711])
    interp_str = cfg["dataset"].get("interpolation", "bicubic")
    interp = T.InterpolationMode.BICUBIC if interp_str == "bicubic" else T.InterpolationMode.BILINEAR
    # Val transform: Resize to 256 (= image_size * 256/224) → CenterCrop to 224
    resize_size = int(image_size * 256 / 224)
    return T.Compose([
        T.Resize(resize_size, interpolation=interp, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_model(cfg, device):
    import timm
    name = cfg["model"]["name"]
    model = timm.create_model(name, pretrained=cfg["model"].get("pretrained", True))
    model.eval().to(device)
    return model


def load_sae(sae_dir, layer_name, device):
    """Load latest SAE checkpoint for a given layer."""
    ckpts = sorted(glob.glob(os.path.join(sae_dir, layer_name, "step_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found for {layer_name} in {sae_dir}")
    ckpt_path = ckpts[-1]
    state = torch.load(ckpt_path, map_location="cpu")
    sae_cfg = dict(state["sae_config"])
    sae_cfg["act_size"] = state["act_size"]
    sae_cfg["device"] = str(device)
    sae_type = sae_cfg.pop("sae_type", "topk")
    sae = create_sae(sae_type, sae_cfg)
    sae.load_state_dict(state["sae_state"])
    sae.eval().to(device)
    return sae, ckpt_path


def get_layer_output(model, layer_name, x):
    """Forward x through model, capture output of layer_name."""
    acts = {}

    def hook(module, input, output):
        acts["out"] = output

    # Register hook on target layer
    # layer_name may start with "model." (training convention) — strip it
    layer_key = layer_name.removeprefix("model.")
    parts = layer_key.split(".")
    module = model
    for p in parts:
        module = getattr(module, p)
    handle = module.register_forward_hook(hook)

    with torch.no_grad():
        model(x)
    handle.remove()

    return acts["out"]  # [1, num_tokens, dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="outputs/clip_imagenet_topk_index")
    parser.add_argument("--sae_dir",   default="outputs/clip_imagenet_topk")
    parser.add_argument("--config",    default="configs/clip_imagenet_topk_index.yaml")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--layers", nargs="+", default=None,
                        help="layers to verify, e.g. model.blocks.0 model.blocks.5")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── Load offline_meta (sample_id → path) ──
    meta_files = glob.glob(os.path.join(args.index_dir, "offline_meta", "parquet", "**", "*.parquet"), recursive=True)
    meta_df = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    meta_df = meta_df.drop_duplicates("sample_id").set_index("sample_id")
    print(f"offline_meta loaded: {len(meta_df)} samples")

    # ── Load ledger ──
    ledger_files = glob.glob(os.path.join(args.index_dir, "deciles", "**", "*.parquet"), recursive=True)
    ledger_df = pd.concat([pd.read_parquet(f) for f in ledger_files], ignore_index=True)
    print(f"ledger loaded: {len(ledger_df)} rows across {ledger_df.layer.nunique()} layers")

    # Filter to requested layers
    if args.layers:
        ledger_df = ledger_df[ledger_df.layer.isin(args.layers)]

    # Sample rows to verify (prefer high scores for robust testing)
    sample_df = ledger_df.sample(n=min(args.n_samples, len(ledger_df)), random_state=42)

    # ── Load model ──
    print("Loading ViT model...")
    model = load_model(cfg, device)
    transform = build_transform(cfg)

    # ── Verify per layer ──
    layers_to_check = sample_df.layer.unique().tolist()
    sae_cache = {}

    errors = []
    max_abs_err = []
    rel_errs = []

    for i, row in enumerate(sample_df.itertuples()):
        layer = row.layer
        unit  = int(row.unit)
        score = float(row.score)
        sid   = int(row.sample_id)
        tok_x = int(row.x)  # token index (0 = CLS for CLIP)

        # Get image path
        if sid not in meta_df.index:
            print(f"[SKIP] sample_id {sid} not in offline_meta")
            continue
        img_path = meta_df.loc[sid, "path"]

        # Load SAE for this layer (cached)
        if layer not in sae_cache:
            sae, ckpt = load_sae(args.sae_dir, layer, device)
            sae_cache[layer] = sae
            print(f"  SAE loaded: {layer} ({ckpt})")

        sae = sae_cache[layer]

        # Forward pass
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        layer_out = get_layer_output(model, layer, x)  # [1, T, 768]
        acts_layer = layer_out[0]  # [T, 768]

        # SAE encode
        with torch.no_grad():
            feat_acts = sae.encode(acts_layer)  # [T, D]

        recomputed = float(feat_acts[tok_x, unit].item())
        abs_err = abs(recomputed - score)
        rel_err = abs_err / (abs(score) + 1e-8)

        max_abs_err.append(abs_err)
        rel_errs.append(rel_err)

        status = "OK" if abs_err < 1e-3 else ("WARN" if abs_err < 0.1 else "FAIL")
        if status != "OK" or i < 5:
            print(f"[{status}] layer={layer} unit={unit:5d} tok={tok_x:3d} "
                  f"stored={score:.5f} recomputed={recomputed:.5f} "
                  f"abs_err={abs_err:.2e} rel_err={rel_err:.2e}  {os.path.basename(img_path)}")

    print("\n── Summary ──")
    print(f"  n verified : {len(max_abs_err)}")
    print(f"  max abs err: {max(max_abs_err):.4e}")
    print(f"  mean abs err: {sum(max_abs_err)/len(max_abs_err):.4e}")
    print(f"  mean rel err: {sum(rel_errs)/len(rel_errs):.4e}")
    n_fail = sum(1 for e in max_abs_err if e >= 0.1)
    n_warn = sum(1 for e in max_abs_err if 1e-3 <= e < 0.1)
    print(f"  FAIL (err≥0.1) : {n_fail}")
    print(f"  WARN (0.001≤err<0.1): {n_warn}")
    print(f"  OK   (err<0.001): {len(max_abs_err) - n_fail - n_warn}")


if __name__ == "__main__":
    main()
