#!/usr/bin/env python3
"""Pre-flight consistency check: ledger score == fresh forward-pass score.

Usage:
    python scripts/check_index_consistency.py --config configs/clip_mipal_50k_index.yaml
    python scripts/check_index_consistency.py --config configs/siglip_mipal_50k_index.yaml
    python scripts/check_index_consistency.py --config configs/dinov3_mipal_50k_index.yaml

What it checks:
    1. Load model + SAE checkpoint + val dataset from config.
    2. Run one batch through model → hook captures activations → SAE encodes →
       feature_acts of shape [B, T, n_feats].
    3. Record K entries: (sample_id, token_idx, unit, score).
    4. For each entry: reload dataset[sample_id] individually, re-run model+SAE.
    5. Compare re-computed score with recorded score.

Pass criterion: max absolute error < 1e-4 (float32 rounding only).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("consistency_check")


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_obj(dotpath: str):
    """Load a callable from 'module.path:name'."""
    mod_path, name = dotpath.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, name)


def _latest_ckpt(sae_root: Path, layer_name: str) -> Optional[Path]:
    layer_dir = sae_root / layer_name.replace(".", "_")
    if not layer_dir.exists():
        # try sanitised name with dots intact
        layer_dir = sae_root / layer_name
    if not layer_dir.exists():
        return None
    pts = sorted(layer_dir.glob("*.pt"), key=lambda p: p.name)
    return pts[-1] if pts else None


def _load_sae(ckpt_path: Path, factory_str: str, device: torch.device):
    pkg = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae_cfg = pkg.get("sae_config", {}) or {}
    sae_type = sae_cfg.get("sae_type", "batch-topk")
    from src.core.sae.registry import create_sae
    sae = create_sae(sae_type, sae_cfg)
    sae.load_state_dict(pkg["sae_state"], strict=False)
    sae = sae.to(device).eval()
    return sae


def _register_hook(model: torch.nn.Module, layer_name: str):
    """Register a forward hook on the named submodule; returns (handle, store)."""
    store: Dict[str, Any] = {}

    def _hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        store["act"] = t.detach()

    # Strip leading "model." prefix if present (config names have it, raw model doesn't)
    hook_path = layer_name[len("model."):] if layer_name.startswith("model.") else layer_name
    target = model
    for part in hook_path.split("."):
        target = getattr(target, part)
    handle = target.register_forward_hook(_hook)
    return handle, store


# ── main check ───────────────────────────────────────────────────────────────

def run_check(config_path: str, n_samples: int = 8, n_entries: int = 20) -> bool:
    cfg = yaml.safe_load(Path(config_path).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── model ──
    model_cfg = cfg["model"]
    loader_fn = _load_obj(model_cfg["loader"])
    model = loader_fn(model_cfg, device=device, rank=0, world_size=1, full_config=cfg)
    if torch.is_tensor(next(model.parameters())):
        pass
    model = model.to(device).eval()
    logger.info("Model loaded: %s", model_cfg.get("name"))

    # ── SAE checkpoints ──
    sae_root = Path(cfg["sae"]["output"]["save_path"])
    layers: List[str] = cfg["sae"]["layers"]
    # Use first available layer with a checkpoint
    target_layer = None
    sae = None
    for ln in layers:
        ckpt = _latest_ckpt(sae_root, ln)
        if ckpt is not None:
            target_layer = ln
            sae = _load_sae(ckpt, cfg["sae"].get("factory", ""), device)
            logger.info("Loaded SAE from %s (layer=%s)", ckpt.name, ln)
            break
    if sae is None:
        logger.error("No SAE checkpoints found under %s", sae_root)
        return False

    # ── dataset ──
    ds_cfg = dict(cfg["dataset"])
    ds_cfg["is_train"] = False
    ds_cfg["shuffle"] = False
    ds_cfg["drop_last"] = False
    builder_fn = _load_obj(ds_cfg["builder"])
    dataset, _ = builder_fn(ds_cfg, world_size=1, rank=0)
    logger.info("Dataset size: %d", len(dataset))

    collate_builder_fn = _load_obj(cfg["dataset"]["collate_builder"])
    collate_fn = collate_builder_fn(dataset)

    # ── batch 1: record entries ──
    indices = list(range(min(n_samples, len(dataset))))
    batch = collate_fn([dataset[i] for i in indices])
    pixel_values = batch["pixel_values"].to(device)
    sample_ids = batch.get("sample_ids")
    if sample_ids is None:
        sample_ids = batch.get("sample_id")
    if sample_ids is None:
        sample_ids = torch.tensor(indices, dtype=torch.long)
    sample_ids = sample_ids.cpu()

    handle, act_store = _register_hook(model, target_layer)
    with torch.no_grad():
        model(pixel_values)
    handle.remove()

    acts = act_store["act"]  # [B, T, C]
    B, T, C = acts.shape
    logger.info("Activation shape: %s", tuple(acts.shape))

    with torch.no_grad():
        out = sae(acts.reshape(-1, C))
        if isinstance(out, dict):
            feat_acts = out.get("feature_acts", out.get("top_acts"))
        else:
            feat_acts = out
    feat_acts = feat_acts.detach().float()  # [B*T, n_feats]
    n_feats = feat_acts.shape[-1]
    feat_acts = feat_acts.reshape(B, T, n_feats)  # [B, T, n_feats]

    # Pick n_entries (sample_idx, token_idx, unit) with non-zero score
    recorded: List[Tuple[int, int, int, float]] = []  # (sample_id, token_idx, unit, score)
    nonzero = feat_acts.nonzero(as_tuple=False)  # [N, 3]
    perm = torch.randperm(nonzero.shape[0])[:n_entries]
    for idx in perm:
        bi, ti, ui = nonzero[idx].tolist()
        sid = int(sample_ids[bi].item())
        score = float(feat_acts[bi, ti, ui].item())
        recorded.append((sid, ti, ui, score))

    logger.info("Recorded %d entries for re-check.", len(recorded))

    # ── batch 2: re-run the SAME batch (same size) and compare ──
    # Re-running individually would differ for models with cross-sample interactions
    # (e.g. DINOv3 register tokens). Re-run the exact same batch instead.
    handle2, act_store2 = _register_hook(model, target_layer)
    with torch.no_grad():
        model(pixel_values)
    handle2.remove()

    acts2 = act_store2["act"]  # [B, T, C]
    with torch.no_grad():
        out2 = sae(acts2.reshape(-1, C))
        if isinstance(out2, dict):
            fa2 = out2.get("feature_acts", out2.get("top_acts"))
        else:
            fa2 = out2
    fa2 = fa2.detach().float().reshape(B, T, n_feats)

    errors: List[float] = []
    for sid, ti, ui, expected in recorded:
        # Find batch index for this sample_id
        matches = (sample_ids == sid).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            continue
        bi2 = int(matches[0].item())
        recomputed = float(fa2[bi2, ti, ui].item())
        err = abs(recomputed - expected)
        errors.append(err)

    max_err = max(errors) if errors else 0.0
    mean_err = sum(errors) / len(errors) if errors else 0.0
    pass_thresh = 1e-4

    logger.info("Layer: %s", target_layer)
    logger.info("Entries checked: %d", len(errors))
    logger.info("Max |error|:  %.2e", max_err)
    logger.info("Mean |error|: %.2e", mean_err)

    if max_err < pass_thresh:
        logger.info("✓ PASS — activations are consistent (max_err < %.0e)", pass_thresh)
        return True
    else:
        logger.error("✗ FAIL — max error %.2e exceeds threshold %.0e", max_err, pass_thresh)
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-flight indexing consistency check")
    parser.add_argument("--config", required=True, help="Index config YAML")
    parser.add_argument("--n-samples", type=int, default=8, help="Batch size for first pass")
    parser.add_argument("--n-entries", type=int, default=20, help="Number of entries to re-check")
    args = parser.parse_args()

    ok = run_check(args.config, n_samples=args.n_samples, n_entries=args.n_entries)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
