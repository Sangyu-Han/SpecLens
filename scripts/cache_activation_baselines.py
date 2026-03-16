#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.core.hooks.spec import parse_spec
from src.core.indexing.registry_utils import sanitize_layer_name
from src.core.runtime.activation_baselines import ActivationBaselineCache, RunningMeanAccumulator
from src.core.runtime.capture import LayerCapture
from src.core.runtime.wrappers import wrap_target_layer_with_sae
from src.utils.utils import load_obj, resolve_module


PACK_DEFAULT_CONFIGS = {
    "sam2": "configs/sam2_sav_feature_index.yaml",
    "clip": "configs/clip_imagenet_index.yaml",
    "dinov2": "configs/dinov2_imagenet_index.yaml",
}
DEFAULT_PACK = "sam2"

PACK_ADAPTERS = {
    "sam2": "src.packs.sam2.models.adapters:SAM2EvalAdapter",
    "clip": "src.packs.clip.models.adapters:CLIPVisionAdapter",
    "dinov2": "src.packs.dinov2.models.adapters:Dinov2VisionAdapter",
}


# ----------------------------- Helpers -----------------------------
def _anchor_attr_name(spec: str) -> str:
    parsed = parse_spec(spec)
    attr = (parsed.attr or "").lower()
    if attr in {"latent", "acts", "activation"}:
        return "acts"
    if attr in {"error_coeff", "error", "residual_coeff"}:
        return "error_coeff"
    if attr in {"residual", "sae_error"}:
        return "residual"
    return attr or "acts"


def _candidate_layer_dirs(root: Path, layer: str) -> List[Path]:
    candidates = [root / sanitize_layer_name(layer)]
    parsed = parse_spec(layer)
    if parsed.method is not None:
        candidates.append(root / sanitize_layer_name(parsed.base_with_branch))
    return candidates


def _load_sae_for_layer(cfg: Dict[str, Any], layer: str, device: torch.device) -> torch.nn.Module:
    root = Path(cfg["output"]["save_path"])
    ckpts: List[Path] = []
    for candidate in _candidate_layer_dirs(root, layer):
        files = sorted(candidate.glob("*.pt"))
        if files:
            ckpts = files
            break
    if not ckpts:
        raise FileNotFoundError(f"No SAE checkpoints found under {root} for layer '{layer}'")
    pkg = torch.load(ckpts[-1], map_location="cpu")
    sae_cfg = dict(pkg.get("sae_config", {}))
    act_size = int(pkg.get("act_size") or sae_cfg.get("act_size", 0))
    if act_size <= 0:
        raise RuntimeError(f"SAE checkpoint at {ckpts[-1]} is missing act_size metadata")
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    create_sae = load_obj(cfg["factory"])
    sae = create_sae(sae_cfg.get("sae_type", "batch-topk"), sae_cfg)
    state = pkg.get("sae_state", pkg.get("state_dict", {}))
    sae.load_state_dict(state, strict=False)
    sae.to(device).eval()
    for param in sae.parameters():
        param.requires_grad = False
    return sae


def _resolve_config(args: argparse.Namespace) -> str:
    if args.config:
        return args.config
    pack = args.pack or DEFAULT_PACK
    if pack not in PACK_DEFAULT_CONFIGS:
        raise ValueError(f"Unknown pack '{pack}'. Available: {sorted(PACK_DEFAULT_CONFIGS)}")
    return PACK_DEFAULT_CONFIGS[pack]


def _list_packs() -> None:
    print("Available pack defaults:")
    for name, path in sorted(PACK_DEFAULT_CONFIGS.items()):
        print(f"  {name:>8}: {path}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_layers(raw: Optional[str | Sequence[str]], default: Sequence[str]) -> List[str]:
    if raw is None:
        return [str(s) for s in default]
    if isinstance(raw, str):
        parts = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                parts.append(tok)
        return parts
    return [str(v) for v in raw]


def _load_model(model_cfg: Dict[str, Any], device: torch.device, logger: logging.Logger) -> torch.nn.Module:
    loader = load_obj(model_cfg["loader"])
    try:
        model = loader(model_cfg, device=device, logger=logger)
    except TypeError:
        model = loader(model_cfg, device)
    return model.to(device).eval()


def _build_dataloader(cfg: Dict[str, Any], *, num_workers_override: Optional[int] = None) -> Tuple[DataLoader, Optional[Any]]:
    ds_cfg = dict(cfg["dataset"])
    builder = load_obj(ds_cfg["builder"])
    collate_builder = load_obj(ds_cfg["collate_builder"]) if "collate_builder" in ds_cfg else None
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        ds, sampler = builder(ds_cfg, world_size=world_size, rank=rank, full_config=cfg)
    except TypeError:
        ds, sampler = builder(ds_cfg, world_size=world_size, rank=rank)
    collate_fn = collate_builder(ds) if collate_builder is not None else None
    batch_size = int(ds_cfg.get("batch_size", 1))
    num_workers = int(num_workers_override if num_workers_override is not None else ds_cfg.get("num_workers", 4))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, collate_fn


@dataclass
class AnchorInfo:
    spec: str
    branch: Any
    attr_name: str


def _setup_ddp() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache layer-wise activation baselines via moving average.")
    parser.add_argument("--config", type=str, default=None, help="Path to indexing/training config YAML.")
    parser.add_argument("--pack", type=str, choices=sorted(PACK_DEFAULT_CONFIGS.keys()), default='clip', help="Use default config for pack.")
    parser.add_argument("--list-packs", action="store_true", help="List supported pack defaults and exit.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer specs (default: use config).")
    parser.add_argument("--out", type=Path, default=None, help="Output path for cached baselines (.pt).")
    parser.add_argument("--device", type=str, default=None, help="Torch device to run on.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on number of batches to process.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    if args.list_packs:
        _list_packs()
        return

    config_path = Path(_resolve_config(args))
    cfg = _load_yaml(config_path)
    pack = args.pack or DEFAULT_PACK
    adapter_path = PACK_ADAPTERS.get(pack)
    if adapter_path is None:
        raise RuntimeError(f"No adapter registered for pack '{pack}'")
    adapter_cls = load_obj(adapter_path)

    rank, world, local = _setup_ddp()
    logger = logging.getLogger("activation_baseline_cache")
    logger.setLevel(logging.INFO)
    device = torch.device(args.device or cfg.get("model", {}).get("device", f"cuda:{local}" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model = _load_model(cfg["model"], device, logger)
    for p in model.parameters():
        p.requires_grad = False

    loader, _collate = _build_dataloader(cfg, num_workers_override=args.num_workers)

    target_layers = _parse_layers(args.layers, cfg["sae"]["layers"])
    attr_by_spec = {spec: _anchor_attr_name(spec) for spec in target_layers}

    anchors: Dict[str, AnchorInfo] = {}
    restore_handles: List[Any] = []
    frame_getter = None

    # Adapter handles preprocessing and any frame bookkeeping (SAM2).
    try:
        adapter = adapter_cls(model, device=device, collate_fn=None)
    except TypeError:
        adapter = adapter_cls(model, device=device)
    frame_getter = getattr(adapter, "current_frame_idx", None)

    for spec in target_layers:
        sae = _load_sae_for_layer(cfg["sae"], spec, device)
        capture = LayerCapture(spec)
        owner = resolve_module(model, capture.base)
        handle, branch = wrap_target_layer_with_sae(
            owner,
            capture=capture,
            sae=sae,
            controller=None,
            frame_getter=frame_getter,
        )
        restore_handles.append(handle)
        anchors[spec] = AnchorInfo(spec=spec, branch=branch, attr_name=attr_by_spec[spec])

    accumulator = RunningMeanAccumulator()
    steps = 0
    max_batches = None if args.max_batches is None else max(0, int(args.max_batches))

    progress = tqdm(total=len(loader), disable=rank != 0, desc="batches")
    with torch.no_grad():
        for bidx, raw_batch in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            batch_on_dev = adapter.preprocess_input(raw_batch)
            adapter.forward(batch_on_dev)
            for spec, anchor in anchors.items():
                ctx = anchor.branch.sae_context()
                tensor = ctx.get(anchor.attr_name)
                if tensor is not None:
                    accumulator.update(spec, anchor.attr_name, tensor)
                anchor.branch.clear_context()
            steps += 1
            if rank == 0:
                progress.update(1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    progress.close()

    accumulator.reduce_distributed()

    meta = {
        "config": str(config_path),
        "pack": pack,
        "device": str(device),
        "layers": target_layers,
        "steps": steps,
        "max_batches": max_batches,
        "world_size": world,
        "rank": rank,
    }
    cache = accumulator.to_cache(meta=meta)
    out_path = args.out or Path("outputs/activation_baselines") / f"{config_path.stem}.pt"
    if rank == 0:
        cache.save(out_path)

        print(f"[baseline] saved {len(cache.layers)} layers to {out_path}")
        for layer, attrs in cache.layers.items():
            for attr_name, stat in attrs.items():
                print(f"  - {layer}#{attr_name}: tokens={stat.count} shape={tuple(stat.mean.shape)}")

    for h in reversed(restore_handles):
        try:
            h()
        except Exception:
            pass
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
