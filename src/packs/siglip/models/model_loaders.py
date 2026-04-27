from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import timm
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def _load_state_dict(model: nn.Module, ckpt_path: Path, *, strict: bool = False) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    LOGGER.info(
        "[siglip] load_state_dict strict=%s missing=%d unexpected=%d", strict, len(missing), len(unexpected)
    )


def load_siglip_model(
    model_cfg: Dict[str, Any],
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> nn.Module:
    """Create a timm SigLIP Vision Transformer and move it to the requested device."""
    name = model_cfg.get("name", "vit_base_patch16_siglip_224")
    pretrained = bool(model_cfg.get("pretrained", True))
    init_kwargs = dict(model_cfg.get("init_kwargs") or {})

    # Match the training/indexing environment used for the SigLIP 50k runs.
    # In newer timm versions this model name resolves to the `v2_webli` weights,
    # but older local timm builds may still default to the older `webli` entry.
    # Force the pretrained source explicitly so local validation uses the same
    # variant as the server-side training/indexing setup.
    if str(name) == "vit_base_patch16_siglip_224":
        overlay = dict(init_kwargs.get("pretrained_cfg_overlay") or {})
        overlay.setdefault("hf_hub_id", "timm/vit_base_patch16_siglip_224.v2_webli")
        overlay.setdefault("tag", "v2_webli")
        overlay.setdefault("hf_hub_filename", "pytorch_model.bin")
        init_kwargs["pretrained_cfg_overlay"] = overlay

    ckpt_path = model_cfg.get("ckpt") or model_cfg.get("checkpoint")
    if ckpt_path:
        init_kwargs.setdefault("pretrained", False)
    else:
        init_kwargs.setdefault("pretrained", pretrained)

    try:
        model = timm.create_model(name, **init_kwargs)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to create timm model '{name}' with args {init_kwargs}") from exc

    model.eval()
    model.to(device)

    if ckpt_path:
        path = Path(ckpt_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        _load_state_dict(model, path, strict=bool(model_cfg.get("strict_load", False)))
    elif pretrained and rank == 0:
        LOGGER.info("[siglip] Loaded pretrained weights for %s", name)

    dtype = model_cfg.get("dtype")
    if dtype:
        try:
            model = model.to(getattr(torch, dtype))
        except AttributeError as exc:
            raise ValueError(f"Unsupported dtype '{dtype}' for SigLIP model conversion") from exc

    return model


__all__ = ["load_siglip_model"]
