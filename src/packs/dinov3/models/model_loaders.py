from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoConfig, AutoModel

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _resolve_dtype(dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "torch.float16", "half"):
        return torch.float16
    if key in ("fp32", "float32", "torch.float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_dinov3_model(
    model_cfg: Dict[str, Any],
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
):
    """Load a DINOv3 model from HuggingFace Transformers."""
    cfg = model_cfg or {}
    device = torch.device(device)
    name = cfg.get("hf_path") or cfg.get("name") or DEFAULT_MODEL
    pretrained = bool(cfg.get("pretrained", True))
    trust_remote = bool(cfg.get("trust_remote_code", True))
    revision = cfg.get("revision")
    attn_impl = cfg.get("attn_impl") or cfg.get("attn_implementation")
    dtype = _resolve_dtype(cfg.get("dtype") or (full_config or {}).get("dtype"))
    low_cpu_mem = bool(cfg.get("low_cpu_mem", True))
    device_map = cfg.get("device_map")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote,
    }
    if revision is not None:
        model_kwargs["revision"] = revision
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if low_cpu_mem:
        model_kwargs["low_cpu_mem_usage"] = True

    if pretrained:
        model = AutoModel.from_pretrained(name, **model_kwargs)
    else:
        base_cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote, revision=revision)
        model = AutoModel.from_config(base_cfg, trust_remote_code=trust_remote)
        if dtype is not None:
            model = model.to(dtype)

    model.eval()
    if device_map is None:
        model.to(device)

    if dtype is not None and device_map is None:
        model = model.to(dtype)

    if pretrained and rank == 0:
        LOGGER.info("[dinov3] Loaded pretrained weights for %s", name)

    return model


__all__ = ["load_dinov3_model"]
