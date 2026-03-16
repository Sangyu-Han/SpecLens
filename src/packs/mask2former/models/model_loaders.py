# src/packs/mask2former/models/model_loaders.py
"""Mask2Former model loader using detectron2 or Hugging Face."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def load_mask2former(
    model_cfg: Dict[str, Any],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> nn.Module:
    """
    Load a Mask2Former model using detectron2's configuration system.
    
    Args:
        model_cfg: Dictionary containing:
            - config_file: Path to detectron2 YAML config
            - weights: Path to model checkpoint (optional, can be in config)
            - opts: Additional config overrides (optional)
        device: Target device
        logger: Optional logger
        rank: DDP rank
        world_size: DDP world size
        full_config: Full SAE config (optional)
    
    Returns:
        Loaded Mask2Former model in eval mode
    """
    log = logger or LOGGER
    
    try:
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.projects.deeplab import add_deeplab_config
    except ImportError as e:
        raise ImportError(
            "detectron2 is required for Mask2Former. "
            "Please install it: python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        ) from e
    
    # Import mask2former config extension
    try:
        from mask2former import add_maskformer2_config
    except ImportError:
        # Fallback: try to import from third_party
        import sys
        mask2former_path = Path(__file__).parents[4] / "third_party" / "Mask2Former-main"
        if str(mask2former_path) not in sys.path:
            sys.path.insert(0, str(mask2former_path))
        from mask2former import add_maskformer2_config
    
    # Build detectron2 config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # Load config file
    config_file = model_cfg.get("config_file")
    if config_file:
        config_path = Path(config_file).expanduser()
        if not config_path.is_absolute():
            # Try relative to project root
            project_root = Path(__file__).parents[4]
            config_path = project_root / config_file
        if config_path.exists():
            cfg.merge_from_file(str(config_path))
        else:
            raise FileNotFoundError(f"Mask2Former config not found: {config_path}")
    
    # Apply additional options
    opts = model_cfg.get("opts", [])
    if opts:
        cfg.merge_from_list(opts)
    
    # Freeze config
    cfg.freeze()
    
    # Build model
    model = build_model(cfg)
    model.to(device)
    
    # Load weights
    weights_path = model_cfg.get("weights") or model_cfg.get("ckpt")
    if weights_path:
        weights_path = Path(weights_path).expanduser()
        if not weights_path.is_absolute():
            project_root = Path(__file__).parents[4]
            weights_path = project_root / weights_path
        
        if weights_path.exists():
            checkpointer = DetectionCheckpointer(model)
            checkpointer.load(str(weights_path))
            if rank == 0:
                log.info("[mask2former] Loaded weights from %s", weights_path)
        else:
            if rank == 0:
                log.warning("[mask2former] Weights not found: %s", weights_path)
    else:
        # Try loading from config default
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        if rank == 0:
            log.info("[mask2former] Loaded weights from config: %s", cfg.MODEL.WEIGHTS)
    
    model.eval()
    if rank == 0:
        log.info("[mask2former] Model loaded on device=%s", device)
    
    return model


def load_mask2former_hf(
    model_cfg: Dict[str, Any],
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> nn.Module:
    """
    Load a Mask2Former model from Hugging Face Transformers.

    Args:
        model_cfg: Dictionary containing:
            - hf_model: HF model id or local path
            - local_files_only: Whether to avoid network downloads (default False)
            - revision: Optional HF revision
            - torch_dtype: Optional dtype string (e.g., "float16")
        device: Target device
        logger: Optional logger
        rank: DDP rank
        world_size: DDP world size
        full_config: Full SAE config (optional)

    Returns:
        Loaded Mask2Former model in eval mode
    """
    log = logger or LOGGER
    try:
        from transformers import Mask2FormerForUniversalSegmentation
    except ImportError as e:
        raise ImportError(
            "transformers is required for HF Mask2Former. "
            "Please install it: python -m pip install transformers"
        ) from e

    model_id = (
        model_cfg.get("hf_model")
        or model_cfg.get("model_id")
        or model_cfg.get("pretrained")
        or model_cfg.get("name")
    )
    if not model_id:
        raise KeyError("config['model']['hf_model'] (or model_id/pretrained/name) must be provided.")

    local_files_only = bool(model_cfg.get("local_files_only", False))
    revision = model_cfg.get("revision")
    dtype = None
    dtype_spec = model_cfg.get("torch_dtype")
    if isinstance(dtype_spec, str):
        dtype = getattr(torch, dtype_spec, None)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    if rank == 0:
        log.info("[mask2former.hf] Loaded %s on device=%s", model_id, device)
    return model


def load_mask2former_simple(
    config_file: str,
    weights: Optional[str] = None,
    device: str = "cuda",
) -> nn.Module:
    """
    Simplified loader for quick experiments.
    
    Args:
        config_file: Path to detectron2 config YAML
        weights: Optional path to checkpoint
        device: Target device string
    
    Returns:
        Loaded Mask2Former model
    """
    cfg = {
        "config_file": config_file,
        "weights": weights,
    }
    return load_mask2former(cfg, torch.device(device))


__all__ = ["load_mask2former", "load_mask2former_hf", "load_mask2former_simple"]
