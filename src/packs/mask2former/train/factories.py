# src/packs/mask2former/train/factories.py
"""
Factory functions for Mask2Former SAE training and indexing.

This module provides the standard interface expected by:
- scripts/train_sae_config.py (SAE training)
- scripts/sae_index_main.py (SAE indexing)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.packs.mask2former.dataset.builders import build_sav_dataset, mask2former_collate_fn
from src.packs.mask2former.models.adapters import create_mask2former_store
from src.packs.mask2former.models.model_loaders import load_mask2former

LOGGER = logging.getLogger(__name__)


def _prepend_sys_path(path_like: str | Path) -> None:
    """Add path to sys.path if not already present."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def setup_env(
    config: Optional[Dict[str, Any]] = None, 
    *, 
    project_root: Optional[Path] = None
) -> None:
    """
    Set up environment for Mask2Former.
    
    Adds necessary paths to sys.path:
    - third_party/Mask2Former-main (for mask2former module)
    - detectron2 paths if needed
    
    Args:
        config: Configuration dict with optional 'sys_paths' list
        project_root: Project root directory
    """
    cfg = config or {}
    
    # Default paths to add
    default_paths = [
        "third_party/Mask2Former-main",
    ]
    
    # Add paths from config
    paths = cfg.get("sys_paths", default_paths)
    
    if project_root is None:
        project_root = Path(__file__).parents[4]  # Go up to project root
    
    for rel in paths:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            _prepend_sys_path(rel_path)
        else:
            _prepend_sys_path(project_root / rel_path)
    
    LOGGER.debug("[mask2former] Environment setup complete")


def load_model(
    model_cfg: Dict[str, Any],
    *positional,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> nn.Module:
    """
    Load Mask2Former model.
    
    Supports both keyword and legacy positional arguments for compatibility.
    
    Args:
        model_cfg: Model configuration dict containing:
            - config_file: Path to detectron2 config YAML
            - weights: Path to model checkpoint
            - opts: Additional config overrides (optional)
        device: Target device (can also be first positional arg)
        rank: DDP rank
        world_size: DDP world size
        full_config: Full SAE config (optional)
        **kwargs: Additional arguments passed to loader
    
    Returns:
        Loaded Mask2Former model in eval mode
    """
    # Handle legacy positional arguments: (model_cfg, device, logger)
    if device is None:
        if not positional:
            raise TypeError("'device' must be provided either as positional or keyword argument")
        device, *positional = positional
    
    # Extract optional logger from positional args
    if positional:
        kwargs.setdefault("logger", positional[0])
    
    return load_mask2former(
        model_cfg,
        device=device,
        rank=rank,
        world_size=world_size,
        full_config=full_config,
        **kwargs,
    )


def build_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
    device: Optional[torch.device] = None,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> Dict[str, Any]:
    """
    Build dataset for SAE training.
    
    Args:
        dataset_cfg: Dataset configuration
        rank: DDP rank
        world_size: DDP world size
        device: Device (unused, for interface compatibility)
        full_config: Full config dict
    
    Returns:
        Dict with 'dataset', 'collate_fn', 'sampler' keys
    """
    return build_sav_dataset(
        dataset_cfg,
        rank=rank,
        world_size=world_size,
        full_config=full_config,
    )


def create_store(
    *,
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset,
    sampler,
    collate_fn,
    on_batch_generated=None,
    **_,
):
    """
    Create activation store for Mask2Former.
    
    Args:
        model: Mask2Former model
        cfg: Store configuration
        dataset: Dataset instance
        sampler: Data sampler
        collate_fn: Collate function
        on_batch_generated: Callback for batch metadata
    
    Returns:
        UniversalActivationStore configured for Mask2Former
    """
    return create_mask2former_store(
        model=model,
        cfg=cfg,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["setup_env", "load_model", "build_dataset", "create_store"]
