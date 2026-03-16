from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.packs.gemma_2b.dataset.builders import build_text_dataset
from src.packs.gemma_2b.models.adapters import create_gemma_store
from src.packs.gemma_2b.models.model_loaders import load_gemma_model

LOGGER = logging.getLogger(__name__)


def _prepend_sys_path(path_like: str | Path) -> None:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _resolve_with_root(target: str | Path, *, project_root: Optional[Path]) -> Path:
    path = Path(target).expanduser()
    if project_root is not None and not path.is_absolute():
        path = project_root / path
    return path


def setup_env(config: Optional[Dict[str, Any]] = None, *, project_root: Optional[Path] = None) -> None:
    """
    Prepare sys.path and HF cache env vars for Gemma runs.
    """
    cfg = config or {}
    paths = cfg.get("sys_paths") or []
    for rel in paths:
        target = _resolve_with_root(rel, project_root=project_root)
        _prepend_sys_path(target)

    # Optional HuggingFace cache override to keep downloads local to the project
    hf_home = cfg.get("hf_home") or cfg.get("hf_cache")
    transformers_cache = cfg.get("transformers_cache")
    datasets_cache = cfg.get("datasets_cache") or cfg.get("hf_datasets_cache")

    if hf_home:
        home = _resolve_with_root(hf_home, project_root=project_root)
        os.environ.setdefault("HF_HOME", str(home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(home / "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(home / "datasets"))
        LOGGER.info("[gemma_2b.setup_env] HF_HOME=%s", home)
    if transformers_cache:
        cache = _resolve_with_root(transformers_cache, project_root=project_root)
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
        LOGGER.info("[gemma_2b.setup_env] TRANSFORMERS_CACHE=%s", cache)
    if datasets_cache:
        cache = _resolve_with_root(datasets_cache, project_root=project_root)
        os.environ.setdefault("HF_DATASETS_CACHE", str(cache))
        LOGGER.info("[gemma_2b.setup_env] HF_DATASETS_CACHE=%s", cache)


def load_model(
    model_cfg: Dict[str, Any],
    *positional,
    device: Optional[torch.device] = None,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Load a Gemma causal LM. Mirrors CLIP/SAM2 factory signatures for compatibility.
    """
    if device is None:
        if not positional:
            raise TypeError("'device' must be provided either as a positional or keyword argument")
        device, *positional = positional

    # Support legacy positional logger slot (ignored).
    if positional:
        kwargs.setdefault("logger", positional[0])

    loaded = load_gemma_model(
        model_cfg,
        device=device,
        rank=rank,
        world_size=world_size,
        full_config=full_config,
        **kwargs,
    )
    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, tokenizer = loaded
        try:
            setattr(model, "tokenizer", tokenizer)
        except Exception:
            pass
        return model
    return loaded


def build_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int,
    world_size: int,
    device: torch.device | None = None,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
):
    """
    Build a simple text dataset + collate_fn for Gemma language models.
    """
    tokenizer_name = dataset_cfg.get("tokenizer") or dataset_cfg.get("tokenizer_name")
    if tokenizer_name is None:
        model_cfg = (full_config or {}).get("model", {}) or {}
        tokenizer_name = model_cfg.get("tokenizer") or model_cfg.get("hf_path") or model_cfg.get("name")

    return build_text_dataset(
        dataset_cfg,
        rank=rank,
        world_size=world_size,
        tokenizer_name=tokenizer_name,
        device=device,
    )


def create_store(
    *,
    model,
    cfg: Dict[str, Any],
    dataset,
    sampler,
    collate_fn,
    on_batch_generated=None,
    **_,
):
    return create_gemma_store(
        model=model,
        cfg=cfg,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["setup_env", "load_model", "build_dataset", "create_store"]
