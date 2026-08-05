from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.base.adapters import ModelAdapter
from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class PerceiverVisionAdapter(ModelAdapter):
    """Adapter for ImageNet Perceiver models with non-spatial latent tokens."""

    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        self.model = model.eval()
        unwrapped = _unwrap_module(model)
        self.device = torch.device(device) if device is not None else next(unwrapped.parameters()).device
        self._current_sample_ids: Optional[torch.Tensor] = None
        self.current_meta: Optional[Dict[str, Any]] = None

    def get_hook_points(self) -> List[str]:
        model = _unwrap_module(self.model)
        encoder = model.perceiver.encoder
        return [f"model.{encoder._speclens_depth_tap_path}"]

    def preprocess_input(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        model_dtype = next(_unwrap_module(self.model).parameters()).dtype
        pixel_values = raw_batch["pixel_values"].to(
            self.device,
            dtype=model_dtype,
            non_blocking=True,
        )
        labels = raw_batch.get("labels", raw_batch.get("label"))
        if torch.is_tensor(labels):
            labels = labels.to(self.device, non_blocking=True)
        sample_ids = raw_batch.get("sample_ids", raw_batch.get("sample_id"))
        if torch.is_tensor(sample_ids):
            self._current_sample_ids = sample_ids.detach().to(torch.long).cpu()
        else:
            self._current_sample_ids = None
        paths = list(raw_batch.get("paths", raw_batch.get("path", [])) or [])
        self.current_meta = {
            "sample_ids": self._current_sample_ids,
            "labels": labels.detach().cpu() if torch.is_tensor(labels) else labels,
            "paths": paths,
        }
        return {"pixel_values": pixel_values, "labels": labels, "paths": paths}

    def forward(self, batch: Dict[str, Any]) -> None:
        pixel_values = batch["pixel_values"]
        if torch.is_grad_enabled():
            self.model(inputs=pixel_values)
        else:
            with torch.no_grad():
                self.model(inputs=pixel_values)

    def get_provenance_spec(self) -> Dict[str, Any]:
        # y=-1 and x=latent index; x is not an input-image coordinate.
        cols = ("sample_id", "y", "x")
        return {"cols": cols, "num_cols": len(cols)}


def create_perceiver_store(
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset=None,
    sampler=None,
    collate_fn: Optional[Any] = None,
    on_batch_generated: Optional[Any] = None,
    **_: Any,
) -> UniversalActivationStore:
    del collate_fn
    adapter = PerceiverVisionAdapter(model, device=cfg.get("device"))
    return UniversalActivationStore(
        model,
        cfg,
        adapter,
        dataset,
        sampler,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["PerceiverVisionAdapter", "create_perceiver_store"]
