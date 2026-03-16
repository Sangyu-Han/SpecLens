from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.base.adapters import ModelAdapter
from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore


@dataclass
class Dinov2BatchMeta:
    sample_ids: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    paths: Sequence[str]


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class Dinov2VisionAdapter(ModelAdapter):
    """Adapter to run DINOv2 vision encoders inside the universal activation store."""

    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        self.model = model.eval()
        unwrapped = _unwrap_module(model)
        if device is None:
            device = next(unwrapped.parameters()).device
        self.device = torch.device(device)
        self._current_sample_ids: Optional[torch.Tensor] = None
        self.current_meta: Optional[Dict[str, Any]] = None

    def get_hook_points(self) -> List[str]:
        base: List[str] = []
        model = _unwrap_module(self.model)

        if hasattr(model, "embeddings"):
            base.append("embeddings")

        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            layers = None
            prefix = None
            if hasattr(encoder, "layer"):
                layers = encoder.layer
                prefix = "encoder.layer"
            elif hasattr(encoder, "layers"):
                layers = encoder.layers
                prefix = "encoder.layers"
            if layers is not None and prefix is not None:
                try:
                    n_blocks = len(layers)
                except TypeError:
                    n_blocks = 0
                for idx in range(n_blocks):
                    base.append(f"{prefix}.{idx}@0")

        if hasattr(model, "blocks"):
            try:
                n_blocks = len(model.blocks)  # type: ignore[arg-type]
            except TypeError:
                n_blocks = 0
            for idx in range(n_blocks):
                base.append(f"blocks.{idx}")

        if hasattr(model, "layernorm"):
            base.append("layernorm")
        elif hasattr(model, "norm"):
            base.append("norm")

        return base

    def preprocess_input(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        pixel_values = raw_batch["pixel_values"].to(self.device, non_blocking=True)
        labels = raw_batch.get("label")
        if labels is None:
            labels = raw_batch.get("labels")
        if torch.is_tensor(labels):
            labels = labels.to(self.device, non_blocking=True)
        sample_ids = raw_batch.get("sample_id")
        if sample_ids is None:
            sample_ids = raw_batch.get("sample_ids")
        if torch.is_tensor(sample_ids):
            self._current_sample_ids = sample_ids.detach().to(torch.long).cpu()
        else:
            self._current_sample_ids = None

        paths = raw_batch.get("path") or raw_batch.get("paths") or []
        self.current_meta = {
            "sample_ids": sample_ids.detach().cpu() if torch.is_tensor(sample_ids) else sample_ids,
            "labels": labels.detach().cpu() if torch.is_tensor(labels) else labels,
            "paths": list(paths),
        }
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "sample_ids": sample_ids,
            "paths": paths,
        }

    def forward(self, batch: Dict[str, Any]) -> None:
        pixel_values = batch["pixel_values"]
        if torch.is_grad_enabled():
            _ = self.model(pixel_values=pixel_values)
        else:
            with torch.no_grad():
                _ = self.model(pixel_values=pixel_values)

    def get_provenance_spec(self) -> Dict[str, Any]:
        cols = ("sample_id", "y", "x")
        return {"cols": cols, "num_cols": len(cols)}


def create_dinov2_store(
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset=None,
    sampler=None,
    collate_fn: Optional[Any] = None,
    on_batch_generated: Optional[Any] = None,
) -> UniversalActivationStore:
    adapter = Dinov2VisionAdapter(model, device=cfg.get("device"))
    if collate_fn is not None:
        adapter.collate_fn = collate_fn
    return UniversalActivationStore(
        model,
        cfg,
        adapter,
        dataset,
        sampler,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["Dinov2VisionAdapter", "create_dinov2_store", "Dinov2BatchMeta"]
