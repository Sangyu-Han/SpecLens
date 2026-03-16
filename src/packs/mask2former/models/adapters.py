# src/packs/mask2former/models/adapters.py
"""Mask2Former adapter for UniversalActivationStore."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.base.adapters import ModelAdapter
from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore


@dataclass
class Mask2FormerBatchMeta:
    """Metadata for a batch of images processed by Mask2Former."""
    sample_ids: Optional[torch.Tensor]  # (B,) int64
    image_paths: Sequence[str]          # List of image paths
    image_sizes: Optional[List[tuple]]  # Original (H, W) for each image
    

def _unwrap_module(model: nn.Module) -> nn.Module:
    """Unwrap DDP/DataParallel wrapper."""
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


class Mask2FormerAdapter(ModelAdapter):
    """
    Adapter for Mask2Former models (detectron2 or Hugging Face).

    Mask2Former processes images directly without prompts, making it simpler
    than SAM2. The forward pass takes a list of dicts with "image" tensors
    (detectron2) or a dict of pixel values (HF).
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        device: Optional[Union[str, torch.device]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model.eval()
        unwrapped = _unwrap_module(model)
        
        if device is None:
            # detectron2 models store pixel_mean as buffer
            if hasattr(unwrapped, "pixel_mean"):
                device = unwrapped.pixel_mean.device
            else:
                device = next(unwrapped.parameters()).device
        self.device = torch.device(device)

        self._is_hf = bool(
            getattr(getattr(unwrapped, "config", None), "model_type", None) == "mask2former"
        )
        self._hf_rescale = None
        self._hf_mean = None
        self._hf_std = None
        if self._is_hf:
            try:
                from transformers import Mask2FormerImageProcessor

                proc = Mask2FormerImageProcessor()
                self._hf_rescale = float(proc.rescale_factor)
                self._hf_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
                self._hf_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
            except Exception:
                self._hf_rescale = 1.0 / 255.0
                self._hf_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                self._hf_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.collate_fn = collate_fn
        self.current_meta: Optional[Mask2FormerBatchMeta] = None
        self._current_sample_ids: Optional[torch.Tensor] = None

    def preprocess_input(self, raw_batch: Dict[str, Any]) -> Any:
        """
        Convert batch dict to detectron2's expected format.
        
        Input format (from our collate_fn):
            {
                "images": Tensor[B, C, H, W],
                "sample_ids": Tensor[B],
                "image_paths": List[str],
                "heights": List[int],
                "widths": List[int],
            }
        
        Output format (detectron2):
            List of dicts, each with:
                "image": Tensor[C, H, W]
                "height": int (original)
                "width": int (original)
        """
        images = raw_batch["images"]  # (B, C, H, W)
        sample_ids = raw_batch.get("sample_ids")
        image_paths = raw_batch.get("image_paths", [])
        heights = raw_batch.get("heights", [])
        widths = raw_batch.get("widths", [])
        
        B = images.shape[0]
        
        # Store metadata
        if torch.is_tensor(sample_ids):
            self._current_sample_ids = sample_ids.detach().to(torch.long).cpu()
        else:
            self._current_sample_ids = torch.arange(B, dtype=torch.long)
        
        image_sizes = []
        for i in range(B):
            h = heights[i] if i < len(heights) else images.shape[2]
            w = widths[i] if i < len(widths) else images.shape[3]
            image_sizes.append((h, w))
        
        self.current_meta = Mask2FormerBatchMeta(
            sample_ids=self._current_sample_ids.clone(),
            image_paths=list(image_paths),
            image_sizes=image_sizes,
        )
        
        if self._is_hf:
            pixel_values = images.to(self.device)
            if pixel_values.dtype != torch.float32:
                pixel_values = pixel_values.float()
            pixel_values = pixel_values[:, [2, 1, 0], :, :]  # BGR -> RGB
            rescale = self._hf_rescale or (1.0 / 255.0)
            pixel_values = pixel_values * rescale
            if self._hf_mean is not None and self._hf_std is not None:
                mean = self._hf_mean.to(pixel_values.device, dtype=pixel_values.dtype)
                std = self._hf_std.to(pixel_values.device, dtype=pixel_values.dtype)
                pixel_values = (pixel_values - mean) / std
            pixel_mask = raw_batch.get("pixel_mask")
            if torch.is_tensor(pixel_mask):
                pixel_mask = pixel_mask.to(self.device)
            return {"pixel_values": pixel_values, "pixel_mask": pixel_mask}

        # Convert to detectron2 format
        batched_inputs = []
        for i in range(B):
            item = {
                "image": images[i].to(self.device),
            }
            if i < len(heights):
                item["height"] = heights[i]
                item["width"] = widths[i]
            batched_inputs.append(item)

        return batched_inputs
    
    @torch.no_grad()
    def forward(self, batch: Any) -> Any:
        """
        Forward pass through Mask2Former.
        
        In eval mode, Mask2Former returns predictions.
        We run forward to trigger hooks, output is not directly used.
        """
        if isinstance(batch, dict):
            return self.model(**batch)
        return self.model(batch)
    
    def get_provenance_spec(self) -> Dict[str, Any]:
        """Return provenance column specification."""
        cols = ("sample_id", "y", "x")
        return {"cols": cols, "num_cols": len(cols)}


def create_mask2former_store(
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset=None,
    sampler=None,
    collate_fn: Optional[Callable] = None,
    on_batch_generated: Optional[Callable] = None,
) -> UniversalActivationStore:
    """
    Create a UniversalActivationStore for Mask2Former.
    
    Args:
        model: Mask2Former model
        cfg: Store configuration dict
        dataset: Dataset instance
        sampler: DistributedSampler or similar
        collate_fn: Batch collation function
        on_batch_generated: Callback for batch metadata (e.g., offline ledger)
    Returns:
        Configured UniversalActivationStore
    """
    adapter = Mask2FormerAdapter(
        model=model,
        device=cfg.get("device"),
        collate_fn=collate_fn,
    )

    return UniversalActivationStore(
        model=model,
        cfg=cfg,
        adapter=adapter,
        dataset=dataset,
        sampler=sampler,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["Mask2FormerAdapter", "Mask2FormerBatchMeta", "create_mask2former_store"]
