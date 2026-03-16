from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.core.base.adapters import ModelAdapter
from src.core.base.layout import LayoutSpec, DimRole
from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore
from src.packs.gemma_2b.models.model_loaders import format_resid_hook_name, parse_layer_index


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


class GemmaLMAdapter(ModelAdapter):
    """
    Adapter that runs Gemma LMs inside the UniversalActivationStore.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        tokenizer=None,
        device: Optional[Union[str, torch.device]] = None,
        collate_fn: Optional[Any] = None,
        prepend_bos: bool = False,
    ):
        self.model = model.eval()
        unwrapped = _unwrap_module(model)
        if device is None:
            device = next(unwrapped.parameters()).device
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.prepend_bos = bool(prepend_bos)
        self.collate_fn = collate_fn

        self._current_sample_ids: Optional[torch.Tensor] = None
        self._current_attention_mask: Optional[torch.Tensor] = None
        self.current_meta: Optional[Dict[str, Any]] = None

    def _layer_count(self) -> int:
        model = _unwrap_module(self.model)
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            return int(cfg.num_hidden_layers)
        layers = getattr(getattr(model, "model", model), "layers", None)
        if layers is not None:
            try:
                return len(layers)
            except TypeError:
                pass
        return 0

    def get_hook_points(self) -> List[str]:
        n_layers = self._layer_count()
        return [format_resid_hook_name(i) for i in range(n_layers)]

    def _maybe_prepend_bos(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.prepend_bos:
            return batch
        if "input_ids" not in batch:
            return batch
        if self.tokenizer is None:
            return batch
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            return batch

        input_ids: torch.Tensor = batch["input_ids"]
        attn = batch.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(input_ids)

        bos_col = torch.full((input_ids.shape[0], 1), bos_id, dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([bos_col, input_ids], dim=1)
        attn = torch.cat([torch.ones_like(bos_col), attn], dim=1)
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attn
        return batch

    def preprocess_input(self, raw_batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = dict(raw_batch)

        # Optional on-the-fly tokenisation if only text is provided.
        if "input_ids" not in batch and self.tokenizer is not None and "text" in batch:
            encoded = self.tokenizer(
                batch["text"],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch.update(encoded)

        batch = self._maybe_prepend_bos(batch)

        # Move tensors to device
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(self.device, non_blocking=True)

        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            input_ids = batch.get("input_ids")
            texts = batch.get("text")
            if torch.is_tensor(input_ids):
                bs = int(input_ids.shape[0])
            elif isinstance(texts, (list, tuple)):
                bs = len(texts)
            elif isinstance(texts, str):
                bs = 1
            else:
                bs = 1
            sample_ids = torch.arange(bs, dtype=torch.long)
        if torch.is_tensor(sample_ids):
            self._current_sample_ids = sample_ids.detach().to(torch.long).cpu()
        else:
            self._current_sample_ids = torch.as_tensor(sample_ids, dtype=torch.long)

        attn = batch.get("attention_mask")
        if torch.is_tensor(attn):
            self._current_attention_mask = attn.detach().to(torch.long).cpu()
        else:
            self._current_attention_mask = None

        texts = batch.get("text")
        self.current_meta = {
            "sample_ids": self._current_sample_ids,
            "text": list(texts) if isinstance(texts, (list, tuple)) else texts,
        }

        # Drop metadata keys before forwarding through the model.
        batch.pop("sample_id", None)
        batch.pop("text", None)
        batch["use_cache"] = batch.get("use_cache", False)
        return batch

    def forward(self, batch: Dict[str, Any]) -> None:
        if torch.is_grad_enabled():
            _ = self.model(**batch)
        else:
            with torch.no_grad():
                _ = self.model(**batch)

    def get_layout_spec(self, act_name: str, tensor: torch.Tensor):
        from src.core.base.layout import LayoutSpec, DimRole

        adapter_ref = self  # capture for closure

        def _gemma_enrich(columns, dim_indices):
            """Apply attention mask: masked tokens get t=-1."""
            mask = getattr(adapter_ref, '_current_attention_mask', None)
            if mask is not None and DimRole.TIME in dim_indices:
                b_idx = dim_indices[DimRole.BATCH]
                t_idx = dim_indices[DimRole.TIME]
                B_mask = mask.shape[0]
                T_mask = mask.shape[1] if mask.ndim > 1 else 1
                flat_mask = mask.reshape(-1)
                flat_pos = (b_idx * T_mask + t_idx).clamp(max=flat_mask.numel() - 1)
                mask_vals = flat_mask[flat_pos.to(flat_mask.device)]
                # Where mask==0 (padding), set the time index to -1
                new_t = torch.where(mask_vals == 0, torch.full_like(t_idx, -1), t_idx)
                columns["t"] = new_t.to(t_idx.device)
            return columns

        return LayoutSpec(
            dims=(DimRole.BATCH, DimRole.TIME, DimRole.FEATURE),
            enrich_fn=_gemma_enrich,
            col_name_overrides={DimRole.TIME: "t"},
        )

    def get_provenance_spec(self) -> Dict[str, Any]:
        cols = ("sample_id", "t")
        return {"cols": cols, "num_cols": len(cols)}


def create_gemma_store(
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset=None,
    sampler=None,
    collate_fn: Optional[Any] = None,
    on_batch_generated: Optional[Any] = None,
) -> UniversalActivationStore:
    def _canon(name: str) -> str:
        try:
            layer_idx = parse_layer_index(name)
        except Exception:
            return name
        if "model.layers" in name:
            return name
        return format_resid_hook_name(layer_idx)

    # Normalize hook/per_layer keys so configs using blocks.{i}.hook_resid_post still work.
    store_cfg = dict(cfg)
    if store_cfg.get("hook_points"):
        store_cfg["hook_points"] = [_canon(h) if isinstance(h, str) else h for h in store_cfg["hook_points"]]
    if store_cfg.get("per_layer"):
        store_cfg["per_layer"] = {_canon(k) if isinstance(k, str) else k: v for k, v in store_cfg["per_layer"].items()}

    tokenizer = cfg.get("tokenizer", None) or getattr(model, "tokenizer", None)
    adapter = GemmaLMAdapter(
        model,
        device=cfg.get("device"),
        tokenizer=tokenizer,
        collate_fn=collate_fn,
        prepend_bos=bool(cfg.get("prepend_bos", False)),
    )
    return UniversalActivationStore(
        model,
        store_cfg,
        adapter,
        dataset,
        sampler,
        on_batch_generated=on_batch_generated,
    )


__all__ = ["GemmaLMAdapter", "create_gemma_store"]
