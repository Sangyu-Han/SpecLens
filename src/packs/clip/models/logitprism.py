from __future__ import annotations

import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.vision_transformer import Attention as TimmAttention
except Exception:  # pragma: no cover - timm not available in some environments
    TimmAttention = None


_ATTENTION_TYPES: Tuple[type, ...] = tuple(t for t in (TimmAttention,) if t is not None)


def _attn_embed_dim(attn: nn.Module) -> int:
    cand = getattr(attn, "embed_dim", None)
    if isinstance(cand, int):
        return cand
    if hasattr(attn, "qkv"):
        qkv = attn.qkv
        if hasattr(qkv, "in_features"):
            return int(qkv.in_features)
        if hasattr(qkv, "weight"):
            return int(qkv.weight.shape[1])
    if hasattr(attn, "proj") and hasattr(attn.proj, "weight"):
        return int(attn.proj.weight.shape[0])
    raise AttributeError("Unable to infer attention embed_dim")


class VisionPrismCache:
    """Cache clean-run stats and apply logit-prism-style patches for timm ViT models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._saved_inputs: Dict[int, torch.Tensor] = {}
        self._attn_qk: Dict[int, Dict[str, torch.Tensor]] = {}
        self._handles: List[Any] = []
        self._orig_fwd: Dict[nn.Module, Callable[..., Any]] = {}
        self._orig_bias: Dict[nn.Linear, Optional[torch.nn.Parameter]] = {}

    def clear(self) -> None:
        self._saved_inputs.clear()
        self._attn_qk.clear()

    # ----------------------------- capture ---------------------------------
    def _make_save_hook(self, module: nn.Module):
        key = id(module)

        def _hook(_m, inputs):
            if not inputs:
                return
            x = inputs[0]
            if torch.is_tensor(x):
                self._saved_inputs[key] = x.detach().cpu()

        return _hook

    def _make_attn_hook(self, module: nn.Module):
        key = id(module)

        def _hook(_m, inputs):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            if not hasattr(module, "qkv"):
                return
            with torch.no_grad():
                qkv = module.qkv(x)
                B, N, _ = qkv.shape
                qkv = qkv.reshape(B, N, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                q_norm = getattr(module, "q_norm", None)
                k_norm = getattr(module, "k_norm", None)
                if q_norm is not None:
                    q = q_norm(q)
                if k_norm is not None:
                    k = k_norm(k)
                self._attn_qk[key] = {
                    "q": q.detach().cpu(),
                    "k": k.detach().cpu(),
                }
                self._saved_inputs[key] = x.detach().cpu()

        return _hook

    def register_save_hooks(self) -> None:
        self.clear()
        for mod in self.model.modules():
            if isinstance(mod, nn.LayerNorm):
                h = mod.register_forward_pre_hook(self._make_save_hook(mod), with_kwargs=False)
                self._handles.append(h)
            elif isinstance(mod, nn.GELU):
                h = mod.register_forward_pre_hook(self._make_save_hook(mod), with_kwargs=False)
                self._handles.append(h)
            elif _ATTENTION_TYPES and isinstance(mod, _ATTENTION_TYPES):
                h = mod.register_forward_pre_hook(self._make_attn_hook(mod), with_kwargs=False)
                self._handles.append(h)

    def remove_hooks(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def capture(self, batch: Dict[str, torch.Tensor]) -> None:
        self.register_save_hooks()
        with torch.no_grad():
            _ = self.model(batch["pixel_values"])
        self.remove_hooks()

    # ----------------------------- patch/unpatch ---------------------------
    def _toggle_bias(self, remove: bool) -> None:
        for mod in self.model.modules():
            if isinstance(mod, nn.Linear):
                if remove:
                    if mod.bias is not None:
                        self._orig_bias.setdefault(mod, mod.bias)
                        mod.bias = None
                elif mod in self._orig_bias:
                    mod.bias = self._orig_bias[mod]
        if not remove:
            self._orig_bias.clear()

    def patch(self) -> None:
        self._toggle_bias(True)
        for mod in self.model.modules():
            if isinstance(mod, nn.LayerNorm):
                self._orig_fwd[mod] = mod.forward  # type: ignore[assignment]
                mod._prism_cache = self  # type: ignore[attr-defined]
                mod.forward = types.MethodType(VisionPrismCache._ln_forward, mod)  # type: ignore[assignment]
            elif isinstance(mod, nn.GELU):
                self._orig_fwd[mod] = mod.forward  # type: ignore[assignment]
                mod._prism_cache = self  # type: ignore[attr-defined]
                mod.forward = types.MethodType(VisionPrismCache._gelu_forward, mod)  # type: ignore[assignment]
            elif _ATTENTION_TYPES and isinstance(mod, _ATTENTION_TYPES):
                self._orig_fwd[mod] = mod.forward  # type: ignore[assignment]
                mod._prism_cache = self  # type: ignore[attr-defined]
                mod.forward = types.MethodType(VisionPrismCache._attn_forward, mod)  # type: ignore[assignment]

    def unpatch(self) -> None:
        for mod, fwd in list(self._orig_fwd.items()):
            try:
                mod.forward = fwd  # type: ignore[assignment]
            except Exception:
                pass
            if hasattr(mod, "_prism_cache"):
                try:
                    delattr(mod, "_prism_cache")
                except AttributeError:
                    pass
        self._orig_fwd.clear()
        self._toggle_bias(False)

    # ----------------------------- patched fwd -----------------------------
    @staticmethod
    def _ln_forward(self: nn.LayerNorm, x: torch.Tensor):  # type: ignore[override]
        cache = getattr(self, "_prism_cache", None)
        saved = cache._saved_inputs.get(id(self)) if cache is not None else None  # type: ignore[attr-defined]
        if saved is not None:
            saved_cast = saved.to(device=x.device, dtype=x.dtype)
            std = (saved_cast.var(dim=-1, unbiased=False, keepdim=True) + self.eps).sqrt()
            mu = x.mean(dim=-1, keepdim=True)
            y = (x - mu) / std
            if self.weight is not None:
                y = y * self.weight
            return y
        if cache is not None and self in cache._orig_fwd:  # type: ignore[attr-defined]
            return cache._orig_fwd[self](x)  # type: ignore[index]
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)

    @staticmethod
    def _gelu_forward(self: nn.GELU, x):  # type: ignore[override]
        cache = getattr(self, "_prism_cache", None)
        approx = getattr(self, "approximate", "none")
        saved = cache._saved_inputs.get(id(self)) if cache is not None else None  # type: ignore[attr-defined]
        if saved is not None:
            saved_cast = saved.to(device=x.device, dtype=x.dtype)
            denom = saved_cast + 1e-6
            ratio = F.gelu(saved_cast, approximate=approx) / denom
            return ratio * x
        if cache is not None and self in cache._orig_fwd:  # type: ignore[attr-defined]
            return cache._orig_fwd[self](x)  # type: ignore[index]
        return F.gelu(x, approximate=approx)

    @staticmethod
    def _attn_forward(self, x, attn_mask=None, **kwargs):  # type: ignore[override]
        cache = getattr(self, "_prism_cache", None)
        saved = cache._attn_qk.get(id(self)) if cache is not None else None  # type: ignore[attr-defined]
        if saved is None:
            orig = cache._orig_fwd.get(self) if cache is not None else None  # type: ignore[attr-defined]
            return orig(x, attn_mask=attn_mask, **kwargs) if orig is not None else NotImplemented
        q = saved["q"].to(device=x.device, dtype=x.dtype)
        k = saved["k"].to(device=x.device, dtype=x.dtype)
        qkv = self.qkv(x)
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        v = qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask.to(dtype=attn.dtype, device=attn.device)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        embed_dim = _attn_embed_dim(self)
        out = out.transpose(1, 2).reshape(B, N, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class SecondOrderLensRuntime:
    """Second-order lens propagation for timm ViT blocks (attention-only path)."""

    def __init__(self, model: nn.Module, cache: VisionPrismCache) -> None:
        self.model = model
        self.cache = cache
        self.blocks: List[nn.Module] = list(getattr(model, "blocks", []))
        self.final_norm = getattr(model, "norm", None)

    def _apply_layernorm(self, delta: torch.Tensor, module: nn.LayerNorm, *, norm_term: Optional[float] = None) -> torch.Tensor:
        saved = self.cache._saved_inputs.get(id(module))
        denom = float(norm_term) if norm_term is not None and norm_term > 0 else None
        if saved is not None and torch.is_tensor(saved):
            saved_cast = saved.to(device=delta.device, dtype=delta.dtype)
            std = (saved_cast.var(dim=-1, unbiased=False, keepdim=True) + module.eps).sqrt()
            mu = saved_cast.mean(dim=-1, keepdim=True)
        else:
            std = (delta.var(dim=-1, unbiased=False, keepdim=True) + module.eps).sqrt()
            mu = delta.mean(dim=-1, keepdim=True)
        centered = delta - (mu / denom if denom else mu)
        out = centered / std
        if module.weight is not None:
            out = out * module.weight
        if module.bias is not None:
            out = out + (module.bias / denom if denom else module.bias)
        return out

    def _attn_weights(self, attn: nn.Module, *, device, dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        qkv_weight = attn.qkv.weight  # (3D, D)
        dim = min(_attn_embed_dim(attn), qkv_weight.shape[0] // 3, qkv_weight.shape[1])
        W_Q = qkv_weight[:dim, :].T.to(device=device, dtype=dtype)
        W_K = qkv_weight[dim : 2 * dim, :].T.to(device=device, dtype=dtype)
        W_V = qkv_weight[2 * dim :, :].T.to(device=device, dtype=dtype)
        qkv_bias = getattr(attn.qkv, "bias", None)
        if qkv_bias is not None:
            qkv_bias = qkv_bias.to(device=device, dtype=dtype)
        return W_Q, W_K, W_V, qkv_bias

    def _one_block(self, delta: torch.Tensor, blk: nn.Module, layer_idx: int, num_blocks: int, device, dtype) -> torch.Tensor:
        ln1 = getattr(blk, "norm1", None)
        attn = getattr(blk, "attn", None)
        delta_norm = delta
        if isinstance(ln1, nn.LayerNorm):
            norm_term = (1.0 + 2.0 * float(layer_idx)) * float(delta.shape[-1])
            delta_norm = self._apply_layernorm(delta, ln1, norm_term=norm_term)
        if attn is None:
            return delta
        B, S, _ = delta_norm.shape
        _, _, W_V, qkv_bias = self._attn_weights(attn, device=device, dtype=dtype)
        bias_term = None
        if qkv_bias is not None and qkv_bias.numel() >= 3 * W_V.shape[0]:
            v_bias = qkv_bias[2 * W_V.shape[0] : 3 * W_V.shape[0]].reshape(-1)
            bias_term = v_bias
        delta_v = delta_norm @ W_V
        if bias_term is not None:
            norm_bias = (1.0 + 2.0 * float(layer_idx)) * float(delta.shape[-1])
            delta_v = delta_v + bias_term.to(device=device, dtype=dtype) / norm_bias
        delta_v = delta_v.view(B, S, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
        saved = self.cache._attn_qk.get(id(attn))
        if saved is not None:
            q = saved["q"].to(device=device, dtype=dtype)
            k = saved["k"].to(device=device, dtype=dtype)
        else:
            src = self.cache._saved_inputs.get(id(attn))
            src_tensor = src if torch.is_tensor(src) else delta_norm
            qkv = attn.qkv(src_tensor.to(device=device, dtype=dtype))
            qkv = qkv.reshape(B, S, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q_norm = getattr(attn, "q_norm", None)
            k_norm = getattr(attn, "k_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            if k_norm is not None:
                k = k_norm(k)
        attn_scores = (q * attn.scale) @ k.transpose(-2, -1)
        attn_probs = attn_scores.softmax(dim=-1)
        delta_ctx = attn_probs @ delta_v
        embed_dim = _attn_embed_dim(attn)
        delta_ctx = delta_ctx.permute(0, 2, 1, 3).reshape(B, S, embed_dim)
        W_O = attn.proj.weight.T.to(device=device, dtype=dtype)
        delta_out = delta_ctx @ W_O
        if getattr(attn, "proj", None) is not None:
            proj_bias = getattr(attn.proj, "bias", None)
            if proj_bias is not None:
                norm_bias = (1.0 + 2.0 * float(layer_idx)) * float(attn.num_heads) * float(delta.shape[-1])
                delta_out = delta_out + proj_bias.to(device=device, dtype=dtype) / norm_bias
        return delta_out + delta

    def __call__(self, delta: torch.Tensor, start_block: int) -> torch.Tensor:
        delta_cur = delta
        device, dtype = delta.device, delta.dtype
        num_blocks = len(self.blocks)
        for idx in range(max(0, start_block), num_blocks):
            delta_cur = self._one_block(delta_cur, self.blocks[idx], idx, num_blocks, device, dtype)
        if isinstance(self.final_norm, nn.LayerNorm):
            last_idx = max(0, num_blocks - 1)
            last_heads = getattr(getattr(self.blocks[last_idx], "attn", None), "num_heads", 1)
            norm_term = (1.0 + 2.0 * float(last_idx)) * (1.0 + 2.0 * float(num_blocks)) * float(last_heads) * float(delta_cur.shape[-1])
            delta_cur = self._apply_layernorm(delta_cur, self.final_norm, norm_term=norm_term)
        elif self.final_norm is not None:
            delta_cur = self.final_norm(delta_cur)
        pooled = self.model.pool(delta_cur) if hasattr(self.model, "pool") else delta_cur[:, 0]
        ln_post = getattr(self.model, "ln_post", None)
        if isinstance(ln_post, nn.LayerNorm):
            last_idx = max(0, num_blocks - 1)
            last_heads = getattr(getattr(self.blocks[last_idx], "attn", None), "num_heads", 1)
            norm_term = (1.0 + 2.0 * float(last_idx)) * (1.0 + 2.0 * float(num_blocks)) * float(last_heads) * float(delta_cur.shape[-1])
            pooled = self._apply_layernorm(pooled, ln_post, norm_term=norm_term)
        elif ln_post is not None:
            pooled = ln_post(pooled)
        fc_norm = getattr(self.model, "fc_norm", None)
        if isinstance(fc_norm, nn.LayerNorm):
            pooled = self._apply_layernorm(pooled, fc_norm)
        elif fc_norm is not None:
            pooled = fc_norm(pooled)
        if hasattr(self.model, "head_drop"):
            pooled = self.model.head_drop(pooled)
        head = getattr(self.model, "head", None)
        if head is not None and hasattr(head, "weight"):
            return pooled @ head.weight.T
        return pooled


__all__ = ["VisionPrismCache", "SecondOrderLensRuntime"]
