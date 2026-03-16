from __future__ import annotations

import math
from functools import wraps
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.backbones.hieradet import MultiScaleAttention
from sam2.modeling.memory_attention import MemoryAttentionLayer
from sam2.modeling.sam.transformer import Attention, RoPEAttention
from sam2.modeling.sam2_utils import LayerNorm2d, MLP


def _stop_gradient(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


class FullGradLayerNorm(nn.LayerNorm):
    """LayerNorm that detaches the variance term (FullGrad style)."""

    @classmethod
    def from_layer(cls, layer: nn.LayerNorm) -> "FullGradLayerNorm":
        new = cls(layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.elementwise_affine)
        if layer.elementwise_affine and layer.weight is not None:
            new.weight = nn.Parameter(layer.weight.detach().clone())
            if layer.bias is not None:
                new.bias = nn.Parameter(layer.bias.detach().clone())
            # Preserve the original requires_grad setting so patched modules stay frozen.
            new.weight.requires_grad = layer.weight.requires_grad
            if new.bias is not None and layer.bias is not None:
                new.bias.requires_grad = layer.bias.requires_grad
            new.to(device=layer.weight.device, dtype=layer.weight.dtype)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps).detach()
        y = (x - mean) / std
        if self.elementwise_affine:
            y = y * self.weight + self.bias
        return y


class FullGradLayerNorm2d(nn.Module):
    """LayerNorm2d variant that detaches the variance term."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    @classmethod
    def from_layer(cls, layer: LayerNorm2d) -> "FullGradLayerNorm2d":
        new = cls(layer.weight.shape[0], eps=layer.eps)
        new.weight = nn.Parameter(layer.weight.detach().clone())
        new.bias = nn.Parameter(layer.bias.detach().clone())
        new.weight.requires_grad = layer.weight.requires_grad
        new.bias.requires_grad = layer.bias.requires_grad
        new.to(device=layer.weight.device, dtype=layer.weight.dtype)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps).detach()
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FullGradGELU(nn.Module):
    """GELU with detached gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gate = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))
        gate = gate.detach()
        return x * gate


class FullGradReLU(nn.Module):
    """ReLU with detached gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gate = (x > 0).to(x.dtype).detach()
        return x * gate


class LinearGammaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, gamma):
        ctx.save_for_backward(input, weight, bias)
        ctx.gamma = gamma
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        gamma = ctx.gamma
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                output = torch.nn.functional.linear(input, weight, bias)
                contrib = input.unsqueeze(-1) * weight.t()
                contrib_pos = torch.clamp(contrib, min=0)
                bias_pos = torch.clamp(bias, min=0)
                denom = output.unsqueeze(-2) + gamma * (contrib_pos.sum(dim=-2, keepdim=True) + bias_pos)
                term = (contrib + gamma * contrib_pos) / denom
                term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                grad_input = torch.einsum("...j,...ij->...i", grad_output, term)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1) @ input
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_input, grad_weight, grad_bias, None

    @staticmethod
    def jvp(ctx, *grad_inputs):
        input_tangent = grad_inputs[0] if len(grad_inputs) > 0 else None
        weight_tangent = grad_inputs[1] if len(grad_inputs) > 1 else None
        bias_tangent = grad_inputs[2] if len(grad_inputs) > 2 else None
        if input_tangent is None and weight_tangent is None and bias_tangent is None:
            return None

        input, weight, bias = ctx.saved_tensors
        gamma = ctx.gamma
        out_tangent = None

        if input_tangent is not None:
            with torch.no_grad():
                output = torch.nn.functional.linear(input, weight, bias)
                contrib = input.unsqueeze(-1) * weight.t()
                contrib_pos = torch.clamp(contrib, min=0)
                bias_pos = torch.clamp(bias, min=0)
                denom = output.unsqueeze(-2) + gamma * (contrib_pos.sum(dim=-2, keepdim=True) + bias_pos)
                term = (contrib + gamma * contrib_pos) / denom
                term = torch.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
                out_tangent = torch.einsum("...ij,...i->...j", term, input_tangent)

        if weight_tangent is not None:
            contrib = torch.nn.functional.linear(input, weight_tangent, None)
            out_tangent = contrib if out_tangent is None else out_tangent + contrib

        if bias_tangent is not None:
            bias_term = bias_tangent
            if bias_term.dim() < input.dim():
                bias_term = bias_term.view(*([1] * (input.dim() - 1)), -1)
                bias_term = bias_term.expand(*input.shape[:-1], bias_term.shape[-1])
            out_tangent = bias_term if out_tangent is None else out_tangent + bias_term

        return out_tangent


class LinearGamma(nn.Module):
    """Linear layer that applies the gamma-rule."""

    def __init__(self, linear: nn.Linear, gamma: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        bias = (
            torch.zeros(linear.weight.shape[0], device=linear.weight.device, dtype=linear.weight.dtype)
            if linear.bias is None
            else linear.bias
        )
        self.bias = nn.Parameter(bias.detach().clone())
        self.weight.requires_grad = linear.weight.requires_grad
        self.bias.requires_grad = False if linear.bias is None else linear.bias.requires_grad
        self.gamma = float(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return LinearGammaFunction.apply(x, self.weight, self.bias, self.gamma)


class LibragradAttention(nn.Module):
    """Attention wrapper that detaches the softmax map before multiplying with V."""

    def __init__(self, attn: Attention) -> None:
        super().__init__()
        self.embedding_dim = attn.embedding_dim
        self.kv_in_dim = attn.kv_in_dim
        self.internal_dim = attn.internal_dim
        self.num_heads = attn.num_heads
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.dropout_p = attn.dropout_p

    @property
    def head_dim(self) -> int:
        return int(self.internal_dim // self.num_heads)

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, c // self.num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q)
        k = self._separate_heads(k)
        v = self._separate_heads(v)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        attn = _stop_gradient(attn)
        out = torch.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class LibragradRoPEAttention(RoPEAttention):
    """RoPEAttention variant with detached attention map (inherits RoPEAttention for isinstance checks)."""

    def __init__(self, attn: RoPEAttention) -> None:
        nn.Module.__init__(self)
        # Copy fields from wrapped attention
        self.embedding_dim = attn.embedding_dim
        self.kv_in_dim = attn.kv_in_dim
        self.internal_dim = attn.internal_dim
        self.num_heads = attn.num_heads
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.dropout_p = attn.dropout_p
        self.compute_cis = attn.compute_cis
        self.freqs_cis = attn.freqs_cis
        self.rope_k_repeat = attn.rope_k_repeat
        self.dropout_p = attn.dropout_p

    @property
    def head_dim(self) -> int:
        return int(self.internal_dim // self.num_heads)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:  # type: ignore[override]
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        # Align rope span: if k has extra tokens (memory etc.), exclude them from RoPE by default.
        if q.shape[-2] != k.shape[-2] and num_k_exclude_rope == 0 and k.shape[-2] > q.shape[-2]:
            num_k_exclude_rope = k.shape[-2] - q.shape[-2]
        if q.shape[-2] != k.shape[-2] and num_k_exclude_rope == 0:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary(q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        attn = _stop_gradient(attn)
        out = torch.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


def apply_rotary(q: torch.Tensor, k: torch.Tensor, *, freqs_cis: torch.Tensor, repeat_freqs_k: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from sam2.modeling.position_encoding import apply_rotary_enc

    return apply_rotary_enc(q, k, freqs_cis=freqs_cis, repeat_freqs_k=repeat_freqs_k)


class LibragradMultiScaleAttention(nn.Module):
    """MultiScaleAttention variant that detaches the attention map."""

    def __init__(self, attn: MultiScaleAttention) -> None:
        super().__init__()
        self.dim = attn.dim
        self.dim_out = attn.dim_out
        self.num_heads = attn.num_heads
        self.q_pool = attn.q_pool
        self.qkv = attn.qkv
        self.proj = attn.proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)

        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        head_dim = q.shape[-1]
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = _stop_gradient(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, H, W, -1)
        out = self.proj(out)
        return out


def do_pool(x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    return x


def _convert_activation(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.GELU):
        return FullGradGELU()
    if isinstance(module, nn.ReLU):
        return FullGradReLU()
    return None


def _convert_activation_fn(fn: Callable) -> Optional[Callable]:
    if fn is F.gelu:
        gelu_mod = FullGradGELU()
        return lambda x, m=gelu_mod: m(x)
    if fn is F.relu or getattr(fn, "__name__", "") == "relu":
        relu_mod = FullGradReLU()
        return lambda x, m=relu_mod: m(x)
    return None


def _convert_layer_norm(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.LayerNorm):
        return FullGradLayerNorm.from_layer(module)
    if isinstance(module, LayerNorm2d):
        return FullGradLayerNorm2d.from_layer(module)
    return None


def _convert_attention(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, RoPEAttention) and not isinstance(module, LibragradRoPEAttention):
        return LibragradRoPEAttention(module)
    if isinstance(module, Attention) and not isinstance(module, LibragradAttention):
        return LibragradAttention(module)
    if isinstance(module, MultiScaleAttention) and not isinstance(module, LibragradMultiScaleAttention):
        return LibragradMultiScaleAttention(module)
    return None


def _replace_attr(owner: object, name: str, new_value: object, record: List[Callable[[], None]]) -> None:
    old = getattr(owner, name, None)
    if old is None or old is new_value:
        return
    if isinstance(old, type(new_value)):
        return
    setattr(owner, name, new_value)
    if isinstance(old, nn.Module) and isinstance(new_value, nn.Module):
        new_value.train(old.training)
    record.append(lambda o=owner, n=name, v=old: setattr(o, n, v))


def _replace_modulelist(module_list: nn.ModuleList, idx: int, new_mod: nn.Module, record: List[Callable[[], None]]) -> None:
    old = module_list[idx]
    if old is new_mod:
        return
    if isinstance(old, type(new_mod)):
        return
    new_mod.train(old.training)
    module_list[idx] = new_mod
    record.append(lambda ml=module_list, i=idx, v=old: ml.__setitem__(i, v))


def enable_sam2_libragrad(model: nn.Module, *, gamma: Optional[float] = None) -> Callable[[], None]:
    """
    Apply libragrad-style patches to SAM2 modules:
      - LayerNorm/LayerNorm2d -> detach variance
      - GELU/ReLU -> detach gate
      - Attention (including RoPE/MultiScale) -> detach softmax map before V multiply
      - Optional gamma-rule for linear layers inside SAM2 MLPs and memory MLP.

    Returns:
        restore handle that reverts patched modules.
    """
    if getattr(model, "_libragrad_enabled", False):
        return lambda: None

    restores: List[Callable[[], None]] = []

    def _walk(mod: nn.Module) -> None:
        # Module-specific overrides
        if isinstance(mod, MLP):
            act_repl = _convert_activation(mod.act)
            if act_repl is not None:
                _replace_attr(mod, "act", act_repl, restores)
            if gamma is not None:
                for i, layer in enumerate(mod.layers):
                    if isinstance(layer, nn.Linear) and not isinstance(layer, LinearGamma):
                        _replace_modulelist(mod.layers, i, LinearGamma(layer, gamma), restores)

        if isinstance(mod, MemoryAttentionLayer):
            fn = _convert_activation_fn(mod.activation)
            if fn is not None:
                _replace_attr(mod, "activation", fn, restores)
            if gamma is not None:
                if isinstance(mod.linear1, nn.Linear) and not isinstance(mod.linear1, LinearGamma):
                    _replace_attr(mod, "linear1", LinearGamma(mod.linear1, gamma), restores)
                if isinstance(mod.linear2, nn.Linear) and not isinstance(mod.linear2, LinearGamma):
                    _replace_attr(mod, "linear2", LinearGamma(mod.linear2, gamma), restores)

        # Child traversal and generic replacements
        for child_name, child in mod.named_children():
            repl_norm = _convert_layer_norm(child)
            if repl_norm is not None:
                _replace_attr(mod, child_name, repl_norm, restores)
                continue

            repl_attn = _convert_attention(child)
            if repl_attn is not None:
                _replace_attr(mod, child_name, repl_attn, restores)
                continue

            repl_act = _convert_activation(child)
            if repl_act is not None:
                _replace_attr(mod, child_name, repl_act, restores)
                continue

            _walk(child)

    _walk(model)
    setattr(model, "_libragrad_enabled", True)

    def _restore() -> None:
        for fn in reversed(restores):
            try:
                fn()
            except Exception:
                pass
        try:
            delattr(model, "_libragrad_enabled")
        except Exception:
            pass

    return _restore


def enable_sae_libragrad(sae: nn.Module) -> None:
    """
    Patch SAE preprocessing to detach mean/std so gradients stop at input normalization.
    """
    if getattr(sae, "_libragrad_enabled", False):
        return
    if not hasattr(sae, "preprocess_input"):
        setattr(sae, "_libragrad_enabled", True)
        return
    orig_preprocess = sae.preprocess_input

    @wraps(orig_preprocess)
    def _preprocess(x: torch.Tensor, *args, **kwargs):
        out = orig_preprocess(x, *args, **kwargs)
        if not isinstance(out, tuple) or len(out) < 3:
            return out
        x_out, x_mean, x_std = out
        if torch.is_tensor(x_mean):
            x_mean = x_mean.detach()
        if torch.is_tensor(x_std):
            x_std = x_std.detach()
        return x_out, x_mean, x_std

    sae.preprocess_input = _preprocess  # type: ignore[assignment]
    setattr(sae, "_libragrad_enabled", True)


__all__ = [
    "enable_sam2_libragrad",
    "enable_sae_libragrad",
    "FullGradLayerNorm",
    "FullGradLayerNorm2d",
    "FullGradGELU",
    "FullGradReLU",
    "LibragradAttention",
    "LibragradRoPEAttention",
    "LibragradMultiScaleAttention",
    "LinearGamma",
]
