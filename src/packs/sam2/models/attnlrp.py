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


class _IdentityRuleFunction(torch.autograd.Function):
    """
    Implements the identity rule from AttnLRP via the Gradient*Input framework.
    Scales the backward pass by output / (input + eps).
    """

    @staticmethod
    def forward(fn: Callable[[torch.Tensor], torch.Tensor], input: torch.Tensor, epsilon: float = 1e-10):
        return fn(input)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, input, epsilon = inputs
        eps = float(epsilon) if not torch.is_tensor(epsilon) else epsilon
        if torch.is_tensor(input) and torch.is_tensor(output):
            scale = output / (input + eps)
            ctx.save_for_backward(scale)
            ctx.save_for_forward(scale)
        else:
            ctx.save_for_backward()
            ctx.save_for_forward()

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.saved_tensors:
            return None, grad_output, None
        (scale,) = ctx.saved_tensors
        grad_input = scale * grad_output
        return None, grad_input, None

    @staticmethod
    def jvp(ctx, *grad_inputs):
        input_tangent = grad_inputs[1] if len(grad_inputs) > 1 else None
        if input_tangent is None:
            return None
        if ctx.saved_for_forward:
            (scale,) = ctx.saved_for_forward
            return scale * input_tangent
        if ctx.saved_tensors:
            (scale,) = ctx.saved_tensors
            return scale * input_tangent
        return input_tangent


def _identity_rule(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    return _IdentityRuleFunction.apply(fn, x, epsilon)


class _DivideGradientFunction(torch.autograd.Function):
    """Uniform rule: divide incoming gradients by a constant factor."""

    @staticmethod
    def forward(input: torch.Tensor, factor: float = 2.0):
        if not torch.is_tensor(input) and torch.is_tensor(factor):
            input, factor = factor, input
        return input

    @staticmethod
    def setup_context(ctx, inputs, output):
        if len(inputs) >= 2:
            inp, fac = inputs[0], inputs[1]
            if not torch.is_tensor(inp) and torch.is_tensor(fac):
                inp, fac = fac, inp
            ctx.factor = float(fac)
        else:
            ctx.factor = 2.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output / ctx.factor, None

    @staticmethod
    def jvp(ctx, *grad_inputs):
        input_tangent = grad_inputs[0] if grad_inputs else None
        if input_tangent is None:
            return None
        return input_tangent / ctx.factor


def _divide_gradient(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    return _DivideGradientFunction.apply(x, factor)


def _sdpa_or_none(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    training: bool,
) -> Optional[torch.Tensor]:
    """
    Try to run scaled dot-product attention using PyTorch's SDPA kernels.

    SDPA can route to flash/memory-efficient kernels that substantially reduce
    peak memory compared to the explicit q@k^T materialisation used below.
    We keep a defensive fallback to the manual implementation if SDPA is not
    available or rejects the inputs.
    """
    sdpa = getattr(F, "scaled_dot_product_attention", None)
    if sdpa is None:
        return None
    drop = float(dropout_p) if training else 0.0
    try:
        return sdpa(q, k, v, dropout_p=drop, is_causal=False)
    except TypeError:
        # Older PyTorch versions may not accept some kwargs.
        try:
            return sdpa(q, k, v, drop, False)  # type: ignore[misc]
        except Exception:
            return None
    except RuntimeError:
        # Some SDPA kernels can reject shapes/dtypes; fall back gracefully.
        return None


class AttnLRPLayerNorm(nn.LayerNorm):
    """LayerNorm that stops gradients through the variance term (identity rule)."""

    @classmethod
    def from_layer(cls, layer: nn.LayerNorm) -> "AttnLRPLayerNorm":
        new = cls(layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.elementwise_affine)
        if layer.elementwise_affine and layer.weight is not None:
            new.weight = nn.Parameter(layer.weight.detach().clone())
            if layer.bias is not None:
                new.bias = nn.Parameter(layer.bias.detach().clone())
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


class AttnLRPLayerNorm2d(nn.Module):
    """LayerNorm2d variant that detaches the variance term."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    @classmethod
    def from_layer(cls, layer: LayerNorm2d) -> "AttnLRPLayerNorm2d":
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


class AttnLRPGELU(nn.Module):
    """GELU with AttnLRP identity rule applied to the activation."""

    def __init__(self, gelu: nn.GELU) -> None:
        super().__init__()
        self.approximate = getattr(gelu, "approximate", "none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _identity_rule(lambda t: F.gelu(t, approximate=self.approximate), x)


class AttnLRPReLU(nn.Module):
    """ReLU with AttnLRP identity rule applied to the activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _identity_rule(F.relu, x)


class AttnLRPAttention(nn.Module):
    """Attention wrapper that applies AttnLRP uniform rule to Q/K/V."""

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

        # Divide gradients before attention; the internal 1/sqrt(d) scaling
        # in SDPA commutes with this constant-factor rule.
        q = _divide_gradient(q, 4.0)
        k = _divide_gradient(k, 4.0)
        v = _divide_gradient(v, 2.0)

        out = _sdpa_or_none(q, k, v, dropout_p=self.dropout_p, training=self.training)
        if out is None:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.dropout_p > 0.0 and self.training:
                attn = F.dropout(attn, p=self.dropout_p)
            out = torch.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, *, freqs_cis: torch.Tensor, repeat_freqs_k: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    from sam2.modeling.position_encoding import apply_rotary_enc

    return apply_rotary_enc(q, k, freqs_cis=freqs_cis, repeat_freqs_k=repeat_freqs_k)


class AttnLRPRoPEAttention(RoPEAttention):
    """RoPEAttention variant with AttnLRP uniform rule (inherits RoPEAttention for isinstance checks)."""

    def __init__(self, attn: RoPEAttention) -> None:
        nn.Module.__init__(self)
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

    @property
    def head_dim(self) -> int:
        return int(self.internal_dim // self.num_heads)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0
    ) -> torch.Tensor:  # type: ignore[override]
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
        if q.shape[-2] != k.shape[-2] and num_k_exclude_rope == 0 and k.shape[-2] > q.shape[-2]:
            num_k_exclude_rope = k.shape[-2] - q.shape[-2]
        if q.shape[-2] != k.shape[-2] and num_k_exclude_rope == 0:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        q = _divide_gradient(q, 4.0)
        k = _divide_gradient(k, 4.0)
        v = _divide_gradient(v, 2.0)

        out = _sdpa_or_none(q, k, v, dropout_p=self.dropout_p, training=self.training)
        if out is None:
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.dropout_p > 0.0 and self.training:
                attn = F.dropout(attn, p=self.dropout_p)
            out = torch.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


def do_pool(x: torch.Tensor, pool: nn.Module) -> torch.Tensor:
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    return x


class AttnLRPMultiScaleAttention(nn.Module):
    """MultiScaleAttention variant that applies AttnLRP uniform rule to Q/K/V."""

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
        q = _divide_gradient(q, 4.0)
        k = _divide_gradient(k, 4.0)
        v = _divide_gradient(v, 2.0)

        out = _sdpa_or_none(q, k, v, dropout_p=0.0, training=self.training)
        if out is None:
            scale = head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, H, W, -1)
        out = self.proj(out)
        return out


def _convert_activation(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.GELU):
        return AttnLRPGELU(module)
    if isinstance(module, nn.ReLU):
        return AttnLRPReLU()
    return None


def _convert_activation_fn(fn: Callable) -> Optional[Callable]:
    if fn is F.gelu:
        return lambda x, f=F.gelu: _identity_rule(f, x)
    if fn is F.relu or getattr(fn, "__name__", "") == "relu":
        return lambda x, f=F.relu: _identity_rule(f, x)
    return None


def _convert_layer_norm(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.LayerNorm):
        return AttnLRPLayerNorm.from_layer(module)
    if isinstance(module, LayerNorm2d):
        return AttnLRPLayerNorm2d.from_layer(module)
    return None


def _convert_attention(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, RoPEAttention) and not isinstance(module, AttnLRPRoPEAttention):
        return AttnLRPRoPEAttention(module)
    if isinstance(module, Attention) and not isinstance(module, AttnLRPAttention):
        return AttnLRPAttention(module)
    if isinstance(module, MultiScaleAttention) and not isinstance(module, AttnLRPMultiScaleAttention):
        return AttnLRPMultiScaleAttention(module)
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


def enable_sam2_attnlrp(model: nn.Module) -> Callable[[], None]:
    """
    Apply AttnLRP-style patches to SAM2 modules:
      - LayerNorm/LayerNorm2d -> detach variance
      - GELU/ReLU -> identity rule
      - Attention (including RoPE/MultiScale) -> divide gradients on Q/K/V before matmuls

    Returns:
        restore handle that reverts patched modules.
    """
    if getattr(model, "_attnlrp_enabled", False):
        return lambda: None

    restores: List[Callable[[], None]] = []

    def _walk(mod: nn.Module) -> None:
        if isinstance(mod, MLP):
            act_repl = _convert_activation(mod.act)
            if act_repl is not None:
                _replace_attr(mod, "act", act_repl, restores)

        if isinstance(mod, MemoryAttentionLayer):
            fn = _convert_activation_fn(mod.activation)
            if fn is not None:
                _replace_attr(mod, "activation", fn, restores)

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
    setattr(model, "_attnlrp_enabled", True)

    def _restore() -> None:
        for fn in reversed(restores):
            try:
                fn()
            except Exception:
                pass
        try:
            delattr(model, "_attnlrp_enabled")
        except Exception:
            pass

    return _restore


def enable_sae_attnlrp(sae: nn.Module) -> None:
    """
    Patch SAE preprocessing to detach mean/std so gradients stop at input normalization.
    """
    if getattr(sae, "_attnlrp_enabled", False):
        return
    if not hasattr(sae, "preprocess_input"):
        setattr(sae, "_attnlrp_enabled", True)
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
    setattr(sae, "_attnlrp_enabled", True)


__all__ = [
    "enable_sam2_attnlrp",
    "enable_sae_attnlrp",
    "AttnLRPLayerNorm",
    "AttnLRPLayerNorm2d",
    "AttnLRPGELU",
    "AttnLRPReLU",
    "AttnLRPAttention",
    "AttnLRPRoPEAttention",
    "AttnLRPMultiScaleAttention",
]
