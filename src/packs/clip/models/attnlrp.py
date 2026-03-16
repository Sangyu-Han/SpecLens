from __future__ import annotations

from functools import wraps
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention


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
        # inputs: (fn, input, epsilon)
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
        # functorch/jvp paths can swap arg order; defensively swap if needed.
        if not torch.is_tensor(input) and torch.is_tensor(factor):
            input, factor = factor, input
        return input

    @staticmethod
    def setup_context(ctx, inputs, output):
        # inputs is a tuple containing (input, factor)
        # Some functorch paths may wrap inputs in an additional tuple layer; only last element is factor.
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


class AttnLRPLayerNorm(nn.LayerNorm):
    """LayerNorm that stops gradients through the variance term (identity rule)."""

    @classmethod
    def from_layer(cls, layer: nn.LayerNorm) -> "AttnLRPLayerNorm":
        new = cls(layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.elementwise_affine)
        if layer.elementwise_affine and layer.weight is not None:
            new.weight = nn.Parameter(layer.weight.detach().clone())
            if layer.bias is not None:
                new.bias = nn.Parameter(layer.bias.detach().clone())
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


class AttnLRPGELU(nn.Module):
    """GELU with AttnLRP identity rule applied to the activation."""

    def __init__(self, gelu: nn.GELU) -> None:
        super().__init__()
        self.approximate = getattr(gelu, "approximate", "none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _identity_rule(lambda t: F.gelu(t, approximate=self.approximate), x)


class AttnLRPQuickGELU(nn.Module):
    """QuickGELU with AttnLRP identity rule applied to the activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _identity_rule(lambda t: t * torch.sigmoid(1.702 * t), x)


class AttnLRPAttention(nn.Module):
    """
    Attention wrapper that applies AttnLRP uniform rule to Q/K/V.
    Gradients on Q/K are divided by 4, on V by 2 before matmuls.
    """

    def __init__(self, attn: Attention) -> None:
        super().__init__()
        self.qkv = attn.qkv
        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.scale = attn.scale

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:  # type: ignore[override]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = _divide_gradient(q * self.scale, 4.0)
        k = _divide_gradient(k, 4.0)
        v = _divide_gradient(v, 2.0)

        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            # Broadcast mask to (B, heads, N, N) if given as (N, N) or (B, N, N).
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask.to(dtype=attn.dtype, device=attn.device)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


def _is_quick_gelu(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return name in {"quickgelu", "quick_gelu"}


def _convert_activation(module: nn.Module) -> torch.nn.Module | None:
    if isinstance(module, nn.GELU):
        return AttnLRPGELU(module)
    if _is_quick_gelu(module):
        return AttnLRPQuickGELU()
    return None


def _convert_layer_norm(module: nn.Module) -> torch.nn.Module | None:
    if isinstance(module, nn.LayerNorm):
        return AttnLRPLayerNorm.from_layer(module)
    return None


def _apply_block_overrides(
    block: nn.Module,
    *,
    record: List[Tuple[nn.Module, str, nn.Module]],
) -> None:
    def _replace(owner: nn.Module, name: str, new_mod: nn.Module) -> None:
        old = getattr(owner, name, None)
        if old is None or old is new_mod:
            return
        if isinstance(old, type(new_mod)):
            return
        setattr(owner, name, new_mod)
        if isinstance(old, nn.Module):
            new_mod.train(old.training)
        record.append((owner, name, old))

    for norm_name in ("norm", "norm1", "norm2"):
        mod = getattr(block, norm_name, None)
        repl = _convert_layer_norm(mod) if isinstance(mod, nn.LayerNorm) else None
        if repl is not None:
            _replace(block, norm_name, repl)

    attn = getattr(block, "attn", None)
    if isinstance(attn, Attention) and not isinstance(attn, AttnLRPAttention):
        _replace(block, "attn", AttnLRPAttention(attn))

    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        act = getattr(mlp, "act", None)
        repl_act = _convert_activation(act)
        if repl_act is not None:
            _replace(mlp, "act", repl_act)


def apply_attnlrp(model: nn.Module) -> Callable[[], None]:
    """
    In-place AttnLRP-style patch for timm/CLIP-like ViT blocks:
      - LayerNorm -> detach variance term
      - GELU/QuickGELU -> identity rule
      - Attention -> divide gradients on Q/K/V before matmuls

    Returns:
        restore handle that reverts patched modules.
    """
    if getattr(model, "_attnlrp_enabled", False):
        return lambda: None

    replaced: List[Tuple[nn.Module, str, nn.Module]] = []

    def _replace(owner: nn.Module, name: str, new_mod: nn.Module) -> None:
        old = getattr(owner, name, None)
        if old is None or old is new_mod:
            return
        if isinstance(old, type(new_mod)):
            return
        setattr(owner, name, new_mod)
        if isinstance(old, nn.Module):
            new_mod.train(old.training)
        replaced.append((owner, name, old))

    def _walk(mod: nn.Module) -> None:
        _apply_block_overrides(mod, record=replaced)

        for child_name, child in mod.named_children():
            ln = _convert_layer_norm(child)
            if ln is not None:
                _replace(mod, child_name, ln)
                continue
            if isinstance(child, Attention) and not isinstance(child, AttnLRPAttention):
                _replace(mod, child_name, AttnLRPAttention(child))
                continue
            act_mod = _convert_activation(child)
            if act_mod is not None:
                _replace(mod, child_name, act_mod)
                continue
            _walk(child)

    _walk(model)
    setattr(model, "_attnlrp_enabled", True)

    def _restore() -> None:
        for owner, name, old in reversed(replaced):
            try:
                setattr(owner, name, old)
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
    "apply_attnlrp",
    "enable_sae_attnlrp",
    "AttnLRPLayerNorm",
    "AttnLRPGELU",
    "AttnLRPQuickGELU",
    "AttnLRPAttention",
]
