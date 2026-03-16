from __future__ import annotations

from functools import wraps
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention


def _stop_gradient(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


class FullGradLayerNorm(nn.LayerNorm):
    """LayerNorm that detaches the variance term (FullGrad style)."""

    @classmethod
    def from_layer(cls, layer: nn.LayerNorm) -> "FullGradLayerNorm":
        new = cls(layer.normalized_shape, eps=layer.eps, elementwise_affine=layer.elementwise_affine)
        if layer.elementwise_affine and layer.weight is not None:
            # Recreate parameters on the same device/dtype to avoid host/device mismatches.
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


class FullGradGELU(nn.Module):
    """GELU with detached gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gate = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=x.dtype, device=x.device))))
        gate = gate.detach()
        return x * gate


class FullGradQuickGELU(nn.Module):
    """QuickGELU with detached gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gate = torch.sigmoid(1.702 * x).detach()
        return x * gate


class FullGradNormalize(nn.Module):
    """Normalize layer with detached norm."""

    def __init__(self, dim: int = -1, eps: float = 1e-12) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    @classmethod
    def from_module(cls, mod: nn.Module) -> "FullGradNormalize":
        dim = getattr(mod, "dim", -1)
        eps = getattr(mod, "eps", 1e-12)
        return cls(dim=dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        denom = torch.norm(x, dim=self.dim, keepdim=True)
        denom = torch.clamp(denom, min=self.eps).detach()
        return x / denom


class LinearGammaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, gamma):
        ctx.save_for_backward(input, weight, bias)
        ctx.gamma = gamma
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, gamma = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.gamma = gamma

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


class LinearAlphaBetaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, alpha, beta):
        ctx.save_for_backward(input, weight, bias)
        ctx.alpha = alpha
        ctx.beta = beta
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weight, bias, alpha, beta = inputs
        ctx.save_for_backward(input, weight, bias)
        ctx.alpha = alpha
        ctx.beta = beta

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        alpha = ctx.alpha
        beta = ctx.beta
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                contrib = input.unsqueeze(-1) * weight.t()
                contrib_pos = torch.clamp(contrib, min=0)
                contrib_neg = torch.clamp(contrib, max=0)
                bias_pos = torch.clamp(bias, min=0)
                bias_neg = torch.clamp(bias, max=0)
                denom_pos = contrib_pos.sum(dim=-2, keepdim=True) + bias_pos
                denom_neg = contrib_neg.sum(dim=-2, keepdim=True) + bias_neg
                term_pos = contrib_pos / denom_pos
                term_neg = contrib_neg / denom_neg
                term_pos = torch.nan_to_num(term_pos, nan=0.0, posinf=0.0, neginf=0.0)
                term_neg = torch.nan_to_num(term_neg, nan=0.0, posinf=0.0, neginf=0.0)
                scaled = alpha * term_pos - beta * term_neg
                scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
                grad_input = torch.einsum("...j,...ij->...i", grad_output, scaled)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1) @ input
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_input, grad_weight, grad_bias, None, None


class LinearGamma(nn.Module):
    """Linear layer that applies the gamma-rule (optional)."""

    def __init__(self, linear: nn.Linear, gamma: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            bias = torch.zeros(linear.weight.shape[0], device=linear.weight.device)
        else:
            bias = linear.bias.detach().clone()
        self.bias = nn.Parameter(bias)
        self.gamma = float(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return LinearGammaFunction.apply(x, self.weight, self.bias, self.gamma)


class LinearAlphaBeta(nn.Module):
    """Linear layer that applies the alpha-beta rule (optional)."""

    def __init__(self, linear: nn.Linear, alpha: float, beta: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            bias = torch.zeros(linear.weight.shape[0], device=linear.weight.device)
        else:
            bias = linear.bias.detach().clone()
        self.bias = nn.Parameter(bias)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return LinearAlphaBetaFunction.apply(x, self.weight, self.bias, self.alpha, self.beta)


_LINEAR_RULE_TYPES = (LinearGamma, LinearAlphaBeta)


class LibragradAttention(nn.Module):
    """
    Attention wrapper that detaches the softmax map before multiplying with V.
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

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask.to(dtype=attn.dtype, device=attn.device)
        attn = attn.softmax(dim=-1)
        attn = _stop_gradient(attn)
        attn = self.attn_drop(attn)
        attn_out = attn @ v

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out


def _is_quick_gelu(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return name in {"quickgelu", "quick_gelu"}


def _convert_activation(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.GELU):
        return FullGradGELU()
    if _is_quick_gelu(module):
        return FullGradQuickGELU()
    return None


def _convert_layer_norm(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.LayerNorm):
        return FullGradLayerNorm.from_layer(module)
    return None


def _convert_normalize(module: nn.Module) -> Optional[nn.Module]:
    if module.__class__.__name__.lower() == "normalize":
        return FullGradNormalize.from_module(module)
    return None


def _resolve_linear_rule(
    linear: nn.Linear,
    *,
    gamma: Optional[float],
    alpha: Optional[float],
    beta: Optional[float],
) -> Optional[nn.Module]:
    if gamma is not None:
        return LinearGamma(linear, gamma)
    if alpha is not None and beta is not None:
        return LinearAlphaBeta(linear, alpha, beta)
    return None


def _apply_block_overrides(
    block: nn.Module,
    *,
    gamma: Optional[float],
    alpha: Optional[float],
    beta: Optional[float],
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
    if isinstance(attn, Attention) and not isinstance(attn, LibragradAttention):
        _replace(block, "attn", LibragradAttention(attn))

    mlp = getattr(block, "mlp", None)
    if mlp is not None:
        act = getattr(mlp, "act", None)
        repl_act = _convert_activation(act)
        if repl_act is not None:
            _replace(mlp, "act", repl_act)
        if gamma is not None or alpha is not None or beta is not None:
            for linear_name in ("fc1", "fc2"):
                linear = getattr(mlp, linear_name, None)
                if isinstance(linear, nn.Linear) and not isinstance(linear, _LINEAR_RULE_TYPES):
                    repl = _resolve_linear_rule(linear, gamma=gamma, alpha=alpha, beta=beta)
                    if repl is not None:
                        _replace(mlp, linear_name, repl)


def apply_libragrad(
    model: nn.Module,
    *,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> Callable[[], None]:
    """
    In-place FullGrad-style patch for timm/CLIP-like ViT blocks:
      - LayerNorm -> detach variance
      - GELU/QuickGELU -> detach gate
      - Normalize -> detach norm
      - Attention -> detach softmax map before V multiply
      - Optional gamma-rule for linear layers (fc1/fc2/head) when gamma is provided.
      - Optional alpha-beta rule for linear layers when alpha and beta are provided.

    Returns:
        restore handle that reverts patched modules.
    """
    if gamma is not None and (alpha is not None or beta is not None):
        raise ValueError("gamma and alpha/beta are mutually exclusive.")
    if (alpha is None) != (beta is None):
        raise ValueError("alpha and beta must be provided together.")
    if getattr(model, "_libragrad_enabled", False):
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
        _apply_block_overrides(mod, gamma=gamma, alpha=alpha, beta=beta, record=replaced)

        for child_name, child in mod.named_children():
            ln = _convert_layer_norm(child)
            if ln is not None:
                _replace(mod, child_name, ln)
                continue
            norm_mod = _convert_normalize(child)
            if norm_mod is not None:
                _replace(mod, child_name, norm_mod)
                continue
            if isinstance(child, Attention) and not isinstance(child, LibragradAttention):
                _replace(mod, child_name, LibragradAttention(child))
                continue
            act_mod = _convert_activation(child)
            if act_mod is not None:
                _replace(mod, child_name, act_mod)
                continue
            _walk(child)

    _walk(model)

    if (gamma is not None or alpha is not None or beta is not None) and hasattr(model, "head"):
        head = getattr(model, "head")
        if isinstance(head, nn.Linear) and not isinstance(head, _LINEAR_RULE_TYPES):
            repl = _resolve_linear_rule(head, gamma=gamma, alpha=alpha, beta=beta)
            if repl is not None:
                _replace(model, "head", repl)

    setattr(model, "_libragrad_enabled", True)

    def _restore() -> None:
        for owner, name, old in reversed(replaced):
            try:
                setattr(owner, name, old)
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


def enable_clip_libragrad(
    model: nn.Module,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
):
    """
    Apply libragrad patches to a CLIP ViT model (kept for backwards-compat).
    """
    return apply_libragrad(model, gamma=gamma, alpha=alpha, beta=beta)


# Backwards-compatible aliases for downstream imports
LibragradLayerNorm = FullGradLayerNorm


__all__ = [
    "apply_libragrad",
    "enable_clip_libragrad",
    "enable_sae_libragrad",
    "FullGradLayerNorm",
    "FullGradGELU",
    "FullGradQuickGELU",
    "FullGradNormalize",
    "LibragradLayerNorm",
    "LibragradAttention",
    "LinearGamma",
    "LinearAlphaBeta",
]
