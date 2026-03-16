from __future__ import annotations

from functools import wraps
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


from transformers.models.gemma2.modeling_gemma2 import (  # type: ignore
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2MLP,
    Gemma2RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
    ALL_ATTENTION_FUNCTIONS,
)


def _stop_gradient(t: torch.Tensor) -> torch.Tensor:
    return t.detach()


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
                pos = torch.clamp(output, min=0).unsqueeze(-2)
                neg = torch.clamp(output, max=0).unsqueeze(-2)
                contrib = input.unsqueeze(-1) * weight.t()
                contrib_pos = torch.clamp(contrib, min=0)
                contrib_neg = torch.clamp(contrib, max=0)
                bias_pos = torch.clamp(bias, min=0)
                bias_neg = torch.clamp(bias, max=0)
                term_pos = (contrib + gamma * contrib_pos) / (
                    output.unsqueeze(-2) + gamma * (contrib_pos.sum(dim=-2, keepdim=True) + bias_pos)
                )
                term_pos = torch.nan_to_num(term_pos, nan=0.0, posinf=0.0, neginf=0.0)
                term_neg = (contrib + gamma * contrib_neg) / (
                    output.unsqueeze(-2) + gamma * (contrib_neg.sum(dim=-2, keepdim=True) + bias_neg)
                )
                term_neg = torch.nan_to_num(term_neg, nan=0.0, posinf=0.0, neginf=0.0)
                scaled = pos * term_pos + neg * term_neg
                scaled = torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
                grad_input = torch.einsum("...j,...ij->...i", grad_output, scaled)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1) @ input
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))

        return grad_input, grad_weight, grad_bias, None


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


class LibragradGemma2RMSNorm(Gemma2RMSNorm):
    """
    RMSNorm variant that detaches the variance term (FullGrad-style).
    """

    @classmethod
    def from_layer(cls, layer: Gemma2RMSNorm) -> "LibragradGemma2RMSNorm":
        new = cls(layer.weight.shape[0], eps=layer.eps)
        if layer.weight is not None:
            new.weight = nn.Parameter(layer.weight.detach().clone())
            new.to(device=layer.weight.device, dtype=layer.weight.dtype)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps).detach()
        if self.weight is not None:
            normed = normed * (1.0 + self.weight)
        return normed


class LibragradGemma2Attention(nn.Module):
    """
    Attention wrapper that detaches the softmax map before multiplying with V.
    Mirrors Gemma2Attention forward while stopping gradients through the attention map.
    """

    def __init__(self, attn: Gemma2Attention) -> None:
        super().__init__()
        self.config = attn.config
        self.layer_idx = attn.layer_idx
        self.head_dim = attn.head_dim
        self.num_key_value_groups = attn.num_key_value_groups
        self.scaling = attn.scaling
        self.attention_dropout = attn.attention_dropout
        self.is_causal = attn.is_causal
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.attn_logit_softcapping = attn.attn_logit_softcapping
        self.sliding_window = attn.sliding_window

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                # fall back to eager when sdpa cannot return attentions
                pass
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        softcap = self.attn_logit_softcapping
        if softcap is not None:
            attn_weights = attn_weights / softcap
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * softcap
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = _stop_gradient(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1))
        return attn_output, attn_weights


class LibragradGemma2MLP(Gemma2MLP):
    """
    MLP variant that detaches the non-linear gate (Libra Gated Activation).
    Equivalent to x * gate.detach() for GELU-like activations.
    """

    @classmethod
    def from_layer(cls, mlp: Gemma2MLP) -> "LibragradGemma2MLP":
        new = cls(mlp.config)
        new.gate_proj = nn.Linear(
            mlp.gate_proj.in_features, mlp.gate_proj.out_features, bias=mlp.gate_proj.bias is not None
        )
        new.up_proj = nn.Linear(
            mlp.up_proj.in_features, mlp.up_proj.out_features, bias=mlp.up_proj.bias is not None
        )
        new.down_proj = nn.Linear(
            mlp.down_proj.in_features, mlp.down_proj.out_features, bias=mlp.down_proj.bias is not None
        )
        new.act_fn = mlp.act_fn

        # copy weights
        new.gate_proj.weight = nn.Parameter(mlp.gate_proj.weight.detach().clone())
        if mlp.gate_proj.bias is not None:
            new.gate_proj.bias = nn.Parameter(mlp.gate_proj.bias.detach().clone())

        new.up_proj.weight = nn.Parameter(mlp.up_proj.weight.detach().clone())
        if mlp.up_proj.bias is not None:
            new.up_proj.bias = nn.Parameter(mlp.up_proj.bias.detach().clone())

        new.down_proj.weight = nn.Parameter(mlp.down_proj.weight.detach().clone())
        if mlp.down_proj.bias is not None:
            new.down_proj.bias = nn.Parameter(mlp.down_proj.bias.detach().clone())

        # keep dtype/device consistent
        new.to(device=mlp.down_proj.weight.device, dtype=mlp.down_proj.weight.dtype)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        gate = self.act_fn(self.gate_proj(x))
        gate = gate.detach()
        up = self.up_proj(x)
        down_proj = self.down_proj(gate * up)
        return down_proj


def _replace(owner: nn.Module, name: str, new_mod: nn.Module, record: List[Tuple[nn.Module, str, nn.Module]]) -> None:
    old = getattr(owner, name, None)
    if old is None or old is new_mod:
        return
    if isinstance(old, type(new_mod)):
        return
    setattr(owner, name, new_mod)
    if isinstance(old, nn.Module):
        new_mod.train(old.training)
    record.append((owner, name, old))


def _convert_rmsnorm(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, Gemma2RMSNorm):
        return LibragradGemma2RMSNorm.from_layer(module)
    return None


def _convert_mlp(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, Gemma2MLP) and not isinstance(module, LibragradGemma2MLP):
        return LibragradGemma2MLP.from_layer(module)
    return None


def _convert_attention(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, Gemma2Attention) and not isinstance(module, LibragradGemma2Attention):
        return LibragradGemma2Attention(module)
    return None


def _apply_decoder_overrides(
    block: Gemma2DecoderLayer,
    *,
    gamma: Optional[float],
    record: List[Tuple[nn.Module, str, nn.Module]],
) -> None:
    for norm_name in (
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        mod = getattr(block, norm_name, None)
        repl = _convert_rmsnorm(mod)
        if repl is not None:
            _replace(block, norm_name, repl, record)

    attn = getattr(block, "self_attn", None)
    repl_attn = _convert_attention(attn)
    if repl_attn is not None:
        _replace(block, "self_attn", repl_attn, record)

    mlp = getattr(block, "mlp", None)
    repl_mlp = _convert_mlp(mlp)
    if repl_mlp is not None:
        _replace(block, "mlp", repl_mlp, record)
        mlp = repl_mlp

    if gamma is not None and isinstance(mlp, Gemma2MLP):
        for linear_name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(mlp, linear_name, None)
            if isinstance(linear, nn.Linear) and not isinstance(linear, LinearGamma):
                _replace(mlp, linear_name, LinearGamma(linear, gamma), record)


def enable_gemma_libragrad(model: nn.Module, *, gamma: Optional[float] = None) -> Callable[[], None]:
    """
    Apply libragrad patches to Gemma-2 style models:
      - RMSNorm -> detach variance term
      - Attention -> detach softmax map before V multiply
      - Optional gamma-rule for MLP linears when gamma is provided

    Returns:
        restore handle to revert all patches.
    """
    if getattr(model, "_libragrad_enabled", False):
        return lambda: None

    replaced: List[Tuple[nn.Module, str, nn.Module]] = []

    def _walk(mod: nn.Module) -> None:
        if isinstance(mod, Gemma2DecoderLayer):
            _apply_decoder_overrides(mod, gamma=gamma, record=replaced)
        for child_name, child in mod.named_children():
            repl_norm = _convert_rmsnorm(child)
            if repl_norm is not None:
                _replace(mod, child_name, repl_norm, replaced)
                continue
            repl_attn = _convert_attention(child)
            if repl_attn is not None:
                _replace(mod, child_name, repl_attn, replaced)
                continue
            _walk(child)

    _walk(model)
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


__all__ = [
    "LibragradGemma2RMSNorm",
    "LibragradGemma2Attention",
    "LibragradGemma2MLP",
    "enable_gemma_libragrad",
    "enable_sae_libragrad",
    "LinearGamma",
]
