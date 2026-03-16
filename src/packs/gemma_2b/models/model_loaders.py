from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.core.sae.variants.jumprelu import JumpReLUSAE

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-2-2b"
DEFAULT_MODEL_IT = "google/gemma-2-2b-it"


def _format_width_token(width: int | str) -> str:
    """
    Gemma-Scope releases use width_16k / width_131k naming. Accept either int (e.g., 16000)
    or already-formatted strings (e.g., "16k").
    """
    if isinstance(width, str):
        return width.lower()
    # Map e.g., 16000 -> 16k, 131000 -> 131k when divisible by 1000.
    if width >= 1000 and width % 1000 == 0:
        return f"{width // 1000}k"
    return str(width)


def _resolve_dtype(dtype: Optional[str | torch.dtype]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "torch.float16", "half"):
        return torch.float16
    if key in ("fp32", "float32", "torch.float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_gemma_tokenizer(
    name_or_path: str,
    *,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    padding_side: Optional[str] = None,
):
    tok_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        tok_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **tok_kwargs)
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    return tokenizer


def load_gemma_model(
    model_cfg: Dict[str, Any],
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    load_tokenizer: bool = False,
    **_,
):
    """
    Load Gemma-2B / 2B-IT from HuggingFace Transformers.
    """
    cfg = model_cfg or {}
    device = torch.device(device)
    instruct = bool(cfg.get("instruct", cfg.get("it", False)))
    name = cfg.get("hf_path") or cfg.get("name")
    if name is None:
        name = DEFAULT_MODEL_IT if instruct else DEFAULT_MODEL

    trust_remote = bool(cfg.get("trust_remote_code", False))
    revision = cfg.get("revision")
    attn_impl = cfg.get("attn_impl")
    dtype = _resolve_dtype(cfg.get("dtype") or (full_config or {}).get("dtype"))
    low_cpu_mem = bool(cfg.get("low_cpu_mem", True))
    pretrained = bool(cfg.get("pretrained", True))
    device_map = cfg.get("device_map")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote,
        "low_cpu_mem_usage": low_cpu_mem,
    }
    if revision is not None:
        model_kwargs["revision"] = revision
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    else:
        base_cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote, revision=revision)
        model = AutoModelForCausalLM.from_config(base_cfg, **{k: v for k, v in model_kwargs.items() if k != "torch_dtype"})
        if dtype is not None:
            model = model.to(dtype)

    model.eval()
    if device_map is None:
        model.to(device)

    if dtype is not None and device_map is None:
        model = model.to(dtype)

    tokenizer = None
    if load_tokenizer or bool(cfg.get("return_tokenizer", False)):
        tokenizer = load_gemma_tokenizer(
            name,
            revision=revision,
            trust_remote_code=trust_remote,
            padding_side=cfg.get("padding_side"),
        )
        if cfg.get("bos_token") and tokenizer.bos_token is None:
            tokenizer.bos_token = cfg["bos_token"]

    if tokenizer is not None:
        return model, tokenizer
    return model


def _build_local_sae_cfg(cfg_dict: Dict[str, Any], *, device: torch.device, dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    act_size = int(cfg_dict.get("d_in") or cfg_dict.get("act_size"))
    dict_size = int(cfg_dict.get("d_sae") or cfg_dict.get("dict_size"))
    bandwidth = cfg_dict.get("bandwidth", cfg_dict.get("jumprelu_bandwidth", cfg_dict.get("jumprelu_bandwith", 0.05)))
    threshold_init = cfg_dict.get("init_threshold", cfg_dict.get("threshold", 0.0))

    return {
        "act_size": act_size,
        "dict_size": dict_size,
        "dtype": dtype or _resolve_dtype(cfg_dict.get("dtype")) or torch.float32,
        "device": device,
        "input_unit_norm": False,  # gemma-scope canonical: normalize_activations handles scaling instead
        "apply_b_dec_to_input": bool(cfg_dict.get("apply_b_dec_to_input", False)),
        "normalize_activations": cfg_dict.get("normalize_activations", cfg_dict.get("activation_norm", "none")),
        "jumprelu_bandwidth": bandwidth,
        "threshold_init": threshold_init,
        "jumprelu_init_threshold": threshold_init,
        "jumprelu_sparsity_loss_mode": cfg_dict.get("jumprelu_sparsity_loss_mode", cfg_dict.get("sparsity_loss_mode", "step")),
        "l1_coeff": cfg_dict.get("l1_coeff", cfg_dict.get("l1", 0.0)),
    }


def convert_saelens_to_local(
    sae,
    cfg_dict: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> JumpReLUSAE:
    """
    Convert a SAELens JumpReLU SAE to the local JumpReLUSAE implementation.
    """
    target_device = device or torch.device(cfg_dict.get("device", sae.W_dec.device))
    target_dtype = dtype or _resolve_dtype(cfg_dict.get("dtype")) or sae.W_dec.dtype
    local_cfg = _build_local_sae_cfg(cfg_dict, device=target_device, dtype=target_dtype)
    local = JumpReLUSAE(local_cfg).eval()

    # Map overlapping parameters; keep local defaults for extras like num_batches_not_active.
    src_state = sae.state_dict()
    dst_state = local.state_dict()
    mapped = {}
    for key, value in src_state.items():
        if key in dst_state:
            mapped[key] = value.detach().to(device=target_device, dtype=target_dtype)
    missing = set(dst_state.keys()) - set(mapped.keys())
    if missing:
        LOGGER.debug("[gemma_2b] SAE convert missing keys left at default: %s", sorted(missing))
    local.load_state_dict(mapped, strict=False)
    local.to(device=target_device, dtype=target_dtype)
    local.requires_grad_(False)
    return local


def load_sae_for_layer(
    layer: int,
    width: int | str,
    model_size: str,
    *,
    instruct: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    cache: Optional[Dict[int, JumpReLUSAE]] = None,
):
    """
    Load a Gemma-Scope JumpReLU SAE for a specific layer/width and convert it to the local implementation.
    """
    try:
        from sae_lens import SAE
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "sae_lens is required to load Gemma-Scope SAEs. "
            "Ensure third_party/SAELens-main is on sys.path or install the package."
        ) from exc

    release = f"gemma-scope-{model_size}-{'it' if instruct else 'pt'}-res-canonical"
    width_token = _format_width_token(width)
    sae_id = f"layer_{layer}/width_{width_token}/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
    )
    if device is not None:
        device = torch.device(device)
    local = convert_saelens_to_local(sae, cfg_dict, device=device, dtype=dtype)
    if cache is not None:
        cache[layer] = local
    return local, cfg_dict, sparsity


def parse_layer_index(layer: int | str) -> int:
    if isinstance(layer, int):
        return layer
    if isinstance(layer, str):
        import re

        if layer.isdigit():
            return int(layer)

        # Prefer an integer that follows "layers" or "blocks"
        tokens = re.split(r"[.@/]", layer)
        for i, tok in enumerate(tokens):
            if tok.isdigit() and i > 0 and tokens[i - 1] in {"layers", "blocks"}:
                return int(tok)

        # Fallback: first integer token in the string
        for tok in tokens:
            if tok.isdigit():
                return int(tok)
    raise ValueError(f"Could not parse layer index from '{layer}'")


def format_resid_hook_name(layer_idx: int, *, template: Optional[str] = None) -> str:
    if template:
        return template.format(layer=layer_idx, idx=layer_idx)
    return f"model.layers.{layer_idx}@0"


def build_layer_sae_map(
    layers: Sequence[int | str],
    *,
    width: int | str | Dict[int, int | str],
    model_size: str,
    instruct: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    hook_name_template: Optional[str] = None,
) -> Dict[str, JumpReLUSAE]:
    """
    Convenience utility to fetch SAEs for multiple layers and return a hook_name -> SAE mapping.
    """
    result: Dict[str, JumpReLUSAE] = {}
    width_map = width if isinstance(width, dict) else {}
    cache: Dict[int, JumpReLUSAE] = {}

    for spec in layers:
        idx = parse_layer_index(spec)
        if isinstance(width, dict):
            w = width_map.get(idx, width)
        else:
            w = width
        hook_name = format_resid_hook_name(idx, template=hook_name_template)
        sae, _, _ = load_sae_for_layer(
            idx,
            w,
            model_size,
            instruct=instruct,
            device=device,
            dtype=dtype,
            cache=cache,
        )
        result[hook_name] = sae
    return result


def smoke_test_sae_parity(
    layer: int,
    width: int,
    model_size: str,
    *,
    instruct: bool = False,
    batch_size: int = 2,
    seq_len: int = 4,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, Any]:
    """
    Quick parity check between SAELens JumpReLU and the local conversion.
    Returns max |diff| for encode/decode.
    """
    try:
        from sae_lens import SAE
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("sae_lens must be importable for the parity smoke test") from exc

    device = torch.device(device or "cpu")
    dtype = _resolve_dtype(dtype) or torch.float32

    release = f"gemma-scope-{model_size}-{'it' if instruct else 'pt'}-res-canonical"
    sae_id = f"layer_{layer}/width_{width}/canonical"
    sae, cfg_dict, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id)
    sae = sae.to(device=device, dtype=dtype)
    local = convert_saelens_to_local(sae, cfg_dict, device=device, dtype=dtype)

    d_in = int(cfg_dict.get("d_in") or cfg_dict.get("act_size") or local.W_dec.shape[-1])
    x = torch.randn(batch_size, seq_len, d_in, device=device, dtype=dtype)

    with torch.no_grad():
        acts_ref = sae.encode(x)
        recon_ref = sae.decode(acts_ref)

        acts_local = local.encode(x)
        recon_local = local.decode(acts_local)

    return {
        "encode_max_diff": float((acts_ref - acts_local).abs().max().item()),
        "decode_max_diff": float((recon_ref - recon_local).abs().max().item()),
        "sparsity": sparsity,
    }


__all__ = [
    "load_gemma_model",
    "load_gemma_tokenizer",
    "load_sae_for_layer",
    "build_layer_sae_map",
    "parse_layer_index",
    "format_resid_hook_name",
    "convert_saelens_to_local",
    "smoke_test_sae_parity",
]
