from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PerceiverForImageClassificationFourier

LOGGER = logging.getLogger(__name__)


class PerceiverLatentTap(nn.Module):
    """Observable tap that can remove a fixed null-image latent baseline."""

    def __init__(self, *, visual_residual: bool = False):
        super().__init__()
        self.visual_residual = bool(visual_residual)
        self.register_buffer("baseline", torch.empty(0), persistent=False)

    def set_baseline(self, baseline: torch.Tensor) -> None:
        if baseline.ndim != 3 or baseline.shape[0] != 1:
            raise ValueError(f"Expected baseline [1, latent, channel], got {tuple(baseline.shape)}")
        self.baseline = baseline.detach().clone()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.visual_residual and self.baseline.numel() > 0:
            return hidden_states - self.baseline.to(hidden_states)
        return hidden_states


def install_depth_tap(
    model: nn.Module,
    *,
    after_block: int = 4,
    visual_residual: bool = False,
) -> str:
    """Expose one repeated Perceiver block output without changing computation.

    Hugging Face's Perceiver reuses the same six self-attention modules for
    every block. A normal module hook therefore mixes all eight depths. This
    function inserts an Identity module after the final self-attention layer
    on exactly one repetition and returns its stable module path.
    """
    encoder = model.perceiver.encoder  # type: ignore[attr-defined]
    num_blocks = int(model.config.num_blocks)  # type: ignore[attr-defined]
    if not 1 <= after_block <= num_blocks:
        raise ValueError(f"after_block must be in [1, {num_blocks}], got {after_block}")
    if getattr(encoder, "_speclens_depth_tap_installed", False):
        installed = int(encoder._speclens_depth_tap_after_block)
        installed_residual = bool(encoder._speclens_depth_tap_visual_residual)
        if installed != after_block or installed_residual != bool(visual_residual):
            raise ValueError(
                "Perceiver depth tap already installed with "
                f"block={installed}, visual_residual={installed_residual}"
            )
        return str(encoder._speclens_depth_tap_path)

    suffix = "visual_residual_tap" if visual_residual else "tap"
    tap_name = f"depth_{after_block}_{suffix}"
    tap = PerceiverLatentTap(visual_residual=visual_residual)
    encoder.add_module(tap_name, tap)

    state = {"call": 0}
    last_self_attend = encoder.self_attends[-1]
    original_forward = last_self_attend.forward

    def reset_counter(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
        state["call"] = 0

    def tapped_forward(*args: Any, **kwargs: Any):
        outputs = original_forward(*args, **kwargs)
        state["call"] += 1
        if state["call"] != after_block:
            return outputs
        # The tap output is observable by hooks but is deliberately excluded
        # from the model's forward path.
        tap(outputs[0])
        return outputs

    encoder.register_forward_pre_hook(reset_counter)
    last_self_attend.forward = tapped_forward
    encoder._speclens_depth_tap_installed = True
    encoder._speclens_depth_tap_after_block = after_block
    encoder._speclens_depth_tap_visual_residual = bool(visual_residual)
    path = f"perceiver.encoder.{tap_name}"
    encoder._speclens_depth_tap_path = path
    LOGGER.info(
        "Installed Perceiver depth tap after repeated block %d (visual_residual=%s)",
        after_block,
        visual_residual,
    )
    return path


def _initialize_visual_baseline(
    model: nn.Module,
    *,
    tap_path: str,
    device: torch.device,
    image_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    tap = model.get_submodule(tap_path)
    if not isinstance(tap, PerceiverLatentTap) or not tap.visual_residual:
        return
    captured: list[torch.Tensor] = []
    handle = tap.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    parameter_dtype = next(model.parameters()).dtype
    null_image = torch.zeros(1, 3, image_size, image_size, device=device, dtype=parameter_dtype)
    try:
        amp = torch.autocast("cuda", dtype=amp_dtype) if use_amp and device.type == "cuda" else torch.no_grad()
        with torch.inference_mode(), amp:
            model(inputs=null_image)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one Perceiver tap output, captured {len(captured)}")
    tap.set_baseline(captured[0])
    LOGGER.info("Initialized null-image latent baseline with shape %s", tuple(captured[0].shape))


def _resolve_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    key = str(value or "float32").lower()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in aliases:
        raise ValueError(f"Unsupported Perceiver dtype: {value}")
    return aliases[key]


def load_perceiver_model(
    model_cfg: Dict[str, Any],
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> nn.Module:
    del rank, world_size
    name = model_cfg.get("hf_model", "deepmind/vision-perceiver-fourier")
    dtype = _resolve_dtype(model_cfg.get("torch_dtype", model_cfg.get("dtype", "float32")))
    cache_dir = model_cfg.get("cache_dir")
    model = PerceiverForImageClassificationFourier.from_pretrained(
        name,
        cache_dir=cache_dir,
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tap_mode = str(model_cfg.get("tap_mode", "raw")).lower()
    if tap_mode not in {"raw", "visual_residual"}:
        raise ValueError(f"Unsupported Perceiver tap_mode: {tap_mode}")
    tap_path = install_depth_tap(
        model,
        after_block=int(model_cfg.get("tap_after_block", 4)),
        visual_residual=tap_mode == "visual_residual",
    )
    model.eval().to(device)
    if tap_mode == "visual_residual":
        training_cfg = ((full_config or {}).get("sae") or {}).get("training") or {}
        amp_dtype = _resolve_dtype(training_cfg.get("amp_dtype", "float16"))
        _initialize_visual_baseline(
            model,
            tap_path=tap_path,
            device=device,
            image_size=int(model_cfg.get("image_size", 224)),
            use_amp=bool(training_cfg.get("use_amp", False)),
            amp_dtype=amp_dtype,
        )
    return model


__all__ = ["PerceiverLatentTap", "install_depth_tap", "load_perceiver_model"]
