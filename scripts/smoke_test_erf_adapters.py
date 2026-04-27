from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.packs.clip.attribution.erf_adapter import create_clip_erf_adapter
from src.packs.clip.models.model_loaders import load_clip_model
from src.packs.dinov3.attribution.erf_adapter import create_dinov3_erf_adapter
from src.packs.dinov3.models.model_loaders import load_dinov3_model
from src.packs.siglip.attribution.erf_adapter import create_siglip_erf_adapter
from src.packs.siglip.models.model_loaders import load_siglip_model


@dataclass
class SmokeResult:
    pack: str
    model_name: str
    status: str
    block_idx: int | None = None
    num_patches: int | None = None
    masked_all_ones_max_abs_err: float | None = None
    alpha_one_max_abs_err: float | None = None
    block_shape: tuple[int, ...] | None = None
    note: str | None = None


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _run_smoke(
    *,
    pack: str,
    model_name: str,
    loader: Callable,
    adapter_factory: Callable,
) -> SmokeResult:
    device = torch.device("cpu")
    try:
        model = loader({"name": model_name, "pretrained": False}, device=device)
    except RuntimeError as exc:
        cause = getattr(exc, "__cause__", None)
        if "Unknown model" in str(exc) or (cause is not None and "Unknown model" in str(cause)):
            return SmokeResult(
                pack=pack,
                model_name=model_name,
                status="skipped",
                note=f"model registry unavailable in current timm build: {exc}",
            )
        raise

    adapter = adapter_factory(model)
    x = torch.randn(1, 3, 224, 224, device=device)
    block_idx = min(3, len(adapter.base_model.blocks) - 1)
    capture = adapter.capture_block0_input_and_target_block_out(x, block_idx=block_idx)

    do_forward_masked, get_block_out = adapter.make_masked_forward(x, capture, block_idx=block_idx)
    do_forward_masked(torch.ones(capture.num_patches, device=device))
    masked_full = get_block_out().detach()

    set_alpha, do_forward, _get_h_alpha, get_alpha_block_out = adapter.make_alpha_forward(
        x,
        capture,
        block_idx=block_idx,
    )
    set_alpha(1.0)
    do_forward()
    alpha_full = get_alpha_block_out().detach()

    # Also ensure the fully masked path is executable.
    do_forward_masked(torch.zeros(capture.num_patches, device=device))
    fully_masked = get_block_out().detach()

    return SmokeResult(
        pack=pack,
        model_name=model_name,
        status="ok",
        block_idx=block_idx,
        num_patches=capture.num_patches,
        masked_all_ones_max_abs_err=_max_abs_err(masked_full, capture.target_block_output),
        alpha_one_max_abs_err=_max_abs_err(alpha_full, capture.target_block_output),
        block_shape=tuple(int(v) for v in fully_masked.shape),
        note=None,
    )


def main() -> None:
    torch.manual_seed(0)

    results = [
        _run_smoke(
            pack="clip",
            model_name="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
            loader=load_clip_model,
            adapter_factory=create_clip_erf_adapter,
        ),
        _run_smoke(
            pack="dinov3",
            model_name="vit_small_patch16_dinov3",
            loader=load_dinov3_model,
            adapter_factory=create_dinov3_erf_adapter,
        ),
        _run_smoke(
            pack="siglip",
            model_name="vit_base_patch16_siglip_224",
            loader=load_siglip_model,
            adapter_factory=create_siglip_erf_adapter,
        ),
    ]

    print(json.dumps([asdict(r) for r in results], indent=2))

    bad = []
    for result in results:
        if result.status != "ok":
            continue
        if result.masked_all_ones_max_abs_err is None or result.alpha_one_max_abs_err is None:
            bad.append(f"{result.pack}: missing error metrics")
            continue
        if result.masked_all_ones_max_abs_err > 1e-5:
            bad.append(
                f"{result.pack}: masked all-ones mismatch {result.masked_all_ones_max_abs_err:.3e}"
            )
        if result.alpha_one_max_abs_err > 1e-5:
            bad.append(f"{result.pack}: alpha=1 mismatch {result.alpha_one_max_abs_err:.3e}")

    if bad:
        raise SystemExit("ERF smoke test failed:\n" + "\n".join(bad))


if __name__ == "__main__":
    main()
