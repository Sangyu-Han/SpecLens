from __future__ import annotations

import torch.nn as nn

from src.core.attribution.erf_adapter import VisionTransformerERFAdapter


class Dinov3ERFAdapter(VisionTransformerERFAdapter):
    pack_name = "dinov3"


def create_dinov3_erf_adapter(model: nn.Module) -> Dinov3ERFAdapter:
    return Dinov3ERFAdapter(model)


__all__ = ["Dinov3ERFAdapter", "create_dinov3_erf_adapter"]
