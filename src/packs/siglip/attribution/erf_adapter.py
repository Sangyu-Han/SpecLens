from __future__ import annotations

import torch.nn as nn

from src.core.attribution.erf_adapter import VisionTransformerERFAdapter


class SiglipERFAdapter(VisionTransformerERFAdapter):
    pack_name = "siglip"


def create_siglip_erf_adapter(model: nn.Module) -> SiglipERFAdapter:
    return SiglipERFAdapter(model)


__all__ = ["SiglipERFAdapter", "create_siglip_erf_adapter"]
