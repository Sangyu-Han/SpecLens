from __future__ import annotations

import torch.nn as nn

from src.core.attribution.erf_adapter import VisionTransformerERFAdapter


class CLIPERFAdapter(VisionTransformerERFAdapter):
    pack_name = "clip"


def create_clip_erf_adapter(model: nn.Module) -> CLIPERFAdapter:
    return CLIPERFAdapter(model)


__all__ = ["CLIPERFAdapter", "create_clip_erf_adapter"]
