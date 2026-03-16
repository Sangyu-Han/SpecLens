"""Backbone wrapper that *imports* ActivationTracer from utils."""
from __future__ import annotations
from typing import Sequence
import torch.nn as nn
from src.utils.hook_wrap import ActivationTracer


class BaseBackbone(nn.Module):
    def __init__(self, target_layers: Sequence[str] | None = None):
        super().__init__()
        self.model = self._build_model()
        self.target_layers = list(target_layers or self.default_target_layers())
        self.tracer = ActivationTracer(self.model, self.target_layers)

    # --------------------------------------------------
    def forward(self, *args, **kw):
        return self.model(*args, **kw)

    # must be overridden
    def _build_model(self) -> nn.Module: ...
    def default_target_layers(self) -> Sequence[str]: ...
