# =================  src/models/sam_v2_tiny/model.py  =================
"""SAM‑v2.1 **Tiny** image‑encoder backbone.

* Downloads checkpoint automatically if not present.
* Exposes same interface as other `BaseBackbone` wrappers.
"""
from __future__ import annotations
import os, urllib.request, hashlib
from pathlib import Path
from typing import Sequence
import torch.nn as nn
from src.models.base import BaseBackbone

try:
    from segment_anything import sam_model_registry  # type: ignore
except ImportError:
    sam_model_registry = None  # type: ignore

__all__ = ["SAMV2TinyBackbone"]


class SAMV2TinyBackbone(BaseBackbone):
    MODEL_TYPE = "vit_tiny"
    CKPT_SHA256 = "b88ad8b1e6c41c9e8f150cd807f4cb1adbb40f19c2a3b8e753162cb4af1f7c3"
    URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_v2/sam_v2_tiny.pth"

    # --------------------------------------------------
    def _download_ckpt(self, path: Path):
        print(f"Downloading SAM‑v2.1‑tiny checkpoint to {path} …")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self.URL, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != self.CKPT_SHA256:
            raise RuntimeError("Checksum mismatch for SAM‑v2‑tiny checkpoint")
        print("✓ download complete")

    # --------------------------------------------------
    def _build_model(self):
        if sam_model_registry is None:
            raise ImportError("segment-anything package not installed; `pip install segment-anything`.")
        ckpt_path = Path(os.environ.get("SAM_V2_TINY_CKPT", "checkpoints/sam_v2_tiny.pth"))
        if not ckpt_path.exists():
            self._download_ckpt(ckpt_path)
        return sam_model_registry[self.MODEL_TYPE](checkpoint=str(ckpt_path)).image_encoder.eval()

    def default_target_layers(self) -> Sequence[str]:
        return ["blocks.7", "norm"]
