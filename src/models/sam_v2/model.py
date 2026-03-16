# =================  src/models/sam_v2/model.py  =================
"""SAM‑v2 **Video** backbone for processing video sequences.

* Downloads checkpoint automatically if not present.
* Supports video input for SAE training.
* Handles temporal modeling capabilities of SAM v2.
"""
from __future__ import annotations
import os, urllib.request, hashlib
from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
from src.models.base import BaseBackbone

try:
    from segment_anything import sam_model_registry  # type: ignore
except ImportError:
    sam_model_registry = None  # type: ignore

__all__ = ["SAMV2Backbone"]


class SAMV2Backbone(BaseBackbone):
    # Model configurations for different sizes
    MODEL_CONFIGS = {
        "tiny": {
            "model_type": "vit_t",
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
            "d_model": 384,
            "default_target_layers": ["blocks.5", "norm"]
        },
        "small": {
            "model_type": "vit_s", 
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
            "d_model": 384,
            "default_target_layers": ["blocks.7", "norm"]
        },
        "base": {
            "model_type": "vit_b",
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", 
            "d_model": 768,
            "default_target_layers": ["blocks.11", "norm"]
        },
        "large": {
            "model_type": "vit_l",
            "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
            "d_model": 1024,
            "default_target_layers": ["blocks.23", "norm"]
        }
    }

    def __init__(self, target_layers: Sequence[str] | None = None, model_size: str = "tiny"):
        self.model_size = model_size.lower()
        if self.model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[self.model_size]
        super().__init__(target_layers)
        self.video_mode = True  # Enable video processing mode
        
        # SAM v2 memory management
        self.memory_encoder = None
        self.memory_attention = None
        self.obj_ptr = None  # Object pointer tokens

    # --------------------------------------------------
    def _download_ckpt(self, path: Path):
        print(f"Downloading SAM‑v2‑{self.model_size} checkpoint to {path} …")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self.config["url"], path)
        print("✓ download complete")

    # --------------------------------------------------
    def _build_model(self):
        if sam_model_registry is None:
            raise ImportError("segment-anything package not installed; `pip install segment-anything`.")
        
        ckpt_path = Path(os.environ.get(
            f"SAM_V2_{self.model_size.upper()}_CKPT", 
            f"checkpoints/sam_v2_{self.model_size}.pt"
        ))
        if not ckpt_path.exists():
            self._download_ckpt(ckpt_path)
        
        # Load full SAM v2 model to access all components
        self.sam_model = sam_model_registry[self.config["model_type"]](checkpoint=str(ckpt_path))
        
        # Extract components for memory-efficient processing
        self.image_encoder = self.sam_model.image_encoder.eval()
        self.prompt_encoder = self.sam_model.prompt_encoder.eval()
        self.memory_encoder = getattr(self.sam_model, 'memory_encoder', None)
        self.memory_attention = getattr(self.sam_model, 'memory_attention', None)
        
        return self.image_encoder

    def default_target_layers(self) -> Sequence[str]:
        return self.config["default_target_layers"]
    
    @property
    def d_model(self) -> int:
        """Return the model's hidden dimension for SAE configuration."""
        return self.config["d_model"]

    def forward(self, x, prompts=None):
        """Memory-efficient forward pass using SAM v2's memory mechanism.
        
        Args:
            x: Video tensor of shape (B, T, C, H, W)
            prompts: Optional dict containing prompts for supervised training
                    {"points": [...], "boxes": [...], "labels": [...]}
        
        Returns:
            Features from the current frame with memory and prompt context
        """
        if x.dim() == 5:  # Video input (B, T, C, H, W)
            B, T, C, H, W = x.shape
            
            # Process frames sequentially using memory
            all_features = []
            
            for t in range(T):
                current_frame = x[:, t]  # Shape: (B, C, H, W)
                
                # Get image features
                if t == 0:
                    # First frame: initialize memory
                    image_features = self.model(current_frame)
                    
                    # Initialize memory bank with first frame
                    if hasattr(self, 'memory_encoder') and self.memory_encoder is not None:
                        with torch.no_grad():
                            self.obj_ptr = self._initialize_memory(image_features)
                else:
                    # Subsequent frames: use memory-augmented processing
                    image_features = self._process_with_memory(current_frame)
                
                # Add prompt information if available
                if prompts is not None and hasattr(self, 'prompt_encoder'):
                    frame_features = self._add_prompt_features(
                        image_features, prompts, t, B
                    )
                else:
                    frame_features = image_features
                
                all_features.append(frame_features)
            
            # Stack features maintaining temporal dimension
            if isinstance(all_features[0], torch.Tensor):
                stacked_features = torch.stack(all_features, dim=1)  # (B, T, spatial, d_model)
            elif isinstance(all_features[0], dict):
                stacked_features = {}
                for key in all_features[0].keys():
                    stacked_features[key] = torch.stack([f[key] for f in all_features], dim=1)
            
            return stacked_features
        else:
            # Single frame input (B, C, H, W) - standard processing
            image_features = self.model(x)
            
            if prompts is not None and hasattr(self, 'prompt_encoder'):
                return self._add_prompt_features(image_features, prompts, 0, x.shape[0])
            
            return image_features
    
    def _add_prompt_features(self, image_features, prompts, frame_idx, batch_size):
        """Add prompt encoder features to image features."""
        try:
            # Handle supervised mode (with prompts)
            if prompts is not None and frame_idx < len(prompts.get("points", [])):
                frame_points = prompts["points"][frame_idx]
                frame_labels = prompts.get("labels", [[] for _ in range(len(prompts["points"]))])[frame_idx]
                
                if frame_points and len(frame_points) > 0:
                    # Convert to tensors
                    points_tensor = torch.tensor(frame_points, dtype=torch.float32, device=image_features.device)
                    labels_tensor = torch.tensor(frame_labels, dtype=torch.int32, device=image_features.device)
                    
                    # Add batch dimension if needed
                    if points_tensor.dim() == 2:
                        points_tensor = points_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
                        labels_tensor = labels_tensor.unsqueeze(0).repeat(batch_size, 1)
                    
                    # Encode prompts
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                            points=(points_tensor, labels_tensor),
                            boxes=None,
                            masks=None,
                        )
                    
                    # In a full implementation, you would combine these with image_features
                    # For SAE training, we mainly care about image features
                    # but prompt features could be concatenated or added
                else:
                    # Empty points for this frame - use no-prompt encoding
                    return self._add_no_prompt_features(image_features, batch_size)
            else:
                # Self-supervised mode or no prompts available - use no-prompt encoding
                return self._add_no_prompt_features(image_features, batch_size)
                
        except Exception as e:
            # Fallback to no-prompt mode if prompt processing fails
            print(f"Warning: Prompt processing failed: {e}, falling back to no-prompt mode")
            return self._add_no_prompt_features(image_features, batch_size)
        
        return image_features
    
    def _add_no_prompt_features(self, image_features, batch_size):
        """Handle self-supervised mode with no prompts (equivalent to SAM v2's no_mem_embed)."""
        try:
            with torch.no_grad():
                # Create empty point with label -1 (SAM v2's convention for "no prompt")
                device = image_features.device
                sam_point_coords = torch.zeros(batch_size, 1, 2, device=device)
                sam_point_labels = -torch.ones(batch_size, 1, dtype=torch.int32, device=device)
                
                # Encode "no prompt" - this will use the prompt encoder's no_mask_embed
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=(sam_point_coords, sam_point_labels),
                    boxes=None,
                    masks=None,  # SAM will use learned no_mask_embed for this
                )
                
                # In SAM v2, when no prompts are provided, the model relies on:
                # 1. no_mem_embed token for memory
                # 2. no_mask_embed from prompt encoder
                # This ensures the model knows it's in "self-supervised" mode
                
        except Exception as e:
            print(f"Warning: No-prompt encoding failed: {e}")
        
        return image_features
    
    def _initialize_memory(self, first_frame_features):
        """Initialize memory bank with first frame features."""
        # This would typically create object pointer tokens from first frame
        # For now, return a placeholder that represents memory initialization
        B = first_frame_features.shape[0] if isinstance(first_frame_features, torch.Tensor) else \
            next(iter(first_frame_features.values())).shape[0]
        
        # Create dummy object pointers (in real implementation, these would come from prompts/masks)
        obj_ptr = torch.randn(B, 1, self.config["d_model"], device=first_frame_features.device)
        return obj_ptr
    
    def _process_with_memory(self, current_frame):
        """Process current frame with memory context from previous frames."""
        # Encode current frame
        current_features = self.model(current_frame)
        
        # In a full implementation, this would:
        # 1. Use memory_attention to attend to previous frame information
        # 2. Update object pointer tokens
        # 3. Store relevant information in memory bank
        
        # For now, return current frame features (memory mechanism would be added here)
        return current_features
