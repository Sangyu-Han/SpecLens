# src/packs/mask2former/dataset/builders.py
"""
Dataset builders for Mask2Former on SA-V dataset.

Unlike SAM2, Mask2Former doesn't require prompt inputs (points, boxes).
It processes frames directly for instance/semantic/panoptic segmentation.
This makes the data pipeline much simpler.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

LOGGER = logging.getLogger(__name__)

# Default normalization values (ImageNet statistics, detectron2 style)
# Note: detectron2 uses BGR and different scale (0-255)
DEFAULT_PIXEL_MEAN = [123.675, 116.280, 103.530]  # BGR
DEFAULT_PIXEL_STD = [58.395, 57.120, 57.375]


class SAVImageDataset(Dataset):
    """
    Simple image dataset for SA-V frames.
    
    Unlike the video-based SAM2 dataset, this treats each frame independently.
    No prompts, no temporal context - just images and optional masks.
    
    Directory structure expected:
        img_folder/
            video_id_1/
                frame_001.jpg
                frame_002.jpg
                ...
            video_id_2/
                ...
        
        gt_folder/ (optional, for training/evaluation)
            video_id_1/
                frame_001.png  (mask)
                ...
    """
    
    def __init__(
        self,
        img_folder: str,
        gt_folder: Optional[str] = None,
        file_list_txt: Optional[str] = None,
        resize: int = 1024,
        pixel_mean: List[float] = DEFAULT_PIXEL_MEAN,
        pixel_std: List[float] = DEFAULT_PIXEL_STD,
        is_train: bool = False,
        max_frames_per_video: Optional[int] = None,
        frame_stride: int = 1,
    ):
        """
        Args:
            img_folder: Root folder containing video subdirectories with frames
            gt_folder: Root folder containing ground truth masks (optional)
            file_list_txt: Optional text file listing video IDs to include
            resize: Target size for resizing images
            pixel_mean: Mean for normalization (BGR, 0-255 scale)
            pixel_std: Std for normalization (BGR, 0-255 scale)
            is_train: Whether this is for training (affects augmentation)
            max_frames_per_video: Limit frames per video (None = all)
            frame_stride: Sample every N-th frame
        """
        self.img_folder = Path(img_folder)
        self.gt_folder = Path(gt_folder) if gt_folder else None
        self.resize = resize
        self.pixel_mean = torch.tensor(pixel_mean).view(3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(3, 1, 1)
        self.is_train = is_train
        self.max_frames_per_video = max_frames_per_video
        self.frame_stride = frame_stride
        
        # Build frame list
        self.frames: List[Dict[str, Any]] = []
        self._build_frame_list(file_list_txt)
        
        LOGGER.info(
            "[SAVImageDataset] Loaded %d frames from %s (is_train=%s)",
            len(self.frames), self.img_folder, self.is_train
        )
    
    def _build_frame_list(self, file_list_txt: Optional[str] = None):
        """Build list of all frames to process."""
        # Get video IDs
        if file_list_txt and Path(file_list_txt).exists():
            with open(file_list_txt, "r") as f:
                video_ids = [line.strip() for line in f if line.strip()]
        else:
            # List all subdirectories
            video_ids = []
            if self.img_folder.exists():
                for item in sorted(self.img_folder.iterdir()):
                    if item.is_dir():
                        video_ids.append(item.name)
        
        # Collect frames from each video
        sample_idx = 0
        for video_id in video_ids:
            video_path = self.img_folder / video_id
            if not video_path.exists():
                continue
            
            # Get all image files
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(video_path.glob(ext))
            image_files = sorted(image_files)
            
            # Apply stride and limit
            image_files = image_files[::self.frame_stride]
            if self.max_frames_per_video:
                image_files = image_files[:self.max_frames_per_video]
            
            for img_path in image_files:
                frame_info = {
                    "sample_id": sample_idx,
                    "video_id": video_id,
                    "frame_name": img_path.stem,
                    "image_path": str(img_path),
                }
                
                # Check for GT mask
                if self.gt_folder:
                    mask_path = self.gt_folder / video_id / f"{img_path.stem}.png"
                    if mask_path.exists():
                        frame_info["mask_path"] = str(mask_path)
                
                self.frames.append(frame_info)
                sample_idx += 1
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame_info = self.frames[idx]
        
        # Load image
        image = Image.open(frame_info["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        
        # Resize
        if self.resize:
            # Maintain aspect ratio, resize shorter side
            scale = self.resize / min(orig_h, orig_w)
            new_h = int(orig_h * scale)
            new_w = int(orig_w * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Convert to tensor (C, H, W), float32, 0-255 range (detectron2 style)
        image_tensor = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float()
        
        # Convert RGB to BGR (detectron2 convention)
        image_tensor = image_tensor[[2, 1, 0], :, :]
        
        # Note: normalization is done by the model, not here
        # detectron2 models handle normalization internally
        
        result = {
            "image": image_tensor,
            "sample_id": frame_info["sample_id"],
            "image_path": frame_info["image_path"],
            "video_id": frame_info["video_id"],
            "frame_name": frame_info["frame_name"],
            "height": orig_h,
            "width": orig_w,
        }
        
        # Load mask if available
        if "mask_path" in frame_info:
            mask = Image.open(frame_info["mask_path"])
            if self.resize:
                mask = mask.resize((new_w, new_h), Image.NEAREST)
            result["mask"] = torch.from_numpy(__import__("numpy").array(mask)).long()
        
        return result


def mask2former_collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for Mask2Former.
    
    Handles variable-size images by padding to the largest size in the batch.
    Returns format compatible with Mask2FormerAdapter.preprocess_input().
    """
    items = list(batch)
    if not items:
        return {}
    
    # Find max dimensions
    max_h = max(item["image"].shape[1] for item in items)
    max_w = max(item["image"].shape[2] for item in items)
    
    # Pad to common size (divisible by 32 for Mask2Former)
    pad_h = ((max_h + 31) // 32) * 32
    pad_w = ((max_w + 31) // 32) * 32
    
    # Stack images with padding + build pixel masks (1=valid, 0=pad)
    images = []
    pixel_masks = []
    for item in items:
        img = item["image"]
        c, h, w = img.shape
        padded = torch.zeros(c, pad_h, pad_w, dtype=img.dtype)
        padded[:, :h, :w] = img
        images.append(padded)
        mask = torch.zeros(pad_h, pad_w, dtype=torch.bool)
        mask[:h, :w] = True
        pixel_masks.append(mask)
    
    images = torch.stack(images, dim=0)
    
    # Collect metadata
    sample_ids = torch.tensor([item["sample_id"] for item in items], dtype=torch.long)
    image_paths = [item["image_path"] for item in items]
    heights = [item["height"] for item in items]
    widths = [item["width"] for item in items]
    
    result = {
        "images": images,
        "sample_ids": sample_ids,
        "image_paths": image_paths,
        "heights": heights,
        "widths": widths,
        "pixel_mask": torch.stack(pixel_masks, dim=0),
    }
    
    # Include masks if present
    if "mask" in items[0]:
        masks = []
        for item in items:
            mask = item["mask"]
            h, w = mask.shape
            padded = torch.zeros(pad_h, pad_w, dtype=mask.dtype)
            padded[:h, :w] = mask
            masks.append(padded)
        result["masks"] = torch.stack(masks, dim=0)
    
    return result


def build_sav_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> Dict[str, Any]:
    """
    Build SA-V dataset for Mask2Former training.
    
    Args:
        dataset_cfg: Dataset configuration containing:
            - img_folder: Path to images
            - gt_folder: Path to masks (optional)
            - file_list_txt: Video ID list (optional)
            - resize: Target image size
            - batch_size: Batch size
            - is_train: Training mode flag
        rank: DDP rank
        world_size: DDP world size
        full_config: Full config dict (optional)
    
    Returns:
        Dict with 'dataset', 'collate_fn', and 'sampler'
    """
    img_folder = dataset_cfg["img_folder"]
    gt_folder = dataset_cfg.get("gt_folder")
    file_list_txt = dataset_cfg.get("file_list_txt")
    resize = int(dataset_cfg.get("resize", 1024))
    is_train = bool(dataset_cfg.get("is_train", True))
    max_frames = dataset_cfg.get("max_frames_per_video")
    frame_stride = int(dataset_cfg.get("frame_stride", 1))
    
    # Pixel normalization (optional override)
    pixel_mean = dataset_cfg.get("pixel_mean", DEFAULT_PIXEL_MEAN)
    pixel_std = dataset_cfg.get("pixel_std", DEFAULT_PIXEL_STD)
    
    dataset = SAVImageDataset(
        img_folder=img_folder,
        gt_folder=gt_folder,
        file_list_txt=file_list_txt,
        resize=resize,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        is_train=is_train,
        max_frames_per_video=max_frames,
        frame_stride=frame_stride,
    )
    
    shuffle = is_train
    drop_last = is_train
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    
    return {
        "dataset": dataset,
        "collate_fn": mask2former_collate_fn,
        "sampler": sampler,
    }


def build_indexing_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    world_size: int,
    rank: int,
    **_,
) -> Tuple[Dataset, DistributedSampler]:
    """
    Build dataset for SAE indexing (compatible with sae_index_main).
    
    This is the entry point used by the indexing pipeline.
    Returns (dataset, sampler) tuple.
    """
    # Force eval mode for indexing
    cfg = dict(dataset_cfg)
    cfg["is_train"] = False
    
    result = build_sav_dataset(
        cfg,
        rank=rank,
        world_size=world_size,
    )
    
    return result["dataset"], result["sampler"]


def build_collate_fn(dataset: Any) -> Callable:
    """
    Build collate function for the dataset.
    
    Required by indexing pipeline registry.
    """
    return mask2former_collate_fn


__all__ = [
    "SAVImageDataset",
    "mask2former_collate_fn",
    "build_sav_dataset",
    "build_indexing_dataset",
    "build_collate_fn",
    "DEFAULT_PIXEL_MEAN",
    "DEFAULT_PIXEL_STD",
]
