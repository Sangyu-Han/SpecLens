from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def _resolve_relative(path_like: str | Path, *, project_root: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def setup_env(config: Optional[Dict[str, Any]] = None, *, project_root: Path) -> None:
    cfg = config or {}
    sam2_path = cfg.get("sam2_path") or cfg.get("path") or "third_party/sam2_src"
    target = _resolve_relative(sam2_path, project_root=project_root)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


def load_model(
    model_cfg: Dict[str, Any],
    *,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> nn.Module:
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from training.utils import checkpoint_utils

    yaml_field = model_cfg.get("yaml")
    if not yaml_field:
        raise FileNotFoundError("config['model']['yaml'] must point to a SAM2 model yaml file.")
    yaml_path = Path(yaml_field).expanduser()
    if not yaml_path.exists():
        raise FileNotFoundError(f"SAM2 model yaml not found: {yaml_path}")

    hydra_cfg = OmegaConf.load(str(yaml_path))
    if "trainer" in hydra_cfg and "model" in hydra_cfg.trainer:
        hydra_model_cfg = hydra_cfg.trainer.model
    elif "model" in hydra_cfg:
        hydra_model_cfg = hydra_cfg.model
    else:
        raise KeyError("SAM2 yaml is missing 'model' or 'trainer.model' section.")

    model = instantiate(hydra_model_cfg)
    model.to(device).eval()

    ckpt_path = model_cfg.get("ckpt")
    if ckpt_path is None:
        try:
            ckpt_path = hydra_cfg.trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path
        except Exception as exc:
            raise FileNotFoundError(
                "Checkpoint path must be provided via config['model']['ckpt'] or the yaml trainer checkpoint block."
            ) from exc

    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    try:
        weights = checkpoint_utils.load_checkpoint_and_apply_kernels(
            checkpoint_path=str(ckpt_path),
            ckpt_state_dict_keys=["model"],
        )
        state_dict = weights.get("model", weights)
    except TypeError:
        raw = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(raw, dict) and "model" in raw:
            state_dict = raw["model"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
            state_dict = {k.replace("module.", "", 1): v for k, v in sd.items()}
        else:
            state_dict = raw

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        LOGGER.info("[sam2.train] model load missing=%d unexpected=%d", len(missing), len(unexpected))
    return model


def build_dataset(
    dataset_cfg: Dict[str, Any],
    *,
    full_config: Optional[Dict[str, Any]] = None,
    **_,
) -> Dict[str, Any]:
    from training.dataset import transforms as T
    from training.dataset.vos_raw_dataset import JSONRawDataset
    from src.packs.sam2.dataset.sa_v.repro_vosdataset import ReproVOSDataset, make_repro_collate
    from src.packs.sam2.dataset.sa_v.safe_samplers import SafeRandomUniformSampler

    img_folder = dataset_cfg["img_folder"]
    gt_folder = dataset_cfg["gt_folder"]
    file_list = dataset_cfg.get("file_list_txt")

    num_frames = int(dataset_cfg.get("num_frames", 8))
    max_num_objects = int(dataset_cfg.get("max_num_objects", 3))
    resolution = int(dataset_cfg.get("resize", 1024))
    multiplier = int(dataset_cfg.get("multiplier", 1))

    base = JSONRawDataset(
        img_folder=img_folder,
        gt_folder=gt_folder,
        file_list_txt=file_list,
        ann_every=4,
    )
    sampler = SafeRandomUniformSampler(
        num_frames=num_frames,
        max_num_objects=max_num_objects,
        reverse_time_prob=dataset_cfg.get("reverse_time_prob", 0.5),
    )

    transforms = [
        T.RandomHorizontalFlip(consistent_transform=True),
        T.RandomAffine(degrees=25, shear=20, image_interpolation="bilinear", consistent_transform=True),
        T.RandomResizeAPI(sizes=resolution, square=True, consistent_transform=True),
        T.ColorJitter(consistent_transform=True, brightness=0.1, contrast=0.03, saturation=0.03, hue=None),
        T.RandomGrayscale(p=0.05, consistent_transform=True),
        T.ColorJitter(consistent_transform=False, brightness=0.1, contrast=0.05, saturation=0.05, hue=None),
        T.ToTensorAPI(),
        T.NormalizeAPI(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    dataset = ReproVOSDataset(
        transforms=transforms,
        training=True,
        video_dataset=base,
        sampler=sampler,
        multiplier=multiplier,
        always_target=True,
        target_segments_available=True,
    )

    prompt_policy = (dataset_cfg.get("prompt_policy") or {}).copy()
    training_cfg = (full_config or {}).get("sae", {}).get("training", {}) or {}
    default_seed = int(training_cfg.get("seed", 0))
    prompt_seed = int(prompt_policy.get("seed", default_seed))
    collate_fn = make_repro_collate(
        dataset,
        run_seed=prompt_seed,
        prompt_policy=prompt_policy,
    )

    return {
        "dataset": dataset,
        "collate_fn": collate_fn,
    }


def create_store(
    *,
    model: nn.Module,
    cfg: Dict[str, Any],
    dataset,
    sampler,
    collate_fn,
    **_,
) -> "UniversalActivationStore":
    from src.packs.sam2.models.adapters import create_sam2eval_store

    if collate_fn is None:
        raise ValueError("collate_fn must be provided to create the SAM2 activation store.")
    return create_sam2eval_store(
        model=model,
        cfg=cfg,
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn,
    )


__all__ = ["setup_env", "load_model", "build_dataset", "create_store"]