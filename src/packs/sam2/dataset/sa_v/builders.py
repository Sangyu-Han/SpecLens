# src/data/sa_v/builders.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Any
from torch.utils.data.distributed import DistributedSampler as DistSampler

from training.dataset.vos_raw_dataset import JSONRawDataset
from training.dataset import transforms as T

from .repro_vosdataset import ReproVOSDataset
from .safe_samplers import SafeRandomUniformSampler


def build_indexing_dataset(cfg_ds: Dict[str, Any], *, world_size: int, rank: int) -> Tuple[ReproVOSDataset, DistSampler]:
    """
    인덱싱(검증 스타일)용 데이터셋/샘플러 빌더.
    - 증강 없이 고정 리사이즈/정규화만 수행
    - 프레임 샘플링은 SafeRandomUniformSampler 사용
    - DDP 분산 샘플러는 shuffle=False, drop_last=False로 글로벌 스텝 일치 보장

    Args:
        cfg_ds: {
            img_folder, gt_folder, file_list_txt(선택),
            num_frames(=8), max_num_objects(=3),
            resize(=1024), multiplier(=1),
            ann_every(=4), reverse_time_prob(=0.0),
            mean/std(선택)
        }
        world_size: DDP world size
        rank:       내 rank

    Returns:
        (dataset, ddp_sampler)
    """
    # 필수 경로
    img_folder = cfg_ds["img_folder"]
    gt_folder  = cfg_ds["gt_folder"]
    file_list  = cfg_ds.get("file_list_txt", None)

    # 하이퍼
    num_frames       = int(cfg_ds.get("num_frames", 8))
    max_num_objects  = int(cfg_ds.get("max_num_objects", 3))
    resolution       = int(cfg_ds.get("resize", 1024))
    multiplier       = int(cfg_ds.get("multiplier", 1))
    ann_every        = int(cfg_ds.get("ann_every", 4))
    reverse_time_prob= float(cfg_ds.get("reverse_time_prob", 0.0))

    mean = cfg_ds.get("mean", [0.485, 0.456, 0.406])
    std  = cfg_ds.get("std",  [0.229, 0.224, 0.225])

    # 원시 JSON 어노테이션 데이터셋
    base = JSONRawDataset(
        img_folder=img_folder,
        gt_folder=gt_folder,
        file_list_txt=file_list,
        ann_every=ann_every,
    )

    # annotated frame 수가 num_frames 미만인 비디오 제거 (NCCL hang 방지)
    import os as _os, logging as _logging
    _logger = _logging.getLogger(__name__)
    before = len(base.video_names)
    base.video_names = [
        v for v in base.video_names
        if len(_os.listdir(_os.path.join(img_folder, v))) >= num_frames
    ]
    filtered = before - len(base.video_names)
    if filtered:
        _logger.info(f"[build_indexing_dataset] Filtered {filtered} videos with < {num_frames} frames.")

    # 프레임/오브젝트 샘플러 (결정적, reverse off)
    sampler = SafeRandomUniformSampler(
        num_frames=num_frames,
        max_num_objects=max_num_objects,
        reverse_time_prob=reverse_time_prob,
    )

    # 검증/인덱싱용 변환(증강 없음)
    transforms = [
        T.RandomResizeAPI(sizes=resolution, square=True, consistent_transform=True),
        T.ToTensorAPI(),
        T.NormalizeAPI(mean=mean, std=std),
    ]

    # ReproVOSDataset 구성
    ds = ReproVOSDataset(
        transforms=transforms,
        training=False,
        video_dataset=base,
        sampler=sampler,
        multiplier=multiplier,
        always_target=True,              # 마스크 없는 케이스도 슬롯 유지
        target_segments_available=True,  # GT 세그먼트 사용
    )

    # DDP 분산 샘플러 (글로벌 순서 고정)
    ddp_sampler = DistSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,     # 진행바/체크포인트 일관성
        drop_last=False
    )
    return ds, ddp_sampler
