# src/packs/sam2/offline/bvd_builders.py
from __future__ import annotations
from typing import Dict, List, Tuple

import hashlib
import torch

from training.utils.data_utils import Frame, Object, VideoDatapoint, BatchedVideoDatapoint
from src.packs.sam2.dataset.sa_v.repro_vosdataset import OfflinePromptMeta
from training.dataset import transforms as T
from PIL import Image
from pathlib import Path
# ---------- image I/O ----------
def load_frames_from_disk(image_root: Path, video_id: str, seq_full: List[int]) -> List[Image.Image]:
    outs = []
    for t in seq_full:
        p = None
        for pat in (f"{t:05d}.jpg", f"{t:05d}.png", f"{t}.jpg", f"{t}.png"):
            cand = image_root / video_id / pat
            if cand.exists():
                p = cand
                break
        if p is None:
            raise FileNotFoundError(f"frame not found: {video_id}/{t} under {image_root}")
        outs.append(Image.open(p).convert("RGB"))
    return outs

def build_vos_datapoint(frames_pil: List[Image.Image], seq_full: List[int], video_id: str) -> VideoDatapoint:
    frames = []
    for img, t_abs in zip(frames_pil, seq_full):
        obj = Object(object_id=0, frame_index=int(t_abs), segment=None)
        frames.append(Frame(data=img, objects=[obj]))
    w, h = frames_pil[0].size
    return VideoDatapoint(frames=frames, video_id=video_id, size=(h, w))

def apply_indexing_transforms(sample: VideoDatapoint, target_res: int) -> VideoDatapoint:
    tfms = [
        T.RandomResizeAPI(sizes=target_res, square=True, consistent_transform=True),
        T.ToTensorAPI(),
        T.NormalizeAPI(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ]
    for tfm in tfms:
        sample = tfm(sample, epoch=0)
    return sample

def frames_chw_from_datapoint(sample_after_tfms: VideoDatapoint) -> List[torch.Tensor]:
    out = []
    for fr in sample_after_tfms.frames:
        ten = fr.data
        assert torch.is_tensor(ten) and ten.ndim == 3, "Expect CHW tensor per frame after transforms"
        out.append(ten.clone().detach())
    return out

def build_offline_prompt_meta_from_prompts_rows(rows) -> OfflinePromptMeta:
    """
    prompts parquet/table 의 행(iterable)을 받아 OfflinePromptMeta 구성.
    rows[i]는 다음 필드를 포함해야 합니다:
      - uid (int)
      - points_x (List[float])
      - points_y (List[float])
      - point_labels (List[int])
    """
    by_uid: Dict[str, Dict] = {}
    for r in rows:
        uid = int(r["uid"])
        xs = r.get("points_x") or []
        ys = r.get("points_y") or []
        lbs = r.get("point_labels") or []
        coords = [[float(x), float(y)] for x, y in zip(xs, ys)]
        by_uid[str(uid)] = {"point_coords": coords, "point_labels": [int(v) for v in lbs]}
    # t=0 기본 주입(외부에서 조정 가능)
    return OfflinePromptMeta(
        version=1,
        use_pt_input=True,
        pt_sampling_for_eval="center",
        init_cond_frames=[0],
        frames_to_add_correction_pt=[],
        per_frame={0: {"by_uid": by_uid}},
    )


def make_single_bvd_with_prompt(
    frames_chw: List[torch.Tensor],
    name: str,
    seq_full: List[int],
    sample_id: int,
    prompt_id: int,
    # PyArrow Table 혹은 dict rows(iterable) 모두 허용
    prompt_rows,
    t_prompt: int,
    dict_key: str,
) -> Tuple[BatchedVideoDatapoint, Dict[int, int]]:
    """
    오프라인 prompt가 있는 케이스.
    - frames_chw: [T] of CHW float tensor (정규화 완료)
    - prompt_rows: Arrow Table 또는 [{'uid', 'points_x', 'points_y', 'point_labels'}...]
    반환:
      - BatchedVideoDatapoint
      - uid -> lane index 매핑
    """
    T = len(frames_chw)
    H, W = int(frames_chw[0].shape[-2]), int(frames_chw[0].shape[-1])

    # (T, 1, 3, H, W)
    img_stack = torch.stack(frames_chw, dim=0)
    img_batch = img_stack.unsqueeze(1).contiguous()

    # Arrow Table/rows 호환적으로 uid리스트 확보
    if hasattr(prompt_rows, "num_rows"):
        # PyArrow Table
        uids = [int(prompt_rows.column("uid")[i].as_py()) for i in range(prompt_rows.num_rows)]
        def row_at(i):
            return {
                "uid": int(prompt_rows.column("uid")[i].as_py()),
                "points_x": prompt_rows.column("points_x")[i].as_py() or [],
                "points_y": prompt_rows.column("points_y")[i].as_py() or [],
                "point_labels": prompt_rows.column("point_labels")[i].as_py() or [],
            }
        rows_iter = [row_at(i) for i in range(prompt_rows.num_rows)]
    else:
        # iterable of dict
        uids = [int(r["uid"]) for r in prompt_rows]
        rows_iter = list(prompt_rows)

    if not uids:
        raise RuntimeError("No prompt rows (uid) to reconstruct BVD.")

    N = len(uids)

    # lane 매핑: parquet에 기록된 uid 순서를 그대로 lane 순서로
    uid_to_lane = {int(u): i for i, u in enumerate(uids)}

    # obj_to_frame_idx: (T, N, 2)  (t, b)
    obj_to_frame_idx = torch.zeros((T, N, 2), dtype=torch.long)
    for t in range(T):
        obj_to_frame_idx[t, :, 0] = t
        obj_to_frame_idx[t, :, 1] = 0  # 배치 내 index=0

    # 더미 mask (모델 forward에서 사용 안 함)
    masks = torch.zeros((T, N, H, W), dtype=torch.bool)

    # ---- metadata 구성 ----
    class SimpleMeta: pass
    md = SimpleMeta()

    # unique_objects_identifier: (T, N, 3) — [bidx, objid, abs_frame]
    md.unique_objects_identifier = torch.zeros((T, N, 3), dtype=torch.long)
    for t in range(T):
        for n, _uid in enumerate(uids):
            md.unique_objects_identifier[t, n, 0] = 0
            md.unique_objects_identifier[t, n, 1] = n
            md.unique_objects_identifier[t, n, 2] = int(seq_full[t])

    md.frame_orig_size = torch.tensor([[[H, W]]], dtype=torch.long).expand(T, N, 2).contiguous()

    md.mask_uid_by_t = torch.full((T, N), -1, dtype=torch.long)
    md.prompt_uid_by_t = torch.full((T, N), -1, dtype=torch.long)
    for n, uid in enumerate(uids):
        md.mask_uid_by_t[t_prompt, n] = uid
        md.prompt_uid_by_t[t_prompt, n] = uid
    for t in range(T):
        if t != t_prompt:
            md.mask_uid_by_t[t] = md.mask_uid_by_t[t_prompt]
            md.prompt_uid_by_t[t] = md.prompt_uid_by_t[t_prompt]

    md.names_by_b = [name]
    md.seq_full_by_b = [list(map(int, seq_full))]
    md.sample_id_by_b = torch.tensor([int(sample_id)], dtype=torch.long)
    md.prompt_id_by_bt = torch.full((1, T), int(prompt_id), dtype=torch.long)
    md.epoch_idx = 0
    md.run_seed = 0
    md.image_size = (H, W)
    md.object_key_by_t = torch.tensor(uids, dtype=torch.long).unsqueeze(0).repeat(T, 1).contiguous()

    # OfflinePromptMeta (t_prompt 기준)
    offline = build_offline_prompt_meta_from_prompts_rows(rows_iter)
    offline.init_cond_frames = [int(t_prompt)]
    offline.per_frame = {int(t_prompt): offline.per_frame.get(int(t_prompt), offline.per_frame.get(0, {}))}
    md.offline_prompt = offline

    # 배치 시그니처(간단 해시)
    md.batch_sig = hashlib.sha1(md.mask_uid_by_t.numpy().tobytes()).hexdigest()[:16]

    # BVD
    bvd = BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=md,
        dict_key=dict_key,
        batch_size=[T],
    )

    return bvd, uid_to_lane


def make_single_bvd_no_prompt(
    frames_chw: List[torch.Tensor],
    name: str,
    seq_full: List[int],
    sample_id: int,
    dict_key: str,
    prompt_id_value: int = -1,
) -> Tuple[BatchedVideoDatapoint, Dict[int, int]]:
    """
    오프라인 prompt가 없는 케이스(encoder 레이어 등).
    - 더미 lane 1개(N=1)를 구성하고, adapter가 프롬프트 주입을 건너뛰도록 설정.
    """
    T = len(frames_chw)
    H, W = int(frames_chw[0].shape[-2]), int(frames_chw[0].shape[-1])

    img_stack = torch.stack(frames_chw, dim=0)
    img_batch = img_stack.unsqueeze(1).contiguous()  # (T,1,3,H,W)

    N = 1
    obj_to_frame_idx = torch.zeros((T, N, 2), dtype=torch.long)
    for t in range(T):
        obj_to_frame_idx[t, 0, 0] = t
        obj_to_frame_idx[t, 0, 1] = 0

    masks = torch.zeros((T, N, H, W), dtype=torch.bool)

    class SimpleMeta: pass
    md = SimpleMeta()

    md.unique_objects_identifier = torch.zeros((T, N, 3), dtype=torch.long)
    for t in range(T):
        md.unique_objects_identifier[t, 0, 2] = int(seq_full[t])

    md.frame_orig_size = torch.tensor([[[H, W]]], dtype=torch.long).expand(T, N, 2).contiguous()
    md.mask_uid_by_t = torch.full((T, N), -1, dtype=torch.long)
    md.prompt_uid_by_t = torch.full((T, N), -1, dtype=torch.long)

    md.names_by_b = [name]
    md.seq_full_by_b = [list(map(int, seq_full))]
    md.sample_id_by_b = torch.tensor([int(sample_id)], dtype=torch.long)
    md.prompt_id_by_bt = torch.full((1, T), int(prompt_id_value), dtype=torch.long)  # -1
    md.epoch_idx = 0
    md.run_seed = 0
    md.image_size = (H, W)
    md.object_key_by_t = torch.full((T, N), -1, dtype=torch.long)

    md.offline_prompt = None  # 어댑터가 프롬프트 주입을 건너뜀
    md.batch_sig = hashlib.sha1(md.mask_uid_by_t.numpy().tobytes()).hexdigest()[:16]

    bvd = BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=md,
        dict_key=dict_key,
        batch_size=[T],
    )
    return bvd, {}  # uid 매핑 없음
