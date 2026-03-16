from typing import List, Any, Dict, Optional, Tuple, Callable, Set
import torch
import itertools
from src.utils.utils import stable_u64
from dataclasses import dataclass
from training.dataset.vos_dataset import VOSDataset as _BaseVOSDataset  # 예시
import hashlib
from sam2.modeling.sam2_utils import get_next_point
from training.utils.data_utils import Frame, Object, VideoDatapoint, BatchedVideoDatapoint
import json


def _sha1_uids(mask_uid_by_t: torch.Tensor) -> str:
    b = mask_uid_by_t.contiguous().cpu().numpy().tobytes()
    return hashlib.sha1(b).hexdigest()[:16]


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


@dataclass
class OfflinePromptMeta:
    version: int
    use_pt_input: bool
    pt_sampling_for_eval: str
    init_cond_frames: List[int]
    frames_to_add_correction_pt: List[int]   # 우리는 항상 [] 권장
    # per_frame[t]["by_uid"][str(uid)] = {"point_coords": [[x,y], ...], "point_labels": [1, ...]}
    per_frame: Dict[int, Dict[str, Any]]              # {t: {"by_uid": {...}}}


@dataclass
class BatchedVideoMetaData:
    # 기본 식별/치수 메타
    unique_objects_identifier: torch.Tensor          # (T,N,3) [orig_video_id, orig_obj_id, orig_frame_idx]
    frame_orig_size: torch.Tensor                    # (T,N,2) [H,W]
    image_size: Tuple[int, int]                      # (H,W)

    # 오프라인 프롬프트 메타/UID
    mask_uid_by_t: Optional[torch.Tensor] = None     # (T,N)
    offline_prompt: Optional[OfflinePromptMeta] = None

    # 재현/조인용 확장 메타
    names_by_b: Optional[List[str]] = None           # 길이 B
    seq_full_by_b: Optional[List[List[int]]] = None  # 길이 B의 리스트
    sample_id_by_b: Optional[torch.Tensor] = None    # (B) int64
    prompt_id_by_bt: Optional[torch.Tensor] = None   # (B,T) int64
    object_key_by_t: Optional[torch.Tensor] = None   # (T,N) int64 (프레임 불변 오브젝트 키)
    prompt_uid_by_t: Optional[torch.Tensor] = None   # (T,N) int64 (t0 UID를 전 프레임에 매핑)

    # 실험 축
    epoch_idx: int = 0
    run_seed: int = 0
    batch_sig: Optional[str] = None                  # 예: sha1(mask_uid_by_t)


def _sample_uniform_global_points(
    mask_slice: torch.Tensor,
    *,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """샘플 전체 이미지에서 균등하게 좌표 하나를 뽑는다."""
    n, h, w = mask_slice.shape
    if n == 0 or h == 0 or w == 0:
        return (
            torch.zeros((n, 1, 2), dtype=torch.float32),
            torch.ones((n, 1), dtype=torch.int64),
        )
    generator = rng if rng is not None else torch.Generator()
    xs = torch.randint(0, w, (n, 1), generator=generator)
    ys = torch.randint(0, h, (n, 1), generator=generator)
    coords = torch.stack([xs, ys], dim=2).to(torch.float32)
    labels = torch.ones((n, 1), dtype=torch.int64)
    return coords, labels


def _build_offline_prompt_meta_uidkey(
    masks: torch.Tensor,
    mask_uid_by_t: torch.Tensor,
    init_cond_frames=(0,),
    method: str = "center",
    method_per_uid: Optional[Dict[int, str]] = None,
    rng: Optional[torch.Generator] = None,
) -> OfflinePromptMeta:
    T, N, H, W = masks.shape
    frames: List[int] = []
    for t in init_cond_frames:
        ti = int(t)
        if 0 <= ti < T and ti not in frames:
            frames.append(ti)
    if not frames:
        frames = [0]
    per_uid = {int(k): str(v) for k, v in (method_per_uid or {}).items()}
    per_frame: Dict[int, Dict[str, Any]] = {}
    methods_used: Set[str] = set()
    default_method = str(method)
    for t in frames:
        if masks[t].shape[0] == 0:
            per_frame[int(t)] = {"by_uid": {}}
            continue
        gt = masks[t].unsqueeze(1)
        uids = mask_uid_by_t[t].cpu().tolist()
        frame_methods = {per_uid.get(int(uid), default_method) for uid in uids}
        frame_methods = {str(m) for m in frame_methods}
        frame_methods.add(default_method)
        method_outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for m_name in frame_methods:
            if m_name == "uniform":
                pts, lbl = _sample_uniform_global_points(
                    masks[t], rng=rng
                )
                pts = pts.cpu().to(torch.float32)
                lbl = lbl.cpu().to(torch.int64)
            else:
                pts, lbl = get_next_point(
                    gt_masks=gt,
                    pred_masks=None,
                    method=m_name,
                )
                pts = pts.cpu().to(torch.float32)
                lbl = lbl.cpu().to(torch.int64)
            method_outputs[m_name] = (pts, lbl)
        by_uid: Dict[str, Dict[str, Any]] = {}
        for n, uid in enumerate(uids):
            chosen = per_uid.get(int(uid), default_method)
            coords_tensor, labels_tensor = method_outputs[chosen]
            by_uid[str(int(uid))] = {
                "point_coords": coords_tensor[n].tolist(),
                "point_labels": labels_tensor[n].tolist(),
            }
        methods_used.update(frame_methods)
        per_frame[int(t)] = {"by_uid": by_uid}
    if not methods_used:
        methods_used.add(default_method)
    sampling_tag = (
        default_method if len(methods_used) == 1 else "mixed:" + ",".join(sorted(methods_used))
    )
    return OfflinePromptMeta(
        version=1,
        use_pt_input=True,
        pt_sampling_for_eval=sampling_tag,
        init_cond_frames=frames,
        frames_to_add_correction_pt=[],  # correction OFF
        per_frame=per_frame,
    )

def _prompt_id_by_bt_from_uidkey(
    offline: OfflinePromptMeta,
    mask_uid_by_t: torch.Tensor,          # [T,N]
    obj_to_frame_idx: torch.Tensor,       # [T,N,2] = (t,b)
    B: int, T: int,
) -> torch.Tensor:
    """
    각 (b,t)에서 t가 init_cond_frames에 포함될 때, 해당 프롬프트 세트의 UID→좌표/라벨을 JSON 직렬화 후 stable_u64로 ID 생성.
    init_cond_frames 외 t는 0을 채움(주입 프레임만 세트 존재).
    """
    pid = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        for t in offline.init_cond_frames:  # 보통 [0]
            n_mask = (obj_to_frame_idx[t, :, 1] == b)
            if not torch.any(n_mask):
                pid[b, t] = 0
                continue
            uids = mask_uid_by_t[t, n_mask].cpu().tolist()
            uids_sorted = sorted(int(u) for u in uids)
            by_uid = offline.per_frame.get(int(t), {}).get("by_uid", {})
            rows = []
            for uid in uids_sorted:
                rec = by_uid.get(str(uid))
                if rec is None:
                    rows.append([uid, [], []])
                else:
                    rows.append([uid, rec["point_coords"], rec["point_labels"]])
            s = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
            pid[b, t] = stable_u64(s)
    return pid


def _sample_id_by_b_from_names_seq(names_by_b: List[str], seq_full_by_b: List[List[int]]) -> torch.Tensor:
    vals = []
    for name, seq in zip(names_by_b, seq_full_by_b):
        key = f"{name}||seq={','.join(map(str, seq))}"
        vals.append(stable_u64(key))
    return torch.tensor(vals, dtype=torch.long)


def _extract_seq_frame_idx_from_vdp(dp) -> List[int]:
    seq: List[int] = []
    for f in getattr(dp, 'frames', []):
        objs = getattr(f, 'objects', None)
        if objs and len(objs) > 0 and getattr(objs[0], 'frame_index', None) is not None:
            seq.append(int(objs[0].frame_index))
        else:
            seq.append(-1)
    return seq if seq else [-1]


def _u64_from_bytes(b: bytes) -> int:
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)  # 63-bit 양수


class ReproVOSDataset(_BaseVOSDataset):
    """
    부모 VOSDataset을 그대로 쓰되, 어댑터가 쓸 수 있도록 video_names 같은 힌트만 노출.
    __getitem__은 그대로 상속(=원 동작 유지).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = getattr(self, 'video_names', None)
        if names is None and hasattr(self, 'video_dataset'):
            names = getattr(self.video_dataset, 'video_names', None)
        self.video_names = names  # 어댑터에서 참조 가능


def _mask_uid_bool(mask: torch.Tensor, *, add_shape=True) -> int:
    """
    콘텐츠 기반 UID. CPU bool tensor를 가정 (collate 이전에 CPU일 가능성이 큼).
    shape까지 포함해 안전성을 높일 수 있음(add_shape=True).
    """
    m = mask
    if m.dtype != torch.bool:
        m = m > 0
    m = m.contiguous()
    payload = (str(tuple(m.shape)).encode("utf-8") + m.numpy().tobytes()) if add_shape else m.numpy().tobytes()
    return _u64_from_bytes(payload)


def make_repro_collate(
    ds,
    *,
    run_seed: int = 0,
    epoch_provider: Optional[Callable[[], int]] = None,
    prompt_policy: Optional[Dict[str, Any]] = None,
):
    """
    ds: ReproVOSDataset (video_names 접근 가능)
    반환: collate_fn(batch) -> BatchedVideoDatapoint
    """
    # 비디오 이름 리졸버 (video_id -> name). 안전한 폴백 포함.
    video_names = getattr(ds, "video_names", None)
    prompt_cfg = (prompt_policy or {}).copy()
    dataset_ratio = float(prompt_cfg.get("dataset_prompt_ratio", 0.05))
    dataset_ratio = max(0.0, min(1.0, dataset_ratio))
    dataset_method = str(prompt_cfg.get("dataset_method", "center"))
    random_method = str(prompt_cfg.get("random_method", "uniform"))
    init_frame_candidates = [int(t) for t in prompt_cfg.get("init_cond_frames", [0])]
    seed_val = int(prompt_cfg.get("seed", run_seed))
    prompt_rng = torch.Generator()
    prompt_rng.manual_seed(seed_val)
    meta_run_seed = int(run_seed if run_seed else seed_val)
    
    def _name_from_video(video, default):
        vid = getattr(video, "video_id", None)
        if video_names is not None:
            if isinstance(video_names, (list, tuple)):
                if isinstance(vid, int) and 0 <= vid < len(video_names):
                    return str(video_names[vid])
            if isinstance(video_names, dict) and vid in video_names:
                return str(video_names[vid])
        return str(getattr(video, "video_path", getattr(video, "name", getattr(video, "video_id", default))))

    def collate_fn(batch: List[Any]) -> BatchedVideoDatapoint:
        # === 기존 collate 로직 (원 코드 유지) ===
        img_batch = []
        for video in batch:
            img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]  # (T,3,H,W)
        img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))  # (T,B,3,H,W)
        T = img_batch.shape[0]
        B = img_batch.shape[1]

        step_t_objects_identifier = [[] for _ in range(T)]
        step_t_frame_orig_size    = [[] for _ in range(T)]
        step_t_masks              = [[] for _ in range(T)]
        step_t_obj_to_frame_idx   = [[] for _ in range(T)]
        step_t_mask_uids          = [[] for _ in range(T)]

        for video_idx, video in enumerate(batch):
            orig_video_id  = video.video_id
            orig_frame_size = video.size  # (H, W)
            for t, frame in enumerate(video.frames):
                for obj in frame.objects:
                    orig_obj_id   = obj.object_id
                    orig_frame_idx = obj.frame_index

                    seg = obj.segment.to(torch.bool).contiguous()
                    if seg.ndim == 3 and seg.shape[0] == 1:
                        seg = seg[0]

                    uid = _mask_uid_bool(seg)

                    step_t_obj_to_frame_idx[t].append(torch.tensor([t, video_idx], dtype=torch.int))
                    step_t_masks[t].append(seg)
                    step_t_objects_identifier[t].append(torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx], dtype=torch.long))
                    step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size, dtype=torch.long))
                    step_t_mask_uids[t].append(torch.tensor(uid, dtype=torch.long))

        obj_to_frame_idx = torch.stack([torch.stack(v, dim=0) for v in step_t_obj_to_frame_idx], dim=0)  # (T,N,2)
        masks            = torch.stack([torch.stack(v, dim=0) for v in step_t_masks], dim=0)             # (T,N,H,W)
        objects_id       = torch.stack([torch.stack(v, dim=0) for v in step_t_objects_identifier], dim=0)# (T,N,3)
        frame_orig_size  = torch.stack([torch.stack(v, dim=0) for v in step_t_frame_orig_size], dim=0)   # (T,N,2)
        mask_uid_by_t    = torch.stack([torch.stack(v, dim=0) for v in step_t_mask_uids], dim=0)         # (T,N)

        dict_key = "train"  # 필요시 외부에서 주입 받도록 변경

        # (1) names_by_b / seq_full_by_b
        names_by_b = [_name_from_video(v, f"vid_{i}") for i, v in enumerate(batch)]
        seq_full_by_b = [_extract_seq_frame_idx_from_vdp(v) for v in batch]

        # (2) offline 프롬프트 생성 (prepare_prompt_inputs 흉내)
        valid_init_frames = sorted({t for t in init_frame_candidates if 0 <= t < T})
        if not valid_init_frames:
            valid_init_frames = [0]
        if B > 0:
            dataset_choice_by_b = torch.rand(B, generator=prompt_rng).lt(dataset_ratio).tolist()
        else:
           dataset_choice_by_b = []
        method_per_uid: Dict[int, str] = {}
        for t in valid_init_frames:
            b_indices = obj_to_frame_idx[t, :, 1].tolist()
            uids_t = mask_uid_by_t[t].tolist()
            for uid, b_idx in zip(uids_t, b_indices):
                if b_idx >= len(dataset_choice_by_b):
                    continue
                chosen = dataset_method if dataset_choice_by_b[b_idx] else random_method
                method_per_uid[int(uid)] = chosen
        offline_prompt = _build_offline_prompt_meta_uidkey(
            masks=masks,
            mask_uid_by_t=mask_uid_by_t,
            init_cond_frames=tuple(valid_init_frames),
            method=dataset_method,
            method_per_uid=method_per_uid,
            rng=prompt_rng,
        )

        # (3) ID들 계산
        prompt_id_by_bt = _prompt_id_by_bt_from_uidkey(
            offline_prompt, mask_uid_by_t, obj_to_frame_idx, B, T
        )  # (B,T) — init_cond_frames만 값 존재
        sample_id_by_b = _sample_id_by_b_from_names_seq(names_by_b, seq_full_by_b)
        batch_sig = _sha1_bytes(mask_uid_by_t.contiguous().cpu().numpy().tobytes())

        T_, N = masks.shape[:2]

        # (a) 프레임 불변 오브젝트 키: (orig_video_id, orig_obj_id)로 생성
        object_key_by_t = torch.empty(T_, N, dtype=torch.long)
        for t in range(T_):
            vids = objects_id[t, :, 0].tolist()
            oids = objects_id[t, :, 1].tolist()
            keys = [stable_u64(f"{int(v)}:{int(o)}") for v, o in zip(vids, oids)]
            object_key_by_t[t] = torch.tensor(keys, dtype=torch.long)

        # (b) 초기 프레임(t0)의 키→UID 맵 만들고, 모든 프레임에 복제
        t0 = int(offline_prompt.init_cond_frames[0]) if offline_prompt.init_cond_frames else 0
        t0 = max(0, min(T_ - 1, t0))
        
        key2uid0: Dict[int, int] = {}
        for n in range(N):
            key = int(object_key_by_t[t0, n])
            uid0 = int(mask_uid_by_t[t0, n])  # t0의 콘텐츠 UID == prompts의 uid
            key2uid0[key] = uid0

        prompt_uid_by_t = torch.full((T_, N), -1, dtype=torch.long)
        for t in range(T_):
            for n in range(N):
                key = int(object_key_by_t[t, n])
                if key in key2uid0:
                    prompt_uid_by_t[t, n] = key2uid0[key]

        # (4) 메타 구성
        H, W = int(img_batch.shape[-2]), int(img_batch.shape[-1])
        meta = BatchedVideoMetaData(
            unique_objects_identifier=objects_id,
            frame_orig_size=frame_orig_size,
            object_key_by_t=object_key_by_t,
            prompt_uid_by_t=prompt_uid_by_t,
            mask_uid_by_t=mask_uid_by_t,
            names_by_b=names_by_b,
            seq_full_by_b=seq_full_by_b,
            sample_id_by_b=sample_id_by_b,
            prompt_id_by_bt=prompt_id_by_bt,
            epoch_idx=int(epoch_provider()) if epoch_provider else 0,
            run_seed=meta_run_seed,
            image_size=(H, W),
            offline_prompt=offline_prompt,
            batch_sig=batch_sig,
        )

        return BatchedVideoDatapoint(
            img_batch=img_batch,
            obj_to_frame_idx=obj_to_frame_idx,
            masks=masks,
            metadata=meta,
            dict_key=dict_key,
            batch_size=[T],
        )

    return collate_fn
