# offline_meta_ledger.py
from __future__ import annotations
import os, time, uuid, json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from src.packs.clip.offline.offline_meta_parquet import OfflineMetaParquetLedger as ClipOfflineMetaParquetLedger

# 프로젝트 유틸 (경로 맞춰 주세요)
from src.utils.utils import stable_u64

# ---- Arrow 스키마
# TODO : sample_id, prompt_id duplicate 방지
SAMPLES_SCHEMA = pa.schema([
    ("sample_id", pa.int64()),
    ("dict_key", pa.string()),
    ("name", pa.string()),
    ("seq_full", pa.list_(pa.int32())),
    ("image_h", pa.int32()),
    ("image_w", pa.int32()),
    ("epoch_idx", pa.int32()),
    ("run_seed", pa.int32()),
    ("batch_sig", pa.string()),
    # 변경: prompt_ids → prompt_sets(list<struct<frame_idx, prompt_id>>)
    ("prompt_sets", pa.list_(pa.struct([
        pa.field("frame_idx", pa.int32()),
        pa.field("prompt_id", pa.int64()),
    ]))),
    ("extra_json", pa.string()),
    # 파티션 컬럼(part)은 write 시 append
])

# prompts: 최소필드만 유지 (배치/실험 축·이미지 크기 등 제거)
PROMPTS_SCHEMA = pa.schema([
    ("sample_id", pa.int64()),
    ("prompt_id", pa.int64()),
    ("frame_idx", pa.int32()),
    ("uid", pa.int64()),  # 또는 object_key를 쓰려면 uid 대신 object_key로 교체
    ("points_x", pa.list_(pa.float32())),
    ("points_y", pa.list_(pa.float32())),
    ("point_labels", pa.list_(pa.int32())),
    ("point_steps", pa.list_(pa.int32())),
    ("version", pa.int32()),
    # dict_key/epoch_idx/part 는 partition_cols 로만 사용 (테이블에는 파티션 컬럼으로 추가)
])
# OfflineMetaParquetLedger 내부에 추가
from collections import defaultdict



def _basename(prefix: str, rank: int | None = None) -> str:
    r = rank if rank is not None else (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)
    pid = os.getpid()
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}-{uid}-r{r}-p{pid}-t{ts}"


# ---- 행 변환 유틸

def _rows_from_bvd(bvd) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    md = bvd.metadata
    assert md is not None, "metadata required"

    T = int(bvd.num_frames)
    B = int(bvd.num_videos)

    dict_key: str = getattr(bvd, "dict_key", "unknown")
    H, W = int(md.image_size[0]), int(md.image_size[1])

    names_by_b = list(md.names_by_b or [""] * B)
    seq_full_by_b = list(md.seq_full_by_b or [[-1] * T for _ in range(B)])

    # (B) / (B,T)
    sample_id_by_b = md.sample_id_by_b.cpu().tolist() if isinstance(md.sample_id_by_b, torch.Tensor) else list(md.sample_id_by_b)
    prompt_id_by_bt_tensor = md.prompt_id_by_bt
    assert prompt_id_by_bt_tensor is not None, "prompt_id_by_bt missing"
    prompt_id_by_bt = prompt_id_by_bt_tensor.cpu().tolist()

    # UID 소스 (폴백 포함)
    prompt_uid_by_t = getattr(md, "prompt_uid_by_t", None)
    if prompt_uid_by_t is None:
        prompt_uid_by_t = md.mask_uid_by_t
    prompt_uid_by_t = prompt_uid_by_t.cpu()  # [T,N]

    obj_to_frame_idx = bvd.obj_to_frame_idx.cpu()   # [T,N,2] = (t, b)

    offline = md.offline_prompt
    assert offline is not None, "offline_prompt missing in metadata"
    init_frames = [int(t) for t in (offline.init_cond_frames or [0])]
    by_frame = offline.per_frame  # {t: {"by_uid": {...}}}

    # --- samples rows (스키마 변경 반영: prompt_sets 생성) ---
    sample_rows: List[Dict[str, Any]] = []
    for b in range(B):
        sid = int(sample_id_by_b[b])
        sets: List[Dict[str, Any]] = []
        for t in init_frames:
            # 과거 호환: prompt_id_by_bt[b]가 스칼라일 수 있음
            pid_t = int(prompt_id_by_bt[b][t]) if isinstance(prompt_id_by_bt[b], (list, tuple)) else int(prompt_id_by_bt[b])
            sets.append({"frame_idx": int(t), "prompt_id": pid_t})

        row = {
            "sample_id": sid,
            "dict_key": dict_key,
            "name": str(names_by_b[b]),
            "seq_full": [int(x) for x in (seq_full_by_b[b] or [])],
            "image_h": int(H),
            "image_w": int(W),
            "epoch_idx": int(md.epoch_idx),
            "run_seed": int(md.run_seed),
            "batch_sig": str(md.batch_sig or ""),
            "prompt_sets": sets,
            "extra_json": "",
        }
        sample_rows.append(row)

    # --- prompts rows (최소필드) ---
    prompt_rows: List[Dict[str, Any]] = []
    for t in init_frames:
        by_uid = (by_frame.get(int(t), {}) or {}).get("by_uid", {})

        # 프롬프트 UID는 t 프레임의 lane 순서를 사용
        uids_t = prompt_uid_by_t[t].tolist()          # 길이 N (t 프레임의 prompt UID들)
        b_idx_t = obj_to_frame_idx[t, :, 1].tolist()  # lane → 배치 b

        for n, uid_val in enumerate(uids_t):
            uid_str = str(int(uid_val))
            rec = by_uid.get(uid_str, None)
            if rec is None:
                # 초기 프레임의 프롬프트 메타에 없는 UID는 스킵
                continue

            b = int(b_idx_t[n])
            sid = int(sample_id_by_b[b])
            # (B,T) 텐서에서 t 인덱싱
            pid = int(prompt_id_by_bt[b][t]) if isinstance(prompt_id_by_bt[b], (list, tuple)) else int(prompt_id_by_bt[b])

            coords = rec["point_coords"]
            labels = rec["point_labels"]
            if isinstance(coords, list) and coords and isinstance(coords[0], (int, float)):
                coords = [coords]
            if isinstance(labels, int):
                labels = [labels]

            xs = [float(c[0]) for c in coords]
            ys = [float(c[1]) for c in coords]
            steps = [0 for _ in xs]  # correction OFF 가정

            prompt_rows.append({
                "sample_id": sid,
                "prompt_id": pid,
                "frame_idx": int(t),
                "uid": int(uid_val),               # (sample_id, prompt_id, frame_idx, uid) == unique
                "points_x": xs,
                "points_y": ys,
                "point_labels": [int(v) for v in labels],
                "point_steps": steps,
                "version": 1,
                # dict_key/epoch_idx/part는 partition_cols로만 사용(아래 write에서 주입)
            })

    return sample_rows, prompt_rows


def _to_table(rows: List[Dict[str, Any]], schema: pa.Schema) -> pa.Table:
    if not rows:
        arrays = [pa.array([], type=f.type) for f in schema]
        return pa.table(arrays, schema=schema)
    cols = {}
    for f in schema:
        name = f.name
        if pa.types.is_list(f.type):
            cols[name] = pa.array([r.get(name, []) for r in rows], type=f.type)
        else:
            cols[name] = pa.array([r.get(name, None) for r in rows], type=f.type)
    return pa.table(cols, schema=schema)


# ---- 메인 레저

class OfflineMetaParquetLedger:
    """
    ReproVOSDataset 전용 오프라인 메타 레저 (samples/prompts 파케 테이블).
    - 멀티프로세스/DDP 안전: 각 프로세스가 고유 basename으로 파일 append
    - write_dataset + 하이브 파티셔닝(dict_key, epoch_idx, part)
    """
    def __init__(
        self,
        root_dir: str | Path,
        *,
        rows_per_shard: Optional[int] = None,     # 예: 200_000
        row_group_size: Optional[int] = None,     # 예: 100_000
        compression: str = "zstd",
        compression_level: Optional[int] = 5,
        use_bloom_filter: bool = False,           # PyArrow 버전 지원 시
        part_modulus: int = 128,
    ):
        self.root = Path(root_dir)
        self.samples_dir = self.root / "samples"
        self.prompts_dir = self.root / "prompts"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self.rows_per_shard = rows_per_shard
        self.part_modulus = int(part_modulus) 
        
        # 공통 Parquet 옵션
        self._pq_common_kwargs = {
            "compression": compression,
            "compression_level": compression_level,
            "row_group_size": row_group_size,
            "write_statistics": True,
        }

        # 컬럼별 dictionary encoding (문자열 위주)
        self._pq_samples_kwargs = {
            **self._pq_common_kwargs,
            "use_dictionary": ["dict_key", "name", "extra_json"],
        }
        self._pq_prompts_kwargs = {
            **self._pq_common_kwargs,
            "use_dictionary": [],  # 최소 문자열만 partition 컬럼으로 빠짐
        }

        # Bloom filter (지원되는 PyArrow에서만)
        if use_bloom_filter:
            try:
                self._pq_samples_kwargs.update({
                    "bloom_filter": True,
                    "bloom_filter_columns": ["sample_id"],
                })
                self._pq_prompts_kwargs.update({
                    "bloom_filter": True,
                    "bloom_filter_columns": ["sample_id", "prompt_id", "uid"],
                })
            except TypeError:
                pass
    def _part_col(self, sample_id: int) -> int:   # ★ 변경
        return int(sample_id) % self.part_modulus
    
    def _shard_rows(self, rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not self.rows_per_shard or self.rows_per_shard <= 0:
            return [rows]
        out = []
        step = int(self.rows_per_shard)
        for i in range(0, len(rows), step):
            out.append(rows[i:i+step])
        return out
    def _ds_samples(self):
        return ds.dataset(str(self.samples_dir), format="parquet", partitioning="hive")

    def _ds_prompts(self):
        return ds.dataset(str(self.prompts_dir), format="parquet", partitioning="hive")

    def _sample_exists(self, sample_id: int) -> bool:
        p = self._part_col(sample_id)
        f = (ds.field("part") == p) & (ds.field("sample_id") == int(sample_id))
        tbl = self._ds_samples().to_table(filter=f, columns=["sample_id"])
        return tbl.num_rows > 0

    def _existing_prompt_uids(self, sample_id: int, prompt_id: int, frame_idx: int) -> set[int]:
        p = self._part_col(sample_id)
        f = (
            (ds.field("part") == p) &
            (ds.field("sample_id") == int(sample_id)) &
            (ds.field("prompt_id") == int(prompt_id)) &
            (ds.field("frame_idx") == int(frame_idx))
        )
        tbl = self._ds_prompts().to_table(filter=f, columns=["uid"])
        if tbl.num_rows == 0:
            return set()
        # Arrow -> 파이썬 set
        return set(int(v.as_py()) for v in tbl.column("uid"))

    def _claim(self, kind: str, relkey: str) -> bool:
        """
        원자적 '한 번만 생성' 마커 파일.
        성공(True) 시 이번 프로세스가 최초 획득 → 쓰기 진행.
        실패(False) 시 이미 다른 프로세스/이전 실행에서 기록됨 → 스킵.
        """
        path = self.root / ".seen" / kind / relkey
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            return False
    # ---- public API

    def write_from_batch(self, bvd) -> Tuple[int, int]:
        """인입 배치(기존 BatchedVideoDatapoint)를 rows로 변환해 저장."""
        sample_rows, prompt_rows = _rows_from_bvd(bvd)
        n_s = self.write_samples(sample_rows, dict_key=bvd.dict_key, epoch_idx=bvd.metadata.epoch_idx)
        n_p = self.write_prompts(prompt_rows, dict_key=bvd.dict_key, epoch_idx=bvd.metadata.epoch_idx)
        return n_s, n_p

    def write_from_bvd(self, bvd) -> Tuple[int, int]:
        """Legacy alias; prefer write_from_batch."""
        return self.write_from_batch(bvd)

    def write_samples(
        self,
        rows: List[Dict[str, Any]],
        *,
        dict_key: str,
        epoch_idx: int,
        skip_if_exists: bool = True,   # ← 추가: 기본적으로 중복 스킵
        use_claims: bool = True        # ← 추가: 멀티프로세스 안전 모드
    ) -> int:
        if not rows:
            return 0

        if skip_if_exists:
            filtered = []
            for r in rows:
                sid = int(r["sample_id"])
                part = self._part_col(sid)
                if use_claims:
                    # 마커로 선점 → 실패하면 이미 기록됨으로 간주
                    if not self._claim("samples", f"p={part}/sid={sid}"):
                        continue
                else:
                    # 단일 프로세스/테스트 상황이면 존재 조회만
                    if self._sample_exists(sid):
                        continue
                filtered.append(r)
            rows = filtered

        if not rows:
            return 0

        total = 0
        schema = SAMPLES_SCHEMA.append(pa.field("part", pa.int32()))
        for chunk in self._shard_rows(rows):
            for r in chunk:
                r["dict_key"] = dict_key
                r["epoch_idx"] = int(epoch_idx)
                r["part"] = self._part_col(r["sample_id"])
            tbl = _to_table(chunk, schema)
            base = _basename("samples")
            pq.write_to_dataset(
                tbl,
                str(self.samples_dir),
                partition_cols=["dict_key", "epoch_idx", "part"],
                basename_template=f"{base}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                **self._pq_samples_kwargs,
            )
            total += len(chunk)
        return total


    def write_prompts(
        self,
        rows: List[Dict[str, Any]],
        *,
        dict_key: str,
        epoch_idx: int,
        skip_if_exists: bool = True,   # ← 추가
        use_claims: bool = True        # ← 추가
    ) -> int:
        if not rows:
            return 0

        if skip_if_exists:
            # (sid, pid, t) 그룹별로 기존 uid를 한 번만 조회
            groups = defaultdict(list)
            for r in rows:
                key = (int(r["sample_id"]), int(r["prompt_id"]), int(r["frame_idx"]))
                groups[key].append(r)

            filtered: List[Dict[str, Any]] = []
            for (sid, pid, t), group in groups.items():
                part = self._part_col(sid)
                existing_uids: set[int] = set()
                if not use_claims:
                    existing_uids = self._existing_prompt_uids(sid, pid, t)

                for r in group:
                    uid = int(r["uid"])
                    if use_claims:
                        # uid 단위 마커로 중복 방지
                        if not self._claim("prompts", f"p={part}/sid={sid}/pid={pid}/t={t}/uid={uid}"):
                            continue
                    else:
                        if uid in existing_uids:
                            continue
                    filtered.append(r)

            rows = filtered

        if not rows:
            return 0

        total = 0
        schema = (
            PROMPTS_SCHEMA
            .append(pa.field("dict_key", pa.string()))
            .append(pa.field("epoch_idx", pa.int32()))
            .append(pa.field("part", pa.int32()))
        )
        for chunk in self._shard_rows(rows):
            for r in chunk:
                r["dict_key"] = dict_key
                r["epoch_idx"] = int(epoch_idx)
                r["part"] = self._part_col(r["sample_id"])
            tbl = _to_table(chunk, schema)
            base = _basename("prompts")
            pq.write_to_dataset(
                tbl,
                str(self.prompts_dir),
                partition_cols=["dict_key", "epoch_idx", "part"],
                basename_template=f"{base}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                **self._pq_prompts_kwargs,
            )
            total += len(chunk)
        return total


    # ---- dataset / 조회

    def as_dataset(self):
        """pyarrow.dataset으로 두 테이블 핸들 반환."""
        ds_samples = ds.dataset(str(self.samples_dir), format="parquet", partitioning="hive")
        ds_prompts = ds.dataset(str(self.prompts_dir), format="parquet", partitioning="hive")
        return ds_samples, ds_prompts

    # 프루닝 포함 샘플 조회
    def find_sample(self, sample_id: int) -> pa.Table:
        ds_samples, _ = self.as_dataset()
        p = self._part_col(sample_id)
        f = (ds.field("part") == p) & (ds.field("sample_id") == int(sample_id))
        return ds_samples.to_table(filter=f)
    
    def find_prompts_by_set(self, sample_id: int, prompt_id: int) -> pa.Table:
        _, ds_prompts = self.as_dataset()
        p = self._part_col(sample_id)
        f = (
            (ds.field("part") == p) &
            (ds.field("sample_id") == int(sample_id)) &
            (ds.field("prompt_id") == int(prompt_id))
        )
        return ds_prompts.to_table(filter=f)


    def _find_prompts_exact(
        self,
        ds_prompts,
        sample_id: int,
        prompt_id: int,
        frame_idx: int,
        uid: Optional[int] = None,
    ) -> pa.Table:
        p = self._part_col(sample_id)
        filt = (
            (ds.field("part") == p) &
            (ds.field("sample_id") == int(sample_id)) &
            (ds.field("prompt_id") == int(prompt_id)) &
            (ds.field("frame_idx") == int(frame_idx))
        )
        if uid is not None:
            filt = filt & (ds.field("uid") == int(uid))
        return ds_prompts.to_table(filter=filt)

    def _sample_prompt_frame_map(self, sample_id: int) -> Dict[int, int]:
        tbl = self.find_sample(sample_id)
        if tbl.num_rows == 0:
            return {}
        row = tbl.to_pylist()[0]
        prompt_sets = row.get("prompt_sets") or []
        out: Dict[int, int] = {}
        for item in prompt_sets:
            try:
                fidx = int(item["frame_idx"])
                pid = int(item["prompt_id"])
            except Exception:
                continue
            out[fidx] = pid
        return out

    def _resolve_prompt_frame_idx(
        self,
        ds_prompts,
        sample_id: int,
        prompt_id: int,
        frame_idx: int,
    ) -> Optional[int]:
        """
        Resolve which frame holds prompt rows for a given (sample_id, prompt_id).

        Important for recurrent SAM2: provenance frame_idx tracks the forward step
        where activation was observed, while prompt rows usually live on init frames.
        """
        fmap = self._sample_prompt_frame_map(sample_id)
        if fmap:
            if int(frame_idx) in fmap and int(fmap[int(frame_idx)]) == int(prompt_id):
                return int(frame_idx)
            frames_for_pid = sorted(
                int(fidx) for fidx, pid in fmap.items() if int(pid) == int(prompt_id)
            )
            if frames_for_pid:
                return frames_for_pid[0]

        # Fallback: infer from prompts table directly (pick earliest available frame).
        p = self._part_col(sample_id)
        filt = (
            (ds.field("part") == p) &
            (ds.field("sample_id") == int(sample_id)) &
            (ds.field("prompt_id") == int(prompt_id))
        )
        tbl = ds_prompts.to_table(filter=filt, columns=["frame_idx"])
        if tbl.num_rows == 0:
            return None
        try:
            return min(int(v) for v in tbl.column("frame_idx").to_pylist())
        except Exception:
            return None

    # 프루닝 + 키 일치로 prompts 조회 (컨트랙트 준수)
    def find_prompts(self, sample_id: int, prompt_id: int, frame_idx: int) -> pa.Table:
        _, ds_prompts = self.as_dataset()
        exact = self._find_prompts_exact(
            ds_prompts, sample_id, prompt_id, frame_idx, uid=None
        )
        if exact.num_rows > 0:
            return exact
        resolved = self._resolve_prompt_frame_idx(
            ds_prompts, sample_id, prompt_id, frame_idx
        )
        if resolved is None or int(resolved) == int(frame_idx):
            return exact
        return self._find_prompts_exact(
            ds_prompts, sample_id, prompt_id, int(resolved), uid=None
        )

    # (선택) UID로 한 객체만 찾고 싶을 때
    def find_prompt_for_uid(self, sample_id: int, prompt_id: int, frame_idx: int, uid: int) -> pa.Table:
        _, ds_prompts = self.as_dataset()
        exact = self._find_prompts_exact(
            ds_prompts, sample_id, prompt_id, frame_idx, uid=uid
        )
        if exact.num_rows > 0:
            return exact
        resolved = self._resolve_prompt_frame_idx(
            ds_prompts, sample_id, prompt_id, frame_idx
        )
        if resolved is not None and int(resolved) != int(frame_idx):
            remapped = self._find_prompts_exact(
                ds_prompts, sample_id, prompt_id, int(resolved), uid=uid
            )
            if remapped.num_rows > 0:
                return remapped
        # Last fallback: ignore frame_idx if prompt row exists only with UID.
        p = self._part_col(sample_id)
        f = (
            (ds.field("part") == p) &
            (ds.field("sample_id") == int(sample_id)) &
            (ds.field("prompt_id") == int(prompt_id)) &
            (ds.field("uid") == int(uid))
        )
        return ds_prompts.to_table(filter=f)


# ---- 복원 유틸(컨트랙트 예시)

def reconstruct_prompt_set(ledger: OfflineMetaParquetLedger, sample_id: int, frame_idx: int = 0):
    """
    Contract:
      1) sample_row 1건 읽어 prompt_sets에서 (frame_idx→prompt_id) 맵 획득
      2) 해당 (sample_id, prompt_id, frame_idx)로 prompts 전체를 읽어서 객체별 포인트/라벨 복구
    반환: (prompt_id, prompts_table)
    """
    tbl_s = ledger.find_sample(sample_id)
    assert tbl_s.num_rows == 1, "unique sample_id required"
    # [{"frame_idx":..,"prompt_id":..}, ...]
    prompt_sets = tbl_s.column("prompt_sets")[0].as_py()
    mp = {int(e["frame_idx"]): int(e["prompt_id"]) for e in prompt_sets}
    assert frame_idx in mp, f"no prompt set for frame {frame_idx}"
    pid = mp[frame_idx]
    tbl_p = ledger.find_prompts(sample_id, pid, frame_idx)
    return pid, tbl_p


class Sam2ClipParquetLedger(ClipOfflineMetaParquetLedger):
    """
    CLIP-style offline meta ledger (sample_id/path) for SAM2 runs that only need fast path lookup.
    Keeps the prompt-aware OfflineMetaParquetLedger available for richer metadata.
    """
