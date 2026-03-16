from src.core.base.adapters import ModelAdapter
import torch
import torch.nn as nn
try:
    from training.utils.data_utils import collate_fn as sam2_collate_fn
except Exception:
    sam2_collate_fn = None
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from training.utils.data_utils import Frame, Object, VideoDatapoint, BatchedVideoDatapoint
from src.core.sae.activation_stores.universal_activation_store import UniversalActivationStore
class SAM2TrainModelAdapter(ModelAdapter):
    """
    training.model.sam2.SAM2Train 전용 어댑터
    - batch: dict (official collate_fn(dict_key="all") 결과)
    - forward: model(batch) 또는 model(**batch)
    """
    def __init__(self, model: nn.Module, device: Optional[Union[str, torch.device]] = None):
        self.model = model.eval()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        if sam2_collate_fn is not None:
            self.collate_fn = lambda items: sam2_collate_fn(items, dict_key="all")
        else:
            self.collate_fn = None

    def get_hook_points(self) -> List[str]:
        return []

    def _to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(self._to_device(v) for v in obj)
        return obj.to(self.device) if hasattr(obj, "to") else obj

    def preprocess_input(self, raw_batch: Any) -> Any:
        # side-car: 배치 메타 저장
        try:
            names = []
            labels = []
            # VOSDataset(collate_fn(dict_key="all")) 기준 예시 키
            vids = raw_batch.get("video_id", None)
            vpaths = raw_batch.get("video_path", None)
            lbls = raw_batch.get("labels", None) or raw_batch.get("class", None)
            if isinstance(vids, list):
                names = vids
            elif isinstance(vpaths, list):
                names = vpaths
            else:
                # 최후: 인덱스 기반
                bs = len(next(iter(raw_batch.values())))
                names = [f"sample_{i}" for i in range(bs)]
            if isinstance(lbls, list):
                labels = [int(x) for x in lbls]
            else:
                labels = None
            extra = {
                "seq_frame_idx": raw_batch.get("seq_frame_idx", None),
                "video_size": raw_batch.get("video_size", None),
                "ann_start": raw_batch.get("ann_start", None),
                "time_reversed": raw_batch.get("time_reversed", None),
            }
            self.current_meta = BatchMeta(input_names=names, labels=labels, extra=extra)
        except Exception:
            self.current_meta = None
        return self._to_device(raw_batch)

    @torch.no_grad()
    def forward(self, batch: Any) -> None:
        _ = self.model(batch)


# === PARQUET PROMPT LEDGER ===============================================
# requires: pip install pyarrow
from dataclasses import dataclass
@dataclass
class BatchMeta:
    # 숫자 전용
    sample_id_by_b: torch.Tensor   # (B,) int64
    prompt_id_by_bt: torch.Tensor  # (B,T) int64
    object_key_by_t: Optional[torch.Tensor] # (T,N) int64
    mask_uid_by_t: torch.Tensor    # (T,N) int64
    prompt_uid_by_t: torch.Tensor  # (T,N) int64

    # 주입 프레임(여러 개 가능). 없으면 [0]로 둠.
    init_cond_frames: List[int]

    # 오프라인 조인용
    names_by_b: List[str]
    seq_full_by_b: List[List[int]]
    epoch_idx: int
    run_seed: int
    image_size: Tuple[int, int]


import contextvars, inspect, types
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

class SAM2EvalAdapter(ModelAdapter):
    def __init__(self, model, device=None, collate_fn=None):
        self.model = model.eval()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.collate_fn = collate_fn
        self.current_meta: Optional[BatchMeta] = None
        
        
        # [NEW] click sampler cache & patch state
        self._click_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._click_patch_mode: Optional[str] = None  # "record" | "replay" | None
        self._click_patch_active: bool = False
        self._click_patch_targets: list[tuple[object, str, callable]] = []  # [(module, 'get_next_point', orig)]
        self._click_patch_key_extras: dict = {}  # (필요시) 추가 키 요소 저장
        self._prompt_cache: Optional[dict[int, dict[str, torch.Tensor]]] = None
        self._prompt_patch_mode: Optional[str] = None  # "record" | "replay" | None        
        # === NEW: 프레임별 lane 캐시와 uid0 맵, 컨텍스트 변수
        self._stage_objinfo: dict[int, dict[str, torch.Tensor]] = {}  # {t: {"bidx": Long[N_t], "objid": Long[N_t]}}
        self._uid0_by_b_objid: dict[tuple[int,int], int] = {}         # {(b, orig_obj_id): uid_at_t0}
        self._objinfo_var: contextvars.ContextVar = contextvars.ContextVar("sam2_objinfo", default=None)

        # === 컨텍스트 변수: forward 동안 현재 frame_idx를 추적
        self._frame_idx_var: contextvars.ContextVar[int] = contextvars.ContextVar("sam2_frame_idx", default=-1)
        self._orig_track_step = None
        self._patch_track_step()  # ← 여기서 track_step 래핑
        self._patch_prepare_prompt_inputs()  # ← prepare_prompt_inputs 래핑. collate_fn이 생성한 프롬프트를 주입하도록 세팅.
        
    # -----------------------------------------------------------------------------
    # [NEW] Click sampler monkey-patch (get_next_point) with record/replay cache. used in jvp path
    # -----------------------------------------------------------------------------
    def _click_cache_key(self, *, B: int, num_pt: int, method: str) -> tuple:
        t = max(0, int(self.current_frame_idx()))
        return (t, int(B), int(num_pt), str(method or ""))

    def _install_click_patch(self, mode: str):
        """Replace all visible `get_next_point` symbols with a wrapper."""
        import importlib, sys, inspect, types as _types
        if self._click_patch_active:
            return
        self._click_patch_mode = mode
        self._click_patch_targets.clear()

        # 1) 원본 함수 확보(정규 경로)
        try:
            utils_mod = importlib.import_module("sam2.modeling.sam2_utils")
        except Exception:
            # 다른 패키징 변형 케이스 대비
            utils_mod = importlib.import_module("sam2_utils")
        orig_fn = getattr(utils_mod, "get_next_point", None)
        if not callable(orig_fn):
            return

        # 2) 래퍼 정의
        adapter_self = self
        def _wrapped_get_next_point(gt_masks, pred_masks=None, method="uniform", *args, **kwargs):
            # gt_masks: [B,1,H,W] (bool)
            B = int(gt_masks.shape[0]) if torch.is_tensor(gt_masks) else 1
            # num_pt는 하위 함수 기본이 1이므로 kwargs/args에서 유추 시도
            num_pt = int(kwargs.get("num_pt", 1))
            key = adapter_self._click_cache_key(B=B, num_pt=num_pt, method=str(method))
            if adapter_self._click_patch_mode == "replay":
                if key not in adapter_self._click_cache:
                    raise RuntimeError(f"[click-replay] cache miss at key={key} (record 프리패스를 먼저 수행하세요)")
                pts_cpu, lbl_cpu = adapter_self._click_cache[key]
                dev = gt_masks.device if torch.is_tensor(gt_masks) else adapter_self.device
                return pts_cpu.to(dev), lbl_cpu.to(dev)
            # record 모드: 원본을 호출(NumPy/cv2 가능), 결과를 CPU에 캐시
            pts, lbl = orig_fn(gt_masks, pred_masks, method, *args, **kwargs)
            adapter_self._click_cache[key] = (pts.detach().to("cpu"), lbl.detach().to("cpu"))
            return pts, lbl

        # 3) 심볼 교체: a) 정의 모듈, b) sys.modules 전체에서 동일 심볼이 바인딩된 모듈
        def _maybe_patch(mod):
            if hasattr(mod, "get_next_point"):
                fn = getattr(mod, "get_next_point")
                # 동일 함수 객체만 교체 (이미 패치된 경우엔 건너뜀)
                if fn is orig_fn:
                    setattr(mod, "get_next_point", _wrapped_get_next_point)
                    self._click_patch_targets.append((mod, "get_next_point", fn))

        _maybe_patch(utils_mod)
        for _name, _mod in list(sys.modules.items()):
            try:
                _maybe_patch(_mod)
            except Exception:
                continue
        self._click_patch_active = True

    def _remove_click_patch(self):
        if not self._click_patch_active:
            return
        # 역순 복원
        for mod, attr, orig in reversed(self._click_patch_targets):
            try:
                setattr(mod, attr, orig)
            except Exception:
                pass
        self._click_patch_targets.clear()
        self._click_patch_active = False
        self._click_patch_mode = None

    from contextlib import contextmanager
    @contextmanager
    def clicks_cache(self, mode: str):
        """
        mode="record": 원본 get_next_point로 포인트를 계산하고 CPU 캐시.
        mode="replay": 캐시를 되돌려주며 원본 호출을 막음(NumPy/cv2 미사용 → JVP 안전).
        """
        assert mode in ("record", "replay")
        try:
            self._install_click_patch(mode)
            yield
        finally:
            self._remove_click_patch()
            
    @contextmanager
    def prompt_inputs_cache(self, mode: str):
        """
        mode="record": prepare_prompt_inputs가 구성한 point_inputs_per_frame를 CPU에 저장
        mode="replay": 저장된 값을 그대로 주입(내부에서 리스트/np 접근 없음) — JVP-safe
        """
        assert mode in ("record", "replay")
        old = self._prompt_patch_mode
        try:
            self._prompt_patch_mode = mode
            yield
        finally:
            self._prompt_patch_mode = old
    # --- DP/DDP 안전 target
    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def get_provenance_spec(self) -> dict:
        cols = ("sample_id", "frame_idx", "y", "x", "prompt_id", "uid")
        return {"cols": cols, "num_cols": len(cols)}

    # --- 현재 frame_idx 읽기 (스토어/라이터가 요청할 때 사용)
    def current_frame_idx(self) -> int:
        try:
            return int(self._frame_idx_var.get())
        except Exception:
            return -1

    # --- track_step 패치
    def _patch_track_step(self):
        target = self._unwrap_model()
        if getattr(target, "_ua_track_step_patched", False):
            return
        if not hasattr(target, "track_step"):
            # SAM2 구현체마다 이름이 다르면 여기서 확장: 예) "track_step_v2" 등
            return

        bound = target.track_step
        orig_fn = bound.__func__ if hasattr(bound, "__func__") else bound
        sig = None
        try:
            sig = inspect.signature(orig_fn)
        except Exception:
            pass

        frame_idx_var = self._frame_idx_var  # 클로저 캡쳐
        outer_self = self  # 캡쳐
        
        def _extract_fidx(args, kwargs):
            # 1) 키워드 우선
            if "frame_idx" in kwargs: return kwargs["frame_idx"]
            if "stage_id"  in kwargs: return kwargs["stage_id"]
            # 2) 위치 인자 추정 (self 다음이 frame_idx 라면)
            try:
                if sig is not None:
                    pos_params = list(sig.parameters.keys())
                    # 메서드: pos_params[0]은 self 가정
                    if len(pos_params) >= 2 and pos_params[1] in ("frame_idx", "stage_id"):
                        # args[0]는 self, args[1]가 frame_idx/stage_id
                        if len(args) >= 2:
                            return args[1]
                        # 혹시 self 바운드 함수가 아닌 경우를 대비
                        if len(args) >= 1 and pos_params[0] in ("frame_idx","stage_id"):
                            return args[0]
            except Exception:
                pass
            return -1

        def _wrapped_track_step(self_obj, *args, **kwargs):
            fidx = _extract_fidx(args, kwargs)
            try: fidx = int(fidx)
            except Exception: fidx = -1

            tk1 = frame_idx_var.set(fidx)
            # === NEW: 현재 프레임의 lane 캐시를 컨텍스트에 올림
            tk2 = outer_self._objinfo_var.set(outer_self._stage_objinfo.get(int(fidx), None))
            try:
                return orig_fn(self_obj, *args, **kwargs)
            finally:
                outer_self._objinfo_var.reset(tk2)
                frame_idx_var.reset(tk1)
        # 실제로 바인딩
        target.track_step = types.MethodType(_wrapped_track_step, target)
        target._ua_track_step_patched = True
        self._orig_track_step = bound

    # --- 필요 시 원복(옵션)
    def _unpatch_track_step(self):
        target = self._unwrap_model()
        if getattr(target, "_ua_track_step_patched", False) and self._orig_track_step is not None:
            target.track_step = self._orig_track_step
            target._ua_track_step_patched = False
            self._orig_track_step = None

    def _to_device(self, x):
        if torch.is_tensor(x): return x.to(self.device, non_blocking=True)
        if isinstance(x, dict): return {k: self._to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = type(x); return t(self._to_device(v) for v in x)
        return x
    def _patch_prepare_prompt_inputs(self):
        target = self._unwrap_model()
        if getattr(target, "_ua_ppi_patched", False) or not hasattr(target, "prepare_prompt_inputs"):
            return

        bound = target.prepare_prompt_inputs
        orig = bound.__func__ if hasattr(bound, "__func__") else bound

        def _wrapped(self_obj, backbone_out, input, start_frame_idx=0):
            out = orig(self_obj, backbone_out, input, start_frame_idx)

            md = getattr(input, "metadata", None)
            offline = getattr(md, "offline_prompt", None) if md is not None else None
            mask_uid_by_t = getattr(md, "mask_uid_by_t", None) if md is not None else None
            if not offline or mask_uid_by_t is None:
                return out

            # 1) point 입력 강제 + correction OFF
            out["use_pt_input"] = True
            T = out.get("num_frames", getattr(input, "num_frames", None))
            if T is None:
                T = len(out.get("gt_masks_per_frame", {}))
            init = [t for t in offline.init_cond_frames if 0 <= t < T] or [0]
            out["init_cond_frames"] = init
            out["frames_not_in_init_cond"] = [t for t in range(start_frame_idx, T) if t not in init]
            out["frames_to_add_correction_pt"] = []
            out["mask_inputs_per_frame"] = {}
            
            
            # [REPLAY 모드] 이미 계산된 point_inputs_per_frame를 그대로 주입하고,
            # provenance용 _stage_objinfo 구축도 생략
            if self._prompt_patch_mode == "replay" and self._prompt_cache is not None:
                dev = next(self_obj.parameters()).device
                pipf = {}
                for t, rec in self._prompt_cache.items():
                    pipf[int(t)] = {
                        "point_coords": rec["point_coords"].to(dev),
                        "point_labels": rec["point_labels"].to(dev),
                    }
                out["point_inputs_per_frame"] = pipf
                return out
            
            # === NEW: 프레임 전역 lane 캐시와 t0 uid 맵 만들기
            with torch.no_grad():
                obj_to_frame_idx = input.obj_to_frame_idx.detach().cpu().long()          # [T,N,2], (..., b)
                objects_id       = md.unique_objects_identifier.detach().cpu().long()     # [T,N,3], (..., orig_obj_id)
                # JVP-safe: current_meta가 있으면 그 값을 우선 사용 (이미 CPU/detach 상태)
                mask_uid_cpu = None
                if self.current_meta is not None and getattr(self.current_meta, "mask_uid_by_t", None) is not None:
                    mask_uid_cpu = self.current_meta.mask_uid_by_t.to(torch.long)
                else:
                    mask_uid_cpu = mask_uid_by_t.detach().cpu().long()

                # 프레임별 lane 캐시 (bidx, objid)
                self._stage_objinfo.clear()
                for t in range(T):
                    bidx_t  = obj_to_frame_idx[t, :, 1].clone()       # [N_t]
                    objid_t = objects_id[t, :, 1].clone()             # [N_t] orig_obj_id
                    self._stage_objinfo[int(t)] = {"bidx": bidx_t, "objid": objid_t}

                # 초기 프레임 t0 기준 (b,obj_id)->uid0 맵
                t0 = int(init[0])
                self._uid0_by_b_objid.clear()
                bidx0  = obj_to_frame_idx[t0, :, 1].tolist()
                objid0 = objects_id[t0, :, 1].tolist()
                uid0   = mask_uid_cpu[t0].tolist()
                for b, oid, u in zip(bidx0, objid0, uid0):
                    self._uid0_by_b_objid[(int(b), int(oid))] = int(u)
            # 2) UID 순서로 point_inputs_per_frame 재구성
            dev = next(self_obj.parameters()).device
            pipf = {}
            for t in init:
                uids = mask_uid_by_t[t].cpu().tolist()        # 현재 N의 순서 # 여기가 문제인거같은데..? 재조립한 BatchedVideoDatapoint에는 mask_uid_by_t가 없늑거같은데.
                by_uid = offline.per_frame.get(int(t), {}).get("by_uid", {})

                coords_list, labels_list = [], []
                K_max = 0
                for uid in uids:
                    rec = by_uid.get(str(int(uid)), None)
                    if rec is None:
                        # 빠졌다면 더미(길이 0) → 아래에서 패딩
                        coords_list.append([])
                        labels_list.append([])
                        continue
                    coords = rec["point_coords"]  # (K,2) 또는 (1,2)
                    labels = rec["point_labels"]  # (K,)  또는 (1,)
                    # 표준화
                    if isinstance(coords[0], (int, float)):
                        coords = [coords]  # (2,) -> (1,2)
                    if isinstance(labels, int):
                        labels = [labels]
                    coords_list.append(coords)
                    labels_list.append(labels)
                    K_max = max(K_max, len(coords))

                # K 길이 맞추기(부족하면 마지막 값 반복 패딩, 없으면 (0,0)/0)
                def _pad_coords(cs):
                    if len(cs) == 0:
                        return [[0.0, 0.0]] * K_max
                    return cs + [cs[-1]] * (K_max - len(cs))
                def _pad_labels(ls):
                    if len(ls) == 0:
                        return [0] * K_max
                    return ls + [ls[-1]] * (K_max - len(ls))

                coords_tensor = torch.tensor([_pad_coords(c) for c in coords_list], dtype=torch.float32, device=dev)  # [N,K,2]
                labels_tensor = torch.tensor([_pad_labels(l) for l in labels_list], dtype=torch.int64, device=dev)    # [N,K]

                pipf[int(t)] = {"point_coords": coords_tensor, "point_labels": labels_tensor}

            out["point_inputs_per_frame"] = pipf
            
            if self._prompt_patch_mode == "record":
                self._prompt_cache = {
                    int(t): {
                        "point_coords": rec["point_coords"].detach().to("cpu"),
                        "point_labels": rec["point_labels"].detach().to("cpu"),
                    }
                    for t, rec in pipf.items()
                }            
            
            return out

        target.prepare_prompt_inputs = types.MethodType(_wrapped, target)
        target._ua_ppi_patched = True




    def preprocess_input(self, raw_batch: BatchedVideoDatapoint) -> BatchedVideoDatapoint:
        md = raw_batch.metadata
        assert md is not None, "metadata is required"
        init_frames = list(getattr(md.offline_prompt, "init_cond_frames", [0]))
        self.current_meta = BatchMeta(
            sample_id_by_b = md.sample_id_by_b.detach().cpu(),            # (B,)
            prompt_id_by_bt = md.prompt_id_by_bt.detach().cpu(),          # (B,T)
            object_key_by_t = md.object_key_by_t.detach().cpu(),          # (T,N)
            mask_uid_by_t   = md.mask_uid_by_t.detach().cpu(),            # (T,N)
            prompt_uid_by_t = md.prompt_uid_by_t.detach().cpu(),          # (T,N)
            init_cond_frames = init_frames,
            names_by_b = list(md.names_by_b or []),
            seq_full_by_b = [list(seq) for seq in (md.seq_full_by_b or [])],
            epoch_idx = int(md.epoch_idx),
            run_seed = int(md.run_seed),
            image_size = tuple(md.image_size),
        )

        def _to_dev(x):
            return x.to(self.device, non_blocking=True) if torch.is_tensor(x) else x
        batch_on_dev = raw_batch.apply(_to_dev)
        return batch_on_dev

    def forward(self, batch: Dict[str, Any]) -> Any:
        return self.model(batch)

    def _get_sample_ids(self) -> torch.Tensor:
        """SAM2 stores sample IDs as sample_id_by_b in BatchMeta.
        Respects _current_sample_ids if set (used by fused B*T expansion)."""
        sids = getattr(self, '_current_sample_ids', None)
        if sids is not None:
            return sids
        if self.current_meta is not None:
            return self.current_meta.sample_id_by_b.to(torch.long)
        return torch.zeros(1, dtype=torch.long)

    # ---- provenance 생성: fidx 힌트를 우선 사용
    def build_token_provenance(
        self,
        *,
        act_name: str,
        raw_output: Any,
        flattened_tokens: torch.Tensor,
        fidx_hint: Optional[Union[int, torch.Tensor]] = None,
        **_: Any,
    ) -> torch.Tensor:
        """
        반환: int64 [N,6] = (sample_id, frame_idx, y, x, prompt_id, uid)
        - head 경로(= track_step 컨텍스트 존재): lane m → (b, obj_id) → uid0 매핑
        * (N,C,H,W): 각 lane m에 대해 (H*W) 위치로 전개 → (y,x) = 픽셀 좌표
        * (N,L,C):   각 lane m에 대해 L 토큰 → (y=-1, x=토큰인덱스)
        * (N,C):     각 lane m당 하나 → (y=-1, x=-1)
        - fallback 경로: 기존 (b,t,y,x) 추정 + uid = -1
        """
        assert self.current_meta is not None, "meta missing"
        B   = int(self.current_meta.sample_id_by_b.shape[0])
        N   = int(flattened_tokens.shape[0])  # 최종 반환 행 수
        dev = raw_output.device if torch.is_tensor(raw_output) else torch.device("cpu")

        # -----------------------------
        # 공통 헬퍼
        # -----------------------------
        def _resolve_frame_idx(n_rows: int) -> torch.Tensor:
            """fidx_hint → 컨텍스트(frame_idx) → 0"""
            if isinstance(fidx_hint, torch.Tensor):
                t = fidx_hint.to(torch.long).view(-1)
                if t.numel() == 1:
                    return t.expand(n_rows)
                if t.numel() == n_rows:
                    return t
            elif isinstance(fidx_hint, int):
                return torch.full((n_rows,), int(fidx_hint), dtype=torch.long)
            # 컨텍스트
            t_ctx = self.current_frame_idx()
            if t_ctx >= 0:
                return torch.full((n_rows,), t_ctx, dtype=torch.long)
            return torch.zeros((n_rows,), dtype=torch.long)

        def _clip_bt_and_fetch_ids(b: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            b = b.clamp_(min=0, max=B-1).to('cpu')
            t = t.to('cpu')

            sid_b  = self.current_meta.sample_id_by_b          # (B,)
            pid_bt = self.current_meta.prompt_id_by_bt         # (B,T)

            sample_id = sid_b.index_select(0, b)              # (N,)

            # 1) 우선 (b,t)로 뽑음
            prompt_id = pid_bt[b, t]                           # (N,)

            # 2) 주입 프레임 폴백 (0 또는 첫 주입 프레임)
            init_frames = getattr(self.current_meta, "init_cond_frames", [0])
            t0 = int(init_frames[0]) if len(init_frames) > 0 else 0
            t0_vec = torch.full_like(b, t0)
            prompt_id_t0 = pid_bt[b, t0_vec]                   # (N,)
            # 값이 0(즉, 해당 t에 세트 없음)이면 t0의 값으로 대체
            prompt_id = torch.where(prompt_id > 0, prompt_id, prompt_id_t0)

            return b, sample_id, prompt_id


        # -----------------------------
        # head 경로인지 결정: fidx가 유효하면 head로 간주
        # -----------------------------
        is_head_ctx = (self.current_frame_idx() >= 0) or (isinstance(fidx_hint, int) and fidx_hint >= 0) \
                    or (isinstance(fidx_hint, torch.Tensor) and fidx_hint.numel() == 1 and int(fidx_hint.item()) >= 0)

        if is_head_ctx and torch.is_tensor(raw_output):
            # lane 캐시(현재 프레임의 bidx/objid)가 있어야 head 경로 가능
            info = None
            try:
                info = self._objinfo_var.get()
            except Exception:
                info = None

            if info is not None and "bidx" in info and "objid" in info:
                bidx_t: torch.Tensor = info["bidx"]   # [N_lane]
                objid_t: torch.Tensor = info["objid"] # [N_lane]

                # --- 모양별로 (b, t, y, x, uid0) 시퀀스 생성
                if raw_output.ndim == 4:
                    # (N_lane, C, H, W)
                    M, _, H, W = raw_output.shape
                    if M == bidx_t.numel() == objid_t.numel():
                        plane = H * W
                        idx = torch.arange(M * plane, device=dev)
                        lane = idx // plane               # [M*plane]
                        r    = idx % plane
                        y    = r // W
                        x    = r % W

                    else:
                        # 불일치 → fallback
                        info = None

                elif raw_output.ndim == 3:
                    # (N_lane, L, C)
                    M, L, _ = raw_output.shape
                    if M == bidx_t.numel() == objid_t.numel():
                        idx  = torch.arange(M * L, device=dev)
                        lane = idx // L                   # [M*L]
                        tok  = idx %  L
                        y    = torch.full_like(tok, -1)
                        x    = tok
                    else:
                        info = None

                elif raw_output.ndim == 2:
                    # (N_lane, C)  → lane당 1행
                    M, _ = raw_output.shape
                    if M == bidx_t.numel() == objid_t.numel():
                        lane = torch.arange(M, device=dev)
                        y    = torch.full_like(lane, -1)
                        x    = torch.full_like(lane, -1)
                    else:
                        info = None
                else:
                    # 다른 차원 수 → fallback
                    info = None

                if info is not None:
                    # lane별 (b, obj_id)
                    b_lane   = bidx_t.to(dev)[lane]
                    obj_lane = objid_t.to(dev)[lane]

                    # lane별 uid0 룩업
                    # 파이썬 dict 루프 → 텐서
                    uid0_list = [ self._uid0_by_b_objid.get((int(bb), int(oo)), -1)
                                for bb, oo in zip(b_lane.tolist(), obj_lane.tolist()) ]
                    uid0 = torch.tensor(uid0_list, dtype=torch.long, device=dev)

                    # frame_idx(t) 확정
                    t = _resolve_frame_idx(lane.numel()).to('cpu')


                    # (b,t) 보정 → sample_id, prompt_id
                    b_lane, sample_id, prompt_id = _clip_bt_and_fetch_ids(b_lane.contiguous(), t.contiguous())
                    sample_id = sample_id.to('cpu').view(-1)
                    t        = t.to('cpu').view(-1)
                    y        = y.to('cpu').view(-1)
                    x        = x.to('cpu').view(-1)
                    prompt_id= prompt_id.to('cpu').view(-1)
                    uid0     = uid0.to('cpu').view(-1)


                    return torch.stack([sample_id, t, y, x, prompt_id, uid0], dim=1).contiguous()

        # -----------------------------
        # fallback 경로 (백본/비 head 레이어 등)
        # — base class generic provenance + 6-col expansion
        # -----------------------------
        # Image encoder produces fused (B*T, C, H, W) tensors.
        # Expand sample_ids from (B,) to (B*T,) so base class maps correctly.
        sids_b = self.current_meta.sample_id_by_b.to(torch.long)
        saved_sids = getattr(self, '_current_sample_ids', None)
        if torch.is_tensor(raw_output) and raw_output.ndim == 4 and raw_output.shape[0] > B:
            BT = raw_output.shape[0]
            T_est = max(1, BT // B) if B > 0 else 1
            self._current_sample_ids = sids_b.repeat_interleave(T_est)[:BT]
        else:
            self._current_sample_ids = sids_b

        prov_base = super().build_token_provenance(
            act_name=act_name,
            raw_output=raw_output,
            flattened_tokens=flattened_tokens,
            fidx_hint=fidx_hint,
        )
        self._current_sample_ids = saved_sids  # restore

        # prov_base from base class: typically (N, 3) = (sample_id, y, x)
        # but could be (N, 1) for 2D tensors or (N, 6) if fallback used SAM2's provenance_spec.
        # Expand to 6-col: (sample_id, frame_idx, y, x, prompt_id, uid)
        N = prov_base.shape[0]
        C = prov_base.shape[1]

        # Extract columns safely
        col_sid = prov_base[:, 0] if C >= 1 else torch.zeros(N, dtype=torch.long)
        col_y   = prov_base[:, 1] if C >= 2 else torch.zeros(N, dtype=torch.long)
        col_x   = prov_base[:, 2] if C >= 3 else torch.zeros(N, dtype=torch.long)

        # For fused B*T tensors, compute per-token frame_idx from shape
        if torch.is_tensor(raw_output) and raw_output.ndim == 4 and raw_output.shape[0] > B:
            BT = raw_output.shape[0]
            T_est = max(1, BT // B) if B > 0 else 1
            H_est, W_est = raw_output.shape[2], raw_output.shape[3]
            plane = H_est * W_est
            idx = torch.arange(N, dtype=torch.long)
            bt = idx // plane
            frame_idx = (bt % T_est).to('cpu')
        else:
            frame_idx = _resolve_frame_idx(N).to('cpu')

        prompt_id = torch.full((N,), -1, dtype=torch.long)
        uid = torch.full((N,), -1, dtype=torch.long)

        return torch.stack([
            col_sid,           # sample_id
            frame_idx,         # frame_idx
            col_y,             # y
            col_x,             # x
            prompt_id,
            uid,
        ], dim=1).contiguous()



# ============================== #
#          Factory API           #
# ============================== #
def create_sam2eval_store(
    model: torch.nn.Module,
    cfg: Dict,
    dataset=None,
    sampler=None,
    collate_fn: Optional[Callable] = None,
    on_batch_generated: Optional[Callable] = None
) -> UniversalActivationStore:
    adapter = SAM2EvalAdapter(model, device=cfg.get("device"), collate_fn=collate_fn)
    # prompts 옵션이 있으면 Parquet 레저 활성화

    return UniversalActivationStore(model, cfg, adapter, dataset, sampler, on_batch_generated=on_batch_generated)


def create_sam2train_store(
    model: torch.nn.Module,
    cfg: Dict,
    dataset=None,
    sampler=None
) -> UniversalActivationStore:
    adapter = SAM2TrainModelAdapter(model, device=cfg.get("device"))
    return UniversalActivationStore(model, cfg, adapter, dataset, sampler)
