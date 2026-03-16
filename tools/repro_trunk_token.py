#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_encoder.trunk@3 전용 SAE 검증 스크립트 (VOS 파이프라인 + SAM2EvalAdapter 재현)

- CSV의 기록과 동일한 전처리/리사이즈/인덱싱으로 feature vector를 집어 뽑아
  SAE 재인코딩 값이 CSV의 value와 atol 내에서 일치하는지 strict 검증
- 디코더(head)보다 단순한 인코더 스테이지로 먼저 sanity-check

주의:
- CSV에서 layer == image_encoder.trunk@3 인 행을 선택해야 함.
- prompts_timeline이 비어 있으면 parquet는 필요 없음.
"""

import os, json, csv, logging, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
from training.utils import checkpoint_utils

# VOS 파이프라인 자료형/변환
from training.utils.data_utils import Frame, Object, VideoDatapoint
from training.dataset import transforms as T

# optional parquet (프롬프트 타임라인 있을 때만)
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

# Eval 어댑터
from src.sae.activation_stores.universal_activation_store import SAM2EvalAdapter
from src.sae.registry import create_sae

logger = logging.getLogger("verify_trunk3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPECTED_LAYER = "image_encoder.trunk@3"

# ---------------------------
# CSV 유틸
# ---------------------------
def parse_csv_meta_header(csv_path: Path) -> Dict[str, Any]:
    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("# META"):
        return json.loads(first.split("\t", 1)[1])
    raise RuntimeError("CSV 첫 줄에 META 헤더가 없습니다.")

def read_csv_row(csv_path: Path, row_index: int) -> Dict[str, Any]:
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(data_lines)
    for i, row in enumerate(reader):
        if i == row_index:
            return row
    raise IndexError(f"row_index {row_index} out of range")

# ---------------------------
# 모델/SAE 로더
# ---------------------------
def load_sam2(model_yaml: str, model_ckpt: Optional[str], device: torch.device):
    hydra_cfg = OmegaConf.load(model_yaml)
    model_cfg = hydra_cfg.trainer.model if "trainer" in hydra_cfg and "model" in hydra_cfg.trainer else hydra_cfg.model
    model = instantiate(model_cfg).to(device).eval()
    if model_ckpt:
        try:
            weights = checkpoint_utils.load_checkpoint_and_apply_kernels(
                checkpoint_path=str(model_ckpt),
                ckpt_state_dict_keys=["model"]
            )
            state_dict = weights.get("model", weights)
        except TypeError:
            raw = torch.load(model_ckpt, map_location="cpu")
            if isinstance(raw, dict) and "model" in raw:
                state_dict = raw["model"]
            elif isinstance(raw, dict) and "state_dict" in raw:
                sd = raw["state_dict"]; state_dict = {k.replace("module.","",1): v for k,v in sd.items()}
            else:
                state_dict = raw
        model.load_state_dict(state_dict, strict=False)
    return model

def load_sae(sae_ckpt_path: str, device: torch.device):
    pkg = torch.load(sae_ckpt_path, map_location="cpu")
    act_size = int(pkg.get("act_size") or pkg.get("sae_config", {}).get("act_size", 0))
    if act_size <= 0:
        raise RuntimeError("SAE ckpt에 act_size가 없습니다.")
    sae_cfg = dict(pkg.get("sae_config", {}))
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    sae = create_sae(sae_cfg.get("sae_type", "batch-topk"), sae_cfg)
    sae.load_state_dict(pkg.get("sae_state", {}), strict=False)
    sae.to(device).eval()
    dict_size = int(sae_cfg.get("dict_size", int(sae_cfg.get("expansion_factor", 8)) * act_size))
    if dict_size <= 0:
        raise RuntimeError("SAE dict_size 추정 실패.")
    return sae, act_size, dict_size

# ---------------------------
# 모듈 경로 해결(path@k)
# ---------------------------
def _index_into(obj, idx: int):
    if isinstance(obj, (nn.Sequential, nn.ModuleList, list, tuple)):
        return obj[idx]
    raise AttributeError(f"Cannot index [{idx}] into object of type {type(obj)}")

def _key_into(obj, key: str):
    if isinstance(obj, nn.ModuleDict):
        if key in obj: return obj[key]
        raise AttributeError(f"Key '{key}' not in ModuleDict")
    if hasattr(obj, key): return getattr(obj, key)
    raise AttributeError(f"Attribute '{key}' not found on {type(obj)}")

def resolve_module_path(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for part in path.split("."):
        if not part: continue
        if part.isdigit():
            cur = _index_into(cur, int(part))
        else:
            cur = _key_into(cur, part)
    return cur

def get_module_and_out_index(model: nn.Module, layer_spec: str):
    layer_spec = str(layer_spec).strip()
    if "@" in layer_spec:
        path, out_sel = layer_spec.rsplit("@", 1)
        out_idx = (int(out_sel) if out_sel.strip() != "" else None)
    else:
        path, out_idx = layer_spec, None
    return resolve_module_path(model, path.strip()), out_idx

# ---------------------------
# 데이터 준비 (VOS와 동일 전처리)
# ---------------------------
def load_frames_from_disk(image_root: Path, video_id: str, seq_full: List[int]) -> List[Image.Image]:
    outs = []
    for t in seq_full:
        p = None
        for pat in (f"{t:05d}.jpg", f"{t:05d}.png", f"{t}.jpg", f"{t}.png"):
            cand = image_root / video_id / pat
            if cand.exists(): p = cand; break
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

def make_repro_collate():
    def _seq_idx(dp) -> list[int]:
        out = []
        for f in getattr(dp, 'frames', []):
            objs = getattr(f, 'objects', None)
            out.append(int(objs[0].frame_index) if objs else -1)
        return out if out else [-1]
    def collate(items: List[VideoDatapoint]) -> Dict[str, Any]:
        Tlen = len(items[0].frames)
        H, W = items[0].frames[0].data.shape[-2:]
        pixel_values = torch.stack(
            [torch.stack([f.data for f in it.frames], dim=0) for it in items], dim=0
        ).contiguous()
        masks_per_t = [[] for _ in range(Tlen)]
        for it in items:
            acc_per_t = []
            for t in range(Tlen):
                acc_per_t.append(torch.zeros(H, W, dtype=torch.bool))
            for t in range(Tlen):
                masks_per_t[t].append(acc_per_t[t])
        masks = [torch.stack(masks_per_t[t], dim=0) for t in range(Tlen)]
        return dict(
            pixel_values=pixel_values,
            masks=masks,
            video_path=[str(getattr(items[0],'video_id',''))],
            seq_frame_idx=[_seq_idx(items[0])],
            video_size=[(int(H), int(W))],
            time_reversed=[False],
            video_id=[-1],
        )
    return collate

# ---------------------------
# 프레임 컨텍스트 & 훅
# ---------------------------
class FrameContext:
    def __init__(self): self.cur_fidx = -1

def patch_track_step_with_ctx(model, ctx: FrameContext):
    target = model.module if hasattr(model, "module") else model
    if not hasattr(target, "track_step"): return None
    orig = target.track_step
    def wrapped(*args, **kwargs):
        fidx = kwargs.get("frame_idx", kwargs.get("stage_id", -1))
        if fidx is None and len(args) >= 2: fidx = args[1]
        try: ctx.cur_fidx = int(fidx)
        except Exception: ctx.cur_fidx = -1
        return orig(*args, **kwargs)
    target.track_step = wrapped
    return orig

class LayerCapture:
    def __init__(self, model, layer_spec: str, ctx=None):
        self.ctx = ctx
        self.mod, self.out_idx = get_module_and_out_index(model, layer_spec)
        self.records: List[Tuple[Tuple[int, ...], torch.Tensor, int]] = []
        self.handle = self.mod.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        out = output
        if self.out_idx is not None:
            if isinstance(output, (tuple, list)):
                out = output[self.out_idx]
            elif self.out_idx == 0:
                out = output
            else:
                raise TypeError(f"Output type {type(output)} cannot take index @{self.out_idx}")
        out = out.detach().contiguous()
        fidx = getattr(self.ctx, "cur_fidx", -1) if self.ctx is not None else -1
        self.records.append((tuple(out.shape), out, fidx))

    def close(self):
        try: self.handle.remove()
        except Exception: pass

def pick_vector_from_records(records, seq_idx: int, y: int, x: int, B: int, T: int):
    """
    image_encoder.trunk@3는 보통 (B*T,C,H,W) 또는 (B,T,C,H,W) 또는 (B,C,H,W) per-frame.
    혹시 토큰화된 (B,M,C) 케이스도 지원.
    """
    for shape, ten, fidx in records:
        if len(shape) == 5:  # (B,T,C,H,W)
            b,t,c,h,w = shape
            if b != B or t != T: continue
            if not (0 <= seq_idx < t): continue
            if y < 0:
                raise IndexError(f"y<0 for 5D spatial tensor: need y/x on feature grid; got y={y},x={x}")
            if not (0 <= y < h and 0 <= x < w):
                raise IndexError(f"(y,x)=({y},{x}) out of bounds for shape {shape}")
            return ten[0, seq_idx, :, y, x].contiguous(), ("5D", fidx, shape)

        elif len(shape) == 4:  # (B*T,C,H,W) 또는 (B,C,H,W)
            bt,c,h,w = shape
            if bt == B*T:  # (BT,C,H,W)
                if y < 0:
                    raise IndexError(f"y<0 for 4D spatial tensor: need y/x on feature grid")
                return ten.view(B,T,c,h,w)[0, seq_idx, :, y, x].contiguous(), ("4D_BT", fidx, shape)
            else:  # (B,C,H,W) per-frame (fidx로 구분)
                if fidx == seq_idx:
                    if y < 0:
                        raise IndexError(f"y<0 for BCHW tensor")
                    return ten[0, :, y, x].contiguous(), ("4D_BCHW_ctx", fidx, shape)

        elif len(shape) == 3:  # (B,M,C) per-frame 토큰화
            b,m,c = shape
            if b != B: continue
            if fidx == seq_idx:
                if not (0 <= x < m):
                    raise IndexError(f"x={x} out of bounds for shape {shape}")
                return ten[0, x, :].contiguous(), ("3D_BMC_ctx", fidx, shape)

    raise RuntimeError(f"대상 벡터를 찾지 못했습니다: seq_idx={seq_idx}, (y,x)=({y},{x}), "
                       f"hook_shapes={[r[0] for r in records]}")

# ---------------------------
# 핵심 검증
# ---------------------------
def verify_row_trunk3(
    row: Dict[str, Any],
    csv_meta: Dict[str, Any],
    cfg_path: Path,
    image_root: Path,
    device: torch.device,
    atol: float = 5e-3,
) -> bool:

    # 1) 행 파싱 + 레이어 가드
    layer_csv = str(row["layer"]).strip()
    if layer_csv != EXPECTED_LAYER:
        raise RuntimeError(f"이 스크립트는 {EXPECTED_LAYER} 전용입니다. CSV layer={layer_csv}")

    dim = int(row["dim"])
    v_csv = float(row["value"])
    video_id = str(row["video_path"])
    seq_idx = int(row["seq_idx"])
    y = int(row["y"]); x = int(row["x"])

    try:
        seq_full = [int(v) for v in json.loads(row["seq_full"])]
    except Exception as e:
        raise ValueError(f"seq_full column malformed: {e}")

    # 2) 설정/경로
    if not cfg_path.exists(): raise FileNotFoundError(cfg_path)
    cfg = OmegaConf.load(str(cfg_path))

    model_yaml = csv_meta["fp"]["model_yaml"]
    model_ckpt = csv_meta["fp"].get("model_ckpt")
    sae_ckpt   = csv_meta["fp"]["sae_ckpt"]

    # 해상도
    try:
        target_res = int(cfg.dataset.resize)
    except Exception:
        target_res = int(row.get("video_h") or 1024)

    # 3) 모델/SAE
    model = load_sam2(model_yaml, model_ckpt, device)
    sae, act_size, dict_size = load_sae(sae_ckpt, device)

    # 4) 프레임 & 전처리
    frames_pil = load_frames_from_disk(Path(image_root), video_id, seq_full)
    vdp = build_vos_datapoint(frames_pil, seq_full, video_id)
    vdp = apply_indexing_transforms(vdp, target_res)

    # 5) 배치 합치기
    collate = make_repro_collate()
    batch = collate([vdp])  # pixel_values: (1,T,3,H,W)
    B, T = int(batch["pixel_values"].shape[0]), int(batch["pixel_values"].shape[1])
    if B != 1: raise RuntimeError("검증은 B=1 배치만 지원합니다.")
    if not (0 <= seq_idx < T):
        raise IndexError(f"seq_idx={seq_idx} out of bounds for T={T}")

    # 6) 어댑터 (프롬프트 주입 없음)
    model_for_adapter = model.module if hasattr(model, "module") else model
    adapter = SAM2EvalAdapter(model=model_for_adapter, device=str(device))
    if hasattr(adapter, "prompts_enabled"):
        adapter.prompts_enabled = False
    if hasattr(adapter, "_inject_prompts_into_batch"):
        adapter._inject_prompts_into_batch = (lambda _wrapped: None)

    # 타임라인이 없으면 프롬프트 전부 None (CSV 샘플이 그런 케이스라고 했지)
    batch["multistep_point_inputs"] = [None] * T

    wrapped = adapter.preprocess_input(batch)

    # 7) 훅 & 컨텍스트
    ctx = FrameContext()
    orig_track = patch_track_step_with_ctx(model, ctx)
    cap = LayerCapture(model, EXPECTED_LAYER, ctx)

    with torch.no_grad():
        _ = model(wrapped)

    if orig_track is not None:
        (model.module if hasattr(model, "module") else model).track_step = orig_track

    # 8) 벡터 인덱싱
    vec, chosen_info = pick_vector_from_records(cap.records, seq_idx, y, x, B=1, T=T)
    cap.close()

    if vec.dim() != 1:
        raise RuntimeError(f"선택된 벡터 차원 오류: {vec.shape}")
    if vec.numel() != act_size:
        raise RuntimeError(f"SAE act_size 불일치: vecC={vec.numel()} vs SAE.act_size={act_size}")

    # 9) SAE 재인코딩 비교
    z = sae.encode(vec.unsqueeze(0).to(device).float())
    if z.shape[-1] != dict_size:
        raise RuntimeError(f"SAE dict_size 불일치: got z.D={z.shape[-1]} vs {dict_size}")
    val = float(z[0, int(dim)].item())

    diff = abs(val - float(v_csv))
    if diff > atol:
        raise AssertionError(f"불일치: csv={float(v_csv):.6g} vs reenc={val:.6g} |Δ|={diff:.4g} > atol={atol}")

    print("\n=== Verification OK (image_encoder.trunk@3) ===")
    print(f"Picked: kind={chosen_info[0]} ctx_fidx={chosen_info[1]} shape={chosen_info[2]}")
    print(f"Vector(C)={vec.numel()}  SAE(D)={z.shape[-1]}")
    print(f"CSV value={float(v_csv):.6g}  Re-encoded={val:.6g}  |Δ|={diff:.3g} (≤ {atol})")
    return True

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default='configs/sam2_sav_feature_index_test.yaml', help="인덱싱에 사용한 YAML 경로")
    ap.add_argument("--csv", type=str, default='test_trunk.csv', help="Decile CSV 경로")
    ap.add_argument("--row", type=int, default=0, help="CSV 데이터 행 인덱스(0-based)")
    ap.add_argument("--image_root", type=str, default='/data/SA-V/sav_val/JPEGImages_24fps', )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--atol", type=float, default=5e-3)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists(): raise FileNotFoundError(csv_path)

    csv_meta = parse_csv_meta_header(csv_path)
    row = read_csv_row(csv_path, args.row)

    ok = verify_row_trunk3(
        row=row,
        csv_meta=csv_meta,
        cfg_path=Path(args.config),
        image_root=Path(args.image_root),
        device=torch.device(args.device),
        atol=args.atol,
    )
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
