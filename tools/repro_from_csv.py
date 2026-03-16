#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head 레이어 검증 (memory/mask encoder/decoder) with parquet prompts

CSV/Parquet -> 샘플 복원 -> SAM2EvalAdapter(preprocess + prompt inject) -> SAM v2 -> 훅 인덱싱 -> SAE 검증

사용 예:
python verify_head_layers.py \
  --config configs/sam2_sav_feature_index_test.yaml \
  --csv your_deciles.csv \
  --row 42 \
  --layer sam_mask_encoder.transformer.1@0 \
  --image_root /data/SA-V/sav_val/JPEGImages_24fps \
  --parquet_dir /path/to/prompts_parquet_dir \
  --device cuda
"""

import os, json, csv, argparse, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from hydra.utils import instantiate
from training.utils import checkpoint_utils

# VOS types/transforms
from training.utils.data_utils import Frame, Object, VideoDatapoint
from training.dataset import transforms as T

# parquet (프롬프트 타임라인 복구)
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

# SAM2 Eval adapter & SAE
from src.sae.activation_stores.universal_activation_store import SAM2EvalAdapter
from src.sae.registry import create_sae

logger = logging.getLogger("verify_head")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------
# CSV utils
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
# Model / SAE loaders
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
# Resolve "path@k"
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
# Data prep (VOS-like)
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
            for t in range(Tlen):
                masks_per_t[t].append(torch.zeros(H, W, dtype=torch.bool))
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
# Prompt timeline -> per-frame points (strict)
# ---------------------------
def build_multistep_prompts_for_sequence(
    parquet_root: Optional[Path],
    video_id: str,
    seq_full: List[int],
    prompts_timeline: Dict[str, str],
    device: torch.device,
) -> List[Optional[Dict[str, torch.Tensor]]]:
    """
    timeline(t->uid) 있으면 parquet에서 해당 uid의 최신 step 좌표를 복구.
    없으면 모두 None.
    """
    Tlen = len(seq_full)
    out: List[Optional[Dict[str, torch.Tensor]]] = [None] * Tlen
    want = {int(k): str(v) for k, v in (prompts_timeline or {}).items() if str(v)}
    if not want:
        return out

    if parquet_root is None:
        raise RuntimeError("prompts_timeline이 있는데 --parquet_dir가 필요합니다.")
    if pq is None:
        raise RuntimeError("pyarrow가 설치되어야 합니다 (prompts_timeline 사용 중).")
    if not Path(parquet_root).exists():
        raise FileNotFoundError(f"parquet_root not found: {parquet_root}")

    shards = sorted(Path(parquet_root).glob("prompts-*.parquet"))
    if not shards:
        raise RuntimeError("parquet_root에 prompts-*.parquet 가 없습니다.")

    rows_by_t: Dict[int, Dict[str, Any]] = {}
    for shard in shards:
        tbl = pq.read_table(shard, columns=[
            "prompt_uid","video_id","video_path","seq_abs",
            "points_x","points_y","point_labels","point_steps"
        ])
        pd = tbl.to_pydict()
        n = tbl.num_rows
        for i in range(n):
            uid = (pd["prompt_uid"][i] or "")
            if uid not in want.values(): continue
            vid = pd.get("video_id", [""])[i] or pd.get("video_path", [""])[i] or ""
            if str(vid) != str(video_id): continue
            seq_abs = int(pd["seq_abs"][i]) if pd.get("seq_abs") else -1
            try:
                t_local = seq_full.index(seq_abs)
            except ValueError:
                continue
            rows_by_t.setdefault(t_local, {
                "uid": uid,
                "points_x": pd.get("points_x", [[]])[i] or [],
                "points_y": pd.get("points_y", [[]])[i] or [],
                "point_labels": pd.get("point_labels", [[]])[i] or [],
                "point_steps": pd.get("point_steps", [[]])[i] or [],
            })

    for t, r in rows_by_t.items():
        px = r["points_x"]; py = r["points_y"]
        pl = r["point_labels"]; ps = r["point_steps"]
        if not px: continue
        max_step = max(ps) if ps else 0
        best_coords, best_labels = [], []
        for x, y, l, s in zip(px, py, pl, ps):
            if int(s) == int(max_step):
                best_coords.append([float(x), float(y)])
                best_labels.append(int(l))
        if best_coords:
            out[t] = {
                "point_coords": torch.tensor(best_coords, dtype=torch.float32, device=device).unsqueeze(0),
                "point_labels": torch.tensor(best_labels, dtype=torch.long, device=device).unsqueeze(0),
            }

    missing_ts = sorted([t for t, uid in want.items() if uid and (t not in rows_by_t or out[t] is None)])
    if missing_ts:
        raise RuntimeError(f"timeline 지정 프레임 {missing_ts} 에 대한 prompt 좌표를 parquet에서 복구하지 못했습니다.")
    return out

# ---------------------------
# Frame context & hook
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

def pick_vector_from_records(
    records: List[Tuple[Tuple[int, ...], torch.Tensor, int]],
    seq_idx: int, y: int, x: int, B: int, T: int
) -> Tuple[torch.Tensor, Tuple[str, int, Tuple[int, ...]]]:
    """
    여러 호출 중 '가장 마지막' 호출을 우선으로 역순 검색.
    지원: (B,T,C,H,W), (B*T,C,H,W), per-frame (B,C,H,W), token형 (B,M,C).
    """
    for shape, ten, fidx in reversed(records):
        if len(shape) == 5:  # (B,T,C,H,W)
            b,t,c,h,w = shape
            if b != B or t != T: continue
            if y < 0:  # 토큰형 인덱스를 공간 텐서에 쓰려는 경우는 skip
                continue
            if not (0 <= seq_idx < t and 0 <= y < h and 0 <= x < w): continue
            return ten[0, seq_idx, :, y, x].contiguous(), ("5D", fidx, shape)

        elif len(shape) == 4:
            bt,c,h,w = shape
            if bt == B*T:  # (B*T,C,H,W)
                if y < 0:  # 토큰형 인덱스면 skip
                    continue
                if not (0 <= y < h and 0 <= x < w): continue
                return ten.view(B,T,c,h,w)[0, seq_idx, :, y, x].contiguous(), ("4D_BT", fidx, shape)
            else:          # per-frame (B,C,H,W)
                if fidx == seq_idx and y >= 0 and 0 <= x < w and 0 <= y < h:
                    return ten[0, :, y, x].contiguous(), ("4D_BCHW_ctx", fidx, shape)

        elif len(shape) == 3:  # token형 (B,M,C)
            b,m,c = shape
            if b != B: continue
            if fidx == seq_idx and y < 0 and 0 <= x < m:
                return ten[0, x, :].contiguous(), ("3D_BMC_ctx", fidx, shape)

    raise RuntimeError(
        f"대상 벡터를 찾지 못했습니다: seq_idx={seq_idx}, (y,x)=({y},{x}), "
        f"hook_shapes={[r[0] for r in records]}"
    )

# ---------------------------
# Core verify
# ---------------------------
def verify_row_head(
    row: Dict[str, Any],
    csv_meta: Dict[str, Any],
    cfg_path: Path,
    image_root: Path,
    parquet_dir: Optional[Path],
    layer_spec: str,
    device: torch.device,
    use_prompts: bool = True,
    atol: float = 5e-3,
) -> bool:

    # 1) parse row
    dim = int(row["dim"])
    v_csv = float(row["value"])
    video_id = str(row["video_path"])
    seq_idx = int(row["seq_idx"])
    y = int(row["y"]); x = int(row["x"])
    try:
        seq_full = [int(v) for v in json.loads(row["seq_full"])]
    except Exception as e:
        raise ValueError(f"seq_full column malformed: {e}")

    try:
        prompts_timeline = json.loads(row.get("prompts_timeline") or "{}")
    except Exception:
        prompts_timeline = {}

    # 2) config / paths
    if not cfg_path.exists(): raise FileNotFoundError(cfg_path)
    cfg = OmegaConf.load(str(cfg_path))

    model_yaml = csv_meta["fp"]["model_yaml"]
    model_ckpt = csv_meta["fp"].get("model_ckpt")
    sae_ckpt   = csv_meta["fp"]["sae_ckpt"]

    # resolution
    try:
        target_res = int(cfg.dataset.resize)
    except Exception:
        target_res = int(row.get("video_h") or 1024)

    # 3) load model/sae
    model = load_sam2(model_yaml, model_ckpt, device)
    sae, act_size, dict_size = load_sae(sae_ckpt, device)

    # 4) frames & transforms
    frames_pil = load_frames_from_disk(Path(image_root), video_id, seq_full)
    vdp = build_vos_datapoint(frames_pil, seq_full, video_id)
    vdp = apply_indexing_transforms(vdp, target_res)

    # 5) collate
    collate = make_repro_collate()
    batch = collate([vdp])  # pixel_values: (1,T,3,H,W)
    B, T = int(batch["pixel_values"].shape[0]), int(batch["pixel_values"].shape[1])
    if B != 1: raise RuntimeError("검증은 B=1 배치만 지원합니다.")
    if not (0 <= seq_idx < T):
        raise IndexError(f"seq_idx={seq_idx} out of bounds for T={T}")

    # 6) prompts (timeline -> parquet -> per-frame)
    if use_prompts:
        msteps = build_multistep_prompts_for_sequence(
            parquet_root=(Path(parquet_dir) if parquet_dir else None),
            video_id=video_id, seq_full=seq_full,
            prompts_timeline=prompts_timeline, device=device
        )
    else:
        msteps = [None] * T
    batch["multistep_point_inputs"] = msteps

    # 7) adapter (프롬프트 주입 사용)
    model_for_adapter = model.module if hasattr(model, "module") else model
    adapter = SAM2EvalAdapter(model=model_for_adapter, device=str(device))
    if hasattr(adapter, "prompts_enabled"):
        adapter.prompts_enabled = bool(use_prompts)
    wrapped = adapter.preprocess_input(batch)

    # 8) hook & ctx
    ctx = FrameContext()
    orig_track = patch_track_step_with_ctx(model, ctx)
    cap = LayerCapture(model, layer_spec, ctx)

    with torch.no_grad():
        _ = model(wrapped)

    if orig_track is not None:
        (model.module if hasattr(model, "module") else model).track_step = orig_track

    # 9) pick vector
    vec, chosen_info = pick_vector_from_records(cap.records, seq_idx, y, x, B=1, T=T)
    cap.close()

    # 10) SAE encode & compare
    if vec.dim() != 1:
        raise RuntimeError(f"선택된 벡터 차원 오류: {vec.shape}")
    if vec.numel() != act_size:
        raise RuntimeError(f"SAE act_size 불일치: vecC={vec.numel()} vs SAE.act_size={act_size}")

    z = sae.encode(vec.unsqueeze(0).to(device).float())
    if z.shape[-1] != dict_size:
        raise RuntimeError(f"SAE dict_size 불일치: got z.D={z.shape[-1]} vs {dict_size}")
    val = float(z[0, int(dim)].item())

    diff = abs(val - float(v_csv))
    if diff > atol:
        raise AssertionError(f"불일치: csv={float(v_csv):.6g} vs reenc={val:.6g} |Δ|={diff:.4g} > atol={atol}")

    print("\n=== Verification OK (head) ===")
    print(f"Layer: {layer_spec}")
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
    ap.add_argument("--csv", type=str, default='test_indexing.csv', help="Decile CSV 경로")
    ap.add_argument("--row", type=int, default=0, help="CSV 데이터 행 인덱스(0-based)")
    ap.add_argument("--image_root", type=str, default='/data/SA-V/sav_val/JPEGImages_24fps', )
    ap.add_argument("--layer", type=str, default='sam_mask_decoder.transformer.layers.1@0', help="예: memory_encoder@vision_feature, sam_mask_encoder.transformer.1@0")
    ap.add_argument("--parquet_dir", type=str, default='outputs/sae_index_test/.state/prompts', help="prompts-*.parquet가 있는 디렉토리 (timeline 있을 때 필수)")
    ap.add_argument("--no_prompts", default=False, help="프롬프트 주입을 끕니다(디버그용)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--atol", type=float, default=5e-3)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists(): raise FileNotFoundError(csv_path)

    csv_meta = parse_csv_meta_header(csv_path)
    row = read_csv_row(csv_path, args.row)

    ok = verify_row_head(
        row=row,
        csv_meta=csv_meta,
        cfg_path=Path(args.config),
        image_root=Path(args.image_root),
        parquet_dir=(Path(args.parquet_dir) if args.parquet_dir else None),
        layer_spec=args.layer,
        device=torch.device(args.device),
        use_prompts=(not args.no_prompts),
        atol=args.atol,
    )
    if not ok:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
