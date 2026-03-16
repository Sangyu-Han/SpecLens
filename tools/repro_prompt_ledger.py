#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-trip reconstruction WITHOUT batch scanning (Parquet-only):
- Pick one decile row (layer, unit)
- Reconstruct single-sample BatchedVideoDatapoint from (video name, seq_full)
- If prompt_id>=0: rebuild offline_prompt from prompts parquet (prompt_id, uid)
- Run once, hook layer, index activation at (frame, y, x) with stride
- SAE encode and compare with stored score

Supports:
  * Dataset-style transforms (T.RandomResizeAPI / ToTensorAPI / NormalizeAPI)
  * No-prompt path (prompt_id == -1) for image-encoder layers
"""

import os, json, yaml, math, hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
import pyarrow as pa
from PIL import Image
import numpy as np

# === project imports ===
from training.dataset import transforms as T
from training.utils.data_utils import Frame, Object, VideoDatapoint, BatchedVideoDatapoint
from src.data.sa_v.repro_vosdataset import OfflinePromptMeta

# ---------- minimal utils ----------
def load_obj(dotted: str):
    if ":" in dotted:
        mod, name = dotted.split(":", 1)
    else:
        mod, name = dotted.rsplit(".", 1)
    import importlib
    m = importlib.import_module(mod)
    return getattr(m, name)

def split_layer_and_branch(layer: str) -> Tuple[str, Optional[int]]:
    if "@" in layer:
        base, idx = layer.rsplit("@", 1)
        try:
            return base, int(idx)
        except ValueError:
            return layer, None
    return layer, None

def sanitize_layer_name(layer: str) -> str:
    return layer.replace("/", "_").replace(":", "_").replace(" ", "_")

def sha256_of(path: str | Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""

def stable_u64(s: str) -> int:
    h = hashlib.sha256(str(s).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False) & ((1 << 63) - 1)

# ---------- decile pick ----------
def pick_best_top1_row(dec_ledger, layer: str, unit: int, num_deciles: int) -> Optional[Dict[str, Any]]:
    best = None
    for seg in range(num_deciles):
        tbl = dec_ledger.topn_for(layer=layer, unit=unit, decile=seg, n=1)
        if tbl is None or tbl.num_rows == 0:
            continue
        row = {c: tbl.column(c)[0].as_py() for c in tbl.column_names}
        if (best is None) or (float(row.get("score", 0.0)) > float(best.get("score", 0.0))):
            best = row
    return best

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
    # transforms 적용 후 Frame.data 가 torch.Tensor[3,H,W] 로 들어있음
    out = []
    for fr in sample_after_tfms.frames:
        ten = fr.data
        assert torch.is_tensor(ten) and ten.ndim == 3, "Expect CHW tensor per frame after transforms"
        out.append(ten.clone().detach())
    return out

# (간단 모드) PIL → CHW (dataset transform 대신 쓸 수 있는 폴백)
def apply_transforms_simple(img: Image.Image, resize: int,
                            mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)) -> torch.Tensor:
    img = img.copy()
    w, h = img.size
    if max(w, h) != resize:
        scale = resize / max(w, h)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    w2, h2 = img.size
    pad_l = (resize - w2) // 2
    pad_t = (resize - h2) // 2
    canvas = Image.new("RGB", (resize, resize), (0,0,0))
    canvas.paste(img, (pad_l, pad_t))
    arr = np.asarray(canvas).astype(np.float32) / 255.0
    chw = torch.from_numpy(arr).permute(2,0,1)  # [3,H,W]
    mean_t = torch.tensor(mean).view(3,1,1)
    std_t  = torch.tensor(std).view(3,1,1)
    return (chw - mean_t) / std_t

# ---------- Offline prompt builder ----------
def build_offline_prompt_meta_from_prompts_table(tbl_p: pa.Table, t: int) -> OfflinePromptMeta:
    by_uid: Dict[str, Dict[str, Any]] = {}
    for i in range(tbl_p.num_rows):
        uid = int(tbl_p.column("uid")[i].as_py())
        xs = tbl_p.column("points_x")[i].as_py() or []
        ys = tbl_p.column("points_y")[i].as_py() or []
        lbs = tbl_p.column("point_labels")[i].as_py() or []
        coords = [[float(x), float(y)] for x, y in zip(xs, ys)]
        by_uid[str(uid)] = {"point_coords": coords, "point_labels": [int(v) for v in lbs]}
    return OfflinePromptMeta(
        version=1,
        use_pt_input=True,
        pt_sampling_for_eval="provided",
        init_cond_frames=[int(t)],
        frames_to_add_correction_pt=[],
        per_frame={int(t): {"by_uid": by_uid}},
    )

# ---------- BVD builders ----------
def make_single_bvd_with_prompt(
    frames_chw: List[torch.Tensor],
    name: str,
    seq_full: List[int],
    sample_id: int,
    prompt_id: int,
    prompt_tbl: pa.Table,
    t_prompt: int,
    dict_key: str,
) -> Tuple[Any, Dict[int,int]]:
    """
    With offline_prompt (prompt_id >= 0)
    """
    T = len(frames_chw)
    H, W = int(frames_chw[0].shape[-2]), int(frames_chw[0].shape[-1])
    img_stack = torch.stack(frames_chw, dim=0)      # (T,3,H,W)
    img_batch = img_stack.unsqueeze(1).contiguous() # (T,1,3,H,W)

    # uids at t_prompt
    uids = [int(prompt_tbl.column("uid")[i].as_py()) for i in range(prompt_tbl.num_rows)]
    uids = sorted(uids)
    if not uids:
        raise RuntimeError("No prompt rows (uid) at t_prompt; cannot reconstruct.")
    N = len(uids)

    obj_to_frame_idx = torch.zeros((T, N, 2), dtype=torch.long)
    for t in range(T):
        obj_to_frame_idx[t, :, 0] = t
        obj_to_frame_idx[t, :, 1] = 0
    masks = torch.zeros((T, N, H, W), dtype=torch.bool)

    class SimpleMeta: pass
    md = SimpleMeta()
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
    md.object_key_by_t = torch.tensor(uids, dtype=torch.long).unsqueeze(0).repeat(T,1).contiguous()

    md.offline_prompt = build_offline_prompt_meta_from_prompts_table(prompt_tbl, t_prompt)
    md.batch_sig = hashlib.sha1(md.mask_uid_by_t.numpy().tobytes()).hexdigest()[:16]

    bvd = BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=md,
        dict_key=dict_key,
        batch_size=[T],
    )
    uid_to_lane = {int(u): i for i, u in enumerate(uids)}
    return bvd, uid_to_lane

def make_single_bvd_no_prompt(
    frames_chw: List[torch.Tensor],
    name: str,
    seq_full: List[int],
    sample_id: int,
    dict_key: str,
    prompt_id_value: int = -1,
) -> Tuple[Any, Dict[int,int]]:
    """
    No offline_prompt path (prompt_id == -1), for encoder-side layers.
    Build a minimal BVD with N=1 dummy lane; adapter will ignore prompt injection.
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

    md.offline_prompt = None  # 중요: 어댑터가 프롬프트 주입을 건너뜀
    md.batch_sig = hashlib.sha1(md.mask_uid_by_t.numpy().tobytes()).hexdigest()[:16]

    bvd = BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        metadata=md,
        dict_key=dict_key,
        batch_size=[T],
    )
    uid_to_lane = {}  # no uid
    return bvd, uid_to_lane

# ---------- activation capture ----------
class LayerCapture:
    def __init__(self, layer_name: str):
        self.base, self.branch = split_layer_and_branch(layer_name)
        self.by_frame: dict[int, torch.Tensor] = {}
        self.last: Optional[torch.Tensor] = None

    def hook(self, adapter, *, frame_var_name: str = "_frame_idx_var"):
        from src.sae.activation_stores.hook_helper import walk_tensors, SEP
        def _fn(_module, _inp, out):
            ten = None
            if self.branch is not None:
                pairs = walk_tensors(self.base, out, allowed_prefixes=[f"{self.base}{SEP}{self.branch}"])
                if pairs:
                    ten = pairs[0][1]
            if ten is None:
                if isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                    ten = out[0]
                elif torch.is_tensor(out):
                    ten = out
                else:
                    pairs = walk_tensors(self.base, out)
                    ten = pairs[0][1] if pairs else None
            if ten is None:
                return
            self.last = ten.detach()
            try:
                fidx = int(getattr(adapter, frame_var_name).get())
            except Exception:
                fidx = -1
            self.by_frame[fidx] = self.last
        return _fn

# ---------- main ----------
def main():
    import argparse, random
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default='configs/sam2_sav_feature_index.yaml',
                    help="same config used for indexing(run)")
    ap.add_argument("--layer", type=str, default='sam_mask_decoder.transformer.layers.1@0',
                    help="e.g., sam_mask_decoder.transformer.layers.1@0 or image_encoder.trunk@3")
    ap.add_argument("--unit", type=int, default=0, help="SAE feature index m")
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--resize", type=int, default=1024)
    ap.add_argument("--use_dataset_tfms", action="store_true",
                    help="Use the exact indexing transforms (T.RandomResizeAPI/ToTensor/Normalize)")
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    # cfg & seed
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("indexing", {}).get("seed", 12345))
    random.seed(seed); __import__("numpy").random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ledgers
    try:
        from src.indexing.decile_parquet_ledger import DecileParquetLedger
    except Exception:
        # fallback path name
        from src.indexing.decile_parquet_ledger import DecileParquetLedger
    dec_ledger = DecileParquetLedger(cfg["indexing"]["out_dir"])

    from src.data.sa_v.offline_meta_ledger import OfflineMetaParquetLedger
    meta_ledger = OfflineMetaParquetLedger(cfg["indexing"]["offline_meta_root"])

    # 1) choose row
    layer_name = args.layer
    unit = int(args.unit)
    row = pick_best_top1_row(dec_ledger, layer=layer_name, unit=unit, num_deciles=int(args.deciles))
    assert row, f"No decile rows for layer={layer_name}, unit={unit}"
    sid   = int(row["sample_id"])
    t_tok = int(row["frame_idx"])
    y     = int(row["y"]); x = int(row["x"])
    pid   = int(row["prompt_id"])            # may be -1
    uid   = int(row.get("uid", -1))          # may be -1
    stride_step = int(row.get("stride_step", 1))
    stored_score = float(row["score"])
    print("[decile row]", json.dumps(row, ensure_ascii=False, indent=2))

    # 2) fetch sample (+ possibly prompts)
    tbl_s = meta_ledger.find_sample(sid)
    assert tbl_s.num_rows == 1, f"sample_id={sid} not unique or missing"
    name = tbl_s.column("name")[0].as_py()
    seq_full = tbl_s.column("seq_full")[0].as_py()
    dict_key = tbl_s.column("dict_key")[0].as_py()
    prompt_sets = tbl_s.column("prompt_sets")[0].as_py()
    mp = {e["frame_idx"]: e["prompt_id"] for e in prompt_sets} if prompt_sets else {}
    t_prompt = t_tok if t_tok in mp else 0

    # prompts rows (only if pid >= 0)
    tbl_p = None
    if pid >= 0:
        tbl_p = meta_ledger.find_prompts(sid, pid, t_prompt)
        if tbl_p.num_rows == 0:
            raise RuntimeError(f"prompt rows not found for sample_id={sid}, prompt_id={pid}, frame_idx={t_prompt}")

    # 3) load frames & apply transforms
    img_root = Path(cfg["dataset"]["img_folder"])
    pil_frames = load_frames_from_disk(img_root, name, [int(v) for v in seq_full])
    if args.use_dataset_tfms:
        vd = build_vos_datapoint(pil_frames, [int(v) for v in seq_full], video_id=name)
        vd_tf = apply_indexing_transforms(vd, target_res=int(args.resize))
        frames = frames_chw_from_datapoint(vd_tf)
    else:
        frames = [apply_transforms_simple(im, resize=int(args.resize)) for im in pil_frames]

    # 4) build single BVD (with or without prompt)
    if pid >= 0:
        bvd, uid2lane = make_single_bvd_with_prompt(
            frames, name, [int(v) for v in seq_full], sid, pid, tbl_p, t_prompt, dict_key
        )
        lane_idx = uid2lane.get(uid, 0)  # uid 없으면 lane 0
    else:
        bvd, _uid2lane = make_single_bvd_no_prompt(
            frames, name, [int(v) for v in seq_full], sid, dict_key, prompt_id_value=-1
        )
        lane_idx = 0  # no prompt/uid → single lane

    # 5) build model + adapter + SAE
    model_loader = load_obj(cfg["model"]["loader"])
    model = model_loader(cfg["model"], device).eval()

    # 주: 어댑터는 당신이 패치한 SAM2EvalAdapter 경로에 맞춰 주세요.
    from src.sae.activation_stores.universal_activation_store import SAM2EvalAdapter
    adapter = SAM2EvalAdapter(model, device=device, collate_fn=None)

    # SAE
    sae_root = Path(cfg["sae"]["output"]["save_path"])
    sae_dir = sae_root / sanitize_layer_name(layer_name)
    ckpts = sorted(sae_dir.glob("*.pt"))
    assert ckpts, f"No SAE ckpt at {sae_dir}"
    pkg = torch.load(ckpts[-1], map_location="cpu")
    act_size = int(pkg.get("act_size", 0)) or 0
    create_sae = load_obj(cfg["sae"]["factory"])
    sae_cfg = pkg.get("sae_config", {}) or {}
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    sae = create_sae(sae_cfg.get("sae_type", "batch-topk"), sae_cfg).to(device).eval()
    try:
        sae.load_state_dict(pkg.get("sae_state", {}), strict=False)
    except Exception:
        pass

    def sae_encode_vec(vec_c: torch.Tensor) -> float:
        with torch.no_grad():
            acts = vec_c.unsqueeze(0).to(device)  # [1,C]
            out = sae.encode(acts) if hasattr(sae, "encode") else sae(acts)
            return float(out.detach().cpu()[0, unit].item())

    # 6) hook target layer
    capture = LayerCapture(layer_name)  # '...@branch' 지원
    target_mod = None
    # 경로 해석(숫자 인덱스 포함)
    def resolve_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
        cur = model
        for tok in dotted.split("."):
            if tok.isdigit():
                cur = cur[int(tok)]
            else:
                if not hasattr(cur, tok):
                    try:
                        cur = dict(cur.named_modules())[dotted]
                        return cur
                    except Exception:
                        raise AttributeError(f"Cannot resolve module token '{tok}' in path '{dotted}'")
                else:
                    cur = getattr(cur, tok)
        return cur
    target_mod = resolve_module(adapter._unwrap_model(), split_layer_and_branch(layer_name)[0])
    h = target_mod.register_forward_hook(capture.hook(adapter))

    # 7) preprocess + forward (once)
    batch_on_dev = adapter.preprocess_input(bvd)  # current_meta 갱신 + to(device)
    with torch.no_grad():
        _ = adapter.forward(batch_on_dev)

    # 8) pick activation
    out = capture.by_frame.get(t_tok, None)
    if out is None:
        out = capture.last  # encoder 계열은 frame 컨텍스트가 없을 수 있음
        print("[warn] frame context missing; using last activation tensor.")
    assert torch.is_tensor(out), "No activation captured"

    # 9) extract vector @ (lane, y, x) with stride
    vec_c = None
    if out.ndim == 4:
        # (lanes, C, H', W')  — encoder/decoder spatial
        M, C, Hm, Wm = out.shape
        yy = max(0, min(Hm-1, y // max(1, stride_step)))
        xx = max(0, min(Wm-1, x // max(1, stride_step)))
        lane = max(0, min(M-1, lane_idx))
        vec_c = out[lane, :, yy, xx].detach().cpu().float()
    elif out.ndim == 3:
        # (lanes, L, C) — token 레이어
        M, L, C = out.shape
        lane = max(0, min(M-1, lane_idx))
        tok = max(0, min(L-1, x if x >= 0 else 0))
        vec_c = out[lane, tok, :].detach().cpu().float()
    elif out.ndim == 2:
        # (lanes, C) — lane당 하나
        M, C = out.shape
        lane = max(0, min(M-1, lane_idx))
        vec_c = out[lane, :].detach().cpu().float()
    else:
        raise RuntimeError(f"Unsupported activation shape: {tuple(out.shape)}")

    # 10) SAE encode and compare
    measured = sae_encode_vec(vec_c)
    atol = float(args.atol); rtol = float(args.rtol)
    ok = math.isclose(measured, float(stored_score), rel_tol=rtol, abs_tol=atol)
    print(f"[compare] stored={stored_score:.6g}  measured={measured:.6g}  -> {'OK' if ok else 'MISMATCH'} "
          f"(rtol={rtol}, atol={atol})")
    if not ok:
        raise SystemExit(1)
    print("[PASS] round-trip reconstruction matched.")

    # cleanup
    h.remove()

if __name__ == "__main__":
    main()
