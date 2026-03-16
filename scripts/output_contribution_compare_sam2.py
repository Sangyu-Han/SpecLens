#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable, TextIO

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.core.attribution.backends.gradients import BACKENDS as GRADIENT_BACKENDS
from src.core.hooks.spec import parse_spec
from src.core.indexing.registry_utils import sanitize_layer_name
from src.core.runtime.activation_baselines import ActivationBaselineCache
from src.core.runtime.capture import LayerCapture, MultiAnchorCapture
from src.core.runtime.controllers import AllTokensFeatureOverrideController, install_controller_autoreset_hooks
from src.core.runtime.specs import OverrideSpec
from src.core.runtime.wrappers import wrap_target_layer_with_sae
from src.packs.sam2.offline.bvd_builders import (
    apply_indexing_transforms,
    build_vos_datapoint,
    frames_chw_from_datapoint,
    load_frames_from_disk,
    make_single_bvd_no_prompt,
    make_single_bvd_with_prompt,
)
from src.packs.sam2.offline.offline_meta_ledger import OfflineMetaParquetLedger
from src.packs.sam2.models.adapters import SAM2EvalAdapter
from src.packs.sam2.models.attnlrp import enable_sam2_attnlrp, enable_sae_attnlrp
from src.packs.sam2.models.libragrad import enable_sam2_libragrad, enable_sae_libragrad
from src.utils.utils import load_obj, resolve_module
from src.packs.sam2.dataset.sa_v import repro_vosdataset as repro_ds


# ---------------------------------------------------------------------------
# Data classes / logging helpers
# ---------------------------------------------------------------------------
@dataclass
class AnchorInfo:
    spec: str
    branch: Any
    attr_name: str
    controller: AllTokensFeatureOverrideController


@dataclass
class MaskDumpConfig:
    root: Path
    sample_id: int
    frame_idx: int
    lane_idx: Optional[int]
    decile: Optional[int]
    method: str
    curve: str


class _LibragradContext:
    """
    Context manager to apply/revert SAM2 libragrad patches (model + SAEs).
    """

    def __init__(self, model: torch.nn.Module, anchors: Dict[str, AnchorInfo], gamma: Optional[float] = None) -> None:
        self.model = model
        self.anchors = anchors
        self.gamma = gamma
        self._restore: Optional[Callable[[], None]] = None

    def __enter__(self):
        self._restore = enable_sam2_libragrad(self.model, gamma=self.gamma)
        for anchor in self.anchors.values():
            sae = getattr(anchor.branch, "sae", None)
            if sae is not None:
                enable_sae_libragrad(sae)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                pass
        return False


class _AttnLRPContext:
    """
    Context manager to apply/revert SAM2 AttnLRP patches (model + SAEs).
    """

    def __init__(self, model: torch.nn.Module, anchors: Dict[str, AnchorInfo]) -> None:
        self.model = model
        self.anchors = anchors
        self._restore: Optional[Callable[[], None]] = None

    def __enter__(self):
        self._restore = enable_sam2_attnlrp(self.model)
        for anchor in self.anchors.values():
            sae = getattr(anchor.branch, "sae", None)
            if sae is not None:
                enable_sae_attnlrp(sae)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                pass
        return False


_LOG_FH: Optional[TextIO] = None


def _set_log_file(fh: Optional[TextIO]) -> None:
    global _LOG_FH
    _LOG_FH = fh


def _log_line(msg: str, *, stdout: bool = True) -> None:
    if _LOG_FH is not None:
        _LOG_FH.write(msg + "\n")
        _LOG_FH.flush()
    if stdout:
        print(msg)


def _debug_print(msg: str, *, enabled: bool) -> None:
    if enabled:
        _log_line(msg)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_baseline_cache(path: Optional[Path]) -> Optional[ActivationBaselineCache]:
    if path is None:
        return None
    return ActivationBaselineCache.load(path)


def _ensure_list(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _parse_csv_list(value: Optional[Any]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items: List[str] = []
        for entry in value:
            if entry is None:
                continue
            for tok in str(entry).split(","):
                tok = tok.strip()
                if tok:
                    items.append(tok)
        return items
    return [tok.strip() for tok in str(value).split(",") if tok.strip()]


def _normalize_method_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _split_method_variant(name: str) -> Tuple[str, bool, bool]:
    """
    Parse per-method suffix variants:
      *_sc / *_sign / *_signed / *_abs    -> force abs ranking
      *_pos / *_positive                   -> force positive-only ranking
    """
    key = _normalize_method_name(name)
    force_abs = False
    force_positive = False

    for suffix in ("_sc", "_sign", "_signed", "_abs"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            force_abs = True
            break
    for suffix in ("_pos", "_positive"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            force_positive = True
            force_abs = False
            break

    return key, force_abs, force_positive


def _parse_anchor_specs(raw: str | Sequence[str]) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = []
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                parts.append(tok)
        return parts
    specs: List[str] = []
    for val in raw:
        if val is None:
            continue
        specs.extend(_parse_anchor_specs(str(val)))
    return specs


def _expand_anchor_specs(specs: List[str]) -> List[str]:
    expanded: List[str] = []
    seen: set[str] = set()
    for raw in specs:
        parsed = parse_spec(raw)
        attr_lower = (parsed.attr or "").lower()
        candidates: List[str] = []
        base = parsed.base_with_branch
        meth = parsed.method

        if meth is None:
            # No method specified: default to SAE layer latent + error_coeff.
            if attr_lower in {"", "latent", "acts", "activation"}:
                candidates = [
                    f"{base}::sae_layer#latent",
                    f"{base}::sae_layer#error_coeff",
                ]
            elif attr_lower in {"error_coeff", "error", "residual_coeff"}:
                candidates = [f"{base}::sae_layer#error_coeff"]
            else:
                candidates = [raw]
        else:
            # Method specified: add latent + error_coeff variants when attr omitted or is latent.
            if attr_lower in {"", "latent", "acts", "activation"}:
                candidates = [
                    f"{base}::{meth}#latent",
                    f"{base}::{meth}#error_coeff",
                ]
            elif attr_lower in {"error_coeff", "error", "residual_coeff"}:
                candidates = [f"{base}::{meth}#error_coeff"]
            else:
                candidates = [raw]
        for cand in candidates:
            if cand not in seen:
                expanded.append(cand)
                seen.add(cand)
    return expanded


def _strip_attr_suffix(spec: str) -> str:
    if not spec:
        return ""
    return spec.split("#", 1)[0].strip()


def _is_ig_active(spec: str, ig_active: Optional[Sequence[str]]) -> bool:
    if not ig_active:
        return True
    spec_full = str(spec or "").strip()
    spec_base = _strip_attr_suffix(spec_full)
    for raw in ig_active:
        active = str(raw or "").strip()
        if not active:
            continue
        if "#" in active:
            if spec_full == active or spec_full.startswith(active) or active.startswith(spec_full):
                return True
        else:
            if spec_base == active or spec_base.startswith(active) or active.startswith(spec_base):
                return True
    return False


def _anchor_attr_name(spec: str) -> str:
    parsed = parse_spec(spec)
    attr = (parsed.attr or "").lower()
    if attr in {"latent", "acts", "activation"}:
        return "acts"
    if attr in {"error_coeff", "error", "residual_coeff"}:
        return "error_coeff"
    if attr in {"residual", "sae_error"}:
        return "residual"
    return attr or "acts"


def _candidate_layer_dirs(root: Path, layer: str) -> List[Path]:
    candidates = [root / sanitize_layer_name(layer)]
    parsed = parse_spec(layer)
    if parsed.method is not None:
        candidates.append(root / sanitize_layer_name(parsed.base_with_branch))
    return candidates


def _load_sae_for_layer(cfg: Dict[str, Any], layer: str, device: torch.device) -> torch.nn.Module:
    root = Path(cfg["output"]["save_path"])
    ckpts: List[Path] = []
    for candidate in _candidate_layer_dirs(root, layer):
        files = sorted(candidate.glob("*.pt"))
        if files:
            ckpts = files
            break
    if not ckpts:
        raise FileNotFoundError(f"No SAE checkpoints under {root} for layer '{layer}'")
    pkg = torch.load(ckpts[-1], map_location="cpu")
    sae_cfg = dict(pkg.get("sae_config", {}))
    act_size = int(pkg.get("act_size") or sae_cfg.get("act_size", 0))
    if act_size <= 0:
        raise RuntimeError(f"SAE checkpoint at {ckpts[-1]} is missing act_size metadata")
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    create_sae = load_obj(cfg["factory"])
    sae = create_sae(sae_cfg.get("sae_type", "batch-topk"), sae_cfg)
    state = pkg.get("sae_state", pkg.get("state_dict", {}))
    sae.load_state_dict(state, strict=False)
    sae.to(device).eval()
    for param in sae.parameters():
        param.requires_grad = False
    return sae


# ---------------------------------------------------------------------------
# Dataset helpers (borrowed from debug scripts)
# ---------------------------------------------------------------------------
def _table_first_row(table) -> Dict[str, Any]:
    rows = table.to_pylist()
    if not rows:
        raise RuntimeError("Requested table is empty")
    return rows[0]


def _frames_from_sample(ds_cfg: Dict[str, Any], name: str, seq_full: Sequence[int]) -> List[torch.Tensor]:
    img_root = Path(ds_cfg["img_folder"])
    pil_frames = load_frames_from_disk(img_root, name, [int(v) for v in seq_full])
    datapoint = build_vos_datapoint(pil_frames, [int(v) for v in seq_full], video_id=name)
    resized = apply_indexing_transforms(datapoint, target_res=int(ds_cfg.get("resize", 1024)))
    return frames_chw_from_datapoint(resized)


def _resolve_prompt_frame(row: Dict[str, Any], prompt_sets: List[Dict[str, Any]]) -> Tuple[int, Dict[int, int]]:
    frame_map = {int(entry["frame_idx"]): int(entry["prompt_id"]) for entry in (prompt_sets or [])}
    frame_idx = int(row.get("frame_idx", 0))
    if frame_idx not in frame_map and frame_map:
        frame_idx = sorted(frame_map.keys())[0]
    return frame_idx, frame_map


_GT_JSON_CACHE: Dict[str, Dict[str, Any]] = {}


def _resolve_gt_json_path(gt_root: Path, name: str) -> Path:
    candidates = [
        gt_root / f"{name}.json",
        gt_root / f"{name}_manual.json",
        gt_root / f"{name}_manual_0.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"GT json not found for '{name}' under {gt_root}")


def _load_gt_json(gt_root: Path, name: str) -> Dict[str, Any]:
    key = f"{str(gt_root)}::{name}"
    cached = _GT_JSON_CACHE.get(key)
    if cached is not None:
        return cached
    path = _resolve_gt_json_path(gt_root, name)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    _GT_JSON_CACHE[key] = data
    return data


def _decode_sav_rle(rle: Dict[str, Any]) -> np.ndarray:
    try:
        import pycocotools.mask as mask_utils  # type: ignore
    except Exception as exc:
        raise RuntimeError("pycocotools is required to decode SA-V masks.") from exc
    rle_dict = dict(rle)
    counts = rle_dict.get("counts")
    if isinstance(counts, str):
        rle_dict["counts"] = counts.encode("ascii")
    mask = mask_utils.decode(rle_dict)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def _gt_uid_masks_for_frame(
    *,
    ds_cfg: Dict[str, Any],
    name: str,
    frame_abs: int,
) -> Dict[int, torch.Tensor]:
    gt_root = Path(ds_cfg["gt_folder"])
    img_root = Path(ds_cfg["img_folder"])
    data = _load_gt_json(gt_root, name)
    masklet = data.get("masklet") or []
    if frame_abs < 0:
        raise IndexError(f"frame index out of range: {frame_abs} (masklet len={len(masklet)})")
    frame_idx = int(frame_abs)
    if frame_idx >= len(masklet):
        # SA-V metadata often stores absolute frame indices (e.g., 200, 204, ...)
        # while `masklet` is stored on a downsampled timeline (e.g., length 65).
        # Map absolute frame id into masklet index using video_frame_count.
        vfc = data.get("video_frame_count", None)
        mapped_idx: Optional[int] = None
        try:
            vfc_f = float(vfc) if vfc is not None else None
        except Exception:
            vfc_f = None
        if vfc_f is not None and vfc_f > 1.0 and len(masklet) > 1:
            scale = float(len(masklet) - 1) / float(vfc_f - 1.0)
            mapped_idx = int(round(float(frame_idx) * scale))
        if mapped_idx is None or mapped_idx < 0 or mapped_idx >= len(masklet):
            raise IndexError(
                f"frame index out of range: abs={frame_abs} mapped={mapped_idx} "
                f"(masklet len={len(masklet)} video_frame_count={vfc})"
            )
        frame_idx = int(mapped_idx)
    frame_masks = masklet[int(frame_idx)]
    if not frame_masks:
        raise RuntimeError(f"No GT masks found for {name} frame={frame_abs}")

    # Load the corresponding image so transforms can align masks to resized frames.
    frames_pil = load_frames_from_disk(img_root, name, [int(frame_abs)])
    if not frames_pil:
        raise RuntimeError(f"Failed to load frame {name}:{frame_abs} from {img_root}")
    frame_pil = frames_pil[0]

    try:
        from training.utils.data_utils import Frame, Object, VideoDatapoint
    except Exception as exc:
        raise RuntimeError("training.utils.data_utils is required to build GT masks.") from exc

    objects: List[Any] = []
    for idx, rle in enumerate(frame_masks):
        if rle is None:
            continue
        mask_np = _decode_sav_rle(rle)
        seg = torch.from_numpy(mask_np).to(torch.bool)
        objects.append(Object(object_id=int(idx), frame_index=int(frame_abs), segment=seg))
    if not objects:
        raise RuntimeError(f"No decodable GT masks for {name} frame={frame_abs}")

    w, h = frame_pil.size
    vdp = VideoDatapoint(frames=[Frame(data=frame_pil, objects=objects)], video_id=name, size=(h, w))
    vdp = apply_indexing_transforms(vdp, target_res=int(ds_cfg.get("resize", 1024)))

    uid_to_mask: Dict[int, torch.Tensor] = {}
    for obj in vdp.frames[0].objects:
        seg = obj.segment
        if seg is None:
            continue
        seg = seg.to(torch.bool).contiguous()
        uid = repro_ds._mask_uid_bool(seg)
        uid_to_mask[int(uid)] = seg
    if not uid_to_mask:
        raise RuntimeError(f"No UID masks produced for {name} frame={frame_abs}")
    return uid_to_mask


def _prompt_rows_from_gt_masks(
    *,
    prompt_rows: Sequence[Dict[str, Any]],
    ds_cfg: Dict[str, Any],
    name: str,
    seq_full: Sequence[int],
    t_prompt: int,
    method: str = "center",
) -> List[Dict[str, Any]]:
    if not prompt_rows:
        raise RuntimeError("prompt_rows is empty; cannot rebuild GT prompts")
    uids = [int(r["uid"]) for r in prompt_rows]
    frame_abs = int(seq_full[t_prompt]) if 0 <= t_prompt < len(seq_full) else int(t_prompt)
    if frame_abs < 0:
        raise RuntimeError(f"Invalid frame index for GT prompt: {frame_abs}")
    uid_to_mask = _gt_uid_masks_for_frame(ds_cfg=ds_cfg, name=name, frame_abs=frame_abs)
    masks_all: List[torch.Tensor] = [m for m in uid_to_mask.values() if torch.is_tensor(m)]
    if not masks_all:
        raise RuntimeError("No GT masks available to rebuild prompts.")

    h = int(masks_all[0].shape[-2])
    w = int(masks_all[0].shape[-1])
    areas: List[int] = []
    centroids: List[Tuple[float, float]] = []
    for m in masks_all:
        yy, xx = torch.where(m)
        if yy.numel() == 0:
            areas.append(0)
            centroids.append((float(h) * 0.5, float(w) * 0.5))
            continue
        areas.append(int(yy.numel()))
        centroids.append((float(yy.float().mean().item()), float(xx.float().mean().item())))
    largest_idx = int(np.argmax(np.asarray(areas, dtype=np.int64))) if areas else 0

    # Match each ledger prompt row to a GT object by point containment; fallback to nearest centroid.
    masks: List[torch.Tensor] = []
    for row in prompt_rows:
        pxs = row.get("points_x") or []
        pys = row.get("points_y") or []
        picked_idx: Optional[int] = None
        if pxs and pys:
            x = int(round(float(pxs[0])))
            y = int(round(float(pys[0])))
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            containing = [i for i, m in enumerate(masks_all) if bool(m[y, x])]
            if containing:
                picked_idx = max(containing, key=lambda i: areas[i])
            else:
                best_i = 0
                best_d = float("inf")
                for i, (cy, cx) in enumerate(centroids):
                    d = (cy - float(y)) ** 2 + (cx - float(x)) ** 2
                    if d < best_d:
                        best_d = d
                        best_i = i
                picked_idx = int(best_i)
        if picked_idx is None:
            picked_idx = int(largest_idx)
        masks.append(masks_all[picked_idx])

    gt_masks = torch.stack(masks, dim=0).unsqueeze(1)  # [N,1,H,W]
    try:
        from sam2.modeling.sam2_utils import get_next_point
    except Exception as exc:
        raise RuntimeError("sam2.modeling.sam2_utils.get_next_point is required for GT prompts.") from exc
    pts, lbl = get_next_point(gt_masks=gt_masks, pred_masks=None, method=str(method))
    pts = pts.detach().cpu()
    lbl = lbl.detach().cpu()

    rebuilt: List[Dict[str, Any]] = []
    for idx, uid in enumerate(uids):
        coords = pts[idx].tolist()
        labels = lbl[idx].tolist()
        if coords and isinstance(coords[0], (int, float)):
            coords = [coords]
        if isinstance(labels, (int, np.integer)):
            labels = [labels]
        xs = [float(c[0]) for c in coords]
        ys = [float(c[1]) for c in coords]
        lbs = [int(v) for v in labels]
        rebuilt.append(
            {
                "uid": int(uid),
                "points_x": xs,
                "points_y": ys,
                "point_labels": lbs,
            }
        )
    return rebuilt


def _build_bvd_for_row(
    *,
    row: Dict[str, Any],
    meta_ledger: OfflineMetaParquetLedger,
    ds_cfg: Dict[str, Any],
    prompt_source: str = "ledger",
    gt_prompt_method: str = "center",
) -> Tuple[Any, int]:
    sample_id = int(row["sample_id"])
    tbl_sample = meta_ledger.find_sample(sample_id)
    sample = _table_first_row(tbl_sample)

    name = sample.get("name") or ""
    seq_full = [int(v) for v in (sample.get("seq_full") or [])]
    dict_key = sample.get("dict_key", "unknown")
    prompt_sets = sample.get("prompt_sets") or []
    t_prompt, prompt_map = _resolve_prompt_frame(row, prompt_sets)
    prompt_id = int(row.get("prompt_id", -1))
    if prompt_id < 0:
        prompt_id = prompt_map.get(t_prompt, -1)

    frames = _frames_from_sample(ds_cfg, name, seq_full)

    if prompt_id >= 0:
        tbl_prompts = meta_ledger.find_prompts(sample_id, prompt_id, t_prompt)
        prompt_rows = tbl_prompts
        if str(prompt_source).lower() == "gt":
            if hasattr(tbl_prompts, "num_rows"):
                rows_list = tbl_prompts.to_pylist()
            else:
                rows_list = list(tbl_prompts) if tbl_prompts is not None else []
            prompt_rows = _prompt_rows_from_gt_masks(
                prompt_rows=rows_list,
                ds_cfg=ds_cfg,
                name=name,
                seq_full=seq_full,
                t_prompt=t_prompt,
                method=str(gt_prompt_method),
            )
        bvd, uid_map = make_single_bvd_with_prompt(
            frames,
            name,
            seq_full,
            sample_id,
            prompt_id,
            prompt_rows,
            t_prompt,
            dict_key,
        )
    else:
        bvd, uid_map = make_single_bvd_no_prompt(
            frames,
            name,
            seq_full,
            sample_id,
            dict_key,
            prompt_id_value=-1,
        )
    lane_idx = uid_map.get(int(row.get("uid", -1)), 0)
    return bvd, lane_idx


def _record_adapter_caches(adapter: SAM2EvalAdapter, batch_on_dev) -> None:
    with torch.no_grad():
        with adapter.clicks_cache("record"), adapter.prompt_inputs_cache("record"):
            adapter.model(batch_on_dev)


def _make_forward_runner(adapter: SAM2EvalAdapter, batch_on_dev):
    def _run(require_grad: bool = False):
        with torch.set_grad_enabled(require_grad):
            with adapter.clicks_cache("replay"), adapter.prompt_inputs_cache("replay"):
                return adapter.model(batch_on_dev)

    return _run


# ---------------------------------------------------------------------------
# Objective helpers (mask -> scalar objective)
# ---------------------------------------------------------------------------
def _choose_lane(lane_idx: Optional[int], tensor: torch.Tensor) -> int:
    if not torch.is_tensor(tensor):
        return 0
    lane = int(lane_idx if lane_idx is not None else 0)
    if tensor.shape[0] <= lane:
        lane = max(0, tensor.shape[0] - 1)
    return lane


def _infer_frame_idx(spec_name: str, parsed: Optional[Any] = None) -> int:
    parsed = parsed or parse_spec(spec_name)
    # Priority: explicit int branch > base_branch > method_branch
    for candidate in (parsed.branch, getattr(parsed, "base_branch", None), getattr(parsed, "method_branch", None)):
        if isinstance(candidate, int):
            return int(candidate)
    # Fallback: parse trailing @token in base if numeric (handles model@0@pred_masks_high_res)
    base_tokens = (parsed.base or "").split("@")
    for tok in reversed(base_tokens):
        if tok.isdigit():
            return int(tok)
    return 0


def _objective_from_masks(
    mask_lists: Dict[str, List[torch.Tensor]],
    lane_idx: Optional[int],
    threshold: float,
    *,
    ref_spec: Optional[str] = None,
    fixed_mask: Optional[List[torch.Tensor]] = None,
    fixed_ref_logits: Optional[List[torch.Tensor]] = None,
    objective_mode: str = "mask",
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    if not mask_lists:
        raise RuntimeError("No mask tensors captured; check mask specs.")

    # Group tensors by frame index inferred from branch tokens (model@0, model@1, ...).
    frame_buckets: Dict[int, List[torch.Tensor]] = {}
    for spec_name, tensors in mask_lists.items():
        if not tensors:
            continue
        parsed = parse_spec(spec_name)
        frame_idx = _infer_frame_idx(spec_name, parsed)
        frame_buckets.setdefault(frame_idx, []).extend(tensors)

    # Choose reference frames: prefer ref_spec tensors if provided; otherwise first tensor per frame sorted by index.
    ref_frames: List[Tuple[int, torch.Tensor]] = []
    for f_idx in sorted(frame_buckets.keys()):
        ref_frames.append((f_idx, frame_buckets[f_idx][0]))

    if not ref_frames:
        raise RuntimeError("No reference masks available to build objective.")

    # Build per-frame masks (either computed or reused from fixed_mask).
    fixed_list = fixed_mask if isinstance(fixed_mask, list) else ([fixed_mask] if fixed_mask is not None else None)
    fixed_ref_list = (
        fixed_ref_logits if isinstance(fixed_ref_logits, list) else ([fixed_ref_logits] if fixed_ref_logits is not None else None)
    )
    masks_per_frame: List[torch.Tensor] = []
    ref_logits_per_frame: List[torch.Tensor] = []
    for idx, (frame_idx, ref_frame) in enumerate(ref_frames):
        lane_slice = ref_frame[_choose_lane(lane_idx, ref_frame)]
        if lane_slice.dim() == 3 and lane_slice.shape[0] == 1:
            lane_slice = lane_slice[0]
        if fixed_ref_list is not None and fixed_ref_list:
            src = fixed_ref_list[min(idx, len(fixed_ref_list) - 1)]
            ref_logits = src.detach()
        else:
            ref_logits = lane_slice.detach()
        ref_logits_per_frame.append(ref_logits)
        if fixed_list is not None and fixed_list:
            src = fixed_list[min(idx, len(fixed_list) - 1)]
            mask = src.to(device=lane_slice.device, dtype=lane_slice.dtype)
        else:
            prob_map = torch.sigmoid(ref_logits.to(device=lane_slice.device, dtype=lane_slice.dtype))
            mask = (prob_map > float(threshold)).to(dtype=prob_map.dtype)
        masks_per_frame.append(mask)

    if not masks_per_frame:
        raise RuntimeError("No masks constructed for objective.")

    mode = (objective_mode or "mask").strip().lower()
    # SAM2 uses -1024.0 as a sentinel for "no object" regions
    NO_OBJ_SCORE = -1024.0
    NO_OBJ_THRESHOLD = NO_OBJ_SCORE + 1.0  # values <= this are considered sentinel

    total = torch.zeros((), device=masks_per_frame[0].device, dtype=masks_per_frame[0].dtype)
    for idx, obj_mask in enumerate(masks_per_frame):
        frame_idx = ref_frames[idx][0]
        tensors_for_frame = frame_buckets.get(frame_idx, [])
        for t in tensors_for_frame:
            lane = _choose_lane(lane_idx, t)
            lane_slice = t[lane]
            if lane_slice.dim() == 3 and lane_slice.shape[0] == 1:
                lane_slice = lane_slice[0]

            # Filter out NO_OBJ_SCORE sentinel values (-1024)
            valid_mask = (lane_slice > NO_OBJ_THRESHOLD).to(dtype=obj_mask.dtype)
            combined_mask = obj_mask * valid_mask

            if mode in {"mask", "masked"}:
                total = total + (lane_slice.to(device=obj_mask.device, dtype=obj_mask.dtype) * combined_mask).sum()
            elif mode in {"output_positive", "mask_positive"}:
                total = total + (lane_slice.to(device=obj_mask.device, dtype=obj_mask.dtype) * combined_mask).clamp(min=0).sum()
            elif mode in {"logit_sum", "logitsum", "logits"}:
                total = total + (lane_slice.to(device=obj_mask.device, dtype=obj_mask.dtype) * valid_mask).sum()
            elif mode in {"logit_dot", "logitdot", "dot"}:
                ref_logits = ref_logits_per_frame[min(idx, len(ref_logits_per_frame) - 1)]
                ref_logits = ref_logits.to(device=lane_slice.device, dtype=lane_slice.dtype)
                ref_valid = (ref_logits > NO_OBJ_THRESHOLD).to(dtype=lane_slice.dtype)
                both_valid = valid_mask * ref_valid
                total = total + (lane_slice * ref_logits * both_valid).sum()
            elif mode in {"cos", "cosine", "cosine_similarity", "cosinesim"}:
                ref_logits = ref_logits_per_frame[min(idx, len(ref_logits_per_frame) - 1)]
                ref_logits = ref_logits.to(device=lane_slice.device, dtype=lane_slice.dtype)
                ref_valid = (ref_logits > NO_OBJ_THRESHOLD).to(dtype=obj_mask.dtype)
                both_valid = combined_mask * ref_valid
                masked_pred = lane_slice * both_valid
                masked_ref = ref_logits * both_valid
                flat_pred = masked_pred.reshape(-1)
                flat_ref = masked_ref.reshape(-1)
                if flat_pred.numel() == 0 or flat_ref.numel() == 0:
                    continue
                total = total + F.cosine_similarity(flat_pred, flat_ref, dim=0, eps=1e-8)
            else:
                raise ValueError(f"Unsupported objective_mode '{objective_mode}'")
    return total, masks_per_frame, ref_logits_per_frame


def _measure_objective_value(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    *,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    fixed_mask: Optional[List[torch.Tensor]] = None,
    fixed_ref_logits: Optional[List[torch.Tensor]] = None,
    objective_mode: str = "mask",
    mask_callback: Optional[Callable[[Dict[str, List[torch.Tensor]]], None]] = None,
    debug: bool = False,
    label: str = "",
    return_mask: bool = False,
    return_ref_logits: bool = False,
) -> Tuple[float, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    mask_capture.release_step_refs()
    mask_capture.clear_tapes()
    forward_runner(require_grad=False)
    mask_lists = mask_capture.get_tensor_lists(detach=True)
    if mask_callback is not None:
        try:
            mask_callback(mask_lists)
        except Exception as exc:
            _debug_print(f"[debug] mask_callback error: {exc}", enabled=debug)
    obj, obj_mask, ref_logits = _objective_from_masks(
        mask_lists,
        lane_idx,
        threshold,
        ref_spec=ref_mask_spec,
        fixed_mask=fixed_mask,
        fixed_ref_logits=fixed_ref_logits,
        objective_mode=objective_mode,
    )
    val = float(obj.detach().cpu().item())
    mask_capture.release_step_refs()
    _debug_print(
        f"[objective]{f'[{label}]' if label else ''} lane={lane_idx} value={val:.6f} masks={ {k: tuple(v[0].shape) if v else None for k,v in mask_lists.items()} }",
        enabled=debug,
    )
    detached_masks = [m.detach() for m in obj_mask] if (return_mask and obj_mask is not None) else None
    detached_ref = [r.detach() for r in ref_logits] if (return_ref_logits and ref_logits is not None) else None
    return val, detached_masks, detached_ref


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------
def _apply_mask(
    t: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    base: Optional[Any],
    *,
    baseline_cache: Optional[ActivationBaselineCache],
    spec: str,
    attr_name: str,
    frame_idx: Optional[int] = None,
) -> torch.Tensor:
    work_mask = mask.to(device=t.device)
    while work_mask.dim() < t.dim():
        work_mask = work_mask.unsqueeze(0)
    baseline_t: Optional[torch.Tensor] = None
    if baseline_cache is not None:
        baseline_t = baseline_cache.broadcast(spec, attr_name, t)
    if mode == "deletion":
        if baseline_t is None:
            return t * (~work_mask)
        return torch.where(work_mask, baseline_t, t)
    if mode == "insertion":
        base_sel = _select_base_for_frame(base, frame_idx)
        if base_sel is None:
            raise RuntimeError("Insertion requires a base tensor.")
        base_t = base_sel.to(device=t.device, dtype=t.dtype)
        while base_t.dim() < t.dim():
            base_t = base_t.unsqueeze(0)
        filler = baseline_t
        if filler is None:
            filler = torch.zeros_like(base_t)
        return torch.where(work_mask, base_t, filler)
    raise ValueError(mode)


def _select_base_for_frame(base: Optional[Any], frame_idx: Optional[int]) -> Optional[torch.Tensor]:
    if base is None:
        return None
    if torch.is_tensor(base):
        return base
    if isinstance(base, dict):
        if frame_idx is not None:
            candidate = base.get(int(frame_idx))
            if torch.is_tensor(candidate):
                return candidate
        for value in base.values():
            if torch.is_tensor(value):
                return value
    return None


def _base_tensor_for_mask(base: Optional[Any]) -> Optional[torch.Tensor]:
    if base is None:
        return None
    if torch.is_tensor(base):
        return base
    if isinstance(base, dict):
        tensors = [t for t in base.values() if torch.is_tensor(t)]
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=0)
    return None


def _base_feat_dim(base: Optional[Any]) -> int:
    base_tensor = _base_tensor_for_mask(base)
    if base_tensor is None:
        return 0
    return int(base_tensor.shape[-1])


def _describe_base_shapes(bases: Dict[str, Any]) -> Dict[str, Any]:
    desc: Dict[str, Any] = {}
    for spec, base in bases.items():
        if torch.is_tensor(base):
            desc[spec] = tuple(base.shape)
            continue
        if isinstance(base, dict):
            frames: Dict[int, Any] = {}
            for key, value in base.items():
                if torch.is_tensor(value):
                    frames[int(key)] = tuple(value.shape)
            desc[spec] = frames if frames else "dict"
            continue
        desc[spec] = type(base).__name__
    return desc


def _frame_map_from_tape(tape, *, detach: bool, to_cpu: bool) -> Dict[int, torch.Tensor]:
    frame_map: Dict[int, torch.Tensor] = {}
    for rec in tape.frames():
        tensor = rec.tensor.detach() if detach else rec.tensor
        if to_cpu:
            tensor = tensor.cpu()
        frame_map[int(rec.frame_idx)] = tensor
    return frame_map


def _capture_anchor_bases(anchors: Dict[str, AnchorInfo]) -> Dict[str, Any]:
    bases: Dict[str, Any] = {}
    for spec, anchor in anchors.items():
        anchor.branch.clear_anchor_overrides()
        tape = None
        if hasattr(anchor.branch, "sae_tape"):
            try:
                tape = anchor.branch.sae_tape(anchor.attr_name)
            except Exception:
                tape = None
        if tape is not None and getattr(tape, "frame_count", None) is not None and tape.frame_count() > 0:
            frame_map = _frame_map_from_tape(tape, detach=True, to_cpu=True)
            if frame_map:
                bases[spec] = frame_map
                continue
        out = anchor.branch.sae_context().get(anchor.attr_name)
        if torch.is_tensor(out):
            bases[spec] = out.detach().cpu()
    return bases


def _apply_feature_drop(
    t: torch.Tensor,
    mask_1d: torch.Tensor,
    *,
    baseline_cache: Optional[ActivationBaselineCache],
    spec: str,
    attr_name: str,
) -> torch.Tensor:
    work_mask = mask_1d.to(device=t.device)
    while work_mask.dim() < t.dim():
        work_mask = work_mask.unsqueeze(0)
    baseline_t: Optional[torch.Tensor] = None
    if baseline_cache is not None:
        baseline_t = baseline_cache.broadcast(spec, attr_name, t)
    if baseline_t is None:
        return t * (~work_mask)
    return torch.where(work_mask, baseline_t, t)


def _reset_anchor_controllers(anchors: Dict[str, AnchorInfo], *, debug: bool = False, reason: str = "") -> None:
    _debug_print(f"[debug] reset anchor controllers{f' ({reason})' if reason else ''}", enabled=debug)
    for anchor in anchors.values():
        ctrl = anchor.controller
        if ctrl is not None:
            ctrl.release_cached_activations()
            ctrl.clear_override()
            # Also clear frame/index bookkeeping to aggressively drop stale refs.
            try:
                ctrl.clear()
            except Exception:
                pass
        anchor.branch.clear_anchor_overrides()


def _clear_anchor_contexts(anchors: Dict[str, AnchorInfo], *, debug: bool = False, reason: str = "") -> None:
    _debug_print(f"[debug] clear anchor contexts{f' ({reason})' if reason else ''}", enabled=debug)
    for anchor in anchors.values():
        anchor.branch.clear_context()
    _reset_anchor_controllers(anchors, debug=debug, reason=reason or "clear-anchor-contexts")


def _gpu_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _select_active_mask(base: Any, threshold: float, max_features: Optional[int]) -> torch.Tensor:
    base_tensor = _base_tensor_for_mask(base)
    if base_tensor is None:
        return torch.zeros(0, dtype=torch.bool)
    flat = base_tensor.detach().abs().reshape(-1, base_tensor.shape[-1]).sum(dim=0)
    if threshold <= 0:
        mask = flat > 0
    else:
        mask = flat > float(threshold)
    if max_features is not None and mask.sum().item() > max_features:
        values = flat * mask.float()
        topk = torch.topk(values, k=max_features)
        limited = torch.zeros_like(mask, dtype=torch.bool)
        limited[topk.indices] = True
        mask = limited
    return mask


def _sanitize_path_token(tok: str) -> str:
    safe = []
    for ch in tok:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _sample_random_bvd_rows(
    *,
    meta_ledger: OfflineMetaParquetLedger,
    n: int,
    seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    ds_samples, _ = meta_ledger.as_dataset()
    table = ds_samples.to_table(columns=["sample_id"])
    sample_ids = table.column("sample_id").to_pylist() if table is not None else []
    if not sample_ids:
        raise RuntimeError("No samples available for random BVD selection.")
    rng = random.Random(int(seed))
    if n <= 0:
        return []
    if n >= len(sample_ids):
        rng.shuffle(sample_ids)
        chosen_ids = list(sample_ids)
    else:
        chosen_ids = rng.sample(sample_ids, k=int(n))

    rows_all: List[Tuple[int, Dict[str, Any]]] = []
    for sample_id in chosen_ids:
        tbl_sample = meta_ledger.find_sample(int(sample_id))
        rows = tbl_sample.to_pylist() if tbl_sample is not None else []
        if not rows:
            continue
        sample_row = rows[0]
        prompt_sets = sample_row.get("prompt_sets") or []
        frame_idx = 0
        prompt_id = -1
        if prompt_sets:
            choice = rng.choice(prompt_sets)
            try:
                frame_idx = int(choice.get("frame_idx", 0))
            except Exception:
                frame_idx = 0
            try:
                prompt_id = int(choice.get("prompt_id", -1))
            except Exception:
                prompt_id = -1
        uid = -1
        if prompt_id >= 0:
            tbl_prompt = meta_ledger.find_prompts(int(sample_id), int(prompt_id), int(frame_idx))
            prompt_rows = tbl_prompt.to_pylist() if tbl_prompt is not None else []
            if prompt_rows:
                uid = int(rng.choice(prompt_rows).get("uid", -1))
        row = {
            "sample_id": int(sample_id),
            "frame_idx": int(frame_idx),
            "prompt_id": int(prompt_id),
            "uid": int(uid),
        }
        rows_all.append((-1, row))
    return rows_all


def _tensor_hw_slice(t: torch.Tensor) -> Optional[torch.Tensor]:
    data = t
    while data.dim() > 2 and data.shape[0] == 1:
        data = data[0]
    if data.dim() == 3 and data.shape[0] > 1:
        data = data[0]
    if data.dim() != 2:
        try:
            h, w = data.shape[-2], data.shape[-1]
            data = data.reshape(h, w)
        except Exception:
            return None
    return data


def _to_uint8_scaled(arr: np.ndarray) -> np.ndarray:
    work = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(work)
    if not finite.any():
        return np.zeros_like(work, dtype=np.uint8)
    vmin = work[finite].min()
    vmax = work[finite].max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(work, dtype=np.uint8)
    norm = (work - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _dump_mask_images(
    mask_lists: Dict[str, List[torch.Tensor]],
    cfg: MaskDumpConfig,
    *,
    step_idx: int,
    label: str,
    removed: int,
    total: int,
    lane_idx: Optional[int],
    threshold: float,
) -> None:
    lane_for_path = lane_idx if lane_idx is not None else 0
    base_dir = cfg.root
    if cfg.decile is not None:
        base_dir = base_dir / f"decile_{cfg.decile}"
    sample_dir = base_dir / f"sample_{cfg.sample_id}_frame_{cfg.frame_idx}_lane_{lane_for_path}"
    method_dir = sample_dir / f"{cfg.method}_{cfg.curve}"
    method_dir.mkdir(parents=True, exist_ok=True)

    for spec, tensors in mask_lists.items():
        if not tensors:
            continue
        spec_safe = _sanitize_path_token(spec)
        frame_from_spec = _infer_frame_idx(spec)
        for idx, tensor in enumerate(tensors):
            if not torch.is_tensor(tensor):
                continue
            lane = _choose_lane(lane_idx, tensor)
            lane_slice = tensor[lane]
            hw = _tensor_hw_slice(lane_slice)
            if hw is None:
                continue
            logits = hw.detach().cpu().float()
            probs = torch.sigmoid(logits)
            mask = (probs > float(threshold)).to(dtype=torch.uint8)

            logits_img = Image.fromarray(_to_uint8_scaled(logits.numpy()))
            probs_img = Image.fromarray(np.clip(np.nan_to_num(probs.numpy(), nan=0.0) * 255.0, 0, 255).astype(np.uint8))
            mask_img = Image.fromarray((mask.numpy() * 255).astype(np.uint8))

            stem = f"step{step_idx:04d}_rm{removed:04d}_of{total:04d}_{label}_{spec_safe}_f{frame_from_spec}_i{idx}"
            logits_img.save(method_dir / f"{stem}_logit.png")
            probs_img.save(method_dir / f"{stem}_prob.png")
            mask_img.save(method_dir / f"{stem}_mask.png")


# ---------------------------------------------------------------------------
# Curves and rankings
# ---------------------------------------------------------------------------
def _run_objective_curve(
    mode: str,
    *,
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    objective_mode: str = "mask",
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    anchors: Dict[str, AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[ActivationBaselineCache],
    ranking: List[Tuple[str, int, float]],
    chunk_size: int,
    max_steps: Optional[int],
    active_masks: Optional[Dict[str, torch.Tensor]],
    norm_ref: float,
    mask_dump: Optional[MaskDumpConfig] = None,
    debug: bool = False,
    curve_label: str = "",
) -> List[Tuple[float, float]]:
    masks: Dict[str, torch.Tensor] = {}
    for spec, base in bases.items():
        feat_dim = _base_feat_dim(base)
        if feat_dim <= 0:
            continue
        masks[spec] = torch.zeros(feat_dim, dtype=torch.bool)
    results: List[Tuple[float, float]] = []
    removed = 0
    last_norm = 1.0
    if active_masks:
        filtered: List[Tuple[str, int, float]] = []
        skipped = 0
        for spec, feat_idx, score in ranking:
            mask = active_masks.get(spec)
            if mask is None:
                filtered.append((spec, feat_idx, score))
                continue
            if feat_idx < mask.numel() and bool(mask[feat_idx].item()):
                filtered.append((spec, feat_idx, score))
            else:
                skipped += 1
        if debug and skipped:
            _debug_print(
                f"[debug] filtered inactive features for curve '{curve_label or mode}': kept={len(filtered)} skipped={skipped}",
                enabled=debug,
            )
        ranking = filtered
    total = len(ranking)
    curve_name = curve_label or mode
    measure_idx = 0
    _debug_print(
        f"[debug] start {mode} curve '{curve_name}': total_features={total}, chunk_size={chunk_size}, max_steps={max_steps}",
        enabled=debug,
    )
    if max_steps is not None and total > 0:
        max_covered = min(int(total), int(max_steps) * int(chunk_size))
        if max_covered < int(total):
            covered_pct = (float(max_covered) / float(total)) * 100.0
            _log_line(
                f"[curve][{curve_name}] max-steps={int(max_steps)} chunk-size={int(chunk_size)} "
                f"limits coverage to {max_covered}/{total} ({covered_pct:.1f}%) features"
            )

    def _set_overrides():
        for spec, anchor in anchors.items():
            mask = masks.get(spec)
            base = bases.get(spec)
            if mask is None or base is None:
                continue
            if mode == "deletion":
                anchor.branch.set_anchor_override(
                    anchor.attr_name,
                    lambda t, m=mask, s=spec, a=anchor.attr_name, br=anchor.branch: _apply_mask(
                        t,
                        m,
                        "deletion",
                        base=None,
                        baseline_cache=baseline_cache,
                        spec=s,
                        attr_name=a,
                        frame_idx=br.current_frame_idx(),
                    ),
                )
            else:
                anchor.branch.set_anchor_override(
                    anchor.attr_name,
                    lambda t, m=mask, b=base, s=spec, a=anchor.attr_name, br=anchor.branch: _apply_mask(
                        t,
                        m,
                        "insertion",
                        base=b,
                        baseline_cache=baseline_cache,
                        spec=s,
                        attr_name=a,
                        frame_idx=br.current_frame_idx(),
                    ),
                )

    def _measure(label: str):
        nonlocal measure_idx

        def _mask_cb(mask_lists: Dict[str, List[torch.Tensor]]):
            if mask_dump is None or mode != "insertion":
                return
            try:
                _dump_mask_images(
                    mask_lists,
                    mask_dump,
                    step_idx=measure_idx,
                    label=label,
                    removed=removed,
                    total=total,
                    lane_idx=lane_idx,
                    threshold=threshold,
                )
            except Exception as exc:
                _debug_print(f"[debug] mask dump failed ({label}): {exc}", enabled=debug)

        val, _, _ = _measure_objective_value(
            forward_runner,
            mask_capture,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            fixed_mask=objective_mask,
            fixed_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            mask_callback=_mask_cb,
            debug=debug,
            label=f"{curve_name}:{label}",
        )
        norm = val / (norm_ref if norm_ref not in (None, 0.0) else 1e-12)
        clamped = float(max(0.0, min(norm, 1.0)))
        results.append((removed / len(ranking) * 100.0 if ranking else 0.0, clamped))
        _debug_print(
            f"[objective][{curve_name}][{label}] removed={removed}/{total} obj={val:.6f} norm={norm:.4f} clamped={clamped:.4f}",
            enabled=debug or label.endswith("start"),
        )
        nonlocal last_norm
        last_norm = clamped
        measure_idx += 1
        _reset_anchor_controllers(anchors, debug=debug, reason=f"{curve_name}-{label}")
        _gpu_gc()

    _set_overrides()
    _measure(f"{mode}-start")

    idx = 0
    step_limit = max_steps if max_steps is not None else (total + chunk_size - 1) // chunk_size
    while idx < total and (max_steps is None or len(results) - 1 < step_limit):
        changed = False
        for _ in range(chunk_size):
            if idx >= total:
                break
            spec, feat_idx, _score = ranking[idx]
            removed += 1
            active_mask = active_masks.get(spec) if active_masks else None
            is_active = True
            if active_mask is not None and feat_idx < active_mask.numel():
                is_active = bool(active_mask[feat_idx].item())
            if spec in masks and feat_idx < masks[spec].numel() and is_active:
                masks[spec][feat_idx] = True
                changed = True
            idx += 1
        if changed:
            _set_overrides()
            _measure(mode)
        else:
            results.append((removed / len(ranking) * 100.0 if ranking else 0.0, last_norm))

    _reset_anchor_controllers(anchors, debug=debug, reason=f"{curve_name}-{mode}-end")
    _clear_anchor_contexts(anchors, debug=debug, reason=f"{curve_name}-{mode}-end")
    _gpu_gc()
    return results


def _stack_from_tape(tape, *, detach: bool) -> Tuple[Optional[torch.Tensor], Dict[int, int]]:
    if tape is None or tape.frame_count() == 0:
        return None, {}
    frame_map: Dict[int, torch.Tensor] = {}
    for rec in tape.frames():
        tensor = rec.tensor.detach() if detach else rec.tensor
        frame_map[int(rec.frame_idx)] = tensor
    if not frame_map:
        return None, {}
    frame_ids = sorted(frame_map.keys())
    stack = torch.stack([frame_map[fid] for fid in frame_ids], dim=0)
    index_map = {fid: idx for idx, fid in enumerate(frame_ids)}
    return stack, index_map


def _collect_anchor_stack(
    anchor: AnchorInfo,
    forward_runner: Optional[Callable[[bool], Any]],
) -> Tuple[Optional[torch.Tensor], Dict[int, int]]:
    tape = None
    if hasattr(anchor.branch, "sae_tape"):
        try:
            tape = anchor.branch.sae_tape(anchor.attr_name)
        except Exception:
            tape = None
    if tape is None or tape.frame_count() == 0:
        if forward_runner is not None:
            with torch.no_grad():
                forward_runner(require_grad=False)
        if hasattr(anchor.branch, "sae_tape"):
            try:
                tape = anchor.branch.sae_tape(anchor.attr_name)
            except Exception:
                tape = None
    if tape is not None and tape.frame_count() > 0:
        stack, index_map = _stack_from_tape(tape, detach=False)
        if stack is not None:
            return stack, index_map
    out = anchor.branch.sae_context().get(anchor.attr_name)
    if torch.is_tensor(out):
        return out, {}
    return None, {}


def _select_frame_slice(stack: torch.Tensor, frame_map: Dict[int, int], frame_idx: Optional[int]) -> torch.Tensor:
    idx = frame_map.get(int(frame_idx), 0) if frame_idx is not None else 0
    if idx < 0 or idx >= stack.shape[0]:
        idx = 0
    return stack[idx]


def _build_grad_ranking(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    *,
    objective_mode: str = "mask",
    use_abs: bool = False,
    use_positive: bool = False,
) -> List[Tuple[str, int, float]]:
    ranking: List[Tuple[str, int, float]] = []
    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()
        anchor.branch.clear_context()

    param_states: List[Tuple[torch.nn.Parameter, bool]] = []
    for anchor in anchors.values():
        for param in anchor.branch.sae.parameters():
            param_states.append((param, param.requires_grad))
            param.requires_grad_(True)

    try:
        mask_capture.release_step_refs()
        mask_capture.clear_tapes()
        forward_runner(require_grad=True)
        mask_lists = mask_capture.get_tensor_lists(detach=False)
        objective, _, _ = _objective_from_masks(
            mask_lists,
            lane_idx,
            threshold,
            ref_spec=ref_mask_spec,
            fixed_mask=objective_mask,
            fixed_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
        )
        tape_tensors: List[torch.Tensor] = []
        tape_specs: List[str] = []
        for spec, anchor in anchors.items():
            tape = anchor.branch.sae_tape(anchor.attr_name)
            if tape is None or tape.frame_count() == 0:
                continue
            for rec in tape.frames():
                if not torch.is_tensor(rec.tensor):
                    continue
                tape_tensors.append(rec.tensor)
                tape_specs.append(spec)
        if not tape_tensors:
            return []
        grads = torch.autograd.grad(
            objective,
            tape_tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        scores_by_spec: Dict[str, torch.Tensor] = {}
        for spec, grad in zip(tape_specs, grads):
            if grad is None:
                continue
            flat = grad.reshape(-1, grad.shape[-1])
            if use_positive:
                flat = flat.clamp(min=0)
            elif use_abs:
                flat = flat.abs()
            score = flat.sum(dim=0)
            if spec in scores_by_spec:
                scores_by_spec[spec] = scores_by_spec[spec] + score
            else:
                scores_by_spec[spec] = score
        for spec, score in scores_by_spec.items():
            for idx, val in enumerate(score):
                ranking.append((spec, idx, float(val.item())))
    finally:
        for param, prev in param_states:
            param.requires_grad_(prev)
        mask_capture.release_step_refs()
        _reset_anchor_controllers(anchors, reason="grad-tape")
        _gpu_gc()

    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


def _build_inputxgrad_ranking(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    *,
    objective_mode: str = "mask",
    use_abs: bool = False,
    use_positive: bool = False,
) -> List[Tuple[str, int, float]]:
    ranking: List[Tuple[str, int, float]] = []
    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()
        anchor.branch.clear_context()

    param_states: List[Tuple[torch.nn.Parameter, bool]] = []
    for anchor in anchors.values():
        for param in anchor.branch.sae.parameters():
            param_states.append((param, param.requires_grad))
            param.requires_grad_(True)

    try:
        mask_capture.release_step_refs()
        mask_capture.clear_tapes()
        forward_runner(require_grad=True)
        mask_lists = mask_capture.get_tensor_lists(detach=False)
        objective, _, _ = _objective_from_masks(
            mask_lists,
            lane_idx,
            threshold,
            ref_spec=ref_mask_spec,
            fixed_mask=objective_mask,
            fixed_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
        )
        tape_tensors: List[torch.Tensor] = []
        tape_specs: List[str] = []
        for spec, anchor in anchors.items():
            tape = anchor.branch.sae_tape(anchor.attr_name)
            if tape is None or tape.frame_count() == 0:
                continue
            for rec in tape.frames():
                if not torch.is_tensor(rec.tensor):
                    continue
                tape_tensors.append(rec.tensor)
                tape_specs.append(spec)
        if not tape_tensors:
            return []
        grads = torch.autograd.grad(
            objective,
            tape_tensors,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        scores_by_spec: Dict[str, torch.Tensor] = {}
        for (spec, acts), grad in zip(zip(tape_specs, tape_tensors), grads):
            if grad is None:
                continue
            if use_abs:
                contrib = (acts.detach() * grad.abs()).reshape(-1, acts.shape[-1])
            else:
                contrib = (acts.detach() * grad).reshape(-1, acts.shape[-1])
                if use_positive:
                    contrib = contrib.clamp(min=0)
            contrib = contrib.sum(dim=0)
            if spec in scores_by_spec:
                scores_by_spec[spec] = scores_by_spec[spec] + contrib
            else:
                scores_by_spec[spec] = contrib
        for spec, score in scores_by_spec.items():
            for idx, val in enumerate(score):
                ranking.append((spec, idx, float(val.item())))
    finally:
        for param, prev in param_states:
            param.requires_grad_(prev)
        mask_capture.release_step_refs()
        _reset_anchor_controllers(anchors, reason="inputxgrad-tape")
        _gpu_gc()

    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


def _noise_grad_scores(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    *,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    objective_mode: str = "mask",
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    use_abs: bool = False,
    use_positive: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute per-feature |grad| scores for a single noisy sample."""
    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()

    acts_cache: Dict[str, torch.Tensor] = {}
    frame_maps: Dict[str, Dict[int, int]] = {}
    for spec, anchor in anchors.items():
        stack, frame_map = _collect_anchor_stack(anchor, forward_runner)
        if stack is None:
            continue
        if not frame_map:
            stack = stack.unsqueeze(0)
        stack = stack.detach().requires_grad_(True)
        acts_cache[spec] = stack
        frame_maps[spec] = frame_map
        anchor.branch.set_anchor_override(
            anchor.attr_name,
            lambda _t, a=stack, m=frame_map, br=anchor.branch: _select_frame_slice(a, m, br.current_frame_idx()),
        )

    scores: Dict[str, torch.Tensor] = {}
    if not acts_cache:
        return scores

    mask_capture.release_step_refs()
    mask_capture.clear_tapes()
    forward_runner(require_grad=True)
    mask_lists = mask_capture.get_tensor_lists(detach=False)
    objective, _, _ = _objective_from_masks(
        mask_lists,
        lane_idx,
        threshold,
        ref_spec=ref_mask_spec,
        fixed_mask=objective_mask,
        fixed_ref_logits=objective_ref_logits,
        objective_mode=objective_mode,
    )
    objective.backward()

    for spec, acts in acts_cache.items():
        if acts.grad is None:
            continue
        grad = acts.grad.detach()
        flat = grad.reshape(-1, grad.shape[-1])
        if use_positive:
            flat = flat.clamp(min=0)
        elif use_abs:
            flat = flat.abs()
        score = flat.sum(dim=0)
        scores[spec] = score

    mask_capture.release_step_refs()
    _reset_anchor_controllers(anchors, reason="smoothgrad-sample")
    _gpu_gc()
    return scores


def _build_smooth_like_ranking(
    mode: str,
    *,
    batch_on_dev: Any,
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    objective_mode: str = "mask",
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    samples: int,
    noise_sigma: float,
    use_abs: bool = False,
    use_positive: bool = False,
) -> List[Tuple[str, int, float]]:
    samples = max(1, int(samples))
    noise_sigma = float(noise_sigma)

    base_img = getattr(batch_on_dev, "img_batch", None)
    if not torch.is_tensor(base_img):
        raise RuntimeError("SmoothGrad requires batch_on_dev.img_batch to be a tensor.")
    base_img = base_img.detach()

    base_std = float(base_img.float().std().item()) if base_img.numel() > 1 else 1.0
    scale = noise_sigma * (base_std if base_std > 0 else 1.0)

    per_spec_scores: Dict[str, List[torch.Tensor]] = {}
    for _ in range(samples):
        noise = torch.randn_like(base_img, dtype=torch.float32) * scale
        noise = noise.to(device=base_img.device, dtype=base_img.dtype)
        try:
            batch_on_dev.img_batch = base_img + noise
            scores = _noise_grad_scores(
                forward_runner,
                mask_capture,
                anchors,
                lane_idx=lane_idx,
                threshold=threshold,
                ref_mask_spec=ref_mask_spec,
                objective_mask=objective_mask,
                objective_mode=objective_mode,
                objective_ref_logits=objective_ref_logits,
                use_abs=use_abs,
                use_positive=use_positive,
            )
        finally:
            batch_on_dev.img_batch = base_img
        for spec, score in scores.items():
            per_spec_scores.setdefault(spec, []).append(score.detach().cpu())

    ranking: List[Tuple[str, int, float]] = []
    for spec, stacks in per_spec_scores.items():
        if not stacks:
            continue
        stacked = torch.stack(stacks, dim=0)
        if mode == "smoothgrad":
            agg = stacked.mean(dim=0)
        elif mode == "vargard":
            agg = stacked.var(dim=0, unbiased=False)
        else:
            raise ValueError(mode)
        for idx, val in enumerate(agg):
            ranking.append((spec, idx, float(val.item())))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


def _build_ig_ranking(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    steps: int,
    *,
    objective_mode: str = "mask",
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    use_abs: bool = False,
    use_positive: bool = False,
    ig_active: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int, float]]:
    del ig_active
    ranking: List[Tuple[str, int, float]] = []
    steps = max(1, int(steps))
    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()
    with torch.no_grad():
        forward_runner(require_grad=False)
    for spec, anchor in anchors.items():
        stack, frame_map = _collect_anchor_stack(anchor, forward_runner=None)
        if stack is None:
            continue
        if not frame_map:
            stack = stack.unsqueeze(0)
        base = stack.detach()
        baseline = torch.zeros_like(base)
        total = torch.zeros_like(base, device=base.device, dtype=base.dtype)
        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            scaled = baseline + (alpha * (base - baseline))
            scaled = scaled.detach().clone().requires_grad_(True)
            anchor.branch.set_anchor_override(
                anchor.attr_name,
                lambda _t, a=scaled, m=frame_map, br=anchor.branch: _select_frame_slice(a, m, br.current_frame_idx()),
            )
            mask_capture.release_step_refs()
            mask_capture.clear_tapes()
            forward_runner(require_grad=True)
            mask_lists = mask_capture.get_tensor_lists(detach=False)
            objective, _, _ = _objective_from_masks(
                mask_lists,
                lane_idx,
                threshold,
                ref_spec=ref_mask_spec,
                fixed_mask=objective_mask,
                fixed_ref_logits=objective_ref_logits,
                objective_mode=objective_mode,
            )
            grad = torch.autograd.grad(objective, scaled, retain_graph=False, create_graph=False)[0]
            total = total + grad * (base - baseline)
            _reset_anchor_controllers(anchors, reason="ig-step")
            _gpu_gc()
        anchor.branch.clear_anchor_overrides()
        attribution = total / float(steps)
        flat = attribution.detach().reshape(-1, attribution.shape[-1])
        if use_positive:
            flat = flat.clamp(min=0)
        elif use_abs:
            flat = flat.abs()
        score = flat.sum(dim=0)
        for idx, val in enumerate(score):
            ranking.append((spec, idx, float(val.item())))
    mask_capture.release_step_refs()
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


def _build_ig_anchor_ranking(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Optional[List[torch.Tensor]],
    steps: int,
    *,
    objective_mode: str = "mask",
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    use_abs: bool = False,
    use_positive: bool = False,
    ig_active: Optional[Sequence[str]] = None,
) -> List[Tuple[str, int, float]]:
    ranking: List[Tuple[str, int, float]] = []
    steps = max(1, int(steps))
    ig_active_list = [str(v) for v in ig_active] if ig_active else []
    active_flags = {spec: _is_ig_active(spec, ig_active_list) for spec in anchors.keys()}
    inactive_specs = [spec for spec, active in active_flags.items() if not active]

    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()
        anchor.branch.clear_ig_attr_alpha()
        anchor.branch.clear_context()
        if hasattr(anchor.controller, "clear_override"):
            anchor.controller.clear_override()

    baseline_map: Dict[str, torch.Tensor] = {}
    if inactive_specs:
        # Keep inactive anchors fixed at their target values so delta=0.
        for anchor in anchors.values():
            anchor.branch.clear_ig_attr_alpha()
        _clear_anchor_contexts(anchors, reason="ig-baseline")
        mask_capture.release_step_refs()
        mask_capture.clear_tapes()
        forward_runner(require_grad=False)
        for spec in inactive_specs:
            anchor = anchors.get(spec)
            if anchor is None:
                continue
            stack, _frame_map = _collect_anchor_stack(anchor, forward_runner=None)
            if stack is None:
                continue
            baseline_map[spec] = stack.detach().clone()
        _clear_anchor_contexts(anchors, reason="ig-baseline")
        mask_capture.release_step_refs()
        mask_capture.clear_tapes()

    branch_specs: Dict[Any, List[Tuple[str, str]]] = {}
    for spec, anchor in anchors.items():
        branch_specs.setdefault(anchor.branch, []).append((spec, anchor.attr_name))

    def _set_alpha(alpha: float) -> None:
        # Apply IG alpha directly to live SAE attrs (no cached replacement).
        for branch, specs in branch_specs.items():
            attrs: set[str] = set()
            for spec, attr_name in specs:
                if not active_flags.get(spec, True):
                    continue
                attrs.add(attr_name)
            if attrs:
                branch.set_ig_attr_alpha(
                    alpha=alpha,
                    active_attrs=attrs,
                    baselines=None,
                    logical_base=None,
                )
            else:
                branch.clear_ig_attr_alpha()

    def _anchor_tensors_getter() -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for spec, anchor in anchors.items():
            stack, _frame_map = _collect_anchor_stack(anchor, forward_runner=None)
            if stack is None:
                continue
            out[spec] = stack
        return out

    def _objective_getter() -> torch.Tensor:
        mask_lists = mask_capture.get_tensor_lists(detach=False)
        objective, _, _ = _objective_from_masks(
            mask_lists,
            lane_idx,
            threshold,
            ref_spec=ref_mask_spec,
            fixed_mask=objective_mask,
            fixed_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
        )
        return objective

    def _do_forward(require_grad: bool = True) -> None:
        forward_runner(require_grad=bool(require_grad))

    def _release_step_refs() -> None:
        mask_capture.release_step_refs()
        mask_capture.clear_tapes()
        _clear_anchor_contexts(anchors, reason="ig-step")
        _gpu_gc()

    _clear_anchor_contexts(anchors, reason="ig-start")
    mask_capture.release_step_refs()
    mask_capture.clear_tapes()

    backend = GRADIENT_BACKENDS["ig"](
        anchor_tensors_getter=_anchor_tensors_getter,
        objective_getter=_objective_getter,
        steps=steps,
        set_alpha=_set_alpha,
        do_forward=_do_forward,
        release_step_refs=_release_step_refs,
        anchor_baselines=baseline_map,
        allow_missing_grad=True,
    )
    try:
        contrib = backend()
    finally:
        for anchor in anchors.values():
            anchor.branch.clear_ig_attr_alpha()
        _clear_anchor_contexts(anchors, reason="ig-end")
        mask_capture.release_step_refs()
        mask_capture.clear_tapes()
        _gpu_gc()

    for spec, tensor in contrib.items():
        if not torch.is_tensor(tensor):
            continue
        flat = tensor.detach().reshape(-1, tensor.shape[-1])
        if use_positive:
            flat = flat.clamp(min=0)
        elif use_abs:
            flat = flat.abs()
        score = flat.sum(dim=0)
        for idx, val in enumerate(score):
            ranking.append((spec, idx, float(val.item())))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking


def _build_activation_patching_ranking(
    forward_runner: Callable[[bool], Any],
    mask_capture: MultiAnchorCapture,
    anchors: Dict[str, AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[ActivationBaselineCache],
    lane_idx: Optional[int],
    mask_threshold: float,
    feature_threshold: float,
    ref_mask_spec: Optional[str],
    max_features: Optional[int],
    *,
    objective_mode: str = "mask",
    objective_ref_logits: Optional[List[torch.Tensor]] = None,
    base_objective: Optional[float] = None,
    objective_mask: Optional[List[torch.Tensor]] = None,
    progress: bool = False,
) -> Tuple[List[Tuple[str, int, float]], Dict[str, torch.Tensor]]:
    scores: Dict[str, torch.Tensor] = {}
    active_masks: Dict[str, torch.Tensor] = {}
    base_val = base_objective
    if base_val is None:
        base_val, _, _ = _measure_objective_value(
            forward_runner,
            mask_capture,
            lane_idx=lane_idx,
            threshold=mask_threshold,
            ref_mask_spec=ref_mask_spec,
            fixed_mask=objective_mask,
            fixed_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
        )
    _reset_anchor_controllers(anchors, reason="activation-patch-base")
    for spec, anchor in anchors.items():
        base = bases.get(spec)
        if base is None:
            continue
        mask = _select_active_mask(base, feature_threshold, max_features)
        active_masks[spec] = mask
        active_count = int(mask.sum().item())
        if active_count == 0:
            continue
        feat_dim = _base_feat_dim(base)
        if feat_dim <= 0:
            continue
        feat_scores = torch.zeros(feat_dim, dtype=torch.float32)
        sae_param = next(anchor.branch.sae.parameters(), None)
        base_tensor = _base_tensor_for_mask(base)
        if base_tensor is None:
            continue
        device = sae_param.device if sae_param is not None else base_tensor.device
        feature_mask = torch.zeros(feat_dim, dtype=torch.bool, device=device)
        measured = 0
        pct_step = 10 if active_count > 20 else 1
        last_pct = -1
        for idx_val in range(feat_dim):
            if not mask[idx_val]:
                continue
            feature_mask.zero_()
            feature_mask[int(idx_val)] = True
            anchor.branch.set_anchor_override(
                anchor.attr_name,
                lambda t, m=feature_mask, s=spec, a=anchor.attr_name: _apply_feature_drop(
                    t,
                    m,
                    baseline_cache=baseline_cache,
                    spec=s,
                    attr_name=a,
                ),
            )
            val, _, _ = _measure_objective_value(
                forward_runner,
                mask_capture,
                lane_idx=lane_idx,
                threshold=mask_threshold,
                ref_mask_spec=ref_mask_spec,
                fixed_mask=objective_mask,
                fixed_ref_logits=objective_ref_logits,
                objective_mode=objective_mode,
            )
            feat_scores[int(idx_val)] = float((base_val - val))
            measured += 1
            if progress and active_count > 0:
                pct = int(measured * 100 / active_count)
                if pct >= last_pct + pct_step or measured == active_count:
                    _log_line(f"[activation_patch][{spec}] measured {measured}/{active_count} ({pct}%)")
                    last_pct = pct
            _reset_anchor_controllers(anchors, reason="activation-patch-iter")
            _gpu_gc()
        anchor.branch.clear_anchor_overrides()
        scores[spec] = feat_scores
    _clear_anchor_contexts(anchors, reason="activation-patch-end")
    ranking: List[Tuple[str, int, float]] = []
    for spec, tensor in scores.items():
        for idx, val in enumerate(tensor):
            ranking.append((spec, idx, float(val.item())))
    ranking.sort(key=lambda x: x[2], reverse=True)
    return ranking, active_masks


def _compute_auc(curve: List[Tuple[float, float]]) -> float:
    if not curve:
        return 0.0
    xs = np.array([pt for pt, _ in curve], dtype=float) / 100.0
    ys = np.array([val for _, val in curve], dtype=float)
    if xs.shape[0] < 2:
        return float(ys.mean())
    return float(np.trapz(ys, xs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SAMv2 mask objectives via SAE feature ablations.")
    parser.add_argument("--attr-config", type=Path, default=Path("configs/sam2_attr_index_v2.yaml"))
    parser.add_argument("--anchors", type=str, default='model.sam_mask_decoder.transformer.layers.0@1', help="Anchor layer specs (comma-separated). Defaults to the attr config layer.")
    parser.add_argument("--mask-specs", type=str, default="model@0@pred_masks_high_res,model@1@pred_masks_high_res,model@2@pred_masks_high_res,model@3@pred_masks_high_res")
    parser.add_argument("--mask-threshold", type=float, default=0.2, help="Probability threshold for the objective mask.")
    parser.add_argument(
        "--feature-active-threshold",
        type=float,
        default=1e-6,
        help="Threshold for marking SAE features as active in activation_patch (applied to summed |activation|).",
    )
    parser.add_argument("--dump-insertion-masks-dir", type=Path, default=None, help="If set, save per-step insertion logits/prob/mask images to this directory.")
    parser.add_argument("--ref-mask-spec", type=str, default=None, help="Optional mask spec to derive the objective mask from.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stop-after", type=int, default=None, help="Stop after N rows overall.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Batch size for scoring/curves.")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit deletion/insertion steps (None = full).")
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument(
        "--ig-active",
        "--ig-anchor",
        dest="ig_active",
        type=str,
        default=None,
        help="Comma-separated specs to apply IG alpha to (default: anchors).",
    )
    parser.add_argument("--smoothgrad-samples", type=int, default=8, help="Number of noise samples for smoothgrad/vargard.")
    parser.add_argument("--smoothgrad-sigma", type=float, default=0.01, help="Noise scale (multiplied by input std) for smoothgrad/vargard.")
    parser.add_argument("--max-features", type=int, default=None, help="Limit active features considered in activation patching.")
    parser.add_argument("--rank-abs", action="store_true", help="Use absolute values when scoring feature rankings.")
    parser.add_argument(
        "--rank-positive",
        action="store_true",
        help="Keep only positive contributions when scoring feature rankings.",
    )
    parser.add_argument(
        "--rank-exclude",
        type=str,
        default='activation_patch',
        help="Comma-separated methods to ignore --rank-abs/--rank-positive (e.g. activation_patch,ig_anchor).",
    )
    parser.add_argument("--objective-mode", type=str, default="mask", help="Objective mode for ranking (mask, output_positive, logit_sum, etc.).")
    parser.add_argument("--methods", type=str, default="ig,libragrad_ig,libragrad_input_x_grad,grad,input_x_grad,attnlrp_input_x_grad", help="Methods to run.")
    parser.add_argument("--libragrad-gamma", type=float, default=None, help="Optional gamma for libragrad linear layers.")
    parser.add_argument("--baseline-cache", type=Path, default=None, help="Optional ActivationBaselineCache (.pt) path.")
    parser.add_argument("--log-file", type=Path, default=Path("logs/output_contribution_compare_sam2.txt"), help="Path to save logs.")
    parser.add_argument("--lane", type=int, default=None, help="Optional override for lane index (default: use ledger UID map).")
    parser.add_argument(
        "--random-bvd-samples",
        type=int,
        default=100,
        help="Number of random BVD samples from offline meta ledger.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling and torch.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    debug = bool(args.debug)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    log_fh = _open_log_file(args.log_file) if args.log_file is not None else None
    _set_log_file(log_fh)
    baseline_cache = _load_baseline_cache(args.baseline_cache)
    if baseline_cache is not None and args.baseline_cache is not None:
        _log_line(f"[info] loaded activation baseline cache: {args.baseline_cache}")

    attr_cfg = _load_yaml(args.attr_config)
    indexing_cfg = _load_yaml(Path(attr_cfg["indexing"]["config"]))
    layer_default = attr_cfg["indexing"]["layer"]
    anchor_specs_raw = _parse_anchor_specs(args.anchors if args.anchors is not None else layer_default)
    anchor_specs = _expand_anchor_specs(anchor_specs_raw)
    ig_active_specs = _parse_anchor_specs(args.ig_active) if args.ig_active else None
    mask_specs = [s for s in (args.mask_specs or "").split(",") if s]
    ref_mask_spec = args.ref_mask_spec or (mask_specs[0] if mask_specs else None)
    dump_root = args.dump_insertion_masks_dir

    device = torch.device(args.device or indexing_cfg.get("model", {}).get("device", "cuda"))
    model_loader = load_obj(indexing_cfg["model"]["loader"])
    model = model_loader(indexing_cfg["model"], device=device).eval()
    for p in model.parameters():
        p.requires_grad = False
    adapter = SAM2EvalAdapter(model, device=device, collate_fn=None)

    anchors: Dict[str, AnchorInfo] = {}
    restore_handles: List[Any] = []
    for spec in anchor_specs:
        controller = AllTokensFeatureOverrideController(spec=OverrideSpec(lane_idx=None, unit_indices=None), frame_getter=getattr(adapter, "current_frame_idx", None))
        sae = _load_sae_for_layer(indexing_cfg["sae"], spec, device)
        capture = LayerCapture(spec)
        owner = resolve_module(model, capture.base)
        handle, branch = wrap_target_layer_with_sae(
            owner,
            capture=capture,
            sae=sae,
            controller=controller,
            frame_getter=getattr(adapter, "current_frame_idx", None),
        )
        restore_handles.append(handle)
        attr = _anchor_attr_name(spec)
        anchors[spec] = AnchorInfo(spec=spec, branch=branch, attr_name=attr, controller=controller)

    autoreset_handle = install_controller_autoreset_hooks(
        model,
        [a.controller for a in anchors.values()],
        branches=[a.branch for a in anchors.values()],
    )
    if autoreset_handle is not None:
        restore_handles.append(autoreset_handle.remove)

    mask_capture = MultiAnchorCapture(frame_getter=getattr(adapter, "current_frame_idx", None))
    mask_capture.register_from_specs(mask_specs, resolve_module_fn=lambda name: resolve_module(model, name))

    methods = [tok.strip().lower() for tok in (args.methods or "").split(",") if tok.strip()]
    objective_mode = str(args.objective_mode).strip().lower()
    use_abs = bool(args.rank_abs)
    use_positive = bool(args.rank_positive)
    if use_abs and use_positive:
        _log_line("[warn] both --rank-abs and --rank-positive set; using --rank-positive only.")
        use_abs = False
    rank_exclude = {_normalize_method_name(tok) for tok in _parse_csv_list(args.rank_exclude)}

    meta_ledger = OfflineMetaParquetLedger(indexing_cfg["indexing"]["offline_meta_root"])

    method_aucs: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    rows_all: List[Tuple[int, Dict[str, Any]]] = _sample_random_bvd_rows(
        meta_ledger=meta_ledger,
        n=int(args.random_bvd_samples),
        seed=int(args.seed),
    )
    _log_line(f"[info] using random BVD samples: n={len(rows_all)}")

    # Process rows
    processed_rows = 0
    stop_now = False
    total_rows = len(rows_all)
    for row_idx, (dec, row) in enumerate(rows_all, start=1):
            sample_id = int(row["sample_id"])
            frame_idx = int(row.get("frame_idx", 0))
            bvd, lane_idx = _build_bvd_for_row(
                row=row,
                meta_ledger=meta_ledger,
                ds_cfg=indexing_cfg["dataset"],
            )
            if args.lane is not None:
                lane_idx = int(args.lane)
            batch_on_dev = adapter.preprocess_input(bvd)

            _record_adapter_caches(adapter, batch_on_dev)
            forward_runner = _make_forward_runner(adapter, batch_on_dev)
            for anchor in anchors.values():
                try:
                    anchor.controller.spec.lane_idx = lane_idx
                except Exception:
                    pass

            _log_line(
                f"\n=== sample {row_idx}/{total_rows} | decile={dec} sample_id={sample_id} frame={frame_idx} lane={lane_idx} ==="
            )
            base_obj, objective_mask, objective_ref_logits = _measure_objective_value(
                forward_runner,
                mask_capture,
                lane_idx=lane_idx,
                threshold=float(args.mask_threshold),
                ref_mask_spec=ref_mask_spec,
                debug=debug,
                label="base",
                return_mask=True,
                return_ref_logits=True,
            )
            _log_line(f"[base] objective={base_obj:.6f}")

            bases = _capture_anchor_bases(anchors)
            _debug_print(f"[debug] captured bases: {_describe_base_shapes(bases)}", enabled=debug)

            curve_records: List[Dict[str, Any]] = []

            for method in methods:
                _log_line(f"\n--- method: {method} ---")
                method_key = _normalize_method_name(method)
                dispatch_method, force_abs, force_positive = _split_method_variant(method_key)
                method_use_abs = (use_abs and dispatch_method not in rank_exclude) or force_abs
                method_use_positive = (use_positive and dispatch_method not in rank_exclude) or force_positive
                if method_use_positive:
                    method_use_abs = False
                if dispatch_method != method_key:
                    _debug_print(
                        f"[debug] method variant: raw={method_key} -> base={dispatch_method} "
                        f"(abs={method_use_abs}, pos={method_use_positive})",
                        enabled=debug,
                    )
                _clear_anchor_contexts(anchors, debug=debug, reason=f"method:{method}")
                ranking: List[Tuple[str, int, float]] = []
                active_masks: Dict[str, torch.Tensor] = {}
                if dispatch_method == "activation_patch":
                    ranking, active_masks = _build_activation_patching_ranking(
                        forward_runner,
                        mask_capture,
                        anchors,
                        bases,
                        baseline_cache,
                        lane_idx,
                        float(args.mask_threshold),
                        float(args.feature_active_threshold),
                        ref_mask_spec,
                        args.max_features,
                        objective_mode=objective_mode,
                        objective_ref_logits=objective_ref_logits,
                        base_objective=base_obj,
                        objective_mask=objective_mask,
                        progress=True,
                    )
                    variants = [(method, ranking, active_masks)]
                elif dispatch_method == "grad":
                    ranking = _build_grad_ranking(
                        forward_runner,
                        mask_capture,
                        anchors,
                        lane_idx,
                        float(args.mask_threshold),
                        ref_mask_spec,
                        objective_mask,
                        objective_ref_logits=objective_ref_logits,
                        objective_mode=objective_mode,
                        use_abs=method_use_abs,
                        use_positive=method_use_positive,
                    )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "input_x_grad":
                    ranking = _build_inputxgrad_ranking(
                        forward_runner,
                        mask_capture,
                        anchors,
                        lane_idx,
                        float(args.mask_threshold),
                        ref_mask_spec,
                        objective_mask,
                        objective_ref_logits=objective_ref_logits,
                        objective_mode=objective_mode,
                        use_abs=method_use_abs,
                        use_positive=method_use_positive,
                    )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "ig":
                    ranking = _build_ig_ranking(
                        forward_runner,
                        mask_capture,
                        anchors,
                        lane_idx,
                        float(args.mask_threshold),
                        ref_mask_spec,
                        objective_mask,
                        steps=args.ig_steps,
                        objective_ref_logits=objective_ref_logits,
                        objective_mode=objective_mode,
                        use_abs=method_use_abs,
                        use_positive=method_use_positive,
                        ig_active=ig_active_specs,
                    )
                    variants = [(method, ranking, None)]
                elif dispatch_method in {"ig_anchor", "ig_anchored", "anchored_ig", "ig-anchor", "ig-anchored"}:
                    ranking = _build_ig_anchor_ranking(
                        forward_runner,
                        mask_capture,
                        anchors,
                        lane_idx,
                        float(args.mask_threshold),
                        ref_mask_spec,
                        objective_mask,
                        steps=args.ig_steps,
                        objective_ref_logits=objective_ref_logits,
                        objective_mode=objective_mode,
                        use_abs=method_use_abs,
                        use_positive=method_use_positive,
                        ig_active=ig_active_specs,
                    )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "smoothgrad":
                    ranking = _build_smooth_like_ranking(
                        "smoothgrad",
                        batch_on_dev=batch_on_dev,
                        forward_runner=forward_runner,
                        mask_capture=mask_capture,
                        anchors=anchors,
                        lane_idx=lane_idx,
                        threshold=float(args.mask_threshold),
                        ref_mask_spec=ref_mask_spec,
                        objective_mask=objective_mask,
                        objective_mode=objective_mode,
                        objective_ref_logits=objective_ref_logits,
                        samples=args.smoothgrad_samples,
                        noise_sigma=args.smoothgrad_sigma,
                        use_abs=method_use_abs,
                        use_positive=method_use_positive,
                    )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "vargard":
                    ranking = _build_smooth_like_ranking(
                        "vargard",
                        batch_on_dev=batch_on_dev,
                        forward_runner=forward_runner,
                        mask_capture=mask_capture,
                        anchors=anchors,
                        lane_idx=lane_idx,
                        threshold=float(args.mask_threshold),
                        ref_mask_spec=ref_mask_spec,
                        objective_mask=objective_mask,
                        objective_mode=objective_mode,
                        objective_ref_logits=objective_ref_logits,
                        samples=args.smoothgrad_samples,
                        noise_sigma=args.smoothgrad_sigma,
                        use_abs=method_use_abs,
                        use_positive=method_use_positive,
                    )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "libragrad_input_x_grad":
                    with _LibragradContext(model, anchors, gamma=args.libragrad_gamma):
                        ranking = _build_inputxgrad_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                        )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "libragrad_ig":
                    with _LibragradContext(model, anchors, gamma=args.libragrad_gamma):
                        ranking = _build_ig_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            steps=args.ig_steps,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                            ig_active=ig_active_specs,
                        )
                    variants = [(method, ranking, None)]
                elif dispatch_method in {"libragrad_ig_anchor", "libragrad_ig_anchored", "ligrad_ig_anchor"}:
                    with _LibragradContext(model, anchors, gamma=args.libragrad_gamma):
                        ranking = _build_ig_anchor_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            steps=args.ig_steps,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                            ig_active=ig_active_specs,
                        )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "attnlrp_grad":
                    with _AttnLRPContext(model, anchors):
                        ranking = _build_grad_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                        )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "attnlrp_input_x_grad":
                    with _AttnLRPContext(model, anchors):
                        ranking = _build_inputxgrad_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                        )
                    variants = [(method, ranking, None)]
                elif dispatch_method == "attnlrp_ig":
                    with _AttnLRPContext(model, anchors):
                        ranking = _build_ig_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            steps=args.ig_steps,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                            ig_active=ig_active_specs,
                        )
                    variants = [(method, ranking, None)]
                elif dispatch_method in {"attnlrp_ig_anchor", "attnlrp_ig_anchored"}:
                    with _AttnLRPContext(model, anchors):
                        ranking = _build_ig_anchor_ranking(
                            forward_runner,
                            mask_capture,
                            anchors,
                            lane_idx,
                            float(args.mask_threshold),
                            ref_mask_spec,
                            objective_mask,
                            steps=args.ig_steps,
                            objective_mode=objective_mode,
                            objective_ref_logits=objective_ref_logits,
                            use_abs=method_use_abs,
                            use_positive=method_use_positive,
                            ig_active=ig_active_specs,
                        )
                    variants = [(method, ranking, None)]
                else:
                    _log_line(f"[skip] unknown method '{method}'")
                    continue

                for variant_label, rank_list, mask_map in variants:
                    if not rank_list:
                        _log_line(f"[warn] no ranking for {variant_label}")
                        continue
                    _debug_print(
                        f"[debug] variant={variant_label} ranking_len={len(rank_list)} active_masks={ {k: int(v.sum().item()) for k, v in (mask_map or {}).items()} }",
                        enabled=debug,
                    )
                    _log_line(f"[curve] deletion ({variant_label})")
                    del_curve = _run_objective_curve(
                        "deletion",
                        forward_runner=forward_runner,
                        mask_capture=mask_capture,
                        lane_idx=lane_idx,
                        threshold=float(args.mask_threshold),
                        ref_mask_spec=ref_mask_spec,
                        objective_mask=objective_mask,
                        objective_ref_logits=objective_ref_logits,
                        anchors=anchors,
                        bases=bases,
                        baseline_cache=baseline_cache,
                        ranking=rank_list,
                        chunk_size=args.chunk_size,
                        max_steps=args.max_steps,
                        active_masks=mask_map,
                        norm_ref=base_obj,
                        debug=debug,
                        curve_label=variant_label,
                    )
                    _log_line(f"[curve] insertion ({variant_label})")
                    mask_dump_cfg = None
                    if dump_root is not None:
                        mask_dump_cfg = MaskDumpConfig(
                            root=dump_root,
                            sample_id=sample_id,
                            frame_idx=frame_idx,
                            lane_idx=lane_idx,
                            decile=dec,
                            method=variant_label,
                            curve="insertion",
                        )
                    ins_curve = _run_objective_curve(
                        "insertion",
                        forward_runner=forward_runner,
                        mask_capture=mask_capture,
                        lane_idx=lane_idx,
                        threshold=float(args.mask_threshold),
                        ref_mask_spec=ref_mask_spec,
                        objective_mask=objective_mask,
                        objective_ref_logits=objective_ref_logits,
                        anchors=anchors,
                        bases=bases,
                        baseline_cache=baseline_cache,
                        ranking=rank_list,
                        chunk_size=args.chunk_size,
                        max_steps=args.max_steps,
                        active_masks=mask_map,
                        norm_ref=base_obj,
                        mask_dump=mask_dump_cfg,
                        debug=debug,
                        curve_label=variant_label,
                    )
                    del_auc = _compute_auc(del_curve)
                    ins_auc = _compute_auc(ins_curve)
                    _log_line(f"  AUC deletion={del_auc:.4g}, insertion={ins_auc:.4g}")
                    method_aucs[variant_label].append((del_auc, ins_auc))
                    curve_records.append(
                        {
                            "sample_id": sample_id,
                            "frame_idx": frame_idx,
                            "method": variant_label,
                            "deletion": del_curve,
                            "insertion": ins_curve,
                            "del_auc": del_auc,
                            "ins_auc": ins_auc,
                        }
                    )

                _reset_anchor_controllers(anchors, debug=debug, reason=f"post-method:{method}")
                model.zero_grad(set_to_none=True)
                _gpu_gc()

            if curve_records:
                _log_line("\n=== AUC summary ===")
                for rec in curve_records:
                    _log_line(f"sample={rec['sample_id']} frame={rec['frame_idx']} {rec['method']}: del={rec['del_auc']:.4f}, ins={rec['ins_auc']:.4f}")

            _clear_anchor_contexts(anchors, debug=debug, reason="row-end")
            mask_capture.release_step_refs()
            model.zero_grad(set_to_none=True)
            _gpu_gc()
            processed_rows += 1
            max_rows = args.stop_after if args.stop_after is not None else None
            if max_rows is not None and processed_rows >= max_rows:
                stop_now = True
                break
    if stop_now:
        pass

    if method_aucs:
        _log_line("\n=== Global AUC mean/std over rows ===")
        for method_name, vals in method_aucs.items():
            if not vals:
                continue
            del_vals = np.asarray([v[0] for v in vals], dtype=float)
            ins_vals = np.asarray([v[1] for v in vals], dtype=float)
            mean_del = float(del_vals.mean()) if del_vals.size else 0.0
            mean_ins = float(ins_vals.mean()) if ins_vals.size else 0.0
            std_del = float(del_vals.std()) if del_vals.size > 1 else 0.0
            std_ins = float(ins_vals.std()) if ins_vals.size > 1 else 0.0
            _log_line(
                f"{method_name}: mean_del={mean_del:.4f} std_del={std_del:.4f}, mean_ins={mean_ins:.4f} std_ins={std_ins:.4f} (n={len(vals)})"
            )

    try:
        mask_capture.clear()
    except Exception:
        pass
    for handle in reversed(restore_handles):
        handle()


def _open_log_file(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("a", encoding="utf-8")
    return fh


if __name__ == "__main__":
    main()
