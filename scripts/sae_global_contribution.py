#!/usr/bin/env python3
"""
Compute global SAE feature output contributions from decile-indexed samples.

Workflow:
1) Load decile parquet ledger and pick top-N samples per decile.
2) Merge across deciles and select the global top-N.
3) If fewer than N samples (or missing files), skip the unit.
4) Compute per-sample contributions for that N-sized batch and global gain.
5) Save per-sample contribution vectors + per-feature mean/global_gain vectors.

Example:
  python scripts/sae_global_contribution.py \
    --config configs/clip_imagenet_index.yaml \
    --deciles 1,2,3,4,5,6,7,8,9,10 --decile-base 1 \
    --topn-per-decile 10 --global-gain-topn 10 \
    --methods logitprism,second_order_lens,activation_patch,activation_patch_sweep,ig,ig_target,activation_patch_input_x_grad_avg,grad,input_x_grad,smoothgrad_input_x_grad,attnlrp_input_x_grad,libragrad_input_x_grad,libragrad_ig \
    --contrib-captures model.head --class-topk 100
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

# Ensure repo root is on sys.path and set as CWD so `experiments.*` imports work in any launch dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    os.chdir(REPO_ROOT)
except Exception:
    pass

from experiments import topk_contribution_alignment as tca
from src.core.hooks.spec import parse_spec
from src.core.indexing.decile_parquet_ledger import DecileParquetLedger
from src.core.indexing.offline_meta import build_offline_ledger
from src.core.runtime.activation_baselines import ActivationBaselineCache
from src.core.runtime.capture import LayerCapture
from src.core.runtime.controllers import AllTokensFeatureOverrideController, install_controller_autoreset_hooks
from src.core.runtime.specs import OverrideSpec
from src.core.runtime.wrappers import wrap_target_layer_with_sae
from src.packs.clip.circuit_runtime import _load_sae_for_layer
from src.packs.clip.dataset.builders import build_clip_transform
from src.packs.clip.models.adapters import CLIPVisionAdapter
from src.utils.utils import load_obj, resolve_module
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOG = logging.getLogger("sae_global_contribution")


@dataclass
class SampleRecord:
    sample_id: int
    decile: int
    rank_in_decile: int
    score: float
    frame_idx: int
    y: int
    x: int
    prompt_id: int
    uid: int
    stride_step: int
    run_id: str
    path: str = ""


@dataclass
class MethodState:
    method: str
    root: Path
    per_feature_root: Path
    feature_entries: List[Dict[str, Any]]
    mean_vectors: List[np.ndarray]
    global_gains: List[np.ndarray]
    mean_acts: List[float]
    decile_vectors: List[Dict[int, np.ndarray]]
    decile_mean_acts: List[Dict[int, float]]
    layer_feature_counts: Dict[str, int]
    layer_order: List[str]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_csv_list(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def _parse_float_list(raw: Optional[str]) -> List[float]:
    if not raw:
        return []
    out: List[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _unique_lower(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _parse_deciles(raw: str, *, base: int) -> List[int]:
    if not raw:
        return []
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = int(tok)
        if base == 1:
            val -= 1
        if val < 0:
            raise ValueError(f"decile '{tok}' maps to negative index (base={base})")
        out.append(val)
    return sorted(set(out))


def _layer_to_latent_spec(layer: str) -> str:
    parsed = parse_spec(layer)
    if parsed.method is None and parsed.attr is None:
        return f"{parsed.base_with_branch}::sae_layer#latent"
    if parsed.method is not None and parsed.attr is None:
        return f"{parsed.base_with_branch}::{parsed.method}#latent"
    return layer


def _unwrap_tensor(val) -> torch.Tensor:
    if torch.is_tensor(val):
        return val
    if isinstance(val, dict):
        for v in val.values():
            if torch.is_tensor(v):
                return v
            if isinstance(v, (dict, list, tuple)):
                t = _unwrap_tensor(v)
                if torch.is_tensor(t):
                    return t
    if isinstance(val, (list, tuple)):
        for v in val:
            if torch.is_tensor(v):
                return v
            if isinstance(v, (dict, list, tuple)):
                t = _unwrap_tensor(v)
                if torch.is_tensor(t):
                    return t
    raise TypeError(f"Unsupported contribution payload type: {type(val)}")


def _reduce_class_vec(tensor_like, *, keep_batch: bool = False) -> torch.Tensor:
    tensor = _unwrap_tensor(tensor_like)
    vec = tensor.detach()
    if vec.dim() == 0:
        return vec.reshape(1)
    if vec.dim() == 1:
        return vec
    if not keep_batch:
        vec = vec.reshape(-1, vec.shape[-1])
        return vec.sum(dim=0)
    if vec.dim() == 2:
        return vec
    vec = vec.reshape(vec.shape[0], -1, vec.shape[-1])
    return vec.sum(dim=1)


def _coerce_batch_contrib(
    vec: np.ndarray,
    *,
    expected_batch: int,
    expected_classes: Optional[int],
) -> Optional[np.ndarray]:
    if expected_batch <= 0:
        return None
    arr = np.asarray(vec)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if expected_batch == 1:
            return arr.reshape(1, -1)
        if expected_classes is not None and arr.size == expected_batch * expected_classes:
            return arr.reshape(expected_batch, expected_classes)
        return None
    if arr.ndim == 2:
        if arr.shape[0] == expected_batch:
            if expected_classes is None or arr.shape[1] == expected_classes:
                return arr
            return None
        if arr.shape[1] == expected_batch:
            arr = arr.T
            if expected_classes is None or arr.shape[1] == expected_classes:
                return arr
            return None
        if expected_classes is not None and arr.size == expected_batch * expected_classes:
            return arr.reshape(expected_batch, expected_classes)
        return None
    if expected_batch in arr.shape:
        batch_axis = list(arr.shape).index(expected_batch)
        if batch_axis != 0:
            arr = np.moveaxis(arr, batch_axis, 0)
    else:
        if expected_classes is not None and arr.size == expected_batch * expected_classes:
            return arr.reshape(expected_batch, expected_classes)
        return None
    if expected_classes is None:
        return arr.reshape(expected_batch, -1)
    if arr.shape[-1] == expected_classes:
        if arr.ndim == 2:
            return arr
        return arr.reshape(expected_batch, -1, expected_classes).sum(axis=1)
    if expected_classes in arr.shape[1:]:
        class_axis = 1 + list(arr.shape[1:]).index(expected_classes)
        arr = np.moveaxis(arr, class_axis, -1)
        return arr.reshape(expected_batch, -1, expected_classes).sum(axis=1)
    remaining = int(np.prod(arr.shape[1:]))
    if remaining == expected_classes:
        return arr.reshape(expected_batch, expected_classes)
    return None


def _topk_with_values(arr: np.ndarray, k: int, *, largest: bool = True) -> List[Dict[str, Any]]:
    if arr.ndim != 1 or arr.size == 0:
        return []
    k = max(1, min(k, arr.shape[0]))
    idx = np.argpartition(-arr if largest else arr, k - 1)[:k]
    idx = idx[np.argsort(-arr[idx] if largest else arr[idx])]
    return [{"class_id": int(i), "score": float(arr[i])} for i in idx]


def _topk_bundle(mean_vec: np.ndarray, *, max_abs: float, mean_act: float, k: int) -> Dict[str, Any]:
    eps = np.finfo(np.float32).eps
    norm_max = mean_vec / max(max_abs, eps)
    norm_act = mean_vec / max(mean_act, eps)
    return {
        "k": int(k),
        "raw": {
            "pos": _topk_with_values(mean_vec, k, largest=True),
            "neg": _topk_with_values(mean_vec, k, largest=False),
        },
        "norm_max": {
            "pos": _topk_with_values(norm_max, k, largest=True),
            "neg": _topk_with_values(norm_max, k, largest=False),
            "denominator": "max_abs",
        },
        "norm_act": {
            "pos": _topk_with_values(norm_act, k, largest=True),
            "neg": _topk_with_values(norm_act, k, largest=False),
            "denominator": "mean_abs_activation",
        },
    }


def _topn_indices_by_score(rows: Sequence[SampleRecord], n: int) -> np.ndarray:
    count = len(rows)
    if n <= 0 or count <= n:
        return np.arange(count, dtype=np.int64)
    scores = np.array([r.score for r in rows], dtype=np.float32)
    if scores.size == 0:
        return np.arange(count, dtype=np.int64)
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def _compute_global_gain(contrib_arr: np.ndarray, activation_vals: np.ndarray) -> np.ndarray:
    if contrib_arr.ndim != 2 or contrib_arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    acts = np.array(activation_vals, dtype=np.float32, copy=False).reshape(-1)
    contrib = np.array(contrib_arr, dtype=np.float32, copy=False)
    if acts.size == 0 or acts.shape[0] != contrib.shape[0]:
        return np.zeros((contrib.shape[1],), dtype=np.float32)
    denom = float(np.dot(acts, acts))
    if denom <= np.finfo(np.float32).eps:
        return np.zeros((contrib.shape[1],), dtype=np.float32)
    numer = (acts[:, None] * contrib).sum(axis=0, dtype=np.float32)
    return (numer / denom).astype(np.float32, copy=False)


def _lens_score_matrix(sae: Any, model: Any) -> Optional[torch.Tensor]:
    dec = getattr(sae, "W_dec", None)
    head = getattr(model, "head", None)
    head_w = getattr(head, "weight", None) if head is not None else None
    if dec is None or head_w is None:
        return None
    try:
        scores = torch.matmul(dec.detach(), head_w.detach().T)
    except Exception:
        return None
    return scores.detach().cpu()


def _build_image_batch(
    *,
    transform,
    image_path: Path,
    sample_id: int,
) -> Dict[str, Any]:
    with Image.open(image_path.expanduser()) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0)
    return {
        "pixel_values": tensor,
        "label": None,
        "sample_id": torch.tensor([int(sample_id)], dtype=torch.long),
        "path": [str(image_path)],
    }


def _build_image_batch_multi(
    *,
    transform,
    rows: Sequence[SampleRecord],
) -> tuple[Optional[Dict[str, Any]], List[SampleRecord]]:
    tensors: List[torch.Tensor] = []
    sample_ids: List[int] = []
    paths: List[str] = []
    kept_rows: List[SampleRecord] = []
    for row in rows:
        if not row.path:
            continue
        img_path = Path(row.path).expanduser()
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as img:
                tensor = transform(img.convert("RGB"))
        except Exception:
            continue
        tensors.append(tensor)
        sample_ids.append(int(row.sample_id))
        paths.append(str(img_path))
        kept_rows.append(row)
    if not tensors:
        return None, []
    batch_tensor = torch.stack(tensors, dim=0)
    return {
        "pixel_values": batch_tensor,
        "label": None,
        "sample_id": torch.tensor(sample_ids, dtype=torch.long),
        "path": paths,
    }, kept_rows


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    size = max(1, int(batch_size))
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _unit_activation_stats(
    base: torch.Tensor,
    *,
    unit: int,
    expected_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros(int(expected_count), dtype=np.float32)
    if not torch.is_tensor(base):
        return zeros, zeros.copy()
    try:
        base_t = base.detach()
        if base_t.dim() == 0:
            return zeros, zeros.copy()
        if base_t.shape[-1] <= int(unit):
            return zeros, zeros.copy()
        unit_vals = base_t[..., int(unit)]
        if unit_vals.dim() == 0:
            unit_vals = unit_vals.reshape(1, 1)
        elif unit_vals.dim() == 1:
            unit_vals = unit_vals.unsqueeze(-1)
        else:
            unit_vals = unit_vals.reshape(unit_vals.shape[0], -1)
        act_mag = unit_vals.abs().mean(dim=1).detach().cpu().float().numpy()
        act_val = unit_vals.mean(dim=1).detach().cpu().float().numpy()
    except Exception:
        return zeros, zeros.copy()
    if act_mag.shape[0] < expected_count:
        pad = int(expected_count - act_mag.shape[0])
        act_mag = np.concatenate([act_mag, np.zeros(pad, dtype=np.float32)])
        act_val = np.concatenate([act_val, np.zeros(pad, dtype=np.float32)])
    elif act_mag.shape[0] > expected_count:
        act_mag = act_mag[:expected_count]
        act_val = act_val[:expected_count]
    return act_mag, act_val


def _collect_decile_rows(
    ledger: DecileParquetLedger,
    *,
    layer: str,
    unit: int,
    deciles: Sequence[int],
    topn_per_decile: int,
) -> List[SampleRecord]:
    rows: List[SampleRecord] = []
    for dec in deciles:
        tbl = ledger.topn_for(
            layer=layer,
            unit=int(unit),
            decile=int(dec),
            n=int(topn_per_decile),
        )
        if tbl.num_rows == 0:
            continue
        cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
        n_rows = tbl.num_rows
        for idx in range(n_rows):
            rows.append(
                SampleRecord(
                    sample_id=int(cols.get("sample_id", [0] * n_rows)[idx]),
                    decile=int(cols.get("decile", [dec] * n_rows)[idx]),
                    rank_in_decile=int(cols.get("rank_in_decile", [idx] * n_rows)[idx]),
                    score=float(cols.get("score", [0.0] * n_rows)[idx]),
                    frame_idx=int(cols.get("frame_idx", [0] * n_rows)[idx]),
                    y=int(cols.get("y", [-1] * n_rows)[idx]),
                    x=int(cols.get("x", [-1] * n_rows)[idx]),
                    prompt_id=int(cols.get("prompt_id", [0] * n_rows)[idx]),
                    uid=int(cols.get("uid", [-1] * n_rows)[idx]),
                    stride_step=int(cols.get("stride_step", [0] * n_rows)[idx]),
                    run_id=str(cols.get("run_id", [""] * n_rows)[idx]),
                )
            )
    return rows


def _lookup_paths(
    ledger: Any,
    sample_ids: Sequence[int],
    *,
    cache: Dict[int, str],
) -> Dict[int, str]:
    missing = [int(sid) for sid in sample_ids if int(sid) not in cache]
    if missing and ledger is not None:
        try:
            resolved = ledger.lookup(missing)
        except Exception:
            resolved = {}
        for sid in missing:
            cache[int(sid)] = str(resolved.get(int(sid), ""))
    return {int(sid): cache.get(int(sid), "") for sid in sample_ids}


def _save_npz(path: Path, *, compress: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _resolve_unit_paths(layer_dir: Path, unit: int, save_format: str) -> tuple[Path, Optional[Path]]:
    unit_id = int(unit)
    pt_path = layer_dir / f"unit_{unit_id}.pt"
    npz_path = layer_dir / f"unit_{unit_id}.npz"
    preferred = pt_path if save_format == "pt" else npz_path
    fallback = npz_path if save_format == "pt" else pt_path
    existing = preferred if preferred.exists() else (fallback if fallback.exists() else None)
    return preferred, existing


def _to_numpy(value, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if value is None:
        arr = np.array([], dtype=dtype or np.float32)
    elif torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _load_unit_payload(unit_path: Path) -> Optional[Dict[str, Any]]:
    try:
        if unit_path.suffix == ".npz":
            with np.load(unit_path, allow_pickle=False) as npz:
                if "mean" not in npz:
                    return None
                mean_vec = _to_numpy(npz["mean"], dtype=np.float32)
                gain_vec = _to_numpy(npz.get("global_gain"), dtype=np.float32)
                if gain_vec.size == 0 and mean_vec.size > 0:
                    gain_vec = np.zeros_like(mean_vec)
                mean_act_arr = _to_numpy(npz.get("mean_activation", [0.0]), dtype=np.float32).reshape(-1)
                mean_act = float(mean_act_arr[0]) if mean_act_arr.size > 0 else 0.0
                dec_ids = _to_numpy(npz.get("decile_ids"), dtype=np.int16)
                dec_means = _to_numpy(npz.get("decile_means"))
                dec_mean_acts = _to_numpy(npz.get("decile_mean_acts"), dtype=np.float32)
                sample_ids = _to_numpy(npz.get("sample_ids"), dtype=np.int64)
        else:
            payload = torch.load(unit_path, map_location="cpu")
            if not isinstance(payload, dict) or "mean" not in payload:
                return None
            mean_vec = _to_numpy(payload.get("mean"), dtype=np.float32)
            gain_vec = _to_numpy(payload.get("global_gain"), dtype=np.float32)
            if gain_vec.size == 0 and mean_vec.size > 0:
                gain_vec = np.zeros_like(mean_vec)
            mean_act_arr = _to_numpy(payload.get("mean_activation", [0.0]), dtype=np.float32).reshape(-1)
            mean_act = float(mean_act_arr[0]) if mean_act_arr.size > 0 else 0.0
            dec_ids = _to_numpy(payload.get("decile_ids"), dtype=np.int16)
            dec_means = _to_numpy(payload.get("decile_means"))
            dec_mean_acts = _to_numpy(payload.get("decile_mean_acts"), dtype=np.float32)
            sample_ids = _to_numpy(payload.get("sample_ids"), dtype=np.int64)
        dec_vecs: Dict[int, np.ndarray] = {}
        dec_mean_acts_map: Dict[int, float] = {}
        for didx, dec_id in enumerate(dec_ids.tolist() if dec_ids.size > 0 else []):
            if dec_means.size > 0 and didx < dec_means.shape[0]:
                dec_vecs[int(dec_id)] = np.array(dec_means[didx], dtype=np.float32, copy=False)
            if dec_mean_acts.size > 0 and didx < dec_mean_acts.shape[0]:
                dec_mean_acts_map[int(dec_id)] = float(dec_mean_acts[didx])
        max_abs = float(np.max(np.abs(mean_vec))) if mean_vec.size > 0 else 0.0
        sample_count = int(sample_ids.shape[0]) if sample_ids.size > 0 else 0
        return {
            "mean_vec": mean_vec,
            "global_gain": gain_vec,
            "mean_act": mean_act,
            "dec_vecs": dec_vecs,
            "dec_mean_acts": dec_mean_acts_map,
            "max_abs": max_abs,
            "sample_count": sample_count,
        }
    except Exception:
        return None


def _save_unit_payload(path: Path, *, compress: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        payload = {k: torch.as_tensor(v) for k, v in arrays.items()}
        torch.save(payload, path)
    else:
        _save_npz(path, compress=compress, **arrays)


def _load_baseline_cache(path: Optional[Path]) -> Optional[ActivationBaselineCache]:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.exists():
        LOG.warning("Baseline cache not found: %s", resolved)
        return None
    if resolved.is_dir():
        LOG.warning("Baseline cache path is a directory (expected file): %s", resolved)
        return None
    try:
        cache = ActivationBaselineCache.load(resolved)
    except Exception as exc:
        LOG.warning("Failed to load baseline cache %s: %s", resolved, exc)
        return None
    if not isinstance(cache, ActivationBaselineCache):
        LOG.warning("Loaded baseline cache has unexpected type %s (path=%s)", type(cache), resolved)
        return None
    return cache


def _configure_logger(log_file: Optional[Path]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def _write_method_outputs(
    *,
    state: MethodState,
    args: argparse.Namespace,
    deciles: Sequence[int],
    contrib_captures: Sequence[str],
    contrib_ig_active: Sequence[str],
    activation_patch_sweep: Sequence[float],
    final_root: Optional[Path],
) -> Optional[Dict[str, Any]]:
    write_decile_outputs = bool(args.write_decile_outputs)

    def _load_unit_npz(unit_path: Path):
        return _load_unit_payload(unit_path)

    def _add_feature_to_lookup(
        *,
        entry: Dict[str, Any],
        mean_vec: np.ndarray,
        global_gain: np.ndarray,
        mean_act: float,
        dec_vecs: Dict[int, np.ndarray],
        dec_mean_acts: Dict[int, float],
        max_abs: float,
        feature_lookup: Dict[tuple[str, int], Dict[str, Any]],
    ) -> None:
        if not write_decile_outputs:
            dec_vecs = {}
            dec_mean_acts = {}
        key = (entry["layer"], int(entry["unit"]))
        feature_lookup[key] = {
            "entry": entry,
            "mean_vec": np.array(mean_vec, dtype=np.float32, copy=False),
            "global_gain": np.array(global_gain, dtype=np.float32, copy=False),
            "mean_act": float(mean_act),
            "dec_vecs": dec_vecs,
            "dec_mean_acts": dec_mean_acts,
            "max_abs": float(max_abs),
        }

    feature_lookup: Dict[tuple[str, int], Dict[str, Any]] = {}
    for entry, mean_vec, global_gain, mean_act, dec_vecs, dec_mean_acts in zip(
        state.feature_entries,
        state.mean_vectors,
        state.global_gains,
        state.mean_acts,
        state.decile_vectors,
        state.decile_mean_acts,
    ):
        _add_feature_to_lookup(
            entry=entry,
            mean_vec=mean_vec,
            global_gain=global_gain,
            mean_act=mean_act,
            dec_vecs=dec_vecs,
            dec_mean_acts=dec_mean_acts,
            max_abs=entry.get("max_abs", 0.0),
            feature_lookup=feature_lookup,
        )

    expected_features = None
    if state.layer_feature_counts:
        expected_features = sum(int(v) for v in state.layer_feature_counts.values())
        for layer_name, count in state.layer_feature_counts.items():
            layer_dir = state.per_feature_root / layer_name
            for unit in range(int(count)):
                key = (layer_name, int(unit))
                if key in feature_lookup:
                    continue
                unit_path, existing_path = _resolve_unit_paths(
                    layer_dir, int(unit), str(args.unit_save_format)
                )
                if existing_path is None:
                    continue
                payload = _load_unit_npz(existing_path)
                if payload is None:
                    continue
                entry = {
                    "layer": layer_name,
                    "layer_ledger": layer_name,
                    "spec": layer_name,
                    "unit": int(unit),
                    "sample_count": payload["sample_count"],
                    "max_abs": payload["max_abs"],
                    "mean_activation": payload["mean_act"],
                    "decile_mean_activations": payload["dec_mean_acts"] if write_decile_outputs else {},
                }
                state.feature_entries.append(entry)
                state.mean_vectors.append(payload["mean_vec"])
                state.global_gains.append(payload["global_gain"])
                state.mean_acts.append(payload["mean_act"])
                state.decile_vectors.append(payload["dec_vecs"])
                state.decile_mean_acts.append(payload["dec_mean_acts"])
                _add_feature_to_lookup(
                    entry=entry,
                    mean_vec=payload["mean_vec"],
                    global_gain=payload["global_gain"],
                    mean_act=payload["mean_act"],
                    dec_vecs=payload["dec_vecs"],
                    dec_mean_acts=payload["dec_mean_acts"],
                    max_abs=payload["max_abs"],
                    feature_lookup=feature_lookup,
                )

    if not state.mean_vectors:
        LOG.warning("No contributions saved for method '%s'", state.method)
        return None

    expected_features = None
    if state.layer_feature_counts:
        expected_features = sum(int(v) for v in state.layer_feature_counts.values())
    actual_features = len(state.feature_entries)
    if (
        expected_features is not None
        and int(args.unit_shard_count) > 1
        and actual_features < expected_features
    ):
        LOG.warning(
            "Skipping aggregate outputs for method '%s': partial shard output (%d/%d features). "
            "Re-run with unit_shard_count=1 after all shards complete to write full summaries.",
            state.method,
            actual_features,
            expected_features,
        )
        return None

    def _build_classes(matrix: np.ndarray, topk: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        num_classes = int(matrix.shape[1])
        for cls_idx in range(num_classes):
            vals = matrix[:, cls_idx]
            k = min(topk, vals.shape[0])
            if k <= 0:
                out[str(cls_idx)] = {"pos": [], "neg": []}
                continue
            pos_idx = np.argpartition(-vals, k - 1)[:k]
            pos_idx = pos_idx[np.argsort(-vals[pos_idx])]
            neg_idx = np.argpartition(vals, k - 1)[:k]
            neg_idx = neg_idx[np.argsort(vals[neg_idx])]
            out[str(cls_idx)] = {
                "pos": [{"feature_id": int(i), "score": float(vals[i])} for i in pos_idx],
                "neg": [{"feature_id": int(i), "score": float(vals[i])} for i in neg_idx],
            }
        return out

    try:
        num_classes = int(state.mean_vectors[0].shape[-1])
    except Exception:
        LOG.warning("Could not infer class dimension for method '%s'", state.method)
        return None

    decile_ids: set[int] = set()
    for payload in feature_lookup.values():
        decile_ids.update(int(d) for d in payload["dec_vecs"].keys())
    decile_ids = sorted(decile_ids)
    if not write_decile_outputs:
        decile_ids = []

    features_list: List[Dict[str, Any]] = []
    if state.layer_feature_counts:
        layer_order = state.layer_order or sorted(state.layer_feature_counts.keys())
        layer_offsets: Dict[str, int] = {}
        total_features = 0
        for layer_name in layer_order:
            count = int(state.layer_feature_counts.get(layer_name, 0))
            layer_offsets[layer_name] = total_features
            total_features += count

        mean_matrix = np.zeros((total_features, num_classes), dtype=np.float32)
        global_gain_matrix = np.zeros((total_features, num_classes), dtype=np.float32)
        max_abs_arr = np.zeros(total_features, dtype=np.float32)
        mean_act_arr = np.zeros(total_features, dtype=np.float32)
        decile_matrices: Dict[int, np.ndarray] = {
            int(dec): np.zeros((total_features, num_classes), dtype=np.float32) for dec in decile_ids
        }
        decile_maxabs: Dict[int, np.ndarray] = {int(dec): np.zeros(total_features, dtype=np.float32) for dec in decile_ids}
        decile_mean_acts: Dict[int, np.ndarray] = {int(dec): np.zeros(total_features, dtype=np.float32) for dec in decile_ids}

        for (layer_name, unit), payload in feature_lookup.items():
            offset = layer_offsets.get(layer_name)
            if offset is None:
                LOG.warning("Feature layer '%s' missing from feature count map; skipping unit %s", layer_name, unit)
                continue
            row_idx = offset + int(unit)
            if row_idx < 0 or row_idx >= mean_matrix.shape[0]:
                LOG.warning(
                    "Feature index %d out of bounds for layer %s (size=%d)", row_idx, layer_name, mean_matrix.shape[0]
                )
                continue
            mean_vec = payload["mean_vec"]
            if mean_vec.shape[0] != num_classes:
                LOG.warning(
                    "Class dim mismatch for layer=%s unit=%d in method=%s: got %d expected %d",
                    layer_name,
                    unit,
                    state.method,
                    int(mean_vec.shape[0]),
                    num_classes,
                )
                continue
            mean_matrix[row_idx] = mean_vec
            gain_vec = payload.get("global_gain")
            if gain_vec is not None:
                gain_vec_np = np.array(gain_vec, dtype=np.float32, copy=False)
                if gain_vec_np.shape[0] == num_classes:
                    global_gain_matrix[row_idx] = gain_vec_np
                else:
                    LOG.warning(
                        "Global gain dim mismatch for layer=%s unit=%d in method=%s: got %d expected %d",
                        layer_name,
                        unit,
                        state.method,
                        int(gain_vec_np.shape[0]),
                        num_classes,
                    )
            max_abs_arr[row_idx] = float(payload.get("max_abs", np.max(np.abs(mean_vec)) if mean_vec.size > 0 else 0.0))
            mean_act_arr[row_idx] = float(payload["mean_act"])
            for dec in decile_ids:
                dec_vec = payload["dec_vecs"].get(int(dec))
                if dec_vec is not None:
                    dec_vec_np = np.array(dec_vec, dtype=np.float32, copy=False)
                    if dec_vec_np.shape[0] == num_classes:
                        decile_matrices[int(dec)][row_idx] = dec_vec_np
                        decile_maxabs[int(dec)][row_idx] = (
                            float(np.max(np.abs(dec_vec_np))) if dec_vec_np.size > 0 else 0.0
                        )
                    else:
                        LOG.warning(
                            "Decile class dim mismatch for layer=%s unit=%d decile=%d in method=%s",
                            layer_name,
                            unit,
                            int(dec),
                            state.method,
                        )
                decile_mean_acts[int(dec)][row_idx] = float(payload["dec_mean_acts"].get(int(dec), 0.0))

        for layer_name in layer_order:
            offset = layer_offsets[layer_name]
            count = int(state.layer_feature_counts.get(layer_name, 0))
            for unit in range(count):
                row_idx = offset + unit
                payload = feature_lookup.get((layer_name, unit))
                base_entry = payload["entry"] if payload else {}
                dec_mean_map = payload["dec_mean_acts"] if payload else {}
                features_list.append(
                    {
                        "id": row_idx,
                        "layer": layer_name,
                        "spec": base_entry.get("spec", layer_name),
                        "unit": unit,
                        "sample_count": base_entry.get("sample_count", 0),
                        "max_abs": float(max_abs_arr[row_idx]),
                        "mean_activation": float(mean_act_arr[row_idx]),
                        "decile_mean_activations": (
                            {str(k): float(v) for k, v in dec_mean_map.items()} if write_decile_outputs else {}
                        ),
                    }
                )
    else:
        mean_matrix = np.stack(state.mean_vectors, axis=0).astype(np.float32, copy=False)
        if len(state.global_gains) == mean_matrix.shape[0]:
            global_gain_matrix = np.stack(state.global_gains, axis=0).astype(np.float32, copy=False)
        else:
            LOG.warning(
                "Global gain count mismatch for method '%s' (gains=%d, features=%d); filling zeros.",
                state.method,
                len(state.global_gains),
                mean_matrix.shape[0],
            )
            global_gain_matrix = np.zeros_like(mean_matrix)
        max_abs_arr = np.array([f.get("max_abs", 0.0) for f in state.feature_entries], dtype=np.float32)
        mean_act_arr = np.array(state.mean_acts or [0.0] * len(state.mean_vectors), dtype=np.float32)
        decile_matrices = {}
        decile_maxabs = {}
        decile_mean_acts = {}
        for dec in decile_ids:
            dec_means: List[np.ndarray] = []
            dec_maxabs: List[float] = []
            dec_meanacts: List[float] = []
            for idx, dec_map in enumerate(state.decile_vectors):
                vec = dec_map.get(int(dec))
                if vec is None:
                    vec = np.zeros_like(state.mean_vectors[idx])
                dec_vec = np.array(vec, dtype=np.float32, copy=False)
                dec_means.append(dec_vec)
                dec_maxabs.append(float(np.max(np.abs(dec_vec))) if dec_vec.size > 0 else 0.0)
                dec_meanacts.append(float(state.decile_mean_acts[idx].get(int(dec), 0.0)))
            decile_matrices[int(dec)] = np.stack(dec_means, axis=0).astype(np.float32, copy=False)
            decile_maxabs[int(dec)] = np.array(dec_maxabs, dtype=np.float32)
            decile_mean_acts[int(dec)] = np.array(dec_meanacts, dtype=np.float32)
        features_list = [
            {
                "id": idx,
                "layer": f["layer"],
                "spec": f["spec"],
                "unit": f["unit"],
                "sample_count": f["sample_count"],
                "max_abs": f["max_abs"],
                "mean_activation": f.get("mean_activation", 0.0),
                "decile_mean_activations": f.get("decile_mean_activations", {}) if write_decile_outputs else {},
            }
            for idx, f in enumerate(state.feature_entries)
        ]

    min_sample_count = int(args.min_sample_count)
    global_gain_topn = int(args.global_gain_topn)
    required_sample_count = int(global_gain_topn) if int(global_gain_topn) > 0 else int(min_sample_count)
    features_meta = {
        "config_path": str(args.config),
        "deciles": list(deciles),
        "topn_per_decile": int(args.topn_per_decile),
        "min_sample_count": int(min_sample_count),
        "min_decile0_count": int(min_sample_count),
        "required_sample_count": int(required_sample_count),
        "global_gain_topn": int(global_gain_topn),
        "write_decile_outputs": bool(write_decile_outputs),
        "method": state.method,
        "baseline_cache": str(args.baseline_cache) if args.baseline_cache else None,
        "contrib_captures": list(contrib_captures),
        "contrib_ig_active": list(contrib_ig_active),
        "baseline": str(args.baseline),
        "ig_steps": int(args.ig_steps),
        "smoothgrad_samples": int(args.smoothgrad_samples),
        "smoothgrad_sigma": float(args.smoothgrad_sigma),
        "smoothgrad_noise_mode": str(args.smoothgrad_noise_mode),
        "batch_size": int(args.batch_size),
        "unit_save_format": str(args.unit_save_format),
        "write_class_rankings": bool(args.write_class_rankings),
        "activation_patch_sweep": list(activation_patch_sweep),
        "num_features": int(mean_matrix.shape[0]),
        "num_classes": int(mean_matrix.shape[1]),
        "features": features_list,
    }
    eps = np.finfo(np.float32).eps
    denom_max = np.maximum(max_abs_arr[:, None], eps)
    denom_act = np.maximum(mean_act_arr[:, None], eps)
    norm_max_matrix = mean_matrix / denom_max
    norm_act_matrix = mean_matrix / denom_act

    class_rankings_scenarios: Optional[Dict[str, Any]] = None
    topk = max(1, int(args.class_topk))
    if args.write_class_rankings:
        class_rankings_scenarios = {}
        # Overall scenarios
        class_rankings_scenarios["all"] = {
            "normalizations": {
                "raw": _build_classes(mean_matrix, topk),
                "max_abs": _build_classes(norm_max_matrix, topk),
                "mean_activation": _build_classes(norm_act_matrix, topk),
            }
        }
        # Decile-specific
        for dec in decile_ids:
            dec_matrix = decile_matrices[int(dec)]
            dec_max = np.maximum(decile_maxabs[int(dec)][:, None], eps)
            dec_act = np.maximum(decile_mean_acts[int(dec)][:, None], eps)
            class_rankings_scenarios[f"decile_{dec}"] = {
                "normalizations": {
                    "raw": _build_classes(dec_matrix, topk),
                    "max_abs": _build_classes(dec_matrix / dec_max, topk),
                    "mean_activation": _build_classes(dec_matrix / dec_act, topk),
                }
            }

    def _write_artifacts(root: Path) -> Dict[str, str]:
        root.mkdir(parents=True, exist_ok=True)
        mean_path = root / "mean_contributions.npy"
        np.save(mean_path, mean_matrix)
        gain_path = root / "global_gain.npy"
        np.save(gain_path, global_gain_matrix)
        np.save(root / "max_abs.npy", max_abs_arr)
        np.save(root / "mean_activation.npy", mean_act_arr)

        for dec in decile_ids:
            dec_matrix = decile_matrices[int(dec)]
            np.save(root / f"mean_contributions_decile{dec}.npy", dec_matrix)
            np.save(root / f"max_abs_decile{dec}.npy", decile_maxabs[int(dec)])
            np.save(root / f"mean_activation_decile{dec}.npy", decile_mean_acts[int(dec)])

        features_path = root / "features.json"
        with features_path.open("w", encoding="utf-8") as f:
            json.dump(features_meta, f, indent=2, ensure_ascii=False)

        outputs = {
            "features": str(features_path),
            "mean": str(mean_path),
            "global_gain": str(gain_path),
        }

        if class_rankings_scenarios is not None:
            class_path = root / "class_rankings.json"
            class_rankings = {
                "top_k": topk,
                "features_path": str(features_path),
                "scenarios": class_rankings_scenarios,
            }
            with class_path.open("w", encoding="utf-8") as f:
                json.dump(class_rankings, f, indent=2, ensure_ascii=False)
            outputs["class_rankings"] = str(class_path)
        return outputs

    outputs = _write_artifacts(state.root)
    if final_root is not None:
        final_method_root = final_root / state.method
        outputs["final_root"] = str(final_method_root)
        outputs["final"] = _write_artifacts(final_method_root)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute global SAE feature contributions.")
    parser.add_argument("--config", type=Path, default=Path("configs/clip_imagenet_index.yaml"))
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--layers", type=str, default='model.blocks.5', help="Comma-separated layer list (default: config).")
    parser.add_argument("--units", type=str, default=None, help="Comma-separated units (only for single layer).")
    parser.add_argument("--max-units", type=int, default=None, help="Optional cap on units per layer.")
    parser.add_argument(
        "--unit-shard-count",
        type=int,
        default=1,
        help="Total number of unit shards for manual parallelism (use >1 to split units across runs).",
    )
    parser.add_argument(
        "--unit-shard-idx",
        type=int,
        default=0,
        help="Index of the unit shard to process (0-based; must be < unit-shard-count).",
    )
    parser.add_argument(
        "--deciles",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Deciles to sample (comma-separated).",
    )
    parser.add_argument(
        "--decile-base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Decile indexing base (default=1).",
    )
    parser.add_argument(
        "--topn-per-decile",
        type=int,
        default=10,
        help="Rows to pull per decile for global top-N selection.",
    )
    parser.add_argument(
        "--min-sample-count",
        "--min-decile0-count",
        dest="min_sample_count",
        type=int,
        default=1,
        help="Minimum samples required before using a unit (<=0 disables check).",
    )
    parser.add_argument(
        "--global-gain-topn",
        type=int,
        default=10,
        help="Global top-N after merging per-decile topn (<=0 uses all).",
    )
    parser.add_argument(
        "--write-decile-outputs",
        action="store_true",
        help="Write decile mean contribution artifacts (disabled by default).",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Samples per forward/contribution batch.")
    parser.add_argument(
        "--methods",
        type=str,
        default="logitprism,second_order_lens,activation_patch,ig,grad,input_x_grad,smoothgrad_input_x_grad,attnlrp_input_x_grad,libragrad_input_x_grad,libragrad_ig,lens",
        help="Comma-separated contribution methods to compute.",
    )
    parser.add_argument(
        "--contrib-method",
        type=str,
        default=None,
        help="Deprecated alias for --methods with a single method.",
    )
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--baseline", type=str, default="zeros", help="Baseline for forward contribution runtime.")
    parser.add_argument("--baseline-cache", type=Path, default=None, help="Optional baseline cache for activation_patch.")
    parser.add_argument("--libragrad-gamma", type=float, default=None, help="Gamma for libragrad (optional).")
    parser.add_argument("--libragrad-alpha", type=float, default=None, help="Alpha for libragrad (optional).")
    parser.add_argument("--libragrad-beta", type=float, default=None, help="Beta for libragrad (optional).")
    parser.add_argument("--contrib-captures", type=str, default="model.head")
    parser.add_argument("--contrib-ig-active", type=str, default=None)
    parser.add_argument(
        "--smoothgrad-samples",
        type=int,
        default=8,
        help="Noise samples for smoothgrad/vargrad.",
    )
    parser.add_argument(
        "--smoothgrad-sigma",
        type=float,
        default=0.5,
        help="Noise std (fixed) or scale factor (proportional/std).",
    )
    parser.add_argument(
        "--smoothgrad-noise-mode",
        type=str,
        default="fixed",
        choices=["proportional", "fixed", "std"],
        help="Noise std mode for SAE features (smoothgrad/vargrad).",
    )
    parser.add_argument(
        "--activation-patch-sweep",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5",
        help="Comma-separated multipliers for activation_patch_sweep.",
    )
    parser.add_argument("--samples-dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument(
        "--unit-save-format",
        type=str,
        default="pt",
        choices=["npz", "pt"],
        help="Per-unit contribution file format.",
    )
    parser.add_argument("--compress", action="store_true", help="Use np.savez_compressed for npz outputs.")
    parser.add_argument("--skip-existing", default=True, help="Skip units that already have saved outputs.")
    parser.add_argument(
        "--write-class-rankings",
        action="store_true",
        help="Write class_rankings.json (can be large).",
    )
    parser.add_argument("--class-topk", type=int, default=100, help="Top-K features per class to store.")
    parser.add_argument("--progress-every", type=int, default=50, help="Log progress every N units.")
    parser.add_argument("--tqdm", action="store_true", help="Show tqdm progress per layer (requires tqdm).")
    parser.add_argument("--reverse-units", action="store_true", help="Process resolved units in reverse order.")
    parser.add_argument(
        "--final-out-dir",
        type=Path,
        default=Path("final"),
        help="Write lightweight final artifacts under this directory (relative to --out-dir unless absolute).",
    )
    parser.add_argument(
        "--skip-final-out",
        action="store_true",
        help="Do not write the final artifact bundle.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sae_global_contributions"))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    _configure_logger(args.log_file)

    methods_raw = args.contrib_method or args.methods
    methods = _unique_lower(_parse_csv_list(methods_raw) if methods_raw else [])
    if not methods:
        raise ValueError("No methods parsed; use --methods")

    activation_patch_sweep = _parse_float_list(args.activation_patch_sweep)
    if not activation_patch_sweep:
        activation_patch_sweep = [0.0]

    index_cfg = _load_yaml(args.config)
    device = torch.device(
        args.device
        or index_cfg.get("model", {}).get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dataset_cfg = dict(index_cfg.get("dataset", {}))
    transform = build_clip_transform(dataset_cfg, is_train=False)

    model_loader = load_obj(index_cfg["model"]["loader"])
    model = model_loader(index_cfg["model"], device=device).eval()
    adapter = CLIPVisionAdapter(model, device=device)

    ledger_root = Path(index_cfg.get("indexing", {}).get("out_dir", ""))
    if not ledger_root:
        raise ValueError("index config missing indexing.out_dir for decile ledger")
    dec_ledger = DecileParquetLedger(ledger_root)
    try:
        offline_ledger = build_offline_ledger(index_cfg)
    except Exception as exc:
        raise RuntimeError(f"Failed to build offline ledger: {exc}") from exc

    if args.layers:
        layers = _parse_csv_list(args.layers)
    else:
        layers = [str(l) for l in index_cfg.get("sae", {}).get("layers", [])]
    if not layers:
        raise ValueError("No layers resolved from config or --layers")

    units_filter: Optional[List[int]] = None
    if args.units:
        units_filter = [int(tok) for tok in _parse_csv_list(args.units)]
        if len(layers) != 1:
            raise ValueError("--units only supported with a single layer")
    shard_count = max(1, int(args.unit_shard_count))
    shard_idx = int(args.unit_shard_idx)
    if shard_idx < 0 or shard_idx >= shard_count:
        raise ValueError(f"--unit-shard-idx must be in [0, {shard_count})")
    if shard_count > 1:
        LOG.warning(
            "Unit sharding active (idx=%d/%d); aggregated outputs will include only this shard's units. "
            "Run an aggregate pass after all shards complete if you need full-layer summaries.",
            shard_idx,
            shard_count,
        )

    deciles = _parse_deciles(args.deciles, base=int(args.decile_base))
    if not deciles:
        raise ValueError("No deciles parsed; use --deciles")
    min_sample_count = int(args.min_sample_count)
    global_gain_topn = int(args.global_gain_topn)
    write_decile_outputs = bool(args.write_decile_outputs)
    if write_decile_outputs:
        LOG.info(
            "Decile outputs reflect the global top-N subset (not per-decile top-N)."
        )
    if int(args.topn_per_decile) <= 0:
        raise ValueError("--topn-per-decile must be > 0 for global top-N selection")
    if global_gain_topn > 0 and int(args.topn_per_decile) < global_gain_topn:
        LOG.warning(
            "topn_per_decile (%d) < global_gain_topn (%d); global top-N may be incomplete.",
            int(args.topn_per_decile),
            int(global_gain_topn),
        )

    baseline_cache = _load_baseline_cache(args.baseline_cache)
    if baseline_cache is not None:
        LOG.info("Loaded baseline cache: %s", args.baseline_cache)

    contrib_captures = _parse_csv_list(args.contrib_captures) or ["model.head"]
    contrib_ig_active = _parse_csv_list(args.contrib_ig_active) if args.contrib_ig_active else contrib_captures

    default_out_dir = Path("outputs/sae_global_contributions")
    out_dir = Path(args.out_dir)
    if len(layers) == 1 and out_dir == default_out_dir:
        anchor_spec = _layer_to_latent_spec(layers[0])
        out_dir = out_dir / anchor_spec
        LOG.info("Using default out_dir scoped by anchor: %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_root = out_dir / "methods"
    method_root.mkdir(parents=True, exist_ok=True)
    final_out_dir: Optional[Path] = None
    if not args.skip_final_out:
        final_out_dir = Path(args.final_out_dir)
        if not final_out_dir.is_absolute():
            final_out_dir = out_dir / final_out_dir
        final_out_dir.mkdir(parents=True, exist_ok=True)

    sample_dtype = np.float16 if args.samples_dtype == "float16" else np.float32
    batch_size = max(1, int(args.batch_size))
    force_single_batch = global_gain_topn > 0
    effective_batch_size = int(global_gain_topn) if force_single_batch else int(batch_size)
    if force_single_batch and int(batch_size) != int(global_gain_topn):
        LOG.info(
            "Forcing single batch of size %d for global top-N (ignoring --batch-size=%d).",
            int(global_gain_topn),
            int(batch_size),
        )
    args.batch_size = int(effective_batch_size)
    path_cache: Dict[int, str] = {}
    method_states: Dict[str, MethodState] = {}
    num_classes: Optional[int] = None

    for method in methods:
        method_dir = method_root / method
        method_states[method] = MethodState(
            method=method,
            root=method_dir,
            per_feature_root=method_dir / "per_feature",
            feature_entries=[],
            mean_vectors=[],
            global_gains=[],
            mean_acts=[],
            decile_vectors=[],
            decile_mean_acts=[],
            layer_feature_counts={},
            layer_order=[],
        )

    for layer in layers:
        parsed_layer = parse_spec(layer)
        layer_key = parsed_layer.base_with_branch or parsed_layer.base or layer
        layer_spec = _layer_to_latent_spec(layer)
        LOG.info("Layer %s (ledger=%s, spec=%s)", layer, layer_key, layer_spec)
        sae = _load_sae_for_layer(index_cfg["sae"], layer_spec, device)
        controller = AllTokensFeatureOverrideController(
            spec=OverrideSpec(lane_idx=None, unit_indices=None),
            frame_getter=None,
        )
        capture = LayerCapture(layer_spec)
        owner = resolve_module(model, capture.base)
        restore_handle, branch = wrap_target_layer_with_sae(
            owner,
            capture=capture,
            sae=sae,
            controller=controller,
            frame_getter=None,
        )
        anchors: Dict[str, tca.AnchorInfo] = {
            layer_spec: tca.AnchorInfo(
                spec=layer_spec,
                attr_name=tca._anchor_attr_name(layer_spec),
                controller=controller,
                branch=branch,
                sae=sae,
                owner=owner,
                capture=capture,
                restore=restore_handle,
            )
        }
        autoreset_handle = install_controller_autoreset_hooks(
            model,
            [controller],
            branches=[branch],
        )

        feature_count: Optional[int] = None
        for attr in ("W_dec", "W_enc"):
            mat = getattr(sae, attr, None)
            if mat is not None and hasattr(mat, "shape"):
                try:
                    feature_count = int(mat.shape[0])
                    break
                except Exception:
                    pass
        if feature_count is None and hasattr(sae, "d_hidden"):
            try:
                feature_count = int(sae.d_hidden)
            except Exception:
                pass
        if feature_count is None and hasattr(sae, "decoder"):
            dec = getattr(sae, "decoder")
            if hasattr(dec, "weight"):
                try:
                    feature_count = int(dec.weight.shape[0])
                except Exception:
                    pass
        units = dec_ledger.units_for_layer(layer_key)
        if feature_count is None and units:
            try:
                feature_count = int(max(int(u) for u in units)) + 1
                LOG.warning(
                    "Feature count fallback for layer %s: using ledger max unit + 1 = %d",
                    layer,
                    feature_count,
                )
            except Exception:
                feature_count = None
        if feature_count is None:
            LOG.warning("Could not determine feature count for layer %s; falling back to observed features only", layer)
        else:
            for state in method_states.values():
                if layer not in state.layer_feature_counts:
                    state.layer_feature_counts[layer] = int(feature_count)
                    state.layer_order.append(layer)
        if units_filter is not None:
            units = [u for u in units if u in units_filter]
        if args.max_units is not None:
            units = units[: max(0, int(args.max_units))]
        if shard_count > 1:
            units = [u for u in units if u % shard_count == shard_idx]
            LOG.info(
                "Unit sharding: shard %d/%d -> %d units for layer %s",
                shard_idx,
                shard_count,
                len(units),
                layer,
            )
        if args.reverse_units:
            units = list(reversed(units))

        if not units:
            LOG.warning("No units found for layer %s", layer)
            if autoreset_handle is not None:
                try:
                    autoreset_handle.remove()
                except Exception:
                    pass
            for anchor in anchors.values():
                try:
                    anchor.restore()
                except Exception:
                    pass
            continue

        total_units = len(units)
        eligible_units: List[int] = []
        unit_iter = enumerate(units, start=1)
        if args.tqdm and tqdm is not None:
            unit_iter = tqdm(
                enumerate(units, start=1),
                total=total_units,
                desc=f"{layer_key} shard {shard_idx}/{shard_count}",
                leave=False,
            )
        for idx_unit, unit in unit_iter:
            rows = _collect_decile_rows(
                dec_ledger,
                layer=layer_key,
                unit=int(unit),
                deciles=deciles,
                topn_per_decile=int(args.topn_per_decile),
            )
            if not rows:
                continue
            if int(global_gain_topn) > 0:
                if len(rows) < int(global_gain_topn):
                    LOG.info(
                        "Skip unit=%d for layer=%s: need top-%d rows but only have %d",
                        int(unit),
                        layer_key,
                        int(global_gain_topn),
                        int(len(rows)),
                    )
                    continue
                top_idx = _topn_indices_by_score(rows, int(global_gain_topn))
                rows = [rows[int(i)] for i in top_idx]
            elif min_sample_count > 0 and len(rows) < int(min_sample_count):
                LOG.info(
                    "Skip unit=%d for layer=%s: sample count below %d (count=%d)",
                    int(unit),
                    layer_key,
                    int(min_sample_count),
                    int(len(rows)),
                )
                continue

            sample_ids = [r.sample_id for r in rows]
            path_map = _lookup_paths(offline_ledger, sample_ids, cache=path_cache)
            for r in rows:
                r.path = path_map.get(int(r.sample_id), "")

            single_raw_batch: Optional[Dict[str, Any]] = None
            single_rows: List[SampleRecord] = []
            if force_single_batch:
                raw_batch, batch_rows = _build_image_batch_multi(transform=transform, rows=rows)
                if raw_batch is None or len(batch_rows) != len(rows):
                    LOG.info(
                        "Skip unit=%d for layer=%s: missing %d/%d images for global top-N batch",
                        int(unit),
                        layer_key,
                        int(len(rows) - len(batch_rows)),
                        int(len(rows)),
                    )
                    continue
                if int(global_gain_topn) > 0 and len(batch_rows) != int(global_gain_topn):
                    LOG.info(
                        "Skip unit=%d for layer=%s: need %d images but loaded %d",
                        int(unit),
                        layer_key,
                        int(global_gain_topn),
                        int(len(batch_rows)),
                    )
                    continue
                single_raw_batch = raw_batch
                single_rows = list(batch_rows)
            eligible_units.append(int(unit))

            processing_logged = False
            for method in methods:
                if method == "lens":
                    continue  # handled separately without samples
                state = method_states[method]
                layer_dir = state.per_feature_root / layer_key
                layer_dir.mkdir(parents=True, exist_ok=True)
                unit_path, existing_path = _resolve_unit_paths(
                    layer_dir, int(unit), str(args.unit_save_format)
                )
                meta_path = layer_dir / f"unit_{int(unit)}.json"
                if args.skip_existing and existing_path is not None:
                    payload = _load_unit_payload(existing_path)
                    recompute = False
                    if payload is None:
                        LOG.warning("Missing mean in %s; recomputing unit %d", existing_path, unit)
                        recompute = True
                    else:
                        mean_vec = payload["mean_vec"]
                        count = int(payload["sample_count"])
                        if int(global_gain_topn) > 0 and count != int(global_gain_topn):
                            LOG.warning(
                                "Sample count mismatch in %s (got %d, expected %d); recomputing unit %d",
                                existing_path,
                                int(count),
                                int(global_gain_topn),
                                unit,
                            )
                            recompute = True
                        if mean_vec.size > 0 and num_classes is not None and mean_vec.shape[0] != num_classes:
                            LOG.warning(
                                "Class dim mismatch in %s (got %d, expected %d); recomputing unit %d",
                                existing_path,
                                int(mean_vec.shape[0]),
                                int(num_classes),
                                unit,
                            )
                            recompute = True
                        global_gain = payload.get("global_gain", np.array([], dtype=np.float32))
                        if global_gain.size > 0 and mean_vec.size > 0 and global_gain.shape[0] != mean_vec.shape[0]:
                            LOG.warning(
                                "Global gain dim mismatch in %s (got %d, expected %d); recomputing unit %d",
                                existing_path,
                                int(global_gain.shape[0]),
                                int(mean_vec.shape[0]),
                                unit,
                            )
                            recompute = True
                    if not recompute and payload is not None:
                        max_abs = float(payload.get("max_abs", 0.0))
                        mean_act = float(payload.get("mean_act", 0.0))
                        if global_gain.size == 0 and mean_vec.size > 0:
                            global_gain = np.zeros_like(mean_vec)
                        decile_vectors = {} if not write_decile_outputs else payload.get("dec_vecs", {})
                        decile_mean_acts = {} if not write_decile_outputs else payload.get("dec_mean_acts", {})
                        state.feature_entries.append(
                            {
                                "layer": layer,
                                "layer_ledger": layer_key,
                                "spec": layer_spec,
                                "unit": int(unit),
                                "sample_count": count,
                                "max_abs": max_abs,
                                "mean_activation": mean_act,
                                "decile_mean_activations": decile_mean_acts,
                            }
                        )
                        state.mean_vectors.append(mean_vec)
                        state.global_gains.append(global_gain)
                        state.mean_acts.append(mean_act)
                        state.decile_vectors.append(decile_vectors)
                        state.decile_mean_acts.append(decile_mean_acts)
                        if num_classes is None and mean_vec.size > 0:
                            num_classes = int(mean_vec.shape[0])
                        continue
                    if recompute:
                        LOG.info(
                            "Recomputing method=%s layer=%s unit=%d (path=%s)",
                            method,
                            layer,
                            int(unit),
                            existing_path,
                        )

                contrib_list: List[np.ndarray] = []
                kept_rows: List[SampleRecord] = []
                activation_mags: List[float] = []
                activation_vals: List[float] = []
                ctx_mgr = tca._method_context(
                    method,
                    model,
                    anchors,
                    libragrad_gamma=args.libragrad_gamma,
                    libragrad_alpha=args.libragrad_alpha,
                    libragrad_beta=args.libragrad_beta,
                )
                with ctx_mgr:
                    def _iter_raw_batches():
                        if force_single_batch:
                            if single_raw_batch is None or not single_rows:
                                return
                            yield single_raw_batch, single_rows
                            return
                        for batch_chunk in _iter_batches(rows, batch_size):
                            raw_batch, batch_rows = _build_image_batch_multi(
                                transform=transform,
                                rows=batch_chunk,
                            )
                            if not batch_rows:
                                continue
                            yield raw_batch, batch_rows

                    for raw_batch, batch_rows in _iter_raw_batches():
                        batch = adapter.preprocess_input(raw_batch)
                        tca._clear_anchor_contexts(anchors, debug=False, reason=f"method-start:{method}")
                        with torch.no_grad():
                            logits = tca._extract_logits(model(batch["pixel_values"]))
                        base_logits_vec = _reduce_class_vec(logits, keep_batch=True).detach().cpu()
                        if base_logits_vec.dim() == 0:
                            base_logits_vec = base_logits_vec.reshape(1, 1)
                        elif base_logits_vec.dim() == 1:
                            base_logits_vec = base_logits_vec.reshape(1, -1)
                        anchor = anchors[layer_spec]
                        base = anchor.branch.sae_context().get(anchor.attr_name)
                        if base is None or not torch.is_tensor(base):
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        feat_dim = int(base.shape[-1])
                        if int(unit) >= feat_dim:
                            LOG.warning(
                                "Skip unit=%d for %s: feature dim %d",
                                unit,
                                layer_spec,
                                feat_dim,
                            )
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        mask = torch.zeros(feat_dim, dtype=torch.bool)
                        mask[int(unit)] = True
                        try:
                            _total_vec, per_feature = tca._compute_method_contributions(
                                method=method,
                                model=model,
                                adapter=adapter,
                                batch=batch,
                                anchors=anchors,
                                masks={layer_spec: mask},
                                forward_steps=int(args.ig_steps),
                                baseline=str(args.baseline),
                                contrib_captures=contrib_captures,
                                contrib_ig_active=contrib_ig_active,
                                baseline_cache=baseline_cache,
                                base_logits_vec=base_logits_vec,
                                activation_patch_sweep=activation_patch_sweep,
                                smoothgrad_samples=int(args.smoothgrad_samples),
                                smoothgrad_sigma=float(args.smoothgrad_sigma),
                                smoothgrad_noise_mode=str(args.smoothgrad_noise_mode),
                                debug=False,
                                keep_batch=True,
                            )
                        except Exception as exc:
                            LOG.warning(
                                "Contribution failed method=%s layer=%s unit=%d batch_size=%d: %s",
                                method,
                                layer,
                                unit,
                                len(batch_rows),
                                exc,
                            )
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        vec = None
                        for entry in per_feature:
                            if entry.get("spec") == layer_spec and int(entry.get("unit")) == int(unit):
                                vec = entry.get("vector")
                                break
                        if vec is None:
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        if not processing_logged:
                            LOG.info(
                                "Layer %s shard %d/%d processing unit=%d (%d/%d)",
                                layer,
                                shard_idx,
                                shard_count,
                                unit,
                                idx_unit,
                                total_units,
                            )
                            processing_logged = True
                        vec_np = vec.detach().cpu().float().numpy()
                        expected_classes = None
                        if base_logits_vec.dim() == 2 and base_logits_vec.shape[0] == len(batch_rows):
                            expected_classes = int(base_logits_vec.shape[1])
                        elif base_logits_vec.dim() == 1 and len(batch_rows) == 1:
                            expected_classes = int(base_logits_vec.shape[0])
                        elif num_classes is not None:
                            expected_classes = int(num_classes)
                        if expected_classes is None:
                            LOG.warning(
                                "Skip unit=%d method=%s: unable to infer class dim for batch=%d",
                                unit,
                                method,
                                int(len(batch_rows)),
                            )
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        vec_np = _coerce_batch_contrib(
                            vec_np,
                            expected_batch=int(len(batch_rows)),
                            expected_classes=expected_classes,
                        )
                        if vec_np is None:
                            LOG.warning(
                                "Skip unit=%d method=%s: contribution shape mismatch (batch=%d, classes=%d)",
                                unit,
                                method,
                                int(len(batch_rows)),
                                int(expected_classes),
                            )
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        if vec_np.shape[0] != len(batch_rows):
                            LOG.warning(
                                "Skip unit=%d method=%s: batch alignment mismatch (%d outputs for %d samples)",
                                unit,
                                method,
                                int(vec_np.shape[0]),
                                int(len(batch_rows)),
                            )
                            tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")
                            continue
                        act_mag_batch, act_val_batch = _unit_activation_stats(
                            base,
                            unit=int(unit),
                            expected_count=len(batch_rows),
                        )
                        for idx, row in enumerate(batch_rows):
                            vec_row = vec_np[idx]
                            if num_classes is None:
                                num_classes = int(vec_row.shape[0])
                            elif vec_row.shape[0] != num_classes:
                                LOG.warning(
                                    "Skip sample_id=%d for unit=%d: class dim %d != %d",
                                    row.sample_id,
                                    unit,
                                    int(vec_row.shape[0]),
                                    int(num_classes),
                                )
                                continue
                            contrib_list.append(vec_row)
                            kept_rows.append(row)
                            activation_mags.append(float(act_mag_batch[idx]))
                            activation_vals.append(float(act_val_batch[idx]))
                        tca._clear_anchor_contexts(anchors, debug=False, reason=f"post-batch:{method}")

                if not contrib_list:
                    continue

                contrib_arr = np.stack(contrib_list, axis=0)
                mean_vec = contrib_arr.mean(axis=0, dtype=np.float32)
                max_abs = float(np.max(np.abs(mean_vec))) if mean_vec.size > 0 else 0.0
                mean_act = float(np.mean(activation_mags)) if activation_mags else 0.0
                global_gain = _compute_global_gain(contrib_arr, np.array(activation_vals, dtype=np.float32))
                decile_mean_vecs: Dict[int, np.ndarray] = {}
                decile_mean_acts: Dict[int, float] = {}
                if write_decile_outputs:
                    decile_ids = sorted({r.decile for r in kept_rows})
                    deciles_arr = np.array([r.decile for r in kept_rows], dtype=np.int16)
                    for dec in decile_ids:
                        mask_dec = deciles_arr == int(dec)
                        if not np.any(mask_dec):
                            continue
                        contrib_dec = contrib_arr[mask_dec]
                        act_dec = np.array(activation_mags)[mask_dec] if activation_mags else np.array([])
                        dec_mean = contrib_dec.mean(axis=0, dtype=np.float32)
                        decile_mean_vecs[int(dec)] = dec_mean
                        decile_mean_acts[int(dec)] = float(act_dec.mean()) if act_dec.size > 0 else 0.0
                decile_ids_np = np.array(list(decile_mean_vecs.keys()), dtype=np.int16)
                decile_means_np = (
                    np.stack([decile_mean_vecs[d] for d in decile_mean_vecs.keys()], axis=0)
                    if decile_mean_vecs
                    else np.zeros((0, mean_vec.shape[0]), dtype=np.float32)
                )
                decile_mean_acts_np = (
                    np.array([decile_mean_acts.get(int(d), 0.0) for d in decile_mean_vecs.keys()], dtype=np.float32)
                    if decile_mean_acts
                    else np.zeros(0, dtype=np.float32)
                )
                topk_classes = min(5, mean_vec.shape[0]) if mean_vec.ndim == 1 else 0
                top_bundle_overall = _topk_bundle(mean_vec, max_abs=max_abs, mean_act=mean_act, k=topk_classes)
                decile_top_bundles = (
                    {
                        str(dec): _topk_bundle(
                            decile_mean_vecs[dec],
                            max_abs=float(np.max(np.abs(decile_mean_vecs[dec])))
                            if decile_mean_vecs[dec].size > 0
                            else 0.0,
                            mean_act=decile_mean_acts.get(dec, 0.0),
                            k=topk_classes,
                        )
                        for dec in decile_mean_vecs
                    }
                    if write_decile_outputs
                    else {}
                )

                _save_unit_payload(
                    unit_path,
                    compress=bool(args.compress),
                    contrib=contrib_arr.astype(sample_dtype, copy=False),
                    mean=mean_vec.astype(np.float32, copy=False),
                    global_gain=global_gain.astype(np.float32, copy=False),
                    mean_activation=np.array([mean_act], dtype=np.float32),
                    decile_ids=decile_ids_np,
                    decile_means=decile_means_np.astype(np.float32, copy=False),
                    decile_mean_acts=decile_mean_acts_np,
                    sample_ids=np.array([r.sample_id for r in kept_rows], dtype=np.int64),
                    deciles=np.array([r.decile for r in kept_rows], dtype=np.int16),
                    ranks=np.array([r.rank_in_decile for r in kept_rows], dtype=np.int16),
                )
                meta = {
                    "layer": layer,
                    "layer_ledger": layer_key,
                    "spec": layer_spec,
                    "unit": int(unit),
                    "method": method,
                    "mean_top_classes": {"overall": top_bundle_overall, "deciles": decile_top_bundles},
                    "mean_activation": float(mean_act),
                    "decile_mean_activations": {str(k): float(v) for k, v in decile_mean_acts.items()},
                    "samples": [
                        {
                            "sample_id": r.sample_id,
                            "decile": r.decile,
                            "rank_in_decile": r.rank_in_decile,
                            "score": r.score,
                            "path": r.path,
                            "frame_idx": r.frame_idx,
                            "y": r.y,
                            "x": r.x,
                            "prompt_id": r.prompt_id,
                            "uid": r.uid,
                            "stride_step": r.stride_step,
                            "run_id": r.run_id,
                        }
                        for r in kept_rows
                    ],
                }
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)
                rank_path = layer_dir / f"unit_{int(unit)}_topk.json"
                rank_payload = {
                    "layer": layer,
                    "layer_ledger": layer_key,
                    "spec": layer_spec,
                    "unit": int(unit),
                    "method": method,
                    "mean_top_classes": meta["mean_top_classes"],
                }
                with rank_path.open("w", encoding="utf-8") as f:
                    json.dump(rank_payload, f, indent=2, ensure_ascii=False)

                state.feature_entries.append(
                    {
                        "layer": layer,
                        "layer_ledger": layer_key,
                        "spec": layer_spec,
                        "unit": int(unit),
                        "sample_count": int(len(kept_rows)),
                        "max_abs": max_abs,
                        "mean_activation": mean_act,
                        "decile_mean_activations": decile_mean_acts,
                    }
                )
                state.mean_vectors.append(mean_vec)
                state.global_gains.append(global_gain)
                state.mean_acts.append(mean_act)
                state.decile_vectors.append(decile_mean_vecs)
                state.decile_mean_acts.append(decile_mean_acts)

            if args.progress_every > 0 and (idx_unit % int(args.progress_every) == 0 or idx_unit == total_units):
                LOG.info(
                    "Progress layer=%s shard=%d/%d: %d/%d units done",
                    layer,
                    shard_idx,
                    shard_count,
                    idx_unit,
                    total_units,
                )

        if autoreset_handle is not None:
            try:
                autoreset_handle.remove()
            except Exception:
                pass
        for anchor in anchors.values():
            try:
                anchor.restore()
            except Exception:
                pass
        # Lens method: projection-only (no samples). Compute after other methods to reuse sae.
        if "lens" in methods:
            lens_state = method_states["lens"]
            layer_dir = lens_state.per_feature_root / layer_key
            layer_dir.mkdir(parents=True, exist_ok=True)
            scores = _lens_score_matrix(sae, model)
            if scores is None:
                LOG.warning("Lens scores unavailable for layer %s (missing W_dec or head.weight)", layer_key)
            else:
                scores_np = scores.float().numpy()
                num_classes = num_classes or int(scores_np.shape[1])
                units_lens = eligible_units
                for unit in units_lens:
                    if int(unit) >= scores_np.shape[0]:
                        continue
                    unit_path, existing_path = _resolve_unit_paths(
                        layer_dir, int(unit), str(args.unit_save_format)
                    )
                    meta_path = layer_dir / f"unit_{int(unit)}.json"
                    rank_path = layer_dir / f"unit_{int(unit)}_topk.json"
                    if args.skip_existing and existing_path is not None:
                        payload = _load_unit_payload(existing_path)
                        recompute = False
                        if payload is not None:
                            mean_vec = payload["mean_vec"]
                            if num_classes is not None and mean_vec.size > 0 and mean_vec.shape[0] != num_classes:
                                LOG.warning(
                                    "Lens skip-existing class dim mismatch in %s (got %d, expected %d); recomputing",
                                    existing_path,
                                    int(mean_vec.shape[0]),
                                    int(num_classes),
                                )
                                recompute = True
                            global_gain = payload.get("global_gain", np.array([], dtype=np.float32))
                            if global_gain.size > 0 and mean_vec.size > 0 and global_gain.shape[0] != mean_vec.shape[0]:
                                LOG.warning(
                                    "Lens global_gain dim mismatch in %s (got %d, expected %d); recomputing",
                                    existing_path,
                                    int(global_gain.shape[0]),
                                    int(mean_vec.shape[0]),
                                )
                                recompute = True
                            if not recompute:
                                max_abs = float(payload.get("max_abs", 0.0))
                                if global_gain.size == 0 and mean_vec.size > 0:
                                    global_gain = np.zeros_like(mean_vec)
                                lens_state.feature_entries.append(
                                    {
                                        "layer": layer,
                                        "layer_ledger": layer_key,
                                        "spec": layer_spec,
                                        "unit": int(unit),
                                        "sample_count": 0,
                                        "max_abs": max_abs,
                                        "mean_activation": 0.0,
                                        "decile_mean_activations": {},
                                    }
                                )
                                lens_state.mean_vectors.append(mean_vec)
                                lens_state.global_gains.append(global_gain)
                                lens_state.mean_acts.append(0.0)
                                lens_state.decile_vectors.append({})
                                lens_state.decile_mean_acts.append({})
                                LOG.info(
                                    "Skip existing method=%s layer=%s unit=%d (path=%s)",
                                    "lens",
                                    layer,
                                    int(unit),
                                    existing_path,
                                )
                                continue
                    mean_vec = scores_np[int(unit)].astype(np.float32, copy=False)
                    if num_classes is not None and mean_vec.size > 0 and mean_vec.shape[0] != num_classes:
                        LOG.warning(
                            "Lens skip unit=%d: class dim %d != %d",
                            unit,
                            int(mean_vec.shape[0]),
                            int(num_classes),
                        )
                        continue
                    max_abs = float(np.max(np.abs(mean_vec))) if mean_vec.size > 0 else 0.0
                    global_gain = np.zeros_like(mean_vec)
                    topk_classes = min(5, mean_vec.shape[0]) if mean_vec.ndim == 1 else 0
                    top_bundle = _topk_bundle(mean_vec, max_abs=max_abs, mean_act=0.0, k=topk_classes)
                    _save_unit_payload(
                        unit_path,
                        compress=bool(args.compress),
                        contrib=np.zeros((0, mean_vec.shape[0]), dtype=sample_dtype),
                        mean=mean_vec.astype(np.float32, copy=False),
                        global_gain=global_gain.astype(np.float32, copy=False),
                        mean_activation=np.array([0.0], dtype=np.float32),
                        decile_ids=np.zeros(0, dtype=np.int16),
                        decile_means=np.zeros((0, mean_vec.shape[0]), dtype=np.float32),
                        decile_mean_acts=np.zeros(0, dtype=np.float32),
                        sample_ids=np.zeros(0, dtype=np.int64),
                        deciles=np.zeros(0, dtype=np.int16),
                        ranks=np.zeros(0, dtype=np.int16),
                    )
                    meta = {
                        "layer": layer,
                        "layer_ledger": layer_key,
                        "spec": layer_spec,
                        "unit": int(unit),
                        "method": "lens",
                        "mean_top_classes": {"overall": top_bundle, "deciles": {}},
                        "mean_activation": 0.0,
                        "decile_mean_activations": {},
                        "samples": [],
                    }
                    with meta_path.open("w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                    rank_payload = {
                        "layer": layer,
                        "layer_ledger": layer_key,
                        "spec": layer_spec,
                        "unit": int(unit),
                        "method": "lens",
                        "mean_top_classes": meta["mean_top_classes"],
                    }
                    with rank_path.open("w", encoding="utf-8") as f:
                        json.dump(rank_payload, f, indent=2, ensure_ascii=False)
                    lens_state.feature_entries.append(
                        {
                            "layer": layer,
                            "layer_ledger": layer_key,
                            "spec": layer_spec,
                            "unit": int(unit),
                            "sample_count": 0,
                            "max_abs": max_abs,
                            "mean_activation": 0.0,
                            "decile_mean_activations": {},
                        }
                    )
                    lens_state.mean_vectors.append(mean_vec)
                    lens_state.global_gains.append(global_gain)
                    lens_state.mean_acts.append(0.0)
                    lens_state.decile_vectors.append({})
                    lens_state.decile_mean_acts.append({})
        try:
            del sae
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest: Dict[str, Any] = {
        "config_path": str(args.config),
        "deciles": list(deciles),
        "topn_per_decile": int(args.topn_per_decile),
        "min_sample_count": int(min_sample_count),
        "min_decile0_count": int(min_sample_count),
        "required_sample_count": int(global_gain_topn) if int(global_gain_topn) > 0 else int(min_sample_count),
        "global_gain_topn": int(global_gain_topn),
        "write_decile_outputs": bool(write_decile_outputs),
        "unit_save_format": str(args.unit_save_format),
        "write_class_rankings": bool(args.write_class_rankings),
        "final_out_dir": str(final_out_dir) if final_out_dir is not None else None,
        "methods": {},
    }
    for method, state in method_states.items():
        outputs = _write_method_outputs(
            state=state,
            args=args,
            deciles=deciles,
            contrib_captures=contrib_captures,
            contrib_ig_active=contrib_ig_active,
            activation_patch_sweep=activation_patch_sweep,
            final_root=final_out_dir,
        )
        if outputs is None:
            continue
        manifest["methods"][method] = {
            "root": str(state.root),
            **outputs,
        }

    if not manifest["methods"]:
        LOG.warning("No contributions saved for any method; exiting.")
        return

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    LOG.info("Done. manifest=%s", manifest_path)


if __name__ == "__main__":
    main()
