#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Ensure repo root on sys.path and set CWD for src/ imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    os.chdir(REPO_ROOT)
except Exception:
    pass

from scripts import output_contribution_compare_sam2 as base
from scripts import output_contribution_compare_sam2_minfeat as minfeat
from scripts import output_contribution_compare_sam2_viz as viz
from src.core.attribution.smoothgrad import add_sae_noise, precompute_sae_noise_scales
from src.core.indexing.decile_parquet_ledger import DecileParquetLedger
from src.core.runtime.attribution_runtime import AnchorConfig, AttributionRuntime, ForwardConfig, RuntimeTarget


@dataclass
class MethodSpec:
    label: str
    base: str
    use_abs: bool
    use_positive: bool


def _sanitize_token(token: str) -> str:
    safe: List[str] = []
    for ch in str(token):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return ("".join(safe).strip("_")) or "value"


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


def _select_row(
    *,
    attr_cfg: Dict[str, Any],
    index_cfg: Dict[str, Any],
    sample_unit: Optional[int],
    decile: int,
    rank: int,
) -> Dict[str, Any]:
    ledger = DecileParquetLedger(index_cfg["indexing"]["out_dir"])
    parsed_layer = base.parse_spec(attr_cfg["indexing"]["layer"])
    ledger_layer = parsed_layer.base_with_branch
    unit = int(sample_unit) if sample_unit is not None else int(attr_cfg["indexing"]["unit"])
    table = ledger.topn_for(layer=ledger_layer, unit=unit, decile=int(decile), n=int(rank) + 1)
    rows = table.to_pylist() if table is not None else []
    if not rows or int(rank) >= len(rows):
        raise RuntimeError(f"No rows found for layer={ledger_layer} unit={unit} decile={decile} rank={rank}")
    return rows[int(rank)]


def _filter_active_ranking(
    ranking: List[Tuple[str, int, float]],
    active_masks: Optional[Dict[str, torch.Tensor]],
) -> List[Tuple[str, int, float]]:
    if not active_masks:
        return ranking
    filtered: List[Tuple[str, int, float]] = []
    for spec, feat_idx, score in ranking:
        mask = active_masks.get(spec)
        if mask is None:
            filtered.append((spec, feat_idx, score))
            continue
        if feat_idx < mask.numel() and bool(mask[feat_idx].item()):
            filtered.append((spec, feat_idx, score))
    return filtered


def _build_empty_masks(bases: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for spec, base_tensor in bases.items():
        feat_dim = base._base_feat_dim(base_tensor)
        if feat_dim <= 0:
            continue
        masks[spec] = torch.zeros(feat_dim, dtype=torch.bool)
    return masks


def _clear_masks(masks: Dict[str, torch.Tensor]) -> None:
    for mask in masks.values():
        mask.zero_()


def _set_insertion_overrides(
    *,
    anchors: Dict[str, base.AnchorInfo],
    bases: Dict[str, Any],
    masks: Dict[str, torch.Tensor],
    baseline_cache: Optional[base.ActivationBaselineCache],
) -> None:
    for spec, anchor in anchors.items():
        mask = masks.get(spec)
        base_tensor = bases.get(spec)
        if mask is None or base_tensor is None:
            continue
        attr_name = anchor.attr_name
        anchor.branch.set_anchor_override(
            attr_name,
            lambda t, m=mask, b=base_tensor, s=spec, a=attr_name, br=anchor.branch: base._apply_mask(
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


def _release_anchor_caches_keep_overrides(anchors: Dict[str, base.AnchorInfo]) -> None:
    for anchor in anchors.values():
        ctrl = anchor.controller
        if ctrl is not None:
            try:
                ctrl.release_cached_activations()
            except Exception:
                pass
        try:
            anchor.branch.clear_context()
        except Exception:
            pass


def _measure_objective(
    *,
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    anchors: Dict[str, base.AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    debug: bool = False,
) -> float:
    mask_capture.release_step_refs()
    mask_capture.clear_tapes()
    val, _, _ = base._measure_objective_value(
        forward_runner,
        mask_capture,
        lane_idx=lane_idx,
        threshold=threshold,
        ref_mask_spec=ref_mask_spec,
        fixed_mask=list(objective_mask),
        fixed_ref_logits=list(objective_ref_logits),
        objective_mode=objective_mode,
        debug=debug,
    )
    _release_anchor_caches_keep_overrides(anchors)
    mask_capture.release_step_refs()
    return float(val)


def _measure_objective_and_map(
    *,
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    anchors: Dict[str, base.AnchorInfo],
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    output_spec: str,
    num_frames: Optional[int],
    num_lanes: Optional[int],
    debug: bool = False,
) -> Tuple[float, torch.Tensor]:
    mask_capture.release_step_refs()
    mask_capture.clear_tapes()
    forward_runner(require_grad=False)
    mask_lists = mask_capture.get_tensor_lists(detach=True)
    obj, _, _ = base._objective_from_masks(
        mask_lists,
        lane_idx,
        threshold,
        ref_spec=ref_mask_spec,
        fixed_mask=list(objective_mask),
        fixed_ref_logits=list(objective_ref_logits),
        objective_mode=objective_mode,
    )
    output_tensor = viz._extract_spec_tensor(mask_lists, output_spec)
    if output_tensor is None:
        raise RuntimeError(f"Output spec '{output_spec}' not found in capture keys: {list(mask_lists.keys())}")
    output_map = viz._extract_output_map(
        output_tensor,
        lane_idx=lane_idx,
        num_frames=num_frames,
        num_lanes=num_lanes,
        debug=debug,
    )
    output_stack = viz._ensure_heat_stack(output_map.detach())
    _release_anchor_caches_keep_overrides(anchors)
    mask_capture.release_step_refs()
    return float(obj.detach().cpu().item()), output_stack


def _rankdata(values: Sequence[float]) -> List[float]:
    if len(values) == 0:
        return []
    pairs = sorted(((val, idx) for idx, val in enumerate(values)), key=lambda x: x[0])
    ranks = [0.0] * len(values)
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[pairs[k][1]] = rank
        i = j
    return ranks


def _pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.shape != yv.shape:
        return 0.0
    xm = float(xv.mean())
    ym = float(yv.mean())
    dx = xv - xm
    dy = yv - ym
    denom = float(np.sqrt((dx * dx).sum() * (dy * dy).sum()))
    if denom <= 0:
        return 0.0
    return float((dx * dy).sum() / denom)


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson_corr(rx, ry)


def _sign(val: float, eps: float) -> int:
    if val > eps:
        return 1
    if val < -eps:
        return -1
    return 0


def _scores_for_candidates(
    ranking: Sequence[Tuple[str, int, float]],
    candidates: Sequence[Tuple[str, int]],
) -> List[float]:
    score_map: Dict[Tuple[str, int], float] = {}
    for spec, unit, score in ranking:
        score_map[(str(spec), int(unit))] = float(score)
    return [float(score_map.get((str(spec), int(unit)), 0.0)) for spec, unit in candidates]


def _read_candidate_list(path: Path) -> List[Tuple[str, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("candidates", data.get("features", data))
        out: List[Tuple[str, int]] = []
        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    out.append((str(entry[0]), int(entry[1])))
                elif isinstance(entry, dict):
                    spec = entry.get("spec") or entry.get("layer") or entry.get("anchor")
                    unit = entry.get("unit") or entry.get("feature") or entry.get("idx")
                    if spec is None or unit is None:
                        continue
                    out.append((str(spec), int(unit)))
        return out
    out: List[Tuple[str, int]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = line.split("\t")
        elif "," in line:
            parts = line.split(",")
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        out.append((str(parts[0]).strip(), int(parts[1])))
    return out


def _parse_method_specs(
    raw: str,
    *,
    use_abs: bool,
    use_positive: bool,
    rank_exclude: Sequence[str],
) -> List[MethodSpec]:
    methods = [tok.strip() for tok in (raw or "").split(",") if tok.strip()]
    excluded = {base._normalize_method_name(tok) for tok in rank_exclude}
    specs: List[MethodSpec] = []
    for method in methods:
        label = method.strip()
        name = base._normalize_method_name(label)
        flag_abs = False
        flag_pos = False
        # Strip sign-correction suffixes (these don't affect ranking, only method maps)
        for sc_suffix in ("_sc", "_sign", "_signed"):
            if name.endswith(sc_suffix):
                name = name[: -len(sc_suffix)]
                break
        for suffix, flag in (("_abs", "abs"), ("_pos", "pos"), ("_positive", "pos")):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                if flag == "abs":
                    flag_abs = True
                else:
                    flag_pos = True
                break
        if flag_abs and flag_pos:
            flag_pos = True
            flag_abs = False
        if not flag_abs and not flag_pos:
            if name not in excluded:
                if use_positive:
                    flag_pos = True
                elif use_abs:
                    flag_abs = True
        specs.append(MethodSpec(label=label, base=name, use_abs=flag_abs, use_positive=flag_pos))
    return specs


def _compute_method_ranking(
    *,
    spec: MethodSpec,
    model: torch.nn.Module,
    anchors: Dict[str, base.AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[base.ActivationBaselineCache],
    base_objective: Optional[float],
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    batch_on_dev: Any,
    lane_idx: Optional[int],
    threshold: float,
    feature_active_threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    max_features: Optional[int],
    ig_steps: int,
    ig_active: Optional[Sequence[str]],
    smoothgrad_samples: int,
    smoothgrad_sigma: float,
    libragrad_gamma: Optional[float],
) -> Tuple[List[Tuple[str, int, float]], Optional[Dict[str, torch.Tensor]]]:
    base._clear_anchor_contexts(anchors, reason=f"method:{spec.label}")
    ranking: List[Tuple[str, int, float]] = []
    active_masks: Optional[Dict[str, torch.Tensor]] = None
    method = spec.base
    if method == "activation_patch":
        ranking, active_masks = base._build_activation_patching_ranking(
            forward_runner,
            mask_capture,
            anchors,
            bases,
            baseline_cache,
            lane_idx,
            threshold,
            feature_active_threshold,
            ref_mask_spec,
            max_features=max_features,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            base_objective=base_objective,
            objective_mask=list(objective_mask),
            progress=False,
        )
        if spec.use_positive:
            ranking = [(s, u, float(max(0.0, sc))) for s, u, sc in ranking]
        elif spec.use_abs:
            ranking = [(s, u, float(abs(sc))) for s, u, sc in ranking]
        ranking.sort(key=lambda x: x[2], reverse=True)
        return ranking, active_masks
    if method == "grad":
        ranking = base._build_grad_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            objective_ref_logits=list(objective_ref_logits),
            objective_mode=objective_mode,
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
        )
        return ranking, None
    if method == "input_x_grad":
        ranking = base._build_inputxgrad_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            objective_ref_logits=list(objective_ref_logits),
            objective_mode=objective_mode,
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
        )
        return ranking, None
    if method == "ig":
        ranking = base._build_ig_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            steps=ig_steps,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
            ig_active=ig_active,
        )
        return ranking, None
    if method in {"ig_anchor", "ig_anchored", "anchored_ig", "ig-anchor", "ig-anchored"}:
        ranking = base._build_ig_anchor_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            steps=ig_steps,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
            ig_active=ig_active,
        )
        return ranking, None
    if method in {"smoothgrad", "vargard"}:
        ranking = base._build_smooth_like_ranking(
            method,
            batch_on_dev=batch_on_dev,
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=list(objective_mask),
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            samples=smoothgrad_samples,
            noise_sigma=smoothgrad_sigma,
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
        )
        return ranking, None
    if method == "libragrad_input_x_grad":
        with base._LibragradContext(model, anchors, gamma=libragrad_gamma):
            ranking = base._build_inputxgrad_ranking(
                forward_runner,
                mask_capture,
                anchors,
                lane_idx,
                threshold,
                ref_mask_spec,
                list(objective_mask),
                objective_ref_logits=list(objective_ref_logits),
                objective_mode=objective_mode,
                use_abs=spec.use_abs,
                use_positive=spec.use_positive,
            )
        return ranking, None
    if method == "libragrad_ig":
        with base._LibragradContext(model, anchors, gamma=libragrad_gamma):
            ranking = base._build_ig_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            steps=ig_steps,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
            ig_active=ig_active,
            )
        return ranking, None
    if method in {"libragrad_ig_anchor", "libragrad_ig_anchored", "ligrad_ig_anchor"}:
        with base._LibragradContext(model, anchors, gamma=libragrad_gamma):
            ranking = base._build_ig_anchor_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            steps=ig_steps,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
            ig_active=ig_active,
            )
        return ranking, None
    if method == "attnlrp_grad":
        with base._AttnLRPContext(model, anchors):
            ranking = base._build_grad_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            objective_ref_logits=list(objective_ref_logits),
            objective_mode=objective_mode,
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
        )
        return ranking, None
    if method == "attnlrp_input_x_grad":
        with base._AttnLRPContext(model, anchors):
            ranking = base._build_inputxgrad_ranking(
                forward_runner,
                mask_capture,
                anchors,
                lane_idx,
                threshold,
                ref_mask_spec,
                list(objective_mask),
                objective_ref_logits=list(objective_ref_logits),
                objective_mode=objective_mode,
                use_abs=spec.use_abs,
                use_positive=spec.use_positive,
            )
        return ranking, None
    if method == "attnlrp_ig":
        with base._AttnLRPContext(model, anchors):
            ranking = base._build_ig_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            steps=ig_steps,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
            ig_active=ig_active,
            )
        return ranking, None
    if method in {"attnlrp_ig_anchor", "attnlrp_ig_anchored"}:
        with base._AttnLRPContext(model, anchors):
            ranking = base._build_ig_anchor_ranking(
            forward_runner,
            mask_capture,
            anchors,
            lane_idx,
            threshold,
            ref_mask_spec,
            list(objective_mask),
            steps=ig_steps,
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            use_abs=spec.use_abs,
            use_positive=spec.use_positive,
            ig_active=ig_active,
            )
        return ranking, None
    print(f"[skip] unknown method '{spec.label}'")
    return [], None


def _select_candidates(
    *,
    method_rankings: Dict[str, Dict[str, Any]],
    method_order: Sequence[str],
    method_aliases: Optional[Dict[str, str]],
    candidate_method: Optional[str],
    candidate_topk: int,
    candidate_union: bool,
    candidate_max: Optional[int],
    active_only: bool,
) -> Tuple[List[Tuple[str, int]], Dict[Tuple[str, int], List[str]]]:
    if not method_order:
        return [], {}

    def _filtered(method_label: str) -> List[Tuple[str, int, float]]:
        payload = method_rankings.get(method_label, {})
        ranking = payload.get("ranking") or []
        active_masks = payload.get("active_masks")
        return _filter_active_ranking(ranking, active_masks) if active_only else list(ranking)

    sources: Dict[Tuple[str, int], List[str]] = {}
    if candidate_union:
        rank_pos: Dict[Tuple[str, int], int] = {}
        for label in method_order:
            ranking = _filtered(label)
            for pos, (spec, unit, _score) in enumerate(ranking, start=1):
                if pos > candidate_topk:
                    break
                key = (str(spec), int(unit))
                if key not in sources:
                    sources[key] = []
                sources[key].append(label)
                if key not in rank_pos or pos < rank_pos[key]:
                    rank_pos[key] = pos
        ordered = sorted(rank_pos.items(), key=lambda kv: kv[1])
        candidates = [key for key, _pos in ordered]
        if candidate_max is not None:
            candidates = candidates[: int(candidate_max)]
        return candidates, {k: sources.get(k, []) for k in candidates}

    if candidate_method is None and method_order:
        candidate_method = str(method_order[0])
    if candidate_method not in method_rankings:
        alias = None
        if candidate_method and method_aliases:
            alias = method_aliases.get(base._normalize_method_name(candidate_method))
        if alias and alias in method_rankings:
            candidate_method = alias
        else:
            fallback = method_order[0] if method_order else None
            print(f"[warn] candidate_method '{candidate_method}' not found; using '{fallback}'")
            candidate_method = fallback
    ranking = _filtered(candidate_method) if candidate_method is not None else []
    candidates = [(str(spec), int(unit)) for spec, unit, _score in ranking[: int(candidate_topk)]]
    if candidate_max is not None:
        candidates = candidates[: int(candidate_max)]
    return candidates, {c: [str(candidate_method)] for c in candidates}


def _select_minfeat_candidates(
    *,
    method_rankings: Dict[str, Dict[str, Any]],
    method_order: Sequence[str],
    method_aliases: Optional[Dict[str, str]],
    candidate_method: Optional[str],
    candidate_union: bool,
    candidate_max: Optional[int],
    active_only: bool,
    metric: str,
    target: float,
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    anchors: Dict[str, base.AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[base.ActivationBaselineCache],
    chunk_size: int,
    max_steps: Optional[int],
    base_value: float,
    debug: bool,
) -> Tuple[
    List[Tuple[str, int]],
    Dict[Tuple[str, int], List[str]],
    Dict[str, Any],
    Optional[str],
    Dict[str, int],
]:
    if not method_order:
        return [], {}, {}, None

    def _resolve_method(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        if name in method_rankings:
            return name
        alias = None
        if method_aliases:
            alias = method_aliases.get(base._normalize_method_name(name))
        if alias and alias in method_rankings:
            return alias
        return None

    def _compute_for(label: str) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
        payload = method_rankings.get(label, {})
        ranking = payload.get("ranking") or []
        active_masks = payload.get("active_masks")
        min_feat, min_pct, last_val, last_iou, last_precision, last_leakage = minfeat._find_min_features(
            metric,
            target,
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=list(objective_mask),
            objective_mode=objective_mode,
            objective_ref_logits=list(objective_ref_logits),
            anchors=anchors,
            bases=bases,
            baseline_cache=baseline_cache,
            ranking=list(ranking),
            chunk_size=int(chunk_size),
            max_steps=max_steps,
            active_masks=active_masks,
            base_value=base_value,
            debug=debug,
            label=f"cand:{label}",
        )
        info = {
            "min_features": int(min_feat) if min_feat is not None else None,
            "min_fraction": float(min_pct) if min_pct is not None else None,
            "last_value": float(last_val),
            "last_iou": float(last_iou) if last_iou is not None else None,
            "last_precision": float(last_precision) if last_precision is not None else None,
            "last_leakage": float(last_leakage) if last_leakage is not None else None,
        }
        if min_feat is None or min_feat <= 0:
            return [], info
        filtered = _filter_active_ranking(ranking, active_masks) if active_only else list(ranking)
        feats = [(str(spec), int(unit)) for spec, unit, _score in filtered[: int(min_feat)]]
        return feats, info

    minfeat_info: Dict[str, Any] = {}
    minfeat_selected: Optional[str] = None
    sources: Dict[Tuple[str, int], List[str]] = {}

    computed: Dict[str, List[Tuple[str, int]]] = {}
    method_counts: Dict[str, int] = {}
    for label in method_order:
        feats, info = _compute_for(label)
        minfeat_info[label] = info
        if feats:
            computed[label] = feats
        method_counts[label] = len(feats)

    if candidate_union:
        candidates: List[Tuple[str, int]] = []
        for label in method_order:
            feats = computed.get(label, [])
            for key in feats:
                if key not in sources:
                    sources[key] = []
                sources[key].append(label)
                if key not in candidates:
                    candidates.append(key)
        if candidate_max is not None:
            candidates = candidates[: int(candidate_max)]
            sources = {k: sources.get(k, []) for k in candidates}
        return candidates, sources, minfeat_info, None, method_counts

    resolved = _resolve_method(candidate_method) if candidate_method else None
    if resolved is None:
        best_label = None
        best_min = None
        for label in method_order:
            info = minfeat_info.get(label, {})
            min_feat = info.get("min_features")
            if min_feat is None:
                continue
            if best_min is None or int(min_feat) < int(best_min):
                best_min = int(min_feat)
                best_label = label
        if best_label is None:
            best_label = method_order[0]
        resolved = best_label
        if candidate_method is not None and resolved != candidate_method:
            print(f"[warn] candidate_method '{candidate_method}' not found; using '{resolved}'")

    feats = computed.get(resolved, [])
    if candidate_max is not None:
        feats = feats[: int(candidate_max)]
    sources = {k: [resolved] for k in feats}
    return feats, sources, minfeat_info, resolved, method_counts


def _estimate_shapley(
    *,
    candidates: Sequence[Tuple[str, int]],
    anchors: Dict[str, base.AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[base.ActivationBaselineCache],
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    output_spec: Optional[str],
    num_frames: Optional[int],
    num_lanes: Optional[int],
    permutations: int,
    seed: int,
    debug: bool,
    shapley_map: bool = False,
    shapley_map_device: str = "cpu",
    shapley_map_dtype: str = "float16",
    shapley_map_topk: Optional[int] = None,
    progress_every: int = 0,
) -> Tuple[List[float], float, float, Optional[torch.Tensor], Optional[List[int]]]:
    if not candidates:
        return [], 0.0, 0.0, None, None

    masks = _build_empty_masks(bases)
    _clear_masks(masks)
    _set_insertion_overrides(
        anchors=anchors,
        bases=bases,
        masks=masks,
        baseline_cache=baseline_cache,
    )

    map_enabled = bool(shapley_map)
    map_scores: Optional[torch.Tensor] = None
    map_indices: Optional[List[int]] = None
    map_device = torch.device(shapley_map_device)
    map_dtype = torch.float16 if str(shapley_map_dtype).lower() in {"float16", "fp16"} else torch.float32
    if map_enabled:
        if output_spec is None:
            raise RuntimeError("shapley_map requested but output_spec is None.")
        map_topk = int(shapley_map_topk) if shapley_map_topk is not None and shapley_map_topk > 0 else len(candidates)
        map_topk = max(1, min(map_topk, len(candidates)))
        map_indices = list(range(map_topk))
        map_slot = {idx: slot for slot, idx in enumerate(map_indices)}

    if map_enabled:
        base_value, base_map = _measure_objective_and_map(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            output_spec=str(output_spec),
            num_frames=num_frames,
            num_lanes=num_lanes,
            debug=debug,
        )
        base_map = base_map.to(device=map_device, dtype=map_dtype)
        map_shape = tuple(base_map.shape)
        map_scores = torch.zeros((len(map_indices),) + map_shape, dtype=map_dtype, device=map_device)
    else:
        base_value = _measure_objective(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            debug=debug,
        )

    _clear_masks(masks)
    for spec, unit in candidates:
        mask = masks.get(spec)
        if mask is None or unit < 0 or unit >= mask.numel():
            continue
        mask[int(unit)] = True
    if map_enabled:
        full_value, _full_map = _measure_objective_and_map(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            output_spec=str(output_spec),
            num_frames=num_frames,
            num_lanes=num_lanes,
            debug=debug,
        )
    else:
        full_value = _measure_objective(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            debug=debug,
        )

    scores = [0.0 for _ in candidates]
    order = list(range(len(candidates)))
    rng = random.Random(int(seed))
    total_perms = int(permutations)
    for idx_perm in range(total_perms):
        rng.shuffle(order)
        _clear_masks(masks)
        current = base_value
        current_map = base_map if map_enabled else None
        for idx in order:
            spec, unit = candidates[idx]
            mask = masks.get(spec)
            if mask is None or unit < 0 or unit >= mask.numel():
                continue
            mask[int(unit)] = True
            if map_enabled:
                val, new_map = _measure_objective_and_map(
                    forward_runner=forward_runner,
                    mask_capture=mask_capture,
                    anchors=anchors,
                    lane_idx=lane_idx,
                    threshold=threshold,
                    ref_mask_spec=ref_mask_spec,
                    objective_mask=objective_mask,
                    objective_ref_logits=objective_ref_logits,
                    objective_mode=objective_mode,
                    output_spec=str(output_spec),
                    num_frames=num_frames,
                    num_lanes=num_lanes,
                    debug=debug,
                )
                new_map = new_map.to(device=map_device, dtype=map_dtype)
                if map_scores is not None and current_map is not None:
                    slot = map_slot.get(idx) if map_enabled else None
                    if slot is not None:
                        map_scores[slot] += new_map - current_map
                current_map = new_map
            else:
                val = _measure_objective(
                    forward_runner=forward_runner,
                    mask_capture=mask_capture,
                    anchors=anchors,
                    lane_idx=lane_idx,
                    threshold=threshold,
                    ref_mask_spec=ref_mask_spec,
                    objective_mask=objective_mask,
                    objective_ref_logits=objective_ref_logits,
                    objective_mode=objective_mode,
                    debug=debug,
                )
            delta = float(val - current)
            scores[idx] += delta
            current = val
        if _should_log_progress(idx_perm + 1, total_perms, progress_every):
            print(f"[shapley] perm {idx_perm + 1}/{total_perms}")
        elif debug and (idx_perm + 1) % max(1, int(permutations) // 10) == 0:
            print(f"[shapley][debug] perm {idx_perm + 1}/{permutations}")

    if permutations > 0:
        scores = [float(val) / float(permutations) for val in scores]
        if map_scores is not None:
            map_scores = map_scores / float(permutations)

    base._reset_anchor_controllers(anchors, reason="shapley-end")
    return scores, base_value, full_value, map_scores, map_indices


def _sample_subset_sizes(
    *,
    n_features: int,
    n_samples: int,
    size_range: Tuple[int, int],
    rng: random.Random,
) -> List[int]:
    if n_samples <= 0 or n_features <= 0:
        return []
    low, high = size_range
    low = max(1, min(low, n_features))
    high = max(low, min(high, n_features))
    return [rng.randint(low, high) for _ in range(int(n_samples))]


def _parse_subset_size(raw: str) -> Tuple[int, int]:
    text = str(raw or "").strip()
    if not text:
        return (1, 1)
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid subset size range: {raw}")
        return int(parts[0]), int(parts[1])
    return int(text), int(text)


def _evaluate_subset_fidelity(
    *,
    candidates: Sequence[Tuple[str, int]],
    scores: Sequence[float],
    anchors: Dict[str, base.AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[base.ActivationBaselineCache],
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    base_value: float,
    subset_samples: int,
    subset_size: Tuple[int, int],
    seed: int,
    debug: bool,
) -> Dict[str, Any]:
    if subset_samples <= 0 or not candidates:
        return {
            "samples": 0,
            "pearson": 0.0,
            "r2": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        }
    masks = _build_empty_masks(bases)
    _clear_masks(masks)
    _set_insertion_overrides(
        anchors=anchors,
        bases=bases,
        masks=masks,
        baseline_cache=baseline_cache,
    )

    preds: List[float] = []
    targets: List[float] = []
    rng = random.Random(int(seed))
    sizes = _sample_subset_sizes(
        n_features=len(candidates),
        n_samples=subset_samples,
        size_range=subset_size,
        rng=rng,
    )
    for idx_sample, k in enumerate(sizes, start=1):
        subset = rng.sample(range(len(candidates)), int(k))
        _clear_masks(masks)
        pred = 0.0
        for idx in subset:
            spec, unit = candidates[idx]
            mask = masks.get(spec)
            if mask is None or unit < 0 or unit >= mask.numel():
                continue
            mask[int(unit)] = True
            pred += float(scores[idx])
        val = _measure_objective(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            debug=debug,
        )
        preds.append(pred)
        targets.append(float(val - base_value))
        if debug and idx_sample % max(1, subset_samples // 5) == 0:
            print(f"[subset] {idx_sample}/{subset_samples}")

    pearson = _pearson_corr(preds, targets)
    r2 = pearson * pearson
    diff = np.asarray(preds, dtype=float) - np.asarray(targets, dtype=float)
    rmse = float(np.sqrt((diff * diff).mean())) if diff.size else 0.0
    mae = float(np.abs(diff).mean()) if diff.size else 0.0

    base._reset_anchor_controllers(anchors, reason="subset-fidelity-end")
    return {
        "samples": int(len(preds)),
        "pearson": float(pearson),
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def _save_shapley_pt(
    out_dir: Path,
    *,
    candidates: Sequence[Tuple[str, int]],
    shapley_scores: Sequence[float],
    meta: Dict[str, Any],
    sources: Dict[Tuple[str, int], List[str]],
) -> Optional[Path]:
    records: List[Dict[str, Any]] = []
    for (spec, unit), phi in zip(candidates, shapley_scores):
        records.append(
            {
                "spec": str(spec),
                "unit": int(unit),
                "phi": float(phi),
                "sources": list(sources.get((spec, unit), [])),
            }
        )
    payload = {
        "meta": dict(meta),
        "records": records,
        "phi": torch.tensor(list(shapley_scores), dtype=torch.float32),
    }
    path = out_dir / "shapley_scores.pt"
    try:
        torch.save(payload, path)
    except Exception as exc:
        print(f"[warn] failed to save shapley pt: {exc}")
        return None
    return path


def _save_shapley_panel(
    out_dir: Path,
    *,
    candidates: Sequence[Tuple[str, int]],
    shapley_scores: Sequence[float],
    topk: int,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] shapley panel skipped (matplotlib not available): {exc}")
        return None

    scores = np.asarray(list(shapley_scores), dtype=float)
    if scores.size == 0:
        return None
    order = np.argsort(np.abs(scores))[::-1]
    topk = max(1, int(topk))
    order = order[: min(topk, order.size)]

    labels = [f"{candidates[idx][0]}:{int(candidates[idx][1])}" for idx in order]
    vals = scores[order]
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]

    height = max(2.0, 0.35 * len(vals) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, height))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.0, color="#444", linewidth=0.8)
    ax.set_xlabel("Shapley value")
    ax.set_title(f"Top-{len(vals)} Shapley contributions (abs-sorted)")
    ax.invert_yaxis()
    fig.tight_layout()

    path = out_dir / f"shapley_panel_top{len(vals)}.png"
    try:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] failed to save shapley panel: {exc}")
        try:
            plt.close(fig)
        except Exception:
            pass
        return None
    return path


def _save_shapley_map_pt(
    out_dir: Path,
    *,
    candidates: Sequence[Tuple[str, int]],
    map_scores: torch.Tensor,
    map_indices: Sequence[int],
    meta: Dict[str, Any],
) -> Optional[Path]:
    if map_scores is None or map_indices is None:
        return None
    records: List[Dict[str, Any]] = []
    for slot, idx in enumerate(map_indices):
        spec, unit = candidates[idx]
        records.append(
            {
                "spec": str(spec),
                "unit": int(unit),
                "index": int(idx),
                "slot": int(slot),
                "shape": tuple(map_scores[slot].shape),
            }
        )
    payload = {
        "meta": dict(meta),
        "records": records,
        "maps": map_scores.detach().cpu(),
    }
    path = out_dir / "shapley_maps.pt"
    try:
        torch.save(payload, path)
    except Exception as exc:
        print(f"[warn] failed to save shapley maps pt: {exc}")
        return None
    return path


def _save_shapley_map_panels(
    out_dir: Path,
    *,
    candidates: Sequence[Tuple[str, int]],
    map_scores: torch.Tensor,
    map_indices: Sequence[int],
    base_stack: torch.Tensor,
    sample_id: int,
    topk: int,
    prompt_points: Sequence[Tuple[int, int]] = (),
    prompt_color: Tuple[int, int, int] = (0, 255, 0),
    overlay_on_base: bool = False,
    apply_sigmoid: bool = True,
    debug: bool = False,
) -> List[Path]:
    if map_scores is None or map_indices is None:
        return []
    if topk <= 0:
        return []
    maps_cpu = map_scores.detach().cpu()
    base_stack = base_stack.detach().cpu()
    scores = maps_cpu.abs().reshape(maps_cpu.shape[0], -1).mean(dim=1)
    order = torch.argsort(scores, descending=True).tolist()
    order = order[: min(int(topk), len(order))]

    max_abs = float(maps_cpu.abs().max().item()) if maps_cpu.numel() else 1.0
    out_paths: List[Path] = []
    for rank, slot in enumerate(order, start=1):
        idx = map_indices[slot]
        spec, unit = candidates[idx]
        heat = maps_cpu[slot]
        normed, _ = viz._normalize_contrib(heat, denom=max_abs)
        normed, base_aligned = viz._align_frames(normed, base_stack, debug=debug, label="shapley_map")
        stub = _sanitize_token(f"shapley_map_rank{rank:03d}_idx{idx}_unit{unit}")
        viz.save_output_contribution_overlay(
            out_dir=out_dir,
            sid=int(sample_id),
            score_suffix="",
            heat_stack=normed,
            target_tensor=base_aligned,
            prompt_points=prompt_points,
            prompt_color=prompt_color,
            overlay_alpha=0.4,
            overlay_cmap="bwr",
            use_abs_overlay=False,
            overlay_on_base=bool(overlay_on_base),
            apply_sigmoid=bool(apply_sigmoid),
            file_stub=stub,
        )
        out_paths.append(out_dir / f"sid{int(sample_id)}_panel__{stub}.jpeg")
    return out_paths


def _resolve_target_index(
    candidates: Sequence[Tuple[str, int]],
    *,
    target_spec: Optional[str],
    target_unit: int,
) -> int:
    matches: List[int] = []
    for idx, (spec, unit) in enumerate(candidates):
        if int(unit) != int(target_unit):
            continue
        if target_spec is None or str(spec) == str(target_spec):
            matches.append(idx)
    if not matches:
        raise RuntimeError(f"Target unit {target_unit} not found in candidates.")
    if len(matches) > 1:
        specs = [candidates[i][0] for i in matches]
        raise RuntimeError(
            f"Target unit {target_unit} is ambiguous across specs {specs}; "
            "specify --shapley-map-target-spec."
        )
    return matches[0]


def _ensure_target_candidate(
    candidates: Sequence[Tuple[str, int]],
    *,
    target_unit: int,
    target_spec: Optional[str],
    anchor_specs: Sequence[str],
    bases: Dict[str, Any],
    candidate_sources: Dict[Tuple[str, int], List[str]],
) -> Tuple[List[Tuple[str, int]], Dict[Tuple[str, int], List[str]], bool]:
    for spec, unit in candidates:
        if int(unit) != int(target_unit):
            continue
        if target_spec is None or str(spec) == str(target_spec):
            return list(candidates), candidate_sources, False

    spec_choice = target_spec
    if spec_choice is None:
        unique_specs = sorted({str(spec) for spec, _ in candidates})
        if len(unique_specs) == 1:
            spec_choice = unique_specs[0]
        else:
            # Prefer anchor spec with latent/acts attr if available, otherwise first anchor spec.
            preferred: Optional[str] = None
            for anchor_spec in anchor_specs:
                parsed = base.parse_spec(anchor_spec)
                attr = (parsed.attr or "").lower()
                if attr in {"latent", "acts", "activation"}:
                    preferred = str(anchor_spec)
                    break
            if preferred is None and anchor_specs:
                preferred = str(anchor_specs[0])
            if preferred is not None:
                spec_choice = preferred
            else:
                raise RuntimeError(
                    f"Target unit {target_unit} not found in candidates and spec is ambiguous; "
                    "use --shapley-map-target-spec."
                )

    masks = _build_empty_masks(bases)
    mask = masks.get(spec_choice)
    if mask is None or int(target_unit) >= int(mask.numel()):
        raise RuntimeError(f"Target unit {target_unit} not found in bases for spec '{spec_choice}'.")

    updated = list(candidates) + [(str(spec_choice), int(target_unit))]
    candidate_sources[(str(spec_choice), int(target_unit))] = ["target"]
    return updated, candidate_sources, True


def _subset_units_from_indices(
    candidates: Sequence[Tuple[str, int]],
    indices: Sequence[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx in indices:
        if idx < 0 or idx >= len(candidates):
            continue
        spec, unit = candidates[int(idx)]
        records.append({"spec": str(spec), "unit": int(unit), "index": int(idx)})
    return records


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = float(torch.norm(a_flat) * torch.norm(b_flat) + 1e-12)
    if denom <= 0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _should_log_progress(step: int, total: int, every: int) -> bool:
    if every <= 0:
        return False
    if step <= 1 or step >= total:
        return True
    return step % every == 0


def _log_cuda_mem(prefix: str, device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        free, total = torch.cuda.mem_get_info(device)
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(
            f"{prefix} cuda_mem free={free/1e9:.2f}GB total={total/1e9:.2f}GB "
            f"alloc={alloc/1e9:.2f}GB reserved={reserved/1e9:.2f}GB"
        )
    except Exception:
        pass


def _cosine_sim_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    denom = float(torch.norm(a_flat) * torch.norm(b_flat) + 1e-12)
    if denom <= 0:
        return 0.0
    return float(torch.dot(a_flat, b_flat) / denom)


def _align_maps(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.dim() != 4 or b.dim() != 4:
        raise RuntimeError(f"Unexpected map dims: a={tuple(a.shape)} b={tuple(b.shape)}")
    if int(a.shape[0]) != int(b.shape[0]):
        raise RuntimeError(f"Frame mismatch: a={tuple(a.shape)} b={tuple(b.shape)}")
    return a, b


def _method_forward_config(method: str, ig_steps: int, smoothgrad_samples: int) -> Tuple[str, int]:
    key = str(method or "").strip().lower()
    key = key.replace("-", "_")
    for prefix in ("libragrad_", "attnlrp_"):
        if key.startswith(prefix):
            key = key.split(prefix, 1)[1]
            break
    for suffix in ("_abs", "_pos", "_positive", "_sc", "_sign", "_signed"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    if key in {"inputxgrad"}:
        key = "input_x_grad"
    if key in {"ig_anchored", "ig_anchor"}:
        key = "ig"
    if key in {"smoothgrad", "vargrad"}:
        return key, int(max(1, smoothgrad_samples))
    if key in {"grad", "input_x_grad"}:
        return key, 1
    return key, int(max(1, ig_steps))


def _compute_method_output_maps(
    *,
    methods: Sequence[str],
    model: torch.nn.Module,
    adapter: base.SAM2EvalAdapter,
    forward_runner,
    sae: torch.nn.Module,
    target_spec: str,
    target_unit: int,
    mask_specs: Sequence[str],
    output_spec: str,
    lane_idx: Optional[int],
    num_frames: Optional[int],
    num_lanes: Optional[int],
    ig_steps: int,
    smoothgrad_samples: int,
    smoothgrad_sigma: float,
    forward_baseline: str,
    debug: bool,
    log_debug: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    sign_objective_mode: str = "output_sum",
    sign_mask_threshold: float = 0.2,
) -> Dict[str, torch.Tensor]:
    if log_debug:
        print(
            f"[map][debug] target={target_spec}:{int(target_unit)} output_spec={output_spec} "
            f"methods={list(methods)}"
        )
    map_device = next(model.parameters()).device
    runtime = AttributionRuntime(
        model=model,
        adapter=adapter,
        forward_fn=lambda: forward_runner(require_grad=True),
        sae_module=sae,
        target=RuntimeTarget(
            layer=str(target_spec),
            unit=int(target_unit),
            override_mode="all_tokens",
            objective_aggregation="sum",
        ),
    )
    runtime.configure_anchors(AnchorConfig(capture=list(mask_specs), ig_active=list(mask_specs), stop_grad=()))
    runtime.set_override_lane(lane_idx)

    attr_name = base._anchor_attr_name(str(target_spec))
    try:
        noise_scales = precompute_sae_noise_scales(sae)
    except Exception:
        noise_scales = {}

    maps: Dict[str, torch.Tensor] = {}
    try:
        for method in methods:
            name = str(method or "").strip()
            if not name:
                continue
            base_name = name.replace("-", "_").lower()
            for suffix in ("_abs", "_pos", "_positive", "_sc", "_sign", "_signed"):
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            use_sign_correction = any(
                name.replace("-", "_").lower().endswith(s)
                for s in ("_sc", "_sign", "_signed")
            )
            if base_name == "activation_patch":
                try:
                    def _capture_output_map() -> torch.Tensor:
                        runtime.anchor_capture.release_step_refs()
                        runtime.anchor_capture.clear_tapes()
                        runtime._run_forward(require_grad=False)
                        mask_lists = runtime.anchor_capture.get_tensor_lists(detach=True)
                        output_tensor = viz._extract_spec_tensor(
                            mask_lists,
                            output_spec,
                            num_frames=num_frames,
                            debug=debug,
                        )
                        if output_tensor is None:
                            raise RuntimeError(
                                f"Output spec '{output_spec}' not found in capture keys: {list(mask_lists.keys())}"
                            )
                        output_map = viz._extract_output_map(
                            output_tensor,
                            lane_idx=lane_idx,
                            num_frames=num_frames,
                            num_lanes=num_lanes,
                            debug=debug,
                        )
                        return viz._ensure_heat_stack(output_map.detach().cpu())

                    base_stack = _capture_output_map()
                    base_latent: Optional[torch.Tensor] = None
                    try:
                        base_latent = runtime.controller.last_encoded_all()
                    except Exception:
                        base_latent = None
                    if not torch.is_tensor(base_latent):
                        ctx = runtime._target_sae_branch.sae_context()
                        base_latent = ctx.get(attr_name) if ctx else None
                    if not torch.is_tensor(base_latent):
                        raise RuntimeError("Base latent not available for activation_patch map.")
                    if base_latent.dim() == 3:
                        base_latent = base_latent.unsqueeze(0)
                    if base_latent.shape[-1] <= int(target_unit):
                        raise RuntimeError(
                            f"Target unit {int(target_unit)} out of range for base latent shape {tuple(base_latent.shape)}"
                        )
                    replacement = base_latent.detach().clone().to(device=map_device)
                    replacement[..., int(target_unit)] = 0.0
                    try:
                        runtime.controller.set_override_all(replacement)
                    except Exception:
                        runtime.controller.set_override(replacement)
                    try:
                        dropped_stack = _capture_output_map()
                    finally:
                        try:
                            runtime.controller.clear_override()
                        except Exception:
                            pass
                    maps[name] = base_stack - dropped_stack
                except Exception as exc:
                    if debug or log_debug:
                        print(f"[map] activation_patch failed: {exc}")
                        if debug and log_debug:
                            traceback.print_exc()
                continue
            forward_method, steps = _method_forward_config(name, ig_steps, smoothgrad_samples)
            if forward_method not in {"grad", "input_x_grad", "ig", "smoothgrad", "vargrad"}:
                if debug or log_debug:
                    print(f"[map] skip unsupported forward method '{name}'")
                continue

            use_libragrad = base_name.startswith("libragrad_")
            use_attnlrp = base_name.startswith("attnlrp_")

            def _perturb():
                ctx = runtime._target_sae_branch.sae_context()
                base = ctx.get(attr_name) if ctx else None
                if not torch.is_tensor(base):
                    return
                noisy = add_sae_noise(
                    base,
                    noise_std=float(smoothgrad_sigma),
                    noise_mode="gaussian",
                    std_scale=noise_scales.get(str(target_spec)),
                )
                runtime._target_sae_branch.set_anchor_override(str(attr_name), lambda _t, v=noisy: v)

            if forward_method in {"smoothgrad", "vargrad"}:
                runtime.set_perturb_fn(_perturb)
            else:
                runtime.set_perturb_fn(None)

            cfg = ForwardConfig(
                enabled=True,
                method=str(forward_method),
                ig_steps=int(steps),
                baseline=str(forward_baseline),
            )
            ctx = (
                viz._LibragradContext(model, sae, gamma=None)
                if use_libragrad
                else viz._AttnLRPContext(model, sae)
                if use_attnlrp
                else None
            )
            if log_debug:
                print(f"[map][debug] method={name} forward={forward_method} steps={steps} baseline={forward_baseline}")

            amp_ctx = (
                torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=True,
                )
                if (
                    bool(amp_enabled)
                    and map_device.type == "cuda"
                    and amp_dtype in {torch.float16, torch.bfloat16}
                )
                else nullcontext()
            )

            # -- sign correction: compute sign vector before contribution --
            if use_sign_correction:
                sign_kwargs = dict(
                    runtime=runtime,
                    output_spec=output_spec,
                    lane_idx=lane_idx,
                    num_frames=num_frames,
                    num_lanes=num_lanes,
                    unit_idx=int(target_unit),
                    baseline=str(forward_baseline),
                    objective_mode=sign_objective_mode,
                    sign_filter=None,
                    threshold=sign_mask_threshold,
                    fixed_mask=None,
                    fixed_ref_logits=None,
                    force_sdpa_math=False,
                    debug=debug,
                )
                try:
                    sign_unit, _, _, _ = viz._compute_forward_grad_sign_unit(**sign_kwargs)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    if debug or log_debug:
                        print(f"[map][warn] sign correction OOM for {name}; retry once after cache clear")
                    gc.collect()
                    if map_device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        sign_unit, _, _, _ = viz._compute_forward_grad_sign_unit(**sign_kwargs)
                    except Exception as retry_exc:
                        if debug or log_debug:
                            print(f"[map] sign correction retry failed for {name}: {retry_exc}")
                        runtime.set_forward_weight_multiplier(None)
                        continue
                except Exception as exc:
                    if debug or log_debug:
                        print(f"[map] sign correction failed for {name}: {exc}")
                    runtime.set_forward_weight_multiplier(None)
                    continue
                if sign_unit is not None:
                    runtime.set_forward_weight_multiplier(
                        viz._make_sign_weight_multiplier(sign_unit, int(target_unit))
                    )
                else:
                    if debug or log_debug:
                        print(f"[map] sign correction returned None for {name}")
                    continue
                # Free sign-correction intermediates before contribution computation
                gc.collect()
                if map_device.type == "cuda":
                    torch.cuda.empty_cache()
            try:
                with (ctx if ctx is not None else nullcontext()):
                    with amp_ctx:
                        contrib_out = runtime.run_forward_contribution(cfg) or {}
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if log_debug:
                        print(f"[map][debug] OOM in method={name} on device={map_device}; skipping this method")
                    try:
                        runtime.set_forward_weight_multiplier(None)
                    except Exception:
                        pass
                    if map_device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                if log_debug:
                    print(f"[map][debug] method={name} failed: {exc}")
                if log_debug and debug:
                    traceback.print_exc()
                continue
            finally:
                if use_sign_correction:
                    runtime.set_forward_weight_multiplier(None)
            payload = contrib_out.get("attr", contrib_out)
            contrib_tensor = viz._extract_spec_tensor(payload, output_spec, num_frames=num_frames, debug=debug)
            if contrib_tensor is None:
                if debug or log_debug:
                    print(f"[map] missing contribution for method={name} spec={output_spec}")
                continue
            contrib_map = viz._extract_output_map(
                contrib_tensor,
                lane_idx=lane_idx,
                num_frames=num_frames,
                num_lanes=num_lanes,
                debug=debug,
            )
            maps[name] = viz._ensure_heat_stack(contrib_map.detach().cpu().float())
            if log_debug:
                shape = tuple(maps[name].shape)
                max_abs = float(maps[name].abs().max().item()) if maps[name].numel() else 0.0
                print(f"[map][debug] method={name} map_shape={shape} max_abs={max_abs:.6f}")
            if map_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    finally:
        try:
            runtime.cleanup()
        except Exception:
            pass
    return maps


def _estimate_shapley_map_owen(
    *,
    candidates: Sequence[Tuple[str, int]],
    target_index: int,
    anchors: Dict[str, base.AnchorInfo],
    bases: Dict[str, Any],
    baseline_cache: Optional[base.ActivationBaselineCache],
    forward_runner,
    mask_capture: base.MultiAnchorCapture,
    lane_idx: Optional[int],
    threshold: float,
    ref_mask_spec: Optional[str],
    objective_mask: Sequence[torch.Tensor],
    objective_ref_logits: Sequence[torch.Tensor],
    objective_mode: str,
    output_spec: str,
    num_frames: Optional[int],
    num_lanes: Optional[int],
    samples: int,
    seed: int,
    convergence_every: int,
    progress_every: int,
    debug: bool,
    sample_callback: Optional[Callable[[int, Sequence[int], float, float, torch.Tensor], None]] = None,
) -> Tuple[torch.Tensor, float, List[Tuple[int, float]]]:
    n = len(candidates)
    if target_index < 0 or target_index >= n:
        raise RuntimeError("Target index out of range.")
    others = [i for i in range(n) if i != target_index]
    m = len(others)
    if samples <= 0:
        raise RuntimeError("samples must be > 0 for Owen sampling.")

    sizes: List[int] = []
    if samples >= (m + 1):
        per_size = samples // (m + 1)
        remainder = samples % (m + 1)
        for k in range(m + 1):
            sizes.extend([k] * per_size)
        for k in range(remainder):
            sizes.append(k % (m + 1))
    else:
        sizes = list(range(m + 1))
        rng = random.Random(int(seed))
        rng.shuffle(sizes)
        sizes = sizes[: int(samples)]

    rng = random.Random(int(seed))
    rng.shuffle(sizes)

    masks = _build_empty_masks(bases)
    _clear_masks(masks)
    _set_insertion_overrides(
        anchors=anchors,
        bases=bases,
        masks=masks,
        baseline_cache=baseline_cache,
    )

    sum_map: Optional[torch.Tensor] = None
    sum_scalar = 0.0
    checkpoints: List[Tuple[int, torch.Tensor]] = []

    total_samples = len(sizes)
    for idx_sample, k in enumerate(sizes, start=1):
        _clear_masks(masks)
        subset = rng.sample(others, k) if k > 0 else []
        for idx in subset:
            spec, unit = candidates[idx]
            mask = masks.get(spec)
            if mask is None or unit < 0 or unit >= mask.numel():
                continue
            mask[int(unit)] = True

        val_s, map_s = _measure_objective_and_map(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            output_spec=output_spec,
            num_frames=num_frames,
            num_lanes=num_lanes,
            debug=debug,
        )

        spec_i, unit_i = candidates[target_index]
        mask_i = masks.get(spec_i)
        if mask_i is None or unit_i < 0 or unit_i >= mask_i.numel():
            raise RuntimeError("Target unit not found in masks.")
        mask_i[int(unit_i)] = True

        val_si, map_si = _measure_objective_and_map(
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            anchors=anchors,
            lane_idx=lane_idx,
            threshold=threshold,
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            output_spec=output_spec,
            num_frames=num_frames,
            num_lanes=num_lanes,
            debug=debug,
        )

        delta_map = map_si - map_s
        # Filter out SAM2 NO_OBJ_SCORE sentinel values (-1024) to avoid skewed aggregates.
        NO_OBJ_THRESHOLD = -1023.0
        invalid = (map_s <= NO_OBJ_THRESHOLD) | (map_si <= NO_OBJ_THRESHOLD)
        if invalid.any():
            delta_map = delta_map.clone()
            delta_map[invalid] = 0.0
        delta_map_cpu = delta_map.detach().cpu().float()
        if sum_map is None:
            sum_map = delta_map_cpu.clone()
        else:
            sum_map.add_(delta_map_cpu)
        delta_scalar = float(val_si - val_s)
        sum_scalar += delta_scalar
        if sample_callback is not None:
            try:
                subset_sorted = sorted(subset)
                sample_callback(
                    int(idx_sample),
                    subset_sorted,
                    float(val_s),
                    float(val_si),
                    delta_map_cpu,
                )
            except Exception as exc:
                print(f"[warn] sample callback failed: {exc}")

        if convergence_every > 0 and idx_sample % int(convergence_every) == 0:
            mean_map = sum_map / float(idx_sample)
            checkpoints.append((idx_sample, mean_map.clone()))
        if _should_log_progress(idx_sample, total_samples, progress_every):
            print(f"[shapley][owen] sample {idx_sample}/{total_samples}")

    if sum_map is None:
        raise RuntimeError("Failed to compute Shapley map (no samples?).")
    mean_map = sum_map / float(len(sizes))
    phi_scalar = float(sum_scalar / float(len(sizes)))

    conv: List[Tuple[int, float]] = []
    if checkpoints:
        for count, mean_at in checkpoints:
            conv.append((count, _cosine_sim(mean_at, mean_map)))

    return mean_map, phi_scalar, conv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Shapley ground-truth contributions for SAMv2/SAM2 and compare attribution methods."
    )
    parser.add_argument("--attr-config", type=Path, default=Path("configs/sam2_attr_index_v2.yaml"))
    parser.add_argument(
        "--anchor",
        type=str,
        default="model.sam_mask_decoder.transformer.layers.0@1::sae_layer#latent",
    )
    parser.add_argument("--unit", type=int, default=73)
    parser.add_argument("--sample-unit", type=int, default=73, help="Unit for ledger row selection.")
    parser.add_argument("--decile", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--mask-specs",
        type=str,
        default="model@0@pred_masks_high_res,model@1@pred_masks_high_res,model@2@pred_masks_high_res,model@3@pred_masks_high_res",
    )
    parser.add_argument("--output-spec", type=str, default=None)
    parser.add_argument("--ref-mask-spec", type=str, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    parser.add_argument(
        "--feature-active-threshold",
        type=float,
        default=1e-6,
        help="Threshold for marking SAE features as active in activation_patch (applied to summed |activation|).",
    )
    parser.add_argument(
        "--objective-mode",
        type=str,
        default="mask",
        choices=["mask", "logit_sum", "logit_dot", "cosine"],
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--lane", type=int, default=None)
    parser.add_argument("--baseline-cache", type=Path, default=None)
    parser.add_argument("--libragrad-gamma", type=float, default=None)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--ig-active", type=str, default=None)
    parser.add_argument("--smoothgrad-samples", type=int, default=8)
    parser.add_argument("--smoothgrad-sigma", type=float, default=0.01)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--rank-abs", action="store_true")
    parser.add_argument("--rank-positive", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--rank-exclude", type=str, default="activation_patch")
    parser.add_argument(
        "--methods",
        type=str,
        default="activation_patch,attnlrp_input_x_grad,libragrad_input_x_grad,input_x_grad,grad,ig,libragrad_ig",
        help="Comma-separated methods to score (suffix _abs/_pos allowed).",
    )
    parser.add_argument("--candidate-method", type=str, default=None)
    parser.add_argument("--candidate-topk", type=int, default=128)
    parser.add_argument("--candidate-union", action="store_true")
    parser.add_argument("--candidate-max", type=int, default=None)
    parser.add_argument("--candidate-list", type=Path, default=None)
    parser.add_argument("--candidate-active-only", action="store_true")
    parser.add_argument("--candidate-minfeat", default=True, action=argparse.BooleanOptionalAction, help="Build candidates via minfeat objective thresholding.")
    parser.add_argument("--candidate-metric", type=str, default="objective", choices=["objective", "iou"])
    parser.add_argument("--candidate-target-fraction", type=float, default=0.99)
    parser.add_argument("--candidate-target-iou", type=float, default=0.95)
    parser.add_argument("--candidate-chunk-size", type=int, default=16)
    parser.add_argument("--candidate-max-steps", type=int, default=None)
    parser.add_argument("--subset-samples", type=int, default=512)
    parser.add_argument("--subset-size", type=str, default="1:32")
    parser.add_argument("--shapley-panel-topk", type=int, default=32, help="Top-K shapley values to render in the panel.")
    parser.add_argument("--no-shapley-panel", action="store_true", help="Skip saving the shapley panel image.")
    parser.add_argument("--no-shapley-pt", action="store_true", help="Skip saving shapley scores as a .pt file.")
    parser.add_argument("--shapley-map", default=True, action=argparse.BooleanOptionalAction, help="Accumulate Shapley maps (output-space) in addition to scalar values.")
    parser.add_argument(
        "--shapley-map-viz-topk",
        type=int,
        default=1,
        help="Save overlay panels for top-K Shapley maps (0 to disable).",
    )
    parser.add_argument("--no-shapley-map-pt", action="store_true", help="Skip saving Shapley maps as a .pt file.")
    parser.add_argument("--shapley-map-samples", type=int, default=256, help="Number of Owen samples for target Shapley map.")
    parser.add_argument("--shapley-map-target-spec", type=str, default=None, help="Spec for target Shapley map (defaults to match candidate with unit).")
    parser.add_argument("--shapley-map-target-unit", type=int, default=None, help="Unit id for target Shapley map (default: --unit).")
    parser.add_argument("--shapley-map-convergence-every", type=int, default=16, help="Checkpoint interval for convergence plot (0 disables).")
    parser.add_argument("--method-map-device", type=str, default=None, help="Device for method map computation (e.g., cpu, cuda:0).")
    parser.add_argument(
        "--method-map-mask-specs",
        type=str,
        default=None,
        help="Comma-separated mask specs to capture for method maps (default: same as --mask-specs).",
    )
    parser.add_argument(
        "--method-map-amp-dtype",
        type=str,
        default="bfloat16",
        help="Autocast dtype for method maps on CUDA: bfloat16|float16|float32|none (default: bfloat16).",
    )
    parser.add_argument("--no-method-map-amp", action="store_true", help="Disable autocast during method map computation.")
    parser.add_argument("--method-map-baseline", type=str, default="zeros", help="Baseline for forward contribution maps.")
    parser.add_argument("--no-method-map-metrics", action="store_true", help="Skip method-vs-Shapley map metrics and panels.")
    parser.add_argument("--method-map-debug", action="store_true", help="Verbose logging for method map computation.")
    parser.add_argument("--no-scalar-shapley", default=True, action=argparse.BooleanOptionalAction, help="Skip scalar Shapley computation and method metrics.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N steps (0 disables).")
    parser.add_argument("--sign-eps", type=float, default=1e-6)
    parser.add_argument("--sign-skip-zero", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sam2_shapley"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.rank_abs and args.rank_positive:
        print("[warn] both --rank-abs and --rank-positive set; using --rank-positive only.")
        args.rank_abs = False

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    debug = bool(args.debug)

    attr_cfg = base._load_yaml(args.attr_config)
    index_cfg = base._load_yaml(Path(attr_cfg["indexing"]["config"]))
    device = torch.device(args.device or index_cfg.get("model", {}).get("device", "cuda"))
    model_loader = base.load_obj(index_cfg["model"]["loader"])
    model = model_loader(index_cfg["model"], device=device).eval()
    for p in model.parameters():
        p.requires_grad = False
    adapter = base.SAM2EvalAdapter(model, device=device, collate_fn=None)

    row = _select_row(
        attr_cfg=attr_cfg,
        index_cfg=index_cfg,
        sample_unit=args.sample_unit,
        decile=int(args.decile),
        rank=int(args.rank),
    )
    meta_ledger = base.OfflineMetaParquetLedger(index_cfg["indexing"]["offline_meta_root"])
    bvd, lane_idx = base._build_bvd_for_row(row=row, meta_ledger=meta_ledger, ds_cfg=index_cfg["dataset"])
    if args.lane is not None:
        lane_idx = int(args.lane)
    batch_on_dev = adapter.preprocess_input(bvd)
    base._record_adapter_caches(adapter, batch_on_dev)
    forward_runner = base._make_forward_runner(adapter, batch_on_dev)

    layer_default = attr_cfg["indexing"]["layer"]
    anchor_specs_raw = base._parse_anchor_specs(args.anchor if args.anchor is not None else layer_default)
    anchor_specs = base._expand_anchor_specs(anchor_specs_raw)

    mask_specs = [s for s in (args.mask_specs or "").split(",") if s]
    if not mask_specs:
        raise ValueError("No mask specs provided.")
    ref_mask_spec = args.ref_mask_spec or (mask_specs[0] if mask_specs else None)
    parsed_anchor = base.parse_spec(args.anchor)
    if args.output_spec:
        output_spec = args.output_spec
    else:
        if parsed_anchor.base_branch is not None:
            output_spec = f"model@{parsed_anchor.base_branch}@pred_masks_high_res"
        else:
            output_spec = mask_specs[0]
    if output_spec not in mask_specs:
        mask_specs.append(output_spec)
    if args.method_map_mask_specs:
        method_map_mask_specs = [tok.strip() for tok in str(args.method_map_mask_specs).split(",") if tok.strip()]
    else:
        method_map_mask_specs = list(mask_specs)
    if not method_map_mask_specs:
        method_map_mask_specs = list(mask_specs) if mask_specs else [output_spec]

    anchors: Dict[str, base.AnchorInfo] = {}
    restore_handles: List[Any] = []
    for spec in anchor_specs:
        controller = base.AllTokensFeatureOverrideController(
            spec=base.OverrideSpec(lane_idx=None, unit_indices=None),
            frame_getter=getattr(adapter, "current_frame_idx", None),
        )
        sae = base._load_sae_for_layer(index_cfg["sae"], spec, device)
        capture = base.LayerCapture(spec)
        owner = base.resolve_module(model, capture.base)
        handle, branch = base.wrap_target_layer_with_sae(
            owner,
            capture=capture,
            sae=sae,
            controller=controller,
            frame_getter=getattr(adapter, "current_frame_idx", None),
        )
        restore_handles.append(handle)
        attr_name = base._anchor_attr_name(spec)
        anchors[spec] = base.AnchorInfo(spec=spec, branch=branch, attr_name=attr_name, controller=controller)

    autoreset_handle = base.install_controller_autoreset_hooks(
        model,
        [a.controller for a in anchors.values()],
        branches=[a.branch for a in anchors.values()],
    )
    if autoreset_handle is not None:
        restore_handles.append(autoreset_handle.remove)

    for anchor in anchors.values():
        try:
            anchor.controller.spec.lane_idx = lane_idx
        except Exception:
            pass

    mask_capture = base.MultiAnchorCapture(frame_getter=getattr(adapter, "current_frame_idx", None))
    mask_capture.register_from_specs(mask_specs, resolve_module_fn=lambda name: base.resolve_module(model, name))

    objective_mode = str(args.objective_mode).strip().lower()

    mask_capture.release_step_refs()
    mask_capture.clear_tapes()
    forward_runner(require_grad=False)
    base_bundle = mask_capture.get_tensor_lists(detach=True)
    base_obj_t, objective_mask, objective_ref_logits = base._objective_from_masks(
        base_bundle,
        lane_idx,
        float(args.mask_threshold),
        ref_spec=ref_mask_spec,
        fixed_mask=None,
        fixed_ref_logits=None,
        objective_mode=objective_mode,
    )
    base_obj = float(base_obj_t.detach().cpu().item())
    objective_mask = [m.detach() for m in objective_mask]
    objective_ref_logits = [r.detach() for r in objective_ref_logits]
    mask_capture.release_step_refs()
    prompt_frames = viz._frames_from_batch(batch_on_dev)
    prompt_points = viz._extract_prompt_points(adapter, prompt_frames, lane_idx, debug=debug)
    prompt_color = (0, 255, 0)
    num_frames, num_lanes = viz._infer_meta_dims(adapter, batch_on_dev)
    base_stack: Optional[torch.Tensor] = None
    if args.shapley_map and args.shapley_map_viz_topk > 0:
        try:
            base_tensor = viz._extract_spec_tensor(base_bundle, output_spec)
            if base_tensor is not None:
                base_map = viz._extract_output_map(
                    base_tensor,
                    lane_idx=lane_idx,
                    num_frames=num_frames,
                    num_lanes=num_lanes,
                    debug=debug,
                )
                base_stack = viz._ensure_heat_stack(base_map.detach().cpu())
        except Exception as exc:
            print(f"[warn] failed to build base stack for shapley map panels: {exc}")

    bases = base._capture_anchor_bases(anchors)
    if not bases:
        raise RuntimeError("Failed to capture SAE base activations.")

    baseline_cache = base._load_baseline_cache(args.baseline_cache)
    if baseline_cache is not None and args.baseline_cache is not None:
        print(f"[info] loaded activation baseline cache: {args.baseline_cache}")

    method_specs = _parse_method_specs(
        args.methods,
        use_abs=bool(args.rank_abs),
        use_positive=bool(args.rank_positive),
        rank_exclude=_parse_csv_list(args.rank_exclude),
    )
    if not method_specs:
        raise RuntimeError("No methods provided.")
    method_aliases: Dict[str, str] = {}
    for spec in method_specs:
        key = base._normalize_method_name(spec.base)
        if key not in method_aliases:
            method_aliases[key] = spec.label

    ig_active_specs = base._parse_anchor_specs(args.ig_active) if args.ig_active else None

    rankings_by_method: Dict[str, Dict[str, Any]] = {}
    method_order: List[str] = []
    for spec in method_specs:
        if args.progress_every > 0:
            print(f"[rank] method={spec.label} (abs={spec.use_abs} pos={spec.use_positive})")
        ranking, active_masks = _compute_method_ranking(
            spec=spec,
            model=model,
            anchors=anchors,
            bases=bases,
            baseline_cache=baseline_cache,
            base_objective=base_obj,
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            batch_on_dev=batch_on_dev,
            lane_idx=lane_idx,
            threshold=float(args.mask_threshold),
            feature_active_threshold=float(args.feature_active_threshold),
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            max_features=args.max_features,
            ig_steps=int(args.ig_steps),
            ig_active=ig_active_specs,
            smoothgrad_samples=int(args.smoothgrad_samples),
            smoothgrad_sigma=float(args.smoothgrad_sigma),
            libragrad_gamma=args.libragrad_gamma,
        )
        if not ranking:
            print(f"[warn] no ranking for {spec.label}")
            continue
        if args.progress_every > 0:
            print(f"[rank] method={spec.label} done (n={len(ranking)})")
        method_order.append(spec.label)
        rankings_by_method[spec.label] = {
            "ranking": ranking,
            "active_masks": active_masks,
        }
        base._reset_anchor_controllers(anchors, reason=f"post-ranking:{spec.label}")
        model.zero_grad(set_to_none=True)
        base._gpu_gc()

    if not method_order:
        raise RuntimeError("No method rankings computed.")

    candidate_rankings = rankings_by_method
    if args.candidate_minfeat and args.rank_positive:
        candidate_rankings = dict(rankings_by_method)
        for spec in method_specs:
            if spec.base == "activation_patch":
                continue
            if spec.use_positive and not spec.use_abs:
                continue
            pos_spec = MethodSpec(label=spec.label, base=spec.base, use_abs=False, use_positive=True)
            ranking, active_masks = _compute_method_ranking(
                spec=pos_spec,
                model=model,
                anchors=anchors,
                bases=bases,
                baseline_cache=baseline_cache,
                base_objective=base_obj,
                forward_runner=forward_runner,
                mask_capture=mask_capture,
                batch_on_dev=batch_on_dev,
                lane_idx=lane_idx,
                threshold=float(args.mask_threshold),
                feature_active_threshold=float(args.feature_active_threshold),
                ref_mask_spec=ref_mask_spec,
                objective_mask=objective_mask,
                objective_ref_logits=objective_ref_logits,
                objective_mode=objective_mode,
                max_features=args.max_features,
                ig_steps=int(args.ig_steps),
                ig_active=ig_active_specs,
                smoothgrad_samples=int(args.smoothgrad_samples),
                smoothgrad_sigma=float(args.smoothgrad_sigma),
                libragrad_gamma=args.libragrad_gamma,
            )
            if ranking:
                candidate_rankings[spec.label] = {
                    "ranking": ranking,
                    "active_masks": active_masks,
                }
                base._reset_anchor_controllers(anchors, reason=f"post-ranking-pos:{spec.label}")
                model.zero_grad(set_to_none=True)
                base._gpu_gc()

    minfeat_info: Dict[str, Any] = {}
    minfeat_counts: Dict[str, int] = {}
    minfeat_selected: Optional[str] = None
    if args.candidate_list is not None:
        candidates = _read_candidate_list(args.candidate_list)
        candidate_sources = {c: ["file"] for c in candidates}
    elif args.candidate_minfeat:
        if args.progress_every > 0:
            print("[candidate] selecting via minfeat")
        metric = str(args.candidate_metric).lower().strip()
        target = float(args.candidate_target_iou if metric == "iou" else args.candidate_target_fraction)
        candidates, candidate_sources, minfeat_info, minfeat_selected, minfeat_counts = _select_minfeat_candidates(
            method_rankings=candidate_rankings,
            method_order=method_order,
            method_aliases=method_aliases,
            candidate_method=args.candidate_method,
            candidate_union=bool(args.candidate_union),
            candidate_max=args.candidate_max,
            active_only=bool(args.candidate_active_only),
            metric=metric,
            target=target,
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            lane_idx=lane_idx,
            threshold=float(args.mask_threshold),
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            anchors=anchors,
            bases=bases,
            baseline_cache=baseline_cache,
            chunk_size=int(args.candidate_chunk_size),
            max_steps=args.candidate_max_steps,
            base_value=base_obj,
            debug=debug,
        )
    else:
        if args.progress_every > 0:
            print("[candidate] selecting via topk")
        candidates, candidate_sources = _select_candidates(
            method_rankings=rankings_by_method,
            method_order=method_order,
            method_aliases=method_aliases,
            candidate_method=args.candidate_method,
            candidate_topk=int(args.candidate_topk),
            candidate_union=bool(args.candidate_union),
            candidate_max=args.candidate_max,
            active_only=bool(args.candidate_active_only),
        )

    masks_for_check = _build_empty_masks(bases)
    filtered_candidates: List[Tuple[str, int]] = []
    for spec_name, unit in candidates:
        mask = masks_for_check.get(spec_name)
        if mask is None or int(unit) >= int(mask.numel()) or int(unit) < 0:
            print(f"[warn] drop candidate spec={spec_name} unit={unit}: not in bases")
            continue
        filtered_candidates.append((str(spec_name), int(unit)))
    candidates = filtered_candidates

    if args.shapley_map:
        target_unit = int(args.shapley_map_target_unit) if args.shapley_map_target_unit is not None else int(args.unit)
        candidates, candidate_sources, added = _ensure_target_candidate(
            candidates,
            target_unit=target_unit,
            target_spec=args.shapley_map_target_spec,
            anchor_specs=anchor_specs,
            bases=bases,
            candidate_sources=candidate_sources,
        )
        if added:
            print(f"[info] appended target unit {target_unit} to candidates for Owen sampling")

    if args.candidate_list is not None:
        print(f"[candidate] source=list total={len(candidates)}")
    elif args.candidate_minfeat:
        for label in method_order:
            count = int(minfeat_counts.get(label, 0))
            info = minfeat_info.get(label, {})
            min_feat = info.get("min_features")
            msg = f"[candidate] minfeat method={label} count={count}"
            if min_feat is not None:
                msg += f" min_feat={min_feat}"
            print(msg)
        if bool(args.candidate_union):
            print(f"[candidate] selected_method=union total={len(candidates)}")
        else:
            chosen = minfeat_selected if minfeat_selected is not None else "unknown"
            chosen_min = None
            if minfeat_selected is not None:
                chosen_min = minfeat_info.get(minfeat_selected, {}).get("min_features")
            extra = f" min_features={chosen_min}" if chosen_min is not None else ""
            print(f"[candidate] selected_method={chosen}{extra} total={len(candidates)}")
    else:
        counts_by_method: Dict[str, int] = {label: 0 for label in method_order}
        for key in candidates:
            for label in candidate_sources.get(key, []):
                counts_by_method[label] = counts_by_method.get(label, 0) + 1
        if counts_by_method:
            summary = ", ".join(f"{label}={counts_by_method.get(label, 0)}" for label in method_order)
            print(f"[candidate] per-method counts: {summary}")
        if bool(args.candidate_union):
            print(f"[candidate] selected_method=union total={len(candidates)}")
        else:
            chosen = "unknown"
            unique_methods: set[str] = set()
            for key in candidates:
                unique_methods.update(candidate_sources.get(key, []))
            if len(unique_methods) == 1:
                chosen = next(iter(unique_methods))
            print(f"[candidate] selected_method={chosen} total={len(candidates)}")

    if not candidates:
        raise RuntimeError("No valid candidates available after filtering.")
    if args.progress_every > 0:
        print(f"[candidate] final count={len(candidates)}")

    objective_dir = f"objective_{_sanitize_token(objective_mode)}"
    out_root = (
        args.out_dir
        / objective_dir
        / f"unit_{int(args.unit)}"
        / f"decile_{int(args.decile)}"
        / f"rank{int(args.rank)}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    panel_dir = out_root / "panels"

    if minfeat_info:
        minfeat_metric = str(args.candidate_metric).lower().strip()
        minfeat_target = float(args.candidate_target_iou if minfeat_metric == "iou" else args.candidate_target_fraction)
        minfeat_summary: Dict[str, Any] = {
            "sample_id": int(row.get("sample_id", -1)),
            "frame_idx": int(row.get("frame_idx", 0)),
            "lane_idx": int(lane_idx) if lane_idx is not None else None,
            "target_unit": int(args.unit),
            "decile": int(args.decile),
            "rank": int(args.rank),
            "metric": minfeat_metric,
            "target": minfeat_target,
            "methods": {},
        }
        best_method = None
        best_min = None
        for label in method_order:
            info = dict(minfeat_info.get(label, {}))
            info["count"] = int(minfeat_counts.get(label, 0))
            minfeat_summary["methods"][label] = info
            min_feat = info.get("min_features")
            if min_feat is None:
                continue
            if best_min is None or int(min_feat) < int(best_min):
                best_min = int(min_feat)
                best_method = label
        if best_method is not None:
            minfeat_summary["best_method"] = {"name": best_method, "min_features": int(best_min)}
        if minfeat_selected is not None:
            minfeat_summary["selected_method"] = str(minfeat_selected)
        (out_root / "minfeat_summary.json").write_text(
            json.dumps(minfeat_summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    print(f"[info] candidates={len(candidates)} method_order={method_order}")
    shapley_maps: Optional[torch.Tensor] = None
    shapley_map_indices: Optional[List[int]] = None
    convergence_trace: List[Tuple[int, float]] = []
    shapley_max_map: Optional[torch.Tensor] = None
    shapley_min_map: Optional[torch.Tensor] = None
    shapley_extremes: Dict[str, Any] = {}
    if not args.no_scalar_shapley and args.progress_every > 0:
        print("[warn] scalar shapley disabled (owen-only)")
    args.no_scalar_shapley = True
    shapley_scores: List[float] = []
    base_value = 0.0
    full_value = 0.0

    if args.shapley_map:
        if args.progress_every > 0:
            print(f"[shapley][owen] samples={int(args.shapley_map_samples)}")
        target_unit = int(args.shapley_map_target_unit) if args.shapley_map_target_unit is not None else int(args.unit)
        target_idx = _resolve_target_index(
            candidates,
            target_spec=args.shapley_map_target_spec,
            target_unit=target_unit,
        )
        target_spec, target_unit = candidates[int(target_idx)]

        max_delta = -float("inf")
        min_delta = float("inf")
        max_subset: Optional[List[int]] = None
        min_subset: Optional[List[int]] = None
        max_vals: Optional[Tuple[float, float]] = None
        min_vals: Optional[Tuple[float, float]] = None
        max_sample_idx: Optional[int] = None
        min_sample_idx: Optional[int] = None
        max_map_cpu: Optional[torch.Tensor] = None
        min_map_cpu: Optional[torch.Tensor] = None

        def _track_extremes(
            idx_sample: int,
            subset_indices: Sequence[int],
            val_s: float,
            val_si: float,
            delta_map_cpu: torch.Tensor,
        ) -> None:
            nonlocal max_delta, min_delta, max_subset, min_subset
            nonlocal max_vals, min_vals, max_sample_idx, min_sample_idx
            nonlocal max_map_cpu, min_map_cpu
            delta = float(val_si - val_s)
            if delta > max_delta:
                max_delta = delta
                max_subset = list(subset_indices)
                max_vals = (float(val_s), float(val_si))
                max_sample_idx = int(idx_sample)
                max_map_cpu = delta_map_cpu.clone()
            if delta < min_delta:
                min_delta = delta
                min_subset = list(subset_indices)
                min_vals = (float(val_s), float(val_si))
                min_sample_idx = int(idx_sample)
                min_map_cpu = delta_map_cpu.clone()

        mean_map, phi_scalar, convergence_trace = _estimate_shapley_map_owen(
            candidates=candidates,
            target_index=target_idx,
            anchors=anchors,
            bases=bases,
            baseline_cache=baseline_cache,
            forward_runner=forward_runner,
            mask_capture=mask_capture,
            lane_idx=lane_idx,
            threshold=float(args.mask_threshold),
            ref_mask_spec=ref_mask_spec,
            objective_mask=objective_mask,
            objective_ref_logits=objective_ref_logits,
            objective_mode=objective_mode,
            output_spec=str(output_spec),
            num_frames=num_frames,
            num_lanes=num_lanes,
            samples=int(args.shapley_map_samples),
            seed=int(args.seed),
            convergence_every=int(args.shapley_map_convergence_every),
            progress_every=int(args.progress_every),
            debug=debug,
            sample_callback=_track_extremes,
        )
        mean_map_cpu = mean_map.detach().cpu()
        shapley_maps = mean_map.unsqueeze(0)
        shapley_map_indices = [int(target_idx)]
        shapley_scores = [0.0 for _ in candidates]
        shapley_scores[target_idx] = float(phi_scalar)
        shapley_max_map = max_map_cpu
        shapley_min_map = min_map_cpu
        if max_subset is not None and max_map_cpu is not None:
            max_units = _subset_units_from_indices(candidates, max_subset)
            target_rec = {"spec": str(target_spec), "unit": int(target_unit), "index": int(target_idx)}
            shapley_extremes["max"] = {
                "sample_idx": int(max_sample_idx) if max_sample_idx is not None else None,
                "delta": float(max_delta),
                "objective_without": float(max_vals[0]) if max_vals is not None else None,
                "objective_with": float(max_vals[1]) if max_vals is not None else None,
                "subset_size": int(len(max_subset)),
                "subset_indices": list(max_subset),
                "subset_units": list(max_units),
                "with_target_indices": list(max_subset) + [int(target_idx)],
                "with_target_units": list(max_units) + [target_rec],
            }
        if min_subset is not None and min_map_cpu is not None:
            min_units = _subset_units_from_indices(candidates, min_subset)
            target_rec = {"spec": str(target_spec), "unit": int(target_unit), "index": int(target_idx)}
            shapley_extremes["min"] = {
                "sample_idx": int(min_sample_idx) if min_sample_idx is not None else None,
                "delta": float(min_delta),
                "objective_without": float(min_vals[0]) if min_vals is not None else None,
                "objective_with": float(min_vals[1]) if min_vals is not None else None,
                "subset_size": int(len(min_subset)),
                "subset_indices": list(min_subset),
                "subset_units": list(min_units),
                "with_target_indices": list(min_subset) + [int(target_idx)],
                "with_target_units": list(min_units) + [target_rec],
            }
        mean_panel_saved = False
        if base_stack is not None and args.shapley_map_viz_topk > 0:
            try:
                normed, _ = viz._normalize_contrib(mean_map_cpu)
                normed, base_aligned = viz._align_frames(
                    normed,
                    base_stack.detach().cpu(),
                    debug=debug,
                    label="shapley_mean_map",
                )
                panel_dir.mkdir(parents=True, exist_ok=True)
                viz.save_output_contribution_overlay(
                    out_dir=panel_dir,
                    sid=int(row.get("sample_id", 0)),
                    score_suffix="",
                    heat_stack=normed,
                    target_tensor=base_aligned,
                    prompt_points=prompt_points,
                    prompt_color=prompt_color,
                    overlay_alpha=0.4,
                    overlay_cmap="bwr",
                    use_abs_overlay=False,
                    overlay_on_base=False,
                    apply_sigmoid=True,
                    file_stub=f"shapley_mean_map_unit{int(target_unit)}",
                )
                mean_panel_saved = True
            except Exception as exc:
                print(f"[warn] failed to save mean_map panel: {exc}")
            if max_map_cpu is not None:
                try:
                    normed, _ = viz._normalize_contrib(max_map_cpu)
                    normed, base_aligned = viz._align_frames(
                        normed,
                        base_stack.detach().cpu(),
                        debug=debug,
                        label="shapley_max_map",
                    )
                    viz.save_output_contribution_overlay(
                        out_dir=panel_dir,
                        sid=int(row.get("sample_id", 0)),
                        score_suffix="",
                        heat_stack=normed,
                        target_tensor=base_aligned,
                        prompt_points=prompt_points,
                        prompt_color=prompt_color,
                        overlay_alpha=0.4,
                        overlay_cmap="bwr",
                        use_abs_overlay=False,
                        overlay_on_base=False,
                        apply_sigmoid=True,
                        file_stub=f"shapley_max_map_unit{int(target_unit)}",
                    )
                except Exception as exc:
                    print(f"[warn] failed to save max_map panel: {exc}")
            if min_map_cpu is not None:
                try:
                    normed, _ = viz._normalize_contrib(min_map_cpu)
                    normed, base_aligned = viz._align_frames(
                        normed,
                        base_stack.detach().cpu(),
                        debug=debug,
                        label="shapley_min_map",
                    )
                    viz.save_output_contribution_overlay(
                        out_dir=panel_dir,
                        sid=int(row.get("sample_id", 0)),
                        score_suffix="",
                        heat_stack=normed,
                        target_tensor=base_aligned,
                        prompt_points=prompt_points,
                        prompt_color=prompt_color,
                        overlay_alpha=0.4,
                        overlay_cmap="bwr",
                        use_abs_overlay=False,
                        overlay_on_base=False,
                        apply_sigmoid=True,
                        file_stub=f"shapley_min_map_unit{int(target_unit)}",
                    )
                except Exception as exc:
                    print(f"[warn] failed to save min_map panel: {exc}")

    method_map_metrics: Dict[str, Any] = {}
    method_map_metrics_extremes: Dict[str, Any] = {}
    if (
        bool(args.shapley_map)
        and shapley_maps is not None
        and shapley_map_indices is not None
        and not bool(args.no_method_map_metrics)
    ):
        # Free GPU memory from earlier phases before running method maps.
        try:
            mask_capture.clear()
        except Exception:
            pass
        for anchor in anchors.values():
            try:
                anchor.branch.clear_context()
            except Exception:
                pass
        anchors = {}
        bases = {}
        baseline_cache = None
        base_bundle = None
        objective_mask = []
        objective_ref_logits = []
        base_stack_cpu = base_stack.detach().cpu() if base_stack is not None else None
        base_stack = None
        if not shapley_map_indices:
            print("[warn] no shapley map indices available for method-map metrics")
        else:
            target_idx = int(shapley_map_indices[0])
            target_spec, target_unit = candidates[target_idx]
            if args.progress_every > 0:
                print(f"[map] computing method maps for target={target_spec}:{int(target_unit)}")

            def _run_method_maps(
                map_device: torch.device,
            ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], List[torch.Tensor], Sequence[Tuple[int, int]]]:
                if args.method_map_debug:
                    print(f"[map][debug] using device={map_device}")
                    _log_cuda_mem("[map][debug] before model load", map_device)
                map_model = model_loader(index_cfg["model"], device=map_device).eval()
                for p in map_model.parameters():
                    p.requires_grad = False
                map_adapter = base.SAM2EvalAdapter(map_model, device=map_device, collate_fn=None)
                map_batch = map_adapter.preprocess_input(bvd)
                base._record_adapter_caches(map_adapter, map_batch)
                def _map_forward_runner(*, require_grad: bool = False, **_kwargs: Any) -> None:
                    # Match viz behavior: respect outer grad mode, only replay cached inputs.
                    with map_adapter.clicks_cache("replay"), map_adapter.prompt_inputs_cache("replay"):
                        map_adapter.model(map_batch)

                map_forward_runner = _map_forward_runner
                map_sae = base._load_sae_for_layer(index_cfg["sae"], str(target_spec), map_device)
                if args.method_map_debug:
                    _log_cuda_mem("[map][debug] after model+sae load", map_device)
                amp_key = str(args.method_map_amp_dtype or "").lower()
                if amp_key in {"bf16", "bfloat16"}:
                    amp_dtype = torch.bfloat16
                elif amp_key in {"float16", "fp16", "half"}:
                    amp_dtype = torch.float16
                elif amp_key in {"float32", "fp32"}:
                    amp_dtype = torch.float32
                else:
                    amp_dtype = None
                amp_enabled = (
                    (not bool(args.no_method_map_amp))
                    and amp_dtype is not None
                    and str(map_device).startswith("cuda")
                )
                method_maps = _compute_method_output_maps(
                    methods=method_order,
                    model=map_model,
                    adapter=map_adapter,
                    forward_runner=map_forward_runner,
                    sae=map_sae,
                    target_spec=str(target_spec),
                    target_unit=int(target_unit),
                    mask_specs=method_map_mask_specs,
                    output_spec=str(output_spec),
                    lane_idx=lane_idx,
                    num_frames=num_frames,
                    num_lanes=num_lanes,
                    ig_steps=int(args.ig_steps),
                    smoothgrad_samples=int(args.smoothgrad_samples),
                    smoothgrad_sigma=float(args.smoothgrad_sigma),
                    forward_baseline=str(args.method_map_baseline),
                    debug=debug,
                    log_debug=bool(args.method_map_debug),
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype or torch.float32,
                )
                frames_for_overlay = viz._frames_from_batch(map_batch)
                prompt_points = viz._extract_prompt_points(
                    map_adapter,
                    frames_for_overlay,
                    lane_idx,
                    debug=bool(args.method_map_debug) or debug,
                )
                activation_stack: Optional[torch.Tensor] = None
                try:
                    act_runtime = AttributionRuntime(
                        model=map_model,
                        adapter=map_adapter,
                        forward_fn=lambda: map_forward_runner(require_grad=False),
                        sae_module=map_sae,
                        target=RuntimeTarget(layer=str(target_spec), unit=int(target_unit)),
                    )
                    act_runtime.configure_anchors(
                        AnchorConfig(capture=list(method_map_mask_specs), ig_active=())
                    )
                    act_runtime.set_override_lane(lane_idx)
                    act_runtime._run_forward(require_grad=False)
                    reshape_meta = act_runtime._target_sae_branch.sae_context().get("reshape_meta")
                    act_stack = act_runtime.controller.post_stack(detach=True)
                    activation_stack = viz._activation_map_stack(
                        act_stack,
                        reshape_meta=reshape_meta if isinstance(reshape_meta, dict) else None,
                        lane_idx=lane_idx,
                        unit_idx=int(target_unit),
                        debug=debug,
                    )
                    if activation_stack is not None and frames_for_overlay:
                        activation_stack = viz._resize_map_stack(
                            activation_stack.detach().cpu(),
                            frames_for_overlay,
                        )
                        activation_stack, frames_for_overlay = viz._align_frames_to_list(
                            activation_stack,
                            frames_for_overlay,
                            debug=debug,
                            label="activation_map",
                        )
                finally:
                    try:
                        act_runtime.cleanup()
                    except Exception:
                        pass
                return method_maps, activation_stack, frames_for_overlay, prompt_points

            if not args.method_map_device or str(args.method_map_device).startswith("cuda"):
                try:
                    model.to("cpu")
                except Exception:
                    pass
            base._gpu_gc()
            map_device = torch.device(args.method_map_device) if args.method_map_device else device
            try:
                method_maps, activation_stack, frames_for_overlay, map_prompt_points = _run_method_maps(map_device)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and str(map_device).startswith("cuda"):
                    print("[warn] method map OOM on CUDA; retrying on CPU")
                    base._gpu_gc()
                    method_maps, activation_stack, frames_for_overlay, map_prompt_points = _run_method_maps(torch.device("cpu"))
                else:
                    raise
            # Explicitly drop heavy objects to release VRAM.
            try:
                map_batch = None
                map_adapter = None
                map_model = None
                map_sae = None
                base._gpu_gc()
            except Exception:
                pass

            shapley_map = mean_map_cpu if "mean_map_cpu" in locals() else shapley_maps[0].detach().cpu()

            def _metrics_for_target(target_map: torch.Tensor) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for method_label, method_map in method_maps.items():
                    smap, mmap = _align_maps(target_map, method_map.detach().cpu())
                    smap_flat = smap.reshape(-1).numpy()
                    mmap_flat = mmap.reshape(-1).numpy()
                    out[method_label] = {
                        "pearson": float(_pearson_corr(mmap_flat, smap_flat)),
                        "spearman": float(_spearman_corr(mmap_flat, smap_flat)),
                        "cosine": float(_cosine_sim_flat(mmap, smap)),
                    }
                return out

            method_map_metrics = _metrics_for_target(shapley_map)
            if shapley_max_map is not None:
                method_map_metrics_extremes["max"] = _metrics_for_target(shapley_max_map)
            if shapley_min_map is not None:
                method_map_metrics_extremes["min"] = _metrics_for_target(shapley_min_map)

            if base_stack_cpu is not None and frames_for_overlay:
                try:
                    panel_dir.mkdir(parents=True, exist_ok=True)
                    viz.save_mask_logits_panels(
                        out_dir=panel_dir,
                        sid=int(row.get("sample_id", 0)),
                        lane_idx=int(lane_idx) if lane_idx is not None else 0,
                        score_suffix="",
                        mask_logits=base_stack_cpu,
                        frames=frames_for_overlay,
                    )
                except Exception as exc:
                    print(f"[warn] failed to save mask panels: {exc}")
            if activation_stack is not None and frames_for_overlay:
                try:
                    stub = f"sae_activation_unit{int(target_unit)}"
                    viz.save_feature_activation_overlay(
                        out_dir=panel_dir,
                        sid=int(row.get("sample_id", 0)),
                        score_suffix="",
                        map_stack=activation_stack,
                        frames=frames_for_overlay,
                        prompt_points=map_prompt_points,
                        prompt_color=prompt_color,
                        overlay_alpha=0.4,
                        overlay_cmap="plasma",
                        file_stub=stub,
                        min_abs=0.0,
                    )
                except Exception as exc:
                    print(f"[warn] failed to save activation panel: {exc}")
            if base_stack_cpu is not None:
                for method_label, method_map in method_maps.items():
                    try:
                        method_cpu = method_map.detach().cpu()
                        if method_cpu.dtype != torch.float32:
                            method_cpu = method_cpu.float()
                        max_abs = float(method_cpu.abs().max().item()) if method_cpu.numel() else 0.0
                        if max_abs > 0:
                            normed = method_cpu / max_abs
                        else:
                            normed = method_cpu
                        normed, base_aligned = viz._align_frames(
                            normed,
                            base_stack_cpu,
                            debug=debug,
                            label=f"method_map:{method_label}",
                        )
                        viz.save_output_contribution_overlay(
                            out_dir=panel_dir,
                            sid=int(row.get("sample_id", 0)),
                            score_suffix="",
                            heat_stack=normed,
                            target_tensor=base_aligned,
                            prompt_points=map_prompt_points,
                            prompt_color=prompt_color,
                            overlay_alpha=0.4,
                            overlay_cmap="bwr",
                            use_abs_overlay=False,
                            overlay_on_base=False,
                            apply_sigmoid=True,
                            file_stub=f"{base._sanitize_path_token(str(method_label))}_unit{int(target_unit)}",
                        )
                    except Exception as exc:
                        print(f"[warn] failed to save method panel {method_label}: {exc}")

    if shapley_scores:
        shapley_sum = float(sum(shapley_scores))
    else:
        shapley_sum = 0.0
    full_delta = float(full_value - base_value) if not args.no_scalar_shapley else 0.0

    metrics: Dict[str, Any] = {
        "meta": {
            "sample_id": int(row.get("sample_id", -1)),
            "frame_idx": int(row.get("frame_idx", 0)),
            "lane_idx": int(lane_idx) if lane_idx is not None else None,
            "objective_mode": objective_mode,
            "base_objective_full": float(base_obj),
            "base_value": float(base_value),
            "full_value": float(full_value),
            "full_delta": float(full_delta),
            "phi_sum": float(shapley_sum),
            "phi_sum_error": float(shapley_sum - full_delta),
            "candidate_mode": "minfeat" if bool(args.candidate_minfeat) else ("list" if args.candidate_list is not None else "topk"),
            "shapley_map": bool(args.shapley_map),
            "shapley_map_samples": int(args.shapley_map_samples),
            "shapley_map_target_spec": args.shapley_map_target_spec,
            "shapley_map_target_unit": int(args.shapley_map_target_unit) if args.shapley_map_target_unit is not None else int(args.unit),
            "no_scalar_shapley": bool(args.no_scalar_shapley),
        },
    }
    if method_map_metrics:
        metrics["method_map_metrics"] = method_map_metrics
    if method_map_metrics_extremes:
        metrics["method_map_metrics_extremes"] = method_map_metrics_extremes
    if shapley_extremes:
        metrics["shapley_map_extremes"] = shapley_extremes
    if minfeat_info:
        metrics["candidate_minfeat"] = {
            "metric": str(args.candidate_metric).lower().strip(),
            "target": float(args.candidate_target_iou if str(args.candidate_metric).lower().strip() == "iou" else args.candidate_target_fraction),
            "chunk_size": int(args.candidate_chunk_size),
            "max_steps": int(args.candidate_max_steps) if args.candidate_max_steps is not None else None,
            "best_method": minfeat_selected,
            "rank_positive_override": bool(args.rank_positive),
            "per_method": minfeat_info,
        }

    score_payload: Dict[str, Any] = {
        "meta": metrics["meta"],
        "candidates": [
            {"spec": spec, "unit": int(unit), "sources": candidate_sources.get((spec, unit), [])}
            for spec, unit in candidates
        ],
        "scores": {},
    }

    shapley_payload = {
        "meta": metrics["meta"],
        "values": [
            {"spec": spec, "unit": int(unit), "phi": float(phi), "rank": int(idx + 1)}
            for idx, ((spec, unit), phi) in enumerate(zip(candidates, shapley_scores))
        ],
    }

    for method_label in method_order:
        ranking = rankings_by_method[method_label]["ranking"]
        method_scores = _scores_for_candidates(ranking, candidates)
        score_payload["scores"][method_label] = [float(val) for val in method_scores]

    candidates_tsv = ["rank\tspec\tunit\tsources"]
    for idx, (spec, unit) in enumerate(candidates, start=1):
        srcs = ",".join(candidate_sources.get((spec, unit), []))
        candidates_tsv.append(f"{idx}\t{spec}\t{int(unit)}\t{srcs}")
    (out_root / "candidates.tsv").write_text("\n".join(candidates_tsv) + "\n", encoding="utf-8")
    (out_root / "shapley.json").write_text(json.dumps(shapley_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_root / "method_scores.json").write_text(json.dumps(score_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if method_map_metrics:
        (out_root / "method_map_metrics.json").write_text(
            json.dumps(method_map_metrics, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    if shapley_extremes:
        (out_root / "shapley_extremes.json").write_text(
            json.dumps(shapley_extremes, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    if shapley_max_map is not None or shapley_min_map is not None:
        try:
            payload = {
                "meta": metrics["meta"],
                "extremes": shapley_extremes,
                "max_map": shapley_max_map.detach().cpu() if shapley_max_map is not None else None,
                "min_map": shapley_min_map.detach().cpu() if shapley_min_map is not None else None,
            }
            torch.save(payload, out_root / "shapley_extreme_maps.pt")
        except Exception as exc:
            print(f"[warn] failed to save shapley extreme maps: {exc}")
    if method_map_metrics_extremes.get("max"):
        (out_root / "method_map_metrics_max.json").write_text(
            json.dumps(method_map_metrics_extremes.get("max"), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    if method_map_metrics_extremes.get("min"):
        (out_root / "method_map_metrics_min.json").write_text(
            json.dumps(method_map_metrics_extremes.get("min"), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    if not args.no_shapley_pt:
        _save_shapley_pt(
            out_root,
            candidates=candidates,
            shapley_scores=shapley_scores,
            meta=metrics["meta"],
            sources=candidate_sources,
        )
    if not args.no_shapley_panel:
        _save_shapley_panel(
            out_root,
            candidates=candidates,
            shapley_scores=shapley_scores,
            topk=int(args.shapley_panel_topk),
        )
    if args.shapley_map and shapley_maps is not None and shapley_map_indices is not None:
        if not args.no_shapley_map_pt:
            _save_shapley_map_pt(
                out_root,
                candidates=candidates,
                map_scores=shapley_maps,
                map_indices=shapley_map_indices,
                meta=metrics["meta"],
            )
        if args.shapley_map_viz_topk > 0 and base_stack is not None and not locals().get("mean_panel_saved", False):
            _save_shapley_map_panels(
                out_dir=panel_dir,
                candidates=candidates,
                map_scores=shapley_maps,
                map_indices=shapley_map_indices,
                base_stack=base_stack,
                sample_id=int(row.get("sample_id", 0)),
                topk=int(args.shapley_map_viz_topk),
                prompt_points=prompt_points,
                prompt_color=prompt_color,
                overlay_on_base=False,
                apply_sigmoid=True,
                debug=debug,
            )
    if convergence_trace:
        conv_path = out_root / "shapley_map_convergence.json"
        conv_payload = [{"samples": int(n), "cosine_to_final": float(val)} for n, val in convergence_trace]
        conv_path.write_text(json.dumps(conv_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            xs = [int(n) for n, _ in convergence_trace]
            ys = [float(v) for _, v in convergence_trace]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, marker="o")
            plt.xlabel("Samples")
            plt.ylabel("Cosine similarity to final map")
            plt.title("Shapley map convergence (Owen sampling)")
            plt.ylim(-1.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_root / "shapley_map_convergence.png", dpi=150)
            plt.close()
        except Exception as exc:
            print(f"[warn] failed to save convergence plot: {exc}")

    try:
        mask_capture.clear()
    except Exception:
        pass
    for handle in reversed(restore_handles):
        handle()


if __name__ == "__main__":
    main()
