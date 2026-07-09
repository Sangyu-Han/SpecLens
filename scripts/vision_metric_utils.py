#!/usr/bin/env python3
"""Shared vision insertion/deletion metric helpers.

Use raw class-probability AUC for insertion/deletion. Do not use the first two
values returned by the archived frontier hard_curves helper for metric
comparison: those are normalized by full-image probability and can exceed 1.
"""
from __future__ import annotations

import warnings

import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", np.trapz)
    return float(fn(np.asarray(y, np.float64), np.asarray(x, np.float64)))


def _validate_raw_auc(ins_auc: float, del_auc: float, eps: float = 1e-5) -> tuple[float, float]:
    if not np.isfinite(ins_auc) or not np.isfinite(del_auc):
        raise ValueError(f"raw-prob AUC is not finite: ins={ins_auc}, del={del_auc}")
    if ins_auc < -eps or ins_auc > 1.0 + eps or del_auc < -eps or del_auc > 1.0 + eps:
        raise ValueError(
            "raw-prob AUC must be in [0,1]. "
            f"Got ins={ins_auc:.6f}, del={del_auc:.6f}; this usually means a normalized/recovery AUC leaked in."
        )
    return float(np.clip(ins_auc, 0.0, 1.0)), float(np.clip(del_auc, 0.0, 1.0))


def raw_auc_from_probs(p_ins: np.ndarray, p_del: np.ndarray) -> tuple[float, float]:
    """Integrate raw insertion/deletion class probabilities over [0,1]."""
    p_ins = np.asarray(p_ins, np.float64).reshape(-1)
    p_del = np.asarray(p_del, np.float64).reshape(-1)
    if p_ins.shape != p_del.shape or p_ins.size < 2:
        raise ValueError(f"invalid raw curves: p_ins={p_ins.shape}, p_del={p_del.shape}")
    xs = np.linspace(0.0, 1.0, p_ins.size)
    return _validate_raw_auc(_trapz(p_ins, xs), _trapz(p_del, xs))


def _prob_curve(runner, masks: np.ndarray, chunk: int | None) -> np.ndarray:
    if chunk is None:
        return np.asarray(runner.prob_curve(masks), np.float64)
    try:
        return np.asarray(runner.prob_curve(masks, chunk), np.float64)
    except TypeError:
        return np.asarray(runner.prob_curve(masks), np.float64)


def raw_curves_manual(runner, score: np.ndarray, n_patches: int | None = None, chunk: int | None = 64) -> tuple[np.ndarray, np.ndarray]:
    """Build hard insertion/deletion masks and read raw probability curves."""
    score = np.asarray(score, np.float32).reshape(-1)
    n = int(n_patches or score.size)
    if score.size != n:
        raise ValueError(f"score length {score.size} does not match n_patches={n}")
    order = np.argsort(-score)
    ins_masks = np.zeros((n + 1, n), np.float32)
    del_masks = np.ones((n + 1, n), np.float32)
    for k, idx in enumerate(order, start=1):
        ins_masks[k] = ins_masks[k - 1]
        ins_masks[k, int(idx)] = 1.0
        del_masks[k] = del_masks[k - 1]
        del_masks[k, int(idx)] = 0.0
    return _prob_curve(runner, ins_masks, chunk), _prob_curve(runner, del_masks, chunk)


def raw_auc_from_hard_curves(frontier, runner, score: np.ndarray, n_patches: int | None = None, chunk: int | None = 64) -> tuple[float, float]:
    """Return corrected raw-prob insertion/deletion AUC.

    If the frontier hard_curves helper exposes raw probability curves as return
    values 2 and 3, use them. Otherwise fall back to building hard curves here.
    """
    score = np.asarray(score, np.float32).reshape(-1)
    try:
        try:
            result = frontier.hard_curves(runner, score, chunk=chunk)
        except TypeError:
            result = frontier.hard_curves(runner, score)
        if isinstance(result, (tuple, list)) and len(result) >= 4:
            return raw_auc_from_probs(result[2], result[3])
        warnings.warn(
            "hard_curves did not return raw probability curves; falling back to manual raw-prob curves.",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception as exc:
        warnings.warn(
            f"hard_curves failed ({type(exc).__name__}: {exc}); falling back to manual raw-prob curves.",
            RuntimeWarning,
            stacklevel=2,
        )
    return raw_auc_from_probs(*raw_curves_manual(runner, score, n_patches=n_patches, chunk=chunk))
