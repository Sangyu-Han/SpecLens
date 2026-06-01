from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


MaskObjective = Callable[[torch.Tensor], torch.Tensor]
ScoreEvaluator = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class FRIConfig:
    """Configuration for Feature Relevance via soft insertion optimization."""

    steps: int = 32
    lr: float = 0.45
    lr_end: float = 0.01
    tv_weight: float = 0.01
    irrelevance_weight: float = 0.05
    init_prob: float = 0.5
    init_scores: Optional[np.ndarray] = None
    reg_warmup_frac: float = 0.0
    restarts: int = 1
    budget_samples: int = 1
    select_best: bool = False
    objective_mode: str = "random_budget_softins"
    optimizer_mode: str = "cautious_adam_cosine"
    fixed_budget_frac: float = 0.10
    seed: int = 0


@dataclass(frozen=True)
class FRIResult:
    scores: np.ndarray
    best_objective: float | None = None


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    mx = float(arr.max()) if arr.size else 0.0
    if mx <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / mx).astype(np.float32)


def _tv_loss_grid(z: torch.Tensor, grid_size: int) -> torch.Tensor:
    g = z.view(int(grid_size), int(grid_size))
    return (g[:, :-1] - g[:, 1:]).abs().sum() + (g[:-1, :] - g[1:, :]).abs().sum()


def _baseline_corrected_recovery(
    value: torch.Tensor,
    full_value: torch.Tensor,
    baseline_value: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    denom = full_value - baseline_value
    denom_safe = torch.where(
        denom.abs() >= eps,
        denom,
        torch.where(denom >= 0, torch.full_like(denom, eps), torch.full_like(denom, -eps)),
    )
    return (value - baseline_value) / denom_safe


def inverse_grad_irrelevance(
    *,
    input_patches: torch.Tensor,
    objective_from_patches: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Compute legacy FRI inverse-gradient irrelevance for patch embeddings.

    `input_patches` is cloned and made differentiable. `objective_from_patches`
    must return the scalar target objective after injecting those patches.
    """

    h_var = input_patches.detach().clone().requires_grad_(True)
    objective = objective_from_patches(h_var)
    objective.backward()
    grad = h_var.grad
    if grad is None:
        raise RuntimeError("FRI inverse-gradient irrelevance failed: missing patch gradient")
    grad_norm = grad[0].norm(dim=-1)
    inv = 1.0 / (grad_norm + 1e-8)
    return (inv / inv.max().clamp(min=1e-8)).detach().reshape(-1)


def run_fri(
    *,
    n_patches: int,
    grid_size: int,
    objective_for_mask: MaskObjective,
    full_objective: torch.Tensor,
    baseline_objective: torch.Tensor,
    irrelevance: torch.Tensor,
    config: FRIConfig | None = None,
    score_evaluator: ScoreEvaluator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FRIResult:
    """Run FRI and return patch attribution scores.

    The solver is model-agnostic: callers provide a differentiable
    `objective_for_mask(mask)` where `mask` is a length-`n_patches` insertion
    mask. This preserves the legacy `run_cautious_cos` algorithm while making
    it reusable across attribution runtimes.
    """

    cfg = config or FRIConfig()
    if device is None:
        device = full_objective.device
    dev = torch.device(device)
    if dtype is None:
        dtype = full_objective.dtype

    n_patches = int(n_patches)
    grid_size = int(grid_size)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    act_orig = full_objective.detach().to(device=dev, dtype=dtype)
    act_base = baseline_objective.detach().to(device=dev, dtype=dtype)
    irr = irrelevance.detach().reshape(-1).to(device=dev, dtype=dtype)
    warmup_steps = int(round(max(0.0, min(1.0, float(cfg.reg_warmup_frac))) * int(cfg.steps)))

    def _init_log_alphas() -> torch.Tensor:
        if cfg.init_scores is not None:
            init_norm = _normalize_scores(np.asarray(cfg.init_scores, dtype=np.float32).reshape(-1))
            init_floor = 0.05
            init_probs = init_floor + (1.0 - 2.0 * init_floor) * init_norm
            return torch.tensor([_logit(float(p)) for p in init_probs], device=dev, dtype=dtype)
        return torch.full((n_patches,), _logit(float(cfg.init_prob)), device=dev, dtype=dtype)

    def _run_once(run_seed: int) -> np.ndarray:
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(run_seed))
        log_alphas = _init_log_alphas()
        m_v = torch.zeros(n_patches, device=dev, dtype=dtype)
        v_v = torch.zeros(n_patches, device=dev, dtype=dtype)
        n_budget_samples = max(1, int(cfg.budget_samples))
        best_scores: Optional[np.ndarray] = None
        best_obj = -float("inf")

        for step in range(int(cfg.steps)):
            frac = step / max(int(cfg.steps) - 1, 1)
            cur_lr = float(cfg.lr_end) + 0.5 * (float(cfg.lr) - float(cfg.lr_end)) * (
                1 + math.cos(math.pi * frac)
            )

            la_req = log_alphas.clone().requires_grad_(True)
            probs = torch.sigmoid(la_req)
            recovery_terms: list[torch.Tensor] = []
            if cfg.objective_mode == "direct_recovery":
                act_masked = objective_for_mask(probs)
                recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                recovery_terms.append(1.0 - recovery)
            elif cfg.objective_mode in {"random_budget_softins", "fixed_budget_softins"}:
                p = probs / (probs.sum() + 1e-8)
                for _ in range(n_budget_samples):
                    if cfg.objective_mode == "random_budget_softins":
                        budget = float(torch.rand(1, generator=generator, device=dev).item() * n_patches)
                    else:
                        budget = float(max(0.0, min(float(cfg.fixed_budget_frac), 1.0)) * n_patches)
                    w = (p * budget).clamp(max=1.0)
                    act_masked = objective_for_mask(w)
                    recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                    recovery_terms.append(1.0 - recovery)
            else:
                raise ValueError(f"Unknown FRI objective_mode: {cfg.objective_mode!r}")

            recovery_loss = torch.stack(recovery_terms).mean()
            eff_irr_weight = 0.0 if step < warmup_steps else float(cfg.irrelevance_weight)
            eff_tv_weight = 0.0 if step < warmup_steps else float(cfg.tv_weight)
            loss = recovery_loss
            loss = loss + eff_irr_weight * (probs * irr).sum()
            loss = loss + eff_tv_weight * _tv_loss_grid(probs, grid_size)
            loss.backward()
            g = la_req.grad.detach()

            t = step + 1
            m_v = beta1 * m_v + (1 - beta1) * g
            v_v = beta2 * v_v + (1 - beta2) * g * g
            m_hat = m_v / (1 - beta1**t)
            v_hat = v_v / (1 - beta2**t)
            adam_dir = m_hat / (v_hat.sqrt() + eps)
            if cfg.optimizer_mode == "cautious_adam_cosine":
                mask = (adam_dir * g > 0).to(dtype=dtype)
                n_active = mask.sum().clamp(min=1.0)
                mask = mask * (n_patches / n_active)
                step_dir = adam_dir * mask
            elif cfg.optimizer_mode == "adam_cosine":
                step_dir = adam_dir
            else:
                raise ValueError(f"Unknown FRI optimizer_mode: {cfg.optimizer_mode!r}")
            log_alphas = log_alphas - cur_lr * step_dir

            if cfg.select_best and score_evaluator is not None and (
                step == int(cfg.steps) - 1 or (step + 1) % 4 == 0
            ):
                scores_np = torch.sigmoid(log_alphas).detach().cpu().numpy().astype(np.float32)
                obj = float(score_evaluator(scores_np))
                if obj > best_obj:
                    best_obj = obj
                    best_scores = scores_np

        final_scores = torch.sigmoid(log_alphas).detach().cpu().numpy().astype(np.float32)
        if cfg.select_best and best_scores is not None:
            return best_scores
        return final_scores

    best_scores: Optional[np.ndarray] = None
    best_obj = -float("inf")
    n_restarts = max(1, int(cfg.restarts))
    for restart_idx in range(n_restarts):
        scores = _run_once(int(cfg.seed) + 9973 * restart_idx)
        if score_evaluator is None:
            obj = 0.0 if best_scores is None else best_obj
        else:
            obj = float(score_evaluator(scores))
        if best_scores is None or obj > best_obj:
            best_obj = obj
            best_scores = scores

    if best_scores is None:
        best_scores = _run_once(int(cfg.seed))
    return FRIResult(scores=best_scores, best_objective=None if score_evaluator is None else best_obj)
