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
    l1_weight: float = 0.0
    init_prob: float = 0.5
    init_scores: Optional[np.ndarray] = None
    init_score_prob_floor: float = 0.05
    init_score_prob_ceiling: float = 0.95
    reg_warmup_frac: float = 0.0
    restarts: int = 1
    budget_samples: int = 1
    select_best: bool = False
    objective_mode: str = "random_budget_softins"
    optimizer_mode: str = "cautious_adam_cosine"
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 0.0
    fixed_budget_frac: float = 0.10
    random_budget_distribution: str = "uniform"
    random_budget_batch_mode: str = "independent"
    random_budget_batch_std: float = 0.08
    random_budget_loss_weight_mode: str = "uniform"
    insertion_weight: float = 1.0
    deletion_weight: float = 1.0
    hybrid_schedule: str = "full"
    budget_norm_grad: str = "full"
    budget_clamp_grad: str = "zero"
    score_mode: str = "final"
    score_modes: tuple[str, ...] = ()
    prune_steps: int = 0
    prune_lr: float = 0.05
    prune_lr_end: float = 0.005
    prune_l1_weight: float = 0.02
    prune_tv_weight: float = 0.0
    prune_tau_min: float = 0.60
    prune_tau_max: float = 0.90
    prune_temperature: float = 0.05
    prune_dropout_keep_prob: float = 1.0
    sinkhorn_temperature: float = 0.25
    sinkhorn_iters: int = 12
    sinkhorn_gumbel_scale: float = 0.0
    threshold_value: float = 0.0
    threshold_value_end: float = float("nan")
    threshold_value_cycle: str = ""
    threshold_value_mixture: str = ""
    threshold_temperature: float = 1.0
    threshold_temperature_end: float = 0.0
    threshold_temperature_cycle: str = ""
    threshold_temperature_mixture: str = ""
    budget_mask_noise_mode: str = ""
    budget_mask_noise_prob: float = 0.0
    budget_mask_noise_density: float = float("nan")
    score_logit_noise_mode: str = ""
    score_logit_noise_prob: float = 0.0
    score_logit_noise_force: float = 4.0
    incumbent_dropout_prob: float = 0.0
    incumbent_dropout_power: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class FRIResult:
    scores: np.ndarray
    best_objective: float | None = None
    score_map: dict[str, np.ndarray] | None = None


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
            init_floor = max(1e-4, min(float(cfg.init_score_prob_floor), 1.0 - 1e-4))
            init_ceiling = max(1e-4, min(float(cfg.init_score_prob_ceiling), 1.0 - 1e-4))
            if init_ceiling < init_floor:
                init_floor, init_ceiling = init_ceiling, init_floor
            init_probs = init_floor + (init_ceiling - init_floor) * init_norm
            return torch.tensor([_logit(float(p)) for p in init_probs], device=dev, dtype=dtype)
        return torch.full((n_patches,), _logit(float(cfg.init_prob)), device=dev, dtype=dtype)

    def _run_once(run_seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(run_seed))
        log_alphas = _init_log_alphas()
        init_log_alphas = log_alphas.detach().clone()
        m_v = torch.zeros(n_patches, device=dev, dtype=dtype)
        v_v = torch.zeros(n_patches, device=dev, dtype=dtype)
        grad_abs_accum = torch.zeros(n_patches, device=dev, dtype=dtype)
        grad_up_accum = torch.zeros(n_patches, device=dev, dtype=dtype)
        rec_support_net_accum = torch.zeros(n_patches, device=dev, dtype=dtype)
        del_support_accum = torch.zeros(n_patches, device=dev, dtype=dtype)
        trajectory_sum = torch.zeros(n_patches, device=dev, dtype=dtype)
        trajectory_log_sum = torch.zeros(n_patches, device=dev, dtype=dtype)
        trajectory_count = 0
        requested_modes = tuple(dict.fromkeys((str(cfg.score_mode), *[str(m) for m in cfg.score_modes])))
        needs_signed_recovery_grad = bool(set(requested_modes) & {
            "signed_grad",
            "signed_final_x_grad",
            "signed_final_plus_grad",
        })
        needs_deletion_recovery_grad = bool(set(requested_modes) & {
            "del_grad_up",
            "final_x_del_grad",
            "alpha_x_del_grad",
            "delta_alpha_x_del_grad",
            "softplus_alpha_x_del_grad",
            "hmean_softplus_alpha_del_grad",
            "min_softplus_alpha_del_grad",
            "hmean_final_del_grad",
            "min_final_del_grad",
            "final_x_grad_x_del",
            "delta_alpha_x_grad_x_del",
            "softplus_alpha_x_grad_x_del",
            "prune_survival_x_del",
            "prune_final_x_del",
        })
        prune_score_modes = {
            "prune_final",
            "prune_survival",
            "prune_survival_x_del",
            "prune_final_x_del",
            "prune_dropout_final",
            "prune_dropout_survival",
            "prune_protect",
            "prune_protect_x_final",
            "prune_dropout_protect",
            "prune_dropout_protect_x_final",
        }
        n_budget_samples = max(1, int(cfg.budget_samples))
        best_scores: Optional[np.ndarray] = None
        best_obj = -float("inf")

        def _norm_t(x: torch.Tensor) -> torch.Tensor:
            x = torch.nan_to_num(x.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            mx = x.max().clamp(min=1e-8)
            return x / mx

        def _product_root(*parts: torch.Tensor) -> torch.Tensor:
            out = torch.ones(n_patches, device=dev, dtype=dtype)
            for part in parts:
                out = out * _norm_t(part)
            return out.clamp(min=0.0).pow(1.0 / max(len(parts), 1))

        def _hmean_t(*parts: torch.Tensor) -> torch.Tensor:
            normed = [_norm_t(part) for part in parts]
            denom = torch.zeros(n_patches, device=dev, dtype=dtype)
            for part in normed:
                denom = denom + 1.0 / part.clamp(min=1e-8)
            return float(len(normed)) / denom.clamp(min=1e-8)

        def _min_t(*parts: torch.Tensor) -> torch.Tensor:
            normed = [_norm_t(part) for part in parts]
            return torch.stack(normed, dim=0).min(dim=0).values

        def _scores_from_state(
            final_scores_t: torch.Tensor,
            final_log_alphas_t: torch.Tensor,
            *,
            mode: str | None = None,
            prune_scores_t: torch.Tensor | None = None,
            prune_survival_t: torch.Tensor | None = None,
            prune_protect_t: torch.Tensor | None = None,
        ) -> np.ndarray:
            mode = str(cfg.score_mode if mode is None else mode)
            direct_sign: torch.Tensor | None = None
            if mode in {"signed_final_x_direct", "signed_final_plus_direct"}:
                ref_mask = final_scores_t.detach().clone().requires_grad_(True)
                direct_value = objective_for_mask(ref_mask)
                direct_recovery = _baseline_corrected_recovery(direct_value, act_orig, act_base)
                direct_grad = torch.autograd.grad(direct_recovery, ref_mask, retain_graph=False)[0]
                direct_sign = torch.where(
                    direct_grad.detach() < -1e-12,
                    torch.full_like(direct_grad.detach(), -1.0),
                    torch.ones_like(direct_grad.detach()),
                )
            delta_alpha_t = final_log_alphas_t - init_log_alphas
            softplus_alpha_t = torch.nn.functional.softplus(final_log_alphas_t)
            if mode == "final":
                out = final_scores_t
            elif mode == "alpha":
                out = final_log_alphas_t
            elif mode == "alpha_pos":
                out = final_log_alphas_t.clamp(min=0.0)
            elif mode == "delta_alpha":
                out = delta_alpha_t
            elif mode == "delta_alpha_pos":
                out = delta_alpha_t.clamp(min=0.0)
            elif mode == "grad_abs":
                out = grad_abs_accum
            elif mode == "grad_up":
                out = grad_up_accum
            elif mode == "signed_grad":
                out = rec_support_net_accum
            elif mode == "final_x_grad":
                out = torch.sqrt(_norm_t(final_scores_t) * _norm_t(grad_abs_accum))
            elif mode == "final_plus_grad":
                out = _norm_t(final_scores_t) + _norm_t(grad_abs_accum)
            elif mode == "trajectory_mean":
                out = trajectory_sum / max(int(trajectory_count), 1)
            elif mode == "trajectory_gmean":
                out = torch.exp(trajectory_log_sum / max(int(trajectory_count), 1))
            elif mode == "final_x_trajectory_mean":
                traj = trajectory_sum / max(int(trajectory_count), 1)
                out = torch.sqrt(_norm_t(final_scores_t) * _norm_t(traj))
            elif mode == "final_x_trajectory_gmean":
                traj = torch.exp(trajectory_log_sum / max(int(trajectory_count), 1))
                out = torch.sqrt(_norm_t(final_scores_t) * _norm_t(traj))
            elif mode == "alpha_x_grad":
                out = torch.sqrt(_norm_t(final_log_alphas_t) * _norm_t(grad_abs_accum))
            elif mode == "softplus_alpha_x_grad":
                out = torch.sqrt(_norm_t(softplus_alpha_t) * _norm_t(grad_abs_accum))
            elif mode == "delta_alpha_x_grad":
                out = torch.sqrt(_norm_t(delta_alpha_t) * _norm_t(grad_abs_accum))
            elif mode == "del_grad_up":
                out = del_support_accum
            elif mode == "final_x_del_grad":
                out = torch.sqrt(_norm_t(final_scores_t) * _norm_t(del_support_accum))
            elif mode == "alpha_x_del_grad":
                out = torch.sqrt(_norm_t(final_log_alphas_t) * _norm_t(del_support_accum))
            elif mode == "softplus_alpha_x_del_grad":
                out = torch.sqrt(_norm_t(softplus_alpha_t) * _norm_t(del_support_accum))
            elif mode == "hmean_softplus_alpha_del_grad":
                out = _hmean_t(softplus_alpha_t, del_support_accum)
            elif mode == "min_softplus_alpha_del_grad":
                out = _min_t(softplus_alpha_t, del_support_accum)
            elif mode == "hmean_final_del_grad":
                out = _hmean_t(final_scores_t, del_support_accum)
            elif mode == "min_final_del_grad":
                out = _min_t(final_scores_t, del_support_accum)
            elif mode == "delta_alpha_x_del_grad":
                out = torch.sqrt(_norm_t(delta_alpha_t) * _norm_t(del_support_accum))
            elif mode == "final_x_grad_x_del":
                out = _product_root(final_scores_t, grad_abs_accum, del_support_accum)
            elif mode == "delta_alpha_x_grad_x_del":
                out = _product_root(delta_alpha_t, grad_abs_accum, del_support_accum)
            elif mode == "softplus_alpha_x_grad_x_del":
                out = _product_root(softplus_alpha_t, grad_abs_accum, del_support_accum)
            elif mode == "prune_final":
                out = final_scores_t if prune_scores_t is None else prune_scores_t
            elif mode == "prune_survival":
                out = final_scores_t if prune_survival_t is None else prune_survival_t
            elif mode == "prune_survival_x_del":
                survival = final_scores_t if prune_survival_t is None else prune_survival_t
                out = torch.sqrt(_norm_t(survival) * _norm_t(del_support_accum))
            elif mode == "prune_final_x_del":
                pruned = final_scores_t if prune_scores_t is None else prune_scores_t
                out = torch.sqrt(_norm_t(pruned) * _norm_t(del_support_accum))
            elif mode == "prune_dropout_final":
                out = final_scores_t if prune_scores_t is None else prune_scores_t
            elif mode == "prune_dropout_survival":
                out = final_scores_t if prune_survival_t is None else prune_survival_t
            elif mode == "prune_protect":
                out = final_scores_t if prune_protect_t is None else prune_protect_t
            elif mode == "prune_protect_x_final":
                protect = final_scores_t if prune_protect_t is None else prune_protect_t
                pruned = final_scores_t if prune_scores_t is None else prune_scores_t
                out = torch.sqrt(_norm_t(protect) * _norm_t(pruned))
            elif mode == "prune_dropout_protect":
                out = final_scores_t if prune_protect_t is None else prune_protect_t
            elif mode == "prune_dropout_protect_x_final":
                protect = final_scores_t if prune_protect_t is None else prune_protect_t
                pruned = final_scores_t if prune_scores_t is None else prune_scores_t
                out = torch.sqrt(_norm_t(protect) * _norm_t(pruned))
            elif mode == "signed_final_x_grad":
                sign = rec_support_net_accum.sign()
                out = sign * torch.sqrt(_norm_t(final_scores_t) * _norm_t(grad_abs_accum))
            elif mode == "signed_final_plus_grad":
                sign = rec_support_net_accum.sign()
                out = sign * (_norm_t(final_scores_t) + _norm_t(grad_abs_accum))
            elif mode == "signed_final_x_direct":
                out = direct_sign * torch.sqrt(_norm_t(final_scores_t) * _norm_t(grad_abs_accum))
            elif mode == "signed_final_plus_direct":
                out = direct_sign * (_norm_t(final_scores_t) + _norm_t(grad_abs_accum))
            else:
                raise ValueError(f"Unknown FRI score_mode: {mode!r}")
            return out.detach().cpu().numpy().astype(np.float32)

        def _budget_distribution(mask_probs: torch.Tensor) -> torch.Tensor:
            denom = mask_probs.sum() + 1e-8
            if cfg.budget_norm_grad == "full":
                return mask_probs / denom
            if cfg.budget_norm_grad == "detach_denom":
                return mask_probs / denom.detach()
            raise ValueError(f"Unknown FRI budget_norm_grad: {cfg.budget_norm_grad!r}")

        def _clamp_budget_weight(raw_w: torch.Tensor) -> torch.Tensor:
            clamped = raw_w.clamp(max=1.0)
            if cfg.budget_clamp_grad == "zero":
                return clamped
            if cfg.budget_clamp_grad == "straight_through":
                return raw_w + (clamped - raw_w).detach()
            raise ValueError(f"Unknown FRI budget_clamp_grad: {cfg.budget_clamp_grad!r}")

        def _budget_mask_density(budget: float) -> float:
            explicit = float(cfg.budget_mask_noise_density)
            if math.isfinite(explicit):
                return max(0.0, min(explicit, 1.0))
            return max(0.0, min(float(budget) / float(max(n_patches, 1)), 1.0))

        def _apply_budget_mask_noise(mask_w: torch.Tensor, budget: float) -> torch.Tensor:
            mode = str(cfg.budget_mask_noise_mode).strip()
            prob = max(0.0, min(float(cfg.budget_mask_noise_prob), 1.0))
            if not mode or prob <= 0.0:
                return mask_w
            if mode == "dropout":
                keep = (
                    torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) >= prob
                ).to(dtype=dtype)
                return mask_w * keep
            if mode == "saltpepper":
                replace = (
                    torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) < prob
                )
                density = _budget_mask_density(budget)
                noise = (
                    torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) < density
                ).to(dtype=dtype)
                return torch.where(replace, noise, mask_w)
            if mode == "force_onoff":
                replace = (
                    torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) < prob
                )
                noise = (
                    torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) < 0.5
                ).to(dtype=dtype)
                return torch.where(replace, noise, mask_w)
            raise ValueError(f"Unknown FRI budget_mask_noise_mode: {mode!r}")

        def _apply_score_logit_noise(logits: torch.Tensor) -> torch.Tensor:
            mode = str(cfg.score_logit_noise_mode).strip()
            prob = max(0.0, min(float(cfg.score_logit_noise_prob), 1.0))
            if not mode or prob <= 0.0:
                return logits
            replace = torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) < prob
            force = float(cfg.score_logit_noise_force)
            if mode == "random":
                signs = torch.where(
                    torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) < 0.5,
                    torch.full((n_patches,), -1.0, device=dev, dtype=dtype),
                    torch.full((n_patches,), 1.0, device=dev, dtype=dtype),
                )
                target = signs * force
            elif mode == "force_on":
                target = torch.full_like(logits, force)
            elif mode == "force_off":
                target = torch.full_like(logits, -force)
            elif mode == "reverse":
                target = -logits.detach()
            elif mode == "forward":
                centered = logits.detach() - logits.detach().mean()
                signs = torch.where(centered >= 0, torch.ones_like(logits), -torch.ones_like(logits))
                target = logits.detach() + signs * force
            else:
                raise ValueError(f"Unknown FRI score_logit_noise_mode: {mode!r}")
            perturbed = torch.where(replace, target, logits)
            return logits + (perturbed - logits).detach()

        def _apply_incumbent_dropout(mask_w: torch.Tensor, mask_probs: torch.Tensor) -> torch.Tensor:
            prob = max(0.0, min(float(cfg.incumbent_dropout_prob), 1.0))
            if prob <= 0.0:
                return mask_w
            ref = mask_probs.detach().clamp(min=0.0)
            ref = ref / ref.max().clamp(min=1e-8)
            power = max(float(cfg.incumbent_dropout_power), 1e-6)
            drop_prob = (prob * ref.pow(power)).clamp(min=0.0, max=1.0)
            keep = (
                torch.rand(n_patches, generator=generator, device=dev, dtype=dtype) >= drop_prob
            ).to(dtype=dtype)
            return mask_w * keep

        def _sample_budget_frac() -> torch.Tensor:
            mode = str(cfg.random_budget_distribution).strip() or "uniform"
            u = torch.rand(1, generator=generator, device=dev, dtype=dtype)[0]
            if mode == "uniform":
                return u
            if mode == "low":
                return u * u
            if mode == "high":
                return 1.0 - (1.0 - u) * (1.0 - u)
            if mode == "mid":
                v = torch.rand(1, generator=generator, device=dev, dtype=dtype)[0]
                return (0.5 + 0.5 * (u - v)).clamp(min=0.0, max=1.0)
            if mode == "bimodal":
                v = torch.rand(1, generator=generator, device=dev, dtype=dtype)[0]
                low = 0.5 * (v * v)
                high = 1.0 - 0.5 * ((1.0 - v) * (1.0 - v))
                return torch.where(u < 0.5, low, high)
            raise ValueError(f"Unknown FRI random_budget_distribution: {mode!r}")

        rank_weights = torch.linspace(1.0, 0.0, steps=n_patches, device=dev, dtype=dtype)

        def _sinkhorn(logits: torch.Tensor) -> torch.Tensor:
            z = logits
            for _ in range(max(1, int(cfg.sinkhorn_iters))):
                z = z - torch.logsumexp(z, dim=1, keepdim=True)
                z = z - torch.logsumexp(z, dim=0, keepdim=True)
            return torch.exp(z)

        def _gumbel_noise(shape: tuple[int, ...]) -> torch.Tensor:
            u = torch.rand(shape, generator=generator, device=dev, dtype=dtype)
            u = u.clamp(min=1e-6, max=1.0 - 1e-6)
            return -torch.log(-torch.log(u))

        def _sample_budget() -> int:
            if cfg.objective_mode.startswith("random_budget"):
                if n_patches <= 1:
                    return 1
                budget = int(round(float(_sample_budget_frac().detach().cpu()) * float(n_patches)))
                return max(1, min(budget, n_patches))
            budget = int(round(max(0.0, min(float(cfg.fixed_budget_frac), 1.0)) * n_patches))
            return max(1, min(budget, n_patches))

        def _sinkhorn_topk_mask(scores: torch.Tensor, budget: int) -> torch.Tensor:
            budget = max(1, min(int(budget), n_patches))
            centered = scores - scores.mean()
            logits = centered.view(-1, 1) * rank_weights.view(1, -1)
            gumbel_scale = float(cfg.sinkhorn_gumbel_scale)
            if gumbel_scale > 0.0:
                logits = logits + gumbel_scale * _gumbel_noise((n_patches, n_patches))
            logits = logits / max(float(cfg.sinkhorn_temperature), 1e-6)
            soft_perm = _sinkhorn(logits)
            return soft_perm[:, :budget].sum(dim=1).clamp(min=0.0, max=1.0)

        for step in range(int(cfg.steps)):
            frac = step / max(int(cfg.steps) - 1, 1)
            cur_lr = float(cfg.lr_end) + 0.5 * (float(cfg.lr) - float(cfg.lr_end)) * (
                1 + math.cos(math.pi * frac)
            )
            budget_batch_mode = str(cfg.random_budget_batch_mode).strip() or "independent"
            budget_center_frac: torch.Tensor | None = None
            if budget_batch_mode == "gaussian":
                budget_center_frac = _sample_budget_frac()
            elif budget_batch_mode not in {"independent", "stratified", "lowmid_stratified"}:
                raise ValueError(f"Unknown FRI random_budget_batch_mode: {budget_batch_mode!r}")

            def _sample_step_budget_frac(sample_idx: int = 0) -> torch.Tensor:
                if budget_batch_mode == "independent":
                    return _sample_budget_frac()
                if budget_batch_mode == "stratified":
                    denom = max(n_budget_samples, 1)
                    center = (float(sample_idx) + 0.5) / float(denom)
                    center_t = torch.as_tensor(center, device=dev, dtype=dtype)
                    std_s = max(float(cfg.random_budget_batch_std), 0.0)
                    if std_s <= 0.0:
                        return center_t.clamp(min=0.0, max=1.0)
                    eps_s = torch.randn(1, generator=generator, device=dev, dtype=dtype)[0]
                    return (center_t + std_s * eps_s).clamp(min=0.0, max=1.0)
                if budget_batch_mode == "lowmid_stratified":
                    denom = max(n_budget_samples, 1)
                    center = (float(sample_idx) + 0.5) / float(2 * denom)
                    center_t = torch.as_tensor(center, device=dev, dtype=dtype)
                    std_s = max(float(cfg.random_budget_batch_std), 0.0)
                    if std_s <= 0.0:
                        return center_t.clamp(min=0.0, max=0.5)
                    eps_s = torch.randn(1, generator=generator, device=dev, dtype=dtype)[0]
                    return (center_t + std_s * eps_s).clamp(min=0.0, max=0.5)
                assert budget_center_frac is not None
                std = max(float(cfg.random_budget_batch_std), 0.0)
                if std <= 0.0:
                    return budget_center_frac
                eps_budget = torch.randn(1, generator=generator, device=dev, dtype=dtype)[0]
                return (budget_center_frac + std * eps_budget).clamp(min=0.0, max=1.0)

            def _sample_step_budget(sample_idx: int = 0) -> tuple[int, torch.Tensor]:
                if cfg.objective_mode.startswith("random_budget"):
                    if n_patches <= 1:
                        frac_t = torch.ones((), device=dev, dtype=dtype)
                        return 1, frac_t
                    frac_t = _sample_step_budget_frac(sample_idx)
                    budget = int(round(float(frac_t.detach().cpu()) * float(n_patches)))
                    return max(1, min(budget, n_patches)), frac_t
                frac_t = torch.as_tensor(
                    max(0.0, min(float(cfg.fixed_budget_frac), 1.0)),
                    device=dev,
                    dtype=dtype,
                )
                budget = int(round(float(frac_t.detach().cpu()) * n_patches))
                return max(1, min(budget, n_patches)), frac_t

            def _periodic_insert_period(schedule: str) -> int:
                raw = schedule.split(":", 1)[1] if ":" in schedule else schedule.rsplit("_", 1)[-1]
                try:
                    return max(2, int(raw))
                except ValueError as exc:
                    raise ValueError(f"Invalid FRI hybrid_schedule: {cfg.hybrid_schedule!r}") from exc

            la_req = log_alphas.clone().requires_grad_(True)
            probs = torch.sigmoid(la_req)
            mask_logits = _apply_score_logit_noise(la_req)
            mask_probs = torch.sigmoid(mask_logits)
            recovery_terms: list[torch.Tensor] = []
            recovery_weights: list[torch.Tensor] = []

            def _append_recovery(term: torch.Tensor, budget_frac: torch.Tensor | None = None) -> None:
                recovery_terms.append(term)
                mode = str(cfg.random_budget_loss_weight_mode).strip() or "uniform"
                if budget_frac is None or mode == "uniform":
                    recovery_weights.append(torch.ones((), device=dev, dtype=dtype))
                    return
                frac_w = budget_frac.detach().to(device=dev, dtype=dtype)
                frac_w = frac_w.clamp(min=1.0 / float(max(n_patches, 1)), max=1.0)
                if mode == "inv_sqrt_budget":
                    recovery_weights.append(torch.rsqrt(frac_w))
                    return
                if mode == "inv_budget":
                    recovery_weights.append(1.0 / frac_w)
                    return
                raise ValueError(f"Unknown FRI random_budget_loss_weight_mode: {mode!r}")

            step_phase = "direct"
            if cfg.objective_mode == "direct_recovery":
                act_masked = objective_for_mask(mask_probs)
                recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                _append_recovery(1.0 - recovery)
            elif cfg.objective_mode in {
                "random_budget_softins",
                "fixed_budget_softins",
                "random_budget_softdel",
                "fixed_budget_softdel",
                "random_budget_hybrid",
                "fixed_budget_hybrid",
            }:
                p = _budget_distribution(mask_probs)
                for sample_i in range(n_budget_samples):
                    if cfg.objective_mode.startswith("random_budget"):
                        budget_frac_t = _sample_step_budget_frac(sample_i)
                        budget = float(budget_frac_t.detach().cpu()) * float(n_patches)
                    else:
                        budget_frac_t = torch.as_tensor(
                            max(0.0, min(float(cfg.fixed_budget_frac), 1.0)),
                            device=dev,
                            dtype=dtype,
                        )
                        budget = float(budget_frac_t.detach().cpu()) * float(n_patches)
                    if cfg.objective_mode.endswith("softins"):
                        step_phase = "ins"
                        w = _apply_incumbent_dropout(
                            _apply_budget_mask_noise(_clamp_budget_weight(p * budget), budget),
                            mask_probs,
                        )
                        act_masked = objective_for_mask(w)
                        recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                        _append_recovery(1.0 - recovery, budget_frac_t)
                    elif cfg.objective_mode.endswith("softdel"):
                        step_phase = "del"
                        del_w = _apply_incumbent_dropout(
                            _apply_budget_mask_noise(_clamp_budget_weight(p * budget), budget),
                            mask_probs,
                        )
                        keep_w = 1.0 - del_w
                        act_masked = objective_for_mask(keep_w)
                        recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                        _append_recovery(recovery, budget_frac_t)
                    elif cfg.objective_mode.endswith("hybrid"):
                        raw_w = p * budget
                        ins_w = _apply_incumbent_dropout(
                            _apply_budget_mask_noise(_clamp_budget_weight(raw_w), budget),
                            mask_probs,
                        )
                        del_w = _apply_incumbent_dropout(
                            _apply_budget_mask_noise(_clamp_budget_weight(raw_w), budget),
                            mask_probs,
                        )
                        keep_w = 1.0 - del_w
                        schedule = str(cfg.hybrid_schedule)
                        if schedule == "full":
                            step_phase = "hybrid"
                            act_ins = objective_for_mask(ins_w)
                            rec_ins = _baseline_corrected_recovery(act_ins, act_orig, act_base)
                            act_del = objective_for_mask(keep_w)
                            rec_del = _baseline_corrected_recovery(act_del, act_orig, act_base)
                            _append_recovery(
                                float(cfg.insertion_weight) * (1.0 - rec_ins)
                                + float(cfg.deletion_weight) * rec_del,
                                budget_frac_t,
                            )
                        elif schedule == "alternating":
                            if step % 2 == 0:
                                step_phase = "ins"
                                act_ins = objective_for_mask(ins_w)
                                rec_ins = _baseline_corrected_recovery(act_ins, act_orig, act_base)
                                _append_recovery(float(cfg.insertion_weight) * (1.0 - rec_ins), budget_frac_t)
                            else:
                                step_phase = "del"
                                act_del = objective_for_mask(keep_w)
                                rec_del = _baseline_corrected_recovery(act_del, act_orig, act_base)
                                _append_recovery(float(cfg.deletion_weight) * rec_del, budget_frac_t)
                        elif schedule.startswith("periodic_insert"):
                            period = _periodic_insert_period(schedule)
                            if step % period == 0:
                                step_phase = "ins"
                                act_ins = objective_for_mask(ins_w)
                                rec_ins = _baseline_corrected_recovery(act_ins, act_orig, act_base)
                                _append_recovery(float(cfg.insertion_weight) * (1.0 - rec_ins), budget_frac_t)
                            else:
                                step_phase = "del"
                                act_del = objective_for_mask(keep_w)
                                rec_del = _baseline_corrected_recovery(act_del, act_orig, act_base)
                                _append_recovery(float(cfg.deletion_weight) * rec_del, budget_frac_t)
                        else:
                            raise ValueError(f"Unknown FRI hybrid_schedule: {cfg.hybrid_schedule!r}")
                    else:
                        raise ValueError(f"Unknown FRI objective_mode: {cfg.objective_mode!r}")
            elif cfg.objective_mode in {
                "random_budget_sinkhorn_softins",
                "fixed_budget_sinkhorn_softins",
                "random_budget_sinkhorn_softdel",
                "fixed_budget_sinkhorn_softdel",
                "random_budget_sinkhorn_hybrid",
                "fixed_budget_sinkhorn_hybrid",
            }:
                for sample_i in range(n_budget_samples):
                    budget, budget_frac_t = _sample_step_budget(sample_i)
                    top_w = _sinkhorn_topk_mask(mask_logits, budget)
                    if cfg.objective_mode.endswith("softins"):
                        step_phase = "ins"
                        act_masked = objective_for_mask(top_w)
                        recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                        _append_recovery(1.0 - recovery, budget_frac_t)
                    elif cfg.objective_mode.endswith("softdel"):
                        step_phase = "del"
                        act_masked = objective_for_mask(1.0 - top_w)
                        recovery = _baseline_corrected_recovery(act_masked, act_orig, act_base)
                        _append_recovery(recovery, budget_frac_t)
                    elif cfg.objective_mode.endswith("hybrid"):
                        schedule = str(cfg.hybrid_schedule)
                        if schedule == "full":
                            step_phase = "hybrid"
                            act_ins = objective_for_mask(top_w)
                            rec_ins = _baseline_corrected_recovery(act_ins, act_orig, act_base)
                            act_del = objective_for_mask(1.0 - top_w)
                            rec_del = _baseline_corrected_recovery(act_del, act_orig, act_base)
                            _append_recovery(
                                float(cfg.insertion_weight) * (1.0 - rec_ins)
                                + float(cfg.deletion_weight) * rec_del,
                                budget_frac_t,
                            )
                        elif schedule == "alternating":
                            if step % 2 == 0:
                                step_phase = "ins"
                                act_ins = objective_for_mask(top_w)
                                rec_ins = _baseline_corrected_recovery(act_ins, act_orig, act_base)
                                _append_recovery(float(cfg.insertion_weight) * (1.0 - rec_ins), budget_frac_t)
                            else:
                                step_phase = "del"
                                act_del = objective_for_mask(1.0 - top_w)
                                rec_del = _baseline_corrected_recovery(act_del, act_orig, act_base)
                                _append_recovery(float(cfg.deletion_weight) * rec_del, budget_frac_t)
                        elif schedule.startswith("periodic_insert"):
                            period = _periodic_insert_period(schedule)
                            if step % period == 0:
                                step_phase = "ins"
                                act_ins = objective_for_mask(top_w)
                                rec_ins = _baseline_corrected_recovery(act_ins, act_orig, act_base)
                                _append_recovery(float(cfg.insertion_weight) * (1.0 - rec_ins), budget_frac_t)
                            else:
                                step_phase = "del"
                                act_del = objective_for_mask(1.0 - top_w)
                                rec_del = _baseline_corrected_recovery(act_del, act_orig, act_base)
                                _append_recovery(float(cfg.deletion_weight) * rec_del, budget_frac_t)
                        else:
                            raise ValueError(f"Unknown FRI hybrid_schedule: {cfg.hybrid_schedule!r}")
                    else:
                        raise ValueError(f"Unknown FRI objective_mode: {cfg.objective_mode!r}")
            elif cfg.objective_mode in {
                "random_budget_threshold_softins",
                "fixed_budget_threshold_softins",
                "random_budget_threshold_softdel",
                "fixed_budget_threshold_softdel",
                "random_budget_threshold_hybrid",
                "fixed_budget_threshold_hybrid",
                "random_budget_threshold_sinkhorn_softins",
                "fixed_budget_threshold_sinkhorn_softins",
                "random_budget_threshold_sinkhorn_softdel",
                "fixed_budget_threshold_sinkhorn_softdel",
                "random_budget_threshold_sinkhorn_hybrid",
                "fixed_budget_threshold_sinkhorn_hybrid",
            }:
                p = _budget_distribution(mask_probs)
                threshold_cycle_raw = str(cfg.threshold_value_cycle).strip()
                if threshold_cycle_raw:
                    thresholds = [
                        float(part)
                        for part in threshold_cycle_raw.replace(";", ",").split(",")
                        if part.strip()
                    ]
                    if not thresholds:
                        raise ValueError(f"Invalid FRI threshold_value_cycle: {cfg.threshold_value_cycle!r}")
                    threshold_val = thresholds[step % len(thresholds)]
                else:
                    threshold_start = float(cfg.threshold_value)
                    threshold_end_raw = float(cfg.threshold_value_end)
                    if math.isfinite(threshold_end_raw):
                        threshold_val = threshold_start + (threshold_end_raw - threshold_start) * float(frac)
                    else:
                        threshold_val = threshold_start
                threshold_t = torch.as_tensor(threshold_val, device=dev, dtype=dtype)
                threshold_mix_raw = str(cfg.threshold_value_mixture).strip()
                if threshold_mix_raw:
                    threshold_vals = [
                        float(part)
                        for part in threshold_mix_raw.replace(";", ",").split(",")
                        if part.strip()
                    ]
                    if not threshold_vals:
                        raise ValueError(f"Invalid FRI threshold_value_mixture: {cfg.threshold_value_mixture!r}")
                else:
                    threshold_vals = [float(threshold_t.detach().cpu())]
                temp_cycle_raw = str(cfg.threshold_temperature_cycle).strip()
                if temp_cycle_raw:
                    temps = [
                        max(float(part), 1e-6)
                        for part in temp_cycle_raw.replace(";", ",").split(",")
                        if part.strip()
                    ]
                    if not temps:
                        raise ValueError(
                            f"Invalid FRI threshold_temperature_cycle: {cfg.threshold_temperature_cycle!r}"
                        )
                    temp_val = temps[step % len(temps)]
                else:
                    temp_start = max(float(cfg.threshold_temperature), 1e-6)
                    temp_end_raw = float(cfg.threshold_temperature_end)
                    if temp_end_raw > 0.0:
                        temp_end = max(temp_end_raw, 1e-6)
                        temp_val = temp_start * ((temp_end / temp_start) ** float(frac))
                    else:
                        temp_val = temp_start
                temp_t = torch.as_tensor(temp_val, device=dev, dtype=dtype)
                temp_mix_raw = str(cfg.threshold_temperature_mixture).strip()
                if temp_mix_raw:
                    temp_vals = [
                        max(float(part), 1e-6)
                        for part in temp_mix_raw.replace(";", ",").split(",")
                        if part.strip()
                    ]
                    if not temp_vals:
                        raise ValueError(
                            f"Invalid FRI threshold_temperature_mixture: {cfg.threshold_temperature_mixture!r}"
                        )
                else:
                    temp_vals = [float(temp_t.detach().cpu())]
                for sample_i in range(n_budget_samples):
                    if "sinkhorn" in cfg.objective_mode:
                        budget_i, budget_frac_t = _sample_step_budget(sample_i)
                        top_w = _sinkhorn_topk_mask(mask_logits, budget_i)
                        keep_w = 1.0 - top_w
                    elif cfg.objective_mode.startswith("random_budget"):
                        budget_frac_t = _sample_step_budget_frac(sample_i)
                        budget = float(budget_frac_t.detach().cpu()) * float(n_patches)
                        raw_w = p * budget
                        top_w = _apply_incumbent_dropout(_clamp_budget_weight(raw_w), mask_probs)
                        keep_w = 1.0 - _apply_incumbent_dropout(_clamp_budget_weight(raw_w), mask_probs)
                    else:
                        budget_frac_t = torch.as_tensor(
                            max(0.0, min(float(cfg.fixed_budget_frac), 1.0)),
                            device=dev,
                            dtype=dtype,
                        )
                        budget = float(budget_frac_t.detach().cpu()) * float(n_patches)
                        raw_w = p * budget
                        top_w = _clamp_budget_weight(raw_w)
                        keep_w = 1.0 - _clamp_budget_weight(raw_w)

                    def _active(mask_w: torch.Tensor) -> torch.Tensor:
                        value = objective_for_mask(mask_w)
                        acts = [
                            torch.sigmoid(
                                (value - torch.as_tensor(tv, device=dev, dtype=dtype))
                                / torch.as_tensor(tp, device=dev, dtype=dtype)
                            )
                            for tv in threshold_vals
                            for tp in temp_vals
                        ]
                        return torch.stack(acts).mean()

                    if cfg.objective_mode.endswith("softins"):
                        step_phase = "ins"
                        _append_recovery(1.0 - _active(top_w), budget_frac_t)
                    elif cfg.objective_mode.endswith("softdel"):
                        step_phase = "del"
                        _append_recovery(_active(keep_w), budget_frac_t)
                    elif cfg.objective_mode.endswith("hybrid"):
                        schedule = str(cfg.hybrid_schedule)
                        if schedule == "full":
                            step_phase = "hybrid"
                            _append_recovery(
                                float(cfg.insertion_weight) * (1.0 - _active(top_w))
                                + float(cfg.deletion_weight) * _active(keep_w),
                                budget_frac_t,
                            )
                        elif schedule == "alternating":
                            if step % 2 == 0:
                                step_phase = "ins"
                                _append_recovery(
                                    float(cfg.insertion_weight) * (1.0 - _active(top_w)),
                                    budget_frac_t,
                                )
                            else:
                                step_phase = "del"
                                _append_recovery(float(cfg.deletion_weight) * _active(keep_w), budget_frac_t)
                        elif schedule.startswith("periodic_insert"):
                            period = _periodic_insert_period(schedule)
                            if step % period == 0:
                                step_phase = "ins"
                                _append_recovery(
                                    float(cfg.insertion_weight) * (1.0 - _active(top_w)),
                                    budget_frac_t,
                                )
                            else:
                                step_phase = "del"
                                _append_recovery(float(cfg.deletion_weight) * _active(keep_w), budget_frac_t)
                        else:
                            raise ValueError(f"Unknown FRI hybrid_schedule: {cfg.hybrid_schedule!r}")
                    else:
                        raise ValueError(f"Unknown FRI objective_mode: {cfg.objective_mode!r}")
            else:
                raise ValueError(f"Unknown FRI objective_mode: {cfg.objective_mode!r}")

            if len(recovery_terms) == 1:
                recovery_loss = recovery_terms[0]
            else:
                terms_t = torch.stack(recovery_terms)
                weights_t = torch.stack(recovery_weights).to(device=dev, dtype=dtype)
                recovery_loss = (terms_t * weights_t).sum() / weights_t.sum().clamp(min=1e-8)
            eff_irr_weight = 0.0 if step < warmup_steps else float(cfg.irrelevance_weight)
            eff_l1_weight = 0.0 if step < warmup_steps else float(cfg.l1_weight)
            eff_tv_weight = 0.0 if step < warmup_steps else float(cfg.tv_weight)
            loss = recovery_loss
            loss = loss + eff_irr_weight * (probs * irr).sum()
            loss = loss + eff_l1_weight * probs.sum()
            loss = loss + eff_tv_weight * _tv_loss_grid(probs, grid_size)
            if needs_signed_recovery_grad or (needs_deletion_recovery_grad and step_phase == "del"):
                rec_g = torch.autograd.grad(recovery_loss, la_req, retain_graph=True)[0].detach()
            else:
                rec_g = None
            loss.backward()
            g = la_req.grad.detach()
            clip_norm = float(cfg.optimizer_grad_clip_norm)
            if clip_norm > 0.0:
                g_norm = g.pow(2).sum().sqrt()
                g = g * (clip_norm / g_norm.clamp(min=clip_norm))
            grad_abs_accum = grad_abs_accum + g.abs()
            grad_up_accum = grad_up_accum + (-g).clamp(min=0.0)
            if rec_g is not None:
                if needs_signed_recovery_grad:
                    rec_support_net_accum = rec_support_net_accum - rec_g
                if step_phase == "del":
                    del_support_accum = del_support_accum + (-rec_g).clamp(min=0.0)

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
            elif cfg.optimizer_mode == "sgd_cosine":
                step_dir = g
            elif cfg.optimizer_mode == "sgd_norm_cosine":
                step_dir = g / g.pow(2).mean().sqrt().clamp(min=1e-8)
            else:
                raise ValueError(f"Unknown FRI optimizer_mode: {cfg.optimizer_mode!r}")
            weight_decay = max(0.0, float(cfg.optimizer_weight_decay))
            if weight_decay > 0.0:
                log_alphas = log_alphas * max(0.0, 1.0 - cur_lr * weight_decay)
            log_alphas = log_alphas - cur_lr * step_dir
            probs_after = torch.sigmoid(log_alphas).detach()
            trajectory_sum = trajectory_sum + probs_after
            trajectory_log_sum = trajectory_log_sum + torch.log(probs_after.clamp(min=1e-8))
            trajectory_count += 1

            if cfg.select_best and score_evaluator is not None and (
                step == int(cfg.steps) - 1 or (step + 1) % 4 == 0
            ):
                scores_np = _scores_from_state(torch.sigmoid(log_alphas).detach(), log_alphas.detach())
                obj = float(score_evaluator(scores_np))
                if obj > best_obj:
                    best_obj = obj
                    best_scores = scores_np

        if cfg.select_best and best_scores is not None:
            return best_scores, {str(cfg.score_mode): best_scores}

        def _prune_from_state(
            start_log_alphas: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            prune_steps = int(cfg.prune_steps)
            if prune_steps <= 0:
                prune_steps = int(cfg.steps)
            prune_log_alphas = start_log_alphas.detach().clone()
            pm = torch.zeros(n_patches, device=dev, dtype=dtype)
            pv = torch.zeros(n_patches, device=dev, dtype=dtype)
            survival = torch.zeros(n_patches, device=dev, dtype=dtype)
            protect = torch.zeros(n_patches, device=dev, dtype=dtype)
            beta1_p, beta2_p = 0.9, 0.999
            tau_min = float(min(cfg.prune_tau_min, cfg.prune_tau_max))
            tau_max = float(max(cfg.prune_tau_min, cfg.prune_tau_max))
            temp = max(float(cfg.prune_temperature), 1e-6)
            dropout_keep_prob = max(0.0, min(float(cfg.prune_dropout_keep_prob), 1.0))
            use_dropout = str(cfg.score_mode) in {
                "prune_dropout_final",
                "prune_dropout_survival",
                "prune_dropout_protect",
                "prune_dropout_protect_x_final",
            }
            for prune_step in range(prune_steps):
                frac_p = prune_step / max(prune_steps - 1, 1)
                cur_prune_lr = float(cfg.prune_lr_end) + 0.5 * (
                    float(cfg.prune_lr) - float(cfg.prune_lr_end)
                ) * (1 + math.cos(math.pi * frac_p))
                la_req_p = prune_log_alphas.clone().requires_grad_(True)
                probs_p = torch.sigmoid(la_req_p)
                if use_dropout and dropout_keep_prob < 1.0:
                    keep_noise = (
                        torch.rand(n_patches, generator=generator, device=dev, dtype=dtype)
                        < dropout_keep_prob
                    ).to(dtype=dtype)
                    objective_mask = probs_p * keep_noise
                else:
                    objective_mask = probs_p
                value_p = objective_for_mask(objective_mask)
                recovery_p = _baseline_corrected_recovery(value_p, act_orig, act_base)
                if tau_max > tau_min:
                    tau = tau_min + (tau_max - tau_min) * float(
                        torch.rand(1, generator=generator, device=dev).item()
                    )
                else:
                    tau = tau_min
                barrier = temp * torch.nn.functional.softplus(
                    (torch.as_tensor(tau, device=dev, dtype=dtype) - recovery_p) / temp
                )
                prune_loss = barrier + float(cfg.prune_l1_weight) * probs_p.sum()
                prune_loss = prune_loss + float(cfg.prune_tv_weight) * _tv_loss_grid(probs_p, grid_size)
                barrier_g = torch.autograd.grad(barrier, la_req_p, retain_graph=True)[0].detach()
                protect = protect + (-barrier_g).clamp(min=0.0)
                prune_loss.backward()
                pg = la_req_p.grad.detach()
                pt = prune_step + 1
                if cfg.optimizer_mode == "sgd_cosine":
                    prune_log_alphas = prune_log_alphas - cur_prune_lr * pg
                elif cfg.optimizer_mode == "sgd_norm_cosine":
                    prune_log_alphas = prune_log_alphas - cur_prune_lr * pg / pg.pow(2).mean().sqrt().clamp(min=1e-8)
                else:
                    pm = beta1_p * pm + (1 - beta1_p) * pg
                    pv = beta2_p * pv + (1 - beta2_p) * pg * pg
                    pm_hat = pm / (1 - beta1_p**pt)
                    pv_hat = pv / (1 - beta2_p**pt)
                    prune_log_alphas = prune_log_alphas - cur_prune_lr * pm_hat / (pv_hat.sqrt() + eps)
                survival = survival + torch.sigmoid(prune_log_alphas).detach()
            return torch.sigmoid(prune_log_alphas).detach(), survival, protect

        final_scores_t = torch.sigmoid(log_alphas).detach()
        final_log_alphas_t = log_alphas.detach()
        if str(cfg.score_mode) in prune_score_modes:
            prune_scores_t, prune_survival_t, prune_protect_t = _prune_from_state(final_log_alphas_t)
            primary = _scores_from_state(
                final_scores_t,
                final_log_alphas_t,
                prune_scores_t=prune_scores_t,
                prune_survival_t=prune_survival_t,
                prune_protect_t=prune_protect_t,
            )
            extras = {
                mode: _scores_from_state(
                    final_scores_t,
                    final_log_alphas_t,
                    mode=mode,
                    prune_scores_t=prune_scores_t,
                    prune_survival_t=prune_survival_t,
                    prune_protect_t=prune_protect_t,
                )
                for mode in requested_modes
            }
            return primary, extras
        primary = _scores_from_state(final_scores_t, final_log_alphas_t)
        extras = {
            mode: _scores_from_state(final_scores_t, final_log_alphas_t, mode=mode)
            for mode in requested_modes
        }
        return primary, extras

    best_scores: Optional[np.ndarray] = None
    best_score_map: dict[str, np.ndarray] | None = None
    best_obj = -float("inf")
    n_restarts = max(1, int(cfg.restarts))
    for restart_idx in range(n_restarts):
        scores, score_map = _run_once(int(cfg.seed) + 9973 * restart_idx)
        if score_evaluator is None:
            obj = 0.0 if best_scores is None else best_obj
        else:
            obj = float(score_evaluator(scores))
        if best_scores is None or obj > best_obj:
            best_obj = obj
            best_scores = scores
            best_score_map = score_map

    if best_scores is None:
        best_scores, best_score_map = _run_once(int(cfg.seed))
    return FRIResult(
        scores=best_scores,
        best_objective=None if score_evaluator is None else best_obj,
        score_map=best_score_map,
    )
