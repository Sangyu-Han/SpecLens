from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
from PIL import Image

from .bootstrap import bootstrap_speclens, register_research_saes
from .config import EvalConfig
from .utils import token_uid


def _coarse_prefix_budgets(n_patches: int) -> list[int]:
    n_patches = int(n_patches)
    if n_patches <= 1:
        return [max(n_patches, 1)]

    budgets: list[int] = []
    current = 1
    while True:
        budgets.append(int(current))
        if current >= n_patches:
            break
        current = min(n_patches, current * 2)
    return budgets


def _search_minimal_support_prefix(
    *,
    n_patches: int,
    threshold: float,
    metric_name: str,
    recovery_cache: dict[int, float],
    evaluate_prefix,
    allow_empty_if_unreached: bool = False,
) -> dict[str, Any]:
    coarse_budgets = _coarse_prefix_budgets(n_patches)
    bracket_low = 0
    bracket_high = int(n_patches)
    threshold_reached = False

    zero_recovery = float(evaluate_prefix(0))
    if zero_recovery >= float(threshold):
        threshold_reached = True
        bracket_high = 0
        max_recovery = max((float(value) for value in recovery_cache.values()), default=0.0)
        recovery_trace = [
            {
                "budget": float(prefix_size),
                "score": float(recovery_cache[prefix_size]),
                metric_name: float(recovery_cache[prefix_size]),
            }
            for prefix_size in sorted(recovery_cache)
        ]
        return {
            "support_size": 0,
            "support_recovery": float(zero_recovery),
            "recovery_trace": recovery_trace,
            "recovery_search_mode": "exponential_binary_prefix",
            "recovery_coarse_budgets": [int(v) for v in coarse_budgets],
            "recovery_dense_limit": int(bracket_high),
            "recovery_bracket_low": int(bracket_low),
            "recovery_bracket_high": int(bracket_high),
            "recovery_eval_count": int(len(recovery_cache)),
            "recovery_threshold_reached": bool(threshold_reached),
            "recovery_max": float(max_recovery),
        }

    for prefix_size in coarse_budgets:
        recovery = evaluate_prefix(prefix_size)
        if recovery >= float(threshold):
            bracket_high = int(prefix_size)
            threshold_reached = True
            break
        bracket_low = int(prefix_size)

    support_size = int(n_patches)
    if threshold_reached:
        left = int(bracket_low) + 1
        right = int(bracket_high)
        while left < right:
            mid = (left + right) // 2
            recovery = evaluate_prefix(mid)
            if recovery >= float(threshold):
                right = mid
            else:
                left = mid + 1
        support_size = int(left)
    max_recovery = max((float(value) for value in recovery_cache.values()), default=0.0)
    support_recovery = float(evaluate_prefix(support_size))
    recovery_trace = [
        {
            "budget": float(prefix_size),
            "score": float(recovery_cache[prefix_size]),
            metric_name: float(recovery_cache[prefix_size]),
        }
        for prefix_size in sorted(recovery_cache)
    ]
    return {
        "support_size": int(support_size),
        "support_recovery": float(support_recovery),
        "recovery_trace": recovery_trace,
        "recovery_search_mode": "exponential_binary_prefix",
        "recovery_coarse_budgets": [int(v) for v in coarse_budgets],
        "recovery_dense_limit": int(bracket_high),
        "recovery_bracket_low": int(bracket_low),
        "recovery_bracket_high": int(bracket_high),
        "recovery_eval_count": int(len(recovery_cache)),
        "recovery_threshold_reached": bool(threshold_reached),
        "recovery_max": float(max_recovery),
    }


def _resolve_support_indices(
    *,
    valid_order: np.ndarray,
    proposed_support_size: int,
    threshold_reached: bool,
    evaluate_prefix,
) -> tuple[list[int], int, float]:
    max_valid = int(valid_order.size)
    if max_valid <= 0:
        if bool(threshold_reached) and int(proposed_support_size) <= 0:
            return [], 0, float(evaluate_prefix(0))
        raise ValueError("valid_order is empty; cannot construct ERF support")

    support_size = int(proposed_support_size)
    if support_size <= 0:
        if bool(threshold_reached):
            return [], 0, float(evaluate_prefix(0))
        support_size = int(max_valid)
    elif not bool(threshold_reached):
        # If the recovery threshold is not reached, keep all patches that survived
        # the attribution cutoff instead of permitting an empty/broken ERF.
        support_size = int(max_valid)
    support_size = max(1, min(int(support_size), int(max_valid)))

    support = valid_order[:support_size].tolist()
    if not support:
        raise ValueError("Resolved empty ERF support unexpectedly")
    support_recovery = float(evaluate_prefix(support_size))
    return support, int(support_size), float(support_recovery)


@dataclass
class ForwardArtifacts:
    x: torch.Tensor
    capture: Any
    patch_out: torch.Tensor


def ledger_row_x_to_token_idx(row_x: int, model_name: str) -> int:
    name = str(model_name).lower()
    # SigLIP deciles store patch-token x directly.
    if "siglip" in name:
        return int(row_x)
    # CLIP-style deciles store x with a +1 offset relative to patch tokens.
    return int(row_x) - 1


class LegacyRuntime:
    def __init__(self, config: EvalConfig):
        self.config = config
        bootstrap_speclens(config)
        register_research_saes(config)
        try:
            import torchvision.ops  # noqa: F401
        except Exception:
            for _name in list(sys.modules):
                if _name == "torchvision" or _name.startswith("torchvision."):
                    sys.modules.pop(_name, None)
            try:
                _tv_lib = torch.library.Library("torchvision", "DEF")
                _tv_lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
            except Exception:
                pass

        self.device = torch.device(config.device)
        model_name = str(config.model_name)
        model_name_lower = model_name.lower()
        if "siglip" in model_name_lower:
            from src.packs.siglip.attribution.erf_adapter import create_siglip_erf_adapter
            from src.packs.siglip.dataset.builders import build_siglip_transform
            from src.packs.siglip.models.model_loaders import load_siglip_model

            self.model = load_siglip_model({"name": model_name, "pretrained": True}, device=self.device)
            self.adapter = create_siglip_erf_adapter(self.model)
            self.transform = build_siglip_transform(
                {
                    "image_size": config.image_size,
                    "interpolation": "bicubic",
                    "is_train": False,
                }
            )
        else:
            from src.packs.clip.attribution.erf_adapter import create_clip_erf_adapter
            from src.packs.clip.dataset.builders import build_clip_transform
            from src.packs.clip.models.model_loaders import load_clip_model

            self.model = load_clip_model({"name": model_name, "pretrained": True}, device=self.device)
            self.adapter = create_clip_erf_adapter(self.model)
            self.transform = build_clip_transform(
                {
                    "image_size": config.image_size,
                    "mean": [0.48145466, 0.4578275, 0.40821073],
                    "std": [0.26862954, 0.26130258, 0.27577711],
                    "interpolation": "bicubic",
                    "is_train": False,
                }
            )
        self.model.eval()
        self.meta_ledger = None
        self._dataset_index: list[str] | None = None
        try:
            from src.packs.clip.offline.offline_meta_parquet import OfflineMetaParquetLedger

            parquet_root = Path(config.offline_meta_root) / "parquet"
            if parquet_root.exists():
                self.meta_ledger = OfflineMetaParquetLedger(str(config.offline_meta_root), part_modulus=128)
        except Exception:
            self.meta_ledger = None
        self._sae_cache: dict[int, torch.nn.Module] = {}

    def row_x_to_token_idx(self, row_x: int) -> int:
        return ledger_row_x_to_token_idx(int(row_x), str(self.config.model_name))

    def load_sae(self, block_idx: int) -> torch.nn.Module:
        block_idx = int(block_idx)
        if block_idx in self._sae_cache:
            return self._sae_cache[block_idx]
        from src.core.sae.registry import create_sae

        ckpt = torch.load(self.config.checkpoint_path(block_idx), map_location="cpu", weights_only=False)
        sae_cfg = dict(ckpt["sae_config"])
        for key, value in list(sae_cfg.items()):
            if isinstance(value, str) and "/" in value:
                path = Path(value)
                if not path.is_absolute():
                    candidate = self.config.repo_root / path
                    if candidate.exists():
                        sae_cfg[key] = str(candidate)
                    else:
                        root = Path("/home/sangyu/Desktop/Master")
                        legacy_candidate = root / path
                        if legacy_candidate.exists():
                            sae_cfg[key] = str(legacy_candidate)
                            continue
                        parts = path.parts
                        if parts and parts[0] == "claude_research_bias":
                            mapped = self.config.legacy_bias_root / Path(*parts[1:])
                            if mapped.exists():
                                sae_cfg[key] = str(mapped)
                                continue
                        if parts and parts[0] == "codex_research_bias":
                            mapped = self.config.legacy_variant_root / Path(*parts[1:])
                            if mapped.exists():
                                sae_cfg[key] = str(mapped)
                                continue
        sae_cfg["device"] = str(self.device)
        sae = create_sae(sae_cfg["sae_type"], sae_cfg)
        sae.load_state_dict(ckpt["sae_state"], strict=False)
        sae.eval()
        sae.to(self.device)
        self._sae_cache[block_idx] = sae
        return sae

    def lookup_paths(self, sample_ids: Sequence[int]) -> dict[int, str]:
        wanted = [int(sample_id) for sample_id in sample_ids]
        if self.meta_ledger is not None:
            try:
                out = self.meta_ledger.lookup(wanted)
            except Exception:
                out = {}
            if len(out) == len(set(wanted)):
                return out
        return self._filesystem_lookup_paths(wanted)

    def _filesystem_lookup_paths(self, sample_ids: Sequence[int]) -> dict[int, str]:
        root = self.config.dataset_root_override
        if root is None:
            return {}
        if self._dataset_index is None:
            root_path = Path(root)
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            paths: list[str] = []
            for class_dir in sorted([path for path in root_path.iterdir() if path.is_dir()]):
                class_files = [
                    path
                    for path in sorted(class_dir.iterdir())
                    if path.is_file() and path.suffix.lower() in image_exts
                ]
                paths.extend(str(path) for path in class_files)
            self._dataset_index = paths
        out: dict[int, str] = {}
        for sample_id in sample_ids:
            if 0 <= int(sample_id) < len(self._dataset_index):
                out[int(sample_id)] = self._dataset_index[int(sample_id)]
        return out

    def load_image_tensor(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def forward_block(self, image_path: str, block_idx: int) -> ForwardArtifacts:
        x = self.load_image_tensor(image_path)
        capture = self.adapter.capture_block0_input_and_target_block_out(x, block_idx=int(block_idx))
        prefix = self.adapter.prefix_count()
        patch_out = capture.target_block_output[:, prefix:, :]
        return ForwardArtifacts(x=x, capture=capture, patch_out=patch_out)

    def feature_activation_map(self, image_path: str, block_idx: int, feature_id: int) -> np.ndarray:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            acts = sae(artifacts.patch_out).get("feature_acts")
        return acts[0, :, int(feature_id)].detach().cpu().numpy().astype(np.float32)

    def feature_activation_map_full(self, image_path: str, block_idx: int, feature_id: int) -> np.ndarray:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            acts = sae(artifacts.capture.target_block_output).get("feature_acts")
        return acts[0, :, int(feature_id)].detach().cpu().numpy().astype(np.float32)

    def feature_activation_map_visible_patches(self, image_path: str, block_idx: int, feature_id: int) -> np.ndarray:
        full = self.feature_activation_map_full(image_path, block_idx, feature_id)
        prefix = int(self.adapter.prefix_count())
        return full[prefix:].astype(np.float32, copy=False)

    def feature_vector_at_token(
        self,
        image_path: str,
        block_idx: int,
        token_idx: int,
        feature_ids: Sequence[int],
    ) -> np.ndarray:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            acts = sae(artifacts.patch_out).get("feature_acts")
        feat_idx = torch.as_tensor([int(v) for v in feature_ids], device=acts.device, dtype=torch.long)
        vals = acts[0, int(token_idx), feat_idx]
        return vals.detach().cpu().numpy().astype(np.float32)

    def validate_feature_token(
        self,
        image_path: str,
        block_idx: int,
        feature_id: int,
        token_idx: int,
        ledger_score: float,
    ) -> dict[str, Any] | None:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            acts = sae(artifacts.patch_out).get("feature_acts")
        feat_map = acts[0, :, int(feature_id)]
        act_at_target = float(feat_map[int(token_idx)].detach().cpu())
        feat_np = feat_map.detach().cpu().numpy()
        argmax_tok = int(np.argmax(feat_np))
        max_act = float(feat_np[argmax_tok])
        ratio = act_at_target / max(max_act, 1e-8)
        if max_act < self.config.min_feature_max_act:
            return None
        if ratio < self.config.target_ratio_min:
            return None
        return {
            "act_at_target": act_at_target,
            "max_act": max_act,
            "argmax_tok": argmax_tok,
            "target_to_max_ratio": ratio,
            "ledger_score": float(ledger_score),
            "abs_score_err": abs(act_at_target - float(ledger_score)),
        }

    def validate_feature_special_token(
        self,
        image_path: str,
        block_idx: int,
        feature_id: int,
        token_x: int,
        ledger_score: float,
    ) -> dict[str, Any] | None:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            acts = sae(artifacts.capture.target_block_output).get("feature_acts")
        feat_map = acts[0, :, int(feature_id)]
        act_at_target = float(feat_map[int(token_x)].detach().cpu())
        feat_np = feat_map.detach().cpu().numpy()
        argmax_tok = int(np.argmax(feat_np))
        max_act = float(feat_np[argmax_tok])
        ratio = act_at_target / max(max_act, 1e-8)
        if max_act < self.config.min_feature_max_act:
            return None
        if ratio < self.config.target_ratio_min:
            return None
        return {
            "act_at_target": act_at_target,
            "max_act": max_act,
            "argmax_tok": argmax_tok,
            "target_to_max_ratio": ratio,
            "ledger_score": float(ledger_score),
            "abs_score_err": abs(act_at_target - float(ledger_score)),
        }

    def token_cosine_map(self, image_path: str, block_idx: int, token_idx: int) -> np.ndarray:
        artifacts = self.forward_block(image_path, block_idx)
        target = artifacts.patch_out[0, int(token_idx), :]
        patches = artifacts.patch_out[0]
        sims = F.cosine_similarity(patches, target.unsqueeze(0), dim=-1)
        return sims.detach().cpu().numpy().astype(np.float32)

    def _objective_metric_name(self, objective_mode: str) -> str:
        if objective_mode == "cosine":
            return "cosine"
        if objective_mode == "unit_ref_dot":
            return "unit_ref_dot_recovery"
        raise ValueError(f"Unsupported ERF objective mode: {objective_mode!r}")

    def _token_recovery_objective(
        self,
        masked_target: torch.Tensor,
        target_vec: torch.Tensor,
        *,
        objective_mode: str,
    ) -> torch.Tensor:
        if objective_mode == "cosine":
            return F.cosine_similarity(masked_target.unsqueeze(0), target_vec.unsqueeze(0), dim=-1).sum()
        if objective_mode == "unit_ref_dot":
            target_unit = F.normalize(target_vec.unsqueeze(0), dim=-1).squeeze(0)
            full_proj = (target_vec * target_unit).sum().clamp(min=1e-8)
            return (masked_target * target_unit).sum() / full_proj
        raise ValueError(f"Unsupported ERF objective mode: {objective_mode!r}")

    def _feature_activation_recovery_objective(
        self,
        patch_out: torch.Tensor,
        *,
        sae: torch.nn.Module,
        token_idx: int,
        feature_id: int,
        full_activation: torch.Tensor,
    ) -> torch.Tensor:
        acts = sae(patch_out).get("feature_acts")
        masked_act = acts[0, int(token_idx), int(feature_id)]
        recovery = masked_act / full_activation.clamp(min=1e-8)
        return torch.minimum(recovery, torch.ones((), device=recovery.device, dtype=recovery.dtype))

    def _feature_activation_recovery_objective_full(
        self,
        block_out: torch.Tensor,
        *,
        sae: torch.nn.Module,
        token_x: int,
        feature_id: int,
        full_activation: torch.Tensor,
        baseline_activation: torch.Tensor,
    ) -> torch.Tensor:
        acts = sae(block_out).get("feature_acts")
        masked_act = acts[0, int(token_x), int(feature_id)]
        denom = (full_activation - baseline_activation).clamp(min=1e-8)
        recovery = (masked_act - baseline_activation) / denom
        return torch.clamp(
            recovery,
            min=torch.zeros((), device=recovery.device, dtype=recovery.dtype),
            max=torch.ones((), device=recovery.device, dtype=recovery.dtype),
        )

    def inverse_grad_irrelevance(
        self,
        image_path: str,
        block_idx: int,
        token_idx: int,
        *,
        objective_mode: str | None = None,
    ) -> torch.Tensor:
        objective_mode = str(objective_mode or self.config.erf_objective_mode)
        artifacts = self.forward_block(image_path, block_idx)
        capture = artifacts.capture
        injected_patches = capture.patch_tokens.detach().clone().requires_grad_(True)
        prefix = capture.prefix_tokens
        target_vec = artifacts.patch_out[0, int(token_idx), :].detach()
        injected = torch.cat([prefix, injected_patches], dim=1)
        buf: list[torch.Tensor] = []
        pre_hook = self.model.blocks[0].register_forward_pre_hook(lambda _m, _args: (injected,))
        blk_hook = self.model.blocks[int(block_idx)].register_forward_hook(
            lambda _m, _i, output: buf.append(output if torch.is_tensor(output) else output[0])
        )
        try:
            self.model(artifacts.x)
        finally:
            pre_hook.remove()
            blk_hook.remove()
        prefix_count = self.adapter.prefix_count()
        masked_target = buf[0][0, prefix_count + int(token_idx), :]
        objective = self._token_recovery_objective(
            masked_target,
            target_vec,
            objective_mode=objective_mode,
        )
        grad = torch.autograd.grad(objective, injected_patches, retain_graph=False, create_graph=False)[0]
        grad_norm = grad[0].norm(dim=-1)
        inv = 1.0 / (grad_norm + 1e-8)
        return inv / inv.max().clamp(min=1e-8)

    def inverse_grad_feature_irrelevance(
        self,
        image_path: str,
        block_idx: int,
        token_idx: int,
        feature_id: int,
    ) -> torch.Tensor:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            full_acts = sae(artifacts.patch_out).get("feature_acts")
            full_activation = full_acts[0, int(token_idx), int(feature_id)].detach()
        injected_patches = artifacts.capture.patch_tokens.detach().clone().requires_grad_(True)
        prefix = artifacts.capture.prefix_tokens
        injected = torch.cat([prefix, injected_patches], dim=1)
        buf: list[torch.Tensor] = []
        pre_hook = self.model.blocks[0].register_forward_pre_hook(lambda _m, _args: (injected,))
        blk_hook = self.model.blocks[int(block_idx)].register_forward_hook(
            lambda _m, _i, output: buf.append(output if torch.is_tensor(output) else output[0])
        )
        try:
            self.model(artifacts.x)
        finally:
            pre_hook.remove()
            blk_hook.remove()
        prefix_count = self.adapter.prefix_count()
        patch_out = buf[0][:, prefix_count:, :]
        objective = self._feature_activation_recovery_objective(
            patch_out,
            sae=sae,
            token_idx=int(token_idx),
            feature_id=int(feature_id),
            full_activation=full_activation,
        )
        grad = torch.autograd.grad(objective, injected_patches, retain_graph=False, create_graph=False)[0]
        grad_norm = grad[0].norm(dim=-1)
        inv = 1.0 / (grad_norm + 1e-8)
        return inv / inv.max().clamp(min=1e-8)

    def inverse_grad_feature_irrelevance_special_token(
        self,
        image_path: str,
        block_idx: int,
        token_x: int,
        feature_id: int,
    ) -> torch.Tensor:
        sae = self.load_sae(block_idx)
        artifacts = self.forward_block(image_path, block_idx)
        with torch.no_grad():
            full_acts = sae(artifacts.capture.target_block_output).get("feature_acts")
            full_activation = full_acts[0, int(token_x), int(feature_id)].detach()
        do_forward_masked, get_block_out = self.adapter.make_masked_forward(
            artifacts.x,
            artifacts.capture,
            block_idx=int(block_idx),
        )
        with torch.no_grad():
            do_forward_masked(torch.zeros(self.config.n_patches, device=artifacts.capture.patch_tokens.device, dtype=artifacts.capture.patch_tokens.dtype))
            baseline_block_out = get_block_out()
            baseline_acts = sae(baseline_block_out).get("feature_acts")
            baseline_activation = baseline_acts[0, int(token_x), int(feature_id)].detach()
        injected_patches = artifacts.capture.patch_tokens.detach().clone().requires_grad_(True)
        prefix = artifacts.capture.prefix_tokens
        injected = torch.cat([prefix, injected_patches], dim=1)
        buf: list[torch.Tensor] = []
        pre_hook = self.model.blocks[0].register_forward_pre_hook(lambda _m, _args: (injected,))
        blk_hook = self.model.blocks[int(block_idx)].register_forward_hook(
            lambda _m, _i, output: buf.append(output if torch.is_tensor(output) else output[0])
        )
        try:
            self.model(artifacts.x)
        finally:
            pre_hook.remove()
            blk_hook.remove()
        objective = self._feature_activation_recovery_objective_full(
            buf[0],
            sae=sae,
            token_x=int(token_x),
            feature_id=int(feature_id),
            full_activation=full_activation,
            baseline_activation=baseline_activation,
        )
        grad = torch.autograd.grad(objective, injected_patches, retain_graph=False, create_graph=False)[0]
        grad_norm = grad[0].norm(dim=-1)
        inv = 1.0 / (grad_norm + 1e-8)
        return inv / inv.max().clamp(min=1e-8)

    def cautious_token_erf(
        self,
        image_path: str,
        block_idx: int,
        token_idx: int,
        *,
        objective_mode: str | None = None,
    ) -> dict[str, Any]:
        objective_mode = str(objective_mode or self.config.erf_objective_mode)
        metric_name = self._objective_metric_name(objective_mode)
        artifacts = self.forward_block(image_path, block_idx)
        capture = artifacts.capture
        do_forward_masked, get_block_out = self.adapter.make_masked_forward(
            artifacts.x,
            capture,
            block_idx=int(block_idx),
        )
        target_vec = artifacts.patch_out[0, int(token_idx), :].detach()
        dtype = capture.patch_tokens.dtype
        dev = capture.patch_tokens.device
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(self.config.cautious_seed))

        def objective_for_mask(mask: torch.Tensor) -> torch.Tensor:
            do_forward_masked(mask)
            block_out = get_block_out()
            prefix = self.adapter.prefix_count()
            masked_target = block_out[0, prefix + int(token_idx), :]
            return self._token_recovery_objective(
                masked_target,
                target_vec,
                objective_mode=objective_mode,
            )

        with torch.no_grad():
            full_objective = float(objective_for_mask(torch.ones(self.config.n_patches, device=dev, dtype=dtype)).item())
        irr = self.inverse_grad_irrelevance(
            image_path,
            block_idx,
            token_idx,
            objective_mode=objective_mode,
        ).to(device=dev, dtype=dtype)
        logit_init = math.log(self.config.cautious_init_prob / (1.0 - self.config.cautious_init_prob))
        log_alphas = torch.full((self.config.n_patches,), logit_init, device=dev, dtype=dtype)
        m = torch.zeros_like(log_alphas)
        v = torch.zeros_like(log_alphas)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def tv_loss(values: torch.Tensor) -> torch.Tensor:
            grid = values.view(self.config.grid_size, self.config.grid_size)
            return (grid[:, :-1] - grid[:, 1:]).abs().sum() + (grid[:-1, :] - grid[1:, :]).abs().sum()

        for step in range(int(self.config.cautious_steps)):
            frac = step / max(int(self.config.cautious_steps) - 1, 1)
            cur_lr = float(self.config.cautious_lr_end) + 0.5 * (
                float(self.config.cautious_lr) - float(self.config.cautious_lr_end)
            ) * (1.0 + math.cos(math.pi * frac))
            budget = float(torch.rand(1, generator=generator, device=dev).item() * self.config.n_patches)
            la_req = log_alphas.clone().requires_grad_(True)
            probs = torch.sigmoid(la_req)
            mass = probs / (probs.sum() + 1e-8)
            soft_mask = (mass * budget).clamp(max=1.0)
            objective = objective_for_mask(soft_mask)
            loss = (1.0 - objective) + float(self.config.cautious_irr_weight) * (probs * irr).sum()
            loss = loss + float(self.config.cautious_tv_weight) * tv_loss(probs)
            loss.backward()
            grad = la_req.grad.detach()
            timestep = step + 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            m_hat = m / (1.0 - beta1**timestep)
            v_hat = v / (1.0 - beta2**timestep)
            adam_dir = m_hat / (v_hat.sqrt() + eps)
            mask = (adam_dir * grad > 0).to(dtype=dtype)
            active = mask.sum().clamp(min=1.0)
            mask = mask * (self.config.n_patches / active)
            log_alphas = log_alphas - cur_lr * adam_dir * mask

        probs = torch.sigmoid(log_alphas).detach().cpu().numpy().astype(np.float32)
        full_order = np.argsort(-probs, kind="mergesort")
        probs_max = float(probs.max()) if probs.size > 0 else 0.0
        normalized_attribution = (probs / probs_max).astype(np.float32) if probs_max > 0 else np.zeros_like(probs)
        min_norm_attr = float(self.config.erf_support_min_normalized_attribution)
        valid_mask = normalized_attribution >= min_norm_attr
        valid_order = full_order[valid_mask[full_order]]
        if valid_order.size == 0:
            valid_order = full_order[:1]
        max_valid = int(valid_order.size)
        recovery_cache: dict[int, float] = {}

        def evaluate_prefix(prefix_size: int) -> float:
            prefix_size = int(prefix_size)
            cached = recovery_cache.get(prefix_size)
            if cached is not None:
                return cached
            hard_mask = torch.zeros(self.config.n_patches, device=dev, dtype=dtype)
            hard_mask[torch.as_tensor(valid_order[:prefix_size], device=dev)] = 1.0
            recovery = float(objective_for_mask(hard_mask).detach().cpu())
            recovery_cache[prefix_size] = recovery
            return recovery

        search_summary = _search_minimal_support_prefix(
            n_patches=max_valid,
            threshold=float(self.config.erf_recovery_threshold),
            metric_name=metric_name,
            recovery_cache=recovery_cache,
            evaluate_prefix=evaluate_prefix,
        )
        support, support_size, support_recovery = _resolve_support_indices(
            valid_order=valid_order,
            proposed_support_size=int(search_summary["support_size"]),
            threshold_reached=bool(search_summary["recovery_threshold_reached"]),
            evaluate_prefix=evaluate_prefix,
        )
        return {
            "objective_mode": objective_mode,
            "objective_metric_name": metric_name,
            "prob_scores": probs.tolist(),
            "normalized_attribution": normalized_attribution.tolist(),
            "ranking": full_order.tolist(),
            "valid_ranking": valid_order.tolist(),
            "support_indices": support,
            "support_size": int(support_size),
            "support_threshold": float(self.config.erf_recovery_threshold),
            "support_min_normalized_attribution": min_norm_attr,
            "support_recovery": float(support_recovery),
            "full_objective": float(full_objective),
            "recovery_trace": list(search_summary["recovery_trace"]),
            "recovery_search_mode": str(search_summary["recovery_search_mode"]),
            "recovery_coarse_budgets": list(search_summary["recovery_coarse_budgets"]),
            "recovery_dense_limit": int(search_summary["recovery_dense_limit"]),
            "recovery_bracket_low": int(search_summary["recovery_bracket_low"]),
            "recovery_bracket_high": int(search_summary["recovery_bracket_high"]),
            "recovery_eval_count": int(search_summary["recovery_eval_count"]),
            "recovery_threshold_reached": bool(search_summary["recovery_threshold_reached"]),
            "recovery_max": float(search_summary["recovery_max"]),
        }

    def cautious_feature_erf(
        self,
        image_path: str,
        block_idx: int,
        token_idx: int,
        feature_id: int,
    ) -> dict[str, Any]:
        artifacts = self.forward_block(image_path, block_idx)
        capture = artifacts.capture
        sae = self.load_sae(block_idx)
        with torch.no_grad():
            full_acts = sae(artifacts.patch_out).get("feature_acts")
            full_activation = full_acts[0, int(token_idx), int(feature_id)].detach()
        do_forward_masked, get_block_out = self.adapter.make_masked_forward(
            artifacts.x,
            capture,
            block_idx=int(block_idx),
        )
        dtype = capture.patch_tokens.dtype
        dev = capture.patch_tokens.device
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(self.config.cautious_seed))
        metric_name = "feature_activation_recovery"

        def objective_for_mask(mask: torch.Tensor) -> torch.Tensor:
            do_forward_masked(mask)
            block_out = get_block_out()
            prefix = self.adapter.prefix_count()
            patch_out = block_out[:, prefix:, :]
            return self._feature_activation_recovery_objective(
                patch_out,
                sae=sae,
                token_idx=int(token_idx),
                feature_id=int(feature_id),
                full_activation=full_activation,
            )

        with torch.no_grad():
            full_objective = float(objective_for_mask(torch.ones(self.config.n_patches, device=dev, dtype=dtype)).item())
        irr = self.inverse_grad_feature_irrelevance(
            image_path,
            block_idx,
            token_idx,
            feature_id,
        ).to(device=dev, dtype=dtype)
        logit_init = math.log(self.config.cautious_init_prob / (1.0 - self.config.cautious_init_prob))
        log_alphas = torch.full((self.config.n_patches,), logit_init, device=dev, dtype=dtype)
        m = torch.zeros_like(log_alphas)
        v = torch.zeros_like(log_alphas)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def tv_loss(values: torch.Tensor) -> torch.Tensor:
            grid = values.view(self.config.grid_size, self.config.grid_size)
            return (grid[:, :-1] - grid[:, 1:]).abs().sum() + (grid[:-1, :] - grid[1:, :]).abs().sum()

        for step in range(int(self.config.cautious_steps)):
            frac = step / max(int(self.config.cautious_steps) - 1, 1)
            cur_lr = float(self.config.cautious_lr_end) + 0.5 * (
                float(self.config.cautious_lr) - float(self.config.cautious_lr_end)
            ) * (1.0 + math.cos(math.pi * frac))
            budget = float(torch.rand(1, generator=generator, device=dev).item() * self.config.n_patches)
            la_req = log_alphas.clone().requires_grad_(True)
            probs = torch.sigmoid(la_req)
            mass = probs / (probs.sum() + 1e-8)
            soft_mask = (mass * budget).clamp(max=1.0)
            objective = objective_for_mask(soft_mask)
            loss = (1.0 - objective) + float(self.config.cautious_irr_weight) * (probs * irr).sum()
            loss = loss + float(self.config.cautious_tv_weight) * tv_loss(probs)
            loss.backward()
            grad = la_req.grad.detach()
            timestep = step + 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            m_hat = m / (1.0 - beta1**timestep)
            v_hat = v / (1.0 - beta2**timestep)
            adam_dir = m_hat / (v_hat.sqrt() + eps)
            mask = (adam_dir * grad > 0).to(dtype=dtype)
            active = mask.sum().clamp(min=1.0)
            mask = mask * (self.config.n_patches / active)
            log_alphas = log_alphas - cur_lr * adam_dir * mask

        probs = torch.sigmoid(log_alphas).detach().cpu().numpy().astype(np.float32)
        full_order = np.argsort(-probs, kind="mergesort")
        probs_max = float(probs.max()) if probs.size > 0 else 0.0
        normalized_attribution = (probs / probs_max).astype(np.float32) if probs_max > 0 else np.zeros_like(probs)
        min_norm_attr = float(self.config.erf_support_min_normalized_attribution)
        valid_mask = normalized_attribution >= min_norm_attr
        valid_order = full_order[valid_mask[full_order]]
        if valid_order.size == 0:
            valid_order = full_order[:1]
        max_valid = int(valid_order.size)
        recovery_cache: dict[int, float] = {}

        def evaluate_prefix(prefix_size: int) -> float:
            prefix_size = int(prefix_size)
            cached = recovery_cache.get(prefix_size)
            if cached is not None:
                return cached
            hard_mask = torch.zeros(self.config.n_patches, device=dev, dtype=dtype)
            hard_mask[torch.as_tensor(valid_order[:prefix_size], device=dev)] = 1.0
            recovery = float(objective_for_mask(hard_mask).detach().cpu())
            recovery_cache[prefix_size] = recovery
            return recovery

        search_summary = _search_minimal_support_prefix(
            n_patches=max_valid,
            threshold=float(self.config.erf_recovery_threshold),
            metric_name=metric_name,
            recovery_cache=recovery_cache,
            evaluate_prefix=evaluate_prefix,
        )
        support, support_size, support_recovery = _resolve_support_indices(
            valid_order=valid_order,
            proposed_support_size=int(search_summary["support_size"]),
            threshold_reached=bool(search_summary["recovery_threshold_reached"]),
            evaluate_prefix=evaluate_prefix,
        )
        return {
            "objective_mode": "feature_activation",
            "objective_metric_name": metric_name,
            "feature_id": int(feature_id),
            "prob_scores": probs.tolist(),
            "normalized_attribution": normalized_attribution.tolist(),
            "ranking": full_order.tolist(),
            "valid_ranking": valid_order.tolist(),
            "support_indices": support,
            "support_size": int(support_size),
            "support_threshold": float(self.config.erf_recovery_threshold),
            "support_min_normalized_attribution": min_norm_attr,
            "support_recovery": float(support_recovery),
            "full_objective": float(full_objective),
            "full_feature_activation": float(full_activation.detach().cpu()),
            "recovery_trace": list(search_summary["recovery_trace"]),
            "recovery_search_mode": str(search_summary["recovery_search_mode"]),
            "recovery_coarse_budgets": list(search_summary["recovery_coarse_budgets"]),
            "recovery_dense_limit": int(search_summary["recovery_dense_limit"]),
            "recovery_bracket_low": int(search_summary["recovery_bracket_low"]),
            "recovery_bracket_high": int(search_summary["recovery_bracket_high"]),
            "recovery_eval_count": int(search_summary["recovery_eval_count"]),
            "recovery_threshold_reached": bool(search_summary["recovery_threshold_reached"]),
            "recovery_max": float(search_summary["recovery_max"]),
        }

    def cautious_feature_erf_special_token(
        self,
        image_path: str,
        block_idx: int,
        token_x: int,
        feature_id: int,
    ) -> dict[str, Any]:
        artifacts = self.forward_block(image_path, block_idx)
        capture = artifacts.capture
        sae = self.load_sae(block_idx)
        with torch.no_grad():
            full_acts = sae(artifacts.capture.target_block_output).get("feature_acts")
            full_activation = full_acts[0, int(token_x), int(feature_id)].detach()
        do_forward_masked, get_block_out = self.adapter.make_masked_forward(
            artifacts.x,
            capture,
            block_idx=int(block_idx),
        )
        with torch.no_grad():
            do_forward_masked(torch.zeros(self.config.n_patches, device=capture.patch_tokens.device, dtype=capture.patch_tokens.dtype))
            baseline_block_out = get_block_out()
            baseline_acts = sae(baseline_block_out).get("feature_acts")
            baseline_activation = baseline_acts[0, int(token_x), int(feature_id)].detach()
        effective_delta = float((full_activation - baseline_activation).detach().cpu())
        dtype = capture.patch_tokens.dtype
        dev = capture.patch_tokens.device
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(self.config.cautious_seed))
        metric_name = "feature_activation_delta_recovery"

        def objective_for_mask(mask: torch.Tensor) -> torch.Tensor:
            do_forward_masked(mask)
            block_out = get_block_out()
            return self._feature_activation_recovery_objective_full(
                block_out,
                sae=sae,
                token_x=int(token_x),
                feature_id=int(feature_id),
                full_activation=full_activation,
                baseline_activation=baseline_activation,
            )

        with torch.no_grad():
            full_objective = float(objective_for_mask(torch.ones(self.config.n_patches, device=dev, dtype=dtype)).item())
            baseline_objective = float(objective_for_mask(torch.zeros(self.config.n_patches, device=dev, dtype=dtype)).item())
        if effective_delta <= 1e-8:
            return {
                "objective_mode": "feature_activation_special_token",
                "objective_metric_name": metric_name,
                "feature_id": int(feature_id),
                "objective_token_x": int(token_x),
                "prob_scores": [],
                "normalized_attribution": [],
                "ranking": [],
                "valid_ranking": [],
                "support_indices": [],
                "support_size": 0,
                "support_threshold": float(self.config.erf_recovery_threshold),
                "support_min_normalized_attribution": float(self.config.erf_support_min_normalized_attribution),
                "support_recovery": 0.0,
                "full_objective": float(full_objective),
                "baseline_objective": float(baseline_objective),
                "full_feature_activation": float(full_activation.detach().cpu()),
                "baseline_feature_activation": float(baseline_activation.detach().cpu()),
                "effective_feature_activation_delta": float(effective_delta),
                "recovery_trace": [],
                "recovery_search_mode": "baseline_dominated_empty_support",
                "recovery_coarse_budgets": [],
                "recovery_dense_limit": 0,
                "recovery_bracket_low": 0,
                "recovery_bracket_high": 0,
                "recovery_eval_count": 0,
                "recovery_threshold_reached": False,
                "recovery_max": 0.0,
                "erf_absent_reason": "baseline_activation_ge_full_activation",
            }
        irr = self.inverse_grad_feature_irrelevance_special_token(
            image_path,
            block_idx,
            token_x,
            feature_id,
        ).to(device=dev, dtype=dtype)
        logit_init = math.log(self.config.cautious_init_prob / (1.0 - self.config.cautious_init_prob))
        log_alphas = torch.full((self.config.n_patches,), logit_init, device=dev, dtype=dtype)
        m = torch.zeros_like(log_alphas)
        v = torch.zeros_like(log_alphas)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def tv_loss(values: torch.Tensor) -> torch.Tensor:
            grid = values.view(self.config.grid_size, self.config.grid_size)
            return (grid[:, :-1] - grid[:, 1:]).abs().sum() + (grid[:-1, :] - grid[1:, :]).abs().sum()

        for step in range(int(self.config.cautious_steps)):
            frac = step / max(int(self.config.cautious_steps) - 1, 1)
            cur_lr = float(self.config.cautious_lr_end) + 0.5 * (
                float(self.config.cautious_lr) - float(self.config.cautious_lr_end)
            ) * (1.0 + math.cos(math.pi * frac))
            budget = float(torch.rand(1, generator=generator, device=dev).item() * self.config.n_patches)
            la_req = log_alphas.clone().requires_grad_(True)
            probs = torch.sigmoid(la_req)
            mass = probs / (probs.sum() + 1e-8)
            soft_mask = (mass * budget).clamp(max=1.0)
            objective = objective_for_mask(soft_mask)
            loss = (1.0 - objective) + float(self.config.cautious_irr_weight) * (probs * irr).sum()
            loss = loss + float(self.config.cautious_tv_weight) * tv_loss(probs)
            loss.backward()
            grad = la_req.grad.detach()
            timestep = step + 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad * grad
            m_hat = m / (1.0 - beta1**timestep)
            v_hat = v / (1.0 - beta2**timestep)
            adam_dir = m_hat / (v_hat.sqrt() + eps)
            mask = (adam_dir * grad > 0).to(dtype=dtype)
            active = mask.sum().clamp(min=1.0)
            mask = mask * (self.config.n_patches / active)
            log_alphas = log_alphas - cur_lr * adam_dir * mask

        probs = torch.sigmoid(log_alphas).detach().cpu().numpy().astype(np.float32)
        full_order = np.argsort(-probs, kind="mergesort")
        probs_max = float(probs.max()) if probs.size > 0 else 0.0
        normalized_attribution = (probs / probs_max).astype(np.float32) if probs_max > 0 else np.zeros_like(probs)
        min_norm_attr = float(self.config.erf_support_min_normalized_attribution)
        valid_mask = normalized_attribution >= min_norm_attr
        valid_order = full_order[valid_mask[full_order]]
        if valid_order.size == 0:
            valid_order = full_order[:1]
        max_valid = int(valid_order.size)
        recovery_cache: dict[int, float] = {}

        def evaluate_prefix(prefix_size: int) -> float:
            prefix_size = int(prefix_size)
            cached = recovery_cache.get(prefix_size)
            if cached is not None:
                return cached
            hard_mask = torch.zeros(self.config.n_patches, device=dev, dtype=dtype)
            hard_mask[torch.as_tensor(valid_order[:prefix_size], device=dev)] = 1.0
            recovery = float(objective_for_mask(hard_mask).detach().cpu())
            recovery_cache[prefix_size] = recovery
            return recovery

        search_summary = _search_minimal_support_prefix(
            n_patches=max_valid,
            threshold=float(self.config.erf_recovery_threshold),
            metric_name=metric_name,
            recovery_cache=recovery_cache,
            evaluate_prefix=evaluate_prefix,
        )
        support, support_size, support_recovery = _resolve_support_indices(
            valid_order=valid_order,
            proposed_support_size=int(search_summary["support_size"]),
            threshold_reached=bool(search_summary["recovery_threshold_reached"]),
            evaluate_prefix=evaluate_prefix,
        )
        return {
            "objective_mode": "feature_activation_special_token",
            "objective_metric_name": metric_name,
            "feature_id": int(feature_id),
            "objective_token_x": int(token_x),
            "prob_scores": probs.tolist(),
            "normalized_attribution": normalized_attribution.tolist(),
            "ranking": full_order.tolist(),
            "valid_ranking": valid_order.tolist(),
            "support_indices": support,
            "support_size": int(support_size),
            "support_threshold": float(self.config.erf_recovery_threshold),
            "support_min_normalized_attribution": min_norm_attr,
            "support_recovery": float(support_recovery),
            "full_objective": float(full_objective),
            "baseline_objective": float(baseline_objective),
            "full_feature_activation": float(full_activation.detach().cpu()),
            "baseline_feature_activation": float(baseline_activation.detach().cpu()),
            "effective_feature_activation_delta": float(effective_delta),
            "recovery_trace": list(search_summary["recovery_trace"]),
            "recovery_search_mode": str(search_summary["recovery_search_mode"]),
            "recovery_coarse_budgets": list(search_summary["recovery_coarse_budgets"]),
            "recovery_dense_limit": int(search_summary["recovery_dense_limit"]),
            "recovery_bracket_low": int(search_summary["recovery_bracket_low"]),
            "recovery_bracket_high": int(search_summary["recovery_bracket_high"]),
            "recovery_eval_count": int(search_summary["recovery_eval_count"]),
            "recovery_threshold_reached": bool(search_summary["recovery_threshold_reached"]),
            "recovery_max": float(search_summary["recovery_max"]),
        }

    def load_decile_frame(self, block_idx: int) -> pd.DataFrame:
        decile_root = self.config.deciles_root / f"layer_part=model.blocks.{int(block_idx)}"
        try:
            dataset = ds.dataset(decile_root, format="parquet")
            table = dataset.to_table(columns=["unit", "score", "sample_id", "x"])
            return table.to_pandas()
        except OSError:
            try:
                import duckdb
            except Exception as exc:  # pragma: no cover - fallback only
                raise RuntimeError(
                    "Failed to read decile parquet with pyarrow and duckdb is unavailable. "
                    "Install duckdb or provide a readable decile export."
                ) from exc
            conn = duckdb.connect()
            try:
                query = (
                    "select unit, score, sample_id, x "
                    f"from read_parquet('{decile_root.as_posix()}/**/*.parquet')"
                )
                return conn.execute(query).df()
            finally:
                conn.close()

    def close(self) -> None:
        for sae in self._sae_cache.values():
            sae.cpu()
        self._sae_cache.clear()
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def token_record_from_row(
    block_idx: int,
    feature_id: int,
    row: Any,
    image_path: str,
    validation: dict[str, Any],
    *,
    token_idx: int | None = None,
) -> dict[str, Any]:
    tok = int(token_idx) if token_idx is not None else int(row.x) - 1
    sid = int(row.sample_id)
    return {
        "block_idx": int(block_idx),
        "feature_id": int(feature_id),
        "sample_id": sid,
        "image_path": image_path,
        "target_patch_idx": tok,
        "tok_max": tok,
        "token_uid": token_uid(block_idx, sid, tok),
        "ledger_score": float(row.score),
        "validation": validation,
    }
