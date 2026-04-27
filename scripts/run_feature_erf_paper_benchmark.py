#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

REPO = Path(os.environ.get("SPECLENS_REPO", str(Path(__file__).resolve().parents[1]))).expanduser()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.attribution.backends.gradients import build_ig_backend, build_ixg_backend
from src.packs.clip.models.attnlrp import (
    AttnLRPAttention,
    AttnLRPGELU,
    AttnLRPLayerNorm,
    AttnLRPQuickGELU,
    enable_sae_attnlrp,
)
from src.packs.clip.models.libragrad import (
    FullGradGELU,
    FullGradLayerNorm,
    FullGradNormalize,
    FullGradQuickGELU,
    LibragradAttention,
    enable_sae_libragrad,
)

try:
    from src.core.attribution.vit_attention_patchers import (
        apply_attention_capture,
        apply_generic_attnlrp,
        apply_generic_libragrad,
        clear_attention_capture_state,
        compute_inflow_rollout,
    )
except Exception:
    import torch.nn as nn
    import torch.nn.functional as F

    def _is_quick_gelu(module: nn.Module) -> bool:
        name = module.__class__.__name__.lower()
        return name in {"quickgelu", "quick_gelu"}

    def _is_qkv_attention_module(module: nn.Module) -> bool:
        required = ("qkv", "attn_drop", "proj", "proj_drop", "num_heads")
        return all(hasattr(module, attr) for attr in required)

    def _ensure_attention_compat(module: nn.Module) -> nn.Module:
        if not hasattr(module, "head_dim"):
            qkv = getattr(module, "qkv", None)
            num_heads = getattr(module, "num_heads", None)
            if qkv is not None and num_heads:
                try:
                    setattr(module, "head_dim", int(qkv.weight.shape[0]) // (3 * int(num_heads)))
                except Exception:
                    pass
        if not hasattr(module, "scale") and hasattr(module, "head_dim"):
            try:
                setattr(module, "scale", float(getattr(module, "head_dim")) ** -0.5)
            except Exception:
                pass
        return module

    def _replace_module(owner: nn.Module, name: str, new_mod: nn.Module, record):
        old = getattr(owner, name, None)
        if old is None or old is new_mod:
            return
        if isinstance(old, type(new_mod)):
            return
        setattr(owner, name, new_mod)
        if isinstance(old, nn.Module):
            new_mod.train(old.training)
        record.append((owner, name, old))

    def apply_generic_attnlrp(model: nn.Module):
        replaced = []

        def _walk(mod: nn.Module):
            for child_name, child in mod.named_children():
                if isinstance(child, nn.LayerNorm):
                    _replace_module(mod, child_name, AttnLRPLayerNorm.from_layer(child), replaced)
                    continue
                if _is_qkv_attention_module(child) and not isinstance(child, AttnLRPAttention):
                    _replace_module(mod, child_name, AttnLRPAttention(_ensure_attention_compat(child)), replaced)
                    continue
                if isinstance(child, nn.GELU):
                    _replace_module(mod, child_name, AttnLRPGELU(child), replaced)
                    continue
                if _is_quick_gelu(child):
                    _replace_module(mod, child_name, AttnLRPQuickGELU(), replaced)
                    continue
                _walk(child)

        _walk(model)

        def _restore():
            for owner, name, old in reversed(replaced):
                try:
                    setattr(owner, name, old)
                except Exception:
                    pass

        return _restore

    def apply_generic_libragrad(model: nn.Module):
        replaced = []

        def _walk(mod: nn.Module):
            for child_name, child in mod.named_children():
                if isinstance(child, nn.LayerNorm):
                    _replace_module(mod, child_name, FullGradLayerNorm.from_layer(child), replaced)
                    continue
                if child.__class__.__name__.lower() == "normalize":
                    _replace_module(mod, child_name, FullGradNormalize.from_module(child), replaced)
                    continue
                if _is_qkv_attention_module(child) and not isinstance(child, LibragradAttention):
                    _replace_module(mod, child_name, LibragradAttention(_ensure_attention_compat(child)), replaced)
                    continue
                if isinstance(child, nn.GELU):
                    _replace_module(mod, child_name, FullGradGELU(), replaced)
                    continue
                if _is_quick_gelu(child):
                    _replace_module(mod, child_name, FullGradQuickGELU(), replaced)
                    continue
                _walk(child)

        _walk(model)

        def _restore():
            for owner, name, old in reversed(replaced):
                try:
                    setattr(owner, name, old)
                except Exception:
                    pass

        return _restore

    class _CaptureAttention(nn.Module):
        def __init__(self, attn: nn.Module) -> None:
            super().__init__()
            self.qkv = attn.qkv
            self.q_norm = getattr(attn, "q_norm", None)
            self.k_norm = getattr(attn, "k_norm", None)
            self.attn_drop = attn.attn_drop
            self.proj = attn.proj
            self.proj_drop = attn.proj_drop
            self.num_heads = int(attn.num_heads)
            self.head_dim = int(getattr(attn, "head_dim", self.qkv.weight.shape[0] // (3 * self.num_heads)))
            self.scale = float(getattr(attn, "scale", self.head_dim ** -0.5))
            self._saved_attn = None
            self._saved_grad = None

        def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, **_: object) -> torch.Tensor:  # type: ignore[override]
            bsz, toks, chans = x.shape
            qkv = self.qkv(x).reshape(bsz, toks, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            raw = (q * self.scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                raw = raw + attn_mask.to(dtype=raw.dtype, device=raw.device)
            attn = raw.softmax(dim=-1)
            self._saved_attn = attn
            if attn.requires_grad:
                attn.register_hook(lambda grad: setattr(self, "_saved_grad", grad))
            out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(bsz, toks, chans)
            return self.proj_drop(self.proj(out))

    def apply_attention_capture(model: nn.Module):
        replaced = []

        def _walk(mod: nn.Module):
            for child_name, child in mod.named_children():
                if _is_qkv_attention_module(child) and not isinstance(child, _CaptureAttention):
                    _replace_module(mod, child_name, _CaptureAttention(child), replaced)
                    continue
                _walk(child)

        _walk(model)

        def _restore():
            for owner, name, old in reversed(replaced):
                try:
                    setattr(owner, name, old)
                except Exception:
                    pass

        return _restore

    def clear_attention_capture_state(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, _CaptureAttention):
                module._saved_attn = None
                module._saved_grad = None

    def compute_inflow_rollout(attentions, biases_1, biases_2):
        attn_s = torch.stack(attentions)
        bias1_s = torch.stack(biases_1)
        bias2_s = torch.stack(biases_2)
        inp_w = bias1_s[:, 0]
        attn_w = bias1_s[:, 1]
        r1_w = bias2_s[:, 0]
        mlp_w = bias2_s[:, 1]
        mat_r1 = attn_s * attn_w.unsqueeze(-2) + torch.diag_embed(inp_w)
        ratio = F.normalize(mlp_w / (r1_w + 1e-8), p=1, dim=-1)
        mat_r2 = torch.diag_embed(ratio * mlp_w + r1_w)
        matrices = mat_r1 @ mat_r2
        matrices = matrices / matrices.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        joint = matrices[0]
        for idx in range(1, len(matrices)):
            joint = matrices[idx] @ joint
        return joint

DEFAULT_OUT_JSON = REPO / "outputs" / "paper_feature_erf" / "results_feature_erf_multimodel_main.json"
BASELINE_CACHE_ROOT = Path(os.environ.get("ERF_BASELINE_CACHE_ROOT", str(REPO / "outputs" / "erf_baselines"))).expanduser()
METHODS = (
    "plain_ixg",
    "ig_block0_abs",
    "libragrad_ig50",
    "naive_attn_rollout",
    "attnlrp_ixg",
    "inflow_erf",
    "cautious_cos",
)
MAIN_METRICS = ("stoch_ins_delta", "insertion_auc", "mas_ins_auc", "rep_idsds")


def _detect_legacy_mm_path() -> Path:
    env_path = os.environ.get("FEATURE_ERF_LEGACY_MM")
    candidates = [
        Path(env_path).expanduser() if env_path else None,
        REPO.parent / "codex_research_softins" / "eval_multimodel_erf.py",
        Path("/media/mipal/1TB/sangyu/codex_research_softins/eval_multimodel_erf.py"),
        Path("/home/mipal/eval_multimodel_erf.py"),
        Path("/media/mipal/1TB/sangyu/eval_multimodel_erf.py"),
        Path("/home/sangyu/Desktop/Master/codex_research_softins/eval_multimodel_erf.py"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate eval_multimodel_erf.py; set FEATURE_ERF_LEGACY_MM")


LEGACY_MM_PATH = _detect_legacy_mm_path()


def _load_legacy_multimodel_module():
    spec = importlib.util.spec_from_file_location("legacy_multimodel_erf", str(LEGACY_MM_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load legacy multimodel ERF module from {LEGACY_MM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mm = _load_legacy_multimodel_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade multimodel feature-ERF benchmark runner.")
    parser.add_argument("--pack", default="all", choices=list(mm.PACK_NAMES) + ["all"])
    parser.add_argument("--blocks", nargs="*", type=int, default=list(mm.BLOCK_IDXS))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--n-features", type=int, default=150)
    parser.add_argument("--n-images", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imagenet-val", type=Path, default=mm.IMAGENET_VAL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--save-manifest-json", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-restore", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store-scores", action="store_true")
    parser.add_argument(
        "--baseline-mode",
        choices=("default", "zero", "mean_image", "global_mean_h", "global_mean_h_plus_pos"),
        default="global_mean_h_plus_pos",
    )
    parser.add_argument(
        "--index-root",
        action="append",
        default=[],
        help="Override decile index root with PACK=PATH, e.g. clip=/path/to/clip_index",
    )
    parser.add_argument(
        "--sae-root",
        action="append",
        default=[],
        help="Override SAE checkpoint root with PACK=PATH, e.g. clip=/path/to/clip_sae",
    )
    parser.add_argument("--stoch-steps", type=int, default=20)
    parser.add_argument("--stoch-samples", type=int, default=5)
    parser.add_argument("--stoch-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--cautious-steps", type=int, default=32)
    parser.add_argument("--cautious-lr", type=float, default=0.45)
    parser.add_argument("--cautious-lr-end", type=float, default=0.01)
    parser.add_argument("--cautious-tv-weight", type=float, default=0.01)
    parser.add_argument("--cautious-irr-weight", type=float, default=0.05)
    parser.add_argument("--cautious-init-prob", type=float, default=0.5)
    parser.add_argument("--cautious-init-mode", choices=("uniform", "plain_ixg"), default="uniform")
    parser.add_argument("--cautious-reg-warmup-frac", type=float, default=0.0)
    parser.add_argument("--cautious-restarts", type=int, default=1)
    parser.add_argument("--cautious-budget-samples", type=int, default=1)
    parser.add_argument("--cautious-select-best", action="store_true")
    parser.add_argument(
        "--cautious-objective-mode",
        choices=("random_budget_softins", "fixed_budget_softins", "direct_recovery"),
        default="random_budget_softins",
    )
    parser.add_argument(
        "--cautious-optimizer-mode",
        choices=("cautious_adam_cosine", "adam_cosine"),
        default="cautious_adam_cosine",
    )
    parser.add_argument("--cautious-fixed-budget-frac", type=float, default=0.10)
    return parser.parse_args()


def _parse_root_overrides(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected PACK=PATH override, got: {item}")
        pack, raw = item.split("=", 1)
        pack = pack.strip()
        if pack not in mm.PACK_NAMES:
            raise ValueError(f"Unknown pack in root override: {pack}")
        out[pack] = Path(raw).expanduser()
    return out


def _run_ig_block0_abs(state: mm.SAEState) -> np.ndarray:
    ig_fn = build_ig_backend(
        anchor_tensors_getter=state.anchor_tensors_getter,
        objective_getter=state.objective_getter_single,
        steps=int(mm.IG_STEPS),
        set_alpha=state.set_alpha,
        do_forward=state.do_forward,
        anchor_baselines={"emb": state.baseline},
    )
    out = ig_fn()
    return out["emb"][0].abs().sum(dim=-1).detach().cpu().numpy().astype(np.float32)


def _collect_attention_payload(state: mm.SAEState, *, with_backward: bool) -> Dict[str, Any]:
    restore = apply_attention_capture(state.model)
    hooks: List[Any] = []
    attn_maps: List[torch.Tensor | None] = []
    block_inputs: List[torch.Tensor] = []
    attn_outputs: List[torch.Tensor] = []
    block_outputs: List[torch.Tensor] = []

    try:
        for blk in state.model.blocks[: state.block_idx + 1]:
            def _block_in_hook(_m, inp, _out, store=block_inputs):
                store.append(inp[0][0].detach().clone())
                return None

            def _attn_hook(mod, _inp, out, store=attn_outputs, amap=attn_maps):
                store.append((out if torch.is_tensor(out) else out[0])[0].detach().clone())
                saved = getattr(mod, "_saved_attn", None)
                amap.append(None if saved is None else saved.detach()[0].clone())
                return None

            def _block_out_hook(_m, _inp, out, store=block_outputs):
                store.append((out if torch.is_tensor(out) else out[0]).detach().clone())
                return None

            hooks.append(blk.register_forward_hook(_block_in_hook))
            hooks.append(blk.attn.register_forward_hook(_attn_hook))
            hooks.append(blk.register_forward_hook(_block_out_hook))

        state.model.zero_grad(set_to_none=True)
        state.sae.zero_grad(set_to_none=True)
        _ = state.model(state.x)

        global_grads: Dict[int, torch.Tensor] = {}
        if with_backward:
            final_block = block_outputs[-1]
            h = final_block[0, state.n_prefix :, :]
            result = state.sae(h)
            acts = result["feature_acts"] if isinstance(result, dict) else result
            scalar = acts[state.tok_max, state.feature_id]
            scalar.backward()
            for idx, blk in enumerate(state.model.blocks[: state.block_idx + 1]):
                grad = getattr(blk.attn, "_saved_grad", None)
                if grad is not None:
                    global_grads[idx] = grad.detach()[0].clone()

        return {
            "attn_maps": attn_maps,
            "block_inputs": block_inputs,
            "attn_outputs": attn_outputs,
            "block_outputs": block_outputs,
            "global_grads": global_grads,
        }
    finally:
        for hook in hooks:
            hook.remove()
        clear_attention_capture_state(state.model)
        restore()


def _run_naive_attn_rollout(state: mm.SAEState) -> np.ndarray:
    payload = _collect_attention_payload(state, with_backward=False)
    eye = torch.eye(int(state.n_prefix + state.h_b0_patches.shape[1]), device=state.x.device)
    rollout = eye.clone()
    used_layers = 0

    for attn_map in payload["attn_maps"]:
        if attn_map is None:
            continue
        attn_avg = attn_map.mean(dim=0)
        attn_res = 0.5 * attn_avg + 0.5 * eye
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = rollout @ attn_res
        used_layers += 1

    if used_layers == 0:
        raise RuntimeError("No attention maps captured for naive rollout")

    tok_full = int(state.tok_max + state.n_prefix)
    return rollout[tok_full, state.n_prefix :].detach().cpu().numpy().astype(np.float32)


def _run_attnlrp_ixg_generic(state: mm.SAEState) -> np.ndarray:
    restore = apply_generic_attnlrp(state.model)
    enable_sae_attnlrp(state.sae)
    try:
        state.set_alpha(1.0)
        state.do_forward()
        ixg_fn = build_ixg_backend(
            anchor_tensors_getter=state.anchor_tensors_getter,
            objective_getter=state.objective_getter_single,
            anchor_baselines={"emb": state.baseline},
        )
        out = ixg_fn()
    finally:
        restore()
    return out["emb"][0].norm(dim=-1).detach().cpu().numpy().astype(np.float32)


def _run_libragrad_ig50_generic(state: mm.SAEState) -> np.ndarray:
    restore = apply_generic_libragrad(state.model)
    enable_sae_libragrad(state.sae)
    try:
        ig_fn = build_ig_backend(
            anchor_tensors_getter=state.anchor_tensors_getter,
            objective_getter=state.objective_getter_single,
            steps=int(mm.IG_STEPS),
            set_alpha=state.set_alpha,
            do_forward=state.do_forward,
            anchor_baselines={"emb": state.baseline},
        )
        out = ig_fn()
    finally:
        restore()
    return out["emb"][0].norm(dim=-1).detach().cpu().numpy().astype(np.float32)


def _run_inflow_erf_generic(state: mm.SAEState) -> np.ndarray:
    payload = _collect_attention_payload(state, with_backward=True)
    num_heads = int(getattr(state.model.blocks[0].attn, "num_heads", 1))
    all_attentions: List[torch.Tensor] = []
    all_biases_1: List[torch.Tensor] = []
    all_biases_2: List[torch.Tensor] = []

    for idx in range(state.block_idx + 1):
        attn_map = payload["attn_maps"][idx]
        if attn_map is None:
            continue
        attn_agg = attn_map.mean(dim=0)
        grad = payload["global_grads"].get(idx)
        if grad is not None:
            importance = (attn_map.transpose(-2, -1) @ grad).abs().mean(dim=(-2, -1))
            importance = importance / (importance.sum() + 1e-8)
            attn_agg = torch.max(attn_map * importance.view(num_heads, 1, 1), dim=0)[0]
        all_attentions.append(attn_agg)

        blk_inp = payload["block_inputs"][idx]
        attn_out = payload["attn_outputs"][idx]
        blk_out = payload["block_outputs"][idx][0]
        resid_1 = blk_inp + attn_out
        mlp_out = blk_out - resid_1

        inp_n = torch.linalg.norm(blk_inp, ord=2, dim=1)
        attn_n = torch.linalg.norm(attn_out, ord=2, dim=1)
        all_biases_1.append(torch.nn.functional.normalize(torch.stack([inp_n, attn_n]), p=1, dim=0))

        r1_n = torch.linalg.norm(resid_1, ord=2, dim=1)
        mlp_n = torch.linalg.norm(mlp_out, ord=2, dim=1)
        all_biases_2.append(torch.nn.functional.normalize(torch.stack([r1_n, mlp_n]), p=1, dim=0))

    if not all_attentions:
        raise RuntimeError("No attention maps captured for InFlow-ERF")

    joint = compute_inflow_rollout(all_attentions, all_biases_1, all_biases_2)
    tok_full = int(state.tok_max + state.n_prefix)
    scores = joint[tok_full, state.n_prefix :].detach().cpu().numpy().astype(np.float32)
    mn, mx = float(scores.min()), float(scores.max())
    return ((scores - mn) / (mx - mn + 1e-8)).astype(np.float32)


def _feature_response(state: mm.SAEState, block_out: torch.Tensor) -> float:
    h = block_out[0, state.n_prefix :, :]
    result = state.sae(h)
    acts = result["feature_acts"] if isinstance(result, dict) else result
    return float(acts[state.tok_max, state.feature_id].item())


def _build_state_with_baseline(
    model,
    sae,
    triple: Dict[str, Any],
    transform: Callable,
    device: str,
    n_prefix: int,
    baseline_mode: str,
) -> mm.SAEState:
    from PIL import Image

    img = Image.open(str(triple["image_path"])).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    h_b0_full, _ = mm._capture_block0_and_target(model, x, int(triple["block_idx"]))
    n_patches = int(h_b0_full.shape[1] - n_prefix)

    if baseline_mode == "default":
        pos_embed = getattr(model, "pos_embed", None)
        if isinstance(pos_embed, torch.Tensor) and int(pos_embed.shape[1]) >= n_patches:
            baseline = pos_embed[:, -n_patches:, :].detach().to(device=device, dtype=h_b0_full.dtype)
        else:
            d_model = int(h_b0_full.shape[-1])
            baseline = torch.zeros(1, n_patches, d_model, device=device, dtype=h_b0_full.dtype)
    elif baseline_mode == "zero":
        d_model = int(h_b0_full.shape[-1])
        baseline = torch.zeros(1, n_patches, d_model, device=device, dtype=h_b0_full.dtype)
    elif baseline_mode == "mean_image":
        x_zero = torch.zeros_like(x)
        h_b0_zero, _ = mm._capture_block0_and_target(model, x_zero, int(triple["block_idx"]))
        baseline = h_b0_zero[:, n_prefix:, :].detach().to(device=device, dtype=h_b0_full.dtype)
    elif baseline_mode in {"global_mean_h", "global_mean_h_plus_pos"}:
        pack = str(triple["pack"])
        baseline_pt = BASELINE_CACHE_ROOT / "global_mean_h" / f"{pack}.pt"
        if not baseline_pt.exists():
            raise FileNotFoundError(
                f"global_mean_h baseline cache missing for {pack}: {baseline_pt}. "
                f"Run scripts/compute_global_mean_h_baseline.py first."
            )
        payload = torch.load(baseline_pt, map_location="cpu")
        mean_patch = payload["mean_patch"]
        if not isinstance(mean_patch, torch.Tensor):
            raise TypeError(f"Invalid mean_patch tensor in {baseline_pt}")
        mean_patch = mean_patch.to(device=device, dtype=h_b0_full.dtype)
        if mean_patch.ndim != 3 or mean_patch.shape[0] != 1 or mean_patch.shape[2] != h_b0_full.shape[-1]:
            raise ValueError(f"Unexpected mean_patch shape in {baseline_pt}: {tuple(mean_patch.shape)}")
        baseline = mean_patch.expand(1, n_patches, -1).contiguous()
        if baseline_mode == "global_mean_h_plus_pos":
            pos_embed = getattr(model, "pos_embed", None)
            if isinstance(pos_embed, torch.Tensor) and int(pos_embed.shape[1]) >= n_patches:
                pos = pos_embed[:, -n_patches:, :].detach().to(device=device, dtype=h_b0_full.dtype)
                pos_mean = pos.mean(dim=1, keepdim=True)
                baseline = baseline - pos_mean + pos
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    return mm.SAEState(
        model=model,
        sae=sae,
        x=x,
        h_b0_full=h_b0_full,
        baseline=baseline,
        block_idx=int(triple["block_idx"]),
        feature_id=int(triple["feature_id"]),
        tok_max=int(triple["tok_max"]),
        n_prefix=n_prefix,
    )


def _soft_weights(scores: np.ndarray, n_patches: int) -> np.ndarray:
    scores_pos = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0)
    total = float(scores_pos.sum())
    if total <= 1e-8:
        return np.ones(n_patches, dtype=np.float64) / n_patches
    return scores_pos / total


def _insertion_budget_schedule(block_idx: int, n_patches: int) -> Optional[List[int]]:
    if int(block_idx) not in {6, 10}:
        return None
    if n_patches <= 0:
        return [0]
    budgets = [0]
    for k in (1, 2, 4, 8, 16):
        if k < n_patches and k not in budgets:
            budgets.append(k)
    k = 24
    while k < n_patches:
        budgets.append(k)
        k += 8
    if budgets[-1] != n_patches:
        budgets.append(n_patches)
    return budgets


def _stochastic_insertion_auc(
    state: mm.SAEState,
    scores: np.ndarray,
    *,
    n_steps: int,
    n_samples: int,
    seed: int,
) -> float:
    n_patches = int(state.h_b0_patches.shape[1])
    budgets = np.linspace(0, n_patches, n_steps + 1)
    probs = _soft_weights(scores, n_patches)
    full_block = mm._run_injected(state.model, state.x, state.prefix_tokens, state.h_b0_patches, state.block_idx)
    full_obj = _feature_response(state, full_block)
    full_obj_safe = max(abs(full_obj), 1e-8)
    vals: List[float] = []
    generator = torch.Generator(device=state.x.device)
    generator.manual_seed(seed)

    for budget in budgets:
        step_probs = np.minimum(probs * budget, 1.0).astype(np.float32)
        step_vals: List[float] = []
        for _ in range(max(1, int(n_samples))):
            z = torch.bernoulli(torch.tensor(step_probs, device=state.x.device), generator=generator).view(
                1, n_patches, 1
            )
            h_mix = z * state.h_b0_patches + (1.0 - z) * state.baseline
            block_out = mm._run_injected(state.model, state.x, state.prefix_tokens, h_mix, state.block_idx)
            step_vals.append(_feature_response(state, block_out) / full_obj_safe)
        vals.append(float(np.mean(step_vals)))

    return float(np.trapz(vals, np.linspace(0.0, 1.0, n_steps + 1)))


def _rep_idsds(scores: np.ndarray, del_effects: np.ndarray) -> Dict[str, float]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    d = np.asarray(del_effects, dtype=np.float64).reshape(-1)
    if s.std() < 1e-10 or d.std() < 1e-10:
        return {"rep_idsds": 0.0, "rep_del_pearson": 0.0}
    try:
        idsds = float(spearmanr(s, d).statistic)
    except Exception:
        idsds = 0.0
    try:
        pear = float(pearsonr(s, d)[0])
    except Exception:
        pear = 0.0
    return {"rep_idsds": idsds, "rep_del_pearson": pear}


def _evaluate_scores(
    state: mm.SAEState,
    scores: np.ndarray,
    *,
    stoch_steps: int,
    stoch_samples: int,
    stoch_seeds: Sequence[int],
) -> Dict[str, Any]:
    n_patches = int(state.h_b0_patches.shape[1])
    insertion_budgets = _insertion_budget_schedule(int(state.block_idx), n_patches)
    full_block = mm._run_injected(state.model, state.x, state.prefix_tokens, state.h_b0_patches, state.block_idx)
    full_obj = _feature_response(state, full_block)
    full_obj_safe = max(abs(full_obj), 1e-8)

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    scores_norm = mm.normalize_scores(scores)
    ins = mm.insertion_auc_n(
        state.model,
        state.sae,
        state.x,
        state.h_b0_patches,
        state.prefix_tokens,
        state.baseline,
        state.feature_id,
        state.tok_max,
        state.block_idx,
        scores_norm,
        n_patches,
        state.n_prefix,
        mode="single",
        budgets=insertion_budgets,
    )
    mas = mm.mas_ins_auc_n(
        state.model,
        state.sae,
        state.x,
        state.h_b0_patches,
        state.prefix_tokens,
        state.baseline,
        state.feature_id,
        state.tok_max,
        state.block_idx,
        scores,
        n_patches,
        state.n_prefix,
        mode="single",
        budgets=insertion_budgets,
    )
    del_effects = mm.single_deletion_effects_n(
        state.model,
        state.sae,
        state.x,
        state.h_b0_patches,
        state.prefix_tokens,
        state.baseline,
        state.feature_id,
        state.tok_max,
        state.block_idx,
        n_patches,
        state.n_prefix,
        mode="single",
    )
    idsds = _rep_idsds(scores, del_effects)

    uniform = np.ones(n_patches, dtype=np.float32) / n_patches
    stoch_auc_seeds: List[float] = []
    stoch_unif_seeds: List[float] = []
    stoch_delta_seeds: List[float] = []
    for seed in stoch_seeds:
        auc = _stochastic_insertion_auc(state, scores, n_steps=stoch_steps, n_samples=stoch_samples, seed=int(seed))
        unif = _stochastic_insertion_auc(
            state,
            uniform,
            n_steps=stoch_steps,
            n_samples=stoch_samples,
            seed=int(seed) + 12345,
        )
        stoch_auc_seeds.append(float(auc))
        stoch_unif_seeds.append(float(unif))
        stoch_delta_seeds.append(float(auc - unif))

    return {
        "stoch_ins_auc": float(np.mean(stoch_auc_seeds)),
        "stoch_ins_unif": float(np.mean(stoch_unif_seeds)),
        "stoch_ins_delta": float(np.mean(stoch_delta_seeds)),
        "stoch_ins_auc_std": float(np.std(stoch_auc_seeds)),
        "stoch_ins_delta_std": float(np.std(stoch_delta_seeds)),
        "stoch_ins_auc_seeds": stoch_auc_seeds,
        "stoch_ins_unif_seeds": stoch_unif_seeds,
        "stoch_ins_delta_seeds": stoch_delta_seeds,
        "insertion_auc": float(ins),
        "mas_ins_auc": float(mas["mas_ins_auc"]),
        "mas_response_auc": float(mas["mas_response_auc"]),
        "mas_penalty_auc": float(mas["mas_penalty_auc"]),
        "full_objective": float(full_obj / full_obj_safe),
        "insertion_budgets": insertion_budgets or list(range(0, n_patches + 1)),
        **idsds,
    }


def _run_method(
    method: str,
    state: mm.SAEState,
    *,
    cautious_steps: int,
    cautious_kwargs: Dict[str, Any],
) -> np.ndarray:
    if method == "plain_ixg":
        return np.asarray(mm.run_plain_ixg(state), dtype=np.float32)
    if method == "ig_block0_abs":
        return _run_ig_block0_abs(state)
    if method == "libragrad_ig50":
        return _run_libragrad_ig50_generic(state)
    if method == "naive_attn_rollout":
        return _run_naive_attn_rollout(state)
    if method == "attnlrp_ixg":
        return _run_attnlrp_ixg_generic(state)
    if method == "inflow_erf":
        return _run_inflow_erf_generic(state)
    if method == "cautious_cos":
        n_patches = int(state.h_b0_patches.shape[1])
        grid = int(round(math.sqrt(n_patches)))
        if grid * grid != n_patches:
            raise ValueError(f"Expected square patch grid, got {n_patches}")
        init_scores = None
        if str(cautious_kwargs.get("init_mode", "uniform")) == "plain_ixg":
            init_scores = np.asarray(mm.run_plain_ixg(state), dtype=np.float32)
        return np.asarray(
            mm.run_cautious_cos(
                state,
                n_patches,
                grid,
                steps=int(cautious_steps),
                lr=float(cautious_kwargs.get("lr", 0.45)),
                lr_end=float(cautious_kwargs.get("lr_end", 0.01)),
                tv_weight=float(cautious_kwargs.get("tv_weight", 0.01)),
                irr_weight=float(cautious_kwargs.get("irr_weight", 0.05)),
                init_prob=float(cautious_kwargs.get("init_prob", 0.5)),
                init_scores=init_scores,
                reg_warmup_frac=float(cautious_kwargs.get("reg_warmup_frac", 0.0)),
                restarts=int(cautious_kwargs.get("restarts", 1)),
                budget_samples=int(cautious_kwargs.get("budget_samples", 1)),
                select_best=bool(cautious_kwargs.get("select_best", False)),
                objective_mode=str(cautious_kwargs.get("objective_mode", "random_budget_softins")),
                optimizer_mode=str(cautious_kwargs.get("optimizer_mode", "cautious_adam_cosine")),
                fixed_budget_frac=float(cautious_kwargs.get("fixed_budget_frac", 0.10)),
                seed=int(cautious_kwargs.get("seed", 0)),
            ),
            dtype=np.float32,
        )
    raise KeyError(f"Unknown method: {method}")


def evaluate_triple(
    model,
    sae,
    triple: Dict[str, Any],
    transform: Callable,
    device: str,
    *,
    methods: Sequence[str],
    stoch_steps: int,
    stoch_samples: int,
    stoch_seeds: Sequence[int],
    cautious_steps: int,
    cautious_kwargs: Dict[str, Any],
    strict_restore: bool,
    store_scores: bool,
    baseline_mode: str,
) -> Dict[str, Any]:
    n_prefix = mm.infer_prefix_count(model)
    state = _build_state_with_baseline(model, sae, triple, transform, device, n_prefix, baseline_mode)
    row = {k: v for k, v in triple.items()}

    _, block_out = mm._capture_block0_and_target(model, state.x, state.block_idx)
    with torch.no_grad():
        h = block_out[0, n_prefix:, :]
        result = sae(h)
        acts = result["feature_acts"] if isinstance(result, dict) else result
        act_at_target = float(acts[state.tok_max, state.feature_id].item())
    ledger_score = float(triple.get("ledger_score", 0.0))
    score_err = abs(act_at_target - ledger_score)
    restore_ok = score_err < max(0.01 * max(abs(ledger_score), 1e-8), 1e-3)
    row["restore_check"] = {
        "act_at_target": act_at_target,
        "ledger_score": ledger_score,
        "abs_err": score_err,
        "restore_ok": restore_ok,
    }

    if act_at_target < 1e-6:
        row["methods"] = {method: {"error": "feature_inactive"} for method in methods}
        return row
    if strict_restore and not restore_ok:
        row["methods"] = {method: {"error": "restore_check_failed"} for method in methods}
        return row

    methods_out: Dict[str, Any] = {}
    for method in methods:
        try:
            scores = _run_method(
                method,
                state,
                cautious_steps=cautious_steps,
                cautious_kwargs=cautious_kwargs,
            )
            metrics = _evaluate_scores(
                state,
                scores,
                stoch_steps=stoch_steps,
                stoch_samples=stoch_samples,
                stoch_seeds=stoch_seeds,
            )
            if store_scores:
                metrics["scores"] = np.asarray(scores, dtype=np.float32).tolist()
            methods_out[method] = metrics
        except Exception as exc:
            logging.getLogger("run_feature_erf_paper_benchmark").exception(
                "[%s] failed for %s", method, triple.get("image_path", "?")
            )
            methods_out[method] = {"error": str(exc)}

    row["methods"] = methods_out
    return row


def _make_run_meta(pack_specs: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    payload = {
        "packs": {
            pack: {
                "train_cfg_path": spec["train_cfg_path"],
                "index_cfg_path": spec["index_cfg_path"],
                "model_cfg": spec["model_cfg"],
                "image_size": spec["image_size"],
                "mean": spec["mean"],
                "std": spec["std"],
            }
            for pack, spec in sorted(pack_specs.items())
        },
        "blocks": list(args.blocks),
        "methods": list(args.methods),
        "n_features": int(args.n_features),
        "n_images": int(args.n_images),
        "seed": int(args.seed),
        "stoch_steps": int(args.stoch_steps),
        "stoch_samples": int(args.stoch_samples),
        "stoch_seeds": list(args.stoch_seeds),
        "cautious_steps": int(args.cautious_steps),
        "cautious_lr": float(args.cautious_lr),
        "cautious_lr_end": float(args.cautious_lr_end),
        "cautious_tv_weight": float(args.cautious_tv_weight),
        "cautious_irr_weight": float(args.cautious_irr_weight),
        "cautious_init_prob": float(args.cautious_init_prob),
        "cautious_init_mode": str(args.cautious_init_mode),
        "cautious_reg_warmup_frac": float(args.cautious_reg_warmup_frac),
        "cautious_restarts": int(args.cautious_restarts),
        "cautious_budget_samples": int(args.cautious_budget_samples),
        "cautious_select_best": bool(args.cautious_select_best),
        "cautious_objective_mode": str(args.cautious_objective_mode),
        "cautious_optimizer_mode": str(args.cautious_optimizer_mode),
        "cautious_fixed_budget_frac": float(args.cautious_fixed_budget_frac),
        "baseline_mode": str(args.baseline_mode),
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("triples") or payload.get("rows") or payload.get("per_triple") or []
    else:
        raise TypeError(f"Unsupported manifest payload in {path}")
    if not isinstance(rows, list):
        raise TypeError(f"Manifest rows must be a list in {path}")
    return [dict(row) for row in rows]


def aggregate_results(rows: Sequence[Dict[str, Any]], stoch_seeds: Sequence[int]) -> Dict[str, Any]:
    by_pack_block_method = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    by_pack_method = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall_method = defaultdict(lambda: defaultdict(list))
    by_pack_block_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    by_pack_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall_seed = defaultdict(lambda: defaultdict(list))

    for row in rows:
        restore = row.get("restore_check") or {}
        if isinstance(restore, dict) and restore.get("restore_ok") is False:
            continue
        pack = str(row.get("pack", "unknown"))
        blk = str(row.get("block_idx", -1))
        for method, metrics in row.get("methods", {}).items():
            if "error" in metrics:
                continue
            for metric in MAIN_METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    by_pack_block_method[pack][blk][method][metric].append(float(value))
                    by_pack_method[pack][method][metric].append(float(value))
                    overall_method[method][metric].append(float(value))

            delta_seeds = metrics.get("stoch_ins_delta_seeds", [])
            if isinstance(delta_seeds, list):
                for seed, value in zip(stoch_seeds, delta_seeds):
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        by_pack_block_seed[pack][blk][method][str(seed)].append(float(value))
                        by_pack_seed[pack][method][str(seed)].append(float(value))
                        overall_seed[method][str(seed)].append(float(value))

    def _agg(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": int(len(arr))}

    def _aggregate_group(group: Dict[str, Dict[str, List[float]]], seed_group: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for method, metrics in group.items():
            out[method] = {metric: _agg(vals) for metric, vals in metrics.items() if vals}
            if method in seed_group:
                out[method]["stoch_ins_delta_by_seed"] = {
                    seed: _agg(vals) for seed, vals in seed_group[method].items() if vals
                }
        return out

    agg_by_pack_block: Dict[str, Any] = {}
    for pack, blocks in by_pack_block_method.items():
        agg_by_pack_block[pack] = {}
        for blk, methods in blocks.items():
            agg_by_pack_block[pack][blk] = _aggregate_group(methods, by_pack_block_seed[pack][blk])

    agg_by_pack: Dict[str, Any] = {
        pack: _aggregate_group(methods, by_pack_seed[pack]) for pack, methods in by_pack_method.items()
    }
    agg_overall = _aggregate_group(overall_method, overall_seed)

    return {
        "aggregate_by_pack_block": agg_by_pack_block,
        "aggregate_by_pack": agg_by_pack,
        "aggregate_overall": agg_overall,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("run_feature_erf_paper_benchmark")
    args = parse_args()

    packs = list(mm.PACK_NAMES) if args.pack == "all" else [args.pack]
    blocks = list(args.blocks or mm.BLOCK_IDXS)
    methods = list(args.methods)
    val_root = Path(args.imagenet_val)
    if not val_root.exists():
        raise FileNotFoundError(f"ImageNet val root not found: {val_root}")

    pack_specs = {pack: mm.build_pack_spec(pack) for pack in packs}
    run_meta = _make_run_meta(pack_specs, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    for pack, root in _parse_root_overrides(args.index_root).items():
        mm.INDEX_ROOTS[pack] = root
    for pack, root in _parse_root_overrides(args.sae_root).items():
        mm.SAE_ROOTS[pack] = root

    manifest_rows = _load_manifest_rows(args.manifest_json) if args.manifest_json is not None else None
    sampled_manifest_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    if args.output_json.exists():
        try:
            existing = json.loads(args.output_json.read_text())
            old_meta = existing.get("_meta", {})
            if old_meta.get("fingerprint") == run_meta["fingerprint"]:
                all_rows = existing.get("per_triple", [])
                log.info("Resuming from %d prior rows", len(all_rows))
        except Exception:
            pass
    done_keys = {(row["pack"], row["block_idx"], row["feature_id"], row["sample_id"]) for row in all_rows}

    for pack in packs:
        pack_spec = pack_specs[pack]
        log.info("=== Pack %s ===", pack)
        model = mm.load_model(pack_spec, args.device)
        transform = mm.load_transform(pack_spec)
        n_prefix = mm.infer_prefix_count(model)
        log.info("[%s] inferred prefix_count=%d", pack, n_prefix)

        sae_cache: Dict[int, Any] = {}

        def get_sae(block_idx: int):
            if block_idx not in sae_cache:
                log.info("[%s] loading SAE for block %d", pack, block_idx)
                sae_cache[block_idx] = mm.load_sae(pack, block_idx, args.device)
            return sae_cache[block_idx]

        for block_idx in blocks:
            log.info("--- Pack=%s Block=%d ---", pack, block_idx)
            if manifest_rows is not None:
                triples = [
                    dict(triple)
                    for triple in manifest_rows
                    if str(triple.get("pack")) == pack and int(triple.get("block_idx", -1)) == int(block_idx)
                ]
            else:
                triples = mm.sample_triples_from_index(
                    pack,
                    block_idx,
                    val_root,
                    n_prefix=n_prefix,
                    n_features=int(args.n_features),
                    n_images=int(args.n_images),
                    seed=int(args.seed),
                )
                sampled_manifest_rows.extend(dict(triple) for triple in triples)
            triples = [
                triple
                for triple in triples
                if (triple["pack"], triple["block_idx"], triple["feature_id"], triple["sample_id"]) not in done_keys
            ]
            if args.limit is not None:
                triples = triples[: args.limit]
            log.info("[%s blk=%d] queued %d triples", pack, block_idx, len(triples))

            sae = get_sae(block_idx)
            for idx, triple in enumerate(triples, start=1):
                log.info(
                    "[%s blk=%d] [%d/%d] feat=%d sid=%d tok=%d",
                    pack,
                    block_idx,
                    idx,
                    len(triples),
                    int(triple["feature_id"]),
                    int(triple["sample_id"]),
                    int(triple["tok_max"]),
                )
                row = evaluate_triple(
                    model,
                    sae,
                    triple,
                    transform,
                    args.device,
                    methods=methods,
                    stoch_steps=int(args.stoch_steps),
                    stoch_samples=int(args.stoch_samples),
                    stoch_seeds=[int(seed) for seed in args.stoch_seeds],
                    cautious_steps=int(args.cautious_steps),
                    cautious_kwargs={
                        "lr": float(args.cautious_lr),
                        "lr_end": float(args.cautious_lr_end),
                        "tv_weight": float(args.cautious_tv_weight),
                        "irr_weight": float(args.cautious_irr_weight),
                        "init_prob": float(args.cautious_init_prob),
                        "init_mode": str(args.cautious_init_mode),
                        "reg_warmup_frac": float(args.cautious_reg_warmup_frac),
                        "restarts": int(args.cautious_restarts),
                        "budget_samples": int(args.cautious_budget_samples),
                        "select_best": bool(args.cautious_select_best),
                        "objective_mode": str(args.cautious_objective_mode),
                        "optimizer_mode": str(args.cautious_optimizer_mode),
                        "fixed_budget_frac": float(args.cautious_fixed_budget_frac),
                        "seed": int(args.seed),
                    },
                    strict_restore=bool(args.strict_restore),
                    store_scores=bool(args.store_scores),
                    baseline_mode=str(args.baseline_mode),
                )
                all_rows.append(row)
                done_keys.add((pack, block_idx, triple["feature_id"], triple["sample_id"]))

                partial = {"_meta": run_meta, "per_triple": all_rows}
                args.output_json.write_text(json.dumps(partial, indent=2))

    summary = aggregate_results(all_rows, [int(seed) for seed in args.stoch_seeds])
    final = {"_meta": run_meta, "per_triple": all_rows, **summary}
    args.output_json.write_text(json.dumps(final, indent=2))
    if args.save_manifest_json is not None:
        manifest_payload = {
            "_meta": {
                "seed": int(args.seed),
                "packs": packs,
                "blocks": blocks,
                "n_features": int(args.n_features),
                "n_images": int(args.n_images),
            },
            "triples": sampled_manifest_rows if manifest_rows is None else manifest_rows,
        }
        args.save_manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_manifest_json.write_text(json.dumps(manifest_payload, indent=2))

    print("\n=== Feature ERF Paper Benchmark ===")
    for method, metrics in final["aggregate_overall"].items():
        line = [f"{method:<18}"]
        for metric in MAIN_METRICS:
            agg = metrics.get(metric)
            if agg:
                line.append(f"{metric}={agg['mean']:.4f}±{agg['std']:.4f}(n={agg['n']})")
        print(" ".join(line))
    print(args.output_json)


if __name__ == "__main__":
    main()
