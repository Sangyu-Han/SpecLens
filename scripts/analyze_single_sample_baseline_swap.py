#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

REPO = Path(os.environ.get("SPECLENS_REPO", str(Path(__file__).resolve().parents[1]))).expanduser()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_legacy_multimodel_module():
    env_path = os.environ.get("FEATURE_ERF_LEGACY_MM")
    candidates = [Path(env_path).expanduser()] if env_path else []
    candidates.extend(
        [
            Path("/media/mipal/1TB/sangyu/codex_research_softins/eval_multimodel_erf.py"),
            Path("/home/sangyu/Desktop/Master/codex_research_softins/eval_multimodel_erf.py"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return _load_module("feature_erf_mm", candidate)
    raise FileNotFoundError("Could not locate eval_multimodel_erf.py")


def _load_paper_runner_module():
    return _load_module("paper_feature_erf_runner", REPO / "scripts" / "run_feature_erf_paper_benchmark.py")


mm = _load_legacy_multimodel_module()
paper = _load_paper_runner_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--feature-id", type=int)
    parser.add_argument("--sample-id", type=int)
    parser.add_argument("--select-mode", choices=["worst_cautious_insertion"], default="worst_cautious_insertion")
    parser.add_argument(
        "--baseline-modes",
        nargs="*",
        default=["default", "mean_image"],
        choices=["default", "zero", "mean_image", "global_mean_h", "global_mean_h_plus_pos"],
    )
    parser.add_argument("--cautious-steps", nargs="*", type=int, default=[32, 64])
    parser.add_argument("--stoch-steps", type=int, default=20)
    parser.add_argument("--stoch-samples", type=int, default=5)
    parser.add_argument("--stoch-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _select_row(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("pack") == args.pack and int(row.get("block_idx")) == int(args.block)
    ]
    if args.feature_id is not None and args.sample_id is not None:
        for row in candidates:
            if int(row.get("feature_id")) == int(args.feature_id) and int(row.get("sample_id")) == int(args.sample_id):
                return row
        raise ValueError("Requested feature/sample pair not found")

    if args.select_mode == "worst_cautious_insertion":
        ranked = []
        for row in candidates:
            metrics = row.get("methods", {}).get("cautious_cos", {})
            ins = metrics.get("insertion_auc")
            if isinstance(ins, (int, float)) and math.isfinite(ins):
                ranked.append((float(ins), row))
        if not ranked:
            raise ValueError("No rows with cautious_cos insertion_auc found")
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    raise ValueError(f"Unsupported select_mode: {args.select_mode}")


def _capture_mean_image_patch_baseline(model, x: torch.Tensor, n_prefix: int, block_idx: int) -> torch.Tensor:
    x_zero = torch.zeros_like(x)
    h_b0_zero, _ = mm._capture_block0_and_target(model, x_zero, block_idx)
    return h_b0_zero[:, n_prefix:, :].detach()


def _build_state_with_baseline(row: Dict[str, Any], model, sae, transform, device: str, baseline_mode: str):
    n_prefix = mm.infer_prefix_count(model)
    img = Image.open(str(row["image_path"])).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    h_b0_full, _ = mm._capture_block0_and_target(model, x, int(row["block_idx"]))
    n_patches = int(h_b0_full.shape[1] - n_prefix)

    if baseline_mode in {"default", "global_mean_h", "global_mean_h_plus_pos"}:
        baseline = mm._get_patch_baseline(
            model,
            n_patches,
            x=x,
            block_idx=int(row["block_idx"]),
            n_prefix=n_prefix,
            pack=str(row["pack"]),
            mode=baseline_mode,
        ).to(device=device, dtype=h_b0_full.dtype)
    elif baseline_mode == "zero":
        d_model = int(h_b0_full.shape[-1])
        baseline = torch.zeros(1, n_patches, d_model, device=device, dtype=h_b0_full.dtype)
    elif baseline_mode == "mean_image":
        baseline = _capture_mean_image_patch_baseline(model, x, n_prefix, int(row["block_idx"])).to(
            device=device,
            dtype=h_b0_full.dtype,
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode}")

    state = mm.SAEState(
        model=model,
        sae=sae,
        x=x,
        h_b0_full=h_b0_full,
        baseline=baseline,
        block_idx=int(row["block_idx"]),
        feature_id=int(row["feature_id"]),
        tok_max=int(row["tok_max"]),
        n_prefix=n_prefix,
    )
    return state


def _hard_curve(state, scores: np.ndarray, budget_fracs: Sequence[float]) -> Dict[str, Any]:
    n_patches = int(state.h_b0_patches.shape[1])
    scores_norm = mm.normalize_scores(np.asarray(scores, dtype=np.float32))
    order = np.argsort(-scores_norm.reshape(-1))
    budgets = sorted({max(0, min(n_patches, int(round(float(frac) * n_patches)))) for frac in budget_fracs})
    block_out_orig = mm._run_injected(state.model, state.x, state.prefix_tokens, state.h_b0_patches, state.block_idx)
    act_orig = mm._sae_act(state.sae, block_out_orig, state.feature_id, state.tok_max, state.n_prefix, "single")
    act_orig_safe = max(abs(act_orig), 1e-8)

    h_cur = state.baseline.clone()
    vals: List[float] = []
    current_k = 0
    for target_k in budgets:
        while current_k < target_k:
            h_cur = h_cur.clone()
            h_cur[0, order[current_k]] = state.h_b0_patches[0, order[current_k]]
            current_k += 1
        block_out = mm._run_injected(state.model, state.x, state.prefix_tokens, h_cur, state.block_idx)
        act_k = mm._sae_act(state.sae, block_out, state.feature_id, state.tok_max, state.n_prefix, "single")
        vals.append(float(act_k / act_orig_safe))
    return {
        "budgets": budgets,
        "budget_fracs": [float(k / n_patches) for k in budgets],
        "responses": vals,
    }


def _hard_deletion_curve(state, scores: np.ndarray, budget_fracs: Sequence[float]) -> Dict[str, Any]:
    n_patches = int(state.h_b0_patches.shape[1])
    scores_norm = mm.normalize_scores(np.asarray(scores, dtype=np.float32))
    order = np.argsort(-scores_norm.reshape(-1))
    budgets = sorted({max(0, min(n_patches, int(round(float(frac) * n_patches)))) for frac in budget_fracs})
    block_out_orig = mm._run_injected(state.model, state.x, state.prefix_tokens, state.h_b0_patches, state.block_idx)
    act_orig = mm._sae_act(state.sae, block_out_orig, state.feature_id, state.tok_max, state.n_prefix, "single")
    act_orig_safe = max(abs(act_orig), 1e-8)

    h_cur = state.h_b0_patches.clone()
    vals: List[float] = []
    current_k = 0
    for target_k in budgets:
        while current_k < target_k:
            h_cur = h_cur.clone()
            h_cur[0, order[current_k]] = state.baseline[0, order[current_k]]
            current_k += 1
        block_out = mm._run_injected(state.model, state.x, state.prefix_tokens, h_cur, state.block_idx)
        act_k = mm._sae_act(state.sae, block_out, state.feature_id, state.tok_max, state.n_prefix, "single")
        vals.append(float(act_k / act_orig_safe))
    fracs = [float(k / n_patches) for k in budgets]
    auc = float(np.trapz(np.asarray(vals, dtype=np.float64), np.asarray(fracs, dtype=np.float64)))
    return {
        "budgets": budgets,
        "budget_fracs": fracs,
        "responses": vals,
        "deletion_auc": auc,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text())
    row = _select_row(payload.get("per_triple", []), args)

    pack_spec = mm.build_pack_spec(args.pack)
    model = mm.load_model(pack_spec, args.device)
    transform = mm.load_transform(pack_spec)
    sae = mm.load_sae(args.pack, int(args.block), args.device)

    out: Dict[str, Any] = {
        "selected_row": {
            "pack": row["pack"],
            "block_idx": row["block_idx"],
            "feature_id": row["feature_id"],
            "sample_id": row["sample_id"],
            "tok_max": row["tok_max"],
            "image_path": row["image_path"],
            "original_cautious_insertion_auc": row.get("methods", {}).get("cautious_cos", {}).get("insertion_auc"),
        },
        "baseline_results": {},
    }

    curve_fracs = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    common_kwargs = {
        "stoch_steps": int(args.stoch_steps),
        "stoch_samples": int(args.stoch_samples),
        "stoch_seeds": [int(x) for x in args.stoch_seeds],
    }

    for baseline_mode in args.baseline_modes:
        state = _build_state_with_baseline(row, model, sae, transform, args.device, baseline_mode)
        n_patches = int(state.h_b0_patches.shape[1])
        grid = int(round(math.sqrt(n_patches)))
        baseline_out: Dict[str, Any] = {}

        plain_scores = np.asarray(mm.run_plain_ixg(state), dtype=np.float32)
        plain_metrics = paper._evaluate_scores(
            state,
            plain_scores,
            stoch_steps=common_kwargs["stoch_steps"],
            stoch_samples=common_kwargs["stoch_samples"],
            stoch_seeds=common_kwargs["stoch_seeds"],
        )
        baseline_out["plain_ixg"] = {
            "metrics": plain_metrics,
            "insertion_curve": _hard_curve(state, plain_scores, curve_fracs),
            "deletion_curve": _hard_deletion_curve(state, plain_scores, curve_fracs),
        }

        for steps in args.cautious_steps:
            scores = np.asarray(
                mm.run_cautious_cos(
                    state,
                    n_patches=n_patches,
                    grid=grid,
                    steps=int(steps),
                    lr=0.45,
                    lr_end=0.01,
                    tv_weight=0.01,
                    irr_weight=0.05,
                    init_prob=0.5,
                    reg_warmup_frac=0.0,
                    restarts=1,
                    budget_samples=1,
                    select_best=False,
                    seed=42,
                ),
                dtype=np.float32,
            )
            metrics = paper._evaluate_scores(
                state,
                scores,
                stoch_steps=common_kwargs["stoch_steps"],
                stoch_samples=common_kwargs["stoch_samples"],
                stoch_seeds=common_kwargs["stoch_seeds"],
            )
            baseline_out[f"cautious_cos_steps{steps}"] = {
                "metrics": metrics,
                "insertion_curve": _hard_curve(state, scores, curve_fracs),
                "deletion_curve": _hard_deletion_curve(state, scores, curve_fracs),
            }

        out["baseline_results"][baseline_mode] = baseline_out

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(args.output_json)


if __name__ == "__main__":
    main()
