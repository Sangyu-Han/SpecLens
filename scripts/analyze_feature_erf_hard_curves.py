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

REPO = Path(os.environ.get("SPECLENS_REPO", str(Path(__file__).resolve().parents[1]))).expanduser()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


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
            spec = importlib.util.spec_from_file_location("feature_erf_mm", candidate)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not locate eval_multimodel_erf.py")


mm = _load_legacy_multimodel_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--methods", nargs="*", default=["plain_ixg", "cautious_cos"])
    parser.add_argument("--budget-fracs", nargs="*", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _compute_curve(state, scores: np.ndarray, budget_fracs: Sequence[float]) -> Dict[str, Any]:
    n_patches = int(state.h_b0_patches.shape[1])
    scores_norm = mm.normalize_scores(np.asarray(scores, dtype=np.float32))
    order = np.argsort(-scores_norm.reshape(-1))
    budgets = []
    for frac in budget_fracs:
        k = int(round(float(frac) * n_patches))
        k = max(0, min(n_patches, k))
        budgets.append(k)
    budgets = sorted(set(budgets))

    block_out_orig = mm._run_injected(state.model, state.x, state.prefix_tokens, state.h_b0_patches, state.block_idx)
    act_orig = mm._sae_act(state.sae, block_out_orig, state.feature_id, state.tok_max, state.n_prefix, "single")
    act_orig_safe = max(abs(act_orig), 1e-8)

    h_cur = state.baseline.clone()
    current_k = 0
    vals: List[float] = []
    for target_k in budgets:
        while current_k < target_k:
            h_cur = h_cur.clone()
            h_cur[0, order[current_k]] = state.h_b0_patches[0, order[current_k]]
            current_k += 1
        block_out = mm._run_injected(state.model, state.x, state.prefix_tokens, h_cur, state.block_idx)
        act_k = mm._sae_act(state.sae, block_out, state.feature_id, state.tok_max, state.n_prefix, "single")
        vals.append(float(act_k / act_orig_safe))

    return {
        "n_patches": n_patches,
        "budgets": budgets,
        "budget_fracs_realized": [float(k / n_patches) for k in budgets],
        "responses": vals,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text())
    rows = [
        row
        for row in payload.get("per_triple", [])
        if row.get("pack") == args.pack and int(row.get("block_idx")) == int(args.block)
    ]
    if not rows:
        raise ValueError(f"No rows found for pack={args.pack} block={args.block}")

    pack_spec = mm.build_pack_spec(args.pack)
    model = mm.load_model(pack_spec, args.device)
    transform = mm.load_transform(pack_spec)
    sae = mm.load_sae(args.pack, int(args.block), args.device)
    n_prefix = mm.infer_prefix_count(model)

    curves_by_method: Dict[str, List[List[float]]] = {method: [] for method in args.methods}
    budgets: List[int] | None = None
    fracs: List[float] | None = None

    for idx, row in enumerate(rows, start=1):
        state = mm.build_state(model, sae, row, transform, args.device, n_prefix)
        for method in args.methods:
            metrics = row.get("methods", {}).get(method, {})
            scores = metrics.get("scores")
            if scores is None:
                raise ValueError(f"Missing stored scores for method={method} row={idx}")
            curve = _compute_curve(state, np.asarray(scores, dtype=np.float32), args.budget_fracs)
            budgets = curve["budgets"]
            fracs = curve["budget_fracs_realized"]
            curves_by_method[method].append(curve["responses"])

    out: Dict[str, Any] = {
        "pack": args.pack,
        "block": int(args.block),
        "n_rows": len(rows),
        "methods": {},
        "budgets": budgets,
        "budget_fracs": fracs,
    }
    for method, curves in curves_by_method.items():
        arr = np.asarray(curves, dtype=np.float64)
        out["methods"][method] = {
            "mean_curve": arr.mean(axis=0).tolist(),
            "std_curve": arr.std(axis=0).tolist(),
            "sem_curve": (arr.std(axis=0) / math.sqrt(max(len(curves), 1))).tolist(),
            "n": int(arr.shape[0]),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print(args.output_json)


if __name__ == "__main__":
    main()
