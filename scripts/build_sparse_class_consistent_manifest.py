from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch

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
except Exception:
    torch = None  # type: ignore[assignment]

from autolabel_eval.config import EvalConfig
from autolabel_eval.legacy import LegacyRuntime
from autolabel_eval.rendering import _normalize_positive_values
from autolabel_eval.utils import feature_key, write_json


def _build_config(args: argparse.Namespace) -> EvalConfig:
    config = EvalConfig()
    overrides: dict[str, Any] = {
        "workspace_root": Path(args.workspace_root).resolve(),
        "model_name": str(args.vision_model_name),
        "blocks": (int(args.block_idx),),
        "features_per_block": int(args.n_features),
        "train_examples_per_feature": int(args.top_k),
        "holdout_examples_per_feature": 0,
        "deciles_root_override": Path(args.deciles_root).resolve(),
        "checkpoints_root_override": Path(args.checkpoints_root).resolve(),
        "checkpoint_relpath_template": str(args.checkpoint_pattern),
        "dataset_root_override": Path(args.dataset_root).resolve(),
        "shuffle_feature_candidates": False,
    }
    return replace(config, **overrides)


def _visible_patch_count(
    runtime: LegacyRuntime,
    image_path: str,
    block_idx: int,
    feature_id: int,
    token_idx: int,
    *,
    grid_size: int = 14,
    activation_threshold: float = 0.24,
) -> int:
    values = runtime.feature_activation_map_visible_patches(image_path, block_idx, feature_id)
    values = np.asarray(values, dtype=np.float32).reshape(grid_size, grid_size)
    scaled = _normalize_positive_values(values, lower_percentile=58.0, upper_percentile=99.5, gamma=0.9)
    mask_grid = (scaled >= float(activation_threshold)).astype(np.uint8)
    token_row, token_col = divmod(int(token_idx), int(grid_size))
    if 0 <= token_row < grid_size and 0 <= token_col < grid_size:
        mask_grid[token_row, token_col] = 1
    if int(mask_grid.sum()) <= 0:
        row, col = divmod(int(np.argmax(values)), int(grid_size))
        mask_grid[row, col] = 1
    return int(mask_grid.sum())


def _synset_from_path(image_path: str) -> str:
    return Path(image_path).parent.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select sparse class-consistent features from full ledger deciles."
    )
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--block-idx", type=int, default=10)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--feature-key", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-same-label-count", type=int, default=4)
    parser.add_argument("--max-visible-patches", type=int, default=5)
    parser.add_argument("--min-sparse-hits", type=int, default=4)
    parser.add_argument("--max-mean-visible-patches", type=float, default=5.0)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--vision-model-name", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    parser.add_argument(
        "--deciles-root",
        default="/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_index/deciles",
    )
    parser.add_argument(
        "--checkpoints-root",
        default="/home/sangyu/Desktop/Master/SpecLens/outputs/spec_lens_store/clip_50k_sae",
    )
    parser.add_argument(
        "--checkpoint-pattern",
        default="model.blocks.{block_idx}/step_0050000_tokens_204800000.pt",
    )
    parser.add_argument("--dataset-root", default="/data/datasets/imagenet/val")
    args = parser.parse_args()

    config = _build_config(args)
    config.ensure_dirs()
    runtime = LegacyRuntime(config)
    try:
        frame = runtime.load_decile_frame(int(args.block_idx))
        grouped = (
            frame.groupby("unit")
            .agg(
                count=("score", "count"),
                mean_score=("score", "mean"),
                max_score=("score", "max"),
            )
            .reset_index()
        )
        grouped["ratio"] = grouped["max_score"] / grouped["mean_score"].clip(lower=1e-8)
        excluded = set(config.exclude_feature_ids.get(int(args.block_idx), []))
        grouped = grouped[~grouped["unit"].isin(excluded)].copy()
        grouped = grouped.sort_values(
            ["ratio", "max_score", "count"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        provided_keys = [str(v).strip() for v in list(args.feature_key) if str(v).strip()]
        if provided_keys:
            provided_units: list[int] = []
            for key in provided_keys:
                block_part, feature_part = key.split("/")
                block_idx = int(block_part.split("_")[1])
                if block_idx != int(args.block_idx):
                    raise SystemExit(f"Feature key {key} does not match --block-idx {args.block_idx}.")
                provided_units.append(int(feature_part.split("_")[1]))
            stat_lookup = {int(row.unit): row for row in grouped.itertuples(index=False)}
            ordered_rows = []
            for unit in provided_units:
                if unit not in stat_lookup:
                    raise SystemExit(f"Feature id {unit} missing from block {args.block_idx} deciles.")
                ordered_rows.append(stat_lookup[unit])
            iter_rows = ordered_rows
        else:
            if int(args.scan_limit) > 0:
                grouped = grouped.head(int(args.scan_limit)).copy()
            iter_rows = list(grouped.itertuples(index=False))

        sid_to_path_cache: dict[int, str] = {}
        selected: list[dict[str, Any]] = []

        for rank, stat_row in enumerate(iter_rows, start=1):
            unit = int(stat_row.unit)
            if rank == 1 or rank % 50 == 0:
                print(
                    f"[scan] rank={rank} selected={len(selected)} unit={unit} "
                    f"ratio={float(stat_row.ratio):.3f} max={float(stat_row.max_score):.3f}",
                    flush=True,
                )
            feature_rows = frame[frame["unit"] == unit].sort_values("score", ascending=False)
            sample_ids = [int(v) for v in feature_rows["sample_id"].astype(int).unique().tolist()]
            missing = [sid for sid in sample_ids if sid not in sid_to_path_cache]
            if missing:
                sid_to_path_cache.update(runtime.lookup_paths(missing))

            accepted_examples: list[dict[str, Any]] = []
            used_sample_ids: set[int] = set()
            for row in feature_rows.itertuples(index=False):
                sample_id = int(row.sample_id)
                if sample_id in used_sample_ids:
                    continue
                tok_idx = runtime.row_x_to_token_idx(int(row.x))
                if tok_idx < 0 or tok_idx >= config.n_patches:
                    continue
                image_path = sid_to_path_cache.get(sample_id, "")
                if not image_path:
                    continue
                validation = runtime.validate_feature_token(
                    image_path=image_path,
                    block_idx=int(args.block_idx),
                    feature_id=unit,
                    token_idx=tok_idx,
                    ledger_score=float(row.score),
                )
                if validation is None:
                    continue
                visible_patch_count = _visible_patch_count(
                    runtime,
                    image_path=image_path,
                    block_idx=int(args.block_idx),
                    feature_id=unit,
                    token_idx=tok_idx,
                )
                synset = _synset_from_path(image_path)
                accepted_examples.append(
                    {
                        "rank": int(len(accepted_examples)),
                        "sample_id": sample_id,
                        "token_idx": tok_idx,
                        "image_path": image_path,
                        "synset": synset,
                        "ledger_score": float(row.score),
                        "validated_act": float(validation["act_at_target"]),
                        "visible_patch_count": int(visible_patch_count),
                    }
                )
                used_sample_ids.add(sample_id)
                if len(accepted_examples) >= int(args.top_k):
                    break

            if len(accepted_examples) < int(args.top_k):
                continue

            synsets = [str(example["synset"]) for example in accepted_examples]
            patch_counts = [int(example["visible_patch_count"]) for example in accepted_examples]
            counter = Counter(synsets)
            mode_label, mode_count = counter.most_common(1)[0]
            sparse_hits = sum(int(v <= int(args.max_visible_patches)) for v in patch_counts)
            mean_visible = float(np.mean(patch_counts))
            if mode_count < int(args.min_same_label_count):
                continue
            if sparse_hits < int(args.min_sparse_hits):
                continue
            if mean_visible > float(args.max_mean_visible_patches):
                continue

            selected.append(
                {
                    "feature_key": feature_key(int(args.block_idx), unit),
                    "block_idx": int(args.block_idx),
                    "feature_id": unit,
                    "selection_rank": int(rank),
                    "selection_stats": {
                        "count": int(stat_row.count),
                        "mean_score": float(stat_row.mean_score),
                        "max_score": float(stat_row.max_score),
                        "ratio": float(stat_row.ratio),
                    },
                    "mode_label": str(mode_label),
                    "mode_count": int(mode_count),
                    "sparse_hits": int(sparse_hits),
                    "mean_visible_patch_count": float(mean_visible),
                    "top5_synsets": synsets,
                    "top5_patch_counts": patch_counts,
                    "label_examples": accepted_examples,
                }
            )
            print(
                f"[selected {len(selected):02d}/{int(args.n_features)}] "
                f"{feature_key(int(args.block_idx), unit)} mode={mode_label} "
                f"mode_count={mode_count} sparse_hits={sparse_hits} mean_visible={mean_visible:.2f}",
                flush=True,
            )
            if len(selected) >= int(args.n_features):
                break

        if len(selected) < int(args.n_features):
            raise SystemExit(
                f"Only found {len(selected)} features matching criteria; need {int(args.n_features)}. "
                "Relax criteria or increase --scan-limit."
            )

        output_json = (
            Path(args.output_json).resolve()
            if str(args.output_json).strip()
            else Path(args.workspace_root).resolve() / "outputs" / "manifests" / f"{args.session_name}.json"
        )
        payload = {
            "session_name": str(args.session_name),
            "source": "full_ledger_sparse_class_consistent",
            "criteria": {
                "block_idx": int(args.block_idx),
                "top_k": int(args.top_k),
                "min_same_label_count": int(args.min_same_label_count),
                "max_visible_patches": int(args.max_visible_patches),
                "min_sparse_hits": int(args.min_sparse_hits),
                "max_mean_visible_patches": float(args.max_mean_visible_patches),
                "scan_limit": int(args.scan_limit),
                "sort": "ratio_desc_then_max_desc_then_count_desc",
                "vision_model_name": str(args.vision_model_name),
                "deciles_root": str(Path(args.deciles_root).resolve()),
                "checkpoints_root": str(Path(args.checkpoints_root).resolve()),
                "checkpoint_pattern": str(args.checkpoint_pattern),
            },
            "selected_feature_keys": [str(row["feature_key"]) for row in selected],
            "features": selected,
            "n_selected": int(len(selected)),
        }
        write_json(output_json, payload)
        print(output_json)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
