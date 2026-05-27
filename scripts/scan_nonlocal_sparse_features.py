from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
from autolabel_eval.utils import feature_key


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _build_config(args: argparse.Namespace) -> EvalConfig:
    config = EvalConfig()
    overrides: dict[str, Any] = {
        "workspace_root": Path(args.workspace_root).resolve(),
        "model_name": str(args.vision_model_name),
        "blocks": (int(args.block_idx),),
        "features_per_block": int(args.n_features_hint),
        "train_examples_per_feature": int(args.top_k),
        "holdout_examples_per_feature": int(args.holdout_k),
        "deciles_root_override": Path(args.deciles_root).resolve(),
        "checkpoints_root_override": Path(args.checkpoints_root).resolve(),
        "checkpoint_relpath_template": str(args.checkpoint_pattern),
        "dataset_root_override": Path(args.dataset_root).resolve(),
        "shuffle_feature_candidates": False,
        "image_size": int(args.image_size),
        "resize_size": int(args.resize_size),
        "grid_size": int(args.grid_size),
        "n_patches": int(args.n_patches),
    }
    return replace(config, **overrides)


def _synset_from_path(image_path: str) -> str:
    return Path(image_path).parent.name


def _visible_patch_count_from_values(
    values: np.ndarray,
    token_idx: int,
    *,
    grid_size: int,
    activation_threshold: float,
) -> int:
    grid = np.asarray(values, dtype=np.float32).reshape(int(grid_size), int(grid_size))
    scaled = _normalize_positive_values(grid, lower_percentile=58.0, upper_percentile=99.5, gamma=0.9)
    mask_grid = (scaled >= float(activation_threshold)).astype(np.uint8)
    token_row, token_col = divmod(int(token_idx), int(grid_size))
    if 0 <= token_row < int(grid_size) and 0 <= token_col < int(grid_size):
        mask_grid[token_row, token_col] = 1
    if int(mask_grid.sum()) <= 0:
        row, col = divmod(int(np.argmax(grid)), int(grid_size))
        mask_grid[row, col] = 1
    return int(mask_grid.sum())


def _ledger_examples_for_unit(
    feature_rows: Any,
    runtime: LegacyRuntime,
    path_lookup: dict[int, str],
    *,
    n_needed: int,
    block_idx: int,
    n_patches: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    used_sample_ids: set[int] = set()
    for row in feature_rows.itertuples(index=False):
        sample_id = int(row.sample_id)
        if sample_id in used_sample_ids:
            continue
        token_idx = runtime.row_x_to_token_idx(int(row.x))
        if token_idx < 0 or token_idx >= int(n_patches):
            continue
        image_path = path_lookup.get(sample_id, "")
        if not image_path:
            continue
        examples.append(
            {
                "rank": int(len(examples)),
                "sample_id": sample_id,
                "token_idx": int(token_idx),
                "image_path": image_path,
                "synset": _synset_from_path(image_path),
                "ledger_score": float(row.score),
                "block_idx": int(block_idx),
            }
        )
        used_sample_ids.add(sample_id)
        if len(examples) >= int(n_needed):
            break
    return examples


def _evaluate_examples_batch(
    runtime: LegacyRuntime,
    examples: list[dict[str, Any]],
    *,
    block_idx: int,
    feature_id: int,
    activation_threshold: float,
) -> list[dict[str, Any]]:
    if torch is None:
        raise RuntimeError("torch is required for feature evaluation")
    if not examples:
        return []

    sae = runtime.load_sae(int(block_idx))
    prefix = int(runtime.adapter.prefix_count())
    evaluated: list[dict[str, Any]] = []
    for example in examples:
        with Image.open(str(example["image_path"])) as image:
            x = runtime.transform(image.convert("RGB")).unsqueeze(0).to(runtime.device)
        with torch.no_grad():
            capture = runtime.adapter.capture_block0_input_and_target_block_out(x, block_idx=int(block_idx))
            acts = sae(capture.target_block_output).get("feature_acts")[0, :, int(feature_id)]
        full_acts = acts.detach().cpu().numpy().astype(np.float32)

        token_idx = int(example["token_idx"])
        patch_values = full_acts[prefix : prefix + int(runtime.config.n_patches)]
        if patch_values.size != int(runtime.config.n_patches):
            continue
        act_at_target = float(patch_values[token_idx])
        argmax_tok = int(np.argmax(patch_values))
        max_act = float(patch_values[argmax_tok])
        target_ratio = act_at_target / max(max_act, 1e-8)
        if max_act < float(runtime.config.min_feature_max_act):
            continue
        if target_ratio < float(runtime.config.target_ratio_min):
            continue
        cls_act = float(full_acts[0]) if prefix > 0 else 0.0
        visible_patch_count = _visible_patch_count_from_values(
            patch_values,
            token_idx,
            grid_size=int(runtime.config.grid_size),
            activation_threshold=float(activation_threshold),
        )
        enriched = dict(example)
        enriched.update(
            {
                "rank": int(len(evaluated)),
                "validated_act": act_at_target,
                "max_act": max_act,
                "argmax_tok": argmax_tok,
                "target_to_max_ratio": float(target_ratio),
                "visible_patch_count": int(visible_patch_count),
                "cls_act": cls_act,
            }
        )
        evaluated.append(enriched)
    return evaluated


def _passes_variant(feature: dict[str, Any], criteria: dict[str, Any]) -> bool:
    if int(feature["mode_count"]) < int(criteria["min_mode_count"]):
        return False
    if int(feature["sparse_hits"]) < int(criteria["min_sparse_hits"]):
        return False
    if float(feature["mean_visible_patch_count"]) > float(criteria["max_mean_visible_patches"]):
        return False
    if int(feature["cls_hits_at_threshold"]) > int(criteria["max_cls_hits_at_threshold"]):
        return False
    return True


def _variant_criteria(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "strict_synset4_sparse5_clsnever": {
            "description": "mode synset >=4/5, >=4/5 examples <=5 visible patches, mean visible <=5, CLS hits 0/5",
            "min_mode_count": 4,
            "max_visible_patches": 5,
            "min_sparse_hits": 4,
            "max_mean_visible_patches": 5.0,
            "max_cls_hits_at_threshold": int(args.max_cls_hits_at_threshold),
        },
        "strict_synset4_sparse8_clsnever": {
            "description": "mode synset >=4/5, >=4/5 examples <=8 visible patches, mean visible <=8, CLS hits 0/5",
            "min_mode_count": 4,
            "max_visible_patches": 8,
            "min_sparse_hits": 4,
            "max_mean_visible_patches": 8.0,
            "max_cls_hits_at_threshold": int(args.max_cls_hits_at_threshold),
        },
        "synset3_sparse8_clsnever": {
            "description": "mode synset >=3/5, >=4/5 examples <=8 visible patches, mean visible <=8, CLS hits 0/5",
            "min_mode_count": 3,
            "max_visible_patches": 8,
            "min_sparse_hits": 4,
            "max_mean_visible_patches": 8.0,
            "max_cls_hits_at_threshold": int(args.max_cls_hits_at_threshold),
        },
    }


def _candidate_for_any_variant(feature: dict[str, Any], variants: dict[str, dict[str, Any]]) -> bool:
    for criteria in variants.values():
        base = dict(criteria)
        base["max_cls_hits_at_threshold"] = 999999
        if _passes_variant(feature, base):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustively scan sparse class-consistent nonlocal candidates.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--session-name", default="clip50k_block10_nonlocal_exhaustive_20260429")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--block-idx", type=int, default=10)
    parser.add_argument("--scan-limit", type=int, default=0, help="0 means scan all grouped units.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--holdout-k", type=int, default=2)
    parser.add_argument("--max-cls-hits-at-threshold", type=int, default=0)
    parser.add_argument("--cls-relative-threshold", type=float, default=0.5)
    parser.add_argument("--activation-threshold", type=float, default=0.24)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--n-features-hint", type=int, default=65536)
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
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=14)
    parser.add_argument("--n-patches", type=int, default=196)
    args = parser.parse_args()

    config = _build_config(args)
    config.ensure_dirs()
    runtime = LegacyRuntime(config)
    start_time = time.time()

    try:
        frame = runtime.load_decile_frame(int(args.block_idx))
        grouped = (
            frame.groupby("unit")
            .agg(count=("score", "count"), mean_score=("score", "mean"), max_score=("score", "max"))
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
        if int(args.scan_limit) > 0:
            grouped = grouped.head(int(args.scan_limit)).copy()

        frame = frame.sort_values(["unit", "score"], ascending=[True, False], kind="mergesort")
        feature_rows_by_unit = {int(unit): rows for unit, rows in frame.groupby("unit", sort=False)}
        all_sample_ids = [int(v) for v in frame["sample_id"].astype(int).unique().tolist()]
        path_lookup = runtime.lookup_paths(all_sample_ids)
        variants = _variant_criteria(args)

        broad_candidates: list[dict[str, Any]] = []
        rejection_counts: Counter[str] = Counter()
        mode_count_histogram: Counter[int] = Counter()
        cls_hits_histogram: Counter[int] = Counter()
        cls_role_histogram: Counter[str] = Counter()

        total = int(len(grouped))
        n_needed = int(args.top_k) + int(args.holdout_k)
        for scan_rank, stat_row in enumerate(grouped.itertuples(index=False), start=1):
            unit = int(stat_row.unit)
            rows = feature_rows_by_unit.get(unit)
            if rows is None:
                rejection_counts["missing_rows"] += 1
                continue

            ledger_examples = _ledger_examples_for_unit(
                rows,
                runtime,
                path_lookup,
                n_needed=n_needed,
                block_idx=int(args.block_idx),
                n_patches=int(args.n_patches),
            )
            if len(ledger_examples) < int(args.top_k):
                rejection_counts["fewer_than_top_k_ledger_examples"] += 1
                continue

            top_synsets = [str(example["synset"]) for example in ledger_examples[: int(args.top_k)]]
            ledger_mode_count = Counter(top_synsets).most_common(1)[0][1]
            if ledger_mode_count < 3:
                rejection_counts["ledger_mode_count_lt3"] += 1
                continue

            evaluated = _evaluate_examples_batch(
                runtime,
                ledger_examples,
                block_idx=int(args.block_idx),
                feature_id=unit,
                activation_threshold=float(args.activation_threshold),
            )
            if len(evaluated) < int(args.top_k):
                rejection_counts["fewer_than_top_k_validated_examples"] += 1
                continue

            label_examples = evaluated[: int(args.top_k)]
            holdout_examples = evaluated[int(args.top_k) : int(args.top_k) + int(args.holdout_k)]
            synsets = [str(example["synset"]) for example in label_examples]
            patch_counts = [int(example["visible_patch_count"]) for example in label_examples]
            cls_acts = [float(example["cls_act"]) for example in label_examples]
            counter = Counter(synsets)
            mode_label, mode_count = counter.most_common(1)[0]
            target_median = float(np.median([float(example["validated_act"]) for example in label_examples]))
            cls_threshold = float(args.cls_relative_threshold) * float(target_median)
            cls_hits = int(sum(float(value) >= cls_threshold for value in cls_acts))
            if cls_hits <= 0:
                cls_role = "cls_never"
            elif cls_hits >= int(args.top_k):
                cls_role = "cls_always"
            else:
                cls_role = "cls_partial"

            mode_count_histogram[int(mode_count)] += 1
            cls_hits_histogram[int(cls_hits)] += 1
            cls_role_histogram[str(cls_role)] += 1

            feature_payload: dict[str, Any] = {
                "feature_key": feature_key(int(args.block_idx), unit),
                "block_idx": int(args.block_idx),
                "feature_id": unit,
                "scan_rank": int(scan_rank),
                "selection_stats": {
                    "count": int(stat_row.count),
                    "mean_score": float(stat_row.mean_score),
                    "max_score": float(stat_row.max_score),
                    "ratio": float(stat_row.ratio),
                },
                "mode_label": str(mode_label),
                "mode_count": int(mode_count),
                "target_median": float(target_median),
                "cls_threshold": float(cls_threshold),
                "cls_relative_threshold": float(args.cls_relative_threshold),
                "cls_hits_at_threshold": int(cls_hits),
                "cls_positive_hits": int(sum(float(value) > 0.0 for value in cls_acts)),
                "cls_role": str(cls_role),
                "mean_cls_act": float(np.mean(cls_acts)),
                "max_cls_act": float(max(cls_acts, default=0.0)),
                "mean_visible_patch_count": float(np.mean(patch_counts)),
                "top5_synsets": synsets,
                "top5_patch_counts": patch_counts,
                "top5_cls_acts": cls_acts,
                "label_examples": label_examples,
                "holdout_examples": holdout_examples,
            }
            # Attach sparse-hits for each variant threshold so downstream filtering is auditable.
            feature_payload["sparse_hits_by_max_visible"] = {
                str(criteria["max_visible_patches"]): int(
                    sum(v <= int(criteria["max_visible_patches"]) for v in patch_counts)
                )
                for criteria in variants.values()
            }
            feature_payload["variant_membership"] = []
            for name, criteria in variants.items():
                local = dict(feature_payload)
                local["sparse_hits"] = int(
                    sum(v <= int(criteria["max_visible_patches"]) for v in patch_counts)
                )
                if _passes_variant(local, criteria):
                    feature_payload["variant_membership"].append(str(name))
            # Broad pool keeps sparse/class candidates regardless of CLS role.
            broad = dict(feature_payload)
            broad["sparse_hits"] = max(int(v) for v in broad["sparse_hits_by_max_visible"].values())
            if _candidate_for_any_variant(broad, variants):
                broad_candidates.append(feature_payload)

            if scan_rank == 1 or scan_rank % int(args.progress_every) == 0:
                elapsed = time.time() - start_time
                print(
                    f"[scan {scan_rank:05d}/{total:05d}] broad={len(broad_candidates)} "
                    f"strict8_clsnever={sum('strict_synset4_sparse8_clsnever' in f['variant_membership'] for f in broad_candidates)} "
                    f"elapsed_min={elapsed / 60.0:.1f}",
                    flush=True,
                )

        variant_payload: dict[str, Any] = {}
        for name, criteria in variants.items():
            members = [feature for feature in broad_candidates if name in feature["variant_membership"]]
            variant_payload[name] = {
                "criteria": criteria,
                "count": int(len(members)),
                "selected_feature_keys": [str(feature["feature_key"]) for feature in members],
                "features": members,
            }

        by_cls_role: dict[str, Any] = {}
        for role in ["cls_never", "cls_partial", "cls_always"]:
            members = [feature for feature in broad_candidates if feature["cls_role"] == role]
            by_cls_role[role] = {
                "count": int(len(members)),
                "selected_feature_keys": [str(feature["feature_key"]) for feature in members],
                "features": members,
            }

        output_json = (
            Path(args.output_json).resolve()
            if str(args.output_json).strip()
            else Path(args.workspace_root).resolve()
            / "outputs"
            / "manifests"
            / f"{args.session_name}.json"
        )
        payload = {
            "session_name": str(args.session_name),
            "source": "exhaustive_block10_sparse_class_clsrole_scan",
            "common_criteria": {
                "block_idx": int(args.block_idx),
                "top_k": int(args.top_k),
                "holdout_k": int(args.holdout_k),
                "scan_limit": int(args.scan_limit),
                "scanned_all_grouped_units": int(args.scan_limit) <= 0,
                "sort": "ratio_desc_then_max_desc_then_count_desc",
                "activation_threshold": float(args.activation_threshold),
                "cls_relative_threshold": float(args.cls_relative_threshold),
                "max_cls_hits_at_threshold": int(args.max_cls_hits_at_threshold),
                "target_ratio_min": float(config.target_ratio_min),
                "min_feature_max_act": float(config.min_feature_max_act),
                "vision_model_name": str(args.vision_model_name),
                "image_size": int(args.image_size),
                "resize_size": int(args.resize_size),
                "grid_size": int(args.grid_size),
                "n_patches": int(args.n_patches),
            },
            "scan_stats": {
                "grouped_units_total": int(len(grouped)),
                "broad_sparseclass_candidates": int(len(broad_candidates)),
                "elapsed_seconds": float(time.time() - start_time),
                "rejection_counts": {str(k): int(v) for k, v in sorted(rejection_counts.items())},
                "mode_count_histogram_after_validation": {
                    str(k): int(v) for k, v in sorted(mode_count_histogram.items())
                },
                "cls_hits_histogram_after_validation": {
                    str(k): int(v) for k, v in sorted(cls_hits_histogram.items())
                },
                "cls_role_histogram_after_validation": {
                    str(k): int(v) for k, v in sorted(cls_role_histogram.items())
                },
            },
            "broad_sparseclass_candidates": broad_candidates,
            "by_cls_role_for_broad_pool": by_cls_role,
            "variants": variant_payload,
        }
        _write_json(output_json, payload)

        stem = output_json.with_suffix("")
        for name, variant in variant_payload.items():
            _write_text(stem.parent / f"{stem.name}__{name}_feature_keys.txt", list(variant["selected_feature_keys"]))
        _write_text(
            stem.parent / f"{stem.name}__broad_sparseclass_feature_keys.txt",
            [str(feature["feature_key"]) for feature in broad_candidates],
        )
        print(str(output_json), flush=True)
        for name, variant in variant_payload.items():
            print(f"{name}: {int(variant['count'])}", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
