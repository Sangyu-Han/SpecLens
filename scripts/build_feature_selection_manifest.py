from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
try:
    sys.path.remove(str(SCRIPT_DIR))
except ValueError:
    pass
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autolabel_eval.config import EvalConfig
from autolabel_eval.feature_bank import build_feature_bank, load_feature_bank


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _build_config_from_args(args: Any) -> EvalConfig:
    config = EvalConfig()
    overrides: dict[str, Any] = {}
    if getattr(args, "workspace_root", None):
        overrides["workspace_root"] = Path(args.workspace_root)
    if getattr(args, "vision_model_name", None):
        overrides["model_name"] = str(args.vision_model_name)
    if getattr(args, "blocks", None):
        overrides["blocks"] = tuple(int(v) for v in args.blocks)
    if getattr(args, "features_per_block", None):
        overrides["features_per_block"] = int(args.features_per_block)
    if getattr(args, "train_per_feature", None):
        overrides["train_examples_per_feature"] = int(args.train_per_feature)
    if getattr(args, "holdout_per_feature", None):
        overrides["holdout_examples_per_feature"] = int(args.holdout_per_feature)
    if getattr(args, "deciles_root", None):
        overrides["deciles_root_override"] = Path(args.deciles_root)
    if getattr(args, "offline_meta_root", None):
        overrides["offline_meta_root_override"] = Path(args.offline_meta_root)
    if getattr(args, "checkpoints_root", None):
        overrides["checkpoints_root_override"] = Path(args.checkpoints_root)
    if getattr(args, "checkpoint_pattern", None):
        overrides["checkpoint_relpath_template"] = str(args.checkpoint_pattern)
    if getattr(args, "dataset_root", None):
        overrides["dataset_root_override"] = Path(args.dataset_root)
    if overrides:
        config = replace(config, **overrides)
    config.ensure_dirs()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a selection manifest from the current feature bank.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--vision-model-name", default="")
    parser.add_argument("--blocks", nargs="*", type=int, default=[2, 6, 10])
    parser.add_argument("--features-per-block", type=int, default=20)
    parser.add_argument("--train-per-feature", type=int, default=5)
    parser.add_argument("--holdout-per-feature", type=int, default=2)
    parser.add_argument("--label-examples-per-feature", type=int, default=5)
    parser.add_argument("--deciles-root", default="")
    parser.add_argument("--offline-meta-root", default="")
    parser.add_argument("--checkpoints-root", default="")
    parser.add_argument("--checkpoint-pattern", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--session-name", default="clip50k_feature_selection_20260421")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--build-feature-bank-if-missing", action="store_true")
    args = parser.parse_args()

    config = _build_config_from_args(args)
    if not config.feature_bank_json.exists():
        if not args.build_feature_bank_if_missing:
            raise SystemExit(
                f"Feature bank missing at {config.feature_bank_json}. "
                "Pass --build-feature-bank-if-missing or build it first."
            )
        build_feature_bank(config)

    feature_bank = load_feature_bank(config)
    label_examples_per_feature = max(1, int(args.label_examples_per_feature))
    per_block = max(1, int(args.features_per_block))
    selected_features: list[dict[str, Any]] = []

    for block_idx in tuple(int(v) for v in args.blocks):
        block_payload = feature_bank["blocks"][str(block_idx)]
        block_features = list(block_payload["features"])[:per_block]
        if len(block_features) < per_block:
            raise SystemExit(
                f"Requested {per_block} features for block {block_idx}, "
                f"but feature bank only has {len(block_features)}."
            )
        for feature in block_features:
            train_rows = list(feature.get("train", []))[:label_examples_per_feature]
            if len(train_rows) < label_examples_per_feature:
                raise SystemExit(
                    f"Feature {feature['feature_key']} has only {len(train_rows)} train rows; "
                    f"need {label_examples_per_feature}."
                )
            selected_features.append(
                {
                    "feature_key": str(feature["feature_key"]),
                    "block_idx": int(feature["block_idx"]),
                    "feature_id": int(feature["feature_id"]),
                    "selection_stats": dict(feature.get("selection_stats", {})),
                    "label_examples": [
                        {
                            "rank": rank,
                            "sample_id": int(row["sample_id"]),
                            "token_idx": int(row["target_patch_idx"]),
                        }
                        for rank, row in enumerate(train_rows)
                    ],
                }
            )

    output_path = (
        Path(args.output_json)
        if str(args.output_json).strip()
        else config.workspace_root / "outputs" / "manifests" / f"{args.session_name}.json"
    )
    payload = {
        "session_name": str(args.session_name),
        "source": "feature_bank",
        "feature_bank_json": str(config.feature_bank_json),
        "vision_model_name": str(config.model_name),
        "blocks": [int(v) for v in args.blocks],
        "features_per_block": per_block,
        "label_examples_per_feature": label_examples_per_feature,
        "selection": [
            {
                "block_idx": int(feature["block_idx"]),
                "feature_id": int(feature["feature_id"]),
                "feature_key": str(feature["feature_key"]),
            }
            for feature in selected_features
        ],
        "features": selected_features,
    }
    _write_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
