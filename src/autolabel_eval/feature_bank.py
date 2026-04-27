from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np

from .config import EvalConfig
from .legacy import LegacyRuntime, token_record_from_row
from .utils import feature_key, read_json, write_json


def _validate_payload_shape(config: EvalConfig, payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"blocks": {}}
    for block_idx in config.blocks:
        features = payload["blocks"][str(block_idx)]["features"]
        train_token_uids = set()
        holdout_token_uids = set()
        overlap = []
        for feature in features:
            if len(feature["train"]) != config.train_examples_per_feature:
                raise ValueError(f"block {block_idx} feature {feature['feature_id']} has wrong train count")
            if len(feature["holdout"]) != config.holdout_examples_per_feature:
                raise ValueError(f"block {block_idx} feature {feature['feature_id']} has wrong holdout count")
            cur_train = {row["token_uid"] for row in feature["train"]}
            cur_holdout = {row["token_uid"] for row in feature["holdout"]}
            inter = sorted(cur_train & cur_holdout)
            if inter:
                overlap.extend(inter)
            train_token_uids.update(cur_train)
            holdout_token_uids.update(cur_holdout)
        diagnostics["blocks"][str(block_idx)] = {
            "n_features": len(features),
            "n_train_tokens": len(train_token_uids),
            "n_holdout_tokens": len(holdout_token_uids),
            "train_holdout_overlap": overlap,
        }
    return diagnostics


def build_feature_bank(config: EvalConfig) -> dict[str, Any]:
    config.ensure_dirs()
    runtime = LegacyRuntime(config)
    payload: dict[str, Any] = {
        "config": config.to_dict(),
        "blocks": {},
    }
    try:
        for block_idx in config.blocks:
            frame = runtime.load_decile_frame(block_idx)
            grouped = (
                frame.groupby("unit")
                .agg(
                    count=("score", "count"),
                    mean_score=("score", "mean"),
                    max_score=("score", "max"),
                )
                .reset_index()
            )
            excluded = set(config.exclude_feature_ids.get(int(block_idx), []))
            filtered = grouped[
                (grouped["mean_score"] >= float(config.min_mean_score))
                & (~grouped["unit"].isin(excluded))
            ].copy()
            if bool(config.shuffle_feature_candidates):
                units = filtered["unit"].astype(int).tolist()
                rng = random.Random(int(config.random_seed) + int(block_idx))
                rng.shuffle(units)
            else:
                units = (
                    filtered.sort_values(["mean_score", "max_score", "count"], ascending=[False, False, False])["unit"]
                    .astype(int)
                    .tolist()
                )

            selected_features: list[dict[str, Any]] = []
            skipped_insufficient: list[dict[str, Any]] = []
            used_token_uids: set[str] = set()
            sid_to_path_cache: dict[int, str] = {}

            for unit in units:
                feature_rows = frame[frame["unit"] == unit].sort_values("score", ascending=False)
                feature_stats = filtered[filtered["unit"] == unit].iloc[0]
                sample_ids = [int(sid) for sid in feature_rows["sample_id"].astype(int).unique().tolist()]
                missing = [sid for sid in sample_ids if sid not in sid_to_path_cache]
                if missing:
                    sid_to_path_cache.update(runtime.lookup_paths(missing))

                accepted: list[dict[str, Any]] = []
                for row in feature_rows.itertuples(index=False):
                    tok_idx = runtime.row_x_to_token_idx(int(row.x))
                    if tok_idx < 0 or tok_idx >= config.n_patches:
                        continue
                    img_path = sid_to_path_cache.get(int(row.sample_id), "")
                    if not img_path:
                        continue
                    candidate_uid = f"block_{int(block_idx)}/sample_{int(row.sample_id)}/tok_{tok_idx}"
                    if candidate_uid in used_token_uids:
                        continue
                    validation = runtime.validate_feature_token(
                        img_path,
                        int(block_idx),
                        int(unit),
                        tok_idx,
                        float(row.score),
                    )
                    if validation is None:
                        continue
                    accepted.append(
                        token_record_from_row(
                            block_idx,
                            int(unit),
                            row,
                            img_path,
                            validation,
                            token_idx=tok_idx,
                        )
                    )
                    if len(accepted) >= config.train_examples_per_feature + config.holdout_examples_per_feature:
                        break

                if len(accepted) < config.train_examples_per_feature + config.holdout_examples_per_feature:
                    skipped_insufficient.append(
                        {
                            "feature_id": int(unit),
                            "accepted": len(accepted),
                            "required": config.train_examples_per_feature + config.holdout_examples_per_feature,
                        }
                    )
                    continue

                train_rows = accepted[: config.train_examples_per_feature]
                holdout_rows = accepted[
                    config.train_examples_per_feature : config.train_examples_per_feature
                    + config.holdout_examples_per_feature
                ]
                for row in train_rows:
                    row["split"] = "train"
                for row in holdout_rows:
                    row["split"] = "holdout"
                used_token_uids.update(row["token_uid"] for row in accepted)
                selected_features.append(
                    {
                        "feature_key": feature_key(block_idx, int(unit)),
                        "block_idx": int(block_idx),
                        "feature_id": int(unit),
                        "selection_stats": {
                            "count": int(feature_stats["count"]),
                            "mean_score": float(feature_stats["mean_score"]),
                            "max_score": float(feature_stats["max_score"]),
                        },
                        "train": train_rows,
                        "holdout": holdout_rows,
                    }
                )
                if len(selected_features) == config.features_per_block:
                    break

            if len(selected_features) != config.features_per_block:
                diagnostic_path = config.diagnostics_root / f"block_{int(block_idx)}_selection_failure.json"
                write_json(
                    diagnostic_path,
                    {
                        "block_idx": int(block_idx),
                        "filtered_candidate_count": int(len(filtered)),
                        "selected_feature_count": int(len(selected_features)),
                        "required_feature_count": int(config.features_per_block),
                        "skipped_insufficient": skipped_insufficient,
                    },
                )
                raise RuntimeError(
                    f"block {block_idx} failed to reach {config.features_per_block} features; "
                    f"selected {len(selected_features)}"
                )

            payload["blocks"][str(block_idx)] = {
                "block_idx": int(block_idx),
                "candidate_pool_size": int(len(filtered)),
                "selected_feature_count": int(len(selected_features)),
                "unique_token_count": int(len(used_token_uids)),
                "skipped_insufficient_count": int(len(skipped_insufficient)),
                "features": selected_features,
            }
    finally:
        runtime.close()

    payload["diagnostics"] = _validate_payload_shape(config, payload)
    write_json(config.feature_bank_json, payload)
    return payload


def load_feature_bank(config: EvalConfig) -> dict[str, Any]:
    return read_json(config.feature_bank_json)


def save_feature_bank(config: EvalConfig, payload: dict[str, Any]) -> None:
    payload["diagnostics"] = _validate_payload_shape(config, payload)
    write_json(config.feature_bank_json, payload)


def all_block_features(payload: dict[str, Any], block_idx: int) -> list[dict[str, Any]]:
    return list(payload["blocks"][str(int(block_idx))]["features"])


def feature_id_order(payload: dict[str, Any], block_idx: int) -> list[int]:
    return [int(feature["feature_id"]) for feature in all_block_features(payload, block_idx)]


def gather_all_instances(payload: dict[str, Any], block_idx: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in all_block_features(payload, block_idx):
        rows.extend(feature["train"])
        rows.extend(feature["holdout"])
    return rows


def gather_unique_instances(payload: dict[str, Any], block_idx: int) -> list[dict[str, Any]]:
    by_uid: dict[str, dict[str, Any]] = {}
    for row in gather_all_instances(payload, block_idx):
        by_uid.setdefault(str(row["token_uid"]), row)
    return list(by_uid.values())


def gather_unique_holdout_tokens(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for block_idx in payload["blocks"]:
        for feature in payload["blocks"][block_idx]["features"]:
            for row in feature["holdout"]:
                out.setdefault(str(row["token_uid"]), row)
    return out
