from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import EvalConfig
from .feature_bank import load_feature_bank
from .legacy import LegacyRuntime
from .metrics import (
    average_precision_binary,
    f1_accuracy_at_threshold,
    roc_auc_binary,
)
from .prompts import axis1_c_conditioned_prompt
from .rendering import (
    save_cosine_overlay_image,
    save_erf_heatmap_image,
    save_original_with_token_box,
    save_support_mask_image,
)
from .utils import read_jsonl, token_uid, write_json, write_jsonl


def _axis1_item_id(feature_key: str, sample_id: int, token_idx: int) -> str:
    return f"{feature_key}::sample_{int(sample_id)}::tok_{int(token_idx)}"


def _evidence_erf_path(payload: dict[str, Any]) -> str:
    path = payload.get("token_erf_support_path") or payload.get("token_erf_path") or payload.get("token_erf90_path")
    if not path:
        raise KeyError("Missing token ERF image path in axis1 evidence payload")
    return str(path)


def build_axis1_c_conditioned_dataset(config: EvalConfig) -> dict[str, Any]:
    feature_bank = load_feature_bank(config)
    runtime = LegacyRuntime(config)
    items: list[dict[str, Any]] = []
    feature_summaries: dict[str, Any] = {}
    try:
        all_features = [
            feature
            for block_payload in feature_bank["blocks"].values()
            for feature in block_payload["features"]
        ]
        total_features = len(all_features)
        for feature_idx, feature in enumerate(all_features, start=1):
            feature_key = str(feature["feature_key"])
            block_idx = int(feature["block_idx"])
            feature_id = int(feature["feature_id"])
            positive_rows = list(feature["holdout"])
            feature_items: list[dict[str, Any]] = []
            for holdout_row in positive_rows:
                image_path = str(holdout_row["image_path"])
                sample_id = int(holdout_row["sample_id"])
                target_idx = int(holdout_row["target_patch_idx"])
                actmap = runtime.feature_activation_map(image_path, block_idx, feature_id)
                target_act = float(actmap[target_idx])
                item_id = _axis1_item_id(feature_key, sample_id, target_idx)
                feature_items.append(
                    {
                        "axis1_item_id": item_id,
                        "feature_key": feature_key,
                        "block_idx": block_idx,
                        "feature_id": feature_id,
                        "sample_id": sample_id,
                        "image_path": image_path,
                        "token_uid": token_uid(block_idx, sample_id, target_idx),
                        "token_idx": target_idx,
                        "label": 1,
                        "feature_activation": target_act,
                        "target_token_activation": target_act,
                        "is_target_token": True,
                        "negative_strategy": None,
                    }
                )

                candidate_indices = [idx for idx in range(config.n_patches) if idx != target_idx]
                if config.axis1_negative_strategy == "lowest_activation":
                    candidate_indices = sorted(candidate_indices, key=lambda idx: (float(actmap[idx]), idx))
                else:
                    raise ValueError(
                        f"Unsupported axis1_negative_strategy={config.axis1_negative_strategy!r}"
                    )
                selected_negatives = candidate_indices[: int(config.axis1_negatives_per_image)]
                for neg_idx in selected_negatives:
                    neg_act = float(actmap[neg_idx])
                    feature_items.append(
                        {
                            "axis1_item_id": _axis1_item_id(feature_key, sample_id, neg_idx),
                            "feature_key": feature_key,
                            "block_idx": block_idx,
                            "feature_id": feature_id,
                            "sample_id": sample_id,
                            "image_path": image_path,
                            "token_uid": token_uid(block_idx, sample_id, neg_idx),
                            "token_idx": int(neg_idx),
                            "label": 0,
                            "feature_activation": neg_act,
                            "target_token_activation": target_act,
                            "is_target_token": False,
                            "negative_strategy": str(config.axis1_negative_strategy),
                        }
                    )
            items.extend(feature_items)
            n_pos = sum(int(item["label"]) for item in feature_items)
            n_neg = len(feature_items) - n_pos
            feature_summaries[feature_key] = {
                "feature_key": feature_key,
                "block_idx": block_idx,
                "feature_id": feature_id,
                "n_items": len(feature_items),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "holdout_sample_ids": [int(row["sample_id"]) for row in positive_rows],
                "negative_strategy": str(config.axis1_negative_strategy),
                "positive_mode": str(config.axis1_positive_mode),
            }
            print(
                f"[axis1-dataset {feature_idx:03d}/{total_features:03d}] "
                f"block={block_idx} feature={feature_id} items={len(feature_items)}",
                flush=True,
            )
    finally:
        runtime.close()

    payload = {
        "config": config.to_dict(),
        "items": items,
        "feature_summaries": feature_summaries,
    }
    write_json(config.axis1_dataset_json, payload)
    return payload


def _load_existing_token_evidence(config: EvalConfig) -> dict[str, dict[str, Any]]:
    if not config.token_evidence_json.exists():
        return {}
    payload = json.loads(config.token_evidence_json.read_text())
    valid: dict[str, dict[str, Any]] = {}
    for key, value in payload.get("tokens", {}).items():
        if "token_erf_path" not in value or "token_erf_json" not in value:
            continue
        if float(value.get("token_erf_support_threshold", -1.0)) != float(config.erf_recovery_threshold):
            continue
        valid[str(key)] = value
    return valid


def build_axis1_c_conditioned_evidence(config: EvalConfig) -> dict[str, Any]:
    if not config.axis1_dataset_json.exists():
        raise FileNotFoundError(
            f"Missing axis1 dataset at {config.axis1_dataset_json}. Run build-axis1-dataset first."
        )
    dataset = json.loads(config.axis1_dataset_json.read_text())
    existing = _load_existing_token_evidence(config)
    by_uid: dict[str, dict[str, Any]] = {}
    for item in dataset["items"]:
        by_uid.setdefault(str(item["token_uid"]), item)

    runtime = LegacyRuntime(config)
    manifest = {"config": config.to_dict(), "tokens": {}}
    try:
        total = len(by_uid)
        for idx, (uid, item) in enumerate(by_uid.items(), start=1):
            if uid in existing:
                manifest["tokens"][uid] = existing[uid]
                print(f"[axis1-evidence {idx:04d}/{total:04d}] reuse {uid}", flush=True)
                continue

            block_idx = int(item["block_idx"])
            token_idx = int(item["token_idx"])
            sample_id = int(item["sample_id"])
            image_path = str(item["image_path"])
            token_dir = (
                config.evidence_root / "axis1_c_conditioned" / f"block_{block_idx}" / "tokens" / uid.replace("/", "__")
            )
            original_path = token_dir / "original_token_box.png"
            cosine_path = token_dir / "token_neighbor_cosine.png"
            erf_path = token_dir / "token_erf.png"
            erf_support_path = token_dir / "token_erf_support.png"
            erf_json = token_dir / "token_erf.json"

            save_original_with_token_box(image_path, original_path, token_idx)
            cosine_map = runtime.token_cosine_map(image_path, block_idx, token_idx)
            save_cosine_overlay_image(image_path, cosine_map, cosine_path, token_idx=token_idx)
            erf = runtime.cautious_token_erf(image_path, block_idx, token_idx)
            save_erf_heatmap_image(
                image_path,
                erf["prob_scores"],
                erf_path,
                token_idx=token_idx,
            )
            save_support_mask_image(
                image_path,
                erf["support_indices"],
                erf_support_path,
                token_idx=token_idx,
            )
            write_json(erf_json, erf)
            manifest["tokens"][uid] = {
                "token_uid": uid,
                "block_idx": block_idx,
                "sample_id": sample_id,
                "target_patch_idx": token_idx,
                "image_path": image_path,
                "original_with_token_box_path": str(original_path),
                "token_neighbor_cosine_path": str(cosine_path),
                "token_erf_path": str(erf_support_path),
                "token_erf_heatmap_path": str(erf_path),
                "token_erf_support_path": str(erf_support_path),
                "token_erf_json": str(erf_json),
                "token_erf_support_threshold": float(erf["support_threshold"]),
            }
            print(
                f"[axis1-evidence {idx:04d}/{total:04d}] build block={block_idx} sample={sample_id} tok={token_idx}",
                flush=True,
            )
    finally:
        runtime.close()

    write_json(config.axis1_evidence_json, manifest)
    return manifest


def _load_baseline_labels(config: EvalConfig) -> dict[str, Any]:
    if not config.baseline_labels_json.exists():
        raise FileNotFoundError(
            f"Missing baseline labels at {config.baseline_labels_json}. "
            f"Import baseline_labels.json before exporting axis1 requests."
        )
    return json.loads(config.baseline_labels_json.read_text())


def export_axis1_c_conditioned_requests(config: EvalConfig) -> dict[str, Any]:
    if not config.axis1_dataset_json.exists():
        raise FileNotFoundError(
            f"Missing axis1 dataset at {config.axis1_dataset_json}. Run build-axis1-dataset first."
        )
    if not config.axis1_evidence_json.exists():
        raise FileNotFoundError(
            f"Missing axis1 evidence at {config.axis1_evidence_json}. Run build-axis1-evidence first."
        )
    dataset = json.loads(config.axis1_dataset_json.read_text())
    evidence = json.loads(config.axis1_evidence_json.read_text()).get("tokens", {})
    labels = _load_baseline_labels(config)
    prompt = axis1_c_conditioned_prompt(config)
    rows: list[dict[str, Any]] = []
    total = len(dataset["items"])
    for idx, item in enumerate(dataset["items"], start=1):
        feature_key = str(item["feature_key"])
        label_payload = labels.get(feature_key)
        if not label_payload:
            raise KeyError(f"Missing baseline label for {feature_key}")
        token_payload = evidence.get(str(item["token_uid"]))
        if not token_payload:
            raise KeyError(f"Missing axis1 evidence for {item['token_uid']}")
        rows.append(
            {
                "task": "axis1_c_conditioned_yes_no",
                "axis1_item_id": str(item["axis1_item_id"]),
                "feature_key": feature_key,
                "block_idx": int(item["block_idx"]),
                "feature_id": int(item["feature_id"]),
                "sample_id": int(item["sample_id"]),
                "token_uid": str(item["token_uid"]),
                "token_idx": int(item["token_idx"]),
                "candidate_label": str(label_payload["label"]),
                "prompt_version": prompt["prompt_version"],
                "system_prompt": prompt["system_prompt"],
                "user_guidelines": prompt["user_guidelines"],
                "evidence_paths": {
                    "original_with_token_box_path": token_payload["original_with_token_box_path"],
                    "token_erf_path": _evidence_erf_path(token_payload),
                    "token_neighbor_cosine_path": token_payload["token_neighbor_cosine_path"],
                },
                "expected_output_schema": {
                    "axis1_item_id": str(item["axis1_item_id"]),
                    "feature_key": feature_key,
                    "block_idx": int(item["block_idx"]),
                    "feature_id": int(item["feature_id"]),
                    "token_uid": str(item["token_uid"]),
                    "score_0_100": "integer 0..100",
                    "decision": "yes|no",
                    "provider": "external model identifier",
                    "prompt_version": prompt["prompt_version"],
                },
            }
        )
        if idx % 128 == 0 or idx == total:
            print(f"[axis1-requests {idx:04d}/{total:04d}]", flush=True)
    write_jsonl(config.axis1_requests_jsonl, rows)
    return {"config": config.to_dict(), "n_requests": len(rows)}


def compute_axis1_c_conditioned_metrics(config: EvalConfig) -> dict[str, Any]:
    if not config.axis1_dataset_json.exists():
        raise FileNotFoundError(
            f"Missing axis1 dataset at {config.axis1_dataset_json}. Run build-axis1-dataset first."
        )
    if not config.axis1_scores_jsonl.exists():
        raise FileNotFoundError(
            f"Missing axis1 scores at {config.axis1_scores_jsonl}. "
            f"Import external axis1 scores before computing metrics."
        )
    dataset = json.loads(config.axis1_dataset_json.read_text())
    score_rows = read_jsonl(config.axis1_scores_jsonl)
    score_map = {str(row["axis1_item_id"]): float(row["score_0_100"]) / 100.0 for row in score_rows}

    grouped_true: dict[str, list[int]] = defaultdict(list)
    grouped_score: dict[str, list[float]] = defaultdict(list)
    grouped_meta: dict[str, dict[str, Any]] = {}
    for item in dataset["items"]:
        item_id = str(item["axis1_item_id"])
        if item_id not in score_map:
            raise KeyError(f"Missing axis1 score for {item_id}")
        feature_key = str(item["feature_key"])
        grouped_true[feature_key].append(int(item["label"]))
        grouped_score[feature_key].append(float(score_map[item_id]))
        grouped_meta[feature_key] = {
            "block_idx": int(item["block_idx"]),
            "feature_id": int(item["feature_id"]),
        }

    per_feature: dict[str, Any] = {}
    macro_ap = []
    macro_auc = []
    macro_f1 = []
    all_true = []
    all_score = []
    for feature_key in sorted(grouped_true):
        y_true = np.asarray(grouped_true[feature_key], dtype=np.int64)
        y_score = np.asarray(grouped_score[feature_key], dtype=np.float32)
        ap = average_precision_binary(y_true, y_score)
        auc = roc_auc_binary(y_true, y_score)
        f1_stats = f1_accuracy_at_threshold(y_true, y_score, threshold=float(config.decision_threshold))
        per_feature[feature_key] = {
            **grouped_meta[feature_key],
            "n_items": int(y_true.size),
            "n_positive": int(y_true.sum()),
            "n_negative": int(y_true.size - y_true.sum()),
            "auprc": ap,
            "auroc": auc,
            **f1_stats,
        }
        macro_ap.append(ap)
        macro_auc.append(auc)
        macro_f1.append(f1_stats["f1_at_0.5"])
        all_true.append(y_true)
        all_score.append(y_score)

    flat_true = np.concatenate(all_true, axis=0) if all_true else np.empty((0,), dtype=np.int64)
    flat_score = np.concatenate(all_score, axis=0) if all_score else np.empty((0,), dtype=np.float32)
    overall = {
        "macro_auprc": float(np.nanmean(np.asarray(macro_ap, dtype=np.float64))) if macro_ap else float("nan"),
        "macro_auroc": float(np.nanmean(np.asarray(macro_auc, dtype=np.float64))) if macro_auc else float("nan"),
        "macro_f1_at_0.5": float(np.nanmean(np.asarray(macro_f1, dtype=np.float64))) if macro_f1 else float("nan"),
        "micro_auprc": average_precision_binary(flat_true, flat_score) if flat_true.size else float("nan"),
        "micro_auroc": roc_auc_binary(flat_true, flat_score) if flat_true.size else float("nan"),
        **(f1_accuracy_at_threshold(flat_true, flat_score, threshold=float(config.decision_threshold)) if flat_true.size else {}),
    }

    summary = {
        "config": config.to_dict(),
        "per_feature": per_feature,
        "overall": overall,
    }
    write_json(config.axis1_metrics_json, summary)
    return summary
