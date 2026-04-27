from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .config import EvalConfig
from .feature_bank import feature_id_order, gather_unique_instances, load_feature_bank
from .legacy import LegacyRuntime
from .utils import read_jsonl, write_json


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    ranked = y_true[order]
    cum_pos = np.cumsum(ranked)
    ranks = np.arange(1, len(ranked) + 1)
    precisions = cum_pos[ranked == 1] / ranks[ranked == 1]
    return float(precisions.mean()) if precisions.size else float("nan")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(y_true.sum())
    n_total = int(y_true.size)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(y_score)
    rank_sum_pos = float(ranks[y_true == 1].sum())
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def f1_accuracy_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = (np.asarray(y_score, dtype=np.float64).reshape(-1) >= float(threshold)).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "f1_at_0.5": float(f1),
        "accuracy_at_0.5": float(accuracy),
        "precision_at_0.5": float(precision),
        "recall_at_0.5": float(recall),
    }


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if int(y_true.sum()) == 0:
        return float("nan")
    k = min(int(k), int(y_true.size))
    order = np.argsort(-y_score, kind="mergesort")[:k]
    rel = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(((2**rel - 1) * discounts).sum())
    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(((2**ideal - 1) * discounts).sum())
    if idcg <= 0.0:
        return float("nan")
    return dcg / idcg


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")[: min(int(k), len(y_score))]
    return float(y_true[order].sum() / positives)


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _summarize_feature_to_token(
    y_true: np.ndarray,
    y_score: np.ndarray,
    decision_threshold: float,
) -> dict[str, Any]:
    per_feature_ap = []
    per_feature_auc = []
    per_feature_f1 = []
    for col in range(y_true.shape[1]):
        yt = y_true[:, col]
        ys = y_score[:, col]
        per_feature_ap.append(average_precision_binary(yt, ys))
        per_feature_auc.append(roc_auc_binary(yt, ys))
        per_feature_f1.append(f1_accuracy_at_threshold(yt, ys, threshold=decision_threshold)["f1_at_0.5"])

    flat_true = y_true.reshape(-1)
    flat_score = y_score.reshape(-1)
    summary = {
        "macro_auprc": _nanmean(per_feature_ap),
        "macro_auroc": _nanmean(per_feature_auc),
        "micro_auprc": average_precision_binary(flat_true, flat_score),
        "micro_auroc": roc_auc_binary(flat_true, flat_score),
        "macro_f1_at_0.5": _nanmean(per_feature_f1),
    }
    summary.update(f1_accuracy_at_threshold(flat_true, flat_score, threshold=decision_threshold))
    return summary


def _summarize_token_to_feature(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    per_token_ap = []
    per_token_ndcg = []
    per_token_r1 = []
    per_token_r5 = []
    per_token_r10 = []
    for row in range(y_true.shape[0]):
        yt = y_true[row, :]
        ys = y_score[row, :]
        per_token_ap.append(average_precision_binary(yt, ys))
        per_token_ndcg.append(ndcg_at_k(yt, ys, k=y_true.shape[1]))
        per_token_r1.append(recall_at_k(yt, ys, 1))
        per_token_r5.append(recall_at_k(yt, ys, 5))
        per_token_r10.append(recall_at_k(yt, ys, 10))
    return {
        "mAP": _nanmean(per_token_ap),
        "nDCG@64": _nanmean(per_token_ndcg),
        "Recall@1": _nanmean(per_token_r1),
        "Recall@5": _nanmean(per_token_r5),
        "Recall@10": _nanmean(per_token_r10),
    }


def _normalize_score(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(max(0.0, min(100.0, float(value))) / 100.0)


def _parse_pairwise_scores(
    config: EvalConfig,
    feature_bank: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    rows = read_jsonl(config.pairwise_scores_jsonl)
    score_map: dict[tuple[int, str, int], float] = {}
    for row in rows:
        key = (int(row["block_idx"]), str(row["token_uid"]), int(row["feature_id"]))
        score_map[key] = _normalize_score(row.get("score_0_100"))

    matrices: dict[int, np.ndarray] = {}
    token_orders: dict[int, list[str]] = {}
    for block_idx in config.blocks:
        block_payload = feature_bank["blocks"][str(block_idx)]
        feature_ids = feature_id_order(feature_bank, block_idx)
        holdout_rows: list[dict[str, Any]] = []
        for feature in block_payload["features"]:
            holdout_rows.extend(feature["holdout"])
        token_orders[block_idx] = [str(row["token_uid"]) for row in holdout_rows]
        matrix = np.full((len(holdout_rows), len(feature_ids)), np.nan, dtype=np.float32)
        for i, row in enumerate(holdout_rows):
            for j, feature_id in enumerate(feature_ids):
                key = (int(block_idx), str(row["token_uid"]), int(feature_id))
                if key not in score_map:
                    raise RuntimeError(f"Missing pairwise score for {key}")
                matrix[i, j] = score_map[key]
        matrices[block_idx] = matrix
    return matrices, token_orders


def build_ground_truth(config: EvalConfig) -> dict[str, Any]:
    feature_bank = load_feature_bank(config)
    runtime = LegacyRuntime(config)
    summary: dict[str, Any] = {"config": config.to_dict(), "blocks": {}}
    try:
        for block_idx in config.blocks:
            feature_ids = feature_id_order(feature_bank, block_idx)
            instances = gather_unique_instances(feature_bank, block_idx)
            activations = np.zeros((len(instances), len(feature_ids)), dtype=np.float32)
            splits: list[str] = []
            token_uids: list[str] = []
            for i, row in enumerate(instances):
                activations[i, :] = runtime.feature_vector_at_token(
                    row["image_path"],
                    int(block_idx),
                    int(row["target_patch_idx"]),
                    feature_ids,
                )
                token_uids.append(str(row["token_uid"]))
                splits.append(str(row["split"]))
                if (i + 1) % 32 == 0 or (i + 1) == len(instances):
                    print(
                        f"[ground-truth block={block_idx}] {i + 1:03d}/{len(instances):03d}",
                        flush=True,
                    )
            thresholds = np.percentile(activations, float(config.p99_percentile), axis=0)
            holdout_mask = np.asarray([split == "holdout" for split in splits], dtype=bool)
            holdout_acts = activations[holdout_mask, :]
            holdout_truth = (holdout_acts >= thresholds.reshape(1, -1)).astype(np.int64)
            threshold_sweep: dict[str, Any] = {}
            for percentile in (90, 95, 97, 99):
                sweep_thr = np.percentile(activations, float(percentile), axis=0)
                sweep_truth = (holdout_acts >= sweep_thr.reshape(1, -1)).astype(np.int64)
                sweep_counts = sweep_truth.sum(axis=0).astype(int)
                threshold_sweep[str(percentile)] = {
                    "positive_density_holdout": float(sweep_truth.mean()) if sweep_truth.size else 0.0,
                    "min_positive_count_per_feature": int(sweep_counts.min()) if sweep_counts.size else 0,
                    "median_positive_count_per_feature": int(np.median(sweep_counts)) if sweep_counts.size else 0,
                    "max_positive_count_per_feature": int(sweep_counts.max()) if sweep_counts.size else 0,
                }
            summary["blocks"][str(block_idx)] = {
                "feature_ids": [int(v) for v in feature_ids],
                "all_token_uids": token_uids,
                "holdout_token_uids": [token_uids[idx] for idx, is_holdout in enumerate(holdout_mask) if is_holdout],
                "thresholds_p99": thresholds.astype(float).tolist(),
                "activation_matrix_all": activations.astype(float).tolist(),
                "activation_matrix_holdout": holdout_acts.astype(float).tolist(),
                "ground_truth_holdout": holdout_truth.astype(int).tolist(),
                "positive_density_holdout": float(holdout_truth.mean()) if holdout_truth.size else 0.0,
                "positive_count_per_feature_holdout": holdout_truth.sum(axis=0).astype(int).tolist(),
                "threshold_sweep": threshold_sweep,
                "degenerate_p99_warning": bool(holdout_truth.sum() <= max(1, len(feature_ids) // 8)),
            }
    finally:
        runtime.close()

    write_json(config.ground_truth_json, summary)
    return summary


def compute_metrics_summary(config: EvalConfig) -> dict[str, Any]:
    if not config.pairwise_scores_jsonl.exists():
        raise FileNotFoundError(
            f"Expected external judge import at {config.pairwise_scores_jsonl}. "
            f"Run export-pairwise-requests first, then import pairwise_scores.jsonl."
        )
    feature_bank = load_feature_bank(config)
    ground_truth = build_ground_truth(config) if not config.ground_truth_json.exists() else None
    if ground_truth is None:
        import json

        ground_truth = json.loads(config.ground_truth_json.read_text())
    score_matrices, _ = _parse_pairwise_scores(config, feature_bank)

    block_summaries: dict[str, Any] = {}
    overall_ft_true = []
    overall_ft_score = []
    overall_tf_true = []
    overall_tf_score = []
    diagnostics: dict[str, Any] = {"sanity": {}, "builder": feature_bank.get("diagnostics", {})}

    for block_idx in config.blocks:
        block_gt = ground_truth["blocks"][str(block_idx)]
        y_true = np.asarray(block_gt["ground_truth_holdout"], dtype=np.int64)
        y_score = score_matrices[int(block_idx)]
        overall_ft_true.append(y_true)
        overall_ft_score.append(y_score)
        overall_tf_true.append(y_true)
        overall_tf_score.append(y_score)
        block_summaries[str(block_idx)] = {
            "feature_to_token": _summarize_feature_to_token(
                y_true,
                y_score,
                decision_threshold=float(config.decision_threshold),
            ),
            "token_to_feature": _summarize_token_to_feature(y_true, y_score),
            "positive_density_holdout": float(block_gt["positive_density_holdout"]),
            "positive_count_per_feature_holdout": block_gt["positive_count_per_feature_holdout"],
            "threshold_sweep": block_gt.get("threshold_sweep", {}),
            "degenerate_p99_warning": bool(block_gt.get("degenerate_p99_warning", False)),
            "n_holdout_tokens": int(y_true.shape[0]),
            "n_candidate_features": int(y_true.shape[1]),
        }

    y_true_all = np.concatenate(overall_ft_true, axis=0)
    y_score_all = np.concatenate(overall_ft_score, axis=0)
    random_rng = np.random.default_rng(seed=0)
    random_scores = random_rng.random(y_score_all.shape, dtype=np.float32)
    oracle_scores = y_true_all.astype(np.float32)
    shuffled_scores = y_score_all[:, random_rng.permutation(y_score_all.shape[1])]

    overall = {
        "feature_to_token": _summarize_feature_to_token(
            y_true_all,
            y_score_all,
            decision_threshold=float(config.decision_threshold),
        ),
        "token_to_feature": _summarize_token_to_feature(y_true_all, y_score_all),
    }
    diagnostics["sanity"] = {
        "random_scores": {
            "feature_to_token": _summarize_feature_to_token(
                y_true_all,
                random_scores,
                decision_threshold=float(config.decision_threshold),
            ),
            "token_to_feature": _summarize_token_to_feature(y_true_all, random_scores),
        },
        "shuffled_columns": {
            "feature_to_token": _summarize_feature_to_token(
                y_true_all,
                shuffled_scores,
                decision_threshold=float(config.decision_threshold),
            ),
            "token_to_feature": _summarize_token_to_feature(y_true_all, shuffled_scores),
        },
        "oracle_scores": {
            "feature_to_token": _summarize_feature_to_token(
                y_true_all,
                oracle_scores,
                decision_threshold=float(config.decision_threshold),
            ),
            "token_to_feature": _summarize_token_to_feature(y_true_all, oracle_scores),
        },
        "oracle_beats_observed_macro_auprc": bool(
            diagnostics["sanity"].get("oracle_scores", {})
            if False
            else _summarize_feature_to_token(
                y_true_all,
                oracle_scores,
                decision_threshold=float(config.decision_threshold),
            )["macro_auprc"]
            >= overall["feature_to_token"]["macro_auprc"]
        ),
    }

    leakage_flags = defaultdict(list)
    for block_idx in config.blocks:
        for feature in feature_bank["blocks"][str(block_idx)]["features"]:
            train = {row["token_uid"] for row in feature["train"]}
            holdout = {row["token_uid"] for row in feature["holdout"]}
            overlap = sorted(train & holdout)
            if overlap:
                leakage_flags[str(block_idx)].extend(overlap)
    diagnostics["leakage_warning"] = {block: vals for block, vals in leakage_flags.items() if vals}
    diagnostics["degenerate_ground_truth_warning"] = any(
        bool(block_summaries[str(block_idx)]["degenerate_p99_warning"]) for block_idx in config.blocks
    )

    summary = {
        "config": config.to_dict(),
        "block_summaries": block_summaries,
        "overall": overall,
        "diagnostics": diagnostics,
    }
    write_json(config.metrics_summary_json, summary)
    return summary
