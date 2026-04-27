from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import EvalConfig
from .feature_bank import gather_unique_holdout_tokens, load_feature_bank, save_feature_bank
from .legacy import LegacyRuntime
from .prompts import load_baseline_prompt, pairwise_prompt
from .rendering import (
    save_activation_region_image,
    save_cosine_overlay_image,
    save_erf_heatmap_image,
    save_original_with_token_box,
    save_support_mask_image,
)
from .utils import feature_key, write_json, write_jsonl


def _evidence_erf_path(evidence: dict[str, Any]) -> str:
    path = evidence.get("token_erf_support_path") or evidence.get("token_erf_path") or evidence.get("token_erf90_path")
    if not path:
        raise KeyError("Missing token ERF image path in evidence payload")
    return str(path)


def export_baseline_label_requests(config: EvalConfig) -> dict[str, Any]:
    payload = load_feature_bank(config)
    runtime = LegacyRuntime(config)
    requests: list[dict[str, Any]] = []
    prompt = load_baseline_prompt(config)
    feature_count = sum(len(block_payload["features"]) for block_payload in payload["blocks"].values())
    feature_idx = 0
    try:
        for block_payload in payload["blocks"].values():
            for feature in block_payload["features"]:
                feature_idx += 1
                feature_dir = config.evidence_root / f"block_{feature['block_idx']}" / f"feature_{feature['feature_id']}"
                image_paths: list[str] = []
                for rank, row in enumerate(feature["train"]):
                    out_path = feature_dir / f"train_{rank:02d}_activation_region.png"
                    actmap = runtime.feature_activation_map(
                        row["image_path"],
                        int(row["block_idx"]),
                        int(row["feature_id"]),
                    )
                    save_activation_region_image(row["image_path"], actmap, out_path)
                    row["baseline_activation_region_path"] = str(out_path)
                    image_paths.append(str(out_path))
                requests.append(
                    {
                        "task": "baseline_label_generation",
                        "feature_key": feature["feature_key"],
                        "block_idx": int(feature["block_idx"]),
                        "feature_id": int(feature["feature_id"]),
                        "prompt_version": prompt["prompt_version"],
                        "system_prompt": prompt["system_prompt"],
                        "user_guidelines": prompt["user_guidelines"],
                        "image_paths": image_paths,
                        "expected_output_schema": {
                            "label": "short feature label",
                            "raw_explanation": "full free-form explanation string",
                            "provider": "external model identifier",
                            "prompt_version": prompt["prompt_version"],
                        },
                    }
                )
                print(
                    f"[label-requests {feature_idx:03d}/{feature_count:03d}] "
                    f"block={feature['block_idx']} feature={feature['feature_id']}",
                    flush=True,
                )
    finally:
        runtime.close()

    save_feature_bank(config, payload)
    bundle = {"config": config.to_dict(), "requests": requests}
    write_json(config.baseline_label_requests_json, bundle)
    return bundle


def build_token_evidence(config: EvalConfig) -> dict[str, Any]:
    payload = load_feature_bank(config)
    runtime = LegacyRuntime(config)
    evidence_manifest: dict[str, Any] = {"config": config.to_dict(), "tokens": {}}
    try:
        unique_tokens = gather_unique_holdout_tokens(payload)
        total = len(unique_tokens)
        for idx, (uid, row) in enumerate(unique_tokens.items(), start=1):
            block_idx = int(row["block_idx"])
            token_idx = int(row["target_patch_idx"])
            token_dir = config.evidence_root / f"block_{block_idx}" / "tokens" / uid.replace("/", "__")
            original_path = token_dir / "original_token_box.png"
            cosine_path = token_dir / "token_neighbor_cosine.png"
            erf_path = token_dir / "token_erf.png"
            erf_support_path = token_dir / "token_erf_support.png"
            erf_json = token_dir / "token_erf.json"

            save_original_with_token_box(row["image_path"], original_path, token_idx)
            cosine_map = runtime.token_cosine_map(row["image_path"], block_idx, token_idx)
            save_cosine_overlay_image(row["image_path"], cosine_map, cosine_path, token_idx=token_idx)
            erf = runtime.cautious_token_erf(row["image_path"], block_idx, token_idx)
            save_erf_heatmap_image(
                row["image_path"],
                erf["prob_scores"],
                erf_path,
                token_idx=token_idx,
            )
            save_support_mask_image(
                row["image_path"],
                erf["support_indices"],
                erf_support_path,
                token_idx=token_idx,
            )
            write_json(erf_json, erf)

            evidence_manifest["tokens"][uid] = {
                "token_uid": uid,
                "block_idx": block_idx,
                "sample_id": int(row["sample_id"]),
                "target_patch_idx": token_idx,
                "image_path": row["image_path"],
                "original_with_token_box_path": str(original_path),
                "token_neighbor_cosine_path": str(cosine_path),
                "token_erf_path": str(erf_support_path),
                "token_erf_heatmap_path": str(erf_path),
                "token_erf_support_path": str(erf_support_path),
                "token_erf_json": str(erf_json),
                "token_erf_support_threshold": float(erf["support_threshold"]),
            }
            print(
                f"[token-evidence {idx:03d}/{total:03d}] block={block_idx} "
                f"sample={row['sample_id']} tok={token_idx} support={erf['support_size']} "
                f"thr={erf['support_threshold']:.2f}",
                flush=True,
            )

        for block_payload in payload["blocks"].values():
            for feature in block_payload["features"]:
                for row in feature["holdout"]:
                    row["token_evidence"] = evidence_manifest["tokens"][str(row["token_uid"])]
    finally:
        runtime.close()

    save_feature_bank(config, payload)
    write_json(config.token_evidence_json, evidence_manifest)
    return evidence_manifest


def export_pairwise_requests(config: EvalConfig) -> dict[str, Any]:
    payload = load_feature_bank(config)
    labels = write_safe_load_labels(config)
    evidence_payload = build_token_evidence(config) if not config.token_evidence_json.exists() else None
    del evidence_payload
    pair_prompt = pairwise_prompt(config)
    rows: list[dict[str, Any]] = []
    for block_key, block_payload in payload["blocks"].items():
        block_idx = int(block_key)
        candidates = [
            {
                "feature_id": int(feature["feature_id"]),
                "feature_key": feature["feature_key"],
                "label": labels[feature["feature_key"]]["label"],
            }
            for feature in block_payload["features"]
        ]
        for feature in block_payload["features"]:
            for row in feature["holdout"]:
                evidence = row.get("token_evidence")
                if not evidence:
                    raise RuntimeError(f"Missing token evidence for {row['token_uid']}")
                for candidate in candidates:
                    rows.append(
                        {
                            "task": "pairwise_token_feature_judgment",
                            "request_id": f"{row['token_uid']}::{candidate['feature_id']}",
                            "block_idx": block_idx,
                            "token_uid": row["token_uid"],
                            "feature_id": candidate["feature_id"],
                            "feature_key": candidate["feature_key"],
                            "candidate_label": candidate["label"],
                            "prompt_version": pair_prompt["prompt_version"],
                            "system_prompt": pair_prompt["system_prompt"],
                            "user_guidelines": pair_prompt["user_guidelines"],
                            "evidence_paths": {
                                "original_with_token_box_path": evidence["original_with_token_box_path"],
                                "token_erf_path": _evidence_erf_path(evidence),
                                "token_neighbor_cosine_path": evidence["token_neighbor_cosine_path"],
                            },
                            "expected_output_schema": {
                                "block_idx": block_idx,
                                "token_uid": row["token_uid"],
                                "feature_id": candidate["feature_id"],
                                "score_0_100": "integer 0..100",
                                "decision": "yes|no",
                                "provider": "external model identifier",
                                "prompt_version": pair_prompt["prompt_version"],
                            },
                        }
                    )
    write_jsonl(config.pairwise_requests_jsonl, rows)
    return {"config": config.to_dict(), "n_requests": len(rows)}


def write_safe_load_labels(config: EvalConfig) -> dict[str, Any]:
    if not config.baseline_labels_json.exists():
        raise FileNotFoundError(
            f"Expected external label import at {config.baseline_labels_json}. "
            f"Run export-label-requests first, then import baseline_labels.json."
        )
    import json

    return json.loads(config.baseline_labels_json.read_text())
