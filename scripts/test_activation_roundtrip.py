#!/usr/bin/env python3
"""
Activation Roundtrip Verification.

Given an existing indexing output (deciles + offline_meta), verify that:
1. sample_id → offline_meta → video name + frame sequence
2. Load frames → build BVD → model forward → extract activation at (y, x)
3. SAE encode → check that the recorded score matches for the given unit

Usage:
    python scripts/test_activation_roundtrip.py \
        --config configs/sam2_sav_feature_index.yaml \
        --index-dir outputs/sae_index_ra-ar \
        --layer "model.sam_mask_decoder.transformer.layers.1@1" \
        --num-checks 5 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.indexing.decile_parquet_ledger import DecileParquetLedger
from src.core.indexing.registry_utils import load_obj, sanitize_layer_name
from src.packs.sam2.offline.offline_meta_ledger import OfflineMetaParquetLedger
from src.packs.sam2.offline.bvd_builders import (
    apply_indexing_transforms,
    build_vos_datapoint,
    frames_chw_from_datapoint,
    load_frames_from_disk,
    make_single_bvd_no_prompt,
    make_single_bvd_with_prompt,
)

logger = logging.getLogger("roundtrip")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def _load_sae(cfg: Dict[str, Any], layer: str, device: torch.device) -> torch.nn.Module:
    """Load SAE for a layer."""
    sae_root = Path(cfg["sae"]["output"]["save_path"])
    sae_type_fallback = cfg["sae"].get("sae_type", "batch-topk")
    ldir = sae_root / sanitize_layer_name(layer)
    ckpts = sorted(ldir.glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No SAE checkpoint for {layer} under {ldir}")
    pkg = torch.load(ckpts[-1], map_location="cpu")
    act_size = int(pkg.get("act_size", 0))
    sae_cfg = pkg.get("sae_config", {}) or {}
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    create_sae = load_obj(cfg["sae"]["factory"])
    sae_type = sae_cfg.get("sae_type") or sae_type_fallback
    sae = create_sae(sae_type, sae_cfg)
    sae.load_state_dict(pkg.get("sae_state", {}), strict=False)
    sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae


def _build_bvd(
    meta_ledger: OfflineMetaParquetLedger,
    ds_cfg: Dict[str, Any],
    sample_id: int,
    frame_idx: int,
    prompt_id: int,
    uid: int,
):
    """Build a BVD from offline_meta for a given sample_id."""
    tbl_sample = meta_ledger.find_sample(sample_id)
    if tbl_sample.num_rows == 0:
        raise RuntimeError(f"sample_id={sample_id} not found in offline_meta")
    d = tbl_sample.to_pydict()
    name = d["name"][0]
    seq_full = [int(v) for v in d["seq_full"][0]]
    dict_key = d.get("dict_key", ["unknown"])[0]
    prompt_sets = d.get("prompt_sets", [[]])[0]

    # Load frames
    img_root = Path(ds_cfg["img_folder"])
    pil_frames = load_frames_from_disk(img_root, name, seq_full)
    datapoint = build_vos_datapoint(pil_frames, seq_full, video_id=name)
    resized = apply_indexing_transforms(datapoint, target_res=int(ds_cfg.get("resize", 1024)))
    frames = frames_chw_from_datapoint(resized)

    # Determine prompt info
    if prompt_id >= 0 and prompt_sets:
        prompt_map = {int(e["frame_idx"]): int(e["prompt_id"]) for e in prompt_sets}
        t_prompt = frame_idx
        if t_prompt not in prompt_map and prompt_map:
            t_prompt = sorted(prompt_map.keys())[0]
        actual_pid = prompt_map.get(t_prompt, prompt_id)
        tbl_prompts = meta_ledger.find_prompts(sample_id, actual_pid, t_prompt)
        bvd, uid_map = make_single_bvd_with_prompt(
            frames, name, seq_full, sample_id, actual_pid,
            tbl_prompts, t_prompt, dict_key,
        )
    else:
        bvd, uid_map = make_single_bvd_no_prompt(
            frames, name, seq_full, sample_id, dict_key,
            prompt_id_value=-1,
        )

    lane_idx = uid_map.get(uid, 0)
    return bvd, lane_idx, name, seq_full


def main():
    parser = argparse.ArgumentParser(description="Activation roundtrip verification")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--num-checks", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--atol", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    layer = args.layer

    # 1. Load decile ledger and get top rows
    logger.info("Loading decile ledger from %s", args.index_dir)
    ledger = DecileParquetLedger(args.index_dir)
    tbl = ledger.topn_for(layer=layer, unit=None, decile=0, n=10000)
    if tbl.num_rows == 0:
        logger.error("No rows found in decile ledger for layer %s", layer)
        return
    logger.info("Decile ledger: %d rows for layer %s", tbl.num_rows, layer)

    rows = tbl.to_pydict()
    # Get unique sample_ids and pick some to check
    unique_sids = list(set(rows["sample_id"]))
    import random
    random.seed(42)
    check_sids = random.sample(unique_sids, min(args.num_checks, len(unique_sids)))

    # 2. Load offline_meta
    meta_root = cfg["indexing"]["offline_meta_root"]
    logger.info("Loading offline_meta from %s", meta_root)
    meta_ledger = OfflineMetaParquetLedger(meta_root)

    # 3. Load model
    logger.info("Loading model...")
    model_loader = load_obj(cfg["model"]["loader"])
    model = model_loader(cfg["model"], device, logger)
    model.eval()

    # 4. Load SAE
    logger.info("Loading SAE for %s", layer)
    sae = _load_sae(cfg, layer, device)

    # 5. Set up activation capture hook
    from src.core.hooks.spec import parse_spec, canonical_name
    from src.packs.sam2.models.adapters import SAM2EvalAdapter

    adapter = SAM2EvalAdapter(model)

    # 6. Run checks
    total_checks = 0
    total_pass = 0
    total_fail = 0
    results = []

    for check_i, sid in enumerate(check_sids):
        # Find rows for this sample_id
        matching = [(i, rows["unit"][i], rows["score"][i], rows["frame_idx"][i],
                      rows["y"][i], rows["x"][i], rows["prompt_id"][i], rows["uid"][i])
                     for i in range(tbl.num_rows) if rows["sample_id"][i] == sid]

        if not matching:
            continue

        # Pick one row
        idx, unit, expected_score, frame_idx, y, x, prompt_id, uid = matching[0]

        logger.info("\n--- Check %d/%d ---", check_i + 1, len(check_sids))
        logger.info("  sample_id=%d, unit=%d, expected_score=%.4f", sid, unit, expected_score)
        logger.info("  frame_idx=%d, y=%d, x=%d, prompt_id=%d, uid=%d", frame_idx, y, x, prompt_id, uid)

        try:
            bvd, lane_idx, name, seq_full = _build_bvd(
                meta_ledger, cfg["dataset"], sid, frame_idx, prompt_id, uid,
            )
            logger.info("  video=%s, seq=%s, lane=%d", name, seq_full, lane_idx)

            # Forward through model and capture activation at the target layer
            batch_on_dev = adapter.bvd_to_device(bvd, device)

            captured_acts = {}
            def make_hook(layer_name):
                def hook_fn(module, input, output):
                    captured_acts[layer_name] = output
                return hook_fn

            # Find the module for the layer
            parsed = parse_spec(layer)
            base_name = parsed.base_with_branch if hasattr(parsed, 'base_with_branch') else layer.split("::")[0]

            # Resolve module from the layer name
            # Handle @branch syntax: model.X@Y means output branch Y of module X
            parts = base_name.split("@")
            module_path = parts[0]
            branch = int(parts[1]) if len(parts) > 1 else None

            # Navigate to the module
            target_module = model
            for part in module_path.split("."):
                if part.isdigit():
                    target_module = target_module[int(part)]
                else:
                    target_module = getattr(target_module, part)

            hook_handle = target_module.register_forward_hook(make_hook(layer))

            with torch.no_grad():
                # Record caches (SAM2 needs this for prompt encoding)
                with adapter.clicks_cache("record"), adapter.prompt_inputs_cache("record"):
                    model(batch_on_dev)

            hook_handle.remove()

            if layer not in captured_acts:
                logger.warning("  SKIP: layer %s not captured", layer)
                continue

            act_tensor = captured_acts[layer]
            if isinstance(act_tensor, tuple):
                if branch is not None:
                    act_tensor = act_tensor[branch]
                else:
                    act_tensor = act_tensor[0]

            # The activation tensor shape depends on the layer:
            # For transformer layers: could be (B, N_tokens, D) or (N_tokens, B, D)
            logger.info("  Captured activation shape: %s", list(act_tensor.shape))

            # Flatten to find the right token by position (x index)
            # x is the token index within the flattened spatial dimension
            # For sam_mask_decoder.transformer.layers: shape is typically (N, B, D) where N=num_tokens
            act_flat = act_tensor.detach().float()
            if act_flat.dim() == 3:
                # Try (N, B, D) format — select lane_idx (batch dim)
                if act_flat.shape[1] == 1:
                    act_flat = act_flat[:, 0, :]  # (N, D)
                else:
                    act_flat = act_flat[:, lane_idx, :]  # (N, D) for this lane

            # x is the token position
            if x >= 0 and x < act_flat.shape[0]:
                token_act = act_flat[x].unsqueeze(0).to(device)  # (1, D)
            else:
                logger.warning("  SKIP: x=%d out of range (shape=%s)", x, list(act_flat.shape))
                continue

            # Encode through SAE
            with torch.no_grad():
                sae_codes = sae.encode(token_act).float().cpu().squeeze(0)  # (dict_size,)

            actual_score = float(sae_codes[unit].item())
            diff = abs(actual_score - expected_score)
            passed = diff <= args.atol

            total_checks += 1
            if passed:
                total_pass += 1
                logger.info("  PASS: actual=%.4f, expected=%.4f, diff=%.6f", actual_score, expected_score, diff)
            else:
                total_fail += 1
                logger.info("  FAIL: actual=%.4f, expected=%.4f, diff=%.6f (> atol=%.4f)",
                           actual_score, expected_score, diff, args.atol)

            results.append({
                "sample_id": sid, "unit": unit, "name": name,
                "expected": expected_score, "actual": actual_score, "diff": diff, "passed": passed,
            })

        except Exception as e:
            logger.error("  ERROR: %s", e, exc_info=True)
            total_fail += 1

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ACTIVATION ROUNDTRIP VERIFICATION")
    logger.info("=" * 60)
    logger.info("  Total checks: %d", total_checks)
    logger.info("  Passed:       %d", total_pass)
    logger.info("  Failed:       %d", total_fail)
    if results:
        diffs = [r["diff"] for r in results]
        logger.info("  Max diff:     %.6f", max(diffs))
        logger.info("  Mean diff:    %.6f", sum(diffs) / len(diffs))
    if total_fail > 0:
        logger.error("  RESULT: FAIL")
        sys.exit(1)
    else:
        logger.info("  RESULT: PASS")


if __name__ == "__main__":
    main()
