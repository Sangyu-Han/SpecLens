#!/usr/bin/env python3
"""
Verify parquet → model → SAE roundtrip.

Reads (unit, sample_id, score, y, x) from decile parquets,
looks up sample metadata from offline_meta,
re-runs model forward + SAE encode,
compares the stored score to the recomputed one.

Usage:
    python scripts/verify_parquet_roundtrip.py \
        --config configs/sam2_sav_feature_index_small_test.yaml \
        --num-checks 20 --atol 1e-3
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pyarrow.dataset as pds
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.packs.sam2.models.adapters import SAM2EvalAdapter

logger = logging.getLogger("verify_roundtrip")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--layer", type=str, default=None,
                        help="Layer to verify (default: first layer in config)")
    parser.add_argument("--num-checks", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Determine layer
    target_layer = args.layer or cfg["sae"]["layers"][0]
    parts = target_layer.split("@")
    module_path = parts[0]
    if len(parts) > 1:
        branch_str = parts[1]
        branch = int(branch_str) if branch_str.isdigit() else branch_str
    else:
        branch = None
    if module_path.startswith("model."):
        module_path = module_path[len("model."):]

    logger.info("Layer: %s (module=%s, branch=%s)", target_layer, module_path, branch)

    # ====================================================================
    # Load model + SAE
    # ====================================================================
    logger.info("Loading model...")
    model_loader = load_obj(cfg["model"]["loader"])
    model = model_loader(cfg["model"], device, logger)
    model.eval()

    logger.info("Loading SAE...")
    sae_root = Path(cfg["sae"]["output"]["save_path"])
    sae_type_fallback = cfg["sae"].get("sae_type", "batch-topk")
    ldir = sae_root / sanitize_layer_name(target_layer)
    ckpts = sorted(ldir.glob("*.pt"))
    if not ckpts:
        logger.error("No SAE checkpoint for %s under %s", target_layer, ldir)
        return
    pkg = torch.load(ckpts[-1], map_location="cpu")
    act_size = int(pkg.get("act_size", 0))
    sae_cfg = pkg.get("sae_config", {}) or {}
    sae_cfg.update({"act_size": act_size, "device": str(device)})
    create_sae = load_obj(cfg["sae"]["factory"])
    sae_type = sae_cfg.get("sae_type") or sae_type_fallback
    sae = create_sae(sae_type, sae_cfg)
    sae.load_state_dict(pkg.get("sae_state", {}), strict=False)
    sae.to(device).eval()
    logger.info("SAE: type=%s, act_size=%d", sae_type, act_size)

    # Navigate to target module
    target_module = model
    for part in module_path.split("."):
        if part.isdigit():
            target_module = target_module[int(part)]
        else:
            target_module = getattr(target_module, part)

    # ====================================================================
    # Read decile parquets — pick random rows to verify
    # ====================================================================
    out_dir = Path(cfg["indexing"]["out_dir"])
    decile_dir = out_dir / "deciles"
    logger.info("Reading decile parquets from %s", decile_dir)

    import pyarrow.compute as pc
    ds = pds.dataset(str(decile_dir), format="parquet", partitioning="hive")
    table = ds.to_table(filter=pc.field("layer") == target_layer)
    logger.info("Rows for layer %s: %d", target_layer, table.num_rows)

    if table.num_rows == 0:
        logger.error("No rows in parquet for layer %s!", target_layer)
        return

    # Sample random rows
    import random
    random.seed(42)
    n = min(args.num_checks, table.num_rows)
    indices = random.sample(range(table.num_rows), n)

    rows = table.to_pydict()
    checks = []
    for i in indices:
        checks.append({
            "unit": int(rows["unit"][i]),
            "sample_id": int(rows["sample_id"][i]),
            "score": float(rows["score"][i]),
            "frame_idx": int(rows["frame_idx"][i]),
            "y": int(rows["y"][i]),
            "x": int(rows["x"][i]),
            "uid": int(rows["uid"][i]) if "uid" in rows else -1,
            "prompt_id": int(rows["prompt_id"][i]) if "prompt_id" in rows else -1,
        })

    logger.info("Selected %d rows to verify", len(checks))
    for c in checks[:5]:
        logger.info("  unit=%d, sid=%d, score=%.4f, frame=%d, y=%d, x=%d",
                    c["unit"], c["sample_id"], c["score"], c["frame_idx"], c["y"], c["x"])

    # ====================================================================
    # Load offline_meta ledger
    # ====================================================================
    meta_root = cfg["indexing"]["offline_meta_root"]
    part_mod = int(cfg["indexing"].get("partition_modulus", 128))
    meta_ledger = OfflineMetaParquetLedger(meta_root, part_modulus=part_mod)
    logger.info("Loaded offline meta from %s", meta_root)

    # ====================================================================
    # Group by sample_id, re-run model, compare scores
    # ====================================================================
    from collections import defaultdict
    by_sample = defaultdict(list)
    for c in checks:
        by_sample[c["sample_id"]].append(c)

    adapter = SAM2EvalAdapter(model, device=device)
    img_root = Path(cfg["dataset"]["img_folder"])
    target_res = int(cfg["dataset"].get("resize", 1024))

    total_checks = 0
    total_pass = 0
    total_fail = 0
    max_diff = 0.0

    for sid, refs in by_sample.items():
        tbl = meta_ledger.find_sample(sid)
        if tbl.num_rows == 0:
            logger.error("sample_id=%d NOT FOUND in offline_meta!", sid)
            total_fail += len(refs)
            continue

        d = tbl.to_pydict()
        name = d["name"][0]
        seq = [int(v) for v in d["seq_full"][0]]
        logger.info("\n--- sample_id=%d, name=%s, seq=%s (%d checks) ---",
                    sid, name, seq, len(refs))

        # Build BVD from disk — with or without prompt
        pil_frames = load_frames_from_disk(img_root, name, seq)
        datapoint = build_vos_datapoint(pil_frames, seq, video_id=name)
        resized = apply_indexing_transforms(datapoint, target_res=target_res)
        frames = frames_chw_from_datapoint(resized)

        # Check if any ref has a valid prompt_id
        ref_prompt_id = refs[0].get("prompt_id", -1) if refs else -1
        if ref_prompt_id >= 0:
            # Retrieve prompt info from offline_meta
            d_sample = tbl.to_pydict()
            dict_key = d_sample.get("dict_key", ["train"])[0]
            prompt_sets = d_sample.get("prompt_sets", [[]])[0]

            # Find the matching prompt set entry
            t_prompt = 0
            if prompt_sets:
                prompt_map = {int(e["frame_idx"]): int(e["prompt_id"]) for e in prompt_sets}
                # Use the prompt_id from our refs
                for fi, pi in prompt_map.items():
                    if pi == ref_prompt_id:
                        t_prompt = fi
                        break

            tbl_prompts = meta_ledger.find_prompts(sid, ref_prompt_id, t_prompt)
            if tbl_prompts.num_rows > 0:
                single_bvd, uid_map = make_single_bvd_with_prompt(
                    frames, name, seq, sid, ref_prompt_id,
                    tbl_prompts, t_prompt, dict_key,
                )
                logger.info("  Built BVD WITH prompt (pid=%d, t_prompt=%d, %d prompt rows)",
                           ref_prompt_id, t_prompt, tbl_prompts.num_rows)
            else:
                logger.warning("  No prompt rows found for pid=%d, falling back to no-prompt", ref_prompt_id)
                single_bvd, uid_map = make_single_bvd_no_prompt(
                    frames, name, seq, sid, "train", prompt_id_value=-1,
                )
        else:
            single_bvd, uid_map = make_single_bvd_no_prompt(
                frames, name, seq, sid, "train", prompt_id_value=-1,
            )

        # Forward pass with hook (accumulate all calls for multi-call layers)
        single_batch = adapter.preprocess_input(single_bvd)
        captured_list = []
        def hook_fn(module, inp, out):
            captured_list.append(out)
        handle = target_module.register_forward_hook(hook_fn)

        with torch.no_grad():
            with adapter.clicks_cache("record"), adapter.prompt_inputs_cache("record"):
                model(single_batch)
        handle.remove()

        if not captured_list:
            logger.error("  Activation not captured!")
            total_fail += len(refs)
            continue

        # Extract branch from each captured output
        import math

        def _extract_branch(raw):
            if isinstance(raw, dict):
                return raw[branch] if branch is not None else list(raw.values())[0]
            elif isinstance(raw, (tuple, list)):
                return raw[branch] if branch is not None else raw[0]
            return raw

        def _flatten_one(t):
            """Flatten a single activation tensor to 2D (tokens, D)."""
            f = t.detach().float()
            if f.dim() == 4:
                f = f.permute(0, 2, 3, 1).contiguous()
                return f.reshape(-1, f.shape[-1])
            elif f.dim() == 3:
                return f.reshape(-1, f.shape[-1])
            return f

        # Concatenate all calls
        acts_per_call = [_extract_branch(raw) for raw in captured_list]
        act_2d_parts = [_flatten_one(a) for a in acts_per_call]
        act_2d = torch.cat(act_2d_parts, dim=0)

        with torch.no_grad():
            codes = sae.encode(act_2d.to(device)).float().cpu()

        # Infer per-frame spatial dims from a single call's shape
        single_act = acts_per_call[0]
        orig_shape = list(single_act.shape)
        n_calls = len(captured_list)

        if len(orig_shape) == 4:
            frames_per_call = orig_shape[0]
            H_per_frame = orig_shape[2]
            W_per_frame = orig_shape[3]
        elif len(orig_shape) == 3:
            frames_per_call = orig_shape[0]
            n_spatial = orig_shape[1]
            side = int(math.isqrt(n_spatial))
            if side * side == n_spatial:
                H_per_frame, W_per_frame = side, side
            else:
                H_per_frame, W_per_frame = n_spatial, 1
        else:
            frames_per_call = 1
            H_per_frame, W_per_frame = act_2d.shape[0], 1

        N_frames = frames_per_call * n_calls
        N_tokens = codes.shape[0]

        # Determine if this is a multi-lane layer:
        # - Single-call layers (image encoder): first dim = B*T frames, no lane offset
        # - Multi-call layers (memory encoder, mask decoder): first dim = N_lane (objects)
        is_multi_call = (n_calls > 1)
        is_token_mode = (len(orig_shape) == 3)  # 3D: (N_lane, L, C)
        is_spatial_4d = (len(orig_shape) == 4)   # 4D: (N_lane, C, H, W)

        if is_multi_call:
            # Multi-call: first dim is N_lane (objects), one call per frame
            M_lanes = orig_shape[0]
            if is_token_mode:
                L_tokens = orig_shape[1]
            else:
                L_tokens = H_per_frame * W_per_frame
            tokens_per_call = M_lanes * L_tokens
        else:
            # Single-call: first dim is B*T (frames), no lane concept
            M_lanes = 1
            L_tokens = H_per_frame * W_per_frame
            tokens_per_call = N_tokens  # all tokens in one call

        logger.info("  Activation: %d calls × shape %s → flat: %s, codes: %s",
                    n_calls, orig_shape, list(act_2d.shape), list(codes.shape))
        logger.info("  multi_call=%s, M_lanes=%d, L_tokens=%d, tokens_per_call=%d, uid_map=%s",
                    is_multi_call, M_lanes, L_tokens, tokens_per_call,
                    uid_map if is_multi_call else "N/A")

        for ref in refs:
            unit = ref["unit"]
            expected = ref["score"]
            frame_idx = ref["frame_idx"]
            y, x = ref["y"], ref["x"]
            ref_uid = ref.get("uid", -1)

            if is_multi_call:
                # Multi-call layer: pos = call_idx * (M * L) + lane_idx * L + spatial_pos
                lane_idx = uid_map.get(ref_uid, 0) if ref_uid >= 0 else 0
                if y == -1:
                    # Token mode: spatial_pos = x (token index within lane)
                    spatial_pos = x
                else:
                    # Spatial mode: spatial_pos = y * W + x
                    spatial_pos = y * W_per_frame + x
                pos = frame_idx * tokens_per_call + lane_idx * L_tokens + spatial_pos
            elif y == -1:
                pos = x
            else:
                # Single-call spatial: pos = frame_idx * H * W + y * W + x
                pos = frame_idx * (H_per_frame * W_per_frame) + y * W_per_frame + x

            if pos < 0 or pos >= N_tokens:
                logger.warning("  SKIP: pos=%d (frame=%d, y=%d, x=%d) out of range [0, %d)",
                              pos, frame_idx, y, x, N_tokens)
                total_fail += 1
                continue

            actual = float(codes[pos, unit].item())
            diff = abs(actual - expected)
            max_diff = max(max_diff, diff)
            total_checks += 1

            if diff <= args.atol:
                total_pass += 1
                logger.info("  PASS: unit=%d, pos=%d (f=%d,y=%d,x=%d), expected=%.4f, actual=%.4f, diff=%.2e",
                           unit, pos, frame_idx, y, x, expected, actual, diff)
            else:
                total_fail += 1
                logger.error("  FAIL: unit=%d, pos=%d (f=%d,y=%d,x=%d), expected=%.4f, actual=%.4f, diff=%.2e",
                            unit, pos, frame_idx, y, x, expected, actual, diff)
                # Debug: show top-5 positions for this unit
                unit_codes = codes[:, unit]
                topk_vals, topk_pos = unit_codes.topk(min(5, (unit_codes > 0).sum().item()))
                logger.error("    Top positions for unit %d: %s",
                            unit, list(zip(topk_pos.tolist(), [f"{v:.4f}" for v in topk_vals.tolist()])))

    # ====================================================================
    # Summary
    # ====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PARQUET ROUNDTRIP SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total checks:   %d", total_checks)
    logger.info("  Passed:         %d", total_pass)
    logger.info("  Failed:         %d", total_fail)
    logger.info("  Max diff:       %.2e (atol=%.2e)", max_diff, args.atol)
    if total_fail > 0:
        logger.error("  RESULT: FAIL")
        sys.exit(1)
    else:
        logger.info("  RESULT: PASS")


if __name__ == "__main__":
    main()
