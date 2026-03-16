#!/usr/bin/env python3
"""
Visualize per-token spatial attribution maps for top SAE features.

For each top feature (ranked by libragrad_input_x_grad_abs score from
minfeat_summary.json), shows WHERE in the spatial grid that feature
contributes most to the mask objective.

Usage:
    python scripts/viz_attribution_spatial_maps.py \
        --unit 1753 --decile 0 --rank 1 --device cuda:1 --top-k 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    os.chdir(REPO_ROOT)
except Exception:
    pass

from scripts import output_contribution_compare_sam2 as base
from scripts import output_contribution_compare_sam2_viz as viz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_minfeat_summary(faith_dir: Path, unit: int, decile: int, rank: int,
                          objective_mode: str = "mask") -> Dict[str, Any]:
    """Load minfeat_summary.json for a given unit/decile/rank."""
    candidates = [
        faith_dir / f"unit_{unit}" / f"decile_{decile}" / f"rank{rank}" / "minfeat_summary.json",
        faith_dir / f"objective_{objective_mode}" / f"unit_{unit}" / f"decile_{decile}" / f"rank{rank}" / "minfeat_summary.json",
    ]
    for p in candidates:
        if p.exists():
            with p.open("r") as f:
                return json.load(f)
    raise FileNotFoundError(f"No minfeat_summary.json found for unit={unit} decile={decile} rank={rank}. Tried: {candidates}")


def _load_bias_set(freq_dir: Path, layer_base: str, threshold_pct: float = 95.0) -> Set[int]:
    """Load bias feature set from frequency parquet."""
    candidates = [
        freq_dir / f"{layer_base}.parquet",
        freq_dir / f"{layer_base.replace('.', '_').replace('@', '_')}.parquet",
    ]
    freq_path = None
    for p in candidates:
        if p.exists():
            freq_path = p
            break
    if freq_path is None:
        sanitized = layer_base.replace(".", "_").replace("@", "_")
        for p in sorted(freq_dir.glob("*.parquet")):
            if sanitized in p.stem or layer_base in p.stem:
                freq_path = p
                break
    if freq_path is None:
        print(f"[warn] No freq parquet for {layer_base}")
        return set()
    df = pd.read_parquet(freq_path)
    col = "freq_pct" if "freq_pct" in df.columns else "freq"
    vals = df[col]
    if vals.max() > 1.5:  # 0-100 scale
        bias_mask = vals >= threshold_pct
    else:
        bias_mask = vals >= (threshold_pct / 100.0)
    unit_col = "unit" if "unit" in df.columns else "feature_id"
    return set(int(u) for u in df.loc[bias_mask, unit_col])


def _token_grid_shape(n_tokens: int) -> Tuple[int, int]:
    """Infer spatial grid shape from token count."""
    sq = int(np.sqrt(n_tokens))
    if sq * sq == n_tokens:
        return sq, sq
    # Try common shapes
    for h in range(sq + 2, 0, -1):
        if n_tokens % h == 0:
            return h, n_tokens // h
    return sq, sq


def _to_spatial(vec: np.ndarray) -> np.ndarray:
    """Reshape 1D token vector to 2D spatial map."""
    h, w = _token_grid_shape(len(vec))
    return vec[:h * w].reshape(h, w)


def _log_transform(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Log1p transform for better visualization of sparse maps."""
    return np.log1p(arr / (eps + arr[arr > 0].min() * 0.1 if (arr > 0).any() else 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial attribution maps for top SAE features")
    parser.add_argument("--attr-config", type=Path, default=Path("configs/sam2_attr_index_v2.yaml"))
    parser.add_argument("--anchor", type=str,
                        default="model.sam_mask_decoder.transformer.layers.0@1::sae_layer#latent")
    parser.add_argument("--unit", type=int, default=1753, help="Target unit for sample selection")
    parser.add_argument("--sample-unit", type=int, default=None, help="Unit for ledger lookup (defaults to --unit)")
    parser.add_argument("--decile", type=int, default=0)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to visualize")
    parser.add_argument("--faith-dir", type=Path, default=Path("outputs/sam2_faithfulness"))
    parser.add_argument("--freq-dir", type=Path,
                        default=Path("outputs/sae_index_ra-batchtopk_v2/feature_freq"))
    parser.add_argument("--objective-mode", type=str, default="mask")
    parser.add_argument("--mask-specs", type=str,
                        default="model@0@pred_masks_high_res,model@1@pred_masks_high_res,"
                                "model@2@pred_masks_high_res,model@3@pred_masks_high_res")
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    parser.add_argument("--libragrad-gamma", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--lane", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/attribution_spatial_maps"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.sample_unit is None:
        args.sample_unit = args.unit

    device = torch.device(args.device)
    debug = bool(args.debug)

    # --- Load minfeat summary to get top features ---
    summary = _load_minfeat_summary(args.faith_dir, args.unit, args.decile, args.rank, args.objective_mode)
    method_data = summary.get("methods", {}).get("libragrad_input_x_grad_abs", {})
    feature_list = method_data.get("feature_list", [])
    if not feature_list:
        raise RuntimeError("No feature_list in minfeat_summary.json")

    top_features = feature_list[:args.top_k]
    print(f"[info] Loaded {len(feature_list)} features, visualizing top {len(top_features)}")
    for i, f in enumerate(top_features[:5]):
        print(f"  #{i}: unit={f['unit']} score={f['score']:.6f}")

    # --- Load bias feature set ---
    parsed_anchor = base.parse_spec(args.anchor)
    layer_base = parsed_anchor.base_with_branch
    bias_set = _load_bias_set(args.freq_dir, layer_base)
    print(f"[info] Loaded {len(bias_set)} bias features (>=95% freq)")

    # --- Setup model and data ---
    attr_cfg = base._load_yaml(args.attr_config)
    index_cfg = base._load_yaml(Path(attr_cfg["indexing"]["config"]))
    model_loader = base.load_obj(index_cfg["model"]["loader"])
    model = model_loader(index_cfg["model"], device=device).eval()
    for p in model.parameters():
        p.requires_grad = False
    adapter = base.SAM2EvalAdapter(model, device=device, collate_fn=None)

    # Select sample row
    ledger = base.DecileParquetLedger(index_cfg["indexing"]["out_dir"])
    parsed_layer = base.parse_spec(attr_cfg["indexing"]["layer"])
    ledger_layer = parsed_layer.base_with_branch
    table = ledger.topn_for(layer=ledger_layer, unit=int(args.sample_unit),
                            decile=int(args.decile), n=int(args.rank) + 1)
    rows = table.to_pylist() if table is not None else []
    if not rows or args.rank >= len(rows):
        raise RuntimeError(f"No row for unit={args.sample_unit} decile={args.decile} rank={args.rank}")
    row = rows[int(args.rank)]

    meta_ledger = base.OfflineMetaParquetLedger(index_cfg["indexing"]["offline_meta_root"])
    bvd, lane_idx = base._build_bvd_for_row(row=row, meta_ledger=meta_ledger, ds_cfg=index_cfg["dataset"])
    if args.lane is not None:
        lane_idx = int(args.lane)
    batch_on_dev = adapter.preprocess_input(bvd)
    base._record_adapter_caches(adapter, batch_on_dev)
    forward_runner = base._make_forward_runner(adapter, batch_on_dev)

    # --- Load original image for overlay ---
    sample_id = int(row.get("sample_id", 0))
    sample_meta = base._table_first_row(meta_ledger.find_sample(sample_id))
    name = sample_meta.get("name", "")
    seq_full = [int(v) for v in (sample_meta.get("seq_full") or [])]
    frame_idx = int(row.get("frame_idx", 0))
    img_root = Path(index_cfg["dataset"]["img_folder"])
    try:
        from src.packs.sam2.offline.bvd_builders import load_frames_from_disk
        pil_frames = load_frames_from_disk(img_root, name, [int(v) for v in seq_full])
        orig_image = pil_frames[frame_idx] if frame_idx < len(pil_frames) else pil_frames[0]
    except Exception as e:
        print(f"[warn] Could not load original image: {e}")
        orig_image = None

    # --- Setup anchor SAE ---
    anchor_specs_raw = base._parse_anchor_specs(args.anchor)
    anchor_specs = base._expand_anchor_specs(anchor_specs_raw)
    target_spec = anchor_specs[0] if anchor_specs else str(args.anchor)

    mask_specs = [s for s in (args.mask_specs or "").split(",") if s]
    ref_mask_spec = mask_specs[0] if mask_specs else None
    if parsed_anchor.base_branch is not None:
        output_spec = f"model@{parsed_anchor.base_branch}@pred_masks_high_res"
    else:
        output_spec = mask_specs[0]
    if output_spec not in mask_specs:
        mask_specs.append(output_spec)

    anchors: Dict[str, base.AnchorInfo] = {}
    restore_handles = []
    for spec in anchor_specs:
        controller = base.AllTokensFeatureOverrideController(
            spec=base.OverrideSpec(lane_idx=None, unit_indices=None),
            frame_getter=getattr(adapter, "current_frame_idx", None),
        )
        sae = base._load_sae_for_layer(index_cfg["sae"], spec, device)
        capture = base.LayerCapture(spec)
        owner = base.resolve_module(model, capture.base)
        handle, branch = base.wrap_target_layer_with_sae(
            owner,
            capture=capture,
            sae=sae,
            controller=controller,
            frame_getter=getattr(adapter, "current_frame_idx", None),
        )
        restore_handles.append(handle)
        attr_name = base._anchor_attr_name(spec)
        anchors[spec] = base.AnchorInfo(spec=spec, branch=branch, attr_name=attr_name, controller=controller)

    autoreset_handle = base.install_controller_autoreset_hooks(
        model,
        [a.controller for a in anchors.values()],
        branches=[a.branch for a in anchors.values()],
    )
    if autoreset_handle is not None:
        restore_handles.append(autoreset_handle.remove)

    for anchor in anchors.values():
        try:
            anchor.controller.spec.lane_idx = lane_idx
        except Exception:
            pass

    mask_capture = base.MultiAnchorCapture(frame_getter=getattr(adapter, "current_frame_idx", None))
    mask_capture.register_from_specs(mask_specs, resolve_module_fn=lambda n: base.resolve_module(model, n))

    num_frames, num_lanes = viz._infer_meta_dims(adapter, batch_on_dev)

    # --- Capture base mask ---
    mask_capture.release_step_refs()
    mask_capture.clear_tapes()
    forward_runner(require_grad=False)
    base_bundle = mask_capture.get_tensor_lists(detach=True)
    base_obj_t, objective_mask, objective_ref_logits = base._objective_from_masks(
        base_bundle, lane_idx, float(args.mask_threshold),
        ref_spec=ref_mask_spec, fixed_mask=None, fixed_ref_logits=None,
        objective_mode=args.objective_mode,
    )
    objective_mask = [m.detach() for m in objective_mask]
    objective_ref_logits = [r.detach() for r in objective_ref_logits]
    # Extract base mask for visualization
    base_tensor = viz._extract_spec_tensor(base_bundle, output_spec)
    if base_tensor is not None:
        base_map = viz._extract_output_map(
            base_tensor, lane_idx=lane_idx, num_frames=num_frames,
            num_lanes=num_lanes, debug=debug,
        )
        _bm = torch.sigmoid(base_map[0]).detach().cpu()
        # Squeeze extra leading dims (e.g., [1, H, W] -> [H, W])
        while _bm.dim() > 2:
            _bm = _bm[0]
        base_mask_np = _bm.numpy()
    else:
        base_mask_np = None
    mask_capture.release_step_refs()

    # --- Compute per-token attribution with libragrad ---
    print("[info] Computing libragrad input_x_grad attribution (per-token)...")
    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()
        anchor.branch.clear_context()

    param_states = []
    for anchor in anchors.values():
        for param in anchor.branch.sae.parameters():
            param_states.append((param, param.requires_grad))
            param.requires_grad_(True)

    per_token_contrib = {}  # spec -> [n_tokens_total, n_features]
    try:
        with base._LibragradContext(model, anchors, gamma=args.libragrad_gamma):
            mask_capture.release_step_refs()
            mask_capture.clear_tapes()
            forward_runner(require_grad=True)
            mask_lists = mask_capture.get_tensor_lists(detach=False)
            objective, _, _ = base._objective_from_masks(
                mask_lists, lane_idx, float(args.mask_threshold),
                ref_spec=ref_mask_spec, fixed_mask=objective_mask,
                fixed_ref_logits=objective_ref_logits,
                objective_mode=args.objective_mode,
            )
            tape_tensors = []
            tape_specs = []
            for spec, anchor in anchors.items():
                tape = anchor.branch.sae_tape(anchor.attr_name)
                if tape is None or tape.frame_count() == 0:
                    continue
                for rec in tape.frames():
                    if not torch.is_tensor(rec.tensor):
                        continue
                    tape_tensors.append(rec.tensor)
                    tape_specs.append(spec)

            if not tape_tensors:
                raise RuntimeError("No tape tensors captured")

            grads = torch.autograd.grad(
                objective, tape_tensors,
                retain_graph=False, create_graph=False, allow_unused=True,
            )

            # Collect per-token contributions (don't sum over tokens!)
            for (spec, acts), grad in zip(zip(tape_specs, tape_tensors), grads):
                if grad is None:
                    continue
                # acts: [n_tokens, n_features] (per frame)
                # contrib: acts * |grad| → [n_tokens, n_features]
                contrib_abs = (acts.detach() * grad.abs()).detach().cpu()
                contrib_signed = (acts.detach() * grad).detach().cpu()
                if spec not in per_token_contrib:
                    per_token_contrib[spec] = {"abs": [], "signed": []}
                per_token_contrib[spec]["abs"].append(contrib_abs)
                per_token_contrib[spec]["signed"].append(contrib_signed)
    finally:
        for param, prev in param_states:
            param.requires_grad_(prev)
        mask_capture.release_step_refs()
        base._reset_anchor_controllers(anchors, reason="attr-spatial")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stack per-frame contributions
    for spec in per_token_contrib:
        per_token_contrib[spec]["abs"] = torch.cat(per_token_contrib[spec]["abs"], dim=0)
        per_token_contrib[spec]["signed"] = torch.cat(per_token_contrib[spec]["signed"], dim=0)
        print(f"  [token contrib] {spec}: shape={per_token_contrib[spec]['abs'].shape}")

    # --- Also capture raw SAE activations for activation maps ---
    print("[info] Capturing raw SAE activations...")
    for anchor in anchors.values():
        anchor.branch.clear_anchor_overrides()
        anchor.branch.clear_context()
    forward_runner(require_grad=False)
    raw_acts = {}
    for spec, anchor in anchors.items():
        tape = anchor.branch.sae_tape(anchor.attr_name)
        if tape is None or tape.frame_count() == 0:
            continue
        frames_list = []
        for rec in tape.frames():
            if torch.is_tensor(rec.tensor):
                frames_list.append(rec.tensor.detach().cpu())
        if frames_list:
            raw_acts[spec] = torch.cat(frames_list, dim=0)  # [n_tokens_total, n_features]
            print(f"  [raw acts] {spec}: shape={raw_acts[spec].shape}")

    # Also get decoder dictionary for cosine similarity maps
    sae_module = anchors[target_spec].branch.sae
    dictionary = sae_module.dictionary.get_dictionary().detach().cpu().numpy()
    print(f"  [dictionary] shape={dictionary.shape}")

    # Get encoder weights for comparison
    W_enc = sae_module.W_enc.detach().cpu().numpy()  # [act_size, dict_size]
    print(f"  [W_enc] shape={W_enc.shape}")

    # --- Capture raw input tokens via hook on SAE branch ---
    print("[info] Capturing raw input tokens via SAE branch hook...")
    raw_input_acts = {}
    _captured_inputs: List[torch.Tensor] = []

    def _capture_sae_input_hook(module, args_tuple, output):
        """Hook on the SAE branch to capture input tensor before SAE processing."""
        if args_tuple and torch.is_tensor(args_tuple[0]):
            # args_tuple[0] is the input tensor to the branch
            inp = args_tuple[0].detach().cpu()
            # Flatten like SAE does
            if inp.dim() > 2:
                inp = inp.reshape(-1, inp.shape[-1])
            _captured_inputs.append(inp)

    branch = anchors[target_spec].branch
    hook_handle = branch.register_forward_hook(_capture_sae_input_hook)
    try:
        for anchor in anchors.values():
            anchor.branch.clear_anchor_overrides()
            anchor.branch.clear_context()
        forward_runner(require_grad=False)
    finally:
        hook_handle.remove()

    if _captured_inputs:
        raw_input_acts[target_spec] = torch.cat(_captured_inputs, dim=0)
        print(f"  [raw input] {target_spec}: shape={raw_input_acts[target_spec].shape}")
    else:
        print("  [warn] No input tokens captured via hook")

    # --- Visualization ---
    out_root = args.out_dir / f"unit_{args.unit}" / f"decile_{args.decile}" / f"rank{args.rank}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Use only the first matching spec
    spec_key = target_spec
    if spec_key not in per_token_contrib:
        # Try the first available
        spec_key = list(per_token_contrib.keys())[0] if per_token_contrib else None
    if spec_key is None:
        raise RuntimeError("No per-token contributions computed")

    abs_contrib = per_token_contrib[spec_key]["abs"].numpy()   # [n_tokens, n_features]
    signed_contrib = per_token_contrib[spec_key]["signed"].numpy()
    n_tokens, n_features = abs_contrib.shape
    print(f"\n[info] Token grid: {n_tokens} tokens, {n_features} features")

    # Use first frame tokens only (if multi-frame)
    # Mask decoder tokens: [n_frames * n_lanes * spatial_tokens, n_features]
    # For SAM2: spatial_tokens = 4096 (64x64), n_lanes = max_objects
    tokens_per_frame = n_tokens // max(1, num_frames or 1) if num_frames else n_tokens
    n_lanes = num_lanes or 1
    spatial_tokens = tokens_per_frame // n_lanes if n_lanes > 0 else tokens_per_frame
    print(f"  tokens_per_frame={tokens_per_frame}, n_lanes={n_lanes}, spatial_tokens={spatial_tokens}")

    # Extract tokens for first frame, correct lane
    frame_0_start = 0
    frame_0_end = tokens_per_frame
    lane_offset = (lane_idx or 0) * spatial_tokens
    lane_slice_start = frame_0_start + lane_offset
    lane_slice_end = lane_slice_start + spatial_tokens
    print(f"  Using tokens [{lane_slice_start}:{lane_slice_end}] (frame=0, lane={lane_idx})")

    abs_contrib_f0 = abs_contrib[lane_slice_start:lane_slice_end]
    signed_contrib_f0 = signed_contrib[lane_slice_start:lane_slice_end]

    acts_np = raw_acts.get(spec_key, raw_acts.get(list(raw_acts.keys())[0]) if raw_acts else None)
    if acts_np is not None:
        acts_np = acts_np.numpy()[lane_slice_start:lane_slice_end]

    # ======== Figure 1: Top-K features spatial attribution (linear + log) ========
    n_show = min(args.top_k, len(top_features))
    n_cols = 5
    n_rows = (n_show + n_cols - 1) // n_cols

    # --- Fig 1a: Linear scale ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Per-token attribution (linear) — unit {args.unit}, sample {sample_id}", fontsize=14)

    for i in range(n_show):
        feat = top_features[i]
        feat_idx = int(feat["unit"])
        score = float(feat["score"])
        is_bias = feat_idx in bias_set
        label = "B" if is_bias else "T"

        ax = axes[i // n_cols, i % n_cols]
        spatial = _to_spatial(abs_contrib_f0[:, feat_idx])
        im = ax.imshow(spatial, cmap="hot", interpolation="bilinear")
        ax.set_title(f"F{feat_idx}[{label}]\n{score:.5f}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off unused axes
    for i in range(n_show, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_root / "01_attribution_linear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_attribution_linear.png")

    # --- Fig 1b: Log scale ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Per-token attribution (log scale) — unit {args.unit}, sample {sample_id}", fontsize=14)

    for i in range(n_show):
        feat = top_features[i]
        feat_idx = int(feat["unit"])
        score = float(feat["score"])
        is_bias = feat_idx in bias_set
        label = "B" if is_bias else "T"

        ax = axes[i // n_cols, i % n_cols]
        spatial = _to_spatial(abs_contrib_f0[:, feat_idx])
        spatial_log = _log_transform(spatial)
        im = ax.imshow(spatial_log, cmap="hot", interpolation="bilinear")
        ax.set_title(f"F{feat_idx}[{label}]\n{score:.5f}", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(n_show, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")

    fig.tight_layout()
    fig.savefig(out_root / "01_attribution_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 01_attribution_log.png")

    # ======== Figure 2: Hero figure — top 8 features, 4 views each ========
    n_hero = min(8, n_show)
    fig = plt.figure(figsize=(24, n_hero * 3))
    gs = gridspec.GridSpec(n_hero, 6, figure=fig, wspace=0.3, hspace=0.4)
    fig.suptitle(f"Attribution hero — unit {args.unit}, sample {sample_id}\n"
                 f"[B]=bias(>=95%freq), [T]=task", fontsize=14, y=0.98)

    for i in range(n_hero):
        feat = top_features[i]
        feat_idx = int(feat["unit"])
        score = float(feat["score"])
        is_bias = feat_idx in bias_set
        label = "B" if is_bias else "T"

        # Col 0: Original image
        ax0 = fig.add_subplot(gs[i, 0])
        if orig_image is not None:
            ax0.imshow(orig_image)
        if i == 0:
            ax0.set_title("Original", fontsize=9)
        ax0.set_ylabel(f"F{feat_idx}[{label}]\nscore={score:.5f}", fontsize=8)
        ax0.set_xticks([])
        ax0.set_yticks([])

        # Col 1: Base mask
        ax1 = fig.add_subplot(gs[i, 1])
        if base_mask_np is not None:
            ax1.imshow(base_mask_np, cmap="gray", vmin=0, vmax=1)
        if i == 0:
            ax1.set_title("Mask", fontsize=9)
        ax1.axis("off")

        # Col 2: Attribution (linear)
        ax2 = fig.add_subplot(gs[i, 2])
        spatial_abs = _to_spatial(abs_contrib_f0[:, feat_idx])
        im2 = ax2.imshow(spatial_abs, cmap="hot", interpolation="bilinear")
        if i == 0:
            ax2.set_title("Attr (linear)", fontsize=9)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Col 3: Attribution (log)
        ax3 = fig.add_subplot(gs[i, 3])
        spatial_log = _log_transform(spatial_abs)
        im3 = ax3.imshow(spatial_log, cmap="hot", interpolation="bilinear")
        if i == 0:
            ax3.set_title("Attr (log)", fontsize=9)
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Col 4: Activation map
        ax4 = fig.add_subplot(gs[i, 4])
        if acts_np is not None:
            act_spatial = _to_spatial(acts_np[:, feat_idx])
            im4 = ax4.imshow(act_spatial, cmap="viridis", interpolation="bilinear")
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        if i == 0:
            ax4.set_title("Activation", fontsize=9)
        ax4.axis("off")

        # Col 5: Cosine similarity (decoder vector vs tokens)
        ax5 = fig.add_subplot(gs[i, 5])
        if raw_input_acts and spec_key in raw_input_acts:
            tokens_np = raw_input_acts[spec_key].numpy()[lane_slice_start:lane_slice_end]
            dec_vec = dictionary[feat_idx]
            dec_norm = dec_vec / (np.linalg.norm(dec_vec) + 1e-8)
            token_norms = np.linalg.norm(tokens_np, axis=1, keepdims=True)
            cos_sim = (tokens_np @ dec_norm) / (token_norms.squeeze() + 1e-8)
            cos_spatial = _to_spatial(cos_sim)
            im5 = ax5.imshow(cos_spatial, cmap="RdBu_r", interpolation="bilinear",
                             vmin=-1, vmax=1)
            plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        if i == 0:
            ax5.set_title("Cos sim\n(dec vec)", fontsize=9)
        ax5.axis("off")

    fig.savefig(out_root / "02_attribution_hero.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 02_attribution_hero.png")

    # ======== Figure 3: Encoder vs Decoder cosine similarity comparison ========
    n_compare = min(6, n_show)
    fig, axes = plt.subplots(n_compare, 4, figsize=(16, n_compare * 3.5))
    if n_compare == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Encoder vs Decoder vector cosine similarity with tokens", fontsize=14)

    for i in range(n_compare):
        feat = top_features[i]
        feat_idx = int(feat["unit"])
        is_bias = feat_idx in bias_set
        label = "B" if is_bias else "T"

        ax_img = axes[i, 0]
        if orig_image is not None:
            ax_img.imshow(orig_image)
        ax_img.set_ylabel(f"F{feat_idx}[{label}]", fontsize=10)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        if i == 0:
            ax_img.set_title("Image", fontsize=10)

        if raw_input_acts and spec_key in raw_input_acts:
            tokens_np = raw_input_acts[spec_key].numpy()[lane_slice_start:lane_slice_end]
            token_norms = np.linalg.norm(tokens_np, axis=1, keepdims=True) + 1e-8

            # Decoder cosine sim
            ax_dec = axes[i, 1]
            dec_vec = dictionary[feat_idx]
            dec_norm = dec_vec / (np.linalg.norm(dec_vec) + 1e-8)
            cos_dec = (tokens_np @ dec_norm) / token_norms.squeeze()
            im_dec = ax_dec.imshow(_to_spatial(cos_dec), cmap="RdBu_r",
                                   interpolation="bilinear", vmin=-1, vmax=1)
            if i == 0:
                ax_dec.set_title("Decoder cos sim", fontsize=10)
            ax_dec.axis("off")
            plt.colorbar(im_dec, ax=ax_dec, fraction=0.046)

            # Encoder cosine sim
            ax_enc = axes[i, 2]
            enc_vec = W_enc[:, feat_idx]  # [act_size]
            enc_norm_val = enc_vec / (np.linalg.norm(enc_vec) + 1e-8)
            cos_enc = (tokens_np @ enc_norm_val) / token_norms.squeeze()
            im_enc = ax_enc.imshow(_to_spatial(cos_enc), cmap="RdBu_r",
                                   interpolation="bilinear", vmin=-1, vmax=1)
            if i == 0:
                ax_enc.set_title("Encoder cos sim", fontsize=10)
            ax_enc.axis("off")
            plt.colorbar(im_enc, ax=ax_enc, fraction=0.046)

            # Difference
            ax_diff = axes[i, 3]
            diff = cos_dec - cos_enc
            max_abs = max(abs(diff.min()), abs(diff.max()), 0.01)
            im_diff = ax_diff.imshow(_to_spatial(diff), cmap="RdBu_r",
                                     interpolation="bilinear", vmin=-max_abs, vmax=max_abs)
            if i == 0:
                ax_diff.set_title("Dec - Enc", fontsize=10)
            ax_diff.axis("off")
            plt.colorbar(im_diff, ax=ax_diff, fraction=0.046)

            # Print encoder-decoder cosine similarity
            enc_dec_cos = np.dot(dec_norm, enc_norm_val)
            print(f"  F{feat_idx}[{label}]: enc-dec cosine = {enc_dec_cos:.4f}")
        else:
            for j in range(1, 4):
                axes[i, j].text(0.5, 0.5, "No input\ntokens", ha="center", va="center")
                axes[i, j].axis("off")

    fig.tight_layout()
    fig.savefig(out_root / "03_enc_vs_dec_cosine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 03_enc_vs_dec_cosine.png")

    # ======== Figure 4: Signed attribution (positive vs negative) ========
    fig, axes = plt.subplots(n_hero, 3, figsize=(12, n_hero * 3))
    if n_hero == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Signed attribution — unit {args.unit}, sample {sample_id}", fontsize=14)

    for i in range(n_hero):
        feat = top_features[i]
        feat_idx = int(feat["unit"])
        is_bias = feat_idx in bias_set
        label = "B" if is_bias else "T"

        signed = signed_contrib_f0[:, feat_idx]
        pos = np.clip(signed, 0, None)
        neg = np.clip(-signed, 0, None)

        # Positive
        ax_pos = axes[i, 0]
        spatial_pos = _to_spatial(pos)
        spatial_pos_log = _log_transform(spatial_pos)
        im_p = ax_pos.imshow(spatial_pos_log, cmap="Reds", interpolation="bilinear")
        ax_pos.set_ylabel(f"F{feat_idx}[{label}]", fontsize=9)
        if i == 0:
            ax_pos.set_title("Positive (log)", fontsize=10)
        ax_pos.axis("off")
        plt.colorbar(im_p, ax=ax_pos, fraction=0.046)

        # Negative
        ax_neg = axes[i, 1]
        spatial_neg = _to_spatial(neg)
        spatial_neg_log = _log_transform(spatial_neg)
        im_n = ax_neg.imshow(spatial_neg_log, cmap="Blues", interpolation="bilinear")
        if i == 0:
            ax_neg.set_title("Negative (log)", fontsize=10)
        ax_neg.axis("off")
        plt.colorbar(im_n, ax=ax_neg, fraction=0.046)

        # Combined signed (diverging colormap)
        ax_comb = axes[i, 2]
        spatial_signed = _to_spatial(signed)
        max_abs = max(abs(spatial_signed.min()), abs(spatial_signed.max()), 1e-8)
        # Apply sign-preserving log transform
        sign = np.sign(spatial_signed)
        log_mag = _log_transform(np.abs(spatial_signed))
        log_signed = sign * log_mag
        max_log = max(abs(log_signed.min()), abs(log_signed.max()), 1e-8)
        im_c = ax_comb.imshow(log_signed, cmap="RdBu_r", interpolation="bilinear",
                              vmin=-max_log, vmax=max_log)
        if i == 0:
            ax_comb.set_title("Signed (log)", fontsize=10)
        ax_comb.axis("off")
        plt.colorbar(im_c, ax=ax_comb, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_root / "04_attribution_signed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 04_attribution_signed.png")

    # ======== Figure 5: Bias vs Task aggregate comparison ========
    bias_features = [(i, f) for i, f in enumerate(top_features) if int(f["unit"]) in bias_set]
    task_features = [(i, f) for i, f in enumerate(top_features) if int(f["unit"]) not in bias_set]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"Bias vs Task feature attribution — unit {args.unit}", fontsize=14)

    for col, (group_name, group) in enumerate([("Bias", bias_features), ("Task", task_features)]):
        if not group:
            for r in range(2):
                axes[r, col * 2].text(0.5, 0.5, f"No {group_name}\nfeatures", ha="center", va="center")
                axes[r, col * 2].axis("off")
                axes[r, col * 2 + 1].text(0.5, 0.5, f"No {group_name}\nfeatures", ha="center", va="center")
                axes[r, col * 2 + 1].axis("off")
            continue

        # Average attribution across group
        group_abs = np.stack([abs_contrib_f0[:, int(f["unit"])] for _, f in group], axis=0)
        mean_abs = group_abs.mean(axis=0)
        max_abs_val = group_abs.max(axis=0)

        # Mean (linear)
        ax = axes[0, col * 2]
        im = ax.imshow(_to_spatial(mean_abs), cmap="hot", interpolation="bilinear")
        ax.set_title(f"{group_name} mean (linear)\nn={len(group)}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Mean (log)
        ax = axes[0, col * 2 + 1]
        im = ax.imshow(_to_spatial(_log_transform(mean_abs)), cmap="hot", interpolation="bilinear")
        ax.set_title(f"{group_name} mean (log)", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Max (linear)
        ax = axes[1, col * 2]
        im = ax.imshow(_to_spatial(max_abs_val), cmap="hot", interpolation="bilinear")
        ax.set_title(f"{group_name} max (linear)", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Max (log)
        ax = axes[1, col * 2 + 1]
        im = ax.imshow(_to_spatial(_log_transform(max_abs_val)), cmap="hot", interpolation="bilinear")
        ax.set_title(f"{group_name} max (log)", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_root / "05_bias_vs_task_aggregate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 05_bias_vs_task_aggregate.png")

    # ======== Summary stats ========
    n_bias_in_top = len(bias_features)
    n_task_in_top = len(task_features)
    summary_stats = {
        "unit": args.unit,
        "sample_id": sample_id,
        "top_k": n_show,
        "n_bias_in_top": n_bias_in_top,
        "n_task_in_top": n_task_in_top,
        "bias_pct": f"{100 * n_bias_in_top / n_show:.1f}%",
        "features": [
            {
                "rank": i,
                "unit": int(f["unit"]),
                "score": float(f["score"]),
                "is_bias": int(f["unit"]) in bias_set,
            }
            for i, f in enumerate(top_features)
        ],
    }
    with open(out_root / "summary.json", "w") as fh:
        json.dump(summary_stats, fh, indent=2)
    print(f"\n  Saved summary.json")

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {out_root}")
    print(f"Top {n_show} features: {n_bias_in_top} bias, {n_task_in_top} task")
    print(f"{'='*60}")

    # Cleanup
    for h in restore_handles:
        try:
            if callable(h):
                h()
        except Exception:
            pass


if __name__ == "__main__":
    main()
