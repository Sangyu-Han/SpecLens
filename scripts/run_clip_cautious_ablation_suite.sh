#!/usr/bin/env bash
set -euo pipefail

REPO="${SPECLENS_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/mipal/miniconda3/envs/sae/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO/outputs/paper_feature_erf_clip_ablation_20260422}"
MANIFEST_JSON="${MANIFEST_JSON:-$OUTPUT_ROOT/manifest_clip_3blocks_300.json}"
IMAGENET_VAL="${IMAGENET_VAL:-/data/datasets/imagenet/val}"
CLIP_INDEX_ROOT="${CLIP_INDEX_ROOT:-/media/mipal/1TB/sangyu/SpecLens_outputs/clip_50k_index}"
CLIP_SAE_ROOT="${CLIP_SAE_ROOT:-/media/mipal/1TB/sangyu/SpecLens_outputs/clip_50k_sae}"
HF_HOME="${HF_HOME:-/media/mipal/1TB/sangyu/hf_cache}"
TMPDIR="${TMPDIR:-/media/mipal/1TB/sangyu/tmp}"
ERF_BASELINE_CACHE_ROOT="${ERF_BASELINE_CACHE_ROOT:-$REPO/outputs/erf_baselines}"
GPU_SOLVER="${GPU_SOLVER:-0}"
GPU_MASKING="${GPU_MASKING:-1}"
MODE="${1:-launch}"

mkdir -p "$OUTPUT_ROOT/logs" "$TMPDIR"

COMMON_ARGS=(
  --pack clip
  --blocks 2 6 10
  --n-features 100
  --n-images 1
  --imagenet-val "$IMAGENET_VAL"
  --index-root "clip=$CLIP_INDEX_ROOT"
  --sae-root "clip=$CLIP_SAE_ROOT"
  --stoch-steps 20
  --stoch-samples 5
  --stoch-seeds 0 1 2
  --cautious-steps 32
  --cautious-lr 0.45
  --cautious-lr-end 0.01
  --cautious-tv-weight 0.01
  --cautious-irr-weight 0.05
  --cautious-init-prob 0.5
  --cautious-init-mode uniform
  --cautious-reg-warmup-frac 0.0
  --cautious-restarts 1
  --cautious-budget-samples 1
)

run_bench() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" \
  SPECLENS_REPO="$REPO" \
  HF_HOME="$HF_HOME" \
  TMPDIR="$TMPDIR" \
  ERF_BASELINE_CACHE_ROOT="$ERF_BASELINE_CACHE_ROOT" \
  "$PYTHON_BIN" "$REPO/scripts/run_feature_erf_paper_benchmark.py" "$@"
}

if [[ ! -f "$MANIFEST_JSON" ]]; then
  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --methods plain_ixg \
    --output-json "$OUTPUT_ROOT/manifest_build.json" \
    --save-manifest-json "$MANIFEST_JSON"
fi

run_solver_queue() {
  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods plain_ixg \
    --output-json "$OUTPUT_ROOT/solver_plain_ixg.json"

  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --output-json "$OUTPUT_ROOT/solver_full.json"

  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --cautious-objective-mode fixed_budget_softins \
    --cautious-fixed-budget-frac 0.10 \
    --output-json "$OUTPUT_ROOT/solver_no_randbudget.json"

  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --cautious-objective-mode direct_recovery \
    --output-json "$OUTPUT_ROOT/solver_no_softins.json"

  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --cautious-optimizer-mode adam_cosine \
    --output-json "$OUTPUT_ROOT/solver_no_cautious.json"

  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --cautious-irr-weight 0.0 \
    --output-json "$OUTPUT_ROOT/solver_no_irr.json"

  run_bench "$GPU_SOLVER" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --cautious-tv-weight 0.0 \
    --output-json "$OUTPUT_ROOT/solver_no_tv.json"
}

run_masking_queue() {
  run_bench "$GPU_MASKING" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h_plus_pos \
    --output-json "$OUTPUT_ROOT/masking_global_mean_h_plus_pos.json"

  run_bench "$GPU_MASKING" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode default \
    --output-json "$OUTPUT_ROOT/masking_default_pos_only.json"

  run_bench "$GPU_MASKING" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode global_mean_h \
    --output-json "$OUTPUT_ROOT/masking_global_mean_h.json"

  run_bench "$GPU_MASKING" \
    "${COMMON_ARGS[@]}" \
    --manifest-json "$MANIFEST_JSON" \
    --methods cautious_cos \
    --baseline-mode zero \
    --output-json "$OUTPUT_ROOT/masking_zero_hidden.json"
}

if [[ "$MODE" == "solver" ]]; then
  run_solver_queue
  exit 0
fi

if [[ "$MODE" == "masking" ]]; then
  run_masking_queue
  exit 0
fi

nohup bash "$0" solver >"$OUTPUT_ROOT/logs/solver_queue.log" 2>&1 &
SOLVER_PID=$!
nohup bash "$0" masking >"$OUTPUT_ROOT/logs/masking_queue.log" 2>&1 &
MASKING_PID=$!

echo "manifest=$MANIFEST_JSON"
echo "solver_pid=$SOLVER_PID"
echo "masking_pid=$MASKING_PID"
echo "output_root=$OUTPUT_ROOT"
