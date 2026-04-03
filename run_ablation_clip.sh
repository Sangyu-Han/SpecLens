#!/bin/bash
# SpecLens CLIP Ablation — 7-variant shared-buffer training
#
# Prerequisites:
#   1. bash scripts/kmeans/core/run_kmeans_clip_ablation.sh
#      (extracts activations + trains K-means centers; required for ra-* variants)
#
# Usage:
#   bash run_ablation_clip.sh
#
# W&B: project=clip-sae  entity=acoexist96  group=clip-ablation-7v
# Checkpoints: outputs/clip_ablation_7v/{layer}/{variant}/step_*.pt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/clip_imagenet_ablation.yaml"
LOG_DIR="outputs/logs"
LOG_FILE="$LOG_DIR/ablation_clip.log"
PYTHON="/home/mipal/miniconda3/envs/sae/bin/python"
TORCHRUN="/home/mipal/miniconda3/envs/sae/bin/torchrun"

mkdir -p "$LOG_DIR"

# Guard: don't double-start
if pgrep -f "train_sae_config.py" > /dev/null 2>&1; then
    echo "[WARN] Training already running:"
    pgrep -la "train_sae_config.py"
    echo "Kill with: pkill -f train_sae_config.py"
    exit 1
fi

# Sanity check: kmeans centers must exist for ra-* variants
CENTERS_DIR="outputs/kmeans_centers/clip_ablation"
if [ ! -d "$CENTERS_DIR" ]; then
    echo "[ERROR] K-means centers not found at: $CENTERS_DIR"
    echo "Run first: bash scripts/kmeans/core/run_kmeans_clip_ablation.sh"
    exit 1
fi

echo "============================================================"
echo " SpecLens CLIP Ablation (7 variants, shared activation buf)"
echo " Config:  $CONFIG"
echo " Log:     $LOG_FILE"
echo " K-means: $CENTERS_DIR"
echo " GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
echo "============================================================"
echo ""

# P2P disable (required on this server)
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4

echo "Starting training... (logs -> $LOG_FILE)"
echo ""

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/train_sae_config.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
