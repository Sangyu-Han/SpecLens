#!/bin/bash
# SpecLens SAMv2 Ablation — 7-variant shared-buffer training (local)
#
# Prerequisites:
#   1. bash scripts/kmeans/core/run_kmeans_sam2_ablation.sh
#      (extracts activations + trains K-means centers; required for ra-* variants)
#
# Usage:
#   bash run_ablation_sam2.sh
#
# W&B: project=sam2-sae  entity=acoexist96  group=sam2-ablation-7v
# Checkpoints: outputs/sam2_sav_ablation_7v/{layer}/{variant}/step_*.pt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/sam2_sav_ablation.yaml"
LOG_DIR="outputs/logs"
LOG_FILE="$LOG_DIR/ablation_sam2.log"
PYTHON="/home/sangyu/anaconda3/envs/py312/bin/python"
TORCHRUN="/home/sangyu/anaconda3/envs/py312/bin/torchrun"

mkdir -p "$LOG_DIR"

# Guard: don't double-start
if pgrep -f "train_sae_config.py" > /dev/null 2>&1; then
    echo "[WARN] Training already running:"
    pgrep -la "train_sae_config.py"
    echo "Kill with: pkill -f train_sae_config.py"
    exit 1
fi

# Sanity check: kmeans centers must exist for ra-* variants
CENTERS_DIR="outputs/kmeans_centers/sam2_ablation"
if [ ! -d "$CENTERS_DIR" ]; then
    echo "[ERROR] K-means centers not found at: $CENTERS_DIR"
    echo "Run first: bash scripts/kmeans/core/run_kmeans_sam2_ablation.sh"
    exit 1
fi

echo "============================================================"
echo " SpecLens SAMv2 Ablation (7 variants, shared activation buf)"
echo " Config:  $CONFIG"
echo " Log:     $LOG_FILE"
echo " K-means: $CENTERS_DIR"
echo " GPUs:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
echo "============================================================"
echo ""

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training... (logs -> $LOG_FILE)"
echo ""

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29502 \
    scripts/train_sae_config.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
