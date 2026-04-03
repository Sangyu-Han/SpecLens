#!/bin/bash
# SpecLens Feature Indexing Script
# Run AFTER training is complete.
# Usage: bash run_index.sh [config_path]
#   Default config: configs/clip_imagenet_topk_index.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/clip_imagenet_topk_index.yaml}"
LOG_DIR="outputs/logs"
LOG_FILE="$LOG_DIR/index_torchrun.log"
TORCHRUN="/home/mipal/miniconda3/envs/sae/bin/torchrun"

mkdir -p "$LOG_DIR"

# Verify SAE checkpoints exist
SAE_PATH="outputs/clip_imagenet_topk"
if [ ! -d "$SAE_PATH" ]; then
    echo "[ERROR] SAE checkpoint directory not found: $SAE_PATH"
    echo "Make sure training completed successfully first."
    exit 1
fi

echo "=============================================="
echo " SpecLens Feature Indexing"
echo " Config: $CONFIG"
echo " SAE:    $SAE_PATH"
echo " Log:    $LOG_FILE"
echo "=============================================="

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4

echo "Starting indexing... (logs -> $LOG_FILE)"
echo ""

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/sae_index_main.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
