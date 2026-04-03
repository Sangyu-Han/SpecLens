#!/bin/bash
# SpecLens TopKSAE Training Script
# Usage: bash run_train.sh [config_path]
#   Default config: configs/clip_imagenet_topk_train.yaml
#
# Handles NCCL P2P hardware issue on this server by using gloo backend.
# If gloo is too slow, try: DIST_BACKEND=nccl bash run_train.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-configs/clip_imagenet_topk_train.yaml}"
LOG_DIR="outputs/logs"
LOG_FILE="$LOG_DIR/train_torchrun.log"
PYTHON="/home/mipal/miniconda3/envs/sae/bin/python"
TORCHRUN="/home/mipal/miniconda3/envs/sae/bin/torchrun"

mkdir -p "$LOG_DIR"

# Check if already running
if pgrep -f "train_sae_config.py" > /dev/null 2>&1; then
    echo "[WARN] Training already running:"
    pgrep -la "train_sae_config.py"
    echo "Kill with: pkill -f train_sae_config.py"
    exit 1
fi

echo "=============================================="
echo " SpecLens TopKSAE Training"
echo " Config: $CONFIG"
echo " Log:    $LOG_FILE"
echo " GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
echo "=============================================="

# 이 서버는 iommu=soft로 CUDA P2P가 비활성화됨.
# NCCL_P2P_DISABLE=1: NCCL이 NVLink P2P 대신 SHM/PCIe 경유로 collective ops 수행.
# torch 2.1.x (NCCL 2.18.x) 필요: 2.5.x의 NCCL 2.21.x는 P2P 체크가 fatal error임.
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4

echo "Starting training... (logs -> $LOG_FILE)"
echo ""

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/train_sae_config.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE"
