#!/bin/bash
# K-means initialization for CLIP ViT-B/16 ablation (7 layers)
# Run this BEFORE clip_imagenet_ablation.yaml training (required for ra-* variants).
#
# Usage: bash scripts/kmeans/core/run_kmeans_clip_ablation.sh
#
# Estimated time: ~1-2h (10M tokens x 7 layers on 2 GPUs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."   # repo root

PYTHON="/home/mipal/miniconda3/envs/sae/bin/python"
CONFIG="configs/clip_imagenet_ablation.yaml"

# ── paths ────────────────────────────────────────────────────────────────────
EXTRACTION_DIR="outputs/kmeans_data/clip_ablation"
CENTERS_DIR="outputs/kmeans_centers/clip_ablation"
# ── layers (must match configs/clip_imagenet_ablation.yaml) ──────────────────
LAYERS=(
    "model.blocks.1"
    "model.blocks.3"
    "model.blocks.5"
    "model.blocks.7"
    "model.blocks.9"
    "model.blocks.10"
    "model.blocks.11"
)
PRIMARY_LAYER="model.blocks.11"
TARGET_TOKENS_PRIMARY=1000000   # 1M tokens; all layers collect equally

# n_clusters = dict_size = act_size(768) * expansion_factor(32) = 24576
N_CLUSTERS=24576

mkdir -p "$EXTRACTION_DIR" "$CENTERS_DIR"

# ── helper ───────────────────────────────────────────────────────────────────
sanitize() { echo "$1"; }   # extraction saves dirs with original layer name (dots preserved)

# ── Phase 1: extract activations ─────────────────────────────────────────────
echo "========================================================"
echo " Phase 1: Extracting activations"
echo "========================================================"

EXTRACT_CMD="$PYTHON scripts/kmeans/core/extract_activations_for_kmeans.py \
    --config $CONFIG \
    --output-dir $EXTRACTION_DIR \
    --primary-layer $PRIMARY_LAYER \
    --target-tokens-primary $TARGET_TOKENS_PRIMARY \
    --auto-probe"

# All layers as a single --layers arg (repeated --layers only keeps the last)
LAYERS_ARG="${LAYERS[*]}"
EXTRACT_CMD="$EXTRACT_CMD --layers $LAYERS_ARG"

# No subsampling — all layers collect equal tokens

echo "Command: $EXTRACT_CMD"
echo ""
eval "$EXTRACT_CMD"

echo ""
echo "Extraction done."

# ── Phase 2: train k-means per layer ─────────────────────────────────────────
echo ""
echo "========================================================"
echo " Phase 2: Training K-means ($N_CLUSTERS clusters/layer)"
echo "========================================================"

for layer in "${LAYERS[@]}"; do
    safe=$(sanitize "$layer")
    data_dir="$EXTRACTION_DIR/$safe"
    centers_subdir="$CENTERS_DIR/$safe"

    echo ""
    echo "--- $layer ---"

    if [ ! -d "$data_dir" ]; then
        echo "  [SKIP] Data dir not found: $data_dir"
        continue
    fi

    $PYTHON scripts/kmeans/core/train_kmeans_centers.py \
        --data-dir "$data_dir" \
        --n-clusters "$N_CLUSTERS" \
        --output centroids.pt \
        --unit-norm    # saves global_mean for b_dec init

    mkdir -p "$centers_subdir"
    cp "$data_dir/centroids.pt" "$centers_subdir/centroids.pt"
    echo "  Saved: $centers_subdir/centroids.pt"
done

echo ""
echo "========================================================"
echo " Done! Centroids saved to: $CENTERS_DIR"
echo " Starting ablation training..."
echo "========================================================"

cd "$(dirname "$0")/../../.."   # back to repo root
bash run_ablation_clip.sh
