#!/bin/bash
# K-means initialization for SAMv2 ablation (3 layers)
# Run this BEFORE sam2_sav_ablation.yaml training (required for ra-* variants).
#
# Usage: bash scripts/kmeans/core/run_kmeans_sam2_ablation.sh
#
# Estimated time: ~30min (1M tokens x 3 layers)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."   # repo root

PYTHON="/home/sangyu/anaconda3/envs/py312/bin/python"
CONFIG="configs/sam2_sav_ablation.yaml"

# ── paths ────────────────────────────────────────────────────────────────────
EXTRACTION_DIR="outputs/kmeans_data/sam2_ablation"
CENTERS_DIR="outputs/kmeans_centers/sam2_ablation"

# ── layers (must match configs/sam2_sav_ablation.yaml) ───────────────────────
LAYERS=(
    "model.image_encoder.trunk@3"
    "model.memory_attention.layers.3"
    "model.sam_mask_decoder.transformer.layers.1@1"
)
PRIMARY_LAYER="model.image_encoder.trunk@3"
TARGET_TOKENS_PRIMARY=1000000   # 1M tokens; all layers collect equally

# n_clusters per layer:
#   Using 4096 for all layers (< dict_size) to keep W=[dict_size, 4096] manageable.
#   trunk@3: W=[24576, 4096]=384MB (was 2.3GB with n_clusters=24576)
#   256-dim layers: W=[8192, 4096]=128MB (was 256MB with n_clusters=8192)
declare -A N_CLUSTERS=(
    ["model.image_encoder.trunk@3"]=4096
    ["model.memory_attention.layers.3"]=4096
    ["model.sam_mask_decoder.transformer.layers.1@1"]=4096
)

mkdir -p "$EXTRACTION_DIR" "$CENTERS_DIR"

# ── helper ───────────────────────────────────────────────────────────────────
sanitize() { echo "$1"; }   # extraction saves dirs with original layer name (dots/@ preserved)

# ── Phase 1: extract activations ─────────────────────────────────────────────
echo "========================================================"
echo " Phase 1: Extracting activations"
echo "========================================================"

LAYERS_ARG="${LAYERS[*]}"

EXTRACT_CMD="$PYTHON scripts/kmeans/core/extract_activations_for_kmeans.py \
    --config $CONFIG \
    --output-dir $EXTRACTION_DIR \
    --primary-layer $PRIMARY_LAYER \
    --target-tokens-primary $TARGET_TOKENS_PRIMARY \
    --layers $LAYERS_ARG \
    --auto-probe"

echo "Command: $EXTRACT_CMD"
echo ""
eval "$EXTRACT_CMD"

echo ""
echo "Extraction done."

# ── Phase 2: train k-means per layer ─────────────────────────────────────────
echo ""
echo "========================================================"
echo " Phase 2: Training K-means per layer"
echo "========================================================"

for layer in "${LAYERS[@]}"; do
    safe=$(sanitize "$layer")
    data_dir="$EXTRACTION_DIR/$safe"
    centers_subdir="$CENTERS_DIR/$safe"
    n_clusters="${N_CLUSTERS[$layer]}"

    echo ""
    echo "--- $layer (n_clusters=$n_clusters) ---"

    if [ ! -d "$data_dir" ]; then
        echo "  [SKIP] Data dir not found: $data_dir"
        continue
    fi

    $PYTHON scripts/kmeans/core/train_kmeans_centers.py \
        --data-dir "$data_dir" \
        --n-clusters "$n_clusters" \
        --output centroids.pt \
        --unit-norm    # saves global_mean for b_dec init

    mkdir -p "$centers_subdir"
    cp "$data_dir/centroids.pt" "$centers_subdir/centroids.pt"
    echo "  Saved: $centers_subdir/centroids.pt"
done

echo ""
echo "========================================================"
echo " Done! Centroids saved to: $CENTERS_DIR"
echo " Starting SAMv2 ablation training..."
echo "========================================================"

cd "$(dirname "$0")/../../.."   # back to repo root
bash run_ablation_sam2.sh
