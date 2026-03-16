#!/bin/bash
# ==============================================================================
# K-means Initialization Pipeline
# ==============================================================================
#
# This script orchestrates the complete K-means initialization workflow:
#   1. Extract activations from inference (with auto-probing)
#   2. Train K-means on extracted activations
#   3. Copy centroids to centers directory
#   4. Generate summary report
#
# Usage:
#   ./scripts/batch_kmeans_pipeline.sh
#
# Configuration:
#   Edit the "USER CONFIGURATION" section below before running.
#
# ==============================================================================
# EXAMPLE CONFIGURATIONS
# ==============================================================================
#
# SAMv2 Example (4096 tokens/inference on primary layer):
# --------------------------------------------------------
#   CONFIG="configs/sam2_sav_train.yaml"
#   DATA_DIR="data/activations"
#   CENTERS_DIR="data/kmeans_centers/sam2"
#   EXTRACTION_NAME="sam2_extraction_10M"
#   PRIMARY_LAYER="image_encoder.trunk.blocks.23"
#   TARGET_TOKENS_PRIMARY=10000000
#   LAYERS=(
#       "image_encoder.trunk.blocks.11"
#       "image_encoder.trunk.blocks.23"
#       "image_encoder.trunk.blocks.35"
#   )
#   declare -A CLUSTERS=(
#       ["image_encoder.trunk.blocks.11"]=16384
#       ["image_encoder.trunk.blocks.23"]=16384
#       ["image_encoder.trunk.blocks.35"]=16384
#   )
#   declare -A SUBSAMPLE_RATES=(
#       ["image_encoder.trunk.blocks.11"]=0.5
#       ["image_encoder.trunk.blocks.35"]=0.5
#   )
#
# CLIP Example (197 tokens/inference on primary layer):
# ------------------------------------------------------
#   CONFIG="configs/clip_sav_train.yaml"
#   DATA_DIR="data/activations"
#   CENTERS_DIR="data/kmeans_centers/clip"
#   EXTRACTION_NAME="clip_extraction_10M"
#   PRIMARY_LAYER="visual.transformer.resblocks.11"
#   TARGET_TOKENS_PRIMARY=10000000
#   LAYERS=(
#       "visual.transformer.resblocks.5"
#       "visual.transformer.resblocks.8"
#       "visual.transformer.resblocks.11"
#   )
#   declare -A CLUSTERS=(
#       ["visual.transformer.resblocks.5"]=8192
#       ["visual.transformer.resblocks.8"]=8192
#       ["visual.transformer.resblocks.11"]=8192
#   )
#
# Mask2Former Example:
# --------------------
#   CONFIG="configs/mask2former_sav_train.yaml"
#   DATA_DIR="data/activations"
#   CENTERS_DIR="data/kmeans_centers/mask2former"
#   EXTRACTION_NAME="m2f_extraction_10M"
#   PRIMARY_LAYER="sem_seg_head.predictor.transformer_self_attention_layers.2"
#   TARGET_TOKENS_PRIMARY=10000000
#   LAYERS=(
#       "sem_seg_head.predictor.transformer_self_attention_layers.0"
#       "sem_seg_head.predictor.transformer_self_attention_layers.2"
#       "sem_seg_head.predictor.transformer_cross_attention_layers.0"
#   )
#   declare -A CLUSTERS=(
#       ["sem_seg_head.predictor.transformer_self_attention_layers.0"]=4096
#       ["sem_seg_head.predictor.transformer_self_attention_layers.2"]=4096
#       ["sem_seg_head.predictor.transformer_cross_attention_layers.0"]=4096
#   )
#
# ==============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

# Configuration file
CONFIG="configs/sam2_sav_train.yaml"

# Directory paths
DATA_DIR="data/activations"
CENTERS_DIR="data/kmeans_centers/sam2"

# Extraction settings
EXTRACTION_NAME="sam2_extraction_10M"
PRIMARY_LAYER="image_encoder.trunk.blocks.23"
TARGET_TOKENS_PRIMARY=10000000

# Layers to extract (space-separated list)
LAYERS=(
    "image_encoder.trunk.blocks.11"
    "image_encoder.trunk.blocks.23"
    "image_encoder.trunk.blocks.35"
)

# Number of clusters per layer
declare -A CLUSTERS=(
    ["image_encoder.trunk.blocks.11"]=16384
    ["image_encoder.trunk.blocks.23"]=16384
    ["image_encoder.trunk.blocks.35"]=16384
)

# Optional: Subsample rates per layer (omit or set empty for no subsampling)
# This allows collecting fewer tokens on non-primary layers
declare -A SUBSAMPLE_RATES=(
    ["image_encoder.trunk.blocks.11"]=0.5
    ["image_encoder.trunk.blocks.35"]=0.5
)

# ==============================================================================
# DERIVED VARIABLES
# ==============================================================================

EXTRACTION_DIR="$DATA_DIR/$EXTRACTION_NAME"
CHECKPOINT_PATH="$EXTRACTION_DIR/checkpoint.json"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

print_header() {
    echo ""
    echo "=============================================================================="
    echo "$1"
    echo "=============================================================================="
    echo ""
}

print_section() {
    echo ""
    echo "------------------------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------------------------"
}

check_error() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1"
        exit 1
    fi
}

sanitize_layer_name() {
    local layer="$1"
    # Replace / with _ and . with _
    layer="${layer//\//_}"
    layer="${layer//./_}"
    echo "$layer"
}

# ==============================================================================
# PHASE 1: EXTRACT ACTIVATIONS
# ==============================================================================

print_header "PHASE 1: EXTRACTING ACTIVATIONS"

echo "Configuration:"
echo "  Config file:          $CONFIG"
echo "  Output directory:     $EXTRACTION_DIR"
echo "  Primary layer:        $PRIMARY_LAYER"
echo "  Target tokens:        $TARGET_TOKENS_PRIMARY"
echo "  Number of layers:     ${#LAYERS[@]}"
echo ""

# Build the extraction command
EXTRACTION_CMD="python scripts/extract_activations_for_kmeans.py \
    --config \"$CONFIG\" \
    --output-dir \"$EXTRACTION_DIR\" \
    --primary-layer \"$PRIMARY_LAYER\" \
    --target-tokens-primary $TARGET_TOKENS_PRIMARY \
    --auto-probe"

# Add layers
for layer in "${LAYERS[@]}"; do
    EXTRACTION_CMD="$EXTRACTION_CMD --layers \"$layer\""
done

# Add subsample rates if specified
for layer in "${LAYERS[@]}"; do
    if [ -n "${SUBSAMPLE_RATES[$layer]:-}" ]; then
        EXTRACTION_CMD="$EXTRACTION_CMD --subsample-rate \"$layer\" ${SUBSAMPLE_RATES[$layer]}"
    fi
done

echo "Running extraction command..."
echo ""
eval $EXTRACTION_CMD
check_error "Activation extraction failed"

print_section "Extraction completed successfully"

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint file not found at $CHECKPOINT_PATH"
    echo "Extraction may have failed silently."
    exit 1
fi

echo "Checkpoint file found: $CHECKPOINT_PATH"

# ==============================================================================
# PHASE 2: TRAIN K-MEANS
# ==============================================================================

print_header "PHASE 2: TRAINING K-MEANS MODELS"

echo "Training K-means for ${#LAYERS[@]} layers..."
echo ""

KMEANS_SUCCESS_COUNT=0
KMEANS_FAIL_COUNT=0

for layer in "${LAYERS[@]}"; do
    print_section "Layer: $layer"

    # Sanitize layer name for filesystem
    layer_safe=$(sanitize_layer_name "$layer")

    # Get number of clusters
    n_clusters=${CLUSTERS[$layer]:-}
    if [ -z "$n_clusters" ]; then
        echo "WARNING: No cluster count specified for layer '$layer', skipping..."
        KMEANS_FAIL_COUNT=$((KMEANS_FAIL_COUNT + 1))
        continue
    fi

    # Check if data directory exists
    layer_data_dir="$EXTRACTION_DIR/$layer_safe"
    if [ ! -d "$layer_data_dir" ]; then
        echo "ERROR: Data directory not found: $layer_data_dir"
        echo "Skipping this layer..."
        KMEANS_FAIL_COUNT=$((KMEANS_FAIL_COUNT + 1))
        continue
    fi

    echo "Data directory:  $layer_data_dir"
    echo "Clusters:        $n_clusters"
    echo ""

    # Run K-means training
    echo "Training K-means..."
    python scripts/train_kmeans_centers.py \
        --data-dir "$layer_data_dir" \
        --n-clusters "$n_clusters" \
        --output centroids.pt

    if [ $? -ne 0 ]; then
        echo "ERROR: K-means training failed for layer '$layer'"
        KMEANS_FAIL_COUNT=$((KMEANS_FAIL_COUNT + 1))
        continue
    fi

    # Verify centroids file exists
    if [ ! -f "$layer_data_dir/centroids.pt" ]; then
        echo "ERROR: Centroids file not created: $layer_data_dir/centroids.pt"
        KMEANS_FAIL_COUNT=$((KMEANS_FAIL_COUNT + 1))
        continue
    fi

    # Copy centroids to centers directory
    centers_subdir="$CENTERS_DIR/$layer_safe"
    mkdir -p "$centers_subdir"
    cp "$layer_data_dir/centroids.pt" "$centers_subdir/"
    check_error "Failed to copy centroids to $centers_subdir"

    echo "Centroids saved to: $centers_subdir/centroids.pt"
    KMEANS_SUCCESS_COUNT=$((KMEANS_SUCCESS_COUNT + 1))
    echo ""
done

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print_header "SUMMARY REPORT"

echo "K-means Training Results:"
echo "  Successful:  $KMEANS_SUCCESS_COUNT / ${#LAYERS[@]}"
echo "  Failed:      $KMEANS_FAIL_COUNT / ${#LAYERS[@]}"
echo ""

# Generate detailed report from checkpoint
if [ -f "$CHECKPOINT_PATH" ]; then
    echo "Extraction Statistics:"
    echo ""
    python -c "
import json
import sys
from pathlib import Path

checkpoint_path = Path('$CHECKPOINT_PATH')
try:
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    print(f'  Inferences completed:  {checkpoint.get(\"inferences_completed\", 0):,}')
    print(f'  Extraction complete:   {checkpoint.get(\"extraction_complete\", False)}')
    print()

    layer_stats = checkpoint.get('layer_stats', {})
    if layer_stats:
        print('  Per-Layer Token Counts:')
        for layer, stats in sorted(layer_stats.items()):
            tokens = stats.get('tokens_collected', 0)
            print(f'    {layer:60s}: {tokens:>12,} tokens')
    else:
        print('  No layer statistics found in checkpoint.')

except FileNotFoundError:
    print(f'ERROR: Checkpoint file not found: {checkpoint_path}', file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f'ERROR: Failed to parse checkpoint JSON: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"
    check_error "Failed to generate extraction statistics"
else
    echo "WARNING: Checkpoint file not found, skipping extraction statistics."
fi

echo ""
echo "Centroids Directory:"
echo "  $CENTERS_DIR"
echo ""

# List all centroid files
if [ -d "$CENTERS_DIR" ]; then
    echo "Available Centroids:"
    find "$CENTERS_DIR" -name "centroids.pt" -type f | while read -r centroid_file; do
        rel_path="${centroid_file#$CENTERS_DIR/}"
        size=$(du -h "$centroid_file" | cut -f1)
        echo "  $rel_path (${size})"
    done
else
    echo "WARNING: Centers directory not found: $CENTERS_DIR"
fi

echo ""

# ==============================================================================
# COMPLETION
# ==============================================================================

if [ $KMEANS_FAIL_COUNT -eq 0 ]; then
    print_header "PIPELINE COMPLETED SUCCESSFULLY"
    echo "All K-means models trained and centroids saved."
    echo ""
    echo "Next steps:"
    echo "  1. Review centroids in: $CENTERS_DIR"
    echo "  2. Update your SAE training config to use these centroids"
    echo "  3. Run SAE training with the initialized centers"
    echo ""
    exit 0
else
    print_header "PIPELINE COMPLETED WITH ERRORS"
    echo "Some K-means models failed to train."
    echo "Review the error messages above and retry failed layers."
    echo ""
    exit 1
fi
