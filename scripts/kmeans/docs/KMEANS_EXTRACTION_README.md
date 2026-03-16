# K-means Activation Extraction for SAE Initialization

This directory contains scripts for extracting activations from trained models to use as initialization for SAE dictionary learning via K-means clustering.

## Overview

The extraction pipeline:
1. **Tracks by inferences** (not tokens) for consistent progress across layers
2. **Supports checkpoint-based resumption** for long-running extractions
3. **Handles multiple layers simultaneously** with per-layer buffer management
4. **Provides flexible subsampling** to balance token counts across layers
5. **Uses atomic checkpoint writes** to prevent corruption

## Files

- **`extract_activations_for_kmeans.py`**: Main extraction script
- **`load_kmeans_activations.py`**: Utilities for loading and inspecting extracted activations
- **`example_extract_activations.sh`**: Example usage script
- **`KMEANS_EXTRACTION_README.md`**: This documentation

## Quick Start

### Basic Usage

```bash
python scripts/extract_activations_for_kmeans.py \
    --config configs/mask2former_sav_train.yaml \
    --output-dir outputs/kmeans_activations \
    --primary-layer "model.pixel_level_module.decoder.mask_projection" \
    --target-tokens-primary 10000000 \
    --layers "model.pixel_level_module.decoder.mask_projection" \
             "model.transformer_module.decoder.layers.0@0" \
    --auto-probe
```

### Resume from Checkpoint

If extraction is interrupted, resume with:

```bash
python scripts/extract_activations_for_kmeans.py \
    --config configs/mask2former_sav_train.yaml \
    --output-dir outputs/kmeans_activations \
    --primary-layer "model.pixel_level_module.decoder.mask_projection" \
    --target-tokens-primary 10000000 \
    --layers "model.pixel_level_module.decoder.layers.0@0" \
    --resume
```

### Load Extracted Activations

```python
from scripts.load_kmeans_activations import load_layer_activations

# Load all activations for a layer
activations = load_layer_activations(
    "outputs/kmeans_activations",
    "model.pixel_level_module.decoder.mask_projection"
)
# Shape: (N_tokens, feature_dim)
```

## Command-Line Arguments

### Required Arguments

- `--config`: Path to training config YAML (e.g., `configs/mask2former_sav_train.yaml`)
- `--output-dir`: Base output directory for activations and checkpoints
- `--primary-layer`: Reference layer name for calculating target inferences

### Optional Arguments

- `--target-tokens-primary`: Target number of tokens for primary layer (default: 10M)
- `--layers`: List of layer names to extract (default: all from config)
- `--auto-probe`: Automatically probe tokens per inference before extraction
- `--subsample-rates`: JSON dict of per-layer subsample rates (e.g., `'{"layer1": 0.5}'`)
- `--flush-every`: Flush buffer every N tokens (default: 16384)
- `--checkpoint-every`: Save checkpoint every N inferences (default: 100)
- `--num-probe-batches`: Number of batches for probing (default: 10)
- `--verbose`: Enable verbose (DEBUG level) logging
- `--resume`: Resume from checkpoint if available

## Key Concepts

### Inference-Based Tracking

Unlike token-based tracking, inference-based tracking:
- **Counts forward passes** through the model, not individual tokens
- **Ensures consistency** across layers with different token yields
- **Simplifies progress estimation** since each inference has similar cost

### Auto-Probing

The `--auto-probe` flag runs a brief probing phase to:
1. Collect 10 inference batches
2. Measure average tokens per inference for each layer
3. Calculate target inference count based on `--target-tokens-primary`

Example probe output:
```
Probing tokens per inference with 10 batches...
  model.pixel_level_module.decoder.mask_projection: 4096.0 tokens/inference
  model.transformer_module.decoder.layers.0@0: 100.0 tokens/inference
Target inferences: 2440
```

### Subsample Rates

Use `--subsample-rates` to downsample high-yield layers:

```bash
--subsample-rates '{"model.pixel_level_module.decoder.mask_projection": 1.0, "model.transformer_module.decoder.layers.0@0": 0.25}'
```

This collects:
- 100% of tokens from `mask_projection`
- 25% of tokens from `layers.0@0`

### Checkpoint Format

Checkpoints are saved as JSON in `<output_dir>/checkpoint.json`:

```json
{
  "primary_layer": "model.pixel_level_module.decoder.mask_projection",
  "target_inferences": 2440,
  "target_tokens_primary": 10000000,
  "inferences_completed": 1500,
  "completed": false,
  "layers": {
    "model.pixel_level_module.decoder.mask_projection": {
      "act_size": 256,
      "tokens_per_inference": 4096.0,
      "subsample_rate": 1.0,
      "tokens_collected": 6144000,
      "chunks_written": 375
    }
  }
}
```

### Output Structure

```
outputs/kmeans_activations/
├── checkpoint.json                                    # Checkpoint metadata
├── logs/
│   └── extraction.log                                 # Detailed logs
├── model.pixel_level_module.decoder.mask_projection/  # Layer directory
│   ├── chunk_000000.pt                                # Activation chunks
│   ├── chunk_000001.pt
│   └── ...
└── model.transformer_module.decoder.layers.0_0/       # Another layer
    ├── chunk_000000.pt
    └── ...
```

## Loading Utilities

### Inspect Extraction

```bash
python scripts/load_kmeans_activations.py outputs/kmeans_activations
```

Output:
```
================================================================================
Extraction Summary: outputs/kmeans_activations
================================================================================
Primary layer: model.pixel_level_module.decoder.mask_projection
Target tokens (primary): 10,000,000
Target inferences: 2,440
Inferences completed: 2,440
Completed: True

Layers extracted: 2

Layer: model.pixel_level_module.decoder.mask_projection
  Act size: 256
  Tokens per inference: 4096.0
  Subsample rate: 1.0
  Tokens collected: 9,994,240
  Chunks written: 610

Layer: model.transformer_module.decoder.layers.0@0
  Act size: 256
  Tokens per inference: 100.0
  Subsample rate: 1.0
  Tokens collected: 244,000
  Chunks written: 15
```

### Load Specific Layer

```bash
python scripts/load_kmeans_activations.py outputs/kmeans_activations \
    --layer "model.pixel_level_module.decoder.mask_projection"
```

### Python API

```python
from scripts.load_kmeans_activations import (
    load_checkpoint,
    load_layer_activations,
    load_layer_activations_lazy,
    prepare_for_kmeans,
)

# Load checkpoint metadata
ckpt = load_checkpoint("outputs/kmeans_activations")
print(f"Completed: {ckpt['completed']}")

# Load all activations for a layer
activations = load_layer_activations(
    "outputs/kmeans_activations",
    "model.pixel_level_module.decoder.mask_projection"
)

# Memory-efficient lazy loading
for chunk in load_layer_activations_lazy("outputs/kmeans_activations", layer_name):
    # Process chunk-by-chunk
    print(chunk.shape)

# Prepare for K-means (sampling + normalization)
kmeans_ready = prepare_for_kmeans(
    activations,
    n_samples=1_000_000,  # Sample 1M tokens
    normalize=True,        # L2 normalize
)
```

## Use Case: K-means Initialization

After extraction, use the activations to initialize SAE dictionaries:

```python
import torch
from sklearn.cluster import MiniBatchKMeans
from scripts.load_kmeans_activations import load_layer_activations, prepare_for_kmeans

# Load activations
activations = load_layer_activations(
    "outputs/kmeans_activations",
    "model.pixel_level_module.decoder.mask_projection"
)

# Prepare for K-means
activations = prepare_for_kmeans(
    activations,
    n_samples=5_000_000,  # Use 5M samples
    normalize=True,
)

# Run K-means
n_clusters = 4096  # Dictionary size
kmeans = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=4096,
    max_iter=100,
    verbose=1,
)
kmeans.fit(activations.numpy())

# Use cluster centers as initial dictionary
W_dec_init = torch.from_numpy(kmeans.cluster_centers_)  # (n_clusters, act_size)
print(f"Initialized dictionary: {W_dec_init.shape}")
```

## Performance Considerations

### Memory Management

- **Flush threshold**: Buffers are flushed every 16K tokens by default
- **Chunk size**: Each chunk is 16K tokens max (configurable)
- **CPU storage**: All activations are stored on CPU to avoid GPU memory issues

### Speed Optimization

- **Batch collection**: Multiple inferences can be collected simultaneously
- **Prefilling**: Store is prefilled with 2 batches before extraction starts
- **Checkpoint frequency**: Checkpoints every 100 inferences (configurable)

### Disk Usage

For a layer with feature dimension D:
- Each token: D × 4 bytes (float32)
- 10M tokens: D × 40 MB

Example: 256-dimensional features × 10M tokens = 10.24 GB

## Troubleshooting

### "No tokens collected" during probing

**Cause**: Layer may not produce activations for small batch sizes.

**Solution**: Increase `--num-probe-batches` or check layer name.

### Extraction is very slow

**Cause**: Model inference is the bottleneck.

**Solution**:
- Reduce `--target-tokens-primary`
- Use subsampling for high-yield layers
- Enable AMP in config (if not already enabled)

### Checkpoint corruption

**Cause**: Process killed during checkpoint write.

**Solution**: The script uses atomic writes (temp + rename), so this should be rare. Delete corrupted checkpoint and restart.

### Out of memory

**Cause**: Buffer accumulation before flush.

**Solution**: Reduce `--flush-every` to flush more frequently.

## Advanced Usage

### Custom Subsample Rates

Calculate subsample rates to balance token counts:

```python
# Target: 10M tokens per layer
target = 10_000_000

# Layer yields (from probing)
yields = {
    "layer1": 4096,  # tokens per inference
    "layer2": 100,
}

# Calculate subsample rates
rates = {}
for layer, tpi in yields.items():
    # How many inferences needed?
    n_inferences = target / tpi
    # What rate achieves this?
    rates[layer] = min(1.0, target / (n_inferences * tpi))

print(rates)
# {"layer1": 1.0, "layer2": 1.0}  # Both need full sampling
```

### Parallel Extraction

For multiple independent extraction jobs, run in separate processes:

```bash
# Terminal 1: Extract layers 0-4
python scripts/extract_activations_for_kmeans.py \
    --config configs/config.yaml \
    --output-dir outputs/kmeans_acts_0_4 \
    --layers layer_0 layer_1 layer_2 layer_3 layer_4 \
    ...

# Terminal 2: Extract layers 5-9
python scripts/extract_activations_for_kmeans.py \
    --config configs/config.yaml \
    --output-dir outputs/kmeans_acts_5_9 \
    --layers layer_5 layer_6 layer_7 layer_8 layer_9 \
    ...
```

## Integration with SAE Training

After extraction, initialize SAE dictionaries:

```python
# In your SAE config or initialization code
from scripts.load_kmeans_activations import load_layer_activations
from sklearn.cluster import MiniBatchKMeans

def initialize_sae_with_kmeans(layer_name: str, dict_size: int):
    # Load activations
    acts = load_layer_activations(
        "outputs/kmeans_activations",
        layer_name,
    )

    # Sample and normalize
    acts = prepare_for_kmeans(acts, n_samples=5_000_000, normalize=True)

    # Run K-means
    kmeans = MiniBatchKMeans(n_clusters=dict_size, batch_size=4096)
    kmeans.fit(acts.numpy())

    # Return as torch tensor
    return torch.from_numpy(kmeans.cluster_centers_)

# Use in SAE initialization
W_dec_init = initialize_sae_with_kmeans("model.layer.0", dict_size=4096)
sae.W_dec.data.copy_(W_dec_init)
```

## Changelog

### v1.0.0 (2025-02-09)
- Initial release
- Inference-based tracking
- Checkpoint resumption
- Multi-layer support
- Auto-probing
- Subsample rates
- Loading utilities

## License

Same as parent project.
