# Random Subsample Implementation

## Overview

Implemented a new `random_subsample` feature in `UniversalActivationStore` to replace stride-based sampling and eliminate spatial bias.

## Problem Statement

The previous stride-based mechanism caused spatial bias in token sampling:
- Example: 64×64 tokens (4096 total) with stride=8
- Only samples from 8 out of 64 columns (12.5% column coverage)
- Creates vertical column patterns in sampled data
- Systematic bias affects feature learning and SAE training

## Solution

Replaced stride logic with deterministic random subsampling per inference, maintaining:
- **Determinism**: Same seed → same samples (reproducible)
- **Per-sample consistency**: Each sample in batch gets independently shuffled
- **Epoch variation**: Different epochs produce different samples
- **Backward compatibility**: Stride logic remains as fallback

## Implementation Details

### Location
File: `/home/sangyu/Desktop/Master/General_SAE_project/src/core/sae/activation_stores/universal_activation_store.py`
Method: `_collect_hook()` (lines 926-1010)

### Configuration Parameters

Two new config parameters added:

1. **`random_subsample_rate`** (float, default=1.0)
   - Rate of tokens to keep (0.0 < rate ≤ 1.0)
   - Example: 0.125 = keep 12.5% of tokens (equivalent to stride=8)

2. **`random_subsample_seed`** (int, default=self.rs_seed)
   - Seed for random number generator
   - Per-layer custom seeds supported

### Priority Logic

```
if random_subsample_rate < 1.0:
    → Use random subsampling
elif stride > 1:
    → Use stride-based sampling (fallback for backward compatibility)
else:
    → No subsampling
```

### Algorithm

#### With Provenance (per-sample shuffle):
```python
for each unique sample_id in batch:
    sample_tokens = get_tokens_for_sample(sample_id)
    n_keep = max(1, int(len(sample_tokens) * subsample_rate))

    # Deterministic shuffle per sample+epoch
    seed = stable_u64(f"{subsample_seed}|{layer_name}|{sample_id}|{epoch}")
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(sample_tokens), generator=rng)[:n_keep]

    keep_tokens_from_sample(indices)
```

#### Without Provenance (global shuffle):
```python
n_keep = max(1, int(total_tokens * subsample_rate))
seed = subsample_seed + epoch
rng = torch.Generator().manual_seed(seed)
indices = torch.randperm(total_tokens, generator=rng)[:n_keep]
```

### Key Features

1. **Per-Sample Independence**: Each sample in batch gets independently shuffled
2. **Epoch Variation**: Seed includes epoch index for different samples per epoch
3. **Layer-Specific**: Different layers can have different seeds/rates
4. **Sorted Output**: Indices are sorted after selection to maintain spatial locality

## Configuration Examples

### YAML Format

#### Basic Usage:
```yaml
queue:
  layers:
    encoder.layer.0.attn.output:
      random_subsample_rate: 0.125  # Keep 12.5% of tokens
      random_subsample_seed: 42      # Optional: custom seed
```

#### Legacy Stride (still works):
```yaml
queue:
  layers:
    encoder.layer.0.attn.output:
      stride: 8  # Fallback to stride-based sampling
```

#### Mixed Strategy:
```yaml
queue:
  layers:
    encoder.layer.0.attn.output:
      random_subsample_rate: 0.125  # Random subsampling
    encoder.layer.1.attn.output:
      stride: 8                      # Legacy stride
    encoder.layer.2.attn.output:
      random_subsample_rate: 0.25    # Different rate
      random_subsample_seed: 99      # Different seed
```

## Testing

Created three test scripts to verify implementation:

### 1. Basic Functionality Test
**File**: `scripts/test_random_subsample.py`

Tests:
- ✅ Random subsampling works correctly
- ✅ Indices are not sequential (scattered distribution)
- ✅ Each sample gets correct number of tokens
- ✅ Reproducibility (same seed → same indices)
- ✅ Epoch variation (different epoch → different indices)
- ✅ Fallback mode (no provenance) works

Results:
```
Original tokens: 4096
Subsample rate: 0.125
Expected tokens: 512
Final subsampled tokens: 512 ✓

First 20 indices: [0, 14, 18, 21, 27, 32, 42, 46, ...] (non-sequential ✓)
Reproducibility: PASSED ✓
Epoch variation: PASSED ✓
```

### 2. Spatial Bias Comparison
**File**: `scripts/test_random_vs_stride.py`

Compares stride vs random subsampling on 64×64 spatial grid:

**Stride Sampling (bias)**:
- Unique columns: 8/64 (12.5%) ❌
- Rows with samples: 64/64
- Columns with samples: 8/64 ← **SPATIAL BIAS**

**Random Subsampling (no bias)**:
- Unique columns: 64/64 (100%) ✅
- Rows with samples: 64/64
- Columns with samples: 64/64 ← **FULL COVERAGE**

### 3. Configuration Integration
**File**: `scripts/test_config_integration.py`

Tests:
- ✅ Config parameter extraction
- ✅ Priority logic (random > stride > none)
- ✅ Backward compatibility
- ✅ Mixed configurations
- ✅ YAML examples

## Benefits

1. **Eliminates Spatial Bias**: All spatial locations equally likely to be sampled
2. **Better Coverage**: Samples distributed across entire feature map
3. **Deterministic**: Reproducible results via seeding
4. **Flexible**: Per-layer rates and seeds
5. **Backward Compatible**: Existing stride configs still work
6. **Epoch Diversity**: Different samples each epoch while maintaining determinism

## Performance

- No significant overhead compared to stride-based sampling
- Tensor operations are efficient (GPU-accelerated)
- Sorting indices maintains spatial locality for better cache performance

## Migration Guide

To migrate from stride to random subsampling:

**Before**:
```yaml
queue:
  layers:
    encoder.layer.0.attn.output:
      stride: 8
```

**After**:
```yaml
queue:
  layers:
    encoder.layer.0.attn.output:
      random_subsample_rate: 0.125  # Equivalent to stride=8
      random_subsample_seed: 42      # For reproducibility
```

## Future Enhancements

Possible future improvements:
1. Adaptive subsampling rates based on loss/gradient magnitude
2. Stratified sampling by spatial regions
3. Importance-weighted sampling based on activation magnitudes
4. Multi-resolution sampling (different rates for different scales)

## Summary

Successfully implemented random subsampling in `UniversalActivationStore`:
- ✅ Eliminates spatial bias from stride-based sampling
- ✅ Deterministic and reproducible
- ✅ Backward compatible
- ✅ Thoroughly tested
- ✅ Production-ready

The implementation provides better spatial coverage while maintaining all benefits of deterministic sampling (reproducibility, epoch variation, per-sample consistency).
