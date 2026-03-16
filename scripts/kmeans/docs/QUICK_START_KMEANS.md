# Quick Start: K-means Activation Extraction

## 1. Extract Activations (30 seconds setup)

```bash
python scripts/extract_activations_for_kmeans.py \
    --config configs/mask2former_sav_train.yaml \
    --output-dir outputs/kmeans_activations \
    --primary-layer "model.pixel_level_module.decoder.mask_projection" \
    --target-tokens-primary 10000000 \
    --layers "model.pixel_level_module.decoder.mask_projection" \
    --auto-probe \
    --verbose
```

**What this does:**
- Loads your model and dataset from config
- Probes to measure tokens per inference
- Extracts 10M tokens for the primary layer
- Saves activations as chunks (resumable)

**Time estimate:** ~30-60 minutes depending on model/dataset

---

## 2. Check Progress

```bash
python scripts/load_kmeans_activations.py outputs/kmeans_activations
```

**Output:**
```
Primary layer: model.pixel_level_module.decoder.mask_projection
Target tokens: 10,000,000
Inferences completed: 2,440 / 2,440
Completed: True
Tokens collected: 9,994,240
```

---

## 3. Initialize SAE with K-means

```bash
python scripts/example_kmeans_init_sae.py \
    --activations-dir outputs/kmeans_activations \
    --layer "model.pixel_level_module.decoder.mask_projection" \
    --dict-size 4096 \
    --n-samples 5000000 \
    --output outputs/kmeans_init_weights.pt
```

**Time estimate:** ~5-10 minutes

---

## 4. Use in SAE Training

```python
import torch

# Load initialized weights
data = torch.load("outputs/kmeans_init_weights.pt")
W_dec_init = data["W_dec"]  # (dict_size, feature_dim)

# Initialize your SAE
sae.W_dec.data.copy_(W_dec_init)
sae.W_enc.data.copy_(W_dec_init.T)  # Transpose for encoder
```

---

## Resume if Interrupted

```bash
# Same command, just add --resume
python scripts/extract_activations_for_kmeans.py \
    --config configs/mask2former_sav_train.yaml \
    --output-dir outputs/kmeans_activations \
    --primary-layer "model.pixel_level_module.decoder.mask_projection" \
    --target-tokens-primary 10000000 \
    --layers "model.pixel_level_module.decoder.mask_projection" \
    --resume
```

---

## Common Issues

### "No tokens collected during probing"
- Layer name might be wrong
- Increase `--num-probe-batches 20`
- Check layer exists: See `outputs/hook_specs.txt`

### "Out of memory"
- Reduce `--flush-every 8192`
- Use fewer layers at once
- Enable GPU memory clearing in config

### "K-means too slow"
- Reduce `--n-samples 1000000`
- Increase `--kmeans-batch-size 8192`
- Use fewer clusters (`--dict-size 2048`)

---

## File Sizes

For a 256-dimensional layer:
- **1M tokens** = ~1 GB
- **10M tokens** = ~10 GB
- **50M tokens** = ~50 GB

Plan disk space accordingly!

---

## Full Documentation

See `KMEANS_EXTRACTION_README.md` for:
- Advanced options (subsample rates, etc.)
- Checkpoint format details
- Python API usage
- Troubleshooting guide
- Performance tuning

---

## Test Installation

```bash
python scripts/test_extraction_pipeline.py
```

Should output: **"All tests passed! ✓"**
