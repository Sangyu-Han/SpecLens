# SpecLens

A universal framework for training **Sparse Autoencoders (SAEs)** on any PyTorch model, with built-in support for feature indexing, attribution maps, circuit analysis, and output contribution.

## Key Design Principles

### 1. Spec-Based Model Hooking — Access Any Layer in Any Model

Traditional SAE libraries (e.g., SAELens) require model-specific adapters for each architecture. This toolkit introduces a **Spec string** system that hooks into any PyTorch module hierarchy without writing custom adapters:

```
"model.blocks.9"                     # hook after blocks[9] forward
"model.blocks.9@2"                   # hook after the 2nd output element
"model.image_encoder.trunk@3"        # named branch output
"model.memory_attention.layers.3"    # deep nested module
"enc.pos::forward_with_coords@0"     # custom method, first return value
"model.blocks.9::sae_layer#latent"   # attribute of a method return
```

The Spec parser (`src/core/hooks/spec.py`) resolves module paths, method calls, and output branches uniformly — no model-specific glue code required.

### 2. Single-Pass Multi-Layer Training

The biggest bottleneck in SAE training is **model forward passes**. Existing approaches train one layer at a time, repeating the full model forward for each layer.

This toolkit trains **all target layers simultaneously in a single model forward pass**:

```
One forward pass → activations captured at all N layers
                 → N SAEs updated in parallel (DDP owner-stream)
                 → ~N× speedup in data throughput
```

This is implemented via `UniversalActivationStore` ([src/core/sae/activation_stores/universal_activation_store.py](src/core/sae/activation_stores/universal_activation_store.py)), which:

- Registers forward hooks for all specified layers at once
- Uses a **block-based CPU queue** (inspired by vLLM paged KV cache) to buffer activations memory-efficiently
- Streams activations to each SAE's owner GPU via P2P transfers (`owner_stream` mode)
- Adapts chunk sizes dynamically based on available GPU memory

### 3. RA-Archetypal SAE Variants

Beyond vanilla TopK and JumpReLU, this toolkit includes **RA-Archetypal** SAEs (`ra-batchtopk`, `ra-jumprelu`) that address bias features by:

- Centering the activation space via global mean subtraction (`input_global_center_norm`)
- K-means initialization of decoder columns (archetypal initialization)
- Per-feature frequency monitoring and EMA-based dead feature reanimation
- Diversity-sampled activation buffers to prevent frequent-token dominance

---

## Supported Models

Out of the box via model packs (`src/packs/`):

| Model | Pack | Notes |
|---|---|---|
| CLIP ViT-B/16 | `clip` | All transformer blocks |
| SAM2 (Hiera-T) | `sam2` | Trunk, memory attention, mask decoder |
| DINOv2 | `dinov2` | Via timm |
| ResNet-18 | `resnet` | Residual blocks |
| Mask2Former | `mask2former` | Pixel decoder, transformer |

Any other PyTorch model can be added by writing a small factory in `src/packs/`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/sae-toolkit.git
cd sae-toolkit

# 2. Install PyTorch with CUDA (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install the toolkit
pip install -e .

# For SAM2 support:
pip install -e ".[sam2]"

# 4. (Optional) Install SAM2 from third_party
pip install -e third_party/sam2_src/
```

---

## Usage

### Training SAEs

Train SAEs on **all layers simultaneously** with a single config:

```bash
# Single GPU
python scripts/train_sae_config.py --config configs/clip_imagenet_ra-batchtopk_debiased.yaml

# Multi-GPU (DDP) — recommended
torchrun --nproc_per_node=4 scripts/train_sae_config.py \
    --config configs/clip_imagenet_ra-batchtopk_debiased.yaml
```

The config specifies which layers to train:

```yaml
sae:
  layers:
    - model.blocks.1
    - model.blocks.3
    - model.blocks.5
    - model.blocks.7
    - model.blocks.9
    - model.blocks.11
  training:
    sae_type: ra-batchtopk
    expansion_factor: 32
    k: 128
    num_training_steps: 200_000
```

All listed layers are trained **simultaneously** — the model runs one forward pass per batch, and all SAEs update in parallel.

#### SAE Types

| `sae_type` | Description |
|---|---|
| `vanilla` | Linear encoder + ReLU, L1 penalty |
| `topk` | Hard top-K sparsity |
| `batch-topk` | Batch-level top-K (more stable) |
| `jumprelu` | Per-feature learned threshold (JumpReLU) |
| `ra-batchtopk` | RA-Archetypal + BatchTopK |
| `ra-jumprelu` | RA-Archetypal + JumpReLU |

### K-Means Initialization (Recommended for RA variants)

RA-Archetypal SAEs use K-means cluster centroids to initialize decoder columns, producing more semantically coherent features:

```bash
# Step 1: Extract activations for K-means
python scripts/kmeans/core/train_kmeans_centers.py \
    --config configs/clip_imagenet_ra-batchtopk_debiased.yaml \
    --out_dir /path/to/kmeans_centers

# Or use the shell script
bash scripts/kmeans/run_kmeans.sh
```

Then point your config to the centroids:

```yaml
sae:
  training:
    kmeans_init:
      enabled: true
      centroids_dir: /path/to/kmeans_centers
```

### Feature Indexing

After training, index which dataset samples activate each feature most strongly:

```bash
# Single GPU
python scripts/sae_index_main.py --config configs/sam2_sav_feature_index.yaml

# Multi-GPU DDP (required for large datasets — use PYTHONUNBUFFERED for live logs)
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 \
    scripts/sae_index_main.py --config configs/sam2_sav_feature_index.yaml
```

Output is a **Parquet ledger** (`indexing.out_dir`) with top-N activating samples per feature, for use in visualization dashboards.

```yaml
indexing:
  out_dir: outputs/sae_index
  mode: topn          # "topn" (fast) or "decile"
  top_n: 300          # top-N samples per feature
  track_frequency: true
```

### Attribution Maps

Compute per-pixel attribution maps showing which input regions drive a target SAE feature:

```python
from src.core.runtime.attribution_runtime import AttributionRuntime

runtime = AttributionRuntime(model, config)
attribution = runtime.compute(
    inputs=image_batch,
    target_spec="model.blocks.9::sae_layer#latent",
    target_unit=42,
    method="integrated_gradients",  # or "gradient", "input_x_gradient"
)
```

#### Spec-Based Target Selection

The same Spec system used for training hooks also selects attribution targets:

```
"model.blocks.9"                  # attribution w.r.t. block 9 output
"model.blocks.9::sae_layer#latent" # attribution w.r.t. feature 42 activation
"model.image_encoder.trunk@3"     # SAM2 trunk stage 3
```

### Output Contribution (JVP)

Compute how much each SAE feature contributes to the final output using JVP (Jacobian-vector products) — no full Jacobian computation required:

```bash
python scripts/output_contribution_compare_sam2.py \
    --config configs/sam2_sav_feature_index.yaml \
    --layer model.image_encoder.trunk@3 \
    --feature_id 42
```

This efficiently decomposes the output into per-feature contributions via the chain rule:

```
output ≈ Σ_i  contribution_i
contribution_i = J_output · (W_dec[:, i] * activation_i)
```

### Circuit / Feature Graph Building

> **Note:** The circuit analysis module is still a work in progress and not yet fully polished. The API may change.

Build a directed attribution graph showing how SAE features at different layers causally influence each other:

```bash
python scripts/build_clip_circuit.py \
    --config configs/clip_circuit_tree_fordebug.yaml
```

The circuit config specifies the graph topology — which layers are nodes, and how edges are weighted by attribution:

```yaml
tree:
  feature_graph:
    layers:
      - model.blocks.9
      - model.blocks.11
    attribution_method: integrated_gradients
    edge_threshold: 0.01
    pruning: topk
    topk_edges: 20
```

The output is a Sankey diagram or JSON edge list for downstream visualization.

---

## Configuration Reference

Configs are YAML files in `configs/`. Key sections:

```yaml
# Which dataset/model to use
dataset:
  builder: src.packs.clip.train.factories:build_dataset
  root: /path/to/imagenet

model:
  loader: src.packs.clip.train.factories:load_model
  name: vit_base_patch16_clip_224

# SAE training
sae:
  layers:
    - model.blocks.9
    - model.blocks.11

  output:
    save_path: /path/to/sae_checkpoints

  training:
    sae_type: ra-batchtopk
    expansion_factor: 32   # dict_size = act_size × expansion_factor
    k: 128                  # sparsity level
    num_training_steps: 200_000
    lr: 3.0e-4

    # Debiased training options
    input_global_center_norm: true   # center + normalize inputs
    b_dec_init: "mean"               # init decoder bias to global mean

    # K-means initialization
    kmeans_init:
      enabled: true
      centroids_dir: /path/to/kmeans_centers

    # Activation buffer
    queue:
      block_size_tokens: 16384
      mix_buffer_batches: 16
      diversity:
        enabled: true    # diversity sampling to reduce frequent-token bias
        k: 128

logging:
  wandb:
    enabled: true
    project: my-sae-project
    entity: null   # Set your W&B entity
```

---

## Architecture Overview

```
src/
├── core/
│   ├── hooks/spec.py              # Spec string parser — model-agnostic layer hooking
│   ├── sae/
│   │   ├── base.py                # BaseAutoencoder
│   │   ├── registry.py            # SAE factory & registration
│   │   ├── variants/              # TopK, BatchTopK, JumpReLU, RA-Archetypal
│   │   ├── activation_stores/
│   │   │   └── universal_activation_store.py  # Single-pass multi-layer buffer
│   │   ├── kmeans/                # Faiss GPU K-means initialization
│   │   └── train/runner.py        # DDP training loop
│   ├── indexing/
│   │   └── index_runner.py        # Feature indexing pipeline (Parquet output)
│   ├── runtime/
│   │   └── attribution_runtime.py # IG / Grad / JVP attribution
│   ├── attribution/               # Attribution backends + feature contributions
│   └── circuits/                  # Feature graph / circuit building
├── models/                        # Model wrapper interfaces
└── packs/                         # Model-specific factories (CLIP, SAM2, DINOv2, ...)

scripts/
├── train_sae_config.py            # Training entry point
├── sae_index_main.py              # Indexing entry point
├── kmeans/                        # K-means center extraction
├── output_contribution_compare_sam2.py   # JVP output contribution
├── build_clip_circuit.py          # Circuit building
├── cache_activation_baselines.py  # Baseline caching for attribution
└── viz_attribution_spatial_maps.py       # Attribution visualization

configs/
├── clip_imagenet_train.yaml               # CLIP ViT-B/16 training
├── clip_imagenet_ra-batchtopk_debiased.yaml  # Debiased RA-BatchTopK
├── sam2_sav_batchtopk_train_batchtopk.yaml   # SAM2 training
├── sam2_sav_feature_index.yaml            # SAM2 feature indexing
└── clip_circuit_tree_fordebug.yaml        # Circuit building
```

---

## Comparison with SAELens

| Feature | SAELens | This Toolkit |
|---|---|---|
| Model support | HuggingFace Transformers (with adapters) | Any PyTorch model via Spec strings |
| Layer training | One layer at a time | All layers in one forward pass |
| Training efficiency | N forward passes for N layers | 1 forward pass for N layers |
| Activation buffer | In-memory | Block-based CPU queue with disk spill |
| Multi-GPU | Limited | DDP with per-layer GPU ownership |
| Feature indexing | Basic | Parquet ledger, top-N or decile modes |
| Attribution | — | IG, Gradient, InputxGrad, JVP |
| Circuit analysis | — | Feature graph with Sankey export |
| SAE variants | TopK, Standard | + RA-Archetypal, JumpReLU debiased |

---

## Citation

If you use SpecLens in your research, please cite:

```bibtex
@software{speclens2025,
  author  = {Han, Sangyu},
  title   = {SpecLens: Universal SAE Training and Attribution for Any PyTorch Model},
  year    = {2025},
  url     = {https://github.com/Sangyu-Han/SpecLens},
}
```

## License

MIT
