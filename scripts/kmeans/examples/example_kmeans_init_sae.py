#!/usr/bin/env python3
"""
Example: Initialize SAE Dictionary with K-means Clustering

This script demonstrates how to:
1. Load extracted activations
2. Run K-means clustering
3. Initialize SAE dictionary weights with cluster centers

Usage:
    python scripts/example_kmeans_init_sae.py \
        --activations-dir outputs/kmeans_activations \
        --layer "model.pixel_level_module.decoder.mask_projection" \
        --dict-size 4096 \
        --n-samples 5000000

Requirements:
    pip install scikit-learn
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from load_kmeans_activations import (
    load_layer_activations,
    prepare_for_kmeans,
    get_layer_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_kmeans_initialization(
    activations_dir: str,
    layer_name: str,
    dict_size: int,
    n_samples: int = None,
    normalize: bool = True,
    kmeans_batch_size: int = 4096,
    max_iter: int = 100,
    random_state: int = 42,
) -> torch.Tensor:
    """
    Run K-means clustering on extracted activations and return cluster centers.

    Args:
        activations_dir: Directory containing extracted activations
        layer_name: Name of layer to initialize
        dict_size: Number of dictionary elements (K-means clusters)
        n_samples: Number of activation samples to use (None = all)
        normalize: Whether to L2 normalize activations before clustering
        kmeans_batch_size: Batch size for MiniBatchKMeans
        max_iter: Maximum iterations for K-means
        random_state: Random seed

    Returns:
        Tensor of shape (dict_size, feature_dim) containing cluster centers
    """
    logger.info("=" * 80)
    logger.info("K-means SAE Dictionary Initialization")
    logger.info("=" * 80)
    logger.info(f"Activations directory: {activations_dir}")
    logger.info(f"Layer: {layer_name}")
    logger.info(f"Dictionary size: {dict_size:,}")

    # Get layer info
    try:
        layer_info = get_layer_info(activations_dir, layer_name)
        logger.info(f"Layer info:")
        logger.info(f"  Feature dimension: {layer_info['act_size']}")
        logger.info(f"  Total tokens: {layer_info['tokens_collected']:,}")
        logger.info(f"  Chunks: {layer_info['chunks_written']}")
    except Exception as e:
        logger.warning(f"Could not load layer info: {e}")

    # Load activations
    logger.info("")
    logger.info("Loading activations...")
    activations = load_layer_activations(activations_dir, layer_name)
    logger.info(f"Loaded: {activations.shape} ({activations.dtype})")

    # Prepare for K-means
    logger.info("")
    logger.info("Preparing activations for K-means...")
    activations = prepare_for_kmeans(
        activations,
        n_samples=n_samples,
        normalize=normalize,
    )

    # Check we have enough samples
    if activations.shape[0] < dict_size:
        raise ValueError(
            f"Not enough samples ({activations.shape[0]}) for dictionary size ({dict_size}). "
            f"Reduce --dict-size or extract more activations."
        )

    logger.info(f"Final shape: {activations.shape}")
    logger.info(f"Memory usage: {activations.element_size() * activations.numel() / 1e9:.2f} GB")

    # Run K-means
    logger.info("")
    logger.info("Running K-means clustering...")
    logger.info(f"  n_clusters: {dict_size:,}")
    logger.info(f"  batch_size: {kmeans_batch_size:,}")
    logger.info(f"  max_iter: {max_iter}")

    kmeans = MiniBatchKMeans(
        n_clusters=dict_size,
        batch_size=kmeans_batch_size,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
        init="k-means++",
        n_init=3,
    )

    # Convert to numpy for sklearn
    activations_np = activations.numpy()

    # Fit K-means
    logger.info("Fitting K-means (this may take several minutes)...")
    kmeans.fit(activations_np)

    # Get cluster centers
    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    logger.info(f"K-means complete! Centers shape: {centers.shape}")

    # Analyze cluster assignments
    logger.info("")
    logger.info("Analyzing cluster assignments...")
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)

    logger.info(f"  Active clusters: {len(unique)} / {dict_size}")
    logger.info(f"  Min cluster size: {counts.min()}")
    logger.info(f"  Max cluster size: {counts.max()}")
    logger.info(f"  Mean cluster size: {counts.mean():.1f}")
    logger.info(f"  Median cluster size: {np.median(counts):.1f}")

    # Warn about empty clusters
    if len(unique) < dict_size:
        empty = dict_size - len(unique)
        logger.warning(f"  {empty} clusters are empty (no samples assigned)")

    # Inertia (sum of squared distances to closest cluster center)
    logger.info(f"  Inertia: {kmeans.inertia_:.2e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Initialization complete!")
    logger.info("=" * 80)

    return centers


def save_initialized_weights(
    centers: torch.Tensor,
    output_path: str,
    layer_name: str,
    metadata: dict = None,
):
    """
    Save initialized dictionary weights to file.

    Args:
        centers: Cluster centers (dict_size, feature_dim)
        output_path: Path to save weights
        layer_name: Layer name (for metadata)
        metadata: Additional metadata to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "W_dec": centers,
        "layer_name": layer_name,
        "dict_size": centers.shape[0],
        "feature_dim": centers.shape[1],
        "initialization_method": "kmeans",
    }

    if metadata:
        data.update(metadata)

    torch.save(data, output_path)
    logger.info(f"Saved initialized weights to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize SAE dictionary with K-means clustering",
    )

    parser.add_argument(
        "--activations-dir",
        type=str,
        required=True,
        help="Directory containing extracted activations",
    )
    parser.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Layer name to initialize",
    )
    parser.add_argument(
        "--dict-size",
        type=int,
        required=True,
        help="Dictionary size (number of clusters)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't L2 normalize activations before clustering",
    )
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=4096,
        help="Batch size for MiniBatchKMeans (default: 4096)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum K-means iterations (default: 100)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save initialized weights (optional)",
    )

    args = parser.parse_args()

    # Run K-means initialization
    try:
        centers = run_kmeans_initialization(
            activations_dir=args.activations_dir,
            layer_name=args.layer,
            dict_size=args.dict_size,
            n_samples=args.n_samples,
            normalize=not args.no_normalize,
            kmeans_batch_size=args.kmeans_batch_size,
            max_iter=args.max_iter,
            random_state=args.random_seed,
        )

        # Save if output path specified
        if args.output:
            metadata = {
                "n_samples": args.n_samples,
                "normalized": not args.no_normalize,
                "kmeans_batch_size": args.kmeans_batch_size,
                "max_iter": args.max_iter,
                "random_seed": args.random_seed,
            }
            save_initialized_weights(centers, args.output, args.layer, metadata)

        # Print example usage
        logger.info("")
        logger.info("Example usage in SAE initialization:")
        logger.info("  # Load initialized weights")
        logger.info(f"  data = torch.load('{args.output or 'init_weights.pt'}')")
        logger.info("  W_dec_init = data['W_dec']  # Shape: (dict_size, feature_dim)")
        logger.info("")
        logger.info("  # Initialize SAE decoder")
        logger.info("  sae.W_dec.data.copy_(W_dec_init)")
        logger.info("  sae.W_enc.data.copy_(W_dec_init.T)  # Transpose for encoder")

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
