#!/usr/bin/env python3
"""
Utility script to load and inspect extracted K-means activations.

This script provides helper functions to:
- Load activation chunks from disk
- Inspect checkpoint metadata
- Concatenate chunks into full tensors
- Prepare activations for K-means clustering

Usage:
    from scripts.load_kmeans_activations import load_layer_activations, load_checkpoint

    # Load checkpoint metadata
    ckpt = load_checkpoint("outputs/kmeans_activations")

    # Load activations for a specific layer
    activations = load_layer_activations(
        "outputs/kmeans_activations",
        "model.pixel_level_module.decoder.mask_projection"
    )

    # activations is a (N, D) tensor where N is tokens and D is feature dimension
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def sanitize_layer_name(layer_name: str) -> str:
    """Convert layer name to filesystem-safe directory name."""
    return layer_name.replace("/", "_").replace(":", "__")


def load_checkpoint(output_dir: str | Path) -> Dict:
    """
    Load checkpoint metadata.

    Args:
        output_dir: Path to extraction output directory

    Returns:
        Dictionary containing checkpoint data
    """
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    with open(checkpoint_path, "r") as f:
        return json.load(f)


def get_layer_info(output_dir: str | Path, layer_name: str) -> Dict:
    """
    Get information about a specific layer from checkpoint.

    Args:
        output_dir: Path to extraction output directory
        layer_name: Name of the layer

    Returns:
        Dictionary with layer info (act_size, tokens_collected, chunks_written, etc.)
    """
    ckpt = load_checkpoint(output_dir)
    layers = ckpt.get("layers", {})

    if layer_name not in layers:
        raise ValueError(
            f"Layer '{layer_name}' not found in checkpoint. "
            f"Available layers: {list(layers.keys())}"
        )

    return layers[layer_name]


def list_layer_chunks(output_dir: str | Path, layer_name: str) -> List[Path]:
    """
    List all chunk files for a layer.

    Args:
        output_dir: Path to extraction output directory
        layer_name: Name of the layer

    Returns:
        Sorted list of chunk file paths
    """
    output_dir = Path(output_dir)
    safe_name = sanitize_layer_name(layer_name)
    layer_dir = output_dir / safe_name

    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

    chunks = sorted(layer_dir.glob("chunk_*.pt"))
    return chunks


def load_layer_activations(
    output_dir: str | Path,
    layer_name: str,
    max_chunks: Optional[int] = None,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Load all activation chunks for a layer and concatenate.

    Args:
        output_dir: Path to extraction output directory
        layer_name: Name of the layer
        max_chunks: Maximum number of chunks to load (for testing)
        device: Device to load tensors to
        dtype: Optional dtype to cast tensors to

    Returns:
        Tensor of shape (N, D) where N is total tokens and D is feature dimension
    """
    chunks = list_layer_chunks(output_dir, layer_name)

    if not chunks:
        raise ValueError(f"No chunks found for layer '{layer_name}'")

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    logger.info(f"Loading {len(chunks)} chunks for layer '{layer_name}'...")

    tensors = []
    total_tokens = 0

    for i, chunk_path in enumerate(chunks):
        tensor = torch.load(chunk_path, map_location=device)

        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        tensors.append(tensor)
        total_tokens += tensor.shape[0]

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            logger.info(f"  Loaded {i+1}/{len(chunks)} chunks, {total_tokens:,} tokens")

    logger.info(f"Concatenating {len(tensors)} tensors...")
    result = torch.cat(tensors, dim=0)

    logger.info(f"Final shape: {result.shape} ({result.shape[0]:,} tokens, {result.shape[1]} features)")
    return result


def load_layer_activations_lazy(
    output_dir: str | Path,
    layer_name: str,
    max_chunks: Optional[int] = None,
):
    """
    Lazy iterator over activation chunks (memory efficient).

    Args:
        output_dir: Path to extraction output directory
        layer_name: Name of the layer
        max_chunks: Maximum number of chunks to yield

    Yields:
        Tensors for each chunk

    Example:
        for chunk in load_layer_activations_lazy(output_dir, layer_name):
            # Process chunk
            print(chunk.shape)
    """
    chunks = list_layer_chunks(output_dir, layer_name)

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    for chunk_path in chunks:
        yield torch.load(chunk_path, map_location="cpu")


def prepare_for_kmeans(
    activations: torch.Tensor,
    n_samples: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Prepare activations for K-means clustering.

    Args:
        activations: (N, D) tensor of activations
        n_samples: If specified, randomly sample this many tokens
        normalize: Whether to L2-normalize each activation vector

    Returns:
        Prepared tensor ready for K-means
    """
    if n_samples is not None and n_samples < activations.shape[0]:
        # Random sampling
        indices = torch.randperm(activations.shape[0])[:n_samples]
        activations = activations[indices]
        logger.info(f"Randomly sampled {n_samples:,} tokens from {activations.shape[0]:,}")

    if normalize:
        # L2 normalize each row
        norms = torch.norm(activations, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)  # Avoid division by zero
        activations = activations / norms
        logger.info("Applied L2 normalization")

    return activations


def summarize_extraction(output_dir: str | Path):
    """
    Print a summary of the extraction results.

    Args:
        output_dir: Path to extraction output directory
    """
    try:
        ckpt = load_checkpoint(output_dir)
    except FileNotFoundError:
        print(f"No checkpoint found in {output_dir}")
        return

    print("=" * 80)
    print(f"Extraction Summary: {output_dir}")
    print("=" * 80)
    print(f"Primary layer: {ckpt.get('primary_layer')}")
    print(f"Target tokens (primary): {ckpt.get('target_tokens_primary', 0):,}")
    print(f"Target inferences: {ckpt.get('target_inferences', 0):,}")
    print(f"Inferences completed: {ckpt.get('inferences_completed', 0):,}")
    print(f"Completed: {ckpt.get('completed', False)}")
    print()

    layers = ckpt.get("layers", {})
    print(f"Layers extracted: {len(layers)}")
    print()

    for layer_name, info in layers.items():
        print(f"Layer: {layer_name}")
        print(f"  Act size: {info.get('act_size', -1)}")
        print(f"  Tokens per inference: {info.get('tokens_per_inference', -1):.1f}")
        print(f"  Subsample rate: {info.get('subsample_rate', 1.0)}")
        print(f"  Tokens collected: {info.get('tokens_collected', 0):,}")
        print(f"  Chunks written: {info.get('chunks_written', 0)}")
        print()

    print("=" * 80)


def main():
    """Example usage of the loading utilities."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and inspect K-means activation extractions"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to extraction output directory",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Specific layer to load (default: show summary only)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum chunks to load (for testing)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show summary, don't load activations",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Show summary
    summarize_extraction(args.output_dir)

    # Load specific layer if requested
    if args.layer and not args.summary_only:
        print(f"Loading activations for layer: {args.layer}")
        print()

        try:
            activations = load_layer_activations(
                args.output_dir,
                args.layer,
                max_chunks=args.max_chunks,
            )

            print(f"Loaded activations shape: {activations.shape}")
            print(f"Data type: {activations.dtype}")
            print(f"Memory usage: {activations.element_size() * activations.numel() / 1e9:.2f} GB")
            print()

            # Basic statistics
            print("Statistics:")
            print(f"  Mean: {activations.mean().item():.6f}")
            print(f"  Std: {activations.std().item():.6f}")
            print(f"  Min: {activations.min().item():.6f}")
            print(f"  Max: {activations.max().item():.6f}")

        except Exception as e:
            print(f"Error loading layer: {e}")


if __name__ == "__main__":
    main()
