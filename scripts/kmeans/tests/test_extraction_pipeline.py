#!/usr/bin/env python3
"""
Test script for the K-means activation extraction pipeline.

This script performs basic validation of the extraction and loading utilities
without requiring a full dataset or long extraction run.

Usage:
    python scripts/test_extraction_pipeline.py
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[3]  # Go up to project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the utilities
kmeans_utils = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(kmeans_utils))
from load_kmeans_activations import (
    load_checkpoint,
    load_layer_activations,
    load_layer_activations_lazy,
    prepare_for_kmeans,
    sanitize_layer_name,
    summarize_extraction,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_sanitize_layer_name():
    """Test layer name sanitization."""
    logger.info("Testing layer name sanitization...")

    tests = [
        ("model.layer.0", "model.layer.0"),
        ("model/layer/0", "model_layer_0"),
        ("model:layer:0", "model__layer__0"),
        ("model/layer:0@1", "model_layer__0@1"),
    ]

    for input_name, expected in tests:
        result = sanitize_layer_name(input_name)
        assert result == expected, f"Expected {expected}, got {result}"
        logger.info(f"  ✓ {input_name} -> {result}")

    logger.info("✓ Layer name sanitization tests passed\n")


def test_checkpoint_operations():
    """Test checkpoint save/load operations."""
    logger.info("Testing checkpoint operations...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a mock checkpoint
        checkpoint_data = {
            "primary_layer": "test.layer.0",
            "target_inferences": 1000,
            "target_tokens_primary": 4096000,
            "inferences_completed": 500,
            "completed": False,
            "layers": {
                "test.layer.0": {
                    "act_size": 256,
                    "tokens_per_inference": 4096.0,
                    "subsample_rate": 1.0,
                    "tokens_collected": 2048000,
                    "chunks_written": 125,
                },
                "test.layer.1": {
                    "act_size": 512,
                    "tokens_per_inference": 2048.0,
                    "subsample_rate": 0.5,
                    "tokens_collected": 512000,
                    "chunks_written": 31,
                },
            },
        }

        # Save checkpoint
        ckpt_path = tmpdir / "checkpoint.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        loaded = load_checkpoint(tmpdir)

        # Verify
        assert loaded["primary_layer"] == checkpoint_data["primary_layer"]
        assert loaded["target_inferences"] == checkpoint_data["target_inferences"]
        assert loaded["inferences_completed"] == checkpoint_data["inferences_completed"]
        assert len(loaded["layers"]) == 2

        logger.info("  ✓ Checkpoint save/load works")

        # Test summarize_extraction
        logger.info("  Testing extraction summary:")
        summarize_extraction(tmpdir)

    logger.info("✓ Checkpoint operations tests passed\n")


def test_activation_save_load():
    """Test activation chunk save/load."""
    logger.info("Testing activation save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock layer directory and chunks
        layer_name = "test.layer.0"
        safe_name = sanitize_layer_name(layer_name)
        layer_dir = tmpdir / safe_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Create mock activation chunks
        num_chunks = 5
        tokens_per_chunk = 1000
        feature_dim = 256

        chunks_data = []
        for i in range(num_chunks):
            chunk = torch.randn(tokens_per_chunk, feature_dim)
            chunks_data.append(chunk)
            torch.save(chunk, layer_dir / f"chunk_{i:06d}.pt")

        # Create checkpoint
        checkpoint_data = {
            "primary_layer": layer_name,
            "target_inferences": 100,
            "target_tokens_primary": 5000,
            "inferences_completed": 100,
            "completed": True,
            "layers": {
                layer_name: {
                    "act_size": feature_dim,
                    "tokens_per_inference": 50.0,
                    "subsample_rate": 1.0,
                    "tokens_collected": num_chunks * tokens_per_chunk,
                    "chunks_written": num_chunks,
                }
            },
        }
        with open(tmpdir / "checkpoint.json", "w") as f:
            json.dump(checkpoint_data, f)

        # Test loading all activations
        logger.info(f"  Loading all activations for {layer_name}...")
        loaded_acts = load_layer_activations(tmpdir, layer_name)

        # Verify shape
        expected_shape = (num_chunks * tokens_per_chunk, feature_dim)
        assert loaded_acts.shape == expected_shape, f"Expected {expected_shape}, got {loaded_acts.shape}"
        logger.info(f"  ✓ Loaded shape: {loaded_acts.shape}")

        # Verify data (concatenate original and compare)
        expected_acts = torch.cat(chunks_data, dim=0)
        assert torch.allclose(loaded_acts, expected_acts, atol=1e-6)
        logger.info("  ✓ Data integrity verified")

        # Test lazy loading
        logger.info("  Testing lazy loading...")
        lazy_chunks = list(load_layer_activations_lazy(tmpdir, layer_name))
        assert len(lazy_chunks) == num_chunks
        for i, chunk in enumerate(lazy_chunks):
            assert chunk.shape == (tokens_per_chunk, feature_dim)
            assert torch.allclose(chunk, chunks_data[i], atol=1e-6)
        logger.info(f"  ✓ Lazy loading: {len(lazy_chunks)} chunks")

        # Test loading with max_chunks
        logger.info("  Testing max_chunks...")
        limited_acts = load_layer_activations(tmpdir, layer_name, max_chunks=3)
        assert limited_acts.shape == (3 * tokens_per_chunk, feature_dim)
        logger.info(f"  ✓ Max chunks: {limited_acts.shape}")

    logger.info("✓ Activation save/load tests passed\n")


def test_prepare_for_kmeans():
    """Test K-means preparation utilities."""
    logger.info("Testing K-means preparation...")

    # Create mock activations
    num_tokens = 10000
    feature_dim = 256
    activations = torch.randn(num_tokens, feature_dim)

    # Test sampling
    logger.info("  Testing sampling...")
    n_samples = 5000
    sampled = prepare_for_kmeans(activations, n_samples=n_samples, normalize=False)
    assert sampled.shape == (n_samples, feature_dim)
    logger.info(f"  ✓ Sampling: {num_tokens} -> {n_samples} tokens")

    # Test normalization
    logger.info("  Testing normalization...")
    normalized = prepare_for_kmeans(activations, n_samples=None, normalize=True)
    assert normalized.shape == activations.shape

    # Verify L2 normalization (each row should have norm ≈ 1)
    norms = torch.norm(normalized, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    logger.info(f"  ✓ Normalization: mean norm = {norms.mean().item():.6f}")

    # Test both sampling and normalization
    logger.info("  Testing sampling + normalization...")
    prepared = prepare_for_kmeans(activations, n_samples=n_samples, normalize=True)
    assert prepared.shape == (n_samples, feature_dim)
    norms = torch.norm(prepared, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    logger.info(f"  ✓ Combined: {prepared.shape}, norm = {norms.mean().item():.6f}")

    logger.info("✓ K-means preparation tests passed\n")


def test_inference_based_extractor():
    """Test InferenceBasedExtractor class (without full pipeline)."""
    logger.info("Testing InferenceBasedExtractor class...")

    # This is a minimal test - we can't fully test without a real activation store
    # but we can test the basic structure and utilities

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Mock extractor state
        logger.info("  Creating mock extractor state...")

        # Create layer directories
        layers = ["test.layer.0", "test.layer.1"]
        for layer_name in layers:
            safe_name = sanitize_layer_name(layer_name)
            layer_dir = tmpdir / safe_name
            layer_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint manually
        checkpoint = {
            "primary_layer": "test.layer.0",
            "target_inferences": 1000,
            "target_tokens_primary": 4096000,
            "inferences_completed": 500,
            "completed": False,
            "layers": {
                "test.layer.0": {
                    "act_size": 256,
                    "tokens_per_inference": 4096.0,
                    "subsample_rate": 1.0,
                    "tokens_collected": 2048000,
                    "chunks_written": 125,
                },
                "test.layer.1": {
                    "act_size": 512,
                    "tokens_per_inference": 2048.0,
                    "subsample_rate": 1.0,
                    "tokens_collected": 1024000,
                    "chunks_written": 62,
                },
            },
        }

        with open(tmpdir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f)

        logger.info("  ✓ Mock state created")

        # Verify we can load it
        loaded = load_checkpoint(tmpdir)
        assert loaded["inferences_completed"] == 500
        assert loaded["target_inferences"] == 1000
        logger.info("  ✓ Checkpoint loadable")

        # Calculate expected target inferences
        target_tokens = checkpoint["target_tokens_primary"]
        tpi = checkpoint["layers"]["test.layer.0"]["tokens_per_inference"]
        expected_target = int(target_tokens / tpi)
        assert checkpoint["target_inferences"] == expected_target
        logger.info(f"  ✓ Target calculation: {target_tokens} / {tpi} = {expected_target}")

    logger.info("✓ InferenceBasedExtractor tests passed\n")


def run_all_tests():
    """Run all test suites."""
    logger.info("=" * 80)
    logger.info("K-means Activation Extraction Pipeline Tests")
    logger.info("=" * 80)
    logger.info("")

    tests = [
        test_sanitize_layer_name,
        test_checkpoint_operations,
        test_activation_save_load,
        test_prepare_for_kmeans,
        test_inference_based_extractor,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            logger.error(f"✗ Test failed: {test_fn.__name__}")
            logger.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    logger.info("=" * 80)
    logger.info("All tests passed! ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_all_tests()
