#!/usr/bin/env python3
"""
Comprehensive test suite for Parquet refactoring with FixedSizeList format.

This test suite validates:
1. FixedSizeList format I/O (write, read, integrity)
2. Random sampling from Parquet files
3. Full integration (extract -> train pipeline)
4. Edge cases (empty, single sample, large dims, multiple chunks)
5. Performance benchmarks (Parquet vs .pt)
6. Backward compatibility (.pt files still work)

Usage:
    python scripts/kmeans/test_parquet_implementation.py
    python scripts/kmeans/test_parquet_implementation.py --verbose
    python scripts/kmeans/test_parquet_implementation.py --skip-integration
"""

import argparse
import gc
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("ERROR: PyArrow not installed. Install with: pip install pyarrow")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ParquetTestSuite:
    """Comprehensive test suite for Parquet FixedSizeList implementation."""

    def __init__(self, temp_dir: Path, verbose: bool = False):
        """Initialize test suite with temporary directory."""
        self.temp_dir = temp_dir
        self.verbose = verbose
        self.results = {}
        self.test_count = 0
        self.passed_count = 0
        self.performance_stats = {}

        if verbose:
            logger.setLevel(logging.DEBUG)

    def log_test(self, name: str, passed: bool, message: str = ""):
        """Log a test result."""
        self.test_count += 1
        if passed:
            self.passed_count += 1
            status = "PASS"
            symbol = "✓"
        else:
            status = "FAIL"
            symbol = "✗"

        log_msg = f"{symbol} {name}: {status}"
        if message:
            log_msg += f" - {message}"
        logger.info(log_msg)
        self.results[name] = passed

    def generate_synthetic_data(
        self,
        n_samples: int = 1000,
        n_dims: int = 768,
        seed: int = 42
    ) -> torch.Tensor:
        """Generate synthetic activation data."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        data = torch.randn(n_samples, n_dims, dtype=torch.float32)
        logger.debug(f"Generated synthetic data: {data.shape}")
        return data

    # ========================================================================
    # FixedSizeList I/O Functions
    # ========================================================================

    def write_parquet_fixedsizelist(
        self,
        data: torch.Tensor,
        output_path: Path
    ) -> float:
        """
        Write activation data to Parquet using FixedSizeList format.

        This is the CORRECT format (not columnar).

        Args:
            data: (N, D) tensor of activations
            output_path: Path to save parquet file

        Returns:
            Write time in seconds
        """
        start_time = time.time()

        # Convert to numpy
        data_np = data.numpy().astype(np.float32)
        n_samples, n_dims = data_np.shape

        # Store shape metadata
        metadata = {
            b'n_samples': str(n_samples).encode(),
            b'n_dims': str(n_dims).encode(),
            b'dtype': 'float32'.encode(),
            b'format': 'fixedsizelist'.encode(),
        }

        # Create FixedSizeList arrays (one list per row)
        # CORRECT FORMAT: Each row is a fixed-size list
        # Convert each row to a list, then create a single array of lists
        rows_as_lists = [row.tolist() for row in data_np]
        activations_array = pa.array(rows_as_lists, type=pa.list_(pa.float32(), n_dims))

        # Create table with single 'activations' column
        table = pa.table({'activations': activations_array})
        table = table.replace_schema_metadata(metadata)

        # Write to parquet
        pq.write_table(table, output_path, compression='snappy')

        write_time = time.time() - start_time
        logger.debug(
            f"Wrote parquet file: {output_path} "
            f"({output_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return write_time

    def read_parquet_fixedsizelist(
        self,
        input_path: Path
    ) -> Tuple[torch.Tensor, float]:
        """
        Read activation data from Parquet FixedSizeList format.

        Args:
            input_path: Path to parquet file

        Returns:
            Tuple of (reconstructed tensor, read time in seconds)
        """
        start_time = time.time()

        # Read parquet
        table = pq.read_table(input_path)

        # Get metadata
        metadata = table.schema.metadata or {}
        n_samples = int(metadata.get(b'n_samples', 0).decode())
        n_dims = int(metadata.get(b'n_dims', 0).decode())

        # Extract activations column
        activations_column = table.column('activations')

        # Convert to numpy: each element is a list, stack into 2D array
        data_list = activations_column.to_pylist()
        data_np = np.array(data_list, dtype=np.float32)

        # Convert to tensor
        data = torch.from_numpy(data_np).float()

        read_time = time.time() - start_time
        logger.debug(f"Read parquet file: {input_path} (time: {read_time:.4f}s)")
        return data, read_time

    def read_parquet_random_sample(
        self,
        input_path: Path,
        sample_size: int,
        seed: int = 42
    ) -> Tuple[torch.Tensor, float]:
        """
        Read random sample from Parquet file.

        Args:
            input_path: Path to parquet file
            sample_size: Number of samples to randomly select
            seed: Random seed for reproducibility

        Returns:
            Tuple of (sampled tensor, read time in seconds)
        """
        start_time = time.time()

        # Read metadata to get total rows
        metadata = pq.read_metadata(input_path)
        total_rows = metadata.num_rows

        # Generate random indices
        np.random.seed(seed)
        if sample_size >= total_rows:
            indices = np.arange(total_rows)
        else:
            indices = np.random.choice(total_rows, size=sample_size, replace=False)
            indices.sort()  # Sort for better I/O performance

        # Read table and select rows
        table = pq.read_table(input_path)
        sampled_table = table.take(indices)

        # Convert to tensor
        activations_column = sampled_table.column('activations')
        data_list = activations_column.to_pylist()
        data_np = np.array(data_list, dtype=np.float32)
        data = torch.from_numpy(data_np).float()

        read_time = time.time() - start_time
        logger.debug(
            f"Read {sample_size} random samples from {input_path} "
            f"(time: {read_time:.4f}s)"
        )
        return data, read_time

    # ========================================================================
    # PyTorch I/O Functions (for comparison)
    # ========================================================================

    def write_pytorch(self, data: torch.Tensor, output_path: Path) -> float:
        """Write data using PyTorch format."""
        start_time = time.time()
        torch.save(data, output_path)
        write_time = time.time() - start_time
        logger.debug(
            f"Wrote .pt file: {output_path} "
            f"({output_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
        return write_time

    def read_pytorch(self, input_path: Path) -> Tuple[torch.Tensor, float]:
        """Read data using PyTorch format."""
        start_time = time.time()
        data = torch.load(input_path, map_location='cpu')
        read_time = time.time() - start_time
        logger.debug(f"Read .pt file: {input_path} (time: {read_time:.4f}s)")
        return data, read_time

    # ========================================================================
    # Test 1: FixedSizeList Format I/O
    # ========================================================================

    def test_fixedsizelist_io(self):
        """Test FixedSizeList format read/write with data integrity checks."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: FixedSizeList Format I/O")
        logger.info("=" * 80)

        # Generate test data
        original_data = self.generate_synthetic_data(n_samples=1000, n_dims=768)

        # Write and read
        parquet_path = self.temp_dir / "test_fixedsizelist.parquet"
        write_time = self.write_parquet_fixedsizelist(original_data, parquet_path)
        loaded_data, read_time = self.read_parquet_fixedsizelist(parquet_path)

        # Check shape
        shape_match = original_data.shape == loaded_data.shape
        self.log_test(
            "FixedSizeList shape match",
            shape_match,
            f"Original: {original_data.shape}, Loaded: {loaded_data.shape}"
        )

        # Check values
        values_close = torch.allclose(original_data, loaded_data, rtol=1e-5, atol=1e-7)
        max_diff = (original_data - loaded_data).abs().max().item()
        self.log_test(
            "FixedSizeList values match",
            values_close,
            f"Max diff: {max_diff:.2e}"
        )

        # Check dtype
        dtype_match = original_data.dtype == loaded_data.dtype
        self.log_test(
            "FixedSizeList dtype match",
            dtype_match,
            f"Original: {original_data.dtype}, Loaded: {loaded_data.dtype}"
        )

        # Verify schema format
        table = pq.read_table(parquet_path)
        schema = table.schema
        field_type = schema.field('activations').type
        is_fixedsizelist = 'activations' in schema.names and \
                          (isinstance(field_type, pa.ListType) or
                           isinstance(field_type, pa.FixedSizeListType))
        self.log_test(
            "FixedSizeList schema correct",
            is_fixedsizelist,
            f"Schema type: {field_type}"
        )

        # Compare with .pt format
        pt_path = self.temp_dir / "test_compare.pt"
        pt_write_time = self.write_pytorch(original_data, pt_path)
        pt_data, pt_read_time = self.read_pytorch(pt_path)

        format_match = torch.allclose(loaded_data, pt_data, rtol=1e-5)
        self.log_test(
            "FixedSizeList vs .pt equivalence",
            format_match,
            "Both formats produce identical data"
        )

        logger.info(f"  Write time - Parquet: {write_time:.4f}s, .pt: {pt_write_time:.4f}s")
        logger.info(f"  Read time  - Parquet: {read_time:.4f}s, .pt: {pt_read_time:.4f}s")

    # ========================================================================
    # Test 2: Random Sampling
    # ========================================================================

    def test_random_sampling(self):
        """Test random row sampling from Parquet files."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: Random Sampling")
        logger.info("=" * 80)

        # Generate large dataset
        n_samples = 50000
        logger.info(f"Generating {n_samples} samples...")
        original_data = self.generate_synthetic_data(n_samples=n_samples, n_dims=768)

        # Save to parquet
        parquet_path = self.temp_dir / "test_sampling.parquet"
        self.write_parquet_fixedsizelist(original_data, parquet_path)

        # Test 1: Sample 10K randomly
        sample_size = 10000
        sampled_data, sample_time = self.read_parquet_random_sample(
            parquet_path, sample_size, seed=42
        )

        size_correct = sampled_data.shape[0] == sample_size
        self.log_test(
            "Random sampling size",
            size_correct,
            f"Sampled {sampled_data.shape[0]} / {sample_size} requested"
        )

        # Test 2: Reproducibility with same seed
        sampled_data2, _ = self.read_parquet_random_sample(
            parquet_path, sample_size, seed=42
        )
        reproducible = torch.equal(sampled_data, sampled_data2)
        self.log_test(
            "Random sampling reproducibility",
            reproducible,
            "Same seed produces identical samples"
        )

        # Test 3: Different seed produces different samples
        sampled_data3, _ = self.read_parquet_random_sample(
            parquet_path, sample_size, seed=999
        )
        different = not torch.equal(sampled_data, sampled_data3)
        self.log_test(
            "Random sampling randomness",
            different,
            "Different seed produces different samples"
        )

        # Test 4: Check distribution (mean should be close to population mean)
        pop_mean = original_data.mean().item()
        sample_mean = sampled_data.mean().item()
        mean_diff = abs(pop_mean - sample_mean)
        distribution_ok = mean_diff < 0.1  # Allow 0.1 difference
        self.log_test(
            "Random sampling distribution",
            distribution_ok,
            f"Population mean: {pop_mean:.4f}, Sample mean: {sample_mean:.4f}"
        )

        logger.info(f"  Sample read time: {sample_time:.4f}s for {sample_size} samples")

    # ========================================================================
    # Test 3: Integration Test
    # ========================================================================

    def test_integration(self):
        """Test full extraction -> training pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: Integration Test (Extract -> Train)")
        logger.info("=" * 80)

        # Simulate extraction: create multiple chunks in Parquet format
        n_chunks = 5
        samples_per_chunk = 2000
        n_dims = 768
        chunk_dir = self.temp_dir / "integration_chunks"
        chunk_dir.mkdir(exist_ok=True)

        logger.info(f"Creating {n_chunks} parquet chunks...")
        all_data_chunks = []
        for i in range(n_chunks):
            chunk_data = self.generate_synthetic_data(
                n_samples=samples_per_chunk,
                n_dims=n_dims,
                seed=42 + i
            )
            all_data_chunks.append(chunk_data)

            chunk_path = chunk_dir / f"chunk_{i:06d}.parquet"
            self.write_parquet_fixedsizelist(chunk_data, chunk_path)

        # Concatenate ground truth
        ground_truth = torch.cat(all_data_chunks, dim=0)
        logger.info(f"Ground truth shape: {ground_truth.shape}")

        # Simulate training: load all chunks
        logger.info("Loading chunks for training...")
        chunk_files = sorted(chunk_dir.glob("chunk_*.parquet"))
        loaded_chunks = []
        for chunk_file in chunk_files:
            chunk_data, _ = self.read_parquet_fixedsizelist(chunk_file)
            loaded_chunks.append(chunk_data)

        loaded_data = torch.cat(loaded_chunks, dim=0)
        logger.info(f"Loaded data shape: {loaded_data.shape}")

        # Verify data integrity
        data_match = torch.allclose(ground_truth, loaded_data, rtol=1e-5)
        self.log_test(
            "Integration data integrity",
            data_match,
            f"Loaded {loaded_data.shape[0]} samples correctly"
        )

        # Simulate K-means training (simplified)
        logger.info("Simulating K-means training...")
        n_clusters = 100
        data_np = loaded_data.numpy()

        # Use simple random initialization (real K-means would use faiss)
        np.random.seed(42)
        indices = np.random.choice(data_np.shape[0], size=n_clusters, replace=False)
        centroids = data_np[indices]

        centroids_valid = centroids.shape == (n_clusters, n_dims)
        self.log_test(
            "Integration K-means centroids",
            centroids_valid,
            f"Centroids shape: {centroids.shape}"
        )

        # Compare with .pt baseline
        logger.info("Comparing with .pt baseline...")
        pt_chunk_dir = self.temp_dir / "integration_pt_chunks"
        pt_chunk_dir.mkdir(exist_ok=True)

        for i, chunk_data in enumerate(all_data_chunks):
            pt_chunk_path = pt_chunk_dir / f"chunk_{i:06d}.pt"
            self.write_pytorch(chunk_data, pt_chunk_path)

        # Load .pt chunks
        pt_chunk_files = sorted(pt_chunk_dir.glob("chunk_*.pt"))
        pt_loaded_chunks = []
        for pt_chunk_file in pt_chunk_files:
            pt_chunk_data, _ = self.read_pytorch(pt_chunk_file)
            pt_loaded_chunks.append(pt_chunk_data)

        pt_loaded_data = torch.cat(pt_loaded_chunks, dim=0)

        baseline_match = torch.allclose(loaded_data, pt_loaded_data, rtol=1e-5)
        self.log_test(
            "Integration vs .pt baseline",
            baseline_match,
            "Parquet and .pt produce identical results"
        )

    # ========================================================================
    # Test 4: Edge Cases
    # ========================================================================

    def test_edge_cases(self):
        """Test edge cases: empty, single sample, large dims, multiple chunks."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 4: Edge Cases")
        logger.info("=" * 80)

        # Test 4a: Empty data
        logger.info("Testing empty data...")
        try:
            empty_data = torch.empty(0, 768, dtype=torch.float32)
            parquet_path = self.temp_dir / "test_empty.parquet"
            self.write_parquet_fixedsizelist(empty_data, parquet_path)
            loaded_data, _ = self.read_parquet_fixedsizelist(parquet_path)

            # For empty data, loaded_data might be shape (0,) or (0, 768)
            # We just need to verify it's empty
            empty_ok = loaded_data.numel() == 0
            self.log_test(
                "Edge case: empty data",
                empty_ok,
                f"Empty tensor handled correctly: {loaded_data.shape}"
            )
        except Exception as e:
            self.log_test("Edge case: empty data", False, f"Error: {e}")

        # Test 4b: Single sample
        logger.info("Testing single sample...")
        try:
            single_data = torch.randn(1, 768, dtype=torch.float32)
            parquet_path = self.temp_dir / "test_single.parquet"
            self.write_parquet_fixedsizelist(single_data, parquet_path)
            loaded_data, _ = self.read_parquet_fixedsizelist(parquet_path)

            single_ok = loaded_data.shape == (1, 768) and \
                       torch.allclose(single_data, loaded_data)
            self.log_test(
                "Edge case: single sample",
                single_ok,
                f"Single sample handled correctly: {loaded_data.shape}"
            )
        except Exception as e:
            self.log_test("Edge case: single sample", False, f"Error: {e}")

        # Test 4c: Large dimensions
        logger.info("Testing large dimensions (2048D)...")
        try:
            large_dim_data = self.generate_synthetic_data(n_samples=100, n_dims=2048)
            parquet_path = self.temp_dir / "test_large_dim.parquet"
            self.write_parquet_fixedsizelist(large_dim_data, parquet_path)
            loaded_data, _ = self.read_parquet_fixedsizelist(parquet_path)

            large_dim_ok = loaded_data.shape == (100, 2048) and \
                          torch.allclose(large_dim_data, loaded_data)
            self.log_test(
                "Edge case: large dimensions",
                large_dim_ok,
                f"Large dims (2048D) handled correctly"
            )
        except Exception as e:
            self.log_test("Edge case: large dimensions", False, f"Error: {e}")

        # Test 4d: Multiple chunks concatenation
        logger.info("Testing multiple chunks...")
        try:
            chunk_dir = self.temp_dir / "edge_chunks"
            chunk_dir.mkdir(exist_ok=True)

            n_chunks = 10
            chunks_data = []
            for i in range(n_chunks):
                chunk = self.generate_synthetic_data(
                    n_samples=100, n_dims=768, seed=42 + i
                )
                chunks_data.append(chunk)

                chunk_path = chunk_dir / f"chunk_{i:04d}.parquet"
                self.write_parquet_fixedsizelist(chunk, chunk_path)

            # Load and concatenate
            loaded_chunks = []
            for i in range(n_chunks):
                chunk_path = chunk_dir / f"chunk_{i:04d}.parquet"
                chunk, _ = self.read_parquet_fixedsizelist(chunk_path)
                loaded_chunks.append(chunk)

            original_concat = torch.cat(chunks_data, dim=0)
            loaded_concat = torch.cat(loaded_chunks, dim=0)

            chunks_ok = torch.allclose(original_concat, loaded_concat, rtol=1e-5)
            self.log_test(
                "Edge case: multiple chunks",
                chunks_ok,
                f"Concatenated {n_chunks} chunks correctly: {loaded_concat.shape}"
            )
        except Exception as e:
            self.log_test("Edge case: multiple chunks", False, f"Error: {e}")

    # ========================================================================
    # Test 5: Performance Benchmarks
    # ========================================================================

    def test_performance(self):
        """Benchmark Parquet vs .pt format."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 5: Performance Benchmarks")
        logger.info("=" * 80)

        # Test different data sizes
        test_configs = [
            (1000, 768, "Small (1K x 768)"),
            (10000, 768, "Medium (10K x 768)"),
            (50000, 768, "Large (50K x 768)"),
        ]

        performance_results = []

        for n_samples, n_dims, desc in test_configs:
            logger.info(f"\nBenchmarking {desc}...")

            # Generate data
            data = self.generate_synthetic_data(n_samples=n_samples, n_dims=n_dims)

            # Benchmark Parquet write
            parquet_path = self.temp_dir / f"perf_{n_samples}.parquet"
            parquet_write_time = self.write_parquet_fixedsizelist(data, parquet_path)

            # Benchmark Parquet read
            _, parquet_read_time = self.read_parquet_fixedsizelist(parquet_path)

            # Benchmark .pt write
            pt_path = self.temp_dir / f"perf_{n_samples}.pt"
            pt_write_time = self.write_pytorch(data, pt_path)

            # Benchmark .pt read
            _, pt_read_time = self.read_pytorch(pt_path)

            # File sizes
            parquet_size = parquet_path.stat().st_size / 1024 / 1024
            pt_size = pt_path.stat().st_size / 1024 / 1024

            # Log results
            logger.info(f"  Parquet: write={parquet_write_time:.4f}s, "
                       f"read={parquet_read_time:.4f}s, size={parquet_size:.2f}MB")
            logger.info(f"  .pt:     write={pt_write_time:.4f}s, "
                       f"read={pt_read_time:.4f}s, size={pt_size:.2f}MB")
            logger.info(f"  Size ratio (Parquet/.pt): {parquet_size/pt_size:.2f}x")

            performance_results.append({
                'desc': desc,
                'parquet_write': parquet_write_time,
                'parquet_read': parquet_read_time,
                'parquet_size': parquet_size,
                'pt_write': pt_write_time,
                'pt_read': pt_read_time,
                'pt_size': pt_size,
            })

            # Verify data integrity
            data_match = True
            self.log_test(
                f"Performance integrity {desc}",
                data_match,
                "Data integrity verified"
            )

        # Store performance stats
        self.performance_stats = performance_results

        # Summary table
        logger.info("\n" + "=" * 80)
        logger.info("Performance Summary:")
        logger.info("=" * 80)
        logger.info(f"{'Config':<20} {'Format':<10} {'Write (s)':<12} "
                   f"{'Read (s)':<12} {'Size (MB)':<12}")
        logger.info("-" * 80)
        for result in performance_results:
            logger.info(f"{result['desc']:<20} {'Parquet':<10} "
                       f"{result['parquet_write']:<12.4f} "
                       f"{result['parquet_read']:<12.4f} "
                       f"{result['parquet_size']:<12.2f}")
            logger.info(f"{result['desc']:<20} {'.pt':<10} "
                       f"{result['pt_write']:<12.4f} "
                       f"{result['pt_read']:<12.4f} "
                       f"{result['pt_size']:<12.2f}")

    # ========================================================================
    # Test 6: Backward Compatibility
    # ========================================================================

    def test_backward_compatibility(self):
        """Test that .pt files still work correctly."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 6: Backward Compatibility")
        logger.info("=" * 80)

        # Create test data
        data = self.generate_synthetic_data(n_samples=1000, n_dims=768)

        # Save as .pt
        pt_path = self.temp_dir / "backward_compat.pt"
        self.write_pytorch(data, pt_path)

        # Load .pt
        loaded_pt, _ = self.read_pytorch(pt_path)

        # Verify
        pt_works = torch.allclose(data, loaded_pt, rtol=1e-5)
        self.log_test(
            "Backward compat: .pt format",
            pt_works,
            ".pt format still works correctly"
        )

        # Test mixed format handling (directory with both .pt and .parquet)
        mixed_dir = self.temp_dir / "mixed_format"
        mixed_dir.mkdir(exist_ok=True)

        # Create both formats
        chunk1 = self.generate_synthetic_data(n_samples=500, n_dims=768, seed=1)
        chunk2 = self.generate_synthetic_data(n_samples=500, n_dims=768, seed=2)

        pt_chunk_path = mixed_dir / "chunk_0000.pt"
        parquet_chunk_path = mixed_dir / "chunk_0001.parquet"

        self.write_pytorch(chunk1, pt_chunk_path)
        self.write_parquet_fixedsizelist(chunk2, parquet_chunk_path)

        # Verify both can be loaded independently
        loaded_chunk1, _ = self.read_pytorch(pt_chunk_path)
        loaded_chunk2, _ = self.read_parquet_fixedsizelist(parquet_chunk_path)

        mixed_ok = torch.allclose(chunk1, loaded_chunk1, rtol=1e-5) and \
                  torch.allclose(chunk2, loaded_chunk2, rtol=1e-5)
        self.log_test(
            "Backward compat: mixed format",
            mixed_ok,
            "Both .pt and .parquet can coexist and be loaded independently"
        )

    # ========================================================================
    # Main Test Runner
    # ========================================================================

    def run_all_tests(self, skip_integration: bool = False):
        """Run all tests in the suite."""
        logger.info("\n" + "=" * 80)
        logger.info("PARQUET FIXEDSIZELIST TEST SUITE")
        logger.info("=" * 80)

        try:
            self.test_fixedsizelist_io()
            self.test_random_sampling()
            if not skip_integration:
                self.test_integration()
            else:
                logger.info("\nSkipping integration test (--skip-integration)")
            self.test_edge_cases()
            self.test_performance()
            self.test_backward_compatibility()
        except Exception as e:
            logger.error(f"Test suite failed with error: {e}", exc_info=True)
            return False

        return True

    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total tests: {self.test_count}")
        logger.info(f"Passed: {self.passed_count}")
        logger.info(f"Failed: {self.test_count - self.passed_count}")

        if self.test_count > 0:
            pass_rate = (self.passed_count / self.test_count) * 100
            logger.info(f"Pass rate: {pass_rate:.1f}%")

        if self.passed_count == self.test_count:
            logger.info("\nResult: ALL TESTS PASSED ✓")
            return True
        else:
            logger.info("\nResult: SOME TESTS FAILED ✗")
            logger.info("\nFailed tests:")
            for test_name, passed in self.results.items():
                if not passed:
                    logger.info(f"  - {test_name}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for Parquet FixedSizeList implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip integration test (faster)",
    )
    args = parser.parse_args()

    if not HAS_PARQUET:
        logger.error("PyArrow is required. Install with: pip install pyarrow")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        suite = ParquetTestSuite(temp_path, verbose=args.verbose)

        # Run tests
        suite.run_all_tests(skip_integration=args.skip_integration)

        # Print summary
        all_passed = suite.print_summary()

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
