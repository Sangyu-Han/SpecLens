#!/usr/bin/env python3
"""
Test script to verify Parquet I/O works correctly for K-means activation data.

This script:
1. Generates synthetic activation data (1000 samples, 768 dims)
2. Saves using Parquet writing logic
3. Loads using Parquet reading logic
4. Verifies data integrity (shapes, values match)
5. Tests random sampling (verify randomness and correctness)
6. Benchmarks performance vs .pt format

Usage:
    python scripts/kmeans/test_parquet_io.py
"""

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("WARNING: PyArrow not installed. Install with: pip install pyarrow")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ParquetIOTester:
    """Test suite for Parquet I/O operations."""

    def __init__(self, temp_dir: Path):
        """Initialize tester with temporary directory."""
        self.temp_dir = temp_dir
        self.results = {}
        self.test_count = 0
        self.passed_count = 0

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

        # Create data with some structure
        data = torch.randn(n_samples, n_dims, dtype=torch.float32)
        logger.info(f"Generated synthetic data: {data.shape}")
        return data

    def write_parquet_simple(self, data: torch.Tensor, output_path: Path) -> float:
        """
        Write activation data to Parquet format.

        Args:
            data: (N, D) tensor of activations
            output_path: Path to save parquet file

        Returns:
            Write time in seconds
        """
        if not HAS_PARQUET:
            raise RuntimeError("PyArrow not available")

        start_time = time.time()

        # Convert to numpy
        data_np = data.numpy().astype(np.float32)
        n_samples, n_dims = data_np.shape

        # Store shape metadata
        metadata = {
            b'n_samples': str(n_samples).encode(),
            b'n_dims': str(n_dims).encode(),
            b'dtype': 'float32'.encode(),
        }

        # Create PyArrow table with data as structured format
        # Store data as individual columns (one per dimension)
        arrays = []
        names = []

        for i in range(n_dims):
            arrays.append(pa.array(data_np[:, i], type=pa.float32()))
            names.append(f"dim_{i}")

        table = pa.table({name: arr for name, arr in zip(names, arrays)})
        table = table.replace_schema_metadata(metadata)

        # Write to parquet
        pq.write_table(table, output_path, compression='snappy')

        write_time = time.time() - start_time
        logger.info(f"Wrote parquet file: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return write_time

    def read_parquet_simple(self, input_path: Path) -> Tuple[torch.Tensor, float]:
        """
        Read activation data from Parquet format.

        Args:
            input_path: Path to parquet file

        Returns:
            Tuple of (reconstructed tensor, read time in seconds)
        """
        if not HAS_PARQUET:
            raise RuntimeError("PyArrow not available")

        start_time = time.time()

        # Read parquet
        table = pq.read_table(input_path)

        # Get metadata
        metadata = table.schema.metadata or {}
        n_samples = int(metadata.get(b'n_samples', 0).decode())
        n_dims = int(metadata.get(b'n_dims', 0).decode())

        # Reconstruct from individual dimension columns
        arrays = []
        for i in range(n_dims):
            col_name = f"dim_{i}"
            arrays.append(table.column(col_name).to_numpy(zero_copy_only=False))

        # Stack into (n_samples, n_dims) array
        data_np = np.column_stack(arrays).astype(np.float32)
        data = torch.from_numpy(data_np).float()

        read_time = time.time() - start_time
        logger.info(f"Read parquet file: {input_path} (time: {read_time:.4f}s)")
        return data, read_time

    def write_pytorch(self, data: torch.Tensor, output_path: Path) -> float:
        """Write data using PyTorch format."""
        start_time = time.time()
        torch.save(data, output_path)
        write_time = time.time() - start_time
        logger.info(f"Wrote .pt file: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return write_time

    def read_pytorch(self, input_path: Path) -> Tuple[torch.Tensor, float]:
        """Read data using PyTorch format."""
        start_time = time.time()
        data = torch.load(input_path, map_location='cpu')
        read_time = time.time() - start_time
        logger.info(f"Read .pt file: {input_path} (time: {read_time:.4f}s)")
        return data, read_time

    def test_data_integrity(self):
        """Test 1: Data integrity - save and load should match."""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: Data Integrity")
        logger.info("="*60)

        # Generate data
        original_data = self.generate_synthetic_data(n_samples=1000, n_dims=768)

        # Write and read parquet
        parquet_path = self.temp_dir / "test_integrity.parquet"
        _ = self.write_parquet_simple(original_data, parquet_path)
        loaded_data, _ = self.read_parquet_simple(parquet_path)

        # Check shapes match
        shape_match = original_data.shape == loaded_data.shape
        self.log_test("Shape match", shape_match,
                     f"Original: {original_data.shape}, Loaded: {loaded_data.shape}")

        # Check values are close (floating point tolerance)
        values_close = torch.allclose(original_data, loaded_data, rtol=1e-5, atol=1e-7)
        self.log_test("Values match", values_close,
                     f"Max diff: {(original_data - loaded_data).abs().max():.2e}")

        # Check dtype
        dtype_match = original_data.dtype == loaded_data.dtype
        self.log_test("Dtype match", dtype_match,
                     f"Original: {original_data.dtype}, Loaded: {loaded_data.dtype}")

        # Check statistics
        stats_match = (
            torch.isclose(original_data.mean(), loaded_data.mean(), rtol=1e-5) and
            torch.isclose(original_data.std(), loaded_data.std(), rtol=1e-5)
        )
        self.log_test("Statistics match", stats_match,
                     f"Mean: {original_data.mean():.6f} vs {loaded_data.mean():.6f}")

    def test_large_dataset(self):
        """Test 2: Large dataset handling."""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Large Dataset Handling")
        logger.info("="*60)

        # Generate larger dataset
        n_samples = 50000
        n_dims = 768
        logger.info(f"Generating large dataset: {n_samples} x {n_dims}")
        large_data = self.generate_synthetic_data(n_samples=n_samples, n_dims=n_dims)

        try:
            # Write parquet
            parquet_path = self.temp_dir / "test_large.parquet"
            write_time = self.write_parquet_simple(large_data, parquet_path)

            # Read parquet
            loaded_data, read_time = self.read_parquet_simple(parquet_path)

            # Verify
            success = large_data.shape == loaded_data.shape and torch.allclose(large_data, loaded_data, rtol=1e-5)
            self.log_test("Large dataset handling", success,
                         f"Write: {write_time:.4f}s, Read: {read_time:.4f}s")
        except Exception as e:
            self.log_test("Large dataset handling", False, str(e))

    def test_random_sampling(self):
        """Test 3: Random sampling correctness."""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Random Sampling")
        logger.info("="*60)

        # Generate data
        original_data = self.generate_synthetic_data(n_samples=1000, n_dims=768)

        # Save
        parquet_path = self.temp_dir / "test_sampling.parquet"
        self.write_parquet_simple(original_data, parquet_path)

        # Load and perform random sampling
        loaded_data, _ = self.read_parquet_simple(parquet_path)

        # Test 3a: Deterministic sampling with seed
        seed = 42
        torch.manual_seed(seed)
        sample_size = 100
        indices1 = torch.randperm(loaded_data.shape[0])[:sample_size]
        sampled1 = loaded_data[indices1]

        torch.manual_seed(seed)
        indices2 = torch.randperm(loaded_data.shape[0])[:sample_size]
        sampled2 = loaded_data[indices2]

        reproducible = torch.equal(sampled1, sampled2)
        self.log_test("Sampling reproducibility", reproducible,
                     f"Same seed produces same samples")

        # Test 3b: Different seeds produce different samples
        torch.manual_seed(seed + 1)
        indices3 = torch.randperm(loaded_data.shape[0])[:sample_size]
        sampled3 = loaded_data[indices3]

        different = not torch.equal(sampled1, sampled3)
        self.log_test("Sampling randomness", different,
                     f"Different seed produces different samples")

        # Test 3c: Sampled data is subset of original
        all_match = torch.all(torch.isin(indices1, torch.arange(loaded_data.shape[0])))
        self.log_test("Sampling validity", all_match,
                     f"All sampled indices are valid")

    def test_multiple_chunks(self):
        """Test 4: Multiple chunk writing and loading."""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Multiple Chunks")
        logger.info("="*60)

        chunk_dir = self.temp_dir / "chunks"
        chunk_dir.mkdir(exist_ok=True)

        # Generate and save multiple chunks
        n_chunks = 5
        chunks_data = []
        chunk_paths = []

        for i in range(n_chunks):
            chunk = self.generate_synthetic_data(n_samples=200, n_dims=768, seed=42 + i)
            chunks_data.append(chunk)

            chunk_path = chunk_dir / f"chunk_{i:04d}.parquet"
            self.write_parquet_simple(chunk, chunk_path)
            chunk_paths.append(chunk_path)

        logger.info(f"Saved {n_chunks} chunks")

        # Load all chunks
        loaded_chunks = []
        for path in chunk_paths:
            chunk, _ = self.read_parquet_simple(path)
            loaded_chunks.append(chunk)

        # Concatenate
        original_concat = torch.cat(chunks_data, dim=0)
        loaded_concat = torch.cat(loaded_chunks, dim=0)

        concat_match = torch.allclose(original_concat, loaded_concat, rtol=1e-5)
        self.log_test("Multiple chunks concatenation", concat_match,
                     f"Concatenated shape: {loaded_concat.shape}")

    def test_performance_comparison(self):
        """Test 5: Performance benchmark vs .pt format."""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Performance Comparison (.pt vs Parquet)")
        logger.info("="*60)

        # Generate data
        data = self.generate_synthetic_data(n_samples=10000, n_dims=768)

        # Test PyTorch format
        pt_path = self.temp_dir / "test_perf.pt"
        pt_write_time = self.write_pytorch(data, pt_path)
        pt_data, pt_read_time = self.read_pytorch(pt_path)
        pt_size = pt_path.stat().st_size / 1024 / 1024

        # Test Parquet format
        parquet_path = self.temp_dir / "test_perf.parquet"
        parquet_write_time = self.write_parquet_simple(data, parquet_path)
        parquet_data, parquet_read_time = self.read_parquet_simple(parquet_path)
        parquet_size = parquet_path.stat().st_size / 1024 / 1024

        # Log comparison
        logger.info("\nPerformance Comparison (10K x 768):")
        logger.info(f"  PyTorch .pt:")
        logger.info(f"    Write time: {pt_write_time:.4f}s")
        logger.info(f"    Read time:  {pt_read_time:.4f}s")
        logger.info(f"    File size:  {pt_size:.2f} MB")
        logger.info(f"  Parquet:")
        logger.info(f"    Write time: {parquet_write_time:.4f}s")
        logger.info(f"    Read time:  {parquet_read_time:.4f}s")
        logger.info(f"    File size:  {parquet_size:.2f} MB")
        logger.info(f"  Size ratio (Parquet/PyTorch): {parquet_size/pt_size:.2f}x")

        # Data integrity check
        data_match = torch.allclose(pt_data, parquet_data, rtol=1e-5)
        self.log_test("Format equivalence", data_match,
                     f"Both formats produce identical data")

    def test_empty_handling(self):
        """Test 6: Edge cases - empty data."""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: Edge Cases - Empty Data")
        logger.info("="*60)

        try:
            # Test empty tensor
            empty_data = torch.empty(0, 768, dtype=torch.float32)
            parquet_path = self.temp_dir / "test_empty.parquet"
            self.write_parquet_simple(empty_data, parquet_path)
            loaded_data, _ = self.read_parquet_simple(parquet_path)

            success = loaded_data.shape[0] == 0 and loaded_data.shape[1] == 768
            self.log_test("Empty data handling", success,
                         f"Empty tensor handled correctly: {loaded_data.shape}")
        except Exception as e:
            self.log_test("Empty data handling", False, str(e))

    def test_single_sample(self):
        """Test 7: Edge cases - single sample."""
        logger.info("\n" + "="*60)
        logger.info("TEST 7: Edge Cases - Single Sample")
        logger.info("="*60)

        try:
            # Test single sample
            single_data = torch.randn(1, 768, dtype=torch.float32)
            parquet_path = self.temp_dir / "test_single.parquet"
            self.write_parquet_simple(single_data, parquet_path)
            loaded_data, _ = self.read_parquet_simple(parquet_path)

            success = loaded_data.shape == (1, 768) and torch.allclose(single_data, loaded_data)
            self.log_test("Single sample handling", success,
                         f"Single sample handled correctly: {loaded_data.shape}")
        except Exception as e:
            self.log_test("Single sample handling", False, str(e))

    def run_all_tests(self):
        """Run all tests."""
        logger.info("\n" + "="*80)
        logger.info("PARQUET I/O TEST SUITE")
        logger.info("="*80)

        if not HAS_PARQUET:
            logger.error("PyArrow not available. Cannot run tests.")
            return False

        try:
            self.test_data_integrity()
            self.test_large_dataset()
            self.test_random_sampling()
            self.test_multiple_chunks()
            self.test_performance_comparison()
            self.test_empty_handling()
            self.test_single_sample()
        except Exception as e:
            logger.error(f"Test suite failed with error: {e}", exc_info=True)
            return False

        return True

    def print_summary(self):
        """Print test summary."""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
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
            return False


def main():
    """Main entry point."""
    if not HAS_PARQUET:
        print("ERROR: PyArrow is required. Install with:")
        print("  pip install pyarrow")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tester = ParquetIOTester(temp_path)

        # Run tests
        tester.run_all_tests()

        # Print summary
        all_passed = tester.print_summary()

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
