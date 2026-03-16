# Parquet Implementation Test Suite

Comprehensive test suite for validating the Parquet FixedSizeList format refactoring in the K-means activation extraction pipeline.

## Overview

This test suite (`test_parquet_implementation.py`) validates the correctness, performance, and robustness of the Parquet-based data storage format used for K-means clustering initialization.

## Test Coverage

### 1. FixedSizeList Format I/O (5 tests)
- **Shape integrity**: Verifies data shape preservation (N x D)
- **Value integrity**: Ensures exact numerical accuracy (max diff < 1e-7)
- **Dtype preservation**: Confirms float32 dtype maintained
- **Schema validation**: Checks for correct `fixed_size_list` Arrow schema
- **Format equivalence**: Compares Parquet vs .pt format outputs

### 2. Random Sampling (4 tests)
- **Sample size**: Validates correct number of samples returned (10K from 50K)
- **Reproducibility**: Same seed produces identical samples
- **Randomness**: Different seeds produce different samples
- **Distribution**: Sample mean approximates population mean (within 0.1)

### 3. Integration Test (3 tests)
- **Data integrity**: Full extraction → training pipeline verification
- **K-means centroids**: Validates centroid computation (100 clusters)
- **Baseline comparison**: Ensures Parquet matches .pt format results

### 4. Edge Cases (4 tests)
- **Empty data**: Handles 0-sample tensors
- **Single sample**: Correctly processes 1-sample tensors
- **Large dimensions**: Supports high-dimensional data (2048D)
- **Multiple chunks**: Concatenates 10 chunks correctly

### 5. Performance Benchmarks (3 configs × 2 formats)
- **Small dataset**: 1K samples × 768 dims
- **Medium dataset**: 10K samples × 768 dims
- **Large dataset**: 50K samples × 768 dims

Metrics tracked:
- Write speed (seconds)
- Read speed (seconds)
- File size (MB)
- Compression ratio

### 6. Backward Compatibility (2 tests)
- **.pt format**: Ensures existing PyTorch format still works
- **Mixed formats**: Validates coexistence of .pt and .parquet files

**Total: 21 tests**

## Usage

### Run all tests
```bash
python scripts/kmeans/test_parquet_implementation.py
```

### Skip integration test (faster)
```bash
python scripts/kmeans/test_parquet_implementation.py --skip-integration
```

### Verbose mode (debug logging)
```bash
python scripts/kmeans/test_parquet_implementation.py --verbose
```

## Requirements

```bash
pip install pyarrow torch numpy
```

## Test Results Summary

### All Tests Pass (21/21)
```
================================================================================
TEST SUMMARY
================================================================================
Total tests: 21
Passed: 21
Failed: 0
Pass rate: 100.0%

Result: ALL TESTS PASSED ✓
```

### Performance Comparison

| Config              | Format  | Write (s) | Read (s) | Size (MB) |
|---------------------|---------|-----------|----------|-----------|
| Small (1K × 768)    | Parquet | 0.0404    | 0.1404   | 3.53      |
| Small (1K × 768)    | .pt     | 0.0022    | 0.0007   | 2.93      |
| Medium (10K × 768)  | Parquet | 0.3433    | 1.5231   | 29.90     |
| Medium (10K × 768)  | .pt     | 0.0201    | 0.0050   | 29.30     |
| Large (50K × 768)   | Parquet | 1.9648    | 7.5322   | 147.11    |
| Large (50K × 768)   | .pt     | 0.0744    | 0.0603   | 146.49    |

**Key Findings**:
- Parquet write is ~25-50x slower than .pt
- Parquet read is ~100-200x slower than .pt
- File size is nearly identical (1.00-1.20x ratio)
- **Tradeoff**: Slower I/O for better random access and interoperability

## FixedSizeList Format Specification

### Write Format
```python
# Convert numpy array to list of lists
data_np = data.numpy().astype(np.float32)  # (N, D) array
n_samples, n_dims = data_np.shape

# Create FixedSizeList array
rows_as_lists = [row.tolist() for row in data_np]
activations_array = pa.array(rows_as_lists, type=pa.list_(pa.float32(), n_dims))

# Create table with metadata
metadata = {
    b'n_samples': str(n_samples).encode(),
    b'n_dims': str(n_dims).encode(),
    b'dtype': 'float32'.encode(),
    b'format': 'fixedsizelist'.encode(),
}
table = pa.table({'activations': activations_array})
table = table.replace_schema_metadata(metadata)

# Write with compression
pq.write_table(table, output_path, compression='snappy')
```

### Read Format
```python
# Read table
table = pq.read_table(input_path)

# Extract activations column
activations_column = table.column('activations')

# Convert to numpy array
data_list = activations_column.to_pylist()
data_np = np.array(data_list, dtype=np.float32)

# Convert to tensor
data = torch.from_numpy(data_np).float()
```

### Random Sampling
```python
# Read metadata
metadata = pq.read_metadata(input_path)
total_rows = metadata.num_rows

# Generate random indices
np.random.seed(seed)
indices = np.random.choice(total_rows, size=sample_size, replace=False)
indices.sort()  # Sort for better I/O performance

# Read and sample
table = pq.read_table(input_path)
sampled_table = table.take(indices)
```

## Integration with Existing Code

### Extraction Script
```python
# In extract_activations_for_kmeans.py
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Write chunk as Parquet
data_np = buffer.numpy().astype(np.float32)
rows_as_lists = [row.tolist() for row in data_np]
activations_array = pa.array(rows_as_lists, type=pa.list_(pa.float32(), data_np.shape[1]))
table = pa.table({'activations': activations_array})
pq.write_table(table, chunk_path, compression='snappy')
```

### Training Script
```python
# In train_kmeans_centers.py

def _load_parquet_chunks(chunk_files: List[Path], max_samples: int = None):
    """Load Parquet chunks with random sampling support."""
    chunks = []
    for chunk_file in chunk_files:
        table = pq.read_table(chunk_file)

        # Optional: random sample from this chunk
        if max_samples and total_loaded + table.num_rows > max_samples:
            # Sample only what's needed
            needed = max_samples - total_loaded
            indices = np.random.choice(table.num_rows, size=needed, replace=False)
            table = table.take(indices)

        # Convert to numpy
        data_list = table.column('activations').to_pylist()
        data_np = np.array(data_list, dtype=np.float32)
        chunks.append(torch.from_numpy(data_np))

    return torch.cat(chunks, dim=0)
```

## Known Limitations

1. **Performance**: Parquet I/O is significantly slower than .pt format
   - Write: 25-50x slower
   - Read: 100-200x slower
   - **Mitigation**: Use for scenarios where random access or interoperability is needed

2. **Memory overhead**: List conversion requires temporary memory
   - Each row must be converted to Python list before Arrow conversion
   - **Mitigation**: Process in batches if memory is constrained

3. **Empty data handling**: Empty tensors may have unexpected shapes
   - Empty (0, 768) becomes (0,) after read
   - **Mitigation**: Check `numel()` instead of exact shape

## Troubleshooting

### ImportError: PyArrow not installed
```bash
pip install pyarrow
```

### Memory issues with large datasets
```bash
# Use smaller batch sizes in write_parquet_fixedsizelist
# Or subsample data before writing
```

### Schema validation errors
```python
# Verify schema is correct
import pyarrow.parquet as pq
schema = pq.read_schema('chunk_000000.parquet')
print(schema)
# Should show: fixed_size_list<element: float>[D]
```

## Future Improvements

1. **Batch processing**: Write data in smaller batches to reduce memory
2. **Streaming reads**: Implement iterator-based reading for very large files
3. **Compression tuning**: Experiment with different compression algorithms
4. **Column chunking**: Investigate Arrow's column chunking for better performance
5. **Memory mapping**: Use Arrow's memory mapping for faster reads

## References

- [Apache Arrow Documentation](https://arrow.apache.org/docs/python/)
- [PyArrow Parquet](https://arrow.apache.org/docs/python/parquet.html)
- [FixedSizeList Type](https://arrow.apache.org/docs/python/api/datatypes.html#pyarrow.list_)
