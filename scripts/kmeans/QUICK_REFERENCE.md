# Parquet Implementation Quick Reference

## TL;DR

**Test Suite**: `test_parquet_implementation.py`
**Status**: ✅ All 21 tests passing
**Format**: FixedSizeList (NOT columnar)
**Performance**: ~100x slower than .pt, same file size

## Run Tests

```bash
# Full test suite (21 tests, ~26s)
python scripts/kmeans/test_parquet_implementation.py

# Quick test (18 tests, ~23s)
python scripts/kmeans/test_parquet_implementation.py --skip-integration

# Debug mode
python scripts/kmeans/test_parquet_implementation.py --verbose
```

## FixedSizeList Format (Correct)

### Write
```python
import pyarrow as pa
import pyarrow.parquet as pq

# Convert to lists
data_np = data.numpy().astype(np.float32)
rows_as_lists = [row.tolist() for row in data_np]

# Create FixedSizeList array
activations_array = pa.array(
    rows_as_lists,
    type=pa.list_(pa.float32(), n_dims)
)

# Write with metadata
table = pa.table({'activations': activations_array})
table = table.replace_schema_metadata({
    b'n_samples': str(n_samples).encode(),
    b'n_dims': str(n_dims).encode(),
    b'format': 'fixedsizelist'.encode(),
})
pq.write_table(table, output_path, compression='snappy')
```

### Read
```python
# Load table
table = pq.read_table(input_path)

# Convert to numpy
data_list = table.column('activations').to_pylist()
data_np = np.array(data_list, dtype=np.float32)

# Convert to tensor
data = torch.from_numpy(data_np).float()
```

### Random Sample
```python
# Get metadata
metadata = pq.read_metadata(input_path)
total_rows = metadata.num_rows

# Generate random indices
np.random.seed(seed)
indices = np.random.choice(total_rows, size=sample_size, replace=False)
indices.sort()  # Better I/O

# Sample
table = pq.read_table(input_path)
sampled_table = table.take(indices)
```

## Performance Cheat Sheet

| Operation      | Parquet    | .pt        | Ratio |
|----------------|------------|------------|-------|
| Write (50K)    | 1.96s      | 0.07s      | 27x   |
| Read (50K)     | 7.53s      | 0.06s      | 126x  |
| Size (50K)     | 147.11 MB  | 146.49 MB  | 1.0x  |
| Random (10K)   | 1.79s      | N/A        | N/A   |

## Common Mistakes

### ❌ Wrong: Creating list per row
```python
# DON'T DO THIS
arrays = [pa.array(row.tolist(), type=pa.list_(...)) for row in data_np]
```

### ✅ Correct: Create single array of lists
```python
# DO THIS
rows_as_lists = [row.tolist() for row in data_np]
activations_array = pa.array(rows_as_lists, type=pa.list_(...))
```

## Validation

```python
# Check schema
schema = pq.read_schema('chunk_000000.parquet')
assert 'activations' in schema.names
assert isinstance(schema.field('activations').type,
                 (pa.ListType, pa.FixedSizeListType))

# Check metadata
metadata = schema.metadata
assert b'format' in metadata
assert metadata[b'format'] == b'fixedsizelist'
```

## Documentation

- **Test Suite**: `test_parquet_implementation.py` (comprehensive tests)
- **Test README**: `TEST_PARQUET_README.md` (detailed test descriptions)
- **Format Guide**: `PARQUET_FORMAT_GUIDE.md` (migration, best practices)
- **Summary**: `TESTING_SUMMARY.md` (results, recommendations)
- **Quick Ref**: `QUICK_REFERENCE.md` (this file)

## Test Results (21/21 Passing)

```
✓ FixedSizeList shape match
✓ FixedSizeList values match (max diff: 0.00e+00)
✓ FixedSizeList dtype match
✓ FixedSizeList schema correct
✓ FixedSizeList vs .pt equivalence
✓ Random sampling size
✓ Random sampling reproducibility
✓ Random sampling randomness
✓ Random sampling distribution
✓ Integration data integrity
✓ Integration K-means centroids
✓ Integration vs .pt baseline
✓ Edge case: empty data
✓ Edge case: single sample
✓ Edge case: large dimensions
✓ Edge case: multiple chunks
✓ Performance integrity (3 configs)
✓ Backward compat: .pt format
✓ Backward compat: mixed format
```

## When to Use

**Use Parquet**:
- Archival storage
- Data sharing (cross-platform)
- Random sampling workflows
- Exploratory analysis

**Use .pt**:
- Training loops (speed critical)
- Real-time inference
- Temporary caching

**Hybrid**:
- Extract → Parquet (canonical)
- Training → Convert to .pt
- Keep both formats

## Install

```bash
pip install pyarrow torch numpy
```

## Help

```bash
python test_parquet_implementation.py --help
```

## Example Output

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
