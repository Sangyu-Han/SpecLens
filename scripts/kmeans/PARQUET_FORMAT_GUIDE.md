# Parquet Format Migration Guide

## Overview

This guide documents the migration from **columnar format** to **FixedSizeList format** for storing activation data in Parquet files.

## Format Comparison

### Old Format: Columnar (DEPRECATED)

Each dimension is stored as a separate column.

**Structure**:
```
Table Schema:
  dim_0: float32
  dim_1: float32
  dim_2: float32
  ...
  dim_767: float32
```

**Code**:
```python
# DEPRECATED - DO NOT USE
def write_parquet_columnar(data: torch.Tensor, output_path: Path):
    data_np = data.numpy().astype(np.float32)
    n_samples, n_dims = data_np.shape

    # Create individual columns for each dimension
    arrays = []
    names = []
    for i in range(n_dims):
        arrays.append(pa.array(data_np[:, i], type=pa.float32()))
        names.append(f"dim_{i}")

    table = pa.table({name: arr for name, arr in zip(names, arrays)})
    pq.write_table(table, output_path)
```

**Problems**:
- Creates 768 separate columns (for 768D data)
- Schema is verbose and difficult to validate
- Random row sampling is inefficient
- Metadata explosion for high-dimensional data
- Difficult to work with in data analysis tools

### New Format: FixedSizeList (RECOMMENDED)

Each row is stored as a fixed-size list of floats.

**Structure**:
```
Table Schema:
  activations: fixed_size_list<element: float>[768]
```

**Code**:
```python
# RECOMMENDED - USE THIS FORMAT
def write_parquet_fixedsizelist(data: torch.Tensor, output_path: Path):
    data_np = data.numpy().astype(np.float32)
    n_samples, n_dims = data_np.shape

    # Metadata
    metadata = {
        b'n_samples': str(n_samples).encode(),
        b'n_dims': str(n_dims).encode(),
        b'dtype': 'float32'.encode(),
        b'format': 'fixedsizelist'.encode(),
    }

    # Create FixedSizeList array - SINGLE COLUMN
    rows_as_lists = [row.tolist() for row in data_np]
    activations_array = pa.array(
        rows_as_lists,
        type=pa.list_(pa.float32(), n_dims)
    )

    # Create table
    table = pa.table({'activations': activations_array})
    table = table.replace_schema_metadata(metadata)

    # Write with compression
    pq.write_table(table, output_path, compression='snappy')
```

**Benefits**:
- Single column schema (clean and simple)
- Efficient random row sampling
- Natural representation of (N, D) tensor data
- Better compatibility with data science tools
- Metadata is compact and readable

## Reading Data

### Old Format (Columnar)
```python
def read_parquet_columnar(input_path: Path):
    table = pq.read_table(input_path)

    # Reconstruct from individual dimension columns
    arrays = []
    for i in range(n_dims):
        col_name = f"dim_{i}"
        arrays.append(table.column(col_name).to_numpy())

    # Stack into (n_samples, n_dims) array
    data_np = np.column_stack(arrays).astype(np.float32)
    data = torch.from_numpy(data_np).float()
    return data
```

### New Format (FixedSizeList)
```python
def read_parquet_fixedsizelist(input_path: Path):
    table = pq.read_table(input_path)

    # Extract activations column
    activations_column = table.column('activations')

    # Convert to numpy (each element is a list)
    data_list = activations_column.to_pylist()
    data_np = np.array(data_list, dtype=np.float32)

    # Convert to tensor
    data = torch.from_numpy(data_np).float()
    return data
```

## Random Sampling

### Old Format (Inefficient)
```python
# Must read all columns first, then sample rows
table = pq.read_table(input_path)
# Convert all columns to numpy...
# Then sample with indices
```

### New Format (Efficient)
```python
# Read metadata to get row count
metadata = pq.read_metadata(input_path)
total_rows = metadata.num_rows

# Generate random indices
np.random.seed(seed)
indices = np.random.choice(total_rows, size=sample_size, replace=False)
indices.sort()  # Better I/O performance

# Read table and sample rows directly
table = pq.read_table(input_path)
sampled_table = table.take(indices)

# Convert single column to numpy
data_list = sampled_table.column('activations').to_pylist()
data_np = np.array(data_list, dtype=np.float32)
```

## Migration Checklist

### For Extraction Scripts
- [ ] Replace columnar write with FixedSizeList write
- [ ] Add format metadata (`b'format': 'fixedsizelist'.encode()`)
- [ ] Update chunk file extension (optional: keep `.parquet`)
- [ ] Test with small dataset first
- [ ] Verify schema with `pq.read_schema()`

### For Training Scripts
- [ ] Update read function to use FixedSizeList format
- [ ] Implement random sampling if needed
- [ ] Update validation to check for `activations` column
- [ ] Test with both formats during transition
- [ ] Update error messages to reference new format

### For Testing
- [ ] Run `test_parquet_implementation.py` to validate implementation
- [ ] Test empty data edge case
- [ ] Test single sample edge case
- [ ] Test large dimensions (2048D+)
- [ ] Benchmark performance vs .pt format
- [ ] Verify backward compatibility with .pt files

## Schema Validation

### Check File Format
```python
import pyarrow.parquet as pq

# Read schema
schema = pq.read_schema('chunk_000000.parquet')
print(schema)

# Expected output (FixedSizeList):
# activations: fixed_size_list<element: float>[768]
#   child 0, element: float
# -- schema metadata --
# n_samples: '1000'
# n_dims: '768'
# dtype: 'float32'
# format: 'fixedsizelist'

# Old output (Columnar) - DEPRECATED:
# dim_0: float
# dim_1: float
# ...
# dim_767: float
```

### Detect Format at Runtime
```python
def detect_parquet_format(file_path: Path) -> str:
    """Detect whether file uses columnar or fixedsizelist format."""
    schema = pq.read_schema(file_path)

    # Check metadata first
    metadata = schema.metadata or {}
    if b'format' in metadata:
        return metadata[b'format'].decode()

    # Fallback: check schema structure
    if 'activations' in schema.names:
        field_type = schema.field('activations').type
        if isinstance(field_type, (pa.ListType, pa.FixedSizeListType)):
            return 'fixedsizelist'

    # Check for columnar format (dim_0, dim_1, ...)
    if 'dim_0' in schema.names and 'dim_1' in schema.names:
        return 'columnar'

    return 'unknown'
```

## Performance Characteristics

### Write Performance
| Format        | 1K samples | 10K samples | 50K samples |
|---------------|------------|-------------|-------------|
| Columnar      | ~0.05s     | ~0.40s      | ~2.0s       |
| FixedSizeList | ~0.04s     | ~0.35s      | ~2.0s       |
| .pt (baseline)| ~0.002s    | ~0.02s      | ~0.07s      |

**Conclusion**: Both Parquet formats are ~25-50x slower than .pt for writes.

### Read Performance
| Format        | 1K samples | 10K samples | 50K samples |
|---------------|------------|-------------|-------------|
| Columnar      | ~0.15s     | ~1.6s       | ~8.0s       |
| FixedSizeList | ~0.14s     | ~1.5s       | ~7.5s       |
| .pt (baseline)| ~0.001s    | ~0.005s     | ~0.06s      |

**Conclusion**: FixedSizeList is ~5-10% faster than columnar for reads.

### File Size
| Format        | 1K samples | 10K samples | 50K samples |
|---------------|------------|-------------|-------------|
| Columnar      | 3.55 MB    | 30.0 MB     | 147.5 MB    |
| FixedSizeList | 3.53 MB    | 29.9 MB     | 147.1 MB    |
| .pt (baseline)| 2.93 MB    | 29.3 MB     | 146.5 MB    |

**Conclusion**: All formats have similar file sizes (~1.0-1.2x ratio).

## Common Issues

### Issue 1: "Could not convert ... with type float"
```python
# WRONG - passing individual values instead of lists
arrays = [pa.array(row.tolist(), type=pa.list_(pa.float32(), n_dims))
          for row in data_np]  # Each row is already a list!

# CORRECT - create single array of lists
rows_as_lists = [row.tolist() for row in data_np]
activations_array = pa.array(rows_as_lists, type=pa.list_(pa.float32(), n_dims))
```

### Issue 2: Empty data has wrong shape
```python
# Empty data may return shape (0,) instead of (0, 768)
# Solution: check numel() instead of exact shape
empty_ok = loaded_data.numel() == 0
```

### Issue 3: Schema validation fails
```python
# Check for both ListType and FixedSizeListType
field_type = schema.field('activations').type
is_list = isinstance(field_type, (pa.ListType, pa.FixedSizeListType))
```

## Best Practices

### 1. Always add metadata
```python
metadata = {
    b'n_samples': str(n_samples).encode(),
    b'n_dims': str(n_dims).encode(),
    b'dtype': 'float32'.encode(),
    b'format': 'fixedsizelist'.encode(),
}
table = table.replace_schema_metadata(metadata)
```

### 2. Use compression
```python
pq.write_table(table, output_path, compression='snappy')
# Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'
```

### 3. Sort indices for random sampling
```python
# Better I/O performance with sorted indices
indices = np.random.choice(total_rows, size=sample_size, replace=False)
indices.sort()  # Add this line
```

### 4. Validate schema after writing
```python
# Verify format is correct
schema = pq.read_schema(output_path)
assert 'activations' in schema.names
assert isinstance(schema.field('activations').type,
                 (pa.ListType, pa.FixedSizeListType))
```

### 5. Handle both formats during transition
```python
def read_parquet_auto(file_path: Path):
    """Auto-detect format and read accordingly."""
    format_type = detect_parquet_format(file_path)

    if format_type == 'fixedsizelist':
        return read_parquet_fixedsizelist(file_path)
    elif format_type == 'columnar':
        return read_parquet_columnar(file_path)
    else:
        raise ValueError(f"Unknown format: {format_type}")
```

## Conversion Script

To convert existing columnar files to FixedSizeList:

```python
#!/usr/bin/env python3
import pyarrow.parquet as pq
from pathlib import Path

def convert_columnar_to_fixedsizelist(input_path: Path, output_path: Path):
    """Convert columnar Parquet file to FixedSizeList format."""
    # Read columnar format
    table = pq.read_table(input_path)

    # Extract data
    n_cols = len(table.column_names)
    data_np = np.column_stack([
        table.column(f'dim_{i}').to_numpy()
        for i in range(n_cols)
    ]).astype(np.float32)

    # Write as FixedSizeList
    write_parquet_fixedsizelist(
        torch.from_numpy(data_np),
        output_path
    )

# Usage
input_dir = Path('old_format/')
output_dir = Path('new_format/')
output_dir.mkdir(exist_ok=True)

for parquet_file in input_dir.glob('chunk_*.parquet'):
    output_file = output_dir / parquet_file.name
    convert_columnar_to_fixedsizelist(parquet_file, output_file)
    print(f"Converted: {parquet_file.name}")
```

## Testing

Run the comprehensive test suite:
```bash
python scripts/kmeans/test_parquet_implementation.py
```

Expected output:
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

## References

- **Test Suite**: `scripts/kmeans/test_parquet_implementation.py`
- **Test README**: `scripts/kmeans/TEST_PARQUET_README.md`
- **Arrow Docs**: https://arrow.apache.org/docs/python/parquet.html
- **FixedSizeList**: https://arrow.apache.org/docs/python/api/datatypes.html#pyarrow.list_
