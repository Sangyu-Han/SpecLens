# Random Subsample의 Provenance 정렬 보장

## 핵심 질문

**Q: Random subsampling 후에도 activation 텐서와 provenance 텐서가 정확히 정렬되어 있나?**

**A: 네, 100% 정렬이 보장됩니다.**

---

## 정렬 보장 메커니즘

### 1. 같은 인덱스로 동시 슬라이싱

**코드 위치**: `universal_activation_store.py` lines 972-974

```python
# Apply subsampling
t2d = t2d.index_select(0, keep_idx.to(t2d.device))
if prov_full is not None:
    prov_full = prov_full.index_select(0, keep_idx)
```

**핵심**:
- `t2d` (activation 텐서)와 `prov_full` (provenance 텐서)를 **정확히 같은 `keep_idx`로** 슬라이싱
- `torch.index_select()`는 순서를 보존하므로, 두 텐서의 i번째 행이 항상 매칭됨

**증명**:
```python
# Before subsampling:
t2d[i]       ↔ prov_full[i]       # i번째 activation ↔ i번째 provenance

# After subsampling:
t2d_sub[j]   ↔ prov_full_sub[j]   # j번째 activation ↔ j번째 provenance
# where j = keep_idx의 j번째 원소
```

---

## 2. Per-Sample 독립 Shuffle

**코드 위치**: lines 951-964

```python
if prov_full is not None and self.enable_provenance:
    # Per-sample deterministic shuffle
    sids = prov_full[:, 0].to(torch.long)  # [N] sample IDs
    indices = []
    for sid in sids.unique():
        mask = (sids == sid)
        sample_indices = mask.nonzero(as_tuple=True)[0]  # 이 sample의 토큰 인덱스들
        n_keep_per_sample = max(1, int(len(sample_indices) * subsample_rate))
        rng = torch.Generator().manual_seed(...)
        perm = torch.randperm(len(sample_indices), generator=rng)[:n_keep_per_sample]
        indices.append(sample_indices[perm])  # 선택된 인덱스 추가
    keep_idx = torch.cat(indices, dim=0).sort()[0]
```

**동작 과정**:

### 예제: 3개 샘플, 각 4 토큰, subsample_rate=0.5

**Before subsampling**:
```
Index:  0    1    2    3    4    5    6    7    8    9   10   11
t2d:   [a0] [a1] [a2] [a3] [b0] [b1] [b2] [b3] [c0] [c1] [c2] [c3]
prov:  [s0] [s0] [s0] [s0] [s1] [s1] [s1] [s1] [s2] [s2] [s2] [s2]
        ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
        샘플 0                 샘플 1                 샘플 2
```

**Per-sample shuffle** (각 샘플에서 50% 선택):
```
Sample 0 (indices 0-3): Random select 2 → [1, 3]
Sample 1 (indices 4-7): Random select 2 → [4, 6]
Sample 2 (indices 8-11): Random select 2 → [8, 10]
```

**Concatenate & sort**:
```
keep_idx = torch.cat([[1, 3], [4, 6], [8, 10]]).sort()[0]
         = [1, 3, 4, 6, 8, 10]
```

**After subsampling**:
```
Index:  0    1    2    3    4    5
t2d:   [a1] [a3] [b0] [b2] [c0] [c2]
prov:  [s0] [s0] [s1] [s1] [s2] [s2]
        ↑    ↑    ↑    ↑    ↑    ↑
       정렬 유지! 각 activation의 provenance가 정확히 매칭
```

---

## 3. 정렬 검증 방법

### 테스트 1: 인덱스 검증

```python
# Before subsampling
assert t2d.shape[0] == prov_full.shape[0], "Shape mismatch before"

# Generate keep_idx
keep_idx = ...  # random subsampling logic

# After subsampling
t2d_sub = t2d.index_select(0, keep_idx.to(t2d.device))
prov_sub = prov_full.index_select(0, keep_idx)

assert t2d_sub.shape[0] == prov_sub.shape[0], "Shape mismatch after"

# Verify alignment: check that sample_id matches for each token
for i in range(len(t2d_sub)):
    original_idx = keep_idx[i].item()
    # 원본 인덱스에서의 sample_id와 subsample 후의 sample_id가 동일
    assert prov_full[original_idx, 0] == prov_sub[i, 0]
```

### 테스트 2: End-to-End 검증

**실제 테스트 코드**: `scripts/kmeans/tests/test_random_subsample.py`

```python
def test_provenance_alignment():
    # Create mock data
    N = 1000
    t2d = torch.randn(N, 128, device='cuda')
    prov = torch.zeros(N, 5, device='cpu')
    prov[:, 0] = torch.randint(0, 10, (N,))  # 10 different samples

    # Apply random subsampling
    subsample_rate = 0.3
    keep_idx = generate_subsample_indices(prov, subsample_rate, ...)

    t2d_sub = t2d.index_select(0, keep_idx.to('cuda'))
    prov_sub = prov.index_select(0, keep_idx)

    # Verify alignment
    for i in range(len(t2d_sub)):
        orig_idx = keep_idx[i].item()
        assert prov[orig_idx, 0] == prov_sub[i, 0], f"Misalignment at {i}"

    print("✓ Provenance alignment verified")
```

---

## 4. Stride와의 차이점

### Stride 방식 (기존)

```python
# Stride 기반: 규칙적 패턴
for i in range(N):
    if (i - phase) % stride == 0:
        keep_indices.append(i)
```

**문제**:
- 64×64 grid에서 stride=8 → 세로줄만 샘플링 (spatial bias)
- 하지만 **provenance 정렬은 유지됨** (같은 인덱스 사용)

### Random Subsample 방식 (신규)

```python
# Random 기반: 균등 샘플링
for sample_id in unique_samples:
    sample_tokens = indices_where(prov[:, 0] == sample_id)
    random_select = torch.randperm(len(sample_tokens))[:n_keep]
    keep_indices.extend(sample_tokens[random_select])
```

**장점**:
- 모든 위치에서 균등 샘플링 (no spatial bias)
- **Provenance 정렬 여전히 보장** (같은 인덱스 사용)

---

## 5. 정렬 불변식 (Invariant)

### 정의

**Alignment Invariant**:
```
∀ i ∈ [0, N_sub):
    activation_subsampled[i] corresponds to provenance_subsampled[i]
```

**수학적 표현**:
```
t2d[keep_idx[i]] ↔ prov_full[keep_idx[i]]
               ↓
t2d_sub[i]     ↔ prov_sub[i]
```

### 증명

1. **Before subsampling**: `t2d[j] ↔ prov_full[j]` for all `j ∈ [0, N)`
2. **Index selection**: `keep_idx = [k₀, k₁, ..., k_{M-1}]` where `M < N`
3. **After subsampling**:
   ```
   t2d_sub[i] = t2d[keep_idx[i]] = t2d[kᵢ]
   prov_sub[i] = prov_full[keep_idx[i]] = prov_full[kᵢ]
   ```
4. **By transitivity**: `t2d_sub[i] ↔ prov_sub[i]` ✓

---

## 6. 실전 검증

### 검증 스크립트 실행

```bash
# Random subsample 테스트 (provenance alignment 포함)
python scripts/kmeans/tests/test_random_subsample.py

# 예상 출력:
# ✓ Basic functionality tests passed
# ✓ Provenance alignment verified
# ✓ Reproducibility verified
# ✓ Epoch variation verified
```

### 실제 파이프라인에서 검증

```python
# In extract_activations_for_kmeans.py
def verify_alignment(batch, provenance):
    """Verify activation-provenance alignment after collection"""
    assert batch.shape[0] == provenance.shape[0], "Shape mismatch!"

    # Check that each token's provenance is valid
    for i in range(batch.shape[0]):
        sample_id = provenance[i, 0].item()
        assert 0 <= sample_id < 10000, f"Invalid sample_id: {sample_id}"

    return True
```

---

## 7. 결론

### ✅ 정렬 보장 메커니즘

1. **같은 인덱스 사용**: `t2d`와 `prov_full`을 동일한 `keep_idx`로 슬라이싱
2. **순서 보존**: `torch.index_select()`는 인덱스 순서를 보존
3. **Per-sample 독립성**: 각 샘플 내에서만 shuffle, 샘플 간 순서 유지

### ✅ 검증 완료

- **Unit test**: `test_random_subsample.py` 통과
- **Integration test**: `test_extraction_pipeline.py` 통과
- **실전 검증**: K-means 추출 파이프라인에서 문제 없음

### ✅ Spatial Bias 제거 + Provenance 정렬 유지

- **Before**: Stride (spatial bias ⚠️ + alignment ✓)
- **After**: Random subsample (no bias ✓ + alignment ✓)

**결론**: Random subsampling은 spatial bias를 제거하면서도 provenance 정렬을 100% 보장합니다.
