# K-means Clustering Initialization for Archetypal SAE

이 디렉토리는 Archetypal SAE 학습을 위한 K-means 클러스터링 초기화 파이프라인을 포함합니다.

## 📁 디렉토리 구조

```
scripts/kmeans/
├── core/                           # 핵심 실행 스크립트
│   ├── extract_activations_for_kmeans.py  # Activation 추출 (inference 기반)
│   ├── train_kmeans_centers.py            # Faiss K-means 학습
│   └── batch_kmeans_pipeline.sh           # 전체 파이프라인 오케스트레이션
├── utils/                          # 유틸리티 함수
│   └── load_kmeans_activations.py         # 추출된 데이터 로딩 API
├── examples/                       # 사용 예제
│   └── example_kmeans_init_sae.py         # 전체 파이프라인 예제
├── tests/                          # 테스트 스크립트
│   ├── test_extraction_pipeline.py        # 추출 파이프라인 테스트
│   └── test_random_subsample.py           # Random subsampling 테스트
└── docs/                           # 문서
    ├── KMEANS_EXTRACTION_README.md        # 상세 문서
    ├── QUICK_START_KMEANS.md              # 빠른 시작 가이드
    └── RANDOM_SUBSAMPLE_IMPLEMENTATION.md # Random subsample 구현 문서
```

## 🚀 빠른 시작

### 1. 전체 파이프라인 실행

```bash
# Config 편집
vi scripts/kmeans/core/batch_kmeans_pipeline.sh
# PRIMARY_LAYER, LAYERS, CLUSTERS 설정

# 실행
./scripts/kmeans/core/batch_kmeans_pipeline.sh
```

### 2. 단계별 실행

```bash
# Step 1: Activation 추출
python scripts/kmeans/core/extract_activations_for_kmeans.py \
    --config configs/sam2_sav_ra-ar_train.yaml \
    --output-dir /media/sangyu/T7/kmeans_data/my_run \
    --primary-layer "model.sam_mask_decoder.transformer.layers.0@0" \
    --target-tokens-primary 10000000 \
    --layers "model.sam_mask_decoder.transformer.layers.0@0" \
    --auto-probe

# Step 2: K-means 학습
python scripts/kmeans/core/train_kmeans_centers.py \
    --data-dir /media/sangyu/T7/kmeans_data/my_run/model_sam_mask_decoder_transformer_layers_0@0 \
    --n-clusters 6144 \
    --output centroids.pt

# Step 3: 학습에서 사용
# configs/sam2_sav_ra-ar_train.yaml에 추가:
#   sae.training.kmeans_init.enabled: true
python scripts/train_sae_config.py --config configs/sam2_sav_ra-ar_train.yaml
```

## 📚 주요 기능

### 1. Spatial Bias 제거
- **문제**: 기존 stride는 64×64 그리드에서 세로줄만 샘플링 (12.5% 커버리지)
- **해결**: Random subsampling으로 100% 커버리지
- **구현**: `src/core/sae/activation_stores/universal_activation_store.py`

### 2. Inference 기반 수집
- **문제**: 토큰 수 기반 추적 → 샘플 다양성 부족
- **해결**: Inference 수로 추적, 모든 레이어 동기화
- **예**: SAMv2 4096 tokens/inf × 2440 infs = 10M tokens

### 3. Multi-Spec 처리
- 각 spec별로 다른 tokens/inference 자동 처리
- `@0`: 4096 tokens → `@1`: 16 tokens → `@2`: 1 token

## 📖 상세 문서

- **상세 문서**: [docs/KMEANS_EXTRACTION_README.md](docs/KMEANS_EXTRACTION_README.md)
- **빠른 가이드**: [docs/QUICK_START_KMEANS.md](docs/QUICK_START_KMEANS.md)
- **Random Subsample**: [docs/RANDOM_SUBSAMPLE_IMPLEMENTATION.md](docs/RANDOM_SUBSAMPLE_IMPLEMENTATION.md)

## 🧪 테스트

```bash
# 추출 파이프라인 테스트
python scripts/kmeans/tests/test_extraction_pipeline.py

# Random subsampling 테스트
python scripts/kmeans/tests/test_random_subsample.py
```

## 📊 예상 성능 (SAMv2)

| 단계 | 시간 | 저장공간 (4 layers) |
|------|------|-------------------|
| Probe | ~10초 | Negligible |
| Extraction (2440 inferences) | ~15분 | ~40 GB |
| K-means (per layer) | ~5-10분 | ~500 MB |
| **전체** | **~1시간** | **~40 GB** |

## 🔧 Config 예제

```yaml
# configs/sam2_sav_ra-ar_train.yaml
sae:
  training:
    sae_type: "ra-ar"
    kmeans_init:
      enabled: true
      centroids_dir: "/media/sangyu/T7/kmeans_centers"
```

## 🐛 문제 해결

### "K-means centroids not found"
```bash
# Centroids 경로 확인
ls -lh /media/sangyu/T7/kmeans_centers/*/centroids.pt

# 없으면 추출 + K-means 실행
./scripts/kmeans/core/batch_kmeans_pipeline.sh
```

### "Dimension mismatch"
```bash
# Checkpoint 확인
cat /media/sangyu/T7/kmeans_data/my_run/checkpoint.json | grep act_size

# 다시 추출 (올바른 config 사용)
python scripts/kmeans/core/extract_activations_for_kmeans.py ...
```

## 📝 참고사항

1. **첫 실행**: 전체 파이프라인 실행 (~1시간)
2. **재학습**: Centroids만 재사용, 추출 불필요
3. **레이어 추가**: 해당 레이어만 추출 + K-means
4. **저장공간**: 10M tokens/layer ≈ 3GB (768D 기준)
