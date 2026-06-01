# FRI 사용법

FRI는 target feature activation을 회복하는 patch soft insertion mask를 최적화하는 attribution 방법입니다. 이전 실험 코드에서는 `cautious_cos`라는 이름으로 불렸고, 현재는 `src.core.attribution.fri`의 공용 구현이 기본입니다.

## Python API

```python
from src.core.attribution.fri import FRIConfig, run_fri

result = run_fri(
    n_patches=196,
    grid_size=14,
    objective_for_mask=objective_for_mask,
    full_objective=full_objective,
    baseline_objective=baseline_objective,
    irrelevance=irrelevance,
    config=FRIConfig(
        steps=32,
        lr=0.45,
        lr_end=0.01,
        tv_weight=0.01,
        irrelevance_weight=0.05,
        seed=0,
    ),
)

scores = result.scores
```

`objective_for_mask(mask)`는 길이 `n_patches`의 soft insertion mask를 받아 scalar objective를 반환해야 합니다. 모델 hook, SAE 호출, baseline mixing은 caller 쪽 runtime이 담당하고, FRI solver는 모델 구조를 알지 않습니다.

## Benchmark에서 사용

새 구현이 기본입니다.

```bash
python scripts/run_feature_erf_paper_benchmark.py \
  --pack clip \
  --blocks 6 \
  --methods fri \
  --n-features 2 \
  --n-images 2
```

기존 method 이름도 호환됩니다.

```bash
python scripts/run_feature_erf_paper_benchmark.py \
  --pack clip \
  --blocks 6 \
  --methods cautious_cos \
  --n-features 2 \
  --n-images 2
```

## Legacy 동일성 검증

legacy 구현과 core 구현의 attribution score가 같은지 확인하려면:

```bash
python scripts/compare_fri_core_legacy.py \
  --pack clip \
  --blocks 2 6 10 \
  --n-features 2 \
  --n-images 2
```

`scripts/run_feature_erf_paper_benchmark.py`에서도 직접 비교할 수 있습니다.

```bash
python scripts/run_feature_erf_paper_benchmark.py \
  --pack clip \
  --blocks 6 \
  --methods cautious_cos \
  --n-features 1 \
  --n-images 1 \
  --fri-implementation compare
```

## 이름 정리

- 권장 이름: `fri`
- 호환 이름: `cautious_cos`
- 공용 구현: `src.core.attribution.fri.run_fri`
- legacy 비교 대상: `/home/sangyu/Desktop/Master/codex_research_softins/eval_multimodel_erf.py`
