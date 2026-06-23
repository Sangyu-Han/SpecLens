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

## Local Deletion Probe

FRI score가 배경처럼 충분하지만 필요하지 않은 patch를 높게 줄 때는 local deletion probe를 후처리로 붙일 수 있습니다. Core helper는 모델을 직접 호출하지 않고, caller가 mask별 scalar 값을 평가한 뒤 넘기는 구조입니다.

```python
from src.core.attribution.fri import (
    LocalProbeConfig,
    local_probe_readout_from_values,
    make_local_probe_masks,
)

cfg = LocalProbeConfig(
    probe_calls=32,
    radii=(1, 2, 1, 3),
    coverage_radius=0,
)

probe = make_local_probe_masks(
    prior=prior_scores,
    n_patches=196,
    grid_size=14,
    config=cfg,
)

# caller가 모델/SAE/objective runtime으로 평가
full_value = evaluate_mask(np.ones(196, dtype=np.float32))
deleted_values = np.asarray([evaluate_mask(mask) for mask in probe.masks], dtype=np.float32)

scores, diagnostics = local_probe_readout_from_values(
    prior=prior_scores,
    full_value=full_value,
    deleted_values=deleted_values,
    groups=probe.groups,
    fallback_scores={
        "final": final_scores,
        "soft": soft_scores,
        "grad_rank": grad_rank_scores,
    },
    readout="adaptive_sparse_content",
    config=cfg,
    drop_mode="relative",  # probability류 objective
    content_scores=patch_content_norms,
    low_fallback="final",
    mid_fallback="grad_rank",
)
```

`drop_mode="relative"`는 probability처럼 scale이 양수인 objective에 맞고, logit/margin처럼 signed scalar를 그대로 비교할 때는 `drop_mode="absolute"`를 쓸 수 있습니다.

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
