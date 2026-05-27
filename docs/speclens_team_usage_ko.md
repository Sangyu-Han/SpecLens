# SpecLens 팀 공유용 사용 가이드

이 문서는 팀원들에게 SpecLens의 기본 사용 흐름을 설명하기 위한 발표자료 초안이다. 핵심 메시지는 간단하다.

> SpecLens에서 layer spec은 모델 내부 activation을 가리키는 주소이며, 같은 주소 체계를 SAE 학습, feature indexing, attribution, circuit 분석에서 반복해서 사용한다.

---

## 1. SpecLens가 해결하는 문제

SAE를 여러 모델에 적용하려면 보통 모델마다 별도 adapter를 작성해야 한다. 모델 구조가 다르면 hook을 걸 위치, output 모양, feature를 읽는 방식도 달라지기 때문이다.

SpecLens는 이 문제를 `Spec string`으로 해결한다. 모델 내부의 PyTorch module path와 output branch를 문자열로 표현하고, 이 문자열을 parser가 해석해서 activation을 수집한다.

예를 들어 CLIP ViT의 9번째 block output을 쓰고 싶으면 다음처럼 쓴다.

```text
model.blocks.9
```

SAM2 trunk의 특정 output branch를 쓰고 싶으면 다음처럼 쓴다.

```text
model.image_encoder.trunk@3
```

즉, SpecLens 사용의 출발점은 “어떤 layer를 볼 것인가?”가 아니라 “어떤 activation 주소를 spec으로 쓸 것인가?”이다.

---

## 2. 전체 사용 흐름

팀에서 가장 자주 쓰는 흐름은 다음과 같다.

```text
1. 모델과 데이터 config 작성
2. SAE를 학습할 layer spec 지정
3. SAE 학습 실행
4. 학습된 SAE feature indexing
5. feature별 top activating sample 확인
6. attribution / ERF / output contribution으로 feature 해석
```

이때 중요한 점은 `sae.layers`에 적은 spec이 이후 분석 단계에서도 기준점이 된다는 것이다.

---

## 3. Layer Spec이란?

Layer spec은 모델 내부 activation 위치를 가리키는 문자열 주소다.

대표 예시는 다음과 같다.

```text
model.blocks.9
model.blocks.9@2
model.image_encoder.trunk@3
enc.pos::forward_with_coords@0
model.blocks.9::sae_layer#latent
```

SpecLens의 parser는 이 문자열을 다음 요소로 나누어 해석한다.

| 요소 | 예시 | 의미 |
|---|---|---|
| base module | `model.blocks.9` | PyTorch module tree에서 hook을 걸 기본 module |
| branch | `@2`, `@3` | module output이 tuple/list/dict일 때 특정 output 선택 |
| method | `::forward_with_coords` | `forward`가 아닌 특정 method 반환값 사용 |
| attribute | `#latent` | 반환 객체나 wrapper 결과에서 특정 attribute 선택 |
| alias | `alias=...`, `... as alias` | 긴 spec에 사람이 읽기 쉬운 이름 부여 |

발표에서는 다음처럼 설명하면 된다.

> layer spec은 논문에서 말하는 layer 이름이 아니라, 실제 PyTorch 모델 안에서 activation tensor를 꺼내는 주소다.

---

## 4. Spec 문법 빠른 해석표

### 4.1 기본 module output

```text
model.blocks.9
```

`model.blocks[9]`에 해당하는 module의 forward output을 가져온다. CLIP, DINO 계열 transformer block에서 가장 자주 쓰는 형태다.

### 4.2 output branch 선택

```text
model.blocks.9@2
```

module output이 tuple, list, dict처럼 여러 값을 담고 있을 때 특정 branch를 선택한다. `@2`는 세 번째 output element를 의미한다.

```text
model.image_encoder.trunk@3
```

SAM2처럼 trunk가 여러 stage output을 반환하는 경우 특정 stage activation을 고를 때 쓴다.

### 4.3 method 반환값 사용

```text
enc.pos::forward_with_coords@0
```

기본 `forward` output이 아니라 특정 method를 호출한 결과를 hook 대상으로 삼는다. `@0`은 그 method 반환값의 첫 번째 branch를 선택한다.

### 4.4 SAE latent feature target

```text
model.blocks.9::sae_layer#latent
```

학습된 SAE wrapper나 runtime에서 `latent` activation을 target으로 잡을 때 쓰는 형태다. attribution에서 “block output”이 아니라 “SAE feature activation”을 보고 싶을 때 사용한다.

---

## 5. Config에서 layer 지정하기

SAE 학습 대상 layer는 config의 `sae.layers`에 적는다.

```yaml
sae:
  layers:
    - model.blocks.0
    - model.blocks.1
    - model.blocks.2
    - model.blocks.3
    - model.blocks.4
    - model.blocks.5
    - model.blocks.6
    - model.blocks.7
    - model.blocks.8
    - model.blocks.9
    - model.blocks.10
    - model.blocks.11
```

이렇게 지정하면 SpecLens는 listed layer 전체에 hook을 등록한다. 모델 forward는 batch마다 한 번만 수행하고, 여러 layer activation을 동시에 수집해서 각 SAE를 업데이트한다.

핵심 포인트:

- `sae.layers`는 SAE별 입력 activation을 결정한다.
- layer 개수가 많아도 model forward를 layer마다 반복하지 않는다.
- spec 문자열이 checkpoint, index, attribution 결과를 이어주는 key 역할을 한다.

---

## 6. SAE 학습 실행

단일 GPU 실행 예시는 다음과 같다.

```bash
python scripts/train_sae_config.py --config configs/clip_imagenet_train.yaml
```

멀티 GPU 실행 예시는 다음과 같다.

```bash
torchrun --nproc_per_node=4 scripts/train_sae_config.py \
  --config configs/clip_imagenet_train.yaml
```

팀원에게 설명할 때는 명령어 자체보다 config 구조를 먼저 보여주는 것이 좋다.

```yaml
dataset:
  builder: src.packs.clip.train.factories:build_dataset
  root: /data/imagenet

model:
  loader: src.packs.clip.train.factories:load_model
  name: vit_base_patch16_clip_224.laion2b_ft_in12k_in1k

sae:
  layers:
    - model.blocks.9
    - model.blocks.11
  training:
    sae_type: batch-topk
    expansion_factor: 16
    k: 32
```

이 config는 다음을 의미한다.

- ImageNet dataset을 사용한다.
- CLIP ViT-B/16 모델을 로드한다.
- `model.blocks.9`, `model.blocks.11` activation에 각각 SAE를 학습한다.
- SAE variant는 `batch-topk`를 사용한다.

---

## 7. Feature Indexing

SAE를 학습한 뒤에는 feature별로 어떤 sample에서 강하게 firing하는지 찾아야 한다. 이 단계가 feature indexing이다.

```bash
python scripts/sae_index_main.py --config configs/clip_mipal_50k_index.yaml
```

멀티 GPU 예시는 다음과 같다.

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 \
  scripts/sae_index_main.py --config configs/clip_mipal_50k_index.yaml
```

indexing 결과는 보통 top-N activating samples 형태로 저장된다.

```yaml
indexing:
  out_dir: outputs/spec_lens_store/clip_50k_index
  mode: topn
  top_n: 300
  track_frequency: true
```

해석할 때는 다음 질문을 던지면 된다.

- 이 feature는 어떤 이미지들에서 가장 강하게 켜지는가?
- 같은 feature의 top samples에 공통 시각 패턴이 있는가?
- 특정 class, texture, object part, position bias에 묶여 있는가?
- firing frequency가 너무 높거나 너무 낮지는 않은가?

---

## 8. Attribution / ERF에서 Spec 쓰기

SpecLens의 장점은 학습 때 사용한 spec 체계를 attribution에서도 그대로 쓴다는 점이다.

예를 들어 block output 자체를 target으로 삼을 수 있다.

```text
model.blocks.9
```

SAE latent feature를 target으로 삼을 수도 있다.

```text
model.blocks.9::sae_layer#latent
```

개념적으로는 다음과 같다.

```python
from src.core.runtime.attribution_runtime import AttributionRuntime

runtime = AttributionRuntime(model, config)
attribution = runtime.compute(
    inputs=image_batch,
    target_spec="model.blocks.9::sae_layer#latent",
    target_unit=42,
    method="integrated_gradients",
)
```

여기서 `target_unit=42`는 해당 layer SAE의 42번 feature를 뜻한다.

발표에서는 다음 식으로 요약하면 된다.

```text
layer spec + feature id = 해석하고 싶은 SAE feature target
```

---

## 9. 새 모델에서 Spec 찾는 법

새 모델을 붙일 때는 먼저 PyTorch module tree를 확인해야 한다.

```python
for name, module in model.named_modules():
    print(name, type(module))
```

이 출력에서 SAE를 학습하고 싶은 module path를 고른 뒤 `sae.layers`에 넣는다.

예를 들어 출력에 다음 module이 있다면:

```text
model.blocks.9 <class '...'>
```

config에는 다음처럼 쓴다.

```yaml
sae:
  layers:
    - model.blocks.9
```

주의할 점:

- 논문 figure의 layer 이름과 코드의 module path가 다를 수 있다.
- `@` branch는 실제 output 구조를 확인한 뒤 정해야 한다.
- output이 tensor가 아니면 `@0`, `@key`, `#attr` 같은 선택자가 필요할 수 있다.
- spec이 길어지면 alias를 붙여 사람이 읽기 쉽게 관리한다.

---

## 10. 자주 하는 실수

### 10.1 module path가 실제 모델과 다름

증상:

```text
Module not found
```

확인 방법:

```python
dict(model.named_modules()).keys()
```

해결:

실제 `named_modules()`에 나오는 이름을 기준으로 spec을 수정한다.

### 10.2 output branch를 잘못 고름

증상:

```text
selected output is not a tensor
```

또는 activation shape이 예상과 다름.

해결:

hook 대상 module의 forward output type과 shape을 먼저 확인한다.

```python
out = module(...)
print(type(out))
```

tuple/list/dict라면 `@0`, `@1`, `@key`를 사용한다.

### 10.3 학습 spec과 분석 spec을 혼동함

예:

```text
model.blocks.9
model.blocks.9::sae_layer#latent
```

둘은 다르다.

- `model.blocks.9`: 원래 model block output
- `model.blocks.9::sae_layer#latent`: 해당 block에 붙은 SAE latent feature activation

attribution에서 feature 단위를 보고 싶으면 보통 후자를 쓴다.

### 10.4 같은 layer라도 모델 pack마다 path가 다름

CLIP, DINO, SAM2, ResNet은 내부 module 구조가 다르다. 따라서 “block 9”라는 개념이 같아 보여도 spec 문자열은 모델마다 달라질 수 있다.

---

## 11. 발표 슬라이드 초안

### Slide 1. 제목

```text
SpecLens Quick Start
Layer Spec으로 SAE 학습부터 Feature 해석까지
```

핵심 문장:

```text
SpecLens는 모델 내부 activation을 spec 문자열로 지정하고,
그 activation에 SAE를 학습한 뒤 feature를 indexing / attribution으로 해석하는 도구다.
```

### Slide 2. 왜 필요한가?

- 모델마다 hook 위치와 output 구조가 다르다.
- SAE 학습을 위해 매번 custom adapter를 작성하면 확장성이 떨어진다.
- SpecLens는 문자열 spec으로 activation 위치를 공통 표현한다.

### Slide 3. 전체 워크플로우

```text
Config
→ Layer Spec
→ SAE Training
→ Feature Indexing
→ Attribution / ERF
→ Interpretation
```

### Slide 4. Layer Spec 개념

```text
model.blocks.9 = model 내부 activation 주소
```

비유:

```text
파일 경로가 파일 위치를 가리키듯,
layer spec은 activation 위치를 가리킨다.
```

### Slide 5. Spec 문법

| 예시 | 의미 |
|---|---|
| `model.blocks.9` | module forward output |
| `model.blocks.9@2` | output branch 2 |
| `enc.pos::method@0` | method output branch 0 |
| `model.blocks.9::sae_layer#latent` | SAE latent activation |

### Slide 6. Config에서 layer 지정

```yaml
sae:
  layers:
    - model.blocks.9
    - model.blocks.11
```

설명:

```text
여기에 적은 spec들이 SAE 입력 activation이 된다.
```

### Slide 7. 학습 실행

```bash
python scripts/train_sae_config.py --config configs/clip_imagenet_train.yaml
```

```bash
torchrun --nproc_per_node=4 scripts/train_sae_config.py \
  --config configs/clip_imagenet_train.yaml
```

### Slide 8. Feature Indexing

```text
각 feature가 어떤 sample에서 가장 강하게 켜지는지 top-N으로 저장한다.
```

보여줄 것:

- top activating images
- activation score
- firing frequency
- feature id

### Slide 9. Attribution / ERF

```text
layer spec + feature id = 분석 target
```

예:

```text
model.blocks.9::sae_layer#latent, unit 42
```

질문:

```text
feature 42를 켜는 이미지 영역은 어디인가?
```

### Slide 10. 팀 사용 규칙

- config 파일에 사용한 layer spec을 명확히 남긴다.
- 결과 디렉토리 이름에 model, dataset, SAE type, layer 범위를 포함한다.
- 새 모델은 먼저 `named_modules()`로 path를 확인한다.
- 분석 결과를 공유할 때는 반드시 `layer spec + feature id`를 함께 적는다.

---

## 12. 팀 내 권장 표기법

Feature를 공유할 때는 다음 형식을 권장한다.

```text
[model / dataset] layer_spec :: feature_id
```

예:

```text
CLIP / MIPAL-50k
model.blocks.9 :: feature 11481
```

Attribution target까지 명확히 쓸 때:

```text
target_spec = model.blocks.9::sae_layer#latent
target_unit = 11481
```

이렇게 적으면 학습 layer, SAE feature, attribution target을 혼동하지 않는다.

---

## 13. 한 줄 요약

SpecLens를 사용할 때 팀원이 반드시 기억해야 할 것은 하나다.

> Spec은 activation 주소다. 학습할 때도, indexing할 때도, attribution할 때도 같은 주소 체계를 쓴다.
