# Banana Scorebank Mining Result - 2026-06-15

## Question

Does the existing research frontier already contain a banana ordering that
avoids the current block0-only OOD collapse?

Current banana failure:

```text
current block64 ~= 1.00
current blur64  ~= 0.24
current drop32  ~= 0
```

## Mining

Scripts added:
- `scripts/mine_banana_scorebank_candidates.py`
- `scripts/audit_banana_scorebank_transfer_all.py`
- `scripts/compose_prefix_shape_candidate_gate.py`

Artifacts:
- `outputs/class_fri/research_frontier/banana_scorebank_mining_20260615.json`
- `outputs/class_fri/research_frontier/banana_scorebank_mining_top80_20260615.scores.npz`
- `outputs/class_fri/research_frontier/banana_scorebank_transfer_all_20260615.json`
- `outputs/class_fri/research_frontier/banana_scorebank_transfer_all_top_20260615.scores.npz`
- `outputs/class_fri/research_frontier/banana_transfer_top10_metrics_20260615.json`
- `outputs/class_fri/research_frontier/qual_banana_transfer_mined_current_top001_top007_20260615.png`

Mined records:

```text
banana score records: 4105
unique banana rank orders: 1063
```

Full k32/k64 transfer audit over all unique orders found exactly one strong
banana transfer candidate:

```text
source:
conditional_random_group_probe_n100_targetseed_pool196_p32p48_s16_g32_keystable.scores.npz
suffix:
condrand_p32_s16_mix25

banana block32 = -0.008
banana block64 =  0.996
banana blur64  =  0.627
blur/block64   =  0.630
```

This is still late-collapse by response shape, but unlike current, the k64
deletion transfers to image-space blur much better.

## Banana exact metric

Corrected paper metrics on banana:

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | banana blur64 |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.2309 | 0.8987 | 0.2503 | 0.9817 | 0.8284 | 0.24 |
| mined transfer top001 / `condrand_p32_s16_mix25` | 0.2574 | 0.8981 | 0.2794 | 0.9815 | 0.6658 | 0.63 |
| metric-best mined top007 | 0.1653 | 0.8616 | 0.1800 | 0.9382 | 0.8241 | 0.24 |

Read:
- The OOD-improved candidate sacrifices HDel and MAS.
- The metric-best mined candidate keeps the old OOD failure.
- Metric and OOD are genuinely pulling apart on banana.

## n100 metrics for the random-group candidate

Full replacement with `condrand_p32_s16_mix25`:

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | raw wins |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.167534 | 0.743911 | 0.202403 | 0.901212 | 0.785240 | 87/73/64 |
| `condrand_p32_s16_mix25` | 0.194341 | 0.742241 | 0.233615 | 0.899100 | 0.649052 | 74/70/52 |

Full replacement is not viable.

## Prefix-shape gated fallback

Gate:

```text
late_drop64 >= 0.90
drop32 / drop64 <= 0.10
```

Selected 4/100 cases:

```text
ILSVRC2012_val_00003508
ILSVRC2012_val_00017714
ILSVRC2012_val_00032327  # banana
ILSVRC2012_val_00047400
```

Candidate fallback:

```text
condrand_p32_s16_mix25
```

Artifacts:
- `outputs/class_fri/research_frontier/prefixshape_condrand_p32s16m25_20260615.scores.npz`
- `outputs/class_fri/research_frontier/prefixshape_condrand_p32s16m25_metrics_20260615.json`
- `outputs/class_fri/research_frontier/ood_transfer_prefixshape_condrand_p32s16m25_hard7_20260615.json`

Corrected n100:

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | raw wins |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.167534 | 0.743911 | 0.202403 | 0.901212 | 0.785240 | 87/73/64 |
| prefixshape + condrand fallback | 0.170344 | 0.744405 | 0.205536 | 0.901598 | 0.779743 | 86/73/64 |

Hard7 OOD:

| method | block64 drop | blur64 drop | valid blur/block | suspicious flags |
|---|---:|---:|---:|---:|
| current-like `necset_v2` | 0.824 | 0.654 | 0.822 | 1/7 |
| prefixshape + condrand fallback | 0.840 | 0.730 | 0.892 | 0/7 |

This is the first candidate that actually removes the banana hard7 OOD flag
without deleting the whole method numerically. It is still not a final endpoint
because HDel and MAS regress.

## Metric frontier plus OOD fallback

Base:

```text
l6sing_dense
```

Fallback:

```text
same prefix-shape gate + condrand_p32_s16_mix25
```

Artifacts:
- `outputs/class_fri/research_frontier/l6dense_prefixshape_condrand_p32s16m25_20260615.scores.npz`
- `outputs/class_fri/research_frontier/l6dense_prefixshape_condrand_p32s16m25_metrics_20260615.json`
- `outputs/class_fri/research_frontier/ood_transfer_l6dense_prefixshape_condrand_p32s16m25_hard7_20260615.json`
- `outputs/class_fri/research_frontier/common_seed_stochastic_l6dense_prefixshape_condrand_p32s16m25_20260615.json`

Corrected n100:

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | raw wins |
|---|---:|---:|---:|---:|---:|---:|
| current | 0.167534 | 0.743911 | 0.202403 | 0.901212 | 0.785240 | 87/73/64 |
| `l6sing_dense` | 0.165083 | 0.746053 | 0.199721 | 0.903217 | 0.786452 | 89/75/66 |
| `l6dense_prefixshape_condrand_p32s16m25` | 0.167894 | 0.746547 | 0.202854 | 0.903603 | 0.780955 | 88/75/66 |

Hard7 OOD:

| method | block64 drop | blur64 drop | valid blur/block | suspicious flags |
|---|---:|---:|---:|---:|
| current-like `necset_v2` | 0.824 | 0.654 | 0.822 | 1/7 |
| `l6dense_prefixshape_condrand_p32s16m25` | 0.887 | 0.743 | 0.779 | 0/7 |

Common-seed stochastic:

| method | SDel | SIns | SInsDelta |
|---|---:|---:|---:|
| current | 0.317548 | 0.850042 | 0.091173 |
| `l6sing_dense` | 0.317335 | 0.850323 | 0.091454 |
| `l6dense_prefixshape_condrand_p32s16m25` | 0.327531 | 0.849664 | 0.090796 |

Read:
- OOD improves.
- Raw/clip HIns and wins stay good.
- HDel falls back close to current.
- MAS and stochastic deletion regress.

## Conclusion

This is the strongest OOD/qualitative tradeoff found so far, but not the final
method.

Useful insight:

```text
random conditional group fallback can create a banana support that transfers to
image-space deletion better than current, while still using only model-internal
response probes as the method signal.
```

Remaining problem:

```text
the fallback broadens support and hurts MAS/stochastic deletion.
```

Next direction:
- compress this into a cheap candidate generator rather than mining 1000+
  historical scorebanks;
- use prefix-shape late-collapse as a detector;
- design a random-group fallback that preserves current mass/order quality,
  possibly by only replacing ranks 33..64 with group-supported candidates or by
  rank-calibrating candidate mass more carefully.
