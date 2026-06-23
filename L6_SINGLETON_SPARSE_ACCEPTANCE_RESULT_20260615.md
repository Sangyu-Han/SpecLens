# L6 Singleton Sparse Acceptance Result - 2026-06-15

## Corrected paper metrics

Metric source:
- `scripts/eval_necset_paper_metrics.py`
- Hard main convention: `raw`
- Extra hard conventions: `clip`, `ours`
- `raw` and `clip` hard insertion are bounded; `ours` is the legacy `p / p_full` convention and is diagnostic only.

Primary metric artifact:
- `outputs/class_fri/research_frontier/l6sing_sparse_accept_denseguard_metrics_20260615.json`

Common-seed stochastic artifact:
- `outputs/class_fri/research_frontier/common_seed_stochastic_l6sing_sparse_accept_denseguard_20260615.json`

## n100 summary

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | raw wins del/ins/both | clip wins del/ins/both |
|---|---:|---:|---:|---:|---:|---:|---:|
| inflow | 0.256496 | 0.695853 | 0.310420 | 0.861576 | 0.711115 | - | - |
| current (`loose_auc04_magree15`) | 0.167534 | 0.743911 | 0.202403 | 0.901212 | 0.785240 | 87/73/64 | 87/64/56 |
| `loose_rg_alpha3_magree15` | 0.166630 | 0.743908 | 0.201389 | 0.901279 | 0.785403 | 87/73/64 | 87/64/56 |
| `l6sing_dense` | 0.165083 | 0.746053 | 0.199721 | 0.903217 | 0.786452 | 89/75/66 | 89/65/57 |

Old FRI reference under the same corrected hard convention:

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | raw wins del/ins/both |
|---|---:|---:|---:|---:|---:|---:|
| `oldfri_l1` | 0.287718 | 0.728720 | 0.352411 | 0.894729 | 0.749412 | 40/70/34 |
| `oldfri_soft` | 0.281283 | 0.723603 | 0.341297 | 0.889158 | 0.750692 | 44/64/33 |

## Over-one audit

Hard insertion AUC values above 1:

| method | raw | clip | legacy ours |
|---|---:|---:|---:|
| inflow | 0 | 0 | 10 |
| current | 0 | 0 | 24 |
| `l6sing_dense` | 0 | 0 | 24 |
| `oldfri_l1` | 0 | 0 | 17 |
| `oldfri_soft` | 0 | 0 | 15 |

Conclusion: the corrected paper metric path fixes the hard insertion >1 issue for `raw` and `clip`. Legacy `ours` can still exceed 1 and should not be used as the headline metric.

## Common-seed stochastic audit

Compared against current with identical random streams:

| method | SDel | SIns | SInsDelta |
|---|---:|---:|---:|
| current | 0.317548 | 0.850042 | 0.091173 |
| `l6sing_dense` | 0.317335 | 0.850323 | 0.091454 |

Pairwise delta (`l6sing_dense - current`):
- dSDel = -0.000212, lower deletion wins 5/100.
- dSIns = +0.000281, higher insertion wins 7/100.
- both = 5/100.

Stochastic insertion is a different estimator and can exceed 1; the corrected hard insertion over-one audit should be read from `hard_ins_auc` and `hard_ins_auc_clip`.

## Interpretation

`l6sing_dense` is the current corrected metric frontier on n100:
- It improves HDel raw from 0.167534 to 0.165083.
- It improves HIns raw from 0.743911 to 0.746053.
- It improves MAS ins from 0.785240 to 0.786452.
- It improves raw wins vs InFlow from 87/73/64 to 89/75/66.

But it is not a final method:
- It uses expensive candidate generation plus sparse acceptance probes.
- It does not address the banana/pointillism OOD failure; selected cases do not include the known qualitative failures.
- Treat it as a selector upper-bound/frontier clue, not the clean 32-forward endpoint.
