# Prefix-Shape Tail Gate Result - 2026-06-15

## Motivation

Soft/log-retention erasure showed that banana is not failing simply because
`rho=0` deletes too much patch-vector information. Current banana has a
late-only block0 collapse:

```text
drop32 ~= 0
drop64 ~= 1
blur64 ~= 0.24
```

So the next test was a cheap response-shape guard: detect rankings whose
deletion effect appears only late, then demote part of the rank-33..64 tail.

Constraints kept:
- no InFlow prior
- no explainer
- no segmentation
- no image-space perturbation in the method
- no spatial neighborhood/radius/blob prior
- no semantic labels

Implementation:
- `scripts/compose_prefix_shape_tail_gate.py`

Inputs:
- base scores: `outputs/class_fri/research_frontier/loose_auc04_magree15.scores.npz`
- sparse probe source: `outputs/class_fri/research_frontier/l6sing_sparse_accept_denseguard_20260615.json`

## Gate

Strict late-collapse gate:

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

Generated variants:

```text
tail33_64_after80
tail33_64_after96
tail33_64_after128
tail49_64_after80
```

Artifacts:
- `outputs/class_fri/research_frontier/prefix_tailgate_d64m90_r10_20260615.scores.npz`
- `outputs/class_fri/research_frontier/prefix_tailgate_d64m90_r10_20260615.json`
- `outputs/class_fri/research_frontier/prefix_tailgate_d64m90_r10_selected4_metrics_20260615.json`
- `outputs/class_fri/research_frontier/ood_transfer_prefix_tailgate_d64m90_r10_hard7_20260615.json`

## Corrected selected4 metrics

The selected4 metric test uses corrected raw/clip hard AUC from
`scripts/eval_necset_paper_metrics.py`.

| method | HDel raw | HIns raw | HDel clip | HIns clip | MAS ins | raw wins vs InFlow |
|---|---:|---:|---:|---:|---:|---:|
| InFlow | 0.3462 | 0.8618 | 0.3812 | 0.9467 | 0.7760 | - |
| current | 0.2292 | 0.8693 | 0.2514 | 0.9562 | 0.8028 | 4/3/3 |
| tail49 | 0.2566 | 0.8704 | 0.2819 | 0.9564 | 0.8031 | 4/3/3 |
| tail80 | 0.3038 | 0.8705 | 0.3340 | 0.9560 | 0.8031 | 3/3/2 |
| tail96 | 0.3734 | 0.8700 | 0.4109 | 0.9552 | 0.8022 | 2/3/1 |
| tail128 | 0.5155 | 0.8688 | 0.5681 | 0.9538 | 0.8003 | 0/3/0 |

Read: every tail demotion variant worsens deletion on the exact cases it
touches.

## OOD transfer on hard7

| method | block64 drop | blur64 drop | blur/block valid | suspicious flags |
|---|---:|---:|---:|---:|
| current-like `necset_v2` | 0.824 | 0.654 | 0.822 | 1/7 |
| tail49 | 0.806 | 0.635 | 0.778 | 1/7 |
| tail80 | 0.806 | 0.635 | 0.778 | 1/7 |
| tail96 | 0.697 | 0.639 | 0.944 | 0/7 |
| tail128 | 0.697 | 0.639 | 0.944 | 0/7 |

Banana behavior:

```text
current/init block64=1.00, blur64=0.24
tail49/tail80 block64=0.76, blur64=-0.04
tail96/tail128 block64~=0.00, blur64~=0.00
```

Read: stronger demotion removes the OOD flag only by eliminating the deletion
effect; weaker demotion preserves more deletion but does not solve the banana
transfer failure.

## Conclusion

The response-shape signal is useful as a detector:

```text
late-only collapse is real and cheaply detectable.
```

But direct tail demotion is not a good generator:

```text
OOD flag removed => deletion power collapses
deletion partly preserved => OOD flag remains
```

Next direction: use `drop32/drop64` as a selection or regularization feature
during candidate generation, not as a posthoc band-demotion operation. The
candidate must find a different early-support ordering rather than merely
moving the late tail down.
