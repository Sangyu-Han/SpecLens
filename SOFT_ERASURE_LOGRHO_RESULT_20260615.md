# Soft-Erasure Log-Retention Result - 2026-06-15

## Question

Can we reduce pointillism/OOD by not erasing patch vectors all the way to the
baseline, but instead optimizing deletion over logarithmic retention levels?

Constraints kept:
- no InFlow prior
- no explainer
- no segmentation
- no image-space blur/mean in the method
- no spatial radius/blob prior

Implementation touched:
- `scripts/research_soft_erasure_fri.py`

Artifacts:
- `outputs/class_fri/research_frontier/qual_soft_erasure_current_logrho_hard8_20260615.png`
- `outputs/class_fri/research_frontier/qual_soft_erasure_current_logrho_hard8_20260615.json`
- `outputs/class_fri/research_frontier/qual_soft_erasure_current_logrho_hard8_20260615.scores.npz`
- `outputs/class_fri/research_frontier/ood_transfer_soft_erasure_current_logrho_hard7_20260615.json`
- `outputs/class_fri/research_frontier/blockstate_ood_proxy_soft_erasure_current_logrho_hard7_20260615.json`
- `outputs/class_fri/research_frontier/BLOCKSTATE_OOD_PROXY_SOFT_ERASURE_CURRENT_LOGRHO_HARD7_20260615.md`
- `outputs/class_fri/research_frontier/qual_soft_erasure_current_early32_hard8_20260615.png`
- `outputs/class_fri/research_frontier/qual_soft_erasure_current_early32_hard8_20260615.json`
- `outputs/class_fri/research_frontier/ood_transfer_soft_erasure_current_early32_hard7_20260615.json`

## Log-retention soft erasure

Run:

```text
python scripts/research_soft_erasure_fri.py \
  --device cuda:1 \
  --seed 42 \
  --steps 35 \
  --ks 8,16,32,48,64,96 \
  --retentions 0,0.01,0.03,0.10,0.30 \
  --loss-mode cvar \
  --insertion-weight 0.12 \
  --gap-weight 0.80 \
  --graph-smooth-weight 0.10 \
  --graph-cut-weight 0.05 \
  --init-scores outputs/class_fri/research_frontier/loose_auc04_magree15.scores.npz:loose_auc04_magree15 \
  --init-name current_loose_auc04_magree15 \
  --out outputs/class_fri/research_frontier/qual_soft_erasure_current_logrho_hard8_20260615.png
```

Hard8 result:

| case | current/init HDel | soft-erasure HDel | delta |
|---|---:|---:|---:|
| zebra/eleph | 0.192 | 0.146 | -0.046 |
| banana | 0.253 | 0.336 | +0.083 |
| lemur | 0.143 | 0.225 | +0.081 |
| eft | 0.368 | 0.474 | +0.106 |
| altar | 0.167 | 0.183 | +0.016 |
| doormat | 0.106 | 0.134 | +0.028 |
| nematode | 0.085 | 0.270 | +0.185 |
| hoopskirt | 0.125 | 0.174 | +0.049 |

Mean HDel moved from 0.180 to 0.243 on this dashboard. Only zebra/elephant
improved.

OOD transfer on the 7 ImageNet dashboard cases:

| method | block64 drop | blur64 drop | blur/block valid | suspicious flags |
|---|---:|---:|---:|---:|
| current/init | 0.840 | 0.675 | 0.827 | 1/7 |
| soft-erasure | 0.645 | 0.438 | 0.680 | 2/7 |

Read: soft-erasure reduced banana block0 collapse, but mostly by losing deletion
power. It also created/worsened suspicious transfer failures on doormat and
nematode.

## Early32 variant

Run:

```text
python scripts/research_soft_erasure_fri.py \
  --device cuda:1 \
  --seed 43 \
  --steps 35 \
  --ks 8,16,24,32 \
  --retentions 0,0.03,0.10,0.30 \
  --loss-mode cvar \
  --insertion-weight 0.10 \
  --gap-weight 0.90 \
  --graph-smooth-weight 0.08 \
  --graph-cut-weight 0.04 \
  --init-scores outputs/class_fri/research_frontier/loose_auc04_magree15.scores.npz:loose_auc04_magree15 \
  --init-name current_loose_auc04_magree15 \
  --out outputs/class_fri/research_frontier/qual_soft_erasure_current_early32_hard8_20260615.png
```

Hard8 mean HDel moved from 0.195 to 0.398. Banana failed badly:

```text
banana current/init HDel = 0.253
banana early32 soft-erasure HDel = 0.937
```

OOD transfer:

| method | block64 drop | blur64 drop | blur/block valid | suspicious flags |
|---|---:|---:|---:|---:|
| current/init | 0.840 | 0.675 | 0.827 | 1/7 |
| early32 soft-erasure | 0.532 | 0.366 | 0.949 | 2/7 |

Read: removing k64 from the objective avoids some late-collapse behavior, but
it mostly destroys deletion ability. It is not a viable replacement.

## Important diagnosis

For current on banana:

```text
drop32 ~= 0
drop64 ~= 1
blur64 ~= 0.24
```

The collapse is robust even when deleted patches retain partial block0
information:

```text
retention 0.00 drop64 = 0.997
retention 0.10 drop64 = 0.999
retention 0.30 drop64 = 0.997
```

So the banana failure is not simply "rho=0 deletes too much information."  It is
more specifically a late-prefix block0-state collapse that does not transfer to
image-space blur.

## Block-state OOD proxy

Using blur only as an audit label, non-image block-state features separated
suspicious records:

```text
flag = block_drop64 > 0.5 and blur_drop64 / block_drop64 < 0.35
```

Best single-threshold rule on the soft-erasure/current hard7 audit:

```text
block_drop32 <= 0.009
precision = 0.833
recall = 0.714
F1 = 0.769
```

This is modality-general because it uses only response shape under the same
representation-level deletion operator. It is a detector, not a generator.

## Conclusion

Soft/log-retention erasure is not the missing generator. It can make maps look
less pointillist, but it loses too much deletion quality and can introduce new
OOD transfer failures.

The useful next direction is not "softer deletion" alone. It is a cheap
block-response-shape guard or objective that prevents late-only collapse:

```text
good necessary set should produce meaningful early drop, not only k64 collapse
```

But a hard early32 objective is too strict. The next candidate should keep the
current ranker and use the early/late response shape as a selector, penalty, or
tail demotion signal rather than replacing the ranker.
