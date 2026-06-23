# New Session Prompt: Necessary-Set Deletion FRI

/goal

SpecLens repo에서 FRI 계열 attribution 연구를 계속한다. 목표는 **vision prior 없이**, **FRI spirit을 유지하면서**, **추가 cost를 대략 32 inference 수준으로 제한**하는 deletion-optimized attribution method를 찾는 것이다.

## Working Directory

Use:

```text
/home/sangyu/Desktop/Master/SpecLens
```

## Core Research Hypothesis

기존 FRI/random-budget 계열은 주로 **cooperation/support set**을 찾는 데 강하다. 그래서 insertion 성능은 좋지만 deletion 성능은 흔들린다.

Deletion은 cooperation set이 아니라 **necessary/disruptive set**을 찾아야 한다. 따라서 기존 insertion-oriented objective를 조금 고치는 수준이 아니라, necessary-set 전용 objective 또는 readout이 필요하다.

중요:

- 각 patch를 하나씩 지우는 single deletion oracle 같은 원시적 방식은 deletion 최적화에 맞지 않는다.
- banana/lemur/nematode에서는 single deletion oracle도 HardDel이 나쁘다.
- random budget처럼 창의적인 coalition/group-level necessary-set objective가 필요하다.
- spatial radius/blob/2D neighborhood 같은 vision prior는 쓰면 안 된다.
- text/token/feature에도 일반화 가능한 방식이어야 한다.

## FRI Spirit Constraints

Method must satisfy:

- training-free
- no external segmentation
- no vision-specific prior
- no patch radius, no 2D blob, no spatial neighborhood assumption
- model-agnostic or minimally model-specific
- differentiable mask optimization and/or forward-probe coalition objective is allowed
- random budget, stochastic coalition, threshold/barrier, contrastive objective, interaction readout are allowed

## Cost Constraint

Final candidate should target roughly:

```text
~32 extra forward inference calls per image
```

Exploration may use 64 calls, but the method must have a plausible 32-call version.

Backward-heavy optimization should stay near the existing FRI 32 to 64 step envelope.

## Existing Files To Read First

Core implementation:

```text
src/core/attribution/fri/solver.py
src/core/attribution/fri/local_probe.py
scripts/run_class_fri_patch_benchmark.py
```

Important result notes:

```text
outputs/class_fri/research_frontier/SPARSE_CONTENT_LOCAL_PROBE_RESULT.md
outputs/class_fri/research_frontier/PART_NECESSITY_ANALYSIS.md
outputs/class_fri/research_frontier/STEP_SIGNAL_AND_COALITION_NOTES.md
outputs/class_fri/research_frontier/DELETION_ONLY_MASKOPT_NOTES.md
```

Important n=100 benchmark files:

```text
outputs/class_fri/softplus_failure_n100_merged_scores.json
outputs/class_fri/softplus_failure_scan_n100.csv
outputs/class_fri/softplus_failure_notes_n100.md
```

## Existing Baselines

Current InFlow baseline on the n=100 hard/failure set:

```text
InFlow n=100:
HardDel   0.3170
HardIns   0.8994
MASIns    0.7111
StochDel  0.5613
```

Existing FRI readouts have generally improved insertion/MAS but not enough deletion. Example:

```text
prob_softplus_x_del n=100:
HardDel   0.3458
HardIns   0.9595
MASIns    0.7507
StochDel  0.5619
```

## Numerical Research Target

Primary benchmark:

Use the n=100 hard/failure set from:

```text
outputs/class_fri/softplus_failure_n100_merged_scores.json
```

or the corresponding case list in:

```text
outputs/class_fri/softplus_failure_scan_n100.csv
```

Primary metric:

```text
HardDel AUC lower is better.
```

Main goal:

```text
Beat InFlow HardDel on the n=100 hard/failure set.
```

Minimum success:

```text
HardDel mean <= 0.3170
wins_del_vs_inflow >= 50/100
HardIns mean >= 0.85
MASIns mean >= 0.70
```

Good success:

```text
HardDel mean <= 0.305
wins_del_vs_inflow >= 55/100
wins_both_vs_inflow >= 35/100
HardIns mean >= 0.87
MASIns mean >= 0.72
StochDel mean <= 0.561
```

Strong success:

```text
HardDel mean <= 0.295
wins_del_vs_inflow >= 60/100
wins_both_vs_inflow >= 40/100
HardIns mean >= 0.88
MASIns mean >= 0.74
StochDel mean <= 0.54
```

Do not optimize only the mean. Always report:

```text
mean HardDel
median HardDel
wins_del_vs_inflow
wins_both_vs_inflow
HardIns mean
MASIns mean
StochDel mean
worst regressions vs InFlow
subgroup behavior on redundancy / thin-core / competitor cases
```

## Fast Diagnostic Cases

Before running n=100, use these 8 cases as a fast failure-mode dashboard:

```text
1. multi_object_zebra_elephant.jpg
   target 386 African elephant

2. /media/sangyu/Dataset/imagenet/val/n07753592/ILSVRC2012_val_00032327.JPEG
   target 954 banana

3. /media/sangyu/Dataset/imagenet/val/n02497673/ILSVRC2012_val_00011144.JPEG
   target 383 Madagascar cat/lemur

4. /media/sangyu/Dataset/imagenet/val/n01631663/ILSVRC2012_val_00028601.JPEG
   target 27 eft

5. /media/sangyu/Dataset/imagenet/val/n02699494/ILSVRC2012_val_00021990.JPEG
   target 406 altar

6. /media/sangyu/Dataset/imagenet/val/n03223299/ILSVRC2012_val_00030383.JPEG
   target 539 doormat

7. /media/sangyu/Dataset/imagenet/val/n01930112/ILSV2012_val_00029702.JPEG
   target 111 nematode

8. /media/sangyu/Dataset/imagenet/val/n03534580/ILSVRC2012_val_00021106.JPEG
   target 601 hoopskirt
```

If path 7 has a typo, use the verified path:

```text
/media/sangyu/Dataset/imagenet/val/n01930112/ILSVRC2012_val_00029702.JPEG
```

Diagnostic interpretation:

- elephant/zebra: competitor/background suppression
- banana: redundancy with apples/background
- Madagascar cat/lemur: target/context drift
- doormat: anti-evidence
- nematode: thin-core
- hoopskirt: thin-core/context

Do not overfit to these 8 cases. They are a smoke test and qualitative dashboard, not the final benchmark.

## Known Negative Results

Avoid repeating these unless you have a new reason:

- single-patch deletion is not enough
- target-prob deletion alone is not enough
- insertion/cooperation optimization does not imply deletion quality
- spatial radius/local blob probe works qualitatively but violates no-vision-prior constraint
- naive random coalition with positive ridge helped elephant/doormat but failed banana/lemur/nematode
- independent deletion-only soft mask helped elephant/doormat/altar but failed banana/lemur/nematode/hoopskirt
- step-derived target-only signals help some cases but fail redundancy/saturation cases

## Key Observations So Far

Good cases for necessary/deletion objective:

```text
elephant/zebra
doormat
altar
some win cases such as eft
```

Hard cases:

```text
banana
Madagascar cat/lemur
nematode
hoopskirt
```

Diagnosis:

- competitor and anti-evidence cases expose useful deletion/contrast signal
- redundancy cases do not expose necessity through single patch or simple target-prob drop
- thin-core cases need an order/coverage notion, but not a vision-specific 2D prior

## Promising Directions

Explore one or more of these. Prefer small decisive experiments.

### 1. Random-budget necessary FRI

Design a necessary-set counterpart to cooperation FRI.

Instead of:

```text
find small inserted support that recovers target
```

try:

```text
find small deleted support that collapses target margin
```

But avoid independent adversarial mask artifacts. Consider stochastic budgets, rank-based deletion distributions, barrier objectives, or survival readouts.

### 2. Contrastive necessary objective

Target probability alone is saturated in banana/lemur/nematode.

Use contrast objectives derived from logits:

```text
target vs top competitor
target vs topK non-target classes
target-specific drop minus context/sibling drop
```

Avoid manually injecting ImageNet semantic labels if possible. Derive contrast set from logits/topK.

### 3. Redundancy-aware coalition objective

Banana/lemur require structured coalition removal. Single patch deletion cannot reveal necessity.

Possible probes:

```text
random sparse coalitions over top FRI candidates
gradient-similarity coalitions
representation-similarity coalitions
optimization co-survival coalitions
topK-logit contrast coalitions
```

No spatial grouping.

### 4. Interaction-aware readout

Linear positive ridge over coalition drops was not enough. Consider:

```text
pair/group interaction score
submodular-style marginal under random context
Shapley-inspired low-sample estimator
rank aggregation over multiple conditional contexts
```

Cost must remain near 32 calls.

### 5. Non-vision continuity

For nematode/hoopskirt, deletion quality needs coherent coverage/order. Do not use 2D continuity.

Possible modality-general replacements:

```text
representation similarity
gradient alignment
co-survival during optimization
coalition co-response
attention-free token interaction if available from model internals
```

## Suggested Workflow

1. Read the notes and current code.
2. Start with the 8 diagnostic cases.
3. Implement candidate in a research script first.
4. Compare against:
   - InFlow
   - existing FRI readouts
   - optionally `fri_top1_local_probe` only as an upper qualitative reference, not as valid final method because it uses radius.
5. If promising, run n=100.
6. Save metrics JSON under:

```text
outputs/class_fri/research_frontier/
```

7. Generate qualitative plots for promising candidates.
8. Write a concise notes file explaining method, cost, metrics, qualitative behavior, and failure modes.

## Required Reporting

For every serious candidate report:

```text
method name
extra inference cost
whether it uses any vision prior
HardDel mean/median
HardIns mean
MASIns mean
StochDel mean
wins_del_vs_inflow
wins_both_vs_inflow
worst 10 regressions vs InFlow
behavior on the 8 diagnostic cases
```

## Success Criteria

The target is not just a nicer qualitative heatmap.

A candidate is meaningful only if it moves toward:

```text
n=100 HardDel <= 0.3170
```

without collapsing:

```text
HardIns >= 0.85
MASIns >= 0.70
```

The strongest target is:

```text
n=100 HardDel <= 0.295
wins_del_vs_inflow >= 60/100
HardIns >= 0.88
MASIns >= 0.74
StochDel <= 0.54
```

## Autonomy

Work autonomously. Do not stop at a proposal.

Implement, run, inspect, and iterate. If a direction fails, record the negative result and pivot.

Keep edits scoped. Use research scripts first before modifying core solver.
