# Handoff — Model-Agnostic Necessity-ERF & the Routing Circularity

You are continuing interpretability research in `/home/sangyu/Desktop/Master/SpecLens`.
This prompt carries the conceptual state, the open problem we ended on, and the
inspiration. Read the memory index (`MEMORY.md`) and
`outputs/class_fri/research_frontier/MASTER_SUMMARY.md` first — they are the canonical record.

## Who you're working with
ML researcher (SAE / mechanistic interpretability, CLIP + SAM v2). Wants **terse,
honest** answers, no trailing summaries; Korean context is fine. He **catches
overclaims** — every claim must be backed by an actual measurement, and favorable
noise (small n) must be re-checked at larger n. He repeatedly course-corrects on
methodology; treat his methodological instincts as usually right (this session he
caught two real confounds — see below).

## Hard constraints (do not violate)
- Interpreter: `/home/sangyu/anaconda3/envs/py312/bin/python` (base conda is broken: torchvision::nms).
- GPU: use `cuda:0`. `cuda:1` sometimes holds the user's own process — leave it.
- **MASKING = MEAN-VECTOR baseline, ALWAYS.** Masking to zero/black inflates deletion via OOD
  (the model collapses on off-manifold inputs → fake "necessity"). Verify on-manifold by the
  **random-masking control**: random-order deletion must NOT kill the target (high random hdel);
  if it does, the baseline is OOD. For ViT the established mean is `global_mean_h_plus_pos`
  (mean patch EMBEDDING + position, in `erf_adapter.py`); at the input-pixel level mean = ImageNet
  mean colour (normalized 0). Decide masking level per-architecture by the random-collapse test.
- Long runs: launch in background and let the harness notify you. Do NOT spawn subagents that
  wait on long runs with Monitor/until-loops — they terminate mid-wait (happened repeatedly).

## The research arc (one paragraph)
**FRI (Feature-Recovery Insertion)** is our model-agnostic SUFFICIENCY tool (mask→output +
autograd, random-budget mask, ~32 steps, no 2D prior — RoPE/tabular-safe). The whole quest:
**find an equally model-agnostic NECESSITY method.** This session nailed the conceptual structure
of necessity vs sufficiency, refuted a tempting shortcut, found a working (but not fully general)
necessity method, and hit the **fundamental obstacle: the routing circularity.** That circularity
is now THE problem to solve.

## The guiding hypothesis (the optimistic frame — keep this front of mind)
**The model has ALREADY computed the necessary set; we only need to REVEAL it.** By the last hidden
layer the model has routed and compressed the inputs into its representation — so "which units/inputs
are necessary for this prediction" is **already ENCODED there.** We are NOT computing necessity from
scratch (that's why brute perturbation / gradient feel wrong); we are decoding what the model already
knows. **hidden-ablation is the first realization of this** — it reveals HIDDEN-space necessity (which
last-hidden positions the readout actually needs) cheaply and exactly, by reading the model's own
aggregated representation. This is the reason to believe a model-agnostic necessity EXISTS despite the
circularity: the answer is sitting in the last hidden state. The circularity is precisely the *remaining*
gap — revealing hidden-space necessity is solved, but DECODING it back to INPUT-space necessity is the
routing (≈identity for 1-to-1 ViT, recursive for CNN/latent). **The bet: find the right readout/decoder
of the last-hidden representation that exposes input-necessity directly, because the information is
already in there.**

## Established findings this session (solid — do not re-derive; cite if used)
1. **FRI = sufficiency** (insertion), model-agnostic. Confirmed across ViT/CNN/Perceiver/DETR
   (FRI wins insertion `hins` everywhere).
2. **The two faces (sufficiency ≠ necessity) are UNIVERSAL — caused by REDUNDANCY = AGGREGATION,
   NOT by attention.** We predicted a CNN (no attention) would COLLAPSE the two faces; it did NOT
   — ResNet shows them clearly (FRI insertion 0.92 excellent, same map's deletion 0.27 poor).
   CNN spatial-pooling and ViT attention both aggregate → both have the gap.
   (`test_twofaces_crossarch.py`, `twofaces_crossarch.json`.)
3. **necessity = the model's input-COMPRESSION / ROUTING structure** — and it is MODEL-SPECIFIC:
   input-aligned ViT → **self-patch** (diagonal dominance, grows with depth); register/CLS →
   attention-flow; latent-bottleneck (Perceiver/DETR/Q-Former/Mask2Former/SAM2) → **cross-attention**
   (NOT Perceiver-specific — all latent models); CNN → **receptive field**. **Sufficiency (FRI) is
   model-AGNOSTIC; necessity is model-SPECIFIC. The two faces are asymmetric in model-dependence.**
4. **"activation map = necessity" REFUTED** (it was a self-patch artifact). First pass looked like
   a clean double dissociation (activation-order deletion beats FRI), but the user's CONTROL
   (forbid deleting the target's own self-patch + neighbours) reversed it: activation hdel 0.033→0.61,
   loses to FRI/grad. Beyond the self-patch the firing map has ~no necessity signal.
   (`test_activation_necessity_deletion.py` with `--keep-self-ring`.)
5. **hidden-ablation necessity** (ablate the hidden position closest to the readout → mean →
   target-logit drop): a MODEL-AGNOSTIC necessity that **beats grad/inflow in ALL of
   resnet/clip/dinov2_reg/siglip**, and **beats input-occlusion for CNN** (where input occlusion
   fails because deep conv re-aggregates the redundancy). Input-occlusion (self-patch) still beats
   it for ViT, but occlusion COLLAPSES on CNN → hidden-ablation is the most ROBUST across
   architectures. Principle: **ablate the representation CLOSEST to the readout — that's where the
   model has aggregated away the redundancy.** Cheap for pooled readouts (GAP/CNN: linear projection,
   1 forward). (`test_hidden_ablation_necessity.py`, `hidden_ablation_necessity.json`.)
6. **Autolabeling insight (user, inspecting ERF panels):** FRI ERF = the CONTEXT that gives a
   feature its identity (whole Humvee → "frame" feature); the firing/activation map = the specific
   part. **ERF is essential for labeling** (loop_005: ERF reveals object identity; firing alone →
   labeler defaults to texture/position). Established ERF-autolabeling pipeline = `src/autolabel_eval/`
   (`cautious_feature_erf` = FRI 90%-recovery; `save_sae_fire_on_original` / `save_support_mask_image`
   / `save_erf_heatmap_image`). Redundant-feature panels: `outputs/class_fri/research_frontier/
   redundant_feature_erf_panels/index.html`.
7. **Redundancy emerges from aggregating substitutable disjoint-support features** (n=30/720):
   single SAE feature concentrated, aggregate redundant; mechanism = OR over disjoint (Jaccard 0.04),
   substitutable (negative super-additivity) supports. (`redundancy_emergence.json`.)

## THE OPEN PROBLEM — the routing circularity (where we stopped)
We want the **input** necessity-ERF for a target T. hidden-ablation gives necessity over a
hidden layer L's spatial grid. To turn that into INPUT necessity we **implicitly assumed each
hidden position i's ERF = its spatial upsampling** (hidden cell i ↔ input region i). **But that
assumption is itself an unsolved necessary-ERF** — *which inputs are necessary to build hidden_L[i]?*
— which is the SAME necessity problem one layer down. So:

> input necessity-ERF  ⟸  hidden-L necessity-ERF  ⟸  hidden-(L-1) necessity-ERF  ⟸ … **(circular / recursive)**

- For **1-to-1 ViT** (self-patch / diagonal dominance), hidden position i's necessary input ≈
  input patch i — the map is ≈ IDENTITY, so the recursion COLLAPSES and ViT escapes. This is *why*
  hidden-ablation looked model-agnostic — it secretly rode the ViT identity map.
- For **CNN** (overlapping receptive field, downsampling) and **latent-bottleneck** (cross-attn,
  no spatial layout), the hidden→input map is non-trivial → the recursion does NOT collapse →
  genuinely circular/expensive. **The CNN "1-to-1" we used (layer4 7×7 ↔ 7×7 input cells) was an
  unverified upsampling assumption, not a solved ERF.**
- **The circularity is the formal proof that "necessity = routing = model-specific."** Solving it
  (computing the input necessity-ERF without the recursion) = a TRULY model-agnostic necessity.

### Inspiration for breaking it (untested — for you to try)
- **Necessity-rollout**: compose per-layer hidden-ablation necessity from the readout back to the
  input (the necessity analogue of attention-rollout). Bounded cost (≈ L×N ablations), model-agnostic
  (any layered net). Expensive but finite — the recursion made concrete. Test whether composing 2–3
  layers already beats single-layer hidden-ablation on CNN, and whether it converges.
- **Fixed-point / closed-form**: attention and convolution are LINEAR given the weights; the
  necessity recursion over a (locally) linear routing may have a closed form (a necessity-flow matrix)
  computable in one pass — distinct from the gradient (which inverts; see Hidden Heroes).
- **Robust input set-perturbation**: solve necessity directly at the input by deleting SETS (the
  combinatorial necessity), handling redundancy the way FRI's random-budget handled sufficiency.
  Prior attempts (5 objectives, budget-Shapley) failed for necessity — but the framing "necessity =
  break the OR over substitutable disjoint supports" (finding #7) is a fresh angle.
- **Depth/resolution sweep** (designed, not yet run): which layer's ablation gives the best
  necessity? Deep layer = low-res, clean (redundancy aggregated, near readout); shallow = high-res
  but redundancy returns. Maps the tradeoff and pins "read necessity at the readout-closest aligned
  layer." Extend `test_hidden_ablation_necessity.py` to ablate at multiple depths.

## Key files
- Record: `outputs/class_fri/research_frontier/MASTER_SUMMARY.md`; memory `MEMORY.md` +
  `project_necessity_routing.md`, `project_erf_two_faces.md`, `project_module_necessity.md`,
  `project_necessity_erf_selfcontained.md`, `project_erf_labeling_findings.md`.
- This session's scripts (all reuse mean-baseline, `cuda:0`):
  `test_hidden_ablation_necessity.py` (the model-agnostic necessity candidate),
  `test_twofaces_crossarch.py` (two-faces vs architecture; token/pixel masking + random-collapse),
  `test_activation_necessity_deletion.py` (`--keep-self-ring` control),
  `build_redundant_feature_erf_panels.py` (ERF-autolabel panels),
  `research_redundancy_emergence.py`, `research_perceiver_necessity.py`, `research_detr_necessity.py`.
- Infra: `src/autolabel_eval/legacy.py` (`LegacyRuntime`: `cautious_feature_erf`=FRI,
  `make_masked_forward`, `input_x_grad_feature_erf`), `src/core/attribution/erf_adapter.py`
  (`make_masked_forward`, baseline `global_mean_h_plus_pos`). CLIP SAE:
  `outputs/spec_lens_store/clip_50k_sae` (batch-topk k=32, dict 12288, blocks 2/6/10), index
  `outputs/spec_lens_store/clip_50k_index`. ImageNet val: `/media/sangyu/Dataset/imagenet/val`.

## Suggested first move next session
Confirm the framing with the user, then run the **depth/resolution sweep** (cheap extension of
`test_hidden_ablation_necessity.py`) and prototype **necessity-rollout** on ResNet — these directly
attack the circularity on the architecture where it bites (CNN). Report honest numbers; expect the
single-layer hidden-ablation to be a ceiling that rollout must beat to justify its cost.

## Continuation update — hidden necessity as query, not map (2026-06-17)

The user corrected the depth-sweep framing: the point is not to choose which
hidden layer should be treated as a patch grid.  The original ERF lesson is
`patch != token`; hidden cells must not be upsampled as if they were input
patches.  Hidden necessity should be a **query/target**, and input necessity
should be decoded by asking which input patches build or preserve that necessary
hidden module.

Implemented:

```text
scripts/research_hidden_ablation_depth_sweep.py
scripts/research_hidden_necessity_input_decoder.py
```

Saved result:

```text
outputs/class_fri/research_frontier/HIDDEN_NECESSITY_INPUT_DECODER_RESULT_20260617.md
```

Depth sweep is diagnostic only:

```text
ResNet50 n=20:
layer2 28x28: hidden 0.283, grad 0.304, occlusion 0.351, random 0.320
layer3 14x14: hidden 0.532, grad 0.544, occlusion 0.517, random 0.686
layer4  7x7: hidden 0.429, grad 0.560, occlusion 0.556, random 0.726
```

Layer4 is the cleanest hidden-ablation result because it beats random robustly,
but this does not solve circularity.

Corrected positive experiment:

```text
ResNet50, layer4 hidden-necessity query, input grid 14x14, n=50:
decoder_repr      hdel 0.3353  hins 0.8774
decoder_contrib   hdel 0.3949  hins 0.8875
decoder_signed    hdel 0.3992  hins 0.8857
grad              hdel 0.5666  hins 0.6541
occlusion         hdel 0.4458  hins 0.8084
random            hdel 0.5989  hins 0.6091
```

Paired `decoder_repr` hdel:

```text
vs grad:      mean -0.2313, wins 46/50, p_less = 1.04e-11
vs occlusion: mean -0.1105, wins 28/50, p_less = 2.76e-4
vs random:    mean -0.2636, wins 48/50, p_less = 1.46e-15
```

Interpretation: first positive signal for the corrected philosophy:
**hidden necessity is not the map; hidden necessity is the query.**  Input patch
necessity can be decoded by mean-masking input patches and measuring damage to
the necessary hidden module representation.  The best current variant is
`decoder_repr`, which scores input patches by representation damage to the
selected necessary hidden module, not by spatially mapping hidden cells to
patches.  A later test of naive hidden-damage fingerprint clustering failed;
see the next update before pursuing any grouped-map variant.

## Continuation update — deletion-attribution map status (2026-06-17)

The "cluster input patches by hidden-module damage fingerprints" idea was
tested and failed in the naive KMeans form:

```text
ResNet50 n=50:
decoder_repr                hdel 0.3353
decoder_repr_group16_mean   hdel 0.4556
decoder_repr_group12_mean   hdel 0.4601
decoder_repr_group16_sum    hdel 0.4953
decoder_repr_group8_mean    hdel 0.5049
decoder_repr_group4_mean    hdel 0.5318
```

All grouping variants were significantly worse than single-patch
`decoder_repr`. Do not keep pursuing naive KMeans grouping as the next step.

Added mean-erasure-direction variants:

```text
decoder_erasedir = movement along full->mean hidden erasure direction
decoder_relnorm  = reduction in distance from hidden mean-vector state
```

They also lost:

```text
ResNet50 n=100, module_frac=0.60:
decoder_repr       hdel 0.3086  hins 0.8684
decoder_contrib    hdel 0.3525  hins 0.8858
decoder_erasedir   hdel 0.3633  hins 0.8509
decoder_signed     hdel 0.3684  hins 0.8864
decoder_relnorm    hdel 0.4158  hins 0.7951
occlusion          hdel 0.4376  hins 0.8100
grad               hdel 0.5376  hins 0.6792
random             hdel 0.5992  hins 0.6035
```

Paired `decoder_repr` hdel:

```text
vs grad:      mean -0.2290, wins 95/100, p_less = 1.17e-23
vs occlusion: mean -0.1290, wins 54/100, p_less = 3.10e-7
vs random:    mean -0.2905, wins 99/100, p_less = 4.85e-31
```

Random-robust subsets:

```text
random > 0.5:  n=69, decoder_repr 0.3916, occlusion 0.5852, grad 0.6422, random 0.7298
random > 0.7:  n=38, decoder_repr 0.5048, occlusion 0.7461, grad 0.7742, random 0.8314
random > 0.85: n=17, decoder_repr 0.6023, occlusion 0.7793, grad 0.8517, random 0.8895
```

module_frac plateau:

```text
n=50:
module_frac=0.40 -> decoder_repr hdel 0.3205
module_frac=0.60 -> decoder_repr hdel 0.3204
module_frac=0.80 -> decoder_repr hdel 0.3204
```

This plateau occurs because hidden-drop weights are clipped to nonnegative
values; after the positive hidden positions are included, extra hidden positions
receive near-zero weight.

Qualitative panels:

```text
outputs/class_fri/research_frontier/hidden_necessity_input_decoder_qual_mf40.png
outputs/class_fri/research_frontier/hidden_necessity_input_decoder_qual_mf60.png
```

Read: `decoder_repr` maps are usually broad and contiguous, not pointillist.
They often cover object body/part regions plus some context.  They are deletion
maps, not clean semantic segmentations.  Some hard failures remain, and at
least one selected qualitative case has occlusion beating the decoder.

Current best working claim:

```text
For ResNet50, a real input deletion-attribution map can be made by using
readout-near hidden necessity as the query and scoring each input patch by
single-patch mean-mask representation damage to that query.
```

This is still a perturbation decoder and costs one hidden capture per input
patch, but it does not use an explainer, InFlow prior, spatial hidden upsampling,
or a semantic prior.  Next real test: port the same query-decoder interface to
ViT and to a latent-bottleneck model.
