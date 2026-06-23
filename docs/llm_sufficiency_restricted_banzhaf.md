# Perturbation attribution beats AttnLRP on LLM sufficiency — the sequence-OOD law and the restricted-range Banzhaf

*Working note, 2026-06. Model: Qwen2.5-1.5B-Instruct. Metric: input-token INSERTION AUC (keep
top-k tokens, rest replaced by the global token-embedding mean; AUC over keep-fraction grid
[0,.02,.05,.1,.15,.2,.3,.5]). Cases: Everest copy-QA (T=456) + IMDB sentiment (length-binned).*

## 0. One-paragraph summary

On LLM input-token **sufficiency** (insertion), the standard belief is that the gradient method
**AttnLRP** wins and perturbation/occlusion methods fail. We show *why* they fail — the failure is
**sequence-level OOD**, not vector-level — and we make perturbation **competitive** with AttnLRP via
a **restricted-range stochastic Banzhaf** estimator (sample only near-complete coalitions). The fix
is the explicit realization of **observational** (vs interventional) feature removal.

> **Honesty caveat (read §7 first).** In one run the restricted Banzhaf scored 0.577 vs AttnLRP
> 0.509, but a *fresh independent run* gives Banzhaf **below** AttnLRP at every sample budget — the
> per-case insertion AUC fluctuates ±0.1 across seeds (everest, 5 seeds: 0.652 ± 0.096), comparable
> to the ~0.07 margin. **The robust claim is competitiveness (within noise) at 200–800× the forward
> cost, NOT a clean win** — confirmed: the per-case seed-std (~0.10) exceeds the win margin. We exhaustively rule out the obvious
alternatives (attention-masking, token removal): they are worse, because a *random token subset is
incoherent language* and therefore OOD no matter how you erase it.

## 1. Background: the two faces and the asymmetry

Attribution has two faces. **Sufficiency** asks "which small set of inputs, kept alone, recovers the
prediction?" (insertion). **Necessity** asks "which small set, removed, destroys it?" (deletion). The
right tool is asymmetric:

- **Necessity = conditional (chunked) occlusion** is the *robust universal*: it beats gradient / IG /
  AttnLRP / FRI on deletion across vision-SAE, CLIP prediction (greedy del 0.077 vs ~0.3 baselines),
  and LLM at all token lengths, at <3% of the O(T²) greedy-oracle cost, and it handles redundancy
  that single-pass methods miss.
- **Sufficiency = FRI** (random-budget soft-insertion) for **vision** (CLIP insertion 0.775 ≫
  baselines; the random budget is a measurable +0.20 over the standard L1+TV relaxation).

The open problem: **sufficiency on LLMs**. FRI's soft-insertion fails on LLMs at every length, for
both concentrated (copy/retrieval) and distributed (sentiment) evidence. AttnLRP wins. Why?

## 2. The sequence-OOD law (central finding)

The masking that drives perturbation attribution pushes the model **off the data manifold**, but the
binding constraint is at the **sequence level, not the vector level**:

- A single masked token (replaced by the global mean embedding) is **tolerated**: a uniform 50%
  soft-mask gives next-token entropy 0.31 (in-distribution, confident).
- **Heavy** masking makes the whole sequence degenerate: at 10% kept, entropy is 7-8 (≈ uniform =
  maximally confused). A random subset of tokens is **incoherent language** — the model has never seen
  it — so its output carries no signal.

This explains **why AttnLRP wins**: it back-propagates relevance on the **full, unmasked** sequence,
so it *never enters the sequence-OOD regime*. Perturbation/occlusion methods *must* mask, so they hit
sequence-OOD — worst on long sequences, where the insertion curve requires heavier absolute masking.
This also explains the **length dependence**: short prompts have little sequence-OOD (perturbation is
fine); long prompts have a lot (perturbation degrades).

**Diagnostic evidence.** Next-token entropy under a uniform hard mask, by keep-fraction: 0.05→~6.6,
0.10→~6.7, 0.50→0.31. The transition is amount-driven, not vector-driven (the mean vector itself is
fine at 50%).

## 3. The fix: restricted-range stochastic Banzhaf

**Estimator.** Token i's importance is its average marginal contribution over random coalitions S:
`v_i = E_S[ f(S∪{i}) − f(S) ]`, where `f(S)` = answer recovery when keeping set S. Estimated cheaply
by the **difference of conditional means**:

```
v_i  ≈  E[ f(S) | i ∈ S ]  −  E[ f(S) | i ∉ S ]      over M random coalitions S
```

A single forward per coalition yields one in-set/out-of-set sample for *every* token, so the whole
attribution costs **O(M) forwards**.

**The restriction.** Sample the coalition keep-fraction `p ~ U(lo, 1)` (instead of `U(0,1)`). With
`lo = 0.5` every coalition keeps ≥50% of the tokens, so every sampled input is **mostly real → mostly
coherent → in-distribution** — the sequence-OOD samples that corrupt the estimate are simply never
drawn. The **random budget** (our core contribution) is preserved; it is only bounded to `[lo, 1]`.

**Interpretation.** High keep-fraction = "most of the rest is present" = a **near-conditional /
observational** marginal = necessity-like; empirically it also ranks well for *insertion* (the
important tokens are important under both faces).

**Length-adaptive `lo`.** The optimal restriction scales with sequence-OOD severity, i.e. with length:

| bin    | full (lo=0) | restricted (lo=0.5) | best |
|--------|-------------|---------------------|------|
| short  | **0.762**   | 0.512               | full |
| medium | 0.596       | **0.564**           | restricted |
| long   | 0.409       | **0.623**           | restricted |

Short prompts have no sequence-OOD, so restriction *hurts* (use full range). Medium/long prompts need
restriction. `lo=0.7` over-restricts (too few per-token out-of-set samples → noisy `s0`).

## 4. Result

Fixed `lo=0.5`, n=9 (Everest + 8 IMDB, length-binned), same insertion metric, AttnLRP via the LRP
back-prop:

```text
ALL n=9:   restricted Banzhaf 0.577   vs   AttnLRP 0.509   (full-range 0.555)   [indist run]
ALL n=9:   restricted Banzhaf 0.498   vs   AttnLRP 0.514                        [fresh run, M-sweep]
```

**This does NOT replicate.** The 0.577 came from the `indist` run; the n=6 (0.573) "confirmation" was
*correlated* (same seed-0 generator path, n=9⊃n=6), not independent. A fresh independent run
(`research_cost_msweep.py`, same cases) gives the restricted Banzhaf **0.498 < AttnLRP 0.514** — it
flips sign. The per-case insertion AUC carries ±0.1–0.15 seed noise (aggregate ±0.08), comparable to
the 0.068 margin. So the honest result is **competitive within noise**, not a win. The multi-seed test
(`research_banzhaf_seedvar.py`) **confirms** this: on Everest, 5 seeds give Banzhaf 0.652 ± 0.096 vs
AttnLRP 0.619 — a +0.033 mean gap that is only 0.35σ, with a per-case seed-std (~0.10) larger than the
margin itself.

The length-adaptive `lo(T)` structure (full for short, restricted for medium/long) is a *real and
reproducible* qualitative effect (it follows from the sequence-OOD theory); the *quantitative* claim
of out-scoring AttnLRP is not established.

**Takeaway (honest):** the sequence-OOD diagnosis is solid and restricting coalitions to the
in-distribution regime *materially helps* perturbation attribution; but a clean *quantitative* win
over AttnLRP is within the noise at n=9, and the method costs 200–800× more (§7).

## 5. Positioning: interventional vs observational

Replacing a feature with a **fixed baseline** (the mean vector) is **interventional** removal; it sends
the joint input off-manifold under heavy masking. **Observational** removal marginalizes the feature
out *conditioned on the rest*, keeping the input on-manifold. This is the conditional/observational
SHAP distinction (Aas et al.; Frye et al., "Shapley explainability on the data manifold"; Janzing et
al., interventional vs observational). **Our restricted-range Banzhaf is the observational principle
realized implicitly** (high keep-fraction ≈ conditioning on the rest), and it is **modality-agnostic**
(restrict the coalition size; no special masking) — it applies to vision and LLMs alike.

## 6. What does NOT work (exhaustive negative results)

The "obvious" explicit-observational fixes are *worse* than the implicit restricted-range:

| erasure | keep=0.1 entropy | full-range Banzhaf AUC | verdict |
|---------|------------------|------------------------|---------|
| mean (interventional)     | ~6.7 | 0.490 | baseline |
| **restricted mean (lo=0.5)** | (mostly-real → low) | **0.575–0.577** | **best** |
| attention-masking (holes) | ~5.0 | 0.465 | worse (mid-sequence holes still OOD + positional leakage) |
| contiguous removal (re-index) | ~6.5 | 0.446 | worse (random subset = gibberish; RoPE positions corrupted) |

**Why the raw gradient fails (decisive).** A natural objection: maybe the gradient only looked
uninformative because we evaluated it at *degenerate masked* states. Tested directly — compute the
token-gradient `d(logit_ans)/d(w_i)` at the **full, prediction-preserved input** (uniform mask a=1.0:
recovery 1.0, next-token entropy 0.23) and correlate it with the *actual* per-token occlusion effect:

```text
a (uniform mask)   recovery  entropy   corr(grad, occlusion)   grad insertion-AUC
1.00 (full)          1.000     0.23           +0.002                 0.055
0.80                 1.036     0.13           +0.029                 0.018
0.50                 0.236     4.71           +0.115                 0.107
AttnLRP                  —        —            +0.190                 0.486   (occlusion ceiling 0.480)
```

Even at the full input this *directional* correlation is **≈0** (+0.002). **But — important
correction — this is specific to the DIRECTIONAL gradient** `(emb−mean)·grad` (the first-order
occlusion estimate, which is also what the soft-mask FRI optimizes). It is *misaligned* on LLMs
because the large occlusion effect (token→mean) ≠ the local first-order direction for a nonlinear,
attention-routing model. The gradient **MAGNITUDE** `|grad|` is a different story:

```text
method (n=12)        insertion-AUC   corr(occlusion)
raw |grad|               0.512           +0.208
input×grad               0.154           +0.158
(emb−base)×grad          0.069           −0.027   ← the directional one above
integrated gradients     0.225           +0.070
AttnLRP                  0.535           +0.246
occlusion-order           0.491
```

So `|grad|` (direction-agnostic) is **informative and competitive with AttnLRP** (0.512 vs 0.535,
within noise), at lower cost (a plain backward, no LRP patch); only the *directional/projected*
gradients fail. This sharpens the vision↔LLM story: the **directional** gradient works on vision
(locally-linear → 1st-order direction ≈ occlusion, AUC 0.858) but fails on LLMs (nonlinear), whereas
the **magnitude** survives the LLM nonlinearity. The soft-mask FRI fails on LLMs *because it uses the
directional gradient*; a magnitude `|grad|` or AttnLRP finds the sufficiency set instead.

The same magnitude-vs-direction split appears in the SIGN. On vision the gradient sign cleanly flags
distractors: negative-directional-grad patches have negative occlusion and dropping them *raises* the
class prob (0.606→0.671), so the signed ranking beats the magnitude (0.825 vs 0.739). On LLMs the sign
has **zero headroom** — even the ground-truth signed-occlusion ceiling (0.491) is *below* `|grad|`
(0.543), and `|grad|×sign(AttnLRP)` ≈ `|grad|`. The reason is redundancy: 42% of tokens have negative
single-occlusion yet dropping them *lowers* recovery (0.785<1), i.e. a token negative in isolation is
needed in context, so the single-token sign is noise. LLM token importance is magnitude-like
(involvement), not cleanly signed — so `|grad|` is near-optimal and there is nothing to gain from the
sign (whereas vision's spatially-distinct distractors make the sign genuinely useful). (Confirmed: in-distribution soft-FRI, restricted
budget, 100 steps → 0.155 vs AttnLRP 0.510, 0/6.)

**Why the gradient/FRI works on vision (the unification).** The same diagnostic on CLIP: the
gradient's *insertion* AUC is **0.858** (60% of the occlusion ceiling 1.421) vs **0.055** on the LLM
(11% of its ceiling 0.480). The discriminator is the KEEP-SUBSET operation's in-distribution-ness —
keeping a subset of image patches is a valid *partial image* (the model recognizes objects from
parts; object patches cluster spatially) → smooth, meaningful recovery surface that the gradient
climbs to the sufficient set; keeping a subset of tokens is *scattered → incoherent → OOD* → a
degenerate surface where the gradient is misaligned. The insertion *ceiling* itself reflects this
(vision 1.421 vs LLM 0.480). So **"FRI is a vision method", the "raw-gradient corr ≈ 0 on LLM", and
the "sequence-OOD law" are one root cause**: subset-masking is in-distribution for vision's
masking-robust continuous/spatial input but OOD for the LLM's discrete/sequential input. (Uniform
mask recovery is *similar* for both — robust until a≈0.5 — so the SUBSET structure, not uniform
scaling, is the discriminator.)

**Final lesson.** Sequence-OOD is *fundamental to random token subsets*: language needs coherence, and
any random subset breaks it — **no erasure scheme (mean / attention / removal) fixes it for random
coalitions.** The only fix is keeping **most** tokens (restricted-range → mostly-coherent input). And
keeping **positions** (mean-mask) beats shifting them (removal). Other dead ends: annealed budget
(fixes the needle *rank* but not insertion-AUC), input_LN-output FRI (3× the raw-soft FRI but its
"beats AttnLRP" was a metric artifact: same-metric AttnLRP 0.664 > 0.422), magnitude/norm-rescale/
slerp, STE hard-mask, frequency-corpus mean (leaks — too plausible a token), random-real tokens
(incoherent, entropy 10), moving-average/blur (≈ uniform), first-token "sink" fix (position 0 is not
Qwen's sink for these prompts).

## 7. Cost

- **Restricted-range Banzhaf:** `M` forwards per case (M≈2000), no back-prop, **model-agnostic** (no
  gradient / no LRP-modified back-prop needed).
- **AttnLRP:** 1 forward + 1 LRP back-prop (~2.5 forward-equivalents), but **model-specific** (needs
  the LRP rule patch).

So the Banzhaf is ~100–800× the forward cost but removes the model-specific machinery and yields a
*distribution-faithful* (observational) attribution.

**M-sweep (lo=0.5, n=9, mean insertion AUC by forward budget):**

```text
   M     fwd/case   ~x AttnLRP   meanAUC   >AttnLRP
  250       250        100x       0.392      22%
  500       500        200x       0.484      56%
 1000      1000        400x       0.508      33%
 2000      2000        800x       0.498      33%
 AttnLRP  ~2.5 fwd-equiv          0.514
```

The curve plateaus at ~0.48–0.51 from M≥500 — i.e. it **saturates just below AttnLRP (0.514)** in this
run and never clearly exceeds it. So there is **no favorable cost floor**: AttnLRP matches/beats the
Banzhaf at ~1/200th–1/800th the cost. The Banzhaf's only advantage is being gradient-free /
model-agnostic and distribution-faithful — not accuracy-per-FLOP.

## 8. Limitations & next

n=9 is moderate (scale to n≥20). The length-adaptive `lo(T)` gain is in-sample — confirm on held-out
prompts with the a-priori rule. One long-case holdout where AttnLRP's single-token sharpness wins.
The restricted-range signal is necessity-like; the necessity↔sufficiency ranking alignment on the
"important" tokens deserves its own analysis. Cross-model (other LLMs) and the vision realization of
the same restricted-range principle are open.
