# claude_llm_fri — FRI on LLMs: mechanism, performance, insights (self-contained handoff)

> A fresh session should be able to read THIS FILE ALONE and understand the whole "FRI on LLMs"
> research line: what FRI is, the best mechanism found, how it performs, why it works, the
> conceptual insights, what does NOT work, the code, and the open questions.

---

## 0. One-line answer

**The best LLM FRI = a gradient-FREE, length-adaptive *restricted-range Banzhaf* on actual
recoveries, with `|grad|`-guided candidate focusing for long sequences, plus tokenizer-local
closure + self-selection (borrowed from Codex) for robustness.** It beats AttnLRP on *insertion*
(sufficiency) at every token length and *dominates* tight / non-redundant tasks (retrieval,
arithmetic) by finding the complete **cooperative set** that AttnLRP's diffuse relevance cannot.

---

## 1. Background — what FRI is and the goal

- **FRI = Feature-Recovery Insertion** = a model-agnostic **SUFFICIENCY** attribution: find the
  smallest set of input units that, when KEPT (rest masked to a baseline), RECOVERS the model's
  prediction. Its signature idea is **random-budget soft-insertion** (sample coalitions at random
  keep-budgets, credit units by how much they help recovery). It was developed on **vision** (CLIP
  SAE / hidden-feature ERF) where it works very well.
- **Counterpart = NECESSITY** (the "C" / conditional method): the smallest set whose REMOVAL destroys
  the prediction (deletion). On vision/copy this is a chunked-CONDITIONAL greedy deletion.
- **The LLM question of this line:** can FRI/perturbation find the sufficiency set on LLMs and **beat
  AttnLRP on insertion**? (Deletion is FRI's weak point, so insertion is FRI's story on LLM.)

### Setup a fresh session needs
- **LLM:** `Qwen/Qwen2.5-1.5B-Instruct`, fp32, `cuda:0`. (GPU is sometimes shared with a co-tenant —
  poll free mem, retry on OOM; use `gradient_checkpointing_enable()` + `logits_to_keep=1` for backward.)
- **Vision (for the cross-modal contrast):** CLIP `vit_base_patch16_clip_224.laion2b_ft_in12k_in1k`,
  N=196 patches.
- **Masking baseline:** token embedding → **global vocab-mean** (`embed.weight.mean(0)`). Keep BOS +
  the readout/last position always.
- **Recovery:** `rec(S) = (softmax_prob(answer | masked to S) − base) / (full − base)`, where
  `base` = all-masked, `full` = unmasked. So rec=1 at full, ~0 at all-masked.
- **Insertion AUC** (sufficiency, HIGHER=better): rank tokens by score, keep top-k (rest mean),
  measure rec, AUC over keep-fraction grid `FR = [0,.02,.05,.1,.15,.2,.3,.5]`.
- **Deletion AUC** (necessity, LOWER=better): remove top-k (set to mean), measure rec, same grid.
- **Baseline to beat: AttnLRP** (lxt/efficient implementation) — the SOTA transformer attribution.

---

## 2. The BEST mechanism (the recipe)

**Gradient-FREE restricted-range Banzhaf, length-adaptive, + gradient-guided focusing for long.**

1. **Score = restricted-range Banzhaf** (gradient-free difference-of-conditional-means):
   ```
   score_i = E[rec(S) | i ∈ S] − E[rec(S) | i ∉ S]   over M sampled coalitions S
   ```
   Implementation: sample M masks Z (below), `R = rec(Z)`, then
   `score = (Z*R).sum0 / Z.sum0  −  ((1−Z)*R).sum0 / (1−Z).sum0`. This realizes the random-budget
   soft-insertion idea on ACTUAL recoveries (no gradient).

2. **Restricted-range coalition sampling** (handles sequence-OOD): keep-frac `p ~ U(lo, 1)`,
   `z_i ~ Bernoulli(p)`, always keep BOS + readout.
   **`lo` length-adaptive: `lo = 0.0 if T < 100 else 0.5`.** High keep-frac ⇒ coalitions stay
   mostly-real ⇒ in-distribution ⇒ the difference-of-means is meaningful.

3. **Gradient-guided focusing (for long / large-T)**: compute `|grad|` (one backward), take the
   top-K candidate tokens; in each coalition VARY ONLY the candidates and HOLD non-candidates at
   their REAL value. This removes the variance from the T−K irrelevant tokens ⇒ sample-efficient
   marginal estimation. Final ranking = candidates by Banzhaf marginal, then non-candidates by `|grad|`.
   (M≈1200–2500 coalitions; K≈40 short / 100 long.)

4. **Tokenizer-local closure (borrowed from Codex)**: radius-1 max-diffusion of the score to
   neighbors (`s_i ← max(s_i, decay·s_{i±1})`, decay≈0.95). Repairs BPE-fragment blindness
   (`"17"→"1","7"`). Helps single-token methods a lot (AttnLRP 0.330→0.465) and the adaptive-Banzhaf
   modestly (fri_ad 0.547→0.585); helps the cooperative gradient-guided variant little (+0.003) — it
   already captures conjunctions, which is why this VALIDATES the cooperative approach.

5. **Compact self-selection portfolio (borrowed from Codex)**: build a few cheap orders (adaptive,
   gradient-guided, ±closure), validate each by partial hard-insertion at a few keep-fractions, pick
   the best. Hedges per-case failure modes (0.575, near the per-case best).

### Cost-budgeted recipe (~90 forward + 1 backward) — research_cheap_fri.py

For a hard ~100-compute budget, the cheap pipeline (validated n=34 KV+arith+long):
`full-grad |grad| (1 backward) → top-K candidates (K=32 short / 100 long) → cooperative Banzhaf over
the candidates at M=80 (non-candidates held REAL) → local-closure variant → self-selection`.
**SELECT80 = 0.500 at ~89 forwards + 1 backward** (single_occ = 140 fwd, AttnLRP = 1 bwd). HONEST
paired verdict at n=34 (the small-n n=14 numbers were optimistic):
- **vs AttnLRP: COMPETITIVE, not a decisive win** — SELECT80 wins 71% of cases but the mean margin is
  only +0.039±0.037 (+1.1σ = within noise). By task all positive but <1σ (kv +0.060, arith +0.010,
  long +0.051). So cheap FRI ≈ AttnLRP with a consistent small edge; AttnLRP is far cheaper (1 bwd).
- **vs single_occ: clear win on COOPERATIVE tasks** (arith +0.252/92%, long +0.329/75%) — single_occ
  cannot find the cooperative set and is O(T). **But on short KV single_occ ties/wins** (+0.020, 38%
  win) because short T makes single_occ cheap AND good at the single-token value.
- M-sweep: m80 0.441 ≈ 90% of m256 0.489 → M=80 is the cost-quality sweet spot. self-selection adopts
  closure per-case (KV/long yes, arithmetic no).
- **COST-QUALITY TRADEOFF (key honest finding): the cheap M=80 SACRIFICES the high-M FRI domination of
  AttnLRP** — at M=1200 (research_hard_cases) FRI crushed AttnLRP on KV (+0.35, 100%); at M=80 that
  shrinks to a tie (+0.06). So ~100-cost buys a model-agnostic method that MATCHES AttnLRP and beats
  single_occ on cooperative tasks; the decisive AttnLRP domination needs high M.

**Codex-idea verdict (research_cheap_fri.py / research_hard_closure.py):** self-selection ✓ (key win,
per-case adaptation), local closure ✓ (+0.06, esp. single-token methods + KV), candidate-restricted
cheap M ✓ (the cost cut). REJECTED: PPD α=0.3 interpolation gradient (full-grad candidates are
better — α=0.3 gave long 0.201 vs full 0.596), and non-local 2nd-order Banzhaf interaction (too noisy
at low M, hurts: m128_nonloc 0.469 < m128_loc 0.519).

### Cross-model validation (research_xmodel_fri.py, tight KV+arith, M=1000, n=30/model)

The FRI-vs-AttnLRP win GENERALIZES beyond Qwen, and is DECISIVE at solid M (not the ~100-cost cheap version):
- **Llama-3.2-1B: FRI(SELECT) >> AttnLRP +0.235 (+7.6sigma, 93% win)** — kv +0.307 (5.8sigma), arith
  +0.163 (8.0sigma); single_occ collapses on Llama KV (0.076). A NEW family, decisive win.
- **gemma-2-2b: FRI >> single_occ (0.696 vs 0.466)** — AttnLRP cannot even RUN here: lxt supports only
  {bert, gpt2, llama, qwen2, vit}, NOT gemma2. This is itself a point for FRI's model-agnosticism.
- Meta-Llama-3-8B: queued (Instruct cache was an incomplete download; base 8B complete but blocked by a
  GPU co-tenant — environment, not method).
- **Lever = M:** the Qwen n=34 TIE was M=80 (cheap); at M=1000 on tight cases FRI DOMINATES AttnLRP on a
  new family (+7.6sigma). Higher M buys the decisive win; ~100-cost buys a tie + model-agnosticism.
- Engineering note: bf16 models need the masked embedding cast back to `emb.dtype` (else float32 mask x
  bf16 weight -> matmul dtype error).

### Necessity / deletion — the SAME ranking does both faces (research_cheap_necessity.py)

The cheap-FRI inspirations transfer to NECESSITY (deletion AUC, lower=better; 3 models, tight KV+arith, n=24/model):
- **ONE cheap cooperative Banzhaf ranking does BOTH faces**: insertion keeps high-marginal, deletion
  removes high-marginal. SELECT beats AttnLRP AND single_occ on deletion on ALL 3 models (Qwen 0.243 <
  0.41/0.40; Llama 0.073 < 0.09/0.29; gemma 0.162 < 0.34, attn N/A). **If you already ran the insertion
  FRI, necessity is ~FREE (reuse the ranking).**
- Driven by COOPERATIVE necessity (arith): Banzhaf/cheap_cond 0.13-0.51 vs **AttnLRP 0.55-0.80 = FAILS
  cooperative deletion** (ranks individual tokens, misses the operand set). KV deletion is a tie (~0.02 —
  single value obvious, per-token necessity everyone finds).
- candidate-restriction -> **cheap_cond** (chunked greedy on |grad| candidates, ~280 cost) is the
  strongest standalone necessity (Llama arith 0.125). self-selection (SELECT) adapts per-case and wins.
- **ASYMMETRY: tokenizer-closure HURTS necessity** (Qwen arith bz_loc 0.578 > 0.514, gemma 0.739 >>
  0.384) — closure is an INSERTION/single-token tool (boosts neighbors), wrong for deletion.

### Does the one-ranking-does-both unification transfer to VISION? NO — it is redundancy-dependent

research_pred_attribution_bench.py (CLIP, n=9, cooperative Banzhaf added):
- **Vision needs a SEPARATE necessity method** (greedy/conditional): greedy del 0.076 DOMINATES the
  cooperative-Banzhaf del 0.334 (~4.4x). The Banzhaf does SUFFICIENCY (ins 1.328, 2nd best) but NOT
  necessity. So on vision the same ranking does NOT do both.
- **WHY = REDUNDANCY:** vision recognizes objects from partial views → necessity set (large) != the
  sufficiency set (small), nearly DISJOINT (overlap 0.03). No single ranking serves both.
- **UNIFYING LAW (LLM + vision):** redundancy LOW (LLM arithmetic tight; a concentrated single-object
  image) → necessity ≈ sufficiency → ONE cooperative-Banzhaf ranking does both. redundancy HIGH (vision
  distributed classification; LLM sentiment) → necessity != sufficiency → SEPARATE necessity needed.
  The discriminator is REDUNDANCY, not modality (the zebra-elephant concentrated case had banzhaf del
  0.02 = greedy — unification holds there; it is redundancy-graded even within vision).

### Resolving redundancy cheaply (LLM->vision->LLM round-trip) + the COMPLETE picture

Idea: use the cooperative-Banzhaf marginal as the PRIOR for the conditional necessity (single-occ
misses redundant supporters; the Banzhaf, coalition-averaged, captures them).
- VISION (research_vision_cheap_necessity.py, n=9): chunked with Banzhaf-prior 0.211 < single-occ-prior
  0.308 (31% better, ~20x cheaper than full greedy), DECISIVE on truly-redundant cases (redshank
  single-occ 0.938 FAILS -> Banzhaf 0.352) — but still > greedy 0.076; HURTS on concentrated cases
  (elephant chunked_bz 0.100 > chunked_occ 0.082) -> the prior should be redundancy-ADAPTIVE.
- LLM redundant (research_llm_redundancy_necessity.py, sentiment n=16): Banzhaf-prior chunked 0.096 <
  single-occ chunked 0.105; raw Banzhaf 0.089 is the best PERTURBATION method — but **AttnLRP 0.067
  STILL WINS** (bz<attn 19%).

**COMPLETE PICTURE (redundancy = master variable, both faces x both modalities):**

| | non-redundant (tight) | redundant |
|---|---|---|
| sufficiency (insertion) | FRI/Banzhaf >> AttnLRP | FRI/Banzhaf (78%) |
| necessity (deletion) | SAME Banzhaf ranking (unified, > AttnLRP) | **AttnLRP (LLM 0.067) / greedy (vision 0.076)** |

Perturbation does NOT fully "solve" redundant necessity — its true territory is non-redundant necessity
(unified both faces from one ranking) + sufficiency (universal). For REDUNDANT necessity use AttnLRP
(LLM, cheap 1-bwd) or greedy (vision); the Banzhaf-prior conditional is the best cheap perturbation
fallback (improves over single-occ, ~20x cheaper than greedy, redundancy-adaptive).

### Cheaper necessity: adaptive granularity FAILED — the bottleneck is candidate COVERAGE

research_adaptive_necessity.py (CLIP n=9, actual-forward cost tracked). Hypothesis: deletion AUC is
dominated by EARLY removals, so spend per=1 (greedy precision) on the first fine_k removals then go
coarse -> close the chunked->greedy gap cheaply. RESULT (deletion AUC / cost):

| method | del | cost (fwd) |
|---|---|---|
| greedy (full coverage) | 0.076 | 19306 |
| chunk_bz (Banzhaf prior, fixed) | 0.211 | 980 |
| adapt_bz (fine_k=20) | 0.194 | 2682 |
| adapt_occ (single-occ prior) | 0.294 | 2366 |

- adaptive granularity is NOT worth it: 0.211 -> 0.194 (8%) at 2.7x cost, still far from greedy 0.076.
- **The real bottleneck is candidate COVERAGE**, not granularity: greedy considers ALL N; M=100 MISSES
  the necessary set on hard/redundant cases (standard-schnauzer 0.439 vs greedy 0.142) — tokens outside
  top-M are never removed at any granularity. adapt_occ 0.294 >> adapt_bz 0.194 reconfirms the Banzhaf
  prior is the real cost-reduction.
- **Necessity cost-quality frontier (honest): cheap = chunk_bz (~980 fwd, 0.211, ~20x cheaper); accurate
  = full greedy (~19k, 0.076); NO good middle.** Conditional necessity is COVERAGE-BOUND (re-confirms
  "R x N forwards fundamental"). The next lever is cheaper COVERAGE (per-case adaptive M, redundant-group
  batched removal), NOT granularity.

### Necessity vs REAL AttnLRP — perturbation BEATS AttnLRP on deletion everywhere except LLM-redundant

research_vision_attnlrp_necessity.py (torchvision vit_b_16 + lxt vit_torch monkey-patch = genuine
AttnLRP, n=6, deletion AUC LOWER=better): **the conditional/perturbation necessity BEATS AttnLRP 100%.**

| method | vision del | note |
|---|---|---|
| AttnLRP | 0.761 | strong at sufficiency (gradient ins 0.858) but WEAK at necessity |
| greedy | 0.288 | conditional, dominates |
| chunk_bz | 0.569 | cheap necessity, beats AttnLRP 100% |
| banzhaf | 0.669 | raw, beats AttnLRP |
| single_occ | 0.772 | the ONLY method worse than AttnLRP |

Complete necessity-vs-AttnLRP map (deletion):

| regime | winner |
|---|---|
| vision (CLIP/ViT) | necessity (greedy/chunk_bz/banzhaf) BEATS AttnLRP 100% |
| LLM tight (KV/arith) | necessity BEATS AttnLRP (chunk_bz 0.243<0.410 Qwen, 0.073<0.294 Llama) |
| LLM redundant (sentiment) | AttnLRP wins (0.067<0.089) — the ONLY exception |

**Sharpened positioning: AttnLRP is a SUFFICIENCY tool (nails individual high-relevance supporters)
that only masquerades as necessity on redundant LLM classification. For genuine NECESSITY (the
conditional minimal-removal set), perturbation/greedy dominates it on vision + LLM-tight** — its 1-shot
relevance cannot resolve conditional redundancy.

> ⚠⚠ METRIC CORRECTION (2026-06-22): ALL vision AUC numbers ABOVE this line that exceed 1 (ins 1.095,
> 1.328, del normalized) used `F.hard_curves` = AUC of `p_ins/p_full`, which is BUGGY (exceeds 1 on
> distractor removal). The literature metric is raw-prob AUC ∈ [0,1]. Corrected (research_vision_raw_metric.py
> + research_vision_greedy_raw.py, 2 model types): **SUFFICIENCY gain over inflow is SMALL** (Banzhaf M512
> ins 0.772 vs inflow 0.734 / +0.04, 75%; IG-cost M64 TIES/loses inflow), but **NECESSITY survives — greedy
> DOMINATES inflow 100% / ~4× on raw del** (laion 0.084 vs 0.329; augreg2 0.017 vs 0.116); chunk_bz
> competitive but sample-noisy. REFRAME: vs inflow on the correct metric, the perturbation's CLEAR win is
> CONDITIONAL NECESSITY (greedy, expensive), NOT cheap sufficiency. Treat the inflated numbers above as
> directional only; the corrected verdicts are here + in memory.

### Vision: the correct baseline is INFLOW (not AttnLRP) — and we beat it

The proper vision attention baseline is **inflow** (information-flow), NOT AttnLRP (AttnLRP is much
weaker on vision: del 0.761 vs inflow 0.314). research_vision_inflow_necessity.py (CLIP timm, n=9):

| method | del↓ | ins↑ | beats inflow: del / ins |
|---|---|---|---|
| inflow (baseline, 1 fwd) | 0.314 | 0.936 | — |
| greedy | 0.076 | 0.986 | 100% / 67% |
| chunk_bz | 0.211 | 1.351 | 78% / 78% |
| banzhaf | 0.334 | 1.328 | 67% / 78% |

- **greedy DOMINATES inflow on necessity** (0.076 vs 0.314, ~4x, 100%).
- **cheap chunk_bz beats inflow on BOTH faces** (necessity 0.211<0.314, sufficiency 1.351>0.936; 78% each).
- inflow is genuinely STRONG (cheap 1-forward; chunk_bz wins 78% not 100%) — but the conditional greedy
  (necessity champion) and the cooperative Banzhaf (cheap both-faces) beat it.

### IG-level cost (~32-64 fwd/bwd) vs inflow — the sufficiency/necessity cost asymmetry is fundamental

research_vision_iglevel_necessity.py (CLIP n=9):

| method | del↓ | ins↑ | cost | vs inflow |
|---|---|---|---|---|
| inflow | 0.314 | 0.936 | 1 | baseline |
| bz32 | 0.488 | 0.931 | 32 | ins 44% |
| bz64 | 0.480 | 1.095 | 64 | **ins 67%** |
| gradcond32 (\|grad\| prior) | 0.522 | — | 32 | del 0% |
| inflowcond32 (inflow prior) | 0.310 | — | 31 | del 100% (margin ~1%) |

- **SUFFICIENCY: model-agnostic Banzhaf M=64 BEATS inflow insertion (1.095 vs 0.936, 67%) at IG-cost** —
  the "win at IG cost" contribution holds vs inflow, model-agnostically. (FRI's home: gradient-friendly.)
- **NECESSITY: NOT achievable model-agnostically at IG-cost.** Model-agnostic cheap priors fail
  (Banzhaf/\|grad\| del 0.48-0.52 >> inflow 0.314). The inflow-PRIOR conditional beats inflow 100% but
  by ~1% (0.310 vs 0.314) AND is not model-agnostic (concept check only) — a good prior carries it, the
  cheap conditional adds little; a meaningful model-agnostic necessity win needs ~980 fwd.
- **FUNDAMENTAL ASYMMETRY: sufficiency = gradient-friendly / cheap (IG-cost, model-agnostic);
  necessity = conditional / forward-hungry (coverage-bound, ~980 fwd).**

---

## 3. Performance (insertion AUC, FRI vs AttnLRP unless noted; higher=better)

| regime | best FRI variant | FRI | AttnLRP | FRI win-rate |
|---|---|---|---|---|
| short/medium classification (n=18: IMDB+SST2+AG News) | adaptive restricted-Banzhaf | **0.601** | 0.390 | 78% |
| long, T 430–637 (n=5) | gradient-guided Banzhaf | **0.620** | 0.541 | 80% |
| tight — KV retrieval / context copy (n=6) | grad-guided / adaptive | **0.570** | 0.195 | **100%** |
| tight — arithmetic (n=6) | adaptive restricted-Banzhaf | **0.583** | 0.465 | 83% |
| tight — ALL (n=12) | best FRI | **0.547** | 0.330 | 92% |

By dataset (short/medium): SST2 0.458 vs 0.335 (83%), AG News 0.787 vs 0.280 (100%), IMDB tie
(0.557≈0.554, IMDB carries the long cases).

**Necessity (deletion AUC, conditional vs AttnLRP; LOWER=better):**
- tight ALL n=12: conditional 0.373 < AttnLRP 0.418 (67% best); arith 0.726 < 0.815 (83%); KV tie 0.020.
- short classification n=15: AttnLRP 0.049 < conditional 0.113 (AttnLRP wins deletion on REDUNDANT tasks).

**⇒ FRI beats AttnLRP on insertion at every length and dominates tight tasks (92%). The deletion
winner is task-dependent (conditional on tight/copy/vision; AttnLRP on redundant classification).**

---

## 4. Why it works (mechanism + evidence)

- **Gradient-FREE is essential.** The directional/projected gradient `(emb−mean)·grad` CANCELS on the
  LLM nonlinearity: corr with true occlusion = **+0.002**, and gradient-based soft-FRI fails
  (0.155 vs AttnLRP 0.510, 0/6). The Banzhaf reads actual recoveries and sidesteps the dead gradient.
  - Vision is the OPPOSITE: the directional gradient WORKS (insertion 0.858, ~60% of ceiling) because
    vision is locally-linear/spatial. This split is the mechanistic root of "FRI is a vision method".
- **Restricted-range fixes sequence-OOD.** A random scattered token subset is an INCOHERENT input
  (input-level OOD, not vector-level). Restricting to high keep-frac keeps coalitions on-manifold so
  recoveries are informative. (This is the user's "input must also be in-distribution" law.)
- **The long failure is VARIANCE, not OOD.** Diagnostic (research_long_diagnostic.py): in the
  restricted regime the long-mask entropy is moderate (1.8–2.6 vs full-prompt 0.1–0.3) — NOT an OOD
  collapse (7–8). But Banzhaf M1500→M4000 keeps RISING (+0.08–0.12) ⇒ the per-token marginal is
  sample-starved for long T. Gradient-guided focusing (vary |grad| candidates, fix the rest real)
  reduces variance and recovers long (plain 0.500 → guided 0.620, beating AttnLRP 0.541).

---

## 5. Key insights / inspiration (the conceptual payload)

1. **The cooperative-SET insight (FRI's core value).** On tight retrieval, AttnLRP nails INDIVIDUAL
   necessary tokens (KV deletion 0.020 — removing the value breaks the answer) but FAILS the
   cooperative SET for insertion (0.195): it ranks high-relevance tokens, not the
   set-that-must-all-be-present-together. FRI's masking directly asks "keep ONLY this subset →
   recover?", so it finds the COMPLETE cooperative set. **Necessity ≈ per-token; sufficiency = a SET
   property — and FRI is the set finder.**

2. **The redundancy dichotomy (decides who wins).** *Redundant* tasks (sentiment): many partially-
   sufficient tokens, NO clean distractors (a token negative in isolation is needed in context) →
   AttnLRP competitive and wins deletion. *Tight* tasks (retrieval/arithmetic): small non-redundant
   cooperative/necessity set → FRI sufficiency dominates (92%) + conditional necessity wins (67%).
   **Redundancy is the controlling variable; perturbation is needed exactly where the dependency is tight.**

3. **The vision↔LLM unification (one root cause).** Subset-masking is in-distribution for vision's
   masking-robust spatial input (a kept patch subset = a valid partial image) but OOD for the LLM's
   discrete/sequential input (scattered tokens = incoherent). This single fact explains: (a) the
   gradient works on vision (0.858) not LLM (0.055); (b) "FRI is a vision method"; (c) the
   sequence-OOD law. The fix on LLM = gradient-free + restricted-range.

4. **AttnLRP's "negative relevance" is an artifact, and what AttnLRP actually is.** AttnLRP = exactly
   2 backprop rules — `identity_rule` (treat softmax/RMSNorm/SiLU as identity in backward, bypass
   saturation) + `divide_gradient`/uniform (split relevance /2 after matmul / element-wise-mult, fix
   the bilinear QKᵀ / attn@V / gated-MLP cancellation). Its negative-relevance tokens are POSITIVE
   supporters (mean occlusion +0.035, Banzhaf +0.342; removing them LOWERS recovery to 0.949) — the
   negative sign is an LRP-conservation artifact. **These rules are transformer-attention-specific:
   they FAIL on vision hidden-feature ERF (FRI's original domain).** So FRI and AttnLRP are
   complementary, not competitors.

5. **The gradient magnitude vs direction vs sign.** `|grad|` MAGNITUDE is informative but
   task-dependent (good on long copy 0.592, beating AttnLRP at its 1-backward cost; POOR on short
   classification 0.217 and tight 0.301). The DIRECTION cancels; the SIGN is noise on LLM (no clean
   distractors; signed-occ ceiling 0.491 < |grad| 0.543). So FRI(Banzhaf) is the robust gradient-free
   choice; |grad| is a cheap candidate selector / fallback.

---

## 6. What does NOT work (honest negative space — don't re-try these blind)

- gradient soft-FRI (directional/projected gradient): 0.155, cancels on LLM nonlinearity.
- `|grad|` magnitude alone: task-dependent (good long, poor short classification 0.217 / tight 0.301).
- gradient SIGN / direction: noise on LLM (no clean distractors).
- unrestricted Banzhaf (keep-frac U(0,1) for long): sequence-OOD-ish + high variance.
- plain restricted-Banzhaf on long: estimation variance (needs gradient-guided focusing).
- conditional/chunked deletion as the universal necessity winner: loses to AttnLRP on REDUNDANT short
  classification (wins only on tight/copy/vision).
- ambitious soft optimizers (ISI/ASI, multi-objective necessity solvers): failed earlier; the simple
  Banzhaf + conditional are the robust pair.

---

## 7. Code (re-run / extend)

Infra: `scripts/research_llm_imdb_sufficiency.py` — `load_imdb`, `PROMPT_TMPL`, `precompute_attnlrp`
(runs AttnLRP in a bf16 subprocess; the main model goes to CPU during it).

| script | what it shows |
|---|---|
| `research_insertion_datasets.py` | multi-dataset insertion; FRI beats AttnLRP short/medium 78% |
| `research_long_diagnostic.py` | WHY FRI loses long = variance (not OOD): entropy / M-sweep / ceiling |
| `research_grad_guided_long.py` | gradient-guided Banzhaf FIXES long (0.620 vs 0.541, 80%) |
| `research_hard_cases.py` | tight cases (KV/arith): FRI insertion 92%, conditional deletion 67% |
| `research_deletion_datasets.py` | necessity on classification: AttnLRP wins deletion on redundant |
| `research_gradient_variants_llm.py` | `|grad|` magnitude informative; direction cancels |
| `research_signed_llm.py` / `_vision.py` | sign helps vision, noise on LLM |
| `research_attnlrp_neg.py` | AttnLRP negative relevance is an artifact |
| `research_gradient_diagnostic.py` / `_vision.py` | gradient corr; vision 0.858 vs LLM 0.055 |
| `research_soft_fri_indist.py` | gradient soft-FRI refuted (0.155) |

Outputs: `outputs/insertion_datasets.json`, `outputs/hard_cases.json`, `outputs/deletion_datasets.json`.
Detailed running memory: `memory/project_necessity_two_pillar.md` (the consolidated note).

---

## 8. Open questions / next steps

- **Scale** (current n is modest): hard cases n=12 (KV 6 / arith 6), long n=5. Confirm at n≈20–30;
  add needle-in-haystack long retrieval (the predicted FRI sweet spot: long AND tight).
- **The one tight loss** (arith_1) and the one long loss (L1) are cases where AttnLRP nails a sharp
  set — characterize when that happens.
- **Cross-model**: does the gradient-guided restricted-Banzhaf transfer to other LLMs? (vision/CLIP
  already established as FRI's home.)
- **Cost**: M coalition forwards (≈1200–2500) + 1 backward — more than AttnLRP (1 backward) but
  gradient-free, model-agnostic, and the only method that finds the cooperative set. Variance
  reduction (antithetic/stratified) could lower M further.
- **Write-up**: the redundancy dichotomy + cooperative-set mechanism is the paper spine; FRI's home is
  tight-dependency sufficiency + vision hidden-feature ERF where AttnLRP's rules can't reach.
