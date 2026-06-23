# Best FRI mechanism for LLM (consolidated)

What the LLM evidence (this research line) converges to as the **best realization of FRI on LLMs**,
why, by regime, and the honest negative space. FRI's claim on LLM = **sufficiency / insertion**
(necessity is the conditional method's job; deletion is FRI's weak point).

## The recipe (best mechanism)

**Gradient-FREE restricted-range Banzhaf, length-adaptive, with gradient-guided focusing for long.**

1. **Score = restricted-range Banzhaf** (gradient-free difference-of-conditional-means):
   `score_i = E[rec | i∈S] − E[rec | i∉S]` over M sampled coalitions S.
   - `rec(S) = (softmax_prob(answer | masked to S) − base) / (full − base)`.
   - masking = token embedding → **global vocab-mean**; `base` = all-masked, `full` = unmasked.
   - This IS the "random-budget soft-insertion" idea, realized gradient-free on actual recoveries.

2. **Restricted-range coalition sampling** (handles sequence-OOD): keep-frac `p ~ U(lo, 1)`,
   `z_i ~ Bernoulli(p)`, always keep BOS + readout. **`lo` length-adaptive: `lo = 0 if T<100 else 0.5`.**
   Keeps coalitions mostly-real ⇒ in-distribution ⇒ the difference-of-means is meaningful.

3. **Gradient-guided focusing (for long / large-T)**: take `|grad|` (1 backward) → top-K candidate
   tokens; in each coalition VARY ONLY the candidates and HOLD non-candidates at their REAL value.
   Removes the T−K irrelevant tokens' variance ⇒ sample-efficient marginal estimation. Rank =
   candidates by Banzhaf marginal, then non-candidates by `|grad|`.

## Why this is the best (mechanism + evidence)

- **Gradient-FREE** — the directional/projected gradient `(emb−mean)·grad` CANCELS on the LLM
  nonlinearity (corr with occlusion +0.002; gradient soft-FRI fails 0.155 vs AttnLRP 0.510). The
  Banzhaf reads ACTUAL recoveries, sidestepping the dead gradient. (Vision is the opposite: the
  directional gradient works, 0.858 — locally linear/spatial. The split is the root cause of
  "FRI is a vision method".)
- **Restricted-range** — unrestricted coalitions = scattered incoherent token subsets = sequence-OOD
  (input-level, not vector-level); restriction to high keep-frac keeps them on-manifold so recoveries
  are informative.
- **Gradient-guided for long** — the plain restricted-Banzhaf fails long NOT from OOD (mask entropy
  moderate 1.8–2.6 in the restricted regime, not the 7–8 collapse) but from ESTIMATION VARIANCE /
  sample-inefficiency (M1500→M4000 still rising). Focusing samples on |grad| candidates + fixing the
  rest real reduces variance and recovers long.

## Performance by regime (insertion AUC, FRI vs AttnLRP; higher = better)

| regime | best FRI variant | FRI | AttnLRP | win |
|---|---|---|---|---|
| short/medium classification (n=18, IMDB+SST2+AG News) | adaptive restricted-Banzhaf | 0.601 | 0.390 | 78% |
| long T 430–637 (n=5) | gradient-guided Banzhaf | 0.620 | 0.541 | 80% |
| tight: KV-retrieval (context copy) (n=6) | grad-guided / adaptive | 0.570 | **0.195** | **100%** |
| tight: arithmetic (n=6) | adaptive restricted-Banzhaf | 0.583 | 0.465 | 83% |

⇒ **FRI beats AttnLRP on insertion at every length AND dominates tight/non-redundant cases (92% overall there).**

## The cooperative-SET insight (why FRI wins where AttnLRP can't)

On tight cases AttnLRP nails INDIVIDUAL necessary tokens (KV deletion 0.020 — removing the value
breaks the answer) but FAILS the cooperative SET for insertion (0.195) — it ranks high-relevance
tokens, not the set-that-must-all-be-present. FRI's masking directly tests "keep only this subset →
recover?" so it finds the COMPLETE cooperative set. This is FRI's core value.

## Redundancy dichotomy (where each method wins)

- **Redundant tasks** (sentiment): many partially-sufficient tokens, no clean distractors → AttnLRP
  competitive and WINS deletion (short classification deletion: AttnLRP 0.049 < conditional 0.113).
- **Tight tasks** (retrieval / arithmetic): small non-redundant cooperative/necessity set → FRI
  sufficiency dominates (92%) + conditional necessity wins (67%); AttnLRP fails insertion.
- Redundancy is the controlling variable; perturbation (FRI + conditional) is needed exactly when the
  dependency is tight.

## What does NOT work (honest negative space)

- gradient soft-FRI (directional/projected gradient): 0.155 — cancels on LLM nonlinearity.
- `|grad|` magnitude alone: task-dependent — informative on long copy (0.592, beats AttnLRP at its
  1-backward cost) but POOR on short classification (0.217) and tight cases (0.301); FRI(Banzhaf) is
  the robust choice.
- gradient SIGN / direction: noise on LLM (no clean distractors — a token negative in isolation is
  needed in context; signed-occ ceiling 0.491 < |grad| 0.543).
- unrestricted Banzhaf: sequence-OOD.
- plain restricted-Banzhaf on long: estimation variance (needs gradient-guided focusing).

## Cost & positioning

- Cost: M coalition forwards (M≈1200–2500) + 1 backward (for the gradient-guided candidates). More
  than AttnLRP (1 backward) but **gradient-free, model-agnostic, and finds the cooperative SET**.
- AttnLRP is cheap and strong on redundant classification + individual necessary tokens, but its
  relevance is DIFFUSE (fails the cooperative set) and its 2 rules are transformer-attention-specific
  (fail on vision hidden-feature ERF — FRI's original domain). FRI and AttnLRP are complementary.

## One-line answer

The best LLM FRI = **gradient-free, length-adaptive restricted-range Banzhaf on actual recoveries,
with |grad|-guided candidate focusing for long sequences** — it beats AttnLRP on insertion at all
lengths and dominates tight/non-redundant tasks by finding the complete cooperative set.
