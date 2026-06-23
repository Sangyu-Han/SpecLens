# claude_necessity_cost — cheap necessity & "the model already knows necessity" (LLM + VISION)

> Consolidated handoff for the NECESSITY cost line: why necessity is expensive, what cost-reductions
> fail vs work, and the validated cross-modal principle "the model's hidden state already encodes the
> necessity/redundancy structure." Companion to `claude_llm_fri.md` (insertion/sufficiency) and the
> memory note `project_necessity_two_pillar.md`.

---

## 1. The problem — why necessity is expensive (and sufficiency is not)

- **Sufficiency (insertion)** = the smallest set that, kept, RECOVERS the prediction. Gradient-friendly,
  cheap: a 1-shot / restricted-Banzhaf at IG-cost (~64–512 fwd) works.
- **Necessity (deletion)** = the smallest set whose REMOVAL destroys the prediction. Needs CONDITIONAL
  deletion (remove a unit *conditioned on what is already removed*) because of **REDUNDANCY**: redundant
  backup units must ALL be removed before the prediction breaks; removing one does nothing.
- Conditional ⇒ forward-hungry: greedy oracle O(N²) (~19k fwd on N=196 patches); the cheap chunked
  variant ~982 fwd. This is a **fundamental sufficiency↔necessity cost asymmetry** (~10–40×).

## 2. Cost-reduction: what FAILS, what WORKS

**FAILS — every attempt to PREDICT the conditional from a STATIC sample collapses to the 1-shot signal:**
- surrogate (2nd-order recovery model fit from Banzhaf coalitions → predict greedy order): ≈ 1-shot Banzhaf.
- 1-pass / KL-marginal / conditional-gradient (LLM): all failed (memory).
- adaptive granularity (fine-early greedy): no gain, bottleneck is candidate COVERAGE not granularity.
- IG-cost (~32 fwd): collapses to 1-shot (≈ inflow), ~5× worse than greedy.
→ The conditional CANNOT be faked from a static/1-shot estimate; redundancy needs real sequential removal.

**WORKS:**
- Banzhaf-prior chunked conditional (`chunk_bz`, ~982 fwd): a better candidate PRIOR than single-occ
  (the cooperative Banzhaf captures redundant supporters single-occ misses), then actual chunked greedy.
- **"The model already knows necessity" — hidden-cluster group-conditional (~70–211 fwd): the main result.**

## 3. "The model already knows necessity" — validated on BOTH modalities

**Principle:** the model's LAST-HIDDEN representation already encodes the redundancy/evidence structure —
units that substitute for each other (redundant) have similar hidden reps. CLUSTER them, then do the
CONDITIONAL removal at the GROUP level (remove a whole redundant cluster at once). This RESOLVES
redundancy at ~G² cost (G clusters) instead of unit-level O(N²). The conditional is **coarsened, not
eliminated** — which is exactly why it works where static-prediction shortcuts fail.

| modality | method | signal | cost | result |
|---|---|---|---|---|
| **LLM** | module-necessity (project_module_necessity / xmodel_transfer) | last-hidden VALUE clusters → evidence module (hitting-set, super-additive removal) | ~70 batched probe | beats inflow on hidden-deletion (n100 0.252, p1e-13); **transfers across 6 architectures** (no-regret); super-additive modules are contrastive/SSL+register-specific |
| **VISION** | cluster-greedy (research_vision_cluster_necessity.py) | last-block PATCH hidden clusters (k-means G=20) → greedy remove whole clusters | ~210 fwd +1 | RAW del: laion 0.131 ≈ chunk_bz 0.137 (at ~5× lower cost), augreg2 0.062 (chunk_bz 0.039 wins); BOTH ≪ banzhaf/inflow; between greedy-gold (0.049) and chunk_bz |

**⚠ CORRECTION (research_llm_cluster_necessity.py, n=12 sentiment) — the cluster-greedy is VISION-SPECIFIC; it does NOT transfer to LLM.** Running the SAME cluster-greedy on LLM token hidden states (raw-prob input-deletion): cluster 0.096 ≈ chunk_bz 0.093 ≈ banzhaf 0.095 (clustering added NOTHING — collapsed to the 1-shot Banzhaf), and ALL are beaten by **AttnLRP 0.059**. So:
- **VISION ✓**: patch hidden clusters = SPATIAL-SIMILARITY = redundancy groups → cluster-greedy resolves redundancy cheaply.
- **LLM ✗**: token redundancy is COMBINATORIAL (a token's backup is not a hidden-similar token but the surrounding context) → hidden clusters group by SEMANTIC ROLE, not redundancy → group removal does not resolve it → cluster-greedy ≈ 1-shot, loses to AttnLRP.

So the "model knows necessity via hidden clusters" is NOT one unified method across modalities. The LLM win in the table (module-necessity) used a DIFFERENT mechanism (VALUE clusters + super-additive, the hidden-DELETION metric, vs INFLOW) — and on the raw-prob input-deletion metric LLM necessity is best served by AttnLRP, not perturbation clustering. **Honest scope: the cluster-greedy cheap necessity is a VISION result; the LLM "model knows necessity" is the separate module-necessity result in its own (hdel/value) framework. The shared PRINCIPLE ("the model's hidden encodes some necessity structure") holds loosely, but the MECHANISM and the winning method are modality-specific.**

## 4. Why this works where the surrogate FAILED

- surrogate = tries to PREDICT the conditional from a static coalition sample → loses the conditioning →
  collapses to the 1-shot marginal.
- cluster-greedy / module-necessity = still does the REAL conditional (actual group removals = real
  forwards); it only COARSENS the granularity using the model's hidden redundancy grouping. So the
  conditional's redundancy-resolving power is kept, its cost drops ~5–10×.

## 5. Honest floor & limits

- **Necessity cost floor ≈ G²-cost cluster-conditional** (~70 LLM / ~210 vision), competitive with the
  ~982 unit-level chunked conditional; both far below the ~19k greedy oracle.
- **NOT IG-cost (~32).** The conditional is required (redundancy's sequential nature); only its
  GRANULARITY is reducible (via the model's clusters), not the conditional itself.
- Vision cluster-greedy is COMPETITIVE not strictly-better than chunk_bz (augreg2 chunk_bz wins); n=5
  modest; tuning (G↑, within-cluster refinement) is open.
- Metric note: use RAW-prob AUC ∈ [0,1] (NOT `p_ins/p_full`, which exceeds 1 on distractor removal).

## 6. One-line

Necessity is irreducibly conditional (redundancy needs sequential removal), so it cannot be made
IG-cheap like sufficiency — BUT the model's own last-hidden representation already groups the redundant
units, and removing those GROUPS conditionally gives a cheap (~70–210 fwd) necessity that is competitive
with the ~982 unit-level conditional and beats attention baselines (inflow) — but ONLY on VISION (patch
hidden = spatial-redundancy clusters). On LLM the cluster-greedy FAILS (token redundancy is
combinatorial; it collapses to the 1-shot Banzhaf and loses to AttnLRP) — LLM necessity is AttnLRP's on
the input-deletion metric, and the prior LLM "module-necessity" win was a separate mechanism (value
clusters + super-additive, hdel metric). The cheap-necessity-from-hidden-clusters result is VISION-SPECIFIC.
