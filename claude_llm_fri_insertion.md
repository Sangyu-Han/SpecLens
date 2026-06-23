# claude_llm_fri_insertion — Insertion / SUFFICIENCY: conclusions, experiments, validation

> Focused handoff for the INSERTION (sufficiency) side of the FRI-vs-AttnLRP work. Companion to
> `claude_llm_fri.md` (full) and `project_necessity_two_pillar.md` (memory). Necessity/deletion is a
> separate story (see those). This file = conclusions + experiments + validation methods for insertion.

---

## 1. CONCLUSIONS (what is true about insertion)

**FRI (cooperative restricted-range Banzhaf) BEATS AttnLRP on insertion across regimes — at solid M.**

| regime | FRI variant | FRI ins-AUC | AttnLRP | win-rate | significance |
|---|---|---|---|---|---|
| short/medium classification (IMDB+SST2+AG News, n=18) | adaptive restricted-Banzhaf | 0.601 | 0.390 | 78% | +0.21 ≈ 8 stderr |
| long, T 430–637 (n=5) | gradient-guided Banzhaf | 0.620 | 0.541 | 80% | small-n |
| tight: KV-retrieval (n=6) | grad-guided / adaptive | 0.570 | 0.195 | 100% | AttnLRP FAILS |
| tight: arithmetic (n=6) | adaptive restricted-Banzhaf | 0.583 | 0.465 | 83% | — |
| tight ALL (n=12) | best FRI | 0.547 | 0.330 | 92% | — |
| cross-model Llama-3.2-1B (tight, n=30) | SELECT, M=1000 | 0.463 | 0.228 | 93% | **+0.235, +7.6σ** |
| cross-model gemma-2-2b (tight, n=30) | SELECT | 0.696 | N/A | — | AttnLRP can't run (lxt no gemma2) |

Key qualifiers (HONEST):
- **M is the lever.** At solid M (1000+) on tight cases FRI DOMINATES AttnLRP (Llama +7.6σ). At the
  cheap M=80 (~89-fwd budget) it only TIES AttnLRP (n=34: +0.039±0.037 = +1.1σ, within noise) — the
  n=14 "+0.15 win" was small-n optimism, corrected.
- **vs single_occ** (the O(T) perturbation baseline): FRI wins decisively on cooperative tasks (arith
  +0.25, long +0.33) but ties on short single-token KV (single_occ is cheap+good there).
- **|grad| is task-dependent**: good on long copy (0.592, beats AttnLRP at 1-bwd cost) but POOR on
  short classification (0.217) and tight (0.301). FRI(Banzhaf) is the robust choice.
- **The cooperative-set insight**: AttnLRP nails individual high-relevance tokens but FAILS the
  cooperative SET for insertion (KV 0.195) — its relevance is diffuse; masking finds the complete
  set-that-must-all-be-present. This is FRI's core value.
- **Redundancy dichotomy (master variable)**: tight/non-redundant → FRI dominates; redundant
  (sentiment) → FRI still wins insertion (78%) but margin smaller.

**One-line:** the best LLM insertion attribution is a gradient-FREE, length-adaptive restricted-range
Banzhaf on actual recoveries (+ |grad|-guided candidates for long, + closure + self-selection); it
beats AttnLRP at solid M and is model-agnostic (works where AttnLRP's rules can't, e.g. gemma2, vision
hidden-ERF).

---

## 2. THE METHOD (validated recipe)

1. **Score = restricted-range Banzhaf** (gradient-free): `score_i = E[rec|i∈S] − E[rec|i∉S]` over M
   coalitions. `rec(S) = (softmax_prob(answer|masked to S) − base)/(full − base)`; masking = token
   embedding → global vocab-mean; keep BOS + readout always.
2. **Restricted-range sampling** (handles sequence-OOD): keep-frac `p ~ U(lo,1)`, `z_i~Bern(p)`;
   **length-adaptive `lo = 0 if T<100 else 0.5`**.
3. **Gradient-guided focusing (long/large-T)**: `|grad|` (1 bwd) → top-K candidates; vary ONLY
   candidates, hold non-candidates REAL (in-distribution + variance reduction).
4. **Tokenizer-local closure** (radius-1 max-diffusion) + **self-selection** portfolio (Codex-borrowed;
   help single-token rankers / hedge; closure is INSERTION-only — it HURTS deletion).
- Cheap budget (~89 fwd + 1 bwd): full-grad candidates → Banzhaf M=80 → local-closure → self-selection.

---

## 3. EXPERIMENTS (scripts + what each established)

| script | established |
|---|---|
| `research_insertion_datasets.py` | multi-dataset insertion; FRI 0.601 vs AttnLRP 0.390 (78%, n=18) |
| `research_long_diagnostic.py` | long failure = estimation VARIANCE, not sequence-OOD (entropy moderate; M-trend rising) |
| `research_grad_guided_long.py` | gradient-guided Banzhaf FIXES long (0.620 vs 0.541, 80%, n=5) |
| `research_hard_cases.py` | tight cases (KV/arith): FRI 92%, KV 100% (AttnLRP fails 0.195) |
| `research_hard_closure.py` | Codex closure/self-selection; closure helps single-token methods, native in cooperative Banzhaf |
| `research_cheap_fri.py` | cost ~89 fwd+1bwd; SELECT80 = 0.500; PPD α=0.3 & non-local 2nd-order REJECTED |
| `research_xmodel_fri.py` | cross-model: Llama-3.2-1B +7.6σ/93%, gemma FRI≫single_occ (AttnLRP N/A) |
| `research_gradient_variants_llm.py` | raw |grad| magnitude informative (0.512); directional cancels (0.055) |
| `research_signed_llm.py` / `_vision.py` | sign helps vision, is noise on LLM |
| `research_attnlrp_neg.py` | AttnLRP negative relevance is an LRP-conservation artifact (supporters) |
| `research_gradient_diagnostic.py` / `_vision.py` | gradient insertion 0.858 vision vs 0.055 LLM (subset-OOD root cause) |
| `research_soft_fri_indist.py` | gradient soft-FRI REFUTED (0.155) — the gradient-free Banzhaf is why FRI works |

Infra: `research_llm_imdb_sufficiency.py` (`load_imdb`, `PROMPT_TMPL`, `precompute_attnlrp` = lxt LRP
subprocess); `research_insertion_datasets.load_cases` (imdb/sst2/ag_news, argmax-correct filtered).
Models: Qwen2.5-1.5B-Instruct, Llama-3.2-1B, gemma-2-2b; CLIP vit_base_patch16 (vision contrast).
Outputs: `outputs/insertion_datasets.json`, `outputs/hard_cases.json`, `outputs/xmodel_fri.json`,
`outputs/cheap_fri.json`.

---

## 4. VALIDATION METHODS (how the claims were checked — the rigor)

- **Metric = insertion AUC**: rank tokens by score, KEEP top-k (rest → vocab-mean), measure
  `rec = (prob_answer − base)/(full − base)`, AUC over keep-fraction grid
  `FR = [0,.02,.05,.1,.15,.2,.3,.5]` (weighted to small-k). HIGHER = better sufficiency. Global
  vocab-mean masking baseline; BOS+readout always kept.
- **Baselines**: AttnLRP (lxt/efficient, bf16 subprocess — the SOTA transformer attribution), single_occ
  (O(T) occlusion), |grad| magnitude. Filtered to Qwen/model-argmax-correct cases only.
- **Statistical honesty (the discipline that caught errors)**:
  - paired per-case diff ± stderr → report in σ (e.g. Llama +0.235±0.031 = +7.6σ) and win-rate.
  - per-case insertion-AUC seed/run std ≈ 0.1 → "wins" within that margin are NOISE; re-ran fresh to
    check. CAUGHT: the restricted-Banzhaf "win" was favorable noise (M-sweep flipped it); the cheap
    SELECT80 n=14 "+0.15" regressed to +0.04 (tie) at n=34. Always re-ran at larger n.
  - corrected several artifacts: input_LN "beats AttnLRP" was a non-comparable metric; "gradient
    uninformative +0.002" was a PROJECTION artifact (raw |grad| is 0.512); "complementary" hand-wave
    rebuked → verified AttnLRP's negative is an artifact concretely.
- **Coverage of the claim**: by-dataset (IMDB/SST2/AG News), by-length bins (short/medium/long),
  cross-model (Qwen→Llama→gemma), cost-swept (M=80…2500), and against the vision contrast (gradient
  works on vision 0.858 / fails LLM 0.055 — the subset-OOD root cause). Diagnosis→fix→re-validate
  chain for the long failure (variance not OOD → gradient-guided → confirmed +7.6σ cross-model).
- **Cost accounting**: forwards + backwards counted explicitly (SELECT80 ≈ 89 fwd + 1 bwd; single_occ
  = T fwd; AttnLRP = 1 bwd). The ~100-budget claim is measured, not estimated.

---

## 5. HONEST OPEN ITEMS (insertion)

- Long n=5 (modest); the long-bin win is real but small-n — scale to n≥20 to firm.
- At the cheap ~100-fwd budget FRI only TIES AttnLRP (the decisive win needs M≥1000); AttnLRP is far
  cheaper (1 bwd). FRI's edge over AttnLRP is solid-M quality + model-agnosticism, not cost.
- Meta-Llama-3-8B insertion still pending (env: GPU co-tenant / incomplete-download).
