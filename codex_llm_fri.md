# Codex LLM FRI: Mechanism and Empirical Summary

## 0. Why This Exists

This note is a handoff for a future session. It explains not only what worked,
but why we tried it, what failed, and what problem the current mechanism solves.

The broader research program is about finding necessity/sufficiency sets cheaply
and generally. In vision, soft/random-budget FRI works well for insertion and
hidden-FRI works well for ViT necessity because the hidden token still maps
cleanly back to its own input patch. In LLMs, that self-patch assumption breaks:
content is copied forward into the readout position very early, so late hidden
positions no longer correspond to their original input tokens.

For LLM insertion, the expectation was stronger: FRI should beat attribution
methods like attention-LRP or full single-token occlusion, especially at long
sequence lengths. The reason is simple: insertion is a cooperative set problem.
If the model needs several tokens together, ranking by one-token effects should
be a weak proxy, while FRI should be able to discover the set directly.

The problem was that naive soft FRI did not reliably work in LLMs.

Main suspected causes:

- interpolation can leave the input distribution even if each replacement vector
  is individually plausible;
- text inputs are discrete and compositional, so partial soft masks can create
  unnatural intermediate sequences;
- long contexts make the gradient signal saturate or diffuse;
- tokenization splits meaningful units into fragments, so the model may need
  `1` and `7` together to represent `17`, or `bas` and `alt` together to
  represent `basalt`;
- some tasks need non-local cooperative evidence, not just one salient token.

This work therefore moved from "make soft masks work directly" to a hard-manifold
FRI style: keep inputs as real token IDs from either the original sequence or a
hard-ID null sequence, and use gradients only to cheaply narrow the search space.

## 0.1 The Concrete Problem Solved Here

The concrete problem was:

> Can we find LLM insertion/sufficiency sets on hard cooperative cases with much
> fewer forward passes than full single-token occlusion?

Full single-occ is a strong baseline but costs `O(T)` forward passes. At 400-500
tokens, that is already expensive, and it still only measures one-token removal
effects. We wanted something closer to `~100` forward passes, ideally less, that
still handles cooperative sets.

The final mechanism solves a narrower but important version of this:

- It uses PPD gradient as a cheap candidate generator.
- It performs causal tests only on the top candidate set.
- It repairs missing tokenizer fragments with local closure.
- It self-validates a small portfolio of cheap rankings by hard insertion.

On the hard cooperative benchmark, this beats full single-occ on average while
using far fewer forward masks.

## 0.2 Key Inspiration

The main inspiration came from three observations during the experiments.

First, gradients were not useless. Even when plain gradient ranking performed
badly, the correct answer/source tokens were often somewhere in its top
candidates. This suggested that gradient should not be the final attribution
score, but it could still be a search-space reducer.

Second, full single-occ was strong because it is causal, not because it scans
every token. If the candidate set is already small, we can get much of its
benefit by running occlusion only inside the candidate set. That gives a cheap
causal verifier.

Third, many "cooperative failure" cases were actually tokenizer-fragment
failures. In arithmetic, the model sees `17` as separate tokens `1` and `7`.
PPD/candidate-occ often found `1` or `2`, but missed the adjacent fragment.
Radius-1 closure fixed this without using arithmetic-specific rules.

The final method is therefore not a fancy continuous relaxation. It is a compact
hard-manifold pipeline:

```text
gradient finds candidates
causal micro-tests rank them
tokenizer closure repairs fragments
hard insertion self-selection chooses the best cheap order
```

## 1. Goal

LLM insertion에서 FRI가 single-token occlusion보다 싸고 강한 sufficiency set을 찾을 수 있는지 검증했다.

핵심 목표는 다음이었다.

- 긴 sequence에서도 cost를 대략 100 forward 이하로 유지한다.
- 단순 copy뿐 아니라 contextual copy, lookup, arithmetic처럼 cooperative set이 필요한 케이스에서도 작동한다.
- 모델별 attention rule이나 task-specific heuristic 없이, hard-manifold input operator 위에서 일반적인 방법을 찾는다.

## 2. Best Mechanism Found

현재 가장 잘 작동한 구조는 `compact candidate-closure FRI`다.

### Step 1. PPD prior

원본 input과 hard-ID null input 사이의 prediction-preserving interpolation point에서 gradient를 읽는다.

- 너무 full에 가까우면 saturation 때문에 source token이 잘 안 보인다.
- 너무 null에 가까우면 prediction이 무너진다.
- 실험상 `rho ~= 0.3` 근처의 낮은 alpha가 후보 생성에 좋았다.

PPD gradient의 역할은 최종 attribution이 아니라 **candidate generator**다.

### Step 2. Candidate causal occlusion

PPD top-M 후보, 보통 `M=32`, 안에서만 hard input occlusion을 수행한다.

- full single-occ cost: `T` forward
- candidate-occ cost: `M ~= 32` forward

이 단계가 gradient의 noisy ranking을 causal signal로 재정렬한다.

### Step 3. Tokenizer-local closure

LLM input은 BPE/tokenizer 때문에 source가 여러 token으로 쪼개진다.

예:

- `17 -> "1", "7"`
- `26 -> "2", "6"`
- `basalt -> "bas", "alt"`

PPD/candidate-occ는 핵심 조각 하나는 잡지만 인접 조각을 놓치는 경우가 많았다.

그래서 radius-1 local closure를 적용했다. 이는 task rule이 아니라 tokenizer closure다. sparse candidate score에서 이웃 token으로 점수를 확산해 missing BPE fragments를 복구한다.

이 closure가 hard arithmetic에서 결정적이었다.

### Step 4. Compact self-selection

여러 cheap order를 만들고, 각 order의 hard insertion curve를 소수 forward로 직접 검증해서 가장 좋은 order를 고른다.

사용 후보:

- `ppd_grad`
- `ppd_local_closure`
- `ppd_cand_occ`
- `ppd_cand_occ_mix`
- `ppd_cand_occ_closure`

선택 비용은 후보 수 x insertion curve point 수다. 현재 설정에서는 평균 total method cost가 약 74 forward였다.

## 3. Main Result: Hard Cooperative Cases

새로 만든 hard cooperative benchmark:

- `lookup2`: winning badge -> key mapping
- `ctxcopy`: person -> token color -> password
- `latercode`: later April entry -> code
- `add2`: 17 + 26 with distractor
- `sub2`: 58 - 19 with distractor

각각 length 128/256 근처에서 평가했다.

Result file:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_hardcoop5_L128_256_compact_select.json`

Aggregate:

| method | ins20 | raw prob ins20 | primary recall20 | mean cost |
|---|---:|---:|---:|---:|
| `compact_select` | 0.707 | 0.417 | 0.768 | 73.8 |
| `ppd_cand_occ_closure` | 0.634 | 0.375 | 0.795 | 32.0 |
| `ppd_cand_occ` | 0.450 | 0.229 | 0.635 | 32.0 |
| `ppd_local_closure` | 0.336 | 0.233 | 0.786 | 0.0 |
| `single_occ` | 0.533 | 0.332 | 0.808 | 191.4 |

`compact_select`는 10 cases 중 8승 2패였다.

| case | selected method | compact ins20 | single_occ ins20 | diff | cost |
|---|---|---:|---:|---:|---:|
| `lookup2 L128` | `ppd_cand_occ` | 1.468 | 1.381 | +0.088 | 77 |
| `lookup2 L256` | `ppd_cand_occ_closure` | 2.553 | 1.572 | +0.980 | 77 |
| `ctxcopy L128` | `ppd_cand_occ_closure` | 0.280 | 0.122 | +0.158 | 77 |
| `ctxcopy L256` | `ppd_cand_occ_closure` | 0.090 | 0.197 | -0.108 | 77 |
| `latercode L128` | `ppd_local_closure` | 0.465 | 0.395 | +0.071 | 45 |
| `latercode L256` | `ppd_cand_occ` | 0.677 | 0.692 | -0.016 | 77 |
| `add2 L128` | `ppd_cand_occ_closure` | 0.337 | 0.230 | +0.107 | 77 |
| `add2 L256` | `ppd_cand_occ_closure` | 0.760 | 0.526 | +0.234 | 77 |
| `sub2 L128` | `ppd_cand_occ_closure` | 0.415 | 0.231 | +0.184 | 77 |
| `sub2 L256` | `ppd_cand_occ_closure` | 0.025 | -0.013 | +0.038 | 77 |

## 4. Earlier Synthetic Sweep

Synthetic simple copy sweep:

- `name`
- `number0`
- `word`
- lengths 64/128/256/448

Result file:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_synth3_len4_rho03_pair98.json`

Aggregate:

| method | ins20 | mean cost |
|---|---:|---:|
| `ppd_pair_span_max` | 0.768 | 94.5 |
| `ppd_pair_span_mix` | 0.763 | 94.5 |
| `single_occ` | 0.764 | 223.3 |
| `ppd_token_fri` | 0.605 | 64.0 |

This was only a weak average win. Case-wise, the fixed pair-span method was 6 wins / 6 losses against single-occ.

The failure mode was clear:

- `name` and `number` often benefited from cooperative phrase/span recovery.
- `word` was a sharp single-token copy task, where single-occ remained very strong.

## 5. Long-Context Check

Long-context questions, 230-459 tokens:

- Q1 copy 8320
- Q2 retrieve Hillary
- Q3 year 53
- Q4 Amazon/Atlantic

Result files:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_longctx_Q1Q3_rho03_pair98_rawprob.json`
- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_longctx_Q2Q4_rho03_pair98_rawprob.json`

Raw probability aggregate over 4 questions:

| method | raw prob ins20 | mean cost |
|---|---:|---:|
| `ppd_pair_span_mix` | 0.968 | 99.2 |
| `ppd_pair_span_max` | 0.964 | 99.2 |
| `ppd_token_fri` | 0.961 | 64.0 |
| `single_occ` | 0.934 | 399.8 |

Important caveat:

Some long-context null baselines already preserved the answer with very high probability (`p_base ~= 0.98-0.999`). In those cases normalized recovery AUC can exceed 1 or become unstable. For long-context, raw probability AUC is more interpretable.

## 6. Negative Results

The following did not solve the hard cooperative cases.

### Plain PPD gradient

PPD gradient often put some source tokens in the candidate set, but the ranking was too noisy for insertion.

Hardcoop aggregate:

| method | ins20 | cost |
|---|---:|---:|
| `ppd_grad` | 0.260 | 0.0 |
| `single_occ` | 0.533 | 191.4 |

### Full-gradient

Gradient at alpha 1, i.e. near the original full input, was worse for arithmetic. It suffered from saturation and did not recover the operand set.

### Wide random CS

Wide random masks over many densities were tested to observe rare conjunctions.

Hard arithmetic result:

| method | ins20 | cost |
|---|---:|---:|
| `wide_cs_p0_rec` | 0.005 | 128 |
| `single_occ` | 0.243 | 191 |

This failed. Broad random covariance did not reliably isolate the cooperative operand set.

### Plain rb-cs

Random-budget compressed sensing worked on some multi-token code cases, but failed on arithmetic and was not robust.

### Pair-span only

Contiguous pair-span helped phrase closure, but it was not reliable for distant operands or selector-value pairs.

## 7. Mechanistic Interpretation

The working mechanism is not "soft FRI is magically better." The useful decomposition is:

1. **Gradient narrows the search space.**
   It has signal, but not enough to trust the ranking.

2. **Causal micro-tests repair the ranking.**
   Candidate-only occlusion gives much of single-occ's causal signal at much lower cost.

3. **Tokenizer closure repairs fragmented evidence.**
   Many apparent cooperative failures were actually BPE-fragment failures.

4. **Self-selection avoids single failure modes.**
   Different cheap scores fail on different cases. A small hard insertion validation step selects a good score without task-specific rules.

Compact form:

```text
PPD prior -> top-M candidates -> candidate hard occlusion -> local tokenizer closure -> compact insertion self-selection
```

## 8. Current Best Recipe

Use:

```text
--rho 0.3
--candidate-occ
--candidate-occ-top-m 32
--local-closure
--closure-radius 1
--closure-decay 0.95
--compact-select
```

Cost estimate:

- PPD gradients: backward passes, not counted as forward masks in the same way.
- Candidate occlusion: about 32 forward masks.
- Compact selection: about 45 forward masks for 5 candidates x 9 curve points.
- Total method cost: about 74 forward masks on average.

## 9. Open Weak Spots

Still weak:

- `ctxcopy L256`
- `latercode L256`

These likely require better non-local closure, not merely local BPE closure. Candidate-occ finds the answer token, but the insertion curve needs more of the selector/value context in the right order.

Next directions:

- non-local closure from candidate co-occurrence, but not simple covariance;
- short conditional group validation over PPD top candidates;
- compact portfolio including pair-span only when self-validation says it helps;
- better null/base handling for long-context cases where `p_base` already contains the answer.

## 10. One-Line Summary

The strongest LLM FRI mechanism found so far is **gradient for candidate generation, hard causal micro-tests for ranking, tokenizer-local closure for fragmented evidence, and compact self-validation for robustness**. This beats full single-occ on the hard cooperative benchmark while using far fewer forward masks.

## 11. What A New Session Should Do Next

Start from the current best command pattern:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,256 \
  --keep-mode query \
  --rho 0.3 \
  --single-baselines \
  --no-token-fri \
  --no-rb-cs \
  --candidate-occ \
  --candidate-occ-top-m 32 \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --compact-select
```

The next research question is not whether local closure works; it does. The next
question is how to handle the two remaining weak cases:

- `ctxcopy L256`
- `latercode L256`

Likely missing ingredient:

- a non-local closure mechanism that can connect selector/value context without
  task-specific rules;
- a compact pair/span validator included in `compact_select`, but only selected
  when hard insertion self-validation says it helps;
- a better candidate expansion mechanism than radius-1 closure, possibly based
  on conditional insertion gain rather than raw adjacency.

Avoid repeating these dead ends unless there is a new reason:

- full-gradient as final attribution;
- plain wide random CS over broad mask densities;
- plain rb-cs without candidate closure;
- pure pair-span as a universal solution.

The strongest current hypothesis is:

> For LLM insertion, the cheap universal path is not a single attribution score.
> It is a small hard-manifold diagnostic pipeline that combines weak signals:
> gradient candidates, causal local verification, tokenizer closure, and
> self-validated selection.

## 12. Claude-Inspired Mechanisms Tested Afterward

Another agent's handoff, `claude_llm_fri.md`, proposed a different best
mechanism:

```text
gradient-free restricted-range Banzhaf
+ |grad|-guided candidate focusing for long sequences
```

The idea is:

- sample many hard insertion coalitions;
- score token `i` by Banzhaf-style difference of conditional means:

```text
score_i = E[recovery(S) | i in S] - E[recovery(S) | i not in S]
```

- for long sequences, vary only top-|grad| candidate tokens and keep all
  non-candidates real, reducing variance.

This was novel relative to the Codex mechanism. The Codex method used gradient
as a candidate generator, then used causal micro-tests and tokenizer closure.
Claude's method uses actual recovery samples directly and tries to estimate a
set marginal.

I reimplemented the Claude mechanisms inside the Codex benchmark, but on the
same **hard-ID null/original operator** used by the Codex experiments, not the
mean-embedding operator from Claude's scripts.

New options added to `scripts/llm_ppd_insertion_benchmark.py`:

```text
--absgrad
--banzhaf
--banzhaf-guided
--banzhaf-samples
--banzhaf-candidates
--banzhaf-lo
```

### 12.1 Hardcoop Result: Banzhaf M=256

Result file:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_hardcoop5_banzhaf256.json`

Aggregate on hardcoop 10 cases:

| method | ins20 | cost |
|---|---:|---:|
| `guided_banzhaf` | 0.739 | 256 |
| `restricted_banzhaf` | 0.452 | 256 |
| `absgrad` | 0.366 | 0 |
| `single_occ` | 0.533 | 191 |
| previous `compact_select` | 0.707 | 74 |

Key observation:

- `guided_banzhaf` is stronger than `compact_select` on average, but much more
  expensive.
- `restricted_banzhaf` without candidate focusing is weak.
- `absgrad` alone is not robust, but sometimes surprisingly strong.

### 12.2 Where Guided Banzhaf Helped Most

It directly fixed two weak cases from the original Codex method.

| case | compact/select or previous best | guided Banzhaf M=256 | single_occ |
|---|---:|---:|---:|
| `ctxcopy L256` | 0.090 | **0.831** | 0.197 |
| `latercode L256` | 0.677 | **0.876** | 0.692 |
| `ctxcopy L128` | 0.280 | **0.601** | 0.122 |
| `latercode L128` | 0.465 | **0.642** | 0.395 |

Mechanistic interpretation:

- Codex candidate closure is excellent for local tokenizer fragments and
  arithmetic BPE closure.
- Guided Banzhaf is excellent when the missing ingredient is **non-local
  selector/value cooperation**.
- Holding non-candidates real makes long context masks much more coherent.
  This matters for `ctxcopy` and `latercode`, where the answer token alone is
  not enough for a strong insertion curve.

### 12.3 Where Guided Banzhaf Did Not Help

Arithmetic did not consistently improve.

| case | Codex `cand_occ_closure` | guided Banzhaf M=256 | single_occ |
|---|---:|---:|---:|
| `add2 L128` | **0.337** | 0.203 | 0.230 |
| `add2 L256` | **0.760** | 0.131 | 0.526 |
| `sub2 L128` | **0.415** | poor | 0.231 |

For arithmetic, the decisive issue was still tokenizer/local fragment closure
over operand pieces, not broad Banzhaf sampling.

### 12.4 Union Portfolio Upper Bound

If we take the best of:

- Codex `compact_select`;
- Codex candidate closure variants;
- Claude-style `guided_banzhaf`;
- `restricted_banzhaf`;
- `absgrad`;

then hardcoop 10-case performance becomes:

| portfolio | mean ins20 | win/loss vs single_occ | mean selected base cost |
|---|---:|---:|---:|
| Codex compact only | 0.707 | 8 / 2 | 73.8 |
| guided Banzhaf only | 0.739 | mixed | 256 |
| union best | **0.902** | **10 / 0** | 166.5 |
| single_occ | 0.533 | - | 191.4 |

This is only an upper bound unless the selection rule is included in the actual
method, but it shows the two methods are complementary.

### 12.5 Cost Sweep On Weak Cases

I also tested cheaper guided Banzhaf on the three weak L256 cases:

Result files:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_hard_weak_guided_banzhaf64.json`
- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_hard_weak_guided_banzhaf128.json`

M=64:

| method | mean ins20 on weak 3 | cost |
|---|---:|---:|
| `guided_banzhaf` | 0.371 | 64 |
| `absgrad` | 0.414 | 0 |
| `single_occ` | 0.292 | 255 |

M=128:

| method | mean ins20 on weak 3 | cost |
|---|---:|---:|
| `guided_banzhaf` | 0.451 | 128 |
| `absgrad` | 0.414 | 0 |
| `single_occ` | 0.292 | 255 |

Case detail:

- `ctxcopy L256`: guided Banzhaf already works at M=64/128.
- `sub2 L256`: M=128 helps.
- `latercode L256`: needs M=256; M=64/128 were not enough.

### 12.6 What To Extract From Claude

Keep these:

1. **Guided Banzhaf as an expensive rescue path.**
   It fixes non-local selector/value cases that local closure misses.

2. **`|grad|` as a candidate focusing signal.**
   Not reliable as final attribution, but useful for choosing which tokens to
   vary while holding the rest real.

3. **High-keep / restricted-range sampling.**
   Varying only part of the sequence while keeping most tokens real improves
   input coherence.

Do not directly adopt these as the default:

1. **Plain restricted Banzhaf.**
   It was weaker than guided Banzhaf and weaker than compact candidate closure.

2. **Large M as the main method.**
   M=256 already costs more than the current compact method; Claude's M=1200+
   would violate the `~100 forward` target.

3. **Gradient-free-only framing.**
   In the Codex hardcoop benchmark, gradient was useful as a candidate generator.
   The better framing is not "gradient-free vs gradient", but:

```text
cheap candidate signal + hard-manifold causal verification + closure/selection
```

### 12.7 Updated Best Hypothesis

The best LLM insertion method is probably a two-tier portfolio:

Tier 1, cheap default:

```text
PPD prior + |grad| -> candidate occlusion -> tokenizer closure -> compact self-select
```

Expected cost after corrected accounting: about `86 forward + 5 backward = 91`
model calls on the hardcoop configuration below. The 5 backward passes are 4
PPD target gradients (`prob,logit,logprob,margin`) plus 1 full-input `absgrad`
candidate signal.

Tier 2, rescue for non-local selector/value cases:

```text
|grad| candidates -> guided restricted Banzhaf with non-candidates held real
```

Expected cost: 64-256 forward masks depending on how hard the case is.

The research frontier is to make Tier 2 adaptive: run it only when Tier 1's
self-validation curve looks weak or unstable.

## 13. Cost-Control Update: `compact+absgrad`

The user's target was roughly `~100 forward/backward` calls, not a 200-300
forward method. I tested the cheapest useful Claude-inspired component:
`absgrad` as another compact selector candidate, without running Banzhaf by
default.

Command:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,256 \
  --keep-mode query \
  --rho 0.3 \
  --single-baselines \
  --no-token-fri \
  --no-rb-cs \
  --candidate-occ \
  --candidate-occ-top-m 32 \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --absgrad \
  --compact-select \
  --compact-select-methods ppd_grad,ppd_local_closure,ppd_cand_occ,ppd_cand_occ_mix,ppd_cand_occ_closure,absgrad \
  --out outputs/class_fri/research_frontier/llm_ppd_insert_bench_hardcoop5_compact_absgrad_costfix.json
```

Result:

| method | ins20 | raw prob ins20 | recall20 | forward | backward | total |
|---|---:|---:|---:|---:|---:|---:|
| `compact_select+absgrad` | **0.777** | **0.469** | 0.793 | 86.0 | 5.0 | **91.0** |
| previous compact only | 0.707 | 0.417 | 0.768 | ~77.0 | 4.0 | ~81.0 |
| adaptive guided Banzhaf M=128 | 0.725 | 0.430 | 0.768 | ~104 | 5.0 | ~109 |
| `single_occ` | 0.533 | 0.332 | 0.808 | 191.4 | 0.0 | 191.4 |

Casewise:

| case | selected method | compact+absgrad | single_occ |
|---|---|---:|---:|
| `lookup2 L128` | `ppd_cand_occ` | **1.468** | 1.381 |
| `lookup2 L256` | `ppd_cand_occ_closure` | **2.553** | 1.572 |
| `ctxcopy L128` | `ppd_cand_occ_closure` | **0.280** | 0.122 |
| `ctxcopy L256` | `absgrad` | **0.486** | 0.197 |
| `latercode L128` | `ppd_local_closure` | **0.465** | 0.395 |
| `latercode L256` | `ppd_cand_occ` | 0.677 | **0.692** |
| `add2 L128` | `ppd_cand_occ_closure` | **0.337** | 0.230 |
| `add2 L256` | `ppd_cand_occ_closure` | **0.760** | 0.526 |
| `sub2 L128` | `ppd_cand_occ_closure` | **0.415** | 0.231 |
| `sub2 L256` | `absgrad` | **0.331** | -0.013 |

Summary: this is the current best cheap default. It wins 9/10 cases against
full `single_occ`, improves mean ins20 by `+0.244`, and stays within the
`~100` call budget. The only remaining miss is `latercode L256`, where the gap
is small (`0.677` vs `0.692`).

Important cost fix: the older `compact_select` JSON undercounted cases where a
zero-forward derived method was selected, because it charged only the selected
method plus selector curves. The corrected script now counts shared candidate
construction once: here, `32` candidate-occlusion forwards plus `6 * 9 = 54`
self-insertion selector forwards, for `86` forward calls per case.

## 14. AttnLRP Comparison With Higher n And More Models

I added AttnLRP directly to `scripts/llm_ppd_insertion_benchmark.py` using the
existing batched bf16 subprocess from `research_llm_imdb_sufficiency.py`. I also
added hardcoop variants so n is not just repeated seeds of the same prompt.

Implementation changes:

- `scripts/llm_hard_coop_cases.py`: `--hard-variants` now changes names,
  colors, passwords, numeric operands, and target answers.
- `scripts/llm_ppd_insertion_benchmark.py`: `--attnlrp` and
  `--attnlrp-closure` add AttnLRP baselines under the exact same insertion
  metric.
- AttnLRP is computed before the fp32 main model is loaded, so the bf16
  AttnLRP subprocess and the main benchmark model do not occupy GPU memory at
  the same time.

### 14.1 Qwen2.5-1.5B, hardcoop n=30

Command output:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_qwen25_1p5b_hardcoop_n30_attnlrp.json`

Case set:

- 5 kinds: `lookup2,ctxcopy,latercode,add2,sub2`
- 3 lengths: `128,192,256`
- 2 variants
- hit-only n = 30 / 30

| method | ins20 | raw prob ins20 | cost |
|---|---:|---:|---:|
| `compact_select` | **0.605** | **0.437** | 86 fwd + 5 bwd = 91 |
| `attnlrp` | 0.334 | 0.283 | 1 bwd |
| `attnlrp_closure` | 0.366 | 0.309 | 1 bwd |

Paired differences:

| baseline | mean diff | SE | t | wins/losses |
|---|---:|---:|---:|---:|
| raw AttnLRP | **+0.271** | 0.084 | 3.23 | 23 / 7 |
| AttnLRP+closure | **+0.239** | 0.082 | 2.92 | 23 / 7 |

### 14.2 Qwen2.5-0.5B, ctxcopy+add2 n=34

The full five-kind hardcoop set was too hard for 0.5B: the model only solved
15/30 in the first hit check. I therefore used the solved regimes
`ctxcopy,add2`, variants=6. Two cases still missed; hit-only n = 34 / 36.

Output:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_qwen25_0p5b_ctxadd_n36_attnlrp.json`

| method | ins20 | raw prob ins20 | cost |
|---|---:|---:|---:|
| `compact_select` | **0.494** | **0.352** | 86 fwd + 5 bwd = 91 |
| `attnlrp` | 0.309 | 0.241 | 1 bwd |
| `attnlrp_closure` | 0.320 | 0.249 | 1 bwd |

Paired differences:

| baseline | mean diff | SE | t | wins/losses |
|---|---:|---:|---:|---:|
| raw AttnLRP | **+0.186** | 0.052 | 3.55 | 25 / 9 |
| AttnLRP+closure | **+0.175** | 0.053 | 3.29 | 28 / 6 |

### 14.3 Qwen2.5-3B, ctxcopy n=30

The original five-kind hardcoop set was also not a clean solved set for 3B
(12/30 hit; arithmetic often produced the wrong first token). For a fair
attribution test I used `ctxcopy` only, variants=10, 3 lengths. Hit-only n =
30 / 30.

Output:

- `outputs/class_fri/research_frontier/llm_ppd_insert_bench_qwen25_3b_ctxcopy_n30_attnlrp.json`

| method | ins20 | raw prob ins20 | cost |
|---|---:|---:|---:|
| `compact_select` | **0.477** | **0.434** | 86 fwd + 5 bwd = 91 |
| `attnlrp` | 0.187 | 0.170 | 1 bwd |
| `attnlrp_closure` | 0.230 | 0.207 | 1 bwd |

Paired differences:

| baseline | mean diff | SE | t | wins/losses |
|---|---:|---:|---:|---:|
| raw AttnLRP | **+0.290** | 0.047 | 6.23 | 26 / 4 |
| AttnLRP+closure | **+0.247** | 0.045 | 5.47 | 25 / 5 |

### 14.4 Pooled Result

Across all solved cases from the three models:

| baseline | n | mean diff | SE | t | wins/losses |
|---|---:|---:|---:|---:|---:|
| raw AttnLRP | 94 | **+0.246** | 0.036 | 6.85 | 74 / 20 |
| AttnLRP+closure | 94 | **+0.218** | 0.035 | 6.19 | 76 / 18 |

This is the strongest evidence so far that the compact hard-manifold FRI
portfolio is not merely matching AttnLRP on LLM insertion; on solved hard
cooperative cases it is clearly ahead, even against a closure-enhanced AttnLRP
baseline.

Caveat: AttnLRP remains far cheaper (`~1 backward` vs `91` calls), and the
extra-model tests used model-specific solved subsets because smaller/larger
Qwen variants did not solve the same five-kind hardcoop set reliably. That is
an attribution-evaluation constraint, not a method win/loss by itself.

### 14.5 Verification Handoff For Insertion Claims

This is the minimal checklist for another agent to verify the insertion claim.
The claim to check is:

```text
On solved hard cooperative LLM insertion cases, compact hard-manifold FRI
beats raw AttnLRP and AttnLRP+closure on mean insertion AUC, across three Qwen
model settings, while staying at about 91 model calls per case.
```

Environment:

- Python: `/home/sangyu/anaconda3/envs/py312/bin/python`
- GPU: `cuda:0`
- Main script: `scripts/llm_ppd_insertion_benchmark.py`
- Case generator: `scripts/llm_hard_coop_cases.py`
- Do not use `cuda:1`.

First sanity check:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python -m py_compile \
  scripts/llm_ppd_insertion_benchmark.py \
  scripts/llm_hard_coop_cases.py
```

Reproduction commands:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --no-token-fri \
  --no-rb-cs \
  --candidate-occ \
  --candidate-occ-top-m 32 \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --absgrad \
  --compact-select \
  --compact-select-methods ppd_grad,ppd_local_closure,ppd_cand_occ,ppd_cand_occ_mix,ppd_cand_occ_closure,absgrad \
  --out outputs/class_fri/research_frontier/verify_qwen25_1p5b_hardcoop_n30_attnlrp.json
```

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds ctxcopy,add2 \
  --hard-lengths 128,192,256 \
  --hard-variants 6 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --no-token-fri \
  --no-rb-cs \
  --candidate-occ \
  --candidate-occ-top-m 32 \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --absgrad \
  --compact-select \
  --compact-select-methods ppd_grad,ppd_local_closure,ppd_cand_occ,ppd_cand_occ_mix,ppd_cand_occ_closure,absgrad \
  --out outputs/class_fri/research_frontier/verify_qwen25_0p5b_ctxadd_n36_attnlrp.json
```

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-3B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds ctxcopy \
  --hard-lengths 128,192,256 \
  --hard-variants 10 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --no-token-fri \
  --no-rb-cs \
  --candidate-occ \
  --candidate-occ-top-m 32 \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --absgrad \
  --compact-select \
  --compact-select-methods ppd_grad,ppd_local_closure,ppd_cand_occ,ppd_cand_occ_mix,ppd_cand_occ_closure,absgrad \
  --out outputs/class_fri/research_frontier/verify_qwen25_3b_ctxcopy_n30_attnlrp.json
```

Expected hit-only counts:

| run | expected hit-only n |
|---|---:|
| Qwen2.5-1.5B hardcoop | 30 / 30 |
| Qwen2.5-0.5B ctxcopy+add2 | about 34 / 36 |
| Qwen2.5-3B ctxcopy | 30 / 30 |

Expected insertion AUC numbers from the original run:

| run | compact | raw AttnLRP | AttnLRP+closure | compact cost |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B hardcoop n=30 | 0.605 | 0.334 | 0.366 | 91 |
| Qwen2.5-0.5B ctx+add n=34 | 0.494 | 0.309 | 0.320 | 91 |
| Qwen2.5-3B ctxcopy n=30 | 0.477 | 0.187 | 0.230 | 91 |

Verifier script for paired statistics:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python - <<'PY'
import json, math, numpy as np

files = {
    "Qwen2.5-1.5B hardcoop": "outputs/class_fri/research_frontier/verify_qwen25_1p5b_hardcoop_n30_attnlrp.json",
    "Qwen2.5-0.5B ctx+add": "outputs/class_fri/research_frontier/verify_qwen25_0p5b_ctxadd_n36_attnlrp.json",
    "Qwen2.5-3B ctxcopy": "outputs/class_fri/research_frontier/verify_qwen25_3b_ctxcopy_n30_attnlrp.json",
}

all_rows = []
for label, path in files.items():
    obj = json.load(open(path))
    rows = [r for r in obj["rows"] if r.get("hit")]
    all_rows.extend(rows)
    agg = obj["aggregate_hit_only"]
    print("\n" + label, "n=", len(rows))
    for m in ["compact_select", "attnlrp", "attnlrp_closure"]:
        a = agg[m]
        print(m, "ins20=", round(a["ins20"], 3), "prob=", round(a["prob_ins20"], 3),
              "fwd+bwd=", round(a["fwd_bwd_mean"], 1))
    for base in ["attnlrp", "attnlrp_closure"]:
        d = np.array([
            r["methods"]["compact_select"]["ins_auc20"] - r["methods"][base]["ins_auc20"]
            for r in rows
        ], float)
        se = d.std(ddof=1) / math.sqrt(len(d))
        print("vs", base, "diff=", round(float(d.mean()), 3), "SE=", round(float(se), 3),
              "wins=", int((d > 0).sum()), "losses=", int((d < 0).sum()))

print("\nPOOLED")
for base in ["attnlrp", "attnlrp_closure"]:
    d = np.array([
        r["methods"]["compact_select"]["ins_auc20"] - r["methods"][base]["ins_auc20"]
        for r in all_rows
    ], float)
    se = d.std(ddof=1) / math.sqrt(len(d))
    print("vs", base, "n=", len(d), "diff=", round(float(d.mean()), 3),
          "SE=", round(float(se), 3), "t=", round(float(d.mean() / se), 2),
          "wins=", int((d > 0).sum()), "losses=", int((d < 0).sum()))
PY
```

Expected pooled result:

| baseline | n | mean diff | SE | t | wins/losses |
|---|---:|---:|---:|---:|---:|
| raw AttnLRP | about 94 | about +0.246 | about 0.036 | about 6.85 | about 74 / 20 |
| AttnLRP+closure | about 94 | about +0.218 | about 0.035 | about 6.19 | about 76 / 18 |

Pass criteria:

- `compact_select` must beat raw AttnLRP and AttnLRP+closure in every
  per-model mean insertion AUC.
- The pooled compact-vs-raw-AttnLRP paired mean difference should stay clearly
  positive; use `+0.20` as a conservative minimum.
- The compact method cost should remain about `86 forward + 5 backward = 91`
  calls per case.
- Do not count metric-evaluation masks as method cost; the JSON reports method
  cost in each method's `meta`.

Known caveats for verification:

- AttnLRP is much cheaper (`~1 backward`), so the claim is performance, not
  cost dominance.
- 0.5B and 3B use model-specific solved subsets. This is deliberate: attribution
  should be evaluated on cases where the model actually made the target
  prediction.
- The deletion/necessity results below are negative for this insertion score
  and should not be used to reject the insertion claim.

## 15. Deletion / Necessity Follow-Up

Question: does the insertion mechanism transfer to deletion/necessity?

Short answer: **partly, but not enough**. The useful transferable idea is:

```text
cheap candidate signal -> hard causal verification -> compact self-selection
```

The non-transferable part is the score itself. The insertion-selected
`compact_select` score is a sufficiency score and performs poorly as a
necessity/deletion order.

### 15.1 Existing Cross-Model Deletion Signal

From the AttnLRP comparison runs:

| run | compact del20 | AttnLRP del20 | AttnLRP+closure del20 |
|---|---:|---:|---:|
| Qwen2.5-1.5B hardcoop n=30 | 0.118 | **0.053** | 0.079 |
| Qwen2.5-0.5B ctx+add n=34 | 0.063 | **0.038** | 0.058 |
| Qwen2.5-3B ctxcopy n=30 | 0.051 | **0.050** | **0.050** |

Lower is better. The insertion portfolio should not be reused for deletion.

### 15.2 Deletion-Specific Candidate Verification

I added a deletion-specific experimental path to
`scripts/llm_ppd_insertion_benchmark.py`:

- `--necessity-occ`: build a candidate set from cheap guides and run hard
  singleton deletion only on those candidates.
- `--necessity-select`: choose the best deletion order by small hard deletion
  self-evaluation.
- `--ixg`: add unpatched input-times-gradient (`ixg`) and `abs_ixg`.

Main run:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --ixg \
  --no-token-fri \
  --no-rb-cs \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --absgrad \
  --necessity-occ \
  --necessity-occ-top-m 48 \
  --necessity-select \
  --necessity-select-methods ppd_grad,absgrad,ixg,abs_ixg,nec_cand_occ,nec_cand_occ_mix,nec_cand_occ_closure \
  --out outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_hardcoop_n30_ixg_necselect.json
```

Result:

| method | del20 | median source rank | cost |
|---|---:|---:|---:|
| AttnLRP | **0.050** | **1.0** | 1 bwd |
| `necessity_select` + ixg | 0.056 | 2.0 | 111 fwd + 7 bwd |
| AttnLRP+closure | 0.080 | **1.0** | 1 bwd |
| absgrad | 0.111 | 5.0 | 1 bwd |
| abs_ixg | 0.113 | 4.5 | 1 bwd |
| PPD grad | 0.104 | 3.0 | 4 bwd |
| insertion `compact_select` | 0.118 | 1.0 | 91 calls |
| `nec_cand_occ` | 0.837 | 103.5 | 48 fwd |

Paired against raw AttnLRP:

| method | mean del20 diff vs AttnLRP | better cases |
|---|---:|---:|
| `necessity_select` without ixg | +0.0249 | 17 / 30 |
| `necessity_select` with ixg | +0.0065 | 18 / 30 |

The ixg version is close, but it is still worse than raw AttnLRP and much more
expensive.

### 15.3 Mechanistic Findings

1. **Insertion success does not imply deletion success.**
   The insertion portfolio often ranks context/scaffold tokens that help build
   the answer. Those are sufficient but not always necessary.

2. **Sharp copy/value cases saturate deletion.**
   For `ctxcopy` and many `latercode` cases, any method that puts the copied
   value token first reaches the deletion floor (`del20 ~= 0.05`). Extra
   verification is wasted there.

3. **Arithmetic/subtraction expose the real gap.**
   AttnLRP directly ranks operand digits. PPD, absgrad, and unpatched ixg often
   drift to prompt structure (`Context`, `:`, names, "has", "tickets").

4. **Hard singleton deletion can chase scaffold artifacts.**
   `nec_cand_occ` failed badly because its top candidates became punctuation or
   filler tokens. This is not useful semantic necessity. It is the LLM analogue
   of a deletion artifact: the model is sensitive to prompt scaffolding, not the
   load-bearing content.

5. **Unpatched input-times-gradient does not explain AttnLRP.**
   `ixg` and `abs_ixg` mostly behave like absgrad and fail to reproduce
   AttnLRP's digit-first arithmetic behavior. The useful part of AttnLRP seems
   to be its attention-aware relevance propagation, not simply `embedding *
   gradient`.

### 15.4 Current Necessity Conclusion

For LLM deletion/necessity, the best current cheap baseline remains raw
AttnLRP or plain sharp input occlusion, depending on budget and whether the
model architecture is supported by AttnLRP.

The insertion FRI work still gives a useful research direction:

```text
do not optimize deletion prob-drop blindly;
first separate content necessity from scaffold sensitivity,
then run causal verification only on content-like candidates.
```

The next promising direction is not more Banzhaf/FRI over arbitrary tokens. It
is a **content-preserving necessity candidate generator**: one that biases
toward copied values, operands, entities, and answer-bearing spans while
excluding prompt-format scaffolding without hand-coded task rules. A possible
route is to use model-internal copy/value transport signals, not local gradients.

### 15.5 Direct Random-Budget Deletion FRI

I then tested whether the random-budget FRI machinery itself could be turned
from sufficiency into necessity by changing the objective from insertion to
deletion.

Implementation added to `scripts/llm_ppd_insertion_benchmark.py`:

```text
--token-fri-objectives delete,both
--cs-objectives ...
```

The underlying `llm_clean_coalition_fri.py` already supported
`objective="delete"` and `objective="both"`; the benchmark simply did not expose
those objectives for hardcoop.

Main run:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --single-baselines \
  --no-rb-cs \
  --token-fri \
  --token-fri-objectives delete,both \
  --rb-steps 8 \
  --rb-samples 8 \
  --local-closure \
  --closure-radius 1 \
  --closure-decay 0.95 \
  --absgrad \
  --ixg \
  --necessity-select \
  --necessity-select-methods ppd_grad,absgrad,ixg,abs_ixg,ppd_token_fri_delete,ppd_token_fri_both,attnlrp,attnlrp_closure,single_occ \
  --out outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_hardcoop_n30_directdel_singleocc.json
```

Aggregate, hit-only n=30:

| method | del20 | ins20 | median source rank | recall20 | cost |
|---|---:|---:|---:|---:|---:|
| AttnLRP | 0.0529 | 0.338 | 1.0 | 0.771 | 1 bwd |
| single_occ | **0.0521** | **0.483** | **1.0** | **0.873** | 190.6 fwd |
| `ppd_token_fri_delete` | 0.0553 | 0.133 | 2.0 | 0.610 | 64 fwd |
| `ppd_token_fri_both` | 0.0889 | 0.273 | 2.0 | 0.563 | 64 fwd |
| PPD grad | 0.1044 | 0.234 | 3.0 | 0.702 | 4 bwd |
| absgrad | 0.1110 | 0.335 | 5.0 | 0.841 | 1 bwd |
| `necessity_select` | **0.0329** | 0.293 | 1.0 | 0.669 | 408.6 calls |

Interpretation:

- Direct deletion-FRI almost matches AttnLRP on average del20, but it has worse
  source rank and source recall.
- The mean is helped by saturated sharp-copy cases where many methods reach
  `del20 ~= 0.05`.
- On arithmetic/subtraction, deletion-FRI often picks prompt/scaffold tokens or
  only one operand fragment. It is not a semantic necessary-set method yet.
- `necessity_select` is a useful oracle-ish diagnostic, but too expensive and
  partially self-evaluates on the deletion metric, so it is not the desired
  cheap method.

Kind breakdown exposed the failure:

| kind | AttnLRP del20 | single_occ del20 | delete-FRI del20 | delete-FRI rank |
|---|---:|---:|---:|---:|
| lookup2 | 0.0490 | 0.0489 | 0.0526 | 2.0 |
| ctxcopy | 0.0500 | 0.0500 | 0.0500 | 2.5 |
| latercode | 0.0500 | 0.0500 | 0.0502 | 1.0 |
| add2 | 0.0443 | 0.0230 | **0.0188** | 4.0 |
| sub2 | **0.0712** | 0.0884 | 0.1049 | 3.0 |

The add2 deletion AUC looks good for delete-FRI, but rank is poor. This is the
same artifact again: low deletion AUC can come from breaking scaffold/context,
not from ranking the intended source set.

### 15.6 Cheap Conditional Deletion Greedy

The stronger necessity-inspired idea was:

```text
cheap prior -> top-M candidates -> set-conditional deletion greedy
```

This differs from candidate singleton deletion. At each round, it evaluates
which candidate most lowers target recovery **given the tokens already deleted**.
With `top_m=32` and `rounds=3`, the cost is:

```text
32 + 31 + 30 = 93 forward masks
```

plus the guide cost, e.g. AttnLRP adds 1 backward. This is inside the user's
rough `~100` call target.

Implementation added:

```text
--conditional-greedy
--conditional-guides attnlrp,attnlrp_closure,ppd_grad
--conditional-top-m 32
--conditional-rounds 3
```

Main run:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --no-token-fri \
  --no-rb-cs \
  --conditional-greedy \
  --conditional-guides attnlrp,attnlrp_closure,ppd_grad \
  --conditional-top-m 32 \
  --conditional-rounds 3 \
  --out outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_hardcoop_n30_condgreedy_attn_ppd.json
```

Aggregate, hit-only n=30:

| method | del20 | ins20 | median source rank | recall20 | cost |
|---|---:|---:|---:|---:|---:|
| AttnLRP | 0.0527 | 0.334 | 1.0 | 0.764 | 1 bwd |
| AttnLRP+closure | 0.0723 | 0.357 | 1.0 | 0.818 | 1 bwd |
| `cond_greedy_attnlrp` | 0.0451 | 0.137 | 1.0 | 0.651 | 93 fwd + 1 bwd |
| `cond_greedy_attnlrp_closure` | **0.0342** | 0.187 | 1.0 | 0.689 | 93 fwd + 1 bwd |
| `cond_greedy_ppd_grad` | 0.0544 | 0.190 | 1.0 | 0.567 | 93 fwd + 4 bwd |
| `cond_greedy_fusion` | 0.9045 | 0.000 | 102.5 | 0.000 | bad |

Paired against raw AttnLRP:

| method | mean del20 diff | SE | better cases |
|---|---:|---:|---:|
| `cond_greedy_attnlrp` | -0.0076 | 0.0158 | 24 / 30 |
| `cond_greedy_attnlrp_closure` | **-0.0185** | 0.0112 | 21 / 30 |
| `cond_greedy_ppd_grad` | +0.0016 | 0.0214 | 22 / 30 |

The result is real but mixed:

- Conditional greedy can improve deletion AUC within the 100-call budget.
- The biggest gains are add2 and some sub2 cases, where deleting one operand
  changes which remaining operand becomes pivotal.
- It hurts insertion, as expected; this is a necessity-only refinement.
- Source-set recall drops compared with raw AttnLRP/single-occ because the
  greedy deletion objective sometimes chooses scaffold tokens after the first
  source token.
- Naive guide fusion is disastrous. Max-fusing AttnLRP, closure, and PPD lets
  scaffold tokens enter top-M, then conditional deletion locks onto them.

Kind breakdown:

| kind | AttnLRP del20 | cond AttnLRP+closure del20 | source recall20 |
|---|---:|---:|---:|
| lookup2 | 0.0490 | 0.0488 | 0.736 |
| ctxcopy | 0.0500 | 0.0500 | 0.442 |
| latercode | 0.0500 | 0.0500 | 0.623 |
| add2 | 0.0444 | **-0.0303** | 0.850 |
| sub2 | 0.0703 | **0.0526** | 0.792 |

Important implementation note: immediately after this run, I fixed the cost
accounting so future `cond_greedy_*` rows include the guide's backward cost.
The saved JSON above reports 93 forward masks, but the real cost for
`cond_greedy_attnlrp_closure` is 93 forward + 1 backward.

### 15.7 Updated Necessity Hypothesis

Current best LLM deletion picture:

1. **If cost is almost free:** raw AttnLRP is still the best baseline. It is
   cheap and source-rank faithful on the hardcoop set.
2. **If about 100 calls are allowed:** AttnLRP-closure-guided conditional
   deletion greedy is the first method here that clearly lowers mean del20
   below raw AttnLRP (`0.034` vs `0.053`), but it sacrifices source-set recall.
3. **If cost T is allowed:** single_occ remains the cleanest causal content
   baseline (`del20 0.052`, recall20 0.873, cost 190.6 on this n=30 set).
4. **Direct deletion-FRI is not enough.** It matches del20 too closely to
   AttnLRP but loses source rank/recall, especially on arithmetic/subtraction.

The research direction should therefore be:

```text
AttnLRP/readout-transport prior
-> top-M source-like candidates
-> conditional deletion greedy
-> source-faithfulness guard, not just del-AUC selection
```

The missing piece is the guard. Deletion AUC alone is not a clean necessity
metric in LLM prompts because scaffold deletion can lower the target probability
without identifying the semantic source. A good next method needs a generic
way to suppress prompt-format/scaffold tokens while preserving copied values,
entities, operands, and answer-bearing spans. This is likely an internal
transport/readout problem, not an insertion-FRI problem.

## 16. MRI Diagnostics For Deletion

After the deletion experiments above, I added two MRI-style diagnostic scripts:

```text
scripts/llm_deletion_mri_diagnostic.py
scripts/llm_group_deletion_mri.py
```

The purpose is to stop judging methods only by deletion AUC. The diagnostics
ask:

- Did the method rank source tokens, distractors, or scaffold/filler?
- Did conditional greedy improve deletion by losing semantic source recall?
- If a distractor/scaffold token was selected, does deleting that group alone
  actually matter?

### 16.1 Offline Ranking MRI

Command:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_deletion_mri_diagnostic.py \
  outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_hardcoop_n30_condgreedy_attn_ppd.json \
  --methods attnlrp,attnlrp_closure,cond_greedy_attnlrp,cond_greedy_attnlrp_closure,cond_greedy_ppd_grad,cond_greedy_fusion,ppd_grad \
  --base attnlrp \
  --out outputs/class_fri/research_frontier/llm_deletion_mri_condgreedy.md
```

Key MRI table:

| method | del20 | recall20 | top1 source | top5 scaffold | selected source | selected scaffold |
|---|---:|---:|---:|---:|---:|---:|
| AttnLRP | 0.053 | 0.764 | **1.000** | 0.420 | - | - |
| AttnLRP+closure | 0.072 | **0.818** | **1.000** | 0.473 | - | - |
| `cond_greedy_attnlrp` | 0.045 | 0.651 | 0.933 | 0.573 | 0.344 | 0.411 |
| `cond_greedy_attnlrp_closure` | **0.034** | 0.689 | 0.933 | 0.580 | 0.333 | 0.422 |
| `cond_greedy_fusion` | 0.905 | 0.000 | 0.000 | **1.000** | 0.000 | **1.000** |
| PPD grad | 0.104 | 0.702 | 0.367 | 0.640 | - | - |

Interpretation:

- Conditional greedy improves deletion AUC, but only about one third of its
  selected tokens are source tokens.
- Roughly 40% of conditional-selected tokens are scaffold/filler.
- Fusion is a clean failure: it selects pure scaffold and should be discarded.
- Raw AttnLRP is the most reliable semantic source detector: top1 source is
  100% on this hardcoop run.

### 16.2 Prior-Weighted Conditional Greedy

Hypothesis:

```text
conditional drop is useful, but it should be regularized by guide prior
```

I added:

```text
--conditional-guide-power
--conditional-prior-floor
```

The selection rule becomes approximately:

```text
reward_i = conditional_drop_i * (prior_floor + guide_i) ** guide_power
```

Targeted arithmetic run:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --no-token-fri \
  --no-rb-cs \
  --conditional-greedy \
  --conditional-guides attnlrp,attnlrp_closure \
  --conditional-top-m 32 \
  --conditional-rounds 3 \
  --conditional-guide-power 1.0 \
  --conditional-prior-floor 0.05 \
  --out outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_arith_n12_condgreedy_priorpow1.json
```

Result on add2/sub2 n=12:

| method | del20 | recall20 | selected source | selected scaffold |
|---|---:|---:|---:|---:|
| AttnLRP | 0.050 | 0.950 | - | - |
| AttnLRP+closure | 0.096 | **1.000** | - | - |
| `cond_greedy_attnlrp` | 0.038 | 0.783 | 0.306 | 0.444 |
| `cond_greedy_attnlrp_closure` | **0.011** | 0.821 | 0.306 | 0.472 |

This did **not** fix the underlying issue. The aggregate is almost the same as
unweighted conditional greedy. Source and distractor/scaffold often have similar
guide prior once they enter top-M, so multiplying by the prior does not stop the
conditional objective from drifting.

### 16.3 Top-M Width Probe

Hypothesis:

```text
scaffold enters through the wide top-32 candidate set; use top-8
```

Targeted arithmetic run:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --attnlrp \
  --attnlrp-closure \
  --no-token-fri \
  --no-rb-cs \
  --conditional-greedy \
  --conditional-guides attnlrp,attnlrp_closure \
  --conditional-top-m 8 \
  --conditional-rounds 3 \
  --out outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_arith_n12_condgreedy_top8.json
```

Result:

| method | del20 | cost |
|---|---:|---:|
| AttnLRP | **0.057** | 1 bwd |
| `cond_greedy_attnlrp` top8 | 0.082 | 21 fwd + 1 bwd |
| `cond_greedy_attnlrp_closure` top8 | 0.253 | 21 fwd + 1 bwd |

Top-8 helps some earlier failure cases, especially sub2 variant 0, by preventing
`not/shelf` drift. But it catastrophically hurts other sub2 variants. So the
candidate-width fix is not robust.

### 16.4 Causal Group MRI

The decisive diagnostic was direct group deletion:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_group_deletion_mri.py \
  outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_arith_n12_condgreedy_priorpow1.json \
  --device cuda:0 \
  --methods attnlrp,attnlrp_closure,cond_greedy_attnlrp,cond_greedy_attnlrp_closure,ppd_grad \
  --out outputs/class_fri/research_frontier/llm_group_deletion_mri_arith_priorpow1.json
```

Aggregate over add2/sub2 n=12:

| deleted group | mean recovery after delete | mean drop |
|---|---:|---:|
| `cond_greedy_*:selected` | **-0.125** | **1.125** |
| `cond_greedy_*:top1` | -0.113 | 1.113 |
| AttnLRP top1 | -0.077 | 1.077 |
| AttnLRP top5 | 0.103 | 0.897 |
| primary source group | 0.119 | 0.881 |
| AttnLRP top3 | 0.159 | 0.841 |
| distractor group | **0.978** | **0.022** |

This is the core MRI finding:

```text
distractor deletion alone does almost nothing,
but source + distractor/scaffold co-deletion can be maximally destructive.
```

So conditional greedy is not necessarily finding a semantic necessary set. It
often finds an **adversarial co-deletion**: once one true source token is gone,
the next most destructive deletion can be a distractor, punctuation, or prompt
scaffold token. This lowers deletion AUC but corrupts source faithfulness.

### 16.5 Updated Mechanism

Why AttnLRP works:

- It reads answer-relevance transport from the model's computation.
- It ranks the semantic source first almost perfectly on hardcoop.
- It is weak mainly at recovering the whole fragmented source set, not at
  identifying the first load-bearing token.

Why direct FRI/delete and conditional greedy fail:

- Their objective is **prediction destruction**, not source recovery.
- In LLM prompts, destruction has shortcuts: prompt scaffold, punctuation,
  distractors, and control words.
- These shortcuts are not necessary content, but they can be highly destructive
  in combination with one real source deletion.

Working principle after MRI:

```text
For LLM necessity, use internal transport/readout to find semantic source.
Use deletion only as a verifier or stopping diagnostic, not as an unconstrained
ranking objective.
```

Next method to try:

```text
AttnLRP/readout-transport source prior
-> source-coherent closure to recover fragmented source spans
-> group deletion only to decide whether the source-coherent set is sufficient
   to destroy the prediction
```

This is different from conditional greedy. Conditional greedy asks "what should
I delete next to break the model fastest?" The MRI suggests the better question
is:

```text
which tokens are on the same answer-transport route as the first source token?
```

That route/coherence signal is the likely LLM analogue of the necessary-ERF.

### 16.6 Hard-Task Deletion MRI: Real Evidence vs Shortcut

I extended `scripts/llm_group_deletion_mri.py` so it can rebuild saved
benchmark cases from the result JSON, including `longctx` cases, and so each
group deletion records:

- raw `p_after_delete`
- raw `prob_drop = p_full - p_after_delete`
- normalized recovery/drop
- top next-token predictions after the deletion

This matters because long-context hard-ID nulls can have `p_base` close to or
above `p_full`. In those cases normalized deletion AUC/recovery can be
misleading; raw probability drop and the post-deletion top prediction are the
clean MRI signals.

Commands:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_group_deletion_mri.py \
  outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_longctx_q1q4_condgreedy_singleocc.json \
  --device cuda:0 \
  --methods attnlrp,attnlrp_closure,single_occ,cond_greedy_attnlrp,cond_greedy_attnlrp_closure,cond_greedy_ppd_grad,ppd_grad \
  --out outputs/class_fri/research_frontier/llm_group_deletion_mri_longctx_q1q4.json

/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_group_deletion_mri.py \
  outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_arith_n12_condgreedy_priorpow1.json \
  --device cuda:0 \
  --methods attnlrp,attnlrp_closure,cond_greedy_attnlrp,cond_greedy_attnlrp_closure,ppd_grad \
  --out outputs/class_fri/research_frontier/llm_group_deletion_mri_arith_priorpow1.json
```

Long-context QA findings:

| case | source deletion raw drop | post-source-deletion top | interpretation |
|---|---:|---|---|
| Q1 copy 8320 | 0.991 | `2` over `0`/space | real evidence deletion; source digit is load-bearing |
| Q2 retrieve Hillary | 0.026 | still `Hillary` at 0.97 | not a real necessity case under this null; high `p_base` already contains the answer |
| Q3 year 1953 | 0.205 | still `5` at 0.78 | partial necessity; raw drop is real, normalized metric is inverted because `p_base > p_full` |
| Q4 Amazon Atlantic | 0.639 | `Atlantic` falls to 0.20; `in`/`Pacific` rise | real evidence deletion |

Conditional greedy on the same long-context run:

| case | conditional selected raw drop | post-selected-deletion top | interpretation |
|---|---:|---|---|
| Q1 | 0.995 | wrong digit `0` | source-like deletion, mostly real |
| Q2 | 0.407 | still `Hillary` at 0.59, `also` rises | confidence shortcut, not semantic necessity |
| Q3 | -0.009 | still `5` at 0.999 | normalized deletion win is artifact/anti-necessity |
| Q4 | 0.719 | generic `ocean` beats `Atlantic` | source plus wording/context co-deletion; destructive but less clean than source deletion |

Arithmetic hardcoop raw MRI confirms the previous shortcut diagnosis:

| deleted group | n | mean p after delete | mean raw prob drop |
|---|---:|---:|---:|
| conditional-greedy selected | 12 | 0.005 | 0.933 |
| AttnLRP top1 | 12 | 0.043 | 0.894 |
| primary/source group | 12 | 0.206 | 0.732 |
| distractor group | 12 | 0.921 | 0.017 |

The top-after-deletion trace is revealing. On add2, distractor deletion leaves
the correct answer near 0.99, so distractors are not necessary. But conditional
greedy often drives the answer probability almost to zero and makes a wrong
digit dominate. This is strong causal destruction, but it is not the same as
recovering the semantic necessary set: it is a source-plus-scaffold or
source-plus-control co-deletion path.

Updated diagnostic conclusion:

```text
Good deletion can mean three different things:
1. real evidence deletion: source deletion lowers p and changes/weakens the answer;
2. partial evidence deletion: source deletion lowers p but top answer remains;
3. shortcut destruction: selected set lowers p by damaging prompt/control/context
   while the true source alone is weak or the top answer remains unchanged.
```

For LLM necessity, raw deletion AUC alone is not enough. The MRI check should be:

```text
source hit/rank
+ raw probability drop
+ post-deletion top prediction
+ source-vs-distractor/scaffold group deletion
```

This strengthens the earlier mechanism claim: deletion should be a verifier, not
the unconstrained search objective. The promising path remains
transport/readout-first necessity, then source-coherent closure, then causal
group deletion only as a guard.

### 16.7 Hardcoop n=30 Shortcut Map

I also ran the same raw/top-after group MRI on the full hardcoop n=30 deletion
run:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_group_deletion_mri.py \
  outputs/class_fri/research_frontier/llm_ppd_deletion_qwen25_1p5b_hardcoop_n30_condgreedy_attn_ppd.json \
  --device cuda:0 \
  --methods attnlrp,attnlrp_closure,cond_greedy_attnlrp,cond_greedy_attnlrp_closure,cond_greedy_ppd_grad,ppd_grad \
  --out outputs/class_fri/research_frontier/llm_group_deletion_mri_hardcoop_n30_condgreedy.json
```

Aggregate:

| group | n | mean p after delete | mean raw prob drop |
|---|---:|---:|---:|
| conditional-greedy selected | 30 | 0.002 | 0.769 |
| AttnLRP top1 | 30 | 0.017 | 0.754 |
| primary/source group | 30 | 0.082 | 0.689 |
| distractor group | 12 | 0.921 | 0.017 |
| PPD grad top1 | 30 | 0.352 | 0.419 |

By kind:

| kind | primary drop | conditional selected drop | gap | interpretation |
|---|---:|---:|---:|---|
| lookup2 | 0.366 | 0.366 | 0.000 | source deletion; no shortcut gap |
| ctxcopy | 0.811 | 0.811 | 0.000 | source deletion |
| latercode | 0.802 | 0.802 | 0.000 | source deletion |
| add2 | 0.814 | 0.992 | 0.178 | conditional co-deletion shortcut |
| sub2 | 0.650 | 0.873 | 0.223 | conditional co-deletion shortcut |

This answers the harder-task question more cleanly:

```text
When the task has a single sharp copied/retrieved value, good deletion is usually
real source deletion.

When the task requires a cooperative arithmetic relation, the semantic source is
real and necessary, but the strongest deletion set can include non-semantic
prompt/control/context tokens. That extra deletion gain is a shortcut.
```

The diagnostic signature of the shortcut is:

- distractor deletion alone has near-zero raw drop (`add2/sub2`: 0.017 mean);
- primary/source deletion has large raw drop (`add2/sub2`: 0.73 mean);
- conditional selected deletion has even larger raw drop (`add2/sub2`: 0.93 mean);
- post-deletion top predictions become high-confidence wrong digits.

So the current best rule is not "maximize deletion." It is:

```text
First identify the answer-transport/source route.
Then use deletion to verify that route, not to freely add destructive tokens.
```

## 17. Model-Agnostic Necessity: Forward-Only Pivotality

Important correction: AttnLRP is useful as an MRI/readout baseline, but it is
not the desired final method because it is not model-agnostic. I added a
forward-only model-agnostic candidate to `scripts/llm_ppd_insertion_benchmark.py`.

The idea:

```text
Sample hard-ID null/original masks at fixed budget densities.
For each density, estimate token pivotality by:

  E[p(answer) | token present, same density]
  - E[p(answer) | token absent, same density]

No gradients. No attention. No model internals. Only forward probabilities.
```

I tested three variants:

- `restricted_banzhaf`: high-density Banzhaf with keep density >= 0.75.
- `ma_pivotal_del`: high-density deletion-like pivotality, 48 forwards.
- `ma_dual_pivotal`: product of low/mid-density insertion-like and high-density
  deletion-like pivotality, 96 forwards.

The dual idea was motivated by filtering shortcuts:

```text
true content source: insertion-positive and deletion-positive
context-only token: insertion-positive but deletion-weak
prompt/scaffold shortcut: deletion-positive but insertion-weak
```

But empirically the insertion branch was too scaffold-biased on hardcoop. So
the dual product often made the ranking worse. The best model-agnostic signal
so far is simply high-density forward-only Banzhaf.

Commands:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds lookup2,ctxcopy,latercode,add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --banzhaf \
  --banzhaf-samples 96 \
  --banzhaf-lo 0.75 \
  --dual-pivotal \
  --dual-pivotal-samples 48 \
  --no-token-fri \
  --no-rb-cs \
  --out outputs/class_fri/research_frontier/llm_modelagnostic_dual_pivotal_hardcoop_n30.json
```

Hardcoop n=30 aggregate:

| method | model-agnostic? | cost | del20 | rank median | recall20 | top1 source | top5 scaffold |
|---|---|---:|---:|---:|---:|---:|---:|
| `restricted_banzhaf` | yes | 96 fwd | 0.102 | 1.0 | 0.627 | 0.767 | 0.587 |
| `ma_pivotal_del` | yes | 48 fwd | 0.115 | 1.0 | 0.562 | 0.533 | 0.667 |
| `ma_dual_pivotal` | yes | 96 fwd | 0.091 | 1.0 | 0.505 | 0.567 | 0.733 |
| `ppd_grad` | no, gradient | 4 bwd | 0.104 | 3.0 | 0.702 | 0.367 | 0.640 |

Arithmetic-only n=12:

| method | cost | del20 | rank median | recall20 | top1 source | top5 scaffold |
|---|---:|---:|---:|---:|---:|---:|
| `restricted_banzhaf` | 96 fwd | 0.116 | 1.0 | 0.867 | 0.750 | 0.450 |
| `ma_pivotal_del` | 48 fwd | 0.181 | 1.0 | 0.867 | 0.667 | 0.567 |
| `single_occ` | ~191 fwd | 0.056 | 1.0 | 1.000 | 0.833 | 0.250 |

Raw group MRI, hardcoop n=30:

| deleted group | n | mean p after delete | raw prob drop |
|---|---:|---:|---:|
| source/primary group | 30 | 0.082 | 0.689 |
| `restricted_banzhaf:top1` | 30 | 0.091 | 0.680 |
| `restricted_banzhaf:top3` | 30 | 0.049 | 0.722 |
| `restricted_banzhaf:top5` | 30 | 0.065 | 0.706 |
| distractor group | 12 | 0.921 | 0.017 |

By kind, raw prob drop:

| kind | primary source | Banzhaf top1 | Banzhaf top3 | interpretation |
|---|---:|---:|---:|---|
| lookup2 | 0.366 | 0.366 | 0.366 | real source deletion |
| ctxcopy | 0.811 | 0.779 | 0.811 | real source deletion |
| latercode | 0.802 | 0.802 | 0.802 | real source deletion |
| add2 | 0.814 | 0.982 | 0.870 | Banzhaf finds destructive source-like digits, sometimes stronger than full annotated source |
| sub2 | 0.650 | 0.470 | 0.762 | top1 unstable, top3 recovers destructive source group |

Interpretation:

```text
There is a real model-agnostic causal signal.
High-density random-coalition Banzhaf can reveal source/necessary anchors with
~100 forward calls and no model internals.
```

But it is not solved:

- It is good at the first necessary anchor (`rank_med = 1.0`).
- It is not clean enough as a full necessary set (`recall20 = 0.627` on n=30).
- The top-k still contains many scaffold/filler tokens (`top5 scaffold = 0.587`).
- The dual insertion+deletion idea did not solve scaffold contamination; the
  insertion branch itself was too scaffold-biased.

Current best model-agnostic direction:

```text
Forward-only high-density Banzhaf -> source anchor
then model-agnostic source-coherence expansion/cleanup
then causal deletion guard.
```

The missing piece is the model-agnostic equivalent of "route coherence." It
cannot be AttnLRP. It likely has to be inferred from black-box causal
co-occurrence:

```text
tokens belong to the same necessary route if they are jointly pivotal across the
same successful/failing random coalitions, while scaffold tokens have broad,
non-specific fragility effects.
```

Next concrete method:

```text
1. Run high-density Banzhaf to get anchor tokens.
2. Reuse the same random coalition table.
3. Compute token-token co-pivotality with the anchor:
   Cov(1[token absent], 1[anchor absent], p_drop) or conditional drop lift.
4. Keep tokens whose co-pivotality is specific to the anchor and whose deletion
   does not look like global prompt fragility.
```

That would stay model-agnostic and may turn the current anchor finder into a
clean necessary-set finder.

### 17.1 Anchor Co-Pivotal Probe

I implemented the first version of that idea as `ma_anchor_copivotal`:

```text
1. Sample high-density hard-ID coalitions.
2. Pick the top pivotal token as the anchor.
3. For every other token j, estimate AND-like lift:

   p(anchor present, j present)
   - p(anchor present, j absent)

   gated by the symmetric lift where j is present and anchor is absent.
4. Rank anchor first, then tokens with high anchor-specific co-pivotality.
```

Command on arithmetic n=12:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/llm_ppd_insertion_benchmark.py \
  --device cuda:0 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --kinds '' \
  --hard-cases \
  --hard-kinds add2,sub2 \
  --hard-lengths 128,192,256 \
  --hard-variants 2 \
  --keep-mode query \
  --rho 0.3 \
  --banzhaf \
  --banzhaf-samples 96 \
  --banzhaf-lo 0.75 \
  --anchor-copivotal \
  --anchor-copivotal-samples 96 \
  --no-token-fri \
  --no-rb-cs \
  --out outputs/class_fri/research_frontier/llm_modelagnostic_anchor_copiv_arith_n12.json
```

Result:

| method | cost | del20 | rank median | recall20 | top1 source | top5 scaffold |
|---|---:|---:|---:|---:|---:|---:|
| `restricted_banzhaf` | 96 fwd | 0.116 | 1.0 | 0.867 | 0.750 | 0.450 |
| `ma_anchor_copivotal` | 96 fwd | 0.081 | 1.0 | 0.763 | 0.667 | 0.500 |
| `ppd_grad` | 4 bwd | 0.183 | 4.5 | 0.917 | 0.167 | 0.567 |

Raw group MRI:

| group | n | p after delete | raw prob drop |
|---|---:|---:|---:|
| primary/source group | 12 | 0.206 | 0.732 |
| distractor group | 12 | 0.921 | 0.017 |
| `restricted_banzhaf:top3` | 12 | 0.195 | 0.742 |
| `restricted_banzhaf:top5` | 12 | 0.118 | 0.820 |
| `ma_anchor_copivotal:top3` | 12 | 0.084 | 0.854 |
| `ma_anchor_copivotal:top5` | 12 | 0.077 | 0.861 |

Interpretation:

```text
Anchor co-pivotality improves deletion AUC and raw drop, but it is not cleaner.
It trades source recall for more destructive co-deletion.
```

Failure mode:

- On sub2, anchor co-pivotality often links the source digit to operation or
  sentence-control tokens (`not`, `changed`, `held`, punctuation, route words).
- Those tokens are genuinely co-destructive with the source under hard deletion,
  but they are not the semantic necessary content set we want.

So co-pivotality alone is not enough. It is another version of the same trap:

```text
black-box causal co-destruction != clean semantic necessity
```

Still, this is a useful result. We now have:

- a model-agnostic **anchor finder**: high-density Banzhaf;
- a model-agnostic **destruction amplifier**: anchor co-pivotality;
- an MRI showing the amplifier is shortcut-prone.

The next model-agnostic cleanup needs a way to distinguish:

```text
specific content co-pivotality
vs
global prompt/control fragility
```

One possible test is a control-target subtraction:

```text
score(j) =
  co-pivotality_with_answer_anchor(j)
  - average co-pivotality_with_random_or_nonanswer_anchors(j)
```

This stays model-agnostic and may subtract broad scaffold fragility while
keeping answer-specific operands/entities.

## 18. Vision necessity return: weak causal correction of hidden-FRI

Motivation:

```text
LLM work suggested that black-box finite perturbation can find pivotal
cooperative structure, but also that using it as the main signal is shortcut
prone. For vision necessity, hidden-FRI is already strong because plain ViT has
self-patch routing. The question was whether cheap input-causal perturbation can
improve deletion without replacing the good hidden prior.
```

Implemented script:

```bash
scripts/research_vision_modelagnostic_pivotal_bench.py
```

Core methods:

- `ma_banzhaf`, `ma_copiv*`: pure model-agnostic random high-density hard patch
  coalitions, using only input masks and target probabilities.
- `hfri_cbanzhaf*`: restrict the random coalitions to the top-M hidden-FRI
  candidate patches; non-candidates stay present.
- `hfri_mix_*`: keep hidden-FRI as the main score and add only a weak causal
  correction:

```text
score = (1 - alpha) * norm(hidden_fri) + alpha * norm(candidate_causal_score)
alpha in {0.05, 0.10, 0.20}
```

Important mechanism:

```text
Pure black-box random coalitions are too noisy/contextual in vision.
Replacing hidden-FRI with candidate causal scores also hurts: the causal score
is a useful correction but a bad primary storage readout. The winning pattern is
"read necessity from the model's routing, then weakly calibrate it with genuine
input deletion."
```

n=16, CLIP ViT-B/16, ImageNet val, deletion lower is better:

| method | cost | hard_del | mas_del | stoch_del |
|---|---:|---:|---:|---:|
| `hidden_fri` | 64 | 0.276 | 0.531 | 0.520 |
| `inflow` | 1 | 0.309 | 0.597 | 0.557 |
| `ma_banzhaf` | 96 | 0.493 | 0.948 | 0.682 |
| `hfri_cbanzhaf_mix` | 160 | 0.291 | 0.607 | 0.629 |
| `hfri_mix_cbanzhaf_local_a20` | 160 | 0.265 | 0.512 | 0.504 |

n=30 confirmation:

| method | cost | hard_del | mas_del | stoch_del |
|---|---:|---:|---:|---:|
| `hidden_fri` | 64 | 0.282 | 0.551 | 0.525 |
| `inflow` | 1 | 0.296 | 0.576 | 0.550 |
| `ma_banzhaf` | 96 | 0.497 | 0.962 | 0.695 |
| `hfri_cbanzhaf_local` | 160 | 0.330 | 0.640 | 0.585 |
| `hfri_mix_cbanzhaf_local_a10` | 160 | 0.271 | 0.528 | 0.511 |
| `hfri_mix_cbanzhaf_local_a20` | 160 | 0.270 | 0.529 | 0.497 |

n=30 low-cost check (`samples=32`, total cost 96 forward):

| method | cost | hard_del | mas_del | stoch_del |
|---|---:|---:|---:|---:|
| `hidden_fri` | 64 | 0.282 | 0.551 | 0.525 |
| `inflow` | 1 | 0.296 | 0.576 | 0.553 |
| `ma_banzhaf` | 32 | 0.529 | 1.028 | 0.761 |
| `hfri_cbanzhaf_local` | 96 | 0.359 | 0.698 | 0.622 |
| `hfri_mix_cbanzhaf_local_a10` | 96 | 0.275 | 0.537 | 0.516 |
| `hfri_mix_cbanzhaf_local_a20` | 96 | 0.280 | 0.548 | 0.515 |

Interpretation:

- The pure model-agnostic route is a negative result for vision deletion.
  Random coalitions do not reveal the clean necessary set when redundancy is
  spatial and spread out.
- The useful signal is not "black-box Banzhaf beats hidden-FRI." It does not.
- The useful signal is that input-causal finite deletion gives a small but real
  value-faithful correction to hidden-FRI when treated as a weak calibration.
- The ~100-forward version is weaker than the 160-forward version, but still
  improves all three deletion metrics over hidden-FRI at n=30.
- This supports the current universal recipe:

```text
cheap routing readout first;
small model-agnostic causal calibration second;
final set judged by genuine input removal.
```

Reproduction:

```bash
/home/sangyu/anaconda3/envs/py312/bin/python scripts/research_vision_modelagnostic_pivotal_bench.py \
  --device cuda:0 \
  --nimg 30 \
  --samples 96 \
  --candidate-m 64 \
  --out outputs/class_fri/research_frontier/vision_ma_pivotal_hybrid_n30

/home/sangyu/anaconda3/envs/py312/bin/python scripts/research_necset_benchmark_eval.py \
  --device cuda:0 \
  --bench outputs/class_fri/research_frontier/vision_hfri_causal_best_n30
```

Note: `vision_hfri_causal_best_n30` was produced from the Stage-A score file by
keeping the best weak-mix candidates. The script now emits those `hfri_mix_*`
scores directly, so a fresh run can evaluate them without the intermediate
manual score-combination step.
