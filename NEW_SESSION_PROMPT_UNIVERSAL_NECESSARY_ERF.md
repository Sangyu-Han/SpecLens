# Research Handoff — A Universal "Necessary-ERF" for Necessity Attribution

## 0. The one-sentence goal
We already have a method (**hidden-FRI**) that finds the **necessity set** cheaply and well **for plain ViTs**. Your job: make a **general necessary-ERF that also works for vision models where the "self-patch" assumption breaks** (latent-bottleneck / register / token-merge / CNN). If it works for those, the same recipe covers *every* architecture (LLMs included) → a universal necessity-attribution method.

A solid fallback paper already exists from the ViT result alone (see §6). This handoff is about reaching **higher**.

---

## 1. Problem framing — necessity vs sufficiency

For an input (image patches / text tokens) and a model prediction:

- **Sufficiency** = the minimal set of inputs that, **inserted** into a blank/mean baseline, **recovers** the prediction. Tool: **FRI** (Feature-Recovery Insertion — a soft-mask gradient solve, random-budget, ~32–64 steps). Cheap, **model-agnostic**. But it answers *"what BUILDS the prediction"* → it includes **context**.
- **Necessity** = the minimal set whose **removal destroys** the prediction. It answers *"what is load-bearing."* This is **causal** (requires actual deletion) and **model-specific**.
- In **vision these two sets are nearly DISJOINT** (CLIP set-overlap ≈ 0.03). Necessity ≠ sufficiency. This is the whole reason the problem is hard.

**Necessity is the harder, more valuable face**, and it is what this research is about.

### The "necessary-ERF" idea
To get input-necessity from the hidden layer: each **necessary hidden token** has an **ERF** (Effective Receptive Field = which input patches it depends on / came from). If we trace the *necessary* hidden tokens back to their *necessary* input support, we get the input-necessity — **even for non-self-patch tokens like a CLS/summary token** (we can see what it actually looked at). That back-trace IS the ERF.

**The catch:** if you compute that ERF by **sufficiency (FRI)**, you get the patches that *build* the hidden token's meaning → which includes **context** (e.g. an "elephant-head" token needs the savanna to be disambiguated). So sufficient-ERF ⇒ context, not the necessary core. We need the **necessary-ERF**, and that is the open problem.

---

## 2. What WORKS (with numbers)

Necessity succeeds when we exploit the **model's routing structure**:

| model / modality | routing (ERF) structure | method that works | result |
|---|---|---|---|
| **plain ViT** (clip / deit3 / augreg) | **self-patch** (hidden token ↔ its own input patch; routing ≈ identity) | **hidden-FRI** — solve in the LAST block by masking patch tokens toward the hidden-mean (FRI delonly, **prob** target), then decide the set by genuine **input** removal | beats inflow on **all** deletion metrics (hard_del 0.276 vs 0.309, mas_del 0.531 vs 0.597, stoch_del 0.524 vs 0.563, n=16), at **~IntegGrad cost** (~64 last-block fwd) |
| vision (exact, expensive) | self-patch | **conditional** (hidden-FRI prior → top-M → greedy-conditional input removal) | near-oracle: hard_del 0.145, mas_del 0.294 (~5k fwd). Oracle itself (full greedy O(N²)) = 0.104, the lowest. |
| vision (cheap, portfolio) | last-hidden value clusters | **module-deletion + est-guard** | beats inflow, hdel 0.252 (n100, p1e-13) |
| **latent-bottleneck** (Perceiver/DETR/Q-Former) | **cross-attention** | cross-attn rollout / gradient (position-free) | nec 0.149, p1.7e-3 |
| **CNN** | **receptive field** | hidden-channel ERF = upsample (implicit) | transfers, no-regret across 6 archs |
| **LLM / natural language** | self-patch *broken*, but input short+sharp | **input-occlusion** (1-pass causal deletion) | necessity-core recall **0.852** |

**Key empirical anchors**: prob (not logit) target; INPUT removal (not hidden) for the final set; necessity is **causal**; for visualization use the **marginal** (per-removal prob drop), not the rank (the greedy order saturates after the set, so the tail is arbitrary noise).

---

## 3. What FAILS (with numbers — don't repeat these)

| failed approach | where | why it fails |
|---|---|---|
| **gradient / input×grad / Integrated Gradients** | every model (LLM 0.41–0.52; worst in vision) | local sensitivity **misses redundancy** — substitutable patches each have low gradient. Necessity needs *removal*, not a derivative. |
| **sufficiency (FRI) used as necessity** | LLM 0.37; vision overlap 0.03 | gives **context** ("what builds it"), not the load-bearing core |
| **hidden-FRI at an LLM late layer** | LLM, overlap 0.056 | self-patch is broken (see §4) — it credits the summary/last position, not the content |
| **position-free necessity on a plain ViT** | ViT | right tool, wrong architecture — self-patch (position-matching) wins there |
| **cheap chunked-conditional (R=2–3)** | vision | coarse chunk-commits ruin the marginal **value**-faithfulness; worse than plain hidden-FRI. No good "cheap middle" tier. |
| **logit target / hidden-masking for the set** | vision multi-object | logit credits context & is robust to removal (huge sets); hidden masking can be an OOD-free *artifact* — verify with genuine input removal |

---

## 4. Root causes — WHY each thing happens (the mechanism)

This is the important part. Understand these before trying anything.

- **The strongest LEAD (not a rule) — the model already KNOWS what is necessary.** During the forward pass the model *compresses* the input into its decision: it has already thrown away the irrelevant and kept the load-bearing, so the necessity is already encoded in the hidden representations (the O(N²) input oracle just *re-discovers* what the model computed for free). **Every cheap win so far came from reading that out** — hidden-FRI, last-hidden module clustering, cross-attn rollout all expose the model's own routing in ~1 forward instead of perturbing the input thousands of times. Memorable form: *don't COMPUTE necessity, REVEAL it.* **Treat this as a promising prior, NOT a fence.** You are free to use cheap perturbation, conditional refinement, hybrids of readout+perturbation, or an approach no one here tried — judge ideas by the metrics, not by whether they fit this slogan. (We even flag in §5 that a pure cheap readout may *not* exist when necessity is redundant AND routing is complex; if so, cleanly characterizing that boundary is itself a contribution.)

- **Why ViT self-patch works** — *it is simply the most readable storage format for that knowledge.* In a ViT's hidden layers, each patch token stores **"I am the patch at position X, and my meaning is Y"** — i.e. position **and** high-dimensional semantics of *its own* patch (e.g. an elephant patch hidden state ≈ "elephant-head, here"). So the hidden token is a faithful proxy for its input patch: **masking hidden token i ≈ removing input patch i** (routing = identity). That is exactly why solving in the rich hidden space and mapping back by position recovers the *input* necessity cheaply. hidden-FRI is literally "the self-patch ERF."

- **Why the LLM breaks (and breaks early).** Causal self-attention **moves/copies** content forward. By **layer 2** in a 1.5B model, the answer-relevant content has already been copied from the content tokens (Eiffel/Tower) into the **last / summary position** (the readout). After that, masking the content token's hidden state does nothing (the info is elsewhere), and the only "necessary" hidden position is the summary. So the hidden token no longer represents its own input → self-patch identity is destroyed → the hidden ERF points at the summary, not the content.

- **Why gradient fails for necessity.** Gradient = infinitesimal sensitivity at the current point. With redundancy (two patches that can substitute for each other), removing *either* alone barely changes the output, so *both* get small gradients — gradient cannot see "necessary as a set." Only **finite causal removal** (occlusion / greedy-conditional deletion) reveals it.

- **Why sufficiency ≠ necessity (context vs load-bearing).** Building a representation needs **context for disambiguation** (savanna ⇒ "this gray thing is an elephant"); that context is **sufficient-contributing** but **redundant** for the prediction once the elephant patches are present, so it is **not necessary**. FRI optimizes "recover when inserted" ⇒ it grabs the context. Necessity asks "destroy when removed" ⇒ the local core.

- **Why NL is *nearly* free but vision is not.** In natural language, **context ≈ necessity** (the necessary tokens basically *are* the informative context; nec–suff overlap 0.345 vs vision 0.03), and inputs are short + the necessary core is usually a single sharp token. So plain 1-pass input-occlusion suffices and you never need a hidden trick. In vision the necessary set is **redundant and spatially spread**, and the input is large (196 patches), so 1-pass occlusion misses redundancy and full input-deletion is O(N²) — hence the need for the hidden-FRI shortcut (when self-patch holds) or conditional (when it doesn't).

- **Why masking MUST use the MEAN vector, never zeros — this is one of our contributions, and the trap that silently invalidates hidden-layer studies.** Activations live on the model's learned **manifold** (the distribution of states it actually produces). **Zeroing** a hidden token — or an input patch — shoves it FAR off-manifold (zero is not a plausible activation), so downstream layers see garbage and the prediction collapses for **OOD reasons, not because you removed that token's information.** You would be measuring *"the model breaks on weird input,"* which looks like a huge necessity signal but is an artifact. The **mean** vector is the *expected / neutral* activation: replacing a token with it keeps the sequence **on-manifold** while **erasing that token's specific information** (it now carries only the average signal). The resulting prediction drop then reflects the *genuine* loss of that token's contribution = true necessity. **So: to successfully erase information in a hidden layer you interpolate toward the hidden-MEAN; for input removal you set pixels to the mean (= normalized 0). Never zero a hidden state, never use noise — if you do, your "necessity" numbers are measuring OOD.** This holds for any new architecture you study.

**Unifying principle:** *necessity = causal removal + the model's routing structure.* The model **always** has the necessity (it made the decision); the **routing structure is just how readably it stored that knowledge.** Simple/known routing (self-patch, cross-attention, conv receptive-field) = the knowledge is in a directly readable form → get necessity cheaply from the hidden layer. Complex routing (LLM info-movement, heavy aggregation) = the knowledge is *there* but stored in a form we can't yet read by position → the hidden shortcut dies and you fall back to input-space (cheap only if necessity is sharp). **Sufficiency (FRI) is model-agnostic and cheap but gives context; necessity is model-specific because recovering it cheaply requires knowing the routing.** So the frontier (§5) is *not* "make the model know what's necessary" — it already does — but **"recover its necessary input support cheaply for a routing we haven't cracked."** Reading the routing is the leading idea; a smarter causal perturbation, a hybrid, or a genuinely new necessity method is equally welcome — there may well be a better approach no one here has found.

---

## 5. YOUR frontier — the actual task

Find a **cheap necessary-ERF for VISION non-self-patch** models (this is the gap; the LLM case is already handled by input-occlusion). **Genuine test beds = architectures with NO usable self-patch tokens to fall back on:** **token-merge** (ToMe = content-merge, position-free, *and the merge is explicit so the routing is readable* / Swin = position-merge, coarse), **latent-bottleneck** (Perceiver / DETR / BLIP-2 Q-Former — latents have no spatial identity), **register**-token models (DINOv2-reg, already in timm).

> **NOT a valid test bed — do not drift here:** a *standard ViT read from its CLS token*. Yes, the CLS is technically a non-self-patch token, but the backbone's **patch tokens still carry self-patch**, so any necessity you "recover from the CLS" is just re-findable from the patches — it proves nothing and is a proxy, not the goal. Study only architectures where self-patch **genuinely does not exist**.

**Frame it correctly (per §4):** a token-merge / latent model **still knows** what's necessary — it produced a correct, confident decision, so the load-bearing input support *is* encoded somewhere in its activations. So you are **not** trying to make the model find necessity; you are trying to **recover** it cheaply. The core question: **how do you recover a necessary hidden unit's (e.g. a ToMe merged token, a Perceiver latent) necessary INPUT support, cheaply, without the sufficiency-context contamination and without the gradient failure?** Reading the model's routing is the most-tested lead — but **a smarter causal perturbation, a hybrid, or an entirely new way to find the necessary set may well beat it. Explore widely and let the metrics (vs the input greedy-conditional oracle) decide; a genuinely new necessity method is the best possible outcome.**

Promising leads (not yet validated for vision non-self-patch):
1. **Read the explicit routing where it exists.** ToMe records its merge bipartite-matching per layer, so a necessary merged token can be traced back to its source input patches *with no perturbation* — the cleanest "read out the routing" case. For attention-based aggregation (latents), use **cross-attention rollback** of a latent/merged token — but *prune to the necessary core*, since raw attention = sufficient-ERF (context).
2. **Prediction-guided causal ERF**: ablate input patch j and measure the change in the hidden token's *prediction-relevant projection* (h · ĝ, ĝ = ∂logit/∂h), not its full value. This drops context that builds h but is prediction-irrelevant. (Gradient alone failed; the *causal* version may not — but watch the cost.)
3. **Layer selection**: solve at the layer where the encoding is rich enough yet self-patch still holds (a sweet spot before context aggregation) — find it with a layer sweep like the LLM one.
4. Honest possibility: there may be **no cheap shortcut** when necessity is redundant AND routing is complex — in which case the contribution is *characterizing that boundary* (cost asymmetry) rather than beating it.

Always compare against the **input greedy-conditional oracle** (the gold necessity) and **inflow** (the baseline to beat), using the value-faithful metrics.

---

## 6. Infrastructure (reuse these — don't rebuild)

- **Interpreter (REQUIRED):** `/home/sangyu/anaconda3/envs/py312/bin/python` (base conda is broken). **GPU: use `cuda:0`** — `cuda:1` has the user's process, leave it alone.
- **hidden-FRI method:** `scripts/viz_necessity_hidden_vit.py` → `HiddenNec` class (hidden-mean masking, FRI solve, input_prob removal, prob/logit target flag).
- **Greedy oracle + marginal viz:** `scripts/viz_greedy_oracle.py`.
- **Benchmark (2-stage, because the metric repo needs a different sys.path):** `scripts/research_necset_benchmark_scores.py` (stage A, computes scores via hg5) → `scripts/research_necset_benchmark_eval.py` (stage B, computes metrics). Cheap-tier + cost tracking: `scripts/research_necset_cheap_bench.py`.
- **Eval harness / baselines:** `scripts/research_hgrad_v5.py` exposes `load_patch_repo()` (gives `inflow`, `load_image`, `get_b0`), `Block0Runner` (`.probs_for_masks`, `.logits_for_masks`), `hard_curves`.
- **Faithfulness metrics:** `~/Desktop/Master/patch-attribution-vit/src/eval/metrics_block0.py` → `evaluate_method_block0(...)` returns `hard_/mas_/stoch_` × `ins/del` AUC. MAS = magnitude-alignment (value-faithful), stoch = sensitivity-n. **Lower = better on deletion, higher = better on insertion.**
- **LLM scripts:** `scripts/llm_layer_sweep.py` (self-patch breakdown curve), `scripts/llm_nec_vs_suff.py`, `scripts/llm_pred_guided_erf.py` (occlusion vs gradient vs FRI). Model: `Qwen/Qwen2.5-1.5B` (non-gated, cached).
- **ImageNet val:** `/media/sangyu/Dataset/imagenet/val`. Multi-object test image: `multi_object_zebra_elephant.jpg` (repo root; CLIP predicts zebra 340, elephant 386).
- **Outputs:** `outputs/class_fri/research_frontier/`.

## 7. Hard constraints / lessons (the user will hold you to these)
- **Mask with the MEAN vector — hidden-mean for hidden tokens, mean-pixel for input — and NEVER zero a hidden state or use noise (see §4 for the manifold reason; this is a core contribution, not a preference).** Zeroing erases nothing cleanly: it just throws the activation off-manifold and you end up measuring OOD model-breakage that masquerades as a huge necessity signal. Any time you mask in a new model, mask toward its mean.
- **Deletion = INPUT removal**, always (the genuine causal test). The hidden domain is for *solving/scoring*, not for the final destruction test.
- **No logit-lens, no plain gradient** for necessity — they don't reflect causal contribution. Prefer causal removal; if you need a hidden readout, use a causal probe.
- **prob target** (not logit) for the destruction criterion in multi-object / strong-prediction cases.
- Report honest numbers, scale n before claiming significance (a 3/3 win flipped to a tie at n=30 once this session), and call out negative results plainly.
