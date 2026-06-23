#!/usr/bin/env python3
"""LONG-CONTEXT LLM necessity: cost + redundancy, OUR conditional methods vs AttnLRP.

Extends scripts/research_llm_necessary_set_qual.py from T~15 to T~456 (the AttnLRP
Mount-Everest passage). The long context exposes two things that do NOT exist at
T~15:
  (a) COST: the greedy necessity ORACLE is ~O(k_nec * T) forwards; at T=456 even a
      few greedy steps cost thousands of forwards, and the full O(T^2) deletion
      order would be ~120k forwards. The CHEAP chunked-conditional (1-pass
      occlusion prior -> top-M -> R rounds) must approximate the oracle deletion-
      AUC at a small fraction of the cost.
  (b) REDUNDANCY: the passage lists many elevations (7,000 / 8,000 / 8,320 / 8,595
      m) and years (1921/1922/1924/1952/...). The answer "8,320" is a near-copy of
      its source span, so single-pass attributions (gradient, AttnLRP) may miss
      necessary tokens that the conditional finds.

TARGET = answer token at the last position. argmax = '3' (the first digit of the
copied "8,320"), prob ~0.995 (verified). We target that token id.

Methods (per-input-token attribution):
  1. greedy_conditional  -- necessity ORACLE. Iteratively MEAN-mask the token whose
     removal most drops the answer prob, until prob <= 0.1*full. Removed set =
     necessary set. COST recorded in #forwards. For the deletion-AUC curve we use
     order = [necessary set, then the rest ranked by 1-pass single-occlusion] so we
     do NOT pay the full O(T^2).
  2. chunked_conditional -- CHEAP. 1-pass single-occlusion prior (full -
     prob(remove i alone); NO locality term, LLMs have no spatial self-patch) ->
     top-M candidates -> R rounds of conditional greedy among them. COST ~ T+R*M.
  3. gradient            -- input_embeds * grad(answer_logit), sum over dim, abs. 1
     backward.
  4. attnlrp             -- AttnLRP (lxt.efficient.monkey_patch on modeling_qwen2):
     relevance = (input_embeds.grad * input_embeds).sum(-1) after backprop from the
     answer logit. 1 backward. Run on a SEPARATE monkey-patched model instance so
     the LRP backward rules do not corrupt the plain occlusion forwards.

METRIC: deletion AUC -- mask tokens in attribution order toward MEAN embedding,
measure answer prob normalized by full, trapz. LOWER = better necessity. The
necessary set is tiny vs T=456, so we evaluate over a FIXED BUDGET of the first
BUDGET_FRAC (=0.20) of ranked tokens (and also a small fixed count) so the
irrelevant tail does not wash out the curve. Insertion AUC reported too (cheap).

CONSTRAINTS: MEAN-embedding masking always (never zero). Freeze the readout (last)
position. No BOS is prepended by Qwen for this raw-string prompt (verified: first
token is 'Context').
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# AttnLRP repo (lxt) lives in third_party next to this checkout.
REPO_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "third_party",
        "LRP-eXplains-Transformers-main (1)",
        "LRP-eXplains-Transformers-main",
    )
)

# Verbatim AttnLRP-example context (the ~350-token Everest passage).
CONTEXT = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
"""

# A different ~230-token passage (Amazon River) for an out-of-domain long-context case.
AMAZON = """Context: The Amazon River in South America is the largest river by discharge volume of water in the world, and the disputed longest river system in the world in comparison to the Nile. The headwaters of the Apurimac River on Nevado Mismi had been considered the most distant source until 2014, when a study found it to be the headwaters of the Mantaro River on the Cordillera Rumi Cruz in Peru. The Amazon flows through Brazil, Peru, and Colombia, and its drainage basin covers about 7,000,000 square kilometres. The river has over 1,100 tributaries, twelve of which are over 1,500 kilometres long. The Amazon represents about one-fifth of the total river flow on Earth. During the wet season, parts of the Amazon exceed 190 kilometres in width. The river discharges into the Atlantic Ocean. The Amazon rainforest, which surrounds the river, is the largest rainforest on the planet. \
"""

# Q1 = the verbatim AttnLRP-example question (copy task: answer "8,320" is copied
#      from the source span -> SHARP single-token necessity; cost story).
# Q2 = a same-passage multi-cue retrieval question (answer "Hillary" is cued BOTH by
#      the in-context copy-source AND by the local "...Norgay and Edmund __" frame).
# Q3 = same-passage year-retrieval (answer "53" copied from "1953" in the passage).
# Q4 = DIFFERENT passage (Amazon): answer " Atlantic" copied from "...the Atlantic Ocean".
PROMPT_Q1 = CONTEXT + "Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"
PROMPT_Q2 = CONTEXT + "Question: Who made the first documented ascent of Everest, in 1953? The first documented ascent was made by Tenzing Norgay and Edmund"
PROMPT_Q3 = CONTEXT + "Question: In which year did Tenzing Norgay and Edmund Hillary make the first documented ascent of Everest? Answer: 19"
PROMPT_Q4 = AMAZON + "Question: Into which ocean does the Amazon River discharge? Answer: The Amazon discharges into the"

QUESTIONS = [
    dict(name="Q1_copy_8320", text=PROMPT_Q1,
         # token spans for redundancy probes (digit-split numbers); see verify run
         span_groups=lambda T: {"src_8320": list(range(223, 229)),   # ' 8,320'
                                "year_1922": list(range(210, 215))},  # ' 1922' (anchor)
         digit_groups=lambda T: {"d3": [226], "d2": [227], "d0": [228]},
         localize={"src_8320": (223, 228), "year_1922": (210, 214), "pushed_upto": (216, 222)}),
    dict(name="Q2_retrieve_Hillary", text=PROMPT_Q2,
         # answer "Hillary": in-passage copy-source ' Hillary' at 325 (preceded by
         # ' Edmund' at 324); question restatement 'Tenzing Norgay' at 451-456;
         # ' 1953' at 438-442. (idx 458 ' Edmund' is the frozen readout -> excluded.)
         span_groups=lambda T: {"passage_Hillary": [325],
                                "passage_Edmund": [324],
                                "q_Tenzing_Norgay": list(range(451, 457)),
                                "q_year_1953": list(range(438, 443))},
         digit_groups=None,
         localize={"passage_EdmundHillary": (324, 325),
                   "q_Tenzing_Norgay": (451, 456), "q_year_1953": (438, 442)}),
    dict(name="Q3_year_53", text=PROMPT_Q3,
         # answer "53" -> first digit '5'. Passage source "1953" digits at 334-338
         # (' ','1','9','5','3'); the '5' source digit is idx 337. Name cues
         # ' Hillary' at 325 / ' Norgay'(... Norgay tokens) earlier.
         span_groups=lambda T: {"src_1953": list(range(334, 339)),
                                "passage_Hillary": [325]},
         digit_groups=lambda T: {"d5": [337], "d3": [338]},
         localize={"src_1953": (334, 338), "passage_Hillary": (325, 325)}),
    dict(name="Q4_amazon_Atlantic", text=PROMPT_Q4,
         # NEW passage. answer ' Atlantic' copy-source at idx 189 (' Ocean' at 190);
         # the 'discharges into' frame in the passage at 186-188.
         span_groups=lambda T: {"src_Atlantic": [189],
                                "src_Ocean": [190],
                                "passage_discharges_into": list(range(186, 189))},
         digit_groups=None,
         localize={"src_Atlantic": (189, 189), "src_Ocean": (190, 190),
                   "passage_discharge_frame": (186, 188)}),
]

FRACS = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0], np.float32)
BUDGET_FRAC = 0.20  # deletion/insertion AUC measured over first 20% of ranked tokens
FIXED_COUNT = 12    # also report deletion over a small fixed token budget


# --------------------------------------------------------------------------- #
# Forward-pass counter
# --------------------------------------------------------------------------- #
class Counter:
    def __init__(self):
        self.n = 0

    def add(self, k=1):
        self.n += k


# --------------------------------------------------------------------------- #
# Occlusion engine (clean model; MEAN-embedding masking)
# --------------------------------------------------------------------------- #
def build_engine(model, ids, ans_id, device, counter):
    emb = model.model.embed_tokens(ids).detach()  # [1,T,D]
    T = int(ids.shape[1])
    # GLOBAL / dataset token-mean baseline = the average token embedding over the
    # whole vocabulary, NOT the prompt-token mean. The prompt-token mean is a
    # degenerate prompt-specific vector the model attends to, which artificially
    # crushed insertion AUC; the global vocab mean is the proper neutral baseline.
    # This single source feeds the occlusion engine (deletion AND insertion), the
    # greedy/chunked conditional (via rec), gradient, and the FRI solve baseline.
    mean_emb = (
        model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
        .to(device=device, dtype=emb.dtype)
    )

    @torch.no_grad()
    def prob_for(keep):
        if not torch.is_tensor(keep):
            keep = torch.as_tensor(keep, device=device, dtype=emb.dtype)
        else:
            keep = keep.to(device=device, dtype=emb.dtype)
        e = emb * keep[None, :, None] + mean_emb * (1.0 - keep[None, :, None])
        lg = model(inputs_embeds=e).logits[0, -1]
        counter.add(1)
        return float(torch.softmax(lg, dim=-1)[ans_id])

    return emb, mean_emb, T, prob_for


def make_recovery(prob_for, T):
    p_full = prob_for(np.ones(T, np.float32))
    p_base = prob_for(np.zeros(T, np.float32))
    den = (p_full - p_base) if abs(p_full - p_base) > 1e-6 else 1e-6

    def rec(keep):
        return (prob_for(keep) - p_base) / den

    return rec, p_full, p_base


# --------------------------------------------------------------------------- #
# 1-pass single-occlusion prior (shared by greedy fill + chunked)
# --------------------------------------------------------------------------- #
def single_occlusion(rec, T, full_rec, keepset):
    """score_i = full_rec - rec(remove token i alone). Higher = more necessary.
    Cost = T forwards."""
    s = np.zeros(T, np.float32)
    for i in range(T):
        if i in keepset:
            s[i] = -1e9
            continue
        m = np.ones(T, np.float32)
        m[i] = 0.0
        s[i] = full_rec - rec(m)
    return s


# --------------------------------------------------------------------------- #
# Method 1: greedy_conditional ORACLE (early-stopped) + cheap fill for the curve
# --------------------------------------------------------------------------- #
def greedy_conditional(rec, T, full_rec, keepset, single_occ_scores, stop=0.1, max_steps=None):
    """Greedy conditional deletion until recovery <= stop*full_rec. Returns the
    necessary SET (in greedy order) and a FULL deletion order = [necessary set,
    then remaining ranked by the 1-pass single-occlusion prior] so we avoid the
    full O(T^2) ordering."""
    keep = np.ones(T, np.float32)
    nec_order = []
    rem = [i for i in range(T) if i not in keepset]
    steps = 0
    cap = max_steps if max_steps is not None else len(rem)
    while rem and steps < cap:
        best, bv = None, 1e9
        for i in rem:
            m = keep.copy()
            m[i] = 0.0
            v = rec(m)
            if v < bv:
                bv, best = v, i
        keep[best] = 0.0
        nec_order.append(best)
        rem.remove(best)
        steps += 1
        if bv <= stop * full_rec:
            break
    # fill the rest by single-occlusion magnitude (cheap, already paid)
    tail = [i for i in np.argsort(-single_occ_scores) if i not in keepset and i not in nec_order]
    full_order = np.array(nec_order + list(tail))
    return np.array(nec_order), full_order


# --------------------------------------------------------------------------- #
# Method 2: chunked_conditional (CHEAP): 1-pass prior -> top-M -> R rounds
# --------------------------------------------------------------------------- #
def chunked_conditional(rec, T, full_rec, keepset, single_occ_scores, M=80, R=6):
    """Restrict conditional greedy to the top-M single-occlusion candidates and run
    R rounds. Cost ~ M*R forwards (the T-cost of the prior is counted separately)."""
    cand = [i for i in np.argsort(-single_occ_scores) if i not in keepset][:M]
    keep = np.ones(T, np.float32)
    order = []
    pool = list(cand)
    for _ in range(min(R, len(pool))):
        best, bv = None, 1e9
        for i in pool:
            m = keep.copy()
            m[i] = 0.0
            v = rec(m)
            if v < bv:
                bv, best = v, i
        keep[best] = 0.0
        order.append(best)
        pool.remove(best)
    # remaining candidates ranked by their single-occ prior, then the rest
    rest_cand = [i for i in cand if i not in order]
    tail = [i for i in np.argsort(-single_occ_scores) if i not in keepset and i not in order and i not in rest_cand]
    full_order = np.array(order + rest_cand + tail)
    return full_order


# --------------------------------------------------------------------------- #
# Method 3: gradient (1 backward)
# --------------------------------------------------------------------------- #
def gradient_scores(model, emb, mean_emb, ans_id, T, device):
    km = torch.ones(T, device=device, requires_grad=True)
    e = emb * km[None, :, None] + mean_emb * (1.0 - km[None, :, None])
    logit = model(inputs_embeds=e).logits[0, -1, ans_id]
    g = torch.autograd.grad(logit, km)[0]
    return g.abs().detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# Method: FRI (random-budget soft-INSERTION solve) -> per-token SUFFICIENCY score
# --------------------------------------------------------------------------- #
def fri_scores(model, emb, mean_emb, ans_id, T, device, p_full, p_base,
               counter, steps=32, restarts=2, keepset=frozenset(),
               budget_mode="uniform"):
    """Exact inline FRI from the first LLM task (1D-token version of
    src.core.attribution.fri.run_fri objective_mode='random_budget_softins',
    optimizer_mode='cautious_adam_cosine'; the square-grid TV term is dropped).

    Soft keep-mask over the input tokens toward the MEAN embedding; each step draws
    a random insertion budget and maximizes the baseline-corrected answer-prob
    recovery. Returns final sigmoid(log_alpha) keep-probs as a per-token SUFFICIENCY
    score. Cost = steps*restarts forwards (each does 1 grad forward).

    budget_mode controls ONLY the per-step random insertion budget (everything else
    -- steps/restarts/cautious-Adam/lr schedule -- is identical):
      - "uniform":  b = u * n_active                 (the original; u~Uniform(0,1))
      - "annealed": b = n_active*(1 - step/(steps-1))*u + 1  (coarse->fine schedule;
        from scripts/research_fri_longseq_diagnostic.py, which restored needle
        localization on long sequences where uniform wastes steps in the
        saturated/flat regions of the step-function recovery)."""
    den = (p_full - p_base) if abs(p_full - p_base) > 1e-6 else 1e-6
    p_base_t = torch.as_tensor(p_base, device=device, dtype=torch.float32)
    den_t = torch.as_tensor(den, device=device, dtype=torch.float32)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    frozen = set(int(i) for i in keepset)
    active_idx = [i for i in range(T) if i not in frozen]
    active_t = torch.as_tensor(active_idx, device=device, dtype=torch.long)
    n_active = len(active_idx)

    def recovery(mask_active):
        mask = torch.ones(T, device=device, dtype=emb.dtype)
        mask[active_t] = mask_active.to(device=device, dtype=emb.dtype)
        e = emb * mask[None, :, None] + mean_emb * (1.0 - mask[None, :, None])
        lg = model(inputs_embeds=e).logits[0, -1]
        counter.add(1)
        prob = torch.softmax(lg, dim=-1)[ans_id]
        return (prob - p_base_t) / den_t

    def run_once(seed):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        log_a = torch.full((n_active,), 0.0, device=device, dtype=torch.float32)  # logit(0.5)=0
        m_v = torch.zeros(n_active, device=device)
        v_v = torch.zeros(n_active, device=device)
        for step in range(steps):
            frac = step / max(steps - 1, 1)
            cur_lr = 0.01 + 0.5 * (0.45 - 0.01) * (1 + np.cos(np.pi * frac))
            la = log_a.clone().requires_grad_(True)
            probs = torch.sigmoid(la)
            p = probs / (probs.sum() + 1e-8)  # budget distribution
            u = float(torch.rand(1, generator=gen, device=device).item())
            if budget_mode == "uniform":
                budget = u * n_active
            elif budget_mode == "annealed":
                budget = n_active * (1.0 - step / max(steps - 1, 1)) * u + 1.0
            else:
                raise ValueError(f"unknown budget_mode {budget_mode!r}")
            w = (p * budget).clamp(max=1.0)  # soft insertion keep-weights
            rec = recovery(w)
            loss = 1.0 - rec
            g = torch.autograd.grad(loss, la)[0].detach()
            t = step + 1
            m_v = beta1 * m_v + (1 - beta1) * g
            v_v = beta2 * v_v + (1 - beta2) * g * g
            m_hat = m_v / (1 - beta1 ** t)
            v_hat = v_v / (1 - beta2 ** t)
            adam_dir = m_hat / (v_hat.sqrt() + eps)
            mask = (adam_dir * g > 0).float()  # cautious: keep aligned coords
            n_update = mask.sum().clamp(min=1.0)
            step_dir = adam_dir * mask * (len(active_idx) / n_update)
            log_a = log_a - cur_lr * step_dir
        return torch.sigmoid(log_a).detach()

    best, best_obj = None, -1e9
    for r in range(max(1, restarts)):
        s = run_once(9973 * r)
        with torch.no_grad():
            obj = float(recovery(s))
        if best is None or obj > best_obj:
            best, best_obj = s, obj
    scores = np.zeros(T, np.float32)
    scores[active_idx] = best.cpu().numpy().astype(np.float32)
    for i in frozen:
        if 0 <= i < T:
            scores[i] = -1e9
    return scores


# --------------------------------------------------------------------------- #
# Method 4: AttnLRP via lxt (separate monkey-patched model)
# --------------------------------------------------------------------------- #
ATTNLRP_WORKER = r'''
import sys, json
import numpy as np
import torch
repo, path, device, ans_id = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ids = json.loads(sys.argv[5])
out_path = sys.argv[6]
sys.path.insert(0, repo)
from transformers.models.qwen2 import modeling_qwen2
from lxt.efficient import monkey_patch
monkey_patch(modeling_qwen2, verbose=False)
model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
input_ids = torch.tensor([ids], device=device)
input_embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
logits = model(inputs_embeds=input_embeds, use_cache=False).logits
target = logits[0, -1, ans_id]
target.backward()
rel = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0].numpy()
np.save(out_path, rel.astype(np.float32))
print("ATTNLRP_OK")
'''


def attnlrp_scores(ids, ans_id, path, device):
    """Compute AttnLRP relevance in a SUBPROCESS so lxt.efficient.monkey_patch
    (which mutates the GLOBAL modeling_qwen2 classes) cannot contaminate the clean
    occlusion model in this process. relevance = (input_embeds.grad*input_embeds)
    .sum(-1) after backprop from the answer logit, on a float32 monkey-patched model
    (the example's bf16+4bit was only for memory; float32 is fairer vs our methods).
    """
    import subprocess
    import tempfile

    id_list = ids[0].tolist()
    with tempfile.TemporaryDirectory() as td:
        worker_py = os.path.join(td, "worker.py")
        out_npy = os.path.join(td, "rel.npy")
        with open(worker_py, "w") as f:
            f.write(ATTNLRP_WORKER)
        cmd = [sys.executable, worker_py, REPO_DIR, path, str(device), str(ans_id),
               json.dumps(id_list), out_npy]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                             cwd=REPO_DIR)
        if "ATTNLRP_OK" not in res.stdout or not os.path.exists(out_npy):
            raise RuntimeError(
                f"AttnLRP worker failed (rc={res.returncode}).\n"
                f"STDOUT tail: {res.stdout[-500:]}\nSTDERR tail: {res.stderr[-800:]}"
            )
        rel = np.load(out_npy)
    return rel.astype(np.float32)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def deletion_auc(rec, order, T, keepset, budget):
    """Mask the first `budget` ranked content tokens, in order. LOWER = better."""
    order = [i for i in order if i not in keepset][:budget]
    if not order:
        return float("nan")
    fr_idx = (FRACS * len(order)).round().astype(int)
    recs = []
    for kk in fr_idx:
        keep = np.ones(T, np.float32)
        if kk > 0:
            keep[order[:kk]] = 0.0
        recs.append(rec(keep))
    return float(np.trapz(recs, FRACS))


def insertion_auc(rec, order, T, keepset, budget):
    """Insert (keep) the first `budget` ranked content tokens; rest stay MEAN.
    Frozen readout kept from start. HIGHER = better sufficiency."""
    order = [i for i in order if i not in keepset][:budget]
    if not order:
        return float("nan")
    fr_idx = (FRACS * len(order)).round().astype(int)
    recs = []
    for kk in fr_idx:
        keep = np.zeros(T, np.float32)
        for j in keepset:
            keep[j] = 1.0
        if kk > 0:
            keep[order[:kk]] = 1.0
        recs.append(rec(keep))
    return float(np.trapz(recs, FRACS))


# --------------------------------------------------------------------------- #
# Redundancy probe
# --------------------------------------------------------------------------- #
def redundancy_probe(rec, T, toks, groups_idx, keepset):
    """Single-group vs joint-group deletion of the answer prob."""
    res = {"groups": {}}
    all_idx = []
    for name, idxs in groups_idx.items():
        idxs = [i for i in idxs if i not in keepset]
        all_idx += idxs
        keep = np.ones(T, np.float32)
        for i in idxs:
            keep[i] = 0.0
        res["groups"][name] = dict(
            idx=idxs, toks=[toks[i] for i in idxs], rec_after_delete=float(rec(keep))
        )
    keep = np.ones(T, np.float32)
    for i in all_idx:
        keep[i] = 0.0
    res["rec_after_joint_delete"] = float(rec(keep))
    singles = [g["rec_after_delete"] for g in res["groups"].values()]
    res["min_single_rec"] = float(min(singles)) if singles else float("nan")
    # redundancy gap = how much MORE the joint deletion destroys vs the worst single
    res["redundancy_gap"] = (
        float(min(singles) - res["rec_after_joint_delete"]) if singles else float("nan")
    )
    # Redundancy signal: EVERY single cue is individually deletable (the answer
    # survives removing any one), yet jointly removing them destroys it far more
    # (large gap). This is the regime where single-pass attributions UNDERESTIMATE
    # necessity. We do NOT require joint->0 (partial redundancy w/ world-knowledge
    # fallback still counts), only: worst single stays high AND a large joint gap.
    res["is_redundant"] = bool(
        len(singles) >= 2
        and min(singles) > 0.5           # even the WORST single cue leaves answer mostly intact
        and res["redundancy_gap"] > 0.3  # joint deletion is much more destructive
    )
    return res


def run_question(q, model, tok, args):
    """Run all methods + probes for one question; return a result dict."""
    ids = tok(q["text"], return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)
    T = int(ids.shape[1])
    toks = [tok.decode(t) for t in ids[0].tolist()]
    keepset = frozenset({T - 1})  # freeze readout (last) position

    with torch.no_grad():
        full_logits = model(ids).logits[0, -1]
        full_probs = torch.softmax(full_logits, dim=-1)
    ans_id = int(full_logits.argmax())  # target the model's own argmax
    pred_tok = tok.decode(ans_id)
    ans_prob = float(full_probs[ans_id])

    print("\n" + "=" * 92)
    print(f"### {q['name']} | T={T} | argmax@last={pred_tok!r} (id {ans_id}) prob={ans_prob:.4f}")
    print("=" * 92)

    counter = Counter()
    emb, mean_emb, _, prob_for = build_engine(model, ids, ans_id, args.device, counter)
    rec, p_full, p_base = make_recovery(prob_for, T)
    full_rec = rec(np.ones(T, np.float32))

    # ---- shared 1-pass single-occlusion prior (cost = T) ----
    c0 = counter.n
    single_occ = single_occlusion(rec, T, full_rec, keepset)
    cost_prior = counter.n - c0

    # ---- Method 1: greedy oracle (early-stopped) ----
    c0 = counter.n
    nec_order, greedy_full = greedy_conditional(
        rec, T, full_rec, keepset, single_occ, stop=0.1, max_steps=args.greedy_max_steps
    )
    cost_greedy = counter.n - c0
    nec_set = list(nec_order)

    # ---- Method 2: chunked conditional (cheap) ----
    c0 = counter.n
    chunk_full = chunked_conditional(
        rec, T, full_rec, keepset, single_occ, M=args.chunk_M, R=args.chunk_R
    )
    cost_chunk = counter.n - c0

    # ---- Method 3: gradient (1 backward) ----
    t0 = time.time()
    grad_s = gradient_scores(model, emb, mean_emb, ans_id, T, args.device)
    grad_order = np.array([i for i in np.argsort(-grad_s) if i not in keepset])
    t_grad = time.time() - t0

    # ---- Method FRI: random-budget soft-insertion solve (SUFFICIENCY score) ----
    # Two budget variants (everything else identical): UNIFORM (original) vs
    # ANNEALED (coarse->fine; restores needle localization on long sequences).
    c0 = counter.n
    fri_u_s = fri_scores(model, emb, mean_emb, ans_id, T, args.device, p_full, p_base,
                         counter, steps=args.fri_steps, restarts=2, keepset=keepset,
                         budget_mode="uniform")
    cost_fri_u = counter.n - c0
    fri_u_order = np.array([i for i in np.argsort(-fri_u_s) if i not in keepset])

    c0 = counter.n
    fri_a_s = fri_scores(model, emb, mean_emb, ans_id, T, args.device, p_full, p_base,
                         counter, steps=args.fri_steps, restarts=2, keepset=keepset,
                         budget_mode="annealed")
    cost_fri_a = counter.n - c0
    fri_a_order = np.array([i for i in np.argsort(-fri_a_s) if i not in keepset])

    # ---- Method 4: AttnLRP (isolated subprocess) ----
    try:
        t0 = time.time()
        attn_s = attnlrp_scores(ids, ans_id, args.model, args.device)
        attn_order = np.array([i for i in np.argsort(-attn_s) if i not in keepset])
        t_attn = time.time() - t0
        attn_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  (AttnLRP failed: {type(e).__name__}: {e})")
        attn_order, t_attn, attn_ok = None, float("nan"), False

    socc_order = np.array([i for i in np.argsort(-single_occ) if i not in keepset])
    n_content = T - len(keepset)
    budget = int(round(BUDGET_FRAC * n_content))

    methods = {
        "greedy_oracle": (greedy_full, cost_prior + cost_greedy, "T(prior)+greedy rounds"),
        "chunked_cond": (chunk_full, cost_prior + cost_chunk,
                         f"T(prior)+M*R≈{args.chunk_M}*{args.chunk_R}"),
        "FRI_uniform": (fri_u_order, cost_fri_u, f"{args.fri_steps}st*2r softins(uniform b)"),
        "FRI_annealed": (fri_a_order, cost_fri_a, f"{args.fri_steps}st*2r softins(annealed b)"),
        "gradient": (grad_order, 1, "1 backward"),
    }
    if attn_ok:
        methods["attnlrp"] = (attn_order, 1, "1 backward (LRP)")
    methods["single_occ"] = (socc_order, cost_prior, "T (1-pass occlusion)")

    rows = {}
    for nm, (order, fwd, costdesc) in methods.items():
        rows[nm] = dict(
            deletion_auc=deletion_auc(rec, order, T, keepset, budget),
            insertion_auc=insertion_auc(rec, order, T, keepset, budget),
            deletion_auc_fixed=deletion_auc(rec, order, T, keepset, FIXED_COUNT),
            attribution_forwards=int(fwd), cost_desc=costdesc,
            top_necessary=[toks[i] for i in list(order)[:8]],
            top_necessary_idx=[int(i) for i in list(order)[:8]],
        )

    # ---- redundancy probes (per-question spans) ----
    redun_main = redundancy_probe(rec, T, toks, q["span_groups"](T), keepset)
    redun_digits = (
        redundancy_probe(rec, T, toks, q["digit_groups"](T), keepset)
        if q.get("digit_groups") else None
    )

    # ---- print ----
    print(f"necessary SET (greedy oracle, {len(nec_set)} toks, in order):")
    print(f"  idx  {nec_set}")
    print(f"  toks {[toks[i] for i in nec_set]}")
    print(f"\n{'method':14s} {'del_auc20(↓)':>12s} {'del_fix(↓)':>11s} {'ins_auc20(↑)':>12s} "
          f"{'#fwd':>7s}  cost")
    for nm in methods:
        r = rows[nm]
        print(f"{nm:14s} {r['deletion_auc']:12.3f} {r['deletion_auc_fixed']:11.3f} "
              f"{r['insertion_auc']:12.3f} {r['attribution_forwards']:7d}  {r['cost_desc']}")
    print(f"\ntop necessary tokens per method (rank 1..8):")
    for nm in methods:
        print(f"  {nm:14s} {rows[nm]['top_necessary']}")

    print(f"\nREDUNDANCY probe (cue groups, single vs joint deletion):")
    for g, gv in redun_main["groups"].items():
        print(f"  delete {g:18s} {gv['toks']} -> recovery {gv['rec_after_delete']:+.3f}")
    print(f"  delete ALL groups jointly -> recovery {redun_main['rec_after_joint_delete']:+.3f}  "
          f"gap(worst_single-joint)={redun_main['redundancy_gap']:+.3f}  "
          f"redundant={redun_main['is_redundant']}")
    if redun_digits:
        print(f"REDUNDANCY (copied answer digits):")
        for g, gv in redun_digits["groups"].items():
            print(f"  delete {g} {gv['toks']} -> recovery {gv['rec_after_delete']:+.3f}")
        print(f"  delete ALL digits -> recovery {redun_digits['rec_after_joint_delete']:+.3f}  "
              f"redundant={redun_digits['is_redundant']}")

    verdict = build_verdict(q, rows, methods, nec_set, toks, redun_main, redun_digits,
                            cost_prior, cost_greedy, cost_chunk, T, args)
    print("\nVERDICT:")
    for v in verdict:
        print(f"  - {v}")

    return dict(
        name=q["name"], text=q["text"], T=T, argmax=pred_tok, answer_id=ans_id,
        answer_prob=ans_prob, p_full=p_full, p_base=p_base,
        budget_frac=BUDGET_FRAC, budget=budget, fixed_count=FIXED_COUNT,
        keepset=[int(i) for i in keepset],
        necessary_set_idx=[int(i) for i in nec_set],
        necessary_set_toks=[toks[i] for i in nec_set],
        cost_prior_forwards=cost_prior, cost_greedy_rounds_forwards=cost_greedy,
        cost_chunk_forwards=cost_chunk,
        full_oracle_forwards_estimate=int(T * (T - 1) // 2),
        grad_seconds=t_grad, attn_seconds=t_attn,
        methods=rows, redundancy=redun_main, redundancy_digits=redun_digits,
        verdict=verdict,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--greedy-max-steps", type=int, default=12,
                    help="cap on oracle greedy steps (each costs ~remaining forwards)")
    ap.add_argument("--chunk-M", type=int, default=80)
    ap.add_argument("--chunk-R", type=int, default=6)
    ap.add_argument("--fri-steps", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        .to(args.device)
        .eval()
    )
    for p in model.parameters():
        p.requires_grad = False

    print("=" * 92)
    print(f"LONG-CONTEXT LLM necessity: cost + redundancy | OUR conditional vs AttnLRP")
    print(f"model={args.model} | masking=GLOBAL vocab-mean embedding (weight.mean(0), "
          f"NOT prompt-mean); readout frozen; no BOS")
    print(f"deletion/insertion AUC over first {int(BUDGET_FRAC*100)}% of ranked tokens "
          f"+ fixed-count={FIXED_COUNT}; LOWER del=better necessity")
    print("=" * 92)

    results = [run_question(q, model, tok, args) for q in QUESTIONS]

    summary = dict(model=args.model, budget_frac=BUDGET_FRAC, fixed_count=FIXED_COUNT,
                   questions=results)
    out = args.out or os.path.join(
        os.path.dirname(__file__), "..", "outputs", "llm_necessary_long_context.json"
    )
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out.replace(".json", ".txt"), "w") as f:
        f.write(format_txt(summary))
    print(f"\nsaved JSON -> {out}")
    print(f"saved TXT  -> {out.replace('.json', '.txt')}")


def build_verdict(q, rows, methods, nec_set, toks, redun_main, redun_digits,
                  cost_prior, cost_greedy, cost_chunk, T, args):
    lines = []
    full_est = T * (T - 1) // 2
    oracle_fwd = rows["greedy_oracle"]["attribution_forwards"]
    chunk_fwd = rows["chunked_cond"]["attribution_forwards"]
    # (a) cost
    lines.append(
        f"COST: full O(T^2) deletion order ~{full_est} forwards; early-stopped oracle "
        f"used {oracle_fwd} ({cost_greedy} greedy rounds + {cost_prior} prior); "
        f"chunked used {chunk_fwd} (~{chunk_fwd/max(oracle_fwd,1):.2f}x oracle, "
        f"{chunk_fwd/max(full_est,1)*100:.1f}% of full O(T^2))."
    )
    od = rows["greedy_oracle"]["deletion_auc"]
    cd = rows["chunked_cond"]["deletion_auc"]
    lines.append(
        f"COST PAYOFF: chunked del_auc20={cd:.3f} vs oracle={od:.3f} (Δ={cd-od:+.3f}) -> "
        + ("chunked MATCHES the oracle at a fraction of the cost (long-context payoff)."
           if abs(cd - od) <= 0.03 else
           "chunked approximates the oracle but with a gap.")
    )
    # (b) redundancy (generic over the question's cue groups)
    rm = redun_main
    grp_str = " / ".join(
        f"{g}={rm['groups'][g]['rec_after_delete']:+.2f}" for g in rm["groups"]
    )
    lines.append(
        f"REDUNDANCY (cue groups): delete-one [{grp_str}] vs delete-all="
        f"{rm['rec_after_joint_delete']:+.2f}, gap(worst_single-joint)={rm['redundancy_gap']:+.2f} -> "
        + (f"REDUNDANT: worst single-cue deletion still leaves recovery "
           f"{rm['min_single_rec']:.2f} but joint deletion drops to "
           f"{rm['rec_after_joint_delete']:.2f} => single-token/single-pass attributions "
           f"(gradient/AttnLRP) UNDERESTIMATE the joint necessity."
           if rm["is_redundant"] else
           "NOT redundant here -- at least one cue is individually necessary "
           "(sharp single-source necessity).")
    )
    if redun_digits:
        rd = redun_digits
        gd_str = " / ".join(
            f"{g}={rd['groups'][g]['rec_after_delete']:+.2f}" for g in rd["groups"]
        )
        lines.append(
            f"REDUNDANCY (answer digits): [{gd_str}] vs all={rd['rec_after_joint_delete']:+.2f} -> "
            + ("a single leading digit is the necessary copy-source (sharp, not redundant)."
               if rd["min_single_rec"] < 0.3 else "digits partially redundant.")
        )
    # (c) do our conditional methods beat gradient / AttnLRP at deletion?
    gd_auc = rows["gradient"]["deletion_auc"]
    gd_fix = rows["gradient"]["deletion_auc_fixed"]
    ad = rows["attnlrp"]["deletion_auc"] if "attnlrp" in rows else float("nan")
    best_ours = min(od, cd)
    beat = []
    if best_ours < gd_auc - 1e-6:
        beat.append(f"gradient({gd_auc:.3f})")
    if "attnlrp" in rows and best_ours < ad - 1e-6:
        beat.append(f"AttnLRP({ad:.3f})")
    lines.append(
        f"METRIC: best OUR conditional del_auc20={best_ours:.3f}; gradient={gd_auc:.3f} "
        f"(fixed={gd_fix:.3f})" + (f", AttnLRP={ad:.3f}" if "attnlrp" in rows else "")
        + (f" -> OUR conditional BEATS {', '.join(beat)} at necessity."
           if beat else " -> cheap single-pass already matches our conditional here.")
    )
    # (d) TWO-PILLAR asymmetry: who tops DELETION (necessity) vs INSERTION (suff)?
    del_leader = min(rows.items(), key=lambda kv: kv[1]["deletion_auc"])
    ins_leader = max(rows.items(), key=lambda kv: kv[1]["insertion_auc"])
    cond = {"greedy_oracle", "chunked_cond", "single_occ"}
    fu_ins = rows["FRI_uniform"]["insertion_auc"]
    fa_ins = rows["FRI_annealed"]["insertion_auc"]
    fa_del = rows["FRI_annealed"]["deletion_auc"]
    ins_top = ins_leader[1]["insertion_auc"]
    fa_ins_rank = 1 + sum(1 for r in rows.values() if r["insertion_auc"] > fa_ins)
    lines.append(
        f"TWO-PILLAR: DELETION(↓nec) topped by {del_leader[0]}({del_leader[1]['deletion_auc']:.3f}); "
        f"INSERTION(↑suff) topped by {ins_leader[0]}({ins_top:.3f})."
    )
    # Q1: annealed vs uniform on insertion
    impr = (fa_ins / fu_ins) if fu_ins > 1e-6 else float("inf")
    lines.append(
        f"FRI annealed vs uniform INSERTION: uniform={fu_ins:.3f} -> annealed={fa_ins:.3f} "
        f"(Δ={fa_ins-fu_ins:+.3f}, {impr:.1f}x) -> "
        + ("annealed budget IMPROVES FRI sufficiency."
           if fa_ins > fu_ins + 1e-3 else "annealed does NOT improve over uniform here.")
    )
    # Q2: is FRI_annealed the TOP insertion method, or do AttnLRP/greedy still beat it?
    fa_is_top = ins_leader[0] == "FRI_annealed"
    lines.append(
        f"FRI_annealed insertion rank = {fa_ins_rank}/{len(rows)} -> "
        + (f"FRI_annealed is the TOP sufficiency method ({fa_ins:.3f})."
           if fa_is_top else
           f"FRI_annealed is NOT top -- {ins_leader[0]} still leads insertion "
           f"({ins_top:.3f} vs FRI_annealed {fa_ins:.3f}).")
    )
    # Q3: do conditional methods stay top on DELETION (necessity)?
    cond_stays = del_leader[0] in cond
    lines.append(
        f"DELETION leader = {del_leader[0]} -> "
        + ("a CONDITIONAL method stays top on necessity (as expected)."
           if cond_stays else
           f"a NON-conditional method ({del_leader[0]}) tops deletion here "
           f"(conditional best = {min(od, cd):.3f}).")
    )
    # overall asymmetry verdict
    asym = fa_is_top and cond_stays
    lines.append(
        "ASYMMETRY VERDICT (annealed): "
        + ("HOLDS -- FRI_annealed(=sufficiency) leads INSERTION while a CONDITIONAL "
           "method(=necessity) leads DELETION."
           if asym else
           f"MIXED -- insertion leader={ins_leader[0]} (FRI_annealed rank {fa_ins_rank}), "
           f"deletion leader={del_leader[0]}; FRI_annealed weak at deletion "
           f"({fa_del:.3f} vs conditional {min(od,cd):.3f}).")
    )
    # necessary-set localization (generic from q['localize'])
    nec_toks = [toks[i].strip() for i in nec_set]
    lines.append(f"NECESSARY SET = {nec_toks}")
    hits = []
    for name, (lo, hi) in q.get("localize", {}).items():
        got = [toks[i] for i in nec_set if lo <= i <= hi]
        hits.append(f"{name}={got}")
    lines.append("necessary-set span hits: " + ", ".join(hits))
    return lines


def format_txt(s):
    out = []
    out.append(f"LONG-CONTEXT LLM necessity: cost + redundancy | OUR conditional vs AttnLRP")
    out.append(f"model={s['model']}; GLOBAL vocab-mean masking (weight.mean(0)); readout frozen; no BOS")
    out.append(f"deletion/insertion AUC over first {int(s['budget_frac']*100)}% of ranked tokens "
               f"+ fixed-count={s['fixed_count']}; LOWER del=better necessity")
    out.append("=" * 84)
    for q in s["questions"]:
        out.append("")
        out.append(f"### {q['name']} | T={q['T']} | argmax={q['argmax']!r} prob={q['answer_prob']:.4f}")
        out.append(f"necessary SET ({len(q['necessary_set_idx'])}): {q['necessary_set_toks']}  "
                   f"(idx {q['necessary_set_idx']})")
        out.append(f"full O(T^2) oracle estimate = {q['full_oracle_forwards_estimate']} forwards")
        out.append(f"{'method':14s} {'del_auc20':>10s} {'del_fixed':>10s} {'ins_auc20':>10s} "
                   f"{'#fwd':>8s}  cost")
        for nm, r in q["methods"].items():
            out.append(f"{nm:14s} {r['deletion_auc']:10.3f} {r['deletion_auc_fixed']:10.3f} "
                       f"{r['insertion_auc']:10.3f} {r['attribution_forwards']:8d}  {r['cost_desc']}")
        out.append("top necessary tokens per method:")
        for nm, r in q["methods"].items():
            out.append(f"  {nm:14s} {r['top_necessary']}")
        rm = q["redundancy"]
        out.append("redundancy (cue groups):")
        for g, gv in rm["groups"].items():
            out.append(f"  del {g} {gv['toks']} -> rec {gv['rec_after_delete']:+.3f}")
        out.append(f"  del all -> {rm['rec_after_joint_delete']:+.3f} gap={rm['redundancy_gap']:+.3f} "
                   f"redundant={rm['is_redundant']}")
        if q.get("redundancy_digits"):
            rd = q["redundancy_digits"]
            out.append("redundancy (answer digits):")
            for g, gv in rd["groups"].items():
                out.append(f"  del {g} {gv['toks']} -> rec {gv['rec_after_delete']:+.3f}")
        out.append("verdict:")
        for v in q["verdict"]:
            out.append(f"  - {v}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
