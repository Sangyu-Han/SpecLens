#!/usr/bin/env python3
"""CHALLENGE AttnLRP on LLM sufficiency with a stochastic COALITION estimator (from scratch).

Hypothesis: AttnLRP wins my mask-optimization FRI because it is a global one-pass relevance,
but it is a LINEAR (1st-order) attribution -> a coalition-based estimator that captures token
INTERACTIONS + directly estimates marginal sufficiency should beat it.

Banzhaf/Shapley-stochastic: score_i = E_S[rec(S+i) - rec(S)] estimated CHEAPLY as
  E[rec | i in S] - E[rec | i not in S]   over M random-BUDGET hard masks (p~U(0,1) per mask).
On-manifold (hard masks), interaction-aware (coalition marginals), same vocab-mean baseline as
the insertion metric, random-budget preserved.

ANALYSIS baselines: single-token marginal (order-1 sufficiency = rec({i})-base; ignores
interactions) vs Banzhaf (coalition) vs AttnLRP -> if Banzhaf >> single, interactions matter.
Metric: standard input-token insertion AUC (FR grid, global vocab-mean). Cases: Everest copy
(AttnLRP ref 0.664) + IMDB distributed (AttnLRP ref 0.541)."""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import BINS, PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "third_party/LRP-eXplains-Transformers-main (1)/LRP-eXplains-Transformers-main/examples/quantized_qwen2.py"
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]


def load_everest_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-imdb", type=int, default=6)
    ap.add_argument("--M", type=int, default=3000, help="Banzhaf coalition samples")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    # ---- build cases: (key, prompt, ans_id or None=argmax) ----
    cases = [("everest", load_everest_prompt(), None)]
    raw, _ = load_imdb(args.n_imdb * 8, 1400)
    per_bin = max(1, args.n_imdb // len(BINS)); binc = {b[0]: 0 for b in BINS}
    for j, (label, text) in enumerate(raw):
        prompt = PROMPT_TMPL.format(text=text)
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1])
        b = next(n for n, lo, hi in BINS if lo <= T < hi)
        if binc[b] >= per_bin + 1:
            continue
        with torch.no_grad():
            pred = int(model(ids).logits[0, -1].argmax())
        gold = pos_id if label == 1 else neg_id
        if pred == gold:
            cases.append((f"r{j}", prompt, gold)); binc[b] += 1
        if len(cases) >= 1 + args.n_imdb and all(v >= 1 for v in binc.values()):
            break
    print(f"cases: {[c[0] for c in cases]}", flush=True)

    print("AttnLRP precompute (offload main to CPU)...", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in cases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()
    print(f"AttnLRP for {len(attn_cache)}/{len(cases)}", flush=True)

    def run(prompt, ans_id):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            logit = model(inputs_embeds=emb).logits[0, -1]
            ans = int(logit.argmax()) if ans_id is None else int(ans_id)
            full = float(F.softmax(logit, -1)[ans])
            base = float(F.softmax(model(inputs_embeds=emb * 0 + gmean).logits[0, -1], -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def probs_batch(Z):  # Z [B,T] -> recovery [B]
            e = emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])
            pr = F.softmax(model(inputs_embeds=e).logits[:, -1], -1)[:, ans]
            return (pr - base) / den

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(probs_batch(m[None])[0])

        def ins_auc(scores):
            order = [i for i in np.argsort(-scores) if i != readout]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        # ---- Banzhaf/Shapley-stochastic: random-budget coalition marginal ----
        # ANTITHETIC variance reduction: each random mask z is paired with its complement
        # (1-z) over content tokens -> every token gets one in-S and one out-S sample per pair,
        # halving the difference-of-means variance (the long-context failure mode).
        Zs = []; Rs = []; j = 0
        while j < args.M:
            nb = max(1, min(args.bs, args.M - j) // 2); zb = []
            for _ in range(nb):
                pf = float(torch.rand(1, generator=gen, device=dev).item())   # random budget
                z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[readout] = 1.0
                za = 1.0 - z; za[readout] = 1.0                                # ANTITHETIC pair
                zb.append(z); zb.append(za)
            Z = torch.stack(zb)
            Rs.append(probs_batch(Z)); Zs.append(Z); j += len(zb)
        Z = torch.cat(Zs); R = torch.cat(Rs)                       # [M,T],[M]
        n1 = Z.sum(0); s1 = (Z * R[:, None]).sum(0) / n1.clamp(min=1)
        s0 = ((1 - Z) * R[:, None]).sum(0) / (args.M - n1).clamp(min=1)
        banzhaf = (s1 - s0).cpu().numpy()

        # ---- single-token marginal (order-1, ignores interactions) ----
        single = np.zeros(T, np.float32)
        for s in range(0, T, args.bs):
            idx = list(range(s, min(s + args.bs, T)))
            Z = torch.zeros(len(idx), T, device=dev); Z[:, readout] = 1.0
            for r, i in enumerate(idx):
                Z[r, i] = 1.0
            single[idx] = probs_batch(Z).cpu().numpy()

        a = attn_cache.get(k_cur)
        ins_attn = ins_auc(np.asarray(a, np.float32)) if a is not None and len(a) == T else float("nan")
        ib, isg = ins_auc(banzhaf), ins_auc(single)
        rc = spearmanr(banzhaf, np.asarray(a, np.float32)).correlation if a is not None and len(a) == T else float("nan")
        return dict(T=T, ins_banzhaf=ib, ins_single=isg, ins_attn=ins_attn,
                    corr_bz_attn=float(rc), cost_banzhaf=args.M)

    rows = []
    for k_cur, prompt, ans_id in cases:
        r = run(prompt, ans_id); r["key"] = k_cur; rows.append(r)
        print(f"  {k_cur:8s} T={r['T']:4d} | banzhaf={r['ins_banzhaf']:.3f} attn={r['ins_attn']:.3f} "
              f"single={r['ins_single']:.3f} | BZ-attn={r['ins_banzhaf']-r['ins_attn']:+.3f} "
              f"corr(BZ,attn)={r['corr_bz_attn']:+.2f}", flush=True)

    print("\n=== MEAN insertion AUC (Banzhaf coalition vs AttnLRP vs single-token order-1) ===")
    ev = [r for r in rows if r["key"] == "everest"]; im = [r for r in rows if r["key"] != "everest"]
    for nm, rs in [("everest", ev), ("imdb", im), ("ALL", rows)]:
        if not rs:
            continue
        bz = np.mean([r["ins_banzhaf"] for r in rs]); at = np.nanmean([r["ins_attn"] for r in rs])
        sg = np.mean([r["ins_single"] for r in rs])
        wr = 100.0 * np.nanmean([float(r["ins_banzhaf"] > r["ins_attn"]) for r in rs if r["ins_attn"] == r["ins_attn"]])
        print(f"  {nm:8s} n={len(rs):2d} | banzhaf={bz:.3f} attnlrp={at:.3f} single={sg:.3f} | BZ>attn {wr:.0f}%", flush=True)
    print(f"(Banzhaf cost = M={args.M} forwards/case; AttnLRP = 1 backward. refs: input_LN FRI ev0.422/im0.337)")
    print("(if banzhaf > attnlrp -> the coalition estimator beats AttnLRP; if banzhaf >> single -> interactions are the reason)")
    import json
    json.dump(rows, open(os.path.join(REPO, "outputs", "banzhaf_vs_attnlrp.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
