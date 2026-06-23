#!/usr/bin/env python3
"""Is the LLM token-gradient GENUINELY uninformative, or did it only LOOK bad because we evaluated
it at degenerate (heavily-masked) states? User's hypothesis: compute the gradient at the FULL /
prediction-preserved state and it may carry signal.

For each uniform-mask level a (input = emb*a + mean*(1-a), w = a*ones):
  - recovery (prob) + next-token entropy  -> is the prediction PRESERVED at this a?
  - gradient g_i = d(logit_ans)/d(w_i)  [one backward at w=a]  -> the directional token-gradient
  - corr(g, OCCLUSION ground-truth Δ_i = full_logit − logit(token i→mean))  -> is g informative?
  - insertion AUC of g (standard mean-baseline metric)
Compare across a ∈ {1.0(full),0.95,0.8,0.5,0.3} and vs AttnLRP. If corr is HIGH at high-a
(prediction preserved) -> the gradient IS informative there (user right; the masked-state was the
problem). If corr is LOW even at a=1 -> the RAW gradient is genuinely uninformative (LRP needed)."""
from __future__ import annotations

import argparse
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
AS = [1.0, 0.95, 0.8, 0.5, 0.3]


def load_everest_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-imdb", type=int, default=2)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    dev = args.device
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]

    def llq_ng(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    raw_all, _ = load_imdb(400, 2000)
    cases = [("everest", load_everest_prompt(), None)]
    per_bin = max(1, args.n_imdb // len(BINS)); binc = {b[0]: 0 for b in BINS}
    for j, (label, text) in enumerate(raw_all):
        ids = tok(PROMPT_TMPL.format(text=text), return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); b = next(n for n, lo, hi in BINS if lo <= T < hi)
        if binc[b] >= per_bin + 1:
            continue
        with torch.no_grad():
            pred = int(llq_ng(model.get_input_embeddings()(ids))[0].argmax())
        if pred == (pos_id if label == 1 else neg_id):
            cases.append((f"r{j}", PROMPT_TMPL.format(text=text), pos_id if label == 1 else neg_id)); binc[b] += 1
        if len(cases) >= 1 + args.n_imdb and all(v >= 1 for v in binc.values()):
            break
    print(f"cases: {[c[0] for c in cases]}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in cases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans_id):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1
        emb = model.get_input_embeddings()(ids).detach()
        nz = [i for i in range(T) if i not in (0, readout)]
        with torch.no_grad():
            lf = llq_ng(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_logit = float(lf[ans]); full_p = float(F.softmax(lf, -1)[ans])
            base_p = float(F.softmax(llq_ng(emb * 0 + gmean)[0], -1)[ans])
        den = max(full_p - base_p, 1e-6)

        # OCCLUSION ground truth: Δ_i = full_logit − logit(token i -> mean), per token (one at a time)
        occ = np.zeros(T, np.float32)
        with torch.no_grad():
            for s in range(0, len(nz), 16):
                idx = nz[s:s + 16]; B = len(idx)
                e = emb.expand(B, -1, -1).clone()
                for r, i in enumerate(idx):
                    e[r, i] = gmean[0, 0]
                lg = llq_ng(e)[:, ans]
                for r, i in enumerate(idx):
                    occ[i] = full_logit - float(lg[r])

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            e = emb * m[None, :, None] + gmean * (1 - m[None, :, None])
            return float((F.softmax(llq_ng(e)[0], -1)[ans] - base_p) / den)

        def ins_auc(scores):
            order = [i for i in np.argsort(-scores) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        rows = {}
        for a in AS:
            w = torch.full((T,), float(a), device=dev, requires_grad=True)
            e = emb * w[None, :, None] + gmean * (1 - w[None, :, None])
            logit = model(inputs_embeds=e).logits[0, -1, ans]
            g = torch.autograd.grad(logit, w)[0].detach().cpu().numpy()
            with torch.no_grad():
                sm = F.softmax(model(inputs_embeds=emb * a + gmean * (1 - a)).logits[0, -1], -1)
                rec_a = float((sm[ans] - base_p) / den); ent_a = float(-(sm * torch.log(sm + 1e-12)).sum())
            rows[a] = dict(rec=rec_a, ent=ent_a, corr=float(spearmanr(g[nz], occ[nz]).correlation),
                           gmag=float(np.abs(g[nz]).mean()), auc=ins_auc(g))
        att = attn_cache.get(key)
        att_corr = float(spearmanr(np.asarray(att, np.float32)[nz], occ[nz]).correlation) if att is not None and len(att) == T else float("nan")
        att_auc = ins_auc(np.asarray(att, np.float32)) if att is not None and len(att) == T else float("nan")
        return dict(T=T, rows=rows, att_corr=att_corr, att_auc=att_auc, occ_auc=ins_auc(occ))

    results = []
    for key, prompt, ans_id in cases:
        r = run(key, prompt, ans_id); r["key"] = key; results.append(r)
        print(f"  {key} T={r['T']} done", flush=True)

    print("\n=== gradient @ uniform-mask level a: is the prediction preserved AND the gradient informative? ===")
    print(f"{'a':>5s} | {'recovery':>8s} {'entropy':>7s} | {'corr(grad,occ)':>14s} {'|grad|':>9s} {'grad insAUC':>11s}")
    for a in AS:
        rec = np.mean([r["rows"][a]["rec"] for r in results]); ent = np.mean([r["rows"][a]["ent"] for r in results])
        corr = np.nanmean([r["rows"][a]["corr"] for r in results]); gm = np.mean([r["rows"][a]["gmag"] for r in results])
        auc = np.mean([r["rows"][a]["auc"] for r in results])
        print(f"{a:5.2f} | {rec:8.3f} {ent:7.2f} | {corr:+14.3f} {gm:9.2e} {auc:11.3f}")
    print(f"\nAttnLRP: corr(occ)={np.nanmean([r['att_corr'] for r in results]):+.3f}  insAUC={np.nanmean([r['att_auc'] for r in results]):.3f}")
    print(f"occlusion-order insAUC (ground-truth ceiling) = {np.mean([r['occ_auc'] for r in results]):.3f}")
    print("(if corr(grad,occ) is HIGH at high-a where recovery~1/entropy low -> gradient informative at prediction-preserved state;")
    print(" if corr LOW even at a=1 -> raw gradient genuinely uninformative, LRP-correction needed)")


if __name__ == "__main__":
    main()
