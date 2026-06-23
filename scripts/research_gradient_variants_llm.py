#!/usr/bin/env python3
"""Trace gradient variants on LLM to (a) see if input×grad (usually meaningful) works, (b) pin WHY
raw gradient fails where AttnLRP works, (c) FIND + demonstrate the real sufficiency set via AttnLRP.

Variants (all at the FULL input, one/few backward passes, NO OOD masking):
  raw|grad|, input×grad (emb·g), (input−baseline)×grad, integrated-gradients (IG, mean→input),
  AttnLRP (LRP-corrected, attention-aware).
For each: insertion AUC (sufficiency-set quality) + corr with the actual occlusion effect. The
gradient family is LOCAL (fixed attention); AttnLRP propagates through attention. Then print the
AttnLRP sufficiency set (top tokens) for Everest and verify its recovery curve."""
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


def load_everest_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-imdb", type=int, default=3)
    ap.add_argument("--ig-steps", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:  # noqa: BLE001
        pass
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

    def grad_at(emb, ids, ans):                # d(logit_ans)/d(emb) at a given input
        e = emb.clone().requires_grad_(True)
        model(inputs_embeds=e).logits[0, -1, ans].backward()
        return e.grad[0].detach()              # [T,D]

    def run(key, prompt, ans_id, show=False):
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
        # occlusion ground truth (logit)
        occ = np.zeros(T, np.float32)
        with torch.no_grad():
            for s in range(0, len(nz), 16):
                idx = nz[s:s + 16]; e = emb.expand(len(idx), -1, -1).clone()
                for r, i in enumerate(idx):
                    e[r, i] = gmean[0, 0]
                lg = llq_ng(e)[:, ans]
                for r, i in enumerate(idx):
                    occ[i] = full_logit - float(lg[r])

        g = grad_at(emb, ids, ans)             # [T,D] at full input
        raw = g.abs().sum(-1).cpu().numpy()
        inxg = (emb[0] * g).sum(-1).cpu().numpy()
        bxg = ((emb[0] - gmean[0, 0]) * g).sum(-1).cpu().numpy()
        ig_acc = torch.zeros_like(g)
        for a in np.linspace(1.0 / args.ig_steps, 1.0, args.ig_steps):
            ig_acc += grad_at(gmean + float(a) * (emb - gmean), ids, ans)
        ig = ((emb[0] - gmean[0, 0]) * (ig_acc / args.ig_steps)).sum(-1).cpu().numpy()
        att = attn_cache.get(key)
        scores = {"raw|grad|": raw, "input×grad": inxg, "(in−base)×grad": bxg, "IG": ig}
        if att is not None and len(att) == T:
            scores["AttnLRP"] = np.asarray(att, np.float32)

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            e = emb * m[None, :, None] + gmean * (1 - m[None, :, None])
            return float((F.softmax(llq_ng(e)[0], -1)[ans] - base_p) / den)

        def ins_auc(sc):
            order = [i for i in np.argsort(-sc) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])
        occ_auc = ins_auc(occ)
        out = {nm: dict(auc=ins_auc(sc), corr=float(spearmanr(sc[nz], occ[nz]).correlation)) for nm, sc in scores.items()}

        if show and "AttnLRP" in scores:       # demonstrate the AttnLRP sufficiency set
            order = [i for i in np.argsort(-scores["AttnLRP"]) if i not in (0, readout)]
            toks = tok.convert_ids_to_tokens([int(ids[0, i]) for i in order[:12]])
            print(f"\n  [{key}] AttnLRP top-12 sufficiency tokens: {toks}")
            print(f"  recovery keeping top-k AttnLRP tokens: " +
                  " ".join(f"k={int(f*len(order))}:{rec_hard(order,f):.2f}" for f in [0.02, 0.05, 0.1, 0.2]))
        return dict(T=T, occ_auc=occ_auc, methods=out)

    results = []
    for i, (key, prompt, ans_id) in enumerate(cases):
        r = run(key, prompt, ans_id, show=(key == "everest")); r["key"] = key; results.append(r)
        print(f"  {key} T={r['T']} done", flush=True)

    print("\n=== gradient VARIANTS vs AttnLRP on LLM: insertion AUC (sufficiency) + corr(occlusion) ===")
    names = ["raw|grad|", "input×grad", "(in−base)×grad", "IG", "AttnLRP"]
    print(f"{'method':>16s} | {'mean insAUC':>11s} | {'mean corr(occ)':>14s}")
    for nm in names:
        aucs = [r["methods"][nm]["auc"] for r in results if nm in r["methods"]]
        corrs = [r["methods"][nm]["corr"] for r in results if nm in r["methods"]]
        if aucs:
            print(f"{nm:>16s} | {np.mean(aucs):11.3f} | {np.nanmean(corrs):+14.3f}")
    print(f"{'occlusion ceiling':>16s} | {np.mean([r['occ_auc'] for r in results]):11.3f} |")
    print("(input×grad/IG are LOCAL gradient -> fail like raw grad; AttnLRP is attention-aware -> ~ceiling = finds the sufficiency set)")


if __name__ == "__main__":
    main()
