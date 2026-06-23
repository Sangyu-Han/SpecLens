#!/usr/bin/env python3
"""Can the SOFT-insertion random-budget FRI (gradient-optimized, ~100 steps) beat AttnLRP on LLM
IF we keep the input in-distribution (restrict the budget to high keep-frac)? Earlier soft-FRI
failed at UNIFORM/low-budget masks (sequence-OOD -> vanishing gradient |grad|~1e-6). Hypothesis:
restricting the random budget to b ~ U(lo,1) keeps every mask near-complete -> in-distribution ->
the model-gradient through the soft mask becomes informative (the conditional/necessity gradient).

Cost: 100 steps x (fwd+bwd) ~= 250 fwd-equiv (8x cheaper than the M=2000 Banzhaf, ~100x AttnLRP).
NOISE-AWARE (lesson learned): K seeds/case, report mean+-std, paired vs AttnLRP. lo=0.5 (restricted)
vs lo=0.0 (full baseline). Insertion AUC = standard mean-baseline metric."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    ap.add_argument("--n-imdb", type=int, default=5)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()  # recompute activations in backward -> fits the contended GPU
        print("gradient checkpointing ON", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"gc enable failed: {e}", flush=True)
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
    print(f"cases: {[c[0] for c in cases]} | K={args.K}, steps={args.steps}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in cases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans_id):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1
        emb = model.get_input_embeddings()(ids).detach()
        fixed = (torch.arange(T, device=dev) == 0) | (torch.arange(T, device=dev) == readout)
        with torch.no_grad():
            lf = llq_ng(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_p = float(F.softmax(lf, -1)[ans])
            m0 = torch.where(fixed, 1.0, 0.0)
            base = float(F.softmax(llq_ng(emb * m0[None, :, None] + gmean * (1 - m0[None, :, None]))[0], -1)[ans])
        den = max(full_p - base, 1e-6)

        def prob_grad(w):                                   # differentiable recovery, soft mask w [T]
            ww = torch.where(fixed, torch.ones_like(w), w)
            e = emb * ww[None, :, None] + gmean * (1 - ww[None, :, None])
            return (F.softmax(model(inputs_embeds=e).logits[0, -1], -1)[ans] - base) / den

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[fixed] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            e = emb * m[None, :, None] + gmean * (1 - m[None, :, None])
            return float((F.softmax(llq_ng(e)[0], -1)[ans] - base) / den)

        def ins_auc(scores):
            order = [i for i in np.argsort(-scores) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        def soft_fri(lo, seed, l1=0.001):
            g = torch.Generator(device=dev); g.manual_seed(seed)
            la = torch.zeros(T, device=dev); mv = torch.zeros(T, device=dev); vv = torch.zeros(T, device=dev)
            for step in range(args.steps):
                lr = 0.01 + 0.5 * (0.4 - 0.01) * (1 + math.cos(math.pi * step / (args.steps - 1)))
                lar = la.clone().requires_grad_(True); p = torch.sigmoid(lar); pn = p / (p.sum() + 1e-6)
                b = (lo + (1 - lo) * float(torch.rand(1, generator=g, device=dev).item())) * T   # RESTRICTED budget
                w = (pn * b).clamp(max=1.0)
                loss = -prob_grad(w) + l1 * p.sum()
                gg = torch.autograd.grad(loss, lar)[0].detach(); t = step + 1
                mv = 0.9 * mv + 0.1 * gg; vv = 0.999 * vv + 0.001 * gg * gg
                adam = (mv / (1 - 0.9 ** t)) / ((vv / (1 - 0.999 ** t)).sqrt() + 1e-8)
                cm = (adam * gg > 0).float(); cm = cm * (T / cm.sum().clamp(min=1.0))
                la = la - lr * adam * cm
            return torch.sigmoid(la).detach().cpu().numpy()

        a = attn_cache.get(key)
        ia = ins_auc(np.asarray(a, np.float32)) if a is not None and len(a) == T else float("nan")
        r05 = [ins_auc(soft_fri(0.5, s)) for s in range(args.K)]
        r00 = ins_auc(soft_fri(0.0, 0))                      # full-budget baseline (1 seed)
        return dict(T=T, ins_attnlrp=ia, soft05_mean=float(np.mean(r05)), soft05_std=float(np.std(r05)),
                    soft05=r05, soft00_full=r00)

    rows = []
    for key, prompt, ans_id in cases:
        try:
            r = run(key, prompt, ans_id); r["key"] = key; rows.append(r)
            print(f"  {key:8s} T={r['T']:4d} | soft-FRI lo0.5 {r['soft05_mean']:.3f}+-{r['soft05_std']:.3f} "
                  f"(full lo0={r['soft00_full']:.3f}) | attnlrp={r['ins_attnlrp']:.3f} | diff={r['soft05_mean']-r['ins_attnlrp']:+.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM skip", flush=True)

    diffs = [r["soft05_mean"] - r["ins_attnlrp"] for r in rows if r["ins_attnlrp"] == r["ins_attnlrp"]]
    print("\n=== VERDICT: in-distribution soft-FRI (restricted budget, 100 steps) vs AttnLRP ===")
    print(f"  soft-FRI lo0.5 = {np.mean([r['soft05_mean'] for r in rows]):.3f}  "
          f"(full lo0 = {np.mean([r['soft00_full'] for r in rows]):.3f})  AttnLRP = {np.nanmean([r['ins_attnlrp'] for r in rows]):.3f}")
    print(f"  paired diff (soft0.5 - attnlrp): mean={np.mean(diffs):+.3f} stderr={np.std(diffs)/np.sqrt(max(len(diffs),1)):.3f} "
          f"| win {int(np.sum(np.array(diffs)>0))}/{len(diffs)}")
    print(f"  avg within-case seed std = {np.mean([r['soft05_std'] for r in rows]):.3f}")
    print(f"  => {'BEATS AttnLRP (mean diff > paired stderr)' if np.mean(diffs)-np.std(diffs)/np.sqrt(max(len(diffs),1))>0 else 'within noise / not beating'}")
    print("  (cost: 100 steps ~250 fwd-equiv, 8x cheaper than M=2000 Banzhaf)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "soft_fri_indist.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
