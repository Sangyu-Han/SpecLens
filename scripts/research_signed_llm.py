#!/usr/bin/env python3
"""Can the gradient SIGN be made usable on LLM (to drop distractors like it does on vision)?
First: does the sign even HELP on LLM? Measure (1) distractor fraction (tokens with negative
occlusion = removing them raises the logit), (2) signed-occ ceiling vs |grad| (the sign headroom),
(3) drop-distractors recovery, (4) can a reliable sign be recovered: signed directional sd
(=(emb−mean)·grad, fails), |grad|×sign(AttnLRP) (magnitude + AttnLRP's sign), AttnLRP. Compare to
vision where negative-sd patches were genuine distractors and the sign helped (0.825 vs 0.739)."""
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
    ap.add_argument("--n-imdb", type=int, default=5)
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

    def llq(e):
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
            pred = int(llq(model.get_input_embeddings()(ids))[0].argmax())
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
            lf = llq(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_logit = float(lf[ans]); full_p = float(F.softmax(lf, -1)[ans])
            base_p = float(F.softmax(llq(emb * 0 + gmean)[0], -1)[ans]); den = max(full_p - base_p, 1e-6)
        so = np.zeros(T, np.float32)
        with torch.no_grad():
            for s in range(0, len(nz), 16):
                idx = nz[s:s + 16]; e = emb.expand(len(idx), -1, -1).clone()
                for r, i in enumerate(idx):
                    e[r, i] = gmean[0, 0]
                lg = llq(e)[:, ans]
                for r, i in enumerate(idx):
                    so[i] = full_logit - float(lg[r])
        eg = emb.clone().requires_grad_(True)
        model(inputs_embeds=eg).logits[0, -1, ans].backward()
        g = eg.grad[0].detach()
        sd = ((emb[0] - gmean[0, 0]) * g).sum(-1).cpu().numpy()
        mag = g.abs().sum(-1).cpu().numpy()
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None

        @torch.no_grad()
        def rec_keep(maskvec):                      # keep where maskvec True
            m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            m[torch.as_tensor([i for i in nz if maskvec[i]], device=dev)] = 1.0
            return float((F.softmax(llq(emb * m[None, :, None] + gmean * (1 - m[None, :, None]))[0], -1)[ans] - base_p) / den)

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float((F.softmax(llq(emb * m[None, :, None] + gmean * (1 - m[None, :, None]))[0], -1)[ans] - base_p) / den)

        def ins_auc(sc):
            order = [i for i in np.argsort(-sc) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        dis = np.array([so[i] < 0 for i in range(T)])           # distractor = negative occlusion
        drop_dis = rec_keep(np.array([not dis[i] for i in range(T)]))   # keep all but distractors
        methods = {"signed sd": sd, "|grad|": mag, "signed-occ(ceil)": so}
        if att is not None:
            methods["AttnLRP"] = att; methods["|grad|×sign(AttnLRP)"] = mag * np.sign(att)
        aucs = {nm: ins_auc(sc) for nm, sc in methods.items()}
        return dict(T=T, corr_sd_so=float(spearmanr(sd[nz], so[nz]).correlation),
                    distractor_frac=float(dis[nz].mean()), full=1.0, drop_dis=drop_dis, aucs=aucs)

    res = []
    for key, prompt, ans_id in cases:
        r = run(key, prompt, ans_id); r["key"] = key; res.append(r)
        print(f"  {key} T={r['T']} distractor%={100*r['distractor_frac']:.0f} done", flush=True)

    print("\n=== LLM: does the gradient SIGN help (like it did on vision)? ===")
    print(f"corr(signed-grad sd, signed-occ) = {np.nanmean([r['corr_sd_so'] for r in res]):+.3f} (vision was +0.087)")
    print(f"distractor fraction (neg occlusion) = {100*np.mean([r['distractor_frac'] for r in res]):.0f}% (vision ~51%)")
    print(f"drop-distractors recovery = {np.mean([r['drop_dis'] for r in res]):.3f} (full=1.0; >1 => distractors were harmful)")
    print("\ninsertion AUC:")
    for nm in ["signed sd", "|grad|", "|grad|×sign(AttnLRP)", "AttnLRP", "signed-occ(ceil)"]:
        vals = [r["aucs"][nm] for r in res if nm in r["aucs"]]
        if vals:
            print(f"  {nm:22s} = {np.mean(vals):.3f}")
    print("(if signed-occ ceiling ~ |grad| AND distractor% low -> the sign has little headroom on LLM; "
          "if |grad|×sign(AttnLRP) > |grad| -> recovering the sign via AttnLRP helps)")


if __name__ == "__main__":
    main()
