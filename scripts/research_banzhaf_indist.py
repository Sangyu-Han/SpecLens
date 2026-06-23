#!/usr/bin/env python3
"""Direct application of the SEQUENCE-OOD insight: the Banzhaf fails on long cases because heavy-
masking coalitions are sequence-OOD (degenerate -> uniform output -> no signal). FIX: sample
coalitions ONLY in the in-distribution regime (high keep-frac) so every sample is a near-complete,
in-distribution sequence. Compare keep-frac ~ U(lo, 1) for lo in {0.0 (full), 0.5 (restricted)}.
uniform-mean baseline (info-less, the best from the baseline sweep). By length bin, vs AttnLRP."""
from __future__ import annotations

import argparse
import json
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
LOS = [0.0, 0.5, 0.7]


def load_everest_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-imdb", type=int, default=5)
    ap.add_argument("--M", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]

    def last_logits(e):
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
            pred = int(last_logits(model.get_input_embeddings()(ids))[0].argmax())
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

        def fix(Z):
            Z[..., readout] = 1.0; Z[..., 0] = 1.0; return Z
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            lf = last_logits(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_p = float(F.softmax(lf, -1)[ans])
            m0 = fix(torch.zeros(1, T, device=dev))
            base = float(F.softmax(last_logits(emb * m0[:, :, None] + gmean * (1 - m0[:, :, None]))[0], -1)[ans])
        den = max(full_p - base, 1e-6)

        @torch.no_grad()
        def sig(Z):
            e = emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])
            return (F.softmax(last_logits(e), -1)[:, ans] - base) / den

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = fix(torch.zeros(1, T, device=dev))[0]
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])

        def ins_auc(scores):
            order = [i for i in np.argsort(-scores) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        def banzhaf(lo):
            Zs = []; Rs = []; j = 0
            while j < args.M:
                nb = min(args.bs, args.M - j); zb = []
                for _ in range(nb):
                    pf = lo + (1 - lo) * float(torch.rand(1, generator=gen, device=dev).item())
                    zb.append(fix((torch.rand(T, generator=gen, device=dev) < pf).float()))
                Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            s1 = (Z * R[:, None]).sum(0) / n1.clamp(min=1)
            s0 = ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)
            return (s1 - s0).cpu().numpy()

        a = attn_cache.get(key)
        ia = ins_auc(np.asarray(a, np.float32)) if a is not None and len(a) == T else float("nan")
        return dict(T=T, ins_attnlrp=ia, ins={f"lo{lo}": ins_auc(banzhaf(lo)) for lo in LOS})

    rows = []
    for key, prompt, ans_id in cases:
        r = run(key, prompt, ans_id); r["key"] = key; rows.append(r)
        ins_str = " ".join(f"lo{lo}={r['ins'][f'lo{lo}']:.3f}" for lo in LOS)
        print(f"  {key:8s} T={r['T']:4d} | {ins_str} | attnlrp={r['ins_attnlrp']:.3f}", flush=True)

    def binof(T):
        return next(n for n, lo, hi in BINS if lo <= T < hi)
    print("\n=== insertion AUC: full-range vs in-distribution-restricted Banzhaf vs AttnLRP ===")
    groups = {"short": [], "medium": [], "long": [], "ALL": []}
    for r in rows:
        if r["key"] != "everest":
            groups[binof(r["T"])].append(r)
        groups["ALL"].append(r)
    for g, rs in groups.items():
        if not rs:
            continue
        means = " ".join(f"lo{lo}={np.mean([r['ins'][f'lo{lo}'] for r in rs]):.3f}" for lo in LOS)
        at = np.nanmean([r["ins_attnlrp"] for r in rs])
        wr = 100.0 * np.nanmean([float(max(r["ins"][f"lo{lo}"] for lo in LOS) > r["ins_attnlrp"]) for r in rs if r["ins_attnlrp"] == r["ins_attnlrp"]])
        wr5 = 100.0 * np.nanmean([float(r["ins"]["lo0.5"] > r["ins_attnlrp"]) for r in rs if r["ins_attnlrp"] == r["ins_attnlrp"]])
        print(f"  {g:7s} n={len(rs):2d} | {means} attnlrp={at:.3f} | lo0.5>attn {wr5:.0f}% best>attn {wr:.0f}%")
    print("(if restricted >> full on LONG -> avoiding sequence-OOD samples fixes the long-case failure)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "banzhaf_indist.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
