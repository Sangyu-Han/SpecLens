#!/usr/bin/env python3
"""GRADIENT-GUIDED long-Banzhaf: the long failure is estimation VARIANCE (not OOD). Reduce it by
focusing the coalition sampling on the |grad|-top-K CANDIDATE tokens while holding the (irrelevant)
non-candidates REAL (in-distribution background). Each candidate's marginal is then estimated
without the variance from the T-K irrelevant tokens -> sample-efficient for long. Rank = candidates
by marginal + non-candidates by |grad|. Compare on long (T>300) vs plain restricted-Banzhaf (vary
ALL tokens), |grad|, AttnLRP. Does gradient-guidance let FRI catch AttnLRP on long?"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--M", type=int, default=2500)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing ON", flush=True)
    except Exception:  # noqa: BLE001
        pass
    dev = args.device
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    raw, _ = load_imdb(600, 3000); cases = []
    for label, text in raw:
        pr = PROMPT_TMPL.format(text=text); ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        if int(ids.shape[1]) < 300:
            continue
        with torch.no_grad():
            if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == (pos_id if label == 1 else neg_id):
                cases.append((f"L{len(cases)}", pr, pos_id if label == 1 else neg_id))
        if len(cases) >= args.n:
            break
    print(f"long cases: {len(cases)}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in cases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0], -1)[ans])
            base = float(F.softmax(llq(emb * 0 + gmean)[0], -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def sig(Z):
            return (F.softmax(llq(emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])), -1)[:, ans] - base) / den

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])

        def ins_auc(order):
            order = [i for i in order if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        eg = emb.clone().requires_grad_(True)
        try:
            out = model(inputs_embeds=eg, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=eg)
        out.logits[0, -1, ans].backward()
        g = eg.grad[0].abs().sum(-1).cpu().numpy()
        cand = [i for i in np.argsort(-g) if i not in (0, readout)][:args.K]
        candset = set(cand)
        cand_t = torch.as_tensor(cand, device=dev)

        def banzhaf(guided):
            Zs = []; Rs = []; j = 0
            while j < args.M:
                nb = min(args.bs, args.M - j); zb = []
                for _ in range(nb):
                    if guided:                                   # non-candidates REAL, vary candidates
                        z = torch.ones(T, device=dev)
                        pf = float(torch.rand(1, generator=gen, device=dev).item())
                        z[cand_t] = (torch.rand(args.K, generator=gen, device=dev) < pf).float()
                    else:                                        # plain restricted: vary ALL, keep-frac U(0.5,1)
                        pf = 0.5 + 0.5 * float(torch.rand(1, generator=gen, device=dev).item())
                        z = (torch.rand(T, generator=gen, device=dev) < pf).float()
                    z[0] = 1.0; z[readout] = 1.0; zb.append(z)
                Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            return ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        marg = banzhaf(True)
        order_guided = sorted(cand, key=lambda i: -marg[i]) + [i for i in np.argsort(-g) if i in nz and i not in candset]
        order_plain = [i for i in np.argsort(-banzhaf(False)) if i in nz]
        order_grad = [i for i in np.argsort(-g) if i in nz]
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        return dict(T=T, guided=ins_auc(order_guided), plain=ins_auc(order_plain), grad=ins_auc(order_grad),
                    attn=ins_auc([i for i in np.argsort(-att) if i in nz]) if att is not None else float("nan"))

    rows = []
    for key, prompt, ans in cases:
        try:
            r = run(key, prompt, ans); r["key"] = key; rows.append(r)
            print(f"  {key} T={r['T']:4d} | guided={r['guided']:.3f} plain={r['plain']:.3f} |grad|={r['grad']:.3f} attn={r['attn']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== LONG insertion: gradient-GUIDED Banzhaf vs plain-restricted vs |grad| vs AttnLRP ===")
    for nm in ["guided", "plain", "grad", "attn"]:
        vals = [r[nm] for r in rows if r[nm] == r[nm]]
        print(f"  {nm:8s} = {np.mean(vals):.3f}")
    wg = 100.0 * np.nanmean([float(r["guided"] > r["attn"]) for r in rows if r["attn"] == r["attn"]])
    print(f"  guided>attn: {wg:.0f}%  (does gradient-guidance let FRI catch AttnLRP on long?)")


if __name__ == "__main__":
    main()
