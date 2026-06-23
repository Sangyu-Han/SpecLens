#!/usr/bin/env python3
"""INSERTION (sufficiency) across MULTIPLE datasets + token lengths: does FRI's random-budget
(realized gradient-free as the restricted-range Banzhaf, length-adaptive lo) BEAT AttnLRP on
insertion at every length/dataset? FRI's claim is insertion (deletion belongs to conditional).
Datasets: IMDB + SST2 (sentiment) + AG News (topic, 4-class). Methods: Banzhaf-adaptive, |grad|
(magnitude), AttnLRP. Metric: standard input-token insertion AUC (global vocab-mean baseline)."""
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
from research_llm_imdb_sufficiency import PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
BINS = [("short", 0, 90), ("medium", 90, 220), ("long", 220, 100000)]
AGN = ["World", "Sports", "Business", "Technology"]
AG_TMPL = ("Article: {text}\nQuestion: Is this news about World, Sports, Business, or Technology? "
           "Answer: The topic is")


def load_cases(name, n, tok, model, dev, llq):
    """Return [(text, prompt, gold_token_id)] argmax-correct, length-diverse."""
    out = []
    if name in ("imdb", "sst2"):
        pos = tok(" positive", add_special_tokens=False).input_ids[0]
        neg = tok(" negative", add_special_tokens=False).input_ids[0]
        if name == "imdb":
            raw, _ = load_imdb(n * 8, 1400)
        else:
            from datasets import load_dataset
            ds = load_dataset("sst2", split="validation")
            raw = [(int(ds[i]["label"]), ds[i]["sentence"]) for i in range(min(len(ds), n * 8))]
        for label, text in raw:
            pr = PROMPT_TMPL.format(text=text); ids = tok(pr, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == (pos if label == 1 else neg):
                    out.append((text, pr, pos if label == 1 else neg))
            if len(out) >= n:
                break
    elif name == "ag_news":
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="test")
        golds = [tok(" " + c, add_special_tokens=False).input_ids[0] for c in AGN]
        idx = np.random.default_rng(0).permutation(len(ds))[: n * 8]
        for i in idx:
            label = int(ds[int(i)]["label"]); text = ds[int(i)]["text"]
            pr = AG_TMPL.format(text=text); ids = tok(pr, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == golds[label]:
                    out.append((text, pr, golds[label]))
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--M", type=int, default=1500)
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

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    allcases = []
    for ds in ["imdb", "sst2", "ag_news"]:
        try:
            cs = load_cases(ds, args.per, tok, model, dev, llq)
            for k, (text, pr, gold) in enumerate(cs):
                allcases.append((f"{ds}_{k}", ds, pr, gold))
            print(f"  {ds}: {len(cs)} cases", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {ds} load failed: {type(e).__name__}: {e}", flush=True)
    print(f"total cases: {len(allcases)}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, _, pr, _ in allcases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full_p = float(F.softmax(llq(emb)[0], -1)[ans])
            m0 = torch.zeros(T, device=dev); m0[0] = 1.0; m0[readout] = 1.0
            base = float(F.softmax(llq(emb * m0[None, :, None] + gmean * (1 - m0[None, :, None]))[0], -1)[ans])
        den = max(full_p - base, 1e-6)

        @torch.no_grad()
        def sig(Z):
            return (F.softmax(llq(emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])), -1)[:, ans] - base) / den

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])

        def ins_auc(sc):
            order = [i for i in np.argsort(-sc) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        lo = 0.0 if T < 100 else 0.5                         # length-adaptive
        Zs = []; Rs = []; j = 0
        while j < args.M:
            nb = min(args.bs, args.M - j); zb = []
            for _ in range(nb):
                pf = lo + (1 - lo) * float(torch.rand(1, generator=gen, device=dev).item())
                z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[0] = 1.0; z[readout] = 1.0
                zb.append(z)
            Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
        Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
        bz = ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()
        eg = emb.clone().requires_grad_(True)
        model(inputs_embeds=eg).logits[0, -1, ans].backward()
        mag = eg.grad[0].abs().sum(-1).cpu().numpy()
        a = attn_cache.get(key); a = np.asarray(a, np.float32) if a is not None and len(a) == T else None
        return dict(T=T, fri=ins_auc(bz), grad=ins_auc(mag), attn=ins_auc(a) if a is not None else float("nan"))

    rows = []
    for key, ds, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r.update(key=key, ds=ds); rows.append(r)
            print(f"  {key:12s} T={r['T']:4d} | FRI={r['fri']:.3f} |grad|={r['grad']:.3f} attn={r['attn']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    def binof(T):
        return next(n for n, lo, hi in BINS if lo <= T < hi)
    print("\n=== INSERTION AUC by dataset (FRI=adaptive-Banzhaf vs |grad| vs AttnLRP) ===")
    for ds in ["imdb", "sst2", "ag_news", "ALL"]:
        rs = [r for r in rows if ds == "ALL" or r["ds"] == ds]
        if not rs:
            continue
        fri = np.mean([r["fri"] for r in rs]); gr = np.mean([r["grad"] for r in rs]); at = np.nanmean([r["attn"] for r in rs])
        wfri = 100.0 * np.nanmean([float(r["fri"] > r["attn"]) for r in rs if r["attn"] == r["attn"]])
        print(f"  {ds:8s} n={len(rs):2d} | FRI={fri:.3f} |grad|={gr:.3f} attn={at:.3f} | FRI>attn {wfri:.0f}%")
    print("\n=== INSERTION by LENGTH bin ===")
    for b, lo, hi in BINS:
        rs = [r for r in rows if lo <= r["T"] < hi]
        if not rs:
            continue
        fri = np.mean([r["fri"] for r in rs]); at = np.nanmean([r["attn"] for r in rs])
        wfri = 100.0 * np.nanmean([float(r["fri"] > r["attn"]) for r in rs if r["attn"] == r["attn"]])
        print(f"  {b:7s}(T~{int(np.mean([r['T'] for r in rs])):4d}) n={len(rs):2d} | FRI={fri:.3f} attn={at:.3f} | FRI>attn {wfri:.0f}%")
    json.dump(rows, open(os.path.join(REPO, "outputs", "insertion_datasets.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
