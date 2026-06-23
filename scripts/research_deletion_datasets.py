#!/usr/bin/env python3
"""C / NECESSITY: does the conditional (chunked) necessary-set win DELETION (lower=better) vs
|grad|, AttnLRP, single-occlusion, across IMDB+SST2+AG News? (FRI=insertion, conditional=deletion.)
Deletion AUC = recovery after removing top-k by each method (token->mean); LOWER = the removed set
was more necessary."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_insertion_datasets import BINS, FR, load_cases  # noqa: E402
from research_llm_imdb_sufficiency import precompute_attnlrp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--Mcand", type=int, default=60)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    allcases = []
    for ds in ["imdb", "sst2", "ag_news"]:
        try:
            for k, (text, pr, gold) in enumerate(load_cases(ds, args.per, tok, model, dev, llq)):
                allcases.append((f"{ds}_{k}", ds, pr, gold))
        except Exception as e:  # noqa: BLE001
            print(f"  {ds} fail: {e}", flush=True)
    print(f"cases: {len(allcases)}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, _, pr, _ in allcases], args.model, dev)
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
        def rec_batch(M):                              # M [B,T] keep-mask -> recovery [B]
            e = emb * M[:, :, None] + gmean * (1 - M[:, :, None])
            return ((F.softmax(llq(e), -1)[:, ans] - base) / den).cpu().numpy()

        @torch.no_grad()
        def occ_prior():                               # single-occ: full - rec(remove i)
            pr = np.zeros(T, np.float32)
            for s in range(0, len(nz), args.bs):
                idx = nz[s:s + args.bs]; M = torch.ones(len(idx), T, device=dev)
                for r, i in enumerate(idx):
                    M[r, i] = 0.0
                rr = rec_batch(M)
                for r, i in enumerate(idx):
                    pr[i] = 1.0 - rr[r]                 # high = removing it drops recovery = necessary
            return pr

        def chunked_order(prior):
            cand = [int(i) for i in np.argsort(-prior) if i in nz][: args.Mcand]
            km = np.ones(T, np.float32); order = []; per = max(1, args.Mcand // args.R)
            while cand:
                M = np.tile(km, (len(cand), 1))
                for r, i in enumerate(cand):
                    M[r, i] = 0.0
                rr = rec_batch(torch.tensor(M, device=dev))
                idx = list(np.argsort(rr)[:per])       # most drop = most necessary
                for j in idx:
                    km[cand[j]] = 0.0; order.append(cand[j])
                for j in sorted(idx, reverse=True):
                    cand.pop(j)
            order += [i for i in np.argsort(-prior) if i in nz and i not in set(order)]
            return order

        @torch.no_grad()
        def del_auc(order):                            # remove top-k -> recovery; LOWER=better
            order = [i for i in order if i in nz]; aucs = []
            for f in FR:
                k = int(round(f * len(order))); m = torch.ones(T, device=dev)
                if k:
                    m[torch.as_tensor(order[:k], device=dev)] = 0.0
                aucs.append(rec_batch(m[None])[0])
            return float(np.trapz(aucs, FR) / FR[-1])

        prior = occ_prior()
        chunked = chunked_order(prior)
        eg = emb.clone().requires_grad_(True)
        model(inputs_embeds=eg).logits[0, -1, ans].backward()
        mag = eg.grad[0].abs().sum(-1).cpu().numpy()
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        return dict(T=T,
                    chunked=del_auc(chunked),
                    singleocc=del_auc([i for i in np.argsort(-prior) if i in nz]),
                    grad=del_auc([i for i in np.argsort(-mag) if i in nz]),
                    attn=del_auc([i for i in np.argsort(-att) if i in nz]) if att is not None else float("nan"))

    rows = []
    for key, ds, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r.update(key=key, ds=ds); rows.append(r)
            print(f"  {key:12s} T={r['T']:4d} | chunked={r['chunked']:.3f} occ={r['singleocc']:.3f} |grad|={r['grad']:.3f} attn={r['attn']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== DELETION AUC (LOWER=better necessity): conditional(chunked) vs baselines ===")
    for ds in ["imdb", "sst2", "ag_news", "ALL"]:
        rs = [r for r in rows if ds == "ALL" or r["ds"] == ds]
        if not rs:
            continue
        ch = np.mean([r["chunked"] for r in rs]); oc = np.mean([r["singleocc"] for r in rs])
        gr = np.mean([r["grad"] for r in rs]); at = np.nanmean([r["attn"] for r in rs])
        wins = 100.0 * np.nanmean([float(r["chunked"] <= min(r["singleocc"], r["grad"], r["attn"] if r["attn"] == r["attn"] else 9)) for r in rs])
        print(f"  {ds:8s} n={len(rs):2d} | chunked={ch:.3f} single-occ={oc:.3f} |grad|={gr:.3f} attn={at:.3f} | chunked best {wins:.0f}%")
    print("(chunked LOWEST => conditional necessary-set is the strongest deletion/necessity method)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "deletion_datasets.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
