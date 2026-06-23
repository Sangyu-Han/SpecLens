#!/usr/bin/env python3
"""LLM REDUNDANCY -> NECESSITY (the culmination): on REDUNDANT LLM tasks (sentiment, where AttnLRP
WON deletion 0.049 < chunked 0.113 earlier), apply the vision-derived idea — use the cooperative
Banzhaf marginal as the PRIOR for the conditional (chunked) necessity, because single-occ MISSES
redundant supporters while the Banzhaf (coalition-averaged) captures them. Question: does the
Banzhaf-prior conditional resolve the redundancy and BEAT AttnLRP on deletion? Deletion AUC LOWER=better.
Methods: chunked_occ (single-occ prior), chunked_bz (Banzhaf prior), banzhaf-raw, AttnLRP, single_occ."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_insertion_datasets import FR, load_cases  # noqa: E402
from research_llm_imdb_sufficiency import precompute_attnlrp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--M", type=int, default=800)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--Mcand", type=int, default=48)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
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
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    allcases = []
    for ds in ["imdb", "sst2"]:                                    # redundant sentiment
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
            full = float(F.softmax(llq(emb)[0], -1)[ans]); base = float(F.softmax(llq(emb * 0 + gmean)[0], -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def rec(M):
            return ((F.softmax(llq(emb * M[:, :, None] + gmean * (1 - M[:, :, None])), -1)[:, ans] - base) / den).cpu().numpy()

        @torch.no_grad()
        def occ_prior():
            pr = np.zeros(T, np.float32)
            for s in range(0, len(nz), args.bs):
                idx = nz[s:s + args.bs]; M = torch.ones(len(idx), T, device=dev)
                for r, i in enumerate(idx):
                    M[r, i] = 0.0
                rr = rec(M)
                for r, i in enumerate(idx):
                    pr[i] = 1.0 - rr[r]
            return pr

        # gradient-guided cooperative Banzhaf prior (captures redundant supporters)
        eg = emb.clone().requires_grad_(True)
        try:
            out = model(inputs_embeds=eg, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=eg)
        out.logits[0, -1, ans].backward()
        g = eg.grad[0].abs().sum(-1).cpu().numpy()
        kk = min(args.K, len(nz)); cand = [i for i in np.argsort(-g) if i in nz][:kk]; cand_t = torch.as_tensor(cand, device=dev)
        Zs = []; Rs = []; j = 0
        while j < args.M:
            nb = min(args.bs, args.M - j); zb = []
            for _ in range(nb):
                z = torch.ones(T, device=dev); pf = float(torch.rand(1, generator=gen, device=dev).item())
                z[cand_t] = (torch.rand(kk, generator=gen, device=dev) < pf).float()
                z[0] = 1.0; z[readout] = 1.0; zb.append(z)
            Z = torch.stack(zb); Rs.append(torch.tensor(rec(Z), device=dev)); Zs.append(Z); j += nb
        Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
        marg = ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()
        bz = np.full(T, -1e9, np.float64)
        for i in cand:
            bz[i] = marg[i]
        for i in nz:
            if i not in set(cand):
                bz[i] = -1e6 + g[i] * 1e-3

        def chunked_order(prior):
            c = [int(i) for i in np.argsort(-prior) if i in nz][: args.Mcand]
            km = np.ones(T, np.float32); order = []; per = max(1, args.Mcand // args.R)
            while c:
                M = np.tile(km, (len(c), 1))
                for r, i in enumerate(c):
                    M[r, i] = 0.0
                rr = rec(torch.tensor(M, device=dev))
                idx = list(np.argsort(rr)[:per])
                for jj in idx:
                    km[c[jj]] = 0.0; order.append(c[jj])
                for jj in sorted(idx, reverse=True):
                    c.pop(jj)
            order += [i for i in np.argsort(-prior) if i in nz and i not in set(order)]
            return order

        @torch.no_grad()
        def del_auc(order):
            order = [i for i in order if i in nz]; aucs = []
            for f in FR:
                k = int(round(f * len(order))); m = torch.ones(T, device=dev)
                if k:
                    m[torch.as_tensor(order[:k], device=dev)] = 0.0
                aucs.append(rec(m[None])[0])
            return float(np.trapz(aucs, FR) / FR[-1])

        prior = occ_prior()
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        return dict(T=T,
                    chunked_occ=del_auc(chunked_order(prior)),
                    chunked_bz=del_auc(chunked_order(bz)),
                    banzhaf=del_auc([i for i in np.argsort(-bz) if i in nz]),
                    single_occ=del_auc([i for i in np.argsort(-prior) if i in nz]),
                    attn=del_auc([i for i in np.argsort(-att) if i in nz]) if att is not None else float("nan"))

    rows = []
    for key, ds, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r.update(key=key, ds=ds); rows.append(r)
            print(f"  {key:10s} T={r['T']:3d} | chunked_occ={r['chunked_occ']:.3f} chunked_bz={r['chunked_bz']:.3f} "
                  f"banzhaf={r['banzhaf']:.3f} | attn={r['attn']:.3f} occ={r['single_occ']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== LLM REDUNDANT (sentiment) NECESSITY (deletion AUC, LOWER=better) — Banzhaf-prior conditional vs AttnLRP ===")
    for ds in ["imdb", "sst2", "ALL"]:
        rs = [r for r in rows if ds == "ALL" or r["ds"] == ds]
        if not rs:
            continue
        agg = {k: np.nanmean([r[k] for r in rs]) for k in ["chunked_occ", "chunked_bz", "banzhaf", "attn", "single_occ"]}
        d = np.array([r["chunked_bz"] - r["attn"] for r in rs if r["attn"] == r["attn"]])
        win = 100.0 * (d < 0).mean() if len(d) else 0
        print(f"  {ds:5s} n={len(rs):2d} | chunked_occ={agg['chunked_occ']:.3f} chunked_bz={agg['chunked_bz']:.3f} "
              f"banzhaf={agg['banzhaf']:.3f} attn={agg['attn']:.3f} occ={agg['single_occ']:.3f} | bz<attn {win:.0f}%")
    print("(does Banzhaf-prior chunked (chunked_bz) resolve redundancy & beat AttnLRP? does it beat single-occ-prior chunked_occ?)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "llm_redundancy_necessity.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
