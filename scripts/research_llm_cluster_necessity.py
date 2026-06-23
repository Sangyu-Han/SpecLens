#!/usr/bin/env python3
"""Run the SAME cluster-greedy necessity (hidden-cluster -> group-conditional removal) on LLM, to test
if 'the model knows necessity' transfers to NL in ONE framework (raw-prob deletion). Hypothesis (user
recalls it under-performed): vision patches cluster by SPATIAL SIMILARITY = redundancy, but LLM token
redundancy is COMBINATORIAL — hidden clusters may group by semantic role, not redundancy — so naive
cluster-greedy may NOT capture LLM necessity (the LLM win came from value-cluster+super-additive, a
different mechanism). Compare cluster-greedy vs chunk_bz / single_occ / banzhaf / AttnLRP on sentiment."""
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


def kmeans(X, G, iters=25, seed=0):
    gen = np.random.default_rng(seed); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    c = X[gen.choice(len(X), min(G, len(X)), replace=False)].copy(); lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None] - c[None]) ** 2).sum(-1); lab = d.argmin(1)
        for g in range(len(c)):
            if (lab == g).any():
                c[g] = X[lab == g].mean(0)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=6); ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--G", type=int, default=16); ap.add_argument("--Mcand", type=int, default=48); ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    dev = args.device; gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    allcases = []
    for ds in ["imdb", "sst2"]:
        for k, (text, pr, gold) in enumerate(load_cases(ds, args.per, tok, model, dev, llq)):
            allcases.append((f"{ds}_{k}", pr, gold))
    print(f"cases: {len(allcases)}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in allcases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0], -1)[ans])
            H = model(ids, output_hidden_states=True).hidden_states[-1][0].float().cpu().numpy()   # [T,D] last-layer hidden

        @torch.no_grad()
        def prob(M):                                            # raw answer prob [0,1]
            e = emb * M[:, :, None] + gmean * (1 - M[:, :, None])
            return F.softmax(llq(e), -1)[:, ans].cpu().numpy()

        @torch.no_grad()
        def del_auc(order):
            order = [i for i in order if i in nz]; ps = []
            for f in FR:
                k = int(round(f * len(order))); m = torch.ones(T, device=dev)
                if k:
                    m[torch.as_tensor(order[:k], device=dev)] = 0.0
                ps.append(prob(m[None])[0])
            return float(np.trapz(ps, FR) / FR[-1])             # RAW prob AUC, lower=better

        @torch.no_grad()
        def occ_prior():
            pr = np.zeros(T, np.float32)
            for s in range(0, len(nz), args.bs):
                idx = nz[s:s + args.bs]; M = torch.ones(len(idx), T, device=dev)
                for r, i in enumerate(idx):
                    M[r, i] = 0.0
                pp = prob(M)
                for r, i in enumerate(idx):
                    pr[i] = full - pp[r]
            return pr

        # cooperative Banzhaf (for chunk_bz prior + banzhaf method)
        Zs = []; Rs = []; j = 0
        while j < args.M:
            nb = min(args.bs, args.M - j); zb = []
            for _ in range(nb):
                pf = float(torch.rand(1, generator=gen, device=dev).item())
                z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[0] = 1.0; z[readout] = 1.0; zb.append(z)
            Z = torch.stack(zb); Rs.append(torch.tensor(prob(Z), device=dev)); Zs.append(Z); j += nb
        Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
        bz = ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        def chunked(prior):
            c = [int(i) for i in np.argsort(-prior) if i in nz][:args.Mcand]; km = np.ones(T, np.float32); order = []; per = max(1, args.Mcand // args.R)
            while c:
                Mm = np.tile(km, (len(c), 1))
                for r, i in enumerate(c):
                    Mm[r, i] = 0.0
                pp = prob(torch.tensor(Mm, device=dev)); idx = list(np.argsort(pp)[:per])
                for jj in idx:
                    km[c[jj]] = 0.0; order.append(c[jj])
                for jj in sorted(idx, reverse=True):
                    c.pop(jj)
            return order + [i for i in np.argsort(-prior) if i in nz and i not in set(order)]

        @torch.no_grad()
        def cluster_greedy():
            nzi = np.array(nz); lab = kmeans(H[nzi], min(args.G, len(nz)))
            norm = np.linalg.norm(H[nzi], axis=1)
            groups = [nzi[lab == g] for g in range(lab.max() + 1) if (lab == g).any()]
            km = np.ones(T, np.float32); order = []; rem = list(range(len(groups)))
            while rem:
                masks = []
                for ci in rem:
                    m = km.copy(); m[groups[ci]] = 0.0; masks.append(m)
                pp = prob(torch.tensor(np.stack(masks), device=dev)); jbest = int(np.argmin(pp)); ci = rem[jbest]
                for p in sorted(groups[ci].tolist(), key=lambda q: -norm[list(nzi).index(q)]):
                    km[p] = 0.0; order.append(int(p))
                rem.pop(jbest)
            return order

        occ = occ_prior(); att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        return dict(T=T,
                    cluster=del_auc(cluster_greedy()),
                    chunk_bz=del_auc(chunked(bz)),
                    single_occ=del_auc([i for i in np.argsort(-occ) if i in nz]),
                    banzhaf=del_auc([i for i in np.argsort(-bz) if i in nz]),
                    attn=del_auc([i for i in np.argsort(-att) if i in nz]) if att is not None else float("nan"))

    rows = []
    for key, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r["key"] = key; rows.append(r)
            print(f"  {key:10s} T={r['T']:3d} | cluster {r['cluster']:.3f} chunk_bz {r['chunk_bz']:.3f} single_occ {r['single_occ']:.3f} "
                  f"banzhaf {r['banzhaf']:.3f} attn {r['attn']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)
    print("\n=== LLM cluster-greedy necessity (RAW-prob del [0,1], LOWER=better) ===")
    a = lambda k: float(np.nanmean([r[k] for r in rows]))
    cc = 100.0 * np.mean([r["cluster"] <= r["chunk_bz"] for r in rows])
    print(f"  n={len(rows)} | cluster {a('cluster'):.3f} chunk_bz {a('chunk_bz'):.3f} single_occ {a('single_occ'):.3f} "
          f"banzhaf {a('banzhaf'):.3f} attn {a('attn'):.3f} | cluster≤chunk_bz {cc:.0f}%")
    print("(KEY: does LLM cluster-greedy ≈ chunk_bz? if WORSE -> token redundancy is combinatorial, not hidden-similarity = vision-specific)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "llm_cluster_necessity.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
