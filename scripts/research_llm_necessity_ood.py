#!/usr/bin/env python3
"""Does solving INPUT-SEQUENCE OOD rescue CHEAP LLM necessity? The sequence-OOD law
(docs/llm_sufficiency_restricted_banzhaf.md): a single masked token is in-dist (entropy ~0.3) but
HEAVY masking is degenerate (entropy 7-8) — so the naive deletion-AUC (vocab-mean fill, full 0..0.5
range) is OOD-contaminated, which may be WHY cheap necessity (cluster/single-occ) lost to AttnLRP/PA-LRP.
Two OOD fixes (user-picked): (A) IN-DIST FILL — replace deleted token by eos / prompt-mean (a real
in-distribution vector) instead of the global vocab-mean; (B) RESTRICTED-RANGE — score necessity only
in the low-removal in-distribution regime. Compare cheap necessity vs LRP under NAIVE vs OOD-SOLVED,
plus an entropy OOD-proxy. Llama-3.2-1B-Instruct; raw-prob deletion AUC (lower=better)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PALRP = "/tmp/PE-AWARE-LRP/NLP"
sys.path.insert(0, PALRP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research_insertion_datasets import load_cases  # noqa: E402

LLAMA = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/"))[-1]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FR_FULL = [0.0, .02, .05, .1, .15, .2, .3, .5]
FR_RESTR = [0.0, .01, .02, .03, .05, .07, .1]      # in-distribution low-removal regime


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
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--G", type=int, default=12); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--tcap", type=int, default=100)
    args = ap.parse_args()
    dev = args.device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    emb_w = model.get_input_embeddings().weight
    gmean = emb_w.mean(0).view(1, 1, -1).detach()
    eos_vec = emb_w[tok.eos_token_id].view(1, 1, -1).detach()

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    cases = []
    for ds in ["imdb", "sst2"]:
        for k, (text, pr, gold) in enumerate(load_cases(ds, args.per, tok, model, dev, llq)):
            if tok(pr, return_tensors="pt").input_ids.shape[1] <= args.tcap:
                cases.append((f"{ds}_{k}", pr, gold))
    print(f"cases: {len(cases)}", flush=True)

    mlrp = LlamaForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True).to(dev).eval()
    mlrp.gradient_checkpointing_enable(); attnlrp.register(mlrp); L = mlrp.config.num_hidden_layers

    def lrp_rel(prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        e = mlrp.get_input_embeddings()(ids)
        pid = torch.arange(0.0, ids.shape[1], device=dev, requires_grad=True, dtype=torch.float32).reshape(1, -1)
        pe = [mlrp.get_input_pos_embeddings()(e, pid) for _ in range(L)]
        pe = [(x[0].requires_grad_(), x[1].requires_grad_()) for x in pe]
        lg = mlrp(inputs_embeds=e.requires_grad_(), position_embeddings=pe, use_cache=False)["logits"]
        t = lg[0, -1, ans]; t.backward(t)
        at = e.grad.float().sum(-1).cpu()[0]; at = at / at.abs().max()
        acc = torch.zeros_like(at)
        for p in pe:
            for i in range(2):
                acc = acc + torch.matmul(p[i].grad.abs(), p[i].transpose(-1, -2).abs()).detach().float().sum(-1).cpu()[0].abs()
        acc = acc / acc.abs().max(); pa = at + acc; pa = pa / pa.abs().max()
        return at.numpy(), pa.numpy()
    lrp_cache = {}
    for key, prompt, ans in cases:
        try:
            lrp_cache[key] = lrp_rel(prompt, ans)
        except Exception as e:  # noqa: BLE001
            print(f"  LRP {key} FAIL {str(e)[:50]}", flush=True)
        mlrp.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    del mlrp; torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        pmean = emb[0, nz].mean(0).view(1, 1, -1)                       # prompt-mean (in-dist)
        FILLS = {"vocab": gmean, "eos": eos_vec, "pmean": pmean}
        with torch.no_grad():
            H = model(ids, output_hidden_states=True).hidden_states[-1][0].float().cpu().numpy()

        @torch.no_grad()
        def prob(M, fill):
            e = emb * M[:, :, None] + fill * (1 - M[:, :, None])
            return F.softmax(llq(e), -1)[:, ans].cpu().numpy()

        @torch.no_grad()
        def entropy_at(f, fill, seed=0):                                # OOD proxy: random-removal entropy
            g = np.random.default_rng(seed); k = int(round(f * len(nz)))
            m = torch.ones(T, device=dev)
            if k:
                m[torch.as_tensor(list(g.choice(nz, k, replace=False)), device=dev)] = 0.0
            e = emb * m[None, :, None] + fill * (1 - m[None, :, None])
            pr = F.softmax(llq(e)[0], -1); return float(-(pr * (pr + 1e-9).log()).sum())

        @torch.no_grad()
        def del_auc(order, fill, frgrid):
            order = [i for i in order if i in nz]; ps = []
            for f in frgrid:
                k = int(round(f * len(order))); m = torch.ones(T, device=dev)
                if k:
                    m[torch.as_tensor(order[:k], device=dev)] = 0.0
                ps.append(prob(m[None], fill)[0])
            return float(np.trapz(ps, frgrid) / frgrid[-1])

        @torch.no_grad()
        def single_occ(fill):
            sc = np.zeros(T)
            for s in range(0, len(nz), args.bs):
                idx = nz[s:s + args.bs]; M = torch.ones(len(idx), T, device=dev)
                for r, i in enumerate(idx):
                    M[r, i] = 0.0
                pp = prob(M, fill)
                for r, i in enumerate(idx):
                    sc[i] = -pp[r]
            return [i for i in np.argsort(-sc) if i in nz]

        @torch.no_grad()
        def cluster(fill):
            nzi = np.array(nz); lab = kmeans(H[nzi], min(args.G, len(nz))); norm = np.linalg.norm(H[nzi], axis=1)
            groups = [nzi[lab == g] for g in range(lab.max() + 1) if (lab == g).any()]
            km = np.ones(T, np.float32); order = []; rem = list(range(len(groups)))
            while rem:
                masks = []
                for ci in rem:
                    mm = km.copy(); mm[groups[ci]] = 0.0; masks.append(mm)
                pp = prob(torch.tensor(np.stack(masks), device=dev), fill); jb = int(np.argmin(pp)); ci = rem[jb]
                for p in sorted(groups[ci].tolist(), key=lambda q: -norm[list(nzi).index(q)]):
                    km[p] = 0.0; order.append(int(p))
                rem.pop(jb)
            return order

        @torch.no_grad()
        def greedy(fill):
            km = np.ones(T, np.float32); order = []; rem = list(nz)
            while rem:
                masks = []
                for i in rem:
                    mm = km.copy(); mm[i] = 0.0; masks.append(mm)
                pp = prob(torch.tensor(np.stack(masks), device=dev), fill); jb = int(np.argmin(pp))
                km[rem[jb]] = 0.0; order.append(rem[jb]); rem.pop(jb)
            return order

        at, pa = lrp_cache.get(key, (None, None))
        od = lambda s: [i for i in np.argsort(-np.asarray(s)) if i in nz]
        gd = greedy(eos_vec) if T <= 55 else None
        out = {"T": T, "ent": {fl: {f: round(entropy_at(f, FILLS[fl]), 2) for f in [0.1, 0.3, 0.5]} for fl in ["vocab", "eos"]}}
        # NAIVE pipeline: vocab-mean fill, FULL range. OOD-SOLVED: eos fill, RESTRICTED range.
        so_v, cl_v = single_occ(gmean), cluster(gmean)
        so_e, cl_e = single_occ(eos_vec), cluster(eos_vec)
        out["naive"] = {  # vocab fill, full range
            "single_occ": del_auc(so_v, gmean, FR_FULL), "cluster": del_auc(cl_v, gmean, FR_FULL),
            "AttnLRP": del_auc(od(at), gmean, FR_FULL) if at is not None else None,
            "PA-LRP": del_auc(od(pa), gmean, FR_FULL) if pa is not None else None}
        out["oodfix"] = {  # eos fill, restricted range
            "single_occ": del_auc(so_e, eos_vec, FR_RESTR), "cluster": del_auc(cl_e, eos_vec, FR_RESTR),
            "greedy": del_auc(gd, eos_vec, FR_RESTR) if gd is not None else None,
            "AttnLRP": del_auc(od(at), eos_vec, FR_RESTR) if at is not None else None,
            "PA-LRP": del_auc(od(pa), eos_vec, FR_RESTR) if pa is not None else None}
        return out

    rows = []
    for key, prompt, ans in cases:
        try:
            r = run(key, prompt, ans); r["key"] = key; rows.append(r)
            n, o = r["naive"], r["oodfix"]
            print(f"  {key:9s} T={r['T']:3d} | ent vocab@.5 {r['ent']['vocab'][0.5]:.1f} eos@.5 {r['ent']['eos'][0.5]:.1f} "
                  f"| NAIVE so {n['single_occ']:.3f} cl {n['cluster']:.3f} Attn {fmt(n['AttnLRP'])} PA {fmt(n['PA-LRP'])} "
                  f"| OODFIX so {o['single_occ']:.3f} cl {o['cluster']:.3f} Attn {fmt(o['AttnLRP'])} PA {fmt(o['PA-LRP'])}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    am = lambda blk, k: float(np.nanmean([r[blk][k] for r in rows if r[blk].get(k) is not None]))
    ent = lambda fl, f: float(np.mean([r["ent"][fl][f] for r in rows]))
    print(f"\n=== LLM necessity under INPUT-SEQUENCE OOD fix (Llama-3.2-1B, raw-prob del, LOWER=better) n={len(rows)} ===")
    print(f"  OOD proxy entropy @50% removal:  vocab-mean {ent('vocab',0.5):.2f}  vs  eos-fill {ent('eos',0.5):.2f}  (lower=more in-dist)")
    print(f"  NAIVE  (vocab fill, full 0..0.5):  single_occ {am('naive','single_occ'):.3f}  cluster {am('naive','cluster'):.3f}  "
          f"AttnLRP {am('naive','AttnLRP'):.3f}  PA-LRP {am('naive','PA-LRP'):.3f}")
    print(f"  OODFIX (eos fill, restricted 0..0.1): single_occ {am('oodfix','single_occ'):.3f}  cluster {am('oodfix','cluster'):.3f}  "
          f"greedy {am('oodfix','greedy'):.3f}  AttnLRP {am('oodfix','AttnLRP'):.3f}  PA-LRP {am('oodfix','PA-LRP'):.3f}")
    nv = pct([min(r["naive"]["single_occ"], r["naive"]["cluster"]) < min(r["naive"]["AttnLRP"], r["naive"]["PA-LRP"]) for r in rows if r["naive"]["PA-LRP"] is not None])
    ov = pct([min(r["oodfix"]["single_occ"], r["oodfix"]["cluster"]) < min(r["oodfix"]["AttnLRP"], r["oodfix"]["PA-LRP"]) for r in rows if r["oodfix"]["PA-LRP"] is not None])
    print(f"  cheap necessity < LRP:  NAIVE {nv:.0f}%   ->   OODFIX {ov:.0f}%   (does solving OOD rescue cheap necessity?)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "llm_necessity_ood.json"), "w"), indent=2)


def fmt(v):
    return " NA  " if v is None else f"{v:.3f}"


def pct(lst):
    return (100.0 * float(np.mean(lst))) if lst else float("nan")


if __name__ == "__main__":
    main()
