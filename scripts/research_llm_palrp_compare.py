#!/usr/bin/env python3
"""Compare the NEW SOTA LRP (PA-LRP, faithful official code) + AttnLRP vs OUR perturbation methods
(FRI sufficiency, chunk_bz necessity, Banzhaf) on ONE insertion/deletion metric (raw-prob AUC).
LRP relevances come from the official PE-AWARE-LRP repo's llama_PE (user-authorized); the metric +
perturbation methods use a standard HF Llama with identical weights. Llama-3.2-1B-Instruct.
KEY Qs: (suff) does FRI insertion beat PA-LRP/AttnLRP? (nec) does chunk_bz deletion beat PA-LRP/AttnLRP?
Hypothesis: PA-LRP's gain over AttnLRP is positional; on content-driven sentiment it ~= AttnLRP."""
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
from research_insertion_datasets import FR, load_cases  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/"))[-1]

TIGHT_PROMPTS = [   # LOCALIZED evidence: answer hinges on ONE value token (perturbation should be sharp)
    "Variables: alpha=7, beta=4, gamma=9, delta=2. The value of gamma is",
    "Variables: alpha=7, beta=4, gamma=9, delta=2. The value of beta is",
    "Mapping: red=3, green=8, blue=5, gray=1. The number for blue is",
    "Mapping: red=3, green=8, blue=5, gray=1. The number for red is",
    "Inventory: apples=6, pears=2, plums=9, figs=4. The count of plums is",
    "Inventory: apples=6, pears=2, plums=9, figs=4. The count of pears is",
    "Note: the access code is 5 today. Please disregard the rest. The access code is",
    "Record: the patient id is 7. All other fields are blank. The patient id is",
    "The meeting is in room 8. Nothing else here matters. The meeting is in room",
    "Fact: the winner is team 3. Ignore the filler text. The winner is team",
]


def tight_cases(tok, model, dev, llq):
    cases = []
    for i, pr in enumerate(TIGHT_PROMPTS):
        ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            ans = int(torch.softmax(llq(model.get_input_embeddings()(ids))[0], -1).argmax())
        cases.append((f"tight_{i}", pr, ans))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--per", type=int, default=5)
    ap.add_argument("--M", type=int, default=256); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--Mcand", type=int, default=48); ap.add_argument("--R", type=int, default=8); ap.add_argument("--tcap", type=int, default=170)
    ap.add_argument("--task", default="sentiment", choices=["sentiment", "tight"])
    args = ap.parse_args()
    dev = args.device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lxt.models.llama_PE import LlamaForCausalLM, attnlrp                  # official PA-LRP (their fork)

    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    allcases = []
    if args.task == "tight":
        allcases = tight_cases(tok, model, dev, llq)
    else:
        for ds in ["imdb", "sst2"]:
            for k, (text, pr, gold) in enumerate(load_cases(ds, args.per, tok, model, dev, llq)):
                T = tok(pr, return_tensors="pt").input_ids.shape[1]
                if T <= args.tcap:
                    allcases.append((f"{ds}_{k}", pr, gold))
    print(f"[{args.task}] cases: {len(allcases)}", flush=True)

    # ---- LRP model (official llama_PE) ----
    mlrp = LlamaForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True).to(dev).eval()
    mlrp.gradient_checkpointing_enable(); attnlrp.register(mlrp)
    L = mlrp.config.num_hidden_layers

    def lrp_rel(prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        emb = mlrp.get_input_embeddings()(ids)
        pos_ids = torch.arange(0.0, ids.shape[1], device=dev, requires_grad=True, dtype=torch.float32).reshape(1, -1)
        pos_emb = [mlrp.get_input_pos_embeddings()(emb, pos_ids) for _ in range(L)]
        pos_emb = [(x[0].requires_grad_(), x[1].requires_grad_()) for x in pos_emb]
        logits = mlrp(inputs_embeds=emb.requires_grad_(), position_embeddings=pos_emb, use_cache=False)["logits"]
        tgt = logits[0, -1, ans]
        tgt.backward(tgt)
        attn = emb.grad.float().sum(-1).cpu()[0]; attn = attn / attn.abs().max()
        acc = torch.zeros_like(attn)
        for pe in pos_emb:
            for i in range(2):
                cr = torch.matmul(pe[i].grad.abs(), pe[i].transpose(-1, -2).abs()).detach().float().sum(-1).cpu()[0]
                acc = acc + cr.abs()
        acc = acc / acc.abs().max()
        pa = attn + acc; pa = pa / pa.abs().max()
        return attn.numpy(), pa.numpy()

    lrp_cache = {}
    for key, prompt, ans in allcases:
        try:
            lrp_cache[key] = lrp_rel(prompt, ans)
        except Exception as e:
            print(f"  LRP {key} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
        mlrp.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    del mlrp; torch.cuda.empty_cache()
    print(f"LRP relevances: {len(lrp_cache)}/{len(allcases)}", flush=True)

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0], -1)[ans])

        @torch.no_grad()
        def prob(M):
            e = emb * M[:, :, None] + gmean * (1 - M[:, :, None])
            return F.softmax(llq(e), -1)[:, ans].cpu().numpy()

        @torch.no_grad()
        def curve(order, insert):
            order = [i for i in order if i in nz]; ps = []
            for f in FR:
                k = int(round(f * len(order)))
                m = (torch.zeros if insert else torch.ones)(T, device=dev); m[0] = 1.0; m[readout] = 1.0
                if insert:
                    if k:
                        m[torch.as_tensor(order[:k], device=dev)] = 1.0
                else:
                    if k:
                        m[torch.as_tensor(order[:k], device=dev)] = 0.0
                ps.append(prob(m[None])[0])
            return float(np.trapz(ps, FR) / FR[-1])

        ins = lambda o: curve(o, True)      # higher=better (sufficiency)
        dele = lambda o: curve(o, False)    # lower=better  (necessity)

        def coalitions(lo):
            Zs, Rs, j = [], [], 0
            while j < args.M:
                nb = min(args.bs, args.M - j); zb = []
                for _ in range(nb):
                    pf = lo + (1 - lo) * float(torch.rand(1, generator=gen, device=dev).item())
                    z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[0] = 1.0; z[readout] = 1.0; zb.append(z)
                Z = torch.stack(zb); Rs.append(torch.tensor(prob(Z), device=dev)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            return ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        bz = coalitions(0.0)                                  # cooperative Banzhaf
        fri = coalitions(0.5 if T >= 100 else 0.0)            # FRI = restricted-budget Banzhaf

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

        occ = occ_prior()

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

        chunk = chunked(bz)

        @torch.no_grad()
        def greedy_del():                                     # O(T^2) oracle conditional necessity
            km = np.ones(T, np.float32); order = []; rem = list(nz)
            while rem:
                masks = []
                for i in rem:
                    m = km.copy(); m[i] = 0.0; masks.append(m)
                pp = prob(torch.tensor(np.stack(masks), device=dev)); j = int(np.argmin(pp))
                km[rem[j]] = 0.0; order.append(rem[j]); rem.pop(j)
            return order

        @torch.no_grad()
        def greedy_ins():                                     # O(T^2) ORACLE conditional sufficiency
            km = np.zeros(T, np.float32); km[0] = 1.0; km[readout] = 1.0; order = []; rem = list(nz)
            while rem:
                masks = []
                for i in rem:
                    m = km.copy(); m[i] = 1.0; masks.append(m)
                pp = prob(torch.tensor(np.stack(masks), device=dev)); j = int(np.argmax(pp))   # most prob INCREASE
                km[rem[j]] = 1.0; order.append(rem[j]); rem.pop(j)
            return order

        greedy = greedy_del() if T <= 60 else None            # only when cheap enough
        greedy_i = greedy_ins() if T <= 60 else None
        attn_r, pa_r = lrp_cache.get(key, (None, None))
        od = lambda s: [i for i in np.argsort(-np.asarray(s)) if i in nz]
        out = {"T": T, "full": round(full, 3)}
        # sufficiency (insertion, higher better)
        out["ins"] = {"greedy_ins": ins(greedy_i) if greedy_i is not None else None,
                      "FRI": ins(od(fri)), "banzhaf": ins(od(bz)),
                      "AttnLRP": ins(od(attn_r)) if attn_r is not None else None,
                      "PA-LRP": ins(od(pa_r)) if pa_r is not None else None}
        # necessity (deletion, lower better)
        out["del"] = {"greedy": dele(greedy) if greedy is not None else None,
                      "chunk_bz": dele(chunk), "single_occ": dele(od(occ)),
                      "AttnLRP": dele(od(attn_r)) if attn_r is not None else None,
                      "PA-LRP": dele(od(pa_r)) if pa_r is not None else None}
        return out

    rows = []
    for key, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r["key"] = key; rows.append(r)
            i, d = r["ins"], r["del"]
            print(f"  {key:9s} T={r['T']:3d} | INS Gi {fmt(i['greedy_ins'])} FRI {i['FRI']:.3f} Attn {fmt(i['AttnLRP'])} PA {fmt(i['PA-LRP'])} "
                  f"| DEL Gd {fmt(d['greedy'])} chunk {d['chunk_bz']:.3f} Attn {fmt(d['AttnLRP'])} PA {fmt(d['PA-LRP'])}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== PA-LRP / AttnLRP vs OURS (Llama-3.2-1B-Instruct, raw-prob AUC) ===")
    am = lambda face, k: float(np.nanmean([r[face][k] for r in rows if r[face][k] is not None]))
    print(f"  n={len(rows)}")
    ngi = [r for r in rows if r["ins"]["greedy_ins"] is not None]
    print(f"  SUFFICIENCY (insertion, HIGHER=better): greedy_ins {am('ins','greedy_ins'):.3f}(n={len(ngi)})  FRI {am('ins','FRI'):.3f}  "
          f"banzhaf {am('ins','banzhaf'):.3f}  PA-LRP {am('ins','PA-LRP'):.3f}  AttnLRP {am('ins','AttnLRP'):.3f}")
    gins_w = 100.0 * np.mean([r["ins"]["greedy_ins"] > max(r["ins"]["PA-LRP"], r["ins"]["AttnLRP"]) for r in ngi if r["ins"]["PA-LRP"] is not None]) if ngi else float("nan")
    print(f"  GREEDY-INS(oracle) ins > both LRP: {gins_w:.0f}%  (the sufficiency-oracle comparison)")
    ng = [r for r in rows if r["del"]["greedy"] is not None]
    print(f"  NECESSITY  (deletion, LOWER=better):   greedy {am('del','greedy'):.3f}(n={len(ng)})  chunk_bz {am('del','chunk_bz'):.3f}  "
          f"single_occ {am('del','single_occ'):.3f}  PA-LRP {am('del','PA-LRP'):.3f}  AttnLRP {am('del','AttnLRP'):.3f}")
    fri_w = 100.0 * np.mean([r["ins"]["FRI"] > max(r["ins"]["PA-LRP"], r["ins"]["AttnLRP"]) for r in rows if r["ins"]["PA-LRP"] is not None])
    nec_w = 100.0 * np.mean([r["del"]["chunk_bz"] < min(r["del"]["PA-LRP"], r["del"]["AttnLRP"]) for r in rows if r["del"]["PA-LRP"] is not None])
    grd_w = 100.0 * np.mean([r["del"]["greedy"] < min(r["del"]["PA-LRP"], r["del"]["AttnLRP"]) for r in ng if r["del"]["PA-LRP"] is not None]) if ng else float("nan")
    print(f"  GREEDY(oracle) del < both LRP: {grd_w:.0f}%  (the FAIR necessity comparison)")
    pa_vs_attn_ins = 100.0 * np.mean([r["ins"]["PA-LRP"] > r["ins"]["AttnLRP"] for r in rows if r["ins"]["PA-LRP"] is not None])
    pa_vs_attn_del = 100.0 * np.mean([r["del"]["PA-LRP"] < r["del"]["AttnLRP"] for r in rows if r["del"]["PA-LRP"] is not None])
    print(f"  FRI ins > both LRP: {fri_w:.0f}% | chunk_bz del < both LRP: {nec_w:.0f}% | PA-LRP>AttnLRP ins {pa_vs_attn_ins:.0f}% del {pa_vs_attn_del:.0f}%")
    json.dump(rows, open(os.path.join(REPO, "outputs", "llm_palrp_compare.json"), "w"), indent=2)


def fmt(v):
    return "  NA " if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
