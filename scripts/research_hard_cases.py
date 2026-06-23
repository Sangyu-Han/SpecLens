#!/usr/bin/env python3
"""HARD / TIGHT-DEPENDENCY cases (vs redundant sentiment): context-reading copy (KV retrieval),
arithmetic, variable-tracking. Here the cooperative (sufficiency) set AND the necessary set are
SMALL and NON-REDUNDANT — every key token must be present (remove one -> wrong answer; keep only
them -> correct). Tests whether FRI finds the COMPLETE set.
INSERTION (sufficiency): FRI adaptive-Banzhaf, FRI gradient-guided, |grad|, AttnLRP.
DELETION  (necessity):  chunked-conditional, |grad|, AttnLRP."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
NAMES = ["Anna", "Ben", "Carl", "Dana", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack", "Kate", "Liam"]
CITIES = ["Paris", "Tokyo", "Rome", "Cairo", "Berlin", "Madrid", "London", "Oslo", "Lima", "Seoul", "Vienna", "Dublin"]


def gen_kv(rng, npairs=5):
    idx = rng.permutation(len(NAMES))[:npairs]; cidx = rng.permutation(len(CITIES))[:npairs]
    facts = " ".join(f"{NAMES[i]} lives in {CITIES[c]}." for i, c in zip(idx, cidx))
    qi = int(rng.integers(npairs))
    p = f"Facts: {facts}\nQuestion: Where does {NAMES[idx[qi]]} live?\nAnswer: {NAMES[idx[qi]]} lives in"
    return p, " " + CITIES[cidx[qi]]


def gen_arith(rng):
    a = int(rng.integers(11, 60)); b = int(rng.integers(11, 60))
    return f"Question: What is {a} plus {b}?\nAnswer: {a} plus {b} equals", " " + str(a + b)


def gen_vartrack(rng):
    x = int(rng.integers(2, 9)); y = int(rng.integers(2, 9)); z = int(rng.integers(2, 9))
    p = f"Let a = {x}. Let b = {y}. Let c = {z}. Let d = a + b + c.\nQuestion: What is d?\nAnswer: d ="
    return p, " " + str(x + y + z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--M", type=int, default=1200)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--Mcand", type=int, default=40)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--bs", type=int, default=12)
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

    # ---- generate + filter Qwen-correct ----
    allcases = []
    for tname, fn in [("kv", gen_kv), ("arith", gen_arith), ("vartrack", gen_vartrack)]:
        rng = np.random.default_rng(0); got = 0; tries = 0
        while got < args.per and tries < args.per * 30:
            tries += 1
            prompt, ans_str = fn(rng)
            gold = tok(ans_str, add_special_tokens=False).input_ids[0]
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == gold:
                    allcases.append((f"{tname}_{got}", tname, prompt, gold)); got += 1
        print(f"  {tname}: {got} correct cases", flush=True)
    print(f"total: {len(allcases)}", flush=True)
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
        def sig(Z):
            return (F.softmax(llq(emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])), -1)[:, ans] - base) / den

        @torch.no_grad()
        def keep_rec(order, f):                       # insertion: keep top-k from mean
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])

        @torch.no_grad()
        def del_rec(order, f):                        # deletion: remove top-k from full
            k = int(round(f * len(order))); m = torch.ones(T, device=dev)
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 0.0
            return float(sig(m[None])[0])

        ins_auc = lambda o: float(np.trapz([keep_rec([i for i in o if i in nz], f) for f in FR], FR) / FR[-1])
        del_auc = lambda o: float(np.trapz([del_rec([i for i in o if i in nz], f) for f in FR], FR) / FR[-1])

        # |grad|
        eg = emb.clone().requires_grad_(True)
        try:
            out = model(inputs_embeds=eg, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=eg)
        out.logits[0, -1, ans].backward()
        g = eg.grad[0].abs().sum(-1).cpu().numpy()
        cand = [i for i in np.argsort(-g) if i not in (0, readout)][:args.K]; candset = set(cand)
        cand_t = torch.as_tensor(cand, device=dev)

        def banzhaf(mode):
            Zs = []; Rs = []; j = 0
            while j < args.M:
                nb = min(args.bs, args.M - j); zb = []
                for _ in range(nb):
                    if mode == "guided":
                        z = torch.ones(T, device=dev)
                        pf = float(torch.rand(1, generator=gen, device=dev).item())
                        z[cand_t] = (torch.rand(len(cand), generator=gen, device=dev) < pf).float()
                    else:                                                 # adaptive: vary all, lo by length
                        lo = 0.0 if T < 100 else 0.5
                        pf = lo + (1 - lo) * float(torch.rand(1, generator=gen, device=dev).item())
                        z = (torch.rand(T, generator=gen, device=dev) < pf).float()
                    z[0] = 1.0; z[readout] = 1.0; zb.append(z)
                Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            return ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        # conditional (chunked) necessity order
        @torch.no_grad()
        def cond_order():
            prior = np.zeros(T, np.float32)
            for s in range(0, len(nz), args.bs):
                idx = nz[s:s + args.bs]; M = torch.ones(len(idx), T, device=dev)
                for r, i in enumerate(idx):
                    M[r, i] = 0.0
                rr = sig(M).cpu().numpy()
                for r, i in enumerate(idx):
                    prior[i] = 1.0 - rr[r]
            c = [int(i) for i in np.argsort(-prior) if i in nz][:args.Mcand]
            km = np.ones(T, np.float32); order = []; per = max(1, args.Mcand // args.R)
            while c:
                M = np.tile(km, (len(c), 1))
                for r, i in enumerate(c):
                    M[r, i] = 0.0
                rr = sig(torch.tensor(M, device=dev)).cpu().numpy()
                ix = list(np.argsort(rr)[:per])
                for jj in ix:
                    km[c[jj]] = 0.0; order.append(c[jj])
                for jj in sorted(ix, reverse=True):
                    c.pop(jj)
            order += [i for i in np.argsort(-prior) if i in nz and i not in set(order)]
            return order

        bz_ad = banzhaf("adapt"); bz_gg_m = banzhaf("guided")
        order_gg = sorted(cand, key=lambda i: -bz_gg_m[i]) + [i for i in np.argsort(-g) if i in nz and i not in candset]
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        a_ord = [i for i in np.argsort(-att) if i in nz] if att is not None else None
        g_ord = [i for i in np.argsort(-g) if i in nz]
        cond = cond_order()
        return dict(T=T,
                    ins_friad=ins_auc([i for i in np.argsort(-bz_ad) if i in nz]),
                    ins_frigg=ins_auc(order_gg), ins_grad=ins_auc(g_ord),
                    ins_attn=ins_auc(a_ord) if a_ord else float("nan"),
                    del_cond=del_auc(cond), del_grad=del_auc(g_ord),
                    del_attn=del_auc(a_ord) if a_ord else float("nan"))

    rows = []
    for key, tname, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r.update(key=key, t=tname); rows.append(r)
            print(f"  {key:12s} T={r['T']:3d} | INS fri-ad={r['ins_friad']:.3f} fri-gg={r['ins_frigg']:.3f} "
                  f"|grad|={r['ins_grad']:.3f} attn={r['ins_attn']:.3f} | DEL cond={r['del_cond']:.3f} "
                  f"|grad|={r['del_grad']:.3f} attn={r['del_attn']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== HARD/TIGHT cases ===")
    print("INSERTION (sufficiency, HIGHER=better): FRI-adaptive / FRI-grad-guided / |grad| / AttnLRP")
    for t in ["kv", "arith", "vartrack", "ALL"]:
        rs = [r for r in rows if t == "ALL" or r["t"] == t]
        if not rs:
            continue
        f1 = np.mean([r["ins_friad"] for r in rs]); f2 = np.mean([r["ins_frigg"] for r in rs])
        gg = np.mean([r["ins_grad"] for r in rs]); at = np.nanmean([r["ins_attn"] for r in rs])
        w = 100.0 * np.nanmean([float(max(r["ins_friad"], r["ins_frigg"]) > r["ins_attn"]) for r in rs if r["ins_attn"] == r["ins_attn"]])
        print(f"  {t:9s} n={len(rs):2d} | fri-ad={f1:.3f} fri-gg={f2:.3f} |grad|={gg:.3f} attn={at:.3f} | bestFRI>attn {w:.0f}%")
    print("DELETION (necessity, LOWER=better): conditional / |grad| / AttnLRP")
    for t in ["kv", "arith", "vartrack", "ALL"]:
        rs = [r for r in rows if t == "ALL" or r["t"] == t]
        if not rs:
            continue
        cd = np.mean([r["del_cond"] for r in rs]); gg = np.mean([r["del_grad"] for r in rs]); at = np.nanmean([r["del_attn"] for r in rs])
        w = 100.0 * np.nanmean([float(r["del_cond"] < min(r["del_grad"], r["del_attn"] if r["del_attn"] == r["del_attn"] else 9)) for r in rs])
        print(f"  {t:9s} n={len(rs):2d} | cond={cd:.3f} |grad|={gg:.3f} attn={at:.3f} | cond best {w:.0f}%")
    json.dump(rows, open(os.path.join(REPO, "outputs", "hard_cases.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
