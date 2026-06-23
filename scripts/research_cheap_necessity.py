#!/usr/bin/env python3
"""DELETION / NECESSITY, applying the cheap-FRI inspirations. Questions:
  (1) Does the SAME cheap cooperative Banzhaf ranking that wins INSERTION also do DELETION well
      (remove high-marginal first)? -> one ~100-cost computation, BOTH faces.
  (2) Does candidate-restriction give a cheap CONDITIONAL necessity (|grad| top-K -> chunked greedy
      on candidates only, ~K*R forwards instead of T*R)?
  (3) Do closure / self-selection transfer to the necessity ranking?
  (4) Cross-model (Qwen + Llama-3.2-1B + gemma-2-2b).
Deletion AUC = recovery after removing top-k by each score (token->mean); LOWER = better necessity."""
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
from research_hard_cases import gen_arith, gen_kv  # noqa: E402
from research_xmodel_fri import closure, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
SEL_FR = [.05, .1, .2]


def run_model(model_path, dtype, per, M, K, R, bs, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = {"torch_dtype": dtype}
    if "gemma" in model_path.lower():
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:  # noqa: BLE001
        pass
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    cases = []
    for tname, fn in [("kv", gen_kv), ("arith", gen_arith)]:
        rng = np.random.default_rng(0); got = 0; tries = 0
        while got < per and tries < per * 40:
            tries += 1
            prompt, ans_str = fn(rng); gold = tok(ans_str, add_special_tokens=False).input_ids[0]
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == gold:
                    cases.append((f"{tname}_{got}", tname, prompt, gold)); got += 1
        print(f"    {tname}: {got}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, _, pr, _ in cases], model_path, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0].float(), -1)[ans]); base = float(F.softmax(llq(emb * 0 + gmean)[0].float(), -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def sig(Z):
            e = (emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])).to(emb.dtype)
            return (F.softmax(llq(e).float(), -1)[:, ans] - base) / den

        @torch.no_grad()
        def del_rec(order, f):                      # remove top-k -> recovery (lower=more necessary)
            k = int(round(f * len(order))); m = torch.ones(T, device=dev)
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 0.0
            return float(sig(m[None])[0])
        order_of = lambda sc: [i for i in np.argsort(-sc) if i in nz]
        del_auc = lambda sc: float(np.trapz([del_rec(order_of(sc), f) for f in FR], FR) / FR[-1])

        eg = emb.clone().requires_grad_(True)
        try:
            out = model(inputs_embeds=eg, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=eg)
        out.logits[0, -1, ans].backward()
        g = eg.grad[0].abs().float().sum(-1).cpu().numpy()
        kk = min(K, len(nz)); cand = [i for i in np.argsort(-g) if i in nz][:kk]; candset = set(cand); cand_t = torch.as_tensor(cand, device=dev)

        # gradient-guided cooperative Banzhaf (SAME signal as the insertion FRI)
        Zs = []; Rs = []; j = 0
        while j < M:
            nb = min(bs, M - j); zb = []
            for _ in range(nb):
                z = torch.ones(T, device=dev); pf = float(torch.rand(1, generator=gen, device=dev).item())
                z[cand_t] = (torch.rand(kk, generator=gen, device=dev) < pf).float()
                z[0] = 1.0; z[readout] = 1.0; zb.append(z)
            Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
        Z = torch.cat(Zs); Rr = torch.cat(Rs); n1 = Z.sum(0)
        marg = ((Z * Rr[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * Rr[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()
        bz = np.full(T, -1e9, np.float64)
        for i in cand:
            bz[i] = marg[i]
        for i in nz:
            if i not in candset:
                bz[i] = -1e6 + g[i] * 1e-3

        # cheap conditional necessity: single-occ prior on candidates -> chunked greedy on candidates
        @torch.no_grad()
        def cheap_cond():
            prior = {}
            for s0 in range(0, kk, bs):
                idx = cand[s0:s0 + bs]; Mb = torch.ones(len(idx), T, device=dev)
                for r0, i in enumerate(idx):
                    Mb[r0, i] = 0.0
                rr = sig(Mb).cpu().numpy()
                for r0, i in enumerate(idx):
                    prior[i] = 1.0 - rr[r0]
            c = sorted(cand, key=lambda i: -prior[i]); km = np.ones(T, np.float32); order = []; per_ = max(1, kk // R)
            while c:
                Mb = np.tile(km, (len(c), 1))
                for r0, i in enumerate(c):
                    Mb[r0, i] = 0.0
                rr = sig(torch.tensor(Mb, device=dev)).cpu().numpy()
                ix = list(np.argsort(rr)[:per_])
                for jj in ix:
                    km[c[jj]] = 0.0; order.append(c[jj])
                for jj in sorted(ix, reverse=True):
                    c.pop(jj)
            order += [i for i in np.argsort(-g) if i in nz and i not in set(order)]
            return order
        cond_order = cheap_cond()

        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        occ = np.zeros(T, np.float32)
        with torch.no_grad():
            for s0 in range(0, len(nz), bs):
                idx = nz[s0:s0 + bs]; Mb = torch.ones(len(idx), T, device=dev)
                for r0, i in enumerate(idx):
                    Mb[r0, i] = 0.0
                rr = sig(Mb).cpu().numpy()
                for r0, i in enumerate(idx):
                    occ[i] = 1.0 - rr[r0]
        cand_arr = np.zeros(T); cand_arr[cand] = 1  # for cond order via list
        orders = {"banzhaf": order_of(bz), "banzhaf_loc": order_of(closure(bz)),
                  "cheap_cond": [i for i in cond_order if i in nz], "single_occ": order_of(occ)}
        part = {nm: -sum(del_rec(o, f) for f in SEL_FR) for nm, o in orders.items()}  # lower del_rec better
        sel = max(part, key=lambda nm: part[nm])
        res = dict(T=T, kk=kk,
                   banzhaf=del_auc(bz), banzhaf_loc=del_auc(closure(bz)),
                   cheap_cond=float(np.trapz([del_rec(orders["cheap_cond"], f) for f in FR], FR) / FR[-1]),
                   single_occ=del_auc(occ), SELECT=float(np.trapz([del_rec(orders[sel], f) for f in FR], FR) / FR[-1]), _sel=sel,
                   attn=del_auc(att) if att is not None else float("nan"))
        res["cost_cheap_nec"] = M + kk + R * kk  # banzhaf M + single-occ-prior kk + chunked R*kk (rough)
        res["cost_full_occ"] = len(nz)
        return res

    rows = []
    for key, tname, prompt, ans in cases:
        try:
            r = run(key, prompt, ans); r.update(key=key, t=tname); rows.append(r)
            print(f"    {key:9s} T={r['T']:3d} | banzhaf={r['banzhaf']:.3f} bz_loc={r['banzhaf_loc']:.3f} cheap_cond={r['cheap_cond']:.3f} SELECT={r['SELECT']:.3f}({r['_sel']}) | attn={r['attn']:.3f} occ={r['single_occ']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"    {key} OOM", flush=True)
    del model; torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--M", type=int, default=800)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--bs", type=int, default=12)
    args = ap.parse_args()
    MODELS = [("Qwen/Qwen2.5-1.5B-Instruct", torch.float32, 12),
              ("meta-llama/Llama-3.2-1B", torch.float32, 12),
              ("google/gemma-2-2b", torch.bfloat16, 12)]
    allres = {}
    for mp, dtype, per in MODELS:
        print(f"\n##### MODEL {mp} #####", flush=True)
        try:
            allres[mp] = run_model(mp, dtype, per, args.M, args.K, args.R, args.bs, args.device)
        except Exception as e:  # noqa: BLE001
            import traceback; print(f"  MODEL FAIL {type(e).__name__}: {e}\n{traceback.format_exc()[-700:]}", flush=True)
    print("\n=== DELETION / NECESSITY (LOWER=better) — cheap cooperative Banzhaf & cheap-conditional vs AttnLRP/single_occ ===")
    for mp, rows in allres.items():
        if not rows:
            continue
        nm = mp.split('/')[-1]
        for t in ["kv", "arith", "ALL"]:
            rs = [r for r in rows if t == "ALL" or r["t"] == t]
            if not rs:
                continue
            agg = {k: np.nanmean([r[k] for r in rs]) for k in ["banzhaf", "banzhaf_loc", "cheap_cond", "SELECT", "attn", "single_occ"]}
            print(f"  {nm:22s} {t:5s} n={len(rs):2d} | banzhaf={agg['banzhaf']:.3f} bz_loc={agg['banzhaf_loc']:.3f} cheap_cond={agg['cheap_cond']:.3f} SELECT={agg['SELECT']:.3f} | attn={agg['attn']:.3f} occ={agg['single_occ']:.3f}")
        c = np.mean([r["cost_cheap_nec"] for r in rows]); o = np.mean([r["cost_full_occ"] for r in rows])
        print(f"      cost: cheap-necessity ~= {c:.0f} fwd +1bwd | single_occ(full) = {o:.0f} fwd")
    json.dump({k: v for k, v in allres.items()}, open(os.path.join(REPO, "outputs", "cheap_necessity.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
