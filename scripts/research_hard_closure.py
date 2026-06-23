#!/usr/bin/env python3
"""Borrow Codex's advantages onto MY FRI (gradient-guided restricted-Banzhaf, which captures
COOPERATIVE effects unlike Codex's single-occ-on-candidates) and test the synthesis on hard/tight
cases:
  (A) tokenizer-local closure  — radius-1 max-diffusion of the score to neighbors, repairs BPE
      fragments ("17"->"1","7"). Codex's standout idea; I lacked it.
  (B) compact self-selection   — validate a few cheap orders by partial hard-insertion, pick best.
Compare: fri-gg, fri-gg+closure, fri-ad+closure, SELECT(portfolio), |grad|+closure, AttnLRP, AttnLRP+closure.
Question: does closure help my FRI (esp. arithmetic multi-token operands)? does self-selection hedge?"""
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
from research_llm_imdb_sufficiency import precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
SEL_FR = [.05, .1, .2]                                   # partial-insertion points for self-selection


def closure(score, radius=1, decay=0.95):
    """radius-r max-diffusion along the sequence: boost a token toward its high-scoring neighbors."""
    s = np.asarray(score, np.float64).copy()
    for _ in range(radius):
        left = np.zeros_like(s); right = np.zeros_like(s)
        left[1:] = s[:-1] * decay; right[:-1] = s[1:] * decay
        s = np.maximum(s, np.maximum(left, right))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--M", type=int, default=1200)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--decay", type=float, default=0.95)
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

    allcases = []
    for tname, fn in [("kv", gen_kv), ("arith", gen_arith)]:
        rng = np.random.default_rng(0); got = 0; tries = 0
        while got < args.per and tries < args.per * 30:
            tries += 1
            prompt, ans_str = fn(rng); gold = tok(ans_str, add_special_tokens=False).input_ids[0]
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == gold:
                    allcases.append((f"{tname}_{got}", tname, prompt, gold)); got += 1
        print(f"  {tname}: {got}", flush=True)
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
        def keep_rec(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])

        def order_of(score):
            return [i for i in np.argsort(-score) if i in nz]
        ins_auc = lambda sc: float(np.trapz([keep_rec(order_of(sc), f) for f in FR], FR) / FR[-1])

        # |grad| + gradient-guided candidates
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
                        z = torch.ones(T, device=dev); pf = float(torch.rand(1, generator=gen, device=dev).item())
                        z[cand_t] = (torch.rand(len(cand), generator=gen, device=dev) < pf).float()
                    else:
                        lo = 0.0 if T < 100 else 0.5; pf = lo + (1 - lo) * float(torch.rand(1, generator=gen, device=dev).item())
                        z = (torch.rand(T, generator=gen, device=dev) < pf).float()
                    z[0] = 1.0; z[readout] = 1.0; zb.append(z)
                Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            return ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        gg = banzhaf("guided"); gg[list(candset)] = gg[list(candset)]  # candidates carry the marginal
        # for gg, non-candidates get -inf so they rank below; reuse |grad| for their relative order
        gg_full = np.full(T, -1e9, np.float32)
        for i in cand:
            gg_full[i] = gg[i]
        for i in nz:
            if i not in candset:
                gg_full[i] = -1e6 + g[i] * 1e-3
        ad = banzhaf("adapt")
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None

        scores = {"fri_gg": gg_full, "fri_ad": ad, "grad": g}
        if att is not None:
            scores["attn"] = att
        # closure variants
        clo = {f"{k}_clo": closure(v, args.radius, args.decay) for k, v in scores.items()}
        scores.update(clo)
        aucs = {k: ins_auc(v) for k, v in scores.items()}

        # compact self-selection over a cheap portfolio (validate by partial insertion)
        port = ["fri_gg", "fri_gg_clo", "fri_ad_clo", "grad_clo"]
        part = {nm: sum(keep_rec(order_of(scores[nm]), f) for f in SEL_FR) for nm in port}
        sel = max(port, key=lambda nm: part[nm])
        aucs["SELECT"] = aucs[sel]
        aucs["_sel"] = sel
        aucs["T"] = T
        return aucs

    rows = []
    for key, tname, prompt, ans in allcases:
        try:
            r = run(key, prompt, ans); r["key"] = key; r["t"] = tname; rows.append(r)
            print(f"  {key:10s} T={r['T']:3d} | gg={r['fri_gg']:.3f} gg+clo={r['fri_gg_clo']:.3f} "
                  f"ad+clo={r['fri_ad_clo']:.3f} SEL={r['SELECT']:.3f}({r['_sel']}) | attn={r.get('attn', float('nan')):.3f} "
                  f"attn+clo={r.get('attn_clo', float('nan')):.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== HARD cases: MY FRI + Codex closure/self-selection (INSERTION, higher=better) ===")
    keys = ["fri_gg", "fri_gg_clo", "fri_ad_clo", "SELECT", "grad_clo", "attn", "attn_clo"]
    for t in ["kv", "arith", "ALL"]:
        rs = [r for r in rows if t == "ALL" or r["t"] == t]
        if not rs:
            continue
        cells = " ".join(f"{k}={np.nanmean([r.get(k, np.nan) for r in rs]):.3f}" for k in keys)
        wbest = 100.0 * np.nanmean([float(max(r['fri_gg_clo'], r['SELECT']) > r.get('attn', -9)) for r in rs])
        print(f"  {t:6s} n={len(rs):2d} | {cells} | best>attn {wbest:.0f}%")
    print("(does closure raise arith? does SELECT match the best per-case? does closure also help attn?)")
    json.dump([{k: v for k, v in r.items() if k != '_sel' or True} for r in rows],
              open(os.path.join(REPO, "outputs", "hard_closure.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
