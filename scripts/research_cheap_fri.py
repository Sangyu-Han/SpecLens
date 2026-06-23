#!/usr/bin/env python3
"""CHEAP cooperative FRI (~100 forward+backward budget) applying BOTH remaining Codex ideas onto my
cooperative Banzhaf, then cutting cost:
  (1) PPD-gradient candidates: |grad| read at an interpolation point alpha*emb+(1-alpha)*mean
      (alpha~0.3) as the candidate generator (ablated vs alpha=1.0 full-gradient).
  (2) NON-LOCAL closure = 2nd-order Banzhaf interaction (REUSE the same coalitions = free): boost a
      candidate that cooperates (positive pair-interaction) with a top-marginal candidate → groups
      selector<->value (Codex's open weak spot).
  COST CUT: cooperative Banzhaf over the K candidates only (non-candidates held REAL), small M.
      Since it is over K (not T), M~80 should suffice; sweep M=80/128/256 to show the plateau.
Methods compared (insertion AUC + COST in forwards): cheap variants vs AttnLRP (1 bwd) and
single_occ (T fwd, the strong but O(T) baseline). Cases: KV-retrieval + arithmetic + a few long."""
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
from research_llm_imdb_sufficiency import PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
SEL_FR = [.05, .1, .2]


def closure(score, radius=1, decay=0.95):
    s = np.asarray(score, np.float64).copy()
    for _ in range(radius):
        left = np.full_like(s, -1e18); right = np.full_like(s, -1e18)
        left[1:] = s[:-1] * decay; right[:-1] = s[1:] * decay
        s = np.maximum(s, np.maximum(left, right))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--nlong", type=int, default=3)
    ap.add_argument("--Mmax", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=1.0)
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
    pos = tok(" positive", add_special_tokens=False).input_ids[0]; neg = tok(" negative", add_special_tokens=False).input_ids[0]
    raw, _ = load_imdb(400, 3000); nl = 0
    for label, text in raw:
        pr = PROMPT_TMPL.format(text=text); ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        if int(ids.shape[1]) < 300:
            continue
        with torch.no_grad():
            if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == (pos if label == 1 else neg):
                allcases.append((f"long_{nl}", "long", pr, pos if label == 1 else neg)); nl += 1
        if nl >= args.nlong:
            break
    print(f"total: {len(allcases)} (long={nl})", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, _, pr, _ in allcases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, tname, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0], -1)[ans]); base = float(F.softmax(llq(emb * 0 + gmean)[0], -1)[ans])
        den = max(full - base, 1e-6)
        K = min(100 if T > 200 else 32, len(nz))

        @torch.no_grad()
        def sig(Z):
            return (F.softmax(llq(emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])), -1)[:, ans] - base) / den

        @torch.no_grad()
        def keep_rec(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])
        order_of = lambda sc: [i for i in np.argsort(-sc) if i in nz]
        ins_auc = lambda sc: float(np.trapz([keep_rec(order_of(sc), f) for f in FR], FR) / FR[-1])

        def ppd_grad(alpha):                                   # candidates at interpolation point
            e = (alpha * emb + (1 - alpha) * gmean).detach().requires_grad_(True)
            try:
                out = model(inputs_embeds=e, logits_to_keep=1)
            except TypeError:
                out = model(inputs_embeds=e)
            out.logits[0, -1, ans].backward()
            return e.grad[0].abs().sum(-1).detach().cpu().numpy()

        g_ppd = ppd_grad(args.alpha); g_full = g_ppd if abs(args.alpha - 1.0) < 1e-9 else ppd_grad(1.0)

        def cand_of(g):
            return [int(i) for i in np.argsort(-g) if i in nz][:K]

        def coalitions(cand):                                  # M_max coalitions over candidates (rest real)
            ct = torch.as_tensor(cand, device=dev); Zs = []; Rs = []; j = 0
            while j < args.Mmax:
                nb = min(args.bs, args.Mmax - j); zb = []
                for _ in range(nb):
                    z = torch.ones(T, device=dev); pf = float(torch.rand(1, generator=gen, device=dev).item())
                    z[ct] = (torch.rand(len(cand), generator=gen, device=dev) < pf).float()
                    z[0] = 1.0; z[readout] = 1.0; zb.append(z)
                Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs)
            return Z[:, ct].cpu().numpy(), R.cpu().numpy()      # Zc [M,K], R [M]

        def marg_at(Zc, R, Mp):
            z = Zc[:Mp]; r = R[:Mp]; n1 = z.sum(0)
            return (z * r[:, None]).sum(0) / np.clip(n1, 1, None) - ((1 - z) * r[:, None]).sum(0) / np.clip(Mp - n1, 1, None)

        def nonlocal_boost(phi, Zc, R, topn=8):                # 2nd-order interaction, free (reuse Zc,R)
            top = list(np.argsort(-phi)[:topn]); boost = np.zeros(len(phi))
            for i in range(len(phi)):
                best = 0.0; a = Zc[:, i]
                for jx in top:
                    if jx == i:
                        continue
                    b = Zc[:, jx]
                    def mm(mask):
                        return R[mask].mean() if mask.any() else 0.0
                    I = mm((a == 1) & (b == 1)) - mm((a == 1) & (b == 0)) - mm((a == 0) & (b == 1)) + mm((a == 0) & (b == 0))
                    best = max(best, float(I))
                boost[i] = best
            return boost

        def full_score(cand, phi, g):
            s = np.full(T, -1e9, np.float64)
            for idx, i in enumerate(cand):
                s[i] = phi[idx]
            cs = set(cand)
            for i in nz:
                if i not in cs:
                    s[i] = -1e6 + g[i] * 1e-3
            return s

        cand = cand_of(g_ppd); Zc, R = coalitions(cand)
        phi256 = marg_at(Zc, R, args.Mmax); phi128 = marg_at(Zc, R, 128); phi80 = marg_at(Zc, R, 80)
        nb = nonlocal_boost(phi256, Zc, R)
        if abs(args.alpha - 1.0) < 1e-9:
            cand_f, phi_full = cand, phi128                 # alpha=1.0: fullcand == m128 (skip redundant work)
        else:
            cand_f = cand_of(g_full); Zcf, Rf = coalitions(cand_f); phi_full = marg_at(Zcf, Rf, 128)

        res = {"T": T, "K": K}
        res["m80"] = ins_auc(full_score(cand, phi80, g_ppd))
        res["m128"] = ins_auc(full_score(cand, phi128, g_ppd))
        res["m256"] = ins_auc(full_score(cand, phi256, g_ppd))
        res["m128_loc"] = ins_auc(closure(full_score(cand, phi128, g_ppd)))
        res["m128_nonloc"] = ins_auc(full_score(cand, phi128 + args.beta * nb, g_ppd))
        res["m128_both"] = ins_auc(closure(full_score(cand, phi128 + args.beta * nb, g_ppd)))
        res["m128_fullcand"] = ins_auc(full_score(cand_f, phi_full, g_full))   # alpha=1.0 ablation
        # self-selection over a cheap portfolio at M=80 budget
        port = {"m80": full_score(cand, phi80, g_ppd),
                "m80_loc": closure(full_score(cand, phi80, g_ppd)),
                "m80_nonloc": full_score(cand, marg_at(Zc, R, 80) + args.beta * nonlocal_boost(phi80, Zc, R), g_ppd)}
        part = {nm: sum(keep_rec(order_of(sc), f) for f in SEL_FR) for nm, sc in port.items()}
        sel = max(port, key=lambda nm: part[nm])
        res["SELECT80"] = ins_auc(port[sel]); res["_sel"] = sel
        res["cost_select"] = 80 + len(port) * len(SEL_FR)        # forwards (+1 bwd for candidates)
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        res["attn"] = ins_auc(att) if att is not None else float("nan")
        # single_occ baseline (cost T)
        occ = np.zeros(T, np.float32)
        with torch.no_grad():
            for s0 in range(0, len(nz), args.bs):
                idx = nz[s0:s0 + args.bs]; M = torch.ones(len(idx), T, device=dev)
                for r0, i in enumerate(idx):
                    M[r0, i] = 0.0
                rr = sig(M).cpu().numpy()
                for r0, i in enumerate(idx):
                    occ[i] = -rr[r0]
        res["single_occ"] = ins_auc(occ); res["cost_occ"] = len(nz)
        return res

    rows = []
    for key, tname, prompt, ans in allcases:
        try:
            r = run(key, tname, prompt, ans); r.update(key=key, t=tname); rows.append(r)
            print(f"  {key:9s} T={r['T']:3d} K={r['K']:3d} | m80={r['m80']:.3f} m128={r['m128']:.3f} "
                  f"both={r['m128_both']:.3f} SEL80={r['SELECT80']:.3f}({r['_sel']}) fullcand={r['m128_fullcand']:.3f} "
                  f"| attn={r['attn']:.3f} occ={r['single_occ']:.3f}(T{r['cost_occ']})", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== CHEAP cooperative FRI (insertion AUC; cost in forwards +1 backward) ===")
    keys = ["m80", "m128", "m256", "m128_loc", "m128_nonloc", "m128_both", "m128_fullcand", "SELECT80", "attn", "single_occ"]
    for t in ["kv", "arith", "long", "ALL"]:
        rs = [r for r in rows if t == "ALL" or r["t"] == t]
        if not rs:
            continue
        cells = " ".join(f"{k}={np.nanmean([r.get(k, np.nan) for r in rs]):.3f}" for k in keys)
        print(f"  {t:5s} n={len(rs):2d} | {cells}")
    sc = np.mean([r["cost_select"] for r in rows]); oc = np.mean([r["cost_occ"] for r in rows])
    print(f"\nCOST: SELECT80 ~= {sc:.0f} forwards + 1 backward  |  single_occ = {oc:.0f} forwards  |  AttnLRP = 1 backward")
    print("(alpha=0.3 PPD vs full: compare m128 vs m128_fullcand; non-local: m128 vs m128_nonloc; target ~100 met by SELECT80)")
    json.dump([{k: v for k, v in r.items()} for r in rows], open(os.path.join(REPO, "outputs", "cheap_fri.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
