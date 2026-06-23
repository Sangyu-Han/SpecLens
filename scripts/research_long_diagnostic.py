#!/usr/bin/env python3
"""WHY does FRI (restricted Banzhaf) lose on LONG insertion while winning short? Distinguish:
(1) SEQUENCE-OOD: entropy of the FRI's own masks (keep-frac ~U(0.5,1)) on long cases — is it still
    OOD (high entropy) even in the restricted regime?
(2) ESTIMATION NOISE: Banzhaf at M=1500 vs M=4000 (subsampled) — does more M close the gap?
(3) CEILING/intrinsic: occlusion-order insertion AUC (ceiling) vs AttnLRP vs FRI — is AttnLRP ~ceiling
    while FRI << ceiling (FRI's estimation is the gap), or is the ceiling itself the limit?
+ corr(Banzhaf, occlusion). Long IMDB cases (T>300)."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--Mbig", type=int, default=4000)
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
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    raw, _ = load_imdb(600, 3000)
    cases = []
    for label, text in raw:
        pr = PROMPT_TMPL.format(text=text); ids = tok(pr, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1])
        if T < 300:
            continue
        with torch.no_grad():
            if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == (pos_id if label == 1 else neg_id):
                cases.append((f"L{len(cases)}", pr, pos_id if label == 1 else neg_id))
        if len(cases) >= args.n:
            break
    print(f"long cases (T>300): {len(cases)}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in cases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0], -1)[ans]); full_logit = float(llq(emb)[0][ans])
            base = float(F.softmax(llq(emb * 0 + gmean)[0], -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def fwd(Z):                                  # Z[B,T] -> (recovery, entropy) batched
            e = emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])
            sm = F.softmax(llq(e), -1)
            return ((sm[:, ans] - base) / den).cpu().numpy(), (-(sm * torch.log(sm + 1e-12)).sum(-1)).cpu().numpy()

        # (1) sequence-OOD: entropy of FRI's own masks (keep-frac U(0.5,1))
        ents = []
        for _ in range(2):
            zb = []
            for _ in range(8):
                pf = 0.5 + 0.5 * float(torch.rand(1, generator=gen, device=dev).item())
                z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[0] = 1.0; z[readout] = 1.0
                zb.append(z)
            ents.append(fwd(torch.stack(zb))[1].mean())
        ent_fri = float(np.mean(ents))
        # real full-prompt entropy (in-distribution reference)
        with torch.no_grad():
            sm0 = F.softmax(llq(emb)[0], -1); ent_full = float(-(sm0 * torch.log(sm0 + 1e-12)).sum())

        # occlusion (signed) for ceiling + corr
        occ = np.zeros(T, np.float32)
        with torch.no_grad():
            for s in range(0, len(nz), 16):
                idx = nz[s:s + 16]; e = emb.expand(len(idx), -1, -1).clone()
                for r, i in enumerate(idx):
                    e[r, i] = gmean[0, 0]
                lg = F.softmax(llq(e), -1)[:, ans].cpu().numpy()
                for r, i in enumerate(idx):
                    occ[i] = full - float(lg[r] * den + base)  # ~ logit-free necessity proxy; use prob drop
                    occ[i] = full - float(lg[r])

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(fwd(m[None])[0][0])

        def ins_auc(sc):
            order = [i for i in np.argsort(-sc) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        # (2) Banzhaf at Mbig, subsample to 1500
        Zs = []; Rs = []; j = 0
        while j < args.Mbig:
            nb = min(args.bs, args.Mbig - j); zb = []
            for _ in range(nb):
                pf = 0.5 + 0.5 * float(torch.rand(1, generator=gen, device=dev).item())
                z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[0] = 1.0; z[readout] = 1.0
                zb.append(z)
            Z = torch.stack(zb); Rs.append(torch.tensor(fwd(Z)[0], device=dev)); Zs.append(Z); j += nb
        Z = torch.cat(Zs); R = torch.cat(Rs)

        def bz_at(Mp):
            Zp = Z[:Mp]; Rp = R[:Mp]; n1 = Zp.sum(0)
            return ((Zp * Rp[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Zp) * Rp[:, None]).sum(0) / (Mp - n1).clamp(min=1)).cpu().numpy()
        bz1500 = bz_at(1500); bz_big = bz_at(args.Mbig)
        att = attn_cache.get(key); att = np.asarray(att, np.float32) if att is not None and len(att) == T else None
        nzi = np.array(nz)
        return dict(T=T, ent_fri=ent_fri, ent_full=ent_full,
                    fri_1500=ins_auc(bz1500), fri_big=ins_auc(bz_big),
                    attn=ins_auc(att) if att is not None else float("nan"),
                    ceiling=ins_auc(occ),
                    corr_bz_occ=float(spearmanr(bz_big[nzi], occ[nzi]).correlation),
                    corr_attn_occ=float(spearmanr(att[nzi], occ[nzi]).correlation) if att is not None else float("nan"))

    rows = []
    for key, prompt, ans in cases:
        try:
            r = run(key, prompt, ans); r["key"] = key; rows.append(r)
            print(f"  {key} T={r['T']:4d} | ent_fri={r['ent_fri']:.2f}(full {r['ent_full']:.2f}) | "
                  f"FRI M1500={r['fri_1500']:.3f} M{args.Mbig}={r['fri_big']:.3f} | attn={r['attn']:.3f} ceil={r['ceiling']:.3f} | "
                  f"corr(bz,occ)={r['corr_bz_occ']:+.2f} corr(attn,occ)={r['corr_attn_occ']:+.2f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print("\n=== LONG diagnostic (mean over cases) ===")
    print(f"(1) SEQUENCE-OOD?  FRI-mask entropy = {np.mean([r['ent_fri'] for r in rows]):.2f}  (real full-prompt = {np.mean([r['ent_full'] for r in rows]):.2f}; high => OOD)")
    print(f"(2) NOISE?         FRI M1500 = {np.mean([r['fri_1500'] for r in rows]):.3f} -> M{args.Mbig} = {np.mean([r['fri_big'] for r in rows]):.3f}  (big jump => estimation noise)")
    print(f"(3) CEILING/intr.  FRI = {np.mean([r['fri_big'] for r in rows]):.3f} | AttnLRP = {np.nanmean([r['attn'] for r in rows]):.3f} | occlusion-ceiling = {np.mean([r['ceiling'] for r in rows]):.3f}")
    print(f"    corr(Banzhaf,occ) = {np.nanmean([r['corr_bz_occ'] for r in rows]):+.3f} vs corr(AttnLRP,occ) = {np.nanmean([r['corr_attn_occ'] for r in rows]):+.3f}")
    print("(diagnosis: high ent_fri=>OOD; M jump=>noise; AttnLRP~ceiling & FRI<<ceiling & corr(bz)<corr(attn)=>FRI's estimate is the gap, not the metric)")


if __name__ == "__main__":
    main()
