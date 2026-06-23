#!/usr/bin/env python3
"""VERIFY: are AttnLRP's NEGATIVE-relevance tokens genuine negative contributors (distractors)?
Ground truth signed contribution per token: (a) signed occlusion so_i = full_logit - logit(i->mean)
[>0 supporter, <0 distractor], (b) signed Banzhaf marginal bz_i = E[rec|i in]-E[rec|i out].
Checks: corr(AttnLRP, so/bz); for the AttnLRP-NEGATIVE tokens, are so/bz actually negative?; does
REMOVING all AttnLRP-negative tokens raise the answer prob (they were opposing)?; decode the most-
negative AttnLRP tokens. If yes -> AttnLRP's sign is real -> use it to improve FRI."""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import BINS, PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "third_party/LRP-eXplains-Transformers-main (1)/LRP-eXplains-Transformers-main/examples/quantized_qwen2.py"


def load_everest_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-imdb", type=int, default=3)
    ap.add_argument("--M", type=int, default=1500)
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

    raw_all, _ = load_imdb(400, 2000)
    cases = [("everest", load_everest_prompt(), None)]
    per_bin = max(1, args.n_imdb // len(BINS)); binc = {b[0]: 0 for b in BINS}
    for j, (label, text) in enumerate(raw_all):
        ids = tok(PROMPT_TMPL.format(text=text), return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); b = next(n for n, lo, hi in BINS if lo <= T < hi)
        if binc[b] >= per_bin + 1:
            continue
        with torch.no_grad():
            pred = int(llq(model.get_input_embeddings()(ids))[0].argmax())
        if pred == (pos_id if label == 1 else neg_id):
            cases.append((f"r{j}", PROMPT_TMPL.format(text=text), pos_id if label == 1 else neg_id)); binc[b] += 1
        if len(cases) >= 1 + args.n_imdb and all(v >= 1 for v in binc.values()):
            break
    print(f"cases: {[c[0] for c in cases]}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, pr, _ in cases], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans_id, show=False):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1
        emb = model.get_input_embeddings()(ids).detach()
        nz = [i for i in range(T) if i not in (0, readout)]
        with torch.no_grad():
            lf = llq(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_logit = float(lf[ans]); full_p = float(F.softmax(lf, -1)[ans])
            base_p = float(F.softmax(llq(emb * 0 + gmean)[0], -1)[ans]); den = max(full_p - base_p, 1e-6)
        so = np.zeros(T, np.float32)
        with torch.no_grad():
            for s in range(0, len(nz), 16):
                idx = nz[s:s + 16]; e = emb.expand(len(idx), -1, -1).clone()
                for r, i in enumerate(idx):
                    e[r, i] = gmean[0, 0]
                lg = llq(e)[:, ans]
                for r, i in enumerate(idx):
                    so[i] = full_logit - float(lg[r])
        # signed Banzhaf marginal (difference of conditional means)
        Zs = []; Rs = []; j = 0
        while j < args.M:
            nb = min(args.bs, args.M - j); zb = []
            for _ in range(nb):
                pf = float(torch.rand(1, generator=gen, device=dev).item())
                z = (torch.rand(T, generator=gen, device=dev) < pf).float(); z[0] = 1.0; z[readout] = 1.0
                zb.append(z)
            Z = torch.stack(zb)
            with torch.no_grad():
                e = emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])
                Rs.append((F.softmax(llq(e), -1)[:, ans] - base_p) / den)
            Zs.append(Z); j += nb
        Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
        bz = ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        att = attn_cache.get(key)
        if att is None or len(att) != T:
            return None
        att = np.asarray(att, np.float32)
        negmask = np.array([att[i] < 0 for i in nz])
        nzi = np.array(nz)
        # remove AttnLRP-negative tokens (keep att>=0 + sink + readout)
        keep = torch.zeros(T, device=dev); keep[0] = 1.0; keep[readout] = 1.0
        keep[torch.as_tensor([i for i in nz if att[i] >= 0], device=dev)] = 1.0
        with torch.no_grad():
            rec_drop_neg = float((F.softmax(llq(emb * keep[None, :, None] + gmean * (1 - keep[None, :, None]))[0], -1)[ans] - base_p) / den)
        if show:
            order_neg = [i for i in np.argsort(att) if i not in (0, readout)][:10]
            print(f"\n  [{key}] most-NEGATIVE AttnLRP tokens: {tok.convert_ids_to_tokens([int(ids[0,i]) for i in order_neg])}")
            print(f"        their occlusion so: {[round(float(so[i]),2) for i in order_neg]} (negative = real distractor)")
        return dict(T=T, corr_att_so=float(spearmanr(att[nzi], so[nzi]).correlation),
                    corr_att_bz=float(spearmanr(att[nzi], bz[nzi]).correlation),
                    neg_frac=float(negmask.mean()),
                    neg_so=float(so[nzi][negmask].mean()) if negmask.any() else float("nan"),
                    neg_bz=float(bz[nzi][negmask].mean()) if negmask.any() else float("nan"),
                    pos_so=float(so[nzi][~negmask].mean()) if (~negmask).any() else float("nan"),
                    so_neg_given_attneg=float((so[nzi][negmask] < 0).mean()) if negmask.any() else float("nan"),
                    rec_drop_neg=rec_drop_neg)

    res = []
    for key, prompt, ans_id in cases:
        r = run(key, prompt, ans_id, show=(key == "everest"))
        if r:
            r["key"] = key; res.append(r); print(f"  {key} T={r['T']} neg%={100*r['neg_frac']:.0f} done", flush=True)

    print("\n=== Are AttnLRP's NEGATIVE tokens real negative contributors? ===")
    print(f"corr(AttnLRP, signed-occ) = {np.nanmean([r['corr_att_so'] for r in res]):+.3f} | "
          f"corr(AttnLRP, signed-Banzhaf) = {np.nanmean([r['corr_att_bz'] for r in res]):+.3f}")
    print(f"AttnLRP-negative tokens: {100*np.mean([r['neg_frac'] for r in res]):.0f}% of tokens")
    print(f"  their mean occlusion = {np.nanmean([r['neg_so'] for r in res]):+.4f} (vs positive-AttnLRP mean occ {np.nanmean([r['pos_so'] for r in res]):+.4f})")
    print(f"  their mean Banzhaf   = {np.nanmean([r['neg_bz'] for r in res]):+.4f}")
    print(f"  fraction of AttnLRP-neg that have NEGATIVE occlusion = {100*np.nanmean([r['so_neg_given_attneg'] for r in res]):.0f}% (50%=random, >50%=sign agrees)")
    print(f"REMOVE all AttnLRP-negative tokens -> recovery {np.mean([r['rec_drop_neg'] for r in res]):+.3f} "
          f"(full=1.0; >1 => they were opposing/harmful = real distractors; <1 => they were actually needed)")


if __name__ == "__main__":
    main()
