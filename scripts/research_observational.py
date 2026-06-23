#!/usr/bin/env python3
"""User's synthesis: not only the masked VECTOR but the whole INPUT must stay in-distribution.
This is the INTERVENTIONAL vs OBSERVATIONAL distinction in attribution:
  - INTERVENTIONAL (replace token with a fixed mean vector): vector in-distribution but heavy
    masking -> whole sequence OOD (the sequence-OOD failure).
  - OBSERVATIONAL (marginalize the token out by NOT attending to it = attention-masking/removal):
    the kept tokens form a valid (shorter) sequence -> input in-distribution at EVERY masking level.
My restricted-range Banzhaf win was a HACK approximating observational (high keep-frac ~ mostly
observed). The clean version = true observational erasure (attention-masking).

TEST: gradedness (next-token entropy at keep-frac) MEAN-mask vs ATTENTION-mask (does attn stay
in-distribution under heavy masking?) + FULL-range Banzhaf for each -> insertion AUC (mean-metric,
for AttnLRP comparison). If attention-masking's full-range Banzhaf ~ or > the restricted-mean win
(0.577) and > AttnLRP, the observational principle is the clean root fix (no range restriction)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import BINS, PROMPT_TMPL, load_imdb, precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "third_party/LRP-eXplains-Transformers-main (1)/LRP-eXplains-Transformers-main/examples/quantized_qwen2.py"
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]


def load_everest_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-imdb", type=int, default=5)
    ap.add_argument("--M", type=int, default=2500)
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

    def llq(e, am=None, pos=None):       # last-position logits, memory-safe
        kw = {}
        if am is not None:
            kw["attention_mask"] = am; kw["position_ids"] = pos
        try:
            return model(inputs_embeds=e, logits_to_keep=1, **kw).logits[:, -1]
        except TypeError:
            return model(inputs_embeds=e, **kw).logits[:, -1]

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

    def run(key, prompt, ans_id):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1
        emb = model.get_input_embeddings()(ids).detach()
        pos = torch.arange(T, device=dev)[None]

        def fix(Z):
            Z[..., readout] = 1.0; Z[..., 0] = 1.0; return Z
        with torch.no_grad():
            lf = llq(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_p = float(F.softmax(lf, -1)[ans])
            m0 = fix(torch.zeros(1, T, device=dev))
            base_m = float(F.softmax(llq(emb * m0[:, :, None] + gmean * (1 - m0[:, :, None]))[0], -1)[ans])
            base_a = float(F.softmax(llq(emb, am=m0, pos=pos)[0], -1)[ans])
        den_m = max(full_p - base_m, 1e-6); den_a = max(full_p - base_a, 1e-6)

        @torch.no_grad()
        def sig_mean(Z):
            sm = F.softmax(llq(emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])), -1)
            return (sm[:, ans] - base_m) / den_m, sm

        @torch.no_grad()
        def sig_attn(Z):
            am = fix(Z.clone()); B = Z.shape[0]
            sm = F.softmax(llq(emb.expand(B, -1, -1), am=am, pos=pos.expand(B, -1)), -1)
            return (sm[:, ans] - base_a) / den_a, sm

        def ent(sm):
            return float((-(sm * torch.log(sm + 1e-12)).sum(-1)).mean())

        grad = {}
        for a in [0.05, 0.1, 0.3]:
            Z = fix((torch.rand(6, T, generator=gen, device=dev) < a).float())
            grad[("mean", a)] = ent(sig_mean(Z)[1]); grad[("attn", a)] = ent(sig_attn(Z)[1])

        @torch.no_grad()
        def rec_hard(order, f):              # mean-baseline metric (fixed, for AttnLRP comparison)
            k = int(round(f * len(order))); m = fix(torch.zeros(1, T, device=dev))[0]
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig_mean(m[None])[0][0])

        def ins_auc(scores):
            order = [i for i in np.argsort(-scores) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        def banzhaf(sigfn):
            Zs = []; Rs = []; j = 0
            while j < args.M:
                nb = min(args.bs, args.M - j); zb = []
                for _ in range(nb):
                    pf = float(torch.rand(1, generator=gen, device=dev).item())
                    zb.append(fix((torch.rand(T, generator=gen, device=dev) < pf).float()))
                Z = torch.stack(zb); Rs.append(sigfn(Z)[0]); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            s1 = (Z * R[:, None]).sum(0) / n1.clamp(min=1)
            s0 = ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)
            return (s1 - s0).cpu().numpy()

        a = attn_cache.get(key)
        ia = ins_auc(np.asarray(a, np.float32)) if a is not None and len(a) == T else float("nan")
        return dict(T=T, grad=grad, ins_attnlrp=ia, ins_mean=ins_auc(banzhaf(sig_mean)),
                    ins_attnmask=ins_auc(banzhaf(sig_attn)))

    rows = []
    for key, prompt, ans_id in cases:
        r = run(key, prompt, ans_id); r["key"] = key; rows.append(r)
        print(f"  {key:8s} T={r['T']:4d} | mean-mask BZ={r['ins_mean']:.3f} attn-mask BZ={r['ins_attnmask']:.3f} "
              f"| attnlrp={r['ins_attnlrp']:.3f} | a=.1 ent mean={r['grad'][('mean',0.1)]:.1f} attn={r['grad'][('attn',0.1)]:.1f}", flush=True)

    print("\n=== gradedness: entropy at keep-frac, INTERVENTIONAL(mean) vs OBSERVATIONAL(attn-mask) ===")
    for a in [0.05, 0.1, 0.3]:
        em = np.mean([r["grad"][("mean", a)] for r in rows]); ea = np.mean([r["grad"][("attn", a)] for r in rows])
        print(f"  keep={a:.2f} | mean ent={em:.2f} | attn-mask ent={ea:.2f} | drop={em-ea:+.2f}")
    print("\n=== FULL-range Banzhaf insertion AUC: interventional vs observational vs AttnLRP ===")
    def binof(T):
        return next(n for n, lo, hi in BINS if lo <= T < hi)
    groups = {"short": [], "medium": [], "long": [], "ALL": []}
    for r in rows:
        if r["key"] != "everest":
            groups[binof(r["T"])].append(r)
        groups["ALL"].append(r)
    for g, rs in groups.items():
        if not rs:
            continue
        bm = np.mean([r["ins_mean"] for r in rs]); ba = np.mean([r["ins_attnmask"] for r in rs])
        at = np.nanmean([r["ins_attnlrp"] for r in rs])
        wa = 100.0 * np.nanmean([float(r["ins_attnmask"] > r["ins_attnlrp"]) for r in rs if r["ins_attnlrp"] == r["ins_attnlrp"]])
        print(f"  {g:7s} n={len(rs):2d} | mean(interv)={bm:.3f} attnmask(observ)={ba:.3f} attnlrp={at:.3f} | observ>attn {wa:.0f}%")
    print("(if attn-mask entropy << mean at heavy masking AND attnmask full-range >> mean full-range / > attnlrp")
    print(" -> observational erasure keeps the INPUT in-distribution at all levels = the clean root fix; ref restricted-mean win=0.577)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "observational.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
