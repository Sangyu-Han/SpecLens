#!/usr/bin/env python3
"""User's two fixes for in-distribution LLM masking:
  (P2) FIRST TOKEN = attention SINK (StreamingLLM): masking it collapses attention -> OOD. FIX it
       (keep position 0) along with the readout. Likely the main driver of heavy-mask entropy 7.97.
  (P1) MOVING-AVERAGE baseline = the LLM analog of the vision BLUR baseline (low-pass over the
       sequence embeddings: removes the specific token, keeps local flow -> in-distribution, non-
       repetitive). vs the single repeated mean (repetition -> OOD).

(A) SINK TEST: uniform-mean masking, entropy at keep-fracs, FIRST-TOKEN-FIXED vs NOT (does fixing
    the sink reduce OOD?). (B) 4 baselines (uniform / freq-corpus / random-real / moving-avg), all
    sink-fixed: gradedness + Banzhaf insertion AUC (uniform-mean metric) vs AttnLRP."""
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
    ap.add_argument("--n-imdb", type=int, default=3)
    ap.add_argument("--M", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    W = model.get_input_embeddings().weight.detach(); Vsz = W.shape[0]
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]
    uniform_mean = W.mean(0)
    raw_all, _ = load_imdb(400, 2000)
    cids = []
    for _, text in raw_all[:200]:
        cids.extend(tok(text, add_special_tokens=False).input_ids[:400])
    freq_mean = W[torch.tensor(cids[:80000], device=dev)].mean(0)

    def last_logits(e):
        try:
            return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
        except TypeError:
            return model(inputs_embeds=e).logits[:, -1]

    cases = [("everest", load_everest_prompt(), None)]
    per_bin = max(1, args.n_imdb // len(BINS)); binc = {b[0]: 0 for b in BINS}
    for j, (label, text) in enumerate(raw_all):
        ids = tok(PROMPT_TMPL.format(text=text), return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); b = next(n for n, lo, hi in BINS if lo <= T < hi)
        if binc[b] >= per_bin + 1:
            continue
        with torch.no_grad():
            pred = int(last_logits(model.get_input_embeddings()(ids))[0].argmax())
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
        w = 6
        ma = torch.stack([emb[0, max(0, i - w):min(T, i + w + 1)].mean(0) for i in range(T)])[None]
        rb = W[torch.randint(0, Vsz, (T,), generator=gen, device=dev)].view(1, T, -1)
        baselines = {"uniform": uniform_mean.view(1, 1, -1), "freq": freq_mean.view(1, 1, -1),
                     "random": rb, "moving_avg": ma}

        def fix(Z, sink=True):           # always keep readout; keep first token (sink) if sink
            Z[..., readout] = 1.0
            if sink:
                Z[..., 0] = 1.0
            return Z

        with torch.no_grad():
            lf = last_logits(emb)[0]
            ans = int(lf.argmax()) if ans_id is None else int(ans_id)
            full_p = float(F.softmax(lf, -1)[ans])
            m0 = fix(torch.zeros(1, T, device=dev))
            base_uni = float(F.softmax(last_logits(emb * m0[:, :, None] + uniform_mean.view(1, 1, -1) * (1 - m0[:, :, None]))[0], -1)[ans])
        den = max(full_p - base_uni, 1e-6)

        @torch.no_grad()
        def sig(Z, B):
            e = emb * Z[:, :, None] + B * (1 - Z[:, :, None])
            sm = F.softmax(last_logits(e), -1)
            return sm[:, ans], float((-(sm * torch.log(sm + 1e-12)).sum(-1)).mean())

        # (A) sink test: uniform-mean, fixed-sink vs not
        sink_test = {}
        for sink in (True, False):
            for a in [0.05, 0.1, 0.3]:
                Z = fix((torch.rand(6, T, generator=gen, device=dev) < a).float(), sink)
                _, e = sig(Z, baselines["uniform"]); sink_test[(sink, a)] = e

        # (B) gradedness 4 baselines (sink fixed)
        grad = {}
        for nm, B in baselines.items():
            for a in [0.05, 0.1, 0.3]:
                Z = fix((torch.rand(6, T, generator=gen, device=dev) < a).float())
                pr, e = sig(Z, B); grad[(nm, a)] = (e, float(((pr - base_uni) / den).mean()))

        @torch.no_grad()
        def rec_hard(order, f):
            k = int(round(f * len(order))); m = fix(torch.zeros(1, T, device=dev))[0]
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float((sig(m[None], baselines["uniform"])[0][0] - base_uni) / den)

        def ins_auc(scores):
            order = [i for i in np.argsort(-scores) if i not in (0, readout)]
            return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

        def banzhaf(B):
            Zs = []; Rs = []; j = 0
            while j < args.M:
                nb = max(1, min(args.bs, args.M - j) // 2); zb = []
                for _ in range(nb):
                    pf = float(torch.rand(1, generator=gen, device=dev).item())
                    z = fix((torch.rand(T, generator=gen, device=dev) < pf).float())
                    za = fix(1.0 - z); zb.append(z); zb.append(za)
                Z = torch.stack(zb); Rs.append((sig(Z, B)[0] - base_uni) / den); Zs.append(Z); j += len(zb)
            Z = torch.cat(Zs); R = torch.cat(Rs); Mt = Z.shape[0]; n1 = Z.sum(0)
            s1 = (Z * R[:, None]).sum(0) / n1.clamp(min=1); s0 = ((1 - Z) * R[:, None]).sum(0) / (Mt - n1).clamp(min=1)
            return (s1 - s0).cpu().numpy()

        a = attn_cache.get(key)
        ia = ins_auc(np.asarray(a, np.float32)) if a is not None and len(a) == T else float("nan")
        return dict(T=T, sink=sink_test, grad=grad, ins_attnlrp=ia,
                    ins={nm: ins_auc(banzhaf(B)) for nm, B in baselines.items()})

    rows = []
    for key, prompt, ans_id in cases:
        r = run(key, prompt, ans_id); r["key"] = key; rows.append(r)
        print(f"  {key:8s} T={r['T']:4d} | BZ uni={r['ins']['uniform']:.3f} freq={r['ins']['freq']:.3f} "
              f"rand={r['ins']['random']:.3f} mavg={r['ins']['moving_avg']:.3f} | attnlrp={r['ins_attnlrp']:.3f}", flush=True)

    print("\n=== (A) SINK TEST: uniform-mean entropy at keep-frac, FIRST-TOKEN fixed vs NOT ===")
    for a in [0.05, 0.1, 0.3]:
        ef = np.mean([r["sink"][(True, a)] for r in rows]); en = np.mean([r["sink"][(False, a)] for r in rows])
        print(f"  keep={a:.2f} | sink-FIXED ent={ef:.2f} | sink-FREE ent={en:.2f} | drop={en-ef:+.2f}")
    print("\n=== (B) gradedness 4 baselines (sink fixed): entropy at keep-frac (LOW=in-distribution) ===")
    print(f"{'keep':>5s} | " + " | ".join(f"{nm:>10s}" for nm in ("uniform", "freq", "random", "moving_avg")))
    for a in [0.05, 0.1, 0.3]:
        print(f"{a:5.2f} | " + " | ".join(f"{np.mean([r['grad'][(nm,a)][0] for r in rows]):10.2f}" for nm in ("uniform", "freq", "random", "moving_avg")))
    print("\n=== Banzhaf insertion AUC (uniform-mean metric) vs AttnLRP ===")
    for nm in ("uniform", "freq", "random", "moving_avg"):
        print(f"  BZ_{nm:10s} = {np.mean([r['ins'][nm] for r in rows]):.3f}")
    print(f"  AttnLRP       = {np.nanmean([r['ins_attnlrp'] for r in rows]):.3f}")
    print("(KEY: does sink-FIX drop entropy? which baseline is most graded? does any Banzhaf beat AttnLRP?)")
    json.dump(rows, open(os.path.join(REPO, "outputs", "meanbaseline.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
