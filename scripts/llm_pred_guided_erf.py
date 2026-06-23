#!/usr/bin/env python3
"""Does a CHEAP prediction-guided ERF recover the necessity CORE (vs sufficiency=context)?
Reference = necessity oracle (greedy-conditional deletion). Compare recall of the necessary
tokens in the top-|nec| of: grad (dy/demb, 1 bwd) / gradxinput / IG (~32) / FRI-sufficiency
(greedy insert). If prediction-guided (grad/IG) >> sufficiency, the fix is cheap for NL.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

PROMPTS = [
    ("The Eiffel Tower is located in the city of", "Paris"),
    ("The capital city of Japan is", "Tokyo"),
    ("The chemical symbol for gold is", "Au"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The first president of the United States was George", "Washington"),
    ("The Great Wall is located in the country of", "China"),
    ("The tallest mountain in the world is Mount", "Everest"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci"),
]
METHODS = ["occlusion", "grad", "gradxinput", "IG", "FRI_suff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    emb_layer = model.model.embed_tokens
    rec = {m: [] for m in METHODS}
    for p, ans in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(args.device); T = ids.shape[1]
        emb0 = emb_layer(ids).detach()
        with torch.no_grad():
            tgt = int(model(inputs_embeds=emb0).logits[0, -1].argmax())
        mean = emb0.mean(1, keepdim=True)

        def y_of(e):
            return model(inputs_embeds=e).logits[0, -1, tgt]

        def probs(masks):
            m = torch.as_tensor(np.stack(masks), device=args.device, dtype=emb0.dtype)[..., None]
            e = emb0 * m + mean * (1 - m)
            with torch.no_grad():
                return torch.softmax(model(inputs_embeds=e).logits[:, -1], -1)[:, tgt].float().cpu().numpy()
        full = float(probs([np.ones(T)])[0]); base = float(probs([np.zeros(T)])[0]); den = abs(full - base) + 1e-6
        occ = full - probs([np.ones(T) - np.eye(T)[i] for i in range(T)])   # 1-pass CAUSAL deletion
        # necessity ORACLE (greedy delete)
        removed = np.zeros(T); nec = []
        while True:
            cand = [i for i in range(T) if removed[i] == 0]
            masks = [(1 - removed).copy() for _ in cand]
            for r, i in enumerate(cand):
                masks[r][i] = 0
            v = probs(masks); j = int(np.argmin(v)); removed[cand[j]] = 1; nec.append(cand[j])
            if (float(v[j]) - base) / den <= 0.1 or not [i for i in range(T) if removed[i] == 0]:
                break
        kn = len(nec); necset = set(nec)
        # grad / gradxinput / IG
        e = emb0.clone().requires_grad_(True)
        g = torch.autograd.grad(y_of(e), e)[0][0]                       # [T,d]
        grad = g.abs().sum(-1).detach().cpu().numpy()
        gxi = ((emb0[0] - mean[0]) * g).sum(-1).abs().detach().cpu().numpy()
        ig_acc = torch.zeros_like(emb0[0])
        for a in np.linspace(1.0 / 32, 1.0, 32):
            ea = (mean + a * (emb0 - mean)).detach().requires_grad_(True)
            ig_acc = ig_acc + torch.autograd.grad(y_of(ea), ea)[0][0].detach()
        ig = ((emb0[0] - mean[0]) * (ig_acc / 32)).sum(-1).abs().cpu().numpy()
        # FRI sufficiency (greedy insert)
        kept = np.zeros(T); suff_order = []
        while True:
            cand = [i for i in range(T) if kept[i] == 0]
            masks = [kept.copy() for _ in cand]
            for r, i in enumerate(cand):
                masks[r][i] = 1
            v = probs(masks); j = int(np.argmax(v)); kept[cand[j]] = 1; suff_order.append(cand[j])
            if (float(v[j]) - base) / den >= 0.9 or not [i for i in range(T) if kept[i] == 0]:
                break
        scores = {"occlusion": occ, "grad": grad, "gradxinput": gxi, "IG": ig}
        toks = [tok.decode(t) for t in ids[0].tolist()]
        line = f"  {ans:11s} nec={[toks[i] for i in sorted(nec)]}"
        for m in ["occlusion", "grad", "gradxinput", "IG"]:
            topk = set(np.argsort(-scores[m])[:kn].tolist())
            rec[m].append(len(topk & necset) / max(kn, 1))
            line += f" | {m}_top={[toks[i] for i in sorted(topk)]}"
        rec["FRI_suff"].append(len(set(suff_order[:kn]) & necset) / max(kn, 1))
        print(line, flush=True)
    print(f"\n=== recall of necessity-core in top-|nec| (n={len(PROMPTS)}) — higher=recovers necessity ===")
    for m in METHODS:
        print(f"   {m:11s} {np.mean(rec[m]):.3f}")


if __name__ == "__main__":
    main()
