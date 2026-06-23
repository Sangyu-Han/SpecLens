#!/usr/bin/env python3
"""Test the NL insight: is context ~= necessity for an LLM? Measure overlap between the
INPUT necessity set (greedy-conditional deletion) and the INPUT sufficiency set (greedy
insertion = FRI-style) on factual prompts. High overlap => for NL, sufficient-ERF ~=
necessary-ERF (so tracing a summary token's ERF recovers necessity). Vision overlap ~0.03.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    emb_layer = model.model.embed_tokens

    def probs(emb, tgt):
        with torch.no_grad():
            return torch.softmax(model(inputs_embeds=emb).logits[:, -1], -1)[:, tgt].float().cpu().numpy()

    overlaps = []; necs = []; suffs = []
    for p, ans in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(args.device)
        T = ids.shape[1]
        with torch.no_grad():
            emb = emb_layer(ids)  # [1,T,d]
            full_lg = model(inputs_embeds=emb).logits[0, -1]
        tgt = int(full_lg.argmax()); mean = emb.mean(1, keepdim=True)
        full = float(torch.softmax(full_lg, -1)[tgt])
        base = float(probs(mean.expand(1, T, -1).clone(), tgt)[0]) if True else 0.0

        def keep_prob(masks):  # masks [M,T] keep=1
            m = torch.as_tensor(np.stack(masks), device=args.device, dtype=emb.dtype)[..., None]
            e = emb * m + mean * (1 - m)
            return probs(e, tgt)
        den = abs(full - base) + 1e-6
        # necessity: greedy delete until recovery<=0.1
        removed = np.zeros(T); nec = []
        while True:
            cand = [i for i in range(T) if removed[i] == 0]
            if not cand:
                break
            masks = []
            for i in cand:
                mm = 1 - removed.copy(); mm[i] = 0; masks.append(mm)
            v = keep_prob(masks); j = int(np.argmin(v)); p_ = cand[j]
            removed[p_] = 1; nec.append(p_)
            if (float(v[j]) - base) / den <= 0.1:
                break
        # sufficiency: greedy insert until recovery>=0.9
        kept = np.zeros(T); suff = []
        while True:
            cand = [i for i in range(T) if kept[i] == 0]
            if not cand:
                break
            masks = []
            for i in cand:
                mm = kept.copy(); mm[i] = 1; masks.append(mm)
            v = keep_prob(masks); j = int(np.argmax(v)); p_ = cand[j]
            kept[p_] = 1; suff.append(p_)
            if (float(v[j]) - base) / den >= 0.9:
                break
        ov = len(set(nec) & set(suff)) / max(len(set(nec) | set(suff)), 1)
        overlaps.append(ov); necs.append(len(nec)); suffs.append(len(suff))
        toks = [tok.decode(t) for t in ids[0].tolist()]
        print(f"  {ans:11s} nec={[toks[i] for i in sorted(nec)]} suff={[toks[i] for i in sorted(suff)]} IoU={ov:.2f}", flush=True)
    print(f"\n=== LLM: necessity vs sufficiency overlap (n={len(PROMPTS)}) ===")
    print(f"  mean IoU = {np.mean(overlaps):.3f}  (vision clip ~0.03)")
    print(f"  nec size {np.mean(necs):.1f} | suff size {np.mean(suffs):.1f} | / {ids.shape[1]} tok")


if __name__ == "__main__":
    main()
