#!/usr/bin/env python3
"""Verify the Everest 'input_LN FRI (0.422) beats AttnLRP (0.38)' claim APPLES-TO-APPLES:
compute AttnLRP on the SAME Everest case with the SAME insertion-AUC metric (FR grid + global
vocab-mean rec_hard) used for input_LN. The 0.38 came from a different (by-length) measurement,
so the 'win' may be a metric artifact -- IMDB (same metric) showed AttnLRP 0.541 >> input_LN
0.337, with AttnLRP's lead GROWING with length, so on T=456 AttnLRP is likely >0.422 here too."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_llm_imdb_sufficiency import precompute_attnlrp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "third_party/LRP-eXplains-Transformers-main (1)/LRP-eXplains-Transformers-main/examples/quantized_qwen2.py"
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]


def load_prompt():
    ns: dict = {}
    exec(re.search(r'(prompt = """.*?""")', EX.read_text(), re.DOTALL).group(1), ns)
    return ns["prompt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad = False
    dev = args.device
    prompt = load_prompt()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(dev)
    T = int(ids.shape[1]); readout = T - 1
    emb = model.get_input_embeddings()(ids).detach()
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    with torch.no_grad():
        ans = int(model(inputs_embeds=emb).logits[0, -1].argmax())
        full = float(F.softmax(model(inputs_embeds=emb).logits[0, -1], -1)[ans])
        base = float(F.softmax(model(inputs_embeds=emb * 0 + gmean).logits[0, -1], -1)[ans])
    print(f"T={T} answer={tok.convert_ids_to_tokens([ans])[0]!r} full={full:.3f}", flush=True)

    @torch.no_grad()
    def rec_hard(order, f):
        k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[readout] = 1.0
        if k:
            m[torch.as_tensor(order[:k], device=dev)] = 1.0
        e = emb * m[None, :, None] + gmean * (1 - m[None, :, None])
        return (float(F.softmax(model(inputs_embeds=e).logits[0, -1], -1)[ans]) - base) / max(full - base, 1e-6)

    def ins_auc(scores):
        order = [i for i in np.argsort(-scores) if i != readout]
        return float(np.trapz([rec_hard(order, f) for f in FR], FR) / FR[-1])

    print("offloading main model to CPU for AttnLRP (bf16 subprocess)...", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn = precompute_attnlrp([("everest", prompt)], args.model, dev)
    model.to(dev); torch.cuda.empty_cache()
    a = attn.get("everest")
    if a is None or len(a) != T:
        print(f"AttnLRP FAILED (got {None if a is None else len(a)} vs T={T})"); return
    ia = ins_auc(np.asarray(a, np.float32))
    print("\n=== Everest insertion AUC, SAME metric ===")
    print(f"  AttnLRP    = {ia:.3f}")
    print(f"  input_LN FRI = 0.422 (from research_fri_inputLN.py, same metric)")
    print(f"  single_occ   = 0.481")
    print(f"VERDICT: AttnLRP {'>' if ia > 0.422 else '<='} input_LN -> the earlier Everest 'win' was "
          f"{'a METRIC ARTIFACT (AttnLRP wins LLM everywhere)' if ia > 0.422 else 'REAL on the copy task (nuanced)'}.", flush=True)


if __name__ == "__main__":
    main()
