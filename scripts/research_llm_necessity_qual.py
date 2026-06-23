#!/usr/bin/env python3
"""QUALITATIVE necessity on COMPLEX reasoning, done RIGHT: Llama-3.1-8B-Instruct + the INSTRUCT CHAT
TEMPLATE (so the model actually solves the problems) and necessity restricted to the CONTENT tokens
(the chat scaffolding is held REAL -> the masked sequence stays in-distribution, partially side-stepping
the sequence-OOD issue). Q: does conditional-necessity highlight the reasoning bridge / chain / operands
and IGNORE distractors better than AttnLRP/PA-LRP? Prints top necessary CONTENT tokens per method."""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PALRP = "/tmp/PE-AWARE-LRP/NLP"
sys.path.insert(0, PALRP)
HUB = "/data/.cache/huggingface/hub"
MODELS = {"1b": "models--meta-llama--Llama-3.2-1B-Instruct", "8b": "models--meta-llama--Llama-3.1-8B-Instruct"}

QUESTIONS = [
    ("multihop", "Maria's father is Carlos. Carlos was born in Lima. In which city was Maria's father born? Reply with only the city name."),
    ("multistep", "There are 20 students in a class. Half of them are boys. Of those boys, 3 wear glasses. How many boys do NOT wear glasses? Reply with only the number."),
    ("logic_rule", "Rule: if a number is even then it is blue. The number 8 is even. What color is the number 8? Reply with only the color."),
    ("compare_chain", "Anna runs faster than Bob. Bob runs faster than Carl. Among Anna, Bob and Carl, who is the slowest runner? Reply with only the name."),
    ("distractor", "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is Madrid. What is the capital of Italy? Reply with only the city name."),
    ("twohop_num", "Tom is 30 years old. His sister is 5 years younger than Tom. How old is Tom's sister? Reply with only the number."),
]


def clean(t):
    return t.replace("Ġ", "").replace("Ċ", "\\n").replace("Ä", "").strip() or t


def content_nz(tok, idl):
    """Return the USER-turn content token indices (skip the auto system date-prompt + chat scaffolding)."""
    sh = tok.convert_tokens_to_ids("<|start_header_id|>"); eh = tok.convert_tokens_to_ids("<|end_header_id|>"); eot = tok.convert_tokens_to_ids("<|eot_id|>")
    toks = tok.convert_ids_to_tokens(idl); i = 0
    while i < len(idl):
        if idl[i] == sh:
            j = idl.index(eh, i); role = tok.decode(idl[i + 1:j]).strip()
            k = idl.index(eot, j) if eot in idl[j + 1:] else len(idl)
            if role == "user":
                return [x for x in range(j + 1, k) if clean(toks[x]) not in ("", "\\n")]
            i = k + 1
        else:
            i += 1
    return []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="8b", choices=["1b", "8b"]); ap.add_argument("--topk", type=int, default=8); ap.add_argument("--no-lrp", action="store_true")
    args = ap.parse_args(); dev = args.device
    path = sorted(glob.glob(f"{HUB}/{MODELS[args.model]}/snapshots/*/"))[-1]
    dt = torch.float32 if args.model == "1b" else torch.bfloat16
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, local_files_only=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    eos_vec = model.get_input_embeddings().weight[tok.eos_token_id].view(1, 1, -1).detach()

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e.to(dt), logits_to_keep=1).logits[:, -1].float()
            except TypeError:
                return model(inputs_embeds=e.to(dt)).logits[:, -1].float()

    cases = []
    for name, q in QUESTIONS:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        idl = ids[0].tolist(); nz = content_nz(tok, idl); T = len(idl)
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            ans = int(F.softmax(llq(emb)[0], -1).argmax())
        cases.append(dict(name=name, q=q, ids=ids, idl=idl, nz=nz, T=T, emb=emb, ans=ans,
                          ans_str=clean(tok.convert_ids_to_tokens([ans])[0])))

    # ---- phase 1: necessity (greedy + single) on CONTENT tokens, eos fill ----
    for c in cases:
        emb, T, nz, ans = c["emb"], c["T"], c["nz"], c["ans"]

        @torch.no_grad()
        def prob(M):
            e = (emb * M[:, :, None] + eos_vec * (1 - M[:, :, None]))
            return F.softmax(llq(e), -1)[:, ans].cpu().numpy()

        @torch.no_grad()
        def greedy():
            km = np.ones(T, np.float32); order = []; rem = list(nz)
            while rem:
                masks = []
                for i in rem:
                    m = km.copy(); m[i] = 0.0; masks.append(m)
                pp = prob(torch.tensor(np.stack(masks), device=dev)); jb = int(np.argmin(pp))
                km[rem[jb]] = 0.0; order.append(rem[jb]); rem.pop(jb)
            return order

        @torch.no_grad()
        def single():
            full = float(prob(torch.ones(1, T, device=dev))[0]); sc = {}
            for i in nz:
                m = torch.ones(T, device=dev); m[i] = 0.0; sc[i] = full - float(prob(m[None])[0])
            return sorted(nz, key=lambda i: -sc[i])
        c["greedy"] = greedy(); c["single"] = single()
        print(f"  necessity done: {c['name']} (T={T}, |content|={len(nz)}, ans={c['ans_str']!r})", flush=True)
    del model; torch.cuda.empty_cache()

    # ---- phase 2: AttnLRP / PA-LRP (official llama_PE), sequential to fit memory ----
    if not args.no_lrp:
        from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
        mlrp = LlamaForCausalLM.from_pretrained(path, torch_dtype=dt, attn_implementation="eager", local_files_only=True).to(dev).eval()
        mlrp.gradient_checkpointing_enable(); attnlrp.register(mlrp); L = mlrp.config.num_hidden_layers
        for c in cases:
            ids = c["ids"]; ans = c["ans"]
            try:
                e = mlrp.get_input_embeddings()(ids)
                pid = torch.arange(0.0, ids.shape[1], device=dev, requires_grad=True, dtype=torch.float32).reshape(1, -1)
                pe = [mlrp.get_input_pos_embeddings()(e, pid) for _ in range(L)]
                pe = [(x[0].requires_grad_(), x[1].requires_grad_()) for x in pe]
                lg = mlrp(inputs_embeds=e.requires_grad_(), position_embeddings=pe, use_cache=False)["logits"]
                t = lg[0, -1, ans].float(); t.backward(t)
                at = e.grad.float().sum(-1).cpu()[0]; at = at / at.abs().max()
                acc = torch.zeros_like(at)
                for p in pe:
                    for i in range(2):
                        acc = acc + torch.matmul(p[i].grad.abs(), p[i].transpose(-1, -2).abs()).detach().float().sum(-1).cpu()[0].abs()
                acc = acc / acc.abs().max(); pa = at + acc; pa = pa / pa.abs().max()
                c["attn"] = at.numpy(); c["pa"] = pa.numpy()
            except Exception as ex:  # noqa: BLE001
                print(f"  LRP {c['name']} FAIL {str(ex)[:60]}", flush=True); c["attn"] = c["pa"] = None
            mlrp.zero_grad(set_to_none=True); torch.cuda.empty_cache()

    # ---- print qualitative ----
    print(f"\n========== QUALITATIVE NECESSITY ({args.model}-Instruct, chat template, content-only) ==========")
    for c in cases:
        toks = tok.convert_ids_to_tokens(c["idl"]); nzset = set(c["nz"])
        top = lambda order: "  ".join(clean(toks[i]) for i in order[:args.topk])
        ford = lambda s: [i for i in np.argsort(-np.asarray(s)) if i in nzset]
        print(f"\n### {c['name']}  | answer = {c['ans_str']!r}")
        print(f"  Q: {c['q']}")
        print(f"  NECESSITY(greedy) : {top(c['greedy'])}")
        print(f"  necessity(single) : {top(c['single'])}")
        if not args.no_lrp and c.get("attn") is not None:
            print(f"  AttnLRP           : {top(ford(c['attn']))}")
            print(f"  PA-LRP            : {top(ford(c['pa']))}")


if __name__ == "__main__":
    main()
