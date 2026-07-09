#!/usr/bin/env python3
"""v8 GAME-CORRECTION prototype (user: fix the GAME, not the rules -> rules should die).
Game: v(S) = sum_k [ log P(ans_k | ctx-mask S, query, ans_<k) ]  -  answer-SEQUENCE, teacher-forced, log scale.
gain(S) = v(S) - v(empty)  ->  three claims tested on realdata-5:
 C1 (real4 target bug): per-answer-token gap_k = logP_k(full)-logP_k(empty) shows '200' is free, '1' carries info;
     the sequence game absorbs this automatically -> no target-token-selection rule needed.
 C2 gain-referencing ABSORBED: gain(empty)=0 by construction -> certificate gain(E) >= tau*gain(N), no R0 rule.
 C3 copy axiom SUBSUMED: literal-copy sentences should maximize solo gain g_s -> enter E0 via lens alone,
     no string matching.
Minimal ruleset here: E0 = {g_s >= 0.3 max g} -> backward-min (drop s if gain stays >= tau*gainN and |dv| <= eps*|gainN|)
-> extend by g order. Constants: tau=0.9, eps=0.02(rel), floor 0.3. NO copy axiom / NO R0 rule / NO direction guard.
3B eager cuda:0."""
from __future__ import annotations

import glob
import re

import numpy as np
import torch
import torch.nn.functional as F

P3B = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/"))[-1]
ARROWS = sorted(glob.glob("/data/.cache/huggingface/datasets/EleutherAI___fineweb-edu-dedup-10b/**/fineweb*.arrow", recursive=True))
dev = "cuda:0"


def split_sents(t):
    t = re.sub(r"\s+", " ", t.strip()); return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 1]


def main():
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P3B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(P3B, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to(dev).eval()
    embl = model.get_input_embeddings(); eos = embl(torch.tensor([[tok.eos_token_id]], device=dev))[0, 0].detach()

    # ---- same number-cloze builder as realdata bench ----
    d = Dataset.from_file(ARROWS[0])
    num_re = re.compile(r"\b(1[0-9]{3}|[2-9][0-9]{2,3})\b")
    cases = []
    for i in range(len(d)):
        if len(cases) >= 40:
            break
        sents_all = split_sents(d[i]["text"])
        if not (8 <= len(sents_all) <= 16):
            continue
        if any(len(s.split()) > 40 for s in sents_all):
            continue
        if any(sum(ch.isascii() for ch in s) / max(len(s), 1) < 0.98 for s in sents_all):
            continue
        tgt = None
        for si, s in enumerate(sents_all):
            m = num_re.search(s)
            if m and len(s[:m.start()].split()) >= 6:
                others = [sj for sj, s2 in enumerate(sents_all) if sj != si and num_re.search(s2)]
                tgt = (si, m, others)
                break
        if tgt is None:
            continue
        si, m, others = tgt
        prefix = sents_all[si][:m.start()].rstrip()
        numstr = m.group(1)
        query = "According to the text, " + prefix
        ids_a = tok(" " + query, add_special_tokens=False).input_ids
        ids_b = tok(" " + query + " " + numstr, add_special_tokens=False).input_ids
        if len(ids_b) <= len(ids_a):
            continue
        cases.append((sents_all, query, ids_b[len(ids_a):], numstr, si))
    print(f"candidates: {len(cases)}", flush=True)

    NF = [0]

    def fwd_v(ids, ctx, masks, ans_pos, batch=14):
        NF[0] += len(masks)
        # v(S) = sum_k log P(ans_k | ...) at teacher-forced positions; returns (B,) and per-token (B,K)
        T = ids.shape[1]; emb = embl(ids).detach(); outs = []; pers = []
        atoks = [ids[0, p].item() for p in ans_pos]
        for b in range(0, len(masks), batch):
            es = []
            for mk in masks[b:b + batch]:
                gt = torch.ones(T, device=dev, dtype=emb.dtype)
                off = [ctx[i] for i in range(len(ctx)) if mk[i] < 0.5]
                if off:
                    gt[torch.tensor(off, device=dev)] = 0.0
                es.append(emb[0] * gt[:, None] + eos[None] * (1 - gt[:, None]))
            with torch.no_grad():
                lg = model(inputs_embeds=torch.stack(es)).logits.float()
            lp = torch.log_softmax(lg, -1)
            pk = torch.stack([lp[:, p - 1, at] for p, at in zip(ans_pos, atoks)], 1).cpu()  # (B,K)
            pers.append(pk); outs.append(pk.sum(1))
        return torch.cat(outs).numpy(), torch.cat(pers).numpy()

    kept = 0; agg = []
    for sents, query, ans_toks, numstr, gt_si in cases:
        idl = [tok.bos_token_id]; ranges = []
        for s in sents:
            t0 = len(idl); idl += tok(" " + s, add_special_tokens=False).input_ids; ranges.append((t0, len(idl)))
        q0 = len(idl); idl += tok(" " + query, add_special_tokens=False).input_ids
        a0 = len(idl); idl += list(ans_toks)
        ids = torch.tensor([idl], device=dev)
        ans_pos = list(range(a0, len(idl)))
        # answer correctness check on FIRST token (same as bench)
        with torch.no_grad():
            lg1 = model(input_ids=ids[:, :a0]).logits[0, -1]
        if int(lg1.argmax()) != ans_toks[0]:
            continue
        kept += 1
        if kept > 20:
            break
        ctx = list(range(1, q0)); L = len(ctx); ns = len(ranges)
        cpos = {t: i for i, t in enumerate(ctx)}
        sm = np.zeros((ns, L))
        for si2, (a2, b2) in enumerate(ranges):
            for t in range(a2, b2):
                sm[si2, cpos[t]] = 1.0
        NF[0] = 0
        vN, pN = fwd_v(ids, ctx, [np.ones(L)], ans_pos)
        v0, p0 = fwd_v(ids, ctx, [np.zeros(L)], ans_pos)
        gainN = float(vN[0] - v0[0])
        toks_str = [tok.decode([t]) for t in ans_toks]
        gaps = (pN[0] - p0[0])
        print(f"\nreal{kept}: ans='{numstr}' tokens={toks_str} gt=s{gt_si}", flush=True)
        print(f"  C1 per-token gap(full-empty): " + " ".join(f"'{w}':{g:+.2f}" for w, g in zip(toks_str, gaps))
              + f"  | gainN={gainN:.2f}", flush=True)
        gs, _ = fwd_v(ids, ctx, [sm[si2] for si2 in range(ns)], ans_pos)
        g = gs - v0[0]
        order = np.argsort(-g)
        copy_true = {si2 for si2, s_ in enumerate(sents) if re.search(r"(?<!\w)" + re.escape(numstr) + r"(?!\w)", s_)}
        print(f"  C3 solo-gain top3: " + " ".join(f"s{int(o)}:{g[int(o)]:.2f}" for o in order[:3])
              + f"  copies(string)={sorted(copy_true)} -> argmax{'==copy YES' if int(order[0]) in copy_true else '!=copy NO'}", flush=True)
        # ---- minimal-rule E assembly on the corrected game ----
        E = sorted([si2 for si2 in range(ns) if g[si2] >= 0.3 * g.max()])
        vE, _ = fwd_v(ids, ctx, [np.clip(sm[E].sum(0), 0, 1)], ans_pos); gE = float(vE[0] - v0[0])
        for si0 in list(np.argsort(g)):
            si0 = int(si0)
            if si0 not in E or len(E) <= 1:
                continue
            E2 = [x for x in E if x != si0]
            v2, _ = fwd_v(ids, ctx, [np.clip(sm[E2].sum(0), 0, 1)], ans_pos); g2 = float(v2[0] - v0[0])
            if g2 >= 0.9 * gainN and (gE - g2) <= 0.02 * abs(gainN):
                E = E2; gE = g2
        for si0 in np.argsort(-g):
            if gE >= 0.9 * gainN:
                break
            if int(si0) not in E:
                E = sorted(E + [int(si0)])
                v2, _ = fwd_v(ids, ctx, [np.clip(sm[E].sum(0), 0, 1)], ans_pos); gE = float(v2[0] - v0[0])
        print(f"  C2 E(no-copy-axiom, no-R0-rule) = {E}  gain(E)/gain(N)={gE / max(gainN, 1e-9):.2f}  "
              f"[gt {'IN' if gt_si in E else 'MISS'}; |E|={len(E)}]", flush=True)
        agg.append((gt_si in E, len(E), int(order[0]) in copy_true, gE / max(gainN, 1e-9), gt_si in [int(order[0])]))
        print(f'  COST real{kept}: {NF[0]} fwd (ns={ns})', flush=True)

    _summary(agg)


def _summary(agg):
    import numpy as _np
    a = _np.array(agg, dtype=float)
    print(f"\n=== v8 N={len(agg)}: gt-in-E {a[:,0].mean():.2f}  |E| mean {a[:,1].mean():.1f}  "
          f"argmax==copy {a[:,2].mean():.2f}  gain-ratio {a[:,3].mean():.2f}  argmax==gt {a[:,4].mean():.2f} ===", flush=True)


if __name__ == "__main__":
    main()
