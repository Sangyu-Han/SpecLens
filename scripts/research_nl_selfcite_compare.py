#!/usr/bin/env python3
"""SelfCite (Chuang et al., ICML 2025) qualitative comparison on BENCH-100 cases.
Their reward, THEIR conventions (verbatim from third_party/SelfCite/longcite_modeling_llama.py +
eval-best-of-n-reranking.py 'log_prob_drop_and_hold'):
  drop-reward(E)  = - mean logP(r | chat-prompt with E TEXT-REMOVED)      [orig logP = constant, ignored]
  hold-reward(E)  = + mean logP(r | chat-prompt with ONLY E kept]
  score(E) = drop + hold; applied per single-sentence E={s} -> sentence attribution map.
Prompt = their citation-instruction document wrapper + llama chat template (their --llama_chat_template path).
Cases rebuilt with bench100 seed: mathctx01/02, needle01, para01, mhop01, mhop05(link-miss case).
Render: cmp26_{case}_selfcite.pdf (sentence-uniform heat, signed). 3B cuda:0."""
from __future__ import annotations

import glob
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/sangyu/Desktop/Master/SpecLens/third_party/PE-AWARE-LRP/NLP")
from nltk.tokenize.punkt import PunktSentenceTokenizer

P3B = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/"))[-1]
ARROWS = sorted(glob.glob("/data/.cache/huggingface/datasets/EleutherAI___fineweb-edu-dedup-10b/**/fineweb*.arrow", recursive=True))
OUT = "/home/sangyu/Desktop/Master/SpecLens/outputs/attr_bench100"
dev = "cuda:0"

NAMES = ["Tom", "Alex", "Bob", "Carl", "Sam", "Max", "Ben", "Leo", "Emma", "Anna", "Sara", "Kate", "Ryan", "Jack", "Mark", "Paul"]
ITEMS = ["apples", "books", "coins", "pencils", "stickers", "marbles", "cards", "shells"]
JOBS = ["nurse", "pilot", "teacher", "doctor", "lawyer", "farmer", "baker", "chef", "singer", "dancer", "painter", "writer"]


def split_sents(t):
    t = re.sub(r"\s+", " ", t.strip()); return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 1]


# ---- verbatim from SelfCite longcite_modeling_llama.py:text_split_by_punctuation ----
def text_split_by_punctuation(original_text, return_dict=False):
    text = original_text
    custom_sent_tokenizer = PunktSentenceTokenizer()
    punctuations = r"([。；！？])"
    separated = custom_sent_tokenizer.tokenize(text)
    separated = sum([re.split(punctuations, s) for s in separated], [])
    for i in range(1, len(separated)):
        if re.match(punctuations, separated[i]):
            separated[i - 1] += separated[i]
            separated[i] = ""
    separated = [s for s in separated if s != ""]
    if len(separated) == 1:
        separated = original_text.split("\n\n")
    separated = [s.strip() for s in separated if s.strip() != ""]
    if not return_dict:
        return separated
    pos = 0; res = []
    for i, sent in enumerate(separated):
        st = original_text.find(sent, pos)
        assert st != -1, sent
        ed = st + len(sent)
        res.append({"c_idx": i, "content": sent, "start_idx": st, "end_idx": ed})
        pos = ed
    return res


# ---- verbatim structure from SelfCite query_log_prob_drop_ablating.get_prompt (drop) / _hold_pruning (keep) ----
SC_PROMPT = ('''Please answer the user's question based on the following document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's question.\n\n[Document Start]\n%s\n[Document End]\n\n%s''')


def sc_prompt(context, question, drop_ids=None, keep_ids=None):
    sents = text_split_by_punctuation(context, return_dict=True)
    splited = ""
    for i, s in enumerate(sents):
        if drop_ids is not None and i in drop_ids:
            continue
        if keep_ids is not None and i not in keep_ids:
            continue
        st, ed = s["start_idx"], s["end_idx"]
        ed = sents[i + 1]["start_idx"] if i < len(sents) - 1 else len(context)
        splited += f"<C{i}>" + context[st:ed]
    return SC_PROMPT % (splited, question), len(sents)


def main():
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from datasets import Dataset
    from lxt.utils import clean_tokens, pdf_heatmap
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P3B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(P3B, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to(dev).eval()

    def sc_logprob(prompt_text, ans_ids):
        msgs = [{"role": "user", "content": prompt_text}]
        pt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pi = tok(pt, return_tensors="pt", add_special_tokens=False).input_ids
        ii = torch.cat([pi, torch.tensor(ans_ids).unsqueeze(0)], -1).to(dev)
        K = len(ans_ids)
        with torch.inference_mode():
            lg = model(input_ids=ii).logits[0, -K - 1:-1].float()
        lp = torch.log_softmax(lg, -1)
        return float(lp[torch.arange(K), torch.tensor(ans_ids, device=dev)].mean())

    # ---- rebuild bench100 cases (same seed/logic, selected only) ----
    d = Dataset.from_file(ARROWS[0])
    ban = set(w.lower() for w in NAMES + JOBS + ITEMS) | {"colleague", "bought", "gave"}
    fillB, docs = [], []
    num_re = re.compile(r"\b(1[0-9]{3}|[2-9][0-9]{2,3})\b")
    for i in range(len(d)):
        ss = split_sents(d[i]["text"])
        if 8 <= len(ss) <= 20 and all(len(s.split()) <= 40 for s in ss) and all(sum(ch.isascii() for ch in s) / max(len(s), 1) > 0.97 for s in ss):
            docs.append(ss)
        for s in ss:
            w = s.split()
            if 6 <= len(w) <= 14 and s[0].isupper() and s.endswith(".") and sum(c.isalpha() or c.isspace() or c in ".,'" for c in s) / len(s) > 0.95:
                if not any(k in ban for k in re.findall(r"[a-z]+", s.lower())) and not re.search(r"\d", s):
                    fillB.append(s)
        if len(docs) > 300 and len(fillB) > 800:
            break
    rng = np.random.default_rng(20260708)
    cases = []
    for k in range(60):
        nm, nm2 = rng.choice(NAMES, 2, replace=False); it, it2 = rng.choice(ITEMS, 2, replace=False)
        v1 = int(rng.integers(6, 20)); v2 = int(rng.integers(2, min(v1, 9)))
        if rng.random() < 0.5:
            p1 = f"{nm} had {v1} {it} in the morning."; p2 = f"Later {nm} gave away {v2} {it}."; ansn = v1 - v2
        else:
            p1 = f"{nm} had {v1} {it} in the morning."; p2 = f"Then {nm} bought {v2} more {it}."; ansn = v1 + v2
        dis = f"{nm2} had {int(rng.integers(3, 15))} {it2}."
        nf = int(rng.integers(6, 11)); fl = [fillB[int(i)] for i in rng.choice(len(fillB), nf, replace=False)]
        sents = list(fl); pos = sorted(rng.choice(nf + 1, 3, replace=True))
        for j, s in enumerate([p1, p2, dis]):
            sents.insert(min(pos[j] + j, len(sents)), s)
        q = f"Question: how many {it} does {nm} have now? Answer: {nm} now has"
        cases.append(("mathctx", sents, q, " " + str(ansn), dict(gt=[sents.index(p1), sents.index(p2)], decoy=[sents.index(dis)])))
    for k in range(70):
        picks = rng.choice(len(docs), int(rng.integers(2, 4)), replace=False)
        sents = []
        for pi in picks:
            sents += docs[int(pi)][:16]
        sents = sents[:42]
        tgt = None
        for si, s in enumerate(sents):
            m = num_re.search(s)
            if m and len(s[:m.start()].split()) >= 6:
                others = [sj for sj, s2 in enumerate(sents) if sj != si and num_re.search(s2)]
                tgt = (si, m, others); break
        if tgt is None:
            continue
        si, m, others = tgt
        cases.append(("needle", sents, "According to the text, " + sents[si][:m.start()].rstrip(), " " + m.group(1), dict(gt=[si], decoy=others[:4])))
    for k in range(50):
        a, b = rng.choice(NAMES, 2, replace=False); jb = JOBS[int(rng.integers(len(JOBS)))]
        c1 = f"{a}'s colleague is {b}."; c2 = f"{b} works as a {jb}."
        nf = int(rng.integers(14, 24)); fl = [fillB[int(i)] for i in rng.choice(len(fillB), nf, replace=False)]
        sents = list(fl)
        for s in (c1, c2):
            sents.insert(int(rng.integers(len(sents) + 1)), s)
        cases.append(("mhop", sents, f"Question: the job of {a}'s colleague? Answer: The colleague of {a} works as a", " " + jb, dict(gt=sorted([sents.index(c1), sents.index(c2)]), link=sents.index(c1))))

    # select the comparison set by replaying the model-correct filter per family
    want = {"mathctx": [1, 2], "needle": [1], "mhop": [1, 5]}
    picked = []
    counts = {}
    for fam, sents, query, ans_str, meta in cases:
        idl = [tok.bos_token_id]
        for s in sents:
            idl += tok(" " + s, add_special_tokens=False).input_ids
        idl += tok(" " + query, add_special_tokens=False).input_ids
        a_ids = tok(ans_str, add_special_tokens=False).input_ids
        ids = torch.tensor([idl], device=dev)
        with torch.no_grad():
            lg1 = model(input_ids=ids).logits[0, -1]
        if int(lg1.argmax()) != a_ids[0]:
            continue
        counts[fam] = counts.get(fam, 0) + 1
        if fam in want and counts[fam] in want[fam]:
            picked.append((f"{fam}{counts[fam]:02d}", sents, query, ans_str, meta))
    print(f"picked: {[p[0] for p in picked]}", flush=True)

    for cname, sents, query, ans_str, meta in picked:
        context = " ".join(sents)
        question = query
        a_ids = tok(ans_str, add_special_tokens=False).input_ids
        sc_sents = text_split_by_punctuation(context, return_dict=True)
        # align their sentences to ours (should be 1:1 for clean periods)
        p_full, nsc = sc_prompt(context, question)
        lp_full = sc_logprob(p_full, a_ids)
        drop_r, hold_r = np.zeros(nsc), np.zeros(nsc)
        for i in range(nsc):
            pd_, _ = sc_prompt(context, question, drop_ids={i})
            ph_, _ = sc_prompt(context, question, keep_ids={i})
            drop_r[i] = lp_full - sc_logprob(pd_, a_ids)   # Prob-Drop (necessity)
            hold_r[i] = sc_logprob(ph_, a_ids) - lp_full   # Prob-Hold (sufficiency, full-C referenced)
        score = drop_r + hold_r                             # their combined reranker quantity
        # map their sentence spans onto our sentence list by containment
        ours_of = []
        for s in sc_sents:
            hit = next((j for j, o in enumerate(sents) if s["content"] in o or o in s["content"]), -1)
            ours_of.append(hit)
        top3 = np.argsort(-score)[:3]
        gtset = set(meta["gt"])
        print(f"[{cname}] gt={sorted(gtset)} SelfCite top3(sent#our): "
              + " ".join(f"{ours_of[int(i)]}({score[int(i)]:+.2f})" for i in top3)
              + f"  drop-top={ours_of[int(np.argmax(drop_r))]}  hold-top={ours_of[int(np.argmax(hold_r))]}"
              + f"  gt-in-top2={int(gtset <= set(ours_of[int(i)] for i in np.argsort(-score)[:max(2, len(gtset))]))}", flush=True)
        # render: token-uniform per our sentence, signed norm
        idl = [tok.bos_token_id]; ranges = []
        for s in sents:
            t0 = len(idl); idl += tok(" " + s, add_special_tokens=False).input_ids; ranges.append((t0, len(idl)))
        idl += tok(" " + query, add_special_tokens=False).input_ids
        arr = np.zeros(len(idl), np.float32)
        pers = np.zeros(len(sents))
        for i, oj in enumerate(ours_of):
            if oj >= 0:
                pers[oj] += score[i]
        mx = np.abs(pers).max() + 1e-9
        for j, (a2, b2) in enumerate(ranges):
            arr[a2:b2] = pers[j] / mx
        words = clean_tokens(list(tok.convert_ids_to_tokens(idl)))
        pdf_heatmap(words, np.clip(arr, -1, 1), path=f"{OUT}/cmp26_{cname}_selfcite.pdf",
                    backend="xelatex", delete_aux_files=True)
        print(f"  -> cmp26_{cname}_selfcite.pdf", flush=True)


if __name__ == "__main__":
    main()
