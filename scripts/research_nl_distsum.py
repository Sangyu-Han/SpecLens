#!/usr/bin/env python3
"""SHOWCASE (brag set): hard cases x 4 methods, qualitative side-by-side PDFs.
Cases: everestMCQ / mathctx01 / needle01 / mhop01 (bench100 seed).
Methods: v8 (ours, PMI game full 4-channel) | contextcite (ORIGINAL context-cite Lasso solver, token toggles,
192 masks, logit target) | onlineBanzhaf (verbatim research_nl_compare.py - user qualitative favorite;
attn prefilter -> adaptive shrink -> p=.5 512-mask Banzhaf) | selfcite (official conventions per-sentence
Drop+Hold, chat template + citation prompt + text splice). -> outputs/attr_showcase/cmp27_{case}_{method}.pdf"""
from __future__ import annotations

import glob
import importlib.util as ilu
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sangyu/Desktop/Master/SpecLens/third_party/PE-AWARE-LRP/NLP")
sys.path.insert(0, "/home/sangyu/Desktop/Master/SpecLens/scripts")
from nltk.tokenize.punkt import PunktSentenceTokenizer

_ccs = ilu.spec_from_file_location("cc_solver", "/home/sangyu/Desktop/Master/SpecLens/third_party/context-cite/context_cite/solver.py")
_ccm = ilu.module_from_spec(_ccs); _ccs.loader.exec_module(_ccm); LassoRegression = _ccm.LassoRegression

P3B = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/"))[-1]
ARROWS = sorted(glob.glob("/data/.cache/huggingface/datasets/EleutherAI___fineweb-edu-dedup-10b/**/fineweb*.arrow", recursive=True))
OUT = "/home/sangyu/Desktop/Master/SpecLens/outputs/attr_showcase"
dev = "cuda:0"
M = 192

NAMES = ["Tom", "Alex", "Bob", "Carl", "Sam", "Max", "Ben", "Leo", "Emma", "Anna", "Sara", "Kate", "Ryan", "Jack", "Mark", "Paul"]
ITEMS = ["apples", "books", "coins", "pencils", "stickers", "marbles", "cards", "shells"]
JOBS = ["nurse", "pilot", "teacher", "doctor", "lawyer", "farmer", "baker", "chef", "singer", "dancer", "painter", "writer"]


def split_sents(t):
    t = re.sub(r"\s+", " ", t.strip()); return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 1]


def relu_norm(x):
    x = np.clip(x, 0, None); return x / (x.max() + 1e-9)


# ---- verbatim from SelfCite (see research_nl_selfcite_compare.py) ----
def sc_split(original_text):
    text = original_text
    tkz = PunktSentenceTokenizer(); punct = r"([。；！？])"
    sep = sum([re.split(punct, s) for s in tkz.tokenize(text)], [])
    for i in range(1, len(sep)):
        if re.match(punct, sep[i]):
            sep[i - 1] += sep[i]; sep[i] = ""
    sep = [s for s in sep if s != ""]
    if len(sep) == 1:
        sep = original_text.split("\n\n")
    sep = [s.strip() for s in sep if s.strip() != ""]
    pos = 0; res = []
    for i, sent in enumerate(sep):
        st = original_text.find(sent, pos); ed = st + len(sent)
        res.append({"content": sent, "start_idx": st, "end_idx": ed}); pos = ed
    return res


SC_PROMPT_STR = """Please answer the user's question based on the following document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's question.\n\n[Document Start]\n%s\n[Document End]\n\n%s"""


def sc_prompt(context, question, drop_ids=None, keep_ids=None):
    sents = sc_split(context)
    sp = ""
    for i, s in enumerate(sents):
        if drop_ids is not None and i in drop_ids:
            continue
        if keep_ids is not None and i not in keep_ids:
            continue
        st = s["start_idx"]
        ed = sents[i + 1]["start_idx"] if i < len(sents) - 1 else len(context)
        sp += f"<C{i}>" + context[st:ed]
    return SC_PROMPT_STR % (sp, question), len(sents), sents


def main():
    os.makedirs(OUT, exist_ok=True)
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from datasets import Dataset
    from lxt.utils import clean_tokens, pdf_heatmap
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research_nl_showcase_v43 import EVEREST
    tok = AutoTokenizer.from_pretrained(P3B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(P3B, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to(dev).eval()
    embl = model.get_input_embeddings(); eos = embl(torch.tensor([[tok.eos_token_id]], device=dev))[0, 0].detach()
    NF = [0]; ORMETA = [None]

    def fwd_v(ids, ctx, masks, ans_pos, batch=12):
        NF[0] += len(masks)
        T = ids.shape[1]; emb = embl(ids).detach(); outs = []
        atoks = [ids[0, p].item() for p in ans_pos]
        pos_prev = [p - 1 for p in ans_pos]
        bsz = max(2, min(batch, (12 * 600) // max(T, 1)))
        for b in range(0, len(masks), bsz):
            es = []
            for mk in masks[b:b + bsz]:
                gt = torch.ones(T, device=dev, dtype=emb.dtype)
                off = [ctx[i] for i in range(len(ctx)) if mk[i] < 0.5]
                if off:
                    gt[torch.tensor(off, device=dev)] = 0.0
                es.append(emb[0] * gt[:, None] + eos[None] * (1 - gt[:, None]))
            with torch.no_grad():
                lg = model(inputs_embeds=torch.stack(es)).logits[:, pos_prev, :].float()
            lp = torch.log_softmax(lg, -1)
            outs.append(torch.stack([lp[:, i, at] for i, at in enumerate(atoks)], 1).sum(1).cpu())
        return torch.cat(outs).numpy()

    # ---- token-toggle fwd (verbatim research_nl_compare.py) ----
    def fwd(ids, toggle, masks, ans, want, batch=16):
        T = ids.shape[1]; emb = embl(ids).detach(); tg = np.array(toggle); out = []
        for b in range(0, len(masks), batch):
            es = []
            for m in masks[b:b + batch]:
                gt = torch.ones(T, device=dev, dtype=emb.dtype); off = tg[np.asarray(m) < 0.5]
                if len(off):
                    gt[torch.tensor(off, device=dev)] = 0.0
                es.append(emb[0] * gt[:, None] + eos[None] * (1 - gt[:, None]))
            with torch.no_grad():
                p = F.softmax(model(inputs_embeds=torch.stack(es)).logits[:, -1].float(), -1)[:, ans].cpu().numpy()
            out.append(p if want == "prob" else (np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(p, 1e-6, 1 - 1e-6))))
        return np.concatenate(out)

    def attn_last(ids):
        with torch.no_grad():
            att = model(input_ids=ids, output_attentions=True).attentions
        return torch.stack([a[0].mean(0) for a in att])[:, -1, :].mean(0).float().cpu().numpy()

    def cc(ids, toggle, ans, MM=M):
        K = len(toggle); Z = (np.random.default_rng(1).random((MM, K)) < 0.5).astype(np.float32)
        w, _ = LassoRegression(0.01).fit(Z, fwd(ids, toggle, Z, ans, "logit"), 1)
        return np.asarray(w, np.float32)

    def marg(ids, toggle, ans, p, MM=512):
        K = len(toggle); r = np.random.default_rng(0); Z = (r.random((MM, K)) < p).astype(np.float32)
        R = fwd(ids, toggle, Z, ans, "prob")
        return (Z * R[:, None]).sum(0) / np.clip(Z.sum(0), 1, None) - ((1 - Z) * R[:, None]).sum(0) / np.clip((1 - Z).sum(0), 1, None)

    def online_banzhaf(ids, content, ans):
        fl = attn_last(ids)[np.array(content)]; K0 = min(len(content), max(16, round(0.4 * len(content))))
        active = [content[i] for i in np.argsort(-fl)[:K0]]; rng = np.random.default_rng(0)
        while len(active) > 8:
            K = len(active); Z = (rng.random((80, K)) < 0.5).astype(np.float32); R = fwd(ids, active, Z, ans, "prob")
            mg = (Z * R[:, None]).sum(0) / np.clip(Z.sum(0), 1, None) - ((1 - Z) * R[:, None]).sum(0) / np.clip((1 - Z).sum(0), 1, None)
            drop = set(np.argsort(mg)[:max(1, round(0.3 * K))]); active = [active[i] for i in range(K) if i not in drop]
        w = marg(ids, active, ans, 0.5, 512); out = np.zeros(len(content)); pos = {c: i for i, c in enumerate(content)}
        for j, t in enumerate(active):
            out[pos[t]] = w[j]
        return out

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

    # ---------------- v8-full pipeline ----------------
    def v81_full(ids, ctx, ranges, qtoks, ans_pos, sents):
        ns = len(ranges); L = len(ctx); cpos = {t: i for i, t in enumerate(ctx)}
        sm = np.zeros((ns, L))
        for si2, (a2, b2) in enumerate(ranges):
            for t in range(a2, b2):
                sm[si2, cpos[t]] = 1.0
        NF[0] = 0
        probes = [np.ones(L), np.zeros(L)] + [sm[s] for s in range(ns)] + [1.0 - sm[s] for s in range(ns)]
        vv = fwd_v(ids, ctx, probes, ans_pos)
        vN, v0 = float(vv[0]), float(vv[1])
        gainN = vN - v0
        g = vv[2:2 + ns] - v0
        nec_raw = vN - vv[2 + ns:2 + 2 * ns]
        nec = np.clip(nec_raw - np.median(nec_raw), 0, None)
        lens = np.maximum(relu_norm(g), relu_norm(nec))
        epsA = 0.02 * max(abs(gainN), 1e-6)
        E = sorted([s for s in range(ns) if lens[s] >= 0.3])
        gE = float(fwd_v(ids, ctx, [np.clip(sm[E].sum(0), 0, 1)], ans_pos)[0] - v0) if E else 0.0
        for sweep in range(3):
            outs = [s for s in range(ns) if s not in E and lens[s] >= 0.15]
            pr = [np.clip(sm[[x for x in E if x != s]].sum(0), 0, 1) for s in E] + \
                 [np.clip(sm[E + [s]].sum(0), 0, 1) for s in outs]
            if not pr:
                break
            vv2 = fwd_v(ids, ctx, pr, ans_pos) - v0
            dm = {E[i]: gE - float(vv2[i]) for i in range(len(E))}
            da = {outs[j]: float(vv2[len(E) + j]) - gE for j in range(len(outs))}
            harmful = [s for s in E if dm[s] < -epsA]
            adds = [s for s in outs if da[s] > epsA]
            neutral = [s for s in E if s not in harmful and abs(dm[s]) <= epsA and lens[s] < 0.15]
            if not harmful and not adds and not neutral:
                break
            E = sorted((set(E) - set(harmful) - set(neutral)) | set(adds))
            gE = float(fwd_v(ids, ctx, [np.clip(sm[E].sum(0), 0, 1)], ans_pos)[0] - v0) if E else 0.0
        for s in np.argsort(-lens):
            if gE >= 0.9 * gainN:
                break
            if int(s) not in E:
                E = sorted(E + [int(s)])
                gE = float(fwd_v(ids, ctx, [np.clip(sm[E].sum(0), 0, 1)], ans_pos)[0] - v0)
        for s in list(np.argsort(lens)):
            s = int(s)
            if s not in E or len(E) <= 1:
                continue
            E2 = [x for x in E if x != s]
            g2 = float(fwd_v(ids, ctx, [np.clip(sm[E2].sum(0), 0, 1)], ans_pos)[0] - v0)
            if g2 >= 0.9 * gainN and (gE - g2) <= epsA:
                E = E2; gE = g2
        # ---- S3 v8.1: SPAN-DESCENT credit (set-then-credit recursed INTO the sentence) ----
        T = ids.shape[1]
        full = np.zeros(T, np.float32)
        etoks = [t for s in E for t in range(*ranges[s])]
        sent_of = {t: s for s in E for t in range(*ranges[s])}
        wt = lambda t: tok.decode([ids[0, t].item()]).strip()
        content = lambda t: (wt(t).isalnum() and len(wt(t)) >= 2) or wt(t).isdigit()
        span_credit = {}
        for s in E:
            a2, b2 = ranges[s]
            toks = list(range(a2, b2))
            solo = sm[s]

            def drop_of(spans):
                pr = []
                for sp in spans:
                    m2 = solo.copy()
                    for t in sp:
                        m2[cpos[t]] = 0.0
                    pr.append(m2)
                vv2 = fwd_v(ids, ctx, pr, ans_pos) - v0
                return [max(0.0, g[s] - float(x)) for x in vv2]

            leaf = {}
            frontier = [toks]
            for depth in range(5):
                if not frontier:
                    break
                drops = drop_of(frontier)
                nxt = []
                for sp, dr in zip(frontier, drops):
                    if len(sp) <= 2 or dr <= epsA:
                        leaf[tuple(sp)] = dr
                        continue
                    h = len(sp) // 2
                    c1, c2 = sp[:h], sp[h:]
                    d1, d2 = drop_of([c1, c2])
                    if max(d1, d2) <= epsA:
                        leaf[tuple(c1)] = dr; leaf[tuple(c2)] = dr   # OR-alternates inherit parent credit
                    else:
                        for c, dc in zip((c1, c2), (d1, d2)):
                            if dc > epsA and len(c) > 2:
                                nxt.append(c)
                            else:
                                leaf[tuple(c)] = dc
                frontier = nxt
            # v8.6 TWO-LENS leaves: OR-alternate signature = solo-sufficient AND deletion-unnecessary.
            # k-way OR safe (each of k copies is individually sufficient), no pairwise machinery, no span heuristics.
            leaves = list(leaf.keys())
            solo_pr = []
            for sp in leaves:
                m2 = np.zeros(L)
                for t in sp:
                    m2[cpos[t]] = 1.0
                solo_pr.append(m2)
            sg = fwd_v(ids, ctx, solo_pr, ans_pos) - v0 if leaves else np.array([])
            for sp, sgain in zip(leaves, sg):
                if float(sgain) > epsA and leaf[sp] <= epsA:
                    leaf[sp] = max(leaf[sp], float(sgain))
            vals = list(leaf.values())
            cbar = float(np.mean(vals)) if vals else 0.0
            mx = max(vals) if vals else 1.0
            for sp, dr in leaf.items():
                share = (dr + cbar) / (mx + cbar + 1e-9)
                for t in sp:
                    span_credit[t] = max(span_credit.get(t, 0.0), lens[s] * share)
        allpos = ctx + qtoks
        loo_t = [t for t in etoks if content(t)] + list(qtoks if len(qtoks) <= 80 else [t for t in qtoks if content(t)])
        lpr = []
        for t in loo_t:
            m2 = np.ones(len(allpos)); m2[allpos.index(t)] = 0.0; lpr.append(m2)
        lv = fwd_v(ids, allpos, lpr, ans_pos) if loo_t else np.array([])
        loo = {t: max(0.0, vN - float(x)) for t, x in zip(loo_t, lv)}
        mxl_c = max([loo[t] for t in loo_t if t in etoks], default=0.0) + 1e-9
        for t in etoks:
            full[t] = float(max(span_credit.get(t, 0.0), loo.get(t, 0.0) / mxl_c))
        # query SPAN-DESCENT (same machinery as E-sentences; world = full, del reading + parent-inheritance)
        qleaf = {}

        def qdrop(spans):
            pr = []
            for sp in spans:
                m2 = np.ones(len(allpos))
                for t in sp:
                    m2[allpos.index(t)] = 0.0
                pr.append(m2)
            vv3 = fwd_v(ids, allpos, pr, ans_pos)
            return [max(0.0, vN - float(x)) for x in vv3]

        qfront = [list(qtoks)]
        for depth in range(6):
            if not qfront:
                break
            qd = qdrop(qfront)
            nxt = []
            for sp, dr in zip(qfront, qd):
                if len(sp) <= 2 or dr <= epsA:
                    qleaf[tuple(sp)] = dr
                    continue
                h = len(sp) // 2
                c1, c2 = sp[:h], sp[h:]
                d1, d2 = qdrop([c1, c2])
                if max(d1, d2) <= epsA:
                    qleaf[tuple(c1)] = dr; qleaf[tuple(c2)] = dr
                else:
                    for c, dc in zip((c1, c2), (d1, d2)):
                        if dc > epsA and len(c) > 2:
                            nxt.append(c)
                        else:
                            qleaf[tuple(c)] = dc
            qfront = nxt
        qvals = np.array([max(loo.get(t, 0.0), max((dr for sp, dr in qleaf.items() if t in sp), default=0.0)) for t in qtoks])
        qn = relu_norm(np.log1p(qvals)) if len(qvals) else qvals
        for i, t in enumerate(qtoks):
            full[t] = float(qn[i])
        cand = [s for s in range(ns) if s not in E]
        if cand:
            hv = fwd_v(ids, ctx, [np.clip(sm[E + [s]].sum(0), 0, 1) for s in cand], ans_pos) - v0
            for s, x in zip(cand, hv):
                if (float(x) - gE) < -epsA:
                    for t in range(*ranges[s]):
                        full[t] = -min(1.0, (gE - float(x)) / max(abs(gainN), 1e-6))
        return full, E, gE, gainN, NF[0]

    # ---------------- cases ----------------
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
    pool = []
    for k in range(40):
        nm, nm2 = rng.choice(NAMES, 2, replace=False)
        it, it2 = rng.choice(ITEMS, 2, replace=False)
        vs = [int(rng.integers(2, 7)) for _ in range(4)]
        tot = sum(vs)
        p1 = f"On Monday, {nm} collected {vs[0]} {it}."
        p2 = f"On Tuesday, {nm} found {vs[1]} more {it}."
        p3 = f"{nm} picked up {vs[2]} {it} on Wednesday."
        p4 = f"On Thursday, {nm} gathered another {vs[3]} {it}."
        dis = f"{nm2} collected {int(rng.integers(2, 9))} {it2} on Friday."
        nf = int(rng.integers(7, 11)); fl = [fillB[int(i)] for i in rng.choice(len(fillB), nf, replace=False)]
        sents = list(fl)
        pos = sorted(rng.choice(nf + 1, 5, replace=True))
        for j, s in enumerate([p1, p2, p3, p4, dis]):
            sents.insert(min(pos[j] + j, len(sents)), s)
        q2 = f"Question: how many {it} did {nm} collect in total over the week? Answer: In total, {nm} collected"
        pool.append(("distsum", sents, q2, " " + str(tot), dict(P1=[sents.index(p1), sents.index(p2)], P2=[sents.index(p3), sents.index(p4)], dis=sents.index(dis))))
    picked = []
    for fam, sents, query, ans_str, meta in pool:
        idl = [tok.bos_token_id]; rgs = []
        for s in sents:
            t0 = len(idl); idl += tok(" " + s, add_special_tokens=False).input_ids; rgs.append((t0, len(idl)))
        q0x = len(idl); idl += tok(" " + query, add_special_tokens=False).input_ids
        a_ids = tok(ans_str, add_special_tokens=False).input_ids
        ids0 = torch.tensor([idl], device=dev)
        with torch.no_grad():
            lg1 = model(input_ids=ids0).logits[0, -1]
        if int(lg1.argmax()) != a_ids[0]:
            if sum(1 for p in pool if p[0] == "orand") <= 30:
                print(f"  [dbg] want='{ans_str}' got='{tok.decode([int(lg1.argmax())])}'", flush=True)
            continue
        ctx0 = list(range(1, q0x)); L0 = len(ctx0); cp0 = {t: i for i, t in enumerate(ctx0)}
        print(f"  valid distsum: gt={sorted(meta['P1'] + meta['P2'])} dis={meta['dis']} ans='{ans_str}'", flush=True)
        picked.append(("distsum01", sents, query, ans_str))
        ORMETA[0] = meta
        break
    print(f"cases: {[p[0] for p in picked]}", flush=True)

    for cname, sents, query, ans_str in picked:
        idl = [tok.bos_token_id]; ranges = []
        for s in sents:
            t0 = len(idl); idl += tok(" " + s, add_special_tokens=False).input_ids; ranges.append((t0, len(idl)))
        q0 = len(idl); idl += tok(" " + query, add_special_tokens=False).input_ids
        a0 = len(idl); a_ids = tok(ans_str, add_special_tokens=False).input_ids
        ids_q = torch.tensor([idl], device=dev)
        ids_full = torch.tensor([idl + list(a_ids)], device=dev)
        ctx = list(range(1, q0)); qtoks = list(range(q0, a0)); ans_pos = list(range(a0, a0 + len(a_ids)))
        ans1 = a_ids[0]
        words_q = clean_tokens(list(tok.convert_ids_to_tokens(idl)))
        maps = {}
        f8, E8, gE8, gainN8, cost8 = v81_full(ids_full, ctx, ranges, qtoks, ans_pos, sents)
        maps["v8"] = f8[:len(idl)]
        m = ORMETA[0]
        diag = []
        smq = np.zeros((len(ranges), len(ctx))); cpq = {t: i for i, t in enumerate(ctx)}
        for si2, (a2, b2) in enumerate(ranges):
            for t in range(a2, b2):
                smq[si2, cpq[t]] = 1.0
        prem = m["P1"] + m["P2"] + [m["dis"]]
        v0d = fwd_v(ids_full, ctx, [np.zeros(len(ctx))], ans_pos)[0]
        vNd = fwd_v(ids_full, ctx, [np.ones(len(ctx))], ans_pos)[0]
        vv2 = fwd_v(ids_full, ctx, [smq[si2] for si2 in prem] + [1.0 - smq[si2] for si2 in prem], ans_pos)
        for j, si2 in enumerate(prem):
            diag.append(f"s{si2}: solo={vv2[j] - v0d:+.2f} del={vNd - vv2[len(prem) + j]:+.2f}")
        print("  PREMISE 2-lens: " + "  ".join(diag), flush=True)
        w_cc = cc(ids_q, ctx, ans1)
        m_cc = np.zeros(len(idl), np.float32); m_cc[np.array(ctx)] = w_cc
        maps["contextcite"] = m_cc
        context = " ".join(sents)
        pfull, nsc, sc_sents = sc_prompt(context, query)
        a_list = list(tok(ans_str, add_special_tokens=False).input_ids)
        lp_full = sc_logprob(pfull, a_list)
        sc_score = np.zeros(nsc)
        for i in range(nsc):
            pd_, _, _ = sc_prompt(context, query, drop_ids={i})
            ph_, _, _ = sc_prompt(context, query, keep_ids={i})
            sc_score[i] = (lp_full - sc_logprob(pd_, a_list)) + (sc_logprob(ph_, a_list) - lp_full)
        m_sc = np.zeros(len(idl), np.float32)
        for i, s in enumerate(sc_sents):
            oj = next((j for j, o in enumerate(sents) if s["content"] in o or o in s["content"]), -1)
            if oj >= 0:
                a2, b2 = ranges[oj]
                m_sc[a2:b2] += sc_score[i]
        maps["selfcite"] = m_sc


        for mname, arr in maps.items():
            aa = np.asarray(arr, np.float32)
            pdf_heatmap(words_q, np.clip(aa / (np.abs(aa).max() + 1e-9), -1, 1),
                        path=f"{OUT}/cmp35_{cname}_{mname}.pdf", backend="xelatex", delete_aux_files=True)
        print(f"  {cname}: v8 E={E8} gain%={gE8 / max(gainN8, 1e-6):.2f} cost={cost8} | 4 maps rendered", flush=True)
    print(f"-> {OUT}/cmp27_*.pdf", flush=True)


if __name__ == "__main__":
    main()
