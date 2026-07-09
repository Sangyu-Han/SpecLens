#!/usr/bin/env python3
"""BENCH-100: harder-than-copy families on the corrected GAME (v8-full = game E-assembly + game S3).
Families: mathctx(30, answer computed NOT copied - AND structure), needle(25, 2-3 docs concat),
para(25, model-generated paraphrase question, 4-gram overlap ban), mhop(20, link sentence w/o answer string
buried in real fillers). All cases filtered by model-answers-correctly. PDFs -> outputs/attr_bench100/.
Game: v(S)=sum_k logP(ans_k|S,ans_<k); gain-referenced; teacher-forced (answer length free).
MODEL env: path glob key (default 3.2-3B); RENDER=0 to skip pdfs. cuda:0."""
from __future__ import annotations

import glob
import json
import os
import re

import sys

import numpy as np
import torch

sys.path.insert(0, "/home/sangyu/Desktop/Master/SpecLens/third_party/PE-AWARE-LRP/NLP")

MKEY = os.environ.get("MODEL", "Llama-3.2-3B-Instruct")
PMODEL = sorted(glob.glob(f"/data/.cache/huggingface/hub/models--meta-llama--{MKEY}/snapshots/*/"))[-1]
ARROWS = sorted(glob.glob("/data/.cache/huggingface/datasets/EleutherAI___fineweb-edu-dedup-10b/**/fineweb*.arrow", recursive=True))
OUT = "/home/sangyu/Desktop/Master/SpecLens/outputs/attr_bench100"
RENDER = os.environ.get("RENDER", "1") == "1"
dev = "cuda:0"

NAMES = ["Tom", "Alex", "Bob", "Carl", "Sam", "Max", "Ben", "Leo", "Emma", "Anna", "Sara", "Kate", "Ryan", "Jack", "Mark", "Paul"]
ITEMS = ["apples", "books", "coins", "pencils", "stickers", "marbles", "cards", "shells"]
JOBS = ["nurse", "pilot", "teacher", "doctor", "lawyer", "farmer", "baker", "chef", "singer", "dancer", "painter", "writer"]


def split_sents(t):
    t = re.sub(r"\s+", " ", t.strip()); return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 1]


def relu_norm(x):
    x = np.clip(x, 0, None); return x / (x.max() + 1e-9)


def main():
    os.makedirs(OUT, exist_ok=True)
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(PMODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(PMODEL, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to(dev).eval()
    embl = model.get_input_embeddings(); eos = embl(torch.tensor([[tok.eos_token_id]], device=dev))[0, 0].detach()
    if RENDER:
        from lxt.utils import clean_tokens, pdf_heatmap
    NF = [0]

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

    # ---------------- fillers + docs ----------------
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

    # ---------------- case builders ----------------
    cases = []  # (family, sents, query, ans_str, meta{gt_sents, decoy_sents})

    # A) mathctx: computed answer (NOT in context); premises AND-structure; distractor entity numbers
    for k in range(60):
        nm, nm2 = rng.choice(NAMES, 2, replace=False)
        it, it2 = rng.choice(ITEMS, 2, replace=False)
        v1 = int(rng.integers(6, 20)); v2 = int(rng.integers(2, min(v1, 9)))
        if rng.random() < 0.5:
            p1 = f"{nm} had {v1} {it} in the morning."; p2 = f"Later {nm} gave away {v2} {it}."; ansn = v1 - v2
        else:
            p1 = f"{nm} had {v1} {it} in the morning."; p2 = f"Then {nm} bought {v2} more {it}."; ansn = v1 + v2
        dis = f"{nm2} had {int(rng.integers(3, 15))} {it2}."
        nf = int(rng.integers(6, 11)); fl = [fillB[int(i)] for i in rng.choice(len(fillB), nf, replace=False)]
        sents = list(fl)
        pos = sorted(rng.choice(nf + 1, 3, replace=True))
        gt_loc = []
        for j, s in enumerate([p1, p2, dis]):
            at = min(pos[j] + j, len(sents)); sents.insert(at, s)
        gt_loc = [sents.index(p1), sents.index(p2)]; dloc = [sents.index(dis)]
        q = f"Question: how many {it} does {nm} have now? Answer: {nm} now has"
        cases.append(("mathctx", sents, q, " " + str(ansn), dict(gt=gt_loc, decoy=dloc)))

    # B) needle: 2-3 concatenated docs, number-cloze in one
    for k in range(70):
        picks = rng.choice(len(docs), int(rng.integers(2, 4)), replace=False)
        sents = []
        for pi in picks:
            sents += docs[int(pi)][:16]
        if len(sents) > 42:
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
        q = "According to the text, " + sents[si][:m.start()].rstrip()
        cases.append(("needle", sents, q, " " + m.group(1), dict(gt=[si], decoy=others[:4])))

    # C) para: model-generated paraphrase question (built later, needs model)
    para_cand = []
    for k in range(90):
        ds = docs[int(rng.integers(len(docs)))]
        if len(ds) > 14:
            ds = ds[:14]
        for si, s in enumerate(ds):
            m = num_re.search(s)
            if m and len(s[:m.start()].split()) >= 5:
                para_cand.append((ds, si, m.group(1)))
                break

    # D) mhop: 2-hop chain, link sentence has NO answer string, buried in many fillers
    for k in range(50):
        a, b = rng.choice(NAMES, 2, replace=False); jb = JOBS[int(rng.integers(len(JOBS)))]
        c1 = f"{a}'s colleague is {b}."; c2 = f"{b} works as a {jb}."
        nf = int(rng.integers(14, 24)); fl = [fillB[int(i)] for i in rng.choice(len(fillB), nf, replace=False)]
        sents = list(fl)
        for s in (c1, c2):
            sents.insert(int(rng.integers(len(sents) + 1)), s)
        q = f"Question: the job of {a}'s colleague? Answer: The colleague of {a} works as a"
        cases.append(("mhop", sents, q, " " + jb, dict(gt=sorted([sents.index(c1), sents.index(c2)]), link=sents.index(c1))))

    # ---- generate paraphrase questions with the model itself ----
    def gen_q(sent, numstr):
        p = (f'Text: "{sent}"\nWrite one short question (max 15 words) asking for the number {numstr} '
             f"mentioned in the text, using different wording than the text.\nQuestion:")
        ii = tok(p, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**ii, max_new_tokens=28, do_sample=False, pad_token_id=tok.eos_token_id)
        g = tok.decode(o[0, ii.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
        return g[:g.index("?") + 1] if "?" in g else None

    def ngrams(t, n=4):
        w = re.findall(r"[a-z0-9]+", t.lower()); return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}

    for ds, si, numstr in para_cand:
        if sum(1 for c in cases if c[0] == "para") >= 40:
            break
        qq = gen_q(ds[si], numstr)
        if not qq or numstr in qq:
            continue
        if ngrams(qq) & ngrams(ds[si]):
            continue
        others = [sj for sj, s2 in enumerate(ds) if sj != si and num_re.search(s2)]
        cases.append(("para", list(ds), "Question: " + qq + " Answer: The number is", " " + numstr, dict(gt=[si], decoy=others[:4])))
    print(f"built cases: " + " ".join(f"{f}:{sum(1 for c in cases if c[0] == f)}" for f in ("mathctx", "needle", "para", "mhop")), flush=True)

    # ---------------- v8-full pipeline ----------------
    def v8_full(ids, ctx, ranges, qtoks, ans_pos, sents):
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

    # ---------------- run ----------------
    rows = []; counts = {}
    for fam, sents, query, ans_str, meta in cases:
        if counts.get(fam, 0) >= {"mathctx": 30, "needle": 25, "para": 25, "mhop": 20}[fam]:
            continue
        idl = [tok.bos_token_id]; ranges = []
        for s in sents:
            t0 = len(idl); idl += tok(" " + s, add_special_tokens=False).input_ids; ranges.append((t0, len(idl)))
        q0 = len(idl); idl += tok(" " + query, add_special_tokens=False).input_ids
        a0 = len(idl); a_ids = tok(ans_str, add_special_tokens=False).input_ids
        idl += a_ids
        ids = torch.tensor([idl], device=dev)
        with torch.no_grad():
            lg1 = model(input_ids=ids[:, :a0]).logits[0, -1]
        if int(lg1.argmax()) != a_ids[0]:
            continue
        counts[fam] = counts.get(fam, 0) + 1
        k = counts[fam]
        ctx = list(range(1, q0)); qtoks = list(range(q0, a0)); ans_pos = list(range(a0, len(idl)))
        full, E, gE, gainN, cost = v8_full(ids, ctx, ranges, qtoks, ans_pos, sents)
        gt = meta["gt"]; dec = meta.get("decoy", [])
        rn = relu_norm(np.clip(full, 0, None))
        dmax = float(max([rn[t] for s in dec for t in range(*ranges[s])], default=0.0))
        row = dict(family=fam, idx=k, gt_in_E=int(all(s in E for s in gt)), E=E, nE=len(E),
                   extra=len([s for s in E if s not in gt]), gainRatio=gE / max(gainN, 1e-6),
                   decoyMax=dmax, cost=cost, ns=len(ranges))
        if fam == "mhop":
            row["linkRet"] = int(meta["link"] in E)
        rows.append(row)
        if RENDER:
            words = clean_tokens(list(tok.convert_ids_to_tokens(idl)))
            arr = np.clip(full / (np.abs(full).max() + 1e-9), -1, 1)
            pdf_heatmap(words, arr, path=f"{OUT}/{fam}{k:02d}.pdf", backend="xelatex", delete_aux_files=True)
        print(f"  {fam}{k:02d}: gt{'✓' if row['gt_in_E'] else '✗'} E={E} extra={row['extra']} "
              f"decoyMax={dmax:.2f} cost={cost} ns={len(ranges)}" + (f" link={'✓' if row.get('linkRet') else '✗'}" if fam == "mhop" else ""), flush=True)
    with open(f"{OUT}/summary_v9_{MKEY.replace('/', '_')}.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n=== BENCH-100 [{MKEY}] ===", flush=True)
    print(f"{'family':<9}{'n':>4}{'gt-in-E':>9}{'|E|':>6}{'extra':>7}{'decoyMax':>9}{'gainR':>7}{'cost':>6}{'linkRet':>8}", flush=True)
    for fam in ("mathctx", "needle", "para", "mhop"):
        rs = [r for r in rows if r["family"] == fam]
        if not rs:
            continue
        lr = np.mean([r["linkRet"] for r in rs]) if fam == "mhop" else float("nan")
        print(f"{fam:<9}{len(rs):>4}{np.mean([r['gt_in_E'] for r in rs]):>9.2f}{np.mean([r['nE'] for r in rs]):>6.1f}"
              f"{np.mean([r['extra'] for r in rs]):>7.2f}{np.mean([r['decoyMax'] for r in rs]):>9.2f}"
              f"{np.mean([r['gainRatio'] for r in rs]):>7.2f}{np.mean([r['cost'] for r in rs]):>6.0f}"
              + (f"{lr:>8.2f}" if fam == "mhop" else ""), flush=True)
    print(f"-> {OUT}/ ({sum(counts.values())} cases)", flush=True)


if __name__ == "__main__":
    main()
