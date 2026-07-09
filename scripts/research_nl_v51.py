#!/usr/bin/env python3
"""FINAL quantitative bench (user: more samples). ~60 generated+real cases x {v22, v42, pipelineCC}.
Tasks: multihop(10)/decoy(16)/redund(10) [harden seeds], kofn(10, 70000+), redund5(8, 40000+ci), everest1922/MCQ.
Labels: intervention-informed (MCQ: Mallory REMOVED from decoy per 5w verdict; decoy=Chinese-team only).
Metrics: keyMean/funcMean/decoyMax/fillMass/queryMass + E recall/precision (v22/v42) + acid(E-keep; CC=0.4-snap)
+ top1. JSON -> bench_v51.json. 3B eager, cuda:0."""
from __future__ import annotations

import glob
import importlib.util as ilu
import json
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/sangyu/Desktop/Master/SpecLens/third_party/PE-AWARE-LRP/NLP")
_ccs = ilu.spec_from_file_location("cc_solver", "/home/sangyu/Desktop/Master/SpecLens/third_party/context-cite/context_cite/solver.py")
_ccm = ilu.module_from_spec(_ccs); _ccs.loader.exec_module(_ccm); LassoRegression = _ccm.LassoRegression
P3B = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/*/"))[-1]
ARROWS = sorted(glob.glob("/data/.cache/huggingface/datasets/EleutherAI___fineweb-edu-dedup-10b/**/fineweb*.arrow", recursive=True))
SDIR = "/home/sangyu/Desktop/Master/SpecLens/outputs/set_discovery"
dev = "cuda:0"
TAU_GATE = 0.05

NAMES = ["Tom", "Alex", "Bob", "Carl", "Sam", "Max", "Ben", "Leo", "Emma", "Anna", "Sara", "Kate", "Ryan", "Jack", "Mark", "Paul"]
JOBS = ["nurse", "pilot", "teacher", "doctor", "lawyer", "farmer", "baker", "chef", "singer", "dancer", "painter", "writer"]
PWORDS = ["banana", "apple", "tiger", "piano", "sunset", "garlic", "marble", "velvet", "pepper", "candle"]
ANIMALS = ["cat", "dog", "rabbit", "turtle", "horse", "fish", "bird", "snake"]
R5_TMPL = ["The magic password is {w}.", "The secret word is {w}.", "The code phrase is {w}.",
           "Remember, the password is {w}.", "The access word is {w}."]
M1 = ["Tom has a {a}.", "Tom owns a {a}.", "There is a {a} at Tom's house."]
M2 = ["Tom's {a} is very playful.", "Tom often feeds his {a}.", "Tom's {a} sleeps a lot."]

EVEREST = ("Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main "
           "climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and "
           "the other from the north in Tibet. While not posing substantial technical climbing challenges on the "
           "standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards "
           "from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 "
           "bodies remain on the mountain and have not been removed due to the dangerous conditions. The first "
           "recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow "
           "foreigners to enter the country at the time, the British made several attempts on the north ridge route "
           "from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m "
           "(22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), "
           "marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one "
           "of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit "
           "attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. "
           "Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the "
           "southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 "
           "Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first "
           "reported ascent of the peak from the north ridge on 25 May 1960.")


def split_sents(t):
    t = re.sub(r"\s+", " ", t.strip()); return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 1]


def relu_norm(x):
    x = np.clip(x, 0, None); return x / (x.max() + 1e-9)


def main():
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(P3B, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(P3B, dtype=torch.bfloat16, local_files_only=True, attn_implementation="eager").to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    embl = model.get_input_embeddings(); eos = embl(torch.tensor([[tok.eos_token_id]], device=dev))[0, 0].detach()
    names = [w for w in NAMES if len(tok(" " + w, add_special_tokens=False).input_ids) == 1]
    jobs = [w for w in JOBS if len(tok(" " + w, add_special_tokens=False).input_ids) == 1]
    pwords = [w for w in PWORDS if len(tok(" " + w, add_special_tokens=False).input_ids) == 1]
    animals = [a for a in ANIMALS if len(tok(" " + a, add_special_tokens=False).input_ids) == 1]
    banG = set(w.lower() for w in NAMES + JOBS + PWORDS) | {"password", "word", "code", "phrase", "secret", "colleague", "magic", "access"}
    banA = set(ANIMALS) | {"animal", "animals", "pet", "pets", "tom"}
    d = Dataset.from_file(ARROWS[0]); fillB = []; fillC = []
    for i in range(len(d)):
        for s in split_sents(d[i]["text"]):
            w = s.split()
            if 6 <= len(w) <= 14 and s[0].isupper() and s.endswith(".") and sum(c.isalpha() or c.isspace() or c in ".,'" for c in s) / len(s) > 0.95:
                if not any(k in banG for k in re.findall(r"[a-z]+", s.lower())):
                    fillB.append(s)
                if not any(k in banA for k in re.findall(r"[a-z]+", s.lower())):
                    fillC.append(s)
        if len(fillB) >= 400 and len(fillC) >= 300:
            break

    nfwd = [0]

    def fwd_logits(ids, ctx, masks, batch=14):
        nfwd[0] += len(masks)
        T = ids.shape[1]; emb = embl(ids).detach(); out = []
        for b in range(0, len(masks), batch):
            es = []
            for m in masks[b:b + batch]:
                gt = torch.ones(T, device=dev, dtype=emb.dtype)
                off = [ctx[i] for i in range(len(ctx)) if m[i] < 0.5]
                if off:
                    gt[torch.tensor(off, device=dev)] = 0.0
                es.append(emb[0] * gt[:, None] + eos[None] * (1 - gt[:, None]))
            with torch.no_grad():
                out.append(model(inputs_embeds=torch.stack(es)).logits[:, -1].float().cpu())
        return torch.cat(out)

    def fwd_p(ids, ctx, masks, ans, want="prob"):
        lg = fwd_logits(ids, ctx, masks)
        p = F.softmax(lg, -1)[:, ans].numpy()
        if want == "prob":
            return p
        return np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(p, 1e-6, 1 - 1e-6))

    def attn_last(ids):
        with torch.no_grad():
            att = model(input_ids=ids, output_attentions=True).attentions
        return torch.stack([a[0].mean(0) for a in att])[:, -1, :].mean(0).float().cpu().numpy()

    def kl(P, Q):
        return float((P * (torch.log(P + 1e-12) - torch.log(Q + 1e-12))).sum())

    def margin_of(lgrow, ans):
        pr = F.softmax(lgrow, -1); pa = float(pr[ans])
        pr2 = pr.clone(); pr2[ans] = 0
        return pa - float(pr2.max())

    def v42_map(ids, ctx, ranges, qtoks, ans, eps=0.02):
        T = ids.shape[1]; L = len(ctx); cpos = {t: i for i, t in enumerate(ctx)}
        ns = len(ranges); sm = np.zeros((ns, L))
        for si, (a, b) in enumerate(ranges):
            for t in range(a, b):
                sm[si, cpos[t]] = 1.0
        lgs = fwd_logits(ids, ctx, [np.ones(L), np.zeros(L)] + [sm[si] for si in range(ns)] + [1.0 - sm[si] for si in range(ns)])
        P = F.softmax(lgs, -1)
        base = float(P[0, ans]); P0 = P[1]
        solokl = np.array([kl(P[2 + si], P0) for si in range(ns)])
        dels = P[2 + ns:2 + 2 * ns, ans].numpy()
        nec_raw = base - dels
        nec = np.clip(nec_raw - np.median(nec_raw), 0, None)
        lens = np.maximum(relu_norm(solokl), relu_norm(nec))
        E = [si for si in range(ns) if lens[si] >= 0.3]
        keep = np.clip(sm[E].sum(0), 0, 1) if E else np.zeros(L)
        lgE = fwd_logits(ids, ctx, [keep])[0] if E else None
        RE = float(F.softmax(lgE, -1)[ans]) if E else 0.0
        ME = margin_of(lgE, ans) if E else -1.0
        changed = True
        while changed and len(E) > 1:
            changed = False
            lgd = fwd_logits(ids, ctx, [np.clip(sm[[e for e in E if e != s]].sum(0), 0, 1) for s in E])
            margs = [margin_of(lgd[i], ans) for i in range(len(E))]
            worst = int(np.argmax(margs))
            if margs[worst] > ME + eps:
                E.pop(worst); ME = margs[worst]; RE = float(F.softmax(lgd[worst], -1)[ans]); keep = np.clip(sm[E].sum(0), 0, 1); changed = True
        changed = True
        while changed and len(E) > 1:
            changed = False
            lowp = [s for s in E if lens[s] < 0.15]
            if lowp:
                lgd = fwd_logits(ids, ctx, [np.clip(sm[[e for e in E if e != s]].sum(0), 0, 1) for s in lowp])
                margs = [margin_of(lgd[i], ans) for i in range(len(lowp))]
                for i in np.argsort(margs)[::-1]:
                    if margs[i] > ME - eps:
                        E.remove(lowp[i]); keep = np.clip(sm[E].sum(0), 0, 1)
                        lgE = fwd_logits(ids, ctx, [keep])[0]
                        RE = float(F.softmax(lgE, -1)[ans]); ME = margin_of(lgE, ans); changed = True
                        break
        # v4.3(a) directional-neutral prune: margin-neutral member whose solo pushes a COMPETING answer,
        # with shared-direction guard (kofn pair members push 'one' like the margin-necessary singles -> protected)
        solo_arg = [int(P[2 + si].argmax()) for si in range(ns)]
        changed = True
        while changed and len(E) > 1:
            changed = False
            lgd = fwd_logits(ids, ctx, [np.clip(sm[[e for e in E if e != s]].sum(0), 0, 1) for s in E])
            margs = [margin_of(lgd[i], ans) for i in range(len(E))]
            nonneutral_dirs = {solo_arg[E[i]] for i in range(len(E)) if margs[i] < ME - eps}
            for i in range(len(E)):
                s = E[i]
                if margs[i] > ME - eps and solo_arg[s] != ans and solo_arg[s] not in nonneutral_dirs:
                    E.pop(i); keep = np.clip(sm[E].sum(0), 0, 1)
                    lgE = fwd_logits(ids, ctx, [keep])[0]
                    RE = float(F.softmax(lgE, -1)[ans]); ME = margin_of(lgE, ans); changed = True
                    break
        for si in np.argsort(-lens):
            if RE >= 0.9 * base:
                break
            if si not in E:
                E.append(int(si)); keep = np.clip(sm[E].sum(0), 0, 1)
                lgE = fwd_logits(ids, ctx, [keep])[0]
                RE = float(F.softmax(lgE, -1)[ans]); ME = margin_of(lgE, ans)
        # v4.3(b) copy-E expansion (restored from v22; the v4 assembly omission behind redund Erec 0.700)
        copy_ctx = {t for t in ctx if ids[0, t].item() == ans}
        n_add = 0
        for t in copy_ctx:
            si = next(s for s in range(ns) if ranges[s][0] <= t < ranges[s][1])
            if si not in E:
                E.append(si); n_add += 1
        if n_add:
            keep = np.clip(sm[E].sum(0), 0, 1)
            lgE = fwd_logits(ids, ctx, [keep])[0]
            RE = float(F.softmax(lgE, -1)[ans]); ME = margin_of(lgE, ans)
        etoks = [t for si in E for t in range(*ranges[si])]
        sent_of = {t: si for si in E for t in range(*ranges[si])}
        probe = []
        for t in etoks:
            m = sm[sent_of[t]].copy(); m[cpos[t]] = 0.0; probe.append(m)
        Pd = F.softmax(fwd_logits(ids, ctx, probe), -1)
        allpos = ctx + qtoks
        looprobe = []
        for t in etoks:
            m = np.ones(len(allpos)); m[allpos.index(t)] = 0.0; looprobe.append(m)
        RLoo = F.softmax(fwd_logits(ids, allpos, looprobe), -1)[:, ans].numpy()
        loo = np.clip(base - RLoo, 0, None)
        loo_n = loo / (loo.max() + 1e-9) if loo.max() > 0 else loo
        full = np.zeros(T, np.float32)
        prof = {}
        for k_, t in enumerate(etoks):
            si = sent_of[t]
            prof.setdefault(si, {})[t] = max(0.0, kl(P[2 + si], P0) - kl(Pd[k_], P0))
        qw = {w.strip(".,?:;()'\"") for w in tok.decode(ids[0, qtoks[0]:].tolist()).split()}
        qw = {w for w in qw if len(w) >= 3}
        wt = lambda t: tok.decode([ids[0, t].item()]).strip()
        ans_str = tok.decode([ans]).strip()
        for si in E:
            ts = list(prof[si].keys()); vv = np.array([prof[si][t] for t in ts])
            vv = 0.25 + 0.75 * (vv / (vv.max() + 1e-9))
            for t, v in zip(ts, vv):
                j = etoks.index(t)
                g = 1.0 if (wt(t) == ans_str or wt(t) in qw or loo_n[j] > 0.05) else 0.35
                full[t] = float(max(lens[si] * v * g, loo_n[j]))
        qprobe = []
        for qt in qtoks:
            m = np.ones(len(allpos)); m[allpos.index(qt)] = 0.0; qprobe.append(m)
        Pq = F.softmax(fwd_logits(ids, allpos, qprobe), -1)
        qd = np.array([kl(P[0], Pq[i]) for i in range(len(qtoks))])
        qdn = relu_norm(qd)
        for i, qt in enumerate(qtoks):
            full[qt] = float(qdn[i])
        cand = [si for si in range(ns) if si not in E]
        if cand:
            Rplus = F.softmax(fwd_logits(ids, ctx, [np.clip(keep + sm[si], 0, 1) for si in cand]), -1)[:, ans].numpy()
            for si, rp in zip(cand, Rplus):
                if (rp - RE) < -0.1 * max(RE, 1e-9):
                    for t in range(*ranges[si]):
                        full[t] = -min(1.0, (RE - rp) / max(RE, 1e-9))
        return full, sorted(E), RE, base


    def v5_map(ids, ctx, ranges, qtoks, ans, eps=0.02):
        """v5: DECISION LAYER = one signed quantity c(s|E)=margin-delta, fixed-point loop (max4 sweeps).
        Replaces margin-prune + participation-prune + directional-prune + extend (4 procedural rules ->
        2 declarative clauses + identity axiom + feasibility). S3 cost cut: content-word-restricted LOO."""
        T = ids.shape[1]; L = len(ctx); cpos = {t: i for i, t in enumerate(ctx)}
        ns = len(ranges); sm = np.zeros((ns, L))
        for si, (a, b) in enumerate(ranges):
            for t in range(a, b):
                sm[si, cpos[t]] = 1.0
        lgs = fwd_logits(ids, ctx, [np.ones(L), np.zeros(L)] + [sm[si] for si in range(ns)] + [1.0 - sm[si] for si in range(ns)])
        P = F.softmax(lgs, -1)
        base = float(P[0, ans]); P0 = P[1]
        solokl = np.array([kl(P[2 + si], P0) for si in range(ns)])
        dels = P[2 + ns:2 + 2 * ns, ans].numpy()
        nec_raw = base - dels
        nec = np.clip(nec_raw - np.median(nec_raw), 0, None)
        lens = np.maximum(relu_norm(solokl), relu_norm(nec))
        solo_arg = [int(P[2 + si].argmax()) for si in range(ns)]
        copy_sents = set()
        for t in ctx:
            if ids[0, t].item() == ans:
                copy_sents.add(next(s for s in range(ns) if ranges[s][0] <= t < ranges[s][1]))
        E = sorted(set([si for si in range(ns) if lens[si] >= 0.3]) | copy_sents)
        keep = np.clip(sm[E].sum(0), 0, 1) if E else np.zeros(L)
        lgE = fwd_logits(ids, ctx, [keep])[0] if E else None
        RE = float(F.softmax(lgE, -1)[ans]) if E else 0.0
        ME = margin_of(lgE, ans) if E else -1.0
        for sweep in range(4):
            outsiders = [si for si in range(ns) if si not in E and lens[si] >= 0.15]
            probes = [np.clip(sm[[e for e in E if e != s]].sum(0), 0, 1) for s in E] + \
                     [np.clip(keep + sm[si], 0, 1) for si in outsiders]
            lgp = fwd_logits(ids, ctx, probes)
            mem_m = [margin_of(lgp[i], ans) for i in range(len(E))]
            out_m = [margin_of(lgp[len(E) + j], ans) for j in range(len(outsiders))]
            changedd = False
            # clause 1: harmful members out (c = ME - margin(E-s) < -eps), batch-safe
            harmful = [E[i] for i in range(len(E)) if (ME - mem_m[i]) < -eps and E[i] not in copy_sents]
            # clause 2: neutral member resolution (participation + direction), max 1 per sweep
            nn_dirs = {solo_arg[E[i]] for i in range(len(E)) if (ME - mem_m[i]) > eps}
            neutral_drops = [E[i] for i in range(len(E))
                             if E[i] not in copy_sents and E[i] not in harmful
                             and abs(ME - mem_m[i]) <= eps
                             and (lens[E[i]] < 0.15 or (solo_arg[E[i]] != ans and solo_arg[E[i]] not in nn_dirs))]
            adds = [outsiders[j] for j in range(len(outsiders)) if (out_m[j] - ME) > eps]
            if harmful or neutral_drops or adds:
                E = sorted((set(E) - set(harmful) - set(neutral_drops)) | set(adds) | copy_sents)
                keep = np.clip(sm[E].sum(0), 0, 1)
                lgE = fwd_logits(ids, ctx, [keep])[0]
                RE = float(F.softmax(lgE, -1)[ans]); ME = margin_of(lgE, ans)
                changedd = True
            if not changedd:
                break
        for si in np.argsort(-lens):
            if RE >= 0.9 * base:
                break
            if si not in E:
                E = sorted(set(E) | {int(si)}); keep = np.clip(sm[E].sum(0), 0, 1)
                lgE = fwd_logits(ids, ctx, [keep])[0]
                RE = float(F.softmax(lgE, -1)[ans]); ME = margin_of(lgE, ans)
        etoks = [t for si in E for t in range(*ranges[si])]
        sent_of = {t: si for si in E for t in range(*ranges[si])}
        wt = lambda t: tok.decode([ids[0, t].item()]).strip()
        content_tok = lambda t: (wt(t).isalnum() and len(wt(t)) >= 2) or wt(t).isdigit()
        probe = []
        for t in etoks:
            m = sm[sent_of[t]].copy(); m[cpos[t]] = 0.0; probe.append(m)
        Pd = F.softmax(fwd_logits(ids, ctx, probe), -1)
        allpos = ctx + qtoks
        loo_targets = [t for t in etoks if content_tok(t)]
        looprobe = []
        for t in loo_targets:
            m = np.ones(len(allpos)); m[allpos.index(t)] = 0.0; looprobe.append(m)
        RLoo = F.softmax(fwd_logits(ids, allpos, looprobe), -1)[:, ans].numpy() if looprobe else np.array([])
        loo_map = {}
        mx = max(np.clip(base - RLoo, 0, None).max(), 1e-9) if len(RLoo) else 1.0
        for t, r in zip(loo_targets, RLoo):
            loo_map[t] = max(0.0, base - float(r)) / mx
        full = np.zeros(T, np.float32)
        prof = {}
        for k_, t in enumerate(etoks):
            si = sent_of[t]
            prof.setdefault(si, {})[t] = max(0.0, kl(P[2 + si], P0) - kl(Pd[k_], P0))
        qw = {w.strip(".,?:;()'\"") for w in tok.decode(ids[0, qtoks[0]:].tolist()).split()}
        qw = {w for w in qw if len(w) >= 3}
        ans_str = tok.decode([ans]).strip()
        for si in E:
            ts = list(prof[si].keys()); vv = np.array([prof[si][t] for t in ts])
            vv = 0.25 + 0.75 * (vv / (vv.max() + 1e-9))
            for t, v in zip(ts, vv):
                ln = loo_map.get(t, 0.0)
                g = 1.0 if (wt(t) == ans_str or wt(t) in qw or ln > 0.05) else 0.35
                full[t] = float(max(lens[si] * v * g, ln))
        q_targets = qtoks if len(qtoks) <= 30 else [qt for qt in qtoks if content_tok(qt)]
        qprobe = []
        for qt in q_targets:
            m = np.ones(len(allpos)); m[allpos.index(qt)] = 0.0; qprobe.append(m)
        Pq = F.softmax(fwd_logits(ids, allpos, qprobe), -1)
        qd = np.array([kl(P[0], Pq[i]) for i in range(len(q_targets))])
        qdn = relu_norm(qd) if len(qd) else qd
        for i, qt in enumerate(q_targets):
            full[qt] = float(qdn[i])
        cand = [si for si in range(ns) if si not in E]
        if cand:
            Rplus = F.softmax(fwd_logits(ids, ctx, [np.clip(keep + sm[si], 0, 1) for si in cand]), -1)[:, ans].numpy()
            for si, rp in zip(cand, Rplus):
                if (rp - RE) < -0.1 * max(RE, 1e-9):
                    for t in range(*ranges[si]):
                        full[t] = -min(1.0, (RE - rp) / max(RE, 1e-9))
        return full, sorted(E), RE, base

    def v22_map(ids, ctx, ranges, qtoks, content, ans, seed=17 * 7 + 3):
        T = ids.shape[1]; L = len(ctx); cpos = {t: i for i, t in enumerate(ctx)}
        ns = len(ranges); sm = np.zeros((ns, L))
        for si, (a, b) in enumerate(ranges):
            for t in range(a, b):
                sm[si, cpos[t]] = 1.0
        base = fwd_p(ids, ctx, [np.ones(L)], ans)[0]
        dels = fwd_p(ids, ctx, [1.0 - sm[si] for si in range(ns)], ans)
        solos = fwd_p(ids, ctx, [sm[si] for si in range(ns)], ans)
        sc_s = np.maximum(relu_norm(base - dels), relu_norm(solos))
        E = [si for si in range(ns) if sc_s[si] >= 0.5]
        keep = np.clip(sm[E].sum(0), 0, 1) if E else np.zeros(L)
        RE = fwd_p(ids, ctx, [keep], ans)[0] if E else 0.0
        for si in np.argsort(-sc_s):
            if RE >= 0.9 * base:
                break
            if si not in E:
                E.append(int(si)); keep = np.clip(sm[E].sum(0), 0, 1); RE = fwd_p(ids, ctx, [keep], ans)[0]
        copy_ctx = {t for t in ctx if ids[0, t].item() == ans}
        for t in copy_ctx:
            si = next(s for s in range(ns) if ranges[s][0] <= t < ranges[s][1])
            if si not in E:
                E.append(si)
        keep = np.clip(sm[E].sum(0), 0, 1); RE = fwd_p(ids, ctx, [keep], ans)[0]
        etoks = [t for si in E for t in range(*ranges[si])]
        toks = etoks + qtoks; nt = len(toks)
        allc = ctx + qtoks; ap = {t: i for i, t in enumerate(allc)}
        lm = []
        for t in toks:
            m = np.ones(len(allc)); m[ap[t]] = 0.0; lm.append(m)
        baseF = fwd_p(ids, allc, [np.ones(len(allc))], ans)[0]
        loo = baseF - fwd_p(ids, allc, lm, ans)
        Zt = (np.random.default_rng(seed).random((192, nt)) < 0.5).astype(np.float32)
        om = np.zeros((192, len(allc)))
        for j, t in enumerate(toks):
            om[:, ap[t]] = Zt[:, j]
        for t in [t for t in ctx if t not in set(etoks)]:
            om[:, ap[t]] = 0.0
        Rt = fwd_p(ids, allc, om, ans)
        bz = (Zt * Rt[:, None]).sum(0) / np.clip(Zt.sum(0), 1, None) - ((1 - Zt) * Rt[:, None]).sum(0) / np.clip((1 - Zt).sum(0), 1, None)
        loo_n = relu_norm(loo); bz_n = relu_norm(bz)
        qw = {w.strip(".,?:;()'\"") for w in tok.decode(ids[0, qtoks[0]:].tolist()).split()}
        qw = {w for w in qw if len(w) >= 3}
        wt = lambda t: tok.decode([ids[0, t].item()]).strip()
        copy_ext = set(copy_ctx) | {t for t in etoks if wt(t) in qw}
        gate = np.array([1.0 if (toks[j] in copy_ext or loo_n[j] > TAU_GATE) else 0.0 for j in range(nt)])
        sc = np.maximum(loo_n, bz_n * gate)
        sent_of = {t: si for si in E for t in range(*ranges[si])}
        for j, t in enumerate(toks):
            if t in sent_of:
                sc[j] = max(sc[j], 0.15 * sc_s[sent_of[t]])
        full = np.zeros(T, np.float32)
        for j, t in enumerate(toks):
            full[t] = sc[j]
        cand = [si for si in range(ns) if si not in E]
        if cand:
            Rplus = fwd_p(ids, ctx, [np.clip(keep + sm[si], 0, 1) for si in cand], ans)
            for si, rp in zip(cand, Rplus):
                if (rp - RE) < -0.1 * max(RE, 1e-9):
                    for t in range(*ranges[si]):
                        full[t] = -min(1.0, (RE - rp) / max(RE, 1e-9))
        return full, sorted(E), RE, base

    def cc_map(ids, content, ans, seed=17 * 7 + 3):
        fl_a = attn_last(ids)[np.array(content)]; K = min(len(content), max(20, round(0.45 * len(content))))
        surv = [content[i] for i in np.argsort(-fl_a)[:K]]
        Z = (np.random.default_rng(seed + 2).random((192, K)) < 0.5).astype(np.float32)
        w_cc, _ = LassoRegression(0.01).fit(Z, fwd_p(ids, list(surv), Z, ans, "logit"), 1)
        out = np.zeros(ids.shape[1], np.float32)
        for j, t in enumerate(surv):
            out[t] = np.asarray(w_cc, np.float32)[j]
        return out

    # ---------------- case generators ----------------
    def gen_std(task, ci):
        rng = np.random.default_rng({"multihop": 10000, "decoy": 20000, "redund": 30000}[task] + ci)
        fidx = rng.choice(len(fillB), 12, replace=False); fl_ = [fillB[int(i)] for i in fidx]
        if task == "redund":
            w = pwords[rng.integers(len(pwords))]
            info = [f"The magic password is {w}.", f"The secret word is {w}.", f"The code phrase is {w}."]
            nf = int(rng.integers(3, 7)); sents = fl_[:nf]
            slots = sorted(rng.choice(nf + 1, 3, replace=True))
            for k, s in enumerate(info):
                sents.insert(min(slots[k] + k, len(sents)), s)
            return sents, "Question: what is the magic password? Answer: The magic password is", " " + w, dict(
                gt_sents=[sents.index(s) for s in info], gt_words={w}, decoy_sents=[], decoy_tok=set())
        X, Y, A, B = [names[int(i)] for i in rng.choice(len(names), 4, replace=False)]
        jb, dj = [jobs[int(i)] for i in rng.choice(len(jobs), 2, replace=False)]
        c1, c2 = f"{X}'s colleague is {Y}.", f"{Y} works as a {jb}."
        info, dsents = [c1, c2], []
        if task == "decoy":
            dsents = [f"{A}'s colleague is {B}.", f"{B} works as a {dj}."]
            info = [c1, c2] + dsents
        nf = int(rng.integers(2, 6)) if task == "decoy" else int(rng.integers(4, 9))
        sents = fl_[:nf]; order = rng.permutation(len(info))
        ins = sorted(rng.choice(nf + 1, len(info), replace=True))
        for k, oi in enumerate(order):
            sents.insert(min(ins[k] + k, len(sents)), info[oi])
        return sents, f"Question: the job of {X}'s colleague? Answer: The colleague of {X} works as a", " " + jb, dict(
            gt_sents=[sents.index(c1), sents.index(c2)], gt_words={Y, jb, "colleague"},
            decoy_sents=[sents.index(s) for s in dsents], decoy_tok=set())

    all_cases = []
    for task, n in (("multihop", 10), ("decoy", 16), ("redund", 10)):
        for ci in range(n):
            s, q, a, m = gen_std(task, ci)
            all_cases.append((task, s, q, a, m))
    for ci in range(20):
        rng = np.random.default_rng(70000 + ci)
        ch_a = [animals[int(i)] for i in rng.choice(len(animals), 2, replace=False)]
        sk = []; rk_ = []
        for k, a in enumerate(ch_a):
            sk.append(M1[int(rng.integers(len(M1)))].format(a=a)); rk_.append(("pair" if k < 1 else "single", a))
            if k < 1:
                sk.append(M2[int(rng.integers(len(M2)))].format(a=a)); rk_.append(("pair", a))
        nfk = int(rng.integers(3, 6)); sents = [fillC[int(i)] for i in rng.choice(len(fillC), nfk, replace=False)]
        irk = [None] * len(sents)
        for s, r in zip(sk, rk_):
            pos = int(rng.integers(len(sents) + 1))
            sents.insert(pos, s); irk.insert(pos, r)
        all_cases.append(("kofn", sents, "Question: how many different animals does Tom have? Answer: Tom has", " two",
                          dict(gt_sents=[i for i, r in enumerate(irk) if r], gt_words=set(a for r in irk if r for a in [r[1]]),
                               decoy_sents=[], decoy_tok=set())))
    for ci in range(8):
        rng5 = np.random.default_rng(40000 + ci)
        w5 = pwords[int(rng5.integers(len(pwords)))]
        infos5 = [t.format(w=w5) for t in R5_TMPL]
        nf5 = int(rng5.integers(3, 6)); sents5 = [fillB[int(i)] for i in rng5.choice(len(fillB), nf5, replace=False)]
        for k, s in enumerate(infos5):
            sents5.insert(int(rng5.integers(len(sents5) + 1)), s)
        all_cases.append(("redund5", sents5, "Question: what is the magic password? Answer: The magic password is", " " + w5,
                          dict(gt_sents=[sents5.index(s) for s in infos5], gt_words={w5}, decoy_sents=[], decoy_tok=set())))
    ev = split_sents(EVEREST)
    i_ev = next(i for i, s in enumerate(ev) if "8,320" in s)
    i_no = next(i for i, s in enumerate(ev) if "8,595" in s)
    i_53 = next(i for i, s in enumerate(ev) if "first documented ascent" in s)
    i_ch = next(i for i, s in enumerate(ev) if "Wang Fuzhou" in s)
    all_cases.append(("everest", ev, "Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,", "320",
                      dict(gt_sents=[i_ev], gt_words={"320", "1922"}, decoy_sents=[i_no], decoy_tok={"000", "595", "970"})))
    all_cases.append(("everest", ev, "Question: Who made the first documented ascent of Everest? Options: A) George Mallory and Andrew Irvine. B) Tenzing Norgay and Edmund Hillary. C) Wang Fuzhou, Gonpo, and Qu Yinhua. D) The 1921 British reconnaissance team. Answer: The correct option is", " B",
                      dict(gt_sents=[i_53], gt_words=None, decoy_sents=[i_ch], decoy_tok=set())))  # Mallory removed per 5w

    rows = []; kept = 0; costs = []; kofn_kept = 0
    for task, sents, query, answord, meta in all_cases:
        if task == "kofn" and kofn_kept >= 10:
            continue
        ids = [tok.bos_token_id]; ranges = []
        for s in sents:
            t0 = len(ids); ids += tok(" " + s, add_special_tokens=False).input_ids; ranges.append((t0, len(ids)))
        q0 = len(ids); ids += tok(" " + query, add_special_tokens=False).input_ids
        ids = torch.tensor([ids], device=dev); T = ids.shape[1]
        ans = tok(answord, add_special_tokens=False).input_ids[0]
        with torch.no_grad():
            lg = model(input_ids=ids).logits[0, -1].float()
        if int(lg.argmax()) != ans:
            continue
        kept += 1
        if task == "kofn":
            kofn_kept += 1
        pA = float(F.softmax(lg, -1)[ans])
        ctx = list(range(1, q0)); qtoks = list(range(q0, T)); content = list(range(1, T))
        word = lambda t: tok.decode([ids[0, t].item()]).strip()
        if meta["gt_words"]:
            key_pos = [t for si in meta["gt_sents"] for t in range(*ranges[si]) if word(t) in meta["gt_words"]]
        else:
            key_pos = [t for si in meta["gt_sents"] for t in range(*ranges[si]) if (word(t)[:1].isupper() or word(t).isdigit()) and len(word(t)) > 2]
        dec_pos = [t for si in meta["decoy_sents"] for t in range(*ranges[si])] + [t for t in ctx if word(t) in meta["decoy_tok"]]
        fil_pos = [t for si in range(len(ranges)) if si not in meta["gt_sents"] + meta["decoy_sents"] for t in range(*ranges[si]) if word(t) not in meta["decoy_tok"]]
        maps = {}
        nfwd[0] = 0
        f5, E5, RE5, base = v5_map(ids, ctx, ranges, qtoks, ans)
        c5 = nfwd[0]; nfwd[0] = 0
        f42, E42, RE42, _ = v42_map(ids, ctx, ranges, qtoks, ans)
        c42 = nfwd[0]
        costs.append((c5, c42))
        maps = {"v51": (f5, E5, RE5), "v42": (f42, E42, RE42)}
        for mname, (arr, E, RE) in maps.items():
            rn = relu_norm(arr)
            if E is not None:
                keep = np.zeros(len(ctx))
                for si2 in E:
                    for t in range(*ranges[si2]):
                        keep[t - 1] = 1.0
                Rk = RE
            else:
                keep = np.zeros(len(ctx))
                for si2 in range(len(ranges)):
                    a2, b2 = ranges[si2]
                    if max(float(arr[t]) for t in range(a2, b2)) >= 0.4 * float(np.abs(arr).max() + 1e-9):
                        for t in range(a2, b2):
                            keep[t - 1] = 1.0
                Rk = fwd_p(ids, ctx, [keep], ans)[0]
            emb = embl(ids).detach()
            gt = torch.ones(T, device=dev, dtype=emb.dtype)
            off = [ctx[i] for i in range(len(ctx)) if keep[i] < 0.5]
            if off:
                gt[torch.tensor(off, device=dev)] = 0.0
            with torch.no_grad():
                lgk = model(inputs_embeds=(emb[0] * gt[:, None] + eos[None] * (1 - gt[:, None])).unsqueeze(0)).logits[0, -1]
            ok = int(lgk.argmax()) == ans
            gtS = set(meta["gt_sents"])
            rows.append(dict(task=task, method=mname,
                             keyMean=float(rn[key_pos].mean()) if key_pos else None,
                             decoyMax=float(rn[dec_pos].max()) if dec_pos else None,
                             fillMass=float(rn[fil_pos].mean()) if fil_pos else 0.0,
                             queryMass=float(rn[qtoks].mean()),
                             Erec=(float(np.mean([si in set(E) for si in gtS])) if E is not None else None),
                             Eprec=(float(np.mean([si in gtS for si in E])) if E else None),
                             acid=float(Rk / max(pA, 1e-9)), top1=int(ok)))
        if kept % 10 == 0:
            print(f"  ... {kept} cases done", flush=True)

    with open(f"{SDIR}/bench_v51.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n=== FINAL BENCH (kept {kept} cases) ===", flush=True)
    print(f"{'task':<9}{'method':<11}{'keyMean':>8}{'decoyMax':>9}{'fillM':>7}{'queryM':>7}{'Erec':>6}{'Eprec':>6}{'acid':>7}{'top1':>6}{'n':>4}", flush=True)
    for task in ("multihop", "decoy", "redund", "kofn", "redund5", "everest"):
        for mname in ("v51", "v42"):
            rs = [r for r in rows if r["task"] == task and r["method"] == mname]
            if not rs:
                continue
            def mv(k):
                v = [r[k] for r in rs if r.get(k) is not None]
                return f"{np.mean(v):.3f}" if v else "  -  "
            print(f"{task:<9}{mname:<11}{mv('keyMean'):>8}{mv('decoyMax'):>9}{mv('fillMass'):>7}{mv('queryMass'):>7}{mv('Erec'):>6}{mv('Eprec'):>6}{mv('acid'):>7}{mv('top1'):>6}{len(rs):>4}", flush=True)
    print(f"cost fwd: v5 mean={np.mean([c[0] for c in costs]):.0f} v4.3 mean={np.mean([c[1] for c in costs]):.0f}", flush=True)
    print(f"-> {SDIR}/bench_v51.json", flush=True)


if __name__ == "__main__":
    main()
