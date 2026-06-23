#!/usr/bin/env python3
"""THE comparison that matters: FULL FRI (gradient-guided restricted-Banzhaf + closure + self-selection
= the documented LLM recipe, M=1000) vs the NEW LRP SOTA PA-LRP + AttnLRP (official llama_PE), on TIGHT
tasks (KV + arithmetic, where FRI is documented to dominate AttnLRP +7.6sigma). Goal: FRI SELECT WINS
insertion (sufficiency); oracle greedy-deletion WINS necessity. Recovery-AUC metric (RISE-normalized;
per-instance winner == raw-prob). Llama-3.2-1B-Instruct. LRP from their fork (user-authorized)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PALRP = "/tmp/PE-AWARE-LRP/NLP"
sys.path.insert(0, PALRP)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from research_xmodel_fri import closure, gen_arith, gen_kv  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = sorted(glob.glob("/data/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/*/"))[-1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
SEL_FR = [.05, .1, .2]
DPA_REPO = "/tmp/dual_path_attribution"


def dpa_scores(cases, tok, dev):
    """Official DPA (Dual Path Attribution) input scores via their tracer (nnsight). Returns {key: [S]}."""
    sys.path.insert(0, DPA_REPO)
    from nnsight import LanguageModel
    from tracer.backend import Llama2Backend
    from tracer.tracer import InputTracer
    nm = LanguageModel(LLAMA, attn_implementation="eager", torch_dtype=torch.bfloat16, device_map=dev, dispatch=True)
    backend = Llama2Backend(nm, scaling_config={"q": 0.25, "k": 0.25, "v": 0.5, "gate": 0.5, "up": 0.5}, cache_device=dev)
    tracer = InputTracer(backend, tok)
    out = {}
    for key, tname, prompt, ans in cases:
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "targets": torch.tensor([ans], device=dev)}
        try:
            sc = tracer.batch_trace(batch)[0]
            out[key] = np.asarray(sc[0].float().cpu().numpy())
        except Exception as e:  # noqa: BLE001
            print(f"  DPA {key} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
    del nm, backend, tracer; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--M", type=int, default=1000); ap.add_argument("--K", type=int, default=32); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--task", default="tight", choices=["tight", "sentiment"]); ap.add_argument("--tcap", type=int, default=120)
    ap.add_argument("--no-dpa", action="store_true")
    args = ap.parse_args()
    dev = args.device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lxt.models.llama_PE import LlamaForCausalLM, attnlrp

    tok = AutoTokenizer.from_pretrained(LLAMA, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    cases = []
    if args.task == "sentiment":
        from research_insertion_datasets import load_cases
        for ds in ["imdb", "sst2"]:
            for k, (text, prompt, gold) in enumerate(load_cases(ds, args.per, tok, model, dev, llq)):
                T = tok(prompt, return_tensors="pt").input_ids.shape[1]
                if T <= args.tcap:
                    cases.append((f"{ds}_{k}", ds, prompt, gold))
        print(f"  sentiment: {len(cases)}", flush=True)
    else:
        for tname, fn in [("kv", gen_kv), ("arith", gen_arith)]:
            rng = np.random.default_rng(0); got = 0; tries = 0
            while got < args.per and tries < args.per * 50:
                tries += 1
                prompt, ans_str = fn(rng); gold = tok(ans_str, add_special_tokens=False).input_ids[0]
                ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == gold:
                        cases.append((f"{tname}_{got}", tname, prompt, gold)); got += 1
            print(f"  {tname}: {got}", flush=True)

    # ---- DPA (disabled by default: official repo env-incompatible, nnsight .source unavailable) ----
    if args.no_dpa:
        dpa_cache = {}; print("DPA skipped", flush=True)
    else:
        print("computing DPA scores (nnsight)...", flush=True)
        dpa_cache = dpa_scores(cases, tok, dev)
        print(f"DPA: {len(dpa_cache)}/{len(cases)}", flush=True)

    # ---- official PA-LRP / AttnLRP (their llama_PE) ----
    mlrp = LlamaForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True).to(dev).eval()
    mlrp.gradient_checkpointing_enable(); attnlrp.register(mlrp)
    L = mlrp.config.num_hidden_layers

    def lrp_rel(prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        emb = mlrp.get_input_embeddings()(ids)
        pos_ids = torch.arange(0.0, ids.shape[1], device=dev, requires_grad=True, dtype=torch.float32).reshape(1, -1)
        pos_emb = [mlrp.get_input_pos_embeddings()(emb, pos_ids) for _ in range(L)]
        pos_emb = [(x[0].requires_grad_(), x[1].requires_grad_()) for x in pos_emb]
        logits = mlrp(inputs_embeds=emb.requires_grad_(), position_embeddings=pos_emb, use_cache=False)["logits"]
        tgt = logits[0, -1, ans]; tgt.backward(tgt)
        attn = emb.grad.float().sum(-1).cpu()[0]; attn = attn / attn.abs().max()
        acc = torch.zeros_like(attn)
        for pe in pos_emb:
            for i in range(2):
                cr = torch.matmul(pe[i].grad.abs(), pe[i].transpose(-1, -2).abs()).detach().float().sum(-1).cpu()[0]
                acc = acc + cr.abs()
        acc = acc / acc.abs().max(); pa = attn + acc; pa = pa / pa.abs().max()
        return attn.numpy(), pa.numpy()

    lrp_cache = {}
    for key, tname, prompt, ans in cases:
        try:
            lrp_cache[key] = lrp_rel(prompt, ans)
        except Exception as e:  # noqa: BLE001
            print(f"  LRP {key} FAIL {str(e)[:60]}", flush=True)
        mlrp.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    del mlrp; torch.cuda.empty_cache()
    print(f"LRP: {len(lrp_cache)}/{len(cases)}", flush=True)

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0].float(), -1)[ans]); base = float(F.softmax(llq(emb * 0 + gmean)[0].float(), -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def sig(Z):
            e = (emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])).to(emb.dtype)
            return (F.softmax(llq(e).float(), -1)[:, ans] - base) / den

        order_of = lambda sc: [i for i in np.argsort(-np.asarray(sc)) if i in nz]

        @torch.no_grad()
        def keep_rec(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])

        @torch.no_grad()
        def del_rec(order, f):
            k = int(round(f * len(order))); m = torch.ones(T, device=dev)
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 0.0
            return float(sig(m[None])[0])

        ins_auc = lambda sc: float(np.trapz([keep_rec(order_of(sc), f) for f in FR], FR) / FR[-1])
        del_auc = lambda sc: float(np.trapz([del_rec(order_of(sc), f) for f in FR], FR) / FR[-1])

        eg = emb.clone().requires_grad_(True)
        try:
            out = model(inputs_embeds=eg, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=eg)
        out.logits[0, -1, ans].backward()
        g = eg.grad[0].abs().float().sum(-1).cpu().numpy()
        kk = min(args.K, len(nz)); cand = [i for i in np.argsort(-g) if i in nz][:kk]; candset = set(cand); cand_t = torch.as_tensor(cand, device=dev)

        def banzhaf(mode):
            Zs = []; Rs = []; j = 0
            while j < args.M:
                nb = min(args.bs, args.M - j); zb = []
                for _ in range(nb):
                    if mode == "guided":
                        z = torch.ones(T, device=dev); pf = float(torch.rand(1, generator=gen, device=dev).item())
                        z[cand_t] = (torch.rand(kk, generator=gen, device=dev) < pf).float()
                    else:
                        lo = 0.0 if T < 100 else 0.5; pf = lo + (1 - lo) * float(torch.rand(1, generator=gen, device=dev).item())
                        z = (torch.rand(T, generator=gen, device=dev) < pf).float()
                    z[0] = 1.0; z[readout] = 1.0; zb.append(z)
                Z = torch.stack(zb); Rs.append(sig(Z)); Zs.append(Z); j += nb
            Z = torch.cat(Zs); R = torch.cat(Rs); n1 = Z.sum(0)
            return ((Z * R[:, None]).sum(0) / n1.clamp(min=1) - ((1 - Z) * R[:, None]).sum(0) / (Z.shape[0] - n1).clamp(min=1)).cpu().numpy()

        gg_m = banzhaf("guided"); gg = np.full(T, -1e9, np.float64)
        for i in cand:
            gg[i] = gg_m[i]
        for i in nz:
            if i not in candset:
                gg[i] = -1e6 + g[i] * 1e-3
        ad = banzhaf("adapt")
        scores = {"gg": gg, "gg_loc": closure(gg), "ad": ad, "ad_loc": closure(ad)}
        part = {nm: sum(keep_rec(order_of(sc), f) for f in SEL_FR) for nm, sc in scores.items()}
        sel = max(part, key=lambda nm: part[nm]); SELECT = scores[sel]

        @torch.no_grad()
        def greedy_del():
            km = np.ones(T, np.float32); order = []; rem = list(nz)
            while rem:
                masks = []
                for i in rem:
                    m = km.copy(); m[i] = 0.0; masks.append(m)
                rr = sig(torch.tensor(np.stack(masks), device=dev)).cpu().numpy(); jb = int(np.argmin(rr))
                km[rem[jb]] = 0.0; order.append(rem[jb]); rem.pop(jb)
            return order

        gdel = greedy_del() if T <= 60 else None
        attn_r, pa_r = lrp_cache.get(key, (None, None))
        dpa_r = dpa_cache.get(key)
        out = {"T": T, "sel": sel}
        out["ins"] = {"FRI_SELECT": ins_auc(SELECT),
                      "DPA": ins_auc(dpa_r) if dpa_r is not None else None,
                      "AttnLRP": ins_auc(attn_r) if attn_r is not None else None,
                      "PA-LRP": ins_auc(pa_r) if pa_r is not None else None}
        out["del"] = {"greedy": float(np.trapz([del_rec(gdel, f) for f in FR], FR) / FR[-1]) if gdel is not None else None,
                      "DPA": del_auc(dpa_r) if dpa_r is not None else None,
                      "AttnLRP": del_auc(attn_r) if attn_r is not None else None,
                      "PA-LRP": del_auc(pa_r) if pa_r is not None else None}
        return out

    rows = []
    for key, tname, prompt, ans in cases:
        try:
            r = run(key, prompt, ans); r.update(key=key, t=tname); rows.append(r)
            i, d = r["ins"], r["del"]
            print(f"  {key:9s} T={r['T']:3d} | INS FRI {i['FRI_SELECT']:.3f}({r['sel']}) DPA {fmt(i['DPA'])} Attn {fmt(i['AttnLRP'])} PA {fmt(i['PA-LRP'])} "
                  f"| DEL greedy {fmt(d['greedy'])} DPA {fmt(d['DPA'])} Attn {fmt(d['AttnLRP'])} PA {fmt(d['PA-LRP'])}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"  {key} OOM", flush=True)

    print(f"\n=== FULL FRI (M={args.M}) vs PA-LRP/AttnLRP — TIGHT tasks (Llama-3.2-1B, recovery AUC) ===")
    am = lambda face, k: float(np.nanmean([r[face][k] for r in rows if r[face][k] is not None]))
    pct = lambda lst: (100.0 * float(np.mean(lst))) if lst else float("nan")
    lrp_ok = lambda r, face: r[face]["PA-LRP"] is not None and r[face]["AttnLRP"] is not None
    fl = pct([r["ins"]["FRI_SELECT"] > max(r["ins"]["PA-LRP"], r["ins"]["AttnLRP"]) for r in rows if lrp_ok(r, "ins")])
    fpa = pct([r["ins"]["FRI_SELECT"] > r["ins"]["PA-LRP"] for r in rows if r["ins"]["PA-LRP"] is not None])
    fd = pct([r["ins"]["FRI_SELECT"] > r["ins"]["DPA"] for r in rows if r["ins"]["DPA"] is not None])
    ng = [r for r in rows if r["del"]["greedy"] is not None]
    gl = pct([r["del"]["greedy"] < min(r["del"]["PA-LRP"], r["del"]["AttnLRP"]) for r in ng if lrp_ok(r, "del")])
    gd = pct([r["del"]["greedy"] < r["del"]["DPA"] for r in ng if r["del"]["DPA"] is not None])
    dpa_on = any(r["ins"]["DPA"] is not None for r in rows)
    print(f"  n={len(rows)}")
    print(f"  SUFFICIENCY (ins, HIGHER=better): FRI_SELECT {am('ins','FRI_SELECT'):.3f}  PA-LRP {am('ins','PA-LRP'):.3f}  AttnLRP {am('ins','AttnLRP'):.3f}" + (f"  DPA {am('ins','DPA'):.3f}" if dpa_on else ""))
    print(f"     -> FRI > BOTH LRP: {fl:.0f}% | FRI > PA-LRP: {fpa:.0f}%" + (f" | FRI > DPA: {fd:.0f}%" if dpa_on else ""))
    print(f"  NECESSITY  (del, LOWER=better):   greedy {am('del','greedy'):.3f}(n={len(ng)})  PA-LRP {am('del','PA-LRP'):.3f}  AttnLRP {am('del','AttnLRP'):.3f}" + (f"  DPA {am('del','DPA'):.3f}" if dpa_on else ""))
    print(f"     -> greedy < BOTH LRP: {gl:.0f}%" + (f" | greedy < DPA: {gd:.0f}%" if dpa_on else ""))
    json.dump(rows, open(os.path.join(REPO, "outputs", "llm_fri_vs_palrp.json"), "w"), indent=2)


def fmt(v):
    return " NA  " if v is None else f"{v:.3f}"


if __name__ == "__main__":
    main()
