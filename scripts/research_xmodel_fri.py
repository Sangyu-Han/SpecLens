#!/usr/bin/env python3
"""Cross-model validation: does FRI (cooperative restricted-Banzhaf + closure + self-selection) BEAT
AttnLRP on TIGHT cooperative cases (KV-retrieval + arithmetic) on NEW LLMs, at higher n?
Models: Llama-3.2-1B + Llama-3.1-8B-Instruct (lxt supports llama; gemma2 NOT supported). Uses a
solid M (the config that dominated AttnLRP on Qwen tight cases), not the ~100-cost cheap version."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from research_hard_cases import gen_arith, gen_kv  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FR = [0.0, .02, .05, .1, .15, .2, .3, .5]
SEL_FR = [.05, .1, .2]

ATTNLRP_WORKER = '''
import sys, json
import numpy as np, torch
repo, path, device, payload_json, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
payload = json.loads(payload_json); sys.path.insert(0, repo)
import importlib
from transformers import AutoTokenizer, AutoConfig
from lxt.efficient import monkey_patch
cfg = AutoConfig.from_pretrained(path); arch = (cfg.architectures or [""])[0]
if "Qwen2" in arch:
    mod = importlib.import_module("transformers.models.qwen2.modeling_qwen2"); cls = mod.Qwen2ForCausalLM
elif "Llama" in arch:
    mod = importlib.import_module("transformers.models.llama.modeling_llama"); cls = mod.LlamaForCausalLM
else:
    raise ValueError("unsupported arch " + arch)
monkey_patch(mod, verbose=False)
tok = AutoTokenizer.from_pretrained(path)
model = cls.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
for pr in model.parameters():
    pr.requires_grad = False
out = {}
for p in payload:
    ids = tok(p["text"], return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    with torch.no_grad():
        ans_id = int(model(ids, use_cache=False).logits[0, -1].argmax())
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb, use_cache=False).logits
    model.zero_grad(set_to_none=True)
    logits[0, -1, ans_id].backward()
    rel = (emb.grad * emb).float().sum(-1).detach().cpu()[0].numpy()
    out[p["key"]] = rel.astype(np.float32).tolist()
    del emb, logits; torch.cuda.empty_cache()
json.dump(out, open(out_path, "w")); print("ATTNLRP_BATCH_OK")
'''


def precompute_attnlrp(prompts, path, device):
    payload = [{"key": k, "text": t} for k, t in prompts]
    with tempfile.TemporaryDirectory() as td:
        worker = os.path.join(td, "w.py"); outp = os.path.join(td, "rel.json")
        open(worker, "w").write(ATTNLRP_WORKER)
        cmd = [sys.executable, worker, str(REPO), path, str(device), json.dumps(payload), outp]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=str(REPO))
        if "ATTNLRP_BATCH_OK" not in res.stdout or not os.path.exists(outp):
            print(f"  (AttnLRP failed rc={res.returncode}; stderr {res.stderr[-400:]})", flush=True)
            return {}
        return {k: np.asarray(v, np.float32) for k, v in json.load(open(outp)).items()}


def closure(score, radius=1, decay=0.95):
    s = np.asarray(score, np.float64).copy()
    for _ in range(radius):
        left = np.full_like(s, -1e18); right = np.full_like(s, -1e18)
        left[1:] = s[:-1] * decay; right[:-1] = s[1:] * decay
        s = np.maximum(s, np.maximum(left, right))
    return s


def run_model(model_path, dtype, per, M, K, bs, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    kw = {"torch_dtype": dtype}
    if "gemma" in model_path.lower():
        kw["attn_implementation"] = "eager"                  # gemma2 soft-capping needs eager
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:  # noqa: BLE001
        pass
    gmean = model.get_input_embeddings().weight.mean(0).view(1, 1, -1).detach()
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    def llq(e):
        with torch.no_grad():
            try:
                return model(inputs_embeds=e, logits_to_keep=1).logits[:, -1]
            except TypeError:
                return model(inputs_embeds=e).logits[:, -1]

    cases = []
    for tname, fn in [("kv", gen_kv), ("arith", gen_arith)]:
        rng = np.random.default_rng(0); got = 0; tries = 0
        while got < per and tries < per * 40:
            tries += 1
            prompt, ans_str = fn(rng); gold = tok(ans_str, add_special_tokens=False).input_ids[0]
            ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                if int(llq(model.get_input_embeddings()(ids))[0].argmax()) == gold:
                    cases.append((f"{tname}_{got}", tname, prompt, gold)); got += 1
        print(f"    {tname}: {got}", flush=True)
    model.to("cpu"); torch.cuda.empty_cache()
    attn_cache = precompute_attnlrp([(k, pr) for k, _, pr, _ in cases], model_path, dev)
    model.to(dev); torch.cuda.empty_cache()

    def run(key, prompt, ans):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        T = int(ids.shape[1]); readout = T - 1; nz = [i for i in range(T) if i not in (0, readout)]
        emb = model.get_input_embeddings()(ids).detach()
        with torch.no_grad():
            full = float(F.softmax(llq(emb)[0].float(), -1)[ans]); base = float(F.softmax(llq(emb * 0 + gmean)[0].float(), -1)[ans])
        den = max(full - base, 1e-6)

        @torch.no_grad()
        def sig(Z):
            e = (emb * Z[:, :, None] + gmean * (1 - Z[:, :, None])).to(emb.dtype)  # bf16-safe
            return (F.softmax(llq(e).float(), -1)[:, ans] - base) / den

        @torch.no_grad()
        def keep_rec(order, f):
            k = int(round(f * len(order))); m = torch.zeros(T, device=dev); m[0] = 1.0; m[readout] = 1.0
            if k:
                m[torch.as_tensor(order[:k], device=dev)] = 1.0
            return float(sig(m[None])[0])
        order_of = lambda sc: [i for i in np.argsort(-sc) if i in nz]
        ins_auc = lambda sc: float(np.trapz([keep_rec(order_of(sc), f) for f in FR], FR) / FR[-1])

        eg = emb.clone().requires_grad_(True)
        try:
            out = model(inputs_embeds=eg, logits_to_keep=1)
        except TypeError:
            out = model(inputs_embeds=eg)
        out.logits[0, -1, ans].backward()
        g = eg.grad[0].abs().float().sum(-1).cpu().numpy()
        kk = min(K, len(nz)); cand = [i for i in np.argsort(-g) if i in nz][:kk]; candset = set(cand); cand_t = torch.as_tensor(cand, device=dev)

        def banzhaf(mode):
            Zs = []; Rs = []; j = 0
            while j < M:
                nb = min(bs, M - j); zb = []
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
        sel = max(part, key=lambda nm: part[nm])
        att = attn_cache.get(key)
        occ = np.zeros(T, np.float32)
        with torch.no_grad():
            for s0 in range(0, len(nz), bs):
                idx = nz[s0:s0 + bs]; Mb = torch.ones(len(idx), T, device=dev)
                for r0, i in enumerate(idx):
                    Mb[r0, i] = 0.0
                rr = sig(Mb).cpu().numpy()
                for r0, i in enumerate(idx):
                    occ[i] = -rr[r0]
        return dict(T=T, gg=ins_auc(gg), gg_loc=ins_auc(scores["gg_loc"]), ad=ins_auc(ad),
                    SELECT=ins_auc(scores[sel]), _sel=sel,
                    attn=ins_auc(att) if att is not None and len(att) == T else float("nan"),
                    single_occ=ins_auc(occ))

    rows = []
    for key, tname, prompt, ans in cases:
        try:
            r = run(key, prompt, ans); r.update(key=key, t=tname); rows.append(r)
            print(f"    {key:9s} T={r['T']:3d} | gg={r['gg']:.3f} gg_loc={r['gg_loc']:.3f} SELECT={r['SELECT']:.3f}({r['_sel']}) | attn={r['attn']:.3f} occ={r['single_occ']:.3f}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"    {key} OOM", flush=True)
    del model; torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--M", type=int, default=1000)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--bs", type=int, default=12)
    args = ap.parse_args()
    MODELS = [("meta-llama/Meta-Llama-3-8B", torch.bfloat16, 12)]   # 3rd model: complete cache (Instruct was .incomplete)
    allres = {}
    for mp, dtype, per in MODELS:
        print(f"\n##### MODEL {mp} (per={per}, dtype={dtype}) #####", flush=True)
        try:
            allres[mp] = run_model(mp, dtype, per, args.M, args.K, args.bs, args.device)
        except Exception as e:  # noqa: BLE001
            import traceback; print(f"  MODEL FAIL {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}", flush=True)
    print("\n=== CROSS-MODEL: FRI(SELECT, M={}) vs AttnLRP vs single_occ (insertion, higher=better) ===".format(args.M))
    for mp, rows in allres.items():
        if not rows:
            continue
        for t in ["kv", "arith", "ALL"]:
            rs = [r for r in rows if t == "ALL" or r["t"] == t]
            if not rs:
                continue
            se = np.mean([r["SELECT"] for r in rs]); at = np.nanmean([r["attn"] for r in rs]); oc = np.mean([r["single_occ"] for r in rs])
            d = np.array([r["SELECT"] - r["attn"] for r in rs if r["attn"] == r["attn"]])
            sig = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-9) if len(d) else 0
            win = 100.0 * (d > 0).mean() if len(d) else 0
            print(f"  {mp.split('/')[-1]:24s} {t:5s} n={len(rs):2d} | SELECT={se:.3f} attn={at:.3f} occ={oc:.3f} | SEL-attn={d.mean() if len(d) else 0:+.3f} ({sig:+.1f}σ, win {win:.0f}%)")
    json.dump({k: v for k, v in allres.items()}, open(os.path.join(REPO, "outputs", "xmodel_fri.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
