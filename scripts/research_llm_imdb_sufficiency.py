#!/usr/bin/env python3
"""Does FRI's SUFFICIENCY (insertion) finally WIN on a DISTRIBUTED-EVIDENCE LLM task?

Motivation: our copy/retrieval tasks let a SINGLE token carry the answer (favoring
1-backward AttnLRP). Sentiment is spread over MANY words, so FRI's random-budget
soft-insertion should find the distributed sufficient set and win insertion. The
AttnLRP paper used IMDB for LLaMA-2; we mirror that with Qwen2.5-1.5B-Instruct.

Setup: GLOBAL token-mean baseline (model.get_input_embeddings().weight.mean(0)),
readout (final token) frozen, cuda:0 (SHARED with a co-tenant job -> per-method OOM
guards + AttnLRP in an isolated bf16 subprocess; skips are documented).

Data: stanfordnlp/imdb test (fallback: hand-written distributed-evidence reviews).
Prompt: "Review: {text}\nQuestion: Is the sentiment of this review positive or
negative? Answer: The sentiment is" -> target prob of ' positive'/' negative'
(single tokens). Keep only argmax-correct; bin by length short/medium/long.

Methods: FRI_annealed, attnlrp, gradient, greedy_oracle, chunked_cond.
Metrics: INSERTION AUC (^ sufficiency -- THE FOCUS) and DELETION AUC (v necessity).

KEY QUESTION: is FRI the INSERTION (sufficiency) winner on IMDB, across all length
bins -- unlike the copy/retrieval tasks? Reported as a by-bin insertion table
(mean+-std, n per bin) + a verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import research_llm_necessary_long_context as L  # noqa: E402


PROMPT_TMPL = ("Review: {text}\nQuestion: Is the sentiment of this review positive "
               "or negative? Answer: The sentiment is")

# Length bins by token count T (inclusive lo, exclusive hi).
BINS = [("short", 0, 90), ("medium", 90, 220), ("long", 220, 100000)]


# Hand-written distributed-evidence fallback reviews (clear pos/neg, multi-word
# sentiment spread across the text), used only if IMDB cannot be loaded offline.
FALLBACK_REVIEWS = [
    (1, "This film was an absolute delight from start to finish. The acting was superb, "
        "the cinematography breathtaking, and the score moved me to tears. A triumph."),
    (0, "A complete waste of time. The plot was incoherent, the dialogue wooden, and the "
        "pacing dreadful. I was bored, annoyed, and ultimately disappointed throughout."),
    (1, "Wonderful, heartfelt, and beautifully crafted. Every performance shines and the "
        "story stayed with me for days. Easily one of the best films I have seen."),
    (0, "Terrible. Cheap effects, a lazy script, and performances that felt phoned in. "
        "Nothing worked. I cannot recommend this dull, forgettable mess to anyone."),
    (1, "Charming and clever, with warm humor and genuine emotional depth. The cast is "
        "wonderful and the direction confident. I left the theater grinning happily."),
    (0, "Dreadful and tedious. The jokes fell flat, the characters were unlikable, and "
        "the ending was both predictable and unsatisfying. A frustrating, joyless slog."),
    (1, "An exhilarating, gorgeous, and deeply moving masterpiece. Brilliant writing, "
        "stunning visuals, and a powerful ending. I was captivated the entire time."),
    (0, "Boring, sloppy, and painfully unfunny. Weak acting, a nonsensical plot, and "
        "ugly editing. I regretted watching it and would warn others to stay away."),
    (1, "Delightful from beginning to end: witty, tender, and visually sumptuous. The "
        "leads have wonderful chemistry and the whole thing is a joyous celebration."),
    (0, "Awful in every respect. Flat jokes, lifeless performances, and a meandering, "
        "pointless story. I found it tiresome, irritating, and thoroughly unpleasant."),
]


def load_imdb(n_cand, max_chars, seed=0):
    """Return list of (label, text), LENGTH-DIVERSE so short/medium/long bins all
    fill. Balanced classes. Falls back to hand-written reviews if offline."""
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("stanfordnlp/imdb", split="test")
        except Exception:
            ds = load_dataset("imdb", split="test")
        n = len(ds)
        rng = np.random.default_rng(seed)
        # sample a big balanced pool, then bucket by CHAR length (proxy for token T)
        per = min(n // 2, 4000)
        neg_pool = rng.choice(np.arange(0, n // 2), size=per, replace=False)
        pos_pool = rng.choice(np.arange(n // 2, n), size=per, replace=False)
        cands = []
        for i in np.concatenate([neg_pool, pos_pool]):
            ex = ds[int(i)]
            cands.append((int(ex["label"]), ex["text"][:max_chars], len(ex["text"])))
        # char-length buckets ~ token bins: short<300 chars, medium 300-900, long>900
        buckets = {"s": [], "m": [], "l": []}
        for lab, txt, clen in cands:
            b = "s" if clen < 300 else ("m" if clen < 900 else "l")
            buckets[b].append((lab, txt))
        for b in buckets:
            rng.shuffle(buckets[b])
        # interleave so the candidate stream spans all lengths (short first to ensure fill)
        out = []
        k = max(len(v) for v in buckets.values())
        for j in range(k):
            for b in ("s", "m", "l"):
                if j < len(buckets[b]):
                    out.append(buckets[b][j])
        return out[: n_cand], "stanfordnlp/imdb"
    except Exception as e:  # noqa: BLE001
        print(f"  (IMDB load failed: {type(e).__name__}: {e}; using fallback reviews)")
        return list(FALLBACK_REVIEWS), "fallback_handwritten"


# --- batched AttnLRP over all kept prompts (one bf16 subprocess, model loaded once) ---
ATTNLRP_BATCH_WORKER = r'''
import sys, json
import numpy as np
import torch
repo, path, device = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.loads(sys.argv[4])   # list of {key, text}
out_path = sys.argv[5]
sys.path.insert(0, repo)
from transformers import AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2
from lxt.efficient import monkey_patch
monkey_patch(modeling_qwen2, verbose=False)
tok = AutoTokenizer.from_pretrained(path)
model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
model.eval()
for pr in model.parameters():
    pr.requires_grad = False
out = {}
for p in prompts:
    ids = tok(p["text"], return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    with torch.no_grad():
        ans_id = int(model(ids, use_cache=False).logits[0, -1].argmax())
    emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=emb, use_cache=False).logits
    model.zero_grad(set_to_none=True)
    logits[0, -1, ans_id].backward()
    rel = (emb.grad * emb).float().sum(-1).detach().cpu()[0].numpy()
    out[p["key"]] = rel.astype(np.float32).tolist()
    del emb, logits
    torch.cuda.empty_cache()
json.dump(out, open(out_path, "w"))
print("ATTNLRP_BATCH_OK")
'''


def precompute_attnlrp(prompts, path, device):
    import subprocess
    import tempfile
    payload = [{"key": k, "text": t} for k, t in prompts]
    with tempfile.TemporaryDirectory() as td:
        worker = os.path.join(td, "w.py")
        outp = os.path.join(td, "rel.json")
        open(worker, "w").write(ATTNLRP_BATCH_WORKER)
        cmd = [sys.executable, worker, L.REPO_DIR, path, str(device), json.dumps(payload), outp]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=L.REPO_DIR)
        if "ATTNLRP_BATCH_OK" not in res.stdout or not os.path.exists(outp):
            print(f"  (batched AttnLRP failed rc={res.returncode}; stderr: {res.stderr[-500:]})")
            return {}
        data = json.load(open(outp))
    return {k: np.asarray(v, np.float32) for k, v in data.items()}


def run_review(key, label, text, model, tok, args, attn_cache):
    prompt = PROMPT_TMPL.format(text=text)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)
    T = int(ids.shape[1])
    keepset = frozenset({T - 1})
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]
    gold_id = pos_id if label == 1 else neg_id

    with torch.no_grad():
        full_logits = model(ids).logits[0, -1]
    pred_id = int(full_logits.argmax())
    if pred_id != gold_id:
        return None  # keep only argmax-correct
    ans_id = gold_id
    ans_prob = float(torch.softmax(full_logits, dim=-1)[ans_id])

    counter = L.Counter()
    emb, mean_emb, _, prob_for = L.build_engine(model, ids, ans_id, args.device, counter)
    rec, p_full, p_base = L.make_recovery(prob_for, T)
    full_rec = rec(np.ones(T, np.float32))

    c0 = counter.n
    single_occ = L.single_occlusion(rec, T, full_rec, keepset)
    cost_prior = counter.n - c0

    c0 = counter.n
    nec_order, greedy_full = L.greedy_conditional(rec, T, full_rec, keepset, single_occ,
                                                  stop=0.1, max_steps=args.greedy_max_steps)
    cost_greedy = counter.n - c0
    nec_set = list(nec_order)

    c0 = counter.n
    chunk_full = L.chunked_conditional(rec, T, full_rec, keepset, single_occ,
                                       M=args.chunk_M, R=args.chunk_R)
    cost_chunk = counter.n - c0

    # gradient (guard OOM)
    try:
        grad_s = L.gradient_scores(model, emb, mean_emb, ans_id, T, args.device)
        grad_order = np.array([i for i in np.argsort(-grad_s) if i not in keepset])
        grad_ok = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        grad_order, grad_ok = None, False

    # FRI_annealed (guard OOM)
    c0 = counter.n
    try:
        fri_a = L.fri_scores(model, emb, mean_emb, ans_id, T, args.device, p_full, p_base,
                             counter, steps=args.fri_steps, restarts=2, keepset=keepset,
                             budget_mode="annealed")
        fri_order = np.array([i for i in np.argsort(-fri_a) if i not in keepset])
        fri_ok = True
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        fri_order, fri_ok = None, False
    cost_fri = counter.n - c0

    attn_s = attn_cache.get(key)
    attn_ok = attn_s is not None and len(attn_s) == T
    attn_order = (np.array([i for i in np.argsort(-attn_s) if i not in keepset])
                  if attn_ok else None)

    n_content = T - len(keepset)
    budget = max(1, int(round(L.BUDGET_FRAC * n_content)))
    fixed = min(L.FIXED_COUNT, n_content)

    methods = {
        "greedy_oracle": (greedy_full, cost_prior + cost_greedy),
        "chunked_cond": (chunk_full, cost_prior + cost_chunk),
    }
    if fri_ok:
        methods["FRI_annealed"] = (fri_order, cost_fri)
    if grad_ok:
        methods["gradient"] = (grad_order, 1)
    if attn_ok:
        methods["attnlrp"] = (attn_order, 1)

    rows = {}
    for nm, (order, fwd) in methods.items():
        rows[nm] = dict(
            insertion_auc=L.insertion_auc(rec, order, T, keepset, budget),
            deletion_auc=L.deletion_auc(rec, order, T, keepset, budget),
            deletion_auc_fixed=L.deletion_auc(rec, order, T, keepset, fixed),
            attribution_forwards=int(fwd),
        )
    return dict(key=key, T=T, label=int(label), answer_prob=ans_prob,
                p_full=p_full, p_base=p_base, n_skipped_methods=5 - len(methods),
                methods=rows)


def bin_of(T):
    for name, lo, hi in BINS:
        if lo <= T < hi:
            return name
    return BINS[-1][0]


def aggregate(results):
    """mean+-std per (bin, method) for each metric + overall."""
    method_order = ["greedy_oracle", "chunked_cond", "FRI_annealed", "gradient", "attnlrp"]
    metrics = ["insertion_auc", "deletion_auc", "deletion_auc_fixed"]
    agg = {}
    groups = {b[0]: [] for b in BINS}
    groups["ALL"] = []
    for r in results:
        groups[bin_of(r["T"])].append(r)
        groups["ALL"].append(r)
    for g, rs in groups.items():
        agg[g] = dict(n=len(rs), T_mean=float(np.mean([r["T"] for r in rs])) if rs else float("nan"),
                      methods={})
        for nm in method_order:
            vals = {m: [r["methods"][nm][m] for r in rs if nm in r["methods"]] for m in metrics}
            n_m = len(vals["insertion_auc"])
            agg[g]["methods"][nm] = {
                "n": n_m,
                **{f"{m}_mean": (float(np.mean(vals[m])) if vals[m] else float("nan")) for m in metrics},
                **{f"{m}_std": (float(np.std(vals[m])) if vals[m] else float("nan")) for m in metrics},
            }
    return agg, method_order


def fmt_table(agg, method_order, metric, label, arrow):
    lines = [f"\n{label}  ({arrow})  [mean +/- std (n)]"]
    bins = [b[0] for b in BINS] + ["ALL"]
    hdr = f"  {'method':14s} " + " ".join(f"{b+'(T~'+str(int(agg[b]['T_mean']))+')':>16s}"
                                          if agg[b]["n"] else f"{b:>16s}" for b in bins)
    lines.append(hdr)
    for nm in method_order:
        cells = []
        for b in bins:
            m = agg[b]["methods"][nm]
            if m["n"] == 0 or m[f"{metric}_mean"] != m[f"{metric}_mean"]:
                cells.append(f"{'-':>16s}")
            else:
                cells.append(f"{m[f'{metric}_mean']:+.3f}+-{m[f'{metric}_std']:.2f}({m['n']})".rjust(16))
        lines.append(f"  {nm:14s} " + " ".join(cells))
    return "\n".join(lines)


def build_verdict(agg, method_order):
    lines = ["\n" + "=" * 96, "VERDICT", "=" * 96]
    bins = [b[0] for b in BINS] + ["ALL"]

    # INSERTION winner per bin
    lines.append("\n[SUFFICIENCY/insertion -- THE FOCUS] winner per length bin:")
    fri_wins = 0
    fri_bins = 0
    for b in bins:
        present = {nm: agg[b]["methods"][nm]["insertion_auc_mean"]
                   for nm in method_order
                   if agg[b]["methods"][nm]["n"] > 0
                   and agg[b]["methods"][nm]["insertion_auc_mean"] == agg[b]["methods"][nm]["insertion_auc_mean"]}
        if not present:
            continue
        win = max(present, key=present.get)
        fri_val = present.get("FRI_annealed", float("nan"))
        if "FRI_annealed" in present:
            fri_bins += 1
            if win == "FRI_annealed":
                fri_wins += 1
        lines.append(f"  {b:7s}(n={agg[b]['n']:2d}): insertion winner = {win} "
                     f"({present[win]:+.3f}); FRI={fri_val:+.3f}; "
                     f"AttnLRP={present.get('attnlrp', float('nan')):+.3f}")
    lines.append(f"  => FRI wins INSERTION in {fri_wins}/{fri_bins} bins where it ran.")

    # DELETION winner per bin (use del_fixed = discriminative necessity)
    lines.append("\n[NECESSITY/deletion] del_fixed winner per length bin (lower=better):")
    cond = ("greedy_oracle", "chunked_cond")
    cond_wins = 0
    for b in bins:
        present = {nm: agg[b]["methods"][nm]["deletion_auc_fixed_mean"]
                   for nm in method_order
                   if agg[b]["methods"][nm]["n"] > 0
                   and agg[b]["methods"][nm]["deletion_auc_fixed_mean"] == agg[b]["methods"][nm]["deletion_auc_fixed_mean"]}
        if not present:
            continue
        win = min(present, key=present.get)
        if win in cond:
            cond_wins += 1
        lines.append(f"  {b:7s}(n={agg[b]['n']:2d}): del_fixed winner = {win} ({present[win]:+.3f})")
    lines.append(f"  => conditional (greedy/chunked) wins del_fixed in {cond_wins}/{len(bins)} bins.")

    # overall headline
    o = agg["ALL"]["methods"]
    fri_ins = o["FRI_annealed"]["insertion_auc_mean"]
    attn_ins = o["attnlrp"]["insertion_auc_mean"]
    grad_ins = o["gradient"]["insertion_auc_mean"]
    gre_ins = o["greedy_oracle"]["insertion_auc_mean"]
    lines.append(f"\n[OVERALL insertion] FRI={fri_ins:+.3f}  AttnLRP={attn_ins:+.3f}  "
                 f"gradient={grad_ins:+.3f}  greedy={gre_ins:+.3f}")
    fri_top_overall = all(
        (fri_ins >= o[nm]["insertion_auc_mean"] - 1e-9)
        for nm in method_order
        if o[nm]["n"] > 0 and o[nm]["insertion_auc_mean"] == o[nm]["insertion_auc_mean"]
    )
    lines.append("VERDICT: FRI-sufficiency on distributed-evidence IMDB -> "
                 + ("WORKS: FRI is the TOP insertion method overall and the asymmetry "
                    "(FRI=sufficiency) is restored on distributed evidence."
                    if fri_top_overall and fri_wins >= max(1, fri_bins // 2) else
                    f"DOES NOT clearly win: FRI insertion overall {fri_ins:+.3f} is not the top "
                    f"(AttnLRP {attn_ins:+.3f}); FRI wins {fri_wins}/{fri_bins} bins. The "
                    f"distributed-evidence hypothesis is {'partly' if fri_wins>0 else 'not'} supported."))
    return lines, dict(fri_insertion_bin_wins=fri_wins, fri_bins=fri_bins,
                       overall_insertion={"FRI_annealed": fri_ins, "attnlrp": attn_ins,
                                          "gradient": grad_ins, "greedy_oracle": gre_ins})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-target", type=int, default=30, help="approx reviews to KEEP (argmax-correct)")
    ap.add_argument("--sample-mult", type=int, default=4, help="oversample factor before filtering")
    ap.add_argument("--max-chars", type=int, default=1400)
    ap.add_argument("--greedy-max-steps", type=int, default=10)
    ap.add_argument("--chunk-M", type=int, default=80)
    ap.add_argument("--chunk-R", type=int, default=6)
    ap.add_argument("--fri-steps", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 96)
    print(f"LLM IMDB SUFFICIENCY (distributed evidence) | model={args.model}")
    print("baseline=GLOBAL vocab-mean; readout frozen; cuda:0 (shared). "
          "Focus: does FRI WIN insertion?")
    print("=" * 96)

    raw, source = load_imdb(args.n_target * args.sample_mult * 3, args.max_chars)
    print(f"data source: {source}; candidate reviews: {len(raw)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = (AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
             .to(args.device).eval())
    for pp in model.parameters():
        pp.requires_grad = False

    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]

    # First pass: filter to argmax-correct with a PER-BIN quota so short/medium/long
    # all fill (IMDB skews long; without a quota the short bin stays empty).
    per_bin_target = max(1, args.n_target // len(BINS))
    binc = {b[0]: 0 for b in BINS}
    kept = []  # (key, label, text, T)
    for j, (label, text) in enumerate(raw):
        prompt = PROMPT_TMPL.format(text=text)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(args.device)
        T = int(ids.shape[1])
        b = bin_of(T)
        if binc[b] >= per_bin_target + 4:  # allow a little overflow per bin
            continue
        with torch.no_grad():
            pred = int(model(ids).logits[0, -1].argmax())
        gold = pos_id if label == 1 else neg_id
        if pred == gold:
            kept.append((f"r{j}", label, text, T))
            binc[b] += 1
        if len(kept) >= args.n_target and all(v >= 1 for v in binc.values()):
            break
    print(f"kept argmax-correct: {len(kept)} | per-bin: {binc}")

    # Precompute AttnLRP in a bf16 subprocess. Move the main model to CPU first so
    # the subprocess gets the full (shared) GPU -- otherwise both fp32+bf16 models
    # would be co-resident and OOM. Move it back afterwards.
    print("offloading main model to CPU for AttnLRP precompute ...")
    model.to("cpu")
    torch.cuda.empty_cache()
    print("precomputing AttnLRP (batched bf16 subprocess) ...")
    attn_cache = precompute_attnlrp(
        [(k, PROMPT_TMPL.format(text=t)) for k, _, t, _ in kept], args.model, args.device)
    print(f"  AttnLRP available for {len(attn_cache)}/{len(kept)} reviews")
    model.to(args.device)
    torch.cuda.empty_cache()

    # Main pass: run all methods per kept review.
    results = []
    for k, label, text, T in kept:
        r = run_review(k, label, text, model, tok, args, attn_cache)
        if r is not None:
            results.append(r)
            sk = f" (skipped {r['n_skipped_methods']} methods)" if r["n_skipped_methods"] else ""
            print(f"  {k} T={r['T']:4d} lab={'pos' if label==1 else 'neg'} "
                  f"p={r['answer_prob']:.2f}{sk}", flush=True)

    agg, method_order = aggregate(results)
    ins_tbl = fmt_table(agg, method_order, "insertion_auc", "INSERTION_AUC", "HIGHER=sufficiency")
    del_tbl = fmt_table(agg, method_order, "deletion_auc", "DELETION_AUC(20%)", "LOWER=necessity")
    delf_tbl = fmt_table(agg, method_order, "deletion_auc_fixed", "DELETION_AUC(fixed)", "LOWER=necessity")
    verdict, vsum = build_verdict(agg, method_order)

    print("\n" + "=" * 96)
    print("BY-LENGTH-BIN AUC TABLES")
    print("=" * 96)
    print(ins_tbl)
    print(del_tbl)
    print(delf_tbl)
    for line in verdict:
        print(line)

    summary = dict(model=args.model, data_source=source, baseline="global_vocab_mean",
                   prompt_template=PROMPT_TMPL, bins=[list(b) for b in BINS],
                   n_kept=len(results), per_bin_counts=binc,
                   aggregate=agg, results=results, verdict=verdict, verdict_summary=vsum)
    out = args.out or os.path.join(os.path.dirname(__file__), "..", "outputs",
                                   "llm_imdb_sufficiency.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out.replace(".json", ".txt"), "w") as f:
        f.write(format_txt(summary, ins_tbl, del_tbl, delf_tbl, verdict))
    print(f"\nsaved JSON -> {out}")
    print(f"saved TXT  -> {out.replace('.json', '.txt')}")


def format_txt(s, ins_tbl, del_tbl, delf_tbl, verdict):
    out = [f"LLM IMDB SUFFICIENCY (distributed evidence) | model={s['model']}",
           f"data={s['data_source']}; baseline={s['baseline']}; readout frozen; cuda:0",
           f"prompt: {s['prompt_template']!r}",
           f"n_kept(argmax-correct)={s['n_kept']}; per-bin={s['per_bin_counts']}",
           "bins: " + ", ".join(f"{b[0]}[{b[1]},{b[2]})" for b in s["bins"]),
           "=" * 84, ins_tbl, del_tbl, delf_tbl, ""]
    out += verdict
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
