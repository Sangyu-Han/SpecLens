#!/usr/bin/env python3
"""Large-n validation: fri_alt steps x ridge-gate (clip convention).

Arms per case (solves shared):
  s32, s16              : bare solves
  s32_g, s16_g          : + ridge gate (64 seeded field probes)
  s32_gsh, s16_gsh      : gate only if split-half beta self-consistency
                          passes (corr(beta_h1, beta_h2) >= tau; zero extra
                          forwards)
Case sets: --cases n100 (failure CSV) | general50 (random val, argmax target).
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = _ilu.spec_from_file_location("hs", REPO / "scripts/research_fri_halfstep.py")
hs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(hs)
ks, ns, q = hs.ks, hs.ns, hs.q
N = 196
CSV = REPO / "outputs/class_fri/softplus_failure_scan_n100.csv"


def ridge_beta_sh(runner, n_states=64, seed=42, ktemp=0.25, lam_scale=1e-3):
    """beta + split-half self-consistency r (zero extra forwards)."""
    masks, sides, margins = hs.seeded_field_states(runner, n_states, seed)
    with torch.no_grad():
        h = runner.base[0].detach().float()
        hn = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S = (hn @ hn.T).cpu().numpy().astype(np.float64)
    Kinv = np.linalg.inv(np.exp((S - 1.0) / ktemp) + 1e-3 * np.eye(N))

    def fit(rows):
        M = masks[rows].astype(np.float64)
        v = margins[rows].astype(np.float64)
        Mc = M - M.mean(0)
        vc = v - v.mean()
        lam = lam_scale * max(len(v), 1)
        return np.linalg.solve(Mc.T @ Mc + lam * Kinv, Mc.T @ vc)

    betas, halves = [], []
    for side in ("ins", "del"):
        rows = np.where(sides == side)[0]
        betas.append(ks.zscore(fit(rows)))
        halves.append((fit(rows[0::2]), fit(rows[1::2])))
    beta = ks.zscore(betas[0] + betas[1])
    r_sh = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in halves]))
    return beta, r_sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cases", choices=("n100", "general50"), default="n100")
    ap.add_argument("--sh-tau", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False

    if args.cases == "n100":
        df = pd.read_csv(CSV)
        cases = [(Path(r["path"]).stem, r["path"], int(r["target"])) for _, r in df.iterrows()]
    else:
        val = Path("/media/sangyu/Dataset/imagenet/val")
        paths = sorted(val.rglob("*.JPEG"))
        rng = random.Random(123)
        rng.shuffle(paths)
        cases = [(p.stem, str(p), -1) for p in paths[:50]]  # -1 -> argmax
    if args.limit:
        cases = cases[: args.limit]

    out_path = args.out or (REPO / f"outputs/class_fri/research_frontier/fri_gate_val_{args.cases}.json")
    arms = ["inflow", "s32", "s32_g", "s32_gsh", "s16", "s16_g", "s16_gsh"]
    rows = {}
    t0 = time.time()
    for n_done, (key, path, target) in enumerate(cases, start=1):
        try:
            x, _ = syms["load_image"](Path(path))
            x = x.to(args.device)
            h_b0, cls_tok, base = syms["get_b0"](model, x)
            if target < 0:
                with torch.no_grad():
                    hold = [torch.cat([cls_tok, h_b0], dim=1)]
                    hk = model.blocks[0].register_forward_pre_hook(lambda m, a: (hold[0],))
                    try:
                        target = int(model(x)[0].argmax().item())
                    finally:
                        hk.remove()
            runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)

            infl = syms["inflow"](model, x, target_class=target).astype(np.float32)
            fri32 = q.fri_alt_prob(runner, steps=32, seed=int(args.seed))["final"]
            fri16 = q.fri_alt_prob(runner, steps=16, seed=int(args.seed))["final"]
            beta, r_sh = ridge_beta_sh(runner, n_states=64, seed=int(args.seed))
            gate_v = 0.1 + 0.9 / (1 + np.exp(-beta / 0.5))
            sh_ok = r_sh >= float(args.sh_tau)

            scores = {
                "inflow": infl,
                "s32": fri32,
                "s32_g": ns._n01(fri32) * gate_v,
                "s32_gsh": ns._n01(fri32) * gate_v if sh_ok else fri32,
                "s16": fri16,
                "s16_g": ns._n01(fri16) * gate_v,
                "s16_gsh": ns._n01(fri16) * gate_v if sh_ok else fri16,
            }
            res = {}
            for nm, sc_ in scores.items():
                _, _, pi, pd_ = q.hard_curves(runner, sc_, chunk=128)
                ci, cd = hs.clip_aucs(pi, pd_)
                res[nm] = {"hdel": cd, "hins": ci}
            res["r_sh"] = r_sh
            res["target"] = int(target)
            rows[key] = res
        except Exception as e:
            print(f"[{n_done:03d}] {key} ERROR {e}", flush=True)
            continue
        if n_done % 10 == 0 or n_done == len(cases):
            print(f"[{n_done:03d}/{len(cases)}] {key} "
                  f"s32={res['s32']['hdel']:.3f} s32g={res['s32_g']['hdel']:.3f} "
                  f"s16g={res['s16_g']['hdel']:.3f} rsh={r_sh:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== {args.cases} (clip) ===")
    print(f"{'arm':8} {'hdel':>7} {'hins':>7} {'winDel':>7} {'winIns':>7} {'winBoth':>8}")
    summary = {}
    for arm in arms:
        d = np.array([rows[k][arm]["hdel"] for k in rows])
        hi = np.array([rows[k][arm]["hins"] for k in rows])
        if arm == "inflow":
            print(f"{arm:8} {d.mean():7.4f} {hi.mean():7.4f}")
            summary[arm] = {"hdel": float(d.mean()), "hins": float(hi.mean())}
            continue
        wd = sum(rows[k][arm]["hdel"] < rows[k]["inflow"]["hdel"] for k in rows)
        wi = sum(rows[k][arm]["hins"] > rows[k]["inflow"]["hins"] for k in rows)
        wb = sum((rows[k][arm]["hdel"] < rows[k]["inflow"]["hdel"]) and
                 (rows[k][arm]["hins"] > rows[k]["inflow"]["hins"]) for k in rows)
        summary[arm] = {"hdel": float(d.mean()), "hins": float(hi.mean()),
                        "win_del": int(wd), "win_ins": int(wi), "win_both": int(wb)}
        print(f"{arm:8} {d.mean():7.4f} {hi.mean():7.4f} {wd:>7} {wi:>7} {wb:>8}")
    rsh = np.array([rows[k]["r_sh"] for k in rows])
    print(f"r_sh mean {rsh.mean():.3f}  gate-on frac {(rsh >= args.sh_tau).mean():.2f}")
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
