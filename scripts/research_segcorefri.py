#!/usr/bin/env python3
"""SegCoreFRI: CoreFRI's dual-sided signed DIM readout over SYNERGY SEGMENTS.

CoreFRI (archive, n100 0.2485) needed 2048 states because it estimated
coalition marginals over 196 patches with spatial blob fields (vision
prior). Replacing the coalition basis with ~10-20 functional synergy
segments cuts the estimation dimension ~10x: 64-128 batched states suffice,
and the blob prior disappears (segments are self-discovered, modality-
general).

  rounds: random-budget segment coalitions C (each segment iid, p~U(0.25,0.75))
    insert side: p_ins = f(union of C on baseline)      [sufficiency]
    delete side: p_del = f(full minus union of C)        [necessity]
  DIM per segment s:
    suff_s = E[p_ins | s in C] - E[p_ins | s not]
    nec_s  = E[p_full - p_del | s in C] - E[..| s not]   (drop when deleted)
  score_s = z(suff) + z(nec)  -> SIGNED (competitors negative -> blue)
  patch score = segment score + tiny within-modulation (singleton jumps)

Renders CoreFRI-style RdBu maps; deletion order = score desc (anti last);
est-guard against the v2 final per case.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = _ilu.spec_from_file_location("nsr", REPO / "scripts/run_necset_n100.py")
nsr = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(nsr)
ns = nsr.ns
q = nsr.q
N, GRID = 196, 14


def cap_merge_segments(segments, h_diff, cap=12):
    """Agglomerate segments to <= cap by content-centroid similarity
    (modality-general; keeps DIM dimension low and coalitions coherent)."""
    X = h_diff[0].detach().float().cpu().numpy()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    segs = [list(s) for s in segments]
    while len(segs) > cap:
        cents = np.stack([Xn[np.asarray(s, int)].mean(0) for s in segs])
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        S = cents @ cents.T
        np.fill_diagonal(S, -np.inf)
        # prefer merging the SMALLEST segment into its nearest neighbor
        si = min(range(len(segs)), key=lambda a: len(segs[a]))
        sj = int(np.argmax(S[si]))
        a, b = (si, sj) if si < sj else (sj, si)
        segs[a] = segs[a] + segs[b]
        del segs[b]
    return segs


def segcore_dim(pc, segments, *, p_full=1.0, rounds=64, seed=0):
    """Dual-sided segment-coalition DIM. Cost: 2*rounds forwards (batched)."""
    nseg = len(segments)
    rng = np.random.default_rng(seed)
    seg_masks = []
    for seg in segments:
        m = np.zeros(N, np.float32)
        m[np.asarray(seg, int)] = 1.0
        seg_masks.append(m)
    seg_masks = np.stack(seg_masks)  # [S, N]
    incl = np.zeros((rounds, nseg), bool)
    states_ins, states_del = [], []
    for r in range(rounds):
        p = rng.uniform(0.25, 0.75)
        c = rng.random(nseg) < p
        if not c.any():
            c[rng.integers(nseg)] = True
        if c.all():
            c[rng.integers(nseg)] = False
        incl[r] = c
        u = (seg_masks[c].sum(0) > 0).astype(np.float32)
        states_ins.append(u)
        states_del.append(1.0 - u)
    v_ins = pc.probs(np.stack(states_ins))
    v_del = pc.probs(np.stack(states_del))
    drop = p_full - v_del

    def dim(vals):
        out = np.zeros(nseg)
        for s in range(nseg):
            a, b = vals[incl[:, s]], vals[~incl[:, s]]
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = np.sqrt(a.var() / max(len(a), 1) + b.var() / max(len(b), 1))
            out[s] = (a.mean() - b.mean()) / max(pooled, 1e-6)
        return out

    return dim(np.asarray(v_ins)), dim(np.asarray(drop))


def segcore_scores(segments, J, suff_z, nec_z, base_scores):
    """Signed patch scores: segment z-sum plateau + tiny within ramp."""
    base_order = ns.order_of(base_scores)
    rank_pos = {int(p): r for r, p in enumerate(base_order)}
    seg_score = suff_z + nec_z
    mx = max(np.abs(seg_score).max(), 1e-8)
    seg_score = seg_score / mx
    scores = np.full(N, -2.0, np.float64)  # non-top floor
    in_top = set()
    for si, seg in enumerate(segments):
        members = sorted(seg, key=lambda i: (-J.get(int(i), 0.0), rank_pos[int(i)]))
        for mi, i in enumerate(members):
            scores[int(i)] = seg_score[si] - 1e-4 * mi
            in_top.add(int(i))
    mids = [i for i in base_order if int(i) not in in_top]
    for mi, i in enumerate(mids):
        scores[int(i)] = -1.0 - 1e-4 * mi  # between positives' tail and antis?
    # place mid BETWEEN smallest positive segment and negative segments:
    # shift: positives stay; mid band at (min_pos_plateau - margin); negatives below mid
    pos_min = min((seg_score[si] for si in range(len(segments)) if seg_score[si] > 0),
                  default=0.1)
    neg_max = max((seg_score[si] for si in range(len(segments)) if seg_score[si] <= 0),
                  default=-0.1)
    mid_lo, mid_hi = neg_max + 0.05, max(pos_min - 0.05, neg_max + 0.06)
    for mi, i in enumerate(mids):
        scores[int(i)] = mid_hi - (mid_hi - mid_lo) * (mi / max(len(mids), 1))
    return scores.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/qual_segcorefri_diag8.png")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "outputs/class_fri/research_frontier/segcorefri_diag8.json")
    args = ap.parse_args()
    import timm

    syms = q.load_patch_repo()
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad = False
    mean = torch.tensor(model.pretrained_cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"]).view(3, 1, 1)

    def up(m):
        return np.kron(np.asarray(m, np.float64).reshape(GRID, GRID),
                       np.ones((16, 16)))

    def draw_signed(ax, img, sc, title):
        ax.imshow(img)
        m = np.asarray(sc, np.float64)
        vmax = max(np.abs(m).max(), 1e-8)
        ax.imshow(up(m), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)

    def draw_jet(ax, img, sc, title, gamma=1.0):
        ax.imshow(img)
        hm = (ns._n01(sc) ** gamma)
        ax.imshow(up(hm), cmap="jet", alpha=0.55, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8)

    rows = {}
    ncol = 5
    fig, axes = plt.subplots(len(ns.CASES), ncol,
                             figsize=(3.1 * ncol, 3.3 * len(ns.CASES)))
    t0 = time.time()
    for ri, (name, path, target) in enumerate(ns.CASES):
        metrics, final_scores, diag = nsr.run_case(
            model, syms, path, target, int(args.seed), args.device)
        x, _ = syms["load_image"](Path(path))
        x = x.to(args.device)
        h_b0, cls_tok, base = syms["get_b0"](model, x)
        runner = q.Block0Runner(model, x, h_b0, cls_tok, base, target)
        img = (x[0].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        p_full, p_base = diag["p_full"], diag["p_base"]

        pc = ns.ProbeCounter(runner)
        segs, J, seg_J, sd = ns.synergy_segments(
            pc, runner, final_scores, top=64, p_full=p_full, seed=int(args.seed))
        segs = cap_merge_segments(segs, runner.h_diff, cap=12)
        sd["n_segments"] = len(segs)
        sd["seg_sizes"] = [len(s_) for s_ in segs]
        suff_z, nec_z = segcore_dim(pc, segs, p_full=p_full,
                                    rounds=int(args.rounds), seed=int(args.seed))
        sg = segcore_scores(segs, J, suff_z, nec_z, final_scores)

        cands = {"v2final": final_scores, "segcore": sg}
        ks_sel = (2, 4, 8, 12, 20, 32, 48, 72, 120)
        best, est = ns.select_prefix(pc, cands, ks=ks_sel,
                                     p_full=p_full, p_base=p_base)
        chosen = cands[best]

        infl = syms["inflow"](model, x, target_class=target).astype(np.float32)
        res = {}
        for nm, s_ in {"inflow": infl, **cands, "chosen": chosen}.items():
            hins, hdel, _, _ = q.hard_curves(runner, s_, chunk=128)
            res[nm] = {"hdel": float(hdel), "hins": float(hins)}
        rows[name] = {"metrics": res, "pick": best,
                      "n_segments": sd["n_segments"],
                      "seg_sizes": sd["seg_sizes"][:8], "probes": pc.n}

        ax = axes[ri, 0]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=8)
        draw_jet(axes[ri, 1], img, infl, f"inflow {res['inflow']['hdel']:.3f}")
        draw_jet(axes[ri, 2], img, final_scores,
                 f"NSD v2 {res['v2final']['hdel']:.3f}", gamma=0.5)
        draw_signed(axes[ri, 3], img, sg,
                    f"SegCoreFRI {res['segcore']['hdel']:.3f}")
        draw_jet(axes[ri, 4], img, chosen,
                 f"chosen[{best}] {res['chosen']['hdel']:.3f}")
        print(f"[{name:<11}] inflow={res['inflow']['hdel']:.3f} "
              f"v2={res['v2final']['hdel']:.3f} segcore={res['segcore']['hdel']:.3f} "
              f"chosen={best} segs={sd['n_segments']} pr={pc.n}", flush=True)

    fd = np.array([rows[k]["metrics"]["chosen"]["hdel"] for k in rows])
    sg_ = np.array([rows[k]["metrics"]["segcore"]["hdel"] for k in rows])
    idl = np.array([rows[k]["metrics"]["inflow"]["hdel"] for k in rows])
    print(f"\nsegcore mean {sg_.mean():.4f} win {(sg_<idl).sum()}/8 | "
          f"chosen mean {fd.mean():.4f} win {(fd<idl).sum()}/8")
    plt.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    args.json_out.write_text(json.dumps(rows, indent=1))
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
