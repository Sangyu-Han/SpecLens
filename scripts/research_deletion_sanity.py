#!/usr/bin/env python3
"""Sanity: is the headless 'good deletion' GENUINE necessity or OOD collapse?

Two concerns (user):
  (1) siglip hdel ~0.52 is HIGH (weak deletion) — is the feature-cosine target
      intrinsically deletion-robust? Check base-cosine (empty pos-only feature
      vs full) = the dynamic range available.
  (2) dinov2 hdel ~0.09 is LOW — is it because block0-deletion pushes the model
      OOD and the feature COLLAPSES (so ANY deletion looks 'necessary')?

Discriminators, per model:
  - hdel(ours) vs hdel(RANDOM order) vs hdel(inflow/attn). If random ≈ ours,
    the low hdel is OOD/fragility, NOT necessity. If random ≫ ours, genuine.
  - feature-norm ratio ||feat(masked)||/||feat(full)|| along deletion. ~1 =
    in-distribution; blow-up/collapse = OOD.
  - base-cosine (headless) = floor of the deletion target.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, REPO / path)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


xm = _load("xm", "scripts/research_xmodel_mechanism.py")
nec = _load("nec", "scripts/research_xmodel_necessity.py")
ne = _load("ne", "scripts/research_necessity_erf.py")
from src.utils.image import load_image_clip
from src.baselines.inflow import inflow_attribution
MODELS = xm.MODELS


def featnorm_traj(r, order, fracs):
    N = r.N
    with torch.no_grad():
        full_feat, _ = r._fwd(torch.ones(1, N, device=r.dev, dtype=r.dtype))
        fn0 = float(full_feat.norm())
        masks = []
        for f in fracs:
            k = int(f * N); m = np.ones(N, np.float32); m[order[:k]] = 0.0; masks.append(m)
        feat, _ = r._fwd(torch.as_tensor(np.stack(masks), device=r.dev, dtype=r.dtype))
        return (feat.norm(dim=-1).cpu().numpy() / fn0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["clip", "dinov2", "siglip"])
    ap.add_argument("--nimg", type=int, default=10)
    ap.add_argument("--baseline", choices=("pos", "mean"), default="pos")
    ap.add_argument("--probe-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/deletion_sanity.json")
    args = ap.parse_args()
    import timm

    paths = sorted(Path("/media/sangyu/Dataset/imagenet/val").rglob("*.JPEG"))
    random.Random(123).shuffle(paths)
    imgs = paths[: args.nimg]
    fracs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    report = {}
    for mk in args.models:
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in MODELS[mk] else {}
        model = timm.create_model(MODELS[mk], pretrained=True, **kw).eval().to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        probe = None
        if args.probe_dir is not None:
            pp = args.probe_dir / f"probe_{mk}.pt"
            if pp.exists():
                probe = torch.load(pp, map_location=args.device)
        acc = {"hdel_ours": [], "hdel_rand": [], "hdel_base": [], "basecos": [],
               "fn_ours": [], "fn_rand": []}
        rng = np.random.default_rng(0)
        for ip in imgs:
            try:
                x, _ = xm.load_image(ip, norm=xm.MODEL_NORM.get(mk, "clip")); x = x.to(args.device)
                r = xm.XRunner(model, x, args.device, baseline=args.baseline, probe=probe)
                N = r.N
                p_full = float(r.target_curve(np.ones((1, N), np.float32))[0])
                p_base = float(r.target_curve(np.zeros((1, N), np.float32))[0])
                acc["basecos"].append(p_base)            # for headless = cos(pos-only, full)
                # our self-contained order (ultra-cheap gradtgt)
                gt = ne.fp_gradtgt(r)
                sc_ours, _ = ne.guarded_cheap(r, gt, p_full, p_base)
                ord_ours = np.argsort(-sc_ours)
                # baselines
                if r.has_head:
                    base_attr = inflow_attribution(model, r.x, target_class=r.target).astype(np.float64)
                else:
                    base_attr = nec.last_attn_localization(r).astype(np.float64)
                # random orders (avg 3)
                hdr = []
                fnr = []
                for s in range(3):
                    ro = rng.permutation(N)
                    hdr.append(nec.hard_curves(r, nec.rankscore(ro, N))[1])
                    fnr.append(featnorm_traj(r, ro, fracs))
                acc["hdel_ours"].append(nec.hard_curves(r, sc_ours)[1])
                acc["hdel_base"].append(nec.hard_curves(r, base_attr)[1])
                acc["hdel_rand"].append(float(np.mean(hdr)))
                acc["fn_ours"].append(featnorm_traj(r, ord_ours, fracs))
                acc["fn_rand"].append(np.mean(fnr, axis=0))
            except Exception as e:
                print(f"[{mk} {ip.name}] ERR {type(e).__name__}: {e}", flush=True)
            finally:
                model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        fn_ours = np.mean(acc["fn_ours"], axis=0); fn_rand = np.mean(acc["fn_rand"], axis=0)
        rep = {"n": len(acc["hdel_ours"]), "has_head": bool(r.has_head),
               "hdel_ours": float(np.mean(acc["hdel_ours"])),
               "hdel_random": float(np.mean(acc["hdel_rand"])),
               "hdel_baseline": float(np.mean(acc["hdel_base"])),
               "base_target_empty": float(np.mean(acc["basecos"])),
               "featnorm_fracs": fracs,
               "featnorm_ours": fn_ours.tolist(), "featnorm_random": fn_rand.tolist()}
        report[mk] = rep
        print(f"\n=== {mk} (head={rep['has_head']}, n={rep['n']}) ===")
        print(f"  hdel: ours {rep['hdel_ours']:.3f} | RANDOM {rep['hdel_random']:.3f} | baseline {rep['hdel_baseline']:.3f}")
        print(f"  base target at EMPTY (headless=cos pos-only vs full): {rep['base_target_empty']:.3f}")
        print(f"  feat-norm ratio along deletion {fracs}:")
        print(f"    ours   " + " ".join(f"{v:.2f}" for v in fn_ours))
        print(f"    random " + " ".join(f"{v:.2f}" for v in fn_rand))
        print(f"  >> OOD test: random≈ours hdel => collapse; random≫ours => genuine necessity", flush=True)
        del model; torch.cuda.empty_cache()
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
