#!/usr/bin/env python3
"""evTokOR vision port v0 (granularity descent): certified structured attribution on CLIP zero-shot.
'sentences' -> patch clusters (1-fwd k-means on vision-tower patch tokens + spatial coords);
S1 displacement lenses over the competing-prompt distribution (soloKL + median-purged nec');
S2 E-assembly = margin-strict prune / participation / directional prune / extend  [no copy axiom: no literal answer in vision];
S3 patch credit = within-cluster dispNec profile x lens (gated) UNION patch-LOO; prompt-word displacement-LOO; harm channel (blue).
Killer demo: zebra/elephant image, target flip zebra<->elephant => E and harm must swap. CLIP-L/14 cuda:0."""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

MODEL = sorted(glob.glob("/data/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/*/"))[-1]
IMG = "/home/sangyu/Desktop/Master/SpecLens/multi_object_zebra_elephant.jpg"
OUT = "/home/sangyu/Desktop/Master/SpecLens/outputs/set_discovery"
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
ALLP = ["a photo of an elephant", "a photo of a zebra", "a photo of a giraffe", "a photo of a lion",
        "a photo of grass", "a photo of the sky", "a photo of water", "a photo of dirt ground"]
KCLUST = 10
GRID = 16
dev = "cuda:0"
NFWD = [0]


def kl(p, q):
    p = np.clip(p, 1e-9, 1); q = np.clip(q, 1e-9, 1)
    return float((p * np.log(p / q)).sum())


def relu_norm(x):
    x = np.clip(x, 0, None)
    return x / (x.max() + 1e-9)


def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from PIL import Image
    from sklearn.cluster import KMeans
    from transformers import CLIPModel, CLIPTokenizerFast
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).to(dev).eval()
    tokz = CLIPTokenizerFast.from_pretrained(MODEL, local_files_only=True)
    for p in model.parameters():
        p.requires_grad = False
    im = Image.open(IMG).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    base_px = torch.tensor(arr.transpose(2, 0, 1), device=dev)[None]
    Np = GRID * GRID
    tin = tokz(ALLP, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        temb0 = model.get_text_features(**tin).float(); temb0 = temb0 / temb0.norm(dim=-1, keepdim=True)
    ls = model.logit_scale.exp().item()

    def img_emb(masks):
        out = []
        for s in range(0, len(masks), 64):
            mc = np.asarray(masks[s:s + 64]); B = len(mc); NFWD[0] += B
            mg = torch.tensor(mc.reshape(B, GRID, GRID), device=dev, dtype=base_px.dtype)
            mpix = mg.repeat_interleave(14, 1).repeat_interleave(14, 2)[:, None]
            with torch.no_grad():
                ie = model.get_image_features(pixel_values=base_px * mpix).float()
                out.append((ie / ie.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out)

    def dist_of(ie, temb=None):
        t = temb0 if temb is None else temb
        lg = ls * ie @ t.T.cpu()
        return torch.softmax(lg, -1).numpy(), lg.numpy()

    # ---- scaffold ladder: model-free (grid / SLIC superpixel) or internal (mid/last layer k-means) ----
    import os as _os
    SCAFFOLD = _os.environ.get("SCAFFOLD", "last")
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    if SCAFFOLD == "grid":                     # dumbest possible: 4x4 blocks, zero model knowledge
        lab = ((yy // 4) * 4 + (xx // 4)).ravel()
    elif SCAFFOLD == "slic":                   # raw-pixel superpixels, zero model knowledge
        from skimage.segmentation import slic as _slic
        seg = _slic(np.asarray(im), n_segments=12, compactness=18, start_label=0)
        lab = np.array([np.bincount(seg[y * 14:(y + 1) * 14, x * 14:(x + 1) * 14].ravel()).argmax()
                        for y in range(GRID) for x in range(GRID)])
        _, lab = np.unique(lab, return_inverse=True)
    else:                                      # internal features: last or mid layer
        with torch.no_grad():
            NFWD[0] += 1
            vout = model.vision_model(pixel_values=base_px, output_hidden_states=(SCAFFOLD == "mid"))
            hs = (vout.hidden_states[12] if SCAFFOLD == "mid" else vout.last_hidden_state)[0, 1:, :].float().cpu().numpy()
        hs = hs / (np.linalg.norm(hs, axis=1, keepdims=True) + 1e-9)
        feat = np.concatenate([hs, 0.6 * np.stack([yy.ravel() / GRID, xx.ravel() / GRID], 1)], 1)
        lab = KMeans(n_clusters=KCLUST, n_init=4, random_state=0).fit_predict(feat)
    KC = int(lab.max()) + 1
    sm = np.zeros((KC, Np), np.float32)
    for c in range(KC):
        sm[c, lab == c] = 1.0

    def run_target(tgt, tag):
        NFWD[0] = 0
        ns = KC
        probes = [np.ones(Np), np.zeros(Np)] + [sm[c] for c in range(ns)] + [1.0 - sm[c] for c in range(ns)]
        P, LG = dist_of(img_emb(probes))
        base = float(P[0, tgt]); P0 = P[1]
        Mfull = float(LG[0, tgt] - np.delete(LG[0], tgt).max())
        eps = max(0.02, 0.01 * abs(Mfull))
        solokl = np.array([kl(P[2 + c], P0) for c in range(ns)])
        nec_raw = base - P[2 + ns:2 + 2 * ns, tgt]
        nec = np.clip(nec_raw - np.median(nec_raw), 0, None)
        lens = np.maximum(relu_norm(solokl), relu_norm(nec))
        solo_arg = [int(P[2 + c].argmax()) for c in range(ns)]

        def stats(E):
            if not E:
                return None, -1e9, 0.0
            keep = np.clip(sm[E].sum(0), 0, 1)
            p, lg = dist_of(img_emb([keep]))
            return keep, float(lg[0, tgt] - np.delete(lg[0], tgt).max()), float(p[0, tgt])

        E = sorted([c for c in range(ns) if lens[c] >= 0.3])
        keep, ME, RE = stats(E)
        # margin-strict prune (decoy)
        changed = True
        while changed and len(E) > 1:
            changed = False
            for c in list(E):
                E2 = [x for x in E if x != c]
                _, M2, _ = stats(E2)
                if M2 > ME + eps:
                    E = E2; keep, ME, RE = stats(E); changed = True
        # participation prune
        for c in list(E):
            if len(E) > 1 and lens[c] < 0.15:
                E2 = [x for x in E if x != c]
                _, M2, R2 = stats(E2)
                if M2 >= ME - eps:
                    E = E2; keep, ME, RE = stats(E)
        # directional prune (neutral & solo-dir competing & dir-unshared)
        nn_dirs = set()
        for c in E:
            _, M2, _ = stats([x for x in E if x != c])
            if ME - M2 > eps:
                nn_dirs.add(solo_arg[c])
        for c in list(E):
            if len(E) > 1 and solo_arg[c] != tgt and solo_arg[c] not in nn_dirs:
                E2 = [x for x in E if x != c]
                _, M2, _ = stats(E2)
                if abs(ME - M2) <= eps:
                    E = E2; keep, ME, RE = stats(E)
        # extend until sufficiency certificate
        for c in np.argsort(-lens):
            if RE >= 0.9 * base:
                break
            if c not in E:
                E = sorted(E + [int(c)]); keep, ME, RE = stats(E)
        # ---- S3: patch credit inside E ----
        full = np.zeros(Np, np.float32)
        etoks = [t for c in E for t in np.where(lab == c)[0]]
        prof_probes, prof_meta = [], []
        for c in E:
            solo = sm[c]
            for t in np.where(lab == c)[0]:
                m = solo.copy(); m[t] = 0.0
                prof_probes.append(m); prof_meta.append((c, t))
        Pd, _ = dist_of(img_emb(prof_probes))
        loo_probes = []
        for t in etoks:
            m = np.ones(Np); m[t] = 0.0
            loo_probes.append(m)
        Pl, _ = dist_of(img_emb(loo_probes))
        loo_raw = np.clip(base - Pl[:, tgt], 0, None)
        loo_n = {t: v for t, v in zip(etoks, loo_raw / (loo_raw.max() + 1e-9))}
        prof = {}
        for (c, t), pd in zip(prof_meta, Pd):
            prof.setdefault(c, {})[t] = max(0.0, kl(P[2 + c], P0) - kl(pd, P0))
        for c in E:
            ts = list(prof[c].keys()); vv = np.array([prof[c][t] for t in ts])
            vv = 0.25 + 0.75 * (vv / (vv.max() + 1e-9))
            for t, v in zip(ts, vv):
                g = 1.0 if loo_n.get(t, 0) > 0.05 else 0.35
                full[t] = float(max(lens[c] * v * g, loo_n.get(t, 0)))
        # harm channel
        harm = np.zeros(Np, np.float32); harm_report = []
        cand = [c for c in range(ns) if c not in E]
        if cand:
            Pp, _ = dist_of(img_emb([np.clip(keep + sm[c], 0, 1) for c in cand]))
            for c, rp in zip(cand, Pp[:, tgt]):
                if (rp - RE) < -0.1 * max(RE, 1e-9):
                    v = min(1.0, (RE - rp) / max(RE, 1e-9))
                    harm[lab == c] = -v
                    harm_report.append((c, float(rp - RE), ALLP[solo_arg[c]].split()[-1]))
        # prompt-word displacement-LOO (query analogue)
        words = ALLP[tgt].split()
        qd = []
        for j in range(len(words)):
            red = " ".join(words[:j] + words[j + 1:])
            ti = tokz([red], padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                NFWD[0] += 1
                te = model.get_text_features(**ti).float(); te = te / te.norm(dim=-1, keepdim=True)
            tmod = temb0.clone(); tmod[tgt] = te[0]
            pm, _ = dist_of(img_emb([np.ones(Np)])[:1], tmod.cpu())
            qd.append(kl(P[0], pm[0]))
        qdn = relu_norm(np.array(qd))
        cost = NFWD[0]
        print(f"[{SCAFFOLD}|{tag}] base={base:.3f} R(E)={RE:.3f} acid={RE / base:.2f} |E|={len(E)}clu/{len(etoks)}patch "
              f"E={E} harm={[(h[0], round(h[1], 3), h[2]) for h in harm_report]} cost={cost}fwd", flush=True)
        print(f"       lens={lens.round(2).tolist()} solo_dir={[ALLP[a].split()[-1] for a in solo_arg]}", flush=True)
        print(f"       prompt-LOO: " + " ".join(f"{w}:{v:.2f}" for w, v in zip(words, qdn)), flush=True)
        return full, harm, E, lens, qdn, words, base, RE

    # ---- baselines for comparison panels ----
    def banzhaf_baseline(tgt, n=512):
        rng = np.random.default_rng(7)
        Z = (rng.random((n, Np)) < 0.5).astype(np.float32)
        Pz, _ = dist_of(img_emb(Z))
        r = Pz[:, tgt]
        on = Z.T @ r / np.clip(Z.sum(0), 1, None)
        off = (1 - Z).T @ r / np.clip((1 - Z).sum(0), 1, None)
        return on - off

    def loo_baseline(tgt):
        masks = np.ones((Np, Np), np.float32)
        masks[np.arange(Np), np.arange(Np)] = 0.0
        Pl, _ = dist_of(img_emb(masks))
        pf, _ = dist_of(img_emb([np.ones(Np)]))
        return pf[0, tgt] - Pl[:, tgt]

    disp = np.asarray(im)
    rb = LinearSegmentedColormap.from_list("rb", [(0, "#1f4fff"), (0.5, "#ffffff"), (1, "#ff2222")])
    results = {}
    for tgt, tag in [(1, "zebra"), (0, "elephant")]:
        full, harm, E, lens, qdn, words, base, RE = run_target(tgt, tag)
        bz = banzhaf_baseline(tgt); lo = loo_baseline(tgt)
        results[tag] = dict(E=E, base=base, RE=RE)
        fig, ax = plt.subplots(1, 5, figsize=(24, 5.2))
        ax[0].imshow(disp); ax[0].imshow(np.kron(lab.reshape(GRID, GRID), np.ones((14, 14))), cmap="tab10", alpha=0.45, vmin=0, vmax=9)
        for c in E:
            ys, xs = np.where(lab.reshape(GRID, GRID) == c)
            ax[0].scatter(xs * 14 + 7, ys * 14 + 7, s=8, c="white", marker="s")
        ax[0].set_title(f"clusters (white=E={E})")
        m = full + harm
        ax[1].imshow(disp); ax[1].imshow(np.kron(m.reshape(GRID, GRID), np.ones((14, 14))), cmap=rb, alpha=0.62, vmin=-1, vmax=1)
        ax[1].set_title(f"evTok-V: red=credit blue=harm (acid {RE / base:.2f})")
        v = np.abs(bz).max() + 1e-9
        ax[2].imshow(disp); ax[2].imshow(np.kron(bz.reshape(GRID, GRID), np.ones((14, 14))), cmap=rb, alpha=0.62, vmin=-v, vmax=v)
        ax[2].set_title("raw Banzhaf-0.5 (512 fwd)")
        v = np.abs(lo).max() + 1e-9
        ax[3].imshow(disp); ax[3].imshow(np.kron(lo.reshape(GRID, GRID), np.ones((14, 14))), cmap=rb, alpha=0.62, vmin=-v, vmax=v)
        ax[3].set_title("per-patch LOO (256 fwd)")
        ax[4].barh(range(len(words))[::-1], qdn[::-1], color="#ff2222")
        ax[4].set_yticks(range(len(words))[::-1]); ax[4].set_yticklabels(words[::-1]); ax[4].set_xlim(0, 1.05)
        ax[4].set_title("prompt-word displacement-LOO")
        for a in ax[:4]:
            a.axis("off")
        plt.suptitle(f"evTokOR vision port [{SCAFFOLD}] — target: {tag}  (E must flip with target; competitor blue)", fontsize=13)
        plt.tight_layout()
        fp = f"{OUT}/vision_evtok_{tag}_{SCAFFOLD}.png"
        plt.savefig(fp, dpi=110); plt.close()
        print(f"  saved -> {fp}", flush=True)
    a, b = results["zebra"]["E"], results["elephant"]["E"]
    print(f"\n[{SCAFFOLD}] FLIP CHECK: E(zebra)={a} vs E(elephant)={b} overlap={sorted(set(a) & set(b))}", flush=True)


if __name__ == "__main__":
    main()
