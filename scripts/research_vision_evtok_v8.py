#!/usr/bin/env python3
"""Vision v8: PMI game + certified E + two model-free scaffolds.
Game: v(S) = log P(target|masked S) - log P(target|empty)  [softmax over competing prompts] = PMI(class; visible patches).
Scaffolds ('sentence unit' without model internals):
  slic     - SLIC superpixels (raw pixels only; prior ladder exp: == last-layer, mid-layer was worst)
  quadtree - CERTIFICATION-DRIVEN GRANULARITY DESCENT: assemble E at 2x2 blocks, split only E members 4-way,
             re-certify, recurse to 2x2-patch grain. Units are DEFINED by the certification process itself.
E-assembly = v8 (lens=max(solo-gain, nec') -> fixed-point eps=0.02|gainN| -> tau=0.9 extend -> neutral-guard backmin).
S3: per-patch LOO within E (v units) red; harm = dv(add)< -eps blue; prompt-word LOO bars.
CLIP-L/14, zebra+elephant image, both targets, flip check. cuda:0."""
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
GRID = 16
dev = "cuda:0"
NF = [0]


def relu_norm(x):
    x = np.clip(x, 0, None); return x / (x.max() + 1e-9)


def main():
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import transformers.utils.import_utils as iu
    iu._torchvision_available = False; iu.is_torchvision_available = lambda *a, **k: False
    from PIL import Image
    from transformers import CLIPModel, CLIPTokenizerFast
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).to(dev).eval()
    tokz = CLIPTokenizerFast.from_pretrained(MODEL, local_files_only=True)
    im = Image.open(IMG).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    base_px = torch.tensor(arr.transpose(2, 0, 1), device=dev)[None]
    Np = GRID * GRID
    tin = tokz(ALLP, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        temb0 = model.get_text_features(**tin).float(); temb0 = temb0 / temb0.norm(dim=-1, keepdim=True)
    ls = model.logit_scale.exp().item()

    def fwd_lp(masks, tgt, temb=None):
        NF[0] += len(masks)
        t = temb0 if temb is None else temb
        out = []
        for s in range(0, len(masks), 64):
            mc = np.asarray(masks[s:s + 64]); B = len(mc)
            mg = torch.tensor(mc.reshape(B, GRID, GRID), device=dev, dtype=base_px.dtype)
            mpix = mg.repeat_interleave(14, 1).repeat_interleave(14, 2)[:, None]
            with torch.no_grad():
                ie = model.get_image_features(pixel_values=base_px * mpix).float()
                ie = ie / ie.norm(dim=-1, keepdim=True)
                lp = torch.log_softmax(ls * ie @ t.T, -1)[:, tgt]
            out.append(lp.cpu().numpy())
        return np.concatenate(out)

    # ---- scaffolds ----
    def units_slic():
        from skimage.segmentation import slic as _slic
        seg = _slic(np.asarray(im), n_segments=12, compactness=18, start_label=0)
        lab = np.array([np.bincount(seg[y * 14:(y + 1) * 14, x * 14:(x + 1) * 14].ravel()).argmax()
                        for y in range(GRID) for x in range(GRID)])
        _, lab = np.unique(lab, return_inverse=True)
        return [np.where(lab == c)[0].tolist() for c in range(lab.max() + 1)]

    def blocks(y0, x0, h):
        return [y * GRID + x for y in range(y0, y0 + h) for x in range(x0, x0 + h)]

    def split4(u):
        ys = sorted({p // GRID for p in u}); xs = sorted({p % GRID for p in u})
        y0, x0, h = ys[0], xs[0], len(ys)
        if h <= 2:
            return [u]
        hh = h // 2
        return [blocks(y0, x0, hh), blocks(y0, x0 + hh, hh), blocks(y0 + hh, x0, hh), blocks(y0 + hh, x0 + hh, hh)]

    # ---- v8 E-assembly over a unit list ----
    def assemble(units, tgt, v0):
        ns = len(units)
        sm = np.zeros((ns, Np))
        for i, u in enumerate(units):
            sm[i, u] = 1.0
        vv = fwd_lp([np.ones(Np)] + [sm[i] for i in range(ns)] + [1.0 - sm[i] for i in range(ns)], tgt)
        vN = float(vv[0]); gainN = vN - v0
        g = vv[1:1 + ns] - v0
        nec_raw = vN - vv[1 + ns:1 + 2 * ns]
        nec = np.clip(nec_raw - np.median(nec_raw), 0, None)
        lens = np.maximum(relu_norm(g), relu_norm(nec))
        epsA = 0.02 * max(abs(gainN), 1e-6)
        E = sorted([i for i in range(ns) if lens[i] >= 0.3])
        gE = float(fwd_lp([np.clip(sm[E].sum(0), 0, 1)], tgt)[0] - v0) if E else 0.0
        for sweep in range(3):
            outs = [i for i in range(ns) if i not in E and lens[i] >= 0.15]
            pr = [np.clip(sm[[x for x in E if x != i]].sum(0), 0, 1) for i in E] + \
                 [np.clip(sm[E + [i]].sum(0), 0, 1) for i in outs]
            if not pr:
                break
            vv2 = fwd_lp(pr, tgt) - v0
            dm = {E[k]: gE - float(vv2[k]) for k in range(len(E))}
            da = {outs[j]: float(vv2[len(E) + j]) - gE for j in range(len(outs))}
            harmful = [i for i in E if dm[i] < -epsA]
            adds = [i for i in outs if da[i] > epsA]
            neutral = [i for i in E if i not in harmful and abs(dm[i]) <= epsA and lens[i] < 0.15]
            if not harmful and not adds and not neutral:
                break
            E = sorted((set(E) - set(harmful) - set(neutral)) | set(adds))
            gE = float(fwd_lp([np.clip(sm[E].sum(0), 0, 1)], tgt)[0] - v0) if E else 0.0
        for i in np.argsort(-lens):
            if gE >= 0.9 * gainN:
                break
            if int(i) not in E:
                E = sorted(E + [int(i)]); gE = float(fwd_lp([np.clip(sm[E].sum(0), 0, 1)], tgt)[0] - v0)
        for i in list(np.argsort(lens)):
            i = int(i)
            if i not in E or len(E) <= 1:
                continue
            E2 = [x for x in E if x != i]
            g2 = float(fwd_lp([np.clip(sm[E2].sum(0), 0, 1)], tgt)[0] - v0)
            if g2 >= 0.9 * gainN and (gE - g2) <= epsA:
                E = E2; gE = g2
        return E, gE, gainN, sm, epsA

    def run(scaffold, tgt, tag):
        NF[0] = 0
        v0 = float(fwd_lp([np.zeros(Np)], tgt)[0])
        if scaffold == "slic":
            units = units_slic()
            E, gE, gainN, sm, epsA = assemble(units, tgt, v0)
        else:  # quadtree certification descent
            units = [blocks(0, 0, 8), blocks(0, 8, 8), blocks(8, 0, 8), blocks(8, 8, 8)]
            for level in range(3):  # 8 -> 4 -> 2
                E, gE, gainN, sm, epsA = assemble(units, tgt, v0)
                new_units = []
                for i, u in enumerate(units):
                    if i in E and len(u) > 4:
                        new_units += split4(u)
                    else:
                        new_units.append(u)
                if len(new_units) == len(units):
                    break
                units = new_units
            E, gE, gainN, sm, epsA = assemble(units, tgt, v0)
        # ---- S3: per-patch LOO within E (v units) + harm ----
        epatch = sorted({p for i in E for p in units[i]})
        keep = np.zeros(Np); keep[epatch] = 1.0
        pr = []
        for p in epatch:
            m = keep.copy(); m[p] = 0.0; pr.append(m)
        lv = fwd_lp(pr, tgt) - v0 if epatch else np.array([])
        cred = np.zeros(Np)
        for p, x in zip(epatch, lv):
            cred[p] = max(0.0, gE - float(x))
        cred = relu_norm(cred)
        harm = np.zeros(Np); cand = [i for i in range(len(units)) if i not in E]
        if cand:
            hv = fwd_lp([np.clip(keep + sm[i], 0, 1) for i in cand], tgt) - v0
            for i, x in zip(cand, hv):
                if (float(x) - gE) < -epsA:
                    harm[units[i]] = -min(1.0, (gE - float(x)) / max(abs(gainN), 1e-6))
        # prompt-word LOO
        words = ALLP[tgt].split(); qd = []
        vN_lp = gainN + v0
        for j in range(len(words)):
            red = " ".join(words[:j] + words[j + 1:])
            ti = tokz([red], padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                te = model.get_text_features(**ti).float(); te = te / te.norm(dim=-1, keepdim=True)
            tmod = temb0.clone(); tmod[tgt] = te[0]
            qd.append(vN_lp - float(fwd_lp([np.ones(Np)], tgt, tmod)[0]))
        qdn = relu_norm(np.array(qd))
        cost = NF[0]
        print(f"[{scaffold}|{tag}] gain(E)/gain(N)={gE / max(gainN, 1e-6):.2f} |E|={len(E)}u/{len(epatch)}p "
              f"units={len(units)} cost={cost}fwd gainN={gainN:.2f}", flush=True)
        return cred, harm, epatch, units, E, qdn, words, gE, gainN, cost

    disp = np.asarray(im)
    rb = LinearSegmentedColormap.from_list("rb", [(0, "#1f4fff"), (0.5, "#ffffff"), (1, "#ff2222")])
    results = {}
    for scaffold in ("slic", "quadtree"):
        for tgt, tag in [(1, "zebra"), (0, "elephant")]:
            cred, harm, epatch, units, E, qdn, words, gE, gainN, cost = run(scaffold, tgt, tag)
            results[(scaffold, tag)] = set(epatch)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
            ax[0].imshow(disp)
            lab_img = np.full(Np, -1)
            for i, u in enumerate(units):
                lab_img[u] = i
            ax[0].imshow(np.kron(lab_img.reshape(GRID, GRID), np.ones((14, 14))), cmap="tab20", alpha=0.35)
            ys, xs = np.where(np.isin(np.arange(Np).reshape(GRID, GRID), epatch))
            ax[0].scatter(xs * 14 + 7, ys * 14 + 7, s=7, c="white", marker="s")
            ax[0].set_title(f"{scaffold} units (white=E: {len(E)}u/{len(epatch)}p)")
            m = cred + harm
            ax[1].imshow(disp); ax[1].imshow(np.kron(m.reshape(GRID, GRID), np.ones((14, 14))), cmap=rb, alpha=0.62, vmin=-1, vmax=1)
            ax[1].set_title(f"PMI-game credit/harm  gain(E)/gain(N)={gE / max(gainN, 1e-6):.2f}  ({cost}fwd)")
            ax[2].barh(range(len(words))[::-1], qdn[::-1], color="#ff2222")
            ax[2].set_yticks(range(len(words))[::-1]); ax[2].set_yticklabels(words[::-1]); ax[2].set_xlim(0, 1.05)
            ax[2].set_title("prompt-word v-LOO")
            for a in ax[:2]:
                a.axis("off")
            plt.suptitle(f"Vision v8 [{scaffold}] target={tag} — PMI game, certification-driven units", fontsize=13)
            plt.tight_layout()
            fp = f"{OUT}/vision_v8_{scaffold}_{tag}.png"
            plt.savefig(fp, dpi=110); plt.close()
            print(f"  -> {fp}", flush=True)
    for sc in ("slic", "quadtree"):
        a, b = results[(sc, "zebra")], results[(sc, "elephant")]
        print(f"FLIP[{sc}]: overlap={len(a & b)}patch (z={len(a)}, e={len(b)})", flush=True)


if __name__ == "__main__":
    main()
