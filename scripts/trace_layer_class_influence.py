#!/usr/bin/env python3
"""Test the hypothesis: does the per-token influence on elephant/zebra at a LATE
layer localize the core, and is that why InFlow works (route it to input)?

For each block-output layer L, compute input x grad of the FINAL elephant logit
(and zebra logit) w.r.t. the residual stream at L: attr_L[i] = (d logit / d h_L)*h_L
summed over channels, per token position. Track ELE/ZEB mass vs depth for:
  - elephant attribution (absolute)
  - elephant - zebra contrast (does the contrast separate even if absolute doesn't?)
References: L=0 ~ IxG (credits zebra), InFlow (clean core).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch

REPO = Path(__file__).resolve().parents[1]
PATCH_REPO = Path("/home/sangyu/Desktop/Master/patch-attribution-vit")
N_PATCHES, GRID = 196, 14
IMAGE = REPO / "multi_object_zebra_elephant.jpg"
ELEPHANT, ZEBRA = 386, 340
DEVICE = "cuda:1"
OUT = REPO / "outputs/class_fri/layer_class_influence"


def _load_syms() -> Dict[str, Any]:
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    sys.path.insert(0, str(PATCH_REPO))
    from src.baselines.inflow import inflow_attribution
    from src.utils.image import load_image_clip
    from src.utils.vit_hooks import get_block0_inputs
    return {"inflow_attribution": inflow_attribution, "load_image_clip": load_image_clip,
            "get_block0_inputs": get_block0_inputs}


def _mass(attr: np.ndarray, region: np.ndarray) -> float:
    a = np.maximum(np.nan_to_num(np.asarray(attr, np.float64).reshape(-1)), 0.0)
    tot = a.sum()
    return float(a[region].sum() / tot) if tot > 1e-12 else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    syms = _load_syms()
    dev = torch.device(DEVICE)
    print(f"[info] loading model on {dev}", flush=True)
    model = timm.create_model("vit_base_patch16_clip_224.laion2b_ft_in12k_in1k", pretrained=True)
    model.eval().to(dev)
    for p_ in model.parameters():
        p_.requires_grad = False

    x, _ = syms["load_image_clip"](IMAGE); x = x.to(dev)
    inf_ele = np.asarray(syms["inflow_attribution"](model, x, target_class=ELEPHANT), np.float32)
    inf_zeb = np.asarray(syms["inflow_attribution"](model, x, target_class=ZEBRA), np.float32)
    ne = inf_ele / (inf_ele.max() + 1e-8); nz = inf_zeb / (inf_zeb.max() + 1e-8)
    ele_top = set(np.argsort(-inf_ele)[:45].tolist()); zeb_top = set(np.argsort(-inf_zeb)[:22].tolist())
    for i in list(ele_top & zeb_top):
        (ele_top if ne[i] >= nz[i] else zeb_top).discard(i)
    ELE = np.array(sorted(ele_top), np.int64); ZEB = np.array(sorted(zeb_top), np.int64)
    print(f"[regions] |ELE|={len(ELE)} |ZEB|={len(ZEB)} | InFlow(ele) mass ELE={_mass(inf_ele,ELE):.3f} ZEB={_mass(inf_ele,ZEB):.3f}", flush=True)

    # forward with block-output capture (in graph), grad of final logits w.r.t. each layer
    outs = []
    hooks = [blk.register_forward_hook(lambda m, i, o: outs.append(o if torch.is_tensor(o) else o[0]))
             for blk in model.blocks]
    x_in = x.detach().requires_grad_(True)
    logits = model(x_in)[0]
    g_ele = torch.autograd.grad(logits[ELEPHANT], outs, retain_graph=True)
    g_zeb = torch.autograd.grad(logits[ZEBRA], outs, retain_graph=True)
    for h in hooks:
        h.remove()

    nb = len(outs)
    ele_attr = {}; zeb_attr = {}; contrast = {}
    for L in range(nb):
        hL = outs[L].detach()[0]                       # [N, C]
        ea = (g_ele[L][0] * hL).sum(-1)[1:].detach().cpu().numpy()    # [196] final-ele input*grad at layer L
        za = (g_zeb[L][0] * hL).sum(-1)[1:].detach().cpu().numpy()
        ele_attr[L] = ea; zeb_attr[L] = za; contrast[L] = ea - za

    rows = {
        "ele_attr": [( _mass(ele_attr[L], ELE), _mass(ele_attr[L], ZEB)) for L in range(nb)],
        "contrast": [( _mass(contrast[L], ELE), _mass(contrast[L], ZEB)) for L in range(nb)],
        "zeb_attr_onZEB": [_mass(zeb_attr[L], ZEB) for L in range(nb)],
    }
    print("\nlayer  ele_attr(ELE/ZEB)   contrast(ELE/ZEB)   zeb_attr->ZEB")
    for L in range(nb):
        e, z = rows["ele_attr"][L]; ce, cz = rows["contrast"][L]
        print(f"  b{L:<2}  {e:.2f}/{z:.2f}          {ce:.2f}/{cz:.2f}          {rows['zeb_attr_onZEB'][L]:.2f}", flush=True)
    inflow_ratio = _mass(inf_ele, ELE) / max(_mass(inf_ele, ZEB), 1e-6)
    print(f"\n[ref] IxG(=layer0 ele_attr) ELE/ZEB = {rows['ele_attr'][0][0]:.2f}/{rows['ele_attr'][0][1]:.2f} "
          f"(ratio {rows['ele_attr'][0][0]/max(rows['ele_attr'][0][1],1e-6):.1f})", flush=True)
    print(f"[ref] InFlow ELE/ZEB ratio = {inflow_ratio:.1f}", flush=True)
    pen = nb - 2
    print(f"[hyp] penultimate (b{pen}) ele_attr ratio = {rows['ele_attr'][pen][0]/max(rows['ele_attr'][pen][1],1e-6):.1f} | "
          f"contrast ratio = {rows['contrast'][pen][0]/max(rows['contrast'][pen][1],1e-6):.1f}", flush=True)

    from PIL import Image
    img = Image.open(IMAGE).convert("RGB").resize((224, 224))
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, :2])
    L = list(range(nb))
    ax.plot(L, [e for e, _ in rows["ele_attr"]], "o-", color="tab:blue", label="ele_attr ELE")
    ax.plot(L, [z for _, z in rows["ele_attr"]], "o--", color="tab:blue", alpha=0.4, label="ele_attr ZEB")
    ax.plot(L, [e for e, _ in rows["contrast"]], "^-", color="tab:green", label="contrast ELE")
    ax.plot(L, [z for _, z in rows["contrast"]], "^--", color="tab:green", alpha=0.4, label="contrast ZEB")
    ax.axhline(_mass(inf_ele, ELE), color="k", ls=":", label="InFlow ELE")
    ax.set_xlabel("layer L (block output)"); ax.set_ylabel("mass fraction")
    ax.set_title("final-logit input*grad localization by layer"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # maps at layer0(IxG), mid(b6), penultimate(b10), inflow
    for a, (ttl, arr) in zip([fig.add_subplot(gs[1, j]) for j in range(4)] + [fig.add_subplot(gs[0, 2])],
                             [("ele_attr b0 (IxG)", ele_attr[0]), ("ele_attr b6", ele_attr[6]),
                              ("ele_attr b10 (penult)", ele_attr[10]), ("contrast b10", contrast[10]),
                              ("InFlow(ele)", inf_ele)]):
        a.imshow(img, extent=(0, 224, 224, 0))
        vv = np.maximum(arr, 0); vv = vv / (vv.max() + 1e-8)
        a.imshow(vv.reshape(GRID, GRID), cmap="magma", alpha=0.6, extent=(0, 224, 224, 0))
        a.set_title(ttl, fontsize=9); a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "layer_influence.png", dpi=130); plt.close(fig)
    (OUT / "trace.json").write_text(json.dumps({"rows": rows, "inflow_mass_ele": _mass(inf_ele, ELE),
                                                "inflow_mass_zeb": _mass(inf_ele, ZEB)}, indent=2))
    print(f"\n[info] wrote {OUT}/layer_influence.png trace.json", flush=True)


if __name__ == "__main__":
    main()
