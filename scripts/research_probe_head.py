#!/usr/bin/env python3
"""Train a linear-probe ImageNet head on a FROZEN backbone (dinov2/siglip) so
necessity can use a SHARP class-prob target instead of the weak feature-cosine.

val-only (no train): split each class's images into probe-train / probe-test.
Extract pooled features (same pooling the necessity pipeline uses: CLS for
dinov2, GAP for siglip), fit a linear classifier, report top-1, save the probe
(W,b + feature mean/std) for reuse in the necessity experiments.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_spec = _ilu.spec_from_file_location("xm", REPO / "scripts/research_xmodel_mechanism.py")
xm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(xm)
from src.utils.image import load_image_clip
MODELS = xm.MODELS


class FeatDS(torch.utils.data.Dataset):
    def __init__(self, items, norm="clip"):
        self.items = items; self.norm = norm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        x, _ = xm.load_image(p, norm=self.norm)
        return x.squeeze(0), y


def extract(model, items, dev, bs=128, workers=8, norm="clip"):
    dl = torch.utils.data.DataLoader(FeatDS(items, norm), batch_size=bs, num_workers=workers)
    feats, ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            f = model(xb.to(dev))
            if not torch.is_tensor(f):
                f = f[0]
            feats.append(f.float().cpu()); ys.append(yb)
    return torch.cat(feats), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="+", default=["dinov2", "siglip"])
    ap.add_argument("--ntrain", type=int, default=20)
    ap.add_argument("--ntest", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", type=Path, default=REPO / "outputs/class_fri/research_frontier/probe_heads")
    args = ap.parse_args()
    import timm

    args.out.mkdir(parents=True, exist_ok=True)
    val = Path("/media/sangyu/Dataset/imagenet/val")
    classes = sorted([d for d in val.iterdir() if d.is_dir()])
    train_items, test_items = [], []
    for ci, c in enumerate(classes):
        imgs = sorted(c.glob("*.JPEG"))
        train_items += [(p, ci) for p in imgs[: args.ntrain]]
        test_items += [(p, ci) for p in imgs[args.ntrain: args.ntrain + args.ntest]]
    print(f"train {len(train_items)}  test {len(test_items)}  ({len(classes)} classes)", flush=True)

    report = {}
    for mk in args.models:
        kw = dict(img_size=224, dynamic_img_size=True) if "dinov2" in MODELS[mk] else {}
        model = timm.create_model(MODELS[mk], pretrained=True, num_classes=0, **kw).eval().to(args.device)
        for p in model.parameters():
            p.requires_grad = False
        import time
        t0 = time.time()
        norm = xm.MODEL_NORM.get(mk, "clip")
        print(f"[{mk}] using {norm}-norm preprocessing", flush=True)
        Xtr, ytr = extract(model, train_items, args.device, norm=norm)
        Xte, yte = extract(model, test_items, args.device, norm=norm)
        C = Xtr.shape[1]
        print(f"[{mk}] feat dim {C}  extract {time.time()-t0:.0f}s", flush=True)
        # standardize
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xtr = ((Xtr - mu) / sd).to(args.device); Xte = ((Xte - mu) / sd).to(args.device)
        ytr = ytr.to(args.device); yte = yte.to(args.device)
        W = torch.nn.Linear(C, 1000).to(args.device)
        opt = torch.optim.AdamW(W.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = torch.nn.CrossEntropyLoss()
        n = Xtr.shape[0]; bs = 4096
        for ep in range(args.epochs):
            perm = torch.randperm(n, device=args.device)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                opt.zero_grad(); loss = lossf(W(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        with torch.no_grad():
            tr_acc = float((W(Xtr).argmax(1) == ytr).float().mean())
            te_logits = W(Xte)
            te_acc = float((te_logits.argmax(1) == yte).float().mean())
            te_acc5 = float((te_logits.topk(5, 1).indices == yte[:, None]).any(1).float().mean())
        report[mk] = {"model": MODELS[mk], "feat_dim": C, "n_train": len(train_items),
                      "n_test": len(test_items), "train_acc": tr_acc, "test_acc": te_acc, "test_acc5": te_acc5}
        torch.save({"W": W.weight.detach().cpu(), "b": W.bias.detach().cpu(),
                    "mu": mu, "sd": sd, "acc": te_acc}, args.out / f"probe_{mk}.pt")
        print(f"[{mk}] PROBE top1 {te_acc*100:.1f}%  top5 {te_acc5*100:.1f}%  (train {tr_acc*100:.1f}%) -> probe_{mk}.pt", flush=True)
        del model, Xtr, Xte; torch.cuda.empty_cache()
    (args.out / "probe_report.json").write_text(json.dumps(report, indent=1))
    print(f"[done] -> {args.out}/probe_report.json")


if __name__ == "__main__":
    main()
