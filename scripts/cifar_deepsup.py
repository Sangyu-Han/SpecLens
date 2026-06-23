"""Leverage the discriminative features: deep-supervision fine-tuning. Attach an
auxiliary class head on the pooled activation of EARLY layers (conv1..layer3) so
each layer is pushed to be class-discriminative (amplify the features that encode
label differences), weighted toward the confusable classes.

Tests both: (a) does using the discriminative signal raise accuracy, and
(b) are EARLIER layers useful for retraining (aux there vs only main loss).

Compares: baseline | CE-only fine-tune | deep-sup (aux on conv1..layer3).

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_deepsup.py --epochs 15
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.cifar_contrastive import confused_pairs, evaluate
from src.packs.cifar_cnn.models.model_loaders import load_cifar_cnn_model

MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
TRAIN_TF = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
AUX_LAYERS = [("conv1", 32), ("layer1", 32), ("layer2", 64), ("layer3", 128)]


def finetune(model, root, device, epochs, deepsup, target, wcls=True):
    caps = {}
    for name, _ in AUX_LAYERS:
        getattr(model, name).register_forward_hook(
            lambda m, i, o, n=name: caps.__setitem__(n, o.mean((2, 3))))   # GAP -> [B,C]
    heads = nn.ModuleDict({n: nn.Linear(c, 100) for n, c in AUX_LAYERS}).to(device) if deepsup else None
    params = list(model.parameters()) + (list(heads.parameters()) if heads else [])
    opt = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = DataLoader(datasets.CIFAR100(root, train=True, download=False, transform=TRAIN_TF),
                    batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    tset = torch.zeros(100, device=device); tset[list(target)] = 1.0
    for _ in range(epochs):
        model.train()
        if heads:
            heads.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            w = torch.where(tset[y] > 0, 2.0, 1.0) if wcls else torch.ones_like(y, dtype=torch.float)
            logits = model(x)
            loss = (nn.functional.cross_entropy(logits, y, reduction="none", label_smoothing=0.1) * w).mean()
            if heads:
                for n, _ in AUX_LAYERS:
                    aux = heads[n](caps[n])
                    loss = loss + 0.3 * (nn.functional.cross_entropy(aux, y, reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    classes = datasets.CIFAR100(args.data_root, train=True, download=False).classes

    base = load_cifar_cnn_model({"ckpt": args.ckpt}, device=device).eval()
    pairs = confused_pairs(base, args.data_root, device, classes, topk=20)
    target = sorted({c for _, i, j in pairs for c in (i, j)})
    b_acc, b_t = evaluate(base, args.data_root, device, target)
    print(f"[baseline]                 acc {b_acc:.4f} | confusable({len(target)}) {b_t:.4f}")
    t0 = time.time()
    for name, ds_flag, wcls in [("CE-only uniform", False, False),
                                ("DEEP-SUP uniform", True, False),
                                ("DEEP-SUP confusable-weighted", True, True)]:
        m = load_cifar_cnn_model({"ckpt": args.ckpt}, device=device)
        m = finetune(m, args.data_root, device, args.epochs, ds_flag, target, wcls)
        a, at = evaluate(m, args.data_root, device, target)
        print(f"[{name:30s}] acc {a:.4f} ({100*(a-b_acc):+.2f}pp) | "
              f"confusable {at:.4f} ({100*(at-b_t):+.2f}pp)")
    print(f"[deepsup] done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
