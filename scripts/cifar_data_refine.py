"""Data-centric test: can SAE-feature evidence clean the training set and improve
accuracy after RETRAINING?

Flag training samples where the model's feature-evidence confidently DISAGREES
with the label (likely label-noise / ambiguous), drop them, retrain from scratch.
Honest control: also retrain after dropping the SAME NUMBER of RANDOM samples
(dropping data alone hurts -> cleaning only counts if suspects-drop beats
random-drop).

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_data_refine.py --k 2000 --epochs 40
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from src.packs.cifar_cnn.models.model_loaders import load_cifar_cnn_model
from src.packs.cifar_cnn.models.model import CifarResNet

NORM = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
TRAIN_TF = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.ToTensor(), NORM])


@torch.no_grad()
def flag_suspects(model, data_root, device, k):
    ds = datasets.CIFAR100(data_root, train=True, download=False, transform=EVAL_TF)
    loader = DataLoader(ds, batch_size=512, num_workers=4)
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu()); labels.append(y)
    logits = torch.cat(logits); labels = torch.cat(labels)
    preds = logits.argmax(1)
    margin = logits.gather(1, preds[:, None]).squeeze(1) - logits.gather(1, labels[:, None]).squeeze(1)
    mis = preds != labels
    margin[~mis] = -1e9                                  # only confidently-wrong samples
    suspects = torch.topk(margin, k).indices.numpy()
    return suspects, preds.numpy(), labels.numpy()


def train_subset(keep_idx, data_root, device, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    full = datasets.CIFAR100(data_root, train=True, download=False, transform=TRAIN_TF)
    test = datasets.CIFAR100(data_root, train=False, download=False, transform=EVAL_TF)
    tr = DataLoader(Subset(full, keep_idx), batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    te = DataLoader(test, batch_size=256, num_workers=4)
    model = CifarResNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True); crit(model(x), y).backward(); opt.step()
        sched.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in te:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cifar_speclens/cnn.pt")
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    model = load_cifar_cnn_model({"ckpt": args.ckpt}, device=device).eval()
    suspects, preds, labels = flag_suspects(model, args.data_root, device, args.k)
    N = 50000
    print(f"[refine] flagged {len(suspects)} confident-disagreement train samples (of {N})")
    all_idx = np.arange(N)
    keep_clean = np.setdiff1d(all_idx, suspects)
    rng = np.random.default_rng(0)
    keep_rand = np.setdiff1d(all_idx, rng.choice(all_idx, len(suspects), replace=False))

    t = time.time()
    acc_clean = train_subset(keep_clean, args.data_root, device, args.epochs)
    acc_rand = train_subset(keep_rand, args.data_root, device, args.epochs)
    print(f"[refine] baseline (full 50k)         : 0.7096  (existing cnn.pt)")
    print(f"[refine] drop {len(suspects)} SUSPECTS -> retrain: {acc_clean:.4f}")
    print(f"[refine] drop {len(suspects)} RANDOM   -> retrain: {acc_rand:.4f}  (control)")
    print(f"[refine] cleaning effect (suspect - random) = {100*(acc_clean-acc_rand):+.2f} pp   "
          f"[{time.time()-t:.0f}s]")
    print("[refine] NOTE single-seed; flagging is single-model (mild circularity). "
          "Positive (suspect>random) => SAE-evidence cleaning helps beyond just data loss.")


if __name__ == "__main__":
    main()
