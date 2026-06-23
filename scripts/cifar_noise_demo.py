"""Controlled demo: does feature/confidence-based cleaning recover accuracy when
the data REALLY has label noise? Inject known noise, train, DETECT it via
confident-disagreement, relabel, retrain, and measure recovery + detection P/R.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/cifar_noise_demo.py --noise 0.15 --epochs 40
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.packs.cifar_cnn.dataset.builders import CIFAR100_MEAN, CIFAR100_STD
from src.packs.cifar_cnn.models.model import CifarResNet

NORM = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
TRAIN_TF = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.ToTensor(), NORM])


class Relabeled(datasets.CIFAR100):
    def __init__(self, root, targets, transform):
        super().__init__(root, train=True, download=False, transform=transform)
        self.targets = [int(t) for t in targets]


def train_labels(root, targets, device, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(Relabeled(root, targets, TRAIN_TF), batch_size=128, shuffle=True,
                    num_workers=4, drop_last=True)
    te = DataLoader(datasets.CIFAR100(root, train=False, download=False, transform=EVAL_TF),
                    batch_size=256, num_workers=4)
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
    model.eval(); c = t = 0
    with torch.no_grad():
        for x, y in te:
            c += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); t += y.numel()
    return model, c / t


@torch.no_grad()
def confident_disagree(model, root, targets, device):
    ds = Relabeled(root, targets, EVAL_TF)
    loader = DataLoader(ds, batch_size=512, num_workers=4)
    lg, yy = [], []
    for x, y in loader:
        lg.append(model(x.to(device)).cpu()); yy.append(y)
    lg = torch.cat(lg); yy = torch.cat(yy); pr = lg.argmax(1)
    margin = lg.gather(1, pr[:, None]).squeeze(1) - lg.gather(1, yy[:, None]).squeeze(1)
    margin[pr == yy] = -1e9
    return pr.numpy(), margin.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/sangyu/Desktop/Master/CBM_test/data")
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    clean = np.array(datasets.CIFAR100(args.data_root, train=True, download=False).targets)
    N = len(clean); rng = np.random.default_rng(0)
    n_noise = int(args.noise * N)
    noisy_idx = rng.choice(N, n_noise, replace=False)
    noisy = clean.copy()
    for i in noisy_idx:
        noisy[i] = (clean[i] + rng.integers(1, 100)) % 100          # forced-wrong label
    is_noise = np.zeros(N, bool); is_noise[noisy_idx] = True
    print(f"[noise] injected {n_noise} ({args.noise:.0%}) wrong labels into {N} train samples")

    t = time.time()
    m_noisy, acc_noisy = train_labels(args.data_root, noisy, device, args.epochs)
    pr, margin = confident_disagree(m_noisy, args.data_root, noisy, device)
    flagged = np.argsort(margin)[::-1][:n_noise]                    # flag top n_noise by confidence
    inter = is_noise[flagged].sum()
    P = inter / len(flagged); R = inter / n_noise
    # relabel flagged to the model's prediction
    fixed = noisy.copy(); fixed[flagged] = pr[flagged]
    recovered = (fixed[noisy_idx] == clean[noisy_idx]).mean()       # of injected noise, fraction relabeled correctly
    m_clean, acc_clean = train_labels(args.data_root, fixed, device, args.epochs)

    print(f"[noise] train-on-NOISY      -> test acc {acc_noisy:.4f}")
    print(f"[noise] detection of injected noise: precision {P:.2f}  recall {R:.2f}")
    print(f"[noise] relabel flagged->pred: {recovered:.0%} of injected-noise labels restored to TRUE")
    print(f"[noise] train-on-CLEANED    -> test acc {acc_clean:.4f}   (recovered +{100*(acc_clean-acc_noisy):.2f}pp)")
    print(f"[noise] reference clean-data acc 0.7096   [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
