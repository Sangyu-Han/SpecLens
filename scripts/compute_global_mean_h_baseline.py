#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

import scripts.run_feature_erf_paper_benchmark as bench


BASELINE_CACHE_ROOT = Path(
    bench.os.environ.get("ERF_BASELINE_CACHE_ROOT", str(bench.REPO / "outputs" / "erf_baselines"))
).expanduser()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute global mean block0 patch embedding baseline.")
    p.add_argument("--pack", default="all", choices=list(bench.mm.PACK_NAMES) + ["all"])
    p.add_argument("--imagenet-val", type=Path, default=bench.mm.IMAGENET_VAL)
    p.add_argument("--n-images", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", type=Path, default=BASELINE_CACHE_ROOT / "global_mean_h")
    return p.parse_args()


def _capture_block0_input(model, x: torch.Tensor) -> torch.Tensor:
    buf: List[torch.Tensor] = []
    hook = model.blocks[0].register_forward_pre_hook(lambda _m, args: buf.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            model(x)
    finally:
        hook.remove()
    return buf[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("compute_global_mean_h_baseline")
    args = parse_args()

    packs = list(bench.mm.PACK_NAMES) if args.pack == "all" else [args.pack]
    val_root = Path(args.imagenet_val)
    args.out_root.mkdir(parents=True, exist_ok=True)

    for pack in packs:
        spec = bench.mm.build_pack_spec(pack)
        model = bench.mm.load_model(spec, args.device)
        transform = bench.mm.load_transform(spec)
        n_prefix = bench.mm.infer_prefix_count(model)

        ds = ImageFolder(str(val_root), transform=transform)
        n_total = len(ds)
        n_take = min(int(args.n_images), n_total)
        g = torch.Generator()
        g.manual_seed(int(args.seed))
        perm = torch.randperm(n_total, generator=g)[:n_take].tolist()
        subset = Subset(ds, perm)
        loader = DataLoader(
            subset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=True,
        )

        token_sum = None
        token_count = 0
        n_patches = None
        d_model = None

        for images, _ in loader:
            images = images.to(args.device, non_blocking=True)
            h_b0 = _capture_block0_input(model, images)
            patch_tokens = h_b0[:, n_prefix:, :]
            if token_sum is None:
                n_patches = int(patch_tokens.shape[1])
                d_model = int(patch_tokens.shape[2])
                token_sum = patch_tokens.sum(dim=(0, 1))
            else:
                token_sum = token_sum + patch_tokens.sum(dim=(0, 1))
            token_count += int(patch_tokens.shape[0] * patch_tokens.shape[1])

        if token_sum is None or token_count <= 0:
            raise RuntimeError(f"Failed to accumulate block0 patch tokens for {pack}")

        mean_vec = (token_sum / float(token_count)).detach().cpu().to(torch.float32).view(1, 1, -1)
        payload: Dict[str, object] = {
            "pack": pack,
            "n_images": n_take,
            "token_count": token_count,
            "n_patches": int(n_patches),
            "d_model": int(d_model),
            "n_prefix": int(n_prefix),
            "seed": int(args.seed),
            "imagenet_val": str(val_root),
            "mean_patch": mean_vec,
            "mean_norm": float(mean_vec.norm().item()),
        }
        out_pt = args.out_root / f"{pack}.pt"
        out_json = args.out_root / f"{pack}.json"
        torch.save(payload, out_pt)
        out_json.write_text(
            json.dumps(
                {k: (v if not isinstance(v, torch.Tensor) else {"shape": list(v.shape)}) for k, v in payload.items()},
                indent=2,
            )
        )
        log.info(
            "[%s] saved %s (%d images, %d tokens, n_patches=%d, d=%d, norm=%.6f)",
            pack,
            out_pt,
            n_take,
            token_count,
            int(n_patches),
            int(d_model),
            float(payload["mean_norm"]),
        )


if __name__ == "__main__":
    main()
