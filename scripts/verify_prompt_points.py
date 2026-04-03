"""
GT 마스크 위에 point prompt가 올바르게 찍히는지 시각적으로 검증.
- center 방법: 마스크 중심 (RITM distance transform) → 마스크 위
- uniform 방법: 이미지 전체 랜덤 → 마스크 밖일 수 있음

Usage:
    python scripts/verify_prompt_points.py \
        --config configs/sam2_sav_batchtopk_train.yaml \
        --n_samples 16 \
        --out_dir outputs/verify_prompt_points
"""

import argparse, sys, yaml, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def denormalize(img_tensor):
    """(3,H,W) normalized tensor → (H,W,3) uint8"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = img_tensor.cpu().float() * std + mean
    return (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def overlay_mask(img_rgb, mask_bool, color=(0, 200, 0), alpha=0.4):
    out = img_rgb.copy().astype(np.float32)
    out[mask_bool] = out[mask_bool] * (1 - alpha) + np.array(color) * alpha
    return out.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/sam2_sav_batchtopk_train.yaml")
    parser.add_argument("--n_samples",  type=int, default=16)
    parser.add_argument("--out_dir",    default="outputs/verify_prompt_points")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for p in cfg.get("pack", {}).get("sys_paths", []):
        sys.path.insert(0, p)

    from src.packs.sam2.train.factories import build_dataset
    print("Building dataset...")
    ds_out     = build_dataset(cfg["dataset"], full_config=cfg)
    dataset    = ds_out["dataset"]
    collate_fn = ds_out["collate_fn"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"on": 0, "off": 0}

    for sample_idx in range(args.n_samples):
        batch = collate_fn([dataset[sample_idx]])

        # ── 이미지 (첫 프레임, 첫 비디오) ──────────────────────────────
        img_np = denormalize(batch.img_batch[0, 0])   # (H,W,3)
        H, W   = img_np.shape[:2]

        # ── 마스크 / UID (첫 프레임) ────────────────────────────────────
        masks_t0    = batch.masks[0].cpu()              # (N, H, W) bool
        uid_t0      = batch.metadata.mask_uid_by_t[0]  # (N,)
        b_idx_t0    = batch.obj_to_frame_idx[0, :, 1]  # (N,) — 비디오 인덱스

        # ── 프롬프트 (t=0) ──────────────────────────────────────────────
        by_uid = (batch.metadata.offline_prompt.per_frame
                  .get(0, {}).get("by_uid", {}))

        # 비디오 0에 속하는 객체만 추려서 그리기
        n_objs = int((b_idx_t0 == 0).sum())
        ncols  = max(1, n_objs)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        obj_counter = 0
        for n in range(masks_t0.shape[0]):
            if int(b_idx_t0[n]) != 0:
                continue

            mask = masks_t0[n].bool().numpy()   # (H, W)
            uid  = str(int(uid_t0[n]))
            vis  = overlay_mask(img_np, mask)

            ax   = axes[obj_counter]
            ax.imshow(vis)

            point_info = by_uid.get(uid)
            on_mask_flag = None
            if point_info is not None:
                for (x, y), lbl in zip(point_info["point_coords"],
                                       point_info["point_labels"]):
                    xi, yi = int(round(x)), int(round(y))
                    color  = "lime" if lbl == 1 else "red"
                    ax.plot(xi, yi, "o", color=color,
                            markersize=11, markeredgecolor="white", markeredgewidth=1.5)
                    if 0 <= yi < H and 0 <= xi < W:
                        on_mask_flag = bool(mask[yi, xi])
                        if on_mask_flag:
                            stats["on"]  += 1
                        else:
                            stats["off"] += 1

            title = f"obj {n}"
            if on_mask_flag is True:
                title += "\n✓ ON mask"
            elif on_mask_flag is False:
                title += "\n✗ OFF mask"
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            obj_counter += 1

        # 사용되지 않은 axes 숨기기
        for ax in axes[obj_counter:]:
            ax.axis("off")

        fig.legend(
            handles=[mpatches.Patch(color="lime", label="positive (label=1)"),
                     mpatches.Patch(color="red",  label="negative (label=0)")],
            loc="lower center", ncol=2, fontsize=8
        )
        fig.suptitle(f"sample {sample_idx}", fontsize=10)
        fig.tight_layout()
        save_path = out_dir / f"sample_{sample_idx:03d}.png"
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[{sample_idx+1}/{args.n_samples}] → {save_path}")

    total = stats["on"] + stats["off"]
    print()
    print("=" * 40)
    print(f"총 point 수  : {total}")
    print(f"마스크 위    : {stats['on']}  ({100*stats['on']/max(total,1):.1f}%)")
    print(f"마스크 밖    : {stats['off']} ({100*stats['off']/max(total,1):.1f}%)")
    print(f"저장 위치    : {out_dir}/")


if __name__ == "__main__":
    main()
