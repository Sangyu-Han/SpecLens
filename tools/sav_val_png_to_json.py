# tools/sav_val_png_to_json.py
import argparse, json, os
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm

try:
    import pycocotools.mask as mask_utils
except ImportError:
    raise SystemExit("pip install pycocotools 필요")

def encode_rle(bin_mask: np.ndarray):
    rle = mask_utils.encode(np.asfortranarray(bin_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default='/data/SA-V/sav_val')
    ap.add_argument("--out", default='/data/SA-V/sav_val_json')
    ap.add_argument("--split_list", default="sav_val.txt")
    ap.add_argument("--ann_every", type=int, default=4)  # 24fps→6fps
    args = ap.parse_args()

    root = Path(args.root)
    img_root = root / "JPEGImages_24fps"
    gt_root  = root / "Annotations_6fps"
    vid_list_f = root / args.split_list
    vids = [x.strip() for x in open(vid_list_f)] if vid_list_f.exists() else \
            sorted([p.name for p in img_root.iterdir() if p.is_dir()])

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for vid in tqdm(vids, desc="convert"):
        frames = sorted((img_root/vid).glob("*.jpg"))
        if not frames: 
            print(f"[skip] no frames: {vid}"); continue
        H, W = cv2.imread(str(frames[0])).shape[:2]
        T = len(frames)

        # 객체 디렉토리 수집
        obj_dirs = sorted([p for p in (gt_root/vid).glob("*") if p.is_dir()])
        obj_ids = [int(p.name) for p in obj_dirs]  # [0,1,2,...]
        obj_id_to_idx = {oid:i for i,oid in enumerate(obj_ids)}
        O = len(obj_ids)

        # 주석이 있는 실제 프레임 인덱스들(24fps)의 합집합
        ann_fidxs = sorted({int(p.stem) for d in obj_dirs for p in d.glob("*.png")})
        if not ann_fidxs:
            print(f"[warn] no masks: {vid}")
            continue

        # 6fps 인덱스 길이
        N = max(ann_fidxs)//args.ann_every + 1

        # masklet[i][j] = i번째(6fps) 프레임의 j번째 객체 RLE (없으면 None)
        masklet = [[None]*O for _ in range(N)]

        # 객체별 프레임 카운트
        per_obj_count = [0]*O

        for obj_dir in obj_dirs:
            j = obj_id_to_idx[int(obj_dir.name)]
            for png in sorted(obj_dir.glob("*.png")):
                fidx = int(png.stem)           # 24fps 인덱스(예: 0,4,8,…)
                i6 = fidx // args.ann_every    # 6fps 인덱스로 압축
                if i6 < 0 or i6 >= N: 
                    continue
                m = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if m is None: 
                    continue
                rle = encode_rle(m > 0)
                if masklet[i6][j] is None:
                    per_obj_count[j] += 1
                masklet[i6][j] = rle

        js = dict(
            video_id=vid,
            video_duration=float(T)/24.0,
            video_frame_count=T,
            video_height=H,
            video_width=W,
            video_resolution=H*W,
            video_environment="Unknown",  # 문자열로
            video_split="val",
            masklet=masklet,               # List[List[RLE or None]], 길이=N(6fps)
            masklet_id=obj_ids,            # 객체 id 리스트
            masklet_type=["manual"]*O,
            masklet_frame_count=per_obj_count,  # 객체별 카운트 배열
            masklet_num=O,
        )
        with open(out_dir/f"{vid}_manual.json", "w") as f:
            json.dump(js, f)

if __name__ == "__main__":
    main()