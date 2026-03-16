#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local SA-V frame extractor (no SLURM/submitit)

- 입력:  --sav-vid-dir 아래의 비디오 파일(.mp4 기본)
- 출력:  --output-dir/<video_stem>/{00000.jpg, 00004.jpg, ...}
         (파일명은 원본 프레임 인덱스를 5자리 zero-pad로 유지)
- 샘플링: --sav-frame-sample-rate N => 매 N번째 프레임만 저장 (서브샘플)
- 삭제:  --delete-after 를 주면 성공 저장 후 원본 비디오 삭제

사용 예:
  python sav_frame_extraction_local.py \
    --sav-vid-dir /data/SA-V/sav_000/sav_train \
    --output-dir /data/sav_frames/JPEGImages_sav \
    --sav-frame-sample-rate 1 \
    --n-workers 8 \
    --recursive \
    --delete-after
"""

import argparse
import os
from pathlib import Path
from typing import List
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        description="[SA-V Preprocessing | Local] Extract JPEG frames without SLURM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # DATA
    p.add_argument(
        "--sav-vid-dir", type=str, required=True,
        help="Root directory containing videos",
    )
    p.add_argument(
        "--sav-frame-sample-rate", type=int, default=4,
        help="Sub-sample rate (N): save 1 frame every N frames",
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="Recursively search for videos under sav-vid-dir (rglob). "
             "If not set, only matches one-level deep like */*.mp4",
    )
    p.add_argument(
        "--exts", type=str, default=".mp4",
        help="Comma-separated video extensions to include (e.g., .mp4,.mov,.mkv)",
    )
    # OUTPUT
    p.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to dump extracted JPEG frames",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=90,
        help="OpenCV JPEG quality (1-100)",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if output folder exists",
    )
    p.add_argument(
        "--delete-after", action="store_true",
        help="Delete the source video after successful extraction (safe: only if >=1 frame written and no write errors)",
    )
    # EXECUTION
    p.add_argument(
        "--n-workers", type=int, default=8,
        help="Parallel workers (processes)",
    )
    return p.parse_args()


def list_videos(root: Path, recursive: bool, exts: List[str]) -> List[Path]:
    if recursive:
        paths = []
        for ext in exts:
            paths += sorted(root.rglob(f"*{ext}"))
        return paths
    else:
        # match only one level deep: */*.ext (mimic original glob behavior)
        paths = []
        for ext in exts:
            paths += sorted(root.glob(f"*/*{ext}"))
        return paths


def extract_one(
    video_path: Path,
    save_root: Path,
    sample_rate: int,
    jpeg_quality: int,
    overwrite: bool,
    delete_after: bool,
) -> str:
    out_dir = save_root / video_path.stem
    if out_dir.exists() and not overwrite:
        # Skip if already extracted (has any jpg)
        if any(out_dir.glob("*.jpg")):
            return f"[SKIP] {video_path} (exists)"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"[ERROR] Cannot open {video_path}"

    # Stream frames; write when i % sample_rate == 0
    i = 0
    wrote = 0
    write_fail = 0

    ok, frame = cap.read()
    while ok:
        if i % sample_rate == 0:
            out_name = f"{i:05d}.jpg"  # preserve original index like fid*sample_rate
            out_path = out_dir / out_name
            ok_write = cv2.imwrite(
                str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok_write:
                wrote += 1
            else:
                write_fail += 1
        i += 1
        ok, frame = cap.read()

    cap.release()

    msg = (
        f"[DONE] {video_path} -> {out_dir} "
        f"(written={wrote}, fails={write_fail}, sample_rate={sample_rate})"
    )

    # Safe delete: only when requested AND at least 1 frame written AND no write errors
    if delete_after and wrote > 0 and write_fail == 0:
        try:
            os.remove(video_path)
            msg += " [DELETED SOURCE VIDEO]"
        except Exception as e:
            msg += f" [DELETE FAILED: {e}]"

    return msg


def main():
    args = parse_args()
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    try:
        # Avoid OpenCV oversubscription in multi-proc
        cv2.setNumThreads(0)
    except Exception:
        pass

    root = Path(args.sav_vid_dir)
    save_root = Path(args.output_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]

    videos = list_videos(root, args.recursive, exts)
    if not videos:
        print(f"No videos found under {root} (recursive={args.recursive}, exts={exts})")
        return

    print(f"Found {len(videos)} videos under {root}")
    worker = partial(
        extract_one,
        save_root=save_root,
        sample_rate=args.sav_frame_sample_rate,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
        delete_after=args.delete_after,
    )

    # Use processes to bypass GIL for decode
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(worker, v): v for v in videos}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            msg = fut.result()
            print(msg)

    print(f"All done. Outputs at: {save_root}")


if __name__ == "__main__":
    main()
