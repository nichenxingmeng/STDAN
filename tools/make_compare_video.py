"""Build comparison frames: LR (bicubic-upscaled) | SR, side by side.

Writes side-by-side PNGs to an output dir; ffmpeg then muxes them into mp4.
"""
import argparse
import os
from glob import glob

import cv2
import numpy as np


def label(img, text):
    cv2.rectangle(img, (0, 0), (10 + 22 * len(text), 60), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr-dir', required=True)
    ap.add_argument('--sr-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sr_files = sorted(glob(os.path.join(args.sr_dir, '*.png')))
    n = 0
    for sr_path in sr_files:
        name = os.path.basename(sr_path)
        lr_path = os.path.join(args.lr_dir, name)
        if not os.path.exists(lr_path):
            continue
        sr = cv2.imread(sr_path)
        lr = cv2.imread(lr_path)
        h, w = sr.shape[:2]
        lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        lr_up = label(lr_up.copy(), 'LR (bicubic x4)')
        sr_l = label(sr.copy(), 'STDAN x4')
        # thin white divider
        div = np.full((h, 6, 3), 255, np.uint8)
        combo = np.hstack([lr_up, div, sr_l])
        cv2.imwrite(os.path.join(args.out_dir, name), combo)
        n += 1
    print(f'wrote {n} side-by-side frames to {args.out_dir}')


if __name__ == '__main__':
    main()
