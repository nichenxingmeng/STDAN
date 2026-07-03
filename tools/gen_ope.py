"""Generate the OPE (omni-positional encoding) maps for STDAN.

The STDAN backbone reads a 4th LQ channel as the OPE positional cue
(`condition = lqs[:, :, 3]`) for the STCA spatial alignment. Following the
paper, the vertical component is a per-latitude cosine weight — the same
equirectangular (ERP) weighting used by WS-PSNR:

    w(j) = cos((j - H/2 + 0.5) * pi / H)      for row j of an H-row frame

This script mirrors the LR frame folder layout: for every LR frame
`{lr_dir}/{clip}/{frame}.png` it writes a single-channel grayscale map
`{out_dir}/{clip}/{frame}.png` holding `w(j)` scaled to 0..255. Maps of the
same height are identical, so they are cached and reused across frames.

The OPE maps are needed for BOTH training and inference (unlike the
training-only saliency maps), because STCA uses them in the forward pass.

Usage:
    python tools/gen_ope.py --lr-dir data/360Video/training/LR_BIx4 \
                            --out-dir data/360Video/ope
"""
import argparse
import math
import os
from glob import glob

import cv2
import numpy as np


def genERP(j, N):
    """Equirectangular per-latitude cosine weight for row ``j`` of ``N`` rows."""
    return math.cos((j - (N / 2) + 0.5) * math.pi / N)


def build_ope_map(height, width):
    """Build the H x W OPE map (uint8, 0..255) for a given frame size.

    The cosine weight is in [0, 1]; we scale to [0, 255] and store as an
    8-bit grayscale image. `RescaleToZeroOne` in the pipeline maps it back
    to [0, 1] at load time.
    """
    col = np.array([genERP(j, height) for j in range(height)], dtype=np.float64)
    col = np.clip(col, 0.0, 1.0)  # cos is >= 0 over the ERP latitude range
    ope = np.repeat(col[:, None], width, axis=1)
    return (ope * 255.0).round().astype(np.uint8)


def generate(lr_dir, out_dir, exts):
    cache = {}
    count = 0
    for clip in sorted(os.listdir(lr_dir)):
        clip_dir = os.path.join(lr_dir, clip)
        if not os.path.isdir(clip_dir):
            continue
        frames = []
        for ext in exts:
            frames += glob(os.path.join(clip_dir, f'*.{ext}'))
        for frame_path in sorted(frames):
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'  [warn] cannot read {frame_path}, skipped')
                continue
            h, w = img.shape[:2]
            if (h, w) not in cache:
                cache[(h, w)] = build_ope_map(h, w)
            ope = cache[(h, w)]

            rel = os.path.relpath(frame_path, lr_dir)
            dst = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, ope)
            count += 1
            if count % 500 == 0:
                print(f'  {count} maps written')
    print(f'Done. {count} OPE maps written to {out_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr-dir', required=True,
                        help='LR frame root; one sub-folder per sequence')
    parser.add_argument('--out-dir', required=True,
                        help='output root for OPE maps (mirrors --lr-dir)')
    parser.add_argument('--exts', nargs='+', default=['png'],
                        help='frame file extensions to look for')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate(args.lr_dir, args.out_dir, args.exts)
