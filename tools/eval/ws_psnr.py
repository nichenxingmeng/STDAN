"""Weighted-to-Spherically-uniform PSNR / SSIM evaluation for 360 video.

WS-PSNR and WS-SSIM weight the per-pixel error by the area each pixel
occupies on the sphere (equirectangular projection), which is the standard
fidelity metric for omnidirectional (360-degree) content.

Usage:
    python ws_psnr.py --gt-dir GT --pred-dir PRED [--num-seq N] [--metric all]

The GT and prediction folders are expected to hold one sub-folder per
sequence (named ``0000``, ``0001``, ... by default), each containing the
frames as ``*.png``. Frames are matched by file name.
"""
import argparse
import math
import os
from glob import glob

import numpy as np
from imageio.v2 import imread

from ws_ssim import ws_ssim


def genERP(j, N):
    """Equirectangular weight for row ``j`` of an image with ``N`` rows."""
    val = math.pi / N
    return math.cos((j - (N / 2) + 0.5) * val)


def compute_map_ws(height, width):
    """Build the spherical weighting map for a ``height x width`` frame."""
    equ = np.zeros((1, height, width))
    for j in range(height):
        equ[0, j, :] = genERP(j, height)
    return equ


def getGlobalWSMSEValue(mx, my, mw):
    val = np.sum(np.multiply((mx - my) ** 2, mw))
    return val / np.sum(mw)


def rgb2y(input_im):
    """ITU-R style luma: Y = 0.257 R + 0.507 G + 0.098 B + 16.

    Returns a (H, W) float array (the WS-MSE uses only the Y channel).
    """
    im = np.asarray(input_im, dtype=np.float64)
    y = 0.257 * im[..., 0] + 0.507 * im[..., 1] + 0.098 * im[..., 2] + 16.0
    return y


def ws_psnr(image1, image2, mw):
    image1_y = rgb2y(image1)
    image2_y = rgb2y(image2)
    ws_mse = getGlobalWSMSEValue(image1_y, image2_y, mw)
    try:
        return 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        return np.inf


def evaluate(gt_dir, pred_dir, num_seq, seq_fmt, metric):
    total = {'ws_psnr': 0.0, 'ws_ssim': 0.0}
    count = 0
    mw = None

    for i in range(num_seq):
        seq = seq_fmt.format(i)
        gt_folder = os.path.join(gt_dir, seq)
        pred_folder = os.path.join(pred_dir, seq)
        if not os.path.isdir(gt_folder) or not os.path.isdir(pred_folder):
            continue

        for a_file in sorted(glob(os.path.join(gt_folder, '*.png'))):
            name = os.path.basename(a_file)
            pred_file = os.path.join(pred_folder, name)
            if not os.path.exists(pred_file):
                continue
            image1 = imread(a_file)
            image2 = imread(pred_file)
            if mw is None or mw.shape[1:] != image1.shape[:2]:
                mw = compute_map_ws(image1.shape[0], image1.shape[1])

            if metric in ('ws_psnr', 'all'):
                total['ws_psnr'] += ws_psnr(image1, image2, mw)
            if metric in ('ws_ssim', 'all'):
                total['ws_ssim'] += ws_ssim(image1, image2, mw[0])
            count += 1
            if count % 100 == 0:
                print(f'{count} frames processed')

    if count == 0:
        print('No matching frames found. Check --gt-dir / --pred-dir.')
        return

    if metric in ('ws_psnr', 'all'):
        print('ALL WS-PSNR:{:.4f}'.format(total['ws_psnr'] / count))
    if metric in ('ws_ssim', 'all'):
        print('ALL WS-SSIM:{:.4f}'.format(total['ws_ssim'] / count))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-dir', required=True,
                        help='folder with ground-truth sequences')
    parser.add_argument('--pred-dir', required=True,
                        help='folder with predicted/restored sequences')
    parser.add_argument('--num-seq', type=int, default=25,
                        help='number of sequences to iterate over')
    parser.add_argument('--seq-fmt', default='{:04d}',
                        help='format string for the sequence sub-folder name')
    parser.add_argument('--metric', default='all',
                        choices=['ws_psnr', 'ws_ssim', 'all'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.gt_dir, args.pred_dir, args.num_seq, args.seq_fmt,
             args.metric)
