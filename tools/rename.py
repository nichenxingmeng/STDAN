"""Shift frame indices of every sequence in a folder down by one.

Renames ``0001.png .. 0100.png`` to ``0000.png .. 0099.png`` inside each
sub-folder of ``directory``. Useful when frames are exported 1-indexed but
the dataset expects 0-indexed names.

Usage:
    python rename.py --directory /path/to/LR
"""
import argparse
import os


def rename_files(directory, num_frames=100):
    for image_name in os.listdir(directory):
        seq_dir = os.path.join(directory, image_name)
        if not os.path.isdir(seq_dir):
            continue
        for i in range(1, num_frames + 1):
            src = os.path.join(seq_dir, '{:04d}.png'.format(i))
            dst = os.path.join(seq_dir, '{:04d}.png'.format(i - 1))
            if os.path.exists(src):
                os.rename(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True,
                        help='folder containing one sub-folder per sequence')
    parser.add_argument('--num-frames', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rename_files(args.directory, args.num_frames)
