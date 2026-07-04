# Copyright (c) OpenMMLab. All rights reserved.
import glob
import math
import os.path as osp

import torch

from mmedit.datasets.pipelines import Compose

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def _forward_tiled(model, lq, device, tile_size, tile_pad):
    """Run the model tile-by-tile and stitch the ×4 SR result.

    STDAN's IMFR head emits 9 channels per frame (interlaced prev/cur/next
    RGB); the restored current frame is channels 3:6. Tiling keeps the peak
    memory bounded for high-resolution 360 frames. ``tile_size <= 0`` (or a
    tile larger than the frame) processes the whole frame in one pass.
    """
    batch, length, channel, height, width = lq.shape
    out = lq.new_zeros((batch, length, 9, height * 4, width * 4)).cpu()

    if tile_size is None or tile_size <= 0:
        tiles_x = tiles_y = 1
        tile_size = max(height, width)
    else:
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

    for y in range(tiles_y):
        for x in range(tiles_x):
            in_sx = x * tile_size
            in_ex = min(in_sx + tile_size, width)
            in_sy = y * tile_size
            in_ey = min(in_sy + tile_size, height)

            # padded input tile
            in_sx_pad = max(in_sx - tile_pad, 0)
            in_ex_pad = min(in_ex + tile_pad, width)
            in_sy_pad = max(in_sy - tile_pad, 0)
            in_ey_pad = min(in_ey + tile_pad, height)

            tile_w = in_ex - in_sx
            tile_h = in_ey - in_sy

            in_tile = lq[:, :, :, in_sy_pad:in_ey_pad, in_sx_pad:in_ex_pad]
            out_tile = model(lq=in_tile.to(device), test_mode=True)['output']
            out_tile = out_tile.cpu()

            # place the unpadded region into the full-size output
            osx, oex = in_sx * 4, in_ex * 4
            osy, oey = in_sy * 4, in_ey * 4
            tsx = (in_sx - in_sx_pad) * 4
            tsy = (in_sy - in_sy_pad) * 4
            out[:, :, :, osy:oey, osx:oex] = out_tile[
                :, :, :, tsy:tsy + tile_h * 4, tsx:tsx + tile_w * 4]

    return out


def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None,
                                tile_size=0,
                                tile_pad=32):
    """Inference a video/image sequence with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input frames (one image per frame).
            Encoded video files (.mp4/.mov) are not supported; extract them to
            a folder of frames first.
        window_size (int): The window size used in sliding-window framework.
            A value <= 0 means using the recurrent framework.
        start_idx (int): The index of the first frame in the sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes at once. If the sequence is longer, it is split into
            segments. None processes the whole sequence at once.
        tile_size (int): Spatial tile size for the ×4 forward pass. 0 (or a
            value larger than the frame) processes the whole frame at once.
        tile_pad (int): Padding around each tile to avoid seams.

    Returns:
        Tensor: The predicted restoration result, shape (1, T, 3, 4H, 4W).
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # only frame folders (not encoded video files) are supported.
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        raise ValueError(
            'Encoded video-file input is not supported. Please extract the '
            'video to a folder of frames (0000.png, 0001.png, ...) and pass '
            'that folder instead.')

    # the first element in the pipeline must be 'GenerateSegmentIndices'
    if test_pipeline[0]['type'] != 'GenerateSegmentIndices':
        raise TypeError('The first element in the pipeline must be '
                        f'"GenerateSegmentIndices", but got '
                        f'"{test_pipeline[0]["type"]}".')

    # specify start_idx and filename_tmpl
    test_pipeline[0]['start_idx'] = start_idx
    test_pipeline[0]['filename_tmpl'] = filename_tmpl

    # prepare data
    sequence_length = len(glob.glob(osp.join(img_dir, '*')))
    # split into parent folder + clip name. Using dirname/basename (rather than
    # splitting on the separator and re-joining) keeps a leading '/' on
    # absolute paths.
    img_dir = img_dir.rstrip('/\\')
    key = osp.basename(img_dir)
    lq_folder = osp.dirname(img_dir)
    data = dict(
        lq_path=lq_folder,
        gt_path='',
        key=key,
        sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu

    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding-window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(0, data.size(1) - 2 * (window_size // 2)):
                data_i = data[:, i:i + window_size].to(device)
                out = model(lq=data_i, test_mode=True)['output'].cpu()
                result.append(out)
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                result = _forward_tiled(model, data, device, tile_size,
                                        tile_pad)
            else:
                result_list = []
                for i in range(0, data.size(1), max_seq_len):
                    seg = data[:, i:i + max_seq_len]
                    result_list.append(
                        _forward_tiled(model, seg, device, tile_size,
                                       tile_pad))
                result = torch.cat(result_list, dim=1)

    # STDAN's IMFR head emits 9 channels (interlaced prev/cur/next RGB);
    # the restored current frame is channels 3:6.
    return result[:, :, 3:6, :, :]
