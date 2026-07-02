# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
import re
from functools import reduce

import mmcv
import numpy as np
import torch

from mmedit.datasets.pipelines import Compose

import math

VIDEO_EXTENSIONS = ('.mp4', '.mov')

import time

def pad_sequence(data, window_size):
    padding = window_size // 2

    data = torch.cat([
        data[:, 1 + padding:1 + 2 * padding].flip(1), data,
        data[:, -1 - 2 * padding:-1 - padding].flip(1)
    ],
                     dim=1)

    return data


def restoration_video_inference(model,
                                img_dir,
                                window_size,
                                start_idx,
                                filename_tmpl,
                                max_seq_len=None):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img_dir (str): Directory of the input video.
        window_size (int): The window size used in sliding-window framework.
            This value should be set according to the settings of the network.
            A value smaller than 0 means using recurrent framework.
        start_idx (int): The index corresponds to the first frame in the
            sequence.
        filename_tmpl (str): Template for file name.
        max_seq_len (int | None): The maximum sequence length that the model
            processes. If the sequence length is larger than this number,
            the sequence is split into multiple segments. If it is None,
            the entire sequence is processed at once.

    Returns:
        Tensor: The predicted restoration result.
    """

    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get('demo_pipeline', None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get('test_pipeline', None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # check if the input is a video
    file_extension = osp.splitext(img_dir)[1]
    if file_extension in VIDEO_EXTENSIONS:
        video_reader = mmcv.VideoReader(img_dir)
        # load the images
        data = dict(lq=[], lq_path=None, key=img_dir)
        for frame in video_reader:
            data['lq'].append(np.flip(frame, axis=2))

        # remove the data loading pipeline
        tmp_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] not in [
                    'GenerateSegmentIndices', 'LoadImageFromFileList'
            ]:
                tmp_pipeline.append(pipeline)
        test_pipeline = tmp_pipeline
    else:
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
        img_dir_split = re.split(r'[\\/]', img_dir)
        key = img_dir_split[-1]
        lq_folder = reduce(osp.join, img_dir_split[:-1])
        data = dict(
            lq_path=lq_folder,
            gt_path='',
            key=key,
            sequence_length=sequence_length)

    # compose the pipeline
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['lq'].unsqueeze(0)  # in cpu
    
    cnt = 0
    time_all = 0
    # forward the model
    with torch.no_grad():
        if window_size > 0:  # sliding window framework
            data = pad_sequence(data, window_size)
            result = []
            for i in range(0, data.size(1) - 2 * (window_size // 2)):
                data_i = data[:, i:i + window_size].to(device)
                result.append(model(lq=data_i, test_mode=True)['output'].cpu())
            result = torch.stack(result, dim=1)
        else:  # recurrent framework
            if max_seq_len is None:
                start = time.time()
                
                tile_size = 1024
                tile_pad = 32
                
                lq=data.to(device)
                batch, length, channel, height, width = lq.shape
                output_height = height * 4
                output_width = width * 4
                output_shape = (batch, length, 9, output_height, output_width)
                result = lq.new_zeros(output_shape).cpu()
                tiles_x = math.ceil(width / tile_size)
                tiles_y = math.ceil(height / tile_size)
                
                # loop over all tiles
                for y in range(tiles_y):
                    for x in range(tiles_x):
                # extract tile from input image
                        ofs_x = x * tile_size
                        ofs_y = y * tile_size
                # input tile area on total image
                        input_start_x = ofs_x
                        input_end_x = min(ofs_x + tile_size, width)
                        input_start_y = ofs_y
                        input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                        input_start_x_pad = max(input_start_x - tile_pad, 0)
                        input_end_x_pad = min(input_end_x + tile_pad, width)
                        input_start_y_pad = max(input_start_y - tile_pad, 0)
                        input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                        input_tile_width = input_end_x - input_start_x
                        input_tile_height = input_end_y - input_start_y
                        tile_idx = y * tiles_x + x + 1
                        input_tile = lq[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                        result_tile = model(lq=input_tile, test_mode=True)['output'].cpu()

                # output tile area on total image
                        output_start_x = input_start_x * 4
                        output_end_x = input_end_x * 4
                        output_start_y = input_start_y * 4
                        output_end_y = input_end_y * 4

                # output tile area without padding
                        output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                        output_end_x_tile = output_start_x_tile + input_tile_width * 4
                        output_start_y_tile = (input_start_y - input_start_y_pad) * 4
                        output_end_y_tile = output_start_y_tile + input_tile_height * 4  

                # put tile into output image
                        result[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[:, :, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

                
                #result = model(
                    #lq=data.to(device), test_mode=True)['output'].cpu()
                
                elapsed = (time.time() - start)
                time_all += elapsed
                cnt += 1
                print('{:.4f}'.format(time_all))
                print(cnt)
            else:
                result_list = []
                for i in range(0, data.size(1), max_seq_len):              
                    tile_size = 512
                    tile_pad = 32
                
                    lq=data[:, i:i + max_seq_len].to(device)
                    batch, length, channel, height, width = lq.shape
                    output_height = height * 4
                    output_width = width * 4
                    output_shape = (batch, length, 9, output_height, output_width)
                    result = lq.new_zeros(output_shape).cpu()
                    tiles_x = math.ceil(width / tile_size)
                    tiles_y = math.ceil(height / tile_size)
                
                # loop over all tiles
                    for y in range(tiles_y):
                        for x in range(tiles_x):
                # extract tile from input image
                            ofs_x = x * tile_size
                            ofs_y = y * tile_size
                # input tile area on total image
                            input_start_x = ofs_x
                            input_end_x = min(ofs_x + tile_size, width)
                            input_start_y = ofs_y
                            input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                            input_start_x_pad = max(input_start_x - tile_pad, 0)
                            input_end_x_pad = min(input_end_x + tile_pad, width)
                            input_start_y_pad = max(input_start_y - tile_pad, 0)
                            input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                            input_tile_width = input_end_x - input_start_x
                            input_tile_height = input_end_y - input_start_y
                            tile_idx = y * tiles_x + x + 1
                            input_tile = lq[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                            print(input_tile.shape)
                            result_tile = model(lq=input_tile, test_mode=True)['output'].cpu()

                # output tile area on total image
                            output_start_x = input_start_x * 4
                            output_end_x = input_end_x * 4
                            output_start_y = input_start_y * 4
                            output_end_y = input_end_y * 4

                # output tile area without padding
                            output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                            output_end_x_tile = output_start_x_tile + input_tile_width * 4
                            output_start_y_tile = (input_start_y - input_start_y_pad) * 4
                            output_end_y_tile = output_start_y_tile + input_tile_height * 4  

                # put tile into output image
                            result[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = result_tile[:, :, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
                    
                    result_list.append(result)
                result = torch.cat(result_list, dim=1)

    return result[:, :, 3:6, :, :]
