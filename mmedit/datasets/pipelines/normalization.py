# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from ..registry import PIPELINES

from torch.nn.modules.utils import _pair
import torch
import math

@PIPELINES.register_module()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys" and these keys with postfix '_norm_cfg'.
    It also supports normalizing a list of images.

    Args:
        keys (Sequence[str]): The images to be normalized.
        mean (np.ndarray): Mean values of different channels.
        std (np.ndarray): Std values of different channels.
        to_rgb (bool): Whether to convert channels from BGR to RGB.
    """

    def __init__(self, keys, mean, std, to_rgb=False, save_original=False):
        self.keys = keys
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.save_original = save_original

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                if self.save_original:
                    results[key + '_unnormalised'] = [
                        v.copy() for v in results[key]
                    ]
                results[key] = [
                    mmcv.imnormalize(v, self.mean, self.std, self.to_rgb)
                    for v in results[key]
                ]
            else:
                if self.save_original:
                    results[key + '_unnormalised'] = results[key].copy()
                results[key] = mmcv.imnormalize(results[key], self.mean,
                                                self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, mean={self.mean}, std={self.std}, '
                     f'to_rgb={self.to_rgb})')

        return repr_str

def genERP(j, N):
    val = math.pi / N
    w = math.cos((j - (N / 2) + 0.5) * val)

    return w

def compute_map_ws(H, W):
    equ = np.zeros((H, W))

    for i in range(0, equ.shape[0]):
        for j in range(0, equ.shape[1]):
            equ[i, j] = genERP(i, equ.shape[0])

    return equ

@PIPELINES.register_module()
class RescaleToZeroOne:
    """Transform the images into a range between 0 and 1.

    Required keys are the keys in attribute "keys", added or modified keys are
    the keys in attribute "keys".
    It also supports rescaling a list of images.

    Args:
        keys (Sequence[str]): The images to be transformed.
    """

    def __init__(self, keys):
        self.keys = keys
        # Position/latitude channels are generated on the fly from each frame's
        # own size (see __call__). Cache them per (H, W) so we build each map
        # only once.
        self._condition_cache = {}
        self._latitude_cache = {}

    def _get_condition(self, h, w):
        """cos-latitude OPE channel appended to the LQ frames (STCA)."""
        if (h, w) not in self._condition_cache:
            cond = get_condition(h, w, 'cos_latitude').reshape((h, w, 1))
            self._condition_cache[(h, w)] = np.array(cond)
        return self._condition_cache[(h, w)]

    def _get_latitude(self, h, w):
        """WS latitude weight channel appended to the GT frames (LSA loss)."""
        if (h, w) not in self._latitude_cache:
            self._latitude_cache[(h, w)] = np.expand_dims(
                compute_map_ws(h, w), axis=2)
        return self._latitude_cache[(h, w)]

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for key in self.keys:
            if isinstance(results[key], list):
                out = []
                for v in results[key]:
                    v = v.astype(np.float32) / 255.
                    h, w = v.shape[:2]
                    if key == 'lq':
                        extra = self._get_condition(h, w)
                    else:
                        extra = self._get_latitude(h, w)
                    out.append(np.concatenate((v, extra), axis=2))
                results[key] = out
            else:
                results[key] = results[key].astype(np.float32) / 255.
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

def get_condition(h, w, condition_type):
    if condition_type is None:
        return 0.
    elif condition_type == 'cos_latitude':
        return torch.cos(make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2,0,1) * math.pi / 2)
    elif condition_type == 'latitude':
        return make_coord([h]).unsqueeze(1).repeat([1, w, 1]).permute(2, 0, 1) * math.pi / 2
    elif condition_type == 'coord':
        return make_coord([h, w]).permute(2, 0, 1)
    else:
        raise RuntimeError('Unsupported condition type')

def make_coord(shape, ranges=(-1, 1), flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret