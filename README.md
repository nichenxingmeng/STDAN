# STDAN: Spatio-Temporal Distortion Aware Omnidirectional Video Super-Resolution

Official code for the AAAI 2026 paper
**"Spatio-Temporal Distortion Aware Omnidirectional Video Super-Resolution"**.

> Hongyu An, Xinfeng Zhang, Shijie Zhao, Li Zhang, Ruiqin Xiong.
> *University of Chinese Academy of Sciences · ByteDance Inc. · Peking University.*

Omnidirectional videos (ODVs / 360° videos) suffer from spatial projection
distortions and temporal flickering that make general video super-resolution (VSR)
methods ineffective. STDAN is a Spatio-Temporal Distortion Aware Network with joint
spatio-temporal alignment and reconstruction, comprising:

- **STCA** — Spatio-Temporal Continuous Alignment, to mitigate discrete geometric
  artifacts alongside temporal alignment.
- **IMFR** — Interlaced Multi-Frame Reconstruction, to enhance temporal consistency.
- **LSA** — Latitude-Saliency Adaptive weighting, focusing reconstruction on regions
  of higher texture complexity and human viewing interest.

The codebase is built on [MMEditing](https://github.com/open-mmlab/mmediting) /
[BasicVSR++](https://arxiv.org/abs/2104.13371) (see [Acknowledgements](#acknowledgements)).

## Demo

<p align="center">
  <img src="assets/0.gif" width="45%"/>
  <img src="assets/7.gif" width="45%"/><br/>
  <img src="assets/8.gif" width="45%"/>
  <img src="assets/14.gif" width="45%"/>
</p>

## ODV-SR dataset

The proposed ODV-SR **training** set has two parts: 210 sequences following
[ODV360](https://github.com/360SR/360SR-Challenge), and 70 sequences collected by us
([Baidu Drive](https://pan.baidu.com/s/1SEREx1hjqFHXQ0lTmmxGow), access code: `tiw7`).

The proposed ODV-SR **test** set:
[Google Drive](https://drive.google.com/drive/folders/1jhrDFtKBBpvRjqOhLtVXil7zdJ0vV0kT?usp=sharing).

## What's in this repository

All three paper components are implemented inside the recurrent backbone (they are
woven into the BasicVSR++-style propagation/reconstruction under generic module names
rather than files literally named STCA/IMFR):

| Component | Where it lives |
| --- | --- |
| **STCA** — spatio-temporal continuous alignment (OPE-conditioned spatial DCN on top of second-order deformable temporal alignment) | `offset_conv` + `DCNv2Pack` in [`basicvsr_pp.py`](mmedit/models/backbones/sr_backbones/basicvsr_pp.py), applied in `propagate()`; the OPE positional cue is the 4th LQ channel read at `forward()` (`condition = lqs[:, :, 3]`), loaded by [`LoadImageFromFileList_ope`](mmedit/datasets/pipelines/loading.py) |
| **IMFR** — interlaced multi-frame reconstruction | the 3-way interlaced path in `upsample()` of [`basicvsr_pp.py`](mmedit/models/backbones/sr_backbones/basicvsr_pp.py) (`reconstruction` takes `3*5*mid_channels`, `conv_last` emits `3*3` channels; prev/cur/next residual), consumed by `pred.chunk(3)` in the loss |
| **LSA** — latitude + saliency weighted loss | [`charbonnier_loss`](mmedit/models/losses/pixelwise_loss.py) + saliency loader [`LoadImageFromFileList_saliency`](mmedit/datasets/pipelines/loading.py) |
| ODV-SR dataset (multi-GT, recurrent) | [`mmedit/datasets/sr_ODI_multiple_gt_dataset.py`](mmedit/datasets/sr_ODI_multiple_gt_dataset.py) |
| WS-PSNR / WS-SSIM evaluation | [`tools/eval/ws_psnr.py`](tools/eval/ws_psnr.py) |
| Training config | [`configs/stdan_odi_360vsr.py`](configs/stdan_odi_360vsr.py) |

> **On the OPE (omni-positional encoding) channel.** The STCA module needs a
> **4-channel LQ input** — RGB plus a single-channel OPE map that the backbone reads
> as `condition = lqs[:, :, 3]`. The OPE maps (cosine-based vertical PE + sinusoidal
> horizontal PE, per the paper) are **precomputed offline** and stored the same way
> as the frames; they are loaded and concatenated as the 4th LQ channel by
> [`LoadImageFromFileList_ope`](mmedit/datasets/pipelines/loading.py), which is
> already referenced in the config's pipelines. Put the maps under
> `data/360Video/ope/` mirroring the LR folder layout (see below). **TODO: the
> offline OPE-generation script will be added here.**

## Installation

1. Install [PyTorch](https://pytorch.org).
2. `pip install openmim`
3. `mim install mmcv-full`
4. `git clone <this-repo> && cd STDAN`
5. `pip install -v -e .`

## Data preparation

The configs expect data under `data/360Video/` with this layout (paths are set in
[`configs/stdan_odi_360vsr.py`](configs/stdan_odi_360vsr.py) and can be changed there):

```
data/360Video/
├── training/
│   ├── LR_BIx4/0000/0000.png ...     # low-resolution frames
│   └── HR/0000/0000.png ...          # ground-truth frames
├── validation/{LR,HR}/...
├── testing/{LR,HR}/...
├── saliency/0000/0000.png ...        # grayscale saliency maps, mirror of HR/
└── ope/0000/0000.png ...             # grayscale OPE maps, mirror of LR/
```

The `saliency_folder` / `ope_folder` (and the `path_split_token` for each) are set
in the config's pipelines — change them there if your layout differs.

Frames are 0-indexed `NNNN.png`. If your frames are 1-indexed, shift them with
[`tools/rename.py`](tools/rename.py):

```bash
python tools/rename.py --directory data/360Video/training/LR_BIx4
```

### Saliency maps

The LSA loss reads one grayscale saliency map per HR frame. The loader
([`LoadImageFromFileList_saliency`](mmedit/datasets/pipelines/loading.py)) maps an
HR frame path to its saliency map via the `saliency_folder` and `path_split_token`
config fields: the part of the HR path after `path_split_token` (default `'HR'`) is
appended to `saliency_folder`, so `.../HR/0000/0000.png` →
`{saliency_folder}/0000/0000.png`.

> **Note:** The procedure for generating the saliency maps will be documented here. (TODO)

## Training

```bash
sh tools/dist_train.sh configs/stdan_odi_360vsr.py ${NGPUS}
```

## Inference

```bash
python demo/restoration_video_demo.py \
    configs/stdan_odi_360vsr.py \
    stdan.ckpt \
    ${INPUT_PATH} \
    ${OUTPUT_PATH}
```

The pretrained checkpoint `stdan.ckpt` is **not tracked in git** (~330 MB).
Download it separately and place it at the repository root, or point the command at
your own path.

> **Download:** [Google Drive](https://drive.google.com/file/d/16nYbqbC3cyNkx6y22kOuLFuEcSPjjpvL/view?usp=sharing)

## Evaluation (WS-PSNR / WS-SSIM)

WS-PSNR / WS-SSIM (Weighted-to-Spherically-uniform metrics) are the standard fidelity
measures for equirectangular content.

```bash
python tools/eval/ws_psnr.py \
    --gt-dir   data/360Video/testing/HR \
    --pred-dir results/stdan_odi_360vsr \
    --num-seq  25 \
    --metric   all
```

Each of `--gt-dir` and `--pred-dir` should contain one sub-folder per sequence
(`0000`, `0001`, ...); frames are matched by file name.

## Acknowledgements

This work is built upon [MMEditing](https://github.com/open-mmlab/mmediting) and
[BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus). Please follow and
star those repositories.

## Citation

```bibtex
@inproceedings{an2026spatio,
  title={Spatio-temporal distortion aware omnidirectional video super-resolution},
  author={An, Hongyu and Zhang, Xinfeng and Zhao, Shijie and Zhang, Li and Xiong, Ruiqin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={4},
  pages={2309--2317},
  year={2026}
}
```

Please also cite BasicVSR++ and MMEditing:

```bibtex
@inproceedings{chan2022basicvsr++,
  title={Basicvsr++: Improving video super-resolution with enhanced propagation and alignment},
  author={Chan, Kelvin CK and Zhou, Shangchen and Xu, Xiangyu and Loy, Chen Change},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={5972--5981},
  year={2022}
}
```
