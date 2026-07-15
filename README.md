# STDAN: Spatio-Temporal Distortion Aware Omnidirectional Video Super-Resolution

Official code for the AAAI 2026 paper
**"Spatio-Temporal Distortion Aware Omnidirectional Video Super-Resolution"**.

> Hongyu An, Xinfeng Zhang, Shijie Zhao, Li Zhang, Ruiqin Xiong.
> *University of Chinese Academy of Sciences · ByteDance Inc. · Peking University.*
> [[Paper (arXiv)]](https://arxiv.org/abs/2410.11506)

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
| **STCA** — spatio-temporal continuous alignment (OPE-conditioned spatial DCN on top of second-order deformable temporal alignment) | `offset_conv` + `DCNv2Pack` in [`basicvsr_pp.py`](mmedit/models/backbones/sr_backbones/basicvsr_pp.py), applied in `propagate()`; the OPE positional cue is the 4th LQ channel read at `forward()` (`condition = lqs[:, :, 3]`), generated on the fly as a cos-latitude map by [`RescaleToZeroOne`](mmedit/datasets/pipelines/normalization.py) |
| **IMFR** — interlaced multi-frame reconstruction | the 3-way interlaced path in `upsample()` of [`basicvsr_pp.py`](mmedit/models/backbones/sr_backbones/basicvsr_pp.py) (`reconstruction` takes `3*5*mid_channels`, `conv_last` emits `3*3` channels; prev/cur/next residual), consumed by `pred.chunk(3)` in the loss |
| **LSA** — latitude + saliency weighted loss | [`charbonnier_loss`](mmedit/models/losses/pixelwise_loss.py) + saliency loader [`LoadImageFromFileList_saliency`](mmedit/datasets/pipelines/loading.py) |
| ODV-SR dataset (multi-GT, recurrent) | [`mmedit/datasets/sr_ODI_multiple_gt_dataset.py`](mmedit/datasets/sr_ODI_multiple_gt_dataset.py) |
| WS-PSNR / WS-SSIM evaluation | [`tools/eval/ws_psnr.py`](tools/eval/ws_psnr.py) |
| Training config | [`configs/stdan_odi_360vsr.py`](configs/stdan_odi_360vsr.py) |

> **On the OPE (omni-positional encoding) channel.** The STCA module reads a
> **4-channel LQ input** — RGB plus a single-channel OPE map that the backbone reads
> as `condition = lqs[:, :, 3]`. This OPE map is the per-latitude ERP cosine weight
> (the same weighting used by WS-PSNR) and is generated **on the fly** from each
> frame's own size by [`RescaleToZeroOne`](mmedit/datasets/pipelines/normalization.py) —
> no external files or preprocessing are needed. The GT frames similarly get a
> latitude-weight channel appended for the LSA loss during training.

## Installation

STDAN builds on MMEditing 0.14, which needs the `mmcv-full` 1.x API (the CUDA
deformable-convolution op used by STCA). The following combination is verified
to run inference end-to-end on an NVIDIA A6000 (CUDA 11.8):

```bash
# Python 3.10 environment (conda / venv / uv all fine)
pip install "numpy<2" opencv-python-headless imageio scipy addict "yapf==0.32.0"

# PyTorch 2.0.1 + CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# prebuilt mmcv-full wheel for torch2.0 / cu118 (source builds are slow/fragile)
pip install "mmcv-full==1.7.2" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# STDAN itself
git clone <this-repo> && cd STDAN
pip install -v -e .
```

Notes:
- Use **Python 3.10** — the old torch / mmcv-full 1.x wheels are not published
  for Python 3.13.
- Keep **numpy < 2** — the mmcv wheel may pull in numpy 2.x, which breaks the
  torch 2.0 ABI and the `np.bool8` usage in the old mmedit code.
- Pin **yapf 0.32.0** — newer yapf ships a lib2to3 grammar cache that mmcv 1.7's
  config module can fail to load (`EOFError: Ran out of input`).
- For other CUDA/torch versions, pick the matching prebuilt wheel from the
  [OpenMMLab index](https://download.openmmlab.com/mmcv/dist/index.html).

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
└── saliency/0000/0000.png ...        # grayscale saliency maps, mirror of HR/ (training only)
```

The `saliency_folder` (and its `path_split_token`) is set in the training
pipeline — change it there if your layout differs. The OPE position channel is
generated on the fly, so it needs no folder.

Frames are 0-indexed `NNNN.png`. If your frames are 1-indexed, shift them with
[`tools/rename.py`](tools/rename.py):

```bash
python tools/rename.py --directory data/360Video/training/LR_BIx4
```

### Saliency maps

> **Training only.** The saliency maps are used *only* by the LSA loss during
> training. They are **not** needed for inference — the LSA weighting is
> precomputed offline and adds zero cost at test time, so you do not need saliency
> maps to run the demo / evaluation.

The LSA loss reads one grayscale saliency map per HR frame. The loader
([`LoadImageFromFileList_saliency`](mmedit/datasets/pipelines/loading.py)) maps an
HR frame path to its saliency map via the `saliency_folder` and `path_split_token`
config fields: the part of the HR path after `path_split_token` (default `'HR'`) is
appended to `saliency_folder`, so `.../HR/0000/0000.png` →
`{saliency_folder}/0000/0000.png`.

**How the maps are generated (offline, once, before training).** Following the
paper, each HR frame's saliency map is estimated with an off-the-shelf 360°
saliency predictor and saved as a grayscale image:

1. Project the ERP frame to a **cubemap** (6 faces).
2. Run the **SalEMA** 360° video saliency predictor
   ([Linardos et al. 2019](https://github.com/Linardos/SalEMA); the paper uses a
   two-branch variant that combines a global-context attention branch with a
   local-viewpoint projection branch, multiplying the two to obtain the
   saliency-oriented weight `W_sal(u, v)`).
3. **Inverse-project** the per-face saliency back to ERP and save it as a grayscale
   PNG mirroring the HR folder layout, under `saliency_folder`.

This is a preprocessing step done once over the training set; it is independent of
this repository (it reuses SalEMA), so no saliency-generation script is bundled
here. Point `saliency_folder` in the config at the resulting maps.

## Training

```bash
sh tools/dist_train.sh configs/stdan_odi_360vsr.py ${NGPUS}
```

## Inference

Inference takes a **folder of frames** (`0000.png`, `0001.png`, ...). Encoded
video files (`.mp4`/`.mov`) are not supported — extract them to a frame folder
first. The OPE position channel is added automatically, so there is no
preprocessing step.

```bash
python demo/restoration_video_demo.py \
    configs/stdan_odi_360vsr.py \
    stdan.ckpt \
    ${INPUT_PATH} \
    ${OUTPUT_PATH}
```

`${INPUT_PATH}` is the clip folder; `${OUTPUT_PATH}` is an output folder (or an
`.mp4` path to write a video). Use `--max-seq-len` to bound memory on long
sequences.

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
