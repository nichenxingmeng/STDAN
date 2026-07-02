# Configs

- [`stdan_odi_360vsr.py`](stdan_odi_360vsr.py) — STDAN training/testing config for
  360° video super-resolution. It uses the `SRODIMultipleGTDataset`, the
  saliency-aware frame loader (`LoadImageFromFileList_saliency`), and the
  saliency + latitude weighted Charbonnier loss.

Edit the `data/360Video/...` paths and the `saliency_folder` field in the config
to point at your dataset. See the top-level [README](../README.md) for details.
