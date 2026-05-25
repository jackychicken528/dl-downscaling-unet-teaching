# WRF U-Net Downscaling Teaching Bundle

This folder is self-contained for the teaching notebooks. Open the notebooks
from this directory so their local imports and relative paths resolve together.

## Contents

- `WRF_UNet_Training_Testing_Colab.ipynb` - merged preprocessing, training,
  testing, and inference workflow. The notebook writes model-ready train/test
  NPZ files before training, then loads them with an inline lightweight dataset.
- `Network.py` - U-Net model used by the training/testing/inference notebook.
- `config.py` - compact teaching configuration, including normalization stats.
- `plot_utils.py` - plotting helpers used by the notebook.
- `data/wrf_train_coarse_raw.npz` and `data/wrf_train_fine_raw.npz` - raw
  training fields.
- `data/wrf_test_coarse_raw.npz` and `data/wrf_test_fine_raw.npz` - raw testing
  fields.
- `data/wrf_static_raw.npz` - raw static channel file expected by the notebook.
- `data/wrf_train_processed.npz` and `data/wrf_test_processed.npz` - generated
  model-ready files written by the notebook.
- `outputs/` - checkpoints, training plots, and prediction NPZ files are written
  here.
- `backup/WRF_UNet_Inference_Colab.ipynb` - previous standalone inference
  workflow kept for reference.
- `backup/means_stds.json` - original raw weather normalization statistics.
- `backup/means_stds_residual_fine_coarse.json` - original residual
  normalization statistics.

## Notes

The original workspace did not include real `data/static/` source files. To keep
the notebooks runnable without reaching outside this folder, `static_d02.npz`
contains placeholder `LSM` and `HGT` channels on the d02 grid. Replace this file
with real static fields when teaching terrain and land-sea-mask effects.

The heavy monthly NetCDF files are not copied here because this notebook uses
the smaller predefined train/test NPZ files.
