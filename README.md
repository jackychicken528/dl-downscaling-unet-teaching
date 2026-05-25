# WRF U-Net Downscaling Teaching Bundle

This folder is self-contained for the teaching notebooks. Open the notebooks
from this directory so their local imports and relative paths resolve together.

## Contents

- `WRF_UNet_Training_Testing_Colab.ipynb` - small training, testing, and
  inference workflow.
- `WRF_Data_Preprocessing_Colab.ipynb` - four-part walkthrough of grid
  matching, data transformation, residual targets, and normalization.
- `dataset.py` - stats loading, yearly NPZ weather loading, static-channel
  loading, time features, normalization, and residual reconstruction.
- `Network.py` - U-Net model used by the training/testing/inference notebook.
- `config.py` - compact teaching configuration, including normalization stats.
- `utils.py` - plotting and output helpers.
- `data/wrf_npz_coarse/` - coarse yearly NPZ files.
- `data/wrf_npz_fine/` - fine yearly NPZ files.
- `data/wrf_npz_static/static_d02.npz` - static channel file expected by the
  notebooks.
- `outputs/` - checkpoints and training plots are written here.
- `data/wrf_npz_predicted/` - inference predictions are written here.
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

The heavy monthly NetCDF files are not copied here because these notebooks use
the smaller yearly NPZ files.
