"""Small Colab-friendly configuration for the teaching notebooks.

The original project config contains many paths and options for the full
research workflow. This copy keeps only the values used by the notebooks.
"""

from pathlib import Path


# Folder containing this teaching bundle.
PROJECT_DIR = str(Path(__file__).resolve().parent)

# Main data and output folders used by the notebooks.
DATA_DIR = f"{PROJECT_DIR}/data"
OUTPUT_DIR = f"{PROJECT_DIR}/outputs"

# Raw train/test files used by the preprocessing cells.
TRAIN_COARSE_RAW_NPZ_PATH = f"{DATA_DIR}/wrf_train_coarse_raw.npz"
TRAIN_FINE_RAW_NPZ_PATH = f"{DATA_DIR}/wrf_train_fine_raw.npz"
TEST_COARSE_RAW_NPZ_PATH = f"{DATA_DIR}/wrf_test_coarse_raw.npz"
TEST_FINE_RAW_NPZ_PATH = f"{DATA_DIR}/wrf_test_fine_raw.npz"
STATIC_NPZ_PATH = f"{DATA_DIR}/wrf_static_raw.npz"

# Processed train/test files created by the U-Net notebook.
TRAIN_PROCESSED_NPZ_PATH = f"{DATA_DIR}/wrf_train_processed.npz"
TEST_PROCESSED_NPZ_PATH = f"{DATA_DIR}/wrf_test_processed.npz"

# Model output paths.
BEST_CHECKPOINT_PATH = f"{OUTPUT_DIR}/unet_npz_best.pt"
LOSS_CURVE_PATH = f"{OUTPUT_DIR}/unet_loss_curve.png"

# Weather variables used as model inputs and outputs.
IN_OUT_PARAMS = ["T2", "TD2", "MSLP", "U10", "V10", "LN_ACC_6H_PRECIP"]

# Static variables are appended to the model input.
CONSTANT_PARAMS = ["LSM", "HGT"]

# Pressure levels are kept for compatibility with helper functions.
PRESSURE_LEVELS = [925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0]

# All coarse and fine fields are resized to this grid.
HIGH_RES_SHP = [48, 48]

# Training settings chosen to be small enough for a notebook demo.
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.00003

# Raw-field statistics used to normalize coarse model inputs.
RAW_STATS = {
    "mean": {
        "T2": 295.5968695903861,
        "TD2": 18.442357518912704,
        "MSLP": 1013.2178737452913,
        "U10": -1.2645954073956605,
        "V10": -0.007613408405587873,
        "LN_ACC_6H_PRECIP": -4.155085741457742,
    },
    "std": {
        "T2": 7.604735189400204,
        "TD2": 8.229461663124336,
        "MSLP": 7.777784548349449,
        "U10": 2.8270035106731903,
        "V10": 4.530224128381111,
        "LN_ACC_6H_PRECIP": 3.274468550630762,
    },
}

# Residual statistics used to normalize fine-minus-coarse targets.
RESIDUAL_STATS = {
    "mean": {
        "T2": 0.06438512355089188,
        "TD2": 0.057380132377147675,
        "MSLP": -0.020381242036819458,
        "U10": -0.03487271070480347,
        "V10": 0.007763346191495657,
        "LN_ACC_6H_PRECIP": -0.4649406969547272,
    },
    "std": {
        "T2": 0.714195966720581,
        "TD2": 0.6818221807479858,
        "MSLP": 0.24972990155220032,
        "U10": 0.6844680309295654,
        "V10": 0.720545768737793,
        "LN_ACC_6H_PRECIP": 1.1577991247177124,
    },
}

# Small U-Net settings for the teaching notebook.
MODEL_KWARGS = {
    "model_channels": 32,
    "channel_mult": [1, 2, 2],
    "num_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
}

# Latitude-longitude bounds for plotting the teaching domain.
DL_DOMAIN = {
    "max_lat": 24.75,
    "min_lat": 21.0,
    "min_lon": 112.25,
    "max_lon": 116.0,
}

# Colormaps used when plotting each weather variable.
PARAM_COLORMAP = {
    "T2": "coolwarm",
    "TD2": "YlGnBu",
    "MSLP": "viridis",
    "U10": "PuOr",
    "V10": "PuOr",
    "LN_ACC_6H_PRECIP": "custom_rainfall",
}

# Discrete rainfall levels in millimeters for the custom precipitation map.
RAINFALL_LEVELS = [
    0.0, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 25, 30,
    40, 50, 70, 100, 150, 200, 300, 400,
]

# Colors paired with the rainfall levels above.
RAINFALL_COLORS = [
    "#fefefe",
    "#e5e5e5",
    "#bfbfbf",
    "#defadd",
    "#b3f2b6",
    "#b3f2b6",
    "#fdfcaf",
    "#fbfab1",
    "#eed0a4",
    "#edd0a9",
    "#eea1a4",
    "#f49da6",
    "#a9a8e7",
    "#be9bed",
    "#a77ddd",
    "#b9189d",
    "#d42fba",
    "#e66cd1",
    "#f99bda",
]
