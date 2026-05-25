"""Small Colab-friendly configuration for the teaching notebooks.

The original project config contains many paths and options for the full
research workflow. This copy keeps only the values used by the notebooks.
"""

from pathlib import Path


PROJECT_DIR = str(Path(__file__).resolve().parent)
DATA_DIR = f"{PROJECT_DIR}/data"
OUTPUT_DIR = f"{PROJECT_DIR}/outputs"

DL_OUT_SINGLE_LAYER_PARAMS = ["T2", "TD2", "MSLP", "U10", "V10", "LN_ACC_6H_PRECIP"]
DL_OUT_UPPER_PARAMS = []

CONSTANT_PARAMS = ["LSM", "HGT"]
PRESSURE_LEVELS = [925.0, 850.0, 700.0, 600.0, 500.0, 400.0, 300.0, 250.0, 200.0]

TRAIN_YEARS = [2019]
TEST_YEARS = [2018]
INFERENCE_YEARS = [2018]

HIGH_RES_SHP = [48, 48]

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.00003

STATIC_NPZ_PATH = f"{DATA_DIR}/wrf_npz_static/static_d02.npz"
BEST_CHECKPOINT_PATH = f"{OUTPUT_DIR}/unet_npz_best.pt"
FINAL_CHECKPOINT_PATH = f"{OUTPUT_DIR}/unet_npz_final.pt"
CHECKPOINT_PATH = BEST_CHECKPOINT_PATH
LOSS_CURVE_PATH = f"{OUTPUT_DIR}/unet_loss_curve.png"

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

MODEL_KWARGS = {
    "model_channels": 32,
    "channel_mult": [1, 2, 2],
    "num_blocks": 1,
    "attn_resolutions": [],
    "dropout": 0.0,
}

DL_DOMAIN = {
    "max_lat": 24.75,
    "min_lat": 21.0,
    "min_lon": 112.25,
    "max_lon": 116.0,
}

PARAM_COLORMAP = {
    "T2": "coolwarm",
    "TD2": "YlGnBu",
    "MSLP": "viridis",
    "U10": "PuOr",
    "V10": "PuOr",
    "LN_ACC_6H_PRECIP": "custom_rainfall",
}

RAINFALL_LEVELS = [
    0.0, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 25, 30,
    40, 50, 70, 100, 150, 200, 300, 400,
]

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
