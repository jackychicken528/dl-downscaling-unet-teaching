"""Data loading and dataset code for the WRF U-Net teaching notebooks."""

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, Normalize
from torchvision.transforms.functional import resize

import config as cfg


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------


def channel_normalizer(mean, std):
    """Build a torchvision channel normalizer for tensors shaped (..., C, H, W)."""
    return Normalize(mean=mean, std=std)


def inverse_channel_normalizer(mean, std):
    """Build the inverse of a torchvision channel normalizer."""
    # If x_norm = (x - mean) / std, then x = x_norm * std + mean.
    # torchvision Normalize uses (x - mean) / std, so these values undo it.
    inverse_mean = []
    inverse_std = []

    for i in range(len(mean)):
        inverse_mean.append(-mean[i] / std[i])
        inverse_std.append(1.0 / std[i])

    return Normalize(mean=inverse_mean, std=inverse_std)


# -----------------------------------------------------------------------------
# Weather and static variable loading
# -----------------------------------------------------------------------------

def read_weather_channel(npz_file, variable_name):
    """Read one weather channel from a yearly WRF NPZ file."""
    if variable_name == "LN_ACC_6H_PRECIP":
        precip = npz_file["ACC_6H_PRECIP"].astype(np.float32)
        precip = np.maximum(precip, 0.0)
        return np.log(precip + 0.001)

    return npz_file[variable_name].astype(np.float32)


def resize_grid(tensor, size, mode):
    """Resize a tensor while keeping all leading dimensions unchanged."""
    if mode == "nearest":
        interpolation = InterpolationMode.NEAREST
    else:
        interpolation = InterpolationMode.BILINEAR

    return resize(tensor, list(size), interpolation=interpolation, antialias=False)


# -----------------------------------------------------------------------------
# Time-conditioning features
# -----------------------------------------------------------------------------

def build_time_features(times):
    """Create seasonal and daily cycle features from numpy datetime64 values."""
    times = times.astype("datetime64[h]")
    dates = times.astype("datetime64[D]")
    years = times.astype("datetime64[Y]")

    day_of_year = (dates - years.astype("datetime64[D]")).astype(int)
    hour_of_day = (times - dates.astype("datetime64[h]")).astype("timedelta64[h]").astype(int)

    features = np.stack(
        [
            np.sin(2 * np.pi * day_of_year / 365.0),
            np.cos(2 * np.pi * day_of_year / 365.0),
            np.sin(2 * np.pi * hour_of_day / 24.0),
            np.cos(2 * np.pi * hour_of_day / 24.0),
        ],
        axis=1,
    )
    return torch.tensor(features, dtype=torch.float32)


def load_weather_year(year, resolution, data_dir):
    """Load one year of coarse or fine WRF fields from a yearly NPZ file."""
    weather_variables = list(cfg.DL_OUT_SINGLE_LAYER_PARAMS + cfg.DL_OUT_UPPER_PARAMS)

    path = f"{data_dir}/wrf_npz_{resolution}/wrf_{int(year)}_{resolution}.npz"
    with np.load(path, allow_pickle=False) as npz_file:
        channels = []
        for name in weather_variables:
            channel = read_weather_channel(npz_file, name)
            channels.append(channel)

        data = np.stack(channels, axis=1)
        time = npz_file["time"]

    return torch.tensor(data, dtype=torch.float32), time


# -----------------------------------------------------------------------------
# Static-variable handling
# -----------------------------------------------------------------------------

def load_static_channels(static_npz_path, output_shape):
    """Load terrain/land-sea channels and resize them to the model grid."""
    static_variables = list(cfg.CONSTANT_PARAMS)

    if not static_variables:
        empty_shape = (0, output_shape[0], output_shape[1])
        return torch.empty(empty_shape, dtype=torch.float32)

    channels = []
    with np.load(static_npz_path, allow_pickle=False) as npz_file:
        for name in static_variables:
            # Add one channel dimension so torchvision treats this as a single image.
            channel = torch.tensor(npz_file[name].astype(np.float32))[None]
            mode = "nearest" if name == "LSM" else "bilinear"
            channel = resize_grid(channel, output_shape, mode=mode)[0]

            if name != "LSM":
                channel = (channel - channel.mean()) / (channel.std() + 1e-6)
            channels.append(channel)

    return torch.stack(channels, dim=0)


# -----------------------------------------------------------------------------
# Dataset assembly
# -----------------------------------------------------------------------------

class WRFNPZDataset(torch.utils.data.Dataset):
    """Small teaching dataset for WRF downscaling from yearly NPZ files."""

    def __init__(
        self,
        years,
        data_dir,
        static_npz_path,
        raw_mean,
        raw_std,
        residual_mean,
        residual_std,
        output_shape=(48, 48),
    ):
        selected_years = []
        for year in years:
            selected_years.append(int(year))

        output_shape = tuple(output_shape)

        coarse_years = []
        fine_years = []
        time_years = []
        for year in selected_years:
            coarse, coarse_time = load_weather_year(year, "coarse", data_dir)
            fine, fine_time = load_weather_year(year, "fine", data_dir)
            coarse_years.append(coarse)
            fine_years.append(fine)
            time_years.append(coarse_time)

        coarse = torch.cat(coarse_years, dim=0)
        fine = torch.cat(fine_years, dim=0)
        self.time = np.concatenate(time_years)

        # 1. Grid matching: put coarse and fine fields on the same 48 x 48 grid.
        self.coarse = resize_grid(coarse, output_shape, mode="nearest")
        self.fine = resize_grid(fine, output_shape, mode="bilinear")

        # 2. Static features: append terrain and land/sea mask to every timestep.
        static_channels = load_static_channels(static_npz_path, output_shape)
        static_channels = static_channels.expand(self.coarse.shape[0], -1, -1, -1)

        # 3. Normalization: use torchvision to normalize each weather channel.
        self.input_normalizer = channel_normalizer(raw_mean, raw_std)
        self.residual_normalizer = channel_normalizer(residual_mean, residual_std)
        self.residual_inverse_normalizer = inverse_channel_normalizer(residual_mean, residual_std)

        weather_inputs = self.input_normalizer(self.coarse)
        residual = self.fine - self.coarse
        self.targets = self.residual_normalizer(residual)

        # 4. Final model inputs: normalized weather channels plus static channels.
        self.inputs = torch.cat([weather_inputs, static_channels], dim=1)

        self.conditions = build_time_features(self.time)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return {
            "inputs": self.inputs[index],
            "targets": self.targets[index],
            "coarse": self.coarse[index],
            "fine": self.fine[index],
            "condition": self.conditions[index],
        }

    def inverse_normalize_residual(self, residual_norm):
        return self.residual_inverse_normalizer(residual_norm)

    def residual_to_fine(self, residual_norm, coarse):
        return coarse + self.inverse_normalize_residual(residual_norm)
