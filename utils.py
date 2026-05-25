import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

import config as cfg


def load_channel_stats(stats, variables):
    """Return channel-wise means and standard deviations as Python lists."""
    means = []
    stds = []
    for variable in variables:
        if isinstance(variable, (list, tuple)):
            name, pressure_level = variable
            level_index = cfg.PRESSURE_LEVELS.index(pressure_level)
            means.append(stats["mean"][name][level_index])
            stds.append(stats["std"][name][level_index])
        else:
            means.append(stats["mean"][variable])
            stds.append(stats["std"][variable])

    return means, stds


def plot_loss_curve(train_losses, test_losses, num_epochs=None, title="Training Curve"):
    """Plot train/test loss using the epochs completed so far."""
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, marker="o", label="Train")
    ax.plot(epochs, test_losses, marker="o", label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if num_epochs is not None:
        ax.set_xlim(1, max(num_epochs, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


@torch.no_grad()
def plot_prediction_example(model, dataset, weather_variables, device, index=0):
    """Compare coarse input, model prediction, and fine truth for one timestep."""
    model.eval()
    sample = dataset[index]

    inputs = sample["inputs"].unsqueeze(0).to(device)
    condition = sample["condition"].unsqueeze(0).to(device)
    residual_prediction = model(inputs, class_labels=condition).cpu()
    predicted_fine = dataset.residual_to_fine(residual_prediction, sample["coarse"].unsqueeze(0))[0]

    rows = [
        ("Coarse", sample["coarse"]),
        ("Predicted", predicted_fine),
        ("Fine", sample["fine"]),
    ]
    return _plot_three_row_weather(rows, weather_variables)


def save_predicted_year_npz(data_dir, year, dataset, predicted):
    """Save predicted fine fields for one year."""
    output_dir = f"{data_dir}/wrf_npz_predicted"
    os.makedirs(output_dir, exist_ok=True)
    lat, lon = grid_lat_lon(predicted.shape[-2:])
    variable_names = np.array(cfg.DL_OUT_SINGLE_LAYER_PARAMS + cfg.DL_OUT_UPPER_PARAMS)
    path = f"{output_dir}/wrf_{year}_predicted.npz"
    np.savez_compressed(
        path,
        data=predicted,
        time=dataset.time.astype(str),
        lat=lat,
        lon=lon,
        variable_names=variable_names,
    )
    return path


def plot_saved_prediction(outputs, weather_variables, year, index=0):
    """Compare saved coarse, predicted, and fine arrays for one timestep."""
    result = outputs[year]
    rows = [
        ("Coarse", result["coarse"][index]),
        ("Predicted", result["predicted"][index]),
        ("Fine", result["fine"][index]),
    ]
    return _plot_three_row_weather(
        rows,
        weather_variables,
        lat=result.get("lat"),
        lon=result.get("lon"),
    )


def custom_rainfall_cmap_norm():
    """Return the classroom rainfall colormap and discrete level normalizer."""
    cmap = mcolors.ListedColormap(cfg.RAINFALL_COLORS, name="custom_rainfall")
    norm = mcolors.BoundaryNorm(cfg.RAINFALL_LEVELS, cmap.N)
    return cmap, norm


def grid_lat_lon(shape):
    """Build latitude and longitude arrays for a grid covering the teaching domain."""
    lat = np.linspace(cfg.DL_DOMAIN["max_lat"], cfg.DL_DOMAIN["min_lat"], shape[0])
    lon = np.linspace(cfg.DL_DOMAIN["min_lon"], cfg.DL_DOMAIN["max_lon"], shape[1])
    return lat, lon


def plot_weather_map(
    ax,
    field,
    variable_name,
    title,
    lat=None,
    lon=None,
    colorbar_label=None,
    precip_is_log=True,
):
    """Plot one weather field on a Cartopy map with coastline context."""
    image = np.asarray(field)
    cmap = cfg.PARAM_COLORMAP.get(variable_name, "viridis")
    norm = None
    if variable_name == "LN_ACC_6H_PRECIP":
        if precip_is_log:
            image = np.exp(image) - 0.001
        image = np.maximum(image, 0.0)
        cmap, norm = custom_rainfall_cmap_norm()
        if colorbar_label is None:
            colorbar_label = "mm"

    if lat is None or lon is None:
        lat, lon = grid_lat_lon(image.shape)

    mesh = ax.pcolormesh(
        lon,
        lat,
        image,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="10m", linewidth=0.6)
    ax.set_extent([min(lon), max(lon), min(lat), max(lat)], crs=ccrs.PlateCarree())
    ax.set_title(title)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    return mesh, colorbar_label


def _plot_three_row_weather(rows, weather_variables, lat=None, lon=None):
    subplot_kw = {"projection": ccrs.PlateCarree()}
    fig, axes = plt.subplots(
        3,
        len(weather_variables),
        figsize=(3.3 * len(weather_variables), 8),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )
    if len(weather_variables) == 1:
        axes = axes[:, None]
    for row_index, (row_name, data) in enumerate(rows):
        for col_index, variable in enumerate(weather_variables):
            title = variable
            if variable == "LN_ACC_6H_PRECIP":
                title = "PRECIP"

            ax = axes[row_index, col_index]
            mesh, colorbar_label = plot_weather_map(
                ax,
                data[col_index],
                variable,
                title if row_index == 0 else row_name,
                lat=lat,
                lon=lon,
            )
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    return fig
