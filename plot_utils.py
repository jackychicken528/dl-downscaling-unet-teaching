import math

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import config as cfg


# -----------------------------------------------------------------------------
# Training plot
# -----------------------------------------------------------------------------

def plot_loss_curve(train_losses, test_losses, num_epochs=None, title="Training Curve"):
    """Plot train/test loss using the epochs completed so far."""
    # Epoch numbers start at 1 so the plot matches classroom notation.
    epochs = np.arange(1, len(train_losses) + 1)

    # Draw train and test losses on one figure for quick comparison.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, marker="o", label="Train")
    ax.plot(epochs, test_losses, marker="o", label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    # Keep the x-axis stable while the plot updates during training.
    if num_epochs is not None:
        ax.set_xlim(1, max(num_epochs, 1))
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


# -----------------------------------------------------------------------------
# Weather map helpers
# -----------------------------------------------------------------------------

def custom_rainfall_cmap_norm():
    """Return the classroom rainfall colormap and discrete level normalizer."""
    # Use discrete rainfall levels so the map resembles operational rain maps.
    cmap = mcolors.ListedColormap(cfg.RAINFALL_COLORS, name="custom_rainfall")
    norm = mcolors.BoundaryNorm(cfg.RAINFALL_LEVELS, cmap.N)
    return cmap, norm


def grid_lat_lon(shape):
    """Build latitude and longitude arrays for a grid covering the teaching domain."""
    # Latitude decreases from north to south; longitude increases west to east.
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
    # Convert tensors or lists to NumPy so Matplotlib can plot them.
    image = np.asarray(field)

    # Use the configured colormap when available.
    cmap = cfg.PARAM_COLORMAP.get(variable_name, "viridis")
    norm = None
    if variable_name == "LN_ACC_6H_PRECIP":
        # Rainfall is usually stored as log(precip + 0.001) in this workflow.
        if precip_is_log:
            image = np.exp(image) - 0.001
        image = np.maximum(image, 0.0)
        cmap, norm = custom_rainfall_cmap_norm()
        if colorbar_label is None:
            colorbar_label = "mm"

    # Fall back to the configured teaching domain if coordinates are omitted.
    if lat is None or lon is None:
        lat, lon = grid_lat_lon(image.shape)

    # pcolormesh draws the gridded field on a latitude-longitude map.
    mesh = ax.pcolormesh(
        lon,
        lat,
        image,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    # Add simple geographic context for Hong Kong region maps.
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


def plot_weather_grid(items, ncols=None, figsize_per_panel=(3.2, 3.0), lat=None, lon=None):
    """Plot a list of weather-map panels.

    Each item is a dictionary with at least field, variable, and title.
    Optional item keys: lat, lon, colorbar_label, precip_is_log.
    """
    # Default to one row when the caller does not choose a column count.
    if ncols is None:
        ncols = len(items)

    # Compute the number of rows needed for all requested panels.
    nrows = math.ceil(len(items) / ncols)

    # Every panel uses the same latitude-longitude projection.
    subplot_kw = {"projection": ccrs.PlateCarree()}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    # Draw one map panel for each item.
    for index in range(len(items)):
        ax = axes[index]
        item = items[index]
        mesh, colorbar_label = plot_weather_map(
            ax,
            item["field"],
            item["variable"],
            item["title"],
            lat=item.get("lat", lat),
            lon=item.get("lon", lon),
            colorbar_label=item.get("colorbar_label"),
            precip_is_log=item.get("precip_is_log", True),
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    # Hide unused panels when the grid has extra slots.
    for ax in axes[len(items):]:
        ax.set_visible(False)

    return fig


def plot_weather_rows(rows, weather_variables, lat=None, lon=None):
    """Plot rows of weather variables, such as coarse/predicted/fine fields."""
    # Create one row per data source and one column per weather variable.
    subplot_kw = {"projection": ccrs.PlateCarree()}
    fig, axes = plt.subplots(
        len(rows),
        len(weather_variables),
        figsize=(3.3 * len(weather_variables), 2.7 * len(rows)),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )

    # Normalize axes shape so indexing works for one-row or one-column plots.
    axes = np.asarray(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif len(rows) == 1:
        axes = axes[None, :]
    elif len(weather_variables) == 1:
        axes = axes[:, None]

    # Fill the plot grid with weather maps.
    for row_index, (row_name, data) in enumerate(rows):
        for col_index, variable in enumerate(weather_variables):
            title = variable
            if variable == "LN_ACC_6H_PRECIP":
                title = "PRECIP"

            ax = axes[row_index, col_index]
            panel_title = row_name
            if row_index == 0:
                panel_title = title

            # Draw each individual map and attach its own colorbar.
            mesh, colorbar_label = plot_weather_map(
                ax,
                data[col_index],
                variable,
                panel_title,
                lat=lat,
                lon=lon,
            )
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    return fig
