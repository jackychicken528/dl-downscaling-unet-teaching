import math

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
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
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
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


def one_degree_ticks(values):
    """Return whole-degree ticks within a latitude or longitude coordinate range."""
    min_value = min(values)
    max_value = max(values)
    start = math.ceil(min_value)
    stop = math.floor(max_value)
    return np.arange(start, stop + 1, 1)


def plot_weather_map(
    ax,
    field,
    variable_name,
    title,
    lat=None,
    lon=None,
    colorbar_label=None,
):
    """Plot one weather field on a Cartopy map with coastline context."""
    # Convert tensors or lists to NumPy so Matplotlib can plot them.
    image = np.asarray(field)

    # Use the configured colormap when available.
    cmap = cfg.PARAM_COLORMAP.get(variable_name, "viridis")
    norm = None
    if variable_name == "ACC_6H_PRECIP":
        # ACC_6H_PRECIP is expected in millimeters for display.
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
    gl.xlocator = mticker.FixedLocator(one_degree_ticks(lon))
    gl.ylocator = mticker.FixedLocator(one_degree_ticks(lat))
    gl.top_labels = False
    gl.right_labels = False
    return mesh, colorbar_label


def plot_weather_grid(
    items,
    ncols=None,
    figsize_per_panel=(3.2, 3.0),
    lat=None,
    lon=None,
    weather_variables=None,
):
    """Plot several weather fields as Cartopy map panels.

    There are two supported input styles.

    1. Explicit panel style:

        items = [
            {"field": rain_2d_array, "variable": "ACC_6H_PRECIP", "title": "Predicted rain"},
            {"field": temp_2d_array, "variable": "T2", "title": "Temperature"},
        ]
        fig = plot_weather_grid(items, ncols=2)

       Required item keys:
       - field: 2-D array-like data to plot, shaped as (lat, lon).
       - variable: variable name used to choose the colormap.
       - title: subplot title.

       Optional item keys:
       - lat, lon: 1-D coordinates for this panel.
       - colorbar_label: label beside that panel's colorbar.

    2. Row style, useful for comparing coarse/predicted/fine fields:

        rows = [("Coarse", coarse_data), ("Predicted", pred_data), ("Fine", fine_data)]
        weather_variables = ["ACC_6H_PRECIP", "T2", "U10"]
        fig = plot_weather_grid(rows, weather_variables=weather_variables)

       In row style, each data object must be indexable by variable position:
       data[col_index] is plotted for weather_variables[col_index]. The first
       row shows variable names as titles; later rows show the row name.

    Args:
        items: Either explicit panel dictionaries, or row-style
            (row_name, data) pairs when weather_variables is provided.
        ncols: Number of subplot columns. Defaults to all panels in one row, or
            len(weather_variables) in row style.
        figsize_per_panel: Width and height, in inches, for each subplot panel.
        lat: Shared 1-D latitude coordinates. Optional.
        lon: Shared 1-D longitude coordinates. Optional.
        weather_variables: Variable names for row style. Leave as None when
            using explicit panel dictionaries.

    Returns:
        matplotlib.figure.Figure: The completed figure. Display it with
        plt.show(), save it with fig.savefig(...), or log it to your notebook.
    """
    if weather_variables is not None:
        grid_items = []
        for row_index, (row_name, data) in enumerate(items):
            for col_index, variable in enumerate(weather_variables):
                title = variable
                if variable == "ACC_6H_PRECIP":
                    title = "PRECIP"

                panel_title = title
                if row_index > 0:
                    panel_title = row_name

                grid_items.append({
                    "field": data[col_index],
                    "variable": variable,
                    "title": panel_title,
                })

        items = grid_items
        if ncols is None:
            ncols = len(weather_variables)

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
        dpi=200,
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
        )
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)

    # Hide unused panels when the grid has extra slots.
    for ax in axes[len(items):]:
        ax.set_visible(False)

    return fig


# -----------------------------------------------------------------------------
# Evaluation plots and tables
# -----------------------------------------------------------------------------

DEFAULT_UNIT_LABELS = {
    "T2": "K",
    "TD2": "deg C",
    "MSLP": "hPa",
    "U10": "m s-1",
    "V10": "m s-1",
    "ACC_6H_PRECIP": "mm",
}


def plot_case_study_comparison(
    coarse,
    fine,
    predicted,
    lat,
    lon,
    variable_names,
    time_label=None,
    coarse_stride=3,
    wind_stride=3,
):
    """Plot coarse, fine, and predicted weather maps for one timestep.

    The map follows the reference case-study plot: precipitation is shaded,
    sea-level pressure is contoured, and 10-m wind is shown with barbs.
    """
    variable_names = list(variable_names)
    precip_idx = variable_names.index("ACC_6H_PRECIP")
    pressure_idx = variable_names.index("MSLP")
    u_idx = variable_names.index("U10")
    v_idx = variable_names.index("V10")

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    rain_cmap, rain_norm = custom_rainfall_cmap_norm()

    panel_data = [
        ("WRF D01 Coarse", np.asarray(coarse), coarse_stride),
        ("WRF D02 Fine", np.asarray(fine), 1),
        ("DL Model Downscaled", np.asarray(predicted), 1),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        dpi=200,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )

    precip_mesh = None
    for ax, (title, data, map_stride) in zip(axes, panel_data):
        row_slice = slice(None, None, map_stride)
        col_slice = slice(None, None, map_stride)
        wind_slice = slice(1, None, wind_stride)

        panel_lon = lon_grid[row_slice, col_slice]
        panel_lat = lat_grid[row_slice, col_slice]
        panel_precip = np.maximum(data[precip_idx, row_slice, col_slice], 0.0)
        panel_pressure = data[pressure_idx, row_slice, col_slice]

        precip_mesh = ax.contourf(
            panel_lon,
            panel_lat,
            panel_precip,
            levels=cfg.RAINFALL_LEVELS,
            cmap=rain_cmap,
            norm=rain_norm,
            transform=ccrs.PlateCarree(),
            extend="max",
        )

        pressure_min = np.floor(np.nanmin(panel_pressure))
        pressure_max = np.ceil(np.nanmax(panel_pressure))
        pressure_levels = np.arange(pressure_min, pressure_max + 1, 1)
        if len(pressure_levels) > 1:
            pressure_contours = ax.contour(
                panel_lon,
                panel_lat,
                panel_pressure,
                levels=pressure_levels,
                colors="tab:blue",
                linewidths=0.8,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(pressure_contours, inline=True, fontsize=7)

        if map_stride == 1:
            barb_lon = lon_grid[wind_slice, wind_slice]
            barb_lat = lat_grid[wind_slice, wind_slice]
            barb_u = data[u_idx, wind_slice, wind_slice]
            barb_v = data[v_idx, wind_slice, wind_slice]
        else:
            barb_lon = panel_lon
            barb_lat = panel_lat
            barb_u = data[u_idx, row_slice, col_slice]
            barb_v = data[v_idx, row_slice, col_slice]

        ax.barbs(
            barb_lon,
            barb_lat,
            barb_u,
            barb_v,
            length=5,
            linewidth=0.6,
            transform=ccrs.PlateCarree(),
        )

        full_title = title
        if time_label is not None:
            full_title = f"{time_label} | {title}"
        ax.set_title(full_title, fontsize=10)
        ax.coastlines(resolution="10m", linewidth=0.7)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )
        gl.xlocator = mticker.FixedLocator(one_degree_ticks(lon))
        gl.ylocator = mticker.FixedLocator(one_degree_ticks(lat))
        gl.top_labels = False
        gl.right_labels = False

    cbar = fig.colorbar(
        precip_mesh,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
        ticks=cfg.RAINFALL_LEVELS,
    )
    cbar.set_label("Precipitation (mm/6h)")
    return fig


def plot_density_qq_grid(fine, predicted, variable_names, unit_labels=None, ncols=3, gridsize=200, cmap="inferno"):
    """Plot density scatter and Q-Q diagnostics for all variables in one figure."""
    variable_names = list(variable_names)
    nrows = math.ceil(len(variable_names) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.2 * nrows),
        dpi=200,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)

    for channel_index, variable_name in enumerate(variable_names):
        ax = axes[channel_index]
        fine_values = np.asarray(fine[:, channel_index], dtype=np.float64).ravel()
        pred_values = np.asarray(predicted[:, channel_index], dtype=np.float64).ravel()
        valid = np.isfinite(fine_values) & np.isfinite(pred_values)
        fine_values = fine_values[valid]
        pred_values = pred_values[valid]

        fine_sorted = np.sort(fine_values)
        pred_sorted = np.sort(pred_values)
        data_min = min(fine_values.min(), pred_values.min())
        data_max = max(fine_values.max(), pred_values.max())
        span = data_max - data_min
        if span == 0:
            span = 1.0

        if variable_name == "ACC_6H_PRECIP":
            min_lim = 0.0
        else:
            min_lim = data_min - span / 20.0
        max_lim = data_max + span / 20.0

        hexbin = ax.hexbin(
            fine_values,
            pred_values,
            gridsize=gridsize,
            bins="log",
            mincnt=1,
            cmap=cmap,
            extent=(min_lim, max_lim, min_lim, max_lim),
        )
        ax.plot([min_lim, max_lim], [min_lim, max_lim], color="red", linestyle="--", linewidth=1, label="y=x")
        ax.plot(fine_sorted, pred_sorted, color="limegreen", linewidth=2, label="Q-Q")

        unit = ""
        if unit_labels is not None:
            unit = unit_labels.get(variable_name, "")
        unit_suffix = f" ({unit})" if unit else ""
        ax.set_title(variable_name, fontsize=11)
        ax.set_xlabel(f"WRF D02{unit_suffix}")
        ax.set_ylabel(f"DL prediction{unit_suffix}")
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.legend(fontsize=8)
        fig.colorbar(hexbin, ax=ax, label="Count")

    for ax in axes[len(variable_names):]:
        ax.set_visible(False)

    return fig
