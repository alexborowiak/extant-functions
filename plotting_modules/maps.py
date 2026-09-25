"""Anything on a projection. The only module that needs cartopy.

Layers:
  Primitive   setup_polar_ax, draw_polar_contour — draw on one axes.
  Selection   select, panel_keys, select_panel, prepare — turn a DataArray plus
              sel/row/col into the coordinates core.panel_grid iterates over.
  Figure      polar_grid — DataArray in, figure out.

For mixed figures, pass caller-owned `axes` to polar_grid. For a custom map,
call draw_polar_contour from your own per-panel draw function.
"""

import numpy as np
import matplotlib.path as mpath
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from .core import (
    add_colorbar,
    add_suptitle,
    label_cols,
    label_rows,
    panel_grid,
    tag_panels,
)
from .utils import plot

DEFAULT_EXTENT = (-180, 180, -90, -50)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------

def setup_polar_ax(ax, extent=DEFAULT_EXTENT):
    """Circular boundary, coastlines and gridlines on a polar axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes to configure, expected to use a polar stereographic projection.
    extent : sequence of float
        Extent as (lon_min, lon_max, lat_min, lat_max) in PlateCarree.
    """
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_path = mpath.Path(
        np.column_stack([np.cos(theta), np.sin(theta)]) * 0.5 + 0.5
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    ax.set_boundary(circle_path, transform=ax.transAxes)
    ax.coastlines(linewidth=0.7)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=0.5,
        color="gray",
        alpha=0.6,
        linestyle="--",
    )
    gl.n_steps = 90
    return ax

@plot("axes")
def draw_polar_contour(
    ax,
    da,
    levels,
    cmap="RdBu_r",
    norm=None,
    lat_name="lat",
    lon_name="lon",
    extent=DEFAULT_EXTENT,
):
    """Filled and line contours of a 2-D field on polar axes.

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Target axes.
    da : xarray.DataArray
        Field with latitude and longitude dimensions.
    levels : array_like
        Contour levels.
    cmap : str or Colormap, default="RdBu_r"
        Colormap for filled contours.
    norm : matplotlib.colors.Normalize, optional
        Normalisation used to map values to colours.
    lat_name, lon_name : str
        Latitude and longitude coordinate names.
    extent : sequence of float
        Passed to `setup_polar_ax`.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        Filled contour set for use as a colorbar mappable.
    """
    da = da.transpose(lat_name, lon_name)
    data_cyclic, lons_cyclic = add_cyclic_point(da.values, coord=da[lon_name].values)
    lats = da[lat_name].values

    cf = ax.contourf(
        lons_cyclic,
        lats,
        data_cyclic,
        transform=ccrs.PlateCarree(),
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )

    ax.contour(
        lons_cyclic,
        lats,
        data_cyclic,
        transform=ccrs.PlateCarree(),
        levels=levels,
        colors="k",
        linewidths=0.3,
        alpha=0.5,
    )

    setup_polar_ax(ax, extent)
    return cf

# --------------------------------------------------------------------------
# Turning a DataArray into panel coordinates
# --------------------------------------------------------------------------

def select(da, sel):
    """Apply a sel dict: a sequence subsets and orders, a scalar collapses."""
    for dim, value in sel.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            da = da.sel({dim: list(value)})
        else:
            da = da.sel({dim: value})
    return da

def panel_keys(da, dim):
    """Keys for panels along `dim`, or [None] when it is not a grid dimension."""
    if dim is None or dim not in da.dims:
        return [None]
    return list(np.atleast_1d(da[dim].values))

def select_panel(da, dim, key):
    """Select one panel key; ``None`` means that dimension is not faceted."""
    return da if key is None else da.sel({dim: key})

def prepare(da, sel=None, row_dim=None, col_dim=None, lat_name="lat", lon_name="lon"):
    """Apply sel, drop degenerate dims, return (da, row_keys, col_keys).

    Raises if any dimension is left neither collapsed by `sel` nor mapped to
    rows, columns, latitude or longitude — silently plotting the first element
    of an unmapped dimension is the failure mode this guards against.
    """
    da = select(da, sel) if sel else da
    squeezable = [
        d for d in da.dims if da.sizes[d] == 1 and d not in (lat_name, lon_name)
    ]
    da = da.squeeze(squeezable)

    unmapped = [d for d in da.dims if d not in (lat_name, lon_name, row_dim, col_dim)]
    if unmapped:
        raise ValueError(
            f"unmapped dimensions {unmapped}; pass them in sel, row_dim or col_dim"
        )

    return da, panel_keys(da, row_dim), panel_keys(da, col_dim)

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

@plot("figure")
def polar_grid(da, row_dim=None, col_dim=None, sel=None,
               levels=np.linspace(-3, 3, 13), cmap="RdBu_r", norm=None,
               lat_name="lat", lon_name="lon", label_fmt=None,
               title=None, cbar_label=None, projection=None, tag=True,
               fig=None, spec=None, layout=None, ax=None, axes=None, cax=None,
               cbar_height=None, cbar_gap=None, **layout_kwargs):
    """Grid of polar panels with two dimensions mapped to rows and columns.

    Every dimension other than latitude, longitude, row_dim and col_dim must be
    collapsed by sel.

    Parameters
    ----------
    da : xarray.DataArray
        Field with latitude and longitude dimensions.
    row_dim, col_dim : str or None
        Dimensions mapped to rows and columns; None or a scalar coordinate
        gives one row or column.
    sel : dict or None
        Coordinate selections applied first, e.g. {'experiment': 'historical'};
        a sequence value subsets and orders a dimension, a scalar collapses it.
    levels : array_like
        Contour levels, shared by every panel.
    cmap : str
        Colormap name.
    lat_name, lon_name : str
        Names of the latitude and longitude coordinates.
    label_fmt : dict or None
        Mapping of dimension name to a callable formatting its values.
    title : str or None
        Figure title. Added only when this function owns the figure; caller-
        owned axes and nested layouts leave the figure title to their caller.
    cbar_label : str or None
        Label under the colorbar, e.g. units.
    projection : cartopy.crs.Projection or None
        Defaults to SouthPolarStereo when this function creates the axes.
    tag : bool
        Letter the panels a) b) c).
    ax : cartopy.mpl.geoaxes.GeoAxes or None
        Target for a one-panel grid. Its figure is inferred when `fig` is
        omitted.
    axes : array-like of cartopy.mpl.geoaxes.GeoAxes or None
        Targets for a multi-panel grid, in row-major order. They must already
        use the desired Cartopy projection.
    cax : matplotlib.axes.Axes or None
        Caller-owned colorbar axes. A colorbar is created automatically in a
        reserved GridSpec row only when this function creates its panel axes.
        For a composed figure, create this with ``fig.add_subplot(grid[...])``.
    cbar_height, cbar_gap : float or None
        Thickness and gap above an automatic colorbar, in inches. These apply
        when this function creates a `GridLayout`; nested layouts use
        `cbar_frac`, and a caller-owned `cax` controls its own geometry.
    fig, spec, layout, **layout_kwargs
        Standard figure-function arguments; see core.open_layout.

    Returns
    -------
    core.Panels

    Examples
    --------
    Put signal, noise and S/N into rows of one caller-owned GridSpec::

        import matplotlib.pyplot as plt
        import numpy as np
        import xarray as xr
        import cartopy.crs as ccrs

        fig = plt.figure()
        grid = fig.add_gridspec(4, n_periods, height_ratios=[1, 1, 1, .06])
        axes = np.array([
            [fig.add_subplot(grid[row, col], projection=ccrs.SouthPolarStereo())
             for col in range(n_periods)]
            for row in range(3)
        ])
        cax = fig.add_subplot(grid[3, :])
        fields = xr.concat(
            (signal, noise, sn),
            dim=xr.IndexVariable("kind", ("signal", "noise", "sn")),
        )
        polar_grid(
            fields, row_dim="kind", col_dim="period", axes=axes, cax=cax,
            tag=False,
        )

    The figure owns every position in this example; ``polar_grid`` only draws
    into the supplied axes.
    """
    da, row_keys, col_keys = prepare(da, sel, row_dim, col_dim, lat_name, lon_name)
    fmt = label_fmt or {}
    caller_axes = ax is not None or axes is not None

    layout_kwargs.setdefault("row_labels", any(key is not None for key in row_keys))
    layout_kwargs.setdefault("has_title", title is not None)
    layout_kwargs.setdefault("has_cbar", cax is None)
    layout_kwargs.setdefault("has_cbar_label", cbar_label is not None and cax is None)
    if cbar_height is not None or cbar_gap is not None:
        if caller_axes or cax is not None:
            raise ValueError("cbar_height/cbar_gap do not resize caller-owned cax")
        if spec is not None or layout is not None:
            raise ValueError(
                "cbar_height/cbar_gap need a new GridLayout; use cbar_frac for a "
                "nested layout or configure a supplied layout directly"
            )
        if cbar_height is not None:
            layout_kwargs["cbar_height"] = cbar_height
        if cbar_gap is not None:
            layout_kwargs["cbar_gap"] = cbar_gap

    def draw_panel(ax, row_key, col_key):
        field = select_panel(da, row_dim, row_key)
        field = select_panel(field, col_dim, col_key)
        return draw_polar_contour(ax, field, levels, cmap, norm, lat_name, lon_name)

    panels = panel_grid(
        row_keys,
        col_keys,
        draw_panel,
        fig=fig, spec=spec, layout=layout, ax=ax, axes=axes,
        projection=projection or ccrs.SouthPolarStereo(),
        **layout_kwargs,
    )

    label_cols(panels.axes, col_keys, fmt.get(col_dim, str))
    label_rows(panels.axes, row_keys, fmt.get(row_dim, str))
    if tag:
        tag_panels(panels.axes)

    if cax is None and not caller_axes:
        cax = panels.layout.cbar_ax(panels.fig, 0, len(col_keys) - 1)
    if cax is not None:
        panels.extras["cbar"] = add_colorbar(
            panels.fig,
            panels.artists[0, 0],
            cax,
            levels=levels,
            label=cbar_label,
        )
    if title and not caller_axes and panels.layout.owns_figure:
        add_suptitle(panels.fig, panels.layout, title)

    return panels


import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point


def stipple_mask(p, threshold=0.01):
    """Mask grid cells that fail a significance threshold.

    Args:
        p (xr.DataArray): p-values.
        threshold (float): significance level.

    Returns:
        xr.DataArray: boolean, True where p is not below threshold.
    """
    return p >= threshold


def plot_stipple(ax, mask, every=None, marker='.', size=1.0, color='k'):
    """Scatter stipple markers on a map axis where mask is True.

    Args:
        ax (GeoAxes): axis to draw on.
        mask (xr.DataArray): 2-D boolean mask with lat and lon dims.
        every (int): plot every nth grid cell in lat and lon, or None for all.
        marker (str): matplotlib marker.
        size (float): marker size in points squared.
        color (str): marker colour.

    Returns:
        PathCollection: the scatter artist.
    """
    if every is not None:
        mask = mask.isel(lat=slice(None, None, every), lon=slice(None, None, every))
    points = mask.transpose('lat', 'lon').values
    lon2d, lat2d = np.meshgrid(mask.lon.values, mask.lat.values)
    return ax.scatter(
        lon2d[points],
        lat2d[points],
        s=size,
        marker=marker,
        color=color,
        linewidths=0,
        transform=ccrs.PlateCarree(),
    )


def plot_hatch(ax, mask, hatch='...', color='k', linewidth=0.3):
    """Hatch regions of a map axis where mask is True.

    Args:
        ax (GeoAxes): axis to draw on.
        mask (xr.DataArray): 2-D boolean mask with lat and lon dims.
        hatch (str): matplotlib hatch pattern; repeat to increase density.
        color (str): hatch line colour.
        linewidth (float): hatch line width in points.

    Returns:
        QuadContourSet: the contour artist.
    """
    field = mask.transpose('lat', 'lon').values.astype(float)
    field, lon = add_cyclic_point(field, coord=mask.lon.values)
    with plt.rc_context({'hatch.linewidth': linewidth, 'hatch.color': color}):
        return ax.contourf(
            lon,
            mask.lat.values,
            field,
            levels=[0.5, 1.5],
            colors='none',
            hatches=[hatch],
            transform=ccrs.PlateCarree(),
        )
def map_panels(panels, da, draw, row_dim=None, col_dim=None, sel=None, **kwargs):
    """Apply a drawing function to every panel of a polar_grid figure."""
    if sel:
        da = da.sel(sel)

    artists = np.empty(panels.axes.shape, dtype=object)

    for i in range(panels.axes.shape[0]):
        for j in range(panels.axes.shape[1]):
            indexer = {}
            if row_dim is not None:
                indexer[row_dim] = i
            if col_dim is not None:
                indexer[col_dim] = j

            field = da.isel(indexer) if indexer else da
            artists[i, j] = draw(panels.axes[i, j], field, **kwargs)

    return artists
