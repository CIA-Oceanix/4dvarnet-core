"""
This module provides tools for generating animated maps from netCDF
datasets.

Author: Q. Febvre
"""

from scipy import ndimage
import numpy as np
import xarray as xr
import holoviews as hv
hv.extension('matplotlib')


def _anim_colormap(xr_ds, dvars, deriv=None):
    tpds = xr_ds[dvars]

    if deriv is None:
        clim = (
            tpds
            .to_array()
            .pipe(
                lambda da: (
                    da.quantile(0.005).item(), da.quantile(0.995).item()
                )
            )
        )
        cmap='RdBu'
    elif deriv == 'grad':
        tpds = tpds.pipe(_sobel)
        clim = (0, tpds.to_array().max().item())
        cmap = 'viridis'
    elif deriv == 'lap':
        tpds = tpds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1))
        clim = (
            tpds
            .to_array()
            .pipe(
                lambda da: (
                    da.quantile(0.005).item(), da.quantile(0.995).item()
                )
            )
        )
        cmap='RdGy'
    else:
        raise ValueError(f'unhandled value: `deriv`={deriv}')

    hvds = hv.Dataset(tpds)
    if len(dvars) == 1:
        return (
            hvds
            .to(hv.QuadMesh, ['lon', 'lat'], dvars[0])
            .relabel(dvars[0])
            .options(cmap=cmap, clim=clim, colorbar=True)
        )

    images = hv.Layout([
        (
            hvds
            .to(hv.QuadMesh, ['lon', 'lat'], v)
            .relabel(v)
            .options(cmap=cmap, clim=clim, colorbar=True)
        )
        for v in dvars
    ]).cols(2).opts(sublabel_format="")
    return images


def _sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) / 2
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) / 2
    return np.hypot(dx_ac, dx_al)


def animate_maps(
    xr_ds, dvars, save_location=None, deriv=None, domain=None, time_slice=None,
):
    """
    Generate an animated map from a `xarray` dataset.

    Parameters
    ----------
    xr_ds : xarray.core.dataset.Dataset
        A xarray dataset object.

    dvars : str or list[str]
        The name(s) of the variable(s) from `xr_ds` to show in the
        animation.

    save_location : str or None
        The path and the filename in which the animated content will be
        stored. If `None`, the animation will be printed in the notebook
        if you are using one. The filename defines the output's format
        ("example.gif" (resp. "example.mp4") for a GIF (resp. MP4)).

    deriv : str or None
        Can be either `grad` (display the gradient version of the data)
        or `lap` (apply a laplacian filter).

    domain : dict[str, slice] or None
        Delimits the area to be animated. Example:
        >>> domain = {
        ...     'lat': slice(33, 45), 'lon': slice(-66, -54),
        ... }

    time_slice : slice
        Same as `domain` but for the time (and by index). Example:
        >>> time_slice = slice(0, -1, 2)  # every two days
    """
    if not isinstance(dvars, list):
        if isinstance(dvars, str):
            dvars = [dvars]
        else:
            raise ValueError('`dvars` must be a string or a list of string')
    if domain:
        xr_ds = xr_ds.sel(domain)
    if time_slice:
        xr_ds = xr_ds.isel(time=time_slice)

    img = _anim_colormap(xr_ds, dvars, deriv)

    if save_location:
        hv.save(img, save_location, fps=4, dpi=125)  # save file
    else:
        hv.output(img, holomap='gif', fps=4, dpi=125)  # display in notebook
