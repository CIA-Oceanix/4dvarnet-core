"""
# Using the swath gaussian diff preprocessing for direct interpolation


## multiple ways of doing this:
 Using coarsening
 Using 2d kernels on the gridded nadirs
 Using 1d kernel on the tracks


## Other idea
tri plane decomposition



## So one pre


First idea:
    - project the nadir data on the triplanes
    - apply different levels of nan-average pooling + upsampling
    - direct prediction from the learnt features

"""

import hydra
import seaborn as sns
import xrft

from omegaconf import OmegaConf
import holoviews as hv
import holoviews.plotting.mpl  # noqa
from dvc_main import VersioningCallback
import einops
import scipy.ndimage as ndi
import contextlib
import numpy as np
from torch.nn.modules.conv import Conv2d
import zarr
import matplotlib.pyplot as plt
import xarray as xr
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import re
from hydra.utils import instantiate, get_class, call
from hydra_main import FourDVarNetHydraRunner
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import traceback
import hydra_config
from IPython.display import display, Markdown, Latex, HTML

import kornia
import math
import traceback
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

def get_nadir_slice(path, **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path),
                          group=group, decode_times=False, consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/nadir.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64() / pd.to_timedelta(1, unit=units.strip())
            te = (dt_end - pd.to_datetime(reference_date)) / pd.to_timedelta(1, unit=units.strip())
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon > slice_args.get('lon_min', -360)))).compute()
            )
    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        print(
            f"no data at {path} found for {slice_args} {groups} {pd.date_range(start=dt_start, end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )


def get_cfg(xp_cfg, overrides=None):
    overrides = overrides if overrides is not None else []
    with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp_cfg}',
                'file_paths=dgx_ifremer',
                'entrypoint=train',
            ] + overrides
        )

    return cfg

def get_model(xp_cfg, ckpt, dm=None, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(xp_cfg, setup=True, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    dm = instantiate(cfg.datamodule)
    if setup:
        dm.setup()
    return dm


class NewDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self,):
        ...

def explore():
    try:
        
        cfg_n = 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00'
        overrides = ['+datamodule.dl_kwargs.shuffle=False']
        dm = get_dm(cfg_n, add_overrides=overrides)
        win = 9
        grid_ds = dm.val_ds.datasets[0].gt_ds.ds.isel(time=slice(None, win))
        slice_args = dict(
                # time_min='2012-10-01', 
                time_min= pd.to_datetime(np.min(grid_ds['time']).values).date(),
                # time_max='2012-10-30',
                time_max= pd.to_datetime(np.max(grid_ds['time']).values).date(),
                lat_min=grid_ds['lat'].min().item(),
                lat_max=grid_ds['lat'].max().item(),
                lon_min=grid_ds['lon'].min().item() + 360,
                lon_max=grid_ds['lon'].max().item() + 360,
        )

        swath_data = {nad: get_nadir_slice(f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}', **slice_args) for nad in 
            [ 'en', 'g2', 'j1', 'tpn', ]}

        import pyinterp
        n_coords = 256
        
        t_coords = pd.date_range(grid_ds.time.min().values, grid_ds.time.max().values, periods=n_coords)
        lon_coords = np.linspace(grid_ds.lon.min().values, grid_ds.lon.max().values, endpoint=True, num=n_coords)
        lat_coords = np.linspace(grid_ds.lat.min().values, grid_ds.lat.max().values, endpoint=True, num=n_coords)
        tgt_gridtx = xr.Dataset(coords={'time': ('time', t_coords) , 'lon':('lon', lon_coords)})
        tgt_gridxy = xr.Dataset(coords={'lon':('lon', lon_coords), 'lat': ('lat', lat_coords)})
        tgt_gridyt = xr.Dataset(coords={'lat': ('lat', lat_coords), 'time': ('time', t_coords)})
        binningtx = pyinterp.Binning2D(pyinterp.Axis(tgt_gridtx.time.astype(float).values), pyinterp.Axis(tgt_gridtx.lon.values))
        binningxy = pyinterp.Binning2D(pyinterp.Axis(tgt_gridxy.lon.values), pyinterp.Axis(tgt_gridxy.lat.values))
        binningyt = pyinterp.Binning2D(pyinterp.Axis(tgt_gridxy.lat.values), pyinterp.Axis(tgt_gridyt.time.astype(float).values))

        binningtx.clear()
        binningyt.clear()
        binningyt.clear()
        for nad_data in swath_data.values():


            values = np.ravel(nad_data.ssh_model.values)
            times = np.ravel(nad_data.time.astype(float).values)
            lons = np.ravel(nad_data.lon.values) - 360
            lats = np.ravel(nad_data.lat.values)
            msk = np.isfinite(values)
            binningtx.push(times[msk], lons[msk], values[msk])
            binningxy.push(lons[msk], lats[msk], values[msk])
            binningyt.push(lats[msk], times[msk], values[msk])

        gridded_tx =  (('time', 'lon'), binningtx.variable('mean'))
        gridded_xy =  (('lon', 'lat'), binningxy.variable('mean'))
        gridded_yt =  (('lat', 'time'), binningyt.variable('mean'))
        gridded =  xr.Dataset(
               {'tx':gridded_tx, 'xy':gridded_xy, 'yt':gridded_yt, },
               {'time': t_coords, 'lat': lat_coords, 'lon': lon_coords}
        )
        gridded.tx.plot()
        gridded.xy.plot()
        gridded.yt.plot()

        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter(nda, sigma=sig, order=0, mode='mirror', truncate=3.0), da)


        coarse_grids = [gridded]
        for i in range(int(np.log2(n_coords)) - 1):
            coarse_factor = 2 ** (i+1)
            coarse_grids.append(
                gridded
                .coarsen(time=coarse_factor, lat=coarse_factor, lon=coarse_factor)
                .mean()
            )

        pp_grids = []
        prev_cg = None
        for i, (cg, ncg) in enumerate(zip(coarse_grids[:-1], coarse_grids[1:])):
            
            pp_grids.append(
                    cg
                    .pipe(lambda ds: ds - ncg.interp_like(ds, method='nearest', kwargs={"fill_value": "extrapolate"}))
                    .fillna(0.)
                    .interp_like(gridded, method='linear', kwargs={"fill_value": "extrapolate"})
                    .pipe(lambda ds: ds.rename_vars( {n: f'{n}_{i}'for n in ds}))
            )

        pp_grids.append(
                coarse_grids[-1]
                .pipe(lambda ds: ds - gridded.mean())
                .fillna(0.)
                .interp_like(gridded, method='linear', kwargs={"fill_value": "extrapolate"})
                .pipe(lambda ds: ds.rename_vars( {n: f'{n}_{i+1}'for n in ds}))
        )
        pp_grids.append(
                xr.full_like(gridded, dict(gridded.mean()))
                .pipe(lambda ds: ds.rename_vars( {n: f'{n}_mean' for n in ds}))
        )
        
        merged_pp = xr.merge(pp_grids)
        inp = merged_pp.sel(time=gridded.time.isel(time=win//2 + 1), method='nearest')
        inp.broadcast_like(inp.xy_0).to_array()

        # v = 'xy'
        # v = 'tx'
        v = 'yt'
        da = pp_grids[0][v]
        for ppg in pp_grids[1:]:
            da = da  + ppg[v]
        da.plot()
        grid_ds.ssh.plot.pcolormesh('lon', 'lat', col='time', col_wrap=3)

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        ...
        fn = explore

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    locals().update(main())
