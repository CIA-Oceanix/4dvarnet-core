# %% Load nad_sst model
from scipy import ndimage
import pytorch_lightning as pl
# from models import LitModel, LitModelWithSST
import models
from pathlib import Path
# from new_dataloading import FourDVarNetDataModule
import new_dataloading
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
import xarray as xr
from collections import defaultdict
# from main import FourDVarNetRunner
import main

def get_most_recent_ckpt(config_pkg, xp='no_roll'):
    ckpt_fold = Path(f'dashboard/{xp}/train/{config_pkg}/checkpoints')
    checkpoints = ckpt_fold.glob('*')
    return max(checkpoints, key=lambda x: x.stat().st_ctime)

# ckpt_path= "first_results_dash/train/nad_roll/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.07.ckpt"
config_pkg = 'q.xp_one.low_zeros_glob'
ckpt_path = get_most_recent_ckpt(
        config_pkg=config_pkg,
        # xp='current',
        # xp='no_roll',
        xp='roll',
    )
print(ckpt_path)
runner = main.FourDVarNetRunner(config=config_pkg)

mod = runner._get_model(ckpt_path=ckpt_path)


print(" #### ", config_pkg, " #### ")
# %% Generate maps

trainer = pl.Trainer(gpus=1)
trainer.test(mod, test_dataloaders=runner.dataloaders['test'])

# %%

import matplotlib.pyplot as plt
"""
%matplotlib inline
"""
t_idx = 4
plt.imshow(mod.obs_gt[t_idx,...])
plt.imshow(mod.x_rec[t_idx,...])
plt.imshow(mod.x_oi[t_idx,...])
plt.imshow(mod.obs_inp[t_idx,...])

# %%
# %% Load swath data

import sys
sys.path.append('../research-quentin')


from src.commons.coords_to_dim import coords_to_dim, reindex
from src.data_processing.get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice

test_xr_ds = runner.dataloaders['test'].dataset.datasets[0].gt_ds.ds
# test_xr_ds = runner.dataloaders['test'].dataset.datasets[0].obs_target_ds.ds.rename({'swot': 'ssh'})
# test_xr_ds = mod.test_xr_ds
dt_start = str(pd.to_datetime(test_xr_ds.time.min().data.item()).date())
dt_end = str(pd.to_datetime(test_xr_ds.time.max().data.item()).date())

slice_args = {
    "time_min":dt_start,
    "time_max":dt_end,
    "lat_min": 33,
    "lat_max": 43,
    "lon_min": 295,
    "lon_max": 305,
}

raw_item = {
    'nadirs': xr.concat([get_nadir_slice(f'../research-quentin/data/zarr/nadir/{name}', **slice_args) for name in
                         ['swot', 'en', 'tpn', 'g2', 'j1']], dim='time'),
    'swot': get_swot_slice(f'../research-quentin/data/zarr/swot', **slice_args),
    'swot_nadir': get_nadir_slice(f'../research-quentin/data/zarr/nadir/swot', **slice_args),
    'oi': get_oi_slice('../research-quentin/data/raw/DUACS-OI_maps/ssh_model/ssh_sla_boost_NATL60_en_j1_tpn_g2.nc', **slice_args),
    'natl': get_natl_slice('../research-quentin/data/raw/NATL60_regular_grid/1_10/natl60CH_H.nc', **slice_args),
}

# %%
def clean_oi(ds, var='ssh', thres=10):
    return (
        ds
        .assign({f'clean_{var}': lambda _ds: (_ds[var].dims, np.where(_ds[var] < thres, _ds[var], np.nan))})
        .drop(var)
        .rename({f'clean_{var}': var})
    )

# swot_ds = raw_item['swot']
swot_ds = raw_item['swot'].assign(err=lambda ds:
        ds.roll_err
        # + ds.phase_err
        # + ds.timing_err
        # + ds.bd_err
        )
nadirs = raw_item['nadirs']
swot_nadir = raw_item['swot_nadir']
oi_ds = reindex(clean_oi(raw_item['oi']), ('time', 'lon', 'lat'))
natl_ds = reindex(raw_item['natl'], ('time', 'lon', 'lat'))


# %%

slope = lambda da: (da.diff('nC') / da.x_ac.diff('nC')).mean('nC')

swot_nadir_w_ch = swot_nadir.assign(
    contiguous_chunk=lambda ds: (ds.time.diff('time') / np.timedelta64(1, 's') > 1).cumsum()
)
chunk_nb = swot_nadir_w_ch.groupby('contiguous_chunk').count().ssh_model.argmax('contiguous_chunk')
swot_nadir_chunk = (
    swot_nadir_w_ch.pipe(
        lambda ds: ds.isel(time = ds.contiguous_chunk==chunk_nb)
    )
)

swot_chunk =  swot_ds.sel(time=swot_nadir_chunk.time)
# swot_chunk =  swot_ds.isel(time=slice(1000, 3000))

swot_chunk.ssh_model.T.plot(figsize=(10,3))

# %%
def compute_duacs_cal(swot_ds, oi_ds):
    return  (
        swot_ds
        .assign(uncal=lambda ds: ds.ssh_model + ds.err)
        .assign(
            oi=lambda ds: (
                ds.ssh_model.dims, 
                oi_ds
                .pipe(lambda ds: coords_to_dim(ds, ('lat', 'lon'), drop=('x', 'y')))
                .assign({f'clean_ssh': lambda _ds: (_ds.ssh.dims, np.where(_ds.ssh < 10, _ds.ssh, np.nan))})
                .clean_ssh
                .interp(
                    time=ds.time.broadcast_like(ds.ssh_model),
                    lat=ds.lat.broadcast_like(ds.ssh_model),
                    lon=ds.lon.broadcast_like(ds.ssh_model),
                ).data
            )
        ).pipe(lambda ds: ds.isel(time = ~np.isnan(ds.oi).any('nC')))
        .assign(res=lambda ds: ds.uncal - ds.oi)
        .assign(conti_chunks=lambda ds: (ds.time.dims, np.cumsum(np.diff(ds.time.values.astype(float), prepend=0) > pd.to_timedelta(0.5, 's') / pd.to_timedelta(1, 'ns'))))
        .pipe(
            lambda ds: ds.assign(
                op = (
                    ds.res.dims,
                    slope(ds).to_dataframe()
                    .groupby('conti_chunks')
                    .apply(
                        lambda df: df.assign(op_slope=lambda df: df.res.rolling(225, win_type="gaussian", center=True, min_periods=1).mean(std=75))
                    )
                    .reset_index(level=0, drop=True)
                    .pipe(xr.Dataset.from_dataframe)
                    .pipe(lambda _ds: _ds.op_slope * ds.x_ac)
                )
            )
        )
    )

swot_chunk_with_op = compute_duacs_cal(swot_chunk, raw_item['oi'])
swot_with_op = compute_duacs_cal(swot_ds, raw_item['oi'])

chunk_date = pd.to_datetime(swot_chunk.time)[0].date()
obs_pred_swath = mod.test_xr_ds.sel(time=str(chunk_date)).drop('time').obs_pred.interp(lat=swot_chunk_with_op.lat, lon=swot_chunk_with_op.lon-360,)
natl_swath = mod.test_xr_ds.sel(time=str(chunk_date)).drop('time').gt.interp(lat=swot_chunk_with_op.lat, lon=swot_chunk_with_op.lon-360,)

# %% plots on swath
fmted_chunk = reindex(
        swot_chunk_with_op.assign(
            pred_err=lambda ds: obs_pred_swath.data - ds.ssh_model,
            pred=lambda ds: (ds.ssh_model.dims, obs_pred_swath.data),
            natl_err=lambda ds: natl_swath.data - ds.ssh_model,
            duacs_cal_err=lambda ds: ds.op - ds.err ),
        ('x_ac', 'x_al')
        ).reindex({'x_ac': np.arange(-62, 64, 2)})
size = (10, 2)

vmin, vmax = -0.05, +0.05
fmted_chunk.uncal.plot(figsize=size)
fmted_chunk.ssh_model.plot(figsize=size)
im = fmted_chunk.err.plot(figsize=size)
im.set_clim(vmin, vmax)
im = fmted_chunk.pred_err.plot(figsize=size)
im.set_clim(vmin, vmax)
im = fmted_chunk.natl_err.plot(figsize=size)
im.set_clim(vmin, vmax)
im = fmted_chunk.duacs_cal_err.plot(figsize=size)
im.set_clim(vmin, vmax)
fmted_chunk.op.plot(figsize=size)

fig, ax = plt.subplots()
slope(swot_chunk_with_op).op.plot(ax=ax)
slope(swot_chunk_with_op).err.plot(ax=ax)
fig.show()

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 0), da)
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 1), da)
    return np.hypot(dx_ac, dx_al)

vmin, vmax = 0., 0.07
sobel(fmted_chunk.uncal).plot(figsize=size)
sobel(fmted_chunk.ssh_model).plot(figsize=size)
im = sobel(fmted_chunk.err).plot(figsize=size)
im.set_clim(vmin, vmax)
im = sobel(fmted_chunk.pred_err).plot(figsize=size)
im.set_clim(vmin, vmax)
im = sobel(fmted_chunk.natl_err).plot(figsize=size)
im.set_clim(vmin, vmax)
im = sobel(fmted_chunk.duacs_cal_err).plot(figsize=size)
im.set_clim(vmin, vmax)

def lap(da):
    return xr.apply_ufunc(ndimage.filters.laplace, da)

vmin, vmax = -0.015, 0.015
lap(fmted_chunk.uncal).plot(figsize=size)
lap(fmted_chunk.ssh_model).plot(figsize=size)
im = lap(fmted_chunk.err).plot(figsize=size)
im.set_clim(vmin, vmax)
im = lap(fmted_chunk.pred_err).plot(figsize=size)
im.set_clim(vmin, vmax)
im = lap(fmted_chunk.natl_err).plot(figsize=size)
im.set_clim(vmin, vmax)
im = lap(fmted_chunk.duacs_cal_err).plot(figsize=size)
im.set_clim(vmin, vmax)

# plot on grid
res = dict(
    time= np.timedelta64(1, 'D') / np.timedelta64(1, 'ns'),
    lat=1/20,
    lon=1/20,
)
def grid_ds(ds, resolutions=res):
    return (
        ds.assign(
            **{
                f'grid_{coord}': ds[coord].astype(float) / resolutions[coord] // 1 *
                                 resolutions[coord]
                for coord in resolutions.keys()
            }
        ).pipe(lambda ds: ds.to_dataframe())
            .reset_index()
            .dropna(subset=['ssh_model'])
            .groupby([f'grid_{coord}' for coord in resolutions.keys()])
            .mean(numeric_only=False)
            .pipe(lambda df: xr.Dataset.from_dataframe(df))
    )

gridded_swath_ds = grid_ds(swot_with_op)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
t_idx=3
gridded_swath_ds.ssh_model.isel(grid_time=t_idx).plot(ax=axs[0])
axs[0].set_title('ssh_model')
gridded_swath_ds.uncal.isel(grid_time=t_idx).plot(ax=axs[1])
axs[1].set_title('uncal')
gridded_swath_ds.err.isel(grid_time=t_idx).plot(ax=axs[2])
axs[2].set_title('roll')
fig.show()

def sobel_grid(da):
    dlat = da.pipe(lambda da:  da.groupby('grid_time').apply(lambda da: ndimage.sobel(da, da.dims.index('grid_lat'))))
    dlon = da.pipe(lambda da:  da.groupby('grid_time').apply(lambda da: ndimage.sobel(da, da.dims.index('grid_lon'))))
    return np.hypot(dlat, dlon)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
t_idx=3
sobel_grid(gridded_swath_ds.ssh_model).isel(grid_time=t_idx).plot(ax=axs[0])
axs[0].set_title('ssh_model')
sobel_grid(gridded_swath_ds.uncal).isel(grid_time=t_idx).plot(ax=axs[1])
axs[1].set_title('uncal')
sobel_grid(gridded_swath_ds.err).isel(grid_time=t_idx).plot(ax=axs[2])
axs[2].set_title('roll')
fig.show()

def lapl_grid(da):
    return da.pipe(lambda da:  da.groupby('grid_time').apply(lambda da: ndimage.filters.laplace(da)))

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
t_idx=3
lapl_grid(gridded_swath_ds.ssh_model).isel(grid_time=t_idx).plot(ax=axs[0])
axs[0].set_title('ssh_model')
lapl_grid(gridded_swath_ds.uncal).isel(grid_time=t_idx).plot(ax=axs[1])
axs[1].set_title('uncal')
lapl_grid(gridded_swath_ds.err).isel(grid_time=t_idx).plot(ax=axs[2])
axs[2].set_title('roll')
fig.show()


## %% 
for t in pd.to_datetime(gridded_swath_ds.grid_time):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))
    mod.test_xr_ds.sel(time=str(t.date())).obs_gt.plot(ax=ax1)
    gridded_swath_ds.sel(grid_time=float(t.asm8)).ssh_model.plot(ax=ax2)
    natl_ds.sel(time=str(t)).H.T.plot(ax=ax3)
    fig.show()

for t in pd.to_datetime(test_xr_ds.time):
    fig, (ax1) = plt.subplots()
    test_xr_ds.sel(time=str(t)).ssh.plot(ax=ax1)
    fig.show()

# %% 

# chunk_date = pd.to_datetime(gridded_swath_ds.grid_time)
chunk_date = pd.to_datetime(swot_chunk.time)[0].date()
mod.test_xr_ds.time
chunk_date
obs_pred = mod.test_xr_ds.sel(time=str(chunk_date)).drop('time').obs_pred.interp(lat=swot_chunk.lat, lon=swot_chunk.lon-360,).T.plot(figsize=(10, 3))
natl_swath = mod.test_xr_ds.sel(time=str(chunk_date)).drop('time').obs_pred.interp(lat=swot_chunk.lat, lon=swot_chunk.lon-360,).T.plot(figsize=(10, 3))
swot_chunk.ssh_model.T.plot(figsize=(10,3))

