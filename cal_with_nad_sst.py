# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: region_name,title,-all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [md]
"""
### Objective


* nad_sst_cal:
    ** Generate interpolation from trained model with nadir + sst
    ** Compare with calibration made using DUACS with 4 nadirs

* Compute $\lambda_x$ et $\lambda_t$ for different xps 
"""

# %% Load nad_sst model
import pytorch_lightning as pl
from models import LitModel, LitModelWithSST
from new_dataloading import FourDVarNetDataModule
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
import xarray as xr
from collections import defaultdict


ckpt_path="dashboard/train/nad_sst/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.05.ckpt"


config = __import__("config_q.nad_sst")


cfg = OmegaConf.create(config.params)
shapeData = cfg.shapeData
w_ = np.zeros(cfg.dT)
w_[int(cfg.dT / 2)] = 1.
wLoss = torch.Tensor(w_)
dataloading = config.params['dataloading']
print(dataloading)

dim_range = config.dim_range
slice_win = config.slice_win
strides = config.strides



datamodule = FourDVarNetDataModule(
    slice_win=slice_win,
    dim_range=dim_range,
    strides=strides,
    **config.params['files_cfg'],
)


datamodule.setup()
dataloaders = {
    'train': datamodule.train_dataloader(),
    'val': datamodule.val_dataloader(),
    'test': datamodule.val_dataloader(),
}
mean_Tr = datamodule.norm_stats[0]
mean_Tt = datamodule.norm_stats[0]
mean_Val = datamodule.norm_stats[0]
var_Tr = datamodule.norm_stats[1] ** 2
var_Tt = datamodule.norm_stats[1] ** 2
var_Val = datamodule.norm_stats[1] ** 2
min_lon, max_lon, min_lat, max_lat = datamodule.bounding_box
ds_size_time = datamodule.ds_size['time']
ds_size_lon = datamodule.ds_size['lon']
ds_size_lat = datamodule.ds_size['lat']

lit_cls =LitModelWithSST
mod = LitModelWithSST.load_from_checkpoint(ckpt_path, w_loss=wLoss,
                                        mean_Tr=mean_Tr, mean_Tt=mean_Tt, mean_Val=mean_Val,
                                            var_Tr=var_Tr, var_Tt=var_Tt, var_Val=var_Val,
                                            min_lon=min_lon, max_lon=max_lon,
                                            min_lat=min_lat, max_lat=max_lat,
                                            ds_size_time=ds_size_time,
                                            ds_size_lon=ds_size_lon,
                                            ds_size_lat=ds_size_lat)

# %% Generate maps

trainer = pl.Trainer(gpus=1)
trainer.test(mod, test_dataloaders=dataloaders['test'])



# %%
# %matplotlib inline

# %%


mod.test_figs['maps']

# %%
mod.test_figs['mse']

# %%
grid

# %%
raw_item['natl']

# %%
mod.x_gt.shape

# %% Get cal data

dates = pd.date_range("2012-10-01", "2013-10-01")
dt_start, dt_end = dates[60], dates[60+16]


import sys
sys.path.append('../research-quentin')


from src.commons.coords_to_dim import coords_to_dim, reindex
from src.data_processing.get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice


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
swot_ds = raw_item['swot']
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

# swot_chunk =  swot_ds.sel(time=swot_nadir_chunk.time)
swot_chunk =  swot_ds.isel(time=slice(1000, 3000))


# %%
@st.cache(hash_funcs={xr.core.dataset.Dataset: lambda ds: ''})
def compute_duacs_cal(swot_ds, oi_ds):
    swot_chunk_with_op = (
        swot_ds
        .assign(uncal=lambda ds: ds.ssh_model + ds.roll_err)
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
    return swot_chunk_with_op

swot_chunk_with_op = compute_duacs_cal(swot_chunk, raw_item['oi'])

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
swot_chunk_with_opswot_chunk_with_op.op.T.plot(ax=ax, figsize=(15,3))
fig
