# %% TODOs: [md]
"""
- [x]  Métriques swot_sst
- [x]  Métriques roll_sst
- []  figures laplacien roll vs swot
- []  figures gradient roll vs swot
- []  pourcentage variance expliquée
- []  cartes norm grad vrai, grad swot+roll, grad pred
"""

# %% Soft reload
"""
%load_ext autoreload
%autoreload 2
"""

# %% Run all file
"""
%run 
"""
# %% Hard reload
"""
import sys
del sys.modules['main'], sys.modules['models'], sys.modules['new_dataloading']
"""

# %% [md]
"""
I want to compute metrics on the swath from the models 4Dvarnet with roll
"""
# %% Load nad_sst model
from functools import reduce
import xrft
import einops
import pytorch_lightning as pl
# from models import LitModel, LitModelWithSST
import models
# from new_dataloading import FourDVarNetDataModule
import new_dataloading
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
import xarray as xr
from collections import defaultdict
# from main import FourDVarNetRunner
import main
mods = {}

# Mod sst
def get_most_recent_ckpt(config_pkg):
    ckpt_fold = Path(f'dashboard/current/train/{config_pkg}/checkpoints')
    checkpoints = ckpt_fold.glob('*')
    return max(checkpoints, key=lambda x: x.stat().st_ctime)

for config_pkg in [
        # 'q.nad_sst',
        # 'q.xp_one.low_zeros_glob',
        # 'q.xp_one.low_zeros_loc',
        # 'q.xp_one.high_zeros_glob',
        'q.xp_one.low_obs_glob',
        # 'q.swot',
        # 'q.nad_roll',
        # 'q.roll',
        # 'q.roll_sst',
        # 'q.swot_sst',
    ]:
    ckpt_path =  get_most_recent_ckpt(config_pkg)
    runner = main.FourDVarNetRunner(config=config_pkg)

    mods[config_pkg] = runner._get_model(ckpt_path=ckpt_path)


print(" #### ", config_pkg, " #### ")
# %% Generate maps

trainer = pl.Trainer(gpus=1)
for mod in mods.values():
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

# test_xr_ds = runner.dataloaders['test'].dataset.datasets[0].gt_ds.ds
test_xr_ds = runner.dataloaders['test'].dataset.datasets[0].obs_target_ds.ds.rename({'swot': 'ssh'})
dt_start = str(pd.to_datetime(test_xr_ds.time.min().data).date())
dt_end = str(pd.to_datetime(test_xr_ds.time.max().data).date())

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

swot_ds = raw_item['swot'].assign(err=lambda ds:
        ds.roll_err
        + ds.phase_err
        + ds.timing_err
        + ds.bd_err)

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

swot_chunk_with_op = compute_duacs_cal(swot_chunk, raw_item['oi'])
swot_with_op = compute_duacs_cal(swot_ds, raw_item['oi'])


# %% Compute spatial resolution


def get_swath_psd_score(x_t, x, ref, with_fig=False):
    def psd_score(da: xr.DataArray) -> xr.DataArray:
        err = x_t - da
        psd_x_t = (
            x_t.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='linear'))
                .mean('x_ac')
        ).compute()

        psd_err = (
            err.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='linear'))
                .mean('x_ac')
        ).compute()
        psd_score = 1 - psd_err / psd_x_t
        return psd_score

    ref_score = psd_score(ref)
    model_score = psd_score(x)

    ref_score = ref_score.where(model_score > 0, drop=True).compute()
    model_score = model_score.where(model_score > 0, drop=True).compute()

    psd_plot_data: xr.DataArray = xr.DataArray(
        einops.rearrange([model_score.data, ref_score.data], 'var wl -> var wl'),
        name='PSD score',
        dims=('var', 'wl'),
        coords={
            'wl': ('wl', 1 / model_score.freq_x_al, {'long_name': 'Wavelength', 'units': 'km'}),
            'var': ('var', ['model', 'OI'], {}),
        },
    )

    spatial_resolution_model = (
        xr.DataArray(
            psd_plot_data.wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='model').data}
        ).interp(psd=0.5)
    )

    spatial_resolution_ref = (
        xr.DataArray(
            psd_plot_data.wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='OI').data}
        ).interp(psd=0.5)
    )

    if not with_fig:
        return spatial_resolution_model, spatial_resolution_ref

    fig, ax = plt.subplots()
    psd_plot_data.plot.line(x='wl', ax=ax)

    # Plot vertical line there
    for i, (sr, var) in enumerate([(spatial_resolution_ref, 'OI'), (spatial_resolution_model, 'model')]):
        plt.axvline(sr, ymin=0, color='0.5', ls=':')
        plt.annotate(f"resolution {var}: {float(sr):.2f} km", (sr * 1.1, 0.1 * i))
        plt.axhline(0.5, xmin=0, color='k', ls='--')
        plt.ylim([0, 1])

    plt.close()
    return fig, spatial_resolution_model, spatial_resolution_ref

fmted_chunk = reindex(
        swot_chunk_with_op,
        ('x_ac', 'x_al')
)
size = (10, 2)
x_t = fmted_chunk.ssh_model
x =  fmted_chunk.uncal
ref = fmted_chunk.oi

fig, mod_res, oi_res = get_swath_psd_score(x_t, x, ref, with_fig=True)

# %% Grid swath data
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
            .mean(numeric_only=True)
            .pipe(lambda df: xr.Dataset.from_dataframe(df))
    )

gridded_swath_ds = grid_ds(swot_with_op)
# %% Test grid
target_grid_ds = (
   test_xr_ds.isel(time=slice(2,-2))
   .pipe(lambda ds: 
       ds.assign(
          obs_tgt=(ds.ssh.dims, mod.obs_gt),
          obs_inp=(ds.ssh.dims, mod.obs_inp),
          oi=(ds.ssh.dims, mod.x_oi),
          pred=(ds.ssh.dims, mod.x_rec),
          gt=(ds.ssh.dims, mod.x_gt),
          )
       )
   )


swathed_grid_ds = (
    target_grid_ds.interp(
        time=swot_with_op.time,
        lat=swot_with_op.lat,
        lon=swot_with_op.lon - 360,
    )
)


fig, ax = plt.subplots(figsize=(15,3))
ax.imshow(swathed_grid_ds.obs_tgt.T.data[:, :1000])



# %% DS
tgt_grid = (
    gridded_swath_ds
    .drop(labels=['lat', 'lon'])
    .rename({
        'grid_time': 'time',
        'grid_lat': 'lat',
        'grid_lon': 'lon',
        'oi': 'swath_oi',
    })
    .pipe(lambda ds:
        ds.assign_coords( 
            {
                'time': pd.to_datetime(ds.time) + pd.to_timedelta('12H'),
                'lon': (ds.lon - 360).astype('float64', casting='same_kind'),
                'lat': ds.lat.astype('float64', casting='same_kind'),
            }
        )
    )
    .pipe(lambda ds:
        ds.reindex_like(
            target_grid_ds,
            method='nearest',
            tolerance=1e-3)
        )
    .pipe(lambda ds: 
        xr.merge(
            [
                target_grid_ds,
                ds,
            ]
        )
    )
)
tgt_grid.uncal.isel(time=4).plot()
tgt_grid.obs_tgt.isel(time=4).plot()


gridded_data = {
    'tgt': {
        'swath': tgt_grid.pipe(lambda ds: ds.uncal - ds.err),
        'grid': tgt_grid.ssh,
        },
    'src': {
        'err': tgt_grid.uncal,
        'duacs_cal': tgt_grid.pipe(lambda ds: ds.uncal - ds.op),
        'duacs_4_nad': tgt_grid.oi,
        'natl': tgt_grid.gt,
        '4dvarnet_' + config_pkg: tgt_grid.pred,
        }
}

for tgt_da in gridded_data['tgt'].values():
    tgt_da.isel(time=t_idx).plot() and plt.show()

for src_da in gridded_data['src'].values():
    src_da.isel(time=t_idx).plot() and plt.show()


def mse_score(data):
    scores =[] 
    for tgt, tgt_da in gridded_data['tgt'].items():
        for src, src_da in gridded_data['src'].items():
            err = tgt_da - src_da
            count = (~np.isnan(err)).sum().data
            mse = float(np.mean(err**2))
            fig, ax = plt.subplots()
            err.isel(time=t_idx).plot(ax=ax)
            scores.append({
                'count': count,
                'score': mse,
                'fig': fig,
                'src': src,
                'tgt': tgt,
            })
    print(pd.DataFrame(scores).drop('fig', axis=1).to_markdown())
    return scores


scores = mse_score(gridded_data)


import scipy
from scipy import ndimage

def sobel(da):
    dlat = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lat'))))
    dlon = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lon'))))
    return np.hypot(dlat, dlon)

def lapla(da):
    return da.pipe(lambda da:  da.groupby('time').apply(lambda da: scipy.ndimage.filters.laplace(da)))

dlat = gridded_data['tgt']['grid'][t_idx, ...].pipe(lambda da:  ndimage.sobel(da, da.dims.index('lat')))
plt.imshow(dlat)

dlon = gridded_data['tgt']['grid'][t_idx, ...].pipe(lambda da:  ndimage.sobel(da, da.dims.index('lon')))
plt.imshow(dlon)
plt.imshow(dlat + dlon)
plt.imshow(np.hypot(dlat, dlon))

sob = sobel(gridded_data['tgt']['grid'])[t_idx, ...]
plt.imshow(sobel(gridded_data['tgt']['grid'])[t_idx, ...])

def grad_mse_score(data):

    grad_scores =[] 
    for tgt, tgt_da in gridded_data['tgt'].items():
        for src, src_da in gridded_data['src'].items():
            err = sobel(tgt_da) - sobel(src_da)
            count = (~np.isnan(err)).sum().data
            mse = float(np.mean(err**2))
            fig, ax = plt.subplots()
            err.isel(time=t_idx).plot(ax=ax)
            grad_scores.append({
                'count': count,
                'score': mse,
                'fig': fig,
                'src': src,
                'tgt': tgt,
            })
    print(pd.DataFrame(grad_scores).drop('fig', axis=1).to_markdown())
    return grad_scores


grad_scores = grad_mse_score(gridded_data)

# Plot and save everything

"""
So I want to plot
- ssh
- grad ssh
- lap ssh

For
- natl
- swot obs
- swot obs with roll
- swot obs with roll after cal
"""

fig, axs = plt.subplots(4, 3, figsize=(15, 15))
das =  [
        gridded_data['tgt']['grid'],
        gridded_data['src']['err'],
        gridded_data['src']['duacs_cal'],
        gridded_data['src']['4dvarnet_' + config_pkg],
] 
for i, da in enumerate(das):
    axs[i, 0].imshow(da.isel(time=t_idx).data)
    axs[i, 1].imshow(sobel(da).isel(time=t_idx).data)
    axs[i, 2].imshow(lapla(da).isel(time=t_idx).data)

# %% Results NAD SST[md]
"""
#  Grid scores

## MSE scores
|    |   count |       score | src                | tgt   |
|---:|--------:|------------:|:-------------------|:------|
|  0  |   62615 | 0.000835879 | roll               | swath |
|  1  |   62615 | 0.000138609 | duacs_cal          | swath |
|  2  |   62615 | 0.00198292  | duacs_4_nad        | swath |
|  3 |   62615 | 0.00023718  | natl            | swath |
|  4  |   62615 | 0.000931464 | 4dvarnet_q.nad_sst | swath |
|  5 |   62615 | 0.000626997 | 4dvarnet_q.swot | swath |
|  6 |   62615 | 0.00123173  | 4dvarnet_q.nad_roll | swath |
|  7 |   62615 | 0.00393216  | 4dvarnet_q.roll | swath |
|  4 |   62615 | 0.00075328  | 4dvarnet_q.roll | swath |
|  4 |   62615 | 0.00193471  | 4dvarnet_q.roll_sst | swath |
|  4 |   62615 | 0.00375245  | 4dvarnet_q.swot_sst | swath |
|  8  |   62615 | 0.000837556 | roll               | grid  |
|  9  |   62615 | 0.000139124 | duacs_cal          | grid  |
|  10  |   77260 | 0.00200381  | duacs_4_nad        | grid  |
|  8 |   77260 | 0.000224822 | natl            | grid  |
|  12  |   77259 | 0.000934756 | 4dvarnet_q.nad_sst | grid  |
|  9 |   77260 | 0.00130725  | 4dvarnet_q.nad_roll | grid  |
|  9 |   77260 | 0.000645756 | 4dvarnet_q.swot | grid  |
|  9 |   77260 | 0.00377194  | 4dvarnet_q.roll | grid  |
|  9 |   77260 | 0.000796328 | 4dvarnet_q.roll | grid  |
|  9 |   77260 | 0.00198621  | 4dvarnet_q.roll_sst | grid  |
|  9 |   77260 | 0.00370027  | 4dvarnet_q.swot_sst | grid  |




## Grad MSE scores

|    |   count |       score | src                | tgt   |
|---:|--------:|------------:|:-------------------|:------|
|  0 |   50199 | 0.000495675 | roll               | swath |
|  1 |   50199 | 0.000121385 | duacs_cal          | swath |
|  2 |   50199 | 0.00331929  | duacs_4_nad        | swath |
|  3 |   50199 | 0.000405146 | natl            | swath |
|  4 |   50199 | 0.00168507  | 4dvarnet_q.nad_sst | swath |
|  4 |   50199 | 0.00259283  | 4dvarnet_q.roll_sst | swath |
|  5 |   50199 | 0.00143043  | 4dvarnet_q.swot | swath |
|  6 |   50199 | 0.00217775  | 4dvarnet_q.nad_roll | swath |
|  7 |   50199 | 0.00349608  | 4dvarnet_q.roll | swath |
|  4 |   50199 | 0.00149082  | 4dvarnet_q.roll | swath |
|  4 |   50199 | 0.00228682  | 4dvarnet_q.swot_sst | swath |
|  8 |   50199 | 0.000499668 | roll               | grid  |
|  9 |   50199 | 0.000124958 | duacs_cal          | grid  |
|  10 |   63139 | 0.00330375  | duacs_4_nad        | grid  |
|  11 |   63139 |  0.000387403| natl               | grid  |
|  12 |   63139 | 0.0017559   | 4dvarnet_q.nad_sst | grid  |
|  13 |   63139 | 0.00149355  | 4dvarnet_q.swot | grid  |
|  14 |   63139 | 0.00224537  | 4dvarnet_q.nad_roll | grid  |
|  15 |   63139 | 0.00343332  | 4dvarnet_q.roll | grid  |
|  9 |   63139 | 0.00157915  | 4dvarnet_q.roll | grid  |
|  9 |   63139 | 0.00261241  | 4dvarnet_q.roll_sst | grid  |
|  9 |   63139 | 0.0022965   | 4dvarnet_q.swot_sst | grid  |


"""


"""

training with same weights as glob loss

|    |   count |   grad_mse_score |    mse_score | src                | tgt   |
|  5 |   62615 |  0.000499668 | 0.000837556 | roll            | grid  |noisy obs|
|  9  |   62615|  0.00330375 | 0.000139124 | duacs_cal          | grid  |duacs calibration|
|  9 |   77260 |  0.00343332 | 0.00377194  | 4dvarnet_q.roll | grid  | base 4dvarnet training with noisy obs (no loss term on swath)|
|  9 |   77260 |  0.00149976 | 0.000740362 | 4dvarnet_q.roll | grid  | With a loss term on the swath |
|  9 |   77260 |  0.00145627 | 0.000630234 | 4dvarnet_q.nad_roll | grid  | With the noisy obs separated from nadirs|
|  9 |   77260 |  0.000875856 | 0.000283701 | 4dvarnet_q.swot | grid  | Without perfect obs |

grad

|  5 |   50199 | 0.000499668 | roll            | grid  |
|  10 |  63139 | 0.00330375  | duacs_4_nad        | grid  |
|  15 |  63139 | 0.00343332  | 4dvarnet_q.roll | grid  |
|  9 |   63139 | 0.00149976  | 4dvarnet_q.roll | grid  |
|  9 |   63139 | 0.00145627  | 4dvarnet_q.nad_roll | grid  |
|  9 |   63139 | 0.000875856 | 4dvarnet_q.swot | grid  |


With Perfect Obs:

|  5 |   62615 |  0.000499668 | 0.000837556 | roll            | grid  |noisy obs|
|  9  |   62615|  0.000124958 | 0.000139124 | duacs_cal          | grid  |duacs calibration|

|  9 |   77260 | 0.000253226 | 0.000107937 | 4dvarnet_q.xp_one.low_zeros_glob | grid  |
|  9 |   77260 | 0.00241264  | 0.00158863  | 4dvarnet_q.xp_one.low_zeros_loc | grid  |
|  9 |   77260 | 0.000236198 | 8.48103e-05 | 4dvarnet_q.xp_one.high_zeros_glob | grid  |
|  9 |   77260 | 0.000262276 | 9.32416e-05 | 4dvarnet_q.xp_one.low_obs_glob | grid  |

With roll:

|  9 |   77260 | 0.00029474  | 0.000135757 | 4dvarnet_q.xp_one.low_zeros_glob | grid  |
|  9 |   77260 | 0.0026019   | 0.00172623  | 4dvarnet_q.xp_one.low_zeros_loc | grid  |
|  9 |   77260 | 0.000227642 | 9.36744e-05 | 4dvarnet_q.xp_one.high_zeros_glob | grid  |
|  9 |   77260 | 0.000264168 | 9.90135e-05 | 4dvarnet_q.xp_one.low_obs_glob | grid  |

With all errs

|  5 |   50199 | 0.00253221  | 0.00434421  | err                             | grid  |
|  6 |   50199 | 0.00110035  | 0.00188047  | duacs_cal                       | grid  |
|  8 |   63139 | 0.000387403 | 0.000224822 | natl                            | grid  |
|  7 |   63139 | 0.00330375  | 0.00200381  | duacs_4_nad                     | grid  |

|  9 |   77260 | 0.00012595  | 0.000264325 | 4dvarnet_q.xp_one.low_zeros_glob | grid  |
|  9 |   63139 | 0.00260308  | 0.00174058  | 4dvarnet_q.xp_one.low_zeros_loc | grid  |
|  9 |   63139 | 0.000225968 | 8.58969e-05 | 4dvarnet_q.xp_one.high_zeros_glob | grid  |
|  9 |   63139 | 0.000274901 | 9.16486e-05 | 4dvarnet_q.xp_one.low_obs_glob | grid  |
"""
