import pytorch_lightning as pl
from pathlib import Path
import main

configs = [
    'q.xp_two.err',
    'q.xp_two.err_no_glob',
    'q.xp_two.err_no_loc',
    'q.xp_two.roll',
    'q.xp_two.roll_no_glob',
    'q.xp_two.roll_no_loc',
    'q.xp_two.swot',
    'q.xp_two.swot_no_glob',
    'q.xp_two.swot_no_loc',
]

config_pkg = 'q.xp_two.err'

def get_most_recent_ckpt(config_pkg, xp='no_roll'):
    ckpt_fold = Path(f'dashboard/{xp}/train/{config_pkg}/checkpoints')
    checkpoints = ckpt_fold.glob('*')
    return max(checkpoints, key=lambda x: x.stat().st_ctime)


# %% Global maps ssh + grad 
# %% Swath maps ssh + grad
# %% Global scores mse, grad mse, spat res
# %% Swath scores mse, grad mse, spat mse

# %% components decomposition: gde échelle, anom glob, anom loc
# ckpt_path= "first_results_dash/train/nad_roll/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.07.ckpt"
mods = {}

for config_pkg in configs:
    ckpt_path = get_most_recent_ckpt(
            config_pkg=config_pkg,
            xp='current',
            # xp='no_roll',
            # xp='roll',
        )
    runner = main.FourDVarNetRunner(config=config_pkg)

    mod = runner._get_model(ckpt_path=ckpt_path)

    print(" #### ", config_pkg, " #### ")
    # %% Generate maps

    trainer = pl.Trainer(gpus=1)
    trainer.test(mod, test_dataloaders=runner.dataloaders['test'])
    mods[config_pkg] = mod

# %% global plot


pt_kw = {'figsize': (6, 5)}
t_idx = 4

from scipy import ndimage
import numpy as np
import xarray as xr

def sobel_grid(da):
    dlat = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, da.dims.index('lat')), da)
    dlon = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, da.dims.index('lon')), da)
    return np.hypot(dlat, dlon)


import holoviews as hv
from holoviews.plotting import mpl

class Clim:
    def __init__(self, das):
        self.das = das

    def __getitem__(self, item):
        print(item)
        if 'grad' in item:
            # print(self.das['gt_grad'].min(), self.das['gt_grad'].max())
            return (self.das['gt_grad'].min(), self.das['gt_grad'].max())
        else:
            # print(self.das['gt'].min(), self.das['gt'].max())
            return (self.das['gt'].min(), self.das['gt'].max())
        
to_plot_das = {}
metrics_das = {}

for cfg, mod in mods.items():
    to_plot_das[cfg] = mod.test_xr_ds.isel(time=t_idx).pred
    to_plot_das[f'{cfg}_grad'] = sobel_grid(mod.test_xr_ds.isel(time=t_idx).pred)

    metrics_das[cfg] = mod.test_xr_ds.pred
    metrics_das[f'{cfg}_grad'] = sobel_grid(mod.test_xr_ds.pred)

to_plot_das['gt'] = mod.test_xr_ds.isel(time=t_idx).gt
to_plot_das['gt_grad'] = sobel_grid(mod.test_xr_ds.isel(time=t_idx).gt)

metrics_das['gt'] = mod.test_xr_ds.gt
metrics_das['gt_grad'] = sobel_grid(mod.test_xr_ds.gt)

metrics_das['oi'] = mod.test_xr_ds.oi
metrics_das['oi_grad'] = sobel_grid(mod.test_xr_ds.oi)

to_plot_ds = xr.Dataset(to_plot_das)
metrics_ds = xr.Dataset(metrics_das)

clims = Clim(to_plot_das)
hv_layout = hv.Layout([
    hv.Dataset(
        to_plot_ds, ['lon', 'lat'], var
    ).to(
        hv.QuadMesh, kdims=['lon', 'lat']
    ).relabel(
        f'{var}'
    ).options(
        colorbar=True,
        cmap='PiYG',
        clim=clims[var],
        aspect=1
    )
    for var in to_plot_das
]).cols(2)

fig_swot_preview = hv.render(hv_layout, backend='matplotlib')

# %% global scores
from metrics import get_psd_score

mse = np.mean((metrics_ds[configs] - metrics_ds.gt)**2)
mse_grad = np.mean((metrics_ds[[f'{c}_grad' for c in configs]] - metrics_ds.gt_grad)**2)
spat_res = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi)[0].item() for c in configs }
spat_res_grad = {c:  get_psd_score(metrics_ds.gt_grad, metrics_ds[f'{c}_grad'], metrics_ds.oi_grad)[0].item()for c in configs }

# %% swath scores

mod_swath_das = {}

for cfg, mod in mods.items():
    mod_swath_das[f'{cfg}_glob'] = mod.test_xr_ds.pred
    mod_swath_das[f'{cfg}_loc'] = mod.test_xr_ds.obs_pred

mod_swath_das['gt_glob'] = mod.test_xr_ds.gt
mod_swath_das['gt_loc'] = mod.test_xr_ds.obs_gt
mod_swath_das['oi'] = mod.test_xr_ds.oi

mod_swath_ds = xr.Dataset(mod_swath_das)

# %% Fetch sensor data
import pandas as pd
import sys
sys.path.append('../research-quentin')


from src.commons.coords_to_dim import coords_to_dim, reindex
from src.data_processing.get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice

dt_start = str(pd.to_datetime(metrics_ds.time.min().data.item()).date())
dt_end = str(pd.to_datetime(metrics_ds.time.max().data.item()).date())

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
fmted_chunk = reindex(swot_chunk, ['x_al', 'x_ac'])
chunk_date = pd.to_datetime(swot_chunk.time)[0].date()
swath_ds = mod_swath_ds.sel(time=str(chunk_date)).drop('time').interp(lat=fmted_chunk.lat, lon=fmted_chunk.lon-360,)
import xrft
import einops
import matplotlib.pyplot as plt

# %% Plots swath ssh grad loc glob

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 0), da)
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 1), da)
    return np.hypot(dx_ac, dx_al)

swath_grad_ds = sobel(swath_ds)

def get_swath_fig(to_plot_ds):
    hv_layout = hv.Layout([
        hv.Dataset(
            to_plot_ds, ['x_al', 'x_ac'], var
        ).to(
            hv.QuadMesh, kdims=['x_al', 'x_ac']
        ).relabel(
            f'{var}'
        ).options(
            colorbar=True,
            cmap='PiYG',
            aspect=3
        )
        for var in to_plot_ds
    ]).cols(2)

    return hv.render(hv_layout, backend='matplotlib')

ssh_fig = get_swath_fig(swath_ds)
grad_fig = get_swath_fig(swath_grad_ds)

# %% Metrics 
ssh_metrics_glob = np.mean((swath_ds[[f'{cfg}_glob' for cfg in configs]] - swath_ds.gt_loc)**2)
ssh_metrics_loc = np.mean((swath_ds[[f'{cfg}_loc' for cfg in configs]] - swath_ds.gt_loc)**2)

grad_metrics_glob = np.mean((swath_grad_ds[[f'{cfg}_glob' for cfg in configs]] - swath_grad_ds.gt_loc)**2)
grad_metrics_loc = np.mean((swath_grad_ds[[f'{cfg}_loc' for cfg in configs]] - swath_grad_ds.gt_loc)**2)

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


    idx = (
            (psd_plot_data.sel(var='model').rolling(wl=3, center=True, min_periods=1).mean() < 0.5).cumsum() <=1
    )

    spatial_resolution_model = (
        xr.DataArray(
            psd_plot_data.sel(var='model').isel(wl=idx).wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='model').isel(wl=idx).data}
        ).interp(psd=0.5)
    )

    idx = (
            (psd_plot_data.sel(var='OI').rolling(wl=3, center=True, min_periods=1).mean() < 0.5).cumsum() <=1
    )
    spatial_resolution_ref = (
        xr.DataArray(
            psd_plot_data.sel(var='OI').isel(wl=idx).wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='OI').isel(wl=idx).data}
        ).interp(psd=0.5)
    )

    if not with_fig:
        return spatial_resolution_model, spatial_resolution_ref

    fig, ax = plt.subplots()
    psd_plot_data.rolling(wl=3, center=True).mean().plot.line(x='wl', ax=ax)

    # Plot vertical line there
    for i, (sr, var) in enumerate([(spatial_resolution_ref, 'OI'), (spatial_resolution_model, 'model')]):
        plt.axvline(sr, ymin=0, color='0.5', ls=':')
        plt.annotate(f"resolution {var}: {float(sr):.2f} km", (sr * 1.1, 0.1 * i))
        plt.axhline(0.5, xmin=0, color='k', ls='--')
        plt.ylim([0, 1])

    plt.close()
    return fig, spatial_resolution_model, spatial_resolution_ref

trim_sw_ds = (
    swath_ds
    .sel(x_ac=list(range(-54, -22, 2)) + list(range(22, 58, 2)))
    .pipe(lambda ds: ds.isel(x_al= (~np.isnan(ds.gt_loc)).all('x_ac')))
    .drop(['lat', 'lon'])
)

trim_grad_sw_ds = sobel(trim_sw_ds)

trim_sw_ds.gt_loc.T.plot(figsize=(10, 3))
swath_spat_res_glob = {c:  get_swath_psd_score(trim_sw_ds.gt_loc, trim_sw_ds[f'{c}_glob'], trim_sw_ds.oi)[0].item() for c in configs}
swath_spat_res_fig_glob = {c:  get_swath_psd_score(trim_sw_ds.gt_loc, trim_sw_ds[f'{c}_glob'], trim_sw_ds.oi, with_fig=True)[0] for c in configs}
swath_spat_res_loc = {c:  get_swath_psd_score(trim_sw_ds.gt_loc, trim_sw_ds[f'{c}_loc'], trim_sw_ds.oi)[0].item() for c in configs}

swath_grad_spat_res_glob = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_loc, trim_grad_sw_ds[f'{c}_glob'], trim_grad_sw_ds.oi)[0].item() for c in configs}
swath_grad_spat_res_loc = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_loc, trim_grad_sw_ds[f'{c}_loc'], trim_grad_sw_ds.oi)[0].item() for c in configs}

fig, fig_no_glob, fig_no_loc = list(swath_spat_res_fig_glob.values())[:3]

x_t = trim_sw_ds.gt_loc
x = trim_sw_ds[f'{configs[0]}_glob']

psd_err = (
    (x - x_t).copy()
        .pipe(
        lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='linear'))
        .mean('x_ac')
).compute()

psd_x_t = (
    x_t.copy()
        .pipe(
        lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='linear'))
        .mean('x_ac')
).compute()

psd_score = 1 - psd_err / psd_x_t

psd_plot_data: xr.DataArray = xr.DataArray(
    einops.rearrange([psd_score.data,], 'var wl -> var wl'),
    name='PSD score',
    dims=('var', 'wl'),
    coords={
        'wl': ('wl', 1 / psd_score.freq_x_al, {'long_name': 'Wavelength', 'units': 'km'}),
        'var': ('var', ['model',], {}),
    },
)

idx = (
        (psd_plot_data.sel(var='model').rolling(wl=3, center=True, min_periods=1).mean() < 0.5).cumsum() <=1
)

spatial_resolution_model = (
    xr.DataArray(
        psd_plot_data.sel(var='model').isel(wl=idx).wl,
        dims=['psd'],
        coords={'psd': psd_plot_data.sel(var='model').isel(wl=idx).data}
    ).interp(psd=0.5)
)

