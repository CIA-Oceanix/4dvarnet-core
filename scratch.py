import importlib
import seaborn as sns
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import main

xp_configs = [
    # ('current', 'q.xp_three.err_high'),
    ('current', 'q.xp_three.err_low'),
    # ('current', 'q.xp_three.swot_high'),
    ('current', 'q.xp_three.swot_low'),
    # ('xp2_with_proj_coquille', 'q.xp_two.err'),
    # ('xp2_with_proj_coquille', 'q.xp_two.err_no_glob'),
    ('xp2_with_proj_coquille', 'q.xp_two.err_no_loc'),
    # ('xp2_with_proj_coquille', 'q.xp_two.roll'),
    # ('xp2_with_proj_coquille', 'q.xp_two.roll_no_glob'),
    # ('xp2_with_proj_coquille', 'q.xp_two.roll_no_loc'),
    # ('xp2_with_proj_coquille', 'q.xp_two.swot'),
    # ('xp2_with_proj_coquille', 'q.xp_two.swot_no_glob'),
    ('xp2_with_proj_coquille', 'q.xp_two.swot_no_loc'),
    # ('xp1_all_errs_no_karin', 'q.xp_one.high_zeros_glob'),
    # ('xp1_all_errs_no_karin', 'q.xp_one.low_obs_glob'),
    # ('xp1_all_errs_no_karin', 'q.xp_one.low_zeros_glob'),
    # ('xp1_all_errs_no_karin', 'q.xp_one.low_zeros_loc'),

]

xps, configs = zip(*xp_configs)
xps, configs = list(xps), list(configs)

def get_most_recent_ckpt(config_pkg, xp='no_roll'):
    ckpt_fold = Path(f'dashboard/{xp}/train/{config_pkg}/checkpoints')
    checkpoints = ckpt_fold.glob('*')
    return max(checkpoints, key=lambda x: x.stat().st_ctime)


# %% Global maps ssh + grad 
# %% Swath maps ssh + grad
# %% Global scores mse, grad mse, spat res
# %% Swath scores mse, grad mse, spat mse
# from config_q.xp_three.err_low import params
# params['files_cfg']
# import importlib
# config = importlib.import_module("config_q.xp_three.err_low")
# config = __import__("config_q.xp_three.err_low")
# print(config.params['files_cfg'])
# %% components decomposition: gde échelle, anom glob, anom loc
# ckpt_path= "first_results_dash/train/nad_roll/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.07.ckpt"
mods = {}
for xp, config_pkg in xp_configs:
    ckpt_path = get_most_recent_ckpt(
            config_pkg=config_pkg,
            xp=xp,
            # xp='no_roll',
            # xp='roll',
        )
    runner = main.FourDVarNetRunner(config=config_pkg)
    mod = runner._get_model(ckpt_path=ckpt_path)
    print()
    print(" #### ", config_pkg, " #### ")
    print(runner.cfg.files_cfg)
    print()
    # %% Generate maps

    trainer = pl.Trainer(gpus=1)
    trainer.test(mod, test_dataloaders=runner.dataloaders['test'])
    mods[config_pkg] = mod.cpu()

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
    for var in to_plot_ds
]).cols(2)

fig_swot_preview = hv.render(hv_layout, backend='matplotlib')

# %% global scores
from metrics import get_psd_score

mse = np.mean((metrics_ds[configs] - metrics_ds.gt)**2)
mse_grad = np.mean((metrics_ds[[f'{c}_grad' for c in configs]] - metrics_ds.gt_grad)**2)
spat_res = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi)[0].item() for c in configs }
spat_res_grad = {c:  get_psd_score(metrics_ds.gt_grad, metrics_ds[f'{c}_grad'], metrics_ds.oi_grad)[0].item()for c in configs }


# %% Plots


grid_metrics_df = (
    pd.concat([
        pd.DataFrame(
            { k: [v['data']] for k, v in ds.to_dict()['data_vars'].items()}
        ).T.assign(metric=metr)
        for metr, ds in [
            ('mse', mse),
            ('mse_grad', mse_grad),
            # ('res_spat', spat_res),
            # ('res_spat_grad', spat_res_grad),
        ]] + [
            pd.DataFrame(
                {k : [v] for k, v in d.items()}
                ).T.assign(metric=metr)
        for metr, d in [
            ('res_spat', spat_res),
            ('res_spat_grad', spat_res_grad),
        ] 
    ]).reset_index()
        .rename({0: 'value', 'index': 'xp_long'}, axis=1)
        .assign(noise = lambda df: df.xp_long.map(lambda s: s.split('_')[1]))
        .assign(loss = lambda df: df.xp_long.map(lambda s: 'no glob' if 'no_glob' in s else 'no loc' if 'no_loc' in s else 'all'))
)


# %% swath scores

mod_swath_das = {}

for cfg, mod in mods.items():
    mod_swath_das[f'{cfg}_glob'] = mod.test_xr_ds.pred
    mod_swath_das[f'{cfg}_loc'] = mod.test_xr_ds.obs_pred
    mod_swath_das[f'{cfg}_obs'] = mod.test_xr_ds.obs_inp

mod_swath_das['gt_glob'] = mod.test_xr_ds.gt
mod_swath_das['gt_loc'] = mod.test_xr_ds.obs_gt
mod_swath_das['oi'] = mod.test_xr_ds.oi

mod_swath_ds = xr.Dataset(mod_swath_das)

# %% Fetch sensor data

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
swath_ds = (
    mod_swath_ds.sel(time=str(chunk_date)).drop('time').interp(lat=fmted_chunk.lat, lon=fmted_chunk.lon-360,)
    .assign(gt_loc2=fmted_chunk.ssh_model)
)

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
            aspect=3,
            clim=(-0.05, 0.05)
        )
        for var in to_plot_ds
    ]).cols(2)

    return hv.render(hv_layout, backend='matplotlib',)

ssh_fig = get_swath_fig(swath_ds)
grad_fig = get_swath_fig(swath_grad_ds)


def get_swath_psd_score(x_t, x, ref, with_fig=False):
    def psd_score(da: xr.DataArray) -> xr.DataArray:
        err = x_t - da
        psd_x_t = (
            x_t.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='linear', scaling='density'))
                .mean('x_ac')
        ).compute()

        psd_err = (
            err.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='linear', scaling='density'))
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
            (psd_plot_data.sel(var='model').rolling(wl=3, center=True, min_periods=1).mean() > 0.)
            & (psd_plot_data.sel(var='model').wl > 10)
    )

    if sum(idx).item() <= 1:
        print(psd_plot_data.sel(var='model').rolling(wl=3, center=True, min_periods=1).mean() < 0.5)
        print((psd_plot_data.sel(var='model').rolling(wl=3, center=True, min_periods=1).mean() < 0.5).cumsum())
        print(idx)
        print(psd_plot_data)
        psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean().plot.line('+' ,x='wl',)

    spatial_resolution_model = (
        xr.DataArray(
            # psd_plot_data.sel(var='model').wl,
            psd_plot_data.sel(var='model').isel(wl=idx).wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='model').isel(wl=idx).data}
            # coords={'psd': psd_plot_data.sel(var='model').data}
        ).interp(psd=0.5)
    )

    idx = (
            (psd_plot_data.sel(var='OI').rolling(wl=3, center=True, min_periods=1).mean() > 0.)
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
    # psd_plot_data.plot.line(x='wl', ax=ax)
    psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean().plot.line('+' ,x='wl', ax=ax)

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

for c in configs:
    print(c)
    get_swath_psd_score(trim_grad_sw_ds.gt_glob, trim_grad_sw_ds[f'{c}_glob'], trim_grad_sw_ds.oi)

swath_spat_res_glob_loc = {c:  get_swath_psd_score(trim_sw_ds.gt_loc, trim_sw_ds[f'{c}_glob'], trim_sw_ds.oi)[0].item() for c in configs}
swath_spat_res_loc_loc = {c:  get_swath_psd_score(trim_sw_ds.gt_loc, trim_sw_ds[f'{c}_loc'], trim_sw_ds.oi)[0].item() for c in configs}
swath_spat_res_obs_loc = {c:  get_swath_psd_score(trim_sw_ds.gt_loc, trim_sw_ds[f'{c}_obs'], trim_sw_ds.oi)[0].item() for c in configs}

swath_grad_spat_res_glob_loc = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_loc, trim_grad_sw_ds[f'{c}_glob'], trim_grad_sw_ds.oi)[0].item() for c in configs}
swath_grad_spat_res_loc_loc = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_loc, trim_grad_sw_ds[f'{c}_loc'], trim_grad_sw_ds.oi)[0].item() for c in configs}
swath_grad_spat_res_obs_loc = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_loc, trim_grad_sw_ds[f'{c}_obs'], trim_grad_sw_ds.oi)[0].item() for c in configs}

swath_spat_res_glob_glob = {c:  get_swath_psd_score(trim_sw_ds.gt_glob, trim_sw_ds[f'{c}_glob'], trim_sw_ds.oi)[0].item() for c in configs}
swath_spat_res_loc_glob = {c:  get_swath_psd_score(trim_sw_ds.gt_glob, trim_sw_ds[f'{c}_loc'], trim_sw_ds.oi)[0].item() for c in configs}
swath_spat_res_obs_glob = {c:  get_swath_psd_score(trim_sw_ds.gt_glob, trim_sw_ds[f'{c}_obs'], trim_sw_ds.oi)[0].item() for c in configs}

swath_grad_spat_res_glob_glob = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_glob, trim_grad_sw_ds[f'{c}_glob'], trim_grad_sw_ds.oi)[0].item() for c in configs}
swath_grad_spat_res_loc_glob = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_glob, trim_grad_sw_ds[f'{c}_loc'], trim_grad_sw_ds.oi)[0].item() for c in configs}
swath_grad_spat_res_obs_glob = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_glob, trim_grad_sw_ds[f'{c}_obs'], trim_grad_sw_ds.oi)[0].item() for c in configs}

# figs = {c:  get_swath_psd_score(trim_grad_sw_ds.gt_loc, trim_grad_sw_ds[f'{c}_glob'], trim_grad_sw_ds.oi, with_fig=True)[0].item() for c in configs}
# fig, fig_no_glob, fig_no_loc = list(swath_spat_res_fig_glob.values())[:3]


# %% Metrics 
ssh_metrics_glob_loc = np.mean((swath_ds[[f'{cfg}_glob' for cfg in configs]] - swath_ds.gt_loc)**2)
ssh_metrics_loc_loc = np.mean((swath_ds[[f'{cfg}_loc' for cfg in configs]] - swath_ds.gt_loc)**2)
ssh_metrics_obs_loc = np.mean((swath_ds[[f'{cfg}_obs' for cfg in configs]] - swath_ds.gt_loc)**2)

grad_metrics_glob_loc = np.mean((swath_grad_ds[[f'{cfg}_glob' for cfg in configs]] - swath_grad_ds.gt_loc)**2)
grad_metrics_loc_loc = np.mean((swath_grad_ds[[f'{cfg}_loc' for cfg in configs]] - swath_grad_ds.gt_loc)**2)
grad_metrics_obs_loc = np.mean((swath_grad_ds[[f'{cfg}_obs' for cfg in configs]] - swath_grad_ds.gt_loc)**2)


# ssh_metrics_glob_loc2 = np.mean((swath_ds[[f'{cfg}_glob' for cfg in configs]] - swath_ds.gt_loc)**2)
# ssh_metrics_loc_loc2 = np.mean((swath_ds[[f'{cfg}_loc' for cfg in configs]] - swath_ds.gt_loc)**2)
# ssh_metrics_obs_loc2 = np.mean((swath_ds[[f'{cfg}_obs' for cfg in configs]] - swath_ds.gt_loc)**2)

# grad_metrics_glob_loc2 = np.mean((swath_grad_ds[[f'{cfg}_glob' for cfg in configs]] - swath_grad_ds.gt_loc2)**2)
# grad_metrics_loc_loc2 = np.mean((swath_grad_ds[[f'{cfg}_loc' for cfg in configs]] - swath_grad_ds.gt_loc2)**2)
# grad_metrics_obs_loc2 = np.mean((swath_grad_ds[[f'{cfg}_obs' for cfg in configs]] - swath_grad_ds.gt_loc2)**2)

ssh_metrics_glob_glob = np.mean((swath_ds[[f'{cfg}_glob' for cfg in configs]] - swath_ds.gt_glob)**2)
ssh_metrics_loc_glob = np.mean((swath_ds[[f'{cfg}_loc' for cfg in configs]] - swath_ds.gt_glob)**2)
ssh_metrics_obs_glob = np.mean((swath_ds[[f'{cfg}_obs' for cfg in configs]] - swath_ds.gt_glob)**2)

grad_metrics_glob_glob = np.mean((swath_grad_ds[[f'{cfg}_glob' for cfg in configs]] - swath_grad_ds.gt_glob)**2)
grad_metrics_loc_glob = np.mean((swath_grad_ds[[f'{cfg}_loc' for cfg in configs]] - swath_grad_ds.gt_glob)**2)
grad_metrics_obs_glob = np.mean((swath_grad_ds[[f'{cfg}_obs' for cfg in configs]] - swath_grad_ds.gt_glob)**2)

metrics_df = (
    pd.concat([
        pd.DataFrame(
            { k: [v['data']] for k, v in ds.to_dict()['data_vars'].items()}
        ).T.assign(metric=metric, src=src, tgt=tgt)
        for tgt, src, metric, ds in [
            ('loc', 'glob', 'mse', ssh_metrics_glob_loc),
            ('loc', 'loc', 'mse', ssh_metrics_loc_loc),
            ('loc', 'obs', 'mse', ssh_metrics_obs_loc),
            ('loc', 'glob', 'g_mse', grad_metrics_glob_loc),
            ('loc', 'loc', 'g_mse', grad_metrics_loc_loc),
            ('loc', 'obs', 'g_mse', grad_metrics_obs_loc),
            # ('loc2', 'glob', 'mse', ssh_metrics_glob_loc),
            # ('loc2', 'loc', 'mse', ssh_metrics_loc_loc2),
            # ('loc2', 'obs', 'mse', ssh_metrics_obs_loc2),
            # ('loc2', 'glob', 'g_mse', grad_metrics_glob_loc2),
            # ('loc2', 'loc', 'g_mse', grad_metrics_loc_loc2),
            # ('loc2', 'obs', 'g_mse', grad_metrics_obs_loc2),
            ('glob', 'glob', 'mse', ssh_metrics_glob_glob),
            ('glob', 'loc', 'mse', ssh_metrics_loc_glob),
            ('glob', 'obs', 'mse', ssh_metrics_obs_glob),
            ('glob', 'glob', 'g_mse', grad_metrics_glob_glob),
            ('glob', 'loc', 'g_mse', grad_metrics_loc_glob),
            ('glob', 'obs', 'g_mse', grad_metrics_obs_glob),
        ]] + [
            pd.DataFrame(
                {k : [v] for k, v in d.items()}
                ).T.assign(metric=metric, src=src, tgt=tgt)
        for tgt, src, metric, d in [
            ('loc', 'glob', 'res_spat', swath_spat_res_glob_loc),
            ('loc', 'loc', 'res_spat', swath_spat_res_loc_loc),
            ('loc', 'obs', 'res_spat', swath_spat_res_obs_loc),
            ('loc', 'glob', 'res_spat_grad', swath_grad_spat_res_glob_loc),
            ('loc', 'loc', 'res_spat_grad', swath_grad_spat_res_loc_loc),
            ('loc', 'obs', 'res_spat_grad', swath_grad_spat_res_obs_loc),
            ('glob', 'glob', 'res_spat', swath_spat_res_glob_glob),
            ('glob', 'loc', 'res_spat', swath_spat_res_loc_glob),
            ('glob', 'obs', 'res_spat', swath_spat_res_obs_glob),
            ('glob', 'glob', 'res_spat_grad', swath_grad_spat_res_glob_glob),
            ('glob', 'loc', 'res_spat_grad', swath_grad_spat_res_loc_glob),
            ('glob', 'obs', 'res_spat_grad', swath_grad_spat_res_obs_glob),
        ] 
    ]).reset_index()
        .rename({0: 'value', 'index': 'xp_long'}, axis=1)
        .assign(noise =  lambda df: df.xp_long.map(lambda s: 'None' if 'swot' in s else 'roll' if 'roll' in s else 'all' if 'err' in s else 'obs'))
        .assign(loss = lambda df: df.xp_long.map(lambda s: 'no glob' if 'no_glob' in s else 'no loc' if 'no_loc' in s else 'all'))
)

# %% Analysis 1 : noise impact on the daily estimation

sns.catplot(
    col='metric',
    y='value',
    data=grid_metrics_df.loc[lambda df: df.loss == 'no loc'],
    x='noise',
    # col='loss',
    kind='bar',
    sharey='col',
)

# on the grid
print(grid_metrics_df.loc[lambda df: df.loss == 'no loc'].to_markdown())

sns.catplot(
    col='metric',
    y='value',
    data=grid_metrics_df.loc[lambda df: df.loss == 'no loc'],
    x='noise',
    # col='loss',
    kind='bar',
    sharey='col',
)

"""
Light effect difficult to see if significative
"""
# On the swath
glob_glob_df = (
    metrics_df
    .loc[lambda df:  df.tgt == 'glob' ]
    .loc[lambda df:  df.src == 'glob' ]
    .loc[lambda df:  df.loss == 'no loc' ]
)

print(glob_glob_df.to_markdown())

sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    # col='loss',
    kind='bar',
    sharey='col',
)

"""
Obvious on lse less obvious on spatial resolution, may even have small regularization effect
"""

# ssh_fig = get_swath_fig(swath_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long) + ['gt_glob']].reindex({'x_ac': np.arange(-60, 62, 2)}))
# grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['gt_glob']].reindex({'x_ac': np.arange(-60, 62, 2)}))


# %% Analysis 2 : Adding a local instantaneous estimator

## Global impact

# on the grid
print(grid_metrics_df.loc[lambda df: df.loss != 'no glob'].to_markdown())

import seaborn as sns
sns.catplot(
    col='metric',
    y='value',
    data=grid_metrics_df.loc[lambda df: df.loss != 'no glob'],
    x='noise',
    hue='loss',
    kind='bar',
    sharey='col',
)


# On the swath 1 : evol no loc -> all
glob_glob_df = (
    metrics_df
    .loc[lambda df:  df.tgt == 'glob' ]
    .loc[lambda df:  df.src == 'glob' ]
    .loc[lambda df:  df.loss != 'no glob' ]
    .loc[lambda df:  df.xp_long.map(lambda s: 'high' not in s)]
)

print(glob_glob_df.to_markdown())


sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    hue='loss',
    kind='bar',
    sharey='col',
)
"""
the daily estimation has slightly improved when no noise
slight improvement for roll err except for grad spat res
degradation when all errs
"""

# On the swath 2 : diff src loc vs glob
glob_glob_df = (
    metrics_df
    .loc[lambda df:  df.tgt == 'glob' ]
    # .loc[lambda df:  df.src == 'glob' ]
    .loc[lambda df:  df.loss == 'all' ]
)

print(glob_glob_df.to_markdown())


sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    hue='src',
    kind='bar',
    sharey='col',
)

"""
we see that almost every where the instantaneous estimation further from the daily gt than the daily estimation (ouf)
"""

# On the swath 3 : diff src with tgt loc 
glob_glob_df = (
    metrics_df
    .loc[lambda df:  df.tgt == 'loc' ]
    # .loc[lambda df:  df.src == 'glob' ]
    .loc[lambda df:  df.loss == 'all' ]
)

print(glob_glob_df.to_markdown())


sns.catplot(
    col='metric',
    row='tgt',
    y='value',
    data=glob_glob_df,
    x='noise',
    hue='src',
    kind='bar',
    sharey='col',
)

"""
we see that almost everywhere the instantaneous estimation is closer from the instantaneous gt than the daily estimation (ouf)
better gain for roll than all errs
"""

# ssh_fig = get_swath_fig(swath_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long) + ['gt_glob']].reindex({'x_ac': np.arange(-60, 62, 2)}))
# grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['gt_glob']].reindex({'x_ac': np.arange(-60, 62, 2)}))


## Instantaneous impact

# On the swath
glob_glob_df = (
    metrics_df
    # .loc[lambda df:  df.tgt == 'loc' ]
    # .loc[lambda df:  df.src == 'glob' ]
    .loc[lambda df:  df.loss == 'all' ]
)

# print(glob_glob_df.to_markdown())


sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    row='tgt',
    hue='src',
    kind='bar',
    sharey='col',
)

ssh_fig = get_swath_fig(swath_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long) + ['gt_glob', 'gt_loc']].reindex({'x_ac': np.arange(-60, 62, 2)}))
grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['gt_glob', 'gt_loc']].reindex({'x_ac': np.arange(-60, 62, 2)}))


## Does the local learning improves the daily estimation on the swath ?


glob_glob_df = (
    metrics_df
    .loc[lambda df:  df.tgt == 'loc' ]
    .loc[lambda df:  df.src == 'glob' ]
    .loc[lambda df:  df.loss != 'no glob' ]
)

print(glob_glob_df.to_markdown())


sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    # row='tgt',
    hue='loss',
    kind='bar',
    sharey='col',
)

ssh_fig = get_swath_fig(swath_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long) + ['gt_glob', 'gt_loc']].reindex({'x_ac': np.arange(-60, 62, 2)}))
grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['gt_glob', 'gt_loc']].reindex({'x_ac': np.arange(-60, 62, 2)}))


# loss terms
best_estim_df = (
    metrics_df
    .loc[lambda df: ~((df.loss == 'no glob') & (df.src == 'glob'))]
    .loc[lambda df: ~((df.loss == 'no loc') & (df.src == 'loc'))]
    .append(
        metrics_df
        .loc[lambda df: ((df.loss == 'no glob') & (df.src == 'loc'))]
        .assign(src='glob')
    )
    .append(
        metrics_df
        .loc[lambda df: ((df.loss == 'no loc') & (df.src == 'glob'))]
        .assign(src='loc')
    )
)

glob_glob_df = (
    best_estim_df
    .assign(loss=lambda df: df.apply(lambda row: row.loss if 'obs'!=row['src']  else row['src'], axis=1))
    .assign(src=lambda df: df.src.map(lambda s: 'glob' if s =='glob' else 'loc'))
    .loc[lambda df:  df.tgt == 'loc' ]
    .loc[lambda df:  df.src == 'loc' ]
    # .loc[lambda df:  df.loss != 'no glob' ]
    .loc[lambda df:  df.xp_long.map(lambda s: 'high' not in s)]
)

print(glob_glob_df.to_markdown())


sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    hue='loss',
    kind='bar',
    sharey='col',
)


grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long ) + ['gt_loc']].reindex({'x_ac': np.arange(-60, 62, 2)}))
grad_fig


grad_err_fig = get_swath_fig((swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['xp.two.err_no_loc_obs']] - swath_grad_ds['gt_loc']).reindex({'x_ac': np.arange(-60, 62, 2)}))
grad_err_fig


ssh_err_fig = get_swath_fig((swath_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['xp.two.err_no_loc_obs']] - swath_ds['gt_loc']).reindex({'x_ac': np.arange(-60, 62, 2)}))
ssh_err_fig

swath_grad_ds
