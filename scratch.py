import importlib
import seaborn as sns
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import main

xp_configs = [
    ('current', 'q.xp_five.alpha_10_ssh'),
    ('xp3_4', 'q.xp_four.alpha_eq'),
    ('xp3_4', 'q.xp_four.alpha_grad'),
    ('xp3_4', 'q.xp_four.no_alpha_twenty'),
    ('xp3_4', 'q.xp_four.swot_oi'),
    # ('xp3_4', 'q.xp_three.err_high'),
    ('xp3_4', 'q.xp_three.err_low'),
    # ('xp3_4', 'q.xp_three.swot_high'),
    ('xp3_4', 'q.xp_three.swot_low'),
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

def get_tested_models(xp_configs):
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

    return mods


mods = get_tested_models(xp_configs)
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
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, item):
        print(item)
        if item.endswith('_g'):
            return (self.ds['gt_g'].min().item(), self.ds['gt_g'].max().item())
        else:
            return (self.ds['gt'].min().item(), self.ds['gt'].max().item())
        
metrics_das = {}

for i, (cfg, mod) in enumerate(mods.items()):
    if i==0:
        metrics_das['gt'] = mod.test_xr_ds.gt
        metrics_das['gt_g'] = sobel_grid(mod.test_xr_ds.gt)

        metrics_das['oi'] = mod.test_xr_ds.oi
        metrics_das['oi_g'] = sobel_grid(mod.test_xr_ds.oi)

    metrics_das[cfg] = mod.test_xr_ds.pred
    metrics_das[f'{cfg}_g'] = sobel_grid(mod.test_xr_ds.pred)

metrics_ds = xr.Dataset(metrics_das)
to_plot_ds = metrics_ds.isel(time=t_idx)

clims = Clim(to_plot_ds)
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
mse_grad = np.mean((metrics_ds[[f'{c}_g' for c in configs]] - metrics_ds.gt_g)**2)
spat_res = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi)[0].item() for c in configs }
spat_res_grad = {c:  get_psd_score(metrics_ds.gt_g, metrics_ds[f'{c}_g'], metrics_ds.oi_g)[0].item()for c in configs }


# %% Plots

grid_metrics_df = (
    pd.concat([
        pd.DataFrame(
            { k: [v['data']] for k, v in ds.to_dict()['data_vars'].items()}
        ).T.assign(metric=metr)
        for metr, ds in [
            ('mse', mse),
            ('mse_grad', mse_grad),
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
        .assign(noise = lambda df: df.xp_long.map(lambda s: 'None' if 'swot' in s else 'roll' if 'roll' in s else 'all'))
        .assign(loss = lambda df: df.xp_long.map(lambda s: 'no glob' if 'no_glob' in s else 'no loc' if 'no_loc' in s else 'all'))
        .assign(xp_short = lambda df: df.xp_long.map(lambda s: s.split('.')[-1].strip('_g')))
)

filtered_grid_metric_df = (
        grid_metrics_df
        # .loc[lambda df: df.loss == 'all']
        # .loc[lambda df: df.noise == 'all']
        .loc[lambda df: df.xp_short.map(lambda s: s in [
             # 'alpha_10_ssh',
            # 'alpha_eq',
            # 'alpha_grad',
            # 'no_alpha_twenty',
            # 'swot_oi',
            'err_low',
            'swot_low',
             'err_no_loc',
             'swot_no_loc',
             ]
            )]
)

# # Visualize XP
# sns.catplot(
#     col='metric',
#     # row='noise',
#     x='value',
#     data=filtered_grid_metric_df,
#     y='xp_short',
#     hue='noise',
#     # row='noise',
#     kind='bar',
#     sharex='none',
# )


# Visualize improvements on grid of second loss term
sns.catplot(
    col='metric',
    y='value',
    data=filtered_grid_metric_df,
    x='noise',
    hue='loss',
    kind='bar',
    sharey='col',
)

# %% swath scores

mod_swath_das_loc = {}
mod_swath_das_glob = {}
mod_swath_das_obs = {}

for i, (cfg, mod) in enumerate(mods.items()):
    if i == 0:

        mod_swath_das_loc[f'gt'] = mod.test_xr_ds.obs_gt
        mod_swath_das_glob[f'gt'] = mod.test_xr_ds.gt
        mod_swath_das_obs[f'gt'] = mod.test_xr_ds.obs_gt

    mod_swath_das_loc[f'{cfg}'] = mod.test_xr_ds.pred
    mod_swath_das_glob[f'{cfg}'] = mod.test_xr_ds.obs_pred if mod.hparams.loss_loc else mod.test_xr_ds.pred
    mod_swath_das_obs[f'{cfg}'] = mod.test_xr_ds.obs_inp

mod_swath_ds_loc = xr.Dataset(mod_swath_das_loc)
mod_swath_ds_glob = xr.Dataset(mod_swath_das_glob)
mod_swath_ds_obs = xr.Dataset(mod_swath_das_obs)
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
chunks = swot_nadir_w_ch.groupby('contiguous_chunk').count().ssh_model.pipe(lambda da: da.isel(contiguous_chunk=da>=400))
swath_ds_chunks = []
for chunk in chunks:
    chunk_nb = chunk.contiguous_chunk.item()
    print(chunk.contiguous_chunk.item())

    swot_nadir_chunk = (
        swot_nadir_w_ch.pipe(
            lambda ds: ds.isel(time = ds.contiguous_chunk==chunk_nb)
        )
    )

    swot_chunk =  swot_ds.sel(time=swot_nadir_chunk.time)
    # swot_chunk =  swot_ds.isel(time=slice(1000, 3000))
    fmted_chunk = reindex(swot_chunk, ['x_al', 'x_ac'])
    chunk_date = pd.to_datetime(swot_chunk.time)[0].date()
    swath_ds_chunks.append(
        mod_swath_ds_loc.sel(time=str(chunk_date)).drop('time').interp(lat=fmted_chunk.lat, lon=fmted_chunk.lon-360,).assign_coords(chunk_nb=chunk_nb)
    )
swath_ds = xr.concat(swath_ds_chunks, dim='x_al')

import xrft
import einops
import matplotlib.pyplot as plt

# %% Plots swath ssh grad loc glob

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 0), da)
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 1), da)
    return np.hypot(dx_ac, dx_al)

swath_grad_ds = sobel(swath_ds)

def get_swath_fig(to_plot_ds, clim=None):
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
            **(dict(clim=clim) if clim is not None else dict())
        )
        for var in to_plot_ds
    ]).cols(2)

    return hv.render(hv_layout, backend='matplotlib',)

ssh_fig = get_swath_fig(swath_ds.pipe(lambda ds: ds.isel(x_al=ds.chunk_nb==2)))
grad_fig = get_swath_fig(swath_grad_ds)


def get_swath_psd_score(x_t, x, with_fig=False):
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

    model_score = psd_score(x)

    model_score = model_score.where(model_score > 0, drop=True).compute()

    psd_plot_data: xr.DataArray = xr.DataArray(
        model_score.data,
        name='PSD score',
        dims=('wl'),
        coords={
            'wl': ('wl', 1 / model_score.freq_x_al, {'long_name': 'Wavelength', 'units': 'km'}),
        },
    )
    
    idx = (
            (psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean() > 0.)
            & (psd_plot_data.wl > 10)
    )


    spatial_resolution_model = (
        xr.DataArray(
            # psd_plot_data.sel(var='model').wl,
            psd_plot_data.isel(wl=idx).wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.isel(wl=idx).data}
            # coords={'psd': psd_plot_data.sel(var='model').data}
        ).interp(psd=0.5)
    )

    if not with_fig:
        return spatial_resolution_model

    fig, ax = plt.subplots()
    # psd_plot_data.plot.line(x='wl', ax=ax)
    psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean().plot.line('+' ,x='wl', ax=ax)

    # Plot vertical line there
    for i, (sr, var) in enumerate([(spatial_resolution_model, 'model')]):
        plt.axvline(sr, ymin=0, color='0.5', ls=':')
        plt.annotate(f"resolution {var}: {float(sr):.2f} km", (sr * 1.1, 0.1 * i))
        plt.axhline(0.5, xmin=0, color='k', ls='--')
        plt.ylim([0, 1])

    plt.close()
    return fig, spatial_resolution_model

trim_sw_ds = (
    swath_ds
    .sel(x_ac=list(range(-50, -28, 2)) + list(range(30, 52, 2)))
    .pipe(lambda ds: ds.isel(x_al= (~np.isnan(ds.gt)).all('x_ac')))
    .drop(['lat', 'lon'])
)

trim_grad_sw_ds = sobel(trim_sw_ds)

# TODO compute all spatial resolutions
def get_spat_reses(trim_ds):
    spat_reses = []
    chunks = trim_ds.groupby('chunk_nb')
    for chunk, g in chunks:
        print(chunk)
        for c in configs:
            spat_reses.append(
                {
                    'xp_long': c,
                    'spat_res': get_swath_psd_score(g.gt.drop('chunk_nb'), g[c].drop('chunk_nb')).item(),
                    'chunk_nb':g.chunk_nb.isel(x_al=0).item()
                }
            )
    spat_res_df = pd.DataFrame(spat_reses)
    return spat_res_df
spat_res_df = get_spat_reses(trim_sw_ds) 
spat_res_grad_df = get_spat_reses(trim_grad_sw_ds) 

sns.violinplot(x='spat_res', y='xp_long', data=spat_res_df)
with pd.option_context('mode.use_inf_as_na', True):
    out_df = spat_res_grad_df.pivot(columns='xp_long', index='chunk_nb', values='spat_res').dropna(how='any').stack().reset_index().rename({0: 'spat_res'}, axis=1)
sns.violinplot(x='spat_res', y='xp_long', data=out_df)


# %% Metrics 
ssh_metrics = ((swath_ds[[f'{cfg}' for cfg in configs]] - swath_ds.gt).to_dataframe()**2).mean()[configs]
grad_metrics = ((swath_grad_ds[[f'{cfg}' for cfg in configs]] - swath_grad_ds.gt).to_dataframe()**2).mean()[configs]

ssh_metrics.plot(kind='bar')
grad_metrics.plot(kind='bar')
with pd.option_context('mode.use_inf_as_na', True):
    spat_res_grad_df.pivot(columns='xp_long', index='chunk_nb', values='spat_res').dropna(how='any').mean().plot(kind='bar')

with pd.option_context('mode.use_inf_as_na', True):
    spat_res_df.pivot(columns='xp_long', index='chunk_nb', values='spat_res').dropna(how='any').mean().plot(kind='bar')

metrics_df = (
    pd.concat([
        pd.DataFrame(
            { k: [v['data']] for k, v in ds.to_dict()['data_vars'].items()}
        ).T.assign(metric=metric, src=src, tgt=tgt)
        for tgt, src, metric, ds in [
            ('loc', 'loc', 'mse', ssh_metrics_loc_loc),
            ('loc', 'loc', 'g_mse', grad_metrics_loc_loc),
        ]] + [
            pd.DataFrame(
                {k : [v] for k, v in d.items()}
                ).T.assign(metric=metric, src=src, tgt=tgt)
        for tgt, src, metric, d in [
            ('loc', 'loc', 'res_spat', swath_spat_res_loc_loc),
            ('loc', 'loc', 'res_spat_grad', swath_grad_spat_res_loc_loc),
        ] 
    ]).reset_index()
        .rename({0: 'value', 'index': 'xp_long'}, axis=1)
        .assign(noise =  lambda df: df.xp_long.map(lambda s: 'None' if 'swot' in s else 'roll' if 'roll' in s else 'all' if 'err' in s else 'obs'))
        .assign(loss = lambda df: df.xp_long.map(lambda s: 'no glob' if 'no_glob' in s else 'no loc' if 'no_loc' in s else 'all'))
        .assign(xp_short = lambda df: df.xp_long.map(lambda s: s.split('.')[-1]))
)
# grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['gt_glob']].reindex({'x_ac': np.arange(-60, 62, 2)}))



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
    .loc[lambda df:  df.tgt == 'loc' ]
    .loc[lambda df:  df.src == 'loc' ]
    .loc[lambda df:  df.noise != 'obs' ]
    # .loc[lambda df:  df.loss != 'no glob' ]
)

print(glob_glob_df.to_markdown())

# Visualize improvements on grid of second loss term
sns.catplot(
    col='metric',
    y='value',
    data=glob_glob_df,
    x='noise',
    hue='loss',
    kind='bar',
    sharey='col',
)



sns.catplot(
    col='metric',
    y='xp_short',
    data=glob_glob_df,
    x='value',
    hue='loss',
    kind='bar',
    sharex='none',
)


grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long ) + ['gt_loc']].reindex({'x_ac': np.arange(-60, 62, 2)}))
grad_fig


grad_err_fig = get_swath_fig((swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['xp.two.err_no_loc_obs']] - swath_grad_ds['gt_loc']).reindex({'x_ac': np.arange(-60, 62, 2)}))
grad_err_fig


ssh_err_fig = get_swath_fig((swath_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['xp.two.err_no_loc_obs']] - swath_ds['gt_loc']).reindex({'x_ac': np.arange(-60, 62, 2)}))
ssh_err_fig

swath_grad_ds
