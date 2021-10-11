import xrft
import einops
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import main
import new_dataloading

xp_configs = [
    ('icassp', 'q.xp_icassp.direct_phi'),
    ('icassp', 'q.xp_icassp.direct_vit'),
    ('icassp', 'q.xp_icassp.fourdvarnet_calmap'),
    ('icassp', 'q.xp_icassp.fourdvarnet_calmapgrad'),
    ('icassp', 'q.xp_icassp.fourdvarnet_map'),
]
default_xp = 'q.xp_icassp.fourdvarnet_calmap'

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
            )
        runner = main.FourDVarNetRunner(config=config_pkg)
        mod = runner._get_model(ckpt_path=ckpt_path)
        dataloaders = runner.dataloaders
        trainer = pl.Trainer(gpus=1)
        trainer.test(mod, test_dataloaders=dataloaders['test'])
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
    dlat = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lat')))) / 5
    dlon = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lon')))) / 5
    return np.hypot(dlat, dlon)


import holoviews as hv
from holoviews.plotting import mpl
metrics_das = {}

for i, (cfg, mod) in enumerate(mods.items()):

    metrics_das[cfg] = mod.test_xr_ds.pred
    metrics_das[f'{cfg}_g'] = sobel_grid(mod.test_xr_ds.pred)

metrics_das['gt'] = mods[ default_xp].test_xr_ds.gt
metrics_das['gt_g'] = sobel_grid(mods[ default_xp].test_xr_ds.gt)

metrics_das['oi'] = mods[default_xp].test_xr_ds.oi
metrics_das['oi_g'] = sobel_grid(mods[ default_xp].test_xr_ds.oi)

# metrics_das['oi_swot'] = mods['q.xp_four.swot_oi'].test_xr_ds.oi
# metrics_das['oi_swot_g'] = sobel_grid(mods['q.xp_four.swot_oi'].test_xr_ds.oi)

metrics_ds = xr.Dataset(metrics_das)
to_plot_ds = metrics_ds.isel(time=t_idx)

obs_das = {}
for i, (cfg, mod) in enumerate(mods.items()):
    obs_das[cfg] = mod.test_xr_ds.obs_pred
    obs_das[f'{cfg}_g'] = sobel_grid(mod.test_xr_ds.obs_pred)

obs_das['gt'] = mods[default_xp].test_xr_ds.obs_gt
obs_das['gt_g'] = sobel_grid(mods[default_xp].test_xr_ds.obs_gt)

obs_das['obs'] = mods[default_xp].test_xr_ds.obs_inp
obs_das['obs_g'] = sobel_grid(mods[default_xp].test_xr_ds.obs_inp)


class Clim:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, item):
        print(item)
        if item == 'NOISE':
            lim = max(np.abs(self.ds[item].min().item()), np.abs(self.ds[item].max().item()))
            return (-lim, lim)
        if item.endswith('_g'):
            return (self.ds['gt_g'].min().item(), self.ds['gt_g'].max().item())
        else:
            return (self.ds['gt'].min().item(), self.ds['gt'].max().item())
        
def get_grid_fig(to_plot_ds):
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

    return hv.render(hv_layout, backend='matplotlib')

# cur_ds = to_plot_ds
# fig_grid_preview = get_grid_fig(cur_ds)
# cur_ds = to_plot_ds
# fig_grad = get_grid_fig(cur_ds)

# cur_ds = xr.Dataset(obs_das).assign(noise=lambda ds: ds.obs -ds.gt).isel(time=t_idx)
# fig_obs_preview = get_grid_fig(cur_ds)
# cur_ds = xr.Dataset(obs_das).isel(time=t_idx)
# fig_obs_g_preview = get_grid_fig(cur_ds)
# %% global scores

from metrics import get_psd_score
mse = np.sqrt(np.mean((metrics_ds[configs + ['oi']] - metrics_ds.gt)**2))
mse_grad = np.sqrt(np.mean((metrics_ds[[f'{c}_g' for c in configs] + ['oi_g']] - metrics_ds.gt_g)**2))
spat_res = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi)[0].item() for c in configs + ['oi']}
# spat_res_figs = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi, fig=True)[0] for c in configs }
spat_res_grad = {c:  get_psd_score(metrics_ds.gt_g, metrics_ds[f'{c}_g'], metrics_ds.oi_g)[0].item()for c in configs + ['oi'] }
# spat_res_grad_figs = {c:  get_psd_score(metrics_ds.gt_g, metrics_ds[f'{c}_g'], metrics_ds.oi_g, with_fig=True)[0]for c in configs }

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
print(grid_metrics_df.to_markdown())
print(grid_metrics_df.to_latex())
results = (
        grid_metrics_df[['xp_short', 'value', 'metric']]
        .pivot_table(index='xp_short', columns='metric', values='value')
        
)
print(grid_metrics_df[['xp_short', 'value', 'metric']].pivot_table(index='xp_short', columns='metric', values='value').to_markdown())
print(results[['mse', 'mse_grad']].to_latex())
print(results[['res_spat', 'res_spat_grad']].to_latex())


# %% Visualize psd
import matplotlib.pyplot as plt
list(enumerate(configs))
# %% swath scores

mod_swath_das_loc = {}
mod_swath_das_glob = {}
mod_swath_das_obs = {}

mask = lambda da: (mods[default_xp].test_xr_ds.obs_gt.dims, np.where(~np.isnan(mods[default_xp].test_xr_ds.obs_gt), da, np.nan))
mod_swath_das_loc[f'gt'] = mods[default_xp].test_xr_ds.obs_gt
mod_swath_das_glob[f'gt'] = mods[default_xp].test_xr_ds.gt
mod_swath_das_obs[f'gt'] = mods[default_xp].test_xr_ds.obs_gt
for i, (cfg, mod) in enumerate(mods.items()):

    mod_swath_das_loc[f'{cfg}'] = mask(mod.test_xr_ds.pred)
    mod_swath_das_glob[f'{cfg}'] = mask(mod.test_xr_ds.obs_pred if mod.hparams.loss_loc else mod.test_xr_ds.pred)
    mod_swath_das_obs[f'{cfg}'] = mask(mod.test_xr_ds.obs_inp)

mod_swath_ds_loc = xr.Dataset(mod_swath_das_loc)
mod_swath_ds_glob = xr.Dataset(mod_swath_das_glob)
mod_swath_ds_obs = xr.Dataset(mod_swath_das_obs)
# %% Fetch sensor data



from utils import coords_to_dim, reindex
from get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice
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
        + ds.karin_err
        + ds.bd_err)

swot_ds.err.T.isel(time=slice(None, 1000)).reset_index('time').plot(figsize=(10, 2))
swot_nadir = raw_item['swot_nadir']
swot_nadir_w_ch = swot_nadir.assign(
    contiguous_chunk=lambda ds: (ds.time.diff('time') / np.timedelta64(1, 's') > 1).cumsum()
)

oi_ds = reindex(clean_oi(raw_item['oi']), ('time', 'lon', 'lat'))
natl_ds = reindex(raw_item['natl'], ('time', 'lon', 'lat'))

slope = lambda da: (da.diff('nC') / da.x_ac.diff('nC')).mean('nC')
chunks = swot_nadir_w_ch.groupby('contiguous_chunk').count().ssh_model.pipe(lambda da: da.isel(contiguous_chunk=da>=400))
swath_ds_chunks = []
swath_obs_ds_chunks = []
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
    swath_obs_ds_chunks.append(
        mod_swath_ds_obs.sel(time=str(chunk_date)).drop('time').interp(lat=fmted_chunk.lat, lon=fmted_chunk.lon-360,).assign_coords(chunk_nb=chunk_nb)
    )

swath_ds = xr.concat(swath_ds_chunks, dim='x_al')
swath_obs_ds = xr.concat(swath_obs_ds_chunks, dim='x_al')
print(swath_obs_ds.pipe(lambda ds: ((ds.gt - ds)**2).to_dataframe().mean()).to_markdown())
print(swath_ds.pipe(lambda ds: ((ds.gt - ds)**2).to_dataframe().mean()).to_markdown())

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 0), da) /2
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 1), da) /2
    return np.hypot(dx_ac, dx_al)

swath_grad_ds = sobel(swath_ds)
# cur_grad_ds = sobel(cur_ds)

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



def get_swath_psd_score(x_t, x, with_fig=False, with_data=False):
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
        if with_data:
            return spatial_resolution_model, psd_plot_data
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
    .sel(x_ac=list(range(-46, -24, 2)) + list(range(26, 48, 2)))
    .pipe(lambda ds: ds.isel(x_al= (~np.isnan(ds.gt)).all('x_ac')))
    .drop(['lat', 'lon'])
)

trim_grad_sw_ds = sobel(trim_sw_ds)

# ssh_fig = get_swath_fig(trim_sw_ds.reindex(x_ac=np.arange(-50, 52, 2)).pipe(lambda ds: ds.isel(x_al=ds.chunk_nb==2)))
# grad_fig = get_swath_fig(trim_grad_sw_ds.reindex(x_ac=np.arange(-50, 52, 2)).pipe(lambda ds: ds.isel(x_al= ds.chunk_nb==2)))

def get_spat_reses(trim_ds):
    spat_reses = []
    chunks = trim_ds.groupby('chunk_nb')
    for chunk, g in chunks:
        print(chunk)
        # for c in configs + ['clem_cal_obs']:
        for c in trim_ds :
            if c == 'gt':
                continue
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

mean_sr = (spat_res_df.loc[lambda df: ~np.isnan(df.spat_res) & ~ np.isinf(df.spat_res)].groupby('xp_long').spat_res.mean())
std_sr = spat_res_df.loc[lambda df: ~np.isnan(df.spat_res) & ~ np.isinf(df.spat_res)].groupby('xp_long').spat_res.std()
print(pd.concat([mean_sr.to_frame(), std_sr.to_frame()], axis=1).to_markdown())

mse = (swath_ds.pipe(lambda ds: (ds - ds.gt)**2).to_dataframe().mean())
mse_grad = (sobel(swath_ds).pipe(lambda ds: (ds - ds.gt)**2).to_dataframe().mean())

print(pd.concat([mse.to_frame(), mse_grad.to_frame()], axis=1).to_markdown())

