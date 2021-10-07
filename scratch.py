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
    # ('current', 'q.xp_seven.direct_phi'),
    # ('current', 'q.xp_seven.phi_err_cal'),
    ('current', 'q.xp_seven.direct_phi_err'),
    ('xp6_vit_hparms', 'q.xp_six.vit_swot_more_dropout'),
    # ('xp4_vit_first', 'q.xp_five.alpha_10_ssh'),
    # ('xp3_4', 'q.xp_four.alpha_eq'),
    ('xp3_4', 'q.xp_four.alpha_grad'),
    # ('xp3_4', 'q.xp_four.no_alpha_twenty'),
    ('xp3_4', 'q.xp_four.swot_oi'),
    # ('xp3_4', 'q.xp_three.err_high'),
    ('xp3_4', 'q.xp_three.err_low'),
    # ('xp3_4', 'q.xp_three.swot_high'),
    # ('xp3_4', 'q.xp_three.swot_low'),
    # ('xp2_with_proj_coquille', 'q.xp_two.err'),
    # ('xp2_with_proj_coquille', 'q.xp_two.err_no_glob'),
    ('xp2_with_proj_coquille', 'q.xp_two.err_no_loc'),
    # ('xp2_with_proj_coquille', 'q.xp_two.roll'),
    # ('xp2_with_proj_coquille', 'q.xp_two.roll_no_glob'),
    # ('xp2_with_proj_coquille', 'q.xp_two.roll_no_loc'),
    # ('xp2_with_proj_coquille', 'q.xp_two.swot'),
    # ('xp2_with_proj_coquille', 'q.xp_two.swot_no_glob'),
    # ('xp2_with_proj_coquille', 'q.xp_two.swot_no_loc'),
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
        # config = runner.cfg
        # if config.files_cfg.obs_mask_var == 'nad_swot_roll_phase_bd_timing_karin':
        #     config.files_cfg.obs_mask_path = '/gpfswork/rech/yrf/commun/CalData/cal_data_clem.nc'
        #     config.files_cfg.obs_mask_var = 'nad_swot_clem_err'
        # print(config.files_cfg.obs_mask_var)
        # config_mod = importlib.import_module("config_" + str(config_pkg))
        # dim_range = config_mod.dim_range
        # slice_win = config_mod.slice_win
        # strides = config_mod.strides
        # datamodule = new_dataloading.FourDVarNetDataModule(
        #     slice_win=slice_win,
        #     dim_range=dim_range,
        #     strides=strides,
        #     **config.files_cfg,
        #     **{k: tuple([slice(*dt) for dt in dts]) for k, dts in config.splits.items()},
        # )


        # datamodule.setup()
        # dataloaders = {
        #     'test': datamodule.test_dataloader(),
        # }
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

metrics_das['gt'] = mods[ 'q.xp_three.err_low'].test_xr_ds.gt
metrics_das['gt_g'] = sobel_grid(mods[ 'q.xp_three.err_low'].test_xr_ds.gt)

metrics_das['oi'] = mods['q.xp_three.err_low'].test_xr_ds.oi
metrics_das['oi_g'] = sobel_grid(mods[ 'q.xp_three.err_low'].test_xr_ds.oi)

metrics_das['oi_swot'] = mods['q.xp_four.swot_oi'].test_xr_ds.oi
metrics_das['oi_swot_g'] = sobel_grid(mods['q.xp_four.swot_oi'].test_xr_ds.oi)

metrics_ds = xr.Dataset(metrics_das)
to_plot_ds = metrics_ds.isel(time=t_idx)

obs_das = {}
for i, (cfg, mod) in enumerate(mods.items()):
    obs_das[cfg] = mod.test_xr_ds.obs_pred
    obs_das[f'{cfg}_g'] = sobel_grid(mod.test_xr_ds.obs_pred)

obs_das['gt'] = mods['q.xp_three.err_low'].test_xr_ds.obs_gt
obs_das['gt_g'] = sobel_grid(mods['q.xp_three.err_low'].test_xr_ds.obs_gt)

obs_das['obs'] = mods['q.xp_three.err_low'].test_xr_ds.obs_inp
obs_das['obs_g'] = sobel_grid(mods['q.xp_three.err_low'].test_xr_ds.obs_inp)


class Clim:
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, item):
        print(item)
        if item == 'NOISE':
            lim = max(np.abs(self.ds[item].min().item()), np.abs(self.ds[item].max().item()))
            return (-lim, lim)
        var = 'SWOT SSH'
        # var = 'NATL60'
        lim = max(np.abs(self.ds['SWOT SSH'].min().item()), np.abs(self.ds['SWOT SSH'].max().item()))
        return (-lim, lim)
        # return (self.ds['NATL60'].min().item(), self.ds['NATL60'].max().item())
        return max(self.ds['SWOT SSH'].min().abs().item(), self.ds['SWOT SSH'].max().abs().item())
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

rename_xps = {
        # 'q.xp_seven.phi_err_cal': r'Direct Inversion \Phi_r cal',
        'q.xp_seven.direct_phi_err': r'Direct Inversion \Phi_r',
        # 'q.xp_six.vit_swot_more_dropout': 'Direct Inversion ViT',
        # 'q.xp_four.alpha_grad':'4DVarNet-SLA-SWOT-grad',
        # 'q.xp_four.alpha_grad':'4DVarNet-SLA-SWOT-grad',
        'q.xp_three.err_low': '4DVarNet-Calmap',
        # 'q.xp_two.err_no_loc': '4DVarNet-SLA',
        'gt': 'NATL60',
        'oi_swot': 'DUACS (4nadirs + SWOT)',
}
cur_ds = to_plot_ds.rename(rename_xps)[list( rename_xps.values() )]
fig_grid_preview = get_grid_fig(cur_ds)
cur_ds = to_plot_ds.rename({f'{c}_g': v for c, v in rename_xps.items()})[list( rename_xps.values() )]
fig_grad = get_grid_fig(cur_ds)

rename_obs = {
        'gt': 'SWOT SSH',
        'noise': 'NOISE',
        'obs': 'SWOT SSH + NOISE',
        'q.xp_three.err_low': '4DVarNet-Calmap',
}
cur_ds = xr.Dataset(obs_das).assign(noise=lambda ds: ds.obs -ds.gt).rename(rename_obs).isel(time=t_idx)[list(rename_obs.values())]
fig_obs_preview = get_grid_fig(cur_ds)
cur_ds = xr.Dataset(obs_das).rename({f'{c}_g': v for c, v in rename_obs.items()}).isel(time=t_idx)[list( rename_obs.values() )]
fig_obs_g_preview = get_grid_fig(cur_ds)
# %% global scores

from metrics import get_psd_score
mse = np.sqrt(np.mean((metrics_ds[configs + ['oi' , 'oi_swot']] - metrics_ds.gt)**2))
mse_grad = np.sqrt(np.mean((metrics_ds[[f'{c}_g' for c in configs] + ['oi_g', 'oi_swot_g']] - metrics_ds.gt_g)**2))
spat_res = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi)[0].item() for c in configs + ['oi', 'oi_swot']}
spat_res_figs = {c:  get_psd_score(metrics_ds.gt, metrics_ds[c], metrics_ds.oi_swot, with_fig=True)[0] for c in configs }
spat_res_grad = {c:  get_psd_score(metrics_ds.gt_g, metrics_ds[f'{c}_g'], metrics_ds.oi_swot_g)[0].item()for c in configs + ['oi', 'oi_swot'] }
spat_res_grad_figs = {c:  get_psd_score(metrics_ds.gt_g, metrics_ds[f'{c}_g'], metrics_ds.oi_swot_g, with_fig=True)[0]for c in configs }

spat_res_figs[ 'q.xp_two.err_no_loc']
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
        .rename(
            dict(
                alpha_grad='4DVarNet-SLA-SWOT-grad',
                err_low='4DVarNet-SLA-SWOT',
                err_no_loc='4DVarNet-SLA',
                oi_swot='CAL + DUACS',
        vit_swot_more_dropout= 'Direct Inversion ViT',
            )
        )
)
print(grid_metrics_df[['xp_short', 'value', 'metric']].pivot_table(index='xp_short', columns='metric', values='value').to_markdown())
print(results[['mse', 'mse_grad']].to_latex())
print(results[['res_spat', 'res_spat_grad']].to_latex())

filtered_grid_metric_df = (
        grid_metrics_df
        # .loc[lambda df: df.loss == 'all']
        # .loc[lambda df: df.noise == 'all']
        .loc[lambda df: df.xp_short.map(lambda s: s in [
            # 'vit_swot_more_dropout',
            'alpha_10_ssh',
            'alpha_eq',
            'alpha_grad',
            # 'swot_oi',
            # 'err_high',
            'err_low',
            # 'swot_low',
            'err_no_loc',
            # 'swot_no_loc'
         ]
            )]
)

# Visualize XP
sns.catplot(
    col='metric',
    # row='noise',
    x='value',
    data=filtered_grid_metric_df,
    y='xp_short',
    hue='noise',
    # row='noise',
    kind='bar',
    sharex='none',
)


# Visualize improvements on grid of second loss term
print(
        filtered_grid_metric_df
        .loc[lambda df: df.xp_short.map(lambda s: s in [
            # 'vit_swot_more_dropout',
            # 'alpha_10_ssh',
            # 'alpha_eq',
            # 'alpha_grad',
            # 'swot_oi',
            # 'err_high',
            'err_low',
            'swot_low',
            'err_no_loc',
            'swot_no_loc'
            ]
            )]
        .loc[lambda df: df.metric == 'mse']
        .pivot_table(index='noise', columns='loss', values='value')
        # .pipe(lambda df: df / df.loc['None', 'no loc']*100 -100 )
        .pipe(lambda df: df / mse.oi.item()*100 -100 )
        .applymap(lambda n: f'{n:.2f} %')
        .to_markdown()
)

print(
        filtered_grid_metric_df
        .loc[lambda df: df.xp_short.map(lambda s: s in [
            # 'vit_swot_more_dropout',
            # 'alpha_10_ssh',
            # 'alpha_eq',
            # 'alpha_grad',
            # 'swot_oi',
            # 'err_high',
            'err_low',
            'swot_low',
            'err_no_loc',
            'swot_no_loc'
            ]
            )]
        .loc[lambda df: df.metric == 'mse_grad']
        .pivot_table(index='noise', columns='loss', values='value')
        # .pipe(lambda df: df / df.loc['None', 'no loc']*100 -100 )
        .pipe(lambda df: df / mse_grad.oi_g.item()*100 -100 )
        .applymap(lambda n: f'{n:.2f} %')
        .to_markdown()
)
# %% Visualize psd
import matplotlib.pyplot as plt
xps = filtered_grid_metric_df.loc[lambda df: df.metric=='mse'].xp_long.values
list(enumerate(configs))
# %% swath scores

mod_swath_das_loc = {}
mod_swath_das_glob = {}
mod_swath_das_obs = {}

mask = lambda da: (mods['q.xp_three.err_low'].test_xr_ds.obs_gt.dims, np.where(~np.isnan(mods['q.xp_three.err_low'].test_xr_ds.obs_gt), da, np.nan))
mod_swath_das_loc[f'gt'] = mods['q.xp_three.err_low'].test_xr_ds.obs_gt
mod_swath_das_glob[f'gt'] = mods['q.xp_three.err_low'].test_xr_ds.gt
mod_swath_das_obs[f'gt'] = mods['q.xp_three.err_low'].test_xr_ds.obs_gt
for i, (cfg, mod) in enumerate(mods.items()):

    mod_swath_das_loc[f'{cfg}'] = mask(mod.test_xr_ds.pred)
    mod_swath_das_glob[f'{cfg}'] = mask(mod.test_xr_ds.obs_pred if mod.hparams.loss_loc else mod.test_xr_ds.pred)
    mod_swath_das_obs[f'{cfg}'] = mask(mod.test_xr_ds.obs_inp)

mod_swath_ds_loc = xr.Dataset(mod_swath_das_loc)
mod_swath_ds_glob = xr.Dataset(mod_swath_das_glob)
mod_swath_ds_obs = xr.Dataset(mod_swath_das_obs)
# %% Fetch sensor data

import sys
sys.path.append('../research-quentin')


from src.commons.coords_to_dim import coords_to_dim, reindex
from src.data_processing.get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice
import src.data_processing.get_slice
importlib.reload(src.data_processing.get_slice)
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
# grid_clem_ds = xr.open_dataset( '/gpfswork/rech/yrf/commun/CalData/cal_data_clem.nc')
# grid_clem_ds.isel(time=0).nad_swot_clem_err.plot()
# grid_clem_ds['time'] = pd.to_datetime(grid_clem_ds['time']).date
# grid_clem_ds['time'] = pd.to_datetime(grid_clem_ds['time'])
# grid_clem_ds= grid_clem_ds.sel(time=pd.to_datetime(mod_swath_ds_loc.time))
# grid_clem_ds= grid_clem_ds.sel(lat=mod_swath_ds_loc.lat, method='nearest', tolerance=0.1)
# grid_clem_ds= grid_clem_ds.sel(lon=mod_swath_ds_loc.lon, method='nearest', tolerance=0.1)
# mod_swath_ds_loc =mod_swath_ds_loc.assign(clem_cal_obs= (mod_swath_ds_loc.gt.dims, grid_clem_ds.pipe(lambda ds : ds.nad_swot_clem_err - ds.clem_est).data))
# mod_swath_ds_loc.isel(time=t_idx).clem_cal_obs.plot()
# grid_clem_ds.pipe(lambda ds : ds.nad_swot_clem_err - ds.clem_est).isel(time=t_idx).plot()
# import zarr
# clem_ds = (
#     xr.open_zarr(zarr.DirectoryStore('../clement_cal_all_err/zarr/err'), group='2013/1')
#     .pipe(lambda ds: ds.isel(time=~np.isnan(ds.time)))
#     .pipe(lambda ds: ds.isel(time=ds.time >= swot_nadir.time.min()))
#     .pipe(lambda ds: ds.isel(time=ds.time <= swot_nadir.time.max()))
#     .sel(time=swot_nadir.time, method='nearest',  tolerance=pd.to_timedelta(100, unit='ms'))
#     .assign(
#         total_err=lambda ds: ds['roll'] + ds.phase + ds.karin + ds.bd + ds.timing,
#         total_est=lambda ds: ds.linear + ds.quad + ds.constant,
#     )
#     [['total_err', 'total_est']]
#     # .interp(time=swot_nadir.time)
#     # .interp(time=swot_nadir.time)
#     .load()
# )
print(1)

# clem_ds.pipe(lambda ds: ((ds.total_err - ds.total_est)**2).mean())
# clem_ds.pipe(lambda ds: ((ds.total_err)**2).mean())
# %%

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
cur_das = {
            'SWOT SSH': swath_ds.gt,
            'SWOT SSH + NOISE': swath_obs_ds['q.xp_three.err_low'],
            '4DVarnet-SLA': swath_ds['q.xp_two.err_no_loc'],
            '4DVarnet-SLA-SWOT': swath_ds['q.xp_three.err_low'],
            '4DVarnet-SLA-SWOT-grad': swath_ds[ 'q.xp_four.alpha_grad'],
            'Direct Inversion ViT': swath_ds[ 'q.xp_six.vit_swot_more_dropout'],
            'Direct  phi err': swath_ds[ 'q.xp_seven.direct_phi_err'],
            'Direct  phi err cal': swath_ds[ 'q.xp_seven.phi_err_cal'],
            }
cur_ds = xr.Dataset(
        cur_das
        )
# clem_ds.pipe(lambda ds: ((ds.total_err - ds.total_est)**2).mean())

# fig, ax = plt.subplots()
# clem_ds.sel(time="2013-01-09").pipe(lambda ds: ((ds.total_err)**2).mean('ac')).reset_index('time').plot(ax=ax)
# clem_ds.sel(time="2013-01-09").pipe(lambda ds: ((ds.total_est)**2).mean('ac')).reset_index('time').plot(ax=ax)
# # swot_ds.pipe(lambda ds: ((ds.err)**2).mean('nC')).reset_index('time').plot(ax=ax)

# clem_ds.total_err.T.isel(time=slice(None, 1000)).reset_index('time').plot(figsize=(10, 2))
# swath_obs_ds.assign(total_err=lambda ds: ((ds.gt - ds[xps[0]]))).total_err.T.isel(x_al=slice(None, 1000)).reset_index('x_al').plot(figsize=(10, 2))

# clem_ds.total_err.T.isel(time=slice(None, 1000)).reset_index('time').plot(figsize=(10, 2))

# %% Plots swath ssh grad loc glob

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 0), da) /2
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 1), da) /2
    return np.hypot(dx_ac, dx_al)

def sobel_median_bef(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(ndimage.median_filter(_da, 5), 0), da)
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(ndimage.median_filter(_da, 5), 1), da)
    return np.hypot(dx_ac, dx_al)

def sobel_median_aft(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 0), da)
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, 1), da)
    grad_ds = np.hypot(dx_ac, dx_al)
    return xr.apply_ufunc(lambda _da: ndimage.median_filter(_da, 5), grad_ds)

swath_grad_ds = sobel(swath_ds)
cur_grad_ds = sobel(cur_ds)
# swath_grad2_ds = sobel_median_bef(swath_ds)
# swath_grad3_ds = sobel_median_aft(swath_ds)

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
    # swath_ds
    cur_ds
    .sel(x_ac=list(range(-46, -24, 2)) + list(range(26, 48, 2)))
    # .pipe(lambda ds: ds.isel(x_al= (~np.isnan(ds.gt)).all('x_ac')))
    .pipe(lambda ds: ds.isel(x_al= (~np.isnan(ds['SWOT SSH'])).all('x_ac')))
    # .reindex(x_ac=list(range(-48, -22, 2)) + list(range(24, 50, 2)))
    .drop(['lat', 'lon'])
)

trim_grad_sw_ds = sobel(trim_sw_ds)

ssh_fig = get_swath_fig(trim_sw_ds.reindex(x_ac=np.arange(-50, 52, 2)).pipe(lambda ds: ds.isel(x_al=ds.chunk_nb==2)))
grad_fig = get_swath_fig(trim_grad_sw_ds.reindex(x_ac=np.arange(-50, 52, 2)).pipe(lambda ds: ds.isel(x_al= ds.chunk_nb==2)))

# TODO compute all spatial resolutions
def get_spat_reses(trim_ds):
    spat_reses = []
    chunks = trim_ds.groupby('chunk_nb')
    for chunk, g in chunks:
        print(chunk)
        # for c in configs + ['clem_cal_obs']:
        for c in trim_ds :
            if c == 'SWOT SSH':
                continue
            print(c)
            spat_reses.append(
                {
                    'xp_long': c,
                    'spat_res': get_swath_psd_score(g['SWOT SSH'].drop('chunk_nb'), g[c].drop('chunk_nb')).item(),
                    # 'spat_res': get_swath_psd_score(g.gt.drop('chunk_nb'), g[c].drop('chunk_nb')).item(),
                    'chunk_nb':g.chunk_nb.isel(x_al=0).item()
                }
            )
    spat_res_df = pd.DataFrame(spat_reses)
    return spat_res_df

spat_res_df = get_spat_reses(trim_sw_ds) 
spat_res_grad_df = get_spat_reses(trim_grad_sw_ds) 

mean_sr = np.sqrt(spat_res_df.loc[lambda df: ~np.isnan(df.spat_res) & ~ np.isinf(df.spat_res)].groupby('xp_long').spat_res.mean())
mean_g_sr = np.sqrt(spat_res_grad_df.loc[lambda df: ~np.isnan(df.spat_res) & ~ np.isinf(df.spat_res)].groupby('xp_long').spat_res.mean())
std_sr = spat_res_df.loc[lambda df: ~np.isnan(df.spat_res) & ~ np.isinf(df.spat_res)].groupby('xp_long').spat_res.std()
std_g_sr = spat_res_grad_df.loc[lambda df: ~np.isnan(df.spat_res) & ~ np.isinf(df.spat_res)].groupby('xp_long').spat_res.std()
print(pd.concat([mean_sr.to_frame(), mean_g_sr.to_frame(), std_sr.to_frame(), std_g_sr.to_frame()], axis=1).to_latex())
print(pd.concat([mean_sr.to_frame(), mean_g_sr.to_frame(), std_sr.to_frame(), std_g_sr.to_frame()], axis=1).to_markdown())
((cur_ds - cur_ds['SWOT SSH'])**2).mean()
(trim_grad_sw_ds.pipe(lambda ds: (ds - ds['SWOT SSH'])**2).mean())
(trim_grad_sw_ds.pipe(lambda ds: ds - ds['SWOT SSH']**2).mean())

mse = (cur_ds.pipe(lambda ds: (ds - ds['SWOT SSH'])**2).to_dataframe().mean()).loc[list(cur_das.keys())]
mse_grad = (sobel(cur_ds).pipe(lambda ds: (ds - ds['SWOT SSH'])**2).to_dataframe().mean()).loc[list(cur_das.keys())]

# mse = (swath_ds.pipe(lambda ds: (ds - ds.gt)**2).to_dataframe().mean())
# mse_grad = (sobel(swath_ds).pipe(lambda ds: (ds - ds.gt)**2).to_dataframe().mean())
print(pd.concat([mse.to_frame(), mse_grad.to_frame()], axis=1).to_latex())
print(pd.concat([mse.to_frame(), mse_grad.to_frame()], axis=1).to_markdown())
# figs
spat_data = []
chunks = trim_sw_ds.groupby('chunk_nb')
for chunk, g in chunks:
    print(chunk)
    for c in configs:
        spat_data.append(
         get_swath_psd_score(g.gt.drop('chunk_nb'), g[c].drop('chunk_nb'), with_data=True)[1].assign_coords(chunk=g.chunk_nb.isel(x_al=0).item(), xp_long=c)
        )

spat_data_df = pd.concat([dat.to_dataframe() for dat in spat_data]).reset_index()

xp_to_plot = [
    # 'q.xp_six.vit_swot_more_dropout',
 # 'q.xp_five.alpha_10_ssh',
 # 'q.xp_four.alpha_eq',
 # 'q.xp_four.alpha_grad',
 'q.xp_four.swot_oi',
 # 'q.xp_three.err_high',
 'q.xp_three.err_low',
 # 'q.xp_three.swot_low',
 # 'q.xp_two.err_no_loc',
 # 'q.xp_two.swot_no_loc'
 'clem_cal_obs'
 ]

sns.catplot(
        data=(
            spat_data_df
            .loc[lambda df: df.wl < 10**6]
            # .loc[lambda df: df.chunk < 9]
            .loc[lambda df: df.xp_long.map(lambda s: s in xp_to_plot)]
            .loc[lambda df: df['PSD score'] > 0]
            .assign(
                wl=lambda df: df.wl / 5 // 1 * 5,
                **{'PSD score': lambda df: df['PSD score'] * 20 // 1 / 20,},
                # log_wl=lambda df: np.log10(df.wl)
                )
            ),
        y='PSD score',
        x='wl',
        col='xp_long',
        kind='violin',
)

sns.violinplot(x='spat_res', y='xp_long', data=spat_res_df.loc[lambda df: df.xp_long.map(lambda s: s in xp_to_plot)])
with pd.option_context('mode.use_inf_as_na', True):
    out_df = spat_res_grad_df.pivot(columns='xp_long', index='chunk_nb', values='spat_res').dropna(how='any').stack().reset_index().rename({0: 'spat_res_grad'}, axis=1)
sns.violinplot(x='spat_res_grad', y='xp_long', data=out_df.loc[lambda df: df.xp_long.map(lambda s: s in xp_to_plot)])


# %% Metrics 
# ssh_metrics = ((swath_ds[[f'{cfg}' for cfg in configs]+ ['clem_cal_obs']] - swath_ds.gt).to_dataframe()**2).mean()[configs+ ['clem_cal_obs']]
# grad_metrics = ((swath_grad_ds[[f'{cfg}' for cfg in configs + ['clem_cal_obs']]] - swath_grad_ds.gt).to_dataframe()**2).mean()[configs+ ['clem_cal_obs']]
ssh_metrics = ((swath_ds[[f'{cfg}' for cfg in configs]] - swath_ds.gt).to_dataframe()**2).mean()[configs]
grad_metrics = ((swath_grad_ds[[f'{cfg}' for cfg in configs ]] - swath_grad_ds.gt).to_dataframe()**2).mean()[configs]
# grad2_metrics = ((swath_grad2_ds[[f'{cfg}' for cfg in configs]] - swath_grad2_ds.gt).to_dataframe()**2).mean()[configs]
# grad3_metrics = ((swath_grad3_ds[[f'{cfg}' for cfg in configs]] - swath_grad3_ds.gt).to_dataframe()**2).mean()[configs]

ssh_metrics.loc[['q.xp_four.alpha_eq', 'q.xp_four.alpha_grad', 'q.xp_three.err_low', 'q.xp_two.err_no_loc']].plot(kind='bar')
grad_metrics.loc[['q.xp_four.alpha_eq', 'q.xp_four.alpha_grad', 'q.xp_three.err_low', 'q.xp_two.err_no_loc']].plot(kind='bar')
# grad2_metrics.loc[['q.xp_four.alpha_eq', 'q.xp_four.alpha_grad', 'q.xp_three.err_low', 'q.xp_two.err_no_loc']].plot(kind='bar')
# grad3_metrics.loc[['q.xp_four.alpha_eq', 'q.xp_four.alpha_grad', 'q.xp_three.err_low', 'q.xp_two.err_no_loc']].plot(kind='bar')


with pd.option_context('mode.use_inf_as_na', True):
    spat_res_grad_df_no_nan = spat_res_grad_df.pivot(columns='xp_long', index='chunk_nb', values='spat_res').dropna(how='any').stack().reset_index()

with pd.option_context('mode.use_inf_as_na', True):
    spat_res_df.pivot(columns='xp_long', index='chunk_nb', values='spat_res').dropna(how='any').mean().plot(kind='bar')

metrics_df = (
        pd.concat(
    [
        ssh_metrics.to_frame().assign(metric='mse', tgt='loc', src='loc'),
        grad_metrics.to_frame().assign(metric='mse_grad', tgt='loc', src='loc'),
        spat_res_df.groupby('xp_long').spat_res.mean().to_frame().assign(metric='spat_res_mean', tgt='loc', src='loc').rename({'spat_res': 0}, axis=1),
        spat_res_df.groupby('xp_long').spat_res.median().to_frame().assign(metric='spat_res_median', tgt='loc', src='loc').rename({'spat_res': 0}, axis=1),
        spat_res_grad_df_no_nan.groupby('xp_long')[0].mean().to_frame().assign(metric='spat_res_grad_mean', tgt='loc', src='loc'),
        spat_res_grad_df_no_nan.groupby('xp_long')[0].median().to_frame().assign(metric='spat_res_grad_median', tgt='loc', src='loc'),
    ]
)
        .reset_index()
        .rename({0: 'value', 'index': 'xp_long'}, axis=1)
        .assign(noise =  lambda df: df.xp_long.map(lambda s: 'None' if 'swot' in s else 'roll' if 'roll' in s else 'all' if 'err' in s else 'obs'))
        .assign(loss = lambda df: df.xp_long.map(lambda s: 'no glob' if 'no_glob' in s else 'no loc' if 'no_loc' in s else 'all'))
        .assign(xp_short = lambda df: df.xp_long.map(lambda s: s.split('.')[-1]))
)
# grad_fig = get_swath_fig(swath_grad_ds[list(glob_glob_df.loc[lambda df: df.metric=='mse'].xp_long)+ ['gt_glob']].reindex({'x_ac': np.arange(-60, 62, 2)}))



# loss terms

glob_glob_df = (
        metrics_df.loc[lambda df: df.xp_short.map(lambda s: s in [
# 'vit_swot_more_dropout',
# 'alpha_10_ssh',
# 'alpha_eq',
# 'alpha_grad',
# 'swot_oi',
# 'err_high',
'err_low',
# 'swot_low',
'err_no_loc',
])])

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


phir_mod = mods['q.xp_seven.direct_phi_err']

for xp, config_pkg in xp_configs:
    ckpt_path = get_most_recent_ckpt(
            config_pkg=config_pkg,
            xp=xp,
            # xp='no_roll',
            # xp='roll',
        )
    runner = main.FourDVarNetRunner(config=config_pkg)
    break

batch = next(iter(runner.dataloaders['test']))
import torch
targets_OI, inputs_Mask, inputs_obs, target_obs_GT, targets_GT = batch
state = torch.cat((targets_OI, targets_OI, targets_OI), dim=1)
logger =  
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger(save_dir='comput_graph')
logger.log_graph(phir_mod.model, (state, None, None))
logger.experiment.add_graph(phir_mod.model.phi, state)
phir_mod.logger
