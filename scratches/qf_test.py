import hydra
import pandas as pd
import holoviews as hv
from holoviews import opts
import pytorch_lightning as pl
from pathlib import Path

hv.extension('matplotlib')

import xarray as xr
from scipy import ndimage
import numpy as np
import torch
import importlib
from hydra.utils import instantiate, get_class, call
import runner as runner_mod
import lit_model_augstate

import metpy.calc as mpcalc
importlib.reload(lit_model_augstate)
from utils import get_cfg, get_dm, get_model
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
import hydra_config

fp = "dgx_ifremer"

cfg_set = []
# cfg_n, ckpt = 'qxp23_no_sst_swot_w_oi_aug3_ds2_dT29_13', 'results/xp23/qxp23_no_sst_swot_w_oi_aug3_ds2_dT29_13/version_2/checkpoints/modelCalSLAInterpGF-epoch=173-val_loss=2.5762.ckpt'
cfg_n, ckpt = 'qxp22_5nad_aug3_ds1', 'results/xp22/qxp22_5nad_aug3_ds1/version_0/checkpoints/lx691_lt406_mu959.ckpt'
cfg_set.append((cfg_n, ckpt))

### SOTA OSE 122km 0.877 km
cfg_n, ckpt = 'qxp23_no_sst_5nad_aug3_ds2_dT29_8', 'results/xp23/qxp23_no_sst_5nad_aug3_ds2_dT29_8/version_1/checkpoints/modelCalSLAInterpGF-epoch=344-val_loss=4.5537.ckpt'
cfg_set.append((cfg_n, ckpt))

# cfg_n, ckpt = 'qxp26_no_sst_natl1_14D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl1_14D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=189-val_loss=0.3319.ckpt'
# cfg_set.append((cfg_n, ckpt))
# cfg_n, ckpt = 'qxp26_no_sst_natl1_1D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl1_1D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=189-val_loss=0.5823.ckpt'
# cfg_set.append((cfg_n, ckpt))
# cfg_n, ckpt = 'qxp26_no_sst_natl05_7D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl05_7D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=189-val_loss=0.8418.ckpt'
# cfg_set.append((cfg_n, ckpt))
# cfg_n, ckpt = 'qxp26_no_sst_natl05_1D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl05_1D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=105-val_loss=1.8854.ckpt'
# cfg_set.append((cfg_n, ckpt))
# cfg_n, ckpt = 'qxp26_no_sst_natl025_1D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl025_1D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=196-val_loss=2.6825.ckpt'
# cfg_set.append((cfg_n, ckpt))
# cfg_n, ckpt = 'qxp26_no_sst_base_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_base_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=187-val_loss=4.6854.ckpt'
# cfg_set.append((cfg_n, ckpt))
# cfg_n, ckpt = 'qxp26_no_sst_glorys12_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_glorys12_aug1_ds2_dT29_8/version_1/checkpoints/modelCalSLAInterpGF-epoch=176-val_loss=4.3510.ckpt'


vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()

def get_best_ckpt(xp_dir, version=None):
    print(xp_dir)
    if version is None:
        version_dir = max(xp_dir.glob('version_*'), key=lambda d: int(str(d).split('_')[-1]))
    else:
        version_dir = xp_dir / f'version_{version}'

    latest_ckpt = max(version_dir.glob('checkpoints/*'), key=lambda f: torch.load(f)['global_step'])

    ckpt = torch.load(latest_ckpt)
    best_ckpt = xp_dir / next(iter(ckpt['callbacks'].values()))['best_model_path']
    xp_name = str(version_dir).split('/')[-2]
    return xp_name, str(best_ckpt)




cfg_set = []
for xp_dir in list(Path('results/xp28').glob('*ds2*')):
    cfg_n, ckpt = get_best_ckpt(xp_dir)
    # print(xp_dir, ckpt)
    cfg_set.append((cfg_n, ckpt))
    
cfg_set = []
for xp_dir in list(Path('results/xp29').glob('*ds1*')):
    cfg_n, ckpt = get_best_ckpt(xp_dir)
    # print(xp_dir, ckpt)
    cfg_set.append((cfg_n, ckpt))


ose = [
    # 'params.files_cfg.oi_path=${file_paths.ose_oi_path}',
    # 'params.files_cfg.oi_path=${file_paths.oi_ose_nad_path}',
    # 'params.files_cfg.oi_var=ssh',
    # 'params.files_cfg.obs_mask_path=${file_paths.ose_obs_mask_path}',
    # 'params.files_cfg.obs_mask_var=ssh',
    # 'params.files_cfg.gt_path=${file_paths.ose_gt_path}',
    # 'params.files_cfg.gt_var=ssh',
    '+xp@_global_=qfebvre/duacsfree_ose_osse_nad_gf_wsst',
    # '+params.files_cfg.test_track=${file_paths.ose_test_along_track}',
    # '+splits@datamodule=ose',
]

ose_sst = [
    'datamodule.sst_path=${file_paths.ose_sst_path}',
    'datamodule.sst_var=analysed_sst',
]

no_ds = [
    'datamodule.resize_factor=1',
]

osse_clean = [
    'params.files_cfg.obs_mask_var=ssh_mod',
]

four_nadirs = [
    'params.files_cfg.obs_mask_var=four_nadirs',
]

center_only = [
    'params.patch_weight._target_=lit_model_augstate.get_cropped_hanning_mask',
    'params.patch_weight.crop.time=1',
]


def run_test(cfg_n, ckpt):
    log_dir = [
            f'logger.save_dir=test_logs/{cfg_n}'
    ]
    ors = [
            # *no_ds,
            # *four_nadirs,
            # *center_only,
            *ose,
            *log_dir,
            # *ose_sst
    ]
    # print(OmegaConf.to_yaml(cfg.file_paths))
    dm_osse = get_dm(cfg_n, setup=False,
            add_overrides=[
                f'file_paths={fp}',
                # *ors
            ]
    )
    dm_osse.setup()
    dm = get_dm(cfg_n, setup=False,
            add_overrides=[
                f'file_paths={fp}',
                *ors
            ]
    )

    cfg = get_cfg(
        cfg_n,
        overrides=[
            f'file_paths={fp}',
            *ors
    ])

    mod = get_model(
            cfg_n,
            ckpt,
            add_overrides=[
                f'file_paths={fp}',
                *ors
            ],
            dm=dm)
    # ds = xr.open_dataset(cfg.params.files_cfg.obs_mask_path)
    # ds = xr.open_dataset(cfg.params.files_cfg.gt_path)
    # ds  = xr.open_dataset(cfg.params.files_cfg.oi_path)
    # print()
    # print()
    # print(xr.open_dataset(cfg.params.files_cfg.gt_path))
    cfg.params.files_cfg.oi_path
    cfg.params.files_cfg.obs_mask_path
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    if cfg.get('logger') is not None:
        print('instantiating logger')
        print(OmegaConf.to_yaml(cfg.logger))
        logger = instantiate(cfg.logger)
    else:
        logger=True
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = runner_mod.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, logger=logger)
    print(runner.logger.log_dir)
    mod = runner._get_model(ckpt_path=ckpt)

    # trainer = pl.Trainer(gpus=1)
    # trainer.test(mod, datamodule=dm)

    # batch = next(iter(dm.val_dataloader()))
    # mod.to('cuda')(mod.transfer_batch_to_device(batch, 'cuda', 0))

    mod.patch_weight.data = torch.tensor(hydra.utils.call(cfg.params.patch_weight))
    mod = runner.test(ckpt, _mod=mod, gpus=1, )
    mod.test_figs['psd']
    return mod, cfg, dm_osse, dm


for cfg_n, ckpt in cfg_set:
    print(cfg_n, ckpt)
    mod, cfg, dm_osse, dm = run_test(cfg_n, ckpt)

# mod, cfg, dm_osse, dm = run_test(cfg_n, ckpt)
# self = mod
# animate_maps(self.x_gt, self.obs_inp, self.x_oi, self.x_rec, self.test_lon, self.test_lat, 'animation.mp4')
1/0

def anim(test_xr_ds, deriv=None,  dvars=['ssh_1_14D', 'ssh_1_1D', 'ssh_025_1D', 'ref']):
    def sobel(da):
        dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
        dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
        return np.hypot(dx_ac, dx_al)

    if deriv is None:
        tpds = test_xr_ds
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdBu'


    if deriv == 'grad':
        tpds = test_xr_ds.map(geo_energy)
        # tpds = test_xr_ds.pipe(sobel)
        clim = (0, tpds[dvars].to_array().max().item())
        cmap = 'viridis'
    
    if deriv == 'lap':
        # tpds = test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1)).isel(time=slice(25, 75, 2))
        tpds = test_xr_ds.map(vort)
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdGy'
    hvds = hv.Dataset(tpds)
    # hvds = hv.Dataset(mod.test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=0.1)).isel(time=slice(None, 10, 1)))
    # hvds = hv.Dataset(mod.test_xr_ds.map(ndimage.laplace).isel(time=slice(None, 10, 1)))
    images = hv.Layout([
            hvds
            .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
            .options(
                cmap=cmap,
                clim=clim,
            )
            for v in dvars
    ]).cols(2).opts(sublabel_format="")

    return images



# ds = xr.open_dataset("/raid/localscratch/qfebvre/4dvarnet-core/dashboard/ose_gf_wsst/lightning_logs/version_1/test.nc")
# images = anim(ds)

# simu_ds = dm_osse.train_ds.datasets[0].gt_ds.ds
# deriv = 'grad'
# simu = anim(simu_ds.assign(time= lambda ds: ds.time - ds.time.min()), deriv=deriv, dvars=[dm_osse.gt_var])

# pred = anim(mod.test_xr_ds.assign(time= lambda ds: ds.time - ds.time.min()), deriv=deriv, dvars=['pred'])
# hv.output(simu + pred, holomap='gif', fps=3, dpi=125)
# 1/0

import sys
sys.path.append('4dvarnet-core')
sys.path.append('4dvarnet-core/ose/eval_notebooks')
sys.path.append('ose/eval_notebooks')
import eval_4dvarnet_test_OSSE

import xarray as xr

oi = xr.open_dataset(
    '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc',
).ssh
ds = xr.open_dataset('/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2022-12-02/15-19-18/ose_ssh_rec.nc').assign(
        gt=lambda d: d.rec_ssh, pred=lambda d: d.rec_ssh, oi=lambda d: oi.sel(d.coords), obs_inp=lambda d: d.rec_ssh
).sel(time=slice('2017-01-01', '2017-12-31'))
eval_4dvarnet_test_OSSE.metrics_ose(ds, 'sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc')

# import importlib

# mod.test_xr_ds
# importlib.reload(eval_4dvarnet_test_OSSE)

cfg = get_cfg(
    cfg_set[0][0],
    overrides=[
        f'file_paths={fp}',
        *ose
])

ose_metrics ={}
sorted(list(Path('test_logs').glob('qxp29*/**/test.nc')), key=lambda p: p.stat().st_mtime)
for p,c in zip(sorted(list(Path('test_logs').glob('qxp29*/**/test.nc')), key=lambda p: p.stat().st_mtime), cfg_set):
    print(p,c)
    # continue
    out = eval_4dvarnet_test_OSSE.metrics_ose(
            xr.open_dataset(p), cfg.file_paths.ose_test_along_track)
    ose_metrics[p] = out['Leaderboard']

ose_metrics_df = pd.DataFrame([
    {'path': k, 'cfg': c[0], 'xp': str(k).split('/')[1], **v.loc[1].to_dict()}
    for (k,v),c in zip(ose_metrics.items(), cfg_set)])

ose_metrics_df.columns
print(ose_metrics_df.to_markdown())
print(ose_metrics_df[['cfg','µ(RMSE) ', 'σ(RMSE)', 'λx (km)']].to_markdown())
1/0
# ose_metrics_df.to_csv('tmp/ose_metrics_df0102.csv')
# ose_metrics_df.to_csv('tmp/ose_metrics_df1221.csv')
# ose_metrics_df.to_csv('tmp/ose_metrics_df0411.csv')
ose_metrics_df = pd.read_csv('tmp/ose_metrics_df1221.csv')

training_data = {}
domain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}
for xp in ose_metrics_df.cfg:
    cfg = get_cfg(xp, overrides=[ f'file_paths={fp}'],)
    training_data[xp] = xr.open_dataset(cfg.params.files_cfg.gt_path)[cfg.params.files_cfg.gt_var].sel(domain)



for training_data_path,c in zip(sorted(list(Path('.').glob('version_*/**/test.nc'))), cfg_set):
    print(training_data_path)
    training_data[c[0]] = xr.open_dataset(training_data_path).pred


# testing_data = {}
# for testing_data_path in list(Path('test_logs').glob('**/test.nc')):
#     testing_data[str(testing_data_path).split('/')[1][13:]] = xr.open_dataset(testing_data_path)

# list(testing_data.keys())

def reset_time(ds):
    ds = ds.copy()
    ds['time'] = (ds.time - ds.time.min()) /pd.to_timedelta('1D')
    return ds
import numpy as np


def reset_latlon(ds):
    ds = ds.copy()
    ds['lat'] = np.arange(34, 44, 0.05)
    ds['lon'] = np.arange(-65, -55, 0.05)
    return ds


import metpy.calc as mpcalc
import pyinterp.fill
import pyinterp.backends.xarray


def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

training_ds = xr.Dataset({k: v.pipe(reset_time).pipe(reset_latlon).load().pipe(remove_nan).pipe(vort) for k, v in training_data.items()}).load()
# training_ds.load().map(remove_nan).map(vort).to_array().isel(time=0).plot.pcolormesh(col='variable', col_wrap=4)

# (training_ds[
#     sorted([ k for k in training_ds if ('0_g' not in k) and ('ds2' in k)])
# ]
#  .isel(time=slice(0, 120, 30)).load().map(remove_nan).map(vort).to_array().plot.pcolormesh(col='variable', row='time', figsize=(15,12)))

# testing_ds = xr.Dataset({k: v.pred.pipe(reset_time) for k, v in testing_data.items()})
# (testing_ds[
#     sorted([ k for k in testing_ds if ('0_g' not in k) and ('ds1' in k)])
#     # sorted([ k for k in testing_ds if ('90' not in k) and ('ds1' in k) and ('natl' in k)])
#     ]
#  .isel(time=slice(0, 120, 30)).load().map(remove_nan).map(geo_energy).to_array().plot.pcolormesh(col='variable', row='time', figsize=(15,12)))

# pred = anim(testing_ds.isel(time=slice(100, 150,2)).map(remove_nan), dvars=sorted([ k for k in testing_ds if ('0_g' not in k) and ('ds2' in k)]), deriv='grad')
# hv.output(pred, holomap='gif', fps=2, dpi=125)

# # locals().update(eval_4dvarnet_test_OSSE.metrics_ose(mod.test_xr_ds, cfg.file_paths.ose_test_along_track))

import xrft
tpds = training_ds

psd_fn = lambda da: xrft.isotropic_power_spectrum(
        da, dim=('lat', 'lon'), truncate=True, window='hann')
psds_ds = tpds.isel(time=np.isfinite(tpds.to_array()).all(['variable','lat', 'lon'])).map(lambda da: psd_fn(da).mean('time'))

# list(psds_ds)
# psds_ds[sorted([ k for k in psds_ds if ('0_g' not in k) and ('ds2' in k)])].to_array().plot.line(
#         x='freq_r',
#         hue='variable',
#         xscale='log',
#         yscale='log',
#         figsize=(10,6)
# )

# psds_ds[sorted([ k for k in psds_ds if
#                 # ('90' not in k)
#                 ('90'  in k)
#                 and ('ds1' in k) and ('natl' in k) or ('natl20_aug2_ds1' in k)
#          or ('natl' not in k)
#         ])].to_array().plot.line(
#         x='freq_r',
#         hue='variable',
#         xscale='log',
#         yscale='log',
#         figsize=(10,6)
# )

# psds_ds[[ k for k in psds_ds if ('90' in k) and ('ds1' in k) and ('natl' in k) or ('natl20_aug2_ds1' in k)]].to_array().plot.line(
#         x='freq_r',
#         hue='variable',
#         xscale='log',
#         yscale='log',
#         figsize=(10,6)
# )

# psds_ds[[ k for k in psds_ds if (('90' in k) or ('natl20_g' not in k)) and ('ds1' in k) ]].to_array().plot.line(
#         x='freq_r',
#         hue='variable',
#         xscale='log',
#         yscale='log',
#         figsize=(10,6)
# )

weighted_scale = (

    psds_ds.sum() / (psds_ds * psds_ds.freq_r).sum('freq_r')
)
fmt_ws = (100*weighted_scale).to_array().to_dataframe(name='scale (km)').sort_values(by='scale (km)')
print(fmt_ws.to_markdown())


# simu = ['natl20', 'glo12_free', 'glo12_rea', 'orca25']
# natl_filt = ['natl20', 'natl20_g1', 'natl20_g3', 'natl20_g5', 'natl20_g5', 'natl20_g8']
# natl_filt_bis = ['natl20', 'natl20_g1_90', 'natl20_g3_90', 'natl20_g5_90', 'natl20_g8_90']
# df =(
#         ose_metrics_df
#     .loc[lambda df: df.xp.map(lambda s: 'ds2' in s)]
#     .assign(data=lambda df: df.xp.map(lambda x:'_'.join(x[13:].split('_')[:-4])))
#     # .loc[lambda df: df.data.map(lambda x: '90' in x or '_g' not in x)]
#     # .loc[lambda df: df.data.isin(simu)]
#     .loc[lambda df: df.data.isin(natl_filt)]
#     # .loc[lambda df: df.data.isin(natl_filt_bis)]
#     .assign(**{'λx (km)': lambda df: df['λx (km)'].map(float)})
#     .assign(**{'µ(RMSE) ': lambda df: df['µ(RMSE) '].map(float)})
#     .set_index('xp').sort_values(by='λx (km)').join(fmt_ws)
# ) 
# plot_kwargs = {'x': 'scale (km)', 'y': 'λx (km)', 'label': 'data'}
# # plot_kwargs = {'x': 'scale (km)', 'y': 'µ(RMSE) ', 'label': 'data'}
# ax = (
#     df.plot(**plot_kwargs , kind='scatter')
# )
# df[list(plot_kwargs.values())].apply(lambda row: ax.text(*row),axis=1);

# oi = xr.open_dataset(cfg.file_paths.oi_ose_osse_nad_path)
# print(
#     Leaderboard
#     .set_index('Method')
#     .loc[['4DVarNet']]
#     .assign(simu_scale=1/weighted_scale.mean('time'))
#     .to_markdown()
# )


# import pandas as pd
# ds = xr.open_dataset("lightning_logs/version_1801515/test.nc")
# glorys=xr.open_dataset('../sla-data-registry/GLORYS/preprocessed_glorys_dc_2021.nc')

# glorys_fmt = glorys.coarsen(lat=2, lon=2).mean().zos.assign_coords(
#     time=pd.to_datetime(glorys.time).date
# ).sel(ds.coords, method='nearest', tolerance=0.01)


# test = glorys_fmt.to_dataset(name='pred').assign(
#         gt=lambda dd: dd.pred,
#         oi=lambda dd: dd.pred,
#         obs_inp=lambda dd: dd.pred,
# ).transpose('time', 'lat', 'lon')

# test = glorys.zos.to_dataset(name='pred').assign(
#         gt=lambda dd: dd.pred,
#         oi=lambda dd: dd.pred,
#         obs_inp=lambda dd: dd.pred,
# ).transpose('time', 'lat', 'lon')
# locals().update(
#     eval_4dvarnet_test_OSSE.metrics_ose(
#         test,
#         # glorys_fmt.to_dataset('pred').assign,
#         # ds.assign(
#         #     pred=(ds.pred.dims, glorys_fmt.transpose(*ds.pred.dims).values)
#         # ),
#         cfg.file_paths.ose_test_along_track
#     )
# )

# cal_data = xr.open_dataset(cfg.file_paths.new_noisy_swot)
# cal_data.four_nadirs.isel(time=10).sel(lat=slice(33,43), lon=slice(-65,-55)).pipe(lambda da: np.isfinite(da)).plot()
