import hydra
import pandas as pd
import holoviews as hv
from holoviews import opts
import pytorch_lightning as pl
from pathlib import Path

try:
    hv.extension('matplotlib')
except Exception:
    pass

import xarray as xr
from scipy import ndimage
import numpy as np
import torch
import importlib
from hydra.utils import instantiate, get_class, call
import runner as runner_mod
import lit_model_augstate
import sys
sys.path.append('4dvarnet-core')

import metpy.calc as mpcalc
importlib.reload(lit_model_augstate)
import metrics
importlib.reload(metrics)
from utils import get_cfg, get_dm, get_model
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
import hydra_config


import sys
# sys.path.append('4dvarnet-core')
# sys.path.append('4dvarnet-core/ose/eval_notebooks')
sys.path.append('ose/eval_notebooks')
import eval_4dvarnet_test_OSSE
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
for xp_dir in list(Path('results/xp30').glob('*ds2*')):
    cfg_n, ckpt = get_best_ckpt(xp_dir)
    cfg_set.append((cfg_n, ckpt))
    # if pd.to_datetime(Path(ckpt).stat().st_mtime, unit='s')> pd.to_datetime('2023-01-01'):
    #     print(xp_dir, pd.to_datetime(Path(ckpt).stat().st_mtime, unit='s'))
    #     # print(xp_dir, ckpt)
    #     cfg_set.append((cfg_n, ckpt))


cfg_set = []
for xp_dir in list(Path('results/xp30').glob('*ds2*')):
    version_dir = max(xp_dir.glob('version_*'), key=lambda d: int(str(d).split('_')[-1]))
    cfg_n = str(version_dir).split('/')[-2]
    for ckpt in version_dir.glob('checkpoints/*'):
        cfg_set.append((cfg_n, ckpt))

'''
dashboard/qxp23_sst_5nad_aug8_ds1_dT29_13/version_0/checkpoints/modelCalSLAInterpGF-epoch=241-val_loss=0.4742.ckpt
dashboard/qxp23_sst_5nad_aug8_ds1_dT29_13/version_0/checkpoints/modelCalSLAInterpGF-epoch=307-val_loss=0.4600.ckpt
dashboard/qxp23_sst_5nad_aug8_ds1_dT29_13/version_0/checkpoints/modelCalSLAInterpGF-epoch=335-val_loss=0.4791.ckpt
'''

cfg_set = [('qxp23_sst_5nad_aug8_ds1_dT29_13', 'dashboard/qxp23_sst_5nad_aug8_ds1_dT29_13/version_0/checkpoints/modelCalSLAInterpGF-epoch=307-val_loss=0.4600.ckpt')]
cfg_set = [('qxp20_5nad_sst', 'results/xp20/qxp20_5nad_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=138-val_loss=0.4040.ckpt')]
cfg_set = [('qxp20_5nad_no_sst', 'results/xp20/qxp20_5nad_no_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=85-val_loss=0.7589.ckpt')]
cfg_set = [('qxp23_sst_swot_w_oi_aug8_ds1_dT29_13', 'results/xp23/qxp23_sst_swot_w_oi_aug8_ds1_dT29_13/version_0/checkpoints/modelCalSLAInterpGF-epoch=337-val_loss=0.3767.ckpt')]
cfg_set = [('qxp23_sst_swot_w_oi_aug8_ds1_dT29_8', 'results/xp23/qxp23_sst_swot_w_oi_aug8_ds1_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=278-val_loss=1.1519.ckpt')]
cfg_set = [('qxp20_swot_sst', 'results/xp20/qxp20_swot_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=139-val_loss=0.2836.ckpt')]
cfg_set = [('qxp20_swot_no_sst', 'results/xp20/qxp20_swot_no_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=131-val_loss=0.4958.ckpt')]
cfg_set = [('qxp23_no_sst_swot_w_oi_aug8_ds1_dT29_13', 'results/xp23/qxp23_no_sst_swot_w_oi_aug8_ds1_dT29_13/version_0/checkpoints/modelCalSLAInterpGF-epoch=194-val_loss=1.0944.ckpt')]
# cfg_set = []

# for xp_dir in list(Path('results/xp29').glob('*ds1*')):
#     cfg_n, ckpt = get_best_ckpt(xp_dir)
#     # print(xp_dir, ckpt)
#     cfg_set.append((cfg_n, ckpt))


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

cfg = get_cfg(
    cfg_set[0][0],
    overrides=[
        f'file_paths={fp}',
        *ose
])

ose_metrics = {}
# sorted(list(Path('test_logs').glob('qxp28*/**/test.nc')), key=lambda p: p.stat().st_mtime)
sorted(list(Path('test_logs').glob('qxp30*/**/test.nc')), key=lambda p: p.stat().st_mtime)
for p,c in zip(sorted(list(Path('test_logs').glob('qxp30*/**/test.nc')), key=lambda p: p.stat().st_mtime), cfg_set):
    print(p,c)
    # continue
    data = xr.open_dataset(p)
    # data['pred'] = data.pred.groupby('time').apply(lambda g: xr.apply_ufunc(lambda npa :ndi.median_filter(npa, size=7), g))
    out = eval_4dvarnet_test_OSSE.metrics_ose(
            data, cfg.file_paths.ose_test_along_track)
    print(out['Leaderboard'])
    ose_metrics[p] = out['Leaderboard']
    # break


ose_metrics_df = pd.DataFrame([
    {'path': k, 'cfg': c[0], 'xp': str(k).split('/')[1], **v.loc[1].to_dict()}
    for (k,v),c in zip(ose_metrics.items(), cfg_set)])

ose_metrics_df.columns
print(ose_metrics_df.to_markdown())
print(ose_metrics_df[['cfg','µ(RMSE) ', 'σ(RMSE)', 'λx (km)']].to_markdown())
ose_metrics_df[['µ(RMSE) ', 'σ(RMSE)', 'λx (km)']]= ose_metrics_df[['µ(RMSE) ', 'σ(RMSE)', 'λx (km)']].applymap(float)
print(ose_metrics_df[['cfg','µ(RMSE) ', 'λx (km)']].groupby('cfg').mean().sort_values( 'λx (km)').to_markdown())
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


import xarray as xr

oi = xr.open_dataset(
    '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc',
).ssh

import pandas as pd
# glorys = xr.open_dataset('../sla-data-registry/GLORYS/cmems_mod_glo_phy_my_0.083_P1D-m_1674061226086.nc')
# glo_da = glorys.zos.load().rename(latitude='lat', longitude='lon')
# glo_da['time'] =  pd.to_datetime(glo_da['time']).date

ds = xr.open_dataset('/raid/localscratch/qfebvre/o2o_enatl_wo_tide_rec.nc').sel(time=slice('2017-01-01', '2017-12-31')).assign(
# ds = xr.open_dataset('/raid/localscratch/qfebvre/ose_rec230120.nc').sel(time=slice('2017-01-01', '2017-12-31')).assign(
# ds = xr.open_dataset('/raid/localscratch/qfebvre/ose_enatl_wo_tide_rec230120.nc').assign(
        gt=lambda d: d.rec_ssh, pred=lambda d: d.rec_ssh, oi=lambda d: oi.sel(d.coords), obs_inp=lambda d: d.rec_ssh
        # gt=lambda d: d.rec_ssh_6, pred=lambda d: d.rec_ssh_3, oi=lambda d: oi.sel(d.coords), obs_inp=lambda d: d.rec_ssh_6
        # gt=lambda d: d['median_field'], pred=lambda d: d['median_field'], oi=lambda d: oi.sel(d.coords), obs_inp=lambda d: d['median_field']
        # gt=lambda d: d['median_field'], pred=lambda d: d['median_field'], oi=lambda d: oi.sel(d.coords), obs_inp=lambda d: d['median_field']
        # gt=lambda d: d['median'], pred=lambda d: glo_da.interp(**d.coords), oi=lambda d: oi.sel(d.coords), obs_inp=lambda d: d['median']
)

eval_4dvarnet_test_OSSE.metrics_ose(ds, '../sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc')

ds[['pred', 'oi']].mean('lat').mean('lon').to_array().plot.line(hue='variable')
# import importlib

# mod.test_xr_ds
# importlib.reload(eval_4dvarnet_test_OSSE)

print(ose_metrics_df.loc[lambda df: df.xp.map(lambda s: 'ds2' in s)][['xp','µ(RMSE) ', 'σ(RMSE)', 'λx (km)']].to_markdown())
1/0
# ose_metrics_df.to_csv('tmp/ose_metrics_df0102.csv')
# ose_metrics_df.to_csv('tmp/ose_metrics_df1221.csv')
# ose_metrics_df.to_csv('tmp/ose_metrics_df0411.csv')
# ose_metrics_df.to_csv('tmp/ose_metrics_df1001.csv')
ose_metrics_df = pd.read_csv('tmp/ose_metrics_df1221.csv')
ose_metrics_df = pd.read_csv('tmp/ose_metrics_df0411.csv')
import io
scales = pd.read_csv(
io.StringIO('''variable,scale (km)
qxp28_no_sst_duacs_emul_ose_aug3_ds2_dT29_8,63.309
qxp28_no_sst_enatl_w_tide_aug3_ds2_dT29_8,67.4348
qxp28_no_sst_enatl_wo_tide_aug3_ds2_dT29_8,68.2462
qxp28_no_sst_natl20_aug3_ds2_dT29_8,68.7848
qxp28_no_sst_glo12_rea_aug3_ds2_dT29_8,87.1803
qxp28_no_sst_glo12_free_aug3_ds2_dT29_8,90.1228
qxp28_no_sst_orca25_aug3_ds2_dT29_8,101.948'''
))
scales.columns
df = (
    ose_metrics_df
    .set_index('cfg')
    .join(scales.set_index('variable'))
    .reset_index()
    .assign(data=lambda df: df.cfg.map(lambda x:'_'.join(x[13:].split('_')[:-4])))
    .rename({'scale (km)': 'MES (km)'}, axis=1)
)

toplot = [
    'orca25',
    'glo12_free',
    'glo12_rea',
    # 'enatl_w_tide',
    # 'enatl_wo_tide',
    # 'duacs_emul_ose',
    'natl20'
]

label_map = {
        'orca25': 'ORCA025 (1/4°)',
        'glo12_free': 'GLORYS12 (1/12°)',
        'glo12_rea': 'GLORYS12 with Reanalysis (1/12°)',
        'enatl_w_tide': 'NEMO 1/60° with tide',
        'enatl_wo_tide': 'NEMO 1/60° without tide',
        'duacs_emul_ose': 'emul duacs',
        'natl20': 'NATL60 (1/60°)',
}

plot_kwargs = {'x': 'Mean eddy scale (km)', 'y': 'λx (km)', 'hue': 'Training simulation'}
plot_kwargs = {'x': 'Mean eddy scale (km)', 'y': 'µ(RMSE) ', 'hue': 'data'}
# ax = (
#     df.loc[df.data.isin(toplot)].plot(**plot_kwargs , kind='scatter')
# )
import seaborn as sns

df = df.assign(**{'Training simulation':lambda df: df.data.map(lambda d: label_map[d])})
df = df.rename({'MES (km)': 'Mean eddy scale (km)',  'λx (km)': 'Resolved scale'}, axis=1)
df
sns.scatterplot(data=df.loc[df.data.isin(toplot)].sort_values( 'Mean eddy scale (km)'), **plot_kwargs)


print(df.loc[df.data.isin(toplot)][['Training simulation',  'λx (km)', 'µ(RMSE) ']].set_index('Training simulation').to_latex())
# import matplotlib.pyplot as plt
# ax = df.loc[df.data.isin(toplot)].sort_values('scale (km)').pipe(lambda ddf: plt.errorbar(ddf['scale (km)'], 'µ(RMSE) ', yerr='σ(RMSE)', data=ddf))

df.loc[df.data.isin(toplot)][list(plot_kwargs.values())].apply(lambda row: ax.text(*row),axis=1);

training_data = {}
domain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}
for xp in ose_metrics_df.cfg:
    cfg = get_cfg(xp, overrides=[ f'file_paths={fp}'],)
    training_data[xp] = xr.open_dataset(cfg.params.files_cfg.gt_path)[cfg.params.files_cfg.gt_var].sel(domain)


def reset_time(ds):
    ds = ds.copy()
    ds['time'] = ((pd.to_datetime(ds.time) - pd.to_datetime('2006-01-01')) /pd.to_timedelta('1D')) %366 //1
    ds = ds.sortby('time')
    return ds


def reset_latlon(ds, dx=0.1):
    ds = ds.copy()
    ds['lat'] = np.arange(34, 44, dx)
    ds['lon'] = np.arange(-65, -55, dx)
    return ds

def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da



for training_data_path,c in zip(sorted(list(Path('.').glob('version_*/**/test.nc'))), ose_metrics_df.cfg): #cfg_set):
    print(training_data_path, c)
    # if c == 'qxp28_no_sst_duacs_emul_ose_aug3_ds2_dT29_8': continue
    # continue
    cfg = get_cfg(c, overrides=[ f'file_paths={fp}'],)
    training_data['_'.join(c[13:].split('_')[:-4])] = xr.concat(
            [
                xr.open_dataset(cfg.params.files_cfg.gt_path)[cfg.params.files_cfg.gt_var].isel(time=slice(0,366))
                    .sel(domain).coarsen(lat=2, lon=2).mean()
                    .pipe(reset_time).pipe(reset_latlon)
                    .load().pipe(remove_nan)
                    .expand_dims(dim={'phase':1})
                    .assign_coords(phase=('phase', ['train'])),
                xr.open_dataset(training_data_path).isel(time=slice(0,366))
                    .pred.pipe(reset_time).pipe(reset_latlon).load().pipe(remove_nan)
                    .expand_dims(dim={'phase':1}).assign_coords(phase=('phase', ['pred']))
            ],
            dim='phase')

# testing_data = {}
# for testing_data_path in list(Path('test_logs').glob('**/test.nc')):
#     testing_data[str(testing_data_path).split('/')[1][13:]] = xr.open_dataset(testing_data_path)

# list(testing_data.keys())


import metpy.calc as mpcalc
import pyinterp.fill
import pyinterp.backends.xarray


# training_ds = xr.Dataset({k: v.pipe(vort) for k, v in training_data.items()}).load()
training_ds = xr.Dataset(training_data)

toplot = [
    # 'enatl_w_tide',
    'orca25',
    'glo12_free',
    'glo12_rea',
    # 'enatl_wo_tide',
    # 'duacs_emul_ose',
    'natl20'
]

import scipy.ndimage as ndi

robust=False
(
    training_ds['natl20']
    .isel(time=50)
    .sel(phase='pred')
    # .pipe(vort)
    .pipe(lambda da: xr.apply_ufunc(lambda npa :ndi.median_filter(npa, size=filt),da))
    .pipe(vort)
    .plot(figsize=(6,5), robust=robust)
)


filt = 4
time = 60
filt_ds = training_ds.copy()
filt_ds = filt_ds.isel(time=time).map(geo_energy)
filt_ds.loc[{'phase': 'pred'}]=(
    filt_ds.sel(phase='pred').map(lambda da: xr.apply_ufunc(lambda npa :ndi.median_filter(npa, size=filt), da))
)
toplot = [ 'orca25', 'glo12_free','glo12_rea', 'natl20' ]
titles = [ 'ORCA025 1/4°', 'GLORYS12 1/12°','GLORYS12 1/12° Reanalysis', 'NATL60 1/60°']
ax = filt_ds[toplot].to_array().plot.pcolormesh(
    col='variable', row='phase', figsize=(18,8), robust=False,
    cmap='viridis',
    # cmap='RdGy_r',
    **filt_ds[toplot].to_array().pipe(lambda da:dict(vmin=da.quantile(0.002), vmax=da.quantile(0.998)))
    # **dict(vmin=-5e-5, vmax=5e-5)
)
for ti, axx in zip(titles, ax.axes[0]):
    axx.set_title(ti)

titles = [ 'ORCA025 1/4°', 'GLORYS12 1/12°','GLORYS12 1/12° Reanalysis', 'NATL60 1/60°']
ax3 = filt_ds[toplot].to_array().sel(lat=slice(34,38), lon=slice(-62,-58), phase='pred').plot.pcolormesh(
    col='variable', figsize=(18,5), robust=True, cmap='viridis', #cmap='RdGy_r',
    # **filt_ds[toplot].to_array().pipe(lambda da:dict(vmin=da.quantile(0.002), vmax=da.quantile(0.998)))
    # **dict(vmin=-5e-5, vmax=5e-5)
)
for ti, axx in zip(titles, ax3.axes[0]):
    axx.set_title(ti)

ax2 = (
    filt_ds['duacs_emul_ose']
    .sel(phase='train')
    .pipe(lambda da: xr.apply_ufunc(lambda npa :ndi.median_filter(npa, size=filt), da))
    .plot(vmin=ax.cbar.vmin, vmax=ax.cbar.vmax, cmap=ax.cbar.cmap, figsize=(5,5), add_colorbar=False)
)
ax2.axes.set_title('DUACS')

ax.axes[0][0].set_title('toto')
dir(ax.axes)
toplot = [ 'glo12_free', 'glo12_rea', ]
ax = filt_ds[toplot].to_array().plot.pcolormesh(col='variable', row='phase', figsize=(10,8), robust=True)

toplot = [ 'enatl_w_tide', 'enatl_wo_tide', ]
ax = filt_ds[toplot].to_array().plot.pcolormesh(col='variable', row='phase', figsize=(10,8), robust=True)

(
    filt_ds['glo12_rea']
    .sel(phase='train')
    # .pipe(lambda da: xr.apply_ufunc(lambda npa :ndi.median_filter(npa, size=filt), da))
    .plot.pcolormesh(vmin=ax.cbar.vmin, vmax=ax.cbar.vmax, cmap=ax.cbar.cmap, add_colorbar=False)
)

ax.cmap
dir(ax.cbar.cmap)

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
psd_fn = lambda da: xrft.power_spectrum(
        da, dim='lon', scaling='density', real_dim='lon', truncate=True, window='hann')
psds_ds = tpds.isel(time=np.isfinite(tpds.to_array()).all(['phase', 'variable','lat', 'lon'])).map(lambda da: psd_fn(da).mean('time').mean('lat'))

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


weighted_scale = (
    psds_ds.sum() / (psds_ds * psds_ds.freq_lon).sum('freq_lon')
)
fmt_ws = (100*weighted_scale).to_array().to_dataframe(name='scale (km)').sort_values(by='scale (km)')
print(fmt_ws.to_markdown())

weighted_scale = (
    psds_ds.sum() / (psds_ds * psds_ds.freq_lat).sum('freq_lat')
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
