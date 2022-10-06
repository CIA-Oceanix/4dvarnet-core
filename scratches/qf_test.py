import hydra
import holoviews as hv
from holoviews import opts
import pytorch_lightning as pl

hv.extension('matplotlib')

import xarray as xr
from scipy import ndimage
import numpy as np
import torch
import importlib
from hydra.utils import instantiate, get_class, call
import runner as runner_mod
import lit_model_augstate

importlib.reload(lit_model_augstate)
from utils import get_cfg, get_dm, get_model
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
import hydra_config


# cfg_n, ckpt = 'qxp23_no_sst_swot_w_oi_aug3_ds2_dT29_13', 'results/xp23/qxp23_no_sst_swot_w_oi_aug3_ds2_dT29_13/version_2/checkpoints/modelCalSLAInterpGF-epoch=173-val_loss=2.5762.ckpt'
cfg_n, ckpt = 'qxp22_5nad_aug3_ds1', 'results/xp22/qxp22_5nad_aug3_ds1/version_0/checkpoints/lx691_lt406_mu959.ckpt'


### SOTA OSE 122km 0.877 km
# cfg_n, ckpt = 'qxp23_no_sst_5nad_aug3_ds2_dT29_8', 'results/xp23/qxp23_no_sst_5nad_aug3_ds2_dT29_8/version_1/checkpoints/modelCalSLAInterpGF-epoch=344-val_loss=4.5537.ckpt'

cfg_set = []
cfg_n, ckpt = 'qxp26_no_sst_natl1_14D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl1_14D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=189-val_loss=0.3319.ckpt'
cfg_set.append((cfg_n, ckpt))
cfg_n, ckpt = 'qxp26_no_sst_natl1_1D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl1_1D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=189-val_loss=0.5823.ckpt'
cfg_set.append((cfg_n, ckpt))
cfg_n, ckpt = 'qxp26_no_sst_natl05_7D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl05_7D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=189-val_loss=0.8418.ckpt'
cfg_set.append((cfg_n, ckpt))
cfg_n, ckpt = 'qxp26_no_sst_natl05_1D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl05_1D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=105-val_loss=1.8854.ckpt'
cfg_set.append((cfg_n, ckpt))
cfg_n, ckpt = 'qxp26_no_sst_natl025_1D_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_natl025_1D_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=196-val_loss=2.6825.ckpt'
cfg_set.append((cfg_n, ckpt))
cfg_n, ckpt = 'qxp26_no_sst_base_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_base_aug1_ds2_dT29_8/version_0/checkpoints/modelCalSLAInterpGF-epoch=187-val_loss=4.6854.ckpt'
cfg_set.append((cfg_n, ckpt))
cfg_n, ckpt = 'qxp26_no_sst_glorys12_aug1_ds2_dT29_8', 'results/xp26/qxp26_no_sst_glorys12_aug1_ds2_dT29_8/version_1/checkpoints/modelCalSLAInterpGF-epoch=176-val_loss=4.3510.ckpt'


def run_test(cfg_n, ckpt):
    log_dir = [
            f'logger.save_dir=test_logs/{cfg_n}'
    ]
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
    ors = [
            # *no_ds,
            # *four_nadirs,
            # *center_only,
            *ose,
            *log_dir,
            # *ose_sst
    ]
    dm_osse = get_dm(cfg_n, setup=False,
            add_overrides=[
                'file_paths=jz',
                # *ors
            ]
    )
    dm_osse.setup()
    dm = get_dm(cfg_n, setup=False,
            add_overrides=[
                'file_paths=jz',
                *ors
            ]
    )


    cfg = get_cfg(
        cfg_n,
        overrides=[
            'file_paths=jz',
            *ors
    ])

    # print(OmegaConf.to_yaml(cfg.file_paths))
    mod = get_model(
            cfg_n,
            ckpt,
            add_overrides=[
                'file_paths=jz',
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


# for cfg_n, ckpt in cfg_set:
#     print(cfg_n, ckpt)
#     mod, cfg, dm_osse, dm = run_test(cfg_n, ckpt)

mod, cfg, dm_osse, dm = run_test(cfg_n, ckpt)
# self = mod
# animate_maps(self.x_gt, self.obs_inp, self.x_oi, self.x_rec, self.test_lon, self.test_lat, 'animation.mp4')
1/0
print(mod.test_xr_ds.pipe(lambda da: da.gt - da.oi).pipe(lambda da: da**2).mean().pipe(np.sqrt))

print(mod.logger.log_dir)

def anim(test_xr_ds, deriv=None,  dvars=['ssh_1_14D', 'ssh_1_1D', 'ssh_025_1D', 'ref']):
    def sobel(da):
        dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
        dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
        return np.hypot(dx_ac, dx_al)

    if deriv is None:
        tpds = test_xr_ds.isel(time=slice(None, 20, 1))
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdBu'


    if deriv == 'grad':
        tpds = test_xr_ds.pipe(sobel).isel(time=slice(50, 150, 2))
        clim = (0, tpds[dvars].to_array().max().item())
        cmap = 'viridis'
    
    if deriv == 'lap':
        tpds = test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1)).isel(time=slice(25, 75, 2))
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

simu_ds = dm_osse.train_ds.datasets[0].gt_ds.ds
deriv = 'grad'
simu = anim(simu_ds.assign(time= lambda ds: ds.time - ds.time.min()), deriv=deriv, dvars=[dm_osse.gt_var])

pred = anim(mod.test_xr_ds.assign(time= lambda ds: ds.time - ds.time.min()), deriv=deriv, dvars=['pred'])
hv.output(simu + pred, holomap='gif', fps=3, dpi=125)


import sys
sys.path.append('ose/eval_notebooks')
import eval_4dvarnet_test_OSSE
import importlib

mod.test_xr_ds
importlib.reload(eval_4dvarnet_test_OSSE)

locals().update(eval_4dvarnet_test_OSSE.metrics_ose(mod.test_xr_ds, cfg.file_paths.ose_test_along_track))


import xrft
psd_fn = lambda da: xrft.isotropic_power_spectrum(da, dim=('lat', 'lon'), truncate=True)
domain = {'lat': slice(35, 44), 'lon':slice(-65,-55)}
tpds = simu_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1)).isel(time=slice(None, 10, 1))
psd = psd_fn(
        tpds
        .assign(lat=lambda ds: ds.lat*100, lon=lambda ds: ds.lon*100)
        [dm_osse.gt_var]
)
weighted_scale = psd.sum() / (psd * 1/ (psd.freq_r)).sum(dim='freq_r')
print(
    Leaderboard
    .set_index('Method')
    .loc[['4DVarNet']]
    .assign(simu_scale=1/weighted_scale.mean('time'))
    .to_markdown()
)


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
