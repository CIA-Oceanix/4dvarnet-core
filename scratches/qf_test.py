import hydra
import holoviews as hv
from holoviews import opts

hv.extension('matplotlib')

import xarray as xr
from scipy import ndimage
import numpy as np
import torch
import importlib
from hydra.utils import instantiate, get_class, call
import hydra_main
import lit_model_augstate

importlib.reload(lit_model_augstate)
importlib.reload(hydra_main)
from utils import get_cfg, get_dm, get_model
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
import hydra_config

cfg_n, ckpt = 'qxp22_5nad_aug0', 'results/xp22/qxp22_5nad_aug0/version_0/checkpoints/modelCalSLAInterpGF-epoch=173-val_loss=4.9276.ckpt'
cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_5nad_aug1/version_0/checkpoints/modelCalSLAInterpGF-epoch=49-val_loss=4.8408.ckpt'
cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_5nad_aug1/version_0/checkpoints/modelCalSLAInterpGF-epoch=93-val_loss=4.2820.ckpt'
cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_5nad_aug2/version_0/checkpoints/lx709_lt426_mu959.ckpt'
# cfg_n, ckpt = 'qxp22_swot_aug2', 'results/xp22/qxp22_swot_aug2/version_0/checkpoints/lx666_lt416_mu967.ckpt'
# cfg_n, ckpt = 'qxp22_swot_aug0', 'results/xp22/qxp22_swot_aug0/version_0/checkpoints/modelCalSLAInterpGF-epoch=191-val_loss=3.5165.ckpt'
# cfg_n, ckpt = 'qxp22_swot_aug1', 'results/xp22/qxp22_swot_aug2/version_0/checkpoints/modelCalSLAInterpGF-epoch=181-val_loss=2.8195.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_5nad_aug5/version_0/checkpoints/modelCalSLAInterpGF-epoch=307-val_loss=3.6219.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_5nad_aug5/version_0/checkpoints/modelCalSLAInterpGF-epoch=307-val_loss=3.6219.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_5nad_aug5/version_0/checkpoints/modelCalSLAInterpGF-epoch=271-val_loss=3.6078.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_swot_aug5/version_0/checkpoints/modelCalSLAInterpGF-epoch=224-val_loss=2.6954.ckpt'
# cfg_n, ckpt = 'qxp22_swot_aug1', 'results/xp22/qxp22_swot_aug5/version_0/checkpoints/modelCalSLAInterpGF-epoch=331-val_loss=2.7033.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug1', 'results/xp22/qxp22_swot_aug1/version_0/checkpoints/modelCalSLAInterpGF-epoch=67-val_loss=3.4198.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug2', 'results/xp22/qxp22_swot_aug0/version_0/checkpoints/modelCalSLAInterpGF-epoch=38-val_loss=5.1896.ckpt'
# cfg_n = f"baseline/{cfg_n}"
# cfg_n, ckpt = 'qxp21_5nad_sst_11', 'dashboard/qxp21_5nad_sst_11/version_0/checkpoints/modelCalSLAInterpGF-epoch=17-val_loss=1.3117.ckpt'
# cfg_n, ckpt = 'qxp21_5nad_no_sst_11', 'dashboard/qxp21_5nad_no_sst_11/version_0/checkpoints/modelCalSLAInterpGF-epoch=80-val_loss=1.7154.ckpt'
# cfg_n, ckpt = 'qxp21_5nad_no_sst_11', 'dashboard/qxp21_5nad_no_sst_11/version_0/checkpoints/modelCalSLAInterpGF-epoch=102-val_loss=1.7432.ckpt'
# cfg_n, ckpt = 'qxp21_5nad_no_sst_11', 'dashboard/qxp21_5nad_no_sst_11/version_0/checkpoints/modelCalSLAInterpGF-epoch=102-val_loss=1.7432.ckpt'

# cfg_n, ckpt = 'qfebvre/xp_oi_sst', 'results/xp22/xp_oi_sst/version_2/checkpoints/modelCalSLAInterpGF-epoch=169-val_loss=1.8386.ckpt'
cfg_n, ckpt = 'qfebvre/xp_oi_sst', 'results/xp22/xp_oi_sst/version_2/checkpoints/modelCalSLAInterpGF-epoch=338-val_loss=1.6439.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug3_ds1', 'results/xp22/qxp22_5nad_aug3_ds1/version_0/checkpoints/modelCalSLAInterpGF-epoch=333-val_loss=1.4607.ckpt'
cfg_n, ckpt = 'qxp22_5nad_aug3_ds1', 'results/xp22/qxp22_5nad_aug3_ds1/version_0/checkpoints/modelCalSLAInterpGF-epoch=331-val_loss=1.4724.ckpt'
# cfg_n, ckpt = 'qxp22_5nad_aug3_ds1', 'results/xp22/qxp22_5nad_aug3_ds1/version_0/checkpoints/modelCalSLAInterpGF-epoch=342-val_loss=1.4789.ckpt'

ose = [
    'params.files_cfg.oi_path=${file_paths.ose_oi_path}',
    'params.files_cfg.oi_var=ssh',
    'params.files_cfg.obs_mask_path=${file_paths.ose_obs_mask_path}',
    'params.files_cfg.obs_mask_var=ssh',
    'params.files_cfg.gt_path=${file_paths.ose_gt_path}',
    'params.files_cfg.gt_var=ssh',
    '+params.files_cfg.test_track=${file_paths.ose_test_along_track}',
    '+splits@datamodule=ose',
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

center_only = [
    'params.patch_weight._target_=lit_model_augstate.get_cropped_hanning_mask',
    'params.patch_weight.crop.time=1',
]
ors = [
        # *no_ds,
        # *center_only,
        # *ose,
        # *ose_sst
]
dm = get_dm(cfg_n, setup=False,
        add_overrides=[
            'file_paths=dgx_ifremer',
            *ors
        ]
)

mod = get_model(
        cfg_n,
        ckpt,
        dm=dm)

mod.__class__
cfg = get_cfg(
    cfg_n,
    overrides=[
        'file_paths=dgx_ifremer',
        *ors
        
])
# ds = xr.open_dataset(cfg.params.files_cfg.obs_mask_path)
# ds = xr.open_dataset(cfg.params.files_cfg.gt_path)
# ds  = xr.open_dataset(cfg.params.files_cfg.oi_path)
# print()
# print()
# print(xr.open_dataset(cfg.params.files_cfg.gt_path))
cfg.params.files_cfg.oi_path
print(OmegaConf.to_yaml(cfg))
lit_mod_cls = get_class(cfg.lit_mod_cls)
runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
mod = runner._get_model(ckpt_path=ckpt)
mod.patch_weight.data = torch.tensor(hydra.utils.call(cfg.params.patch_weight))
mod = runner.test(ckpt, _mod=mod, gpus=[2])
mod.test_figs['psd']

# self = mod
# animate_maps(self.x_gt, self.obs_inp, self.x_oi, self.x_rec, self.test_lon, self.test_lat, 'animation.mp4')

mod.test_xr_ds.pipe(lambda da: da.gt - da.oi).pipe(lambda da: da**2).mean().pipe(np.sqrt)

def anim(mod):
    dvars = ['pred', 'oi', 'gt']
    def sobel(da):
        dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
        dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
        return np.hypot(dx_ac, dx_al)
    # tpds = mod.test_xr_ds.pipe(sobel).isel(time=slice(None, 5, 1))
    # clim = (0, tpds[dvars].to_array().max().item())
    # cmap = 'viridis'

    tpds = mod.test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1.5)).isel(time=slice(None, 10, 1))
    # tpds = mod.test_xr_ds.map(lambda da: ndimage.laplace(da)).isel(time=slice(None, 50, 1))
    clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.01).item(), da.quantile(0.99).item()))
    # clim = (-m, m)
    cmap='RdGy'
    # cmap='RdYlBu'
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
            ]).cols(3).opts(sublabel_format="")
    return images

images = anim(mod)
hv.output(images, holomap='gif', fps=4, dpi=125)

