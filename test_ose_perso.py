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

cfg_n, ckpt = 'hgeorg/xp_oi', 'dashboard/xp_oi/version_1/checkpoints/modelCalSLAInterpGF-epoch=190-val_loss=4.8044.ckpt'

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

# ~ ose_sst = [
    # ~ 'datamodule.sst_path=${file_paths.ose_sst_path}',
    # ~ 'datamodule.sst_var=analysed_sst',
# ~ ]

# ~ no_ds = [
    # ~ 'datamodule.resize_factor=1',
# ~ ]

# ~ osse_clean = [
    # ~ 'params.files_cfg.obs_mask_var=ssh_mod',
# ~ ]

# ~ center_only = [
    # ~ 'params.patch_weight._target_=lit_model_augstate.get_cropped_hanning_mask',
    # ~ 'params.patch_weight.crop.time=1',
# ~ ]
ors = [
        # *no_ds,
        # *center_only,
        *ose,
        # *ose_sst
]

dm = get_dm(cfg_n, setup=False,
        add_overrides=[
            'file_paths=dgx_ifremer_perso',
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
        'file_paths=dgx_ifremer_perso',
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
# ~ mod = runner.test(ckpt, _mod=mod, gpus=[2])
mod = runner.test(ckpt, _mod=mod)
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

