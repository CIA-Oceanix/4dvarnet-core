import hydra
import pickle
import seaborn as sns
import xrft

from omegaconf import OmegaConf
import holoviews as hv
import holoviews.plotting.mpl  # noqa
from dvc_main import VersioningCallback
import einops
import scipy.ndimage as ndi
import contextlib
import numpy as np
from torch.nn.modules.conv import Conv2d
import zarr
import matplotlib.pyplot as plt
import xarray as xr
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import re
from hydra.utils import instantiate, get_class, call
from hydra_main import FourDVarNetHydraRunner
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import traceback
import hydra_config
from IPython.display import display, Markdown, Latex, HTML

import kornia
import math
import traceback
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
 
s = """
## TODO
Dev
- [x] Load 5nad sst model
- [x] generate 5nadsst maps
- [x] Load swath data
- [x] interpolate 5nadsst maps on maps
- [x] write coords, gt, obs, x_b functions
- [ ] launch from config
- [ ] Generate full xp data : gridded product + targets
Xps
- [ ] from oi
- [ ] from 4dvarnet 5nad no SST
- [ ] swot only pred (full field tgt + no xb input)
- [ ] OPT - from SST input



"""

pl.seed_everything(0)
display(Markdown(s))

cs = ConfigStore.instance()
cfgs = cs.list('xp')

def bst_ckpt(dirpath, glob='version_*/checkpoints/*', ckpt_fmt='.+val_loss=(.+)\.ckpt'):
    return min(Path(dirpath).glob(glob), key=lambda p: float(re.match(ckpt_fmt, str(p)).group(1)))

cfgs =  [
    # 'qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',
    # 'qxp12_aug1_dp240_5nad_cal_no_sst_ng5x3cas_l1_dp01_01',
    # 'qxp12_aug1_dp240_5nad_cal_no_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug1_dp240_5nad_cal_no_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug1_dp240_5nad_cal_no_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug1_dp240_5nad_cal_sst_ng5x3cas_l1_dp01_01',
    # 'qxp12_aug1_dp240_5nad_cal_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug1_dp240_5nad_cal_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug1_dp240_5nad_cal_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug1_dp240_5nad_map_no_sst_ng5x3cas_l1_dp01_01',
    'qxp12_aug1_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug1_dp240_5nad_map_no_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug1_dp240_5nad_map_no_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug1_dp240_5nad_map_sst_ng5x3cas_l1_dp01_01',
    'qxp12_aug1_dp240_5nad_map_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug1_dp240_5nad_map_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug1_dp240_5nad_map_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug2_dp240_5nad_cal_no_sst_ng5x3cas_l1_dp01_01',
    # 'qxp12_aug2_dp240_5nad_cal_no_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug2_dp240_5nad_cal_no_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug2_dp240_5nad_cal_no_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug2_dp240_5nad_cal_sst_ng5x3cas_l1_dp01_01',
    # 'qxp12_aug2_dp240_5nad_cal_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug2_dp240_5nad_cal_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug2_dp240_5nad_cal_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l1_dp01_01',
    # 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l2_dp025_00',
    # 'qxp12_aug2_dp240_5nad_map_sst_ng5x3cas_l1_dp01_01',
    # 'qxp12_aug2_dp240_5nad_map_sst_ng5x3cas_l1_dp025_00',
    # 'qxp12_aug2_dp240_5nad_map_sst_ng5x3cas_l2_dp01_01',
    # 'qxp12_aug2_dp240_5nad_map_sst_ng5x3cas_l2_dp025_00',
    # 'qxp13_aug2_dp240_swot_map_sst_ng5x3cas',
    # 'qxp13_aug2_dp240_swot_cal_no_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_cal_no_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_cal_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_map_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_map_no_sst_ng5x3cas',
    # 'qxp13_aug2_dp240_swot_cal_sst_ng5x3cas',

]

common_cfg = OmegaConf.create(dict(
    net_cfg=dict(
        nhidden = 128,
        depth = 3,
        kernel_size = 3,
        num_repeat = 1,
        residual = True,
        norm_type = 'lrn',
        act_type = 'silu',
        mix = True,
        mix_residual = False,
        mix_act_type = 'none',
        mix_norm_type = 'none',
    ),
    lit_cfg=dict(
        lr_init=2e-3,
        wd=1e-2,
        loss_w={
            'tot':(5., 3., 3.),
            'rec':(0., 0., 0.,)
        },
    )
))

direct_obs_cfg = OmegaConf.create(dict(
    fourdvar_cfg='qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',
    pred_var='pred',
    fourdvar_mod_ckpt=str(bst_ckpt('results/xpfeb_train/newaug_5nad_1.0_map_sst')),
    cal_mod_ckpt=str(next(Path(f'lightning_logs/baseline_obs_direct/checkpoints').glob('epoch*.ckpt'), None)),
    swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_xb=tuple(),
        sigmas_gt=(0,),
        gt_var='ssh_model',
        ref_var='pred',
        xb_var='zeros',
    ),
))

fivenad_sst_res_cfg = OmegaConf.create(dict(
    fourdvar_cfg='qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',
    pred_var='pred',
    fourdvar_mod_ckpt=str(bst_ckpt('results/xpfeb_train/newaug_5nad_1.0_map_sst')),
    cal_mod_ckpt=str(next(Path(f'lightning_logs/baseline_5nad_sst_res/checkpoints').glob('epoch*.ckpt'), None)),
    swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_gt=(0,),
        gt_var='gt_res',
        ref_var='ref_res',
        xb_var='pred',
    ),
))

oi_res_cfg = OmegaConf.create(dict(
    fourdvar_cfg='qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',
    pred_var='pred',
    fourdvar_mod_ckpt=str(bst_ckpt('results/xpfeb_train/newaug_5nad_1.0_map_sst')),
    cal_mod_ckpt=str(next(Path(f'lightning_logs/baseline_duacs4nad_res/checkpoints').glob('epoch*.ckpt'), None)),
    swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_gt=(0,),
        gt_var='gt_res',
        ref_var='ref_res',
        xb_var='oi',
    ),
))

xp_no_sst = 'qxp12_aug1_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00'
no_sst1_cfg = OmegaConf.create(dict(
    fourdvar_cfg=xp_no_sst,
    pred_var=xp_no_sst,
    fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp12/{xp_no_sst}')),
    cal_mod_ckpt=str(next(Path(f'lightning_logs/1_{xp_no_sst}/version_0/checkpoints').glob('epoch*.ckpt'), None)),
    swath_ds_cfg=dict(
        sigmas_gt=(0,),
        gt_var='gt_res',
        ref_var='ref_res',
        xb_var='pred',
    ),
))

xp_sst = 'qxp12_aug1_dp240_5nad_map_sst_ng5x3cas_l1_dp025_00'
sst1_cfg = OmegaConf.create(dict(
    fourdvar_cfg=xp_sst,
    pred_var=xp_sst,
    fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp12/{xp_sst}')),
    cal_mod_ckpt=str(next(Path(f'lightning_logs/1_{xp_sst}/version_0/checkpoints').glob('epoch*.ckpt'), None)),
    swath_ds_cfg=dict(
        sigmas_gt=(0,),
        gt_var='gt_res',
        ref_var='ref_res',
        xb_var='pred',
    ),
))

size_overrides_cfg = dict(
    pp10x2=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*2 for i in range(10)]),
        sigmas_xb=(0,*[(i+1)*2 for i in range(10)]),
    ))),
    pp20x2=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*2 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*2 for i in range(20)]),
    ))),
    pp40x2=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*2 for i in range(40)]),
        sigmas_xb=(0,*[(i+1)*2 for i in range(40)]),
    ))),
    pp10x5=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*5 for i in range(10)]),
        sigmas_xb=(0,*[(i+1)*5 for i in range(10)]),
    ))),
    pp20x5=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*5 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*5 for i in range(20)]),
    ))),
    pp20x8=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
    ))),
    pp40x8=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(40)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(40)]),
    ))),
    ppcustom=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0, 2, 4, 8, 12, 17, 22, 28, 34, 41, 48, 56, 64, 73, 82, 92, 102, 112, 122, 132, 142),
        sigmas_xb=(0, 2, 4, 8, 12, 17, 22, 28, 34, 41, 48, 56, 64, 73, 82, 92, 102, 112, 122, 132, 142),
    ))),
    no_pp=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,),
        sigmas_xb=(0,),
    ))),
)
overrides_cfg = dict(
    direct_obs=OmegaConf.create(
        dict(swath_ds_cfg=dict(
            sigmas_xb=tuple(),
            gt_var='ssh_model',
            ref_var='pred',
            xb_var='zeros',
        ))),
    duacs_base=OmegaConf.create(
        dict(
            swath_ds_cfg=dict(
            xb_var='oi',
        ))),
    non_residual=OmegaConf.create(dict(net_cfg=dict(residual=False,))),
    no_norm=OmegaConf.create(dict(net_cfg=dict(norm_type='none',))),
    no_mix=OmegaConf.create(dict(net_cfg=dict(mix='False',))),
    relu_act=OmegaConf.create(dict(net_cfg=dict(act_type='relu',))),
    swot_sst_model1=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug2_dp240_swot_map_sst_ng5x3cas')),
    )),
    swot_sst_model2=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug1_dp240_swot_cal_sst_ng5x3cas')),
    )),
    swot_sst_model3=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug2_dp240_swot_cal_sst_ng5x3cas')),
    )),
    swot_no_sst_model1=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug2_dp240_swot_map_no_sst_ng5x3cas')),
    )),
    swot_no_sst_model2=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug1_dp240_swot_cal_no_sst_ng5x3cas')),
    )),
    swot_no_sst_model3=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug2_dp240_swot_cal_no_sst_ng5x3cas')),
    )),
    swot_no_sst_model4=OmegaConf.create(dict(
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp13/qxp13_aug2_dp240_swot_map_no_sst_ng5x3cas')),
    )),
)

    # 'qxp13_aug2_dp240_swot_map_sst_ng5x3cas',
    # 'qxp13_aug2_dp240_swot_cal_no_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_cal_no_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_cal_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_map_sst_ng5x3cas',
    # 'qxp13_aug1_dp240_swot_map_no_sst_ng5x3cas',
    # 'qxp13_aug2_dp240_swot_cal_sst_ng5x3cas',
# Siren imp

# import importlib
# import new_dataloading
# import hydra_main
# importlib.reload(new_dataloading)
# importlib.reload(hydra_main)
# Generate data

def to_swath_data(grid_ds):

        slice_args = dict(
                # time_min='2012-10-01', 
                time_min= pd.to_datetime(np.min(grid_ds['time']).values).date(),
                # time_max='2012-10-30',
                time_max= pd.to_datetime(np.max(grid_ds['time']).values).date(),
                lat_min=grid_ds['lat'].min().item(),
                lat_max=grid_ds['lat'].max().item(),
                lon_min=grid_ds['lon'].min().item() + 360,
                lon_max=grid_ds['lon'].max().item() + 360,
        )

        swath_data = get_swot_slice('../sla-data-registry/sensor_zarr/zarr/new_swot', **slice_args, drop_vars=[
         'bd_err', 'karin_noise', 'karin_err', 'phase_err', 'roll_err','timing_err',
         # 'lat', 'lat_nadir', 'lon', 'lon_nadir',
         'model_index',
          # 'ssh_model',
          # 'syst_error_uncalibrated', 'wet_tropo_res',
         # 'x_ac', 'x_al'
         ])

        # grid_ds.obs_gt.isel(time=6).plot()
        # dl = dm.train_dataloader()
        # dl = dm.val_dataloader()
        # delta_ds = dl.dataset.datasets[0].gt_ds.ds.ssh - grid_ds.gt
        # dl.dataset.datasets[0].gt_ds.ds.ssh.isel(time=3).plot() 
        # grid_ds.gt.isel(time=3).plot() 
        # delta_ds.isel(time=3).plot()
        swathed_grid = grid_ds.interp(
            time=swath_data.time.broadcast_like(swath_data.ssh_model),
            lat=swath_data.lat.broadcast_like(swath_data.ssh_model),
            lon=swath_data.lon.broadcast_like(swath_data.ssh_model) - 360,
        )
        swath_data = (
            swath_data.assign({v: (swath_data.ssh_model.dims, swathed_grid[v].data)for v in swathed_grid} )
            
        )
        return swath_data

def generate_swath_sst_data():
    try:

        cfg_n = next( c  for c  in cfgs if 'no_sst' not in c)
        overrides = ['+datamodule.dl_kwargs.shuffle=False']
        dm = get_dm(cfg_n,add_overrides=overrides)

        grid_ds = dm.val_ds.datasets[0].sst_ds.ds
        swathed_sst = to_swath_data(grid_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.sst).all('nC')))
        swathed_sst.to_netcdf('data/val_swath_sst.nc')

        grid_ds = dm.train_ds.datasets[0].sst_ds.ds
        swathed_sst = to_swath_data(grid_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.sst).all('nC')))
        swathed_sst.to_netcdf('data/train_swath_sst.nc')

        grid_ds = dm.test_ds.datasets[0].sst_ds.ds
        swathed_sst = to_swath_data(grid_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.sst).all('nC')))
        swathed_sst.to_netcdf('data/test_swath_sst.nc')

        swathed_sst.sst.isel(time=slice(None, 1000)).T.reset_index('time').plot(figsize=(15,3))
        swathed_sst.ssh_model.isel(time=slice(None, 1000)).T.reset_index('time').plot(figsize=(15,3))
        
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def generate_data():
    try:

        preds_data = {}
        cs.list('xp')
        for cfg_n in cfgs:
            overrides = ['+datamodule.dl_kwargs.shuffle=False']
            cfg = get_cfg(cfg_n, overrides=overrides)
            dm = get_dm(cfg_n,add_overrides=overrides)
            model = get_model(cfg_n, bst_ckpt(f'results/xp12/{cfg_n}'), dm=dm, add_overrides=overrides)

            trainer = pl.Trainer(gpus=[5])
            # trainer.test(model, dm.train_dataloader())
            trainer.test(model, dm.test_dataloader())
            # trainer.test(model, dm.val_dataloader())
            swath_data = to_swath_data(model.test_xr_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.oi).all('nC')))
            # swath_data.to_netcdf('data/swath_train_data.nc')
            preds_data[cfg_n] = swath_data.pred
            rms = lambda da: np.sqrt(np.mean(da**2))
            swath_data.gt.std()
            swath_data.ssh_model.std()
            print(f' \n\n ############################# {cfg_n} ##########################################')
            print(f'{rms(swath_data.ssh_model - swath_data.oi).item()=:.2e} m')
            print(f'{rms(swath_data.ssh_model - swath_data.pred).item()=:.2e} m')
            print(f'{rms(swath_data.ssh_model - swath_data.gt).item()=:.2e} m')
            print(f'  ############################# {cfg_n} ########################################## \n\n')

        for c, p in preds_data.items():
            print(f'{c} : {rms(swath_data.ssh_model - p).item()=:.2e} m')

        xr.Dataset(preds_data).to_netcdf('data/test_swath_data_xp_12.nc')
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


class SmoothSwathDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            swath_data,
            norm_stats=None,
            sigmas_obs=(0,*[(i+1)*8 for i in range(40)]),
            sigmas_xb=(0,),
            sigmas_gt=(0,),
            gt_var='ssh_model',
            ref_var='pred',
            xb_var='oi',
        ):
        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter1d(nda, axis=0, sigma=sig, order=0, mode='mirror', truncate=3.0), da)
        swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
        # swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time') > 3).cumsum())
        sw_data_w_aug = (
                swath_data
                .groupby('contiguous_chunk')
                .apply(
                    lambda g: g.assign(
                        err =  lambda _g: _g.syst_error_uncalibrated + _g.wet_tropo_res,
                    ).assign(
                        zeros = lambda _g:  xr.zeros_like(_g.err)
                    ).assign(
                        xb =  lambda _g: _g[xb_var],
                    ).assign(
                        obs = lambda _g:  _g.ssh_model + _g.err
                    ).assign(
                        obs_res = lambda _g: _g.obs - _g.xb,
                        gt_res= lambda ds: ds.ssh_model - ds.xb,
                        ref_res= lambda ds: ds.pred - ds.xb
                    ).assign(
                        **{f'obs_{sig}' : lambda _g, sig=sig: xrgf(_g.obs, sig) for sig in sigmas_obs},
                        **({} if len(sigmas_xb)==0 else {f'xb_{sig}' : lambda _g, sig=sig: xrgf(_g.xb, sig) for sig in sigmas_xb}),
                        **{f'gt_{sig}' : lambda _g, sig=sig: xrgf(_g[gt_var], sig) for sig in sigmas_gt},
                    )
                )
        )

        sw_res_data = sw_data_w_aug.assign(
                **{
                    f'dobs_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'obs_{sig1}'] - ds[f'obs_{sig2}']
                    for sig1, sig2 in zip(sigmas_obs[:-1], sigmas_obs[1:])
                },
                **({} if len(sigmas_xb)==0 else {
                    f'dxb_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'xb_{sig1}'] - ds[f'xb_{sig2}']
                    for sig1, sig2 in zip(sigmas_xb[:-1], sigmas_xb[1:])
                }),
                **{
                    f'dgt_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'gt_{sig1}'] - ds[f'gt_{sig2}']
                    for sig1, sig2 in zip(sigmas_gt[:-1], sigmas_gt[1:])
                },
        )

        pp_vars = (
                [f'dobs_{sig2}_{sig1}'for sig1, sig2 in zip(sigmas_obs[:-1], sigmas_obs[1:])] 
                + ([f'obs_{sigmas_obs[-1]}'] if len(sigmas_obs)>0 else [])
                + (([f'dxb_{sig2}_{sig1}' for sig1, sig2 in zip(sigmas_xb[:-1], sigmas_xb[1:])]
                    + [f'xb_{sigmas_xb[-1]}']) if len(sigmas_xb)>0 else [])
        )
        gt_vars = (
                [f'dgt_{sig2}_{sig1}'for sig1, sig2 in zip(sigmas_gt[:-1], sigmas_gt[1:])] + [f'gt_{sigmas_gt[-1]}']
        )

        # gt_vars = ['gt_res']
        # ref_var = 'zeros'

        
        # for v in pp_vars:
        #     p(sw_res_data.isel(time=sw_res_data.contiguous_chunk==2)[v])
        all_vars = gt_vars + pp_vars
        mean, std =  norm_stats  if norm_stats is not None else (
                sw_res_data[all_vars].mean(),
                sw_res_data[all_vars].std(),
        )
        # mean, std =train_ds.stats
        # norm_stats=train_ds.stats
        # print(mean)
        pp_ds = ((sw_res_data[all_vars] - mean) / std).assign(contiguous_chunk=sw_res_data.contiguous_chunk).astype(np.float32)

        min_timestep = 300
        self.stats  = mean, std
        self.chunks = list(
                pp_ds.groupby('contiguous_chunk').count()
                .isel(nC=0).pipe(lambda ds: ds.isel(contiguous_chunk=ds[pp_vars[0]] > min_timestep))
                .contiguous_chunk.values
        )

        self.pp_vars = pp_vars 
        self.gt_vars = gt_vars
        self.gt_var = gt_var
        self.ref_var = ref_var
        self.pp_ds = pp_ds
        self.raw_ds = sw_data_w_aug

        self.return_coords = False

    def __len__(self):
        return len(self.chunks)

    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    def __getitem__(self, idx):
        c = self.chunks[idx]
        pp_item_ds = self.pp_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))
        raw_item_ds = self.raw_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))

        if self.return_coords:
            return raw_item_ds
        return (
            pp_item_ds[self.pp_vars].to_array().data,
            pp_item_ds[self.gt_vars].to_array().data,
            raw_item_ds[[self.gt_var]].to_array().data,
            raw_item_ds[[self.ref_var]].to_array().data
        )

def get_same_pad(h, w, kh, kw, s):
    # The total padding applied along the height and width is computed as:
    if (h % s[0] == 0):
        pad_along_height = max(kh - s[0], 0)
    else:
        pad_along_height = max(kh - (h % s[0]), 0)
    if w % s[1] == 0:
        pad_along_width = max(kw - s[1], 0)
    else:
        pad_along_width = max(kw - (w % s[1]), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return {'left': pad_left, 'right': pad_right, 'top': pad_top, 'bottom': pad_bottom}

    # %% models


class ConvSamePad(torch.nn.Module):
    def __init__(self, apply_per_side=True, *args, **kwargs):
        super().__init__()
        self.apply_per_side = apply_per_side
        self.conv = torch.nn.Conv2d(*args, **kwargs)

    def _forward(self, inp):
        inp_shape = einops.parse_shape(inp, 'bs inc w h')
        kernel_shape = einops.parse_shape(self.conv.weight, 'inc outc w h')
        same_pad = get_same_pad(
            inp_shape['h'],
            inp_shape['w'],
            kernel_shape['h'],
            kernel_shape['w'],
            self.conv.stride,
        )
        return self.conv(F.pad(inp, (same_pad['top'], same_pad['bottom'], same_pad['left'], same_pad['right']), mode='reflect'))

    def forward(self, x):
        if self.apply_per_side:
            sizes = einops.parse_shape(x, 'b c time nc')
            side_limit = sizes['nc'] // 2
            return einops.rearrange(
                [
                    self._forward(x[..., :side_limit]),
                    self._forward(x[..., side_limit:]),
                ],
                'sides b c time hnc -> b c time (hnc sides)'
            )

        return self._forward(x)

def build_net(
    in_channels,
    out_channels,
    nhidden = 128,
    depth = 3,
    kernel_size = 3,
    num_repeat = 1,
    residual = True,
    norm_type = 'lrn',
    act_type = 'silu',
    mix = True,
    mix_residual = False,
    mix_act_type = 'none',
    mix_norm_type = 'none',
):

    def norm(norm_type='bn', nh=nhidden):
        if norm_type=='none':
            return nn.Identity()
        elif norm_type=='bn':
            return nn.BatchNorm2d(num_features=nh)
        elif norm_type=='in':
            return nn.InstanceNorm2d(num_features=nh)
        elif norm_type=='lrn':
            return nn.LocalResponseNorm(size=5)
        else:
            assert False, 'Should not be here'

    def act(act_type='relu'):
        if act_type=='none':
            return nn.Identity()
        elif act_type=='relu':
            return nn.ReLU()
        elif act_type=='silu':
            return nn.SiLU()
        elif act_type=='gelu':
            return nn.GELU()
        else: 
            assert False, 'Should not be here'


    class ResidualBlock(nn.Module):
        def __init__(self, net,  res=True):
            super().__init__()
            self.net = net
            self.res = res

        def forward(self, x):
          if not self.res:
              return self.net(x)
          return x + self.net(x)
    from einops.layers.torch import Rearrange, Reduce

    def mixer(b=True, res=False):
         return ResidualBlock(
             nn.Sequential(
               Rearrange('b c t nC -> b nC t c'),
               ConvSamePad(in_channels=52, out_channels=52, kernel_size=1, apply_per_side=False),
               norm(mix_norm_type, nh=52),
               act(act_type=mix_act_type),
               Rearrange('b nC t c -> b c t nC'),
            ),
         res=res) if b else nn.Identity()

    
    inner_net = nn.Sequential(
        *[ nn.Sequential(
            ResidualBlock(
                nn.Sequential(
                    ConvSamePad(in_channels=nhidden,out_channels=nhidden, kernel_size=kernel_size),
                    norm(norm_type),
                    act(act_type=act_type),
                ), res=residual),
                mixer(mix, res=mix_residual),
        )
        for _ in range(depth) ],
    )
    net = nn.Sequential(
            ConvSamePad(in_channels=in_channels,out_channels=nhidden, kernel_size=1),
            norm(norm_type=norm_type),
            act(act_type=act_type),
            nn.Sequential(
                *[inner_net for _ in range(num_repeat)]
            ),
            ConvSamePad(in_channels=nhidden, out_channels=out_channels, kernel_size=1),
    )
    return net

class LitDirectCNN(pl.LightningModule):
            def __init__(
                    self,
                    net,
                    gt_var_stats,
                    lr_init=1e-3,
                    wd=1e-4,
                    loss_w={'tot':(.1, .1, .1), 'rec':(1., 1., 1.,)},
                    loss_budget_gt_vars=100,
                ):
                super().__init__()
                self.net = net
                self.lr_init = lr_init
                self.wd = wd
                self.loss_budget_gt_vars = loss_budget_gt_vars
                self.loss_w = loss_w
                self.gt_means = nn.Parameter(torch.from_numpy(gt_var_stats[0])[None, :, None, None], requires_grad=False)
                self.gt_stds = nn.Parameter(torch.from_numpy(gt_var_stats[1])[None, :, None, None], requires_grad=False)
                self.save_hyperparameters()

            def forward(self, batch):
                x, *_ = batch 
                return self.net(x)

            def loss(self, t1, t2):
                rmse = ((t1 -t2)**2).mean().sqrt()

                def sob(t):
                    if len(t.shape) == 4:
                        # return kornia.filters.sobel(rearrange(t, 'b d1 d2 c -> b c d1 d2'))
                        return kornia.filters.sobel(t)
                    elif len(t.shape) == 3:
                        return kornia.filters.sobel(rearrange(t, 'b d1 d2 -> b () d1 d2'))
                    else:
                        assert False, 'Should not be here'

                def lap(t):
                    if len(t.shape) == 4:
                        # return kornia.filters.laplacian(rearrange(t, 'b d1 d2 c -> b c d1 d2'), kernel_size=3)
                        return kornia.filters.laplacian(t, kernel_size=3)
                    elif len(t.shape) == 3:
                        return kornia.filters.laplacian(rearrange(t, 'b d1 d2 -> b () d1 d2'), kernel_size=3)
                    else:
                        assert False, 'Should not be here'

                rmse_grad = ((sob(t1) - sob(t2))**2).mean().sqrt()
                rmse_lap = ((lap(t1) - lap(t2))**2).mean().sqrt()

                return rmse, rmse_grad, rmse_lap

            def process_batch(self, batch, phase='val'):
                _, y, raw_gt, raw_ref = batch 
                out = self.forward(batch)
                losses = {}
                losses['err_tot'], losses['g_err_tot'], losses['l_err_tot'] = self.loss(out, y)

                rec_out = (out * self.gt_stds + self.gt_means).sum(dim=1)
                losses['err_rec'], losses['g_err_rec'], losses['l_err_rec'] = self.loss(rec_out, raw_gt)

                for ln, l in losses.items():
                    self.log(f'{phase}_{ln}', l)

                loss_ref, g_loss_ref, l_loss_ref= self.loss(raw_ref, raw_gt)
                self.log(f'{phase}_imp_mse', losses['err_rec'] / loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_grad_mse', losses['g_err_rec'] / g_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_lap_mse', losses['l_err_rec'] / l_loss_ref, prog_bar=True, on_step=False, on_epoch=True)

                loss = (
                    self.loss_w['tot'][0] * losses['err_tot']
                    + self.loss_w['tot'][1] * losses['g_err_tot']
                    + self.loss_w['tot'][2] * losses['l_err_tot']
                    + self.loss_w['rec'][0] * losses['err_rec']
                    + self.loss_w['rec'][1] * losses['g_err_rec']
                    + self.loss_w['rec'][2] * losses['l_err_rec']
                )
                self.log(f'{phase}_loss', loss, prog_bar=False)
                return loss
                
            def training_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='train')

            def validation_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='val')

            def predict_step(self, batch, batch_idx):
                out = self.forward(batch)
                # print(f'{out.isnan().sum()=}')

                rec_out = (out * self.gt_stds + self.gt_means).sum(dim=1)
                return rec_out.cpu().numpy()


            def configure_optimizers(self):
                # opt = torch.optim.AdamW(self.parameters(), lr=self.lr_init, weight_decay=self.wd)
                opt = torch.optim.Adam(
                        [{'params': self.parameters(), 'initial_lr': self.lr_init}],
                        lr=self.lr_init, weight_decay=self.wd)
                # opt = torch.optim.SGD(self.parameters(), lr=self.lr_init)
                return {
                    'optimizer': opt,
                    'lr_scheduler':
                    # torch.optim.lr_scheduler.ReduceLROnPlateau(
                    #     opt, verbose=True, factor=0.5, min_lr=1e-6, cooldown=5, patience=5,
                    # ),
                    # torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=5e-5, T_max=20),
                    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                    #     eta_min=5e-5, T_0=15, T_mult=2, last_epoch=-1),
                    torch.optim.lr_scheduler.CyclicLR(
                        opt, base_lr=5e-5, max_lr=5e-3,  step_size_up=25, step_size_down=25, cycle_momentum=False, mode='triangular2'),
                    'monitor': 'val_loss'
                }

def generate_cal_xrds(ds, lit_mod, trainer, var_name='cal'):
    # ds, lit_mod, trainer, var_name=cal_ds, cal_mod, trainer, 'cal'
    pred_dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    predictions = [p for pred in trainer.predict(lit_mod, pred_dl) for p in pred]
    with ds.get_coords():
        coords = [
           ds[i]
           for i in range(len(ds))
        ]

    return xr.concat(
            [
                coord.assign({var_name: lambda ds: (ds.ssh_model.dims, pred + coord.xb.values)})
                for pred, coord in zip(predictions, coords)
            ], dim='time'
    )


def register_configs():
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from pathlib import Path
    from itertools import product

    cs = ConfigStore.instance()

    XP_NUM = 14 
    cfgs = []
    simple_overrides = [overrides_cfg[o] for o in ['no_norm', 'no_mix']]
    basic_overrides = [overrides_cfg[o] for o in ['no_norm', 'no_mix', 'relu_act']]
    for xp_name, cfg in [
            # ('base_duacs_simple',  [sst1_cfg, size_overrides_cfg['pp20x8'], overrides_cfg['duacs_base'],  *simple_overrides]),
            # ('direct_obs_simple',  [sst1_cfg, size_overrides_cfg['pp20x8'], overrides_cfg['direct_obs'],  *simple_overrides]),
            # ('base_no_sst_simple',  [no_sst1_cfg, size_overrides_cfg['pp20x8'],  *simple_overrides]),
            # ('base_sst_simple',  [sst1_cfg, size_overrides_cfg['pp20x8'], *simple_overrides]),
            # ('no_sst_no_mix',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['no_mix']]),
            # ('no_sst_relu',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['relu_act']]),
            # ('no_sst_non_residual',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['non_residual']]),
            # ('no_sst_no_pp',  [no_sst1_cfg, size_overrides_cfg['no_pp'], *basic_overrides]),
            # ('no_sst_small_pp',  [no_sst1_cfg, size_overrides_cfg['pp10x5'], *basic_overrides]),
            # ('base_duacs',  [sst1_cfg, size_overrides_cfg['pp20x8'], overrides_cfg['duacs_base'], *basic_overrides]),
            # ('direct_obs',  [sst1_cfg, size_overrides_cfg['pp20x8'], overrides_cfg['direct_obs'], *basic_overrides]),
            # ('base_no_sst_swot1',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_no_sst_model1']]),
            # ('base_no_sst_swot3',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_no_sst_model3']]),
            # ('base_no_sst_swot4',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_no_sst_model4']]),
            # ('base_no_sst_swot2',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_no_sst_model2']]),
            # ('base_sst',  [sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides]),
            ('base_no_sst',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides]),
            # ('base_sst_swot1',  [sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_sst_model1']]),
            # ('base_sst_swot2',  [sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_sst_model2']]),
            # ('base_sst_swot3',  [sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides, overrides_cfg['swot_sst_model3']]),
            # ('base_duacs_big',  [sst1_cfg, size_overrides_cfg['pp40x8'], overrides_cfg['duacs_base'], *basic_overrides]),
            # ('direct_obs_big',  [sst1_cfg, size_overrides_cfg['pp40x8'], overrides_cfg['direct_obs'], *basic_overrides]),
            # ('base_no_sst_big',  [no_sst1_cfg, size_overrides_cfg['pp40x8'], *basic_overrides]),
            # ('base_sst_big',  [sst1_cfg, size_overrides_cfg['pp40x8'], *basic_overrides]),
            # *[(f'sst_{pp}',  [sst1_cfg, size_overrides_cfg[pp], *basic_overrides]) for pp in size_overrides_cfg],
            # # 15
            # *[(f'no_sst_{pp}',  [no_sst1_cfg, size_overrides_cfg[pp], *basic_overrides]) for pp in size_overrides_cfg],
            # ('base_no_sst',  [no_sst1_cfg, size_overrides_cfg['pp20x8'], *basic_overrides]),
        ]:
        cfgs.append(f'{XP_NUM}_' + xp_name)
        cs.store(name=cfgs[-1], node=OmegaConf.merge(common_cfg, *cfg), group='xp', package='_global_')
    return XP_NUM, cfgs


def new_train_cal():
    try:

        XP_NUM, cfgs = register_configs()
        for cfgn in cfgs:
            with hydra.initialize():
                cfg = hydra.compose(overrides=[f'+xp={cfgn}'])
            print(OmegaConf.to_yaml(cfg))

            # Get swath model
            swath_data = xr.open_dataset('data/swath_train_data.nc')
            if cfg.pred_var not in swath_data:
                preds_data = xr.open_dataset('data/train_swath_data_xp_12.nc') 
                tr_swath_data = (
                        swath_data.assign(
                            pred=preds_data[cfg.pred_var]
                                .pipe(lambda da: (da.dims, da.values))
                        )
                )
                val_preds_data = xr.open_dataset('data/val_swath_data_xp_12.nc') 
                val_swath_data = (
                    xr.open_dataset('data/swath_val_data.nc')
                    .assign(
                        pred=val_preds_data[cfg.pred_var]
                            .pipe(lambda da: (da.dims, da.values))
                    )
                )
            else:
                tr_swath_data = swath_data
                val_swath_data = xr.open_dataset('data/swath_val_data.nc')

            train_ds = SmoothSwathDataset(tr_swath_data, **cfg.swath_ds_cfg) 
            val_ds = SmoothSwathDataset(val_swath_data, **cfg.swath_ds_cfg, norm_stats=train_ds.stats) 

            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=3)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3)
            
            net = build_net(
                    in_channels=len(train_ds.pp_vars),
                    out_channels=len(train_ds.gt_vars),
                    **cfg.net_cfg
            )
            cal_mod = LitDirectCNN(
                    net,
                    gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
                    **cfg.lit_cfg
                )
            logger = pl.loggers.TensorBoardLogger('lightning_logs', name=cfgn, version='')
            trainer = pl.Trainer(
                gpus=[5],
                logger=logger,
                callbacks=[
                    callbacks.LearningRateMonitor(),
                    callbacks.RichProgressBar(),
                    callbacks.ModelCheckpoint(monitor='val_loss', save_last=True),
                    callbacks.StochasticWeightAveraging(),
                    callbacks.GradientAccumulationScheduler({1: 4, 10: 8, 15: 16, 20: 32, 30: 64}),
                    VersioningCallback()
                ],
                log_every_n_steps=10,
                max_epochs=250,
            )

            print(pl.utilities.model_summary.summarize(cal_mod, max_depth=3))
            trainer.fit(cal_mod,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl
            )
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def make_report():
    try:
        XP_NUM = 14
        grid_metrics = pd.read_csv(f'{XP_NUM}_grid_chain_metrics.csv')
        sw_metrics = pd.read_csv(f'{XP_NUM}_sw_chain_metrics.csv')
        with open(f'{XP_NUM}_figs.pk', 'rb') as f:
            figs = pickle.load(f)
        data = (
                grid_metrics
                .set_index(['xp', 'iter'])
                .drop('Unnamed: 0', axis=1)
                .join(sw_metrics.set_index(['xp', 'iter'])
                .assign(
                    rmse_improvement= lambda df: 1 - df.rmse / df.rmse_pred,
                    grad_rmse_improvement= lambda df: 1 - df.grad_rmse / df.grad_rmse_pred
                )
                .drop('Unnamed: 0', axis=1), rsuffix='_sw')
                .join(pd.DataFrame(figs).set_index(['xp', 'iter']))
                .reset_index().loc[lambda df: df.iter == 0]
                .set_index('xp')
        )

        metrics = [
                'rmse', 'rmse_improvement', 'grad_rmse', 'grad_rmse_improvement',
                'spat_res_mean', 'spat_res_std'
        ]

        grid_xp = lambda df: df.loc[[f'{XP_NUM}_base_duacs', f'{XP_NUM}_direct_obs', f'{XP_NUM}_base_no_sst', f'{XP_NUM}_base_sst']]
        ablation_xp = lambda df: df.loc[[f'{XP_NUM}_base_no_sst', f'{XP_NUM}_no_sst_no_pp', f'{XP_NUM}_no_sst_non_residual']]
        # data.pipe(ablation_xp)[metrics].plot(kind='bar', subplots=True, figsize=(8, 10), layout=(2, 3), legend=False)
        # data.pipe(grid_xp)[metrics].plot(kind='bar', subplots=True, figsize=(8, 10), layout=(2, 3), legend=False)
        with  open('report/violin_all_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].violin_all.savefig(f)

        with  open('report/violin_diff_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].violin_diff.savefig(f)

        with  open('report/grad_ssh_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].ssh.savefig(f)

        with  open('report/err_no_sst.png', 'wb') as f:
            data.loc[f'{XP_NUM}_base_no_sst'].err.savefig(f)


        report = f"""

        ## Results 4dvarnet 5 nadir no sst + CalCNN 
        **Ssh gradients**
        ![](report/grad_ssh_no_sst.png)

        **Residual error after calibration**
        ![](report/err_no_sst.png)

        {data.loc[f'{XP_NUM}_base_no_sst'][metrics].T.to_markdown()}
        
        **Spatial resolution on swath before**
        ![](report/violin_all_no_sst.png)
        ![](report/violin_diff_no_sst.png)

        ## Metrics different grids
        {data.pipe(grid_xp)[metrics].to_markdown()}

        ## Ablation
        {data.pipe(ablation_xp)[metrics].to_markdown()}


        """
        display(Markdown(report))
        print(report)

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()


def to_grid(swath_data, tgt_grid):
    import pyinterp
    ds = swath_data[['pred', 'lat', 'lon', 'time']]
    binning = pyinterp.Binning2D(pyinterp.Axis(tgt_grid.lon.values), pyinterp.Axis(tgt_grid.lat.values))
    grid_day_dses = []

    for t in tgt_grid.time:
        binning.clear()
        tds = ds.isel(time=pd.to_datetime(ds.time.values).date == pd.to_datetime(t.values).date())

        values = np.ravel(tds.pred.values)
        lons = np.ravel(tds.lon.values) - 360
        lats = np.ravel(tds.lat.values)
        msk = np.isfinite(values)
        binning.push(lons[msk], lats[msk], values[msk])
        gridded =  (('time', 'lat', 'lon'), binning.variable('mean').T[None, ...])
        grid_day_dses.append(
           xr.Dataset(
               {'gridded': gridded},
               {'time': [t.values], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
            ).astype('float32', casting='same_kind')
        )
    tgt_ds = xr.concat(grid_day_dses, dim='time')
    return tgt_ds

def regrid_preds():
    try:
        overrides = ['+datamodule.dl_kwargs.shuffle=False']
        cfg_n =  'qxp12_aug2_dp240_5nad_map_sst_ng5x3cas_l1_dp025_00'
        dm = get_dm(cfg_n,add_overrides=overrides)

        preds_data = xr.open_dataset('data/train_swath_data_xp_12.nc') 
        val_preds_data = xr.open_dataset('data/val_swath_data_xp_12.nc') 
        swath_data = (
            xr.open_dataset('data/swath_train_data.nc')
            .assign(pred=preds_data[cfg_n])
        )
        swath_data = (
            xr.open_dataset('data/swath_val_data.nc')
            .assign(pred=val_preds_data[cfg_n])
        )
        tgt_grid = dm.val_ds.datasets[0].obs_mask_ds.ds
        tgt_ds = to_grid(swath_data, tgt_grid)
        plt.imshow(tgt_ds.gridded.isel(time=7))
        obs_ds = dm.val_ds.datasets[0].obs_mask_ds.ds
        obs_ds.five_nadirs.isel(time=5)
        tgt_ds.gridded.isel(time=5).plot()
        obs_ds
        tgt_ds
        tgt_ds.isel(time=5).gridded.reindex_like(obs_ds, method='nearest', tolerance=1e-6).plot()
        (
                xr.merge([obs_ds[['five_nadirs']], tgt_ds.reindex_like(obs_ds, method='nearest', tolerance=1e-6)])
                .to_array()
                .mean('variable')
        ).isel(time=5).plot(figsize=(8,8))

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def test_chain_4dvar_cal():
    try:
        grid_metrics = []
        swath_metrics = []
        figs = []
        rms = lambda da: np.sqrt((da**2).mean())

        XP_NUM, cfgs = register_configs()
        for cfgn in cfgs:
            print(cfgn)
            with hydra.initialize():
                cfg = hydra.compose(overrides=[
                    f'+xp={cfgn}',
                    '++cal_mod_ckpt=' + str(((Path('lightning_logs') / cfgn / 'checkpoints' / 'last.ckpt')))
                    # '++cal_mod_ckpt=' + str(((Path('lightning_logs') / '12_base_no_sst' / 'checkpoints' / 'last.ckpt')))
                ])

            # First estim from 4dvar on grid        
            overrides = ['+datamodule.dl_kwargs.shuffle=False']
            fourdvar_dm = get_dm(cfg.fourdvar_cfg, add_overrides=overrides)

            # Get grid model
            fourdvar_model = get_model(cfg.fourdvar_cfg, cfg.fourdvar_mod_ckpt, dm=fourdvar_dm, add_overrides=overrides)


            # Get swath model
            swath_data = xr.open_dataset('data/swath_train_data.nc')
            preds_data = xr.open_dataset('data/train_swath_data_xp_12.nc') 

            tr_swath_data = (
                swath_data.assign(pred=preds_data[cfg.pred_var])
            )
            train_ds = SmoothSwathDataset(tr_swath_data, **cfg.swath_ds_cfg) 

            net = build_net(
                    in_channels=len(train_ds.pp_vars),
                    out_channels=len(train_ds.gt_vars),
                    **cfg.net_cfg
            )
            cal_mod = LitDirectCNN(
                    net,
                    gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
                    **cfg.lit_cfg
                )

            trainer = pl.Trainer(gpus=[5], logger=False)

            for niter in range(1):
                grid_metrics = grid_metrics + [{
                        'xp': cfgn,
                        'iter': niter,
                        **trainer.test(fourdvar_model, fourdvar_dm.test_dataloader())[0]
                }]
                print(pd.DataFrame(grid_metrics).to_markdown())

                # Convert grid estim to swath
                sw_data = to_swath_data(fourdvar_model.test_xr_ds).pipe(lambda ds: ds.isel(time=np.isfinite(ds.pred).all('nC')))

                # Convert grid estim to swath
                cal_ds = SmoothSwathDataset(sw_data, norm_stats=train_ds.stats, **cfg.swath_ds_cfg) 

                # Estimate on swath using cal model
                print(cal_mod.load_state_dict(torch.load(cfg.cal_mod_ckpt)['state_dict']))
                cal_data = generate_cal_xrds(cal_ds, cal_mod, trainer)[list(sw_data) + ['cal', 'contiguous_chunk']]

                def sobel(da):
                    dx_ac = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 0), da) /2
                    dx_al = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 1), da) /2
                    return np.hypot(dx_ac, dx_al)
               
                # Grid calibrated swath estimation
                obs_ds = fourdvar_dm.test_ds.datasets[0].obs_mask_ds.ds
                tgt_ds = to_grid(cal_data, obs_ds)

                # Generate new obs on grid with calibrated data
                new_obs = (
                        xr.merge([obs_ds[['five_nadirs']],
                            # obs_ds[['swot_no_noise']],
                            tgt_ds.reindex_like(obs_ds, method='nearest', tolerance=1e-6)
                            ]
                        )
                        .to_array()
                        .mean('variable')
                )

                # TODO: Do inference 
                fourdvar_dm.test_ds.datasets[0].obs_mask_ds.ds = obs_ds.assign(cal=new_obs)
                fourdvar_dm.test_ds.datasets[0].obs_mask_ds.var = 'cal'
            

                add_inter_sw = lambda ds:(
                            ds
                        .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
                        .swap_dims(nC='x_ac')
                        .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
                )
                v = 'cal'
                chunk=2
                fig_errs = (
                        cal_data.pipe(add_inter_sw).pipe(lambda d: d.isel(time=d.contiguous_chunk==chunk))
                        .assign(err=lambda d: d[v] - d.ssh_model)
                        .assign(pred_err=lambda d: d.pred - d.ssh_model)
                        [['err', 'pred_err']] 
                        .to_array()
                        .plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 7))
                ).fig
                plt.close(fig_errs)
                fig_ssh = (
                        cal_data.pipe(add_inter_sw).pipe(lambda d: d.isel(time=d.contiguous_chunk==2))
                        [[ 'ssh_model', 'cal', 'pred']] 
                        .map(lambda da: da.pipe(sobel))
                        # .pipe(sobel)
                        .to_array()
                        .plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 11))
                ).fig
                plt.close(fig_ssh)
                spat_res_df = get_spat_reses(
                    cal_data
                    .assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
                    .assign(
                        syst=lambda d: d.ssh_model + d.syst_error_uncalibrated,
                        tropo=lambda d: d.ssh_model + d.wet_tropo_res,
                        obs=lambda d: d.ssh_model + d.wet_tropo_res + d.syst_error_uncalibrated,
                    )
                    .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
                    .swap_dims(time='x_al', nC='x_ac').drop('time')
                )

                fig_violin_all = sns.violinplot(data=spat_res_df, x='xp_long', y='spat_res').figure
                plt.close(fig_violin_all)
                fig_violin_diff = sns.violinplot(data=spat_res_df.loc[lambda df: df.xp_long.isin(['cal', 'pred'])], x='xp_long', y='spat_res').figure
                plt.close(fig_violin_diff)
                figs.append({
                    'xp': cfgn,
                    'iter': niter,
                    'err': fig_errs,
                    'ssh': fig_ssh,
                    'violin_all': fig_violin_all,
                    'violin_diff': fig_violin_diff,
                })
                print(spat_res_df.groupby('xp_long').spat_res.agg(['mean', 'std']).to_markdown())
                swath_metrics = swath_metrics + [{
                    'xp': cfgn,
                    'iter': niter,
                    'rmse': rms(cal_data.cal - cal_data.ssh_model).item(),
                    'rmse_pred': rms(cal_data.pred - cal_data.ssh_model).item(),
                    'grad_rmse': cal_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.cal) - sobel(g.ssh_model)).pipe(rms).item(),
                    'grad_rmse_pred': cal_data.groupby('contiguous_chunk').apply(lambda g: sobel(g.pred) - sobel(g.ssh_model)).pipe(rms).item(),
                **spat_res_df.loc[lambda df: df.xp_long=='cal'].spat_res.agg({'spat_res_mean': 'mean','spat_res_std': 'std'}).to_dict()
                }]
                
                print(pd.DataFrame(swath_metrics).to_markdown())
            pd.DataFrame(swath_metrics).to_csv(f'{XP_NUM}_sw_chain_metrics.csv')
            pd.DataFrame(grid_metrics).to_csv(f'{XP_NUM}_grid_chain_metrics.csv')
            import pickle
            with open(f'{XP_NUM}_figs.pk', 'wb') as f:
                pickle.dump(figs, f)
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()




def get_swath_psd_score(x_t, x, with_fig=False, with_data=False):
    def psd_score(da: xr.DataArray) -> xr.DataArray:
        err = x_t - da
        psd_x_t = (
            x_t.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='constant', scaling='density'))
                .mean('x_ac')
        ).compute()

        psd_err = (
            err.copy()
                .pipe(
                lambda _da: xrft.power_spectrum(_da, dim='x_al', real_dim='x_al', window='hann', detrend='constant', scaling='density'))
                .mean('x_ac')
        ).compute()
        psd_score = 1 - psd_err / psd_x_t
        return psd_score

    model_score = psd_score(x)

    model_score = (
        model_score.where(model_score.freq_x_al > 0, drop=True).compute()
    )

    psd_plot_data: xr.DataArray = xr.DataArray(
        model_score.data,
        name='PSD score',
        dims=('wl'),
        coords={
            'wl': ('wl', 1 / model_score.freq_x_al.data, {'long_name': 'Wavelength', 'units': 'km'}),
        },
    )
    
    idx = (
            (psd_plot_data.rolling(wl=3, center=True, min_periods=1).mean() > 0.)
            & (psd_plot_data.wl > 10)
    )


    spatial_resolution_model = (
        xr.DataArray(
            # psd_plot_data.sel(var='model').wl,
            psd_plot_data.isel(wl=idx).wl.data,
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

def get_spat_reses(trim_ds):
    spat_reses = []
    chunks = trim_ds.groupby('contiguous_chunk')
    for chunk, g in chunks:
        # print(chunk)

        # for c in ['cal', 'pred', 'gt', 'syst', 'tropo', 'obs'] :
        for c in ['ff_cal', 'cal'] :
            spat_reses.append(
                {
                    'xp_long': c,
                    'spat_res': get_swath_psd_score(g.ssh_model, g[c]).item(),
                    'chunk_nb':g.contiguous_chunk.isel(x_al=0).item()
                }
            )
    spat_res_df = pd.DataFrame(spat_reses)
    return spat_res_df


def make_plots():
    try:
        swath_data = xr.open_dataset('data/swath_train_data.nc')
        val_swath_data = xr.merge([
            xr.open_dataset('data/swath_val_data.nc'),
            xr.open_dataset('data/val_swath_data_xp_12.nc'),
            xr.open_dataset('data/val_swath_sst.nc'),
            ]).pipe(lambda ds: ds.isel(time=np.isfinite(ds.x_al)))
        (
            val_swath_data
            .pred
            .pipe(lambda da: da.isel(time=np.isfinite(da).all('nC')))
            .isel(time=slice(None, 1000)).T.reset_index('time').plot(figsize=(10,3))
        )
        (
            val_swath_data
            .sst
            .pipe(lambda da: da.isel(time=np.isfinite(da).all('nC')))
            .isel(time=slice(None, 1000)).T.reset_index('time').plot(figsize=(10,3))
        )

        (
            val_swath_data
            .qxp12_aug1_dp240_5nad_cal_no_sst_ng5x3cas_l1_dp025_00
            .pipe(lambda da: da.isel(time=np.isfinite(da).all('nC')))
            .isel(time=slice(None, 1000)).T.reset_index('time').plot(figsize=(10,3))
        )

        trainer = pl.Trainer(
            gpus=[5],
            callbacks=[
            ],
            log_every_n_steps=10,
            max_epochs=1000,
            logger=False,
            # overfit_batches=2,
        )

        model_version = 174

        checkpoint = next(Path(f'lightning_logs_archives/220317/version_{model_version}/checkpoints').glob('epoch*.ckpt'))
        if model_version == 178:
            ds_kwargs = dict(
                sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
                sigmas_xb=tuple(),
                sigmas_gt=(0,),
                gt_var='ssh_model',
                ref_var='pred',
                xb_var='zeros',
            )
        if model_version == 174:
            ds_kwargs = dict(
                sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
                sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
                sigmas_gt=(0,),
                gt_var='gt_res',
                ref_var='ref_res',
                xb_var='pred',
            )
        if model_version == 176:
            ds_kwargs = dict(
                sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
                sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
                sigmas_gt=(0,),
                gt_var='gt_res',
                ref_var='ref_res',
                xb_var='oi',
            )
        train_ds = SmoothSwathDataset(swath_data, **ds_kwargs) 
        val_ds = SmoothSwathDataset(val_swath_data, norm_stats=train_ds.stats, **ds_kwargs) 
        
        net = build_net(
                in_channels=len(val_ds.pp_vars),
                out_channels=len(val_ds.gt_vars),
                nhidden = 128,
                depth = 3,
                kernel_size = 3,
                num_repeat = 1,
                residual = True,
                norm_type = 'lrn',
                act_type = 'silu',
                mix = True,
                mix_residual = False,
                mix_act_type = 'none',
                mix_norm_type = 'none',
        )
        lit_mod = LitDirectCNN(
                net,
                gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
                lr_init=2e-3,
                wd=1e-2,
                loss_w={
                    'tot':(5., 3., 3.),
                    'rec':(0., 0., 0.,)
                    # 'tot':(1., 1., 1.), 'rec':(0., 0., 0.,)
                },
            )

        lit_mod.load_state_dict(torch.load(checkpoint)['state_dict'])

        print(pl.utilities.model_summary.summarize(lit_mod, max_depth=3))
        # trainer.fit(lit_mod,
        #     train_dataloaders=train_dl,
        #     val_dataloaders=val_dl
        # )
         


        val_data = generate_cal_xrds(val_ds, lit_mod, trainer)[list(swath_data) + ['cal', 'contiguous_chunk']]   
        p = lambda da: da.T.plot(figsize=(15,3))
        add_inter_sw = lambda ds:(
                    ds
                .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
                .swap_dims(nC='x_ac')
                .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
        )
        gf = lambda ds: (
                ds
                .pipe(lambda dds: dds.isel(time=dds.lat_nadir >33))
                .pipe(lambda dds: dds.isel(time=dds.lat_nadir <43))
                .pipe(lambda dds: dds.isel(time=dds.lon_nadir <305))
                .pipe(lambda dds: dds.isel(time=dds.lon_nadir >295))
            )
        strip = lambda ds, n=2: (
                ds.isel(nC=list(np.arange(0 + n, 26 -n)) + list(np.arange(26 + n, 52 -n)))
            )
        chunk = lambda ds, n=2: (
                ds.isel(time=ds.contiguous_chunk==n)
            )
        gf_val_data = (
                val_data
                .pipe(chunk)
                .pipe(add_inter_sw)
                # .pipe(strip)
        ) #strip(gf(val_data), 2)
        rms = lambda da: np.sqrt(np.mean(da**2))

        
        # if model_version == 174:
        
        #     cal_fn = lambda ds :(ds.cal + ds.pred)
        # elif model_version == 176:
        #     cal_fn = lambda ds :(ds.cal + ds.oi)
        # else:

        #     cal_fn = (lambda ds :ds.cal)

        # print(model_version, rms(val_data.oi - val_data.ssh_model))
        # print(model_version, rms(val_data.pred - val_data.ssh_model))
        print(model_version, rms(cal_fn(val_data) - val_data.ssh_model))
                 
        cal = cal_fn(gf_val_data)
        gt = (gf_val_data.ssh_model)
        print(f'{rms(gf_val_data.pred - gt)=}')
        print(f'{rms(gf_val_data.oi - gt)=}')
        val_data.contiguous_chunk
        
        def sobel(da):
            dx_ac = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 0), da) /2
            dx_al = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 1), da) /2
            return np.hypot(dx_ac, dx_al)

        def med(da):
            return xr.apply_ufunc(lambda _da: ndi.median_filter(_da, 5), da)

        def lap(da):
            return xr.apply_ufunc(lambda _da: ndi.laplace(_da, mode='nearest'), da)

        p(sobel(gf_val_data.oi))

        print(f'{rms(med(cal) - gt)=}')
        p(cal)
        p(gf_val_data.wet_tropo_res)
        p(gf_val_data.syst_error_uncalibrated)
        p(sobel(cal))
        p(lap(cal))

        p(sobel(cal))
        p(lap(cal))
        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter1d(nda, axis=0, sigma=sig, order=0, mode='mirror', truncate=3.0), da)
        rms(xrgf(cal_fn(val_data), 0) - val_data.ssh_model)
        rms(val_data.syst_error_uncalibrated + val_data.wet_tropo_res)
        p(gt)
        p(sobel(gt))
        p(lap(cal))
        val_data.cal.plot()

        ch = (
                cal_data
            .isel(time=cal_data.contiguous_chunk==2)
            # .swap_dims(time='x_al')
            .pipe(add_inter_sw).cal
        )
        ch_gt = (
                cal_data
            .isel(time=cal_data.contiguous_chunk==2)
            # .swap_dims(time='x_al')
            .pipe(add_inter_sw).ssh_model

        )

        from scipy import fftpack
        import xrft
        dat = []
        for f_th in [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]:
            for sig in [1, 2, 4, 5, 10, 20]:
                # rms = (
                #         xrft.fft(ch.swap_dims(time='x_al').drop('time'), dim='x_al')
                #         # .isel(x_ac=0)
                #         .pipe(lambda x: x.where(x.freq_x_al < f_th, ndi.gaussian_filter1d(x, sigma=sig, axis=0)))
                #         .pipe(lambda da:xrft.ifft( da, dim='freq_x_al'))
                #         .pipe(np.real)
                #         .pipe(lambda da: np.sqrt(((da - ch_gt.data)**2).mean()))
                # )
                rms = (
                        ch.pipe(fourier_filter(f_th, sig))
                    .pipe(lambda da: np.sqrt(((da - ch_gt.data)**2).mean()))
                )
                print(rms)
                dat.append({'f_th': f_th, 'sig': sig, 'rms':rms.item()})

        ch.dims
        da = ch
        data.shape
        def fourier_filter(f_th, sig):
            def _f(da):
                data = (
                        xrft.fft(da.swap_dims(time='x_al').drop('time'), dim='x_al')
                        # .isel(x_ac=0)
                        .pipe(lambda x: x.where(x.freq_x_al < f_th, ndi.gaussian_filter1d(x, sigma=sig, axis=0)))
                        .pipe(lambda da:xrft.ifft( da, dim='freq_x_al'))
                        .pipe(np.real)
                        .data
                )
                return  xr.DataArray(data, dims=da.dims, coords=da.coords)
            return _f

        dat = []
        for f_th in [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.]:
            for sig in [1, 2, 4, 5, 10, 20]:
                ff_data = (
                    cal_data
                    .groupby('contiguous_chunk')
                    .apply(lambda g: g.assign(ff_cal = fourier_filter(f_th, sig)(g.cal)))
                )
                rms = (
                        ff_data.pipe(lambda ds: np.sqrt(((ds.ff_cal - ds.ssh_model)**2).mean()))
                )
                spat_res_df = get_spat_reses(
                    ff_data
                    .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
                    .swap_dims(time='x_al', nC='x_ac').drop('time')
                )
                dat.append({
                    'f_th': f_th, 'sig': sig, 'rms':rms.item(),
                    'spat_res_mean':spat_res_df.spat_res.mean(),
                    'spat_res_std':spat_res_df.spat_res.std()})

        pd.DataFrame(dat).rms.min()
        pd.DataFrame(dat).spat_res_mean.min()
        pd.DataFrame(dat).spat_res_std.min()
        cal_data.pipe(lambda da: np.sqrt(((da.cal - da.ssh_model)**2).mean()))

        f_th, sig = 0.05, 1
        ff_data = (
            cal_data
            .groupby('contiguous_chunk')
            .apply(lambda g: g.assign(ff_cal = fourier_filter(f_th, sig)(g.cal)))
        )
        rms = (
                ff_data.pipe(lambda ds: np.sqrt(((ds.ff_cal - ds.ssh_model)**2).mean()))
        )
        spat_res_df = get_spat_reses(
            ff_data
            .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
            .swap_dims(time='x_al', nC='x_ac').drop('time')
        )
        fig_violin_diff = sns.violinplot(data=spat_res_df.loc[lambda df: df.xp_long.isin(['cal', 'ff_cal'])], x='xp_long', y='spat_res').figure
        pg(ff_data.pipe(add_inter_sw).ff_cal.isel(time=ff_data.contiguous_chunk==3))
        (
                ff_data
                .pipe(add_inter_sw)[['ff_cal', 'cal']]
                .isel(time=ff_data.contiguous_chunk==2)
        ).map(sobel).to_array().plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 7))

        (
                ff_data
                .pipe(add_inter_sw)[['ssh_model', 'ff_cal', 'cal']]
                .isel(time=ff_data.contiguous_chunk==3)
        ).pipe(lambda ds: ds - ds.ssh_model).drop('ssh_model').to_array().plot.pcolormesh('time', 'x_ac', col='variable', col_wrap=1, figsize=(15, 10))
        
        ch.pipe(lambda da: np.sqrt(((da - ch_gt)**2).mean()))

        ch.pipe(sobel).isel(x_ac=1).plot()
        pg = lambda da: da.pipe(sobel).T.plot(figsize=(15,4))
        pg(ch)
        pg(ch_gt)
        np.unique(ff_data.contiguous_chunk.values)
        #     # 'dobs_2_1',
        #     # 'dobs_5_2',
        #     # 'dobs_10_5',

        #     # 'dobs_25_10',
        #     # 'dobs_30_25',
        #     # 'dobs_35_30',
        #     # 'dobs_40_35',
        #     # 'dobs_50_40',
        #     # 'dobs_75_50',
        #     # 'dobs_100_75',
        #     # 'obs_100',
        #     'dxb_1_0',
        #     'dxb_2_1',
        #     'dxb_5_2',
        #     'dxb_10_5',
        #     'dxb_50_10',
        #     # 'lat',  # raw vars
        #     # 'lat_nadir',
        #     # 'lon',
        #     # 'lon_nadir',
        #     # 'ssh_model',
        #     # 'ssh_obs',
        #     # 'syst_error_uncalibrated',
        #     # 'wet_tropo_res',
        #     # 'x_ac',
        #     # 'x_al',
        #     # 'gt',
        #     # 'oi',
        #     # 'pred',
        #     # 'obs_gt',
        #     # 'obs_pred',
        #     # 'obs_inp',
        # ]
        # to_plot_ds = (
        #         # chunk_raw
        #         chunk_pp
        #         .assign_coords(x_ac=lambda ds: ('nC', chunk_raw.x_ac.isel(time=0).data))
        #         .swap_dims(nC='x_ac')
        #         # .assign_coords(x_ac=lambda ds: chunk_raw.x_ac.isel(time=0).data)
        #         .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
        # )
        # hv.core.options.Store.set_current_backend('matplotlib')

        # hv_layout = hv.Layout([
        #     hv.Dataset(
        #         to_plot_ds, ['time', 'x_ac'], var
        #     ).to(
        #         hv.QuadMesh, kdims=['time', 'x_ac']
        #     ).relabel(
        #         f'{var}'
        #     ).options(
        #         colorbar=True,
        #         cmap='PiYG',
        #         clim=(to_plot_ds[to_plot_vars].min().to_array().min().item(), to_plot_ds[to_plot_vars].max().to_array().max().item()),
        #         aspect=3.5,
        #         fig_size=500,
        #     )
        #     for var in to_plot_vars
        # ]).cols(1)
        # fig = hv.render(hv_layout, backend='matplotlib')
        # fig

        # v = 'dobs_25_10'
        # # to_plot_ds[v].T.plot(figsize=(15,3))
        # to_plot_ds[v].T.plot(figsize=(15,3))
        # list(chunk_raw)

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        ...
        # fn = fn1
        # fn = generate_data
        # fn = calib_test
        # fn = new_train_cal
        # fn = make_plots
        # fn = regrid_preds
        fn = test_chain_4dvar_cal

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()




def get_cfg(xp_cfg, overrides=None):
    overrides = overrides if overrides is not None else []
    with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp_cfg}',
                'file_paths=dgx_ifremer',
                'entrypoint=train',
            ] + overrides
        )

    return cfg

def get_model(xp_cfg, ckpt, dm=None, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(xp_cfg, setup=True, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    dm = instantiate(cfg.datamodule)
    if setup:
        dm.setup()
    return dm



def get_swot_slice(path, drop_vars=('model_index',),
                   **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-09-30"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]
    print(slice_args)
    print(groups)

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path), drop_variables=drop_vars, group=group,
                          decode_times=False,
                          consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/swot.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64().astype(np.float64)
            te = (dt_end - pd.to_datetime(reference_date)).to_timedelta64().astype(np.float64)
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir > slice_args.get('lon_min', -360)))).compute()
            )

    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        print(
            f"no data found at {path} for {slice_args} {groups} {pd.date_range(start=dt_start.replace(day=1), end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )


def get_nadir_slice(path, **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path),
                          group=group, decode_times=False, consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/nadir.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64() / pd.to_timedelta(1, unit=units.strip())
            te = (dt_end - pd.to_datetime(reference_date)) / pd.to_timedelta(1, unit=units.strip())
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon > slice_args.get('lon_min', -360)))).compute()
            )
    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        print(
            f"no data at {path} found for {slice_args} {groups} {pd.date_range(start=dt_start, end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )




# if __name__ == '__main__':
#     locals().update(main())

