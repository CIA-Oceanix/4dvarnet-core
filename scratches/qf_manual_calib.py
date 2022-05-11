import hydra
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
- [ ] find normalization and test on single item convergence
- [ ] to test: network dimensions
- [ ] to test: no modulator
- [ ] to test: loss ponderations
- [ ] to test: grad model = Identity 
- [ ] to test: slice window
- [ ] to test: n grad iter à 0
- [ ] to test: no xb
- [ ] to test: anomaly wrt xb
- [ ] to test: learn ws
- [ ] to test: learnable embbedding per batch
- [ ] to test: improvement wrt OI ?
- [ ] to test: batch selection: one batch = one pass + all cross overs + n next|previous pass
- [ ] to test: Simple preprocessing : low res from x_b, high res from swot
- [ ] to think: NORMALIZATION... what coord system is common across batch, and which isn't
- [ ] to think: LOSSES using derviatives of the SIREN  ?
"""

display(Markdown(s))

cs = ConfigStore.instance()
cfgs = cs.list('xp')


# Siren imp

# import importlib
# import new_dataloading
# import hydra_main
# importlib.reload(new_dataloading)
# importlib.reload(hydra_main)
# Generate data
def generate_data():
    try:

        cs.list('xp')
        cfgs =  [
             'qxp8_5nad_sst_l1_dp01_01.yaml',
             'qxp8_5nad_sst_l1_dp025_00.yaml',
             'qxp8_5nad_sst_l2_dp01_01.yaml',
             'qxp8_5nad_sst_l2_dp025_00.yaml',
            ]
        overrides = ['+datamodule.dl_kwargs.shuffle=False']
        cfg = get_cfg('qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0', overrides=overrides)
        dm = get_dm('qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',add_overrides=overrides)
        model = get_model('qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0', bst_ckpt('results/xpfeb_train/newaug_5nad_1.0_map_sst'), dm=dm, add_overrides=overrides)

        trainer = pl.Trainer(gpus=[5])
        # trainer.test(model, dm.train_dataloader())
        trainer.test(model, dm.val_dataloader())
        slice_args = dict(
                # time_min='2012-10-01', 
                time_min= pd.to_datetime(np.min(model.test_xr_ds['time']).values).date(),
                # time_max='2012-10-30',
                time_max= pd.to_datetime(np.max(model.test_xr_ds['time']).values).date(),
                lat_min=model.test_xr_ds['lat'].min().item(),
                lat_max=model.test_xr_ds['lat'].max().item(),
                lon_min=model.test_xr_ds['lon'].min().item() + 360,
                lon_max=model.test_xr_ds['lon'].max().item() + 360,
        )

        swath_data = get_swot_slice('../sla-data-registry/sensor_zarr/zarr/new_swot', **slice_args, drop_vars=[
         'bd_err', 'karin_noise', 'karin_err', 'phase_err', 'roll_err','timing_err',
         # 'lat', 'lat_nadir', 'lon', 'lon_nadir',
         'model_index',
         # 'ssh_model',
         # 'syst_error_uncalibrated', 'wet_tropo_res', 'x_ac', 'x_al'
         ])

        # model.test_xr_ds.obs_gt.isel(time=6).plot()
        dl = dm.train_dataloader()
        delta_ds = dl.dataset.datasets[0].gt_ds.ds.ssh - model.test_xr_ds.gt
        dl.dataset.datasets[0].gt_ds.ds.ssh.isel(time=3).plot() 
        model.test_xr_ds.gt.isel(time=3).plot() 
        delta_ds.isel(time=3).plot()
        swathed_grid = model.test_xr_ds.interp(
            time=swath_data.time.broadcast_like(swath_data.ssh_model),
            lat=swath_data.lat.broadcast_like(swath_data.ssh_model),
            lon=swath_data.lon.broadcast_like(swath_data.ssh_model) - 360,
        )
        swath_data = (
            swath_data.assign({v: (swath_data.ssh_model.dims, swathed_grid[v].data)for v in swathed_grid} )
            .pipe(lambda ds: ds.isel(time=np.isfinite(ds.oi).all('nC')))
        )
        # swath_data.to_netcdf('data/swath_train_data.nc')

        rms = lambda da: np.sqrt(np.mean(da**2))
        swath_data.gt.std()
        swath_data.ssh_model.std()
        print(f'{cfg}')
        print(f'{rms(swath_data.ssh_model - swath_data.oi).item()=:.2e} m')
        print(f'{rms(swath_data.ssh_model - swath_data.pred).item()=:.2e} m')
        print(f'{rms(swath_data.ssh_model - swath_data.gt).item()=:.2e} m')
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()




# manual calib 
def calib_test():
    try:

        # swath_data = xr.open_dataset('data/swath_val_data.nc')
        swath_data = xr.open_dataset('data/swath_train_data.nc')


        swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time') > 3).cumsum())
        chunk = swath_data.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk==2))
        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter1d(nda, axis=0, sigma=sig, order=0, mode='mirror', truncate=3.0), da)
        p = lambda da: da.T.plot(figsize=(15,3))
        rms = lambda da: np.sqrt(np.mean(da**2))
        swath_data.gt.std()
        swath_data.ssh_model.std()
        rms(swath_data.ssh_model - swath_data.oi)
        rms(swath_data.ssh_model - swath_data.pred)
        rms(swath_data.ssh_model - swath_data.gt)


        def find_best_sigmas(x_b, obs_res, gt):
            best = (-1, (0, 0, 0))
            for sig0 in [0, 1, 2, 5, 8]:
                for sig1 in [0, 1, 2, 5, 8, 10]:
                    for sig2 in [ 5, 10, 25, 30, 35, 40, 50, 75]:

                        aug_pred =  xrgf(x_b, sig0) + (xrgf(obs_res, sig1) - xrgf(obs_res, sig2))

                        aug_pred_err = aug_pred - gt
                        pred_err = x_b - gt
                        imp = 100 - (rms(aug_pred_err) / rms(pred_err)).item() * 100
                        if imp > best[0]:
                            best = imp, (sig0, sig1, sig2)
            
            (sig0, sig1, sig2) = best[1]
            aug_pred =  xrgf(x_b, sig0) + (xrgf(obs_res, sig1) - xrgf(obs_res, sig2))
            aug_pred_err = aug_pred - gt
            pred_err = x_b - gt
            imp = 100 - (rms(aug_pred_err) / rms(pred_err)).item() * 100
            print(f"Improvements wrt 5nad sst: { imp:.1f}%, ({x_b.shape=}, {sig0, sig1, sig2=}")
            return aug_pred

        sw_data_w_aug = (
                swath_data
                .groupby('contiguous_chunk')
                .apply(
                    lambda g: g.assign(
                        err =  lambda _g: _g.syst_error_uncalibrated + _g.wet_tropo_res,
                    ).assign(
                        obs = lambda _g:  _g.ssh_model + _g.err
                    ).assign(
                        obs_res = lambda _g: _g.obs - _g.pred 
                    ).assign(
                        aug_pred = lambda _g: find_best_sigmas(_g.pred, _g.obs_res, _g.ssh_model)
                    )
                )
        )

        eval_ds = (
            sw_data_w_aug
            .pipe( lambda ds: ds.isel(time= ds.lat_nadir > 33 ))
            .pipe( lambda ds: ds.isel(time= ds.lat_nadir < 43 ))
            .pipe( lambda ds: ds.isel(time= ds.lon_nadir > 295 ))
            .pipe( lambda ds: ds.isel(time= ds.lon_nadir < 305 ))
        )
        aug_pred_err = eval_ds.aug_pred - eval_ds.ssh_model
        pred_err = eval_ds.pred - eval_ds.ssh_model
        oi_err = eval_ds.oi - eval_ds.ssh_model
        err = eval_ds.err
        imp = 100 - (rms(aug_pred_err) / rms(pred_err)).item() * 100
        print(f"Improvements wrt 5nad sst: { imp:.1f}%")
        print(f"Improvements wrt 4nad oi: { 100 - (rms(aug_pred_err) / rms(oi_err)).item() * 100:.1f}%")
        print(f"Improvements wrt noisy_obs: { 100 - (rms(aug_pred_err) / rms(err)).item() * 100:.1f}%")


        """
        Aug pred
        Improvements wrt 5nad sst: 33.5%
        Improvements wrt 4nad oi: 65.3%
        Improvements wrt noisy_obs: 99.4%

        Oi aug
        Improvements wrt 5nad sst: -3.4%
        Improvements wrt 4nad oi: 46.1%
        Improvements wrt noisy_obs: 99.0%
        """

