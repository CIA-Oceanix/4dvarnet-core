import hydra
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
def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, use_residual=False, final_activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.use_residual = use_residual

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        
        mods = cast_tuple(mods, self.num_layers)

        for i, (layer, mod) in enumerate(zip(self.layers, mods)):
            res = x
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, f'b d -> b {" ".join(["()" for _ in range(len(x.shape)-2)])} d')

            if 0 < i < (self.num_layers - 1):
                x = x + res

        return self.last_layer(x)

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)


# Lit mod
class LitSirenAE(pl.LightningModule):
    def __init__(self, net, mod, solver_mod, state_dim=128, ngrad=5, use_xb=True, lr_init=1e-4, learn_ws=False):
        super().__init__()
        self.net = net
        self.state_dim = state_dim
        self.mod = mod
        self.state_init = nn.Parameter(torch.randn(self.state_dim))
        self.track_grad = True
        # self.automatic_optimization = False
        self.n_state_update = 5
        self.norm_grad = None
        self.ws = nn.Parameter(torch.tensor([1., 10.], device=self.device, requires_grad=learn_ws)[None, None, :])
        self.solver_mod = solver_mod
        self.use_xb = use_xb
        self.lr_init = lr_init

    def _forward(self, state, coords):
        mods = self.mod(state)
        out = self.net(coords, mods) * self.ws
        return out

    def forward(self, batch):
        coords, obs, x_b, gt = batch 
        with torch.enable_grad():
            # state = torch.zeros(obs.shape[0], self.state_dim, device=self.device).normal_(0, 1e-2).requires_grad_() 
            state = repeat(self.state_init, 'd -> b d', b=obs.shape[0])

            for _ in range(self.n_state_update):
                state.requires_grad_(True)
                out = self._forward(state, coords)
                var_cost = self.var_cost(out, obs, x_b)
                state_grad = torch.autograd.grad(var_cost, state, create_graph=self.track_grad)[0]
                norm_grad = self.norm_grad or torch.sqrt( torch.mean( state_grad**2 + 0.))
                state = self.state_init + self.solver_mod(state_grad / norm_grad) 
                if not self.track_grad:
                    state.detach()

        return self._forward(state, coords)

    def var_cost(self, out, obs, x_b): 
        # TODO: add wegihting
        obs_cost = ((out.sum(-1) - obs)**2).mean()
        var_cost = obs_cost
        if self.use_xb:
            x_b_cost = ((out[..., 0] - x_b)**2).mean()
            var_cost = var_cost + x_b_cost
        return var_cost



    def loss(self, t1, t2):
        # TODO: add wegihting
        rmse = ((t1 -t2)**2).mean().sqrt()
        def sob(t):
            if len(t.shape) == 4:
                return kornia.filters.sobel(rearrange(t, 'b d1 d2 c -> b c d1 d2'))
            elif len(t.shape) == 3:
                return kornia.filters.sobel(rearrange(t, 'b d1 d2 -> b () d1 d2'))
            else:
                assert False, 'Should not be here'

        rmse_grad = ((sob(t1) - sob(t2))**2).mean().sqrt()

        return rmse, rmse_grad

    def process_batch(self, batch, phase='val'):
        coords, obs, x_b, gt = batch 
        out = self.forward(batch)
        losses = {}
        losses['err_tot'], losses['g_err_tot'] = self.loss(out, gt)
        losses['err_obs'], losses['g_err_obs'] = self.loss(out.sum(-1), obs)
        losses['err_x_b'], losses['g_err_x_b'] = self.loss(out[..., 0], x_b)
        losses['err_ssh'], losses['g_err_ssh'] = self.loss(out[..., 0], gt[..., 0])
        losses['err_err'], losses['g_err_err'] = self.loss(out[..., 1], gt[..., 1])

        for ln, l in losses.items():
            self.log(f'{phase}_{ln}', l)

        with torch.no_grad():
            err_init, g_err_init = self.loss(x_b, gt[..., 0])
            self.log(f'{phase}_improvement',  losses['err_ssh'] / err_init, prog_bar=True)
            self.log(f'{phase}_improvement_grad', losses['g_err_ssh'] / g_err_init, prog_bar=True)

                   
        loss = losses['err_tot'] + losses['g_err_tot']
        self.log(f'{phase}_loss', loss, prog_bar=True)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.process_batch(batch, phase='train')


    def validation_step(self, batch, batch_idx):
        return self.process_batch(batch, phase='val')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, factor=0.8, min_lr=1e-6),
            'monitor': 'val_loss'
        }


# Swath ds
class SwathDataset(torch.utils.data.Dataset):
    def __init__(self, ds, sel, ref_var, slice_win, strides, select_vars_fn):
        super().__init__()
        self.ds = ds.sel(**sel)
        
        self.strides=strides
        self.ref_var=ref_var
        self.slice_win=slice_win

        self.ds_size = {
            dim: max((self.ds.dims[dim] - slice_win[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in slice_win
        }
        self.select_vars_fn = select_vars_fn
        self.ds = self.ds.transpose('time', 'nC')
        self.return_coords = False


    def __del__(self):
        self.ds.close()

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
        
    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.slice_win[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        if self.return_coords:
            return self.ds.isel(**sl).coords
        return tuple(var_fn(self.ds.isel(**sl)).astype(np.float32) for var_fn in self.select_vars_fn)


# import importlib
# import new_dataloading
# import hydra_main
# importlib.reload(new_dataloading)
# importlib.reload(hydra_main)
# Generate data
def generate_data():
    try:
        overrides = ['+datamodule.dl_kwargs.shuffle=False']
        cfg = get_cfg('qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0', overrides=overrides)
        dm = get_dm('qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',add_overrides=overrides)
        model = get_model('qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0', bst_ckpt('results/xpfeb_train/newaug_5nad_1.0_map_sst'), dm=dm, add_overrides=overrides)

        trainer = pl.Trainer(gpus=[5])
        trainer.test(model, dm.train_dataloader())
        # trainer.test(model, dm.val_dataloader())
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
        swath_data.to_netcdf('data/swath_train_data.nc')
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

# Training 1
def training1():
    try:
        swath_data = xr.open_dataset('data/swath_calib_data.nc')
        def get_coords(sw_ds):  
            def _pp_time(da, bc): 
                    dt = xr.apply_ufunc(lambda c: (pd.to_datetime(c) - pd.to_datetime('2013-05-05')) , da).broadcast_like(bc)
                    norm_dt = dt / pd.to_timedelta('5D') - 1
                    return norm_dt.data

            def _pp_lat(da, bc): 
                dda = da - 32 
                norm_dda = dda / 6 - 1
                return norm_dda.broadcast_like(bc).data

            def _pp_lon(da, bc): 
                dda = da - (-66) 
                norm_dda = dda / 6 - 1
                return norm_dda.broadcast_like(bc).data

            def _pp_x_al(da, bc): 
                norm_dda = da / (5*10**6) - 1
                return norm_dda.broadcast_like(bc).data

            def _pp_x_ac(da, bc): 
                norm_dda = da / 60
                return norm_dda.broadcast_like(bc).data

            return np.stack(
                [
                    _pp_time(sw_ds.time, bc=sw_ds.ssh_model),
                    _pp_lat(sw_ds.lat, bc=sw_ds.ssh_model),
                    _pp_lon(sw_ds.lon, bc=sw_ds.ssh_model),
                    _pp_x_al(sw_ds.x_al, bc=sw_ds.ssh_model),
                    _pp_x_ac(sw_ds.x_ac, bc=sw_ds.ssh_model),
                ],
                axis=-1
            )

        def get_obs(sw_ds):  
            return (sw_ds.ssh_model + sw_ds.syst_error_uncalibrated + sw_ds.wet_tropo_res).data

        def get_gt(sw_ds):  
            return np.stack(
                [ 
                    sw_ds.ssh_model.data,
                    (sw_ds.syst_error_uncalibrated + sw_ds.wet_tropo_res).data,
                ],
                axis=-1
            )

        def get_x_b(sw_ds):  
            return sw_ds.pred.data        
            # return sw_ds.oi.data        
        train_ds, val_ds = (SwathDataset(
                swath_data,
                sel={'time': sl},
                slice_win={'time': 2000, 'nC': 52},
                strides={'time': 500, 'nC': 52},
                ref_var='ssh_model',
                select_vars_fn=[
                    get_coords,
                    get_obs,
                    get_x_b,
                    get_gt,
                ],
        ) for sl in (slice('2013-02-01', '2013-06-15'), slice('2013-06-15', '2013-08-25')))
        ds = train_ds
        len(ds)
        coords, obs, gt, x_b  = ds[0]
        
        net = SirenNet(
            dim_in = 5,     
            dim_hidden = 256,
            dim_out = 2,   
            num_layers = 4, 
            w0_initial = 30.,
            use_residual=False,
        )
        state_dim = 128
        modulator = Modulator(
            dim_in = state_dim,
            dim_hidden = net.dim_hidden,
            num_layers = net.num_layers
        )

        # solver_mod = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, state_dim),
        # )
        solver_mod = lambda g: 0.2*g
        lit_mod = LitSirenAE(
                net=net,
                mod=modulator,
                solver_mod=solver_mod,
                state_dim=state_dim,
                ngrad=0,
                use_xb=False,
                lr_init=1e-3,
                learn_ws=False,
        )
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)

        trainer = pl.Trainer(
            gpus=[5],
            callbacks=[
                callbacks.LearningRateMonitor(),
                # callbacks.TQDMProgressBar(),
                callbacks.RichProgressBar(),
                callbacks.GradientAccumulationScheduler({1: 1, 10: 3, 30: 7, 60: 7}),
            ],
            log_every_n_steps=10,
            max_epochs=301,
            # overfit_batches=2,
        )
        trainer.fit(lit_mod,
            train_dataloader=train_dl,
            val_dataloaders=val_dl
        )
         

    except Exception as e:
        print('I am here')
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



         

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def prepro():
    try:
        swath_data = xr.open_dataset('data/swath_train_data.nc')
        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter1d(nda, axis=0, sigma=sig, order=0, mode='mirror', truncate=3.0), da)
        p = lambda da: da.T.plot(figsize=(15,3))
        rms = lambda da: np.sqrt(np.mean(da**2))

        class SmoothSwathDataset(torch.utils.data.Dataset):
            def __init__(self, swath_data):
                SIGMAS_OBS = [
                        0, 1, 2, 5, 10, 25, 30, 35, 40, 50, 75, 100
                ]
                SIGMAS_XB = [
                        0, 1, 2, 5, 10, 50
                ]
                swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time') > 3).cumsum())
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
                                **{f'obs_{sig}' : lambda _g, sig=sig: xrgf(_g.obs, sig) for sig in SIGMAS_OBS},
                                **{f'xb_{sig}' : lambda _g, sig=sig: xrgf(_g.pred, sig) for sig in SIGMAS_XB},
                            )
                        )
                )

                sw_res_data = sw_data_w_aug.assign(
                        **{
                            f'dobs_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'obs_{sig2}'] - ds[f'obs_{sig1}']
                            for sig1, sig2 in zip(SIGMAS_OBS[:-1], SIGMAS_OBS[1:])
                        },
                        **{
                            f'dxb_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'xb_{sig2}'] - ds[f'xb_{sig1}']
                            for sig1, sig2 in zip(SIGMAS_XB[:-1], SIGMAS_XB[1:])
                        },
                        **{'gt_res': lambda ds: ds.ssh_model - ds.pred}
                )

                pp_vars = (
                        [f'dobs_{sig2}_{sig1}'for sig1, sig2 in zip(SIGMAS_OBS[:-1], SIGMAS_OBS[1:])]  + [f'obs_{SIGMAS_OBS[-1]}']
                        +[f'dxb_{sig2}_{sig1}'for sig1, sig2 in zip(SIGMAS_XB[:-1], SIGMAS_XB[1:])]+ [f'xb_{SIGMAS_XB[-1]}']
                        )

                # for v in pp_vars:
                #     p(sw_res_data.isel(time=sw_res_data.contiguous_chunk==2)[v])

                gt_vars = ['gt_res']
                sw_res_data[pp_vars].mean()
                sw_res_data[pp_vars].std()
                pp_ds = sw_res_data[pp_vars + gt_vars].pipe(lambda ds: (ds - ds.mean()) / ds.std()).assign(contiguous_chunk=sw_res_data.contiguous_chunk).astype(np.float32)

                min_timestep = 300
                self.chunks = list(pp_ds.groupby('contiguous_chunk').count().isel(nC=0).pipe(lambda ds: ds.isel(contiguous_chunk=ds[pp_vars[0]] > min_timestep)).contiguous_chunk.values)

                self.pp_vars = pp_vars 
                self.gt_vars = gt_vars 
                self.stats = sw_res_data[pp_vars + gt_vars].mean(), sw_res_data[pp_vars + gt_vars].std()
                self.pp_ds = pp_ds
                self.raw_ds = swath_data

            def __len__(self):
                return len(self.chunks)

            def __getitem__(self, idx):
                c = self.chunks[idx]
                item_ds = self.pp_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))

                return item_ds[self.pp_vars].to_array().data, item_ds[self.gt_vars].to_array().data

        ds = SmoothSwathDataset(swath_data) 
        self = ds
        idx=0
        c = self.chunks[idx]
        item_ds = self.pp_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))
        item_ds[self.pp_vars].to_array().data, item_ds[self.gt_vars].to_array().data
        item = ds[0]

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
                return self.conv(F.pad(inp, (same_pad['top'], same_pad['bottom'], same_pad['left'], same_pad['right'])))

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

        nhidden = 1024
        kernel_size = 3
        depth = 12
        residual = True
        # TODO  test mixer apply conv with nC as channels
        class ResidualBlock(nn.Module):
            def __init__(self, net,  res=True):
                super().__init__()
                self.net = net
                self.res = res

            def forward(self, x):
                if not self.res:
                    return self.net(x)
                return x + self.net(x)
        net = nn.Sequential(
                ConvSamePad(in_channels=len(ds.pp_vars),out_channels=nhidden, kernel_size=kernel_size),
                nn.BatchNorm2d(num_features=nhidden),
                nn.SiLU(),
                *[ nn.Sequential(
                    ResidualBlock(
                        nn.Sequential(
                            ConvSamePad(in_channels=nhidden,out_channels=nhidden, kernel_size=kernel_size),
                            nn.BatchNorm2d(num_features=nhidden),
                            nn.SiLU(),
                        ), res=residual),
                    #ConvSamePad(in_channels=nhidden,out_channels=nhidden, kernel_size=kernel_size),
                )
                for _ in range(depth) ],
                ConvSamePad(in_channels=nhidden, out_channels=len(ds.gt_vars), kernel_size=kernel_size),
        )
        
        split = [int(len(ds) * 0.75 // 1), int(len(ds) - (len(ds) * 0.75 //1))]
        train_ds, val_ds = torch.utils.data.random_split(ds, split)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=3)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3)
        ds[0][0].shape
        ds[0][1].shape

        class LitDirectCNN(pl.LightningModule):
            def __init__(self, net, lr_init=1e-3):
                super().__init__()
                self.net = net
                self.lr_init = lr_init

            def forward(self, batch):
                x, y = batch 
                return self.net(x)



            def loss(self, t1, t2):
                # TODO: add wegihting
                rmse = ((t1 -t2)**2).mean().sqrt()
                def sob(t):
                    if len(t.shape) == 4:
                        return kornia.filters.sobel(rearrange(t, 'b d1 d2 c -> b c d1 d2'))
                    elif len(t.shape) == 3:
                        return kornia.filters.sobel(rearrange(t, 'b d1 d2 -> b () d1 d2'))
                    else:
                        assert False, 'Should not be here'

                rmse_grad = ((sob(t1) - sob(t2))**2).mean().sqrt()

                return rmse, rmse_grad

            def process_batch(self, batch, phase='val'):
                x, y = batch 
                out = self.forward(batch)
                losses = {}
                losses['err_tot'], losses['g_err_tot'] = self.loss(out, y)

                for ln, l in losses.items():
                    self.log(f'{phase}_{ln}', l)

                loss_ref, g_loss_ref = self.loss(y, torch.zeros_like(y))
                self.log(f'{phase}_imp_mse', losses['err_tot'] / loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_grad_mse', losses['g_err_tot'] / g_loss_ref, prog_bar=True, on_step=False, on_epoch=True)

                loss = 3. * losses['err_tot'] + losses['g_err_tot']
                self.log(f'{phase}_loss', loss, prog_bar=False)
                return loss
                
            def training_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='train')

            def validation_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='val')

            def configure_optimizers(self):
                opt = torch.optim.Adam(self.parameters(), lr=self.lr_init)
                return {
                    'optimizer': opt,
                    'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, factor=0.8, min_lr=1e-6),
                    'monitor': 'val_loss'
                }

        lit_mod = LitDirectCNN(net, lr_init=4e-4)
        trainer = pl.Trainer(
            gpus=[5],
            callbacks=[
                callbacks.LearningRateMonitor(),
                # callbacks.TQDMProgressBar(),
                callbacks.RichProgressBar(),
                callbacks.GradientAccumulationScheduler({1: 1, 10: 3, 30: 7, 60: 15}),
                callbacks.StochasticWeightAveraging(swa_epoch_start=50),
                # callbacks.GradientAccumulationScheduler({1: 15}),
            ],
            log_every_n_steps=10,
            max_epochs=301,
            # overfit_batches=2,
        )
        trainer.fit(lit_mod,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl
        )
         
        """    it/s loss: 0.456 v_num: 53
        val_imp_mse: 0.677    
        val_imp_grad_mse: 0.63
        train_imp_mse: 0.192 
        train_imp_grad_mse:0.41
        """
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def main():
    try:
        # fn = fn1
        # fn = generate_data
        # fn = calib_test
        fn = prepro

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()



def bst_ckpt(dirpath, glob='*'):
    return min(Path(dirpath).glob('version_*/checkpoints/*'), key=lambda p: float(re.match('.+val_loss=(.+)\.ckpt', str(p)).group(1)))


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

