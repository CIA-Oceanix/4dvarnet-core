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
- [ ] launch from config
- [ ] Generate full xp data : gridded product + targets
Xps
- [ ] from oi
- [ ] from 4dvarnet 5nad no SST
- [ ] swot only pred (full field tgt + no xb input)
- [ ] OPT - from SST input
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
             'qxp8_5nad_sst_l1_dp01_01',
             'qxp8_5nad_sst_l1_dp025_00',
             'qxp8_5nad_sst_l2_dp01_01',
             'qxp8_5nad_sst_l2_dp025_00',
            ]
        for cfg_n in cfgs:
            overrides = ['+datamodule.dl_kwargs.shuffle=False']
            cfg = get_cfg(cfg_n, overrides=overrides)
            dm = get_dm(cfg_n,add_overrides=overrides)
            model = get_model(cfg_n, bst_ckpt(f'results/xp8/{cfg_n}'), dm=dm, add_overrides=overrides)

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
            # delta_ds = dl.dataset.datasets[0].gt_ds.ds.ssh - model.test_xr_ds.gt
            # dl.dataset.datasets[0].gt_ds.ds.ssh.isel(time=3).plot() 
            # model.test_xr_ds.gt.isel(time=3).plot() 
            # delta_ds.isel(time=3).plot()
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
            print(f' \n\n ############################# {cfg_n} ##########################################')
            print(f'{rms(swath_data.ssh_model - swath_data.oi).item()=:.2e} m')
            print(f'{rms(swath_data.ssh_model - swath_data.pred).item()=:.2e} m')
            print(f'{rms(swath_data.ssh_model - swath_data.gt).item()=:.2e} m')
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()





def prepro():
    try:
        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter1d(nda, axis=0, sigma=sig, order=0, mode='mirror', truncate=3.0), da)
        p = lambda da: da.T.plot(figsize=(15,3))
        rms = lambda da: np.sqrt(np.mean(da**2))

        class SmoothSwathDataset(torch.utils.data.Dataset):
            def __init__(self, swath_data, norm_stats=None):
                SIGMAS_OBS = [
                        0,
                        1, 2, 5, 10, 25, 30, 35, 40, 50, 75, 100
                ]
                SIGMAS_XB = [
                        0,
                        1, 2, 5, 10, 50
                ]
                SIGMAS_GT = [
                        0,
                        # 5, 10,
                        # 25,
                        # 50,
                        # 75
                ]
                # GT_VAR, REF_VAR = 'ssh_model', 'pred'
                GT_VAR, REF_VAR = 'gt_res', 'ref_res'
                XB_VAR = 'pred'
                # XB_VAR = 'oi'
                swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time') > 3).cumsum())
                sw_data_w_aug = (
                        swath_data
                        .groupby('contiguous_chunk')
                        .apply(
                            lambda g: g.assign(
                                err =  lambda _g: _g.syst_error_uncalibrated + _g.wet_tropo_res,
                                xb =  lambda _g: _g[XB_VAR],
                            ).assign(
                                obs = lambda _g:  _g.ssh_model + _g.err
                            ).assign(
                                obs_res = lambda _g: _g.obs - _g.xb,
                                **{'gt_res': lambda ds: ds.ssh_model - ds.xb},
                                **{'ref_res': lambda ds: ds.pred - ds[XB_VAR]}
                            ).assign(
                                **{f'obs_{sig}' : lambda _g, sig=sig: xrgf(_g.obs, sig) for sig in SIGMAS_OBS},
                                **{f'xb_{sig}' : lambda _g, sig=sig: xrgf(_g.xb, sig) for sig in SIGMAS_XB},
                                **{f'gt_{sig}' : lambda _g, sig=sig: xrgf(_g[GT_VAR], sig) for sig in SIGMAS_GT},
                            )
                        )
                )

                sw_res_data = sw_data_w_aug.assign(
                        **{
                            f'dobs_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'obs_{sig1}'] - ds[f'obs_{sig2}']
                            for sig1, sig2 in zip(SIGMAS_OBS[:-1], SIGMAS_OBS[1:])
                        },
                        **{
                            f'dxb_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'xb_{sig1}'] - ds[f'xb_{sig2}']
                            for sig1, sig2 in zip(SIGMAS_XB[:-1], SIGMAS_XB[1:])
                        },
                        **{
                            f'dgt_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'gt_{sig1}'] - ds[f'gt_{sig2}']
                            for sig1, sig2 in zip(SIGMAS_GT[:-1], SIGMAS_GT[1:])
                        },
                )

                pp_vars = (
                        [f'dobs_{sig2}_{sig1}'for sig1, sig2 in zip(SIGMAS_OBS[:-1], SIGMAS_OBS[1:])] 
                        + ([f'obs_{SIGMAS_OBS[-1]}'] if len(SIGMAS_OBS)>0 else [])
                        + [f'dxb_{sig2}_{sig1}'for sig1, sig2 in zip(SIGMAS_XB[:-1], SIGMAS_XB[1:])]
                        + ([f'xb_{SIGMAS_XB[-1]}'] if len(SIGMAS_XB)>0 else [])
                )
                gt_vars = (
                        [f'dgt_{sig2}_{sig1}'for sig1, sig2 in zip(SIGMAS_GT[:-1], SIGMAS_GT[1:])] + [f'gt_{SIGMAS_GT[-1]}']
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
                self.chunks = list(pp_ds.groupby('contiguous_chunk').count().isel(nC=0).pipe(lambda ds: ds.isel(contiguous_chunk=ds[pp_vars[0]] > min_timestep)).contiguous_chunk.values)

                self.pp_vars = pp_vars 
                self.gt_vars = gt_vars
                self.gt_var = GT_VAR
                self.ref_var = REF_VAR
                self.pp_ds = pp_ds
                self.raw_ds = sw_data_w_aug

            def __len__(self):
                return len(self.chunks)

            def __getitem__(self, idx):
                c = self.chunks[idx]
                pp_item_ds = self.pp_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))
                raw_item_ds = self.raw_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))

                return (
                    pp_item_ds[self.pp_vars].to_array().data,
                    pp_item_ds[self.gt_vars].to_array().data,
                    raw_item_ds[[self.gt_var]].to_array().data,
                    raw_item_ds[[self.ref_var]].to_array().data
                )

        swath_data = xr.open_dataset('data/swath_train_data.nc')
        train_ds = SmoothSwathDataset(swath_data) 
        val_swath_data = xr.open_dataset('data/swath_val_data.nc')
        val_ds = SmoothSwathDataset(val_swath_data, norm_stats=train_ds.stats) 

        # split = [int(len(ds) * 0.75 // 1), int(len(ds) - (len(ds) * 0.75 //1))]
        # train_ds, val_ds = torch.utils.data.random_split(ds, split, generator=torch.Generator().manual_seed(42))
        # self = ds
        # idx=0
        # c = self.chunks[idx]
        # item_ds = self.pp_ds.pipe(lambda ds: ds.isel(time=ds.contiguous_chunk == c))
        # item_ds[self.pp_vars].to_array().data, item_ds[self.gt_vars].to_array().data
        # item = train_ds[0]

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


        """
        Tests to do:
            - [ ] ablation bn
            - [ ] ablation mix
            - [ ] ablation res
            - [ ] Big grad weight
        """
        
        # nhidden = 1024
        # depth = 12

        # nhidden = 512
        # depth = 8

        nhidden = 128
        depth = 3

        # nhidden = 64
        # depth = 1

        kernel_size = 3
        num_repeat = 1
        residual = True
        norm_type = 'lrn'
        act_type = 'silu'
        mix = True
        mix_residual = False
        mix_act_type = 'none'
        mix_norm_type = 'none'

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
                ConvSamePad(in_channels=len(train_ds.pp_vars),out_channels=nhidden, kernel_size=1),
                norm(norm_type=norm_type),
                act(act_type=act_type),
                nn.Sequential(
                    *[inner_net for _ in range(num_repeat)]
                ),
                ConvSamePad(in_channels=nhidden, out_channels=len(train_ds.gt_vars), kernel_size=1),
        )
        
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=3)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=3)
        b = next(iter(train_dl))
        # ds[0][0].shape
        # ds[0][1].shape

        class LitDirectCNN(pl.LightningModule):
            def __init__(
                    self,
                    net,
                    lr_init=1e-3,
                    wd=1e-4,
                    loss_w={'tot':(.1, .1, .1), 'rec':(1., 1., 1.,)},
                    loss_budget_gt_vars=100,
                    gt_var_stats=train_ds.stats,
                ):
                super().__init__()
                self.net = net
                self.lr_init = lr_init
                self.wd = wd
                self.loss_budget_gt_vars = loss_budget_gt_vars
                self.loss_w = loss_w
                self.gt_means = nn.Parameter(torch.from_numpy(gt_var_stats[0])[None, :, None, None], requires_grad=False)
                self.gt_stds = nn.Parameter(torch.from_numpy(gt_var_stats[1])[None, :, None, None], requires_grad=False)

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

            def configure_optimizers(self):
                opt = torch.optim.AdamW(self.parameters(), lr=self.lr_init, weight_decay=self.wd)
                # opt = torch.optim.SGD(self.parameters(), lr=self.lr_init)
                return {
                    'optimizer': opt,
                    'lr_scheduler':
                    # torch.optim.lr_scheduler.ReduceLROnPlateau(
                    #     opt, verbose=True, factor=0.5, min_lr=1e-6, cooldown=5, patience=5,
                    # ),
                    # torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=1e-8, T_max=15),
                    torch.optim.lr_scheduler.CyclicLR(
                        opt, base_lr=1e-4, max_lr=5e-3,  step_size_up=10, step_size_down=10, cycle_momentum=False),
                    'monitor': 'val_loss'
                }

        lit_mod = LitDirectCNN(
                net,
                lr_init=1e-3,
                wd=1e-2,
                loss_w={
                    'tot':(5., 3., 3.),
                    'rec':(0., 0., 0.,)
                    # 'tot':(1., 1., 1.), 'rec':(0., 0., 0.,)
                },
                gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats]
            )

        trainer = pl.Trainer(
            gpus=[5],
            callbacks=[
                callbacks.LearningRateMonitor(),
                # callbacks.TQDMProgressBar(),
                callbacks.RichProgressBar(),
                callbacks.ModelCheckpoint(monitor='val_loss', save_last=True),
                # callbacks.GradientAccumulationScheduler({1: 1, 10: 3, 30: 7, 60: 15, 100: 25, 150: 40}),
                # callbacks.GradientAccumulationScheduler({1: 1, 10: 3, 30: 8, 60: 16}),
                callbacks.StochasticWeightAveraging(),
                # callbacks.GradientAccumulationScheduler({1: 4, 10: 8, 25: 16}),
                callbacks.GradientAccumulationScheduler({1: 4, 10: 8, 15: 16, 20: 32, 30: 64}),
                VersioningCallback()
            ],
            log_every_n_steps=10,
            max_epochs=50,
            # overfit_batches=2,
        )

        print(pl.utilities.model_summary.summarize(lit_mod, max_depth=3))
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
        """
        loss: 0.14 v_num: 55
        val_imp_mse: 0.684
        val_imp_grad_mse: 0.591
        train_imp_mse: 0.046
        train_imp_grad_mse: 0.254
        """
        """
        loss: 0.582 v_num: 101  
        v l_imp_mse: 0.64       
        val_imp_grad_mse: 0.457 
        val_imp_lap_mse: 0.856  
        train_imp_mse: 0.513    
        train_imp_grad_mse:     0.399
        train_imp_lap_mse: 0.727
        """
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def make_plots():

    ds = xr.Dataset()
    print(f'{rms(ds.raw_ds.syst_error_uncalibrated).item()=:.2e} m')
    print(f'{rms(ds.raw_ds.wet_tropo_res).item()=:.2e} m')
    print(f'{rms(ds.raw_ds.ssh_model - ds.raw_ds.oi).item()=:.2e} m')
    print(f'{rms(ds.raw_ds.ssh_model - ds.raw_ds.pred).item()=:.2e} m')
    print(f'{rms(ds.raw_ds.ssh_model - ds.raw_ds.gt).item()=:.2e} m')
    chunk_pp = ds.pp_ds.isel(time=ds.pp_ds.contiguous_chunk==ds.chunks[0])
    chunk_raw = ds.raw_ds.isel(time=ds.raw_ds.contiguous_chunk==ds.chunks[0])
    
    to_plot_vars = [
        # 'dobs_1_0', # pp_vars
        # 'dobs_2_1',
        # 'dobs_5_2',
        # 'dobs_10_5',
        # 'dobs_25_10',
        # 'dobs_30_25',
        # 'dobs_35_30',
        # 'dobs_40_35',
        # 'dobs_50_40',
        # 'dobs_75_50',
        # 'dobs_100_75',
        # 'obs_100',
        'dxb_1_0',
        'dxb_2_1',
        'dxb_5_2',
        'dxb_10_5',
        'dxb_50_10',
        # 'lat',  # raw vars
        # 'lat_nadir',
        # 'lon',
        # 'lon_nadir',
        # 'ssh_model',
        # 'ssh_obs',
        # 'syst_error_uncalibrated',
        # 'wet_tropo_res',
        # 'x_ac',
        # 'x_al',
        # 'gt',
        # 'oi',
        # 'pred',
        # 'obs_gt',
        # 'obs_pred',
        # 'obs_inp',
    ]
    to_plot_ds = (
            # chunk_raw
            chunk_pp
            .assign_coords(x_ac=lambda ds: ('nC', chunk_raw.x_ac.isel(time=0).data))
            .swap_dims(nC='x_ac')
            # .assign_coords(x_ac=lambda ds: chunk_raw.x_ac.isel(time=0).data)
            .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
    )
    hv.core.options.Store.set_current_backend('matplotlib')

    hv_layout = hv.Layout([
        hv.Dataset(
            to_plot_ds, ['time', 'x_ac'], var
        ).to(
            hv.QuadMesh, kdims=['time', 'x_ac']
        ).relabel(
            f'{var}'
        ).options(
            colorbar=True,
            cmap='PiYG',
            clim=(to_plot_ds[to_plot_vars].min().to_array().min().item(), to_plot_ds[to_plot_vars].max().to_array().max().item()),
            aspect=3.5,
            fig_size=500,
        )
        for var in to_plot_vars
    ]).cols(1)
    fig = hv.render(hv_layout, backend='matplotlib')
    fig

    v = 'dobs_25_10'
    # to_plot_ds[v].T.plot(figsize=(15,3))
    to_plot_ds[v].T.plot(figsize=(15,3))
    list(chunk_raw)

def main():
    try:
        ...
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


if __name__ == '__main__':
    locals().update(main())
