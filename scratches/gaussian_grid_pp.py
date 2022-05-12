"""
# Using the swath gaussian diff preprocessing for direct interpolation


## multiple ways of doing this:
 Using coarsening
 Using 2d kernels on the gridded nadirs
 Using 1d kernel on the tracks


## Other idea
tri plane decomposition



## So one pre


First idea:
    - project the nadir data on the triplanes
    - apply different levels of nan-average pooling + upsampling
    - direct prediction from the learnt features

"""

import pyinterp
import hydra
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

from functools import cache

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

cached_get_nadir_slice = cache(get_nadir_slice)

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


class NewDataset(torch.utils.data.Dataset):
    def __init__(self, tgt_ds, ref_ds, get_nad_fns, n_coords, t_win, s_win, norm_stats=None):
        super().__init__()
        self.tgt_ds = tgt_ds
        self.ref_ds = ref_ds
        self.get_nad_fns = get_nad_fns
        self.t_win = t_win
        self.s_win = s_win
        self.n_coords = n_coords

        if norm_stats is None:
            self.avgs, self.stds =  self.compute_stats()
        else:

            self.avgs, self.stds =  norm_stats


    def __len__(self):
        return len(self.tgt_ds['time'])

    def compute_stats(self):
        self.avgs, self.stds =  0., 1. 
        c = 0
        s = None
        for it in range(len(self)):
            x, *_ = self._getitem(it)
            if s is None:
                s = np.sum(x, axis=(1,2), keepdims=True)
            else:
                s += np.sum(x, axis=(1,2), keepdims=True)
            c+=np.sum(np.isfinite(x), (1,2), keepdims=True)
        avg = s / c

                
        c = 0
        s = None
        for it in range(len(self)):
            x, *_ = self._getitem(it)
            if s is None:
                s = np.sum((x-avg)**2, axis=(1,2), keepdims=True)
            else:
                s += np.sum((x-avg)**2, axis=(1,2), keepdims=True)

            c+=np.sum(np.isfinite(x), (1,2), keepdims=True)
        std = np.sqrt(s / c)

        return avg.astype(np.float32), np.maximum(std.astype(np.float32), 1e-2)
    
    @cache
    def __getitem__(self, item):
        return self._getitem(item)

    def _getitem(self, item):
        tgt_item = self.tgt_ds.isel(time=item)
        ref_item = self.ref_ds.isel(time=item)
        t = self.tgt_ds.time.isel(time=item).values
        slice_args = dict(
                time_min= t - self.t_win,
                time_max= t + self.t_win,
                lat_min=tgt_item['lat'].min().item()- self.s_win,
                lat_max=tgt_item['lat'].max().item() + self.s_win,
                lon_min=tgt_item['lon'].min().item() + 360 - self.s_win,
                lon_max=tgt_item['lon'].max().item() + 360 + self.s_win,
        )
        swath_data = self.get_nad_fns(slice_args)
        
        t_coords = pd.date_range(slice_args['time_min'], slice_args['time_max'], periods=self.n_coords)
        lon_coords = np.linspace(slice_args['lon_min'], slice_args['lon_max'], num=self.n_coords)
        lat_coords = np.linspace(slice_args['lat_min'], slice_args['lat_max'], num=self.n_coords)

        tgt_gridtx = xr.Dataset(coords={'time': ('time', t_coords) , 'lon':('lon', lon_coords)})
        tgt_gridxy = xr.Dataset(coords={'lon':('lon', lon_coords), 'lat': ('lat', lat_coords)})
        tgt_gridyt = xr.Dataset(coords={'lat': ('lat', lat_coords), 'time': ('time', t_coords)})
        binningtx = pyinterp.Binning2D(pyinterp.Axis(tgt_gridtx.time.astype(float).values), pyinterp.Axis(tgt_gridtx.lon.values))
        binningxy = pyinterp.Binning2D(pyinterp.Axis(tgt_gridxy.lon.values), pyinterp.Axis(tgt_gridxy.lat.values))
        binningyt = pyinterp.Binning2D(pyinterp.Axis(tgt_gridxy.lat.values), pyinterp.Axis(tgt_gridyt.time.astype(float).values))

        binningtx.clear()
        binningyt.clear()
        binningyt.clear()
        for nad_data in swath_data.values():
            if nad_data is None:
                continue
            values = np.ravel(nad_data.ssh_model.values)
            times = np.ravel(nad_data.time.astype(float).values)
            lons = np.ravel(nad_data.lon.values) - 360
            lats = np.ravel(nad_data.lat.values)
            msk = np.isfinite(values)
            binningtx.push(times[msk], lons[msk], values[msk])
            binningxy.push(lons[msk], lats[msk], values[msk])
            binningyt.push(lats[msk], times[msk], values[msk])

        gridded_tx =  (('time', 'lon'), binningtx.variable('mean'))
        gridded_xy =  (('lon', 'lat'), binningxy.variable('mean'))
        gridded_yt =  (('lat', 'time'), binningyt.variable('mean'))
        gridded =  xr.Dataset(
               {'tx':gridded_tx, 'xy':gridded_xy, 'yt':gridded_yt, },
               {'time': t_coords, 'lat': lat_coords, 'lon': lon_coords}
        )


        coarse_grids = [gridded]
        for i in range(int(np.log2(self.n_coords)) - 1):
            coarse_factor = 2 ** (i+1)
            coarse_grids.append(
                gridded
                .coarsen(time=coarse_factor, lat=coarse_factor, lon=coarse_factor)
                .mean()
            )

        pp_grids = []
        for i, (cg, ncg) in enumerate(zip(coarse_grids[:-1], coarse_grids[1:])):
            pp_grids.append(
                    cg
                    .pipe(lambda ds: ds - ncg.interp_like(ds, method='nearest', kwargs={"fill_value": "extrapolate"}))
                    .fillna(0.)
                    .interp_like(gridded, method='linear', kwargs={"fill_value": "extrapolate"})
                    .pipe(lambda ds: ds.rename_vars( {n: f'{n}_{i}'for n in ds}))
            )

        pp_grids.append(
                coarse_grids[-1]
                .pipe(lambda ds: ds - gridded.mean())
                .fillna(0.)
                .interp_like(gridded, method='linear', kwargs={"fill_value": "extrapolate"})
                .pipe(lambda ds: ds.rename_vars( {n: f'{n}_{i+1}'for n in ds}))
        )
        pp_grids.append(
                xr.full_like(gridded, dict(gridded.mean()))
                .pipe(lambda ds: ds.rename_vars( {n: f'{n}_mean' for n in ds}))
        )
        
        merged_pp = xr.merge(pp_grids)
        self._merged = merged_pp
        self._tgt = tgt_item.astype(np.float32)
        inp = merged_pp.sel(time=tgt_item.time, method='nearest')
        return (
                (inp.broadcast_like(inp.xy_0).to_array().astype(np.float32).values - self.avgs) / self.stds,
                tgt_item.astype(np.float32).values[None, ...], 
                ref_item.astype(np.float32).values[None, ...], 
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
    mix = False,
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
                    lr_init=1e-3,
                    wd=1e-4,
                    loss_w={'rec':(1., 1., 1.,)},
                ):
                super().__init__()
                self.net = net
                self.lr_init = lr_init
                self.wd = wd
                self.loss_w = loss_w
                self.save_hyperparameters()

            def forward(self, batch):
                x, y, *_ = batch 
                out = self.net(x)
                return F.interpolate(self.net(x), y.size()[2:])

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
                x, y, ref = batch 
                out = self.forward(batch)
                losses = {}

                losses['err_rec'], losses['g_err_rec'], losses['l_err_rec'] = self.loss(out, y)

                for ln, l in losses.items():
                    self.log(f'{phase}_{ln}', l)

                loss_ref, g_loss_ref, l_loss_ref= self.loss(ref, y)
                self.log(f'{phase}_imp_mse', losses['err_rec'] / loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_grad_mse', losses['g_err_rec'] / g_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_lap_mse', losses['l_err_rec'] / l_loss_ref, prog_bar=True, on_step=False, on_epoch=True)

                loss = (
                    self.loss_w['rec'][0] * losses['err_rec']
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
                return {
                    'optimizer': opt,
                    'lr_scheduler':
                    torch.optim.lr_scheduler.CyclicLR(
                        opt, base_lr=5e-5, max_lr=5e-3,  step_size_up=25, step_size_down=25, cycle_momentum=False, mode='triangular2'),
                    'monitor': 'val_loss'
                }

def first_training():
    
    try:
        XP_NUM= "grid_1"
        xp_name= "first"
        cfg_n = 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00'
        cfg = get_cfg(cfg_n)
        print(OmegaConf.to_yaml(cfg))
        gt_ds = xr.open_dataset(cfg.file_paths.natl_ssh_daily).sel(instantiate(cfg.params.test_domain))
        gt_ds.time.attrs["units"] = "seconds since 2012-10-01"
        gt_ds = xr.decode_cf(gt_ds)
        ref_ds = xr.open_dataset(cfg.file_paths.oi_4nadir).sel(instantiate(cfg.params.test_domain))
        t_win = '10D'
        s_win = 2.
        train_ds = NewDataset(
                tgt_ds=gt_ds.ssh.sel(time=instantiate(cfg.datamodule.train_slices[0])),
                ref_ds=ref_ds.ssh_mod.sel(time=instantiate(cfg.datamodule.train_slices[0])),
                # tgt_ds=gt_ds.ssh.sel(time=instantiate(cfg.datamodule.val_slices[0])),
                # ref_ds=ref_ds.ssh_mod.sel(time=instantiate(cfg.datamodule.val_slices[0])),
                get_nad_fns=lambda slkw: {
                    nad: cached_get_nadir_slice(f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}', **slkw) for nad in [ 'en', 'g2', 'j1', 'tpn', ]
                },
                n_coords=256,
                t_win=pd.to_timedelta(t_win),
                s_win=s_win,
        )

        val_ds = NewDataset(
                tgt_ds=gt_ds.ssh.sel(time=instantiate(cfg.datamodule.val_slices[0])),
                ref_ds=ref_ds.ssh_mod.sel(time=instantiate(cfg.datamodule.val_slices[0])),
                get_nad_fns=lambda slkw: {
                    nad: cached_get_nadir_slice(f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}', **slkw) for nad in [ 'en', 'g2', 'j1', 'tpn', ]
                },
                n_coords=256,
                t_win=pd.to_timedelta(t_win),
                s_win=s_win,
                norm_stats=(train_ds.avgs, train_ds.stds)
        )

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
        assert np.all(np.isfinite(train_ds[0][0]))
        assert np.all(np.isfinite(train_ds[0][1]))
        assert np.all(np.isfinite(train_ds[0][2]))

        net = build_net(
                in_channels=train_ds.stds.shape[0],
                out_channels=1,
                nhidden = 128,
                depth = 3,
                kernel_size = 3,
                num_repeat = 1,
                residual = True,
                norm_type = 'lrn',
                act_type = 'relu',
                mix = False,
                mix_residual = False,
                mix_act_type = 'none',
                mix_norm_type = 'none',
        )
        cal_mod = LitDirectCNN(
                net,
                lr_init=1e-4,
                wd=1e-4,
                loss_w={'rec':(1., 1., 1.,)},
            )
        logger = pl.loggers.TensorBoardLogger('lightning_logs', name=f'{XP_NUM}_{xp_name}')
        trainer = pl.Trainer(
            gpus=[6],
            logger=logger,
            callbacks=[
                callbacks.LearningRateMonitor(),
                callbacks.RichProgressBar(),
                callbacks.ModelCheckpoint(monitor='val_loss', save_last=True),
                callbacks.StochasticWeightAveraging(),
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

class NewDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self,):
        ...

def explore():
    try:
        
        cfg_n = 'qxp12_aug2_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00'
        overrides = ['+datamodule.dl_kwargs.shuffle=False']
        dm = get_dm(cfg_n, add_overrides=overrides)
        win = 9
        grid_ds = dm.val_ds.datasets[0].gt_ds.ds.isel(time=slice(None, win))
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

        swath_data = {nad: cached_get_nadir_slice(f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}', **slice_args) for nad in 
            [ 'en', 'g2', 'j1', 'tpn', ]}

        import pyinterp
        n_coords = 256
        
        t_coords = pd.date_range(grid_ds.time.min().values, grid_ds.time.max().values, periods=n_coords)
        lon_coords = np.linspace(grid_ds.lon.min().values, grid_ds.lon.max().values, endpoint=True, num=n_coords)
        lat_coords = np.linspace(grid_ds.lat.min().values, grid_ds.lat.max().values, endpoint=True, num=n_coords)
        tgt_gridtx = xr.Dataset(coords={'time': ('time', t_coords) , 'lon':('lon', lon_coords)})
        tgt_gridxy = xr.Dataset(coords={'lon':('lon', lon_coords), 'lat': ('lat', lat_coords)})
        tgt_gridyt = xr.Dataset(coords={'lat': ('lat', lat_coords), 'time': ('time', t_coords)})
        binningtx = pyinterp.Binning2D(pyinterp.Axis(tgt_gridtx.time.astype(float).values), pyinterp.Axis(tgt_gridtx.lon.values))
        binningxy = pyinterp.Binning2D(pyinterp.Axis(tgt_gridxy.lon.values), pyinterp.Axis(tgt_gridxy.lat.values))
        binningyt = pyinterp.Binning2D(pyinterp.Axis(tgt_gridxy.lat.values), pyinterp.Axis(tgt_gridyt.time.astype(float).values))

        binningtx.clear()
        binningyt.clear()
        binningyt.clear()
        for nad_data in swath_data.values():


            values = np.ravel(nad_data.ssh_model.values)
            times = np.ravel(nad_data.time.astype(float).values)
            lons = np.ravel(nad_data.lon.values) - 360
            lats = np.ravel(nad_data.lat.values)
            msk = np.isfinite(values)
            binningtx.push(times[msk], lons[msk], values[msk])
            binningxy.push(lons[msk], lats[msk], values[msk])
            binningyt.push(lats[msk], times[msk], values[msk])

        gridded_tx =  (('time', 'lon'), binningtx.variable('mean'))
        gridded_xy =  (('lon', 'lat'), binningxy.variable('mean'))
        gridded_yt =  (('lat', 'time'), binningyt.variable('mean'))
        gridded =  xr.Dataset(
               {'tx':gridded_tx, 'xy':gridded_xy, 'yt':gridded_yt, },
               {'time': t_coords, 'lat': lat_coords, 'lon': lon_coords}
        )
        gridded.tx.plot()
        gridded.xy.plot()
        gridded.yt.plot()

        xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(lambda nda: ndi.gaussian_filter(nda, sigma=sig, order=0, mode='mirror', truncate=3.0), da)


        coarse_grids = [gridded]
        for i in range(int(np.log2(n_coords)) - 1):
            coarse_factor = 2 ** (i+1)
            coarse_grids.append(
                gridded
                .coarsen(time=coarse_factor, lat=coarse_factor, lon=coarse_factor)
                .mean()
            )

        pp_grids = []
        prev_cg = None
        for i, (cg, ncg) in enumerate(zip(coarse_grids[:-1], coarse_grids[1:])):
            
            pp_grids.append(
                    cg
                    .pipe(lambda ds: ds - ncg.interp_like(ds, method='nearest', kwargs={"fill_value": "extrapolate"}))
                    .fillna(0.)
                    .interp_like(gridded, method='linear', kwargs={"fill_value": "extrapolate"})
                    .pipe(lambda ds: ds.rename_vars( {n: f'{n}_{i}'for n in ds}))
            )

        pp_grids.append(
                coarse_grids[-1]
                .pipe(lambda ds: ds - gridded.mean())
                .fillna(0.)
                .interp_like(gridded, method='linear', kwargs={"fill_value": "extrapolate"})
                .pipe(lambda ds: ds.rename_vars( {n: f'{n}_{i+1}'for n in ds}))
        )
        pp_grids.append(
                xr.full_like(gridded, dict(gridded.mean()))
                .pipe(lambda ds: ds.rename_vars( {n: f'{n}_mean' for n in ds}))
        )
        
        merged_pp = xr.merge(pp_grids)
        inp = merged_pp.sel(time=gridded.time.isel(time=win//2 + 1), method='nearest')
        inp.broadcast_like(inp.xy_0).to_array().interp_like(grid_ds.isel(time=5))

        # v = 'xy'
        # v = 'tx'
        v = 'yt'
        da = pp_grids[0][v]
        for ppg in pp_grids[1:]:
            da = da  + ppg[v]
        da.plot()
        grid_ds.ssh.plot.pcolormesh('lon', 'lat', col='time', col_wrap=3)

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        ...
        # fn = explore
        fn = first_training

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    locals().update(main())
