import hydra
from einops.layers.torch import Rearrange, Reduce
import pickle
import seaborn as sns
import xrft
from functools import cache
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


class SwotOverlapDataset(torch.utils.data.Dataset):
    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    def __init__(self, spat_time_domain, min_timestep, sigmas, stats=None):
        super().__init__()
        self.return_coords = False
        cycle_period = pd.to_timedelta('1814429992676000ns')
        xrgf = lambda da, sig: (
            da if sig==0 else
            xr.apply_ufunc(
                lambda nda: ndi.gaussian_filter1d(
                    nda,
                    axis=0,
                    sigma=sig,
                    order=0,
                    mode='mirror',
                    truncate=3.0
                ),
                da
            )
        )

        swath_data = get_swot_slice('../sla-data-registry/sensor_zarr/zarr/new_swot', **spat_time_domain, drop_vars=[
            'bd_err', 'karin_noise', 'karin_err', 'phase_err', 'roll_err','timing_err', 'model_index',
            # 'lat', 'lat_nadir', 'lon', 'lon_nadir', 'ssh_model', 'syst_error_uncalibrated', 'wet_tropo_res', 'x_ac', 'x_al'
        ]).assign(
            err =  lambda _g: _g.syst_error_uncalibrated + _g.wet_tropo_res,
        ).assign(
            obs = lambda _g:  _g.ssh_model + _g.err
        ).pipe(lambda ds: ds.isel(time=np.isfinite(ds.ssh_model).all('nC')))

        nadir_data = (
                get_nadir_slice('../sla-data-registry/sensor_zarr/zarr/nadir/swot', **spat_time_domain)
                .sel(time=swath_data.time, method='nearest')
                .assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum().bfill('time'))
        )
        chunks = list(
                nadir_data
                .groupby('contiguous_chunk').count()
                .pipe(lambda ds: ds.isel(contiguous_chunk=ds.ssh_model > min_timestep))
                .contiguous_chunk.values
        )
        first_chunk = int(
            nadir_data.pipe(
                lambda ds: ds.isel(time=ds.contiguous_chunk==chunks[0])
            ).pipe(
                lambda ds: nadir_data.sel(time=ds.time + cycle_period, method='nearest'))
            .contiguous_chunk.values[0]
        )
        last_chunk = int(
            nadir_data.pipe(
                lambda ds: ds.isel(time=ds.contiguous_chunk==chunks[-1])
            ).pipe(
                lambda ds: nadir_data.sel(time=ds.time - cycle_period, method='nearest'))
            .contiguous_chunk.values[0]
        )
        chunk_cycle = chunks.index(first_chunk)
        sw_data_w_aug = (
            swath_data.sel(time=nadir_data.time, method='nearest')
            .assign(contiguous_chunk=(('time',), nadir_data.contiguous_chunk.data))
            .groupby('contiguous_chunk')
            .apply( lambda g: g.assign(
                    **{f'obs_{sig}' : lambda _g, sig=sig: xrgf(_g.obs, sig) for sig in sigmas},
            )).assign({
                    f'dobs_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'obs_{sig1}'] - ds[f'obs_{sig2}']
                    for sig1, sig2 in zip(sigmas[:-1], sigmas[1:])
            })
        )
        
        nad_data_w_aug = (
            nadir_data
            .assign(obs=lambda ds: ds.ssh_model)
            .groupby('contiguous_chunk')
            .apply( lambda g: g.assign(
                    **{f'obs_{sig}' : lambda _g, sig=sig: xrgf(_g.obs, sig) for sig in sigmas},
            )).assign({
                    f'dobs_{sig2}_{sig1}': lambda ds, sig1=sig1, sig2=sig2: ds[f'obs_{sig1}'] - ds[f'obs_{sig2}']
                    for sig1, sig2 in zip(sigmas[:-1], sigmas[1:])
            })
        )


        prev_vars, obs_vars, next_vars = [[f'd{src}_{sig2}_{sig1}'
                for sig1, sig2 in zip(sigmas[:-1], sigmas[1:])]
                for src in ['prev', 'obs', 'next']
        ]
        pp_vars = prev_vars + obs_vars + next_vars
        if stats is not None:
            sw_stats, nad_stats, gt_stats = stats
        else:
            sw_stats =  sw_data_w_aug[obs_vars].pipe(lambda ds: (ds.mean(), ds.std()))#.map(lambda s: np.maximum(s, 1e-2))))
            nad_stats =  nad_data_w_aug[obs_vars].pipe(lambda ds: (ds.mean(), ds.std()))#.map(lambda s: np.maximum(s, 1e-2))))
            gt_stats =  sw_data_w_aug[['ssh_model']].pipe(lambda ds: (ds.mean(), ds.std()))#.map(lambda s: np.maximum(s, 1e-2))))

        self.stats = (sw_stats, nad_stats, gt_stats)

        self._len = chunks.index(last_chunk) - chunk_cycle + 1
        self.nad_stats = nad_stats
        self.sw_stats = sw_stats
        self.gt_stats = gt_stats
        self.sw_data_w_aug = sw_data_w_aug
        self.nad_data_w_aug = nad_data_w_aug
        self.prev_vars = prev_vars
        self.obs_vars = obs_vars
        self.next_vars = next_vars
        self.pp_vars = pp_vars
        self.chunks = chunks
        self.chunk_cycle = chunk_cycle
        self.cycle_period = cycle_period

    def __len__(self):
        return self._len

    @cache
    def __getitem__(self, item):
        nad_stats = self.nad_stats
        gt_stats = self.gt_stats
        sw_data_w_aug = self.sw_data_w_aug
        nad_data_w_aug = self.nad_data_w_aug
        sw_stats = self.sw_stats
        prev_vars = self.prev_vars
        obs_vars = self.obs_vars
        next_vars = self.next_vars
        pp_vars = self.pp_vars
        chunks = self.chunks
        chunk_cycle = self.chunk_cycle
        cycle_period = self.cycle_period

        # sw_cs = []
        # nad_cs = []
        # for c in  [chunks[item], chunks[item + chunk_cycle], chunks[item + 2*chunk_cycle]]:
        #     nad_c.append(nad_data_w_aug.isel(time=nad_data_w_aug.contiguous_chunk==c)
        #             .pipe(lambda ds: ds - nad_stats[0] / nad_stats[1]))
        #     sw_cs.append(sw_data_w_aug.sel(time=nad_cs[-1].time, method='nearest')
        #             .pipe(lambda ds: ds - xr.merge([gt_stats[0], sw_stats[0]]) / xr.merge([gt_stats[1], sw_stats[1]])))
        first_sw_chunk = sw_data_w_aug.isel(time=sw_data_w_aug.contiguous_chunk==chunks[item])
        sw_cs = [
                first_sw_chunk.pipe(lambda ds: ds - xr.merge([gt_stats[0], sw_stats[0]]) / xr.merge([gt_stats[1], sw_stats[1]])),
                sw_data_w_aug.sel(time=first_sw_chunk.time + cycle_period, method='nearest').pipe(lambda ds: ds - xr.merge([gt_stats[0], sw_stats[0]]) / xr.merge([gt_stats[1], sw_stats[1]])),
                sw_data_w_aug.sel(time=first_sw_chunk.time + 2*cycle_period, method='nearest').pipe(lambda ds: ds - xr.merge([gt_stats[0], sw_stats[0]]) / xr.merge([gt_stats[1], sw_stats[1]])),
        ]
        first_nad_chunk = nad_data_w_aug.isel(time=nad_data_w_aug.contiguous_chunk==chunks[item])
        nad_cs = [
                first_nad_chunk.pipe(lambda ds: ds - nad_stats[0] / nad_stats[1]),
                nad_data_w_aug.sel(time=first_nad_chunk.time + cycle_period, method='nearest').pipe(lambda ds: ds - nad_stats[0] / nad_stats[1]),
                nad_data_w_aug.sel(time=first_nad_chunk.time + 2*cycle_period, method='nearest').pipe(lambda ds: ds - nad_stats[0] / nad_stats[1]),
        ]
        sw_chunk = sw_cs[1].assign(
            **{prev_v: lambda ds: (ds[obs_vars[0]].dims, sw_cs[0][obs_v].values) for prev_v, obs_v in zip(prev_vars, obs_vars)},
            **{next_v: lambda ds: (ds[obs_vars[0]].dims, sw_cs[0][obs_v].values) for next_v, obs_v in zip(next_vars, obs_vars)},
        )
        if self.return_coords:
            return sw_chunk
        nad_cs[2]
        nad_chunk = nad_cs[1].assign(
            **{prev_v: lambda ds: (ds[obs_vars[0]].dims, nad_cs[0][obs_v].values) for prev_v, obs_v in zip(prev_vars, obs_vars)},
            **{next_v: lambda ds: (ds[obs_vars[0]].dims, nad_cs[0][obs_v].values) for next_v, obs_v in zip(next_vars, obs_vars)},
        )
        return (
            sw_chunk.ssh_model.values.astype(np.float32),
            nad_chunk[pp_vars].to_array().values.astype(np.float32),
            sw_chunk[pp_vars].to_array().values.astype(np.float32),
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
    split_sides =True,
    norm_type = 'none',
    act_type = 'relu',
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
                    ConvSamePad(in_channels=nhidden,out_channels=nhidden, kernel_size=kernel_size, apply_per_side=split_sides),
                    norm(norm_type),
                    act(act_type=act_type),
                ), res=residual),
                mixer(mix, res=mix_residual),
        )
        for _ in range(depth) ],
    )
    net = nn.Sequential(
            ConvSamePad(in_channels=in_channels,out_channels=nhidden, kernel_size=1, apply_per_side=split_sides),
            norm(norm_type=norm_type),
            act(act_type=act_type),
            nn.Sequential(
                *[inner_net for _ in range(num_repeat)]
            ),
            ConvSamePad(in_channels=nhidden, out_channels=out_channels, kernel_size=1, apply_per_side=split_sides),
    )
    return net

class LitDirectCNN(pl.LightningModule):
            def __init__(
                    self,
                    net_kwargs,
                    len_pp,
                    nad_embed,
                    gt_stats,
                    stats,
                    lr_init=2e-3,
                    wd=1e-4,
                    loss_w=(.5, .3, .1),
                    log_loss=False,
                ):
                super().__init__()
                self.nadnet = nn.Sequential(
                       Rearrange('b c t -> b t c'),
                       nn.Linear(len_pp, 52 * nad_embed),
                       Rearrange('b t (c nC) -> b c t nC', c=nad_embed, nC=52),
                )
                self.loss_pp = lambda l: (l.log() if log_loss else l)
                self.net = build_net(in_channels=len_pp+nad_embed, out_channels=1,**net_kwargs)
                self.lr_init = lr_init
                self.wd = wd
                self.loss_w = loss_w
                self.gt_means = nn.Parameter(torch.from_numpy(gt_stats[0]), requires_grad=False)
                self.gt_stds = nn.Parameter(torch.from_numpy(gt_stats[1]), requires_grad=False)
                self.stats=stats
                self.save_hyperparameters()

            def forward(self, batch):
                gt, nad, sw = batch 
                nad_out = self.nadnet(nad)
                return self.net(torch.cat([nad_out, sw],dim=1))

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
                y, nad, sw  = batch
                out = self.forward(batch).sum(dim=1)
                losses = {}
                losses['err_tot'], losses['g_err_tot'], losses['l_err_tot'] = self.loss(out, y)

                rec_out = (out * self.gt_stds + self.gt_means)
                raw_gt = (y * self.gt_stds + self.gt_means)
                losses['err_rec'], losses['g_err_rec'], losses['l_err_rec'] = self.loss(rec_out, raw_gt)

                for ln, l in losses.items():
                    self.log(f'{phase}_{ln}', l)

                loss = (
                    self.loss_w[0] * self.loss_pp(losses['err_tot'])
                    + self.loss_w[1] * self.loss_pp(losses['g_err_tot'])
                    + self.loss_w[2] * self.loss_pp(losses['l_err_tot'])
                )
                self.log(f'{phase}_loss', loss, prog_bar=False)
                return loss
                
            def training_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='train')

            def validation_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='val')

            def predict_step(self, batch, batch_idx):
                out = self.forward(batch)
                rec_out = (out * self.gt_stds + self.gt_means).sum(dim=1)
                return rec_out.cpu().numpy()


            def configure_optimizers(self):
                opt = torch.optim.Adam(
                        [{'params': self.parameters(), 'initial_lr': self.lr_init}],
                        lr=self.lr_init, weight_decay=self.wd)
                return {
                    'optimizer': opt,
                    'lr_scheduler':
                    torch.optim.lr_scheduler.CyclicLR(
                        opt, base_lr=5e-5, max_lr=2e-3,  step_size_up=30, step_size_down=30, cycle_momentum=False, mode='triangular2'),
                    'monitor': 'val_loss'
                }

def full_swot_training():
    try:
        spat_domain = dict(
               lat_min=32,
               lat_max=44,
               lon_min=-66 + 360,
               lon_max=-54 + 360,
        )
        train_domain = dict(
               # time_min='2013-07-01', 
               time_min='2013-03-01', 
               time_max='2013-09-30',
               **spat_domain,
        )
        val_domain = dict(
               time_min='2012-12-09', 
               # time_max='2013-01-28',
               time_max='2013-02-28',
               **spat_domain,
        )
        min_timestep = 500
        sigmas = (0,*[(i+1)*20 for i in range(30)]) 
        ds = SwotOverlapDataset(train_domain, min_timestep, sigmas)
        val_ds = SwotOverlapDataset(val_domain, min_timestep, sigmas, stats=ds.stats)
        train_dl = torch.utils.data.DataLoader(ds)
        val_dl = torch.utils.data.DataLoader(val_ds)
        nad_embed=128
        net_kwargs = dict(
            nhidden = 128,
            depth = 3,
            kernel_size = 7,
            num_repeat = 1,
            residual = True,
            split_sides =True,
            norm_type = 'none',
            act_type = 'relu',
            mix = False,
            mix_residual = False,
            mix_act_type = 'none',
            mix_norm_type = 'none',
        )
        gt_stats=(ds.gt_stats[0].to_array().values,  ds.gt_stats[1].to_array().values)
        logger = pl.loggers.TensorBoardLogger('lightning_logs', name='swot_only')#, version='')
        trainer = pl.Trainer(
            gpus=[5],
            logger=logger,
            callbacks=[
                callbacks.LearningRateMonitor(),
                callbacks.RichProgressBar(),
                callbacks.ModelCheckpoint(monitor='val_loss', save_last=True),
                # callbacks.StochasticWeightAveraging(),
                callbacks.GradientAccumulationScheduler({1: 4, 10: 8, 15: 16, 200: 32}),#, 300: 50}),
                VersioningCallback()
            ],
            log_every_n_steps=10,
            max_epochs=1500,
        )

        lit_mod = LitDirectCNN(net_kwargs=net_kwargs,
                nad_embed=nad_embed, len_pp=len(ds.pp_vars), gt_stats=gt_stats,
                loss_w=(10., 5., 2.),
                lr_init=2e-4,
                wd=2e-5,
                log_loss=True,
                stats=ds.stats)
        trainer.fit(lit_mod,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl
        )

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def main():
    try:
        ...
        fn = full_swot_training

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()


def test():
    spat_domain = dict(
           lat_min=32,
           lat_max=44,
           lon_min=-66 + 360,
           lon_max=-54 + 360,
    )
    # test_domain = dict(
    #        # time_min='2013-07-01', 
    #        time_min='2012-10-01', 
    #        time_max='2012-12-24',
    #        **spat_domain,
    # )
    test_domain = dict( # VAL
           time_min='2012-12-09', 
           # time_max='2013-01-28',
           time_max='2013-02-28',
           **spat_domain,
    )
    ckpt = 'lightning_logs/swot_only/version_1/checkpoints/epoch=798-step=5183.ckpt'
    # ckpt = 'lightning_logs/swot_only/version_1/checkpoints/last.ckpt'
    lit_mod = LitDirectCNN.load_from_checkpoint(ckpt)
    min_timestep = 500
    sigmas = (0,*[(i+1)*8 for i in range(50)]) 
    test_ds = SwotOverlapDataset(test_domain, min_timestep, sigmas, stats=lit_mod.stats)
    test_dl = torch.utils.data.DataLoader(test_ds, shuffle=False)
    trainer = pl.Trainer(
        gpus=[7],
        logger=False,
        log_every_n_steps=10,
        max_epochs=1500,
    )
    val_pred = [p for pred in trainer.predict(lit_mod, test_dl) for p in pred]
    with test_ds.get_coords():
        coords = [
           test_ds.__getitem__.__wrapped__(test_ds, i)
           # test_ds[i]
           for i in range(len(test_ds))
        ]
    coords[0]
    pred_ds =  xr.concat(
            [
                coord.assign({'cal': lambda ds: (ds.ssh_model.dims, pred )})
                for pred, coord in zip(val_pred, coords)
            ], dim='time'
    ).assign(gt=lambda ds: (ds[['ssh_model']] * test_ds.gt_stats[1] +test_ds.gt_stats[0]).ssh_model)

    def sobel(da):
        dx_ac = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 0), da) /2
        dx_al = xr.apply_ufunc(lambda _da: ndi.sobel(_da, 1), da) /2
        return np.hypot(dx_ac, dx_al)
    (
        pred_ds
        .pipe(lambda ds: np.sqrt(((ds.gt - ds.cal)**2).mean()))
    )

    pred_ds.isel(time=slice(0, 600))[['gt', 'cal']].to_array().T.plot.pcolormesh('time', 'nC', col='variable', col_wrap=1, figsize=(15, 7))
    pred_ds.isel(time=slice(0, 600))[['gt', 'cal']].map(sobel).to_array().T.plot.pcolormesh('time', 'nC', col='variable', col_wrap=1, figsize=(15, 7))
    type(val_pred[0]), val_pred[0].shape
    type(val_pred), len(val_pred)

if __name__ == '__main__':
    locals().update(main())
