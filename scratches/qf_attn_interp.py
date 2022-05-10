# %% Import

import os
from dataclasses import dataclass, field
from pathlib import Path

import holoviews as hv
import holoviews.plotting.mpl  # noqa
import hydra
import math
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from einops import einops
from hydra.conf import ConfigStore
from hydra.experimental import compose, initialize
from hydra.types import TargetConf
from omegaconf import OmegaConf, DictConfig
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import importlib

importlib.reload(pl)
pl.LightningModule
print('done !')
# %% Config

@dataclass
class XpConfig:
    obs_file: str = '../sla-data-registry/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc'
    oi_file: str = '../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc'
    ref_file: str = '../sla-data-registry/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc'

    logger: dict = field(
        default_factory=lambda: dict(
            _target_='pytorch_lightning.loggers.TensorBoardLogger',
            save_dir='tb_logs',
            # version='',
            name='',
            default_hp_metric=False
        )
    )
    train_slice: tuple = (50, 90)
    test_slice: tuple = (65, 75)
    bs: int = 2048
    acc_batch: int = 1
    max_epochs: int = 10

    nmesh: int = 256
    n_signals: int = 1
    n_embd_per_head: int = 2
    n_attn_head: int = 32


cfg = OmegaConf.create({'xp':XpConfig()})


# %% data utils
def make_repeat_expr(data_dims, coords_dims):
    lh_expr = ' '.join(coords_dims)
    rh_expr = ' '.join(data_dims)
    return lh_expr + '->' + rh_expr


def make_flatten_expr(data_dims):
    lh_expr = ' '.join(data_dims)
    rh_expr = '(' + lh_expr + ')'
    return lh_expr + '->' + rh_expr


def make_unflatten_expr(data_dims):
    rh_expr = ' '.join(data_dims)
    lh_expr = '(' + rh_expr + ')'
    return lh_expr + '->' + rh_expr


def coords_to_dim(ds, dims=('time',), drop='x'):
    df = ds.to_dataframe()

    for d in dims:
        df = df.set_index(d, append=True)

    return (
        df
            .droplevel(drop)
            .pipe(lambda ddf: xr.Dataset.from_dataframe(ddf))
    )


def ds_to_data(ds, data_vars=('ssh',), coord_vars=('time', 'lat', 'lon'), coord_norm=None):
    data_dims = ds[data_vars[0]].dims

    data = [
        einops.rearrange(ds[var].data, make_flatten_expr(data_dims))
        for var in data_vars
    ]

    bdcasted_coords = [
        einops.repeat(
            ds[coord_var].data,
            make_repeat_expr(data_dims, ds[coord_var].dims),
            **ds.dims
        ) for coord_var in coord_vars
    ]

    coords = [
        einops.rearrange(bcast_datum, make_flatten_expr(data_dims))
        for bcast_datum in bdcasted_coords
    ]

    _tensor_coords = einops.rearrange([
        torch.from_numpy(dat.astype(np.float32)) for dat in coords
    ], 'c x -> x c')

    if coord_norm is None:
        coords_mean = _tensor_coords.mean(0)
        coords_std = _tensor_coords.std(0)
        coord_norm = lambda t: (t - coords_mean) / coords_std

    tensor_coords = coord_norm(_tensor_coords)

    tensor_data = einops.rearrange([
        torch.from_numpy(dat.astype(np.float32)) for dat in data
    ], 'c x -> x c')

    return tensor_coords, tensor_data, coord_norm


# %% data
class InterpDataset(Dataset):
    def __init__(
            self,
            xrds,
            data_vars=('ssh',),
            coord_vars=('time', 'lat', 'lon'),
            coord_norm=None,
    ):
        super(InterpDataset, self).__init__()
        self.xrds = xrds
            
        self.coords, self.values, self.coord_norm = ds_to_data(self.xrds, data_vars, coord_vars, coord_norm=coord_norm)
        self.ds = TensorDataset(
            self.coords[~torch.isnan(self.values[:, 0]), :],
            self.values[~torch.isnan(self.values[:, 0]), :],
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        return self.ds[item]


# %%  model utils
class Repeat(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Repeat, self).__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return einops.repeat(x, **self.kwargs)


def exists(val):
    return val is not None


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

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
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1., w0_initial=30., use_bias=True,
                 final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias,
                                activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


# %% models

class AttnInterp(torch.nn.Module):
    def __init__(self, k_net, v_net, last_layer, query, attn):
        super().__init__()
        self.k_net = k_net
        self.v_net = v_net
        self.last_layer = last_layer
        self.query = query
        self.attn = attn

    def forward(self, rel_coords, src_values):
        """

        Parameters
        ----------
        rel_coords batch_size x n_obs x n_coords
        src_values n_obs x n_values

        Returns
        -------
            tgt_values: batch_size x n_values
        """
        shape = einops.parse_shape(rel_coords, "bs nobs ncoord")
        key = einops.rearrange(self.k_net(rel_coords), 'bs nobs c_embd -> nobs bs c_embd')
        value = einops.repeat(self.v_net(src_values), 'nobs v_embd -> nobs bs v_embd', bs=shape['bs'])
        query = einops.repeat(self.query, 'nlat c_embd -> nlat bs c_embd', bs=shape['bs'])
        attn_out, _ = self.attn(query, key, value)
        tgt_values = einops.rearrange(attn_out, 'n_lat bs vembd -> bs n_lat vembd')
        return einops.reduce(self.last_layer(tgt_values), 'bs n_lat nval -> bs nval', 'sum')


def to_rel(tgt_coords, src_coords):
    return tgt_coords[:, None, ...] - src_coords[None, ...]


class MeshInterpolator(nn.Module):
    def __init__(self, nmesh, *args, **kwargs):
        super().__init__()
        self.interpolator = AttnInterp(*args, **kwargs)
        self.nmesh = nmesh
        self.mesh_coords = nn.Parameter(torch.rand(nmesh, 3))
        self.mesh_values = nn.Parameter(torch.rand(nmesh, 1))

    def forward(self, tgt_coords, mesh_sample_prop=1.):
        idx_samp = torch.randperm(self.nmesh)[:int(mesh_sample_prop * self.nmesh)]
        rel_coords = to_rel(tgt_coords, self.mesh_coords[idx_samp, ...])
        return self.interpolator.forward(rel_coords, self.mesh_values[idx_samp, ...])


# %% latent models
class LatentAttnInterp(torch.nn.Module):
    def __init__(self, v_latent, k_net, last_layer, query, attn):
        super().__init__()
        self.v_latent = v_latent
        self.k_net = k_net
        self.last_layer = last_layer
        self.query = query
        self.attn = attn

    def forward(self, rel_coords, *args, **kwargs):
        """

        Parameters
        ----------
        rel_coords batch_size x n_obs x n_coords
        src_values n_obs x n_values

        Returns
        -------
            tgt_values: batch_size x n_values
        """
        shape = einops.parse_shape(rel_coords, "bs nobs ncoord")
        key = einops.rearrange(self.k_net(rel_coords), 'bs nobs c_embd -> nobs bs c_embd')
        value = einops.repeat(self.v_latent, 'nobs v_embd -> nobs bs v_embd', bs=shape['bs'])
        query = einops.repeat(self.query, 'nlat c_embd -> nlat bs c_embd', bs=shape['bs'])

        attn_out, _ = self.attn(query, key, value)
        tgt_values = einops.rearrange(attn_out, 'n_lat bs vembd -> bs n_lat vembd')
        return einops.reduce(self.last_layer(tgt_values), 'bs n_lat nval -> bs nval', 'sum')


class LatentMeshInterpolator(nn.Module):
    def __init__(self, nmesh, n_embd, n_coords=3, *args, **kwargs):
        super().__init__()
        self.nmesh = nmesh
        self.mesh_coords = nn.Parameter(torch.rand(nmesh, n_coords))
        self.mesh_values = nn.Parameter(torch.rand(nmesh, n_embd))
        self.interpolator = LatentAttnInterp(v_latent=self.mesh_values, *args, **kwargs)

    def forward(self, tgt_coords, *args, **kwargs):
        rel_coords = to_rel(tgt_coords, self.mesh_coords)
        return self.interpolator.forward(rel_coords, self.mesh_values)


# %% metrics utils


def to_sobel(osse_item: xr.Dataset) -> xr.Dataset:
    def ufunc(da: xr.DataArray):
        nans = np.isnan(da)
        da_wo_na = np.nan_to_num(da)
        grad = np.sqrt(ndimage.sobel(da_wo_na, axis=-1) ** 2 + ndimage.sobel(da_wo_na, axis=-2) ** 2)
        grad[nans] = np.nan
        return grad

    osse_item = osse_item.map(
        lambda da: xr.apply_ufunc(ufunc, da, keep_attrs=True),
        keep_attrs=True
    )
    return osse_item


def metrics(ds):
    se = (ds - ds.gt) ** 2
    mse = se.to_dataframe().mean()._set_name('mse')
    nmse = (se / (np.abs(ds.gt) + 10 ** -3)).to_dataframe().mean()._set_name('nmse')
    grad_ds = ds.pipe(to_sobel)
    grad_se = (grad_ds - grad_ds.gt) ** 2
    grad_mse = grad_se.to_dataframe().mean()._set_name('grad_mse')

    metric_df = pd.concat([
        mse, nmse, grad_mse
    ], axis=1).drop('gt')

    formatted_metrics_df = (
        metric_df.T
            .assign(wrt_oi=lambda df: df.pred / df.oi * 100)
            .drop('oi', axis=1)
            .pipe(lambda df: pd.concat([
            df.pred,
            df[['wrt_oi']].assign(new_idx=lambda df: df.index + '_wrt_oi').set_index('new_idx').wrt_oi
        ])
                  )
    )
    print(metric_df.to_markdown())
    return formatted_metrics_df


# %% pl mod
class LitInterp(pl.LightningModule):
    def __init__(self, gen_mod, disc_mod, cfg, oi_ds=None):
        super(LitInterp, self).__init__()
        self.cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
        self.save_hyperparameters({**self.cfg})

        self.gen_mod = gen_mod
        self.disc_mod = disc_mod
        self.test_ds = None
        self.test_fig = {}
        self.oi_ds = oi_ds
        self.automatic_optimization = False

    def forward(self, coords):
        return self.gen_mod(coords)

    def training_step(self, batch, batch_idx):

        if  (self.current_epoch > 3) and (batch_idx % 50 < 10):

            tgt_coords, tgt_values = batch
            tgt_pred = self.gen_mod(tgt_coords, mesh_sample_prop=1.)

            

            true_obs_score = nn.BCEWithLogitsLoss()(
                    self.disc_mod(torch.cat((tgt_coords, tgt_values), dim=1), mesh_sample_prop=0.6),
                    torch.ones_like(tgt_pred),
            )

            pred_score = nn.BCEWithLogitsLoss()(
                    self.disc_mod(torch.cat((tgt_coords, tgt_pred), dim=1), mesh_sample_prop=0.6),
                    torch.zeros_like(tgt_pred),
            )
            disc_loss = pred_score + true_obs_score
            _, opt_disc = self.optimizers()

            # mse = F.mse_loss(tgt_pred, tgt_values)
            opt_disc.zero_grad()
            # self.manual_backward(disc_loss + mse)
            self.manual_backward(disc_loss)
            opt_disc.step()

            self.log('disc_loss', disc_loss, logger=True, prog_bar=True)
            self.log('true_obs', true_obs_score, logger=True, prog_bar=True)
        
        if True or batch_idx // 50 % 2 == 1:
        
            tgt_coords, tgt_values = batch
            opt_gen, _ = self.optimizers()
            tgt_pred = self.gen_mod(tgt_coords, mesh_sample_prop=0.9)
            gen_loss = nn.BCEWithLogitsLoss()(
                    self.disc_mod(torch.cat((tgt_coords, tgt_pred), dim=1)),
                    torch.ones_like(tgt_pred),
            )

            mse = F.mse_loss(tgt_pred, tgt_values)
            opt_gen.zero_grad()
            # self.manual_backward(2 * gen_loss + mse)
            self.manual_backward(gen_loss + 10 * mse)
            opt_gen.step()
            self.log('gen_loss', gen_loss, logger=True, prog_bar=True)
            self.log('mse', mse, logger=True, prog_bar=True)

        # norm_factor = torch.where(
        #     torch.abs(tgt_values) < 10 ** -1,
        #     torch.full_like(tgt_values, 10 ** -1),
        #     torch.abs(tgt_values)
        # )
        # nmse = torch.mean((tgt_pred - tgt_values) ** 2 / norm_factor)

        # loss = mse# + 2 * mse

        # loss = mse
        # self.log('nmse', nmse, logger=True, prog_bar=True)
        # self.log('loss', loss, logger=True)


    def test_step(self, batch, batch_idx):
        tgt_coords, tgt_values = batch
        tgt_pred = self.gen_mod(tgt_coords)
        self.log('test_loss', F.mse_loss(tgt_pred, tgt_values))
        return {'coords': tgt_coords.detach().cpu(),
                'preds': tgt_pred.detach().cpu(),
                'values': tgt_values.detach().cpu()}

    def test_epoch_end(self, outputs):
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        coords = torch.cat([chunk['coords'] for chunk in outputs]).numpy()
        gt = torch.cat([chunk['values'] for chunk in outputs]).numpy()
        self.test_ds = xr.Dataset(
            data_vars={
                'pred': (('x',), pred[..., 0]),
                'gt': (('x',), gt[..., 0]),
            },
            coords={
                c: (('x',), coords[..., i]) for i, c in enumerate(['time', 'lat', 'lon'])
            }
        ).pipe(lambda ds: coords_to_dim(ds, ['time', 'lat', 'lon']))

        self.test_ds['oi'] = (
            self.oi_ds.ssh_mod.dims,
            self.oi_ds.ssh_mod.data
        )

        to_plot_ds = self.test_ds.pipe(to_sobel).isel(time=1)
        hv_layout = hv.Layout([
            hv.Dataset(
                to_plot_ds, ['lon', 'lat'], var
            ).to(
                hv.QuadMesh, kdims=['lon', 'lat']
            ).relabel(
                f'{var}'
            ).options(
                colorbar=True,
                cmap='PiYG',
                clim=(to_plot_ds.gt.min(), to_plot_ds.gt.max())
            )
            for var in to_plot_ds.data_vars
        ]).cols(3)
        fig = hv.render(hv_layout, backend='matplotlib')
        self.logger.experiment.add_figure('Error reduction', fig, global_step=self.current_epoch)
        self.test_fig['err_red'] = fig
        mdf = metrics(self.test_ds.isel(lat=slice(20, -20), lon=slice(20, -20)))
        self.logger.log_hyperparams(
            {**self.cfg},
            mdf.to_dict(),
        )

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.gen_mod.parameters(), lr=0.001)# , weight_decay=0.001)
        disc_opt = torch.optim.Adam(self.disc_mod.parameters(), lr=0.001, weight_decay=0.001)
        return gen_opt, disc_opt

    # def configure_optimizers(self, ):
    #     opt = torch.optim.Adam(self.parameters(), lr=0.01)# , weight_decay=0.001)
    #     return {
    #         'optimizer': opt,
    #         'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=10,
    #                                                                    threshold=10 ** -3),
    #         'monitor': 'loss'
    #     }



def main():
    domain = {'lat': slice(32, 44), 'lon':slice(-66, -54)}
    # pl.seed_everything(32)
    obs_ds = xr.open_dataset(cfg.xp.obs_file)
    oi_ds = xr.open_dataset(cfg.xp.oi_file).pipe(lambda da: da.where(np.abs(da)<100, np.nan))
    ref_ds = xr.open_dataset(cfg.xp.ref_file, decode_cf=False)
    ref_ds['time'] = obs_ds.time

    # %% instant dataset

    train_slice = slice(*cfg.xp.train_slice)
    test_slice = slice(*cfg.xp.test_slice)
    train_ds = InterpDataset(obs_ds.sel(domain).isel(time=train_slice), data_vars=('ssh_mod',))
    test_ds = InterpDataset(ref_ds.sel(domain).isel(time=test_slice), coord_norm=train_ds.coord_norm)

    train_dl = DataLoader(train_ds, batch_size=cfg.xp.bs, shuffle=True, num_workers=10)
    test_dl = DataLoader(test_ds, batch_size=cfg.xp.bs, num_workers=10)

    # %% instant_models
    n_embd = cfg.xp.n_embd_per_head * cfg.xp.n_attn_head
    k_net = Siren(dim_in=3, dim_out=n_embd, w0=30.)
    query = nn.Parameter(torch.randn(cfg.xp.n_signals, n_embd))
    # last_layer = Siren(dim_in=n_embd, dim_out=1, w0=1.)
    last_layer = SirenNet(dim_in=n_embd, dim_hidden=256, dim_out=1, num_layers=2, w0=1.)
    attn = nn.MultiheadAttention(
        n_embd,
        num_heads=cfg.xp.n_attn_head
    )

    gen_mod = LatentMeshInterpolator(
        nmesh=cfg.xp.nmesh,
        n_embd=n_embd,
        k_net=k_net,
        query=query,
        last_layer=last_layer,
        attn=attn
    )


    n_embd = cfg.xp.n_embd_per_head * cfg.xp.n_attn_head
    k_net = Siren(dim_in=4, dim_out=n_embd, w0=30.)
    query = nn.Parameter(torch.randn(cfg.xp.n_signals, n_embd))
    # last_layer = Siren(dim_in=n_embd, dim_out=1, w0=1.)
    last_layer = SirenNet(dim_in=n_embd, dim_hidden=256, dim_out=1, num_layers=2, w0=1.)
    attn = nn.MultiheadAttention(
        n_embd,
        num_heads=cfg.xp.n_attn_head
    )

    disc_mod = LatentMeshInterpolator(
        nmesh=cfg.xp.nmesh,
        n_coords=4,
        n_embd=n_embd,
        k_net=k_net,
        query=query,
        last_layer=last_layer,
        attn=attn
    )

    # %% training
    lit_mod = LitInterp(gen_mod=gen_mod, disc_mod=disc_mod, cfg=cfg.xp, oi_ds=oi_ds.sel(domain).isel(time=test_slice))

    # %%  Training 1

    # Logger
    logger = hydra.utils.instantiate(cfg.xp.logger)
    callbacks = pl.callbacks.RichProgressBar()
    trainer = pl.Trainer(
        gpus=1,
        accumulate_grad_batches=cfg.xp.acc_batch,
        # callbacks=[pl.callbacks.StochasticWeightAveraging()],
        callbacks=callbacks,
        max_epochs=cfg.xp.max_epochs,
        deterministic=True,
        logger=logger
    )
    trainer.fit(lit_mod, train_dataloader=train_dl)
    trainer.test(lit_mod, test_dataloaders=test_dl)
    #
    return locals()




    
# %%
if __name__ == '__main__':
    # scratch_dir = Path('stages/attn_interp')
    # if not str(Path.cwd()).endswith(str(scratch_dir)):
    #     scratch_dir.mkdir(parents=True, exist_ok=True)
    #     os.chdir(scratch_dir)



    # # %% Xp dvc
    # os.environ['PATH'] = os.environ['PATH'] + ':/home/quentin/research/research-quentin/.conda_env/bin'

    cfg = OmegaConf.create({'xp':XpConfig()})

    # main(cfg)


