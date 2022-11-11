import utils
import pickle
import functorch
import metpy.calc as mpcalc
import itertools as it
import math
import siren_pytorch
import functools as ft
import pl_bolts.models.autoencoders
import pl_bolts.models.autoencoders.components as aecomp
import pl_bolts.models.gans.dcgan.components as gancomp
import einops
import tqdm
from tqdm import tqdm
from einops.layers.torch import Rearrange, Reduce
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import matplotlib.pyplot as plt
import pyinterp
import pyinterp.backends.xarray
import pyinterp.fill
import numpy as np
import xarray as xr
import traceback
import hydra_config
import sys
from omegaconf import OmegaConf
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('scratches')
from qf_base_xr_to_torch import XrDataset, XrConcatDataset
from collections import namedtuple, OrderedDict
base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    f'file_paths={fp}'
]


def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)


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
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

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

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= einops.rearrange(mod, 'b d -> b () d')

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

def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

TrainingItem = namedtuple('TrainingItem', ['ssh'])


def get_constant_crop(patch_size, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_size[d] for d in dim_order], dtype='float32')
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        return patch_weight

def build_dataloaders(path, patch_dims, strides, train_period, val_period, ds=None, batch_size=16):
    inp_ds = xr.open_dataset(path)
    perm = np.random.permutation(inp_ds.dims['time'])

    downsampling = ds or dict()
    _pre_fns = [
            lambda ds: ds.coarsen(downsampling, boundary='trim').mean(),
    ]
    _post_fns = [
        TrainingItem._make
    ]
    pre_fn = ft.partial(ft.reduce,lambda i, f: f(i), _pre_fns)
    post_fn = ft.partial(ft.reduce,lambda i, f: f(i), _post_fns)
    new_ds = inp_ds.assign(
        shuffled_nadir_obs=(
            inp_ds.ssh.dims, 
            np.where(np.isfinite(inp_ds.nadir_obs.values[perm]), inp_ds.ssh.values, np.nan)
        )).assign(dict(
            # ssh=lambda ds: remove_nan(ds.ssh - ds.ssh.mean('time')),
            ssh=lambda ds: remove_nan(ds.ssh),
    )).assign(dict(
            # ssh=lambda ds: (ds.ssh - np.mean(ds.ssh.data)) / np.std(ds.ssh.data),
    ))[[*TrainingItem._fields]]
     
    train_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').pipe(pre_fn).to_array().sel(time=train_period),
        patch_dims=patch_dims, strides=strides,
        # prepro_fn=pre_fn,
        postpro_fn=post_fn,
    )
    val_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').pipe(pre_fn).to_array().sel(time=val_period),
        patch_dims=patch_dims, strides=strides,
        # prepro_fn=pre_fn,
        postpro_fn=post_fn,
    )
    print(f'{len(train_ds)=}, {len(val_ds)=}')
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1), \
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=1)

def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class LitFuncta(pl.LightningModule):
            def __init__(self, net_fn,  params, grad_net, rec_weight, latent_init, patch_dims):
                super().__init__()
                self.net_fn = lambda *args, **kwargs: net_fn(*args, **kwargs)
                self.grad_net = grad_net
                self.rec_weight = nn.Parameter(torch.from_numpy(rec_weight), requires_grad=False)
                self.coords = self.get_coords_mesh(patch_dims)                        
                self.latent_init = nn.Parameter(torch.from_numpy(latent_init), requires_grad=False)
                self.params = nn.ParameterList(list(params))
                self.latents = None
                self.test_data = None

                self.use_latents = False

                self.n_fit_latent_step = 3
                self.msk_prop_fit_latent_step = 0.1
                self.lr_fit_latent = 1e-2 
                self.lr_init, self.lr_min, self.lr_max = 2e-3, 2e-6, 1e-3

            def batchify(self, x, y):
                return einops.repeat(x, '... -> b ...', b=y.size(0))


            def get_coords_mesh(self, patch_dims):
                return nn.Parameter(torch.stack(torch.meshgrid(
                    *(torch.linspace(-1, 1, v) for v in patch_dims.values()), indexing='ij'
                ), dim=-1), requires_grad=False)


            @staticmethod
            def weighted_mse(err, weight):
                err_w = (err * weight[None, ...])
                non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.
                err_num = err.isfinite() & ~non_zeros
                if err_num.sum() == 0:
                    return torch.scalar_tensor(0., device=err_num.device)
                loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
                return loss

            def rec_loss(self, inp, tgt):
                rec = self(inp) 
                return self.weighted_mse(rec - tgt, self.rec_weight)

            def training_step(self, batch, batch_idx, optimizer_idx=None):
                return self.step(batch, 'tr', optimizer_idx, training=True)[0]

            def validation_step(self, batch, batch_idx):
                return self.step(batch, 'val')[0]
            
            def forward(self, latent=None, batch=None, coords=None, training=False):
                if latent is None:
                    latent = self.fit_latent(batch, training=training, msk_prop=self.msk_prop_fit_latent_step)
                if coords is None:
                    coords = self.batchify(self.coords, latent).detach().requires_grad_()

                return self.net_fn(self.params, (coords, latent)), latent


            def fit_latent(self, batch, training=False, msk_prop=0.1):
                tgt = batch.ssh[..., None]
                msk = torch.rand_like(tgt).greater_equal_(msk_prop).int()
                with torch.enable_grad():
                    inp_c = self.batchify(self.coords, tgt).detach().requires_grad_()
                    latent = self.batchify(self.latent_init, tgt).detach().requires_grad_()
                    for _ in range(3):
                        out = self.net_fn(self.params, (inp_c, latent))
                        loss_inner = F.mse_loss(out*msk, tgt*msk)
                        grad_lat = torch.autograd.grad(loss_inner, (latent,), create_graph=True)[0]
                        latent = latent - 0.01 * (grad_lat + self.grad_net(grad_lat))
                        if not training:
                            latent.detach_().requires_grad_()
                return latent

            def step(self, batch, phase='', opt_idx=None, training=False):
                out, latent = self(batch=batch, training=training)
                loss_outer = self.weighted_mse(out.squeeze(-1) - batch.ssh, self.rec_weight)
                self.log(f'{phase}_loss', loss_outer, prog_bar=True, on_step=False, on_epoch=True)
                return loss_outer, out, latent

            def configure_optimizers(self):
                # return torch.optim.Adam(self.params, lr=3e-6)
                opt = torch.optim.Adam(
                        [{'params': self.parameters(), 'initial_lr': 0.002}],
                        lr=0.002)
                # opt = torch.optim.SGD(self.parameters(), lr=self.lr_init)
                return {
                    'optimizer': opt,
                    'lr_scheduler':
                    # torch.optim.lr_scheduler.ReduceLROnPlateau(
                    #     opt, verbose=True, factor=0.5, min_lr=1e-6, cooldown=5, patience=50,
                    # ),
                    # 'monitor': 'tr_loss',
                    # torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=5e-5, T_max=20),
                    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                    #     eta_min=5e-5, T_0=15, T_mult=2, last_epoch=-1),
                    torch.optim.lr_scheduler.CyclicLR(
                        opt, base_lr=2e-6, max_lr=1e-3,  step_size_up=15, step_size_down=25, cycle_momentum=False, mode='triangular'),
                }

            def test_step(self, batch, batch_idx):
                
                if self.use_latents:
                    bs = batch.ssh.shape[0]
                    batch_latents = torch.stack(self.latents[batch_idx *bs: (batch_idx+1) * bs], dim=0).to(self.device)
                    out, latent = self(latent=batch_latents)
                else:
                    out, latent = self(batch=batch)

                return torch.stack([
                    batch.ssh.cpu(),
                    out.squeeze(dim=-1).detach().cpu(),
                ],dim=1), latent.cpu()


            def test_epoch_end(self, outputs):
                rec_data, latent = (it.chain(lt) for lt in zip(*outputs))
                rec_da = (
                    self.trainer
                    .test_dataloaders[0].dataset
                    .reconstruct(rec_data, self.rec_weight.cpu().numpy())
                )
                npa = rec_da.values
                lonidx = ~np.all(np.isnan(npa), axis=tuple([0, 1, 2]))
                latidx = ~np.all(np.isnan(npa), axis=tuple([0, 1, 3]))
                tidx = ~np.all(np.isnan(npa), axis=tuple([0, 2, 3]))

                self.test_data = xr.Dataset({
                    k: rec_da.isel(v0=i, time=tidx, lat=latidx, lon=lonidx)
                    for i,k  in enumerate(['ssh', 'rec_ssh'])
                })

                self.latents = list(it.chain(*latent))



class SirenWrapper(nn.Module):
    def __init__(self, net, mod, patch_dims):
        super().__init__()
        self.net, self.mod = net, mod
        self.patch_dims = patch_dims

        self.flt = Rearrange('b time lat lon ... -> b (time lat lon) ...')
        self.unflt = Rearrange('b (time lat lon) ... -> b time lat lon ...', **patch_dims)

    def forward(self, inputs):
        coords, lat = inputs
        mods = self.mod(lat)
        out = self.net(self.flt(coords), mods)
        return self.unflt(out)


def run1():
    try:
        cfg = utils.get_cfg(base_cfg, overrides=overrides)
        # print(OmegaConf.to_yaml(cfg.file_paths))
        print(cfg.file_paths.data_registry_path)

        xr.open_dataset( f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',).ssh.std()

        strides = dict(time=1, lat=240, lon=240)
        _patch_dims = dict(time=3, lat=240, lon=240)
        ds = dict(time=1, lat=4, lon=4)
        patch_dims = {k: _patch_dims[k]//ds.get(k,1) for k in _patch_dims}
        # crop = dict(time=2, lat=4, lon=4)
        crop = dict()


        train_period = slice('2012-10-01', '2013-06-30')
        # train_period = slice('2013-07-01', '2013-08-30')
        val_period = slice('2013-07-01', '2013-08-30')
        train_dl, val_dl = build_dataloaders(
            f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',
            patch_dims,
            strides,
            train_period,
            val_period,
            ds=ds
        )


        latent_dim = 128
        net = SirenNet(dim_in=3, dim_hidden=256, dim_out=1, num_layers=5)
        mod = Modulator(dim_in=latent_dim, dim_hidden=256, num_layers=5)
        grad_net = nn.Identity()

        # grad_net = nn.Sequential(
        #     nn.BatchNorm1d(latent_dim),
        #     nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )

        coords = torch.stack(torch.meshgrid(
            *(torch.linspace(-1, 1, v) for v in patch_dims.values()), indexing='ij'
        ), dim=-1)
        latent_init = nn.Parameter(torch.zeros(latent_dim))


        net_fn, params = functorch.make_functional(SirenWrapper(net, mod, patch_dims))
         

        vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
        geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()
        rec_weight = get_constant_crop(patch_dims, crop) 
        lit_mod = LitFuncta(
                net_fn=net_fn, params=params, grad_net=grad_net, rec_weight=rec_weight, patch_dims=patch_dims, latent_init=np.zeros(latent_dim, dtype=np.float32))

        pl.seed_everything(333)
        callbacks=[
            plcb.ModelCheckpoint(monitor='val_loss', save_last=True),
            plcb.TQDMProgressBar(),
            # plcb.GradientAccumulationScheduler({10:2, 15:4, 25:8}),
            # plcb.StochasticWeightAveraging(),
            # plcb.RichProgressBar(),
            plcb.ModelSummary(max_depth=2),
            # plcb.GradientAccumulationScheduler({50: 10})
        ]

        # trainer = pl.Trainer(gpus=[1], logger=False, callbacks=callbacks, max_epochs=150,
        #      limit_train_batches=2,
        # )
        # trainer.fit(lit_mod, train_dataloaders=train_dl)

        trainer = pl.Trainer(gpus=[1], logger=False, callbacks=callbacks, max_epochs=450,
             limit_train_batches=10,
        )
        trainer.fit(lit_mod, train_dataloaders=train_dl)

        trainer = pl.Trainer(gpus=[1], logger=False, callbacks=callbacks, max_epochs=50,
             limit_train_batches=15,
        )

        # trainer.fit(lit_mod, train_dataloaders=train_dl, val_dataloaders=val_dl)
        # trainer = pl.Trainer(gpus=[1], callbacks=callbacks, max_epochs=2000,)

        trainer.fit(lit_mod, train_dataloaders=train_dl, val_dataloaders=val_dl)
        # torch.save(lit_mod.state_dict(), 'tmp/siren_functa_model.t')
        # lit_mod.load_state_dict(torch.load('tmp/siren_functa_model.t'))
        # trainer.fit(lit_mod, train_dataloaders=val_dl)#, val_dataloaders=val_dl)
        trainer.checkpoint_callback.best_model_score
        lit_mod.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
        trainer.test(lit_mod, dataloaders=[val_dl])
        # trainer.test(lit_mod, dataloaders=[train_dl], ckpt_path=trainer.checkpoint_callback.best_model_path)
        lit_mod.test_data.to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')
        lit_mod.test_data.map(geo_energy).to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')
        lit_mod.test_data.map(vort).to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')

        # lit_mod.test_data.pipe(lambda ds: ds-ds.ssh).drop('ssh').to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')


        # trainer.test(lit_mod, dataloaders=[train_dl])
        # lit_mod.test_data.to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(col='variable', row='time')
        # lit_mod.test_data.pipe(lambda ds: ds.ssh - ds.rec_ssh).pipe(lambda da: np.sqrt(np.mean(da**2)))

        # _, up_val_dl = build_dataloaders(
        #     f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',
        #     _patch_dims,
        #     strides,
        #     train_period,
        #     val_period,
        #     ds=dict(time=1, lat=1, lon=1),
        #     batch_size=2,
        # )
        # up_batch = next(iter(up_val_dl))
        # up_batch.ssh.shape
        # up_net_fn, _ = functorch.make_functional(SirenWrapper(net, mod, _patch_dims))
        # up_rec_weight = get_constant_crop(_patch_dims, crop) 
        # up_lit_mod = LitFuncta(
        #         net_fn=up_net_fn, params=params,grad_net=grad_net, rec_weight=up_rec_weight, patch_dims=_patch_dims, latent_init=np.zeros(latent_dim, dtype=np.float32))
        
        # state_dict = torch.load('tmp/siren_functa_model.t')

        # del state_dict['rec_weight']
        # del state_dict['coords']
        # up_lit_mod.load_state_dict(state_dict, strict=False)

        # up_lit_mod.latents = lit_mod.latents
        # up_lit_mod.use_latents = True
        # trainer.test(up_lit_mod, dataloaders=[up_val_dl])
        # up_lit_mod.use_latents = False
        # up_lit_mod.test_data.to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(col='variable', row='time')

        # trainer.fit(up_lit_mod, train_dataloaders=up_val_dl)
    

        # plt.imshow(up_lit_mod(latent=lit_mod.latents[0])[0][0,1,...].detach())
        # plt.imshow(lit_mod(latent=lit_mod.latents[0])[0][0,1,...].detach())

        # torch.unbind(0)
        # list(it.chain(map(ft.partial(torch.unbind, dim=0), lit_mod.latents)))[0].shape
        # up_lit_mod.latents[0] 
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def optuna():
    try:
        import optuna

        cfg = utils.get_cfg(base_cfg, overrides=overrides)

        strides = dict(time=1, lat=240, lon=240)
        _patch_dims = dict(time=3, lat=240, lon=240)
        ds = dict(time=1, lat=2, lon=2)
        patch_dims = {k: _patch_dims[k]//ds.get(k,1) for k in _patch_dims}
        crop = dict()


        train_period = slice('2012-10-01', '2013-06-30')
        val_period = slice('2013-07-01', '2013-08-30')
        train_dl, val_dl = build_dataloaders(
            f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',
            patch_dims, strides, train_period, val_period, ds=ds
        )
        coords = torch.stack(torch.meshgrid(
            *(torch.linspace(-1, 1, v) for v in patch_dims.values()), indexing='ij'
        ), dim=-1)

        def opt_obj(trial):
            latent_dim = trial.suggest_int('latent_dim', 32, 1024)
            dim_hidden = trial.suggest_int('dim_hidden', 64, 512)
            num_layers = trial.suggest_int('num_layers', 3, 8, step=2)
            fit_lat_iter = trial.suggest_int('fit_lat_iter', 1, 5)
            mask_prop = trial.suggest_float('mask_prop', 0.1, 0.7)
            train_batches = 1.# trial.suggest_int('train_batches', 2, 20, log=True)
            print(f'{(latent_dim, dim_hidden, num_layers, fit_lat_iter, mask_prop)=}')

            net = SirenNet(dim_in=3, dim_hidden=dim_hidden, dim_out=1, num_layers=num_layers)
            mod = Modulator(dim_in=latent_dim, dim_hidden=dim_hidden, num_layers=num_layers)
            grad_net = nn.Identity()
            net_fn, params = functorch.make_functional(SirenWrapper(net, mod, patch_dims))
             

            rec_weight = get_constant_crop(patch_dims, crop) 
            lit_mod = LitFuncta(
                    net_fn=net_fn,
                    params=params,
                    grad_net=grad_net,
                    rec_weight=rec_weight,
                    patch_dims=patch_dims,
                    latent_init=np.zeros(latent_dim, dtype=np.float32)
            )
            lit_mod.n_fit_latent_step = fit_lat_iter
            lit_mod.msk_prop_fit_latent_step = mask_prop

            pl.seed_everything(333)
            callbacks=[
                plcb.ModelCheckpoint(monitor='val_loss', save_last=True),
                plcb.TQDMProgressBar(),
                plcb.ModelSummary(max_depth=2),
            ]


            trainer = pl.Trainer(gpus=[3], logger=False, callbacks=callbacks, max_epochs=500,
                 limit_train_batches=train_batches,
            )
            try:
                trainer.fit(lit_mod, train_dataloaders=train_dl, val_dataloaders=val_dl)
            except Exception:
                return 100.
            return trainer.checkpoint_callback.best_model_score

        study = optuna.create_study(storage='sqlite:///tmp/optuna.db')
        study.optimize(opt_obj, n_trials=100)
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1
        fn = optuna

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    ...
    locals().update(main())
