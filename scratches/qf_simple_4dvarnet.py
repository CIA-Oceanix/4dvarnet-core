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


def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])


def get_constant_crop(patch_size, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_size[d] for d in dim_order], dtype='float32')
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        return patch_weight

def build_dataloaders(path, patch_dims, strides, train_period, val_period, ds=None, batch_size=4):
    inp_ds = xr.open_dataset(path)
    new_ds = inp_ds.coarsen(ds).mean().assign(
        land_mask=lambda ds: np.isnan(ds.ssh),
        ssh=lambda ds: remove_nan(ds.ssh),
    ).assign(
        input=lambda ds: ds.nadir_obs,
        tgt=lambda ds: ds.ssh,
    )[[*TrainingItem._fields]]
     
    train_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').to_array().sel(time=train_period),
        patch_dims=patch_dims, strides=strides,
        postpro_fn=TrainingItem._make,
    )
    val_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').to_array().sel(time=val_period),
        patch_dims=patch_dims, strides=strides,
        postpro_fn=TrainingItem._make,
    )
    print(f'{len(train_ds)=}, {len(val_ds)=}')
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1), \
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=1)

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight):
        super().__init__()
        self.solver = solver
        self.rec_weight = nn.Parameter(torch.from_numpy(rec_weight), requires_grad=False)
        self.test_data = None

    @staticmethod
    def weighted_mse(err, weight):
        err_w = (err * weight[None, ...])
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            print('Am i here')
            return torch.scalar_tensor(0., device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'tr', training=True)[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')[0]

    def forward(self, batch, training=False):
        return self.solver(batch)

    def step(self, batch, phase='', opt_idx=None, training=False):
        states = self(batch=batch, training=training)
        loss = torch.sum(torch.stack([ 
                self.weighted_mse(state - batch.tgt, self.rec_weight)
                for state in states]))

        out = states[-1]
        rmse = self.weighted_mse(out - batch.tgt, self.rec_weight)**0.5
        self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss, out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def test_step(self, batch, batch_idx):
        out = self(batch=batch)[-1]

        return torch.stack([
            batch.tgt.cpu(),
            out.squeeze(dim=-1).detach().cpu(),
            ],dim=1)

    def test_epoch_end(self, outputs):
        rec_data = (it.chain(lt) for lt in zip(*outputs))
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
            k: rec_da.isel(v0=i,
                           time=tidx, lat=latidx, lon=lonidx
                        )
            for i, k  in enumerate(['ssh', 'rec_ssh'])
        })

class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, cut_graph_freq):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.cut_graph_freq = cut_graph_freq

        self._grad_norm = None

    def solver_step(self, state, batch, prog):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        if self._grad_norm is None:
            self._grad_norm = (grad**2).mean().sqrt()
        
        state_update = 1 / prog *  self.grad_mod(grad / self._grad_norm) + prog * grad
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            _intermediate_states = []
            state = batch.input.nan_to_num().detach().requires_grad_(True)
            self.grad_mod.reset_state(batch.input)
            self._grad_norm = None
            for step in range(self.n_step):
                if step + 1 % self.cut_graph_freq == 0:
                    _intermediate_states.append(state)
                    state = state.detach().requires_grad_(True)
                    self.grad_mod.detach_state()

                state = self.solver_step(state, batch, prog=(step + 1) / self.n_step)

        return [*_intermediate_states, state]



class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden, 4 * dim_hidden,
            kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self._state = []

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._state = [
                torch.zeros(size, device=inp.device),
                torch.zeros(size, device=inp.device),
        ]

    def detach_state(self):
        self._state = [
                s.detach().requires_grad_(True) for s in self._state
        ]

    def forward(self, input):
        hidden, cell = self._state
        gates = self.gates(torch.cat((input, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(torch.sigmoid, [in_gate, remember_gate, out_gate])
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        return self.conv_out(hidden)


class BaseObsCost(nn.Module):
    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3):
        super().__init__()
        self.conv_in = nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_hidden = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)

        self.bilin_1 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_21 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_22 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
    
        self.conv_out = nn.Conv2d(2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size//2)

    def forward_ae(self, x):
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        return self.conv_out(
            torch.cat([self.bilin_1(x), self.bilin_21(x) * self.bilin_21(x)], dim=1)
        )

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))

def run1():
    try:
        cfg = utils.get_cfg(base_cfg, overrides=overrides)
        # print(OmegaConf.to_yaml(cfg.file_paths))
        print(cfg.file_paths.data_registry_path)

        xr.open_dataset( f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',).ssh.std()

        strides = dict(time=1, lat=240, lon=240)
        _patch_dims = dict(time=20, lat=240, lon=240)

        _crop = dict(time=8, lat=20, lon=20)
        ds = dict(time=1, lat=2, lon=2)
        patch_dims = {k: _patch_dims[k]//ds.get(k,1) for k in _patch_dims}
        crop = {k: _crop[k]//ds.get(k,1) for k in _crop}
        # crop = dict()


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

        
        vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
        geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()
        rec_weight = get_constant_crop(patch_dims, crop) 
        lit_mod = Lit4dVarNet(
            solver=GradSolver(
                prior_cost=BilinAEPriorCost(dim_in=patch_dims['time'], dim_hidden=50),
                obs_cost=BaseObsCost(),
                grad_mod=ConvLstmGradModel(dim_in=patch_dims['time'], dim_hidden=150),
                n_step=15,
                cut_graph_freq=5,
            ),
            rec_weight=rec_weight

        )

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

        trainer = pl.Trainer(gpus=[1], logger=False, callbacks=callbacks, max_epochs=100,
             # limit_train_batches=10,
        )
        trainer.fit(lit_mod, train_dataloaders=train_dl, val_dataloaders=val_dl)

        print(trainer.checkpoint_callback.best_model_score)
        lit_mod.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
        trainer.test(lit_mod, dataloaders=[val_dl])


        lit_mod.test_data.to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')
        lit_mod.test_data.map(geo_energy).to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')
        lit_mod.test_data.map(vort).to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(row='variable', col='time')

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    ...
    locals().update(main())
