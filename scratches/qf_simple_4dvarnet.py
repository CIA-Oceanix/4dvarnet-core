import utils
# import metrics
import holoviews as hv
try:
    hv.extension('matplotlib')
except:
    pass
import kornia
# import metpy.calc as mpcalc
import itertools as it
import functools as ft
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
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

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inp_ds, aug_factor, item_cls=TrainingItem):
        self.aug_factor = aug_factor
        self.inp_ds = inp_ds
        self.perm = np.random.permutation(len(self.inp_ds))
        self.item_cls = item_cls

    def __len__(self):
        return len(self.inp_ds) * (self.aug_factor + 1)

    def __getitem__(self, idx):
        tgt_idx = idx % len(self.inp_ds)
        perm_idx = tgt_idx
        for _ in range(idx // len(self.inp_ds)):
            perm_idx = self.perm[perm_idx]
        
        item = self.inp_ds[tgt_idx]
        perm_item = self.inp_ds[perm_idx]

        return self.item_cls(
            **{
                **item._asdict(),
                **{'input': np.where(np.isfinite(perm_item.input),
                             item.tgt, np.full_like(item.tgt,np.nan))
                 }
            }
        )


def build_dataloaders(
        path, patch_dims, strides, train_period, val_period, ds=None, batch_size=4,
        aug_factor=2,

        ):
    inp_ds = xr.open_dataset(path)
    new_ds = inp_ds.coarsen(ds).mean().assign(
        input=lambda ds: ds.nadir_obs,
        tgt=lambda ds: remove_nan(ds.ssh),
    )[[*TrainingItem._fields]]
     
    m, s = new_ds.tgt.mean().values, new_ds.tgt.std().values
    print(m, s)
    post_fn = ft.partial(ft.reduce,lambda i, f: f(i), [
        lambda item: (item - m) / s,
        TrainingItem._make,
    ])

    _train_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').to_array().sel(time=train_period),
        patch_dims=patch_dims, strides=strides,
        postpro_fn=post_fn,
    )
    train_ds = AugmentedDataset(_train_ds, aug_factor)
    val_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').to_array().sel(time=val_period),
        patch_dims=patch_dims, strides=strides,
        postpro_fn=post_fn,
    )
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
            return torch.scalar_tensor(1000., device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        loss, grad_loss = self.step(batch, 'tr', training=True)[0]
        return loss + 50*grad_loss

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')[0]

    def forward(self, batch, training=False):
        return self.solver(batch)

    def step(self, batch, phase='', opt_idx=None, training=False):
        states = self(batch=batch, training=training)
        loss = sum(
                self.weighted_mse(state - batch.tgt, self.rec_weight)
            for state in states)

        grad_loss = sum(
                self.weighted_mse(
                    kornia.filters.sobel(state) - kornia.filters.sobel(batch.tgt),
                    self.rec_weight
                ) for state in states)
        out = states[-1]
        rmse = self.weighted_mse(out - batch.tgt, self.rec_weight)**0.5
        self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_gloss', grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        return [loss, grad_loss], out

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.solver.grad_mod.parameters(), 'lr':1e-3},
            {'params': self.solver.prior_cost.parameters(), 'lr':5e-4}],
            lr=1e-3)

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

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        if self._grad_norm is None:
            self._grad_norm = (grad**2).mean().sqrt()
        
        state_update = 1 / (step + 1)  *  self.grad_mod(grad / self._grad_norm)
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

                state = self.solver_step(state, batch, step=step)

        return [*_intermediate_states, state]



class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.25):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
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

    def forward(self, x):
        hidden, cell = self._state
        x = self.dropout(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(torch.sigmoid, [in_gate, remember_gate, out_gate])
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        hidden = self.dropout(hidden)
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
        _patch_dims = dict(time=29, lat=240, lon=240)

        _crop = dict(time=8, lat=20, lon=20)
        # ds = dict(time=1, lat=2, lon=2)
        ds = dict()
        patch_dims = {k: _patch_dims[k]//ds.get(k,1) for k in _patch_dims}
        crop = {k: _crop[k]//ds.get(k,1) for k in _crop}
        # crop = dict()


        train_period = slice('2013-01-01', '2013-09-30')
        # train_period = slice('2013-07-01', '2013-08-30')
        val_period = slice('2012-10-01', '2012-12-30')
        train_dl, val_dl = build_dataloaders(
            f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',
            patch_dims,
            strides,
            train_period,
            val_period,
            ds=ds
        )

        rec_weight = get_constant_crop(patch_dims, crop) 
        lit_mod = Lit4dVarNet(
            solver=GradSolver(
                prior_cost=BilinAEPriorCost(dim_in=patch_dims['time'], dim_hidden=64),
                obs_cost=BaseObsCost(),
                grad_mod=ConvLstmGradModel(dim_in=patch_dims['time'], dim_hidden=128),
                n_step=15,
                cut_graph_freq=5,
            ),
            rec_weight=rec_weight
        )
        pl.seed_everything(333)
        callbacks=[
            plcb.ModelCheckpoint(monitor='val_loss', save_last=True),
            plcb.TQDMProgressBar(),
            plcb.GradientAccumulationScheduler({20:2, 50:4}),
            # plcb.StochasticWeightAveraging(),
            # plcb.RichProgressBar(),
            plcb.ModelSummary(max_depth=2),
            # plcb.GradientAccumulationScheduler({50: 10})
        ]

        trainer = pl.Trainer(gpus=[2], logger=False, callbacks=callbacks, max_epochs=200,
             # limit_train_batches=10,
        )
        # trainer.fit(lit_mod, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # print(trainer.checkpoint_callback.best_model_score)
        # ckpt = '/raid/localscratch/qfebvre/4dvarnet-core/checkpoints/epoch=53-step=6623.ckpt'
        ckpt = '/raid/localscratch/qfebvre/4dvarnet-starter/logs/default/version_27/checkpoints/epoch=297-step=15499.ckpt'
        # lit_mod.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
        lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
        trainer.test(lit_mod, dataloaders=[val_dl])

        vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
        geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()

        p_slice = dict(time=slice(5, 35, 10))
        lit_mod.test_data.to_array().isel(p_slice).plot.pcolormesh(row='variable', col='time', robust=True)
        lit_mod.test_data.map(remove_nan).map(geo_energy).to_array().isel(p_slice).plot.pcolormesh(row='variable', col='time', robust=True)
        lit_mod.test_data.map(remove_nan).map(vort).to_array().isel(p_slice).plot.pcolormesh(row='variable', col='time', robust=True)
        lit_mod.load_from_checkpoint

        s = train_dl.dataset.inp_ds.da.sel(variable='tgt').std().values
        m = train_dl.dataset.inp_ds.da.sel(variable='tgt').mean().values
        m, s = 0.334645130315896, 0.38879137163517863
        # (rec - val_ds.da.sel(variable='tgt')).mean(['lat', 'lon']).plot()
        print(m, s)
        print(lit_mod.test_data.pipe(lambda ds: (ds.rec_ssh -ds.ssh)*s).pipe(lambda da: da**2).mean().pipe(np.sqrt))

        val_ds = val_dl.dataset
        batches = [b.tgt for b in val_dl]
        rec = val_ds.reconstruct(batches, weight=rec_weight) *s +m
        rec.isel(lat=50, lon=50).plot()
        val_ds.da.sel(variable='tgt').isel(lat=50, lon=50).plot()

        (rec - val_ds.da.sel(variable='tgt')).mean(['lat', 'lon']).plot()

        hv.output((
            anim(lit_mod.test_data.ssh.isel(time=slice(None,30, 2)), 'SSH (m)') +
            anim(lit_mod.test_data.map(remove_nan).map(geo_energy).ssh.isel(time=slice(None,30, 2)), 'SSH. Geostrophic Energy') +
            anim(lit_mod.test_data.map(remove_nan).map(vort).ssh.isel(time=slice(None,30, 2)), 'SSH Vorticity') +
            anim(lit_mod.test_data.rec_ssh.isel(time=slice(None,30, 2)), 'Reconstruction (m)') +
            anim(lit_mod.test_data.map(remove_nan).map(geo_energy).rec_ssh.isel(time=slice(None,30, 2)), 'Rec. Geostrophic Energy') +
            anim(lit_mod.test_data.map(remove_nan).map(vort).rec_ssh.isel(time=slice(None,30, 2)), 'Rec Vorticity')).cols(3),
            holomap='gif', fps=2, dpi=50, size=150)
        tdat = lit_mod.test_data.isel(time=slice(15, -15)) * s + m
        psdda, lx, lt = metrics.psd_based_scores(tdat.rec_ssh, tdat.ssh,)
        lx, lt
        
        metrics.plot_psd_score(psdda)
       
        errt, errmap, mu, sig = metrics.rmse_based_scores(tdat.rec_ssh, tdat.ssh,)
        errt.plot()
        errmap.plot()

        src_ds = xr.open_dataset(cfg.file_paths.data_registry_path+'/qdata/natl20.nc').load()
        ref = src_ds.sel(time=tdat.time).map(remove_nan).ssh
        ref = val_dl.dataset.da.sel(variable='tgt')
        pred = (tdat.ssh).interp(
                time=ref.time.broadcast_like(ref),
                lat=ref.lat.broadcast_like(ref),
                lon=ref.lon.broadcast_like(ref),
        )
        npa = pred.values
        lonidx = ~np.all(np.isnan(npa), axis=tuple([0, 1]))
        latidx = ~np.all(np.isnan(npa), axis=tuple([0, 2]))
        tidx = ~np.all(np.isnan(npa), axis=tuple([1, 2]))

        pred = pred.isel(time=tidx, lat=latidx, lon=lonidx).transpose('time', 'lat', 'lon')
        ref = ref.isel(time=tidx, lat=latidx, lon=lonidx).transpose('time', 'lat', 'lon')
        (pred - ref).isel(time=10).plot()
        pred.isel(lat=40, lon=40).plot()
        ref.isel(lat=40, lon=40).plot()
        pred.pipe(remove_nan).pipe(geo_energy).isel(time=1).plot()
        ref.pipe(remove_nan).pipe(geo_energy).isel(time=1).plot()
        errt, errmap, mu, sig =metrics.rmse_based_scores(pred, ref)
        errt.plot()
        errmap.plot()
        metrics.psd_based_scores(pred, ref)
        import pandas as pd
        # pd.read_csv('../4dvarnet-starter/logs/default/version_27/metrics.csv').val_rmse.dropna().plot()
        # patch_dims = cfg.model.rec_weight.patch_dims
        # crop = cfg.model.rec_weight.crop
        # def rec_weight_fn(t, lt, lg):
             
        #     mt = (t > crop.time) & (t < (patch_dims.time - crop.time))
        #     mlt = (lt > crop.lat) & (lt < (patch_dims.lat - crop.lat))
        #     mlg = (lg > crop.lon) & (lg < (patch_dims.lon - crop.lon))

        #     nz = (patch_dims.time )
        #     te = t 
        #     return (nz / 2 - np.abs( t - nz/2)  -crop.time) * mt * mlt * mlg

        # rec_weight = np.fromfunction(rec_weight_fn, patch_dims.values())
        # import matplotlib.pyplot as plt
        # plt.plot(rec_weight[:, 100, 100])
        # lit_mod.rec_weight.data = torch.from_numpy(
        #         get_constant_crop(patch_size=cfg.datamodule.xrds_kw.patch_dims, crop=dict(time=14, lat=20, lon=20)))
        # lit_mod.rec_weight.data = torch.from_numpy(rec_weight)
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def test_starter():
    import sys
    sys.path.append('../4dvarnet-starter')
    import src
    import inspect
    import importlib
    import src.models
    importlib.reload(src.models)
    import hydra
    from pathlib import Path
    xpdir = '/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2022-11-23/15-23-51'
    ('..' / Path(xpdir).relative_to(Path('..').resolve().absolute()) /'.hydra').exists()

    with hydra.initialize(config_path= Path(xpdir).relative_to(Path('..').resolve().absolute()) /'.hydra', version_base="1.2"):
        cfg = hydra.compose('config.yaml', overrides=['trainer.logger=False'])

    trainer = hydra.utils.call(cfg.trainer)(inference_mode=False, gpus=[3])
    dm = hydra.utils.call(cfg.datamodule)()
    lit_mod = hydra.utils.call(cfg.model)(norm_stats=dm.norm_stats())
    # print(lit_mod.solver.solver_step)
    # print(inspect.getsource(lit_mod.solver.solver_step))
    ckpt = xpdir +'/base/checkpoints/best.ckpt'
    lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
    trainer.test(lit_mod, datamodule=dm)

    print(
            lit_mod
            .test_data.isel(time=slice(7, -7))
            .pipe(lambda ds: (ds.rec_ssh -ds.ssh)).pipe(lambda da: da**2).mean().pipe(np.sqrt)
    )
    tdat = lit_mod.test_data.isel(time=slice(5, -5)) *s +m
    dm.test_ds.da.sel(variable='tgt').pipe(lambda da: da.sel(tdat.coords) - tdat.ssh).isel(time=1).plot()
    psdda, lx, lt = metrics.psd_based_scores(tdat.rec_ssh, tdat.ssh,)
    print(lx, lt)
    errt, errmap, mu, sig = metrics.rmse_based_scores(tdat.rec_ssh, tdat.ssh,)
    print(mu, sig)

    vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
    geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()

    p_slice = dict(time=slice(5, 35, 10))
    lit_mod.test_data.to_array().isel(p_slice).plot.pcolormesh(row='variable', col='time', robust=True)
    lit_mod.test_data.map(remove_nan).map(geo_energy).to_array().isel(p_slice).plot.pcolormesh(row='variable', col='time', robust=True)
    lit_mod.test_data.map(remove_nan).map(vort).to_array().isel(p_slice).plot.pcolormesh(row='variable', col='time', robust=True)



def anim(da, name, climda=None):
    climda = climda if climda is not None else da
    clim = climda.pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
    return  (hv.Dataset(da)
            .to(hv.QuadMesh, ['lon', 'lat']).relabel(name)
            .options(cmap='RdBu',clim=clim, colorbar=True))

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
    # locals().update(main())
