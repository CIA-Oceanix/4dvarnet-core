import numpy as np
import contextlib
import kornia
import lit_model_augstate
import hydra
import dataloading
import einops
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndi
from omegaconf import OmegaConf
import xarray as xr
import pandas as pd
import torch
import pickle
import swath_calib.configs
import swath_calib.utils
import swath_calib.models
import swath_calib.dataset
import swath_calib.versioning_cb
import swath_calib.report
import swath_calib.interp
import pytorch_lightning as pl
from pathlib import Path
import time
import utils
import uuid
import models
import solver
import functools
import importlib
import numcodecs
import torch.nn.functional as F
numcodecs.blosc.use_threads = False
importlib.reload(solver)
import hydra_main
importlib.reload(hydra_main)
importlib.reload(utils)
importlib.reload(lit_model_augstate)


# Generate grided product on swath


"""
TODO:

- Datamodule:
    - load swot obs + gt
- Cal model
    - Test standalone training
    - Test with precomputed normalization

- Joint model
    - Load pretrained 5nad weights
    - Load pretrained Cal mod weights
    

"""

class ModelObsMixedGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 3
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_data, sst_ch])

    def forward(self, x, y, msk):
        xlr, anom_obs, *_ = torch.split(x, self.hparams.dT, dim=1)

        ylr, (_, gc, sv, sc, nv, nc) = y
        msk_lr, _ = msk
        dyoutlr =  (ylr - xlr) * msk_lr

        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        if sv is not None:
            s_x = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr + anom_obs, *gc, *sc)
            s_x_msk = s_x.isfinite()
            dyout =  fna(s_x, s_x_msk) - fna(sv, s_x_msk)
        else:
            dyout = torch.zeros_like(xlr)

        if nv is not None:
            n_x = swath_calib.interp.batch_torch_interpolate_with_fmt(x, *gc, *nc)
            n_x_msk = n_x.isfinite()
            dyout1 =  fna(n_x, n_x_msk) - fna(nv, n_x_msk)
        else:
            dyout1 = torch.zeros_like(xlr)
        return [dyoutlr, dyout, dyout1]



class SensorXrDs(torch.utils.data.Dataset):
    def __init__(self,
            xr_ds,
            nadir_paths=tuple([f'../sla-data-registry/sensor_zarr/zarr/nadir/{name}' for name in ['swot', 'en', 'tpn', 'g2', 'j1']]),
            swot_path=f'../sla-data-registry/sensor_zarr/zarr/new_swot',
            nadir_var='ssh_model',
            swot_gt_vars=('ssh_model',),
            swot_obs_vars=('ssh_model',),
            obs_downsamp=None,
            min_swot_length=None,
        ):
        self.xr_ds = xr_ds
        self.nadir_paths = nadir_paths
        self.swot_path = swot_path
        self.nadir_vars = [nadir_var]
        self.swot_gt_vars = list(swot_gt_vars)
        self.swot_obs_vars = list(swot_obs_vars)
        self.obs_downsamp = obs_downsamp
        self.min_swot_length = min_swot_length
        self.crop = pd.to_timedelta('3D')

    def __len__(self):
        return len(self.xr_ds)

    @staticmethod
    def collate_fn(list_of_items):

        gv, gc, sv, sc, nv, nc = zip(*list_of_items)
        return (
            torch.stack(gv),
            list(map(torch.stack, zip(*gc))),
            None if sv is None else swath_calib.interp.stack(sv),
            None if sc is None else list(map(swath_calib.interp.stack, zip(*sc))),
            None if nv is None else swath_calib.interp.stack(nv),
            None if nc is None else list(map(swath_calib.interp.stack, zip(*nc))),
        )

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, item):
        with self.xr_ds.get_coords():
            coords = self.xr_ds[item]
        gv_ = self.xr_ds[item]

        slice_args = dict(
            time_min= (pd.to_datetime(np.min(coords['time']).values) + self.crop).date(),
            time_max= (pd.to_datetime(np.max(coords['time']).values) - self.crop).date(),
            lat_min=coords['lat'].min().item(),
            lat_max=coords['lat'].max().item(),
            lon_min=coords['lon'].min().item() + 360,
            lon_max=coords['lon'].max().item() + 360,
        )

        if self.swot_path is not None:
            swot = swath_calib.utils.get_swot_slice(self.swot_path, **slice_args)

            # add_cc = lambda ds: ds.assign(ch_nb=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
            # sw = swot.pipe(add_cc)

            sp_ds = swath_calib.interp.stack_passes(*swath_calib.interp.add_nb_ch(swot), min_len=self.min_swot_length)
            spc =  swath_calib.interp.fmt_s_coords(sp_ds)
            spgt = swath_calib.interp.fmt_s_value(sp_ds, self.swot_gt_vars , spc[0].isfinite())
            spv = swath_calib.interp.fmt_s_value(sp_ds, self.swot_obs_vars , spc[0].isfinite())
        else:
            spgt, spv, spc = None, None, (None, None, None)

        if self.nadir_paths[0] is not None:
            nadirs = [swath_calib.utils.get_nadir_slice(p, **slice_args) for p in self.nadir_paths]
            nadirs = [nad for nad in nadirs if nad is not None]
            nads_with_nb_ch, tgt_len = zip(*[swath_calib.interp.add_nb_ch(nad) for nad in nadirs])

            np_ds = xr.concat([swath_calib.interp.stack_passes(nad, max(tgt_len)) for nad in nads_with_nb_ch], 'p')


            npc = swath_calib.interp.fmt_s_coords(np_ds)
            npv = swath_calib.interp.fmt_s_value(np_ds, self.nadir_vars, npc[0].isfinite())
        else: 
            npv, npc = None, (None, None, None)

        return (
                spgt, torch.from_numpy(gv_), swath_calib.interp.fmt_g_coords(xr.Dataset(coords)),
                spv, spc, npv, npc,
        )

class FourDVarMixedGeometryDataset(torch.utils.data.Dataset):
    def __init__(self,
            period,
            gt_path,
            gt_var,
            oi_path,
            oi_var,
            obs_mask_path,
            obs_mask_var,
            grid_kwargs,
            sens_kwargs,
            norm_stats=(0,1),
            ):
        grid_kwargs = {
                **grid_kwargs,
                **{'dim_range': {**grid_kwargs['dim_range'], **{'time': period}}}
        }
        self.gt_ds = dataloading.XrDataset(gt_path, gt_var, interp_na=True, decode=True, auto_padding=False, **grid_kwargs)
        self.oi_ds = dataloading.XrDataset(oi_path, oi_var, interp_na=True, decode=False, auto_padding=False, **grid_kwargs)
        self.grid_obs_ds = dataloading.XrDataset(obs_mask_path, obs_mask_var, decode=True, auto_padding=False, **grid_kwargs)
        self.sens_obs_ds = SensorXrDs(
            self.grid_obs_ds,**sens_kwargs
        )
        self.norm_stats = norm_stats
        self.return_coords = False

    def __len__(self):
        l = len(self.grid_obs_ds)
        print(l)
        return l

    @contextlib.contextmanager
    def get_coords(self):
        try:
            self.return_coords = True
            yield
        finally:
            self.return_coords = False

    @staticmethod
    def collate_fn(list_of_items):
        oi , gt, sgt, go, gc, sv, sc, nv, nc = zip(*list_of_items)
        cstack = lambda ts: None if all(t is None for t in ts) else swath_calib.interp.stack([t for t in ts if t is not None])
        return (
            torch.stack(oi),
            torch.stack(gt),
            cstack(sgt),
            torch.stack(go),
            list(map(torch.stack, zip(*gc))),
            cstack(sv),
            list(map(cstack, zip(*sc))),
            cstack(nv),
            list(map(cstack, zip(*nc))),
        )

    def __getitem__(self, item):
        if self.return_coords:
            with self.gt_ds.get_coords():
                return self.gt_ds[item]
        oi, gt, sgt, go, gc, sv, sc, nv, nc = (self.oi_ds[item], self.gt_ds[item], *self.sens_obs_ds[item])
        pp = lambda t: t if t is None else (t - self.norm_stats[0]) / self.norm_stats[1]
        return pp(torch.from_numpy(oi)), pp(torch.from_numpy(gt)), pp(sgt), pp(go), gc, pp(sv), sc, pp(nv), nc



class FourDVarMixedGeometryDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            ds_kwargs,
            dl_kwargs,
            train_slices= (slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            test_slices= (slice('2013-01-03', "2013-01-27"),),
            val_slices= (slice('2012-11-30', "2012-12-24"),),
    ):
        super().__init__()
        self.ds_kwargs = ds_kwargs
        self.dl_kwargs = dl_kwargs
        self.train_slices, self.test_slices, self.val_slices = train_slices, test_slices, val_slices
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.norm_stats = (0, 1)
        self.norm_stats_sst = None
        self.dim_range = ds_kwargs['grid_kwargs']['dim_range']

    def compute_stats(self, ds):
        sum = 0
        count = 0

        for gt in (_it for _ds in ds.datasets for _it in _ds.gt_ds):
            sum += np.nansum(gt)
            count += np.sum(np.isfinite(gt))
        mean = sum / count
        sum = 0
        for gt in (_it for _ds in ds.datasets for _it in _ds.gt_ds):
            sum += np.nansum((gt - mean)**2)
        std = (sum / count)**0.5
        print(' normstats ', mean, std)
        return mean, std

    def setup(self, stage='test'):
        self.train_ds = torch.utils.data.ConcatDataset(
            [FourDVarMixedGeometryDataset(
                period=sl, **self.ds_kwargs,
            ) for sl in self.train_slices])


        self.val_ds, self.test_ds = [
           torch.utils.data.ConcatDataset(
                [FourDVarMixedGeometryDataset(
                period=sl, **self.ds_kwargs,
            ) for sl in sls]) for sls in (self.val_slices, self.test_slices)
        ]

        self.norm_stats = self.compute_stats(self.train_ds)
        self.set_norm_stats(self.train_ds, self.norm_stats)
        self.set_norm_stats(self.val_ds, self.norm_stats)
        self.set_norm_stats(self.test_ds, self.norm_stats)

        self.bounding_box = self.get_domain_bounds(self.train_ds)
        self.ds_size = self.get_domain_split()

    def set_norm_stats(self, ds, ns):
        for _ds in ds.datasets:
            _ds.norm_stats = ns

    def get_domain_bounds(self, ds):
        min_lon = round(np.min(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        max_lon = round(np.max(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        min_lat = round(np.min(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        max_lat = round(np.max(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        return min_lon, max_lon, min_lat, max_lat

    def get_domain_split(self):
        return self.test_ds.datasets[0].gt_ds.ds_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                self.train_ds,
                collate_fn=self.train_ds.datasets[0].collate_fn,
                **{**dict(shuffle=True), **self.dl_kwargs})

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
                self.val_ds,
                collate_fn=self.val_ds.datasets[0].collate_fn,
                **{**dict(shuffle=False), **self.dl_kwargs})

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
                self.test_ds,
                collate_fn=self.test_ds.datasets[0].collate_fn,
                **{**dict(shuffle=False), **self.dl_kwargs})


def get_4dvarnet_mixgeom(hparams):
    return solver.Solver_Grad_4DVarNN(
                models.Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                ModelObsMixedGeometry(hparams.shape_state[0], hparams=hparams),
                solver.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

def get_4dvarnet_cal(hparams):
    return solver.Solver_Grad_4DVarNN(
                models.Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                CalibrationModelObsSensorGeometry(hparams.shape_state[0], hparams=hparams),
                solver.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

def get_4dvarnet_cal_grid(hparams):
    return solver.Solver_Grad_4DVarNN(
                models.Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                CalibrationModelObsGridGeometry(hparams.shape_state[0], hparams=hparams),
                solver.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

class LitModMixGeom(lit_model_augstate.LitModelAugstate):

    def diag_step(self, batch, batch_idx, log_pref='test'):
        oi, gt, sgt, go, *y = batch
        losses, out, out_w_cal, metrics, metrics_w_cal = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None and log_pref is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mse_w_cal', metrics_w_cal["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_cal', metrics[-1]["mseCal"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_cal_joint', metrics_w_cal['mseCal'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (gt.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (oi.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (go.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out_w_cal.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred_wo_cal' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out=None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, out, state, _metrics = self.compute_loss(
                    batch, phase='train', state_init=state_init, use_swot=False, compute_cal=((_+1)==self.hparams.n_fourdvar_iter)
            )
            state_init = [None if s is None else s.detach() for s in state]
            losses.append(_loss)
            metrics.append(_metrics)


        _loss, out_w_cal, state, metrics_w_cal = self.compute_loss(
                batch,
                phase='train',
                state_init=state_init,
                use_swot=True,
                compute_cal=True
        )
        return losses, out, out_w_cal, metrics, metrics_w_cal


    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()

        losses = []
        metrics = []
        state_init = [None]
        out=None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, out, state, _metrics = self.compute_loss(
                    train_batch, phase='train', state_init=state_init, use_swot=False, compute_cal=((_+1)==self.hparams.n_fourdvar_iter)
            )
            if not self.automatic_optimization:
                self.manual_backward(_loss)
            state_init = [None if s is None else s.detach() for s in state]
            losses.append(_loss)
            metrics.append(_metrics)

        # if not self.automatic_optimization:
        #     opt.step()

        _loss, out_w_cal, state, metrics_w_cal = self.compute_loss(
                train_batch,
                phase='train',
                state_init=state_init,
                use_swot=True,
                compute_cal=True
        )
        if not self.automatic_optimization:
            self.manual_backward(5 * _loss)

        if not self.automatic_optimization:
            opt.step()
        # losses, _, metrics = self(train_batch, phase='train')
        # if losses[-1] is None:
        #     print("None loss")
        #     return None
        # # loss = torch.stack(losses).sum()
        loss = 2*torch.stack(losses).sum() - losses[0]

        self.log("tr_loss_wo_cal", losses[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_loss_w_cal", _loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mse_w_cal", metrics_w_cal['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_cal", metrics[-1]['mseCal'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_cal_joint", metrics_w_cal['mseCal'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        return self.diag_step(batch, batch_idx, log_pref=None)


    def create_model(self, *args, **kwargs):
        if self.hparams.model_name == '4dvarnet':
            return get_4dvarnet_mixgeom(self.hparams)
        elif self.hparams.model_name == '4dvarnet_cal':
            return get_4dvarnet_cal(self.hparams)
        elif self.hparams.model_name == '4dvarnet_cal_grid':
            return get_4dvarnet_cal_grid(self.hparams)

    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        oi, _, _, go, *_ = batch
        msk = go.isfinite()
        
        go_oi = msk.float() * (go.where(msk, torch.zeros_like(go)) - oi)

        init_state = torch.cat((oi,go_oi), dim=1)
        if self.aug_state:
            init_state = torch.cat((init_state, go_oi), dim=1)
        return init_state

    def cal_out(self, y, xb):
        _, gc, sv, sc, _, _ = y
        s_vb = swath_calib.interp.batch_torch_interpolate_with_fmt(xb, *gc, *sc)
        cal_out, cal_msk, cal_idx = self.model.model_H.cal_out(sv, s_vb.detach())

        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        pw = einops.repeat(self.patch_weight, '... -> b ...', b=xb.size(0))
        sw = swath_calib.interp.batch_torch_interpolate_with_fmt(pw, *gc, *sc)
        sw = fna(sw, sw.isfinite())
        return cal_out, cal_msk, cal_idx, sw

    def loss_cal(self, sgt, y, xb):

        _, _, sv, _, _, _ = y
        if sv is None:
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        cal_out, cal_msk, cal_idx, sw = self.cal_out(y, xb)

        t1, t2, tw = cal_out, fna(sgt[:,:, cal_idx, :], cal_msk), fna(sw[:,:, cal_idx, :], cal_msk)
        if tw.sum().item() == 0:
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        mse = ((t1-t2)**2 * tw).sum() / tw.sum()

        def sob(t):
            if len(t.shape) == 4:
                return kornia.filters.sobel(t)
            elif len(t.shape) == 3:
                return kornia.filters.sobel(einops.rearrange(t, 'b d1 d2 -> b () d1 d2'))
            else:
                assert False, 'Should not be here'

        def lap(t):
            if len(t.shape) == 4:
                # return kornia.filters.laplacian(rearrange(t, 'b d1 d2 c -> b c d1 d2'), kernel_size=3)
                return kornia.filters.laplacian(t, kernel_size=3)
            elif len(t.shape) == 3:
                return kornia.filters.laplacian(einops.rearrange(t, 'b d1 d2 -> b () d1 d2'), kernel_size=3)
            else:
                assert False, 'Should not be here'

        mse_grad = ((sob(t1) - sob(t2))**2 * tw).sum() / tw.sum()
        mse_lap = ((lap(t1) - lap(t2))**2 * tw).sum() / tw.sum()

        return mse, mse_grad, mse_lap

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)

        optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.model.model_H.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            ])

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=25, cooldown=10, factor=0.5),
            'monitor': 'val_loss'
        }


    def compute_loss(self, batch, phase, state_init=(None,), use_swot=False, compute_cal=False):
        oi, gt, sgt, *y = batch


        # handle patch with no observation
        gt_wo_nan = gt.where(~ gt.isnan(), oi)
        oi_wo_nan = oi.where(~ oi.isnan(), oi)

        state = self.get_init_state(batch, state_init)

        obs = (oi, y)
        if not use_swot:
            (go, gc, _, sc, nv, nc) = y 
            obs = (oi, (go, gc, None, sc, nv, nc))

        msks = (torch.ones_like(oi), None)

        # gradient norm field
        g_gt_x, g_gt_y = self.gradient_img(gt)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            x_out, hidden_new, cell_new, normgrad = self.model(state, obs, msks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                x_out = x_out.detach()

        if self.aug_state:
            xlr, _, xhr = torch.split(x_out, self.hparams.dT, dim=1)
            outputs = xlr + xhr
        else:
            xlr, xhr = torch.split(x_out, self.hparams.dT, dim=1)
            outputs = xlr + xhr

        # median filter
        if self.median_filter_width > 1:
            outputs = kornia.filters.median_blur(outputs, (self.median_filter_width, self.median_filter_width))


        yGT = torch.cat((oi_wo_nan, gt_wo_nan - xlr), dim=1)
        if self.aug_state:
            yGT = torch.cat((yGT, gt_wo_nan - xlr), dim=1)

        loss_All, loss_GAll = self.sla_loss(outputs, gt_wo_nan)
        loss_OI, loss_GOI = self.sla_loss(oi_wo_nan, gt_wo_nan)
        loss_AE, loss_AE_GT, loss_SR, loss_LR =  self.reg_loss(
            yGT, oi, outputs, xlr, x_out
        )

        if self.hparams.calref =='oi':
            xb = oi_wo_nan
        elif self.hparams.calref =='xlr':
            xb = xlr
        elif self.hparams.calref =='x':
            xb = outputs

        # loss_cal, g_loss_cal, l_loss_cal = self.loss_cal(sgt, y, xb.detach())
        lcal_in = y
        if not compute_cal:
            (go, gc, _, sc, nv, nc) = y
            lcal_in = (go, gc, None, sc, nv, nc)
        loss_cal, g_loss_cal, l_loss_cal = self.loss_cal(sgt, lcal_in, xb)

        # print(loss_All, loss_GAll,loss_AE, loss_AE_GT, loss_SR, loss_LR, loss_cal)
        # total loss
        loss = 0
        loss += self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
        loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
        loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
        loss += self.hparams.alpha_cal * (loss_cal + 0.5*g_loss_cal + 0.5*l_loss_cal)

        # metrics
        # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
        mean_GAll = solver.compute_spatio_temp_weighted_loss(
                torch.hypot(g_gt_x, g_gt_y) , self.grad_crop(self.patch_weight))
        mse = loss_All.detach()
        mseGrad = loss_GAll.detach()
        metrics = dict([
            ('mse', mse.detach()),
            ('mseGrad', mseGrad.detach()),
            ('meanGrad', mean_GAll.detach()),
            ('mseOI', loss_OI.detach()),
            ('mseGOI', loss_GOI.detach()),
            ('mseCal', loss_cal.detach())])

        return loss, outputs, [x_out, hidden_new, cell_new, normgrad], metrics

class CalibrationModelObsGridGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None, min_size=500, sigs=tuple([8* (i+1) for i in range(10)])):
        super().__init__()
        self.hparams = hparams
        sst_ch = hparams.dT
        # self.dim_obs_channel = np.array([shape_data, sst_ch])
        # self.dim_obs = 3
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_data])
        self.min_size = min_size
        self.num_feat = 2*(len(sigs)+1)
        self.norm = torch.nn.BatchNorm2d(num_features=self.num_feat, affine=False, momentum=0.1)

        self.gaussian = StackedGaussian(sigs)
        self.calnet = torch.nn.Sequential(
            self.norm,
            swath_calib.models.build_net(
                in_channels=self.num_feat,
                out_channels=1,
                mix=True,
            ),# nhidden = 32, depth = 2,),
        )
        self.downsamp = self.hparams.obscost_downsamp

        self.patch_weight = torch.nn.Parameter(
                torch.from_numpy(hydra.utils.call(self.hparams.patch_weight)), requires_grad=False)
        self.ns = (0.31446309894037083, 0.3886609494201447)


    def cal_inp(self, sv_uncal, sv_bg):
        fs_xlr = einops.rearrange(sv_bg, 'b p ... -> (b p) () ...')

        fy = einops.rearrange(sv_uncal, 'b p ... -> (b p) () ...')

        msk = fy[:, :, :self.min_size, :].isfinite().all(3).all(2).all(1)
        idx = fs_xlr[ msk, ...].isfinite().all(3).all(1).all(0)

        gy =  self.gaussian(fy[msk][:, :, idx])
        inp_gy = torch.cat((gy.diff(dim=1), -gy[:, -1:, ...]), 1)

        gs_xlr = self.gaussian(fs_xlr[msk][:, :, idx])
        inp_gs_xlr = torch.cat((gy.diff(dim=1), -gs_xlr[:, -1:, ...]), 1)

        cal_input = -torch.cat((inp_gy, inp_gs_xlr), dim=1)
        return cal_input

    def cal_out(self, sv_uncal, sv_bg):
        sh = einops.parse_shape(sv_bg, 'b p ...')
        fs_xlr = einops.rearrange(sv_bg, 'b p ... -> (b p) () ...')

        fy = einops.rearrange(sv_uncal, 'b p ... -> (b p) () ...')

        msk = fy[:, :, :self.min_size, :].isfinite().all(3).all(2).all(1)
        idx = fs_xlr[ msk, ...].isfinite().all(3).all(1).all(0)

        gy =  self.gaussian(fy[msk][:, :, idx])
        inp_gy = torch.cat((gy.diff(dim=1), -gy[:, -1:, ...]), 1)

        gs_xlr = self.gaussian(fs_xlr[msk][:, :, idx])
        inp_gs_xlr = torch.cat((gy.diff(dim=1), -gs_xlr[:, -1:, ...]), 1)

        cal_input = -torch.cat((inp_gy, inp_gs_xlr), dim=1)
        _out_cal = self.calnet(cal_input)
        _out_cal = _out_cal + self.ns[0]/self.ns[1] + fs_xlr[msk][:,:,idx]
        out_cal = torch.zeros_like(fy[:, :, idx])
        out_msk = torch.zeros_like(fy[:, :, idx])
        # print(out_cal.shape, _out_cal.shape, cal_input.shape)
        out_cal.index_add_(0, msk.nonzero(as_tuple=True)[0], _out_cal)
        out_msk.index_add_(0, msk.nonzero(as_tuple=True)[0], torch.ones_like(_out_cal))
        out_cal = einops.rearrange(out_cal, '(b p) ()  ... -> b p ...', **sh)
        out_msk = einops.rearrange(out_msk > 0, '(b p) ()  ... -> b p ...', **sh)
        # print(out_cal.shape, out_cal.isnan().sum())
        
        return out_cal, out_msk, idx

    def cal_cost(self, sv_uncal, sv_bg, sv_ref):
        out_cal, cal_msk, idx = self.cal_out(sv_uncal, sv_bg)
        dyout = cal_msk * (out_cal - sv_ref[:, :, idx, :])
        return dyout

    def forward(self, x, y, ymsk):
        ylr, (go, gc, sv, sc, nv, nc) = y
        xlr, _, anom_rec = torch.split(x, self.hparams.dT, dim=1)

        # if self.hparams.compat_base:
        #     anom_rec = anom_obs

        if nv is not None:
            g_cal, gmsk = swath_calib.interp.batch_interp_to_grid(
                ylr, *gc, nv, *nc
            )
            # dyout1 = gmsk.float() * (g_cal - (xlr + anom_rec))


            # Baseline
            # yy = torch.cat((ylr, (g_cal - ylr).where(gmsk, torch.zeros_like(ylr)), ylr), dim=1) 
            # mm = torch.cat((torch.ones_like(ylr), gmsk, torch.zeros_like(go)), dim=1)
            # dyout = torch.abs((yy -x )  * mm)

            # Move nadir obs to rec
            yy = torch.cat((ylr, ylr, (g_cal - ylr).where(gmsk, torch.zeros_like(ylr))), dim=1) 
            mm = torch.cat((torch.ones_like(ylr), torch.zeros_like(go), gmsk), dim=1)
            dyout = torch.abs((yy -x )  * mm)
        else:
            # dyout =torch.zeros_like(x)
            msk = go.isfinite()
            gooi = (go - ylr).where(msk, torch.zeros_like(ylr))
            yy = torch.cat((ylr, gooi, gooi), dim=1) 

            mm = torch.cat((torch.ones_like(ylr), msk, torch.zeros_like(msk)), dim=1)
            dyout = torch.abs((yy -x )  * mm)

        if sv is not None:
            if self.hparams.calref == 'oi':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(ylr, *gc, *sc)
            elif self.hparams.calref == 'x':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr + anom_rec, *gc, *sc)
            elif self.hparams.calref == 'xlr':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr, *gc, *sc)

            out_cal, cal_msk, cal_idx = self.cal_out(sv, s_xlr.detach())
            g_cal, gmsk = swath_calib.interp.batch_interp_to_grid(
                    ylr, *gc, out_cal.detach(), *map(lambda t: t[:, :, cal_idx, ...], sc)
            )
            g_cal_msk, _ = swath_calib.interp.batch_interp_to_grid(
                    ylr, *gc, cal_msk.float(), *map(lambda t: t[:, :, cal_idx, ...], sc)
            )
            yy = torch.cat((ylr, (g_cal - ylr).where(g_cal_msk.bool() & gmsk, torch.zeros_like(ylr)), ylr), dim=1) 
            mm = torch.cat((torch.ones_like(ylr), self.patch_weight[None, ...] * gmsk.float() * g_cal_msk, torch.zeros_like(go)), dim=1)
            dyout1 = torch.abs((yy -x )  * mm)
            # dyout = self.hparams.swot_obs_w * dyout

        else:
            dyout1 = torch.zeros_like(x)

        return dyout + dyout1
        # return [dyoutlr, dyout, dyout1]
        
class CalibrationModelObsSensorGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None, min_size=500, sigs=tuple([8* (i+1) for i in range(10)])):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 3
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_data, sst_ch])
        self.min_size = min_size
        self.num_feat = 2*(len(sigs)+1)
        self.norm = torch.nn.BatchNorm2d(num_features=self.num_feat, affine=False, momentum=0.1)

        self.gaussian = StackedGaussian(sigs)
        self.calnet = swath_calib.models.build_net(self.num_feat, 1,)# nhidden = 32, depth = 2,)
        self.downsamp = self.hparams.obscost_downsamp


    def cal_cost(self, sv_uncal, sv_bg, sv_ref):
        fs_xlr = einops.rearrange(sv_bg.detach(), 'b p ... -> (b p) () ...')

        fy = einops.rearrange(sv_uncal, 'b p ... -> (b p) () ...')

        msk = fy[:, :, :self.min_size, :].isfinite().all(3).all(2).all(1)
        idx = fs_xlr[ msk, ...].isfinite().all(3).all(1).all(0)

        gy =  self.gaussian(fy[msk][:, :, idx])
        inp_gy = torch.cat((gy.diff(dim=1), gy[:, -1:, ...]), 1)

        gs_xlr = self.gaussian(fs_xlr[msk][:, :, idx])
        inp_gs_xlr = torch.cat((gy.diff(dim=1), gs_xlr[:, -1:, ...]), 1)

        cal_input = torch.cat((inp_gy, inp_gs_xlr), dim=1)
        # print(cal_input.shape)
        out_cal = self.calnet(self.norm(cal_input)) + fs_xlr[msk][:,:,idx]

        f_sx = einops.rearrange(sv_ref, 'b p ... -> (b p) () ...')
        dyout = out_cal - f_sx[msk][:,:,idx]
        return dyout

    def forward(self, x, y, ymsk):
        ylr, (_, gc, sv, sc, nv, nc) = y
        xlr, anom_obs, *_ = torch.split(x, self.hparams.dT, dim=1)

        if sv is not None:
            s_x = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr + anom_obs, *gc, *sc)

            if self.hparams.calref == 'oi':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(ylr, *gc, *sc)
            elif self.hparams.calref == 'x':
                s_xlr = s_x
            elif self.hparams.calref == 'xlr':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr, *gc, *sc)

            dyout = self.cal_cost(sv, s_xlr, s_x)
            # dyout = self.cal_cost(sv, s_x, s_x)
            dyout = self.hparams.swot_obs_w * dyout
            if self.downsamp > 1:
                dyout = einops.reduce(
                        dyout[...,
                            :dyout.shape[-2] // self.downsamp * self.downsamp,
                            :dyout.shape[-2] // self.downsamp * self.downsamp,
                        ], '... (h hds) (w wds) -> ... h w', hds=self.downsamp, wds=self.downsamp,
                        reduction='mean',
                )


        else:
            dyout = torch.zeros_like(xlr)

        if nv is not None:
            xlr, _, anom_rec = torch.split(x, self.hparams.dT, dim=1)
            fna = lambda t, m: t.where(m, torch.zeros_like(t))

            n_x = swath_calib.interp.batch_torch_interpolate_with_fmt(anom_rec, *gc, *nc)
            n_x_msk = n_x.isfinite()
            dyout1 =  fna(n_x, n_x_msk) - fna(nv, n_x_msk)
            if self.downsamp > 1:
                dyout1 = einops.reduce(
                        dyout1[...,
                            :dyout.shape[-2] // self.downsamp * self.downsamp,
                            :,
                        ], '... (h hds) () -> ... h ()', hds=self.downsamp,
                        reduction='mean',
                )
        else:
            dyout1 = torch.zeros_like(xlr)

        msk_lr, _ = ymsk
        dyoutlr =  (ylr - xlr) * msk_lr
        return [dyoutlr, dyout, dyout1]


class StackedGaussian(torch.nn.Module):
    def __init__(self, sigs= tuple([8 * (i+1) for i in range(20)]), size=75, min_size=600):
        super().__init__()
        assert size % 2 ==1, 'same pad not yet implemnted here for even kernels'
        self.size = size
        self.min_size = min_size
        self.kernel = torch.nn.Parameter(torch.stack([
            kornia.filters.get_gaussian_kernel1d(size, sig)
            for sig in sigs
        ])[:, None, :, None], requires_grad=False)

  
    def forward(self, fx):
        px = F.pad(fx, (0,0, self.size//2, self.size//2), mode='replicate')
        return  torch.cat((fx, F.conv2d(px, self.kernel)), dim=1)

if __name__ == '__main__':

    fp = 'dgx_ifremer'
    # cfgn = 'qxp20_swot_sst'
    cfgn = 'xp_aug/xp_repro/full_core_hanning'
    OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
    overrides = [
        # '+datamodule.dl_kwargs.shuffle=False',
        f'file_paths={fp}',
        'params.files_cfg.oi_path=${file_paths.oi_4nadir}',
        'params.files_cfg.obs_mask_path=${file_paths.new_noisy_swot}',
        'params.files_cfg.obs_mask_var=five_nadirs',
        '+params.model_name=4dvarnet_cal_grid',
        'params.val_diag_freq=3',
        '+params.alpha_cal=10',
        '+params.swot_obs_w=0.1',
        '+params.warmup_epochs=0',
        '+params.compat_base=False',
        '+params.obscost_downsamp=1',
        '+params.calref=x',
        'params.lr_update=[0.0005]',
        # '+params.calref=xlr',
        # '+params.calref=oi',
        'params.automatic_optimization=false',
        'params.patch_weight._target_=lit_model_augstate.get_constant_crop',
        'params.dT=11',
        'params.patch_weight.crop.time=3',
    ]
    map_cfg_n, map_ckpt = 'qxp20_5nad_no_sst', 'results/xp20/qxp20_5nad_no_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=85-val_loss=0.7589.ckpt'
    map_cfg_n, map_ckpt = 'qxp20_swot_no_sst', 'results/xp20/qxp20_swot_no_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=131-val_loss=0.4958.ckpt'
    mapmod = utils.get_model(
        map_cfg_n,
        map_ckpt,
        add_overrides=overrides
    )
    
    print(swath_calib.configs.register_configs())
    cal_cfg_n, cal_ckpt = 'ffFalse_swath_calib_qxp20_5nad_no_sst', 'lightning_logs/117_ffFalse_swath_calib_qxp20_5nad_no_sst/checkpoints/last.ckpt'
    cal_cfg = utils.get_cfg(cal_cfg_n)

    net = swath_calib.models.build_net(
            in_channels=22,
            out_channels=1,
            **cal_cfg.net_cfg
    )
    normnet = torch.nn.Sequential(
        torch.nn.BatchNorm2d(num_features=22, affine=False, momentum=0.1),
        net
    )

    cal_mod = swath_calib.models.LitDirectCNN(
            # net,
            normnet,
            # gt_var_stats=[s[train_ds.gt_vars].to_array().data for s in train_ds.stats],
            gt_var_stats=[np.array([0]), np.array([1])],
            **cal_cfg.lit_cfg
        )

    print(cal_mod.load_state_dict(torch.load(cal_ckpt, map_location='cpu')['state_dict']))

    cfg_4dvar = utils.get_cfg(cfgn, overrides=overrides)
    OmegaConf.resolve(cfg_4dvar)
    splits = OmegaConf.masked_copy(cfg_4dvar.datamodule, ['train_slices', 'val_slices', 'test_slices'])
    grid_kwargs = hydra.utils.call(OmegaConf.masked_copy(cfg_4dvar.datamodule, ['slice_win', 'dim_range', 'strides']))
    dl_kwargs = {
        'batch_size': 2,
        'num_workers': 4
    }
    sensor_kwargs =dict(
        nadir_paths=[None],
        # nadir_paths=tuple([f'../sla-data-registry/sensor_zarr/zarr/nadir/{name}' for name in ['swot', 'en', 'tpn', 'g2', 'j1']]),
        # swot_path=None,
        swot_path=f'../sla-data-registry/sensor_zarr/zarr/new_swot',
        min_swot_length=None,
        nadir_var='ssh_model',
        swot_gt_vars=('ssh_model',),
        swot_obs_vars=('ssh_model', 'wet_tropo_res', 'syst_error_uncalibrated'),
    )
    ds_kwargs = dict(
        OmegaConf.masked_copy(cfg_4dvar.datamodule,['gt_path', 'gt_var', 'oi_path', 'oi_var', 'obs_mask_path', 'obs_mask_var']),
        grid_kwargs=grid_kwargs,
        sens_kwargs=sensor_kwargs,
    )
    print(ds_kwargs)
    dm = FourDVarMixedGeometryDatamodule(ds_kwargs=ds_kwargs, dl_kwargs=dl_kwargs, **hydra.utils.call(splits),)
    
    xp_num = 6 

    ################ TRAIN ####################
    def train():
        ckpt = None
        print(ckpt)
        lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])


        # print(lit_mod.model.load_state_dict(mapmod.model.state_dict(), strict=False))
        # print(lit_mod.model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))

        pw = hydra.utils.call(cfg_4dvar.params.patch_weight)
        lit_mod.patch_weight.data = torch.from_numpy(pw)

        vcb = swath_calib.versioning_cb.VersioningCallback()
        lrcb = pl.callbacks.LearningRateMonitor()
        logger = pl.loggers.TensorBoardLogger(
            'mixed_geom_logs',
            name=f'{xp_num}_{int(sensor_kwargs["swot_path"] is None)}_{cfg_4dvar.params.calref}',
        )
        trainer = pl.Trainer(gpus=1, logger=logger, weights_summary='full', callbacks=[vcb, lrcb])
        trainer.fit(lit_mod, datamodule=dm)


    #####               TEST #########################
    def test():
        ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_7/checkpoints').glob('*.ckpt')))

        print(ckpt)
        lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])
        logger = pl.loggers.TensorBoardLogger(
            'mixed_geom_logs',
            name=f'test_{xp_num}_{int(sensor_kwargs["swot_path"] is None)}_{cfg_4dvar.params.calref}',
        )
        trainer = pl.Trainer(gpus=[7], logger=logger, weights_summary='full')
        trainer.test(lit_mod, datamodule=dm)


    train()
    # test()
    # trainer.test(lit_mod, dataloaders=dm.test_dataloader())
    # out = trainer.predict(lit_mod, dataloaders=dm.val_dataloader())
    def s2xr(v, c):                                              
        v, c = v.detach().cpu().numpy(), [cc.detach().cpu().numpy() for cc in c]   
        dims = tuple([f'd{di}' for di, _ in enumerate(v.shape)]) 
        ds = xr.Dataset({                                        
          'value': (dims, v),                                  
          't': (dims, c[0]),                                   
          'x': (dims, c[1]),                                   
          'y': (dims, c[2]),                                   
        })                                                       
        return ds                                                
                                                               
    def g2xr(v, c):                                              
        v, c = v.detach().cpu().numpy(), [cc.detach().cpu().numpy() for cc in c]   
        dims = tuple([f'd{di}' for di, _ in enumerate(v.shape)]) 
        ds = xr.Dataset({                                        
          'value': (dims, v),                                  
          't': ((dims[0], dims[-3]), c[0]),                    
          'x': ((dims[0], dims[-2]), c[1]),                    
          'y': ((dims[0], dims[-1]), c[2]),                    
        })                                                       
        return ds                                                
