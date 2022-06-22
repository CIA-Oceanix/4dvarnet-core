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
            crop_time=0,
        ):
        self.xr_ds = xr_ds
        self.nadir_paths = nadir_paths
        self.swot_path = swot_path
        self.nadir_vars = [nadir_var]
        self.swot_gt_vars = list(swot_gt_vars)
        self.swot_obs_vars = list(swot_obs_vars)
        self.obs_downsamp = obs_downsamp
        self.min_swot_length = min_swot_length
        self.crop = pd.to_timedelta(f'{crop_time}D')

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
           
            if swot is not None:
                sp_ds = swath_calib.interp.stack_passes(*swath_calib.interp.add_nb_ch(swot), min_len=self.min_swot_length)
                if sp_ds is not None:
                    spc =  swath_calib.interp.fmt_s_coords(sp_ds)
                    spgt = swath_calib.interp.fmt_s_value(sp_ds, self.swot_gt_vars , spc[0].isfinite())
                    spv = swath_calib.interp.fmt_s_value(sp_ds, self.swot_obs_vars , spc[0].isfinite())
                else:
                    spgt, spv, spc = None, None, (None, None, None)
            else:
                spgt, spv, spc = None, None, (None, None, None)
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
            torch.tensor([t is not None for t in sv]),
            cstack(nv),
            list(map(cstack, zip(*nc))),
            torch.tensor([t is not None for t in nv]),
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
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter + hparams.joint_step)

class LitModMixGeom(lit_model_augstate.LitModelAugstate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_model = self.create_model(self, *args, **kwargs)

    def on_train_epoch_start(self):
        pass

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

        pred = out if ((self.hparams.joint_step==0) or (self.current_epoch > self.hparams.warmup_epochs)) else out_w_cal 
        return {'gt'    : (gt.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (oi.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (go.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                # 'pred' : (pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                # 'pred_wo_cal' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_T
                }

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out=None
        if phase == 'train':
            if not self.automatic_optimization:
                opt, opt_cal, joint_opt, joint_opt_cal = self.optimizers()
                opt.zero_grad()
        for _ in range(self.hparams.n_fourdvar_iter):
            (_loss, _loss_cal), out, state, _metrics = self.compute_loss(
                    batch, phase=phase, state_init=state_init, use_swot=False, compute_cal=((_+1)==self.hparams.n_fourdvar_iter)
            )
            state_init = [None if s is None else s.detach() for s in state]
            if phase == 'train':
                if not self.automatic_optimization and self.hparams.train_map_mod:
                    self.manual_backward(_loss)
            losses.append(_loss)
            metrics.append(_metrics)

        if phase == 'train':
            if not self.automatic_optimization:

                if self.hparams.train_map_mod:
                    if self.current_epoch > self.hparams.warmup_epochs:
                        if _loss_cal > 0:
                            opt_cal.zero_grad()
                            self.manual_backward(_loss_cal)
                            opt_cal.step()
                    opt.step()

        (_loss, _loss_cal), out_w_cal, state, metrics_w_cal = self.compute_loss(
                batch,
                phase='train',
                state_init=state_init[:1],
                use_swot=True,
                compute_cal=True,
                model=self.joint_model
        )
        if (phase == 'train') and (self.hparams.joint_step>0) and (self.current_epoch > self.hparams.start_join):
            if not self.automatic_optimization:
                joint_opt.zero_grad()
                joint_opt_cal.zero_grad()
                self.manual_backward(_loss)
                if _loss_cal > 0:
                    self.manual_backward(_loss_cal)
                    joint_opt_cal.step()
                joint_opt.step()
        # if (phase == 'train'):
                # opt_cal.step()
        return losses, out, out_w_cal, metrics, metrics_w_cal


    def training_step(self, train_batch, batch_idx):

        # compute loss and metrics
        losses, _, _, metrics, metrics_w_cal = self(train_batch, phase='train')

        self.log("tr_loss", losses[-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mse_w_cal", metrics_w_cal['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_cal", metrics[-1]['mseCal'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_cal_joint", metrics_w_cal['mseCal'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return losses[-1]


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
        
        go_oi = lambda:  msk.float() * (go.where(msk, torch.zeros_like(go)) - oi)
        if self.hparams.rand_init:
            go_oi = lambda:  (go.where(msk, torch.zeros_like(go)) - oi).where(msk, 0.1*torch.randn_like(go))

        init_state = torch.cat((oi,go_oi()), dim=1)
        if self.hparams.aug_cal_obs:
            init_state = torch.cat((init_state, torch.zeros_like(go_oi())), dim=1)
        if self.aug_state:
            init_state = torch.cat((init_state, go_oi()), dim=1)
        return init_state

    def cal_out(self, y, xb):
        _, gc, sv, sc, si, _, _, ni = y
        s_vb = swath_calib.interp.batch_torch_interpolate_with_fmt(xb[si], *[cc[si] for cc in gc], *sc)
        cal_out, cal_msk, cal_idx = self.model.model_H.cal_out(sv, s_vb.detach())

        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        pw = einops.repeat(self.patch_weight, '... -> b ...', b=xb.size(0))
        sw = swath_calib.interp.batch_torch_interpolate_with_fmt(pw, *[cc[si] for cc in gc], *sc)
        sw = fna(sw, sw.isfinite())
        return cal_out, s_vb, cal_msk, cal_idx, sw

    def loss_cal(self, sgt, y, xb):

        _, _, sv, *_= y
        if sv is None:
            return torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        cal_out, _, cal_msk, cal_idx, sw = self.cal_out(y, xb)

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
                            # {'params': self.model.model_H.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            ])
        optimizer_cal = opt([
                            # {'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                            # {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.model.model_H.parameters(), 'lr': 1 * self.hparams.lr_update[0]},
                            # {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            ])

        joint_optimizer = opt([{'params': self.joint_model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.joint_model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                            # {'params': self.model.model_H.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            {'params': self.joint_model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            ])
        joint_optimizer_cal = opt([
                            # {'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                            # {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.joint_model.model_H.parameters(), 'lr': 1 * self.hparams.lr_update[0]},
                            # {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            ])

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=25, cooldown=10, factor=0.5),
                'monitor': 'val_loss'
            },{
                'optimizer': optimizer_cal,
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cal, verbose=True, patience=25, cooldown=10, factor=0.5),
                'monitor': 'val_loss'
            },
            {
                'optimizer': joint_optimizer,
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(joint_optimizer, verbose=True, patience=25, cooldown=10, factor=0.5),
                'monitor': 'val_loss'
            },{
                'optimizer': joint_optimizer_cal,
                'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(joint_optimizer_cal, verbose=True, patience=25, cooldown=10, factor=0.5),
                'monitor': 'val_loss'
            }
        ]


    def compute_loss(self, batch, phase, state_init=(None,), use_swot=False, compute_cal=False, opt_map_for_cal=False, model=None):
        oi, gt, sgt, *y = batch

        # handle patch with no observation
        gt_wo_nan = gt.where(~ gt.isnan(), oi)
        oi_wo_nan = oi.where(~ oi.isnan(), oi)

        state = self.get_init_state(batch, state_init)

        obs = (oi, y)
        if not use_swot:
            (go, gc, _, *_) = y 
            obs = (oi, (go, gc, None, *_))

        msks = (torch.ones_like(oi), None)

        # gradient norm field
        g_gt_x, g_gt_y = self.gradient_img(gt)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            if model is None:
                model = self.model
            x_out, hidden_new, cell_new, normgrad = model(state, obs, msks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                x_out = x_out.detach()

        xlr, *_, xhr = torch.split(x_out, self.hparams.dT, dim=1)
        outputs = xlr + xhr

        # median filter
        if self.median_filter_width > 1:
            outputs = kornia.filters.median_blur(outputs, (self.median_filter_width, self.median_filter_width))



        if self.hparams.calref =='oi':
            xb = oi_wo_nan
        elif self.hparams.calref =='xlr':
            xb = xlr
        elif self.hparams.calref =='x':
            xb = outputs

        lcal_in = y
        if not compute_cal:
            (go, gc, _, *_) = y
            lcal_in = (go, gc, None, *_)

        if (not self.hparams.joint_step) or (not opt_map_for_cal):
            xb = xb.detach()
        loss_cal, g_loss_cal, l_loss_cal = self.loss_cal(sgt, lcal_in, xb)

        yGT = torch.cat((oi_wo_nan, gt_wo_nan - xlr), dim=1)
        if self.aug_state:
            yGT = torch.cat((yGT, gt_wo_nan - xlr), dim=1)
        if self.hparams.aug_cal_obs:
            yGT = torch.cat((yGT, gt_wo_nan - xlr), dim=1)

        loss_All, loss_GAll = self.sla_loss(outputs, gt_wo_nan)
        loss_OI, loss_GOI = self.sla_loss(oi_wo_nan, gt_wo_nan)
        loss_AE, loss_AE_GT, loss_SR, loss_LR =  self.reg_loss(
            yGT, oi, outputs, xlr, x_out
        )

        # print(loss_All, loss_GAll,loss_AE, loss_AE_GT, loss_SR, loss_LR, loss_cal)
        # total loss
        loss = 0
        loss += self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
        loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
        loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
        loss_cal = self.hparams.alpha_cal * (loss_cal + 0.5*g_loss_cal + 0.5*l_loss_cal)

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

        return (loss, loss_cal), outputs, [x_out, hidden_new, cell_new, normgrad], metrics

class CalibrationModelObsGridGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None, min_size=300, sigs=tuple([8* (i+1) for i in range(10)])):
        super().__init__()
        self.hparams = hparams
        sst_ch = hparams.dT
        # self.dim_obs_channel = np.array([shape_data, sst_ch])
        # self.dim_obs = 3
        # self.dim_obs = 2
        # self.dim_obs_channel = np.array([shape_data, shape_data])
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_data])
        self.min_size = min_size
        self.num_feat = 2*(len(sigs)+1)
        self.norm_in = torch.nn.BatchNorm2d(num_features=self.num_feat, affine=True, momentum=0.1)
        # self.norm_out = torch.nn.BatchNorm2d(num_features=1, affine=True, momentum=0.1)
        self.norm_out = torch.nn.Identity()

        self.gaussian = StackedGaussian(sigs)
        self.calnet = torch.nn.Sequential(
            self.norm_in,
            swath_calib.models.build_net(
                in_channels=self.num_feat,
                out_channels=1,
                mix=True,
            ),# nhidden = 32, depth = 2,),
            self.norm_out,
        )
        self.downsamp = self.hparams.obscost_downsamp

        self.patch_weight = torch.nn.Parameter(
                torch.from_numpy(hydra.utils.call(self.hparams.patch_weight)), requires_grad=False)
        self.scale_cal = torch.nn.Parameter(torch.scalar_tensor(self.hparams.scale_cal), requires_grad=self.hparams.train_scale_cal)
        self.ns = (0.31446309894037083, 0.3886609494201447)
        # self.conv11 = torch.nn.Conv2d(shape_data, hparams.dT, (3, 3), padding=1, bias=False)
        # self.conv21 = torch.nn.Conv2d(hparams.dT, hparams.dT, (3, 3), padding=1, bias=False)
        # self.conv_m = torch.nn.Conv2d(hparams.dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        # self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)


    def cal_inp(self, sv_uncal, sv_bg):
        fs_xlr = einops.rearrange(sv_bg, 'b p ... -> (b p) () ...')

        fy = einops.rearrange(sv_uncal, 'b p ... -> (b p) () ...')

        msk = fy[:, :, :self.min_size, :].isfinite().all(3).all(2).all(1)
        idx = fs_xlr[ msk, ...].isfinite().all(3).all(1).all(0)

        gy =  self.gaussian(fy[msk][:, :, idx])
        inp_gy = torch.cat((gy.diff(dim=1), gy[:, -1:, ...]), 1)

        gs_xlr = self.gaussian(fs_xlr[msk][:, :, idx])
        inp_gs_xlr = torch.cat((gs_xlr.diff(dim=1), gs_xlr[:, -1:, ...]), 1)

        cal_input = -torch.cat((inp_gy, inp_gs_xlr), dim=1)
        return cal_input

    def cal_out(self, sv_uncal, sv_bg):
        sh = einops.parse_shape(sv_bg, 'b p ...')
        fs_xlr = einops.rearrange(sv_bg, 'b p ... -> (b p) () ...')

        fy = einops.rearrange(sv_uncal, 'b p ... -> (b p) () ...')

        msk = fy[:, :, :self.min_size, :].isfinite().all(3).all(2).all(1)
        idx = fs_xlr[ msk, ...].isfinite().all(3).all(1).all(0)

        gy =  self.gaussian(fy[msk][:, :, idx])
        inp_gy = torch.cat((gy.diff(dim=1), gy[:, -1:, ...]), 1)

        gs_xlr = self.gaussian(fs_xlr[msk][:, :, idx])
        inp_gs_xlr = torch.cat((gs_xlr.diff(dim=1), gs_xlr[:, -1:, ...]), 1)

        cal_input = -torch.cat((inp_gy, inp_gs_xlr), dim=1)
        _out_cal = self.calnet(cal_input)
        _out_cal = _out_cal  + fs_xlr[msk][:,:,idx]
        # _out_cal = _out_cal / self.scale_cal + self.ns[0]/self.ns[1] + fs_xlr[msk][:,:,idx]
        # _out_cal = _out_cal + self.ns[0]/self.ns[1] + fs_xlr[msk][:,:,idx]
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
        ylr, (go, gc, sv, sc, si, nv, nc, ni) = y
        xlr, *_, anom_rec = torch.split(x, self.hparams.dT, dim=1)


        # if self.hparams.compat_base:
        #     anom_rec = anom_obs

        if nv is not None:
            g_cal, gmsk = swath_calib.interp.batch_interp_to_grid(
                ylr[ni], *[cc[ni] for cc in gc], nv, *nc
            )
            # dyout1 = gmsk.float() * (g_cal - (xlr + anom_rec))


            # Baseline
            # yy = torch.cat((ylr, (g_cal - ylr).where(gmsk, torch.zeros_like(ylr)), ylr), dim=1) 
            # mm = torch.cat((torch.ones_like(ylr), gmsk, torch.zeros_like(go)), dim=1)
            # dyout = torch.abs((yy -x )  * mm)

            # Move nadir obs to rec
            yy = torch.cat((ylr[ni], ylr[ni], (g_cal - ylr[ni]).where(gmsk, torch.zeros_like(ylr[ni]))), dim=1) 
            mm = torch.cat((torch.ones_like(ylr[ni]), torch.zeros_like(go[ni]), gmsk), dim=1)
            dyout = torch.abs((yy -x[ni] )  * mm)
        else:
            # dyout =torch.zeros_like(x)
            msk = go.isfinite()
            gooi = (go - ylr).where(msk, torch.zeros_like(ylr))

            yy = torch.cat((ylr, gooi, gooi), dim=1) 
            mm = torch.cat((torch.ones_like(ylr), msk, torch.zeros_like(msk)), dim=1)

            if self.hparams.aug_cal_obs:
                yy = torch.cat((yy, gooi), dim=1) 
                mm = torch.cat((mm, torch.zeros_like(gooi)), dim=1) 
            dyout = ((yy - x )  * mm)

        if sv is not None:
            sgc  = [cc[si] for cc in gc]
            if self.hparams.calref == 'oi':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(ylr[si], *sgc , *sc)
            elif self.hparams.calref == 'x':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt((xlr + anom_rec)[si],  *sgc, *sc)
            elif self.hparams.calref == 'xlr':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr[si], *sgc, *sc)

            out_cal, cal_msk, cal_idx = self.cal_out(sv, s_xlr.detach())
            g_cal, gmsk = swath_calib.interp.batch_interp_to_grid(
                    ylr[si], *sgc, out_cal.detach(), *map(lambda t: t[:, :, cal_idx, ...], sc)
            )
            g_cal_msk, _ = swath_calib.interp.batch_interp_to_grid(
                    ylr[si], *sgc, cal_msk.float(), *map(lambda t: t[:, :, cal_idx, ...], sc)
            )
            yy = ylr[si]
            mm = torch.ones_like(ylr[si])
            if self.hparams.aug_cal_obs:
                yy = torch.cat((yy, ylr[si]), dim=1) 
                mm = torch.cat((mm, torch.zeros_like(ylr[si])), dim=1) 
            yy = torch.cat((yy, (g_cal - ylr[si]).where(g_cal_msk.bool() & gmsk, ylr[si]), ylr[si]), dim=1) 
            mm = torch.cat((mm, self.patch_weight[None, ...] * gmsk.float() * g_cal_msk, torch.zeros_like(ylr[si])), dim=1)
            # print(yy.shape, mm.shape, x[si].shape)


            dyout1 = ((yy -x[si] )  * mm)
            # dyout = self.hparams.swot_obs_w * dyout

        else:
            dyout1 = torch.zeros_like(x)

        return dyout.abs() +  dyout1.abs()
        # return [dyoutlr, dyout, dyout1]
        
class CalibrationModelObsSensorGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None, min_size=450, sigs=tuple([8* (i+1) for i in range(10)])):
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
    def __init__(self, sigs= tuple([8 * (i+1) for i in range(20)]), size=75, min_size=300):
        super().__init__()
        assert size % 2 ==1, 'same pad not yet implemnted here for even kernels'
        size=max(sigs)*6 //2 *2 +1
        self.size = size
        self.min_size = min_size
        self.kernel = torch.nn.Parameter(torch.stack([
            kornia.filters.get_gaussian_kernel1d(size, sig)
            for sig in sigs
        ])[:, None, :, None], requires_grad=False)

  
    def forward(self, fx):
        px = F.pad(fx.nan_to_num(), (0,0, self.size//2, self.size//2), mode='reflect')
        return  torch.cat((fx, F.conv2d(px, self.kernel)), dim=1)

if __name__ == '__main__':

    fp = 'dgx_ifremer'
    # cfgn = 'qxp20_swot_sst'
    cfgn = 'baseline/full_core_hanning'
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
        '+params.train_map_mod=True',
        '+params.start_join=0',
        '+params.compat_base=False',
        '+params.joint_step=1',
        '+params.rand_init=0',
        '+params.obscost_downsamp=1',
        '+params.aug_cal_obs=0',
        '+params.scale_cal=1',
        '+params.train_scale_cal=False',
        "params.shape_state.0='${mul:${datamodule.slice_win.time},3}'",
        "params.shape_data.0='${mul:${datamodule.slice_win.time},3}'",
        '+params.calref=x',
        # 'params.lr_update=[0.0005]',
        # '+params.calref=xlr',
        # '+params.calref=oi',
        'params.automatic_optimization=false',
        'params.patch_weight._target_=lit_model_augstate.get_constant_crop',
        'params.dT=11',
        'params.patch_weight.crop.time=3',
    ]
    map_cfg_n, map_ckpt = 'qxp21_5nad_no_sst_11', 'dashboard/qxp21_5nad_no_sst_11/version_0/checkpoints/modelCalSLAInterpGF-epoch=80-val_loss=1.7154.ckpt'
    # map_cfg_n, map_ckpt = 'qxp20_swot_no_sst', 'results/xp20/qxp20_swot_no_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=131-val_loss=0.4958.ckpt'
    mapmod = utils.get_model(
        map_cfg_n,
        map_ckpt,
        add_overrides=overrides
    )
    
    print(swath_calib.configs.register_configs())
    cal_cfg_n, cal_ckpt = 'ffFalse_swath_calib_qxp21_5nad_no_sst_11', 'lightning_logs/120_ffFalse_swath_calib_qxp21_5nad_no_sst_11/checkpoints/epoch=114-step=916.ckpt'
    cal_cfg = utils.get_cfg(cal_cfg_n)

    net = swath_calib.models.build_net(
            in_channels=22,
            out_channels=1,
            **cal_cfg.net_cfg
    )
    normnet = torch.nn.Sequential(
        torch.nn.BatchNorm2d(num_features=22, affine=True, momentum=0.1),
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
        min_swot_length=450,
        crop_time=cfg_4dvar.params.patch_weight.crop.time,
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
    
    xp_num = 10 

    ################ TRAIN ####################
    def train():
        ckpt = None
        print(ckpt)
        lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])


        print(lit_mod.model.load_state_dict(mapmod.model.state_dict(), strict=False))
        print(lit_mod.model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))

        print(lit_mod.joint_model.load_state_dict(mapmod.model.state_dict(), strict=False))
        print(lit_mod.joint_model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))

        vcb = swath_calib.versioning_cb.VersioningCallback()
        lrcb = pl.callbacks.LearningRateMonitor()
        from pytorch_lightning import callbacks
        mckptcb = callbacks.ModelCheckpoint(save_top_k=5, monitor='val_loss', save_last=True)
        logger = pl.loggers.TensorBoardLogger(
            'mixed_geom_logs',
            name=f'{xp_num}_{int(sensor_kwargs["swot_path"] is None)}_{cfg_4dvar.params.calref}',
        )
        trainer = pl.Trainer(gpus=1, max_epochs=300, logger=logger, weights_summary='full', callbacks=[vcb, lrcb, mckptcb])
        trainer.fit(lit_mod, datamodule=dm)


    #####               TEST #########################
    def test_one():
        vnum=6
        # ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_10/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_21/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_40/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/6_1_x/version_1/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path(f'mixed_geom_logs/7_0_x/version_{vnum}/checkpoints').glob('last.ckpt')))
        ckpt = str(next(Path(f'mixed_geom_logs/10_0_x/version_{vnum}/checkpoints').glob('last.ckpt')))
        # ckpt = next(Path('mixed_geom_logs/10_0_x/version_5/checkpoints').glob('last.ckpt'))
        # ckpt = str(next(Path('mixed_geom_logs/7_0_x/version_6/checkpoints').glob('epoch=59-step=6840.ckpt')))
        # ckpt = 'joint.ckpt'
        # cfg_4dvar.params.patch_weight._target_='lit_model_augstate.get_cropped_hanning_mask'
        ckpt =None
        metrics ={}
        for cal in [False]:
            lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])
            # print(lit_mod.model.load_state_dict(mapmod.model.state_dict(), strict=False))
            # print(lit_mod.model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))
            lit_mod.hparams.rand_init=0
            ckpt=str(ckpt)
            if cal:
                bak = cfg_4dvar.params.patch_weight.crop.time
                cfg_4dvar.params.patch_weight.crop.time=cfg_4dvar.params.dT//2
                pw = hydra.utils.call(cfg_4dvar.params.patch_weight)
                lit_mod.patch_weight.data = torch.from_numpy(pw)
                cfg_4dvar.params.patch_weight.crop.time=bak

            name = f'test_{xp_num}_{vnum}_{torch.load(ckpt)["epoch"]}_{cal}'
            # name = f'test_baseline'
            import pytorch_lightning.loggers
            logger = pytorch_lightning.loggers.TensorBoardLogger(
                'mixed_geom_logs',
                name=name,
                default_hp_metric=False
            )
            trainer = pl.Trainer(gpus=[6], logger=logger, weights_summary='full')
            print(ckpt)

            test_out = trainer.test(lit_mod, datamodule=dm)
            metrics[name] = dict(
                ckpt=ckpt,
                test_out=test_out,
                test_xrds=lit_mod.test_xr_ds,
                metrics=lit_mod.latest_metrics,
            )
            with open(f'tmp/results_{name}', 'wb') as f:
                pickle.dump(metrics, f)

    def test_all():
        # ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_10/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_21/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/6_0_x/version_40/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/6_1_x/version_1/checkpoints').glob('*.ckpt')))
        # ckpt = str(next(Path('mixed_geom_logs/9_0_x/version_10/checkpoints').glob('last.ckpt')))
        # ckpt = next(Path('mixed_geom_logs/7_0_x/version_5/checkpoints').glob('last.ckpt'))
        # ckpt = str(next(Path('mixed_geom_logs/7_0_x/version_6/checkpoints').glob('epoch=59-step=6840.ckpt')))
        # ckpt = 'joint.ckpt'
        # cfg_4dvar.params.patch_weight._target_='lit_model_augstate.get_cropped_hanning_mask'
        metrics = {}
        for version, joint_step in [
                # (5, 0),
                # (6, 1),
                # (8, 1),
                (2,1),
                # (3,1),
            ]:
            for ckpt in Path(f'mixed_geom_logs/10_0_x/version_{version}/checkpoints').glob('*.ckpt'):
                print(ckpt)

                if ckpt.name == 'last.ckpt': continue
                for cal in [False]:#, True]:
                    lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])
                    print(lit_mod.hparams.rand_init)
                    lit_mod.hparams.rand_init=0
                    print(lit_mod.hparams.joint_step)
                    ckpt=str(ckpt)
                    if cal:
                        bak = cfg_4dvar.params.patch_weight.crop.time
                        cfg_4dvar.params.patch_weight.crop.time=cfg_4dvar.params.dT//2
                        pw = hydra.utils.call(cfg_4dvar.params.patch_weight)
                        lit_mod.patch_weight.data = torch.from_numpy(pw)
                        cfg_4dvar.params.patch_weight.crop.time=bak


                    name = f'test_{xp_num}_{version}_{torch.load(ckpt)["epoch"]}_{cal}'
                    import pytorch_lightning.loggers
                    logger = pytorch_lightning.loggers.TensorBoardLogger(
                        'mixed_geom_logs',
                        name=f'test_{xp_num}_{version}_{torch.load(ckpt)["epoch"]}_{cal}',
                        default_hp_metric=False
                    )
                    trainer = pl.Trainer(gpus=[6], logger=logger, weights_summary='full')
                    print(ckpt)

                    test_out = trainer.test(lit_mod, datamodule=dm)
                    metrics[name] = dict(
                        ckpt=ckpt,
                        test_out=test_out,
                        test_xrds=lit_mod.test_xr_ds,
                        metrics=lit_mod.latest_metrics,
                    )
                    with open('tmp/resultsbis0906', 'wb') as f:
                        pickle.dump(metrics, f)
            return metrics



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

    class LitTestCal(LitModMixGeom):
        def predict_step(self, batch, batch_idx):
            oi, gt, sgt, *y = batch
            _, gc, sv, sc, si, _, _, ni = y
            if sv is None:
                return None
            losses, out, out_w_cal, metrics, metrics_w_cal = self.forward(batch, phase='predict')
            if out is None or sv is None:
                return None
            print(metrics)
            cal_out, sv_b, cal_msk, cal_idx, sw = self.cal_out(y, out)
            fna = lambda t, m: t.where(m, torch.zeros_like(t))
            fmt = lambda t: fna(t[:,:, cal_idx, :], cal_msk)
            cout, cgt, sc, tw, cin = cal_out, fmt(sgt),[fmt(cc) for cc in sc], fmt(sw), fmt(sv_b)

            cal_out_joint, sv_b_joint, *_ = self.cal_out(y, out_w_cal)
            cout_joint, cin_joint = cal_out_joint, fmt(sv_b_joint)

            ds = xr.merge([
                    s2xr(cout, sc).rename(value='cal'),
                    s2xr(cin, sc).rename(value='map'),
                    s2xr(cout_joint, sc).rename(value='cal_joint'),
                    s2xr(cin_joint, sc).rename(value='map_joint'),
                    s2xr(cgt, sc).rename(value='gt'),
                    s2xr(tw, sc).rename(value='weight'),
            ])
            return ds

    def predict():
        # ckpt = str(next(Path('mixed_geom_logs/10_0_x/version_1/checkpoints').glob('*.ckpt')))
        ckpt = None
        ckpt = Path('mixed_geom_logs/10_0_x/version_3/checkpoints/epoch=98-step=43480.ckpt')
        # ckpt = Path('mixed_geom_logs/10_0_x/version_2/checkpoints/epoch=22-step=9870.ckpt')
        # ckpt = str(Path('mixed_geom_logs/7_0_x/version_9/checkpoints/epoch=156-step=17898.ckpt'))
        lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitTestCal'])
        # print(lit_mod.model.load_state_dict(mapmod.model.state_dict(), strict=False))
        # print(lit_mod.model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))
        # bak = cfg_4dvar.params.patch_weight.crop.time
        # cfg_4dvar.params.patch_weight.crop.time=cfg_4dvar.params.dT//2
        # pw = hydra.utils.call(cfg_4dvar.params.patch_weight)
        # lit_mod.patch_weight.data = torch.from_numpy(pw)
        # cfg_4dvar.params.patch_weight.crop.time=bak
        trainer = pl.Trainer(gpus=[6], logger=False, weights_summary='full')

        predictions = trainer.predict(lit_mod, dm.test_dataloader())

    
        ns = (0.31446309894037083, 0.3886609494201447)
        sobel = lambda da: np.hypot(ndi.sobel(da, -1), ndi.sobel(da, -2))
        sums = []
        for p in [p for p in predictions if p is not None]:
            if p.weight.pipe(lambda da: da ==0).all().item(): continue
            sums.append(
                    p[['map', 'cal','map_joint', 'cal_joint', 'gt', 'weight']]
                    .pipe(lambda ds: ds * p.weight)
                    # .map(sobel)
                    .pipe(lambda ds: ds - ds.gt).drop('gt')
                    .pipe(lambda ds: ds**2).sum()
            )


        print(xr.concat(sums, dim='batch').sum().pipe(lambda ds: ds/ds.weight).drop('weight').pipe(np.sqrt).pipe(lambda da: da*ns[1]))
        return predictions

    # train()
    # metrics = test_all()
    # test_one()
    # preds = predict()

    # ckpt = None
    # # ckpt = str(next(Path('mixed_geom_logs/7_0_x/version_6/checkpoints').glob('epoch=125-step=14364.ckpt')))
    # # ckpt = str(Path('mixed_geom_logs/7_0_x/version_9/checkpoints/epoch=156-step=17898.ckpt'))
    # lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitTestCal'])
    # print(lit_mod.model.load_state_dict(mapmod.model.state_dict(), strict=False))
    # print(lit_mod.model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))

    # print(lit_mod.joint_model.load_state_dict(mapmod.model.state_dict(), strict=False))
    # print(lit_mod.joint_model.model_H.calnet.load_state_dict(cal_mod.net.state_dict(), strict=False))
    # trainer = pl.Trainer(gpus=[6], weights_summary='full')
    # trainer.test(lit_mod, dm.test_dataloader())
    # tmp_test = 'tmp/test.nc'
    # lit_mod.test_xr_ds.to_netcdf(tmp_test)
    # new_grid_kwargs = {
    #     'slice_win': {'lat': 200, 'lon': 200, 'time': 11}, 
    #     'strides': {'lat': 200, 'lon': 200, 'time': 1},
    #     'dim_range': {'lat': slice(33.0, 43.0, None), 'lon': slice(-65.0, -55.0, None)}
    # }

    # ns = (0.31446309894037083, 0.3886609494201447)
    # ds = FourDVarMixedGeometryDataset(
    #     slice(None, None),
    #     tmp_test, 'gt',
    #     tmp_test, 'oi',
    #     tmp_test, 'pred',
    #     new_grid_kwargs, sensor_kwargs, norm_stats=ns,
    # )
    # len(ds)
    # # dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
    # # dm.setup()
    # dl = dm.val_dataloader()
    # lit_mod = lit_mod.to('cuda:7')
    # lit_mod.eval()
    # with torch.no_grad():
    #     for batch in dl:
    #         batch = lit_mod.transfer_batch_to_device(batch, lit_mod.device,0)
    #         oi, gt, sgt, *y = batch
    #         (go, gc, sv, sc, si, nv, nc, ni) = y
    #         break

    # 1/0
    # g2xr(go.nan_to_num(), gc).value.isel(dict(d0=1, d1=2)).T.plot(figsize=(5,5))
    # s2xr(sgt.nan_to_num(), sc).value.isel(dict(d0=1, d1=2)).T.plot(figsize=(15,3))
    # s2xr(sv.nan_to_num(), sc).value.isel(dict(d0=1, d1=2)).T.plot(figsize=(15,3))
    # xb=go
    # s_xb = swath_calib.interp.batch_torch_interpolate_with_fmt(xb[si], *[cc[si] for cc in gc], *sc)
    # s2xr(s_xb.nan_to_num(), sc).value.isel(dict(d0=1, d1=2)).T.plot(figsize=(15,3))
    # cal_inp = lit_mod.model.model_H.cal_inp(sv, s_xb)

    # cal_out = lit_mod.model.model_H.calnet(cal_inp)
    # cal_inp.shape
    # v= sv.detach().cpu().numpy()
    # vinn= lit_mod.model.model_H.calnet[0](cal_inp).detach().cpu().numpy()
    # vin = cal_inp.detach().cpu().numpy()
    # vp = cal_out.detach().cpu().numpy()
    # vgt = (sgt - s_xb).detach().cpu().numpy()
    # dims = tuple([f'd{di}' for di, _ in enumerate(v.shape)]) 
    # ds = xr.Dataset({'value': (dims, v)},)                                                       
    # dsin = xr.Dataset({'value': (dims, vin)},)                                                       
    # dsinn = xr.Dataset({ 'value': (dims, vinn), })
    # dsp = xr.Dataset({ 'value': (dims, vp), })
    # dsgt = xr.Dataset({ 'value': (dims, vgt), })
    # idx=0
    # dsgt.isel(d0=0, d1=idx).value.plot.pcolormesh(x='d2', y='d3', robust=True, figsize=(15, 3))
    # dsp.isel(d0=idx, d1=0).value.plot.pcolormesh(x='d2', y='d3', robust=False, figsize=(15, 3))
    # (dsp**2).mean(('d1', 'd2', 'd3')).pipe(np.sqrt).to_array().values
    # (dsgt**2).mean(('d2', 'd3')).pipe(np.sqrt).to_array().values
    # ds.isel(d0=idx, d1=0).value.plot.pcolormesh(x='d2', y='d3', figsize=(15, 3))
    # dsin.isel(d0=idx, d1=slice(None, 11)).value.plot.pcolormesh(x='d2', y='d3', col='d1', col_wrap=1, figsize=(15, 15))
    # dsin.isel(d0=idx, d1=slice(11, None)).value.plot.pcolormesh(x='d2', y='d3', col='d1', col_wrap=1, figsize=(15, 15))
    # dsinn.isel(d0=idx, d1=slice(None, 11)).value.plot.pcolormesh(x='d2', y='d3', col='d1', col_wrap=1, figsize=(15, 15))
    # dsinn.isel(d0=idx, d1=slice(11, None)).value.plot.pcolormesh(x='d2', y='d3', col='d1', col_wrap=1, figsize=(15, 15))
    # list(lit_mod.model.model_H.calnet[0].named_buffers())
    # list(lit_mod.model.model_H.calnet[0].named_parameters())
    # plt.plot([8 * (i+1)*2 for i in range(10)], lit_mod.model.model_H.calnet[0].weight[:10].detach().cpu(), label='obs')
    # plt.plot([8 * (i+1)*2 for i in range(10)], lit_mod.model.model_H.calnet[0].weight[11:-1].detach().cpu(), label='grid product')
    # plt.title("Learnt rescaling if function of $\sigma_1$ (km)")
    # plt.legend()
    #         if sv is None:
    #             continue
    #         xb=go
    #         fna = lambda t, m: t.where(m, torch.zeros_like(t))
            # cal_out, _, cal_msk, cal_idx, sw = lit_mod.cal_out(y, xb)
    #         s_xb = swath_calib.interp.batch_torch_interpolate_with_fmt(xb[si], *[cc[si] for cc in gc], *sc)
    #         t1, t2, ref, tw = cal_out, fna(sgt[:,:, cal_idx, :], cal_msk), fna(s_xb[:,:, cal_idx, :], cal_msk), fna(sw[:,:, cal_idx, :], cal_msk)
    #         if tw.sum().item() == 0:
    #             continue

    #         ((t1 -ref)**2 *tw).sum() /tw.sum()

    #         mse = ((t1-t2)**2 * tw).sum() / tw.sum()
    #         mse_ref = ((t1-ref)**2 * tw).sum() / tw.sum()
    #         print(mse.sqrt() * ns[1])
    #         print(mse_ref.sqrt() * ns[1])
    #         # print(mse.sqrt() * ns[1])
    #         # print(mse.sqrt() * ns[1], tw.sum((2,3)))

    

    
    # xrgf = lambda da, sig: da if sig==0 else xr.apply_ufunc(
    #         lambda nda: ndi.gaussian_filter1d(nda, axis=2, sigma=sig, order=0, mode='mirror', truncate=3.0), da)

    # (xrgf(s2xr(sv.nan_to_num(), sc), sig=80)-xrgf(s2xr(sv.nan_to_num(), sc), sig=70)).value.isel(dict(d0=0, d1=0)).T.plot(figsize=(10,3))
    # s2xr(sgt.nan_to_num(), sc).value.isel(dict(d0=0, d1=0)).T.plot(figsize=(10,3))
    # gaus = StackedGaussian([70,80]).to(sv.device)

    # s2xr(gaus(sv[:1, :1].nan_to_num()).diff(dim=1)[:, -1:], [cc[:1, :1] for cc in sc]).value.isel(dict(d0=0, d1=0)).T.plot(figsize=(10,3))


    # it = val_ds[0]
    # len(it)
    # len(it)
    # x, xb, gt, rgt, rxb = it
    # with val_ds.get_coords():
    #     c = val_ds[0]
    # c[['gt', 'syst_error_uncalibrated', 'wet_tropo_res']].to_array().T.plot.pcolormesh(col='variable', col_wrap=1, figsize=(15,8), robust=True)
    # add_inter_sw = lambda ds:(
    #             ds
    #         .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
    #         .swap_dims(nC='x_ac')
    #         .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
    # )
    # c.pipe(add_inter_sw)[['gt', 'syst_error_uncalibrated', 'wet_tropo_res']].to_array().sum('variable').T.plot(figsize=(10, 3))
    # dm.setup()
    # dl = dm.train_dataloader()
    # dl.dataset.datasets[0].gt_ds.ds.ssh

    # def anim(ds):
    #     def sobel(da):
    #         dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
    #         dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
    #         return np.hypot(dx_ac, dx_al)
    #     hvds = hv.Dataset(ds.assign(grad=sobel(ds.ssh)).isel(time=slice(None, 50, 1 )))
    #     images = hv.Layout([
    #             hvds
    #             .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
    #             .options(cmap={'ssh': 'RdYlBu', 'grad':'viridis'}[v])
    #             for v in ['ssh', 'grad']
    #             ]).cols(2).opts(sublabel_format="")
    #     return images

    # images = anim( dl.dataset.datasets[0].gt_ds.ds)
    # # images
    # hv.output(images, holomap='gif', fps=5)

    # hv.save(images, filename='ssh_anim.gif', fps=5)
    # it.shape

    # cal_inp = lit_mod.model.model_H.cal_inp(sv, s_xb)
    # cal_inp.shape
    # cal_mod=cal_mod.to(lit_mod.device)
    # cal_out = cal_mod.net(cal_inp)

    # cm_cal_inp, cm_s_xb, s_gt, raw_gt, raw_ref = lit_mod.transfer_batch_to_device(
    #     next(iter(torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=0))), lit_mod.device, 0
    # )

    # cal_out = cal_mod.net(cal_inp)
    # cal_inp.std((0,2,3))
    # cal_inp.mean((0,2,3))

    # cm_cal_inp.std((0, 2, 3))
    # plt.imshow(cal_out.squeeze().T.detach().cpu())
    # plt.imshow((s_xb - s_gt).squeeze().T.detach().cpu())
    
    # cal_inp.abs().mean((2,3))
    # cal_inp.std((2,3))
    # my_cal_inp = lit_mod.model.model_H.cal_inp(sv, s_xb)
    # F.mse_loss(s_xb, s_gt)
    # F.mse_loss(s_xb+cal_out, s_gt)


# #     preds[1].weight.any(dim=['d2', 'd3'])
# #     # preds[1].isel(d0=0, d1=2).weight.any(dim=['d2', 'd3'])
# #     preds[1].isel(d0=1, d1=2).weight.T.plot(figsize=(10, 3))
# #     preds[1].isel(d0=1, d1=2).pipe(lambda ds: ds.cal - ds['map']).T.plot(figsize=(10, 3))
# #     preds[1].isel(d0=1, d1=2).pipe(lambda ds: ds.gt - ds['map']).T.plot(figsize=(10, 3))
# #     preds[1].isel(d0=1, d1=2).pipe(lambda ds: ds.gt ).T.plot(figsize=(10, 3))
# #     preds[1].isel(d0=1, d1=2).pipe(lambda ds: ds['map']).T.plot(figsize=(10, 3))
    # # mse = lambda t1, t2, tw: ((t1-t2)**2 * tw).sum() / tw.sum()
    # # np.sqrt(np.mean(
    # #     [1. if p is None else (((p.cal - p.gt)**2 * p.weight).sum() / p.weight.sum()).item() for p in preds 
    # #         if p is not None
    # #         ]
    # # ) )* ns[1]

    # # [np.sqrt(((p.gt - p['map'])**2 * p.weight).sum() / p.weight.sum()).item() for p in preds if p is not None ]


    # # with open('tmp/resultsbis0906', 'rb') as f:
    # #     metrics  = pickle.load(f)

    # # for k in metrics:
    # #     print(metrics[k]['test_out'])

    # pd.DataFrame([{'ckpt': k, **metrics[k]['test_out'][0], **metrics[k]['metrics']} for k in metrics])#.to_csv('tmp/joint_metrics.csv')

    
