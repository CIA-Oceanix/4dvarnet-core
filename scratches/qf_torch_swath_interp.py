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
        ):
        self.xr_ds = xr_ds
        self.nadir_paths = nadir_paths
        self.swot_path = swot_path
        self.nadir_vars = [nadir_var]
        self.swot_gt_vars = list(swot_gt_vars)
        self.swot_obs_vars = list(swot_obs_vars)
        self.obs_downsamp = obs_downsamp

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

    @functools.cache
    def __getitem__(self, item):
        with self.xr_ds.get_coords():
            coords = self.xr_ds[item]
        gv_ = self.xr_ds[item]

        slice_args = dict(
            time_min= pd.to_datetime(np.min(coords['time']).values).date(),
            time_max= pd.to_datetime(np.max(coords['time']).values).date(),
            lat_min=coords['lat'].min().item(),
            lat_max=coords['lat'].max().item(),
            lon_min=coords['lon'].min().item() + 360,
            lon_max=coords['lon'].max().item() + 360,
        )

        if self.swot_path is not None:
            swot = swath_calib.utils.get_swot_slice(self.swot_path, **slice_args)

            # add_cc = lambda ds: ds.assign(ch_nb=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
            # sw = swot.pipe(add_cc)

            sp_ds = swath_calib.interp.stack_passes(*swath_calib.interp.add_nb_ch(swot))
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
        cstack = lambda ts: None if ts[0] is None else swath_calib.interp.stack(ts)
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
        oi, gt, _, go,  *_ = batch
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None and log_pref is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_cal', metrics[-1]["mseCal"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (gt.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (oi.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (go.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

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

    def loss_cal(self, sgt, y, xb):
        _, gc, sv, sc, _, _ = y
        if sv is None:
            return torch.tensor(0.)

        s_vb = swath_calib.interp.batch_torch_interpolate_with_fmt(xb, *gc, *sc)
        cal_out, cal_idx = self.model.model_H.cal_out(sv, s_vb)
        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        t1, t2 = cal_out, fna(sgt[:,:, cal_idx, :], ~(cal_out==0))
        rmse = ((t1 -t2)**2).mean().sqrt()

        def sob(t):
            if len(t.shape) == 4:
                # return kornia.filters.sobel(rearrange(t, 'b d1 d2 c -> b c d1 d2'))
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

        rmse_grad = ((sob(t1) - sob(t2))**2).mean().sqrt()
        rmse_lap = ((lap(t1) - lap(t2))**2).mean().sqrt()

        return rmse, rmse_grad, rmse_lap

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)

        optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                            {'params': self.model.model_H.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                            {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                            ])

        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=25, cooldown=10, factor=0.5),
            'monitor': 'val_loss'
        }

    def compute_loss(self, batch, phase, state_init=(None,)):
        oi, gt, sgt, *y = batch


        # handle patch with no observation
        gt_wo_nan = gt.where(~ gt.isnan(), oi)
        oi_wo_nan = oi.where(~ oi.isnan(), oi)

        state = self.get_init_state(batch, state_init)

        obs = (oi, y)
        if self.current_epoch < self.hparams.warmup_epochs:
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
        loss_cal, g_loss_cal, l_loss_cal = self.loss_cal(sgt, y, xb)

        # print(loss_All, loss_GAll,loss_AE, loss_AE_GT, loss_SR, loss_LR)
        # total loss
        loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
        loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
        loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
        loss += self.hparams.alpha_cal * (loss_cal + g_loss_cal + l_loss_cal)

        # metrics
        # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
        mean_GAll = solver.compute_spatio_temp_weighted_loss(
                torch.hypot(g_gt_x, g_gt_y) , self.grad_crop(self.patch_weight))
        mse = loss_All.detach()
        mseGrad = loss_GAll.detach()
        metrics = dict([
            ('mse', mse),
            ('mseGrad', mseGrad),
            ('meanGrad', mean_GAll),
            ('mseOI', loss_OI.detach()),
            ('mseGOI', loss_GOI.detach()),
            ('mseCal', loss_cal.detach())])

        return loss, outputs, [x_out, hidden_new, cell_new, normgrad], metrics

class CalibrationModelObsGridGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None, min_size=500, sigs=tuple([8* (i+1) for i in range(10)])):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 3
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_data, sst_ch])
        self.min_size = min_size
        self.num_feat = 2*len(sigs) 
        self.norm = torch.nn.BatchNorm2d(num_features=self.num_feat, affine=False, momentum=0.1)

        self.gaussian = StackedGaussian(sigs)
        self.calnet = swath_calib.models.build_net(self.num_feat, 1,)# nhidden = 32, depth = 2,)
        self.downsamp = self.hparams.obscost_downsamp


    def cal_out(self, sv_uncal, sv_bg):
        sh = einops.parse_shape(sv_bg, 'b p ...')
        fs_xlr = einops.rearrange(sv_bg.detach(), 'b p ... -> (b p) () ...')

        fy = einops.rearrange(sv_uncal, 'b p ... -> (b p) () ...')

        msk = fy[:, :, :self.min_size, :].isfinite().all(3).all(2).all(1)
        idx = fs_xlr[ msk, ...].isfinite().all(3).all(1).all(0)

        gy =  self.gaussian(fy[msk][:, :, idx])
        inp_gy = torch.cat((gy.diff(dim=1), gy[:, -1:, ...]), 1)

        gs_xlr = self.gaussian(fs_xlr[msk][:, :, idx])
        inp_gs_xlr = torch.cat((gy.diff(dim=1), gs_xlr[:, -1:, ...]), 1)

        cal_input = torch.cat((inp_gy, inp_gs_xlr), dim=1)
        _out_cal = self.calnet(self.norm(cal_input)) + fs_xlr[msk][:,:,idx]
        out_cal = torch.zeros_like(fy[:, :, idx])
        # print(out_cal.shape, _out_cal.shape, cal_input.shape)
        out_cal.index_add_(0, msk.nonzero(as_tuple=True)[0], _out_cal)
        out_cal = einops.rearrange(out_cal, '(b p) ()  ... -> b p ...', **sh)
        
        return out_cal, idx

    def cal_cost(self, sv_uncal, sv_bg, sv_ref):
        out_cal, idx = self.cal_out(sv_uncal, sv_bg)
        dyout = out_cal - sv_ref[:, :, idx, :]
        return dyout

    def forward(self, x, y, ymsk):
        ylr, (_, gc, sv, sc, nv, nc) = y
        xlr, anom_obs, anom_rec = torch.split(x, self.hparams.dT, dim=1)

        if sv is not None:
            if self.hparams.calref == 'oi':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(ylr, *gc, *sc)
            elif self.hparams.calref == 'x':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr + anom_rec, *gc, *sc)
            elif self.hparams.calref == 'xlr':
                s_xlr = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr, *gc, *sc)

            out_cal, cal_idx = self.cal_out(sv, s_xlr)
            g_cal = swath_calib.interp.batch_interp_to_grid(
                    ylr, *gc, out_cal, *map(lambda t: t[:, :, cal_idx, ...], sc)
            )
            dyout = g_cal - (xlr + anom_obs) 
            dyout = self.hparams.swot_obs_w * dyout

        else:
            dyout = torch.zeros_like(xlr)

        if nv is not None:
            xlr, _, anom_rec = torch.split(x, self.hparams.dT, dim=1)
            g_cal = swath_calib.interp.batch_interp_to_grid(
                ylr, *gc, nv, *nc
            )
            dyout1 = g_cal - (xlr + anom_rec)

        else:
            dyout1 = torch.zeros_like(xlr)

        msk_lr, _ = ymsk
        dyoutlr =  (ylr - xlr) * msk_lr
        return [dyoutlr, dyout, dyout1]
        
class CalibrationModelObsSensorGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None, min_size=500, sigs=tuple([8* (i+1) for i in range(10)])):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 3
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_data, sst_ch])
        self.min_size = min_size
        self.num_feat = 2*len(sigs) 
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
        return  F.conv2d(px, self.kernel)

if __name__ == '__main__':
    fp = 'dgx_ifremer'
    cfgn = 'qxp20_swot_sst'
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
        '+params.alpha_cal=1',
        '+params.swot_obs_w=0.1',
        '+params.warmup_epochs=20',
        '+params.obscost_downsamp=2',
        # '+params.calref=x',
        '+params.calref=xlr',
        # '+params.calref=oi',
        'params.automatic_optimization=false',
    ]
    cfg_4dvar = utils.get_cfg(cfgn, overrides=overrides)
    OmegaConf.resolve(cfg_4dvar)
    splits = OmegaConf.masked_copy(cfg_4dvar.datamodule, ['train_slices', 'val_slices', 'test_slices'])
    grid_kwargs = hydra.utils.call(OmegaConf.masked_copy(cfg_4dvar.datamodule, ['slice_win', 'dim_range', 'strides']))
    dl_kwargs = {
        'batch_size': 2,
        'num_workers': 4
    }
    sensor_kwargs =dict(
        nadir_paths=tuple([f'../sla-data-registry/sensor_zarr/zarr/nadir/{name}' for name in ['swot', 'en', 'tpn', 'g2', 'j1']]),
        # swot_path=None,
        swot_path=f'../sla-data-registry/sensor_zarr/zarr/new_swot',
        nadir_var='ssh_model',
        swot_gt_vars=('ssh_model',),
        swot_obs_vars=('ssh_model', 'wet_tropo_res', 'syst_error_uncalibrated'),
    )
    ds_kwargs = dict(
        OmegaConf.masked_copy(cfg_4dvar.datamodule,['gt_path', 'gt_var', 'oi_path', 'oi_var', 'obs_mask_path', 'obs_mask_var']),
        grid_kwargs=grid_kwargs, sens_kwargs=sensor_kwargs,
    )
    print(ds_kwargs)
    dm = FourDVarMixedGeometryDatamodule(ds_kwargs=ds_kwargs, dl_kwargs=dl_kwargs, **hydra.utils.call(splits),)
    
    # ckpt = 'lightning_logs/version_50/checkpoints/epoch=52-step=6148.ckpt'
    ckpt = None
    lit_mod = utils.get_model(cfgn, ckpt=ckpt, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])
    print(lit_mod.__class__)
    vcb = swath_calib.versioning_cb.VersioningCallback()
    lrcb = pl.callbacks.LearningRateMonitor()
    xp_num = 2 
    logger = pl.loggers.TensorBoardLogger(
        'mixed_geom_logs',
        name=f'{xp_num}_{int(sensor_kwargs["swot_path"] is None)}_{cfg_4dvar.params.calref}',
    )
    trainer = pl.Trainer(gpus=[0], logger=logger, weights_summary='full', callbacks=[vcb, lrcb])
    trainer.fit(lit_mod, datamodule=dm)

    def s2xr(v, c):                                              
        v, c = v.cpu().numpy(), [cc.cpu().numpy() for cc in c]   
        dims = tuple([f'd{di}' for di, _ in enumerate(v.shape)]) 
        ds = xr.Dataset({                                        
          'value': (dims, v),                                  
          't': (dims, c[0]),                                   
          'x': (dims, c[1]),                                   
          'y': (dims, c[2]),                                   
        })                                                       
        return ds                                                
                                                               
    def g2xr(v, c):                                              
        v, c = v.cpu().numpy(), [cc.cpu().numpy() for cc in c]   
        dims = tuple([f'd{di}' for di, _ in enumerate(v.shape)]) 
        ds = xr.Dataset({                                        
          'value': (dims, v),                                  
          't': ((dims[0], dims[-3]), c[0]),                    
          'x': ((dims[0], dims[-2]), c[1]),                    
          'y': ((dims[0], dims[-1]), c[2]),                    
        })                                                       
        return ds                                                

    # oi, gt, sgt, go, gc, sv, sc, nv, nc = next(iter(dm.val_dataloader()))
    # mod_obs = CalibrationModelObsGridGeometry(cfg_4dvar.params.shape_data,
    #         cfg_4dvar.params)

    # out, idx = mod_obs.cal_out(sv, sgt)
    # sv.shape
    # sgt[:, :, idx,:] -out
    # out.shape
    # sv.shape
    # fsv = einops.rearrange(sv, 'b v ... -> (b v) () ...')
    # fsv.shape
   
    # zfsv = torch.zeros_like(fsv)
    # idx = torch.tensor([1, 2, 4])
    # zfsv.index_add_(0, idx, fsv[idx])


    # ggv, fg, wfg = batch_interp_to_grid(gt, *gc, sgt, *sc)
    # g2xr(ggv, gc).value.isel(d0=1, d1=5).plot()
    # g2xr(fg.reshape_as(ggv), gc).value.isel(d0=1, d1=5).plot()
    # g2xr(wfg.reshape_as(ggv), gc).value.isel(d0=1, d1=5).plot()
    # ggv = batch_interp_to_grid(gt, *gc, nv, *nc)
    # g2xr(go, gc).value.isel(d0=0, d1=1).plot()
    # g2xr(go -ggv, gc).value.isel(d0=0, d1=1).plot()

