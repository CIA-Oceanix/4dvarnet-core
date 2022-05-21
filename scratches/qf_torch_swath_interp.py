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
importlib.reload(solver)
fp = 'dgx_ifremer'
cfgn = 'qxp20_swot_sst'
cfgn = 'xp_aug/xp_repro/full_core_hanning'
OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
cfg = utils.get_cfg(f'{cfgn}')
overrides = [
    # '+datamodule.dl_kwargs.shuffle=False',
    f'file_paths={fp}',
    'params.files_cfg.obs_mask_path=${file_paths.new_noisy_swot}',
    # 'params.files_cfg.obs_mask_var=five_nadirs',
]


# Generate grided product on swath
cfg_4dvar = utils.get_cfg(cfgn, overrides=overrides)
OmegaConf.resolve(cfg_4dvar)

rms = lambda da: np.sqrt(np.mean(da**2))

class ModelObsMixedGeometry(torch.nn.Module):

    def __init__(self, shape_data, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 3
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_data, sst_ch])


    def forward(self, x, y, msk):
        xlr, anom, *_ = torch.split(x, self.hparams.dT, dim=1)

        ylr, (_, gc, sv, sc, nv, nc) = y
        msk_lr, _ = msk
        dyoutlr =  (ylr - xlr) * msk_lr
        s_x = swath_calib.interp.batch_torch_interpolate_with_fmt(xlr + anom, *gc, *sc)
        s_x_msk = s_x.isfinite()

        fna = lambda t, m: t.where(m, torch.zeros_like(t))
        dyout =  fna(s_x, s_x_msk) - fna(sv, s_x_msk)
        n_x = swath_calib.interp.batch_torch_interpolate_with_fmt(x, *gc, *nc)
        n_x_msk = n_x.isfinite()
        dyout1 =  fna(n_x, n_x_msk) - fna(nv, n_x_msk)
        return [dyoutlr, dyout, dyout1]

class SensorXrDs(torch.utils.data.Dataset):
    def __init__(self,
            xr_ds,
            nadir_paths=tuple([f'../sla-data-registry/sensor_zarr/zarr/nadir/{name}' for name in ['swot', 'en', 'tpn', 'g2', 'j1']]),
            swot_path=f'../sla-data-registry/sensor_zarr/zarr/new_swot',
            nadir_var='ssh_model',
            swot_vars=('ssh_model',),
        ):
        self.xr_ds = xr_ds
        self.nadir_paths = nadir_paths
        self.swot_path = swot_path
        self.nadir_vars = [nadir_var]
        self.swot_vars = list(swot_vars)

    def __len__(self):
        return len(self.xr_ds)

    @staticmethod
    def collate_fn(list_of_items):

        gv, gc, sv, sc, nv, nc = zip(*list_of_items)
        return (
            torch.stack(gv),
            list(map(torch.stack, zip(*gc))),
            swath_calib.interp.stack(sv),
            list(map(swath_calib.interp.stack, zip(*sc))),
            swath_calib.interp.stack(nv),
            list(map(swath_calib.interp.stack, zip(*nc))),
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

        nadirs = [swath_calib.utils.get_nadir_slice(p, **slice_args) for p in self.nadir_paths]
        nadirs = [nad for nad in nadirs if nad is not None]
        swot = swath_calib.utils.get_swot_slice(self.swot_path, **slice_args)

        add_cc = lambda ds: ds.assign(ch_nb=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
        sw = swot.pipe(add_cc)



        nads_with_nb_ch, tgt_len = zip(*[swath_calib.interp.add_nb_ch(nad) for nad in nadirs])

        np_ds = xr.concat([swath_calib.interp.stack_passes(nad, max(tgt_len)) for nad in nads_with_nb_ch], 'p')

        sp_ds = swath_calib.interp.stack_passes(*swath_calib.interp.add_nb_ch(swot))

        spc, npc = swath_calib.interp.fmt_s_coords(sp_ds), swath_calib.interp.fmt_s_coords(np_ds)
        spv = swath_calib.interp.fmt_s_value(sp_ds, self.swot_vars, spc[0].isfinite())
        npv = swath_calib.interp.fmt_s_value(np_ds, self.nadir_vars, npc[0].isfinite())

        return (
                torch.from_numpy(gv_), swath_calib.interp.fmt_g_coords(xr.Dataset(coords)),
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
        oi , gt, go, gc, sv, sc, nv, nc = zip(*list_of_items)
        return (
            torch.stack(oi),
            torch.stack(gt),
            torch.stack(go),
            list(map(torch.stack, zip(*gc))),
            swath_calib.interp.stack(sv),
            list(map(swath_calib.interp.stack, zip(*sc))),
            swath_calib.interp.stack(nv),
            list(map(swath_calib.interp.stack, zip(*nc))),
        )

    def __getitem__(self, item):
        if self.return_coords:
            with self.gt_ds.get_coords():
                return self.gt_ds[item]
        oi, gt, go, gc, sv, sc, nv, nc = (self.oi_ds[item], self.gt_ds[item], *self.sens_obs_ds[item])
        pp = lambda t: (t - self.norm_stats[0]) / self.norm_stats[1]
        return pp(torch.from_numpy(oi)), pp(torch.from_numpy(gt)), pp(go), gc, pp(sv), sc, pp(nv), nc



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
    print(hparams)
    print('should_be_here')
    return solver.Solver_Grad_4DVarNN(
                models.Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                ModelObsMixedGeometry(hparams.shape_state[0], hparams=hparams),
                solver.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

class LitModMixGeom(lit_model_augstate.LitModelAugstate):
    def diag_step(self, batch, batch_idx, log_pref='test'):
        oi, gt, go,  *_ = batch
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None and log_pref is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (gt.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (oi.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (go.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def create_model(self, *args, **kwargs):
        return get_4dvarnet_mixgeom(self.hparams)

    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        oi, _, go, *_ = batch
        go_oi = go.where(go.isfinite(), oi)

        init_state = torch.cat((oi,go_oi), dim=1)
        if self.aug_state:
            init_state = torch.cat((init_state, go_oi), dim=1)
        return init_state

    def compute_loss(self, batch, phase, state_init=(None,)):
        oi, gt, *y = batch

        # handle patch with no observation
        gt_wo_nan = gt.where(~ gt.isnan(), oi)
        oi_wo_nan = oi.where(~ oi.isnan(), oi)

        state = self.get_init_state(batch, state_init)

        obs = (oi, y)
        msks = (torch.ones_like(oi), None)

        # gradient norm field
        g_gt_x, g_gt_y = self.gradient_img(gt)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(state, obs, msks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR = outputs[:, 0:self.hparams.dT, :, :]
            if self.aug_state:
                outputs = outputsSLR + outputs[:, 2*self.hparams.dT:, :, :]
            else:
                outputs = outputsSLR + outputs[:, self.hparams.dT:2*self.hparams.dT, :, :]

            # median filter
            if self.median_filter_width > 1:
                outputs = kornia.filters.median_blur(outputs, (self.median_filter_width, self.median_filter_width))


            yGT = torch.cat((oi,
                             gt_wo_nan - outputsSLR),
                            dim=1)
            if self.aug_state:
                yGT = torch.cat((yGT, gt_wo_nan - outputsSLR), dim=1)

            loss_All, loss_GAll = self.sla_loss(outputs, gt_wo_nan)
            loss_OI, loss_GOI = self.sla_loss(oi, gt_wo_nan)
            loss_AE, loss_AE_GT, loss_SR, loss_LR =  self.reg_loss(
                yGT, oi, outputs, outputsSLR, outputsSLRHR
            )

            # print(loss_All, loss_GAll,loss_AE, loss_AE_GT, loss_SR, loss_LR)
            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR

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
                ('mseGOI', loss_GOI.detach())])

        return loss, outputs, [outputsSLRHR, hidden_new, cell_new, normgrad], metrics
        
splits = OmegaConf.masked_copy(cfg_4dvar.datamodule, ['train_slices', 'val_slices', 'test_slices'])
grid_kwargs = hydra.utils.call(OmegaConf.masked_copy(cfg_4dvar.datamodule, ['slice_win', 'dim_range', 'strides']))
dl_kwargs = {'batch_size': 2}
sensor_kwargs =dict(
    nadir_paths=tuple([f'../sla-data-registry/sensor_zarr/zarr/nadir/{name}' for name in ['swot', 'en', 'tpn', 'g2', 'j1']]),
    swot_path=f'../sla-data-registry/sensor_zarr/zarr/new_swot',
    nadir_var='ssh_model',
    swot_vars=('ssh_model',),
)
ds_kwargs = dict(
    OmegaConf.masked_copy(cfg_4dvar.datamodule,['gt_path', 'gt_var', 'oi_path', 'oi_var', 'obs_mask_path', 'obs_mask_var']),
    grid_kwargs=grid_kwargs, sens_kwargs=sensor_kwargs,
)

dm = FourDVarMixedGeometryDatamodule(ds_kwargs=ds_kwargs, dl_kwargs=dl_kwargs, **hydra.utils.call(splits),)

lit_mod = utils.get_model(cfgn, ckpt=None, dm=dm, add_overrides=overrides+['lit_mod_cls=__main__.LitModMixGeom'])
trainer = pl.Trainer(gpus=1)
trainer.fit(lit_mod, datamodule=dm)
dl = dm.train_dataloader()

for i, b in enumerate(dm.train_dataloader()):
    oi, gv, gc, sv, sc, nv, nc = b
    print('training', i)
    print('training', oi.shape, sv.shape, nv.shape)
for i, b in enumerate(dm.val_dataloader()):
    print('val', i)
for i, b in enumerate(dm.test_dataloader()):
    print('test', i)

batch = next(iter(dl))
# dataloading.XrDataset(ds_kwargs.oi_path, ds_kwargs.oi_var, decode=False, **hydra.utils.call(grid_kwargs))
# FourDVarMixedGeometryDataset(slice('2013-01-03', "2013-01-27"), **ds_kwargs)

if __name__ == '__main__':
    dm = utils.get_dm(cfgn, add_overrides=overrides)

    s = torch.squeeze
    ms = lambda t: list(map(s, t))
    bsqu = lambda gv, gc, sv, sc, nv, nc: (s(gv), ms(gc), s(sv), ms(sc), s(nv), ms(nc))
    dl = dm.test_dataloader()
    xr_ds = dl.dataset.datasets[0].gt_ds
    obs = dl.dataset.datasets[0].obs_mask_ds


    nadirs_paths = [f'../sla-data-registry/sensor_zarr/zarr/nadir/{name}' for name in ['swot', 'en', 'tpn', 'g2', 'j1']]
    swot_path = f'../sla-data-registry/sensor_zarr/zarr/new_swot'
    sds = swath_calib.interp.SensorXrDs(xr_ds=xr_ds, nadir_paths=nadirs_paths, swot_path=swot_path)
    dl = torch.utils.data.DataLoader(sds, batch_size=2, collate_fn=swath_calib.interp.collate_fn, shuffle=False)
    batch = next(iter(dl))

    gv, gc, sv, sc, nv, nc = batch
    nv1 = swath_calib.interp.batch_torch_interpolate_with_fmt(gv, *gc, *nc)
    sv1 = swath_calib.interp.batch_torch_interpolate_with_fmt(gv, *gc, *sc)

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

    gds = g2xr(gv, gc)
    gds.value.isel(d0=0).plot.pcolormesh('d2', 'd3', col='d1', col_wrap=3, figsize=(10, 10))
    gobs_it = obs[0]
    gobs_it =torch.from_numpy(obs[0])[None, ...]
    gobsds = g2xr(gobs_it, [c[0:1,...] for c in gc])
    gobsds.value.isel(d0=0).plot.pcolormesh('d2', 'd3', col='d1', col_wrap=3, figsize=(10, 10))


    # swot vis
    ds = s2xr(sv, sc)
    ds1 = s2xr(sv1, sc)
    ds.value.isel(d0=0).plot.pcolormesh('d2', 'd3', col='d1', col_wrap=1, figsize=(10, 10))
    ds1 .value.isel(d0=0).plot.pcolormesh('d2', 'd3', col='d1', col_wrap=1, figsize=(10, 10))

    err_ds = s2xr(sv - sv1, sc)
    err_ds.value.isel(d0=1).plot.pcolormesh('d2', 'd3', col='d1', col_wrap=1, figsize=(10, 10))
    
    # nadir vis
    nds = s2xr(nv, nc)
    nds1 = s2xr(nv1, nc)
    nds.value.isel(d0=0).isel(d1=slice(None, None, 5)).plot.line(x='d2', col='d1', col_wrap=1, figsize=(10, 10))
    nds1.value.isel(d0=0).isel(d1=slice(None, None, 5)).plot.line(x='d2', col='d1', col_wrap=1, figsize=(10, 10))

    
    nerr_ds = s2xr(nv - nv1, nc)
    nerr_ds.value.isel(d0=0).isel(d1=slice(None, None, 5)).plot.line(x='d2', col='d1', col_wrap=1, figsize=(10, 10))
