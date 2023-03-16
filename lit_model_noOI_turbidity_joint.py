import einops
import torch.distributed as dist
import kornia
from hydra.utils import instantiate
import pandas as pd
from functools import reduce
from torch.nn.modules import loss
import xarray as xr
from pathlib import Path
from hydra.utils import call
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import solver as NN_4DVar
import metrics
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps_oi, animate_maps, animate_maps_OI, get_psd_score
from models import Model_H, Phi_r_OI, Phi_r, ModelLR, Phi_r_unet, Gradient_img

from lit_model_augstate import LitModelAugstate

def get_4dvarnet_OI(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
                
def get_4dvarnet_OI_phir(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
                
def get_4dvarnet_OI_phir_unet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_unet(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


def get_constant_crop(patch_size, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_size[d] for d in dim_order], dtype='float32')
        print(patch_size, crop)
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        print(patch_weight.sum())
        return patch_weight

def get_hanning_mask(patch_size, **kwargs):

    t_msk =kornia.filters.get_hanning_kernel1d(patch_size['time'])
    s_msk = kornia.filters.get_hanning_kernel2d((patch_size['lat'], patch_size['lon']))

    patch_weight = t_msk[:, None, None] * s_msk[None, :, :]
    return patch_weight.cpu().numpy()

def get_cropped_hanning_mask(patch_size, crop, **kwargs):
    pw = get_constant_crop(patch_size, crop)

    t_msk =kornia.filters.get_hanning_kernel1d(patch_size['time'])
    
class LitModelOI(pl.LightningModule):

    MODELS = {
            '4dvarnet_OI': get_4dvarnet_OI,
            '4dvarnet_OI_phir': get_4dvarnet_OI_phir,
            '4dvarnet_OI_unet': get_4dvarnet_OI_phir_unet,
             }

    def __init__(self,
                 hparam=None,
                 min_lon=None, max_lon=None,
                 min_lat=None, max_lat=None,
                 ds_size_time=None,
                 ds_size_lon=None,
                 ds_size_lat=None,
                 time=None,
                 dX = None, dY = None,
                 swX = None, swY = None,
                 coord_ext = None,
                 test_domain=None,
                 *args, **kwargs):
        super().__init__()
        hparam = {} if hparam is None else hparam
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)

        self.save_hyperparameters({**hparams, **kwargs})
        # self.save_hyperparameters({**hparams, **kwargs}, logger=False)
        self.latest_metrics = {}
        # TOTEST: set those parameters only if provided

        # create longitudes & latitudes coordinates
        self.test_domain=test_domain
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None

        self.patch_weight = torch.nn.Parameter(
                torch.from_numpy(call(self.hparams.patch_weight)), requires_grad=False)

        self.var_ssh_Val = self.hparams.var_Val[0]
        self.var_ssh_Tr = self.hparams.var_Tr[0]
        self.var_ssh_Tt = self.hparams.var_Tt[0]
        
        self.mean_ssh_Val = self.hparams.mean_Val[0]
        self.mean_ssh_Tr = self.hparams.mean_Tr[0]
        self.mean_ssh_Tt = self.hparams.mean_Tt[0]
        
        self.var_sst_Val = self.hparams.var_Val[1]
        self.var_sst_Tr = self.hparams.var_Tr[1]
        self.var_sst_Tt = self.hparams.var_Tt[1]
        
        self.mean_sst_Val = self.hparams.mean_Val[1]
        self.mean_sst_Tr = self.hparams.mean_Tr[1]
        self.mean_sst_Tt = self.hparams.mean_Tt[1]

        # main model

        self.model_name = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        self.use_sst = self.hparams.sst if hasattr(self.hparams, 'sst') else False
        self.aug_state = self.hparams.aug_state if hasattr(self.hparams, 'aug_state') else False
        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.grad_crop = lambda t: t[...,1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
                self.grad_crop(2.*kornia.filters.spatial_gradient(t, normalized=True)), 2)
        # loss weghing wrt time

        # self._w_loss = torch.nn.Parameter(torch.Tensor(self.patch_weight), requires_grad=False)  # duplicate for automatic upload to gpu
        self.w_loss = torch.nn.Parameter(torch.Tensor([0,0,0,1,0,0,0]), requires_grad=False)  # duplicate for automatic upload to gpu
        
        self.x_ssh_gt = None  # variable to store Ground Truth
        self.obs_ssh_inp = None
        self.x_ssh_oi = None  # variable to store OI
        self.x_ssh_rec = None  # variable to store output of test method
        
        self.x_sst_gt = None  # variable to store Ground Truth
        self.obs_sst_inp = None
        self.x_sst_oi = None  # variable to store OI
        self.x_sst_rec = None  # variable to store output of test method
        
        self.test_figs = {}

        self.tr_loss_hist = []
        self.automatic_optimization = self.hparams.automatic_optimization if hasattr(self.hparams, 'automatic_optimization') else False

        self.median_filter_width = self.hparams.median_filter_width if hasattr(self.hparams, 'median_filter_width') else 1

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out=None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, out, state, _metrics = self.compute_loss(batch, phase=phase, state_init=state_init)
            state_init = [None if s is None else s.detach() for s in state]
            losses.append(_loss)
            metrics.append(_metrics)
        return losses, out, metrics
        
    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)
        if self.model_name in {'4dvarnet_OI','4dvarnet_OI_phir','4dvarnet_OI_unet'}:
            optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ])

        return optimizer

    def on_epoch_start(self):
        self.model.n_grad = self.hparams.n_grad

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                    self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

                self.hparams.n_grad = self.hparams.nb_grad_update[indx]
                self.model.n_grad = self.hparams.n_grad
                print("ngrad iter", self.model.n_grad)
                mm = 0
                lrCurrent = self.hparams.lr_update[indx]
                lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
                for pg in opt.param_groups:
                    pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                    mm += 1

    def training_epoch_end(self, outputs):
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if len(best_ckpt_path) > 0:
            def should_reload_ckpt(losses):
                diffs = losses.diff()
                if losses.max() > (10 * losses.min()):
                    print("Reloading because of check", 1)
                    return True

                if diffs.max() > (100 * diffs.abs().median()):
                    print("Reloading because of check", 2)
                    return True

            if should_reload_ckpt(torch.stack([out['loss'] for out in outputs])):
                print('reloading', best_ckpt_path)
                ckpt = torch.load(best_ckpt_path)
                self.load_state_dict(ckpt['state_dict'])

        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics

        losses, _, metrics = self(train_batch, phase='train')
        if losses[-1] is None:
            print("None loss")
            return None
        # loss = torch.stack(losses).sum()
        loss = 2*torch.stack(losses).sum() - losses[0]

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        # log step metric
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True)
        # self.log("tr_min_nobs", train_batch[1].sum(dim=[1,2,3]).min().item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log("tr_n_nobs", train_batch[1].sum().item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse_ssh", metrics[-1]['mseSSH'] / self.var_ssh_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG_ssh", metrics[-1]['mseGradSSH'] / metrics[-1]['meanGradSSH'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_msePhir_ssh", metrics[-1]['msePhirSSH'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mse_sst", metrics[-1]['mseSST'] / self.var_sst_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG_sst", metrics[-1]['mseGradSST'] / metrics[-1]['meanGradSST'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_msePhir_sst", metrics[-1]['msePhirSST'], on_step=False, on_epoch=True, prog_bar=True)
        return loss
        
    def diag_step(self, batch, batch_idx, log_pref='test'):
        
        oi, inputs_Mask, inputs_obs, targets_GT, sst_gt, mask_sst = batch
        
        sst_obs = sst_gt * mask_sst
        
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse_ssh', metrics[-1]["mseSSH"] / self.var_ssh_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mse_sst', metrics[-1]["mseSST"] / self.var_sst_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG_ssh', metrics[-1]['mseGradSSH'] / metrics[-1]['meanGradSSH'], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG_sst', metrics[-1]['mseGradSST'] / metrics[-1]['meanGradSST'], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_msePhir_ssh', metrics[-1]['msePhirSSH'], on_step=False, on_epoch=True, prog_bar=True)
        return {'ssh_gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'ssh_oi' : (oi.detach().cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'ssh_obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'ssh_pred' : (out.detach().cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'sst_gt'    : (sst_gt.detach().cpu() * np.sqrt(self.var_sst_Tr)) + self.mean_sst_Tr,
                'sst_obs_inp'    : (sst_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_sst_Tr)) + self.mean_sst_Tr,
                'sst_pred' : (out.detach().cpu() * np.sqrt(self.var_sst_Tr)) + self.mean_sst_Tr}

    def test_step(self, test_batch, batch_idx):
        return self.diag_step(test_batch, batch_idx, log_pref='test')

    def test_epoch_end(self, step_outputs):
        return self.diag_epoch_end(step_outputs, log_pref='test')

    def validation_step(self, batch, batch_idx):
        return self.diag_step(batch, batch_idx, log_pref='val')

    def validation_epoch_end(self, outputs):
        print(f'epoch end {self.global_rank} {len(outputs)}')
        if (self.current_epoch + 1) % self.hparams.val_diag_freq == 0:
            return self.diag_epoch_end(outputs, log_pref='val')


    def gather_outputs(self, outputs, log_pref):
        data_path = Path(f'{self.logger.log_dir}/{log_pref}_data')
        data_path.mkdir(exist_ok=True, parents=True)
        torch.save(outputs, data_path / f'{self.global_rank}.t')

        if dist.is_initialized():
            dist.barrier()

        if self.global_rank == 0:
            return [torch.load(f) for f in sorted(data_path.glob('*'))]

    def build_test_xr_ds(self, outputs, diag_ds):

        outputs_keys = list(outputs[0][0].keys())
        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i]
               for i in range(len(diag_ds))
            ]

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield tuple(
                                [outputs[bc][b][k][i] for k in outputs_keys]
                        )

        dses =[
                xr.Dataset( {
                    k: (('time', 'lat', 'lon'), x_k) for k, x_k in zip(outputs_keys, xs)
                }, coords=coords)
            for  xs, coords
            in zip(iter_item(outputs), self.test_patch_coords)
        ]

        fin_ds = xr.merge([xr.zeros_like(ds[['time','lat', 'lon']]) for ds in dses])
        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
        )
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
            )

        for ds in dses:
            ds_nans = ds.assign(weight=xr.ones_like(ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)
            xr_weight = xr.DataArray(self.patch_weight.detach().cpu(), ds.coords, dims=ds.gt.dims)
            _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
            fin_ds = fin_ds + _ds


        return (
            (fin_ds.drop('weight') / fin_ds.weight)
            .sel(instantiate(self.test_domain))
            .isel(time=slice(self.hparams.dT //2, -self.hparams.dT //2))
            # .pipe(lambda ds: ds.sel(time=~(np.isnan(ds.gt).all('lat').all('lon'))))
        ).transpose('time', 'lat', 'lon')


    def nrmse_fn(self, pred, ref, gt):
        return (
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - ds.mean())
                .pipe(lambda ds: ds - (self.test_xr_ds[gt].pipe(lambda da: da - da.mean())))
                .pipe(lambda ds: ds ** 2 / self.test_xr_ds[gt].std())
                .to_dataframe()
                .pipe(lambda ds: np.sqrt(ds.mean()))
                .to_frame()
                .rename(columns={0: 'nrmse'})
                .assign(nrmse_ratio=lambda df: df / df.loc[ref])
        )

    def mse_fn(self, pred, ref, gt):
            return(
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - self.test_xr_ds[gt])
                .pipe(lambda ds: ds ** 2)
                .to_dataframe()
                .pipe(lambda ds: ds.mean())
                .to_frame()
                .rename(columns={0: 'mse'})
                .assign(mse_ratio=lambda df: df / df.loc[ref])
        )


    def sla_diag(self, t_idx=3, log_pref='test'):
        path_save0 = self.logger.log_dir + '/maps_ssh.png'
        t_idx = 3
        fig_maps_ssh = plot_maps_oi(
                  self.x_ssh_gt[t_idx],
                self.obs_ssh_inp[t_idx],
                  self.x_ssh_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + '/maps_Grad_ssh.png'
        fig_maps_grad_ssh = plot_maps_oi(
                  self.x_ssh_gt[t_idx],
                self.obs_ssh_inp[t_idx],
                  self.x_ssh_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        path_save02 = self.logger.log_dir + '/maps_sst.png'
        t_idx = 3
        fig_maps_sst = plot_maps_oi(
                  self.x_sst_gt[t_idx],
                self.obs_sst_inp[t_idx],
                  self.x_sst_rec[t_idx],
                  self.test_lon, self.test_lat, path_save02)
        path_save03 = self.logger.log_dir + '/maps_Grad_sst.png'
        fig_maps_grad_sst = plot_maps_oi(
                  self.x_sst_gt[t_idx],
                self.obs_sst_inp[t_idx],
                  self.x_sst_rec[t_idx],
                  self.test_lon, self.test_lat, path_save03, grad=True)
        self.test_figs['maps_ssh'] = fig_maps_ssh
        self.test_figs['maps_grad_ssh'] = fig_maps_grad_ssh
        self.test_figs['maps_sst'] = fig_maps_sst
        self.test_figs['maps_grad_sst'] = fig_maps_grad_sst
        self.logger.experiment.add_figure(f'{log_pref} Maps_ssh', fig_maps_ssh, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad_ssh', fig_maps_grad_ssh, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps_sst', fig_maps_sst, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad_sst', fig_maps_grad_sst, global_step=self.current_epoch)
        ###############
        # animate maps
        ###############
        #print(self.hparams)
        if self.hparams.animate:
            path_save0 = self.logger.log_dir + '/animation_ssh.mp4'
            animate_maps_OI(self.x_ssh_gt, self.obs_ssh_inp, self.x_ssh_rec, self.test_lon, self.test_lat, path_save0)

            path_save0 = self.logger.log_dir + '/animation_grad_ssh.mp4'
            animate_maps_OI(self.x_ssh_gt, self.obs_ssh_inp, self.x_ssh_rec, self.test_lon, self.test_lat, path_save0, grad=True)
            
            path_save0 = self.logger.log_dir + '/animation_sst.mp4'
            animate_maps_OI(self.x_sst_gt, self.obs_sst_inp, self.x_sst_rec, self.test_lon, self.test_lat, path_save0)

            path_save0 = self.logger.log_dir + '/animation_grad_sst.mp4'
            animate_maps_OI(self.x_sst_gt, self.obs_sst_inp, self.x_sst_rec, self.test_lon, self.test_lat, path_save0, grad=True)

        # ~ psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        # ~ psd_fig = metrics.plot_psd_score(psd_ds)
        # ~ self.test_figs['psd'] = psd_fig
        # ~ self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        _, _, mu_ssh, sig_ssh = metrics.rmse_based_scores(self.test_xr_ds.pred_ssh, self.test_xr_ds.ssh_gt)
        _, _, mu_sst, sig_sst = metrics.rmse_based_scores(self.test_xr_ds.pred_sst, self.test_xr_ds.sst_gt)

        md = {
            # ~ f'{log_pref}_lambda_x': lamb_x,
            # ~ f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu_ssh': mu_ssh,
            f'{log_pref}_sigma_ssh': sig_ssh,
            f'{log_pref}_mu_sst': mu_sst,
            f'{log_pref}_sigma_sst': sig_sst,
        }
        print(pd.DataFrame([md]).T.to_markdown())
        return md

    def diag_epoch_end(self, outputs, log_pref='test'):
        full_outputs = self.gather_outputs(outputs, log_pref=log_pref)
        if full_outputs is None:
            print("full_outputs is None on ", self.global_rank)
            return
        if log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise Exception('unknown phase')
        self.test_xr_ds = self.build_test_xr_ds(full_outputs, diag_ds=diag_ds)

        Path(self.logger.log_dir).mkdir(exist_ok=True)
        path_save1 = self.logger.log_dir + f'/test.nc'
        print(path_save1)
        self.test_xr_ds.to_netcdf(path_save1)
        
        self.x_ssh_gt = self.test_xr_ds.ssh_gt.data
        self.obs_ssh_inp = self.test_xr_ds.ssh_obs_inp.data
        self.x_ssh_rec = self.test_xr_ds.ssh_pred.data
        self.x_rec_ssh = self.x_ssh_rec
        
        self.x_sst_gt = self.test_xr_ds.sst_gt.data
        self.obs_sst_inp = self.test_xr_ds.sst_obs_inp.data
        self.x_sst_rec = self.test_xr_ds.sst_pred.data
        self.x_rec_sst = self.x_sst_rec
        
        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data

        # display map
        md = self.sla_diag(t_idx=3, log_pref=log_pref)
        self.latest_metrics.update(md)
        self.logger.log_metrics(md, step=self.current_epoch)

    def teardown(self, stage='test'):

        self.logger.log_hyperparams(
                {**self.hparams},
                self.latest_metrics
    )


    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        _, inputs_Mask, inputs_obs, targets_GT, sst_gt, sst_mask = batch
        init_state = torch.cat((inputs_Mask * inputs_obs,sst_mask*sst_gt,), dim=1)
        return init_state
    def loss_ae(self, state_out):
        return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def sla_loss(self, gt, out):
        g_outputs_x, g_outputs_y = self.gradient_img(out)
        g_gt_x, g_gt_y = self.gradient_img(gt)

        loss = NN_4DVar.compute_spatio_temp_weighted_loss((out - gt), self.patch_weight)
        loss_grad = (
                NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_x - g_gt_x, self.grad_crop(self.patch_weight))
            +    NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_y - g_gt_y, self.grad_crop(self.patch_weight))
        )

        return loss, loss_grad

    def reg_loss(self, y_gt, oi, out, out_lr, out_lrhr):
        l_ae = self.loss_ae(out_lrhr)
        l_ae_gt = self.loss_ae(y_gt)
        l_sr = NN_4DVar.compute_spatio_temp_weighted_loss(out_lr - oi, self.patch_weight)

        gt_lr = self.model_LR(oi)
        out_lr_bis = self.model_LR(out)
        l_lr = NN_4DVar.compute_spatio_temp_weighted_loss(out_lr_bis - gt_lr, self.model_LR(self.patch_weight))

        return l_ae, l_ae_gt, l_sr, l_lr

    def compute_loss(self, batch, phase, state_init=(None,)):
        _, inputs_Mask, inputs_obs, targets_GT, sst_gt, sst_mask = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_GT),
                    torch.cat((torch.zeros_like(targets_GT),
                              torch.zeros_like(targets_GT),
                              torch.zeros_like(targets_GT)), dim=1),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                        ('msePhir', 0.),
                        ])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        sst_gt_wo_nan = sst_gt.where(~sst_gt.isnan(), torch.zeros_like(targets_GT))
        
        state = self.get_init_state(batch, state_init)

        obs = inputs_Mask * inputs_obs
        new_masks =  inputs_Mask
        
        obs = torch.cat((obs,sst_mask*sst_gt), dim=1)
        new_masks = torch.cat((new_masks,sst_mask), dim=1)
        
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)
        g_sst_gt_x, g_sst_gt_y = self.gradient_img(sst_gt)
        
        

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            print('la c ok encore ! !')
            if self.hparams.n_grad>0:
                outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])
                print('rien ne va plus....')
                if (phase == 'val') or (phase == 'test'):
                    outputs = outputs.detach()
            else:
                outputs = self.model.phi_r(obs)
                hidden_new = None
                cell_new = None
                normgrad = None
            
            # ~ loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
            output_ssh = outputs[:,0:self.hparams.dt,:,:]
            output_sst = outputs[:,self.hparams.dt:2*self.hparams.dt,:,:]
            
            outputs_GT_ssh_wo_nan = outputs_ssh.where(~targets_GT.isnan(), torch.zeros_like(outputs_ssh))
            outputs_GT_sst_wo_nan = outputs_sst.where(~sst_gt.isnan(), torch.zeros_like(outputs_sst))
            
            loss_All_ssh, loss_GAll_ssh = self.sla_loss(outputs_GT_ssh_wo_nan, targets_GT_wo_nan)            
            loss_AE_ssh = self.loss_ae(outputs_ssh)
            
            loss_All_sst, loss_GAll_sst = self.sla_loss(outputs_GT_sst_wo_nan, sst_gt_wo_nan)            
            loss_AE_sst = self.loss_ae(outputs_sst)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All_ssh + self.hparams.alpha_mse_g_ssh * loss_GAll_ssh
            loss += 0.5 * self.hparams.alpha_proj * loss_AE_ssh
            loss += self.hparams.alpha_mse_sst * loss_All_sst + self.hparams.alpha_mse_g_sst * loss_GAll_sst
            loss += 0.5 * self.hparams.alpha_proj * loss_AE_sst

            # metrics
            # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mean_GAll_ssh = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_targets_GT_x, g_targets_GT_y) , self.grad_crop(self.patch_weight))
            mean_GAll_ssh = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_sst_gt_x, g_sst_gt_y) , self.grad_crop(self.patch_weight))
            mse_ssh = loss_All_ssh.detach()
            mse_sst = loss_All_sst.detach()
            mseGrad_ssh = loss_GAll_ssh.detach()
            mseGrad_sst = loss_GAll_sst.detach()
            msePhir_ssh = loss_AE_ssh.detach()
            msePhir_sst = loss_AE_sst.detach()
            metrics = dict([
                ('mse_ssh', mse_ssh),
                ('mseGrad_ssh', mseGrad_ssh),
                ('meanGrad_ssh', mean_GAll_ssh),
                ('msePhir_ssh', msePhir_ssh),
                ('mse_sst', mse_sst),
                ('mseGrad_sst', mseGrad_sst),
                ('meanGrad_sst', mean_GAll_sst),
                ('msePhir_sst', msePhir_sst),
                ])
        return loss, outputs, [outputs, hidden_new, cell_new, normgrad], metrics

