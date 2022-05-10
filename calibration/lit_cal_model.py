import einops
import shutil
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
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, get_psd_score
from models import Phi_r, ModelLR, Gradient_img

from calibration.models import get_passthrough, get_vit


class Model_H_with_noisy_Swot(torch.nn.Module):
    """
    state: [oi, anom_glob, anom_swath ] or [oi, anom_glob_obs,  anom_swath, anom_glob_rec]

    obs: [oi, obs]
    mask: [ones, obs_mask]
    """
    def __init__(self, shape_data, shape_obs, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_obs])



    def forward(self, x, y, mask):
        aug_state = self.hparams.get('aug_state', False)
        if aug_state:
            if aug_state == 2:
                output_low_res,  output_anom_glob, output_anom_swath, _, _ = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
            else:
                output_low_res,  output_anom_glob, output_anom_swath, _ = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
        else: 
            output_low_res,  output_anom_glob, output_anom_swath = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
        output_global = output_low_res + output_anom_glob

        if self.hparams.swot_anom_wrt == 'low_res':
            output_swath = output_low_res + output_anom_swath
        elif self.hparams.swot_anom_wrt == 'high_res':
            output_swath = output_global + output_anom_swath


        yhat_glob = torch.cat((output_low_res, output_global), dim=1)
        yhat_swath = torch.cat((output_low_res, output_swath), dim=1)
        dyout_glob = (yhat_glob - y) * mask
        dyout_swath = (yhat_swath - y) * mask

        return dyout_glob + dyout_swath

class Model_H_SST_with_noisy_Swot(torch.nn.Module):
    """
    state: [oi, anom_glob, anom_swath ]
    obs: [[oi, obs], sst]
    mask: [[ones, obs_mask], ones]
    """
    

    def __init__(self, shape_data, shape_obs, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 2
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_data, sst_ch])

        self.conv11 = torch.nn.Conv2d(shape_data, hparams.dT, (3, 3), padding=1, bias=False)
        self.conv21 = torch.nn.Conv2d(sst_ch, hparams.dT, (3, 3), padding=1, bias=False)
        self.conv_m = torch.nn.Conv2d(sst_ch, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def forward(self, x, y, mask):
        y_ssh, y_sst = y
        mask_ssh, mask_sst = mask
        aug_state = self.hparams.get('aug_state', False)
        if aug_state:
            if aug_state == 2:
                output_low_res,  output_anom_glob, output_anom_swath, _, _ = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
            else:
                output_low_res,  output_anom_glob, output_anom_swath, _ = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
        else: 
            output_low_res,  output_anom_glob, output_anom_swath = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
        output_global = output_low_res + output_anom_glob

        if self.hparams.swot_anom_wrt == 'low_res':
            output_swath = output_low_res + output_anom_swath
        elif self.hparams.swot_anom_wrt == 'high_res':
            output_swath = output_global + output_anom_swath
        

        yhat_glob = torch.cat((output_low_res, output_global), dim=1)
        yhat_swath = torch.cat((output_low_res, output_swath), dim=1)
        dyout_glob = (yhat_glob - y_ssh) * mask_ssh
        dyout_swath = (yhat_swath - y_ssh) * mask_ssh

        dyout = dyout_glob + dyout_swath
        dyout1 = self.conv11(x) - self.conv21(y_sst)
        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask_sst))

        return [dyout, dyout1]


def get_4dvarnet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H_with_noisy_Swot(hparams.shape_data[0], hparams.shape_obs[0], hparams=hparams),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_data, hparams.n_grad)


def get_4dvarnet_sst(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H_SST_with_noisy_Swot(hparams.shape_data[0], hparams.shape_obs[0], hparams=hparams),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_data, hparams.n_grad)

def get_phi(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = Phi_r(hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic)

            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks):
            return self.phi(state), None, None, None

    return PhiPassThrough()


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

############################################ Lightning Module #######################################################################

class LitCalModel(pl.LightningModule):


    MODELS = {
            'passthrough': get_passthrough,
            'vit': get_vit,
            '4dvarnet': get_4dvarnet,
            '4dvarnet_sst': get_4dvarnet_sst,
            'phi': get_phi,
        }

    def __init__(self, hparam=None,
                               min_lon=None, max_lon=None,
                               min_lat=None, max_lat=None,
                               ds_size_time=None,
                               ds_size_lon=None,
                               ds_size_lat=None,
                               time=None,
                               dX = None, dY = None,
                               swX = None, swY = None,
                               coord_ext = None,
                               *args, **kwargs):
        super().__init__()
        hparam = {} if hparam is None else hparam
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)

        # self.save_hyperparameters({**hparams, **kwargs})
        self.save_hyperparameters({**hparams, **kwargs}, logger=False)
        self.latest_metrics = {}
        # TOTEST: set those parameters only if provided
        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt

        # create longitudes & latitudes coordinates
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None

        self.patch_weight = call(self.hparams.patch_weight)

        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt
        self.mean_Val = self.hparams.mean_Val
        self.mean_Tr = self.hparams.mean_Tr
        self.mean_Tt = self.hparams.mean_Tt

        # main model

        self.model_name = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        self.use_sst = self.hparams.sst if hasattr(self.hparams, 'sst') else False
        self.aug_state = self.hparams.aug_state if hasattr(self.hparams, 'aug_state') else False
        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.gradient_img = kornia.filters.sobel
        # loss weghing wrt time

        self.w_loss = torch.nn.Parameter(torch.Tensor(self.patch_weight), requires_grad=False)  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.test_figs = {}
        
        self.tr_loss_hist = []
        self.automatic_optimization = False

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, outs, _metrics = self.compute_loss(batch, phase=phase, state_init=state_init)
            state_init = outs.detach()
            losses.append(_loss)
            metrics.append(_metrics)
        return losses, outs, metrics

    def configure_optimizers(self):
        
        if self.model_name == '4dvarnet':
            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ]
                , lr=0., weight_decay=self.hparams.weight_decay)

            return optimizer
        elif self.model_name == '4dvarnet_sst':

            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0., weight_decay=self.hparams.weight_decay)

            return optimizer
        else: 
            opt = optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': opt,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=50,),
            'monitor': 'val_loss'
        }

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
                if (losses.argmax() > losses.argmin()):
                    if (losses.max() > (10 * losses.min())):
                        print("Reloading because of check", 1)
                        return True

                    if diffs.max() > (100 * diffs.abs().median()):
                        print("Reloading because of check", 2)
                        return True

            if should_reload_ckpt(torch.stack([out['loss'] for out in outputs])):
                print('reloading', best_ckpt_path)
                ckpt = torch.load(best_ckpt_path, map_location='cpu')
                self.load_state_dict(ckpt['state_dict'])
                

            
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        opt = self.optimizers()
        # compute loss and metrics    
        if self.hparams.get('rand_mask'):
            rand = torch.rand_like(train_batch[0])
            mask_95 = rand > 0.95
            mask_gt = lambda t, m: t.where(m, torch.zeros_like(t))
            targets_OI, inputs_Mask, inputs_obs, targets_GT, *rest = train_batch
            # loss_95, outs_95, metrics_95 = self.compute_loss(
            #     (targets_OI, mask_95, mask_gt(targets_GT, mask_95), targets_GT, *rest), phase='train'
            # )
            loss_95_obs, outs_95_obs, metrics_95_obs = self.compute_loss(
                (targets_OI, inputs_Mask.logical_or(mask_95), inputs_obs.where(inputs_Mask, mask_gt(targets_GT, mask_95)), targets_GT, *rest),
                phase='train'
            )
            opt.zero_grad()
            self.manual_backward((loss_95_obs) / 2)
            opt.step()

        losses, outs, metrics = self(train_batch, phase='train')
        if losses[-1] is None:
            print("None loss")
            return None
        loss = torch.stack(losses).mean()

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        # log step metric        
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True)
        # self.log("tr_min_nobs", train_batch[1].sum(dim=[1,2,3]).min().item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log("tr_n_nobs", train_batch[1].sum().item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.get('rand_mask'):
            self.log("tr_mse_95", metrics_95_obs['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mse_swath", metrics[-1]['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG_swath", metrics[-1]['mseGradSwath'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)


        return loss

    def predict_step(self, batch, batch_idx):
        return self.diag_step(batch, batch_idx, log_pref=None)

    def diag_step(self, batch, batch_idx, log_pref='test'):
        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, obs_target_item = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, obs_target_item = batch
        losses, outs, metrics = self(batch, phase='test')
        _, out, out_pred = self.get_outputs(batch, outs)
        loss = losses[-1]
        if loss is not None and log_pref is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mse_swath', metrics[-1]['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG_swath', metrics[-1]['mseGradSwath'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'target_obs'    : (obs_target_item.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'inp_obs'    : (inputs_obs.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_pred'    : (out_pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

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

    def diag_epoch_end(self, outputs, log_pref='test'):
        data_path = Path(self.logger.log_dir if self.logger is not None else 'tmp') / f'{log_pref}_data'
        data_path.mkdir(exist_ok=True, parents=True)
        print(len(outputs))
        torch.save(outputs, data_path / f'{self.global_rank}.t')
        if dist.is_initialized():
            dist.barrier() 
        if self.global_rank > 0:
            print(f'Saved data for rank {self.global_rank}')
            return 

        full_outputs = [torch.load(f) for f in sorted(data_path.glob('*'))]
        shutil.rmtree(data_path)

        print(len(full_outputs))
        if log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise Exception('unknown phase')
        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i]
               for i in range(len(diag_ds))
            ]
        self.outputs = full_outputs

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield (
                                outputs[bc][b]['gt'][i],
                                outputs[bc][b]['oi'][i],
                                outputs[bc][b]['preds'][i],
                                outputs[bc][b]['target_obs'][i],
                                outputs[bc][b]['obs_pred'][i],
                                outputs[bc][b]['inp_obs'][i],
                        )
            
        dses =[ 
                xr.Dataset( {
                    'gt': (('time', 'lat', 'lon'), x_gt),
                    'oi': (('time', 'lat', 'lon'), x_oi),
                    'pred': (('time', 'lat', 'lon'), x_rec),
                    'obs_gt': (('time', 'lat', 'lon'), obs_gt),
                    'obs_pred': (('time', 'lat', 'lon'), obs_pred),
                    'obs_inp': (('time', 'lat', 'lon'), obs_inp),
                }, coords=coords)
            for  (x_gt, x_oi, x_rec, obs_gt, obs_pred, obs_inp), coords
            in zip(iter_item(self.outputs), self.test_patch_coords)
        ]
        import time
        t0 = time.time()
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
            xr_weight = xr.DataArray(self.patch_weight, ds.coords, dims=ds.gt.dims) 
            _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
            fin_ds = fin_ds + _ds 
        

        self.test_xr_ds = (
            (fin_ds.drop('weight') / fin_ds.weight)
            .sel(instantiate(self.hparams.test_domain))
            .pipe(lambda ds: ds.sel(time=~(np.isnan(ds.gt).all('lat').all('lon'))))
        ).transpose('time', 'lat', 'lon')

        self.x_gt = self.test_xr_ds.gt.data
        self.obs_inp = self.test_xr_ds.obs_inp.data
        self.obs_gt = self.test_xr_ds.obs_gt.data
        self.obs_pred = self.test_xr_ds.obs_pred.data
        self.x_oi = self.test_xr_ds.oi.data
        self.x_rec = self.test_xr_ds.pred.data
        
        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data
        
        if self.logger is None:
            return
        Path(self.logger.log_dir).mkdir(exist_ok=True)
        # display map
        path_save0 = self.logger.log_dir + f'/{log_pref} maps.png'
        t_idx = 3
        fig_maps = plot_maps(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + f'/{log_pref} maps_Grad.png'
        fig_maps_grad = plot_maps(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure(f'{log_pref} Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        path_save02 = self.logger.log_dir + f'/{log_pref} maps_obs.png'
        fig_maps = plot_maps(
                self.obs_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                self.obs_pred[t_idx],
                self.test_lon, self.test_lat, path_save02, grad=True)
        self.test_figs['maps_obs'] = fig_maps
        self.logger.experiment.add_figure(f'{log_pref} Maps Obs', fig_maps, global_step=self.current_epoch)

        # animate maps
        if self.hparams.animate == True:
            path_save0 = self.logger.log_dir + f'/{log_pref} animation.mp4'
            animate_maps(self.x_gt,
                    self.x_oi,
                    self.x_rec,
                    self.lon, self.lat, path_save0)
            # save NetCDF
        path_save1 = self.logger.log_dir + f'/{log_pref} data.nc'
        # PENDING: replace hardcoded 60
        self.test_xr_ds.to_netcdf(path_save1)
        # save_netcdf(saved_path1=path_save1, pred=self.x_rec,
        #         lon=self.test_lon, lat=self.test_lat, time=self.test_dates, time_units=None)

        # compute nRMSE
        # np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)
        nrmse_fn = lambda pred, ref, gt: (
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
        mse_fn = lambda pred, ref, gt: (
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - self.test_xr_ds[gt])
                .pipe(lambda ds: ds ** 2)
                .to_dataframe()
                .pipe(lambda ds: ds.mean())
                .to_frame()
                .rename(columns={0: 'mse'})
                .assign(mse_ratio=lambda df: df / df.loc[ref])
        )

        nrmse_df = nrmse_fn('pred', 'oi', 'gt')
        mse_df = mse_fn('pred', 'oi', 'gt')

        nrmse_swath_df = nrmse_fn('obs_pred', 'obs_inp', 'obs_gt')
        mse_swath_df = mse_fn('obs_pred', 'obs_inp', 'obs_gt')
        nrmse_swath_df.to_csv(self.logger.log_dir + f'/{log_pref} nRMSE_swath.txt')
        mse_swath_df.to_csv(self.logger.log_dir + f'/{log_pref} MSE_swath.txt')
        # plot nRMSE
        # PENDING: replace hardcoded 60
        path_save3 = self.logger.log_dir + f'/{log_pref} nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,  self.x_oi, self.x_rec, path_save3, time=self.test_dates)
        self.test_figs['nrmse'] = nrmse_fig
        self.logger.experiment.add_figure(f'/{log_pref} NRMSE', nrmse_fig, global_step=self.current_epoch)
        # plot SNR
        path_save4 = self.logger.log_dir + f'/{log_pref} SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_oi, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig

        self.logger.experiment.add_figure(f'/{log_pref} SNR', snr_fig, global_step=self.current_epoch)
        
        fig, spatial_res_model, spatial_res_oi = get_psd_score(self.test_xr_ds.gt, self.test_xr_ds.pred, self.test_xr_ds.oi, with_fig=True)
        self.test_figs['res'] = fig
        self.logger.experiment.add_figure(f'/{log_pref} Spat. Resol', fig, global_step=self.current_epoch)
        # PENDING: Compute metrics on swath
        psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        psd_fig = metrics.plot_psd_score(psd_ds)
        self.logger.experiment.add_figure(f'/{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        self.test_figs['psd'] = fig
        _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)

        _mdf = pd.concat([
            nrmse_df.rename(columns=lambda c: f'{log_pref}_{c}_glob').loc['pred'].T,
            mse_df.rename(columns=lambda c: f'{log_pref}_{c}_glob').loc['pred'].T,
            nrmse_swath_df.rename(columns=lambda c: f'{log_pref}_{c}_swath').loc['obs_pred'].T,
            mse_swath_df.rename(columns=lambda c: f'{log_pref}_{c}_swath').loc['obs_pred'].T,
        ])
        md = {
            f'{log_pref}_spatial_res': float(spatial_res_model),
            f'{log_pref}_spatial_res_imp': float(spatial_res_model / spatial_res_oi),
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
            **_mdf.to_dict(), 
        }
        mdf = pd.DataFrame([md])
        metrics_path =Path(self.logger.log_dir).parent / 'metrics'
        metrics_path.mkdir(exist_ok=True)
        mdf.to_json(metrics_path / f'{log_pref}.json')

        self.latest_metrics.update(md)
        print(mdf.T.to_markdown())
        self.logger.log_metrics(md ,step=self.current_epoch)

        
    def configure_callbacks(self):
        def save_ckpt_metrics(trainer, pl_module, checkpoint):
            mdf = pd.DataFrame(pl_module.latest_metrics)
            metrics_path = Path(pl_module.logger.log_dir).parent / 'metrics'
            mdf.to_json(metrics_path / f'latest_ckpt.json')                                                                                                                                                       
        return [pl.callbacks.LambdaCallback(on_save_checkpoint=save_ckpt_metrics)]

    def teardown(self, stage='test'):
        if self.logger is None:
            return
        self.logger.log_hyperparams(
                {**self.hparams},
                self.latest_metrics
        )

    def get_init_state(self, batch, state):
        if state is not None:
            return state

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch

        anomaly_global = torch.zeros_like(targets_OI)

        if self.hparams.anom_swath_init == 'zeros':
            anomaly_swath = torch.zeros_like(targets_OI)
        elif self.hparams.anom_swath_init == 'obs':
            anomaly_swath = (inputs_obs - targets_OI).detach()

        if self.aug_state:
            if self.aug_state == 2:
                init_state = torch.cat((targets_OI, anomaly_global, anomaly_swath, torch.zeros_like(targets_OI), torch.zeros_like(targets_OI)), dim=1)
            else:
                init_state = torch.cat((targets_OI, anomaly_global, anomaly_swath, torch.zeros_like(targets_OI)), dim=1)
        else:
            init_state = torch.cat((targets_OI, anomaly_global, anomaly_swath), dim=1)
        return init_state

    def get_outputs(self, batch, state_out):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch
        # print(state_out.shape)
        # print(targets_OI.shape)
        if self.aug_state:
            if self.aug_state == 2:
                output_low_res, _,  _, output_anom_swath, output_anom_glob = torch.split(state_out, split_size_or_sections=targets_OI.size(1), dim=1)
            else:
                output_low_res,  _, output_anom_swath, output_anom_glob = torch.split(state_out, split_size_or_sections=targets_OI.size(1), dim=1)
        else:
            output_low_res,  output_anom_glob, output_anom_swath = torch.split(state_out, split_size_or_sections=targets_OI.size(1), dim=1)

        output_global = output_low_res + output_anom_glob

        if self.hparams.swot_anom_wrt == 'low_res':
            output_swath = output_low_res + output_anom_swath
        elif self.hparams.swot_anom_wrt == 'high_res':
            output_swath = output_global + output_anom_swath

        return output_low_res, output_global, output_swath

    def loss_ae(self, state_out):
        return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def compute_loss(self, batch, phase, state_init=None):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch

        #targets_OI, inputs_Mask, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_GT),
                    torch.zeros_like(target_obs_GT),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('mseSwath', 0.),
                        ('mseGradSwath', 0.),
                        ('meanGrad', 1.),
                        ('mseOI', 0.),
                        ('mseGOI', 0.)])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        target_obs_GT_wo_nan = target_obs_GT.where(~target_obs_GT.isnan(), torch.zeros_like(target_obs_GT))

        state = self.get_init_state(batch, state_init)

        #state = torch.cat((targets_OI, inputs_Mask * (targets_GT_wo_nan - targets_OI)), dim=1)
        if not self.use_sst:
            new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask), dim=1)
            obs = torch.cat((targets_OI, inputs_obs), dim=1)
        else:
            new_masks = [
                    torch.cat((torch.ones_like(inputs_Mask), inputs_Mask), dim=1),
                    torch.ones_like(sst_gt)
            ]
            obs = [
                    torch.cat((targets_OI, inputs_obs), dim=1),
                    sst_gt
            ]
        # PENDING: Add state with zeros

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)
        g_targets_obs = self.gradient_img(target_obs_GT_wo_nan)
        g_targets_obs_mask = self.gradient_img(target_obs_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks)

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            # PENDING: reconstruct outputs, outputs LowRes and outputSwath MegaRes

            if self.hparams.swot_anom_wrt == 'low_res':
                gt_anom_swath = targets_OI
            elif self.hparams.swot_anom_wrt == 'high_res':
                gt_anom_swath = targets_GT

            output_low_res, output_global, output_swath = self.get_outputs(batch, outputs)
            # reconstruction losses
            g_output_global = self.gradient_img(output_global)
            g_output_swath = self.gradient_img(output_swath)
            # PENDING: add loss term computed on obs (outputs swath - obs_target)

            _err_swath =(output_swath - target_obs_GT_wo_nan)**2 
            err_swath = _err_swath.where(target_obs_GT.isfinite(), torch.zeros_like(_err_swath))
            _err_g_swath =(g_output_swath - g_targets_obs)**2
            err_g_swath = _err_g_swath.where(g_targets_obs_mask.isfinite(), torch.zeros_like(_err_g_swath))

            loss_swath = NN_4DVar.compute_spatio_temp_weighted_loss(err_swath, self.w_loss)
            # print(f"{loss_swath=}")
            loss_grad_swath = NN_4DVar.compute_spatio_temp_weighted_loss(err_g_swath, self.w_loss)
            # print(f"{loss_grad_swath=}")

            loss_All = NN_4DVar.compute_spatio_temp_weighted_loss((output_global - targets_GT), self.w_loss)
            loss_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_output_global - g_targets_GT, self.w_loss)
            loss_OI = NN_4DVar.compute_spatio_temp_weighted_loss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_spatio_temp_weighted_loss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = self.loss_ae(outputs)

            if self.aug_state:

                if self.aug_state==2:
                    yGT = torch.cat((
                        targets_OI,
                        targets_GT_wo_nan - targets_OI,
                        (target_obs_GT_wo_nan - gt_anom_swath).where(target_obs_GT.isfinite(), torch.zeros_like(target_obs_GT)),
                        targets_GT_wo_nan - targets_OI,
                        (inputs_obs - gt_anom_swath).where(target_obs_GT.isfinite(), torch.zeros_like(target_obs_GT))),
                        dim=1)
                else:
                    yGT = torch.cat((targets_OI, targets_GT_wo_nan - targets_OI, target_obs_GT_wo_nan - gt_anom_swath, targets_GT_wo_nan - targets_OI), dim=1)
            else:

                yGT = torch.cat((targets_OI, targets_GT_wo_nan - targets_OI, target_obs_GT_wo_nan - gt_anom_swath), dim=1)
            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_spatio_temp_weighted_loss(output_low_res - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_spatio_temp_weighted_loss(self.model_LR(output_global) - targets_GTLR, self.model_LR(self.w_loss))

            # total loss
            loss = 0
            if self.hparams.loss_glob:
                loss += self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll

            if (self.hparams.loss_loc if hasattr(self.hparams, 'loss_loc') else 1):
                alpha_mse = self.hparams.alpha_loc_mse_ssh if hasattr(self.hparams, 'alpha_loc_mse_ssh') else self.hparams.alpha_mse_ssh
                alpha_gmse = self.hparams.alpha_loc_mse_gssh if hasattr(self.hparams, 'alpha_loc_mse_gssh') else self.hparams.alpha_mse_gssh
                loss += alpha_mse * loss_swath * 20
                loss += alpha_gmse * loss_grad_swath * 20

            if (self.hparams.loss_proj if hasattr(self.hparams, 'loss_proj') else 1):
                loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            if (self.hparams.loss_low_res if hasattr(self.hparams, 'loss_low_res') else 1):
                loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
            # metrics
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            mseSwath = loss_swath.detach()
            mseGradSwath = loss_grad_swath.detach()
            metrics = dict([
                ('mse', mse),
                ('mseGrad', mseGrad),
                ('mseSwath', mseSwath),
                ('mseGradSwath', mseGradSwath),
                ('meanGrad', mean_GAll),
                ('mseOI', loss_OI.detach()),
                ('mseGOI', loss_GOI.detach())])
            # PENDING: Add new loss term to metrics

        return loss, outputs, metrics


if  __name__ == '__main__':
    import hydra
    import torch
    import re
    import matplotlib.pyplot as plt
    import re
    from hydra.utils import instantiate, get_class, call
    from hydra.core.config_store import ConfigStore
    from hydra_main import FourDVarNetHydraRunner
    import pytorch_lightning as pl
    import pandas as pd
    from pathlib import Path
    import traceback
    import hydra_config
    from IPython.display import display, Markdown, Latex, HTML

    s = """
    ## TODO
    - [ ]  Use term without errors in sst comp
    - [ ]  Use aug state as explicit error terms
    - [ ]  Add normalization term
    """
    display(Markdown(s))


    def main():
        try:
            ...
        except Exception as e:
            print('I am here')
            print(traceback.format_exc()) 
        finally:
            return locals()

        """

        """
