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
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps_oi, animate_maps, get_psd_score
from models import Model_H, Phi_r_OI, Gradient_img

from lit_model_augstate import LitModelAugstate

def get_4dvarnet_OI(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

def get_4dvarnet_OI_sst(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_HwithSST(hparams.shape_state[0], dT=hparams.dT),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


class LitModelOI(LitModelAugstate):
    MODELS = {
            '4dvarnet_OI': get_4dvarnet_OI,
            '4dvarnet_OI_sst': get_4dvarnet_OI_sst,
             }

    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)
        if self.model_name == '4dvarnet_OI':
            optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ])
        elif self.model_name == '4dvarnet_OI_sst':

            optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ])

        return optimizer

    def diag_step(self, batch, batch_idx, log_pref='test'):
        if not self.use_sst:
            _, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            _, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def sla_diag(self, t_idx=3, log_pref='test'):

        path_save0 = self.logger.log_dir + '/maps.png'
        t_idx = 3
        fig_maps = plot_maps_oi(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps_oi(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure(f'{log_pref} Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        '''
        psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        psd_fig = metrics.plot_psd_score(psd_ds)
        self.test_figs['psd'] = psd_fig
        self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        
        md = {
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
        }
        '''
        md = {
            f'{log_pref}_lambda_x': np.nan,
            f'{log_pref}_lambda_t': np.nan,
            f'{log_pref}_mu': np.nan,
            f'{log_pref}_sigma': np.nan,
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
        self.test_xr_ds.to_netcdf(path_save1)

        self.x_gt = self.test_xr_ds.gt.data
        self.obs_inp = self.test_xr_ds.obs_inp.data
        self.x_rec = self.test_xr_ds.pred.data
        self.x_rec_ssh = self.x_rec

        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data

        # display map
        md = self.sla_diag(t_idx=3, log_pref=log_pref)
        self.latest_metrics.update(md)
        self.logger.log_metrics(md, step=self.current_epoch)

    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        _, inputs_Mask, inputs_obs, _ = batch

        init_state = inputs_Mask * inputs_obs
        return init_state

    def compute_loss(self, batch, phase, state_init=(None,)):

        _, inputs_Mask, inputs_obs, targets_GT = batch

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
                        ])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))

        state = self.get_init_state(batch, state_init)

        obs = inputs_Mask * inputs_obs
        new_masks =  inputs_Mask

        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
            loss_AE = self.loss_ae(outputs)
            if self.hparams.supervised==True:
                # total loss
                loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                loss += 0.5 * self.hparams.alpha_proj * loss_AE
            else:
                self.type_loss_supervised = self.hparams.type_loss_supervised if hasattr(self.hparams, 'type_loss_supervised') else 'var_cost'
                if self.type_loss_supervised != "var_cost":
                    # MSE
                    mask = (targets_GT_wo_nan!=0.)
                    iT = int(self.hparams.dT / 2)
                    new_tensor = torch.masked_select(outputs[:,iT,:,:],mask[:,iT,:,:]) - torch.masked_select(targets_GT[:,iT,:,:],mask[:,iT,:,:])
                    loss = NN_4DVar.compute_WeightedLoss(new_tensor, torch.tensor(1.))
                    loss = self.hparams.alpha_mse_ssh * loss  + 0.5 * self.hparams.alpha_proj * loss_AE 
                else:
                    dy = self.model.model_H(outputs,obs,new_masks)
                    dx = outputs - self.model.phi_r(outputs)
                    loss = self.model.model_VarCost(dx,dy)
                loss += 10 * loss_AE

            # metrics
            # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_targets_GT_x, g_targets_GT_y) , self.grad_crop(self.patch_weight))
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            metrics = dict([
                ('mse', mse),
                ('mseGrad', mseGrad),
                ('meanGrad', mean_GAll),
                ])

        return loss, outputs, [outputs, hidden_new, cell_new, normgrad], metrics
