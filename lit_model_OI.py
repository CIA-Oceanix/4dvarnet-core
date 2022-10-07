import einops
import torch.distributed as dist
import kornia
from hydra.utils import instantiate
import pandas as pd
from functools import reduce
from torch.nn.modules import loss
from torch import nn
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
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps_oi, animate_maps, get_psd_score, plot_multi_prior_maps
from models import Model_H, Model_HwithSST, Phi_r_OI,Phi_r_OI_linear, Gradient_img, UNet, Phi_r_UNet, Multi_Prior

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
#4dvarnet with linear phi_r instead of bilinear units
def get_4dvarnet_OI_linear(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI_linear(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
#Unet with batch normalization
def get_4dvarnet_unet_bn(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                nn.Sequential(
            nn.BatchNorm2d(hparams.shape_state[0]),
            Phi_r_UNet(hparams.shape_state[0], hparams.dropout_phi_r, hparams.stochastic,False, shrink_factor=hparams.UNet_shrink_factor)
        ),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

def get_4dvarnet_unet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r_UNet(hparams.shape_state[0], hparams.dropout_phi_r, hparams.stochastic,False, shrink_factor=hparams.UNet_shrink_factor),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)



def get_4dvarnet_unet_sst(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_UNet(hparams.shape_state[0], hparams.dropout_phi_r, hparams.stochastic,False, shrink_factor=hparams.UNet_shrink_factor),
                Model_HwithSST(hparams.shape_state[0], dT=hparams.dT),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

#Direct UNet with no solver 
def get_UNet_direct(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi =  nn.Sequential(
            nn.BatchNorm2d(hparams.shape_state[0]),
            Phi_r_UNet(hparams.shape_state[0], hparams.dropout_phi_r, hparams.stochastic,False, shrink_factor=hparams.UNet_shrink_factor)
        )
            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks, *internal_state):
            return self.phi(state), None, None, None
    return PhiPassThrough()

def get_multi_prior(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Multi_Prior(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.nb_phi, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)




#UNet and a fixed point solver
def get_UNet_fixed_point(hparams):
    return NN_4DVar.FP_Solver(
        nn.Sequential(
            nn.BatchNorm2d(hparams.shape_state[0]),
            Phi_r_UNet(hparams.shape_state[0], hparams.dropout_phi_r, hparams.stochastic, shrink_factor=hparams.UNet_shrink_factor)
        ),
    hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

#4dvarnet with the phi_r_OI and a fixed point solver
def get_phi_r_fixed_point(hparams):
    return NN_4DVar.FP_Solver(
         nn.Sequential(
            nn.BatchNorm2d(hparams.shape_state[0]),
            Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                        hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic)),
    hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

class LitModelOI(LitModelAugstate):
    MODELS = {
        '4dvarnet_OI': get_4dvarnet_OI,
        '4dvarnet_OI_sst': get_4dvarnet_OI_sst,
        '4dvarnet_OI_linear': get_4dvarnet_OI_linear,
        '4dvarnet_UNet_sst': get_4dvarnet_unet_sst,
        '4dvarnet_UNet': get_4dvarnet_unet,
        'UNet_direct': get_UNet_direct,
        'UNet_FP': get_UNet_fixed_point,
        'phi_r_FP': get_phi_r_fixed_point,
        'multi_prior': get_multi_prior
     }

    # def add_model_specific_args(self, parent_parser):
    #     parser = parent_parser.add_argument_group("LitModel_OI")
    #     parser.add_argument("--FP_iterations", type=int, default=self.hparams.n_grad * self.hparams.n_fourdvar_iter)


    def __init__(self, *args, **kwargs):
         super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)
        if self.model_name in ['4dvarnet_OI','4dvarnet_OI_linear', '4dvarnet_OI_sst', '4dvarnet_UNet','4dvarnet_UNet_sst']:
            optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ])
        elif self.model_name in ['4dvarnet_UNet_gradient']:
            optimizer = opt([{'params': self.model.phi_r.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]}
                ])
        elif self.model_name in [ 'UNet_direct','UNet_FP', 'phi_r_FP', 'multi_prior']:
            optimizer = opt([{'params': self.model.parameters(), 'lr': self.hparams.lr_update[0]}])


        return optimizer

    def diag_step(self, batch, batch_idx, log_pref='test'):
        oi, inputs_Mask, inputs_obs, targets_GT, *_= batch
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None and log_pref is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (oi.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def sla_diag(self, t_idx=3, log_pref='test'):
        path_save0 = self.logger.log_dir + '/maps.png'
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
        #Make maps for multi phi modles
        if self.model_name == 'multi_prior':
            path_save03 = self.logger.log_dir + '/prior_maps.png'
            mp_maps = plot_multi_prior_maps(self.x_gt[t_idx],self.obs_inp[t_idx],
             self.model.phi_r.phi_list,self.test_lon, self.test_lat, path_save03)
            path_save04= self.logger.log_dir + '/prior_maps_Grad.png'
            mp_maps_grad = plot_multi_prior_maps(self.x_gt[t_idx],self.obs_inp[t_idx],
             self.model.phi_r.phi_list,self.test_lon, self.test_lat, path_save03, grad=True)

        lamb_x, lamb_t, mu, sig = np.nan, np.nan, np.nan, np.nan
        try:
            psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
            psd_fig = metrics.plot_psd_score(psd_ds)
            self.test_figs['psd'] = psd_fig
            self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
            _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        except:
            print('fail to compute psd scores')
        mse_metrics_pred = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.pred)
        mse_metrics_oi = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.oi)
        var_mse_pred_vs_oi = 100. * ( 1. - mse_metrics_pred['mse'] / mse_metrics_oi['mse'] )
        var_mse_grad_pred_vs_oi = 100. * ( 1. - mse_metrics_pred['mseGrad'] / mse_metrics_oi['mseGrad'] )
        md = {
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
            f'{log_pref}_var_mse_vs_oi': float(var_mse_pred_vs_oi),
            f'{log_pref}_var_mse_grad_vs_oi': float(var_mse_grad_pred_vs_oi),
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

        log_path = Path(self.logger.log_dir).mkdir(exist_ok=True)
        print('########', f'{log_path=}')
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

        _, inputs_Mask, inputs_obs, *_ = batch

        init_state = inputs_Mask * inputs_obs
        return init_state

    def loss_ae(self, state_out):
        #Ignore autoencoder loss for fixed point solver
        if self.model_name in ['UNet_FP']:
            return 0.
        else: 
            #same as in lit_model_augstate
            return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def compute_loss(self, batch, phase, state_init=(None,)):


        if not self.use_sst:
            _, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            _, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

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

        if self.use_sst:
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]

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

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
            loss += 0.5 * self.hparams.alpha_proj * loss_AE

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
