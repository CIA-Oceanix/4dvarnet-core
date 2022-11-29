from traceback import print_tb
import einops
import logging
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

import print_log
log = print_log.get_logger(__name__)


def get_4dvarnet_OI(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

class LitModelOI(LitModelAugstate):
    MODELS = {
            '4dvarnet_OI': get_4dvarnet_OI,
             }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_train = []
        self.log_val = []
        self.print_log_train_val = False
        print(self.hparams)
        #self.save_outputs_path = self.hparams.save_outputs_path

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

        return optimizer

    def diag_step(self, batch, batch_idx, log_pref='test'):
        _, inputs_Mask, inputs_obs, targets_GT = batch
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1] 
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

            if log_pref == 'val' :
                self.log_val.append( [loss, metrics[-1]["mse"] / self.var_Tt, metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'] ] )

        return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def sla_diag(self, t_idx=3, log_pref='test'):
        if hasattr(self.hparams, 'save_outputs_path') : 
            self.save_outputs_path = self.hparams.save_outputs_path if self.hparams.save_outputs_path else self.logger.log_dir
        else : 
            self.save_outputs_path = self.logger.log_dir
        #log.info('Save outputs files in {}'.format(self.save_outputs_path))
        Path(self.save_outputs_path).mkdir(parents = True, exist_ok=True)

        path_save0 = self.save_outputs_path + '/maps.png'
        t_idx = 3
        print(self.x_gt.shape, self.x_gt[3])
        print(self.obs_inp.shape)
        print(self.x_rec.shape)

        fig_maps = plot_maps_oi(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.save_outputs_path + '/maps_Grad.png'
        fig_maps_grad = plot_maps_oi(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure(f'{log_pref} Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        psd_fig = metrics.plot_psd_score(psd_ds, self.save_outputs_path + '/psd_score.png')
        self.test_figs['psd'] = psd_fig
        self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)

        md = {
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
        }
        print(pd.DataFrame([md]).T.to_markdown())
        log.info('Computing additional diags : lambda_x = {}, lambda_t = {}, mu = {}, sigma = {}'.format(lamb_x, lamb_t, mu, sig))
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

        if log_pref == 'test' :
            path_save1 = self.save_outputs_path + f'/outputs_dataset_test.nc'
            self.test_xr_ds.attrs = md 
            self.test_xr_ds.to_netcdf(path_save1)



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
        self.log("tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        self.log_train.append( [loss, metrics[-1]['mse'] / self.var_Tr, metrics[-1]['mseGrad'] / metrics[-1]['meanGrad']] )
        self.print_log_train_val = True
        return loss



    def validation_epoch_end(self, outputs):
        if self.print_log_train_val :
            log_val = np.nanmean(self.log_val, axis = 0)
            log_train = np.nanmean(self.log_train, axis = 0)
            log.info('------------------------------------------------------------------------------------------')
            log.info('Epochs {} - val_mse = {:.4}, val_mseG = {:.4}, tr_loss = {:.4}, tr_mse = {:.4}, tr_mseG = {:.4}'.format(
                self.current_epoch, log_val[1], log_val[2], 
                log_train[0], log_train[1], log_train[2]))

            self.log_val = []
            self.log_train = []

        
        if (self.current_epoch + 1) % self.hparams.val_diag_freq == 0:
            return self.diag_epoch_end(outputs, log_pref='val')
