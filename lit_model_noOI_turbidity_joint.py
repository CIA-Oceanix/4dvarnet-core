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
from models import Model_H, Phi_r_OI, Phi_r, Phi_r_unet, Gradient_img

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

class LitModelOI(LitModelAugstate):
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
        self.log("tr_msePhir", metrics[-1]['msePhir'], on_step=False, on_epoch=True, prog_bar=True)
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
            self.log(f'{log_pref}_msePhir', metrics[-1]['msePhir'], on_step=False, on_epoch=True, prog_bar=True)
        return {'ssh_gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'ssh_oi' : (oi.detach().cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'ssh_obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'ssh_pred' : (out.detach().cpu() * np.sqrt(self.var_ssh_Tr)) + self.mean_ssh_Tr,
                'sst_gt'    : (sst_gt.detach().cpu() * np.sqrt(self.var_sst_Tr)) + self.mean_sst_Tr,
                'sst_obs_inp'    : (sst_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_sst_Tr)) + self.mean_sst_Tr,
                'sst_pred' : (out.detach().cpu() * np.sqrt(self.var_sst_Tr)) + self.mean_sst_Tr}}

    def sla_diag(self, t_idx=3, log_pref='test'):
        path_save0 = self.logger.log_dir + '/maps.png'
        t_idx = 3
        fig_maps = plot_maps_oi(
                  self.ssh_gt[t_idx],
                self.ssh_obs_inp[t_idx],
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
        ###############
        # animate maps
        ###############
        #print(self.hparams)
        if self.hparams.animate:
            path_save0 = self.logger.log_dir + '/animation.mp4'
            animate_maps_OI(self.x_gt, self.obs_inp, self.x_rec, self.test_lon, self.test_lat, path_save0)

            path_save0 = self.logger.log_dir + '/animation_grad.mp4'
            animate_maps_OI(self.x_gt, self.obs_inp, self.x_rec, self.test_lon, self.test_lat, path_save0, grad=True)

        # ~ psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        # ~ psd_fig = metrics.plot_psd_score(psd_ds)
        # ~ self.test_figs['psd'] = psd_fig
        # ~ self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)

        md = {
            # ~ f'{log_pref}_lambda_x': lamb_x,
            # ~ f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
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
                        ('msePhir', 0.),
                        ])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))

        state = self.get_init_state(batch, state_init)

        obs = inputs_Mask * inputs_obs
        new_masks =  inputs_Mask
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)
        

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            if self.hparams.n_grad>0:
                outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])
                if (phase == 'val') or (phase == 'test'):
                    outputs = outputs.detach()
            else:
                outputs = self.model.phi_r(obs)
                hidden_new = None
                cell_new = None
                normgrad = None
            
            # ~ loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
            
            outputs_GT_wo_nan = outputs.where(~targets_GT.isnan(), torch.zeros_like(outputs))
            loss_All, loss_GAll = self.sla_loss(outputs_GT_wo_nan, targets_GT_wo_nan)
            
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
            msePhir = loss_AE.detach()
            metrics = dict([
                ('mse', mse),
                ('mseGrad', mseGrad),
                ('meanGrad', mean_GAll),
                ('msePhir', msePhir),
                ])
        return loss, outputs, [outputs, hidden_new, cell_new, normgrad], metrics

