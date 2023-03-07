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
import ose.solver as NN_4DVar
import metrics
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps_oi, animate_maps, get_psd_score
from models import Model_H, Model_HwithSST, Phi_r_OI, Gradient_img
# additional import for OSE metrics
from ose.src.mod_inout import *
from ose.src.mod_interp import *
from ose.src.mod_stats import *
from ose.src.mod_spectral import *
from ose.src.mod_plot import *
from ose.src.utils import *
import copy

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
         # parameters for ose metrics
         # study area
         self.lon_min = 295.
         self.lon_max = 305.
         self.lat_min = 33.
         self.lat_max = 43.
         self.is_circle = False
         self.time_min = '2017-01-13'
         self.time_max = '2017-12-18'
         # outputs
         self.bin_lat_step = 1.
         self.bin_lon_step = 1.
         self.bin_time_step = '1D'
         # spectral parameter
         # C2 parameter
         #self.c2_file = self.hparams.files_cfg.c2_path
         self.c2_file = "/gpfswork/rech/yrf/uba22to/4dvarnet-core/ose/eval_notebooks/inputs/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
         #self.c2_file = "/users/local/m19beauc/4dvarnet-core/ose/eval_notebooks/inputs/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc"
         self.ds_alongtrack = read_l3_dataset(self.c2_file,
                                           lon_min=self.lon_min,
                                           lon_max=self.lon_max,
                                           lat_min=self.lat_min,
                                           lat_max=self.lat_max,
                                           time_min=self.time_min,
                                           time_max=self.time_max)
         self.c2_delta_t = 0.9434  # s
         self.c2_velocity = 6.77   # km/s
         self.c2_delta_x = self.c2_velocity * self.c2_delta_t
         self.c2_length_scale = 1000 # km

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)

        if self.model_name == '4dvarnet_OI':
            optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                             {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                             {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                             #{'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ])
        elif self.model_name == '4dvarnet_OI_sst':

            optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ])

        return optimizer

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out=None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, out, state, _metrics = self.compute_loss(self.model,batch, phase=phase, state_init=state_init)
            state_init = [None if s is None else s.detach() for s in state]
        losses.append(_loss)
        metrics.append(_metrics)
        return losses, out, metrics

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
        t_idx = 2
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

        # compute ose metrics
        data = xr.open_dataset(path_save1)
        if 'obs_inp' in list(data.keys()):
            data = data.update({'obs':(('time','lat','lon'),data['obs_inp'].values)})
        data = data.update({'ssh':(('time','lat','lon'),data.pred.values)})
        data = data.assign_coords(time=[np.datetime64(day) for day in data.time.values])
        file_4dvarnet = path_save1.replace(".nc","_reformat.nc")
        data.to_netcdf(file_4dvarnet)
        # interpolate 4DVarNet on tracks
        itrp = interp_on_alongtrack(file_4dvarnet,
                     self.ds_alongtrack,
                     lon_min=self.lon_min,
                     lon_max=self.lon_max,
                     lat_min=self.lat_min,
                     lat_max=self.lat_max,
                     time_min=self.time_min,
                     time_max=self.time_max,
                     is_circle=self.is_circle)
        time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_4dvarnet_interp = itrp
        # Compute spatial and temporal statistics
        st_stats = compute_stats(time_alongtrack,
                                 lat_alongtrack,
                                 lon_alongtrack,
                                 ssh_alongtrack,
                                 ssh_4dvarnet_interp,
                                 self.bin_lon_step,
                                 self.bin_lat_step,
                                 self.bin_time_step,
                                 output_filename=self.logger.log_dir + f'/spa_stat_4dvarnet.nc',
                                 output_filename_timeseries=self.logger.log_dir + f'/TS_4dvarnet.nc')
        leaderboard_nrmse_4dvarnet, leaderboard_nrmse_std_4dvarnet = st_stats
        # compute spectral statistics
        compute_spectral_scores(time_alongtrack,
                        lat_alongtrack,
                        lon_alongtrack,
                        ssh_alongtrack,
                        ssh_4dvarnet_interp,
                        self.c2_length_scale,
                        self.c2_delta_x,
                        self.c2_delta_t,
                        self.logger.log_dir + f'/spectrum_4dvarnet.nc')
        leaderboard_psds_score_4dvarnet = plot_psd_score(self.logger.log_dir + f'/spectrum_4dvarnet.nc') 
        # Print leaderboard
        data = np.array([['4DVarNet'],
           [leaderboard_nrmse_4dvarnet],
           [leaderboard_nrmse_std_4dvarnet],             
           [int(leaderboard_psds_score_4dvarnet)]]).T
        Leaderboard = pd.DataFrame(data,
                           columns=['Method',
                                    "µ(RMSE) ",
                                    "σ(RMSE)",
                                    'λx (km)'])
        print("Summary of the leaderboard metrics:")
        print(Leaderboard)
        path_metrics = self.logger.log_dir + f'/metrics.txt'
        Leaderboard.to_csv(path_metrics, header=True, index=None, sep='\t')

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

        if not self.use_sst:
            _, inputs_Mask, inputs_obs, _ = batch
        else:
            _, inputs_Mask, inputs_obs, _, _ = batch

        init_state = inputs_Mask * inputs_obs
        return init_state

    def compute_loss(self, model, batch, phase, state_init=(None,)):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

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
        obs = inputs_Mask * inputs_obs
        new_masks =  inputs_Mask

        if self.use_sst:
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]

        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))

        state = self.get_init_state(batch, state_init)

        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = model(state, obs, new_masks, *state_init[1:])

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
                if self.type_loss_supervised == "loss_on_track":
                    # MSE
                    itime = int(self.hparams.dT/2)
                    gt = torch.where(obs==0,0.,targets_GT.double())
                    pred_track = torch.where(obs==0,0.,outputs.double())
                    mask = (obs!=0)
                    pred_diff = torch.masked_select(pred_track[:,itime,:,:],mask[:,itime,:,:]) - torch.masked_select(gt[:,itime,:,:],mask[:,itime,:,:])
                    loss = NN_4DVar.compute_WeightedLoss(pred_diff, torch.tensor(1.))
                    # add spatial information
                    grad_pred = kornia.filters.sobel(pred_track,3)
                    grad_gt = kornia.filters.sobel(gt,3)
                    laplacian_pred = kornia.filters.laplacian(pred_track,3)
                    laplacian_gt = kornia.filters.laplacian(gt,3)
                    grad_pred_diff = torch.masked_select(grad_pred[:,itime,:,:],mask[:,itime,:,:]) - torch.masked_select(grad_gt[:,itime,:,:],mask[:,itime,:,:])
                    laplacian_pred_diff = torch.masked_select(laplacian_pred[:,itime,:,:],mask[:,itime,:,:]) - torch.masked_select(laplacian_gt[:,itime,:,:],mask[:,itime,:,:])
                    loss_grad = NN_4DVar.compute_WeightedLoss(grad_pred_diff, torch.tensor(1.))
                    loss_laplacian = NN_4DVar.compute_WeightedLoss(laplacian_pred_diff, torch.tensor(1.))
                    loss += 0.1*loss_grad + 0.1*loss_laplacian
                else:
                    dy = model.model_H(outputs,obs,new_masks)
                    dx = outputs - model.phi_r(outputs)
                    loss, loss_prior, loss_obs = model.model_VarCost(dx,dy)
                    loss_LR = NN_4DVar.compute_spatio_temp_weighted_loss(self.model_LR(outputs)-self.model_LR(targets_OI),
                                                                         self.model_LR(self.patch_weight))
                    loss += 10*loss_LR
                    
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
