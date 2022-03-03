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
from xarray.core.common import zeros_like
import solver as NN_4DVar
import metrics
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, get_psd_score
from models import Phi_r, ModelLR, Gradient_img
from calibration.lit_cal_model import LitCalModel
from calibration.models import get_passthrough, get_vit

class ModelHWithExplicitError(torch.nn.Module):
    """
    state x:
    # use_loc_estim = False : (x_lr, x_hr, x_err)
    # use_loc_estim = True : (x_lr, x_hr_glob, x_hr_loc, x_err_glob, x_err_loc)

    obs_y:
    
    # use_sst = False : (y_lr, y_nadirswot)
    # use_sst = True : (y_lr, y_nadirswot), (y_sst)
    """
    def __init__(self, shape_data, shape_obs, hparams=None):
        super().__init__()
        self.use_sst = hparams.sst
        self.dim_obs = 2 if self.use_sst else 1
        self.dim_obs_channel = np.array([shape_obs])
        self.hparams = hparams
        self.use_loc_estim = hparams.loc_estim
        if self.use_loc_estim:
            assert shape_data == 5 * hparams.dT , 'sanity check fail'
        self.err_scaling = torch.nn.Parameter(torch.scalar_tensor(0.), requires_grad=hparams.train_error_scaling)
        if self.use_sst:
            ssh_ch = 2*hparams.dT 
            sst_ch = hparams.dT
            self.dim_obs_channel = np.array([shape_data, sst_ch])

            self.conv11 = torch.nn.Conv2d(ssh_ch, hparams.dT, (3, 3), padding=1, bias=False)
            self.conv21 = torch.nn.Conv2d(sst_ch, hparams.dT, (3, 3), padding=1, bias=False)
            self.conv_m = torch.nn.Conv2d(sst_ch, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
            self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

        # case sst
        # case no sst

    def get_y_hat(self, x, loc=False, err=False):
        x_components = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
        x_lr = x_components[0]
        x_hr = x_components[3] if loc and self.use_loc_estim else x_components[1]
        if err:
            x_err = x_components[4] if loc and self.use_loc_estim else x_components[2]
            x_hr = x_hr + x_err * 10**self.err_scaling

        return torch.cat((x_lr, x_hr), dim=1)


    def sst_cost(self, yhat_wo_err, y_sst, mask_sst):
        dyout = self.conv11(yhat_wo_err) - self.conv21(y_sst)
        dyout = dyout * self.sigmoid(self.conv_m(mask_sst))
        return dyout

    def ssh_cost(self, yhat_w_err, y_ssh, mask_ssh):
        return (yhat_w_err - y_ssh) * mask_ssh


    def forward(self, x, y, mask):
        dyouts = []
        if self.use_sst:
            y_ssh, y_sst = y
            mask_ssh, mask_sst = y

            y_hat_no_err = self.get_y_hat(x, loc=False, err=False)
            dyout1 = self.sst_cost(y_hat_no_err, y_sst, mask_sst)

        else:
            y_ssh = y 
            mask_ssh = mask 

        y_hat_err = self.get_y_hat(x, loc=False, err=True)
        dyouts.append(self.ssh_cost(y_hat_err, y_ssh, mask_ssh))

        if self.use_loc_estim:

            y_hat_loc_err = self.get_y_hat(x, loc=True, err=True)
            dyouts.append(self.ssh_cost(y_hat_loc_err, y_ssh, mask_ssh))
        
        dyouts =  torch.stack(dyouts).sum(0)
        if self.use_sst:
            return dyouts, dyout1

        return dyouts


class PhiRWrapper(torch.nn.Module):
    ERR_PRIORS = ('none', 'same', 'diff')
    def __init__(self, hparams, shape_data, *args, **kwargs):
        super().__init__()
        self.err_prior = hparams.get('err_prior', 'same')
        self.use_loc_estim = hparams.get('loc_estim', hparams.dT *5 == shape_data)
        self.hparams = hparams
        assert self.err_prior in self.ERR_PRIORS, f"err_prior should be one of {self.ERR_PRIORS}"


        if self.err_prior == 'same':
            self.phi_ssh, self.phi_err = Phi_r(shape_data, *args, **kwargs), None
            return

        if self.use_loc_estim:
            shape_data_ssh = shape_data - 2*int(hparams.dT)
            shape_data_err = 2*int(hparams.dT)
        else:
            shape_data_ssh = shape_data - 1*int(hparams.dT)
            shape_data_err = 1*int(hparams.dT)
        
        phi_ssh = Phi_r(shape_data_ssh, *args, **kwargs)
        if self.err_prior == 'none':
            self.phi_ssh, self.phi_err = phi_ssh, None
            return

        phi_err = Phi_r(shape_data_err, *args, **kwargs)
        if self.err_prior == 'diff':
            self.phi_ssh, self.phi_err = phi_ssh, phi_err
            return

        assert False, "Should have return by now"



    def split_state(self, x):
        if self.use_loc_estim:
            x_lr, x_hr_glob, x_err_glob, x_hr_loc, x_err_loc = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
            x_ssh = torch.cat((x_lr, x_hr_glob, x_hr_loc), dim=1)
            x_err = torch.cat((x_err_glob, x_err_loc), dim=1)
        else:
            x_lr, x_hr_glob, x_err_glob = torch.split(x, split_size_or_sections=self.hparams.dT, dim=1)
            x_ssh = torch.cat((x_lr, x_hr_glob), dim=1)
            x_err = x_err_glob

        return x_ssh, x_err

    def join_state(self, x_ssh, x_err):
        if self.use_loc_estim:
            x_lr, x_hr_glob, x_hr_loc = torch.split(x_ssh, split_size_or_sections=self.hparams.dT, dim=1)
            x_err_glob, x_err_loc = torch.split(x_err, split_size_or_sections=self.hparams.dT, dim=1)
            x = torch.cat((x_lr, x_hr_glob, x_err_glob, x_hr_loc, x_err_loc), dim=1)
        else:
            x_lr, x_hr_glob  = torch.split(x_ssh, split_size_or_sections=self.hparams.dT, dim=1)
            x_err_glob = x_err
            x = torch.cat((x_lr, x_hr_glob, x_err_glob), dim=1)
        return x

    def same_forward(self, x):
        return self.phi_ssh(x)

    def none_forward(self, x):
        x_ssh, x_err = self.split_state(x)
        phi_x_ssh = self.phi_ssh(x_ssh)
        return self.join_state(phi_x_ssh, x_err)

    def diff_forward(self, x):
        x_ssh, x_err = self.split_state(x)
        phi_x_ssh = self.phi_ssh(x_ssh)
        phi_x_err = self.phi_err(x_err)
        return self.join_state(phi_x_ssh, phi_x_err)

    def forward(self, x):
        forward_fn =  dict(
            same=self.same_forward,
            none=self.none_forward,
            diff=self.diff_forward,
        )[self.err_prior]

        return forward_fn(x)

class LitModel(LitCalModel):
    def create_model(self):
        hparams = self.hparams
        return NN_4DVar.Solver_Grad_4DVarNN(
            PhiRWrapper(hparams, hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
            ModelHWithExplicitError(hparams.shape_data[0], hparams.shape_obs[0], hparams=hparams),
            NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                hparams.dim_grad_solver, hparams.dropout),
            hparams.norm_obs, hparams.norm_prior, hparams.shape_data, hparams.n_grad
        )


    def get_init_state(self, batch, state):
        if state is not None:
            return state

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch

        anomaly_global = torch.zeros_like(targets_OI)
        err_global = torch.zeros_like(targets_OI)
        
        if self.hparams.loc_estim:
            anomaly_local = (inputs_obs - targets_OI).detach() / 2
            err_local = (inputs_obs - targets_OI).detach() / 2
            init_state = torch.cat((targets_OI, anomaly_global, err_global, anomaly_local, err_local), dim=1)
        else:
            init_state = torch.cat((targets_OI, anomaly_global, err_global), dim=1)

        return init_state

    def get_outputs(self, batch, state_out, with_error=False):
        sum_channels = lambda t: einops.reduce(t, 'b (n t) lat lon -> b t lat lon', t=self.hparams.dT, reduction='sum')
        x_glob = sum_channels(self.model.model_H.get_y_hat(state_out, loc=False, err=with_error))

        x_loc = sum_channels(self.model.model_H.get_y_hat(state_out, loc=True, err=with_error))
        x_lr = state_out[:, :self.hparams.dT, ...]
        return x_lr, x_glob, x_loc

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
                    dict([('mse', 0.), ('mseGrad', 0.), ('mseSwath', 0.),
                        ('mseGradSwath', 0.), ('meanGrad', 1.), ('mseOI', 0.), ('mseGOI', 0.)])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        target_obs_GT_wo_nan = target_obs_GT.where(~target_obs_GT.isnan(), torch.zeros_like(target_obs_GT))
        target_obs_err_GT_wo_nan = (inputs_obs - target_obs_GT).where(~target_obs_GT.isnan(), torch.zeros_like(target_obs_GT))

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

            _, x_glob_w_err, x_loc_w_err = self.get_outputs(batch, outputs, with_error=True)
            x_lr, x_glob_wo_err, x_loc_wo_err = self.get_outputs(batch, outputs, with_error=False)
            # reconstruction losses
            g_output_global = self.gradient_img(x_glob_wo_err)
            g_output_swath = self.gradient_img(x_loc_wo_err)

            _err_swath =(x_loc_wo_err - target_obs_GT_wo_nan)**2 
            err_swath = _err_swath.where(target_obs_GT.isfinite(), torch.zeros_like(_err_swath))
            _err_err_swath =((x_loc_w_err - x_loc_wo_err) - target_obs_err_GT_wo_nan)**2 
            err_err_swath = _err_swath.where(target_obs_GT.isfinite(), torch.zeros_like(_err_swath))
            _err_g_swath =(g_output_swath - g_targets_obs)**2
            err_g_swath = _err_g_swath.where(g_targets_obs_mask.isfinite(), torch.zeros_like(_err_g_swath))

            loss_swath = NN_4DVar.compute_spatio_temp_weighted_loss(err_swath, self.w_loss)
            loss_err_swath = NN_4DVar.compute_spatio_temp_weighted_loss(err_swath, self.w_loss)
            loss_grad_swath = NN_4DVar.compute_spatio_temp_weighted_loss(err_g_swath, self.w_loss)

            loss_All = NN_4DVar.compute_spatio_temp_weighted_loss((x_glob_wo_err - targets_GT), self.w_loss)
            loss_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_output_global - g_targets_GT, self.w_loss)
            loss_OI = NN_4DVar.compute_spatio_temp_weighted_loss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_spatio_temp_weighted_loss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = self.loss_ae(outputs)

            if self.hparams.loc_estim:
                yGT = torch.cat((
                    targets_OI,
                    targets_GT_wo_nan - targets_OI,
                    (inputs_obs - targets_GT_wo_nan).where(target_obs_GT.isfinite(), torch.zeros_like(targets_OI)) / 10**self.model.model_H.err_scaling,
                    (target_obs_GT_wo_nan - targets_OI).where(target_obs_GT.isfinite(), targets_GT_wo_nan - targets_OI),
                    (inputs_obs - target_obs_GT_wo_nan).where(target_obs_GT.isfinite(), torch.zeros_like(targets_OI))/ 10**self.model.model_H.err_scaling,
                    ), dim=1)
            else:
                yGT = torch.cat((
                    targets_OI,
                    targets_GT_wo_nan - targets_OI,
                    (inputs_obs - targets_GT_wo_nan).where(target_obs_GT.isfinite(), torch.zeros_like(targets_OI))/ 10**self.model.model_H.err_scaling,
                    ), dim=1)

            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_spatio_temp_weighted_loss(x_lr - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_spatio_temp_weighted_loss(self.model_LR(x_glob_wo_err) - targets_GTLR, self.model_LR(self.w_loss))

            # total loss
            loss = 0
            if self.hparams.loss_glob:
                loss += self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                # print('#############', loss, 1)


            if (self.hparams.loss_loc if hasattr(self.hparams, 'loss_loc') else 1):
                alpha_mse = self.hparams.alpha_loc_mse_ssh if hasattr(self.hparams, 'alpha_loc_mse_ssh') else self.hparams.alpha_mse_ssh
                alpha_gmse = self.hparams.alpha_loc_mse_gssh if hasattr(self.hparams, 'alpha_loc_mse_gssh') else self.hparams.alpha_mse_gssh
                loss += alpha_mse * loss_swath * 20
                # print('#############', loss, 2)
                loss += alpha_gmse * loss_grad_swath * 20
                # print('#############', loss, 3)

            if self.hparams.loss_err:
                loss += self.hparams.alpha_err_mse_ssh * loss_err_swath
                # print('#############', loss, 4)

            if self.hparams.train_error_scaling:
                loss += 0.1 * max((outputs**2).mean() - (targets_OI**2).mean(), 0)

            if (self.hparams.loss_proj if hasattr(self.hparams, 'loss_proj') else 1):
                loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
                # print('#############', loss, 5)
            if (self.hparams.loss_low_res if hasattr(self.hparams, 'loss_low_res') else 1):
                loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
                # print('#############', loss, 6)
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
