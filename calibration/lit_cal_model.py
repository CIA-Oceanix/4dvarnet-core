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

import lit_model_augstate
class LitCalModel(lit_model_augstate.LitModelAugstate):


    MODELS = {
            'passthrough': get_passthrough,
            'vit': get_vit,
            '4dvarnet': get_4dvarnet,
            '4dvarnet_sst': get_4dvarnet_sst,
            'phi': get_phi,
        }

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

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
                'obs_gt'    : (obs_target_item.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (inputs_obs.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_pred'    : (out_pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def diag_epoch_end(self, outputs, log_pref='test'):
        full_outputs = self.gather_outputs(outputs, log_pref=log_pref)

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
        self.obs_gt = self.test_xr_ds.obs_gt.data
        self.obs_pred = self.test_xr_ds.obs_pred.data
        self.x_oi = self.test_xr_ds.oi.data
        self.x_rec = self.test_xr_ds.pred.data
        
        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data
        
        md = self.sla_diag(t_idx=3, log_pref=log_pref)
        self.logger.log_metrics(md, step=self.current_epoch)

        nrmse_swath_df = self.nrmse_fn('obs_pred', 'obs_inp', 'obs_gt')
        mse_swath_df = self.mse_fn('obs_pred', 'obs_inp', 'obs_gt')
        nrmse_swath_df.to_csv(self.logger.log_dir + f'/{log_pref} nRMSE_swath.txt')
        mse_swath_df.to_csv(self.logger.log_dir + f'/{log_pref} MSE_swath.txt')

        cal_mdf = pd.concat([
            nrmse_swath_df.rename(columns=lambda c: f'{log_pref}_{c}_swath').loc['obs_pred'].T,
            mse_swath_df.rename(columns=lambda c: f'{log_pref}_{c}_swath').loc['obs_pred'].T,
        ])
        md = {
            **md,
            **cal_mdf.to_dict(), 
        }
        mdf = pd.Series(md)
        metrics_path =Path(self.logger.log_dir) / 'metrics'
        metrics_path.mkdir(exist_ok=True)
        mdf.to_json(metrics_path / f'{log_pref}.json')

        self.latest_metrics.update(md)
        print(mdf.T.to_markdown())
        self.logger.log_metrics(md ,step=self.current_epoch)

        
    def configure_callbacks(self):
        def save_ckpt_metrics(trainer, pl_module, checkpoint):
            mdf = pd.Series({
                **pl_module.latest_metrics,
                **checkpoint['hyper_parameters'],
                'epoch': checkpoint['epoch'],
                })
            metrics_path = Path(pl_module.logger.log_dir) / 'metrics'
            metrics_path.mkdir(exist_ok=True)
            mdf.to_json(metrics_path / f'ckpt_{checkpoint["epoch"]}.json', default_handler=str)                                                                                                                                                       
        return [pl.callbacks.LambdaCallback(on_save_checkpoint=save_ckpt_metrics)]

    def teardown(self, stage='test'):
        if self.logger is None:
            return
        self.logger.log_hyperparams(
                {**self.hparams},
                self.latest_metrics
        )

    def get_init_state(self, batch, state):
        if state[0] is not None:
            return state[0]

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch

        anomaly_global = torch.zeros_like(targets_OI)

        anomaly_swath = torch.zeros_like(targets_OI)
        init_state = torch.cat((targets_OI, anomaly_global, anomaly_swath), dim=1)
        if self.aug_state:
            init_state = torch.cat((init_state, inputs_Mask * (inputs_obs - targets_OI)), dim=1)
            if self.aug_state == 2:
                init_state = torch.cat((init_state, inputs_Mask * (inputs_obs - targets_OI)), dim=1)
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

        output_swath = output_low_res + output_anom_swath

        return output_low_res, output_global, output_swath

    def loss_ae(self, state_out):
        return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def cal_loss(self, gt, out, oi):
        msk = gt.isfinite()
        gt_wo_nan =  gt.where(msk, oi)

        gt_x, gt_y = self.gradient_img(gt)
        msk_gt_x, msk_gt_y = gt_x.isfinite(), gt_y.isfinite()
        gt_x, gt_y = self.gradient_img(gt_wo_nan)
        out_x, out_y = self.gradient_img(out)

         
        err = (out - gt_wo_nan).where(msk, torch.zeros_like(out))

        err_g_x = (out_x - gt_x).where(msk_gt_x, torch.zeros_like(gt_x))
        err_g_y = (out_y - gt_y).where(msk_gt_y, torch.zeros_like(gt_y))

        loss_swath = NN_4DVar.compute_spatio_temp_weighted_loss(err, self.patch_weight)
        loss_grad_swath = (NN_4DVar.compute_spatio_temp_weighted_loss(err_g_x, self.grad_crop(self.patch_weight))
                + NN_4DVar.compute_spatio_temp_weighted_loss(err_g_y, self.grad_crop(self.patch_weight)))

        return loss_swath, loss_grad_swath

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
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        target_obs_GT_wo_nan = target_obs_GT.where(~target_obs_GT.isnan(), targets_OI)

        state = self.get_init_state(batch, state_init)

        #state = torch.cat((targets_OI, inputs_Mask * (targets_GT_wo_nan - targets_OI)), dim=1)
        new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask), dim=1)
        obs = torch.cat((targets_OI, inputs_obs), dim=1)
        if  self.use_sst:
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]


        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            # PENDING: reconstruct outputs, outputs LowRes and outputSwath MegaRes

            if self.hparams.swot_anom_wrt == 'low_res':
                gt_anom_swath = targets_OI
            elif self.hparams.swot_anom_wrt == 'high_res':
                gt_anom_swath = targets_GT

            output_low_res, output_global, output_swath = self.get_outputs(batch, outputs)
            # reconstruction losses



            # projection losses

            glob_anom_gt = targets_GT_wo_nan - output_low_res
            loc_anom_gt = (target_obs_GT_wo_nan - output_low_res).where(target_obs_GT.isfinite(), glob_anom_gt)
            yGT = torch.cat((targets_OI,  glob_anom_gt, loc_anom_gt), dim=1)
            if self.aug_state:
                yGT = torch.cat((yGT, glob_anom_gt), dim=1) 
                if self.aug_state==2:
                    yGT = torch.cat((yGT, loc_anom_gt), dim=1)



            loss_All, loss_GAll = self.sla_loss(output_global, targets_GT_wo_nan)
            loss_OI, loss_GOI = self.sla_loss(targets_OI, targets_GT_wo_nan)
            loss_AE, loss_AE_GT, loss_SR, loss_LR =  self.reg_loss(yGT, targets_OI, output_global, output_low_res, outputs) 
            loss_swath, loss_grad_swath =  self.cal_loss(target_obs_GT, output_swath, targets_OI) 
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
            g_targets_GT = torch.hypot(*self.gradient_img(targets_GT))
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.grad_crop(self.patch_weight))
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

        return loss, outputs, [outputs, hidden_new, cell_new, normgrad], metrics

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
