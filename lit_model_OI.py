""" Lit model for interpolation without the OI in initialisation """

from pathlib import Path
import pandas as pd
from hydra.utils import call
import numpy as np
import torch
import solver as NN_4DVar
from metrics import psd_based_scores, rmse_based_scores, plot_psd_score
from models import Model_H, Phi_r_OI

from lit_model_augstate import LitModelAugstate


def get_4dvarnet_oi(hparams):
    """
        Retrieve the 4dvarnet model adapted for experiences
        without the OI in the initialisation
    """
    return NN_4DVar.Solver_Grad_4DVarNN(
        Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW,
                 hparams.dW2, hparams.sS, hparams.nbBlocks,
                 hparams.dropout_phi_r, hparams.stochastic),
        Model_H(hparams.shape_state[0]),
        NN_4DVar.model_GradUpdateLSTM(hparams.shape_state,
                                      hparams.UsePriodicBoundary,
                                      hparams.dim_grad_solver,
                                      hparams.dropout), hparams.norm_obs,
        hparams.norm_prior, hparams.shape_state,
        hparams.n_grad * hparams.n_fourdvar_iter)


class LitModelOI(LitModelAugstate):
    MODELS = dict(LitModelAugstate.MODELS)
    MODELS.update({'4dvarnet_OI': get_4dvarnet_oi})

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):

            def opt(p):
                call(self.hparams.opt, p)

            # opt = lambda p: call(self.hparams.opt, p)
        if self.model_name == '4dvarnet_OI':
            optimizer = opt([
                {
                    'params': self.model.model_Grad.parameters(),
                    'lr': self.hparams.lr_update[0]
                },
                {
                    'params': self.model.model_VarCost.parameters(),
                    'lr': self.hparams.lr_update[0]
                },
                {
                    'params': self.model.model_H.parameters(),
                    'lr': self.hparams.lr_update[0]
                },
                {
                    'params': self.model.phi_r.parameters(),
                    'lr': 0.5 * self.hparams.lr_update[0]
                },
            ])
        else:
            optimizer = opt(self.parameters(), lr=1e-4)

        return optimizer

    def diag_step(self, batch, batch_idx, log_pref='test'):
        targets_oi, inputs_mask, inputs_obs, targets_gt = batch
        losses, out, metric = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse',
                     metric[-1]["mse"] / self.var_Tt,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True)
            self.log(f'{log_pref}_mseG',
                     metric[-1]['mseGrad'] / metric[-1]['meanGrad'],
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True)

        return {
            'gt':
            (targets_gt.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
            'oi':
            (targets_oi.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
            'obs_inp': (inputs_obs.detach().where(
                inputs_mask, torch.full_like(inputs_obs, np.nan)).cpu() *
                        np.sqrt(self.var_Tr)) + self.mean_Tr,
            'pred': (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr
        }

    def sla_diag(self, t_idx=3, log_pref='test'):

        # path_save0 = self.logger.log_dir + '/maps.png'
        # fig_maps = plot_maps_oi(self.x_gt[t_idx], self.obs_inp[t_idx],
        #                         self.x_rec[t_idx], self.test_lon,
        #                         self.test_lat, path_save0)
        # path_save01 = self.logger.log_dir + '/maps_Grad.png'
        # fig_maps_grad = plot_maps_oi(self.x_gt[t_idx],
        #                              self.obs_inp[t_idx],
        #                              self.x_rec[t_idx],
        #                              self.test_lon,
        #                              self.test_lat,
        #                              path_save01,
        #                              grad=True)
        # self.test_figs['maps'] = fig_maps
        # self.test_figs['maps_grad'] = fig_maps_grad
        # self.logger.experiment.add_figure(f'{log_pref} Maps',
        #                                   fig_maps,
        #                                   global_step=self.current_epoch)
        # self.logger.experiment.add_figure(f'{log_pref} Maps Grad',
        #                                   fig_maps_grad,
        #                                   global_step=self.current_epoch)

        psd_ds, lamb_x, lamb_t = psd_based_scores(self.test_xr_ds.pred,
                                                  self.test_xr_ds.gt)
        psd_fig = plot_psd_score(psd_ds)
        self.test_figs['psd'] = psd_fig
        self.logger.experiment.add_figure(f'{log_pref} PSD',
                                          psd_fig,
                                          global_step=self.current_epoch)
        _, _, mean_mu, sig = rmse_based_scores(self.test_xr_ds.pred,
                                               self.test_xr_ds.gt)

        dict_md = {
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mean_mu,
            f'{log_pref}_sigma': sig,
        }
        print(pd.DataFrame([dict_md]).T.to_markdown())
        return dict_md

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
        path_save1 = self.logger.log_dir + '/test.nc'
        self.test_xr_ds.to_netcdf(path_save1)

        self.x_gt = self.test_xr_ds.gt.data
        self.obs_inp = self.test_xr_ds.obs_inp.data
        self.x_oi = self.test_xr_ds.oi.data
        self.x_rec = self.test_xr_ds.pred.data
        self.x_rec_ssh = self.x_rec

        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data

        # display map
        dict_md = self.sla_diag(t_idx=3, log_pref=log_pref)
        self.latest_metrics.update(dict_md)
        self.logger.log_metrics(dict_md, step=self.current_epoch)

    def get_init_state(self, batch, state=(None, )):
        """ Create init state for the compute loss function. """
        if state[0] is not None:
            return state[0]

        _, inputs_mask, inputs_obs, _ = batch

        init_state = inputs_mask * inputs_obs
        return init_state

    def get_obs_state(self, batch):
        """ Create obs state for the compute loss function. """
        _, inputs_mask, inputs_obs, _ = batch
        obs = inputs_mask * inputs_obs
        return obs

    def compute_loss(self, batch, phase, state_init=(None, )):

        _, inputs_mask, _, targets_gt = batch

        # handle patch with no observation
        if inputs_mask.sum().item() == 0:
            return (None, torch.zeros_like(targets_gt),
                    torch.cat((torch.zeros_like(targets_gt),
                               torch.zeros_like(targets_gt),
                               torch.zeros_like(targets_gt)),
                              dim=1),
                    dict([
                        ('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                    ]))
        targets_gt_wo_nan = targets_gt.where(~targets_gt.isnan(),
                                             torch.zeros_like(targets_gt))

        state = self.get_init_state(batch, state_init)

        obs = self.get_obs_state(batch)
        new_masks = inputs_mask

        # gradient norm field
        g_targets_gt_x, g_targets_gt_y = self.gradient_img(targets_gt)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(
                state, obs, new_masks, *state_init[1:])

            if phase in ('val', 'test'):
                outputs = outputs.detach()

            loss_all, loss_gall = self.sla_loss(outputs, targets_gt_wo_nan)
            loss_ae = self.loss_ae(outputs)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_all \
                + self.hparams.alpha_mse_gssh * loss_gall
            loss += 0.5 * self.hparams.alpha_proj * loss_ae

            # metrics
            # mean_gall = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_gt, self.w_loss)
            mean_gall = NN_4DVar.compute_spatio_temp_weighted_loss(
                torch.hypot(g_targets_gt_x, g_targets_gt_y),
                self.grad_crop(self.patch_weight))
            mse = loss_all.detach()
            mse_grad = loss_gall.detach()
            metrics_dict = dict([
                ('mse', mse),
                ('mseGrad', mse_grad),
                ('meanGrad', mean_gall),
            ])

        return loss, outputs, [outputs, hidden_new, cell_new,
                               normgrad], metrics_dict
