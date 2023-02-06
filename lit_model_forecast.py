""" Lit model forecast """

import torch
import pandas as pd
import numpy as np
import solver as NN_4DVar
from metrics import rmse_based_scores, psd_based_scores
from lit_model_augstate import get_constant_crop
from lit_model_OI import LitModelOI
from copy import deepcopy


def get_forecast_crop(patch_size, crop):
    """
        Get the crop for forecasting applications
        input:
            patch_size: [time, latitude, longitute]
            crop: [time, latitude, longitute]
        return:
            patch_weight: time : 0 for past value, 1 for future value
    """
    patch_weight = get_constant_crop(patch_size, crop)
    time_patch_weight = np.concatenate((np.zeros(
        (patch_size['time'] - 1) // 2), np.ones(
            (patch_size['time'] + 1) // 2)),
                                       axis=0)
    print(patch_size, crop)
    final_patch_weight = time_patch_weight[:, None, None] * patch_weight
    return final_patch_weight


def longest_accurate_pred(array, threshold=0.90):
    """
        Find the longest accurate prediction according to a threshold
        for each day of a validation period
        input:
            array: [nb_days_predictions x nb_days_validation]
            threshold: float between 0 and 1
        return:
            value: all the different value of the longest
            accurate prediction sorted
            count: number of times each value appears
    """
    counter = np.zeros_like(array[0])
    array_longest_accurate_pred = np.zeros_like(array[0])
    array_threshold = array >= threshold
    for preds in array_threshold:
        for id_day in range(len(preds)):
            if preds[id_day]:
                counter[id_day] += 1
            else:
                array_longest_accurate_pred[id_day] = max(
                    array_longest_accurate_pred[id_day], counter[id_day])
                counter[id_day] = 0
    for id_value in range(len(counter)):
        array_longest_accurate_pred[id_value] = max(
            array_longest_accurate_pred[id_value], counter[id_value])
    return np.unique(array_longest_accurate_pred, return_counts=True)


class LitModelForecast(LitModelOI):
    """
        Class for the training of the 4dvarnet for forecasting applications
        inherit from LitModelOI class
    """

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

        if log_pref == 'test':
            rmse_t_list = []
            for test_xr_dses in self.test_xr_ds_list:
                psd_ds, lamb_x, lamb_t = psd_based_scores(
                    test_xr_dses.pred, test_xr_dses.gt)
                rmse_t, _, mean_mu, sig = rmse_based_scores(
                    test_xr_dses.pred, test_xr_dses.gt)
                rmse_t_list.append(rmse_t)
                dict_md_temp = {
                    f'{log_pref}_lambda_x': lamb_x,
                    f'{log_pref}_lambda_t': lamb_t,
                    f'{log_pref}_mu': mean_mu,
                    f'{log_pref}_sigma': sig,
                }
                # print(f'{rmse_t>0.90=}')
                # find the number of consecutive days where
                # the error is below a threshold
                # dict_nb_days = {
                #     f'{log_pref}_nb_days_90': nb_days_90,
                #     f'{log_pref}_nb_days_75': nb_days_75,
                #     f'{log_pref}_nb_days_50': nb_days_50,
                # }
                print(pd.DataFrame([dict_md_temp]).T.to_markdown())
                # print(pd.DataFrame([dict_nb_days]).T.to_markdown())
            rmse_t_list = np.array(rmse_t_list)
            rmse_t_list = rmse_t_list[3::, :]
            values_days_90, count_days_90 = longest_accurate_pred(
                rmse_t_list, 0.90)
            values_days_75, count_days_75 = longest_accurate_pred(
                rmse_t_list, 0.75)
            values_days_50, count_days_50 = longest_accurate_pred(
                rmse_t_list, 0.50)
            print(f'{values_days_90=}')
            print(f'{count_days_90=}')
            print(f'{values_days_75=}')
            print(f'{count_days_75=}')
            print(f'{values_days_50=}')
            print(f'{count_days_50=}')
            dict_md = dict_md_temp
        else:
            psd_ds, lamb_x, lamb_t = psd_based_scores(self.test_xr_ds.pred,
                                                      self.test_xr_ds.gt)
            # psd_fig = plot_psd_score(psd_ds)
            # self.test_figs['psd'] = psd_fig
            # self.logger.experiment.add_figure(f'{log_pref} PSD',
            #                                   psd_fig,
            #                                   global_step=self.current_epoch)
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
        # For the test build a [number of days of predictions x time x dims]
        # to estimate metrics
        if log_pref == 'test':
            self.test_xr_ds_list = []
            for i in range(-3, 4):
                small_patch_weight = np.concatenate(
                    (np.zeros((self.hparams.dT // 2 + i, 240, 240)),
                     np.ones((1, 240, 240)),
                     np.zeros((self.hparams.dT // 2 - i, 240, 240))),
                    axis=0)
                outputs_reduced = deepcopy(full_outputs)
                for gpu_rank in range(len(full_outputs)):
                    for batch_nb in range(len(full_outputs[0])):
                        outputs_reduced[gpu_rank][batch_nb][
                            'pred'] *= small_patch_weight
                        outputs_reduced[gpu_rank][batch_nb][
                            'gt'] *= small_patch_weight
                # outputs_reduced = list(
                #     map(
                #         lambda y: map(
                #             lambda x: x.get("pred") * small_patch_weight, y),
                #         full_outputs))
                self.test_xr_ds_list.append(
                    self.build_test_xr_ds(outputs_reduced, diag_ds=diag_ds))
            self.test_xr_ds = self.test_xr_ds_list[0]
        else:
            self.test_xr_ds = self.build_test_xr_ds(full_outputs,
                                                    diag_ds=diag_ds)

        # Path(self.logger.log_dir).mkdir(exist_ok=True)
        # path_save1 = self.logger.log_dir + '/test.nc'
        # self.test_xr_ds.to_netcdf(path_save1)

        self.x_gt = self.test_xr_ds.gt.data
        self.obs_inp = self.test_xr_ds.obs_inp.data
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

        prod = inputs_mask * inputs_obs
        init_state = torch.cat((
            prod[:, 0:(self.hparams.dT - 1) // 2, :, :],
            torch.zeros_like(
                prod[:, (self.hparams.dT - 1) // 2:self.hparams.dT, :, :]),
        ),
                               dim=1)
        return init_state

    def get_obs_state(self, batch):
        """
            Create obs state for the compute loss function.
            Use the obs of t=1 ... (dT-1)/2 to forecast t=(dT-1)/2+1 ... dT
        """
        _, inputs_mask, inputs_obs, _ = batch
        prod = inputs_mask * inputs_obs
        obs = torch.cat((
            prod[:, 0:(self.hparams.dT - 1) // 2, :, :],
            torch.zeros_like(
                prod[:, (self.hparams.dT - 1) // 2:self.hparams.dT, :, :]),
        ),
                        dim=1)
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

        # need to evaluate grad/backward during the evaluation
        # and training phase for phi_r
        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(
                state, obs, new_masks, *state_init[1:])

            if phase in ('val', 'test'):
                outputs = outputs.detach()

            loss_all, loss_gall = self.sla_loss(outputs, targets_gt_wo_nan)
            loss_ae = self.loss_ae(outputs)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_all + \
                self.hparams.alpha_mse_gssh * loss_gall
            loss += 0.5 * self.hparams.alpha_proj * loss_ae

            # metrics
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
