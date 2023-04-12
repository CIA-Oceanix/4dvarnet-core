""" Lit model for forecasting applications """

import torch
import pandas as pd
import numpy as np
import solver as NN_4DVar
from metrics import rmse_based_scores, psd_based_scores
from lit_model_augstate import get_constant_crop
from lit_model_OI import LitModelOI
from copy import deepcopy
from pathlib import Path
import xarray as xr
from hydra.utils import instantiate


def get_forecast_crop(patch_size, crop):
    """
        Get the crop for forecasting applications
        input:
            patch_size: [time, latitude, longitute]
            crop: [time, latitude, longitute]
        return:
            patch_weight: time: 0 for past value, 1 for future value
    """
    patch_weight = get_constant_crop(patch_size, crop)
    time_patch_weight = np.concatenate((np.zeros(
        (patch_size['time'] - 1) // 2), np.ones(
            (patch_size['time'] + 1) // 2)),
                                       axis=0)
    print(patch_size, crop)
    final_patch_weight = time_patch_weight[:, None, None] * patch_weight
    return final_patch_weight


def get_forecast_interp_triangle_crop(patch_size, crop):
    """
        Get the crop for forecasting applications with interpolation
        input:
            patch_size: [time, latitude, longitute]
            crop: [time, latitude, longitute]
        return:
            patch_weight: Linear weight from 0 to 1 for the interpolation
                          Linear weight from 1 to 0.5 for the forecast
    """
    patch_weight = get_constant_crop(patch_size, crop)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1, (patch_size['time'] - 1) // 2),
         np.linspace(1, 0.5, (patch_size['time'] + 1) // 2)),
        axis=0)
    print(patch_size, crop)
    final_patch_weight = time_patch_weight[:, None, None] * patch_weight
    return final_patch_weight


def get_forecast_interp_triangle_crop_asym(patch_size, crop):
    """
        Get the crop for forecasting applications with interpolation
        input:
            patch_size: [time, latitude, longitute]
            crop: [time, latitude, longitute]
        return:
            patch_weight: Linear weight from 0 to 1 for the interpolation
                          Linear weight from 1 to 0.5 for the forecast
                          Only forecast 7 days
    """
    patch_weight = get_constant_crop(patch_size, crop)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1,
                     (patch_size['time'] - 1) // 2), np.linspace(1, 0.5, 7),
         (np.zeros((patch_size['time'] + 1) // 2 - 7))),
        axis=0)
    print(patch_size, crop)
    final_patch_weight = time_patch_weight[:, None, None] * patch_weight
    return final_patch_weight


def get_forecast_triangle_interp_constant_crop(patch_size, crop):
    """
        Get the crop for forecasting applications with interpolation
        input:
            patch_size: [time, latitude, longitute]
            crop: [time, latitude, longitute]
        return:
            patch_weight: Linear weight from 0 to 1 for the interpolation
                          Constant weight to 1 for the forecast
                          7 days of forecast
    """
    patch_weight = get_constant_crop(patch_size, crop)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1, (patch_size['time'] - 1) // 2), np.ones(7),
         (np.zeros((patch_size['time'] + 1) // 2 - 7))),
        axis=0)
    print(patch_size, crop)
    final_patch_weight = time_patch_weight[:, None, None] * patch_weight
    return final_patch_weight


def accurate_pred(array, threshold=0.9):
    """
        Find the number of days of good prediction according
        to a threshold for each day of a validation period
        input:
            array: [nb_days_predictions x nb_days_validation]
            threshold: float between 0 and 1
        return:
            value: all the different value of the longest
            accurate prediction sorted
            count: number of times each value appears
    """
    counter = np.zeros_like(array[0])
    array_accurate_pred = np.zeros_like(array[0])
    array_threshold = array >= threshold
    for preds in array_threshold:
        for id_day in range(len(preds)):
            if preds[id_day]:
                counter[id_day] += 1
            else:
                array_accurate_pred[id_day] = max(array_accurate_pred[id_day],
                                                  counter[id_day])
                # We only want series that start at the beginning
                # -100 is an arbitrary value
                counter[id_day] = -100
    for id_value in range(len(counter)):
        array_accurate_pred[id_value] = max(array_accurate_pred[id_value],
                                            counter[id_value])
    return np.unique(array_accurate_pred, return_counts=True)


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
            dict_md = []
            for test_xr_dses in self.test_xr_ds_list:
                # Reconstructions metrics
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
                dict_md.append(dict_md_temp)
            print(
                pd.DataFrame(dict_md,
                             range(-((self.hparams.dT - 1) // 2 - 1),
                                   7)).T.to_markdown())
            dict_md = dict_md[0]
            rmse_t_list = np.array(rmse_t_list)
            rmse_t_list = rmse_t_list[((self.hparams.dT - 1) // 2 - 1)::, :]
            values_days_90, count_days_90 = accurate_pred(rmse_t_list, 0.90)
            values_days_75, count_days_75 = accurate_pred(rmse_t_list, 0.75)
            values_days_50, count_days_50 = accurate_pred(rmse_t_list, 0.50)
            print(f'{values_days_90=}')
            print(f'{count_days_90=}')
            print(f'{values_days_75=}')
            print(f'{count_days_75=}')
            print(f'{values_days_50=}')
            print(f'{count_days_50=}')
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

    def build_test_xr_ds(self, outputs, diag_ds):

        outputs_keys = list(outputs[0][0].keys())
        with diag_ds.get_coords():
            self.test_patch_coords = [diag_ds[i] for i in range(len(diag_ds))]

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield tuple(
                            [outputs[bc][b][k][i] for k in outputs_keys])

        dses = [
            xr.Dataset(
                {
                    k: (('time', 'lat', 'lon'), x_k)
                    for k, x_k in zip(outputs_keys, xs)
                },
                coords=coords)
            for xs, coords in zip(iter_item(outputs), self.test_patch_coords)
        ]

        fin_ds = xr.merge(
            [xr.zeros_like(ds[['time', 'lat', 'lon']]) for ds in dses])
        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values())))})
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values())))})

        for ds in dses:
            ds_nans = ds.assign(weight=xr.ones_like(
                ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)
            _ds = ds.assign(weight=xr.ones_like(ds.gt)).broadcast_like(
                fin_ds).fillna(0.).where(ds_nans == 0, np.nan)
            fin_ds = fin_ds + _ds

        return ((fin_ds.drop('weight') / fin_ds.weight).sel(
            instantiate(self.test_domain)).isel(
                time=slice(self.hparams.dT // 2, -self.hparams.dT //
                           2))).transpose('time', 'lat', 'lon')

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
            dims = full_outputs[0][0]['gt'].shape
            dim_lat, dim_lon = dims[2:4]
            self.test_xr_ds_list = []
            for i in range(-((self.hparams.dT - 1) // 2 - 1), 7):
                small_patch_weight = np.concatenate(
                    (np.zeros((self.hparams.dT // 2 + i, dim_lat, dim_lon)),
                     np.ones((1, dim_lat, dim_lon)),
                     np.zeros((self.hparams.dT // 2 - i, dim_lat, dim_lon))),
                    axis=0)
                outputs_reduced = deepcopy(full_outputs)
                for gpu_rank in range(len(full_outputs)):
                    for batch_nb in range(len(full_outputs[0])):
                        outputs_reduced[gpu_rank][batch_nb][
                            'pred'] *= small_patch_weight
                        outputs_reduced[gpu_rank][batch_nb][
                            'gt'] *= small_patch_weight
                        outputs_reduced[gpu_rank][batch_nb][
                            'oi'] *= small_patch_weight
                self.test_xr_ds_list.append(
                    self.build_test_xr_ds(outputs_reduced, diag_ds=diag_ds))
            self.test_xr_ds = self.test_xr_ds_list[(
                (self.hparams.dT - 1) // 2 - 2)]
            Path(self.logger.log_dir).mkdir(exist_ok=True)
            for i in range(len(self.test_xr_ds_list)):
                path_save1 = self.logger.log_dir + f'/test_{i:02}.nc'
                self.test_xr_ds_list[i].to_netcdf(path_save1)
        else:
            self.test_xr_ds = self.build_test_xr_ds(full_outputs,
                                                    diag_ds=diag_ds)

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
        # _, _, _, targets_gt = batch
        # prod = targets_gt
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
