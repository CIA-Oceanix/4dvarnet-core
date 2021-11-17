import einops
import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import solver as NN_4DVar
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, get_psd_score
from models import Phi_r, ModelLR, Gradient_img

from calibration.models import get_passthrough, get_vit

class Model_H_with_noisy_Swot(torch.nn.Module):
    """
    state: [oi, anom_glob, anom_swath ]
    obs: [oi, obs]
    mask: [ones, obs_mask]
    """
    def __init__(self, shape_state, shape_obs, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_obs])



    def forward(self, x, y, mask):
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
    

    def __init__(self, shape_state, shape_obs, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.dim_obs = 2
        sst_ch = hparams.dT
        self.dim_obs_channel = np.array([shape_state, sst_ch])

        self.conv11 = torch.nn.Conv2d(shape_state, hparams.dT, (3, 3), padding=1, bias=False)
        self.conv21 = torch.nn.Conv2d(sst_ch, hparams.dT, (3, 3), padding=1, bias=False)
        self.conv_m = torch.nn.Conv2d(sst_ch, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def forward(self, x, y, mask):
        y_ssh, y_sst = y
        mask_ssh, mask_sst = mask
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
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H_with_noisy_Swot(hparams.shape_state[0], hparams.shape_obs[0], hparams=hparams),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad)

def get_4dvarnet_sst(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H_SST_with_noisy_Swot(hparams.shape_state[0], hparams.shape_obs[0], hparams=hparams),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad)

def get_phi(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic)

            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks):
            return self.phi(state), None, None, None

    return PhiPassThrough()
############################################ Lightning Module #######################################################################

class LitCalModel(pl.LightningModule):


    MODELS = {
            'passthrough': get_passthrough,
            'vit': get_vit,
            '4dvarnet': get_4dvarnet,
            '4dvarnet_sst': get_4dvarnet_sst,
            'phi': get_phi,
        }

    def __init__(self, hparam=None,
                               min_lon=None, max_lon=None,
                               min_lat=None, max_lat=None,
                               ds_size_time=None,
                               ds_size_lon=None,
                               ds_size_lat=None,
                               time=None,
                               dX = None, dY = None,
                               swX = None, swY = None,
                               coord_ext = None, *args, **kwargs):
        super().__init__()
        hparam = {} if hparam is None else hparam
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)

        self.save_hyperparameters({**hparams, **kwargs})
        # TOTEST: set those parameters only if provided
        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt

        # create longitudes & latitudes coordinates
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None


        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt
        self.mean_Val = self.hparams.mean_Val
        self.mean_Tr = self.hparams.mean_Tr
        self.mean_Tt = self.hparams.mean_Tt

        # main model

        self.model_name = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        self.use_sst = self.hparams.model if hasattr(self.hparams, 'sst') else False
        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.gradient_img = Gradient_img()
        # loss weghing wrt time

        self.w_loss = torch.nn.Parameter(torch.Tensor(self.hparams.w_loss), requires_grad=False)  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.test_figs = {}

        self.automatic_optimization = self.hparams.automatic_optimization

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, outs, _metrics = self.compute_loss(batch, phase=phase, state_init=state_init)
            state_init = outs.detach()
            losses.append(_loss)
            metrics.append(_metrics)
        return losses, outs, metrics

    def configure_optimizers(self):
        
        if self.model_name == '4dvarnet':
            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ]
                , lr=0.)

            return optimizer
        elif self.model_name == '4dvarnet_sst':

            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0.)

            return optimizer
        else: 
            opt = optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': opt,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=50,),
            'monitor': 'val_loss'
        }

    def on_epoch_start(self):
        self.model.n_grad = self.hparams.n_grad

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                    self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

                self.hparams.n_grad = self.hparams.nb_grad_update[indx]
                self.model.n_grad = self.hparams.n_grad
                print("ngrad iter", self.model.n_grad)
                mm = 0
                lrCurrent = self.hparams.lr_update[indx]
                lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
                for pg in opt.param_groups:
                    pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                    mm += 1

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics    
        losses, outs, metrics = self(train_batch, phase='train')
        if losses[-1] is None:
            print("None loss")
            return None
        loss = torch.stack(losses).mean()
        # log step metric        
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True)
        self.log("tr_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mse_swath", metrics[-1]['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG_swath", metrics[-1]['mseGradSwath'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)


        return loss

    def validation_step(self, val_batch, batch_idx):
        losses, _, metrics = self(val_batch, phase='val')
        loss = torch.stack(losses).mean()
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics[-1]['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse_swath", metrics[-1]['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG_swath", metrics[-1]['mseGradSwath'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def on_test_epoch_start(self):
        self.test_coords = self.trainer.test_dataloaders[0].dataset.datasets[0].gt_ds.ds.coords
        self.test_ds_patch_size = self.trainer.test_dataloaders[0].dataset.datasets[0].gt_ds.ds_size
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].isel(time=slice(self.hparams.dT // 2, - self.hparams.dT // 2 + 1)).data

    def test_step(self, test_batch, batch_idx):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, obs_target_item = test_batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, obs_target_item = test_batch
        loss, outs, metrics = self(test_batch, phase='test')
        _, out, out_pred = self.get_outputs(test_batch, outs[-1])
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics[-1]['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics[-1]['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mse_swath", metrics[-1]['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG_swath", metrics[-1]['mseGradSwath'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return {'gt'    : (targets_GT.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'target_obs'    : (obs_target_item.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'inp_obs'    : (inputs_obs.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_pred'    : (out_pred.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def test_epoch_end(self, outputs):

        self.outputs = outputs
        gt = np.concatenate([chunk['gt'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        oi = np.concatenate([chunk['oi'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        pred = np.concatenate([chunk['preds'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        target_obs = np.concatenate([chunk['target_obs'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        inputs_obs = np.concatenate([chunk['inp_obs'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        obs_pred = np.concatenate([chunk['obs_pred'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])

        ds_size = self.test_ds_patch_size

        gt, oi, pred, inputs_obs, target_obs, obs_pred = map(
                lambda t: einops.rearrange(
                    t,
                    '(lat_idx lon_idx t_idx )  win_lat win_lon -> t_idx  (lat_idx win_lat) (lon_idx win_lon)',
                    t_idx=int(ds_size['time']),
                    lat_idx=int(ds_size['lat']),
                    lon_idx=int(ds_size['lon']),
                    ),
                [gt, oi, pred, inputs_obs, target_obs, obs_pred])

        self.x_gt = gt
        self.x_oi = oi
        self.x_rec = pred
        self.obs_gt = target_obs
        self.obs_inp = np.where(~np.isnan(self.obs_gt), inputs_obs, np.full_like(self.obs_gt, float('nan')))
        self.obs_pred = np.where(~np.isnan(self.obs_gt), obs_pred, np.full_like(self.obs_gt, float('nan')))

        self.test_xr_ds = xr.Dataset(
                {

                    'gt': (('time', 'lat', 'lon'), self.x_gt),
                    'oi': (('time', 'lat', 'lon'), self.x_oi),
                    'pred': (('time', 'lat', 'lon'), self.x_rec),
                    'obs_gt': (('time', 'lat', 'lon'), self.obs_gt),
                    'obs_pred': (('time', 'lat', 'lon'), self.obs_pred),
                    'obs_inp': (('time', 'lat', 'lon'), self.obs_inp),

                    },
                {
                    'time': self.test_coords['time'].isel(time=slice(self.hparams.dT // 2, - self.hparams.dT // 2 + 1)),
                    'lat': self.test_coords['lat'],
                    'lon': self.test_coords['lon'],
                    }
                )

        Path(self.logger.log_dir).mkdir(exist_ok=True)
        # display map

        path_save0 = self.logger.log_dir + '/maps.png'
        t_idx = 3
        fig_maps = plot_maps(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure('Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure('Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        path_save02 = self.logger.log_dir + '/maps_obs.png'
        fig_maps = plot_maps(
                self.obs_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_rec[t_idx],
                self.obs_pred[t_idx],
                self.test_lon, self.test_lat, path_save02, grad=True)
        self.test_figs['maps_obs'] = fig_maps
        self.logger.experiment.add_figure('Maps Obs', fig_maps, global_step=self.current_epoch)

        # animate maps
        if self.hparams.animate == True:
            path_save0 = self.logger.log_dir + '/animation.mp4'
            animate_maps(self.x_gt,
                    self.x_oi,
                    self.x_rec,
                    self.lon, self.lat, path_save0)
            # save NetCDF
        path_save1 = self.logger.log_dir + '/test.nc'
        # PENDING: replace hardcoded 60
        self.test_xr_ds.to_netcdf(path_save1)
        # save_netcdf(saved_path1=path_save1, pred=self.x_rec,
        #         lon=self.test_lon, lat=self.test_lat, time=self.test_dates, time_units=None)

        # compute nRMSE
        # np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)
        nrmse_fn = lambda pred, ref, gt: (
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - ds.mean())
                .pipe(lambda ds: ds - (self.test_xr_ds[gt].pipe(lambda da: da - da.mean())))
                .pipe(lambda ds: ds ** 2 / self.test_xr_ds[gt].std())
                .to_dataframe()
                .pipe(lambda ds: np.sqrt(ds.mean()))
                .to_frame()
                .rename(columns={0: 'nrmse'})
                .assign(nrmse_ratio=lambda df: df / df.loc[ref])
        )
        mse_fn = lambda pred, ref, gt: (
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - self.test_xr_ds[gt])
                .pipe(lambda ds: ds ** 2)
                .to_dataframe()
                .pipe(lambda ds: ds.mean())
                .to_frame()
                .rename(columns={0: 'mse'})
                .assign(mse_ratio=lambda df: df / df.loc[ref])
        )

        nrmse_df = nrmse_fn('pred', 'oi', 'gt')
        mse_df = mse_fn('pred', 'oi', 'gt')
        nrmse_df.to_csv(self.logger.log_dir + '/nRMSE.txt')
        mse_df.to_csv(self.logger.log_dir + '/MSE.txt')

        # compute nRMSE on swath
        path_save23 = self.logger.log_dir + '/nRMSE_swath.txt'

        nrmse_swath_df = nrmse_fn('obs_pred', 'obs_inp', 'obs_gt')
        mse_swath_df = mse_fn('obs_pred', 'obs_inp', 'obs_gt')
        nrmse_df.to_csv(self.logger.log_dir + '/nRMSE_swath.txt')
        mse_df.to_csv(self.logger.log_dir + '/MSE_swath.txt')
        
        print(
            pd.concat(
                [
                    pd.concat([nrmse_df, mse_df], axis=1,),
                    pd.concat([nrmse_swath_df, mse_swath_df], axis=1)
                ], axis=0,
            ).to_markdown()
        )
        # plot nRMSE
        # PENDING: replace hardcoded 60
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,  self.x_oi, self.x_rec, path_save3, time=self.test_dates)
        self.test_figs['nrmse'] = nrmse_fig
        self.logger.experiment.add_figure('NRMSE', nrmse_fig, global_step=self.current_epoch)
        # plot SNR
        path_save4 = self.logger.log_dir + '/SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_oi, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig

        self.logger.experiment.add_figure('SNR', snr_fig, global_step=self.current_epoch)
        
        fig, spatial_res_model, spatial_res_oi = get_psd_score(self.test_xr_ds.gt, self.test_xr_ds.pred, self.test_xr_ds.oi, with_fig=True)
        self.test_figs['res'] = fig
        self.logger.experiment.add_figure('Spat. Resol', fig, global_step=self.current_epoch)
        # PENDING: Compute metrics on swath
        mdf = pd.concat([
            nrmse_df.rename(columns=lambda c: f'{c}_glob').loc['pred'].T,
            mse_df.rename(columns=lambda c: f'{c}_glob').loc['pred'].T,
            nrmse_swath_df.rename(columns=lambda c: f'{c}_swath').loc['obs_pred'].T,
            mse_swath_df.rename(columns=lambda c: f'{c}_swath').loc['obs_pred'].T,
        ])
        print(mdf.to_frame().to_markdown())
        self.logger.log_hyperparams(
                {**self.hparams},
                {
                    'spatial_res': float(spatial_res_model),
                    'spatial_res_imp': float(spatial_res_model / spatial_res_oi),
                    **mdf.to_dict(), },
                )

    def get_init_state(self, batch, state):
        if state is not None:
            return state

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch

        anomaly_global = torch.zeros_like(targets_OI)

        if self.hparams.anom_swath_init == 'zeros':
            anomaly_swath = torch.zeros_like(targets_OI)
        elif self.hparams.anom_swath_init == 'obs':
            anomaly_swath = (inputs_obs - targets_OI).detach()

        return torch.cat((targets_OI, anomaly_global, anomaly_swath), dim=1)

    def get_outputs(self, batch, state_out):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, target_obs_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, target_obs_GT = batch
        output_low_res,  output_anom_glob, output_anom_swath = torch.split(state_out, split_size_or_sections=targets_OI.size(1), dim=1)
        output_global = output_low_res + output_anom_glob

        if self.hparams.swot_anom_wrt == 'low_res':
            output_swath = output_low_res + output_anom_swath
        elif self.hparams.swot_anom_wrt == 'high_res':
            output_swath = output_global + output_anom_swath

        return output_low_res, output_global, output_swath

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
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        target_obs_GT_wo_nan = target_obs_GT.where(~target_obs_GT.isnan(), torch.zeros_like(target_obs_GT))

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
        # PENDING: Add state with zeros

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

            # PENDING: reconstruct outputs, outputs LowRes and outputSwath MegaRes

            if self.hparams.swot_anom_wrt == 'low_res':
                gt_anom_swath = targets_OI
            elif self.hparams.swot_anom_wrt == 'high_res':
                gt_anom_swath = targets_GT

            output_low_res, output_global, output_swath = self.get_outputs(batch, outputs)
            # reconstruction losses
            g_output_global = self.gradient_img(output_global)
            g_output_swath = self.gradient_img(output_swath)
            # PENDING: add loss term computed on obs (outputs swath - obs_target)

            _err_swath =(output_swath - target_obs_GT_wo_nan)**2 
            err_swath = torch.where(target_obs_GT.isnan() | target_obs_GT.isinf(), torch.zeros_like(_err_swath), _err_swath)
            _err_g_swath =(g_output_swath - g_targets_obs)**2
            err_g_swath = torch.where(g_targets_obs_mask.isnan() | g_targets_obs_mask.isinf(), torch.zeros_like(_err_g_swath), _err_g_swath)

            loss_swath = NN_4DVar.compute_WeightedLoss(err_swath, self.w_loss)
            # print(f"{loss_swath=}")
            loss_grad_swath = NN_4DVar.compute_WeightedLoss(err_g_swath, self.w_loss)
            # print(f"{loss_grad_swath=}")

            loss_All = NN_4DVar.compute_WeightedLoss((output_global - targets_GT), self.w_loss)
            loss_GAll = NN_4DVar.compute_WeightedLoss(g_output_global - g_targets_GT, self.w_loss)
            loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = torch.mean((self.model.phi_r(outputs) - outputs) ** 2)

            yGT = torch.cat((targets_OI, targets_GT_wo_nan - targets_OI, target_obs_GT_wo_nan - gt_anom_swath), dim=1)

            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_WeightedLoss(output_low_res - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(output_global) - targets_GTLR, self.w_loss)

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
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT, self.w_loss)
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
