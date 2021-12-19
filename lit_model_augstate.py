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
from metrics import nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps
from models import Phi_r, Model_H, Model_HwithSST, ModelLR, Gradient_img

def get_4dvarnet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(hparams.shapeData[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                  hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
            Model_H(hparams.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(hparams.shapeData, hparams.UsePriodicBoundary,
                                          hparams.dim_grad_solver, hparams.dropout,
                                          hparams.stochastic),
            None, None, hparams.shapeData, hparams.n_grad, hparams.stochastic)

def get_4dvarnet_sst(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(hparams.shapeData[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
            Model_HwithSST(hparams.shapeData[0], hparams.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(hparams.shapeData, hparams.UsePriodicBoundary,
                hparams.dim_grad_solver, hparams.dropout),
            hparams.norm_obs, hparams.norm_prior, hparams.shapeData, hparams.n_grad)

def get_phi(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = Phi_r(hparams.shapeData[0], hparams.DimAE, hparams.dW,
                            hparams.dW2, hparams.sS, hparams.nbBlocks,
                            hparams.dropout_phi_r, hparams.stochastic)

            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks):
            return self.phi(state), None, None, None

    return PhiPassThrough()
############################################ Lightning Module #######################################################################

class LitModelAugState(pl.LightningModule):

    MODELS = {
            '4dvarnet': get_4dvarnet,
            '4dvarnet_sst': get_4dvarnet_sst,
            'phi': get_phi,
        }

    def __init__(self, hparam, *args, **kwargs):
        super().__init__()
        hparam = {} if hparam is None else hparam
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)

        # self.save_hyperparameters({**hparams, **kwargs})
        self.save_hyperparameters(hparams)
        # TOTEST: set those parameters only if provided
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.mean_Val = kwargs['mean_Val']
        self.mean_Tr = kwargs['mean_Tr']
        self.mean_Tt = kwargs['mean_Tt']
        # self.var_Val = self.hparams.var_Val
        # self.var_Tr = self.hparams.var_Tr
        # self.var_Tt = self.hparams.var_Tt
        # self.mean_Val = self.hparams.mean_Val
        # self.mean_Tr = self.hparams.mean_Tr
        # self.mean_Tt = self.hparams.mean_Tt

        self.xmin = kwargs['min_lon']
        self.xmax = kwargs['max_lon']
        self.ymin = kwargs['min_lat']
        self.ymax = kwargs['max_lat']
        self.Nx = 200
        self.Ny = 200
        #self.Nx = int(((self.xmax-self.xmin)/.05)/self.hparams.resize_factor)
        #self.Ny = int(((self.ymax-self.ymin)/.05)/self.hparams.resize_factor)
        self.lon = np.linspace(self.xmin, self.xmax, self.Nx)
        self.lat = np.linspace(self.ymin, self.ymax, self.Ny)
        self.shapeData = [self.hparams.dT*2,self.Ny,self.Nx]
        self.ds_size_time = kwargs['ds_size_time']
        self.ds_size_lon = kwargs['ds_size_lon']
        self.ds_size_lat = kwargs['ds_size_lat']

        self.time = kwargs['time']

        self.dX = kwargs['dX']
        self.dY = kwargs['dY']
        self.swX = kwargs['swX']
        self.swY = kwargs['swY']
        self.lon_ext = (kwargs['coord_ext'])['lon_ext']
        self.lat_ext = (kwargs['coord_ext'])['lat_ext']

        # create longitudes & latitudes coordinates
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None

        # main model
        # self.model_name = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        self.model_name = '4dvarnet'
        # self.use_sst = self.hparams.model if hasattr(self.hparams, 'sst') else False
        self.use_sst = False
        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.gradient_img = Gradient_img()

        # loss weghing wrt time
        # self.w_loss = torch.nn.Parameter(torch.Tensor(self.hparams.w_loss), requires_grad=False)  # duplicate for automatic upload to gpu
        self.w_loss = torch.nn.Parameter(kwargs['w_loss'], requires_grad=False)  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.test_figs = {}

        self.automatic_optimization = self.hparams.automatic_optimization

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self):
        return 1

    def configure_optimizers(self):

        if self.model_name == '4dvarnet':
            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                    ], lr=0.)

            return optimizer
        elif self.model_name == '4dvarnet_sst':

            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                    ], lr=0.)

            return optimizer
        # else:
            # opt = optim.Adam(self.parameters(), lr=1e-4)
        # return {
            # 'optimizer': opt,
            # 'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=50,),
            # 'monitor': 'val_loss'
        # }
        return optimizer

    def on_epoch_start(self):
        self.model.n_grad = self.hparams.n_grad

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                    self.current_epoch, self.hparams.nb_grad_update[indx],
                    self.hparams.lr_update[indx]))

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
        loss, _, metrics = self.compute_loss(train_batch, phase='train')
        # losses, outs, metrics = self(train_batch, phase='train')
        if loss is None:
            print("None loss")
            return None
        # log step metric
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True)
        self.log("tr_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, _, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def on_test_epoch_start(self):
        self.test_coords = self.trainer.test_dataloaders[0].dataset.datasets[0].gt_ds.ds.coords
        self.test_ds_patch_size = self.trainer.test_dataloaders[0].dataset.datasets[0].gt_ds.ds_size
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].isel(time=slice(self.hparams.dT // 2, - self.hparams.dT // 2 + 1)).data

    def test_step(self, test_batch, batch_idx):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT = test_batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt = test_batch
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return {'gt'    : (targets_GT.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs'   : (inputs_obs.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr}
        # return {'gt'    : (targets_GT.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                # 'oi'    : (targets_OI.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                # 'inp_obs'    : (inputs_obs.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                # 'preds' : (out.detach().cpu().numpy() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def test_epoch_end(self, outputs):

        self.outputs = outputs
        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        obs = torch.cat([chunk['obs'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        # gt = np.concatenate([chunk['gt'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        # obs = np.concatenate([chunk['inp_obs'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        # oi = np.concatenate([chunk['oi'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])
        # pred = np.concatenate([chunk['preds'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs])

        ds_size = self.test_ds_patch_size

        gt, obs, oi, pred = map(
            lambda t: einops.rearrange(
                t,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
            ),
            [gt, obs, oi, pred])

        # gt, oi, pred, obs = map(
                # lambda t: einops.rearrange(
                    # t,
                    # '(lat_idx lon_idx t_idx )  win_lat win_lon -> t_idx  (lat_idx win_lat) (lon_idx win_lon)',
                    # t_idx=int(ds_size['time']),
                    # lat_idx=int(ds_size['lat']),
                    # lon_idx=int(ds_size['lon']),
                    # ),
                # [gt, obs, oi, pred])

        # keep only points of the original domain
        iX = np.where( (self.lon_ext>=self.xmin) & (self.lon_ext<self.xmax) )[0]
        iY = np.where( (self.lat_ext>=self.ymin) & (self.lat_ext<self.ymax) )[0]
        gt = (gt[:,:,iY,:])[:,:,:,iX]
        obs = (obs[:,:,iY,:])[:,:,:,iX]
        oi = (oi[:,:,iY,:])[:,:,:,iX]
        pred = (pred[:,:,iY,:])[:,:,:,iX]

        self.x_gt = gt[:, int(self.hparams.dT / 2), :, :]
        self.x_obs = obs[:, int(self.hparams.dT / 2), :, :]
        self.x_oi = oi[:, int(self.hparams.dT / 2), :, :]
        self.x_rec = pred[:, int(self.hparams.dT / 2), :, :]

        # self.x_gt = gt
        # self.x_obs = obs
        # self.x_oi = oi
        # self.x_rec = pred

        self.test_xr_ds = xr.Dataset(
                {
                    'gt': (('time', 'lat', 'lon'), self.x_gt),
                    'obs': (('time', 'lat', 'lon'), self.x_obs),
                    'oi': (('time', 'lat', 'lon'), self.x_oi),
                    'pred': (('time', 'lat', 'lon'), self.x_rec),
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
                  self.x_obs[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps(
                  self.x_gt[t_idx],
                self.x_obs[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure('Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure('Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        # animate maps
        if self.hparams.animate:
            path_save0 = self.logger.log_dir + '/animation.mp4'
            animate_maps(self.x_gt,
                    self.x_obs,
                    self.x_oi,
                    self.x_rec,
                    self.lon, self.lat, path_save0)
        # save NetCDF
        path_save1 = self.logger.log_dir + '/test.nc'
        # PENDING: replace hardcoded 60
        self.test_xr_ds.to_netcdf(path_save1)
        # save_netcdf(saved_path1=path_save1, pred=self.x_rec,
                # lon=self.test_lon, lat=self.test_lat, time=self.test_dates, time_units=None)

        # compute nRMSE
        path_save2 = self.logger.log_dir + '/nRMSE.txt'
        tab_scores = nrmse_scores(gt, oi, pred, path_save2)
        print('*** Display nRMSE scores ***')
        print(tab_scores)

        path_save21 = self.logger.log_dir + '/MSE.txt'
        tab_scores = mse_scores(gt, oi, pred, path_save21)
        print('*** Display MSE scores ***')
        print(tab_scores)

        # plot nRMSE
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,  self.x_oi, self.x_rec, path_save3, time=self.test_dates)
        self.test_figs['nrmse'] = nrmse_fig

        # plot MSE
        path_save31 = self.logger.log_dir + '/MSE.png'
        mse_fig = plot_mse(self.x_gt, self.x_oi, self.x_rec, path_save31, time=self.test_dates)
        self.test_figs['mse'] = mse_fig
        self.logger.experiment.add_figure('Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure('NRMSE', nrmse_fig, global_step=self.current_epoch)
        self.logger.experiment.add_figure('MSE', mse_fig, global_step=self.current_epoch)

        # plot SNR
        path_save4 = self.logger.log_dir + '/SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_oi, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig
        self.logger.experiment.add_figure('SNR', snr_fig, global_step=self.current_epoch)

    def compute_loss(self, batch, phase):
        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_GT),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                        ('mseOI', 0.),
                        ('mseGOI', 0.)])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        if not self.hparams.augment_state:
            state = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI)), dim=1)
        else:
            state = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI), torch.zeros_like(targets_OI)), dim=1)
        # state = torch.cat((targets_OI, inputs_obs), dim=1)
        # state = torch.cat((targets_OI, inputs_Mask * (targets_GT_wo_nan - targets_OI)), dim=1)
        if not self.use_sst:
            if not self.hparams.augment_state:
                new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask), dim=1)
                inputs_obs = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI)), dim=1)
            else:
                # new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask), dim=1)
                # inputs_obs = torch.cat((targets_OI, inputs_obs), dim=1)
                new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask, torch.zeros_like(inputs_Mask)), dim=1)
                inputs_obs = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI), torch.zeros_like(targets_OI)), dim=1)
        else:
            new_masks = [
                    torch.cat((torch.ones_like(inputs_Mask), inputs_Mask, torch.zeros_like(inputs_Mask)), dim=1),
                    torch.ones_like(sst_gt)
            ]
            inputs_obs = [
                    torch.cat((targets_OI, inputs_obs, torch.zeros_like(targets_OI)), dim=1),
                    sst_gt
            ]

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            state = torch.autograd.Variable(state, requires_grad=True)
            outputs, hidden_new, cell_new, normgrad = self.model(state, inputs_obs, new_masks)

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR = outputs[:, 0:self.hparams.dT, :, :]
            if not self.hparams.augment_state:
                outputs = outputsSLR + outputs[:, self.hparams.dT:2*self.hparams.dT, :, :]
            else:
                outputs = outputsSLR + outputs[:, 2*self.hparams.dT:, :, :]

            g_outputs = self.gradient_img(outputs)

            loss_All = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.w_loss)
            loss_GAll = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.w_loss)
            loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)

            if not self.hparams.augment_state:
                # yGT = torch.cat((targets_OI, targets_GT - outputsSLR), dim=1)
                yGT = torch.cat((targets_GT_wo_nan, outputsSLR - targets_GT_wo_nan), dim=1)
            else:
                # yGT = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
                yGT = torch.cat((targets_GT_wo_nan, outputsSLR - targets_GT_wo_nan, outputsSLR - targets_GT_wo_nan), dim=1)

            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.w_loss)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR

            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT, self.w_loss)
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            metrics = dict([
                ('mse', mse),
                ('mseGrad', mseGrad),
                ('meanGrad', mean_GAll),
                ('mseOI', loss_OI.detach()),
                ('mseGOI', loss_GOI.detach())])
            # PENDING: Add new loss term to metrics

        return loss, outputs, metrics
