import einops
import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf

import solver as NN_4DVar
from metrics import save_netcdf, nrmse_scores, mse_scores, plot_nrmse, plot_snr, plot_maps, animate_maps, get_psd_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLinUnit(torch.nn.Module):
    def __init__(self, dimIn, dim, dW, dW2, dropout=0.):
        super(BiLinUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(dimIn, 2 * dim, (2 * dW + 1, 2 * dW + 1), padding=dW, bias=False)
        self.conv2 = torch.nn.Conv2d(2 * dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim, dimIn, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, xin):
        x = self.conv1(xin)
        x = self.dropout(x)
        x = self.conv2(F.relu(x))
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)), dim=1)
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, dimInp, dimAE, dW, dW2, sS, nbBlocks, rateDropout=0.):
        super(Encoder, self).__init__()

        self.NbBlocks = nbBlocks
        self.DimAE = dimAE
        # self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        # self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        self.pool1 = torch.nn.AvgPool2d(sS)
        self.convTr = torch.nn.ConvTranspose2d(dimInp, dimInp, (sS, sS), stride=(sS, sS), bias=False)

        # self.NNtLR    = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        # self.NNHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        self.NNLR = self.__make_BilinNN(dimInp, self.DimAE, dW, dW2, self.NbBlocks, rateDropout)
        self.NNHR = self.__make_BilinNN(dimInp, self.DimAE, dW, dW2, self.NbBlocks, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)

    def __make_BilinNN(self, dimInp, dimAE, dW, dW2, Nb_Blocks=2, dropout=0.):
        layers = []
        layers.append(BiLinUnit(dimInp, dimAE, dW, dW2, dropout))
        for kk in range(0, Nb_Blocks - 1):
            layers.append(BiLinUnit(dimAE, dimAE, dW, dW2, dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, xinp):
        ## LR comlponent
        xLR = self.NNLR(self.pool1(xinp))
        xLR = self.dropout(xLR)
        xLR = self.convTr(xLR)

        # HR component
        xHR = self.NNHR(xinp)

        return xLR + xHR


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return torch.mul(1., x)


class CorrelateNoise(torch.nn.Module):
    def __init__(self, shape_data, dim_cn):
        super(CorrelateNoise, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_cn, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_cn, 2 * dim_cn, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_cn, shape_data, (3, 3), padding=1, bias=False)

    def forward(self, w):
        w = self.conv1(F.relu(w)).to(device)
        w = self.conv2(F.relu(w)).to(device)
        w = self.conv3(w).to(device)
        return w


class RegularizeVariance(torch.nn.Module):
    def __init__(self, shape_data, dim_rv):
        super(RegularizeVariance, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_rv, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_rv, 2 * dim_rv, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_rv, shape_data, (3, 3), padding=1, bias=False)

    def forward(self, v):
        v = self.conv1(F.relu(v)).to(device)
        v = self.conv2(F.relu(v)).to(device)
        v = self.conv3(v).to(device)
        return v


class Phi_r(torch.nn.Module):
    def __init__(self, shape_state, DimAE, dW, dW2, sS, nbBlocks, rateDr, stochastic=False):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(shape_state, DimAE, dW, dW2, sS, nbBlocks, rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shape_state, 10)
        self.regularize_variance = RegularizeVariance(shape_state, 10)
        self.stochastic = stochastic

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.stochastic == True:
            W = torch.randn(x.shape).to(device)
            #  g(W) = alpha(x)*h(W)
            # gW = torch.mul(self.regularize_variance(x),self.correlate_noise(W))
            gW = self.correlate_noise(W)
            # print(stats.describe(gW.detach().cpu().numpy()))
            x = x + gW
        return x


class Model_H(torch.nn.Module):
    def __init__(self, shape_state):
        super(Model_H, self).__init__()
        self.DimObs = 1
        self.dimObsChannel = np.array([shape_state])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class Model_H_with_noisy_Swot(torch.nn.Module):
    """
    state: [oi, anom_glob, anom_swath ]
    obs: [oi, obs]
    mask: [ones, obs_mask]
    """
    def __init__(self, shape_state, shape_obs, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.DimObs = 1
        self.dimObsChannel = np.array([shape_obs])



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


class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.convgx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convgx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.convgy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convgy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),
                requires_grad=False)

        self.eps=10**-3

    def forward(self, im):

        if im.size(1) == 1:
            g_x = self.convgx(im)
            g_y = self.convgy(im)
            g = torch.sqrt(torch.pow(0.5 * g_x, 2) + torch.pow(0.5 * g_y, 2) + self.eps)
        else:

            for kk in range(0, im.size(1)):
                g_x = self.convgx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                g_y = self.convgy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                g_x = g_x.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                g_y = g_y.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                ng = torch.sqrt(torch.pow(0.5 * g_x, 2) + torch.pow(0.5 * g_y, 2)+ self.eps)

                if kk == 0:
                    g = ng.view(-1, 1, im.size(1) - 2, im.size(2) - 2)
                else:
                    g = torch.cat((g, ng.view(-1, 1, im.size(1) - 2, im.size(2) - 2)), dim=1)
        return g

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)


def get_4dvarnet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H_with_noisy_Swot(hparams.shape_state[0], hparams.shape_obs[0], hparams=hparams),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                None, None, hparams.shape_state, hparams.n_grad)

from direct_inversion_grid import get_passthrough, get_vit
############################################ Lightning Module #######################################################################
class LitModel(pl.LightningModule):


    MODELS = {
            'passthrough': get_passthrough,
            'vit': get_vit,
            '4dvarnet': get_4dvarnet,
        }

    def __init__(self, hparam=None, *args, **kwargs):
        super().__init__()
        hparam = {} if hparam is None else hparam
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)
        self.save_hyperparameters({**hparams, **kwargs})

        # TOTEST: set those parameters only if provided
        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt

        # create longitudes & latitudes coordinates
        self.test_lon = np.arange(self.hparams.min_lon, self.hparams.max_lon, 0.05)
        self.test_lat = np.arange(self.hparams.min_lat, self.hparams.max_lat, 0.05)
        self.test_dates = self.hparams.test_dates[self.hparams.dT // 2: -(self.hparams.dT // 2)]
        # print(len(self.test_dates), len(self.hparams.test_dates), self.hparams.dT)
        self.ds_size_time = self.hparams.ds_size_time
        self.ds_size_lon = self.hparams.ds_size_lon
        self.ds_size_lat = self.hparams.ds_size_lat

        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt
        self.mean_Val = self.hparams.mean_Val
        self.mean_Tr = self.hparams.mean_Tr
        self.mean_Tt = self.hparams.mean_Tt

        # main model
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
        model = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        return self.MODELS[model](self.hparams)

    def forward(self):
        return 1

    def configure_optimizers(self):
        if self.hparams.model == '4dvarnet':
            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ], lr=0.)

            return optimizer
        else: 
            opt = optim.Adam(self.parameters(), lr=1e-2)
        return {
            'optimizer': opt,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=10,),
            'monitor': 'val_loss'
        }

    def on_epoch_start(self):
        # enfore acnd check some hyperparameters
        self.model.n_grad = self.hparams.n_grad

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad = self.hparams.n_grad

            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                mm += 1

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics    
        loss, out, _, metrics = self.compute_loss(train_batch, phase='train')
        if loss is None:
            return loss
        # log step metric        
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mse_swath", metrics['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG_swath", metrics['mseGradSwath'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        # initial grad value
        if self.hparams.automatic_optimization == False:
            opt = self.optimizers()
            # backward
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()

                # grad initialization to zero
                opt.zero_grad()

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, out, _, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mse_swath", metrics['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG_swath", metrics['mseGradSwath'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, inputs_obs, obs_target_item, targets_GT = test_batch
        loss, out, out_pred, metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mse_swath", metrics['mseSwath'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG_swath", metrics['mseGradSwath'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'target_obs'    : (obs_target_item.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'inp_obs'    : (inputs_obs.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_pred'    : (out_pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        target_obs = torch.cat([chunk['target_obs'] for chunk in outputs]).numpy()
        inputs_obs = torch.cat([chunk['inp_obs'] for chunk in outputs]).numpy()
        obs_pred = torch.cat([chunk['obs_pred'] for chunk in outputs]).numpy()

        ds_size = {'time': self.ds_size_time,
                'lon': self.ds_size_lon,
                'lat': self.ds_size_lat,
                }

        gt, oi, pred, inputs_obs, target_obs, obs_pred = map(
                lambda t: einops.rearrange(
                    t,
                    '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                    t_idx=int(ds_size['time']),
                    lat_idx=int(ds_size['lat']),
                    lon_idx=int(ds_size['lon']),
                    ),
                [gt, oi, pred, inputs_obs, target_obs, obs_pred])

        self.x_gt = gt[:, int(self.hparams.dT / 2), :, :]
        self.x_oi = oi[:, int(self.hparams.dT / 2), :, :]
        self.x_rec = pred[:, int(self.hparams.dT / 2), :, :]
        self.obs_gt = target_obs[:, int(self.hparams.dT / 2), :, :]
        self.obs_inp = np.where(~np.isnan(self.obs_gt), inputs_obs[:, int(self.hparams.dT / 2), :, :], np.full_like(self.obs_gt, float('nan')))
        self.obs_pred = np.where(~np.isnan(self.obs_gt), obs_pred[:, int(self.hparams.dT / 2), :, :], np.full_like(self.obs_gt, float('nan')))

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
                    'time': (('time',), self.test_dates),
                    'lat': (('lat',), self.test_lat),
                    'lon': (('lon',), self.test_lon),
                    },
                )

        Path(self.logger.log_dir).mkdir(exist_ok=True)
        # display map
        path_save0 = self.logger.log_dir + '/maps.png'
        fig_maps = plot_maps(
                self.x_gt[3],
                self.x_oi[3],
                self.x_rec[3],
                self.test_lon, self.test_lat, path_save0)
        self.test_figs['maps'] = fig_maps
        self.logger.experiment.add_figure('Maps', fig_maps, global_step=self.current_epoch)

        path_save01 = self.logger.log_dir + '/maps_obs.png'
        fig_maps = plot_maps(
                self.obs_gt[3],
                self.obs_inp[3],
                self.obs_pred[3],
                self.test_lon, self.test_lat, path_save01)
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
        save_netcdf(saved_path1=path_save1, pred=self.x_rec,
                lon=self.test_lon, lat=self.test_lat, dates=self.test_dates)

        # compute nRMSE
        path_save2 = self.logger.log_dir + '/nRMSE.txt'
        tab_n_scores = nrmse_scores(self.x_gt, self.x_oi, self.x_rec, path_save2)
        print('*** Display nRMSE scores grid ***')
        print(tab_n_scores.to_markdown())
        path_save22 = self.logger.log_dir + '/MSE.txt'
        tab_scores = mse_scores(self.x_gt, self.x_oi, self.x_rec, path_save22)
        print('*** Display MSE scores grid ***')
        print(tab_scores.to_markdown())

        # compute nRMSE on swath
        path_save23 = self.logger.log_dir + '/nRMSE_swath.txt'
        tab_n_scores_swath = nrmse_scores(self.obs_gt, self.obs_inp, self.obs_pred, path_save23)
        print('*** Display nRMSE scores swath ***')
        print(tab_n_scores_swath.to_markdown())
        path_save24 = self.logger.log_dir + '/MSE_swath.txt'
        tab_scores_swath = mse_scores(self.obs_gt, self.obs_inp, self.obs_pred, path_save24)
        print('*** Display MSE scores swath ***')
        print(tab_scores_swath.to_markdown())

        # plot nRMSE
        # PENDING: replace hardcoded 60
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,  self.x_oi, self.x_rec, path_save3, dates=self.test_dates)
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
        # TODO: log hparams and metrics
        mdf = pd.concat(
                [
                    df.T[['pred', 'ratio']]
                    .rename({'pred': f'{mname}_abs', 'ratio': f'{mname}_imp'}, axis=1)
                    .T['mean']

                    for df, mname in [
                        (tab_n_scores, 'nrmse_glob'),
                        (tab_scores, 'mse_glob'),
                        (tab_n_scores_swath, 'nrmse_swath'),
                        (tab_scores_swath, 'mse_swath'),
                        ]
                    ]
                )
        print(mdf.to_frame().to_markdown())
        self.logger.log_hyperparams(
                {**self.hparams},
                {
                    'spatial_res': float(spatial_res_model),
                    'spatial_res_imp': float(spatial_res_model / spatial_res_oi),
                    **mdf.to_dict(), },

                )

    def compute_loss(self, batch, phase):
        targets_OI, inputs_Mask, inputs_obs, target_obs_GT, targets_GT = batch
        #targets_OI, inputs_Mask, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_GT),
                    dict([('mse', 0.), ('mseGrad', 0.),('mseSwath', 0.), ('mseGradSwath', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                        ('mseGOI', 0.)])
                    )
        new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask), dim=1)
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        target_obs_GT_wo_nan = target_obs_GT.where(~target_obs_GT.isnan(), torch.zeros_like(target_obs_GT))
        anomaly_global = torch.zeros_like(targets_OI)

        if self.hparams.anom_swath_init == 'zeros':
            anomaly_swath = torch.zeros_like(targets_OI)
        elif self.hparams.anom_swath_init == 'obs':
            anomaly_swath = (inputs_obs - targets_OI).detach()

        state = torch.cat((targets_OI, anomaly_global, anomaly_swath), dim=1)

        #state = torch.cat((targets_OI, inputs_Mask * (targets_GT_wo_nan - targets_OI)), dim=1)
        obs = torch.cat((targets_OI, inputs_obs), dim=1)
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
            output_low_res,  output_anom_glob, output_anom_swath = torch.split(outputs, split_size_or_sections=targets_OI.size(1), dim=1)
            output_global = output_low_res + output_anom_glob

            if self.hparams.swot_anom_wrt == 'low_res':
                output_swath = output_low_res + output_anom_swath
                gt_anom_swath = targets_OI
            elif self.hparams.swot_anom_wrt == 'high_res':
                output_swath = output_global + output_anom_swath
                gt_anom_swath = targets_GT

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
            # PENDING: Add loss term

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

        return loss, output_global, output_swath, metrics


class Model_HwithSST(torch.nn.Module):
    def __init__(self, shape_state, dim=5):
        super(Model_HwithSST, self).__init__()

        self.DimObs = 2
        self.dimObsChannel = np.array([shape_state, dim])

        self.conv11 = torch.nn.Conv2d(shape_state, self.dimObsChannel[1], (3, 3), padding=1, bias=False)

        self.conv21 = torch.nn.Conv2d(int(shape_state / 2), self.dimObsChannel[1], (3, 3), padding=1, bias=False)
        self.convM = torch.nn.Conv2d(int(shape_state / 2), self.dimObsChannel[1], (3, 3), padding=1, bias=False)
        self.S = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
        dyout1 = self.conv11(x) - self.conv21(y1)
        dyout1 = dyout1 * self.S(self.convM(mask[1]))

        return [dyout, dyout1]


class Model_H_with_SST_and_noisy_Swot(torch.nn.Module):
    """
    state: [oi, obs, anom_glob, anom_swath ]
    obs: [oi, obs], ~sst
    mask: [ones, obs_mask]
    """
    def __init__(self, shape_state, shape_obs, shape_sst, dim=5):
        super().__init__()

        self.DimObs = 2
        self.dimObsChannel = np.array([shape_state, dim])

        self.conv_state_to_obs = torch.nn.Conv2d(shape_state, shape_obs, (3, 3), padding=1, bias=False)
        self.conv11 = torch.nn.Conv2d(shape_state, self.dimObsChannel[1], (3, 3), padding=1, bias=False)

        self.conv21 = torch.nn.Conv2d(shape_sst, self.dimObsChannel[1], (3, 3), padding=1, bias=False)
        self.convM = torch.nn.Conv2d(shape_sst, self.dimObsChannel[1], (3, 3), padding=1, bias=False)
        self.S = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def forward(self, x, y, mask):

        dyout = (self.conv_state_to_obs(x) - y[0]) * mask[0]

        y1 = y[1] * mask[1]
        dyout1 = self.conv11(x) - self.conv21(y1)
        dyout1 = dyout1 * self.S(self.convM(mask[1]))

        return [dyout, dyout1]

class LitModelWithSST(LitModel):
    def __init__(self, hparam=None, *args, **kwargs):
        super().__init__(hparam, *args, **kwargs)
        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.hparams.shape_state[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks, self.hparams.dropout_phi_r),
            Model_H_with_SST_and_noisy_Swot(self.hparams.shape_state[0], self.hparams.shape_obs[0],  self.hparams.shape_obs[0]//2
                ,self.hparams.shape_state[0]),
            NN_4DVar.model_GradUpdateLSTM(self.hparams.shape_state, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.hparams.shape_state, self.hparams.n_grad)

    def configure_optimizers(self):

        optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0.)

        return optimizer

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad = self.hparams.n_grad

            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent, lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                mm += 1

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, input_obs, target_obs_GT, targets_GT, sst_GT = test_batch
        loss, out, out_pred,  metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return {'gt'    : (targets_GT.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'inp_obs'    : (input_obs.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'target_obs'    : (target_obs_GT.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_pred'    : (out_pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr}

    def compute_loss(self, batch, phase):  ## to be updated
        targets_OI, inputs_Mask, inputs_obs, target_obs_GT, targets_GT, sst_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_GT),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                      ('mseGOI', 0.)])
            )
        new_masks = torch.cat((1. + 0. * inputs_Mask, inputs_Mask), dim=1)
        mask_SST = 1. + 0. * sst_GT
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        target_obs_GT_wo_nan = target_obs_GT.where(~target_obs_GT.isnan(), torch.zeros_like(target_obs_GT))
        anomaly_global = torch.zeros_like(targets_OI)
        anomaly_swath = torch.zeros_like(targets_OI)
        state = torch.cat((targets_OI, inputs_obs, anomaly_global, anomaly_swath), dim=1)
        # PENDING: create state with anomaly full and anomaly swath (pad with zeros)

        #state = torch.cat((targets_OI, inputs_Mask * (targets_GT_wo_nan - targets_OI)), dim=1)
        obs = torch.cat((targets_OI, inputs_obs), dim=1)
        # PENDING: Add state with zeros

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)
        g_targets_obs = self.gradient_img(target_obs_GT_wo_nan)
        g_targets_obs_mask = self.gradient_img(target_obs_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            state = torch.autograd.Variable(state, requires_grad=True)

            outputs, _, _, _ = self.model.forward(state, [obs, sst_GT],
                                                                 [new_masks, mask_SST])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()


            output_low_res, _, output_anom_glob, output_anom_swath = torch.split(outputs, split_size_or_sections=targets_OI.size(1), dim=1)
            output_global = output_low_res + output_anom_glob
            output_swath = output_global + output_anom_swath

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

            # reconstruction losses
            loss_All = NN_4DVar.compute_WeightedLoss((output_global - targets_GT), self.w_loss)
            loss_GAll = NN_4DVar.compute_WeightedLoss(g_output_global - g_targets_GT, self.w_loss)
            loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = torch.mean((self.model.phi_r(outputs) - outputs) ** 2)
            yGT = torch.cat((targets_GT_wo_nan, inputs_obs, targets_GT_wo_nan - output_low_res, target_obs_GT_wo_nan - output_global), dim=1)
            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_WeightedLoss(output_low_res - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(output_global) - targets_GTLR, self.w_loss)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
            if loss_swath > 0:
                pass
                loss += self.hparams.alpha_mse_ssh * loss_swath
            if loss_grad_swath > 0:
                pass
                loss += self.hparams.alpha_mse_gssh * loss_grad_swath
                
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
            # PENDING: Add loss term

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

        return loss, output_global, output_swath, metrics
