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
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps_oi, animate_maps, get_psd_score
from models import Model_H, Phi_r_OI, Gradient_img
# additional import for OSE metrics
from ose.src.mod_inout import *
from ose.src.mod_interp import *
from ose.src.mod_stats import *
from ose.src.mod_spectral import *
from ose.src.mod_plot import *
from ose.src.utils import *

from lit_model_OI import LitModelOI

def get_4dvarnet_OI(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

class LitModelOI_predict(LitModelOI):
    MODELS = {
            '4dvarnet_OI': get_4dvarnet_OI,
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
         self.c2_file = '/users/local/m19beauc/4dvarnet-core/ose/eval_notebooks/inputs/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc'
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
         self.c2_lenght_scale = 1000 # km


    def diag_step(self, batch, batch_idx, log_pref='predict'):

        oi, inputs_mask, inputs_obs, gt, analogs = batch
        n_analogs = analogs.shape[1]

        losses, out, metrics = self([oi,inputs_mask,inputs_obs,gt], phase='test')
        loss = losses[-1]
        '''
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        '''

        outputs_simu = list()
        for i in range(n_analogs):
            # 4DVarNet on analogs
            state = analogs[:,i,:,:,:] * inputs_mask
            obs = analogs[:,i,:,:,:] * inputs_mask
            state = torch.autograd.Variable(state, requires_grad=True)
            losses, outputs_simu_, metrics = self([oi,inputs_mask,obs,gt], phase='test')
            outputs_simu_ = outputs_simu_.detach()
            outputs_simu.append(outputs_simu_)
        # pdf has shape batch*n_analogs*time*lat*lon
        pdf = torch.unsqueeze(out,dim=1) + (analogs-torch.stack(outputs_simu,dim=1))

        # return observations, 4DVarNet and estimated pdf
        return {'gt' : (gt.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs_inp'    : (inputs_obs.detach().where(inputs_mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pred' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'pdf' : (pdf.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def predict_step(self, predict_batch, batch_idx):
        return self.diag_step(predict_batch, batch_idx, log_pref='predict')

    def on_predict_epoch_end(self, outputs):
        return self.diag_epoch_end(outputs, log_pref='predict')

    def diag_epoch_end(self, outputs, log_pref='predict'):
        outputs_0 = [ [ dict((k, outputs[i][j][k]) for k in ['gt','obs_inp', 'pred']) \
                        for j in range(len(outputs[i])) \
                      ] \
                        for i in range(len(outputs)) \
                    ]
        outputs_1 = [ [ dict((k, outputs[i][j][k]) for k in ['pdf']) \
                        for j in range(len(outputs[i])) \
                      ] \
                        for i in range(len(outputs)) \
                    ]
        full_outputs = self.gather_outputs(outputs_0, log_pref=log_pref)
        full_outputs_pdf = self.gather_outputs(outputs_1, log_pref=log_pref)
        if full_outputs is None:
            print("full_outputs is None on ", self.global_rank)
            return
        if log_pref == 'predict':
            diag_ds = self.trainer.predict_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise Exception('unknown phase')
        
        if log_pref=='predict':
            self.test_xr_ds = self.build_test_xr_ds(outputs_0, diag_ds=diag_ds)
            self.test_xr_pdf_ds = self.build_test_xr_pdf_ds(outputs_1, diag_ds=diag_ds)
            self.test_xr_ds = xr.merge([self.test_xr_ds,self.test_xr_pdf_ds])
        else:
            self.test_xr_ds = self.build_test_xr_ds(full_outputs, diag_ds=diag_ds)
            
        Path(self.logger.log_dir).mkdir(exist_ok=True)
        path_save1 = self.logger.log_dir + f'/test.nc'
        self.test_xr_ds.to_netcdf(path_save1)

        self.x_gt = self.test_xr_ds.gt.data
        self.obs_inp = self.test_xr_ds.obs_inp.data
        self.x_rec = self.test_xr_ds.pred.data
        self.x_rec_pdf = self.test_xr_ds.pdf.data

        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data
        
    def build_test_xr_pdf_ds(self, outputs, diag_ds):

        # add test if additional dim for simus
        
        outputs_keys = list(outputs[0][0].keys())
        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i]
               for i in range(len(diag_ds))
            ]

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bm = outputs[0][b]['pdf'].shape[0]
                for i in range(bm):
                     for bc in range(n_batch_chunk):
                        ret = tuple(
                                    [outputs[bc][b][k][i] for k in outputs_keys]
                            )
                        yield ret

        dses =[
                xr.Dataset( {
                    k: (('member','time', 'lat', 'lon'), x_k) for k, x_k in zip(outputs_keys, xs)
                }, coords=coords)
            for  xs, coords
            in zip(iter_item(outputs), self.test_patch_coords)
        ]

        fin_ds = xr.merge([xr.zeros_like(ds[['member', 'time', 'lat', 'lon']]) for ds in dses])
        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
        )
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
            )
        for ds in dses:
            ds_nans = ds.assign(weight=xr.ones_like(ds.pdf)).isnull().broadcast_like(fin_ds).fillna(0.)
            xr_weight_ = xr.DataArray(self.patch_weight.unsqueeze(dim=0).detach().cpu(), ds.coords, dims=ds.pdf.dims)
            xr_weight = xr_weight_
            for _ in range(ds.pdf.shape[0]-1):
                xr_weight = xr.concat([xr_weight, xr_weight_],dim="member")
            _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
            fin_ds = fin_ds + _ds

        return (
            (fin_ds.drop('weight') / fin_ds.weight)
            .sel(instantiate(self.test_domain))
            .isel(time=slice(self.hparams.dT //2, -self.hparams.dT //2))
            # .pipe(lambda ds: ds.sel(time=~(np.isnan(ds.gt).all('lat').all('lon'))))
        ).transpose('member','time', 'lat', 'lon')

