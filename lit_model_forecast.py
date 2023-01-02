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

from lit_model_OI import LitModelOI

def get_4dvarnet_OI(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r_OI(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

class LitModelForecast(LitModelOI):

    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        _, inputs_Mask, inputs_obs, _ = batch

        # Use the obs of t=1 ... (dT-1)/2 to forecast t=(dT-1)/2+1 ... dT
        init_state = torch.cat((torch.index_select(inputs_Mask * inputs_obs, dim=1, index=torch.tensor(range(1, (self.hparams.dT-1)/2))),
                                torch.zeros_like(torch.index_select(inputs_Mask, dim=1, index=torch.tensor(range((self.hparams.dT-1)/2, self.hparams.dT))))
                              ))
        # init_state = inputs_Mask * inputs_obs
        return init_state

    def get_obs_state(self, batch):
        """ Create obs state for the compute loss function. Use the obs of t=1 ... (dT-1)/2 to forecast t=(dT-1)/2+1 ... dT """
        _, inputs_Mask, inputs_obs, _ = batch
        obs = torch.cat((torch.index_select(inputs_Mask * inputs_obs, dim=1, index=torch.tensor(range(1, (self.hparams.dT-1)/2))),
                                torch.zeros_like(torch.index_select(inputs_Mask, dim=1, index=torch.tensor(range((self.hparams.dT-1)/2, self.hparams.dT))))
                       ))
        return obs
