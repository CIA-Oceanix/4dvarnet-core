# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Imports

# %%
from new_dataloading import FourDVarNetDataModule, XrDataset, FourDVarNetDataset
from models import LitModel

# %%
import xarray as xr
import torch
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np
import pytorch_lightning as pl

# %% [markdown]
# ### Config

# %%
params = {
    'data_dir'        : '/gpfsscratch/rech/nlu/commun/large',
    'dir_save'        : '/gpfsscratch/rech/nlu/commun/large/results_maxime',

    'iter_update'     : [0, 20, 40, 60, 100, 150, 800],  # [0,2,4,6,9,15]
    'nb_grad_update'  : [5, 5, 10, 10, 15, 15, 20, 20, 20],  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
    'lr_update'       : [1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7],
    'k_batch'         : 1,
    'n_grad'          : 5,
    'dim_grad_solver' : 10,
    'dropout'         : 0.25,
    'dT'              : 5, ## Time window of each space-time patch
    'dx'              : 1,   ## subsampling step if > 1
    'W'               : 200, # width/height of each space-time patch
    'shapeData'       : [10, 200, 200],
    'dW'              : 3,
    'dW2'             : 1,
    'sS'              : 4,  # int(4/dx),
    'nbBlocks'        : 1,
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 

    # stochastic version
    'stochastic'      : False,

    # animation maps 
    'animate'         : False,

    # NN architectures and optimization parameters
    'batch_size'      : 4, #16#4#4#8#12#8#256#
    'DimAE'           : 50, #10#10#50
    'dimGradSolver'   : 100, # dimension of the hidden state of the LSTM cell

    'alpha_MSE'       : 0.1,
    'alpha_Proj'      : 0.5,
    'alpha_SR'        : 0.5,
    'alpha_LR'        : 0.5,  # 1e4

    # data generation
    'sigNoise'        : 0.,## additive noise standard deviation
    'flagSWOTData'    : True, #False ## use SWOT data or not
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 
    'rnd1'            : 0, ## random seed for patch extraction (space sam)
    'rnd2'            : 100, ## random seed for patch extraction
    'dwscale'         : 1,

    'betaX'           : 42.20436766972647, #None
    'betagX'          : 77.99700321505073, #None

    'UsePriodicBoundary' : False,  # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    'InterpFlag'         : False, # True :> force reconstructed field to observed data after each gradient-based update
    'flagSWOTData'       : True,
    'automatic_optimization' : True,

}
cfg = OmegaConf.create(params)

# %% [markdown]
# ### Init Datamodule

# %%
dim_range = {
    'lat': slice(33, 43),
    'lon': slice(-65, -55),
}

# Specify the batch patch size
slice_win = {
    'time': cfg.dT,
    'lat': 200,
    'lon': 200,
    # 'lat': 20,
    # 'lon': 20,
}
# Specify the stride between two patches
strides = {
    'time': 1,
    'lat': 200,
    'lon': 200,
    # 'lat': 20,
    # 'lon': 20,
}

data_files=dict(
    oi_path='/users/local/DATA/OSSE/GULFTREAM/oi/ssh_NATL60_swot_4nadir.nc',
    oi_var='ssh_mod',
    obs_mask_path='/users/local/DATA/OSSE/GULFTREAM/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc',
    obs_mask_var='ssh_mod',
    gt_path='/users/local/DATA/OSSE/GULFTREAM/ref/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc',
    gt_var='ssh',
#     sst_path='/gpfsscratch/rech/nlu/commun/large/NATL60-CJM165_NATL_sst_y2013.1y.nc',
#     sst_var='sst'
)
datamodule = FourDVarNetDataModule(
    slice_win=slice_win,
    dim_range=dim_range,
    strides=strides,
    **data_files
)

# %%
datamodule.setup()
train_dl = datamodule.train_dataloader()

# %%
targets_OI, inputs_Mask, targets_GT = next(iter(train_dl))

# %%
targets_OI.shape, inputs_Mask.shape, targets_GT.shape

# %%
# %matplotlib inline

# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 4))
axs[0].imshow(targets_OI[0,0,...].detach().cpu().numpy())
axs[1].imshow(inputs_Mask[0,0,...].detach().cpu().numpy())
axs[2].imshow(targets_GT[0,0,...].detach().cpu().numpy())
obs = torch.where(inputs_Mask, targets_GT, torch.zeros_like(targets_GT))
axs[3].imshow(obs[0,0,...].detach().cpu().numpy())
fig.show()

# %% [markdown]
# ### Init Model

# %%
from pathlib import Path

# %%
ckpt_path = "lightning_logs/version_15/checkpoints/modelSLAInterpGF-Exp3-epoch=20-val_loss=0.05.ckpt"

# %%
w_ = np.zeros(cfg.dT)
w_[int(cfg.dT / 2)] = 1.

bb = dict(zip(('min_lon', 'max_lon', 'min_lat', 'max_lat'), datamodule.bounding_box))
model_kwargs=dict(
    w_loss = torch.Tensor(w_),
    var_Tr = datamodule.norm_stats[1] ** 2,
    var_Tt = datamodule.norm_stats[1] ** 2,
    var_Val = datamodule.norm_stats[1] ** 2,
    ds_size_time = datamodule.ds_size['time'],
    ds_size_lon = datamodule.ds_size['lon'],
    ds_size_lat = datamodule.ds_size['lat'],
)
litmod = LitModel.load_from_checkpoint(ckpt_path, hparam=cfg, **model_kwargs, **bb )

# %% [markdown]
# ### Testing

# %%
trainer = pl.Trainer(gpus=1, max_epochs=100, progress_bar_refresh_rate=5)

# %%
trainer.test(litmod, test_dataloaders=datamodule.test_dataloader())

# %%
litmod.test_figs['maps']

# %%
litmod.test_figs['nrmse']

# %%
litmod.test_figs['snr']

# %%
litmod = litmod.to('cuda')

# %%
batch = litmod.transfer_batch_to_device(next(iter(datamodule.test_dataloader())), litmod.device)
loss, outputs, metrics = litmod.compute_loss(batch , phase='train')

# %%
targets_OI, inputs_Mask, targets_GT = batch

# %%
torch.mean((targets_GT[:, 2] - targets_OI[:, 2])**2), torch.mean((targets_GT[:, 2] - outputs[:, 2])**2)

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].imshow(targets_OI[0, 2].detach().cpu().numpy())
axs[1].imshow(targets_GT[0, 2].detach().cpu().numpy())
axs[2].imshow(outputs[0, 2].detach().cpu().numpy())


# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
im1 = axs[0].imshow((targets_GT[0, 2] - targets_OI[0, 2]).detach().cpu().numpy())
fig.colorbar(im1, ax=axs[0])
im2 = axs[1].imshow((targets_GT[0, 2] - outputs[0, 2]).detach().cpu().numpy())
fig.colorbar(im2, ax=axs[1])


# %%
litmod.x_gt.shape, litmod.x_oi.shape, litmod.x_rec.shape

# %%
((litmod.x_gt - litmod.x_oi)**2).mean(), ((litmod.x_gt - litmod.x_rec)**2).mean()

# %%
nrmse = lambda ref, pred: np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)

# %%
nrmse(litmod.x_gt, litmod.x_oi), ((litmod.x_gt - litmod.x_rec)**2).mean()

# %%
gt, oi, pred = litmod.x_gt, litmod.x_oi, litmod.x_rec

nrmse_oi = []
nrmse_pred = []
for i in range(len(oi)):
    nrmse_oi.append(nrmse(gt[i], oi[i]))
    nrmse_pred.append(nrmse(gt[i], pred[i]))

# %%
fig, ax = plt.subplots()
plt.plot(nrmse_oi)
plt.plot(nrmse_pred)

# %%
