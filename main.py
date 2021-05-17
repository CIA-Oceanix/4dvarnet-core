#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import xarray as xr
import argparse
import numpy as np
import datetime
from os.path import join
from os import makedirs
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from netCDF4 import Dataset
import pytorch_lightning as pl
import math
import torch.distributed as dist

from data_utils import extract_image_patches, extract_SpaceTimePatches, save_NetCDF, Gradient_img, compute_betas
from models import Decoder, Phi_r, Model_H, ModelLR, LitModel

################################
############# FLAGS ############
################################

flagRandomSeed     = 0
batch_size         = 12  # 16#4#4#8#12#8#256#8
DimAE              = 10  # 10#10#50
UsePriodicBoundary = False  # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
flagUseObsData     = True  # use along-track data or not
flagUseOI          = True
flagMultiScale     = True
flagSWOTData       = True  # False
dT                 = 5
W                  = 128  # 128
dx                 = 1
dimGradSolver      = 50
Nbpatches          = 10  # 10#25
rnd1               = 0
rnd2               = 100
rateDropout        = 0.25
W                  = int(W / dx)
dwscale            = 1

if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)
    torch.manual_seed(100)

# lat lon grid
idLon = np.arange(400,600)
idLat = np.arange(100,300)

################################
############# DATA #############
################################

# common dir
root_dir = '/gpfsscratch/rech/nlu/commun/large'
dirSAVE  = join(root_dir, 'ResSLANATL60')
makedirs(dirSAVE, exist_ok=True)
ncfile   = Dataset(join(root_dir, "NATL60-CJM165_NATL_ssh_y2013.1y.nc"), "r")
qHR      = ncfile.variables['ssh'][:,idLat,idLon]


if flagSWOTData == True:
    print('.... Use SWOT+4-nadir dataset')
    genFilename = 'resInterpSWOTSLAwOInoSST_' + str('%03d' % (W)) + 'x' + str('%03d' % (W)) + 'x' + str(
        '%02d' % (dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(join(root_dir, "ssh_NATL60_swot_4nadir.nc"), "r")
    qOI = ncfile.variables['ssh_mod'][:,idLat,idLon]
    ncfile.close()
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(join(root_dir, "dataset_nadir_0d_swot.nc"), "r")
    qMask = ncfile.variables['ssh_mod'][:,idLat,idLon]
    qMask = 1.0 - qMask.mask.astype(float)
    ncfile.close()

print('----- MSE OI: %ls.3f' % np.mean((qOI - qHR) ** 2))
print()

dtOI = 10
# training1
ii1 = 10  # temporal start
jj1 = 40  # temporal end
# training2
ii2 = 105 # temporal start
jj2 = 365 # temporal end

## extraction of patches from the SSH field
## TODO: next steps it to remove lines below and to on the fly generation
dataTraining = np.concatenate(
    (
        extract_SpaceTimePatches(qHR, ii1, jj1, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False), # dataTraining1
        extract_SpaceTimePatches(qHR, ii2, jj2, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)  # dataTraining2
    ),
    axis=0
)

dataTrainingMask = np.concatenate(
    (
        extract_SpaceTimePatches(qMask, ii1, jj1, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False), # dataTrainingMask1
        extract_SpaceTimePatches(qMask, ii2, jj2, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)  # dataTrainingMask2
    ),
    axis=0
)

dataTrainingOI = np.concatenate(
    (
        extract_SpaceTimePatches(qOI, ii1, jj1, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False), # dataTrainingOI1
        extract_SpaceTimePatches(qOI, ii2, jj2, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)  # dataTrainingOI2
    ),
    axis=0
)

dataTrainingOI1 = np.concatenate(
    (
        extract_SpaceTimePatches(qOI, ii2 - dtOI, jj2 - dtOI, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False), # dataTrainingOI21
        extract_SpaceTimePatches(qOI, ii1 - dtOI, jj1 - dtOI, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)  # dataTrainingOI11
    ),
    axis=0
)


# test
ii3 = 60 - int(dT / 2)
jj3 = 80 + int(dT / 2)

dataTest      = extract_SpaceTimePatches(qHR,   ii3, jj3, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)
dataTestMask  = extract_SpaceTimePatches(qMask, ii3, jj3, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)
dataTestOI    = extract_SpaceTimePatches(qOI,   ii3, jj3, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)
dataTestOI1   = extract_SpaceTimePatches(qOI,   ii3 - dtOI, jj3 - dtOI, W, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=False)

# get statistics and normalize
meanTr        = np.mean(dataTraining)
dataTraining  = dataTraining - meanTr
stdTr         = np.sqrt(np.mean(dataTraining ** 2))
dataTraining /= stdTr

dataTest      = dataTest - meanTr
stdTt         = np.sqrt(np.mean(dataTest ** 2))  ## TODO: on test set ??? 
dataTest     /= stdTr

dataTrainingOI  = (dataTrainingOI - meanTr) / stdTr
dataTrainingOI1 = (dataTrainingOI1 - meanTr) / stdTr
dataTestOI      = (dataTestOI - meanTr) / stdTr
dataTestOI1     = (dataTestOI1 - meanTr) / stdTr


training_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(dataTrainingOI),
    torch.Tensor(dataTrainingOI1),
    torch.Tensor(dataTrainingMask),
    torch.Tensor(dataTraining)
)

del dataTrainingOI
del dataTrainingOI1
del dataTrainingMask
del dataTraining

val_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(dataTestOI),
    torch.Tensor(dataTestOI1),
    torch.Tensor(dataTestMask),
    torch.Tensor(dataTest)
)

del dataTestOI
del dataTestOI1
del dataTestMask
del dataTest

train_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# keep track of shapes
shapeData      = np.array(dataTraining.shape[1:])
shapeData_test = np.array(dataTest.shape[1:])

# another test set
Nbpatches = 1
qHR = qHR[:, 0:200, 0:200]
qOI = qOI[:, 0:200, 0:200]
qMask = qMask[:, 0:200, 0:200]

dataTest     = extract_SpaceTimePatches(qHR, ii3, jj3, 200, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=True)
dataTestMask = extract_SpaceTimePatches(qMask, ii3, jj3, 200, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=True)
dataTestOI   = extract_SpaceTimePatches(qOI, ii3, jj3, 200, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=True)
dataTestOI1  = extract_SpaceTimePatches(qOI, ii3 - dtOI, jj3 - dtOI, 200, dT, rnd1, rnd2, dx, Nbpatches, NoRndPatches=True)

dataTest = (dataTest - meanTr ) / stdTr
# stdTt         = np.sqrt(np.mean(dataTest ** 2))  ## on test set ??? 
# dataTest     /= stdTt
dataTestOI  = (dataTestOI - meanTr) / stdTr
dataTestOI1 = (dataTestOI1 - meanTr) / stdTr

test_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(dataTestOI),
    torch.Tensor(dataTestOI1),
    torch.Tensor(dataTestMask),
    torch.Tensor(dataTest)
)  

del dataTest
del dataTestMask
del dataTestOI
del dataTestOI1

test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


print('........ Define AE architecture')

if flagMultiScale == True:
    shapeData[0] += shapeData[0]
    shapeData_test[0] += shapeData_test[0]
    model_LR = ModelLR()

# PHI_R model
dW = 5
sS = int(4 / dx)
phi_r = Phi_r(dW, sS, shapeData, DimAE)
print('Number of trainable parameters = %d' % (sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))

# H model
model_H = Model_H(shapeData)
print('Number of trainable parameters = %d' % (sum(p.numel() for p in model_H.parameters() if p.requires_grad)))

# Gradient image module
gradient_img = Gradient_img(shapeData, shapeData_test)

# loss weghing wrt time
w_ = np.zeros(dT)
w_[int(dT / 2)] = 1.
wLoss = torch.Tensor(w_)

betaX, betagX = 42.20436766972647, 77.99700321505073
# recompute the mean loss for OI when train_dataloader change
if betaX is None or betagX is None: 
    betaX, betagX = compute_betas(train_dataloader, gradient_img, wLoss)


# Training parameters
lr_Sampling = 1.
# mean-squared error loss
alpha = np.array([0.1, 0.5])  # np.array([1.,0.1])
alpha_L1Sampling = 0.5
thr_L1Sampling = 1.0  # 0.05#0.025 # 0.05 ()
alpha_SR = 0.5
alpha_LR = 0.5  # 1e4
NbGradIter = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
lrUpdate = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]

NBGradCurrent = NbGradIter[0]
lrCurrent     = lrUpdate[0]
lambda_LRAE   = 0.5
alpha_Grad    = alpha[0]
alpha_AE      = alpha[1]



if __name__ == '__main__':
    profile = False
    if profile:
        from pytorch_lightning.profiler import PyTorchProfiler

        profiler = PyTorchProfiler(
            "results/profile_report",
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=1),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_profile'),
            record_shapes=True
        )
        # profile with max NbGradIter
        NbGradIter[0] = NbGradIter[-1]
        profiler_kwargs = {
            'profiler': profiler,
            'max_epochs': 1,
        }
    else:
        profiler_kwargs = {'max_epochs': 1}

    mod_args = {
        'shapeData'         : shapeData,
        'shapeData_test'    : shapeData_test,
        'UsePriodicBoundary': UsePriodicBoundary,
        'dimGradSolver'     : dimGradSolver,
        'NBGradCurrent'     : NBGradCurrent,
        'rateDropout'       : rateDropout,
        'lambda_LRAE'       : lambda_LRAE,
        'lrCurrent'         : lrCurrent,
        'lr_Sampling'       : lr_Sampling,
        'stdTr'             : stdTr,
        'meanTr'            : meanTr,
        'path_save'         : 'results/test.nc',
        'flagUseObsData'    : flagUseObsData,
        'flagUseOI'         : flagUseOI,
        'flagMultiScale'    : flagMultiScale,
        'dT'                : dT,
        'thr_L1Sampling'    : thr_L1Sampling,
        'alpha_Grad'        : alpha_Grad,
        'betaX'             : betaX,
        'betagX'            : betagX,
        'alpha_AE'          : alpha_AE,
        'alpha_L1Sampling'  : alpha_L1Sampling,
        'alpha_LR'          : alpha_LR,
        'alpha_SR'          : alpha_SR,
        'wLoss'             : wLoss
    }

    mod = LitModel(phi_r, model_H, model_LR, gradient_img, mod_args)
    print(mod)

    # checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='results', save_top_k = 3)
    # training
    trainer = pl.Trainer(gpus=1, distributed_backend="ddp", **profiler_kwargs)
    trainer.fit(mod, train_dataloader, val_dataloader)
    trainer.test(mod, test_dataloaders=test_dataloader)
