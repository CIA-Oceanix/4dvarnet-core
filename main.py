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

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from netCDF4 import Dataset
import pytorch_lightning as pl
import math
import torch.distributed as dist

from data_utils import extract_image_patches, extract_SpaceTimePatches, save_NetCDF
from models import Decoder, Phi_r, Model_H, ModelLR, Model_Sampling

from sklearn.feature_extraction import image


flagRandomSeed = 0
batch_size = 12  # 16#4#4#8#12#8#256#8
DimAE = 10  # 10#10#50
UsePriodicBoundary = False  # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
flagUseObsData = True  # use along-track data or not
flagUseOI = True
flagMultiScale = True
flagSWOTData = True  # False
dT = 5
W = 128  # 128
dx = 1
dimGradSolver = 50
Nbpatches = 10  # 10#25
rnd1 = 0
rnd2 = 100
rateDropout = 0.25
W = int(W / dx)
dwscale = 1

if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)
    torch.manual_seed(100)

idLon = np.arange(400,600)
idLat = np.arange(100,300)

dirSAVE = '/gpfswork/rech/yrf/ueh53pd/ResSLANATL60/'
genSuffixObs = ''

ncfile = Dataset("/gpfswork/rech/yrf/uba22to/DATA/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc","r")
qHR = ncfile.variables['ssh'][:,idLat,idLon]
ncfile.close()

if flagSWOTData == True:
    print('.... Use SWOT+4-nadir dataset')
    genFilename = 'resInterpSWOTSLAwOInoSST_' \
         + str('%03d' % (W)) + 'x' + str('%03d' % (W)) + 'x' + str('%02d' % (dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset("/gpfswork/rech/yrf/uba22to/DATA/NATL/oi/ssh_NATL60_swot_4nadir.nc", "r")
    qOI = ncfile.variables['ssh_mod'][:,idLat,idLon]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(
                "/gpfswork/rech/yrf/uba22to/DATA/NATL/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc",
                "r")
    qMask = ncfile.variables['ssh_mod'][:,idLat,idLon]
    qMask = 1.0 - qMask.mask.astype(float)
    ncfile.close()

print('----- MSE OI: %.3f' % np.mean((qOI - qHR) ** 2))
print()


dtOI = 10
# training1
ii1 = 10  # temporal start
jj1 = 40  # temporal end
# training2
ii2 = 105 # temporal start
jj2 = 365 # temporal end

## extraction of patches from the SSH field

dataTraining  = np.concatenate(
    (
        extract_SpaceTimePatches(qHR, ii1, jj1, W, dT, rnd1, rnd2, dx, NoRndPatches=False), # dataTraining1
        extract_SpaceTimePatches(qHR, ii2, jj2, W, dT, rnd1, rnd2, dx, NoRndPatches=False)  # dataTraining2
    ),
    axis=0)

dataTrainingMask  = np.concatenate(
    (
        extract_SpaceTimePatches(qMask, ii1, jj1, W, dT, rnd1, rnd2, dx, NoRndPatches=False), # dataTrainingMask1
        extract_SpaceTimePatches(qMask, ii2, jj2, W, dT, rnd1, rnd2, dx, NoRndPatches=False)  # dataTrainingMask2
    ),
    axis=0)

dataTrainingOI   = np.concatenate(
    (
        extract_SpaceTimePatches(qOI, ii1, jj1, W, dT, rnd1, rnd2, dx, NoRndPatches=False), # dataTrainingOI1
        extract_SpaceTimePatches(qOI, ii2, jj2, W, dT, rnd1, rnd2, dx, NoRndPatches=False)  # dataTrainingOI2
    ),
    axis=0)

dataTrainingOI1  = np.concatenate(
    (
        extract_SpaceTimePatches(qOI, ii2 - dtOI, jj2 - dtOI, W, dT, rnd1, rnd2, dx, NoRndPatches=False), # dataTrainingOI21
        extract_SpaceTimePatches(qOI, ii1 - dtOI, jj1 - dtOI, W, dT, rnd1, rnd2, dx, NoRndPatches=False)  # dataTrainingOI11
    ),
    axis=0)


# test
ii3 = 60 - int(dT / 2)
jj3 = 80 + int(dT / 2)

dataTest      = extract_SpaceTimePatches(qHR, ii3, jj3, W, dT, rnd1, rnd2, dx, NoRndPatches=False)
dataTestMask  = extract_SpaceTimePatches(qMask, ii3, jj3, W, dT, rnd1, rnd2, dx, NoRndPatches=False)
dataTestOI    = extract_SpaceTimePatches(qOI, ii3, jj3, W, dT, rnd1, rnd2, dx, NoRndPatches=False)
dataTestOI1   = extract_SpaceTimePatches(qOI, ii3 - dtOI, jj3 - dtOI, W, dT, rnd1, rnd2, dx, NoRndPatches=False)

# statistics
meanTr        = np.mean(dataTraining)
dataTraining  = dataTraining - meanTr
stdTr         = np.sqrt(np.mean(dataTraining ** 2))
dataTraining /= stdTr

dataTest      = dataTest - meanTr
stdTt         = np.sqrt(np.mean(dataTest ** 2))  ## TODO: on test set ??? 
dataTest     /= stdTr

shapeData      = np.array(dataTraining.shape[1:])
shapeData_test = np.array(dataTest.shape[1:])

# x_train = dataTraining - meanTr
# x_test = dataTest - meanTr 
# # scale wrt std
# stdTr = np.sqrt(np.mean(x_train ** 2))
# x_train = x_train / stdTr
# x_test = x_test / stdTr
# stdTt = np.sqrt(np.mean(x_test ** 2))

dataTrainingOI  = (dataTrainingOI - meanTr) / stdTr
dataTrainingOI1 = (dataTrainingOI1 - meanTr) / stdTr
dataTestOI      = (dataTestOI - meanTr) / stdTr
dataTestOI1     = (dataTestOI1 - meanTr) / stdTr

# x_trainOI = (dataTrainingOI - meanTr) / stdTr
# x_trainOI1 = (dataTrainingOI1 - meanTr) / stdTr
# x_testOI = (dataTestOI - meanTr) / stdTr
# x_testOI1 = (dataTestOI1 - meanTr) / stdTr

training_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(dataTrainingOI),
    torch.Tensor(dataTrainingOI1),
    torch.Tensor(dataTrainingMask),
    torch.Tensor(dataTraining)
)

# del dataTrainingOI
# del dataTrainingOI1
# del dataTrainingMask
# del dataTraining

val_dataset = torch.utils.data.TensorDataset(
    torch.Tensor(dataTestOI),
    torch.Tensor(dataTestOI1),
    torch.Tensor(dataTestMask),
    torch.Tensor(dataTest)
)

# del dataTestOI
# del dataTestOI1
# del dataTestMask
# del dataTest

train_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

NoRndPatches = True
Nbpatches = 1
qHR = qHR[:, 0:200, 0:200]
qOI = qOI[:, 0:200, 0:200]
qMask = qMask[:, 0:200, 0:200]

dataTest     = extract_SpaceTimePatches(qHR, ii3, jj3, 200, dT, rnd1, rnd2, dx, NoRndPatches=True)
dataTestMask = extract_SpaceTimePatches(qMask, ii3, jj3, 200, dT, rnd1, rnd2, dx, NoRndPatches=True)
dataTestOI   = extract_SpaceTimePatches(qOI, ii3, jj3, 200, dT, rnd1, rnd2, dx, NoRndPatches=True)
dataTestOI1  = extract_SpaceTimePatches(qOI, ii3 - dtOI, jj3 - dtOI, 200, dT, rnd1, rnd2, dx, NoRndPatches=True)

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

#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')

if flagMultiScale == True:
    shapeData[0] += shapeData[0]
    shapeData_test[0] += shapeData_test[0]
    model_LR = ModelLR()

dW = 5
sS = int(4 / dx)

phi_r = Phi_r(dW, sS, shapeData, dimAE)
print('Number of trainable parameters = %d' % (sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))

model_H = Model_H(shapeData)
print('Number of trainable parameters = %d' % (sum(p.numel() for p in model_H.parameters() if p.requires_grad)))

gradient_img = Gradient_img(shapeData, shapeData_test)


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
lrCurrent = lrUpdate[0]
lambda_LRAE = 0.5
alpha_Grad = alpha[0]
alpha_AE = alpha[1]

# loss weghing wrt time
w_ = np.zeros(dT)
w_[int(dT / 2)] = 1.
wLoss = torch.Tensor(w_)

betaX, betagX = 42.20436766972647, 77.99700321505073
# recompute the mean loss for OI when train_dataloader change
if betaX is None or betagX is None:
    running_loss_GOI = 0.
    running_loss_OI = 0.
    num_loss = 0

    for targets_OI, targets_OI1, inputs_Mask, targets_GT in train_dataloader:
        # gradient norm field
        g_targets_GT = gradient_img(targets_GT, phase)
        loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, wLoss)
        loss_GOI = NN_4DVar.compute_WeightedLoss(gradient_img(targets_OI, phase) - g_targets_GT, wLoss)
        running_loss_GOI += loss_GOI.item() * targets_GT.size(0)
        running_loss_OI += loss_OI.item() * targets_GT.size(0)
        num_loss += targets_GT.size(0)

    epoch_loss_GOI = running_loss_GOI / num_loss
    epoch_loss_OI = running_loss_OI / num_loss

    betaX = 1. / epoch_loss_OI
    betagX = 1. / epoch_loss_GOI

    print(".... MSE(Tr) OI %.3f -- MSE(Tr) gOI %.3f " % (epoch_loss_OI, epoch_loss_GOI))
print(f"{(betaX, betagX)=}")



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
        'lrSampling'        : lr_Sampling,
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
        'alpha_SR'          : alpha_SR
    }

    mod = LitModel(phi_r, model_H, model_LR, model_Sampling, gradient_img, mod_args)

    print(mod)
    exit()
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='results', save_top_k = 3)
    # training
    trainer = pl.Trainer(gpus=1, distributed_backend="ddp", **profiler_kwargs)
    trainer.fit(mod, train_dataloader, val_dataloader)
    trainer.test(mod, test_dataloaders=test_dataloader)
