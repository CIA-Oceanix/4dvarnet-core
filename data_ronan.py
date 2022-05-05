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

import os
import sys
#sys.path.append('../4dvarnet-core-old20211112')
#sys.path.append('../4dvarnet-core')
import solver_ronan as NN_4DVar

from sklearn.feature_extraction import image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy import ndimage

import hydra
from hydra.utils import instantiate, get_class, call
def get_cfg(xp_cfg, overrides=None):
    overrides = overrides if overrides is not None else []
    def get():
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp_cfg}',
                'file_paths=jz',
                'entrypoint=train',
            ] + overrides
        )

        return cfg
    try:
        with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
            return get()
    except:
        return get()

def get_model(xp_cfg, ckpt, dm=None, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(xp_cfg, setup=True, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    dm = instantiate(cfg.datamodule)
    if setup:
        dm.setup()
    return dm




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_RND_TRAINING = 0.
USE_RND_TRAINING = False#True#
if USE_RND_TRAINING == False :
    RND_TRAINING = False
else:
    RND_TRAINING = True

print('.... RND Training %s %f'%(USE_RND_TRAINING,VAL_RND_TRAINING))
flagProcess =  0

## NN architectures and optimization parametflagPers
batch_size      = 2#16#4#4#8#12#8#256#
DimAE           = 50#10#10#50
dimGradSolver   = 150#150 # dimension of the hidden state of the LSTM cell
rateDropout     = 0.25 # dropout rate
True#
flag_aug_state = 1#1#1 #False#
dim_aug_state =  0
flag_obs_model = 0#8#1 # 0: no-SST, 1: SST
flag_augment_training_data = True #False#         'modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_01-dgrad150-epoch=20-val_loss=1.95.ckpt''modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_01-dgrad150-epoch=20-val_loss=1.95.ckpt'
flagPhiParam = 1#1#1#4#1
flagAutomOptim = False#True#
width_med_filt_spatial = 1
width_med_filt_temp = 1

scale_dwscaling = 0

# data generation
sigNoise        = 0. ## additive noise standard deviation
flagSWOTData    = True #False ## use SWOT data or not
dT              = 7 ## Time window of each space-time patch
W               = 200#240#200 ## width/height of each space-time patch
dx              = 1  ## subsampling step if > 1
Nbpatches       = 1#5#1#10#10#25 ## number of patches extracted from each time-step 
rnd1            = 0 ## random seed for patch extraction (space sam)
rnd2            = 100 ## random seed for patch extraction
dwscale         = 1
padding_mode = 'reflect'#'zeros' # 
dim_obs_sst = 10

W = int(W/dx)

flag_data_240times240 = False # False# 
#if flag_data_240times240 == True :
#    W = 240
    
def torch_memory(device):
    # Checks and prints GPU memory
    print(f'{torch.cuda.memory_allocated(device)/1024/1024:.2f} MB USED')
    print(f'{torch.cuda.memory_reserved(device)/1024/1024:.2f} MB RESERVED')
    print(f'{torch.cuda.max_memory_allocated(device)/1024/1024:.2f} MB USED MAX')
    print(f'{torch.cuda.max_memory_reserved(device)/1024/1024:.2f} MB RESERVED MAX')
    print('')

UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
InterpFlag         = False # True => force reconstructed field to observed data after each gradient-based update

suffix_exp = "boost-swot"
day_0 = datetime.date(2012,10,1)

if suffix_exp == "exp3" :
    iiVal = 60 
    jjVal = iiVal + 20 + 2*int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "boost-swot" :
    day_val  = datetime.date(2013,1,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    day_test_0  = datetime.date(2012,10,22)
    day_test_1  = datetime.date(2012,12,3)
    iiTest = int((day_test_0 - day_0).days) - int(dT / 2) #90 - int(dT / 2)
    jjTest = int((day_test_1 - day_0).days) + int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = jjVal + 10
    jjTr1 = 365
    
    iiTr2 = 365 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "exp22" :
    day_val  = datetime.date(2012,12,12)
    iiVal = int((day_val - da_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    day_test_0  = datetime.date(2012,10,1)
    day_test_1  = datetime.date(2012,11,12)
    iiTest = int((day_test_0 - day_0).days) #- int(dT / 2) #90 - int(dT / 2)
    jjTest = int((day_test_1 - day_0).days) + int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = jjVal + 10
    jjTr1 = 365
    
    iiTr2 = 365 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "summer" : # Summer
    day_val  = datetime.date(2013,7,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "spring" : # Spring
    day_val  = datetime.date(2013,4,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "winter" : # Winter
    day_val  = datetime.date(2013,1,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "fall" : # Fall
    day_val  = datetime.date(2012,10,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "winter2" : # Winter
    day_val  = datetime.date(2013,1,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 30 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 30 #130 + int(dT / 2)
    jjTr2 = 365
 
############################################## Data generation ###############################################################
print('........ Random seed set to 100')
np.random.seed(100)
torch.manual_seed(100)

dirREF = "/gpfsstore/rech/yrf/commun/NATL60/GULFSTREAM/"
if os.path.isdir(dirREF) == True :            
    dirDATA = dirREF
    dirSAVE = ' ./SLANATL60new3_ChckPt'
else:
    dirDATA = '/users/local/DATA/DataNATL60/GULFSTREAM/'
    #dirSAVE = './SLANATL60new3_ChckPt'
    if flag_data_240times240 == False :    
        dirSAVE = './SLANATL60_GFMMexp_ChckPt'
    else:
        dirSAVE = './SLANATL60_GFMMexp240x240_ChckPt'


if flag_data_240times240 == False :
    ncfile = Dataset(dirDATA+"ref/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc","r")
else:    
    ncfile = Dataset(dirDATA+"ref/NATL60-CJM165_GULFSTREAM240x240_ssh_y2013.1y_new.nc","r")

qHR = ncfile.variables['ssh'][:]
latArea = ncfile.variables['lat'][:]
lonArea = ncfile.variables['lon'][:]
ncfile.close()

if flag_data_240times240 == False :
    ncfile = Dataset(dirDATA+"ref/NATL60-CJM165_GULFSTREAM_sst_y2013.1y.nc","r")
else:
    ncfile = Dataset(dirDATA+"ref/NATL60-CJM165_GULFSTREAM240x240_sst_y2013.1y_new.nc","r")
qSST   = ncfile.variables['sst'][:]
ncfile.close()

if flagSWOTData == True :
    print('.... Use SWOT+4-nadir dataset')
    genFilename  = 'resInterpSLAwSWOT_Exp3_NewSolver_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    if flag_data_240times240 == False :
        ncfile = Dataset(dirDATA+"oi/ssh_NATL60_swot_4nadir.nc","r")
    else :
        ncfile = Dataset(dirDATA+"oi/ssh_NATL60_240x240_swot_4nadir_new.nc","r")
    qOI    = ncfile.variables['ssh_mod'][:]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    #ncfile = Dataset(dirDATA+"data/ssh_NATL60_swot_4nadir.nc","r")
    ncfile = Dataset(dirDATA+"data_new/dataset_nadir_0d_swot.nc","r")
    
    if flag_data_240times240 == True :
        ncfile = Dataset(dirDATA+"data_new/dataset_240x240_nadir_0d_swot.nc","r")
    
    qMask   = ncfile.variables['ssh_mod'][:]
    qMask   = 1.0-qMask.mask.astype(float)

    qObs   = ncfile.variables['ssh_mod'][:]
    #qObs   = ncfile.variables['ssh_obs'][:]
    ncfile.close()

else:
    genFilename  = 'resInterp4NadirSLAwOInoSST_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
    print('.... Use 4-nadir dataset')
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"oi/ssh_NATL60_4nadir.nc","r")
    qOI    = ncfile.variables['ssh_mod'][:]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"data/dataset_nadir_0d.nc","r")
    
    qMask   = ncfile.variables['ssh_mod'][:]
    qMask   = 1.0-qMask.mask.astype(float)
    qObs   = ncfile.variables['ssh_mod'][:]
    ncfile.close()


## extraction of patches from the SSH field
#NoRndPatches = False  
#if ( Nbpatches == 1 ) & ( W == 200 ):
#NoRndPatches = True
print('... No random seed for the extraction of patches')

if flag_data_240times240 == False :
    if 1*0 :
        qHR   = qHR[:,0:200,0:200]
        qOI   = qOI[:,0:200,0:200]
        qMask = qMask[:,0:200,0:200]
        qObs = qObs[:,0:200,0:200]

    else:
        qHR   = qHR[:,1:,1:]
        qOI   = qOI[:,1:,1:]
        qMask = qMask[:,1:,1:]
        qObs = qObs[:,1:,1:]
        
        latArea = latArea[1:]
        lonArea = lonArea[1:]
#else:
if 1*0 :
    qHR   = qHR[:,30:230,30:230]
    qOI   = qOI[:,30:230,30:230]
    qMask = qMask[:,30:230,30:230]
    qObs = qObs[:,30:230,30:230]
    qSST = qSST[:,30:230,30:230]
    
    latArea = latArea[20:220]
    lonArea = lonArea[20:220]


def interpolation_diffusion(x,mask,alpha,n_iter):
    
    x_ = 1. * x
    for iter in range(0,n_iter):
        dx = 0. * x
        for ii in range(1,x.shape[1]-1):
            for jj in range(1,x.shape[2]-1):
                if mask[ii,jj] == 0. :
                    avg_x = x[:,ii,jj+1] + x[:,ii,jj-1] + x[:,ii-1,jj] + + x[:,ii+1,jj]
                    dx[:,ii,jj] = x[:,ii,jj] + alpha * (0.25 * avg_x - x[:,ii,jj])
        
        for ii in range(1,x.shape[1]-1):
            avg_x = x[:,ii,1] + x[:,ii-1,0] + + x[:,ii+1,0]
            dx[:,ii,0] = x[:,ii,0] + alpha * (1./3. * avg_x - x[:,ii,0])
    
            avg_x = x[:,ii,x.shape[2]-1] + x[:,ii-1,-1] + + x[:,ii+1,-1]
            dx[:,ii,-1] =   (1./3. * avg_x - x[:,ii,-1])
    
        for jj in range(1,x.shape[2]-1):
            avg_x = x[:,1,jj] + x[:,0,jj-1] + + x[:,0,jj+1]
            dx[:,0,jj] = x[:,0,jj] + alpha * (1./3. * avg_x - x[:,0,jj])
    
            avg_x = x[:,x.shape[1]-1,jj] + x[:,-1,jj-1] + + x[:,-1,jj+1]
            dx[:,-1,jj] =  (1./3. * avg_x - x[:,-1,jj])
            
        avg_x = x[:,0,1] + x[:,1,0]
        dx[:,0,0] =  (1./2. * avg_x - x[:,0,0])
    
        avg_x = x[:,-1,1] + x[:,x.shape[1]-1,0]
        dx[:,-1,0] =  (1./2. * avg_x - x[:,-1,0])
    
        avg_x = x[:,1,-1] + x[:,0,x.shape[2]-2]
        dx[:,0,-1] =  (1./3. * avg_x - x[:,0,-1])
    
        avg_x = x[:,x.shape[1]-2,-1] + x[:,-1,x.shape[2]-2]
        dx[:,-1,-1] = (1./3. * avg_x - x[:,-1,-1])
        
        x_ = x_ + alpha * x_ * mask
    
    return x
    
def interpolation_diffusion(x,mask,alpha,n_iter):
    
    avg_x = 0. * x
    for iter in range(0,n_iter):
        dx = 0. * x
        
        avg_x[:,1:x.shape[1]-1,1:x.shape[2]-1] = x[:,0:x.shape[1]-2,1:x.shape[2]-1] + x[:,2:x.shape[1],1:x.shape[2]-1]
        avg_x[:,1:x.shape[1]-1,1:x.shape[2]-1] = x[:,1:x.shape[1]-1,0:x.shape[2]-2] + x[:,1:x.shape[1]-1,2:x.shape[2]] + avg_x[:,1:x.shape[1]-1,1:x.shape[2]-1]

        dx[:,1:x.shape[1]-1,1:x.shape[2]-1] = 0.25 * avg_x[:,1:x.shape[1]-1,1:x.shape[2]-1] - x[:,1:x.shape[1]-1,1:x.shape[2]-1]
        
        avg_x[:,1:x.shape[1]-1,0] = x[:,0:x.shape[1]-2,0] + x[:,2:x.shape[1],0]
        avg_x[:,1:x.shape[1]-1,0] = x[:,1:x.shape[1]-1,2] + avg_x[:,1:x.shape[1]-1,0]
        dx[:,1:x.shape[1]-1,0] = ( 1. / 3 ) * avg_x[:,1:x.shape[1]-1,0] - x[:,1:x.shape[1]-1,0]

        avg_x[:,1:x.shape[1]-1,-1] = x[:,0:x.shape[1]-2,-1] + x[:,2:x.shape[1],-1]
        avg_x[:,1:x.shape[1]-1,-1] = x[:,1:x.shape[1]-1,x.shape[2]-2] + avg_x[:,1:x.shape[1]-1,-1]
        dx[:,1:x.shape[1]-1,-1] = ( 1. / 3 ) * avg_x[:,1:x.shape[1]-1,-1] - x[:,1:x.shape[1]-1,-1]

        avg_x[:,0,1:x.shape[2]-1] = x[:,0,0:x.shape[2]-2] + x[:,0,2:x.shape[2]]
        avg_x[:,0,1:x.shape[2]-1] = x[:,2,1:x.shape[2]-1] + avg_x[:,0,1:x.shape[2]-1]
        dx[:,0,1:x.shape[2]-1] = ( 1. / 3 ) * avg_x[:,0,1:x.shape[2]-1] - x[:,0,1:x.shape[2]-1]

        avg_x[:,-1,1:x.shape[2]-1] = x[:,-1,0:x.shape[1]-2] + x[:,-1,2:x.shape[1]]
        avg_x[:,-1,1:x.shape[2]-1] = x[:,x.shape[1]-2,1:x.shape[1]-1] + avg_x[:,-1,1:x.shape[1]-1]
        dx[:,-1,1:x.shape[2]-1] = ( 1. / 3 ) * avg_x[:,1:x.shape[1]-1,0] - x[:,-1,1:x.shape[1]-1]


        x = x + alpha * dx * ( 1. - mask )
    
    return x



if 1*0 :
#mask = 1. + 0. * qHR.mask
#mask[ qHR.mask == True ] = 0.
    qHR[ qHR.mask == True ] = np.nanmean( qHR[ qHR.mask == False ] )
#qHR = interpolation_diffusion(qHR,mask,0.25,1)

    qOI.mask[ np.abs(qOI) > 1e2 ] = True
    qOI[ qOI.mask == True ] = np.nanmean( qOI[ qOI.mask == False ] )
#qOI = interpolation_diffusion(qOI,mask,0.25,1)

    qSST[ qSST.mask == True ] = np.nanmean( qSST[ qSST.mask == False ] )

    qMask[ np.isnan(qMask) ] = 0.
    qObs[ np.isnan(qObs) ] = 0.

    mask_earth = 1. - qHR.mask[0,:,:].astype(float)
else:
    mask_earth = 1. + 0. * qHR[0,0:W,0:W].squeeze()
    qObs[ np.isnan(qObs) ] = 0.
    qMask[ np.isnan(qMask) ] = 0.
    
print('----- MSE OI: %.3f %.3f'%( np.mean((qOI-qHR)**2),np.median((qOI-qHR)**2)))
print()


if width_med_filt_spatial > 1 :
    qHR = ndimage.median_filter(qHR,size=(width_med_filt_temp,width_med_filt_spatial,width_med_filt_spatial))

def extract_SpaceTimePatches(q,i1,i2,W,dT,rnd1,rnd2,D=1):
    dataTraining  = image.extract_patches_2d(np.moveaxis(q[i1:i2,::D,::D], 0, -1),(W,W),max_patches=Nbpatches,random_state=rnd1)
    dataTraining  = np.moveaxis(dataTraining, -1, 1)
    dataTraining  = dataTraining.reshape((Nbpatches,dataTraining.shape[1],W*W)) 
    
    dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),max_patches=None)

    dataTraining  = dataTraining.reshape((dataTraining.shape[0],dataTraining.shape[1],dT,W,W)) 
    dataTraining  = np.moveaxis(dataTraining, 0, -1)
    dataTraining  = np.moveaxis(dataTraining, 0, -1)
    dataTraining  = dataTraining.reshape((dT,W,W,dataTraining.shape[3]*dataTraining.shape[4])) 
    dataTraining  = np.moveaxis(dataTraining, -1, 0)
    return dataTraining     

# training dataset
if jjTr1-iiTr1 >= dT :
    
    print(qHR.shape)
    dataTraining1     = extract_SpaceTimePatches(qHR,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
    dataTrainingMask1 = extract_SpaceTimePatches(qMask,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
    dataTrainingOI1   = extract_SpaceTimePatches(qOI,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
    dataTrainingObs1   = extract_SpaceTimePatches(qObs,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
    dataTrainingSST1 = extract_SpaceTimePatches(qSST,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)

if jjTr2-iiTr2 >= dT :
    dataTraining2     = extract_SpaceTimePatches(qHR,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
    dataTrainingMask2 = extract_SpaceTimePatches(qMask,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
    dataTrainingOI2  = extract_SpaceTimePatches(qOI,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
    dataTrainingObs2  = extract_SpaceTimePatches(qObs,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
    dataTrainingSST2  = extract_SpaceTimePatches(qSST,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)

    
if jjTr1-iiTr1 < dT :
    dataTraining      = dataTraining2
    dataTrainingMask  = dataTrainingMask2
    dataTrainingOI    = dataTrainingOI2    
    dataTrainingObs    = dataTrainingObs2    
    dataTrainingSST   = dataTrainingSST2
elif jjTr2-iiTr2 < dT :
    dataTraining      = dataTraining1
    dataTrainingMask  = dataTrainingMask1
    dataTrainingOI    = dataTrainingOI1    
    dataTrainingObs    = dataTrainingObs1  
    dataTrainingSST   = dataTrainingSST1
else:
    dataTraining      = np.concatenate((dataTraining1,dataTraining2),axis=0)
    dataTrainingMask  = np.concatenate((dataTrainingMask1,dataTrainingMask2),axis=0)
    dataTrainingOI    = np.concatenate((dataTrainingOI1,dataTrainingOI2),axis=0)
    dataTrainingObs    = np.concatenate((dataTrainingObs1,dataTrainingObs2),axis=0)
    dataTrainingSST    = np.concatenate((dataTrainingSST1,dataTrainingSST2),axis=0)

# test dataset
dataTest     = extract_SpaceTimePatches(qHR,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestMask = extract_SpaceTimePatches(qMask,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestOI   = extract_SpaceTimePatches(qOI,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestObs   = extract_SpaceTimePatches(qObs,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestSST  = extract_SpaceTimePatches(qSST,iiTest,jjTest,W,dT,rnd1,rnd2,dx)

# validation dataset
dataVal     = extract_SpaceTimePatches(qHR,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValMask = extract_SpaceTimePatches(qMask,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValOI   = extract_SpaceTimePatches(qOI,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValObs   = extract_SpaceTimePatches(qObs,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValSST  = extract_SpaceTimePatches(qSST,iiVal,jjVal,W,dT,rnd1,rnd2,dx)

meanTr     = np.mean(dataTraining)
x_train    = dataTraining - meanTr
stdTr      = np.sqrt( np.mean( x_train**2 ) )
x_train    = x_train / stdTr

meanSST      = np.mean(dataTrainingSST)
ySST_train   = dataTrainingSST - meanSST
stdSST       = np.sqrt( np.mean( ySST_train**2 ) )
ySST_train   = ySST_train / stdSST

x_trainOI   = (dataTrainingOI - meanTr) / stdTr
x_trainObs  = (dataTrainingObs - meanTr) / stdTr
x_trainMask = dataTrainingMask

x_test     = (dataTest  - meanTr )
stdTt      = np.sqrt( np.mean( x_test**2 ) )
x_test     = x_test / stdTr
x_testOI   = (dataTestOI - meanTr) / stdTr
x_testObs  = (dataTestObs - meanTr) / stdTr
x_testMask  = dataTestMask
ySST_test  = (dataTestSST - meanSST ) / stdSST

x_val     = (dataVal  - meanTr )
stdVal    = np.sqrt( np.mean( x_val**2 ) )
x_val     = x_val / stdTr
x_valOI   = (dataValOI - meanTr) / stdTr
x_valObs   = (dataValObs - meanTr) / stdTr
x_valMask = dataValMask
ySST_val  = (dataValSST - meanSST ) / stdSST

x_trainObs[ x_trainMask == 0 ] = 0.
x_testObs[ x_testMask == 0 ] = 0.
x_valObs[ x_valMask == 0 ] = 0.


if flag_augment_training_data == True :  
    k_augment = 1
    
    for kk in range(0,k_augment):
        # shuffle observation mask for the training data
        ind = np.random.permutation(x_trainOI.shape[0])
   
        x_trainMask_new = x_trainMask[ind,:,:,:]
        x_trainOI_new   = x_trainOI[ind,:,:,:]
        x_train_new = x_train[ind,:,:,:]
        x_trainObs_new = x_train_new * x_trainMask_new
        ySST_train_new = ySST_train[ind,:,:,:]
      
        x_trainOI = np.concatenate((x_trainOI,x_trainOI_new),axis=0)
        x_trainObs = np.concatenate((x_trainObs,x_trainObs_new),axis=0)
        x_trainMask = np.concatenate((x_trainMask,x_trainMask_new),axis=0)
        x_train = np.concatenate((x_train,x_train_new),axis=0)
        ySST_train = np.concatenate((ySST_train,ySST_train_new),axis=0)

print('----- MSE Tr OI: %.6f'%np.mean((dataTrainingOI[:,int(dT/2),:,:]-dataTraining[:,int(dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean((dataTestOI[:,int(dT/2),:,:]-dataTest[:,int(dT/2),:,:])**2))

print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

print('..... Masked points (Tr)) : %.3f'%(np.sum(x_trainMask)/(x_trainMask.shape[0]*x_trainMask.shape[1]*x_trainMask.shape[2]*x_trainMask.shape[3])))
print('..... Masked points (Tt)) : %.3f'%(np.sum(x_testMask)/(x_testMask.shape[0]*x_testMask.shape[1]*x_testMask.shape[2]*x_testMask.shape[3])) )

print('----- MSE Tr OI: %.6f'%np.mean(stdTr**2 * (x_trainOI[:,int(dT/2),:,:]-x_train[:,int(dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean(stdTr**2 * (x_testOI[:,int(dT/2),:,:]-x_test[:,int(dT/2),:,:])**2))

######################### data loaders
if ( flag_obs_model == 0 ) & ( flag_aug_state != 2 ):
    training_dataset   = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainObs),torch.Tensor(x_trainMask),torch.Tensor(x_train)) # create your datset
    val_dataset        = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valObs),torch.Tensor(x_valMask),torch.Tensor(x_val)) # create your datset
    test_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testObs),torch.Tensor(x_testMask),torch.Tensor(x_test))  # create your datset
#elif ( flag_obs_model == 1) |  ( flag_obs_model == 2) |  ( flag_obs_model == 3) |  ( flag_obs_model == 4) :
else:   
    training_dataset   = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainObs),torch.Tensor(x_trainMask),torch.Tensor(ySST_train),torch.Tensor(x_train)) # create your datset
    val_dataset        = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valObs),torch.Tensor(x_valMask),torch.Tensor(ySST_val),torch.Tensor(x_val)) # create your datset
    test_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testObs),torch.Tensor(x_testMask),torch.Tensor(ySST_test),torch.Tensor(x_test))  # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
}            

var_Tr    = np.var( x_train )
var_Tt    = np.var( x_test )
var_Val   = np.var( x_val )

#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')
shapeData      = np.array(x_train.shape[1:])
shapeData_test = np.array(x_test.shape[1:])
if flag_aug_state == 1 :
    shapeData_test[0] += 2*shapeData_test[0]+dim_aug_state
    shapeData[0]  += 2*shapeData[0]+dim_aug_state
elif flag_aug_state == 2 :
    shapeData_test[0] += 3*shapeData_test[0]+dim_aug_state
    shapeData[0]  += 3*shapeData[0]+dim_aug_state
else:
    shapeData[0]  += shapeData[0]
    shapeData_test[0] += shapeData_test[0]
shapeDataSST   = np.array(ySST_train.shape[1:])
        
dW  = 3
dW2 = 1
sS  = int(4/dx)
nbBlocks = 1
rateDr   = 0. * rateDropout

class BiLinUnit(torch.nn.Module):
    def __init__(self,dimIn,dim,dropout=0.):
        super(BiLinUnit, self).__init__()
        
        self.conv1  = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False,padding_mode=padding_mode)
        self.conv2  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.conv3  = torch.nn.Conv2d(2*dim, dimIn, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.dropout  = torch.nn.Dropout(dropout)
        
    def forward(self,xin):
        
        x = self.conv1(xin)
        x = self.dropout(x)
        x = self.conv2( F.relu(x) )
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)),dim=1)
        x = self.dropout(x)
        x = self.conv3( x )
        
        return x    

class BiLinUnit2(torch.nn.Module):
    def __init__(self,dimIn,dim,dropout=0.):
        super(BiLinUnit2, self).__init__()
        
        self.conv1  = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False,padding_mode=padding_mode)
        self.conv2  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.conv3  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False,padding_mode=padding_mode)
        self.dropout  = torch.nn.Dropout(dropout)
        
    def forward(self,xin):
        
        x = self.conv1(xin)
        x = self.dropout(x)
        x = self.conv2( F.relu(x) )
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)),dim=1)
        x = self.dropout(x)
        x = self.conv3( x )
        
        return x    

class Encoder(torch.nn.Module):
    def __init__(self,dimInp,dimAE,rateDropout=0.):
        super(Encoder, self).__init__()

        self.NbBlocks  = nbBlocks
        self.DimAE     = dimAE
        #self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        #self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        self.pool1   = torch.nn.AvgPool2d(sS)
        self.convTr  = torch.nn.ConvTranspose2d(dimInp,dimInp,(sS,sS),stride=(sS,sS),bias=False)          

        #self.NNtLR    = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        #self.NNHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)                      
        self.NNLR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)
        self.NNHR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)                      
        self.dropout  = torch.nn.Dropout(rateDropout)
      
    def __make_BilinNN(self,dimInp,dimAE,Nb_Blocks=2,dropout=0.): 
          layers = []
          layers.append( BiLinUnit(dimInp,dimAE,dropout) )
          for kk in range(0,Nb_Blocks-1):
              layers.append( BiLinUnit(dimInp,dimAE,dropout) )
          return torch.nn.Sequential(*layers)
      
    def forward(self, xinp):
        ## LR comlponent
        xLR = self.NNLR( self.pool1(xinp) )
        xLR = self.dropout(xLR)
        xLR = self.convTr( xLR ) 
        
        # HR component
        xHR = self.NNHR( xinp )

        if ( RND_TRAINING == True ) :
            a = (1. - VAL_RND_TRAINING) * torch.ones(xinp.size(0),1)
            r = torch.bernoulli(a)
            r = r.view(-1,1,1,1)
            r = r.repeat(1,xinp.size(1),xinp.size(2),xinp.size(3))

            r = r.to(device)
            
            return xLR + r*xHR
        else:
            return xLR + xHR
  
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
  
    def forward(self, x):
        return torch.mul(1.,x)



class Encoder2(torch.nn.Module):
    def __init__(self,dimInp,dimAE,rateDropout=0.):
        super(Encoder2, self).__init__()

        self.NbBlocks  = nbBlocks
        self.DimAE     = dimAE
        #self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        #self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        self.scale_factor = sS
        self.pool1   = torch.nn.AvgPool2d(sS)

        self.NNLR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)
        self.NNHR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)                      
        self.dropout  = torch.nn.Dropout(rateDropout)
      
    def __make_BilinNN(self,dimInp,dimAE,Nb_Blocks=2,dropout=0.): 
          layers = []
          layers.append( BiLinUnit(dimInp,dimAE,dropout) )
          for kk in range(0,Nb_Blocks-1):
              layers.append( BiLinUnit(dimInp,dimAE,dropout) )
          return torch.nn.Sequential(*layers)
      
    def forward(self, xinp):
        ## LR comlponent
        xLR = self.NNLR( self.pool1(xinp) )
        xLR = torch.nn.functional.interpolate(xLR, scale_factor=self.scale_factor, mode='bicubic')

        # HR component
        xHR = self.NNHR( xinp )
        
        return xLR + xHR

class Encoder3(torch.nn.Module):
    def __init__(self,dimInp,dimAE,rateDropout=0.):
        super(Encoder3, self).__init__()

        self.NbBlocks  = 1
        self.DimAE     = dimAE
        self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        #self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        self.scale_factor = sS
        self.pool1   = torch.nn.AvgPool2d(sS)
        self.conv_out  = torch.nn.Conv2d(2*dimAE,dimInp, (1,1),padding=0, bias=False)

        self.NNLR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)
        self.NNHR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)                      
        self.dropout  = torch.nn.Dropout(rateDropout)
      
    def __make_BilinNN(self,dimInp,dimAE,Nb_Blocks=2,dropout=0.): 
          layers = []
          layers.append( BiLinUnit2(dimInp,dimAE,dropout) )
          for kk in range(0,Nb_Blocks-1):
              layers.append( BiLinUnit(dimAE,dimAE,dropout) )
          return torch.nn.Sequential(*layers)
      
    def forward(self, xinp):
        ## LR comlponent
        xLR = self.NNLR( self.pool1(xinp) )
        xLR = torch.nn.functional.interpolate(xLR, scale_factor=self.scale_factor, mode='bicubic') 
        xHR = self.NNHR( xinp )
        xout = self.conv_out( torch.cat(( xLR , xHR),dim=1) )
        
        return xout

class Encoder4(torch.nn.Module):
    def __init__(self,dimInp,dimAE,rateDropout=0.):
        super(Encoder4, self).__init__()

        self.NbBlocks  = 1
        self.dw = 2
        self.DimAE     = dimAE

        self.conv_feat  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False,padding_mode=padding_mode)         
        self.conv_feat_hr  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False,padding_mode=padding_mode)         
        
        #self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        self.scale_factor = sS
        self.pool1   = torch.nn.AvgPool2d(sS)
        self.conv_out  = torch.nn.Conv2d(2*self.DimAE,dimInp, (1,1),padding=0, bias=False)
        self.convTr  = torch.nn.ConvTranspose2d(self.DimAE,self.DimAE,(sS,sS),stride=(sS,sS),bias=False)          

        self.conv_lr1  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*self.dw+1,2*self.dw+1),padding=self.dw,bias=False,padding_mode=padding_mode)         
        self.conv_lr2  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*self.dw+1,2*self.dw+1),padding=self.dw,bias=False,padding_mode=padding_mode)         
        self.conv_lr3  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*self.dw+1,2*self.dw+1),padding=self.dw,bias=False,padding_mode=padding_mode)         

        self.conv_hr1  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*self.dw+1,2*self.dw+1),padding=self.dw,bias=False,padding_mode=padding_mode)         
        self.conv_hr2  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*self.dw+1,2*self.dw+1),padding=self.dw,bias=False,padding_mode=padding_mode)         
        self.conv_hr3  = torch.nn.Conv2d(self.DimAE,self.DimAE,(2*self.dw+1,2*self.dw+1),padding=self.dw,bias=False,padding_mode=padding_mode)         

        #self.NNLR1     = self.__make_BilinNN(self.DimAE,self.DimAE,self.NbBlocks,rateDropout)
        #self.NNLR2     = self.__make_BilinNN(self.DimAE,self.DimAE,self.NbBlocks,rateDropout)
        #self.NNHR1     = self.__make_BilinNN(self.DimAE,self.DimAE,self.NbBlocks,rateDropout)                      
        #self.NNHR2     = self.__make_BilinNN(self.DimAE,self.DimAE,self.NbBlocks,rateDropout)                      
        self.dropout  = torch.nn.Dropout(rateDropout)
      
    def __make_BilinNN(self,dimInp,dimAE,Nb_Blocks=2,dropout=0.): 
          layers = []
          layers.append( BiLinUnit(dimInp,dimAE,dropout) )
          for kk in range(0,Nb_Blocks-1):
              layers.append( BiLinUnit(dimAE,dimAE,dropout) )
          return torch.nn.Sequential(*layers)
      
    def forward(self, xinp):
        ## LR comlponent
        x_feat = self.conv_feat( xinp )
        
        x_lr = self.pool1(x_feat)
        x_lr = self.dropout(x_lr)
        #x_lr = x_lr + self.NNLR1( x_lr )
        x_lr = x_lr + F.relu( self.conv_lr1( x_lr ) )
        x_lr = self.dropout(x_lr)
        x_lr = x_lr + F.relu( self.conv_lr2( x_lr ) )
        x_lr = self.dropout(x_lr)
        x_lr = x_lr + F.relu( self.conv_lr3( x_lr ) )
        #x_lr = torch.nn.functional.interpolate(x_lr, scale_factor=self.scale_factor, mode='bicubic')
        x_lr = self.dropout(x_lr)
        x_lr = self.convTr( x_lr )
        
        x_hr = self.conv_feat_hr( x_feat )
        x_hr = self.dropout(x_hr)
        #x_hr = x_hr + self.NNLR1( x_hr )
        x_lr = x_hr + F.relu( self.conv_hr1( x_hr ) )
        x_hr = self.dropout(x_hr)
        x_hr = x_hr + F.relu( self.conv_hr2( x_hr ) )
        x_hr = self.dropout(x_hr)
        x_hr = x_hr + F.relu( self.conv_hr3( x_hr ) )
              
        x_lr = self.dropout(x_lr)
        x_hr = self.dropout(x_hr)
        xout = self.conv_out( torch.cat((x_hr,x_lr),dim=1) )
        
        return xout

class Phi_r(torch.nn.Module):
    def __init__(self):
        super(Phi_r, self).__init__()
        if flagPhiParam == 1 :
            self.encoder = Encoder(shapeData[0],DimAE,rateDr)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

phi_r = Phi_r()

print(phi_r)
print('Number of trainable parameters = %d' % (sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))

class Model_H(torch.nn.Module):
    def __init__(self):
        super(Model_H, self).__init__()
        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData[0]])
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shapeData[0]])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class Model_H1(torch.nn.Module):
    def __init__(self,width_kernel=3,dim=5):
        super(Model_H1, self).__init__()

        self.DimObs = 1
        self.w_kernel = width_kernel
        self.dimObsChannel = np.array([2*dT])

                    
        self.conv11  = torch.nn.Conv2d(2*dT,dT,(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv21  = torch.nn.Conv2d(dT,dT,(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(dT,dT,(3,3),padding=int(self.w_kernel/2),bias=False)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

    def forward(self, x , y , mask):
        
        
        dyout_oi = (x[:,:dT,:] - y[:,:dT,:]) * mask[:,:dT,:]
        
        y1     = y[:,dT:2*dT,:] * mask[:,dT:2*dT,:]
        dyout1 = self.conv11(x[:,0:2*dT,:]) - self.conv21(y1)
        dyout_swot = dyout1 * self.S( self.convM( mask[:,dT:2*dT,:] ) )                  
                
#        if( mask.sizee(1) == 3*dT ):
#            dyout_aug = (x[:,2*dT:3*dT,:] - y[:,2*dT:3*dT,:]) * mask[:,2*dT:3*dT,:]
        dyout = torch.cat((dyout_oi, dyout_swot ), dim=1)
                
        return dyout

class Model_H2(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_H2, self).__init__()

        self.DimObs = 1
        self.w_kernel = width_kernel
        self.dimObsChannel = np.array([shapeData[0]+dim])
                    
        self.conv1_x  = torch.nn.Conv2d(2*dT,dim,(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv1_y  = torch.nn.Conv2d(dT,dim,(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.convM   = torch.nn.Conv2d(dT,dim,(3,3),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        #self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

        #self.conv2_x  = torch.nn.Conv2d(2*dT,dT,(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False)
        #self.conv2_y  = torch.nn.Conv2d(dT,dT,(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False)
        #self.conv2_mask   = torch.nn.Conv2d(dT,dT,(3,3),padding=int(self.w_kernel/2),bias=False)
        #self.conv2_sigmoid = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

    def forward(self, x , y , mask):
                
        dyout = (x - y[0]) * mask[0]    
        
        dyout_swot = self.conv1_x( x[:,:2*dT,:] * mask[0][:,:2*dT,:] ) - self.conv1_y( y[0][:,dT:2*dT,:] * mask[0][:,dT:2*dT,:] )
        dyout_swot = dyout_swot * self.S( self.convM( mask[0][:,dT:2*dT,:] ) )                  

        dyout = torch.cat((dyout, dyout_swot ), dim=1)
                
        return dyout

class Model_HSST(torch.nn.Module):
    def __init__(self,dim=5,dropout=0.25,width_kernel=3):
        super(Model_HSST, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel

        #self.convMssh  = torch.nn.Conv2d(self.dimObsChannel[0],self.dimObsChannel[0],(1,1),padding=0,bias=False,padding_mode=padding_mode)

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

        self.dropout  = torch.nn.Dropout(0.*dropout)

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        #yout = self.convMssh( dyout )
               

        y1     = y[1] * mask[1]
        if scale_dwscaling > 1 :
            y1 = torch.nn.functional.avg_pool2d(y1, (scale_dwscaling,scale_dwscaling))
            y1 = torch.nn.functional.interpolate(y1, scale_factor=scale_dwscaling, mode='bicubic')

        dyout1 = self.conv11(x) - self.conv21(y1)
        #dyout1 = self.dropout( self.conv11(x) - self.conv21(y1) )
        #w = 2. * self.S( self.convM( 1. - mask[1] ) ) - 1.
        #dyout1 = torch.sqrt( 0.1 +  torch.sqrt( 0.1 + dyout1** 2 ) )
        
        w = self.S( self.convM( mask[1] ) )
        #w = self.S( self.convM( 1. - mask[1] ) )

        #print( w[0,:,10,10] )
        #print( w[1,:,20,20] )
        
        dyout1 = dyout1 * w                  
        
        return [dyout,dyout1]

class Model_HSST2(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_HSST2, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM1   = torch.nn.Conv2d(self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        #self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        self.convM2   = torch.nn.Conv2d(self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        
        self.lam_obs_sst  = torch.nn.Parameter(torch.Tensor(1. * np.ones((1,self.dimObsChannel[1]))))
        self.thr_obs_sst  = torch.nn.Parameter(torch.Tensor(0.3 * np.ones((1,self.dimObsChannel[1]))))

        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)        

    def compute_sst_feat(self,y1):
        return self.conv21(y1)
    
    def compute_dyout1(self,x,y1):
        return self.conv11(x) - self.conv21(y1)

    def compute_w(self,dyout1,r=1.):
        for kk in range(0,self.dimObsChannel[1]):
            wkk = ( self.lam_obs_sst[0,kk] * dyout1[:,kk,:,:] ) **2 - self.thr_obs_sst[0,kk]**2
            wkk = wkk.view(-1,1,dyout1.size(2),dyout1.size(3))
            
            if kk == 0 :
                w = 1. * wkk
            else:
                w = torch.cat( (w,wkk) , dim = 1)
                
        #print( torch.nanmean( torch.nanmean( torch.nanmean( self.convM2( 0. * F.relu( r * w ) ) , dim = 3) , dim = 2) , dim = 0 ) )
        #print( torch.nanmean( torch.nanmean( torch.nanmean( self.convM2( - F.relu( r * w ) ) , dim = 3) , dim = 2) , dim = 0 ) )
        #print( w[0,:,10,10] )
        w = self.S( self.convM2( - F.relu( r * w ) ) )

        return w        
    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 

        y1     = y[1] * mask[1]

        if 1*0 :
            y1 = torch.nn.functional.avg_pool2d(y1, (5,5))
            y1 = torch.nn.functional.interpolate(y1, scale_factor=5, mode='bicubic')

        
        dyout1 = self.compute_dyout1(x,y1)

        #w = self.S( self.convM2( F.relu( self.convM1( y1 ) ) ) )
        
        if ( RND_TRAINING == True ) :
            a = (1. - VAL_RND_TRAINING)*torch.ones(y1.size(0),1)
            r = torch.bernoulli(a)
            r = r.view(-1,1,1,1)
            r = r.repeat(1,self.dimObsChannel[1],y1.size(2),y1.size(3))
        else:
            r = torch.ones(y1.size(0),self.dimObsChannel[1],y1.size(2),y1.size(3))
        r = r.to(device)
        #w = self.S( self.convM2( - F.relu( r * self.convM1( dyout1 )**2 - 0.5 ) ) )
        #w = self.S( self.convM2( - F.relu( r * self.convM1( dyout1 )**2 - 0.1 ) ) )
                
        #print('-------')
        #print( torch.mean( torch.mean( torch.mean( dyout1 **2 , dim = 3) , dim = 2) , dim = 0 ) )
        w = self.compute_w(dyout1,r)
        if 1*0 :
            for kk in range(0,self.dimObsChannel[1]):
                wkk = ( self.lam_obs_sst[0,kk] * dyout1[:,kk,:,:] ) **2 - self.thr_obs_sst[0,kk]**2
                wkk = wkk.view(-1,1,dyout1.size(2),dyout1.size(3))
                
                if kk == 0 :
                    w = 1. * wkk
                else:
                    w = torch.cat( (w,wkk) , dim = 1)
                
                w = self.S( self.convM2( - F.relu( r * w ) ) )
        
        dyout1 = dyout1 * w                  
        
        return [dyout,dyout1]

class Model_HSSTfiltered(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_HSSTfiltered, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]
        
        y1 = torch.nn.functional.avg_pool2d(y1, (5,5))
        y1 = torch.nn.functional.interpolate(y1, scale_factor=5, mode='bicubic')
        
        dyout1 = self.conv11(x) - self.conv21( y1)
        dyout1 = dyout1 * self.S( self.convM( mask[1] ) )                  
        
        return [dyout,dyout1]


class Model_HSST_MM(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_HSST_MM, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

        self.lam = torch.nn.Parameter(0.1 * torch.randn(1, self.dimObsChannel[1],1,1), requires_grad=True)#

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]
                
        dyout1 = self.conv11(x) - self.conv21( y1)
        w = self.lam ** 2
        w = w.repeat(y1.size(0),1,y1.size(2),y1.size(3))
        
        
        w = torch.exp( - w * dyout1**2 )
        sw = torch.sum( w , 1 ).view(-1,1,y1.size(2),y1.size(3)).repeat(1,dyout1.size(1),1,1) 
        w = w / sw            

        dyout1 = w * dyout1 
        
        return [dyout,dyout1]

class Model_HSSTBilin(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_HSSTBilin, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel


        self.convx1  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convx2  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.convy1  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convy2  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.convxy  = torch.nn.Conv2d(self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=False,padding_mode=padding_mode)

        self.convM1   = torch.nn.Conv2d(self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)        

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]

        #y1 = torch.nn.functional.avg_pool2d(y1, (4,4))
        #y1 = torch.nn.functional.interpolate(y1, scale_factor=4, mode='bicubic')

        #x = torch.nn.functional.avg_pool2d(y1, (4,4))
        #x = torch.nn.functional.interpolate(y1, scale_factor=4, mode='bicubic')

        dyout1 = self.convy1(y1) + self.convx1(x) + self.convxy( self.convx2(x) * self.convy2(y1) ) 
        #w = self.S( self.convM2( F.relu( self.convM1( y1 ) ) ) )

        w = self.S( self.convM( mask[1] ) )
        
        #if ( RND_TRAINING == True ) :
        #    a = (1. - VAL_RND_TRAINING)*torch.ones(y1.size(0),1)
        #    r = torch.bernoulli(a)
        #    r = r.view(-1,1,1,1)
        #    r = r.repeat(1,2*self.dimObsChannel[1],y1.size(2),y1.size(3))
        #else:
        #    r = torch.ones(y1.size(0),2*self.dimObsChannel[1],y1.size(2),y1.size(3))
        #r = r.to(device)
        #w = self.S( self.convM2( F.relu( 0.1 * r * self.convM1( dyout1**2 ) ) ) )
        #w = torch.nn.functional.avg_pool2d(w, (4,4))
        #w = torch.nn.functional.interpolate(w, scale_factor=4, mode='bicubic')
                
        #print( w[0,:,10,10] )
        #print( w[1,:,20,20] )
        
        dyout1 = dyout1 * w                  
        
        return [dyout,dyout1]

class Model_HSSTBilin2(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_HSSTBilin2, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0],dim])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.convx1  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convx2  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convx3  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.convy1  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convy2  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convy3  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.convxy  = torch.nn.Conv2d(self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=False,padding_mode=padding_mode)

        #self.convM1   = torch.nn.Conv2d(self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],2*self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)    
        

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]

        dyout1 = self.convy1(y1) - self.convx1(x)
        
        dyout2 = self.convx2(x) * self.convy2(y1) + self.convx3(x) * self.convy3(y1) 
        
        dyout1 = torch.cat((dyout1,dyout2),dim=1)

        w = self.S( self.convM( mask[1] ) )
                
        dyout1 = dyout1 * w                  
        
        return [dyout,dyout1]

class Model_HSSTnolin(torch.nn.Module):
    def __init__(self,dim=5,width_kernel=3):
        super(Model_HSSTnolin, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.w_kernel = width_kernel

        self.convx11  = torch.nn.Conv2d(shapeData[0],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convx12  = torch.nn.Conv2d(2*self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.convx21  = torch.nn.Conv2d(2*self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convx22  = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
                    
        self.convy11  = torch.nn.Conv2d(shapeDataSST[0],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convy12  = torch.nn.Conv2d(2*self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        
        self.convy21  = torch.nn.Conv2d(2*self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convy22  = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        
        #self.convy31  = torch.nn.Conv2d(self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convy32  = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)

        self.convM1   = torch.nn.Conv2d(self.dimObsChannel[1],2*self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=True,padding_mode=padding_mode)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False,padding_mode=padding_mode)
        #self.convM2   = torch.nn.Conv2d(2*self.dimObsChannel[1],self.dimObsChannel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)        

    def extract_sst_feature(self,y1):
        y1     = self.convy12( torch.tanh( self.convy11(y1) ) )
        y1     = self.convy22( torch.tanh( self.convy21( torch.tanh(y1) ) ) )
        
        return y1
        
    def extract_state_feature(self,x):
        x1     = self.convx12( torch.tanh( self.convx11(x) ) )
        x1     = self.convx22( torch.tanh( self.convx21( torch.tanh(x1) ) ) )
        
        return x1

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]
        
        if scale_dwscaling > 1 :
            y1 = torch.nn.functional.avg_pool2d(y1, (scale_dwscaling,scale_dwscaling))
            y1 = torch.nn.functional.interpolate(y1, scale_factor=scale_dwscaling, mode='bicubic')

        if 1*0 :
            y1     = self.convy12( F.relu( self.convy11(y1) ) )
            y1     = self.convy22( F.relu( self.convy21( F.relu(y1) ) ) )
                
            x1     = self.convx12( F.relu( self.convx11(x) ) )
            x1     = self.convx22( F.relu( self.convx21( F.relu(x1) ) ) )
        else:
            y1 = self.extract_sst_feature(y1)
            x1 = self.extract_state_feature(x)
            #y1     = self.convy12( F.tanh( self.convy11(y1) ) )
            #y1     = self.convy22( F.tanh( self.convy21( F.tanh(y1) ) ) )
                
            #x1     = self.convx12( F.tanh( self.convx11(x) ) )
            #x1     = self.convx22( F.tanh( self.convx21( F.tanh(x1) ) ) )

        #dyout1 = y1 - self.convx1(x)
        dyout1 = y1 - x1

        w = self.S( self.convM( mask[1] ) )
        
        dyout1 = dyout1 * w                  
        
        return [dyout,dyout1]

class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = 0.25 * np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

        b = 0.25 * np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, im):

        if im.size(1) == 1:
            G_x = self.convGx(im)
            G_y = self.convGy(im)
            nG = torch.sqrt(torch.pow( G_x, 2) + torch.pow( G_y, 2))
        else:

            for kk in range(0, im.size(1)):
                G_x_ = self.convGx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                G_y_ = self.convGy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                G_x_ = G_x_.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                G_y_ = G_y_.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                nG_ = torch.sqrt(torch.pow( G_x_, 2) + torch.pow( G_y_, 2))

                if kk == 0:
                    nG = nG_.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                    Gx = G_x_.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                    Gy = G_y_.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                else:
                    nG = torch.cat((nG, nG_.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
                    Gx = torch.cat((Gx, G_x_.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
                    Gy = torch.cat((Gy, G_y_.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
        #return G
        return nG,Gx,Gy

gradient_img = Gradient_img()

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)

#model_LR = ModelLR()
alpha_MSE     = 0.1
alpha_Proj    = 0.5
alpha_SR      = 0.5
alpha_LR      = 0.5  # 1e4

# loss weghing wrt time
w_ = np.zeros(dT)
w_[int(dT / 2)] = 1.
wLoss = torch.Tensor(w_)


# recompute the MSE for OI on training dataset
# to define weighing parameters in the training
if 1*0 :
    betaX  = None#42.20436766972647
    betagX = None#77.99700321505073
    if betaX is None or betagX is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_loss_GOI = 0.
        running_loss_OI = 0.
        num_loss = 0
        
        gradient_img = gradient_img.to(device)
        wLoss        = wLoss.to(device)
    
        for targets_OI, imputs_obs, inputs_Mask, targets_GT in dataloaders['train']:
            targets_OI     = targets_OI.to(device)
            inputs_Mask    = inputs_Mask.to(device)
            targets_GT     = targets_GT.to(device)
            
            # gradient norm field
            g_targets_GT,gx,gy = gradient_img(targets_GT)
            loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, wLoss)
            loss_GOI = NN_4DVar.compute_WeightedLoss(gradient_img(targets_OI)[0] - g_targets_GT, wLoss)
            running_loss_GOI += loss_GOI.item() * targets_GT.size(0)
            running_loss_OI += loss_OI.item() * targets_GT.size(0)
            num_loss += targets_GT.size(0)
    
        epoch_loss_GOI = running_loss_GOI / num_loss
        epoch_loss_OI = running_loss_OI / num_loss
    
        betaX  = 1. / epoch_loss_OI
        betagX = 1. / epoch_loss_GOI
    
        print(".... MSE(Tr) OI %.3f -- MSE(Tr) gOI %.3f " % (epoch_loss_OI, epoch_loss_GOI))
        print(".... betaX = %.3f -- betagX %.3f " % (betaX, betagX))
    #print(f"{(betaX, betagX)=}")

def save_NetCDF(saved_path1, ind_start,ind_end,ssh_gt , ssh_oi, ssh_obs, ssh_rec,sst_feat,lat,lon):
    
    #extent = [-65., -55., 33., 43.]
    #indLat = 200
    #indLon = 200

    #lon = np.arange(extent[0], extent[1], 1 / (20 / dwscale))
    #lat = np.arange(extent[2], extent[3], 1 / (20 / dwscale))
    #indLat = int(indLat / dwscale)
    #indLon = int(indLon / dwscale)
    #lon = lon[:indLon]
    #lat = lat[:indLat]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    indN_Tt = np.concatenate([np.arange(ind_start, ind_end)])
    time_ = [datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]

    #xrdata = xr.Dataset( \
    #    data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
    #               'latitude': (('lat', 'lon'), mesh_lat), \
    #               'Time': (('time'), time), \
    #               'ssh': (('time', 'lat', 'lon'), x_test_rec[:, int(dT / 2), :, :])}, \
    #    coords={'lon': lon, 'lat': lat, 'time': indN_Tt})

    if sst_feat.all() == None :
        xrdata = xr.Dataset( \
            data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                       'latitude': (('lat', 'lon'), mesh_lat), \
                       'Time': (('time'), time_), \
                       'ssh_gt': (('time', 'lat', 'lon'), ssh_gt), \
                       'ssh_oi': (('time', 'lat', 'lon'), ssh_oi), \
                       'ssh_obs': (('time', 'lat', 'lon'), ssh_obs), \
                       'ssh_rec': (('time', 'lat', 'lon'), ssh_rec)}, \
            coords={'lon': lon, 'lat': lat, 'time': indN_Tt},)
    else:
        xrdata = xr.Dataset( \
            data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                       'latitude': (('lat', 'lon'), mesh_lat), \
                       'Time': (('time'), time_), \
                       'ssh_gt': (('time', 'lat', 'lon'), ssh_gt), \
                       'ssh_oi': (('time', 'lat', 'lon'), ssh_oi), \
                       'ssh_obs': (('time', 'lat', 'lon'), ssh_obs), \
                       'ssh_rec': (('time', 'lat', 'lon'), ssh_rec), \
                       'sst_feat': (('time', 'feat', 'lat', 'lon'), sst_feat)}, \
            coords={'lon': lon, 'lat': lat, 'time': indN_Tt,'feat':np.arange(sst_feat.shape[1])},)
        
        
    xrdata.time.attrs['units'] = 'days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')


## Laplacian
def compute_laplacian(x):
    
    if len( x.shape ) == 2 :
        lap = x[1:-1,1:-1] 
        lap = lap - 0.25 * x[1:-1,0:x.shape[1]-2]
        lap = lap - 0.25 * x[1:-1,2:x.shape[1]]
        lap = lap - 0.25 * x[0:x.shape[0]-2,1:-1]
        lap = lap - 0.25 * x[2:x.shape[0],1:-1]
    else:
        lap = x[:,1:-1,1:-1] 
        lap = lap - 0.25 * x[:,1:-1,0:x.shape[2]-2]
        lap = lap - 0.25 * x[:,1:-1,2:x.shape[2]]
        lap = lap - 0.25 * x[:,0:x.shape[1]-2,1:-1]
        lap = lap - 0.25 * x[:,2:x.shape[1],1:-1]

    
    return lap

from scipy.ndimage import gaussian_filter
def compute_laplacian_metrics(x,x_ref,sig_lap=1):

    lap_ref = compute_laplacian( gaussian_filter(x_ref, sigma=sig_lap))
    lap_rec = compute_laplacian( gaussian_filter(x, sigma=sig_lap))

    mse_lap = np.mean((lap_ref-lap_rec)**2)
    var_lap = np.var(lap_ref)
    
    R2 = np.corrcoef(lap_ref.ravel(), lap_rec.ravel())[0,1]
    
    return {'mse':mse_lap,'var_lap': var_lap,'r_square': R2}

############################################Lightning Module#######################################################################
class HParam:
    def __init__(self):
        self.iter_update     = []
        self.nb_grad_update  = []
        self.lr_update       = []
        self.n_grad          = 5
        self.dim_grad_solver = 10
        self.dropout         = 0.25
        self.median_filter_width = 3
        self.w_loss          = []
        self.automatic_optimization = True
        self.k_batch         = 1
        self.alpha_proj    = 0.5
        self.alpha_sr      = 0.5
        self.alpha_lr      = 0.5  # 1e4
        self.alpha_mse_ssh = 10.
        self.alpha_mse_gssh = 1.
        self.alpha_fft = 1.
        self.dw_loss = 10        
        #self.eps_norm_grad = 0.
        self.max_rate_fft = 1.5
        self.thr_snr_fft = 0.5
        self.ifft_max = 15
        self.obs_model = flag_obs_model
  
hparam = HParam()
print( hparam.obs_model , flush = True)
EPS_NORM_GRAD = 0. * 1.e-20  
class LitModel(pl.LightningModule):
    #def __init__(self,conf=hparam,*args, **kwargs):
    def __init__(self,conf=flag_obs_model,*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        #print( '... %d'%conf.obs_model , flush = True)
        self.hparams.obs_model = conf

        # hyperparameters
        self.hparams.iter_update     = [0, 200, 50, 70, 100, 150, 800]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        self.hparams.k_batch         = 1
        
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        self.hparams.k_n_grad        = 3
        self.hparams.dim_grad_solver = dimGradSolver
        self.hparams.dropout         = rateDropout
        self.hparams.median_filter_width = 1
        
        #self.hparams.eps_norm_grad = 0.
        
        self.hparams.alpha_proj    = 0.5
        self.hparams.alpha_sr      = 0.5
        self.hparams.alpha_lr      = 0.5  # 1e4
        self.hparams.alpha_mse_ssh = 1.e1
        self.hparams.alpha_mse_gssh = 1.e3
        self.hparams.alpha_fft = 0. 
        self.hparams.dw_loss = 5
        self.hparams.alpha_4dvarloss_diff = 0.1
        
        self.hparams.max_rate_fft = 1.5
        self.hparams.thr_snr_fft = 0.5
        self.hparams.ifft_max = 15

        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)
        self.hparams.automatic_optimization = flagAutomOptim

        # main model        
        if self.hparams.obs_model == 0:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_H(), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD,k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad) )#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 1:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSST(dim_obs_sst,dropout=self.hparams.dropout), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 2:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSST2(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 3:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSSTfiltered(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 4:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSST_MM(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 5:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSSTBilin(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 6:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSSTnolin(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)
        elif self.hparams.obs_model == 7:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_HSSTBilin2(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)

        elif self.hparams.obs_model == 8:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_H2(dim_obs_sst), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)

        else:
            self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                             Model_H(), 
                                                             NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode=padding_mode), 
                                                             None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD, k_step_grad = 1. / (self.hparams.n_grad * self.hparams.k_n_grad))#, self.hparams.eps_norm_grad)



        self.model_LR     = ModelLR()
        self.gradient_img = Gradient_img()
        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_rec_ssh        = None # variable to store output of test method
        self.x_rec_ssh_obs = None
        self.x_feat_sst = None
        self.curr = 0
        self.t_mask_earth = torch.Tensor( mask_earth ).to(device)

        self.automatic_optimization = self.hparams.automatic_optimization
        print('Total number of trainable parameters = %d' % (sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        

    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        if 1*1 :
            #optimizer   = optim.Adam(self.model.parameters(), lr = self.lrUpdate[0])
            optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                      {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                                    ], lr=0.)
        elif 1*0:
            optimizer   = optim.RMSprop([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.hparams.lr_update[0]},
                                    ], lr=0.)

        else:
            optimizer   = optim.ASGD([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.hparams.lr_update[0]},
                                    ], lr=0.)
        return optimizer
    
    def on_epoch_start(self):
        # enfore acnd check some hyperparameters 
        #self.model.n_grad   = self.hparams.k_n_grad * self.hparams.n_grad 
        self.model.k_step_grad = 1. / np.maximum( 1.0 * self.hparams.k_n_grad * self.hparams.n_grad , 0.1)
        #self.model.k_step_grad = 1. / self.hparams.n_grad
        
    def on_train_epoch_start(self):
        if USE_RND_TRAINING == True :
                RND_TRAINING = True
        self.model.n_grad   = self.hparams.n_grad 

        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx             = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparams.nb_grad_update[indx],self.hparams.lr_update[indx]))
            
            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad   = self.hparams.n_grad 
            
            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent,lrCurrent,0.5*lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics
        loss, out, metrics,diff_loss_4dvar_init = self.compute_loss(train_batch, phase='train')
                
        if self.hparams.k_n_grad > 1 :
            loss_all = loss + self.hparams.alpha_4dvarloss_diff * diff_loss_4dvar_init
            for kk in range(0,self.hparams.k_n_grad-1):
                loss1, out, metrics,diff_loss_4dvar_init = self.compute_loss(train_batch, phase='train',batch_init=out[2],hidden=out[3],cell=out[4],normgrad=out[5])

                dloss = F.relu(loss1 - loss)
                loss = 1. * loss1                 
                loss_all = loss_all + loss1 +  dloss + self.hparams.alpha_4dvarloss_diff * diff_loss_4dvar_init
                loss_all = loss_all + loss1 + self.hparams.alpha_4dvarloss_diff * diff_loss_4dvar_init

            loss =  loss_all
        
        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mseG", metrics['mseGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.hparams.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        if USE_RND_TRAINING == True :
            RND_TRAINING = False
        #opt = self.optimizers()
                    
        # compute loss and metrics
        #loss, out, metrics = self.compute_loss(val_batch, phase='train')        #opt.zero_grad()
                
        loss, out, metrics, diff_loss_4dvar_init = self.compute_loss(val_batch, phase='val')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics, diff_loss_4dvar_init = self.compute_loss(val_batch, phase='val',batch_init=out[2],hidden=out[3],cell=out[4],normgrad=out[5])
            loss = loss1

        #self.log('val_loss', loss)
        self.log('val_loss', self.hparams.alpha_mse_ssh * metrics['mse'] + self.hparams.alpha_mse_gssh * metrics['mseGrad'] )
        self.log("val_mse", metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mseG", metrics['mseGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        if USE_RND_TRAINING == True :
            RND_TRAINING = True #False

        loss, out, metrics, diff_loss_4dvar_init = self.compute_loss(test_batch, phase='test')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics, diff_loss_4dvar_init = self.compute_loss(test_batch, phase='test',batch_init=out[2],hidden=out[3],cell=out[4],normgrad=out[5])

        #out_ssh,out_ssh_obs = out
        #self.log('test_loss', loss)
        self.log('test_loss', self.hparams.alpha_mse_ssh * metrics['mse'] + self.hparams.alpha_mse_gssh * metrics['mseGrad'] )
        self.log("test_mse", metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mseG", metrics['mseGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mseFFT", metrics['mseFFT'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #return {'preds': out_ssh.detach().cpu(),'obs_ssh': out_ssh_obs.detach().cpu()}
        return {'preds_ssh': out[0].detach().cpu(),'obs_ssh': out[1].detach().cpu(),'feat_sst': out[6].detach().cpu()}
        #if self.hparams.obs_model > 0 : #( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
        ##    return {'preds_ssh': out[0].detach().cpu(),'obs_ssh': out[1].detach().cpu(),'feat_sst': out[6].detach().cpu()}
        #else:
        #    return {'preds_ssh': out[0].detach().cpu(),'obs_ssh': out[1].detach().cpu(),'feat_sst': out[6].detach().cpu()}
            
    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds_ssh'] for chunk in outputs]).numpy()
        x_test_rec = stdTr * x_test_rec + meanTr        
        self.x_rec_ssh = x_test_rec[:,int(dT/2),:,:]

        x_test_ssh_obs = torch.cat([chunk['obs_ssh'] for chunk in outputs]).numpy()
        x_test_ssh_obs[ x_test_ssh_obs == 0. ] = np.float('NaN')
        x_test_ssh_obs = stdTr * x_test_ssh_obs + meanTr
        self.x_rec_ssh_obs = x_test_ssh_obs[:,int(dT/2),:,:]

        #if self.hparams.obs_model > 0 : #( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
        x_test_sst_feat = torch.cat([chunk['feat_sst'] for chunk in outputs]).numpy()
        self.x_feat_sst = x_test_sst_feat

        return [{'mse':meanTr,'preds': meanTr}]

    def compute_4DvarCost(self,x,yobs,mask):
        dy = self.model.model_H(x,yobs,mask)
        dx = x - self.model.phi_r(x)
        
        loss_4dvar = self.model.model_VarCost( dx , dy )
        
        return loss_4dvar

    def compute_loss(self, batch, phase, batch_init = None , hidden = None , cell = None , normgrad = 0.0):

        if ( self.hparams.obs_model == 0 ) & ( flag_aug_state != 2 ) :
            # targets_OI, inputs_obs, inputs_Mask, targets_GT = batch
            targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        else:# self.hparams.obs_model > 0 : #( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
            targets_OI, inputs_obs, inputs_Mask, inputs_SST, targets_GT = batch
            mask_SST  = 1. + 0. * inputs_SST
        
        if flag_aug_state == 1 :
            if dim_aug_state == 0 :
                inputs_init_    = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , inputs_Mask * (inputs_obs - targets_OI)), dim=1)
                inputs_missing = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , 0. * targets_OI), dim=1)
                new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask , 0. * inputs_Mask ), dim=1)
            else  :
                init_aug_state = 0. * torch.randn((inputs_obs.size(0),dim_aug_state,inputs_obs.size(2),inputs_obs.size(3))).to(device)
                inputs_aug_state = 0. * init_aug_state
                inputs_init_    = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , inputs_Mask * (inputs_obs - targets_OI),init_aug_state), dim=1)
                inputs_missing = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , 0. * targets_OI, inputs_aug_state ), dim=1)
                new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask , 0. * inputs_Mask, inputs_aug_state ), dim=1)
        elif flag_aug_state == 2 :
            if dim_aug_state == 0 :
                inputs_init_    = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , inputs_Mask * (inputs_obs - targets_OI),inputs_SST), dim=1)
                inputs_missing = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , 0. * targets_OI,inputs_SST), dim=1)
                new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask , 0. * inputs_Mask ,  1. + 0. * inputs_SST ), dim=1)
            else  :
                init_aug_state = 0. * torch.randn((inputs_obs.size(0),dim_aug_state,inputs_obs.size(2),inputs_obs.size(3))).to(device)
                inputs_aug_state = 0. * init_aug_state
                inputs_init_    = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , inputs_Mask * (inputs_obs - targets_OI),init_aug_state), dim=1)
                inputs_missing = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , 0. * targets_OI, inputs_aug_state ), dim=1)
                new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask , 0. * inputs_Mask, inputs_aug_state ), dim=1)
        else:
            inputs_init_    = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI)), dim=1)
            inputs_missing = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI)), dim=1)
            #inputs_missing = torch.cat((targets_OI, inputs_Mask * (targets_GT - targets_OI)), dim=1)
            #inputs_init    = torch.cat((targets_OI, inputs_Mask * (targets_GT - targets_OI)), dim=1)
            new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask ), dim=1)

        if batch_init == None :
            inputs_init = inputs_init_
        else:
            inputs_init = batch_init.detach()

        # gradient norm field
        g_targets_GT,gx_targets_GT,gy_targets_GT = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
        #with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
            if self.model.n_grad > 0 : 
                if self.hparams.obs_model == 0 :
                    outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, new_masks,hidden = hidden , cell = cell , normgrad = normgrad)
                else: #elif ( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
                    #outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, [inputs_missing,inputs_SST], [new_masks,mask_SST])#,hidden = hidden , cell = cell , normgrad = normgrad)
                    outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, [inputs_missing,inputs_SST], [new_masks,mask_SST],hidden = hidden , cell = cell , normgrad = normgrad)

                #torch_memory(device.index)

                #print(self.model.model_H.lam)
                outputsSLRHR = outputs
                outputsSLR   = outputs[:, 0:dT, :, :]
                if flag_aug_state == True :
                    outputs      = outputsSLR + outputs[:, 2*dT:3*dT, :, :]
                else:
                    outputs      = outputsSLR + outputs[:, dT:2*dT, :, :]

                #if (phase == 'val') or (phase == 'test'):
                #    outputs = outputs.detach()
    
                # median filter
                if self.hparams.median_filter_width > 1 :
                    #outputs = 0.1 * outputs + 0.9 * medianFilter2D.median_blur_copy_padding(outputs, (self.hparams.median_filter_width, self.hparams.median_filter_width))
                    outputs = medianFilter2D.median_blur(outputs, (self.hparams.median_filter_width, self.hparams.median_filter_width))
                    #outputs = outputs_
                
                # reconstruction losses
                g_outputs,gx_outputs,gy_outputs  = self.gradient_img(outputs)
                
                dw = self.hparams.dw_loss
    
                loss_All   = NN_4DVar.compute_WeightedLoss(self.t_mask_earth[dw:outputs.size(2)-dw,dw:outputs.size(3)-dw] * (outputs[:,:,dw:outputs.size(2)-dw,dw:outputs.size(3)-dw] - targets_GT[:,:,dw:outputs.size(2)-dw,dw:outputs.size(3)-dw]), self.w_loss)
    
                loss_GAll  = NN_4DVar.compute_WeightedLoss( self.t_mask_earth[dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] * (gx_outputs[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] - gx_targets_GT[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw]), self.w_loss)
                loss_GAll += NN_4DVar.compute_WeightedLoss( self.t_mask_earth[dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] * (gy_outputs[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] - gy_targets_GT[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw]), self.w_loss)
    
                loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
                loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI)[0] - g_targets_GT, self.w_loss)
    
                # projection losses
                loss_AE     = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
                if flag_aug_state == 1 :
                    if dim_aug_state == 0 :
                        yGT         = torch.cat((targets_OI , targets_GT - outputsSLR, targets_GT - outputsSLR ), dim=1)
                    else:
                        yGT         = torch.cat((targets_OI , targets_GT - outputsSLR, targets_GT - outputsSLR, outputsSLRHR[:,3*dT:,:] ), dim=1)                        
                elif flag_aug_state == 2 :
                    if dim_aug_state == 0 :
                        yGT         = torch.cat((targets_OI , targets_GT - outputsSLR, targets_GT - outputsSLR, inputs_SST ), dim=1)
                    else:
                        yGT         = torch.cat((targets_OI , targets_GT - outputsSLR, targets_GT - outputsSLR, inputs_SST, outputsSLRHR[:,4*dT:,:] ), dim=1)                        
                else:
                    yGT         = torch.cat((targets_GT, targets_GT - outputsSLR), dim=1)
                #yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
                loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)
    
                # low-resolution loss
                loss_SR      = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.w_loss)
                targets_GTLR = self.model_LR(targets_OI)
                loss_LR      = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.w_loss)
    
                # spectral loss
                err = self.t_mask_earth * (outputs[:,int(dT/2),:,:] - targets_GT[:,int(dT/2),:,:]).view(-1,1,targets_GT.size(2),targets_GT.size(3))
                
                #fft_err = torch.fft.rfft2(err,dim=(2,3),norm="ortho")
                #fft_gt = torch.fft.rfft2( targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=(2,3),norm="ortho")
                
                window = torch.hann_window(targets_GT.size(2)).repeat(targets_GT.size(0),1,targets_GT.size(3))
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(device)
                fft_err = torch.fft.rfft(window * err,dim=2,norm="ortho")
                fft_gt = torch.fft.rfft( window * targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=2,norm="ortho")
                #loss_fft1 = torch.mean(fft_err.abs(),dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs(),dim=3) )
                loss_fft1 = torch.mean(fft_err.abs()**2,dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs()**2,dim=3) )
    
                window = torch.hann_window(targets_GT.size(3)).repeat(targets_GT.size(0),targets_GT.size(2),1)
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(device)
                fft_err = torch.fft.rfft(window * err,dim=3,norm="ortho")
                fft_gt = torch.fft.rfft( window * targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=3,norm="ortho")
                #loss_fft2 = torch.mean(fft_err.abs(),dim=2) / (0.*1e-15 + torch.mean(fft_gt.abs(),dim=2) )
                loss_fft2 = torch.mean(fft_err.abs()**2,dim=2) / (0.*1e-15 + torch.mean(fft_gt.abs()**2,dim=2) )
               
                loss_fft1 = loss_fft1[:,0,:self.hparams.ifft_max]
                loss_fft2 = loss_fft2[:,0,:self.hparams.ifft_max]
                
                #print( loss_fft1 )
                #print( loss_fft2 )
                #loss_fft = 0.5 * ( torch.mean( F.relu( self.hparams.max_rate_fft - F.relu( self.hparams.max_rate_fft - loss_fft1 ) - self.hparams.thr_snr_fft ) ) )
                #loss_fft += 0.5 * torch.mean( F.relu( self.hparams.max_rate_fft - F.relu( self.hparams.max_rate_fft - loss_fft2 ) - self.hparams.thr_snr_fft ) )
                            
                
                loss_fft = 0.5 * torch.mean( F.relu( loss_fft1 - 0.5 ) ) + 0.5 * torch.mean( F.relu( loss_fft2 - 0.5 ) )
                #loss_fft += 0.5 * torch.mean( F.relu( self.hparams.max_rate_fft - F.relu( self.hparams.max_rate_fft - loss_fft2 ) - self.hparams.thr_snr_fft ) )

                # total loss
                loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
                loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
                loss += self.hparams.alpha_fft * loss_fft
                print(f'ronan {loss_All=}, {self.hparams.alpha_mse_ssh * loss_All=}')
                print(f'ronan {loss_GAll=}, {self.hparams.alpha_mse_gssh * loss_GAll=}')
                print(f'ronan {loss_AE=}, {0.5 * self.hparams.alpha_proj * loss_AE=} ')
                print(f'ronan {loss_AE_GT=}, {0.5 * self.hparams.alpha_proj * loss_AE_GT=} ')
                print(f'ronan {loss_LR=}, { self.hparams.alpha_lr *loss_LR=}')
                print(f'ronan {loss_SR=}, { self.hparams.alpha_sr *loss_SR=}')
                print(f'ronan {loss_fft=} {self.hparams.alpha_fft * loss_fft=}')
                print(f'ronan {loss=} ')
                # loss 4dVar before/after iteration
                if self.hparams.obs_model == 0 :
                    diff_loss_4dvar_init = self.compute_4DvarCost(outputsSLRHR, inputs_missing, new_masks) - self.compute_4DvarCost(inputs_init, inputs_missing, new_masks)            
                else: #elif ( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
                    diff_loss_4dvar_init = self.compute_4DvarCost(outputsSLRHR, [inputs_missing,inputs_SST], [new_masks,mask_SST]) - self.compute_4DvarCost(inputs_init, [inputs_missing,inputs_SST], [new_masks,mask_SST])            
                diff_loss_4dvar_init = F.relu( diff_loss_4dvar_init )
            else:
                if flag_aug_state > 0 :
                    inputs_direct_inv = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , inputs_SST), dim=1)
                else:
                    inputs_direct_inv = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) ), dim=1)
                    
                outputsSLRHR = None 
                hidden_new = None 
                cell_new = None
                normgrad = 0.
                
                outputs = self.model.phi_r(inputs_direct_inv)
                
                outputs = outputs[:, dT:2*dT, :, :] + targets_OI
    
                if (phase == 'val') or (phase == 'test'):
                    outputs = outputs.detach()
    
                # median filter
                if self.hparams.median_filter_width > 1 :
                    #outputs = 0.1 * outputs + 0.9 * medianFilter2D.median_blur_copy_padding(outputs, (self.hparams.median_filter_width, self.hparams.median_filter_width))
                    outputs = medianFilter2D.median_blur(outputs, (self.hparams.median_filter_width, self.hparams.median_filter_width))
                    #outputs = outputs_
                
                # reconstruction losses
                g_outputs,gx_outputs,gy_outputs  = self.gradient_img(outputs)
                
                dw = self.hparams.dw_loss
    
                loss_All   = NN_4DVar.compute_WeightedLoss((outputs[:,:,dw:outputs.size(2)-dw,dw:outputs.size(3)-dw] - targets_GT[:,:,dw:outputs.size(2)-dw,dw:outputs.size(3)-dw]), self.w_loss)
    
                loss_GAll  = NN_4DVar.compute_WeightedLoss(gx_outputs[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] - gx_targets_GT[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw], self.w_loss)
                loss_GAll += NN_4DVar.compute_WeightedLoss(gy_outputs[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] - gy_targets_GT[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw], self.w_loss)
    
                diff_loss_4dvar_init = 0.
    
                # spectral loss
                err = (outputs[:,int(dT/2),:,:] - targets_GT[:,int(dT/2),:,:]).view(-1,1,targets_GT.size(2),targets_GT.size(3))
                
                window = torch.hann_window(targets_GT.size(2)).repeat(targets_GT.size(0),1,targets_GT.size(3))
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(device)
                fft_err = torch.fft.rfft(window * err,dim=2,norm="ortho")
                fft_gt = torch.fft.rfft( window * targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=2,norm="ortho")
                #loss_fft1 = torch.mean(fft_err.abs(),dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs(),dim=3) )
                loss_fft1 = torch.mean(fft_err.abs()**2,dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs()**2,dim=3) )
    
                window = torch.hann_window(targets_GT.size(3)).repeat(targets_GT.size(0),targets_GT.size(2),1)
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(device)
                fft_err = torch.fft.rfft(window * err,dim=3,norm="ortho")
                fft_gt = torch.fft.rfft( window * targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=3,norm="ortho")
                #loss_fft2 = torch.mean(fft_err.abs(),dim=2) / (0.*1e-15 + torch.mean(fft_gt.abs(),dim=2) )
                loss_fft2 = torch.mean(fft_err.abs()**2,dim=2) / (0.*1e-15 + torch.mean(fft_gt.abs()**2,dim=2) )
               
                loss_fft1 = loss_fft1[:,0,:self.hparams.ifft_max]
                loss_fft1 = loss_fft2[:,0,:self.hparams.ifft_max]
                loss_fft = 0.5 * ( torch.mean( F.relu( self.hparams.max_rate_fft - F.relu( self.hparams.max_rate_fft - loss_fft1 ) - self.hparams.thr_snr_fft ) ) )
                loss_fft += 0.5 * torch.mean( F.relu( self.hparams.max_rate_fft - F.relu( self.hparams.max_rate_fft - loss_fft2 ) - self.hparams.thr_snr_fft ) )
                            
                # total loss
                loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                loss_OI = 0. * loss
                loss_GOI = 0. * loss
                
            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT,self.w_loss)
            mse       = loss_All.detach()
            mseGrad   = loss_GAll.detach()  
            metrics   = dict([('mse',mse),('mseGrad',mseGrad),('meanGrad',mean_GAll),('mseOI',loss_OI.detach()),('mseGOI',loss_GOI.detach()),('mseFFT',loss_fft.detach())])
            #print(mse.cpu().detach().numpy())
            
            if 1*0 :
                if self.current_epoch == 1:
                    self.curr = 1
                    
                if self.curr == self.current_epoch :
                    #print(' %e %e'%(torch.mean( torch.abs(div_gt)),torch.mean( torch.abs(div_rec))))
                    #print(' %e %e'%(torch.mean( torch.abs(curl_gt)),torch.mean( torch.abs(curl_rec))))
                    print(' loss_fft = %f'%(loss_fft.item()))
                    print(loss_fft2[0,:].detach().cpu().numpy()  )
                    #print( np.mean( (div_gt.detach().cpu().numpy())**2 ), flush=True )
                    #print( np.mean( (div_rec.detach().cpu().numpy())**2 ), flush=True )
    
                    self.curr += 1
        if ( self.hparams.obs_model== 1 ) :
            sst_feat = self.model.model_H.conv21( inputs_SST )
            sst_feat = torch.cat( (inputs_SST[:,int(dT/2),:,:].view(-1,1,sst_feat.size(2),sst_feat.size(3)),sst_feat) , dim=1)
        elif ( self.hparams.obs_model== 2 ) :
            dyout1 = self.model.model_H.compute_dyout1( outputsSLRHR.detach() , inputs_SST )
            w = self.model.model_H.compute_w( dyout1 )            
            sst_feat = torch.cat( (self.model.model_H.compute_sst_feat( inputs_SST ) , w ) , dim = 1 )
            if 1*0 :
                print(self.model.model_H.thr_obs_sst )
                print(self.model.model_H.lam_obs_sst )
                #print(self.model.model_VarCost.WReg )
                print(self.model.model_H.conv21 )
        elif ( self.hparams.obs_model== 3 ) :
            filt_SST = torch.nn.functional.avg_pool2d(inputs_SST, (5,5))
            filt_SST = torch.nn.functional.interpolate(filt_SST, scale_factor=5, mode='bicubic')            
            sst_feat = self.model.model_H.conv21( filt_SST )
            sst_feat = torch.cat( (inputs_SST[:,int(dT/2),:,:].view(-1,1,sst_feat.size(2),sst_feat.size(3)),sst_feat) , dim=1)
        elif ( self.hparams.obs_model== 4 ) :
            sst_feat = self.model.model_H.conv21( inputs_SST )
            sst_feat = torch.cat( (inputs_SST[:,int(dT/2),:,:].view(-1,1,sst_feat.size(2),sst_feat.size(3)),sst_feat) , dim=1)
        elif ( self.hparams.obs_model== 5 ) :
            sst_feat = self.model.model_H.convy1( inputs_SST )
            sst_feat = torch.cat( (inputs_SST[:,int(dT/2),:,:].view(-1,1,sst_feat.size(2),sst_feat.size(3)),sst_feat) , dim=1)
        elif ( self.hparams.obs_model== 6 ) :
            sst_feat = self.model.model_H.extract_sst_feature( inputs_SST )
            sst_feat = torch.cat( (inputs_SST[:,int(dT/2),:,:].view(-1,1,sst_feat.size(2),sst_feat.size(3)),sst_feat) , dim=1)
        elif ( self.hparams.obs_model== 7 ) :
            sst_feat = self.model.model_H.convy1( inputs_SST )
        elif ( self.hparams.obs_model== 8 ) :
            sst_feat = inputs_SST
            #dyout1 = self.model.model_H.compute_dyout1( outputsSLRHR.detach() , inputs_SST )
            #w = self.model.model_H.compute_w( dyout1 )            
        else:
            sst_feat = 0. * inputs_obs
        outputs = [outputs,inputs_obs,outputsSLRHR,hidden_new, cell_new, normgrad,sst_feat]#inputs_missing[:,dT:2*dT,:,:]]
        
        
        
        return loss,outputs, metrics, diff_loss_4dvar_init

def compute_metrics(X_test,X_rec,mask = 1.):
    # MSE
        
    mse = np.mean( (X_test - X_rec)**2 * mask )

    # MSE for gradient
    gX_rec = np.gradient(X_rec,axis=[1,2])
    ngX_rec = np.sqrt(gX_rec[0]**2 +  gX_rec[1]**2)
    
    gX_test = np.gradient(X_test,axis=[1,2])
    ngX_test = np.sqrt(gX_test[0]**2 +  gX_test[1]**2)
    
    gmse = np.mean( (gX_test[0] - gX_rec[0])**2+(gX_test[1] - gX_rec[1])**2 )
    ngmse = np.mean( (ngX_test - ngX_rec)**2 )
    ng   = np.mean( (ngX_test)**2 )
    
    return {'mse':mse,'mseNGrad': ngmse,'mseGrad': gmse,'meanGrad': ng}

def rmse_based_scores(ds_oi, ds_ref):
    
    # RMSE(t) based score
    rmse_t = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('lon', 'lat')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('lon', 'lat')))**0.5
    # RMSE(x, y) based score
    # rmse_xy = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
    rmse_xy = (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
    
    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')

    # Temporal stability of the error
    reconstruction_error_stability_metric = rmse_t.std().values

    # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
    leaderboard_rmse = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig']) ** 2).mean()) ** 0.5 / (
        ((ds_ref['sossheig']) ** 2).mean()) ** 0.5

    
    return rmse_t, rmse_xy, np.round(leaderboard_rmse.values, 3), np.round(reconstruction_error_stability_metric, 3)

import numpy
import xrft
import matplotlib.pyplot as plt


def psd_based_scores(ds_oi, ds_ref):
            
    # Compute error = SSH_reconstruction - SSH_true
    err = (ds_oi['sossheig'] - ds_ref['sossheig'])
    err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
    
    # make time vector in days units 
    err['time'] = (err.time - err.time[0]) #/ numpy.timedelta64(1, 'D')
    
    # Rechunk SSH_true
    signal = ds_ref['sossheig'].chunk({"lat":1, 'time': ds_ref['time'].size, 'lon': ds_ref['lon'].size})
    # make time vector in days units
    signal['time'] = (signal.time - signal.time[0]) #/ numpy.timedelta64(1, 'D')

    # Compute PSD_err and PSD_signal
    #psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='linear', window=True).compute()
    #psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='linear', window=True).compute()
    psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='constant', window=True).compute()
    psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='constant', window=True).compute()
    
    # Averaged over latitude
    mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
    mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)
    
    # return PSD-based score
    psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

    # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score 

    level = [0.5]
    
    cs = plt.contour(1./psd_based_score.freq_time.values,1./psd_based_score.freq_lon.values, psd_based_score, level)
    y05, x05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()
    
    #cs = plt.contour(1./psd_based_score.freq_lon.values,1./psd_based_score.freq_time.values, psd_based_score, level)
    #x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    #plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)
    
    if 1*0 :
        mean_psd_err_spatial = 1. -  np.mean( mean_psd_err.values , axis = 1) / np.mean( mean_psd_signal.values , axis = 1)
        print( 1. / psd_signal.freq_lon )
        print('.... spatial scale %f'%np.min( np.min( psd_signal.freq_lon[ np.where( mean_psd_err_spatial >= 0.5 ) ] ) ) )
        
        mean_psd_err_time = 1. - np.mean( mean_psd_err.values , axis = 0) / np.mean( mean_psd_signal.values , axis = 0)
        print('.... time scale %f'%np.min( psd_signal.freq_time[ np.where( mean_psd_err_time >= 0.5 ) ] ) )
    
#    logging.info('          => Leaderboard Spectral score = %s (degree lon)',
#                 numpy.round(shortest_spatial_wavelength_resolved, 2))
#    logging.info('          => shortest temporal wavelength resolved = %s (days)',
#                 numpy.round(shortest_temporal_wavelength_resolved, 2))

    return (1.0 - mean_psd_err/mean_psd_signal), np.round(shortest_spatial_wavelength_resolved, 3), np.round(shortest_temporal_wavelength_resolved, 3)


if __name__ == '__main__':
        
    if flagProcess == 0: ## training model from scratch
    
        loadTrainedModel = False#  True# 

        if loadTrainedModel == True :             

            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_05-boost-swot-dT07-igrad05_02-dgrad150-epoch=104-val_loss=0.79.ckpt'
            
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTobs_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=71-val_loss=0.76.ckpt'
            #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTobs_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=73-val_loss=0.79.ckpt'
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTobs_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=48-val_loss=0.88.ckpt'
            
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_01-boost-swot-dT07-igrad05_03-dgrad150-epoch=97-val_loss=1.00.ckpt'
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_02-boost-swot-dT07-igrad05_03-dgrad150-epoch=76-val_loss=0.80.ckpt'

            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_01-boost-swot-dT07-igrad05_03-dgrad150-epoch=89-val_loss=1.00.ckpt'
            #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=97-val_loss=0.73.ckpt'
            
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate_sst-boost-swot-dT07-igrad05_03-dgrad150-epoch=98-val_loss=0.83.ckpt'
            #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=23-val_loss=1.79.ckpt'
            
            
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=82-val_loss=0.69.ckpt'
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_01-boost-swot-dT07-igrad05_03-dgrad150-epoch=97-val_loss=0.99.ckpt'
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=96-val_loss=0.79.ckpt'
            
            pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=23-val_loss=1.79.ckpt'        
            
            pathCheckPOint = 'modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=23-val_loss=1.79.ckpt' 
            print('.... load pre-trained model :'+pathCheckPOint)
            
            if 1*1 :
                mod = LitModel.load_from_checkpoint(pathCheckPOint,conf = flag_obs_model )
            elif flag_obs_model == 7 :
                mod_ = LitModel.load_from_checkpoint(pathCheckPOint,conf = 1 )
     
                mod = LitModel( conf = flag_obs_model )
                mod.model.model_Grad = mod_.model.model_Grad
                mod.model.model_VarCost = mod_.model.model_VarCost
    
                mod.model.phi_r = mod_.model.phi_r
                #mod.model.model_H = mod_.model.model_H
                mod.model.model_H.conv11 = mod_.model.model_H.conv11
                mod.model.model_H.conv12 = mod_.model.model_H.conv12
                mod.model.model_H.conv21 = mod_.model.model_H.conv21
                mod.model.model_H.conv22 = mod_.model.model_H.conv22
                mod.model.model_H.S = mod_.model.model_H.S
                
                w = torch.cat( (mod_.model.model_H.convM.weight[:dim_obs_sst,:,:,:],0.1 * mod.model.model_H.convM.weight[dim_obs_sst:,:,:,:]),dim=0)
                
                mod.model.model_H.convM.weight = torch.nn.Parameter( w )
                print( mod.model.model_H.convM.weight.size() )
                
            elif flag_obs_model == 7 :
                mod_ = LitModel.load_from_checkpoint(pathCheckPOint,conf = 1 )
     
                mod = LitModel( conf = flag_obs_model )
                mod.model.model_Grad = mod_.model.model_Grad
                mod.model.model_VarCost = mod_.model.model_VarCost
    
                mod.model.phi_r = mod_.model.phi_r
                #mod.model.model_H = mod_.model.model_H
                mod.model.model_H.convx1 = mod_.model.model_H.conv11
                mod.model.model_H.convy1 = mod_.model.model_H.conv21
                mod.model.model_H.S = mod_.model.model_H.S
                                
                w = torch.cat( (mod_.model.model_H.convM.weight[:dim_obs_sst,:,:,:], 1. * mod.model.model_H.convM.weight[dim_obs_sst:,:,:,:]),dim=0)
                mod.model.model_H.convM.weight = torch.nn.Parameter( w )
            else:
                mod_ = LitModel.load_from_checkpoint(pathCheckPOint,conf = 1 )
     
                mod = LitModel( conf = flag_obs_model )
                mod.model.model_Grad = mod_.model.model_Grad
                mod.model.model_VarCost = mod_.model.model_VarCost
    
                mod.model.phi_r = mod_.model.phi_r
                #mod.model.model_H = mod_.model.model_H
                mod.model.model_H.conv11 = mod_.model.model_H.conv11
                mod.model.model_H.conv12 = mod_.model.model_H.conv12
                mod.model.model_H.conv21 = mod_.model.model_H.conv21
                mod.model.model_H.conv22 = mod_.model.model_H.conv22
                mod.model.model_H.S = mod_.model.model_H.S
                
                mod.model.model_H.convM2.bias = torch.nn.Parameter( torch.sum(  torch.sum( torch.sum( mod_.model.model_H.convM.weight , 1) , 1) , 1) )
                
                print(".... trained biases: ")
                print(mod.model.model_H.convM2.bias )
            mod.hparams.n_grad          = 5
            mod.hparams.k_n_grad        = 3
            mod.hparams.iter_update     = [0, 100, 200, 300, 120, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [5, 5, 5, 15, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-3, 1e-5, 1e-6, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-7]

            mod.hparams.alpha_proj    = 0.5
            mod.hparams.alpha_sr      = 0.5
            mod.hparams.alpha_lr      = 0.5  # 1e4
            mod.hparams.alpha_mse_ssh = 5.e1
            mod.hparams.alpha_mse_gssh = 1.e3
            mod.hparams.alpha_4dvarloss_diff = 0.
            
            mod.hparams.alpha_fft = 0.
            mod.max_rate_fft = 1.5
            mod.hparams.thr_snr_fft = 0.5
            mod.hparams.ifft_max = 15
            
            mod.hparams.median_filter_width = 0
            mod.hparams.dw_loss = 20
        else:
            mod = LitModel( conf = flag_obs_model )

            mod.hparams.n_grad          = 5
            mod.hparams.k_n_grad        = 3
            mod.hparams.iter_update     = [0, 200, 200, 320, 380, 400, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]

            mod.hparams.alpha_proj    = 0.5
            mod.hparams.alpha_sr      = 0.5
            mod.hparams.alpha_lr      = 5  # 1e4
            mod.hparams.alpha_mse_ssh = 5.e1
            mod.hparams.alpha_mse_gssh = 1.e3#1.e4#
            mod.hparams.alpha_4dvarloss_diff = 1.
            
            mod.hparams.alpha_fft = 0.
            mod.max_rate_fft = 1.5
            mod.hparams.thr_snr_fft = 0.5
            mod.hparams.ifft_max = 15

            mod.hparams.median_filter_width = 0
            mod.hparams.dw_loss = 20

        #filename_chkpt = 'modelSLA-L2-GF-norm-grad-eps1e-6-'
        filename_chkpt = 'modelSLA-L2-GF-'#'FineTuning2-'#'orig9-'#'OptTrue'
        
        if loadTrainedModel == True :
            filename_chkpt = filename_chkpt+'FT-'
            
        if flagPhiParam > 1 :
            filename_chkpt = filename_chkpt + 'Unet%d-'%flagPhiParam
        elif flagPhiParam == 1 :
            if USE_RND_TRAINING == True :
                filename_chkpt = filename_chkpt+'RndTr%02d-'%(10.*VAL_RND_TRAINING)

        if mod.hparams.automatic_optimization == True :
            filename_chkpt = filename_chkpt + 'AutomOpt-'            
            
        if flag_augment_training_data == True:
            filename_chkpt = filename_chkpt + 'augdata%02d'%k_augment+'-'            
                       
        if flag_aug_state == 1 :
            if dim_aug_state > 0:
                filename_chkpt = filename_chkpt + 'augstate%02d-'%dim_aug_state            
            else:               
                filename_chkpt = filename_chkpt + 'augstate-'
                
            if scale_dwscaling > 1 :
                filename_chkpt = filename_chkpt + '%d-'%scale_dwscaling
        elif flag_aug_state == 2 :
            if dim_aug_state > 0:
                filename_chkpt = filename_chkpt + 'augstate_sst%02d-'%dim_aug_state            
            else:               
                filename_chkpt = filename_chkpt + 'augstate_sst-'            
                
        if flag_obs_model == 1 :
            filename_chkpt = filename_chkpt + 'SSTobs_%02d-'%dim_obs_sst           
        elif flag_obs_model == 2 :
            filename_chkpt = filename_chkpt + 'SSTobs2_%02d-'%dim_obs_sst           
        elif flag_obs_model == 3 :
            filename_chkpt = filename_chkpt + 'MWSSTobs_%02d-'%dim_obs_sst           
        elif flag_obs_model == 4 :
            filename_chkpt = filename_chkpt + 'MWSSTobsMM_%02d-'%dim_obs_sst           
        elif flag_obs_model == 5 :
            filename_chkpt = filename_chkpt + 'SSTbilin_%02d-'%dim_obs_sst           
        elif flag_obs_model == 6 :
            filename_chkpt = filename_chkpt + 'SSTnolin-tanh_%02d-'%dim_obs_sst           
        elif flag_obs_model == 7 :
            filename_chkpt = filename_chkpt + 'SSTbilinv2_%02d-'%dim_obs_sst           
        elif flag_obs_model == 8 :
            filename_chkpt = filename_chkpt + 'HSWOT_%02d-'%dim_obs_sst           
            
        if flagSWOTData == False:
            filename_chkpt = filename_chkpt + 'nadir-only-'            

        if mod.hparams.alpha_fft > 0.:
            filename_chkpt = filename_chkpt + 'fft%03d-'%(int(mod.hparams.alpha_fft))           
            
        if mod.hparams.median_filter_width+width_med_filt_spatial+width_med_filt_temp > 3.:
            filename_chkpt = filename_chkpt + 'median%02d_%02d_%02d-'%(int(mod.hparams.median_filter_width),width_med_filt_spatial,width_med_filt_temp)           

        filename_chkpt = filename_chkpt + suffix_exp
        filename_chkpt = filename_chkpt+'-dT%02d'%dT
        
        if mod.hparams.n_grad > 0.:
            filename_chkpt = filename_chkpt+'-igrad%02d_%02d'%(mod.hparams.n_grad,mod.hparams.k_n_grad)+'-dgrad%d'%dimGradSolver
        else:
            filename_chkpt = filename_chkpt+'-DirectInv'

        #filename_chkpt = filename_chkpt+'-L%.1f_%.1f_%.1fd'%(mod.hparams.p_norm_loss,mod.hparams.q_norm_loss,mod.hparams.r_norm_loss)
        
        print('..... Filename chkpt: '+filename_chkpt)        
        print(mod.hparams)
        print('n_grad = %d'%mod.hparams.n_grad)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= dirSAVE+'-'+suffix_exp,
                                              filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 200 }

        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs,callbacks=[checkpoint_callback])
    
        ## training loop
        from omegaconf import OmegaConf
        OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
        cfg = get_cfg("xp_aug/xp_repro/quentin_repro")

        # dm = instantiate(cfg.datamodule)
        # dm.setup()
        # dataloaders = {
        #         'test': dm.test_dataloader(),
        #         'val': dm.val_dataloader(),
        #         'train': dm.train_dataloader(),
        # }
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, dataloaders=dataloaders['val'])
        
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                
        val_mseRec = compute_metrics(X_val,mod.x_rec_ssh)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, dataloaders=dataloaders['test'])
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
    
            
        test_mseRec = compute_metrics(X_test,mod.x_rec_ssh)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))
    
    elif flagProcess == 1: ## profling 

        mod = LitModel()
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
        mod.NbGradIter[0] = mod.NbGradIter[-1]
        profiler_kwargs = {
            'profiler': profiler,
            'max_epochs': 1,
        }
    
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])

        
    elif flagProcess == 2: ## test trained model with the non-Lighning code
        mod = LitModel()
        fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_100_modelPHI_iter080.mod'
        
        mod.model.phi_r.load_state_dict(torch.load(fileAEModelInit))
        mod.model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelGrad_iter')))
        mod.model.model_VarCost.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelVarCost_iter')))
        mod.model.NGrad = 10
    
        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        #ncfile = Dataset("results/test.nc","r")
        #X_rec    = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                
        val_mseRec = compute_metrics(X_val,mod.x_rec)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
                
        test_mseRec = compute_metrics(X_test,mod.x_rec)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad'])))

    elif flagProcess == 3: ## test trained model with the Lightning code
        
        pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=23-val_loss=1.79.ckpt'                
        pathCheckPOint = 'SLANATL60_GFMMexp240x240_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=04-val_loss=1.33.ckpt'
        
        pathCheckPOint = 'SLANATL60_GFMMexp240x240_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=42-val_loss=1.28.ckpt'
        
        pathCheckPOint = 'SLANATL60_GFMMexp240x240_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=31-val_loss=0.57.ckpt'
        
        pathCheckPOint = 'SLANATL60_GFMMexp240x240_ChckPt-boost-swot/modelSLA-L2-GF-Unet5-augdata01-boost-swot-dT07-DirectInv-epoch=101-val_loss=1.89.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp240x240_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=51-val_loss=0.64.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_02-dgrad150-epoch=51-val_loss=1.81.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_01-boost-swot-dT07-igrad05_03-dgrad150-epoch=57-val_loss=1.05.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_01-boost-swot-dT07-igrad05_03-dgrad150-epoch=97-val_loss=1.00.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_02-boost-swot-dT07-igrad05_03-dgrad150-epoch=76-val_loss=0.80.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=71-val_loss=0.76.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=73-val_loss=0.79.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=48-val_loss=0.88.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_10-boost-swot-dT07-igrad05_02-dgrad150-epoch=92-val_loss=0.78.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTobs_03-boost-swot-dT07-igrad05_02-dgrad150-epoch=98-val_loss=0.80.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-FT-augdata01-augstate-SSTobs_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=01-val_loss=0.78.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate-SSTobs_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=01-val_loss=0.75.ckpt'     
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate-SSTobs_02-boost-swot-dT07-igrad05_03-dgrad150-epoch=04-val_loss=0.78.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-FT-augdata01-augstate-SSTobs_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=01-val_loss=0.78.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate-SSTobs_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=05-val_loss=0.75.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate-SSTobs_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=03-val_loss=0.86.ckpt'

        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate_sst-SSTobs_03-boost-swot-dT07-igrad05_02-dgrad150-epoch=99-val_loss=0.82.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate_sst-SSTobs_05-boost-swot-dT07-igrad05_02-dgrad150-epoch=87-val_loss=0.79.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate_sst-SSTobs_10-boost-swot-dT07-igrad05_02-dgrad150-epoch=99-val_loss=0.80.ckpt'
        
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate_sst-SSTobs_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=60-val_loss=0.89.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate_sst-SSTobs_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=61-val_loss=0.89.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate_sst-SSTobs_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=45-val_loss=0.91.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_02-dgrad150-epoch=51-val_loss=1.81.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=23-val_loss=1.79.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTobs_02-boost-swot-dT07-igrad05_03-dgrad150-epoch=44-val_loss=0.89.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_01-boost-swot-dT07-igrad05_03-dgrad150-epoch=97-val_loss=0.99.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_02-boost-swot-dT07-igrad05_03-dgrad150-epoch=79-val_loss=0.88.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_03-boost-swot-dT07-igrad05_03-dgrad150-epoch=96-val_loss=0.79.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=97-val_loss=0.73.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-augdata01-augstate-SSTnolin-tanh_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=82-val_loss=0.69.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate-SSTnolin-tanh_10-boost-swot-dT07-igrad05_03-dgrad150-epoch=05-val_loss=0.69.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/SaveModels/modelSLA-L2-GF-FT-augdata01-augstate-SSTnolin-tanh_05-boost-swot-dT07-igrad05_03-dgrad150-epoch=14-val_loss=0.71.ckpt'
        
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-augdata01-augstate_sst-boost-swot-dT07-igrad05_03-dgrad150-epoch=98-val_loss=0.83.ckpt'
        #pathCheckPOint = 'SLANATL60_GFMMexp_ChckPt-boost-swot/modelSLA-L2-GF-FT-augdata01-augstate_sst-boost-swot-dT07-igrad05_03-dgrad150-epoch=09-val_loss=0.81.ckpt'
        
        if 1*1 :
            mod = LitModel.load_from_checkpoint(pathCheckPOint,conf = flag_obs_model )  
        elif flag_obs_model == 7 :
            mod_ = LitModel.load_from_checkpoint(pathCheckPOint,conf = 1 )
 
            mod = LitModel( conf = flag_obs_model )
            mod.model.model_Grad = mod_.model.model_Grad
            mod.model.model_VarCost = mod_.model.model_VarCost

            mod.model.phi_r = mod_.model.phi_r
            #mod.model.model_H = mod_.model.model_H
            mod.model.model_H.convx1 = mod_.model.model_H.conv11
            mod.model.model_H.convy1 = mod_.model.model_H.conv21
            mod.model.model_H.S = mod_.model.model_H.S
            
            print( mod.model.model_H.convM.weight.size() )
            
            for kk in range(0,5):
                print( mod_.model.model_H.S( torch.sum( mod_.model.model_H.convM.weight[kk,:,:,:] ) ) )
            w = torch.cat( (mod_.model.model_H.convM.weight[:dim_obs_sst,:,:,:], 1. * mod.model.model_H.convM.weight[dim_obs_sst:,:,:,:]),dim=0)
            mod.model.model_H.convM.weight = torch.nn.Parameter( w )
                        
            for kk in range(0,2*dim_obs_sst):
                print( mod.model.model_H.S( torch.sum( mod.model.model_H.convM.weight[kk,:,:,:] ) ) )

        else:
            mod_ = LitModel.load_from_checkpoint(pathCheckPOint,conf = 1 )
 
            mod = LitModel( conf = 2 )
            mod.model.model_Grad = mod_.model.model_Grad
            mod.model.model_VarCost = mod_.model.model_VarCost

            mod.model.phi_r = mod_.model.phi_r
            #mod.model.model_H = mod_.model.model_H
            mod.model.model_H.conv11 = mod_.model.model_H.conv11
            mod.model.model_H.conv12 = mod_.model.model_H.conv12
            mod.model.model_H.conv21 = mod_.model.model_H.conv21
            mod.model.model_H.conv22 = mod_.model.model_H.conv22
            mod.model.model_H.S = mod_.model.model_H.S
            
            mod.model.model_H.convM2.bias = torch.nn.Parameter( torch.sum(  torch.sum( torch.sum( mod_.model.model_H.convM.weight , 1) , 1) , 1) )
                #mod.model.model_H.convM2.weight = torch.nn.Parameter( 1. * mod.model.model_H.convM2.weight )
            
            #mod.model.model_H.convM = mod_.model.model_H.convM
            #mod = LitModel.load_from_checkpoint(pathCheckPOint,conf = 1 )

            #mod.model.model_H.convM2 = mod_.model.model_H.convM

        mod.hparams.n_grad = 5#5
        mod.hparams.k_n_grad = 3
        
        mod.hparams.alpha_proj    = 0.5
        mod.hparams.alpha_sr      = 0.5
        mod.hparams.alpha_lr      = 0.5  # 1e4
        mod.hparams.alpha_mse_ssh = 5.e1
        mod.hparams.alpha_mse_gssh = 1.e3
        mod.hparams.dw_loss = 20
            
        mod.hparams.alpha_fft = 0.
        mod.hparams.median_filter_width = 1
    
        print('... model: '+pathCheckPOint)
        mod.model.n_grad = mod.hparams.n_grad
        print(' Ngrad = %d / %d'%(mod.hparams.n_grad,mod.model.n_grad))
        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, dataloaders=dataloaders['val'])
        
        width_med_filt = 0
        dw_grad = 20
        if width_med_filt > 2 :
            mod.x_rec_ssh = ndimage.median_filter(mod.x_rec_ssh,size=(1,width_med_filt,width_med_filt))

        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                            
        val_mseRec = compute_metrics(X_val[:,dw_grad:X_val.shape[1]-dw_grad,dw_grad:X_val.shape[2]-dw_grad],mod.x_rec_ssh[:,dw_grad:X_val.shape[1]-dw_grad,dw_grad:X_val.shape[2]-dw_grad])     
        val_mseOI  = compute_metrics(X_val[:,dw_grad:X_val.shape[1]-dw_grad,dw_grad:X_val.shape[2]-dw_grad],X_OI[:,dw_grad:X_val.shape[1]-dw_grad,dw_grad:X_val.shape[2]-dw_grad])     
        val_norm_rmse_ssh = np.sqrt( np.mean( X_val **2 ) )
        
        ds_OI = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(X_OI,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)
        ds_val = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(X_val,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)
        ds_rec = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(mod.x_rec_ssh,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)

        boost_swot_score_rmse_duacs = rmse_based_scores(ds_OI,ds_val )
        boost_swot_score_rmse_rec = rmse_based_scores(ds_rec,ds_val )

        boost_swot_score_psd_duacs = psd_based_scores(ds_OI,ds_val )
        boost_swot_score_psd_rec = psd_based_scores(ds_rec,ds_val )
        
        #boost_swot_score_psd_rec = psd_based_scores({'sossheig':mod.x_rec_ssh}, {'sossheig':X_test})
        #boost_swot_score_rmse_rec = rmse_based_scores({'sossheig':mod.x_rec_ssh}, {'sossheig':X_test})

        print('.... Boost swot RMSE scores (Validation dataset) ......')
        print('.... OI mse scores       : mu = %f  --  sigma = %f '%(boost_swot_score_rmse_duacs[2],boost_swot_score_rmse_duacs[3]) )
        print('.... 4DVarNet mse scores : mu = %f  --  sigma = %f '%(boost_swot_score_rmse_rec[2],boost_swot_score_rmse_rec[3]) )

        print('.... Boost swot PSD scores ......')
        print('.... OI mse scores       : lambda_x = %f  --  lambda_t = %f '%(boost_swot_score_psd_duacs[1],boost_swot_score_psd_duacs[2]) )
        print('.... 4DVarNet mse scores : lambda_x = %f  --  lambda_t = %f '%(boost_swot_score_psd_rec[1],boost_swot_score_psd_rec[2]) )

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, dataloaders=dataloaders['test'])
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
                
        ## postprocessing
        if width_med_filt > 2 :
            mod.x_rec_ssh = ndimage.median_filter(mod.x_rec_ssh,size=(1,width_med_filt,width_med_filt))
            
            #mod.x_rec_ssh = medianFilter2D.median_blur( torch.Tensor(mod.x_rec_ssh).view(-1,1,mod.x_rec_ssh.shape[1],mod.x_rec_ssh.shape[2]), (width_med_filt,width_med_filt))
            #mod.x_rec_ssh = mod.x_rec_ssh.detach().cpu().numpy().squeeze()

        mask_earth_test = np.tile(mask_earth.reshape(1,mask_earth.shape[0],mask_earth.shape[1]),(X_test.shape[0],1,1))
        test_mseRec = compute_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],mod.x_rec_ssh[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],mask_earth_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad])     
        test_mseOI  = compute_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],X_OI[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],mask_earth_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad])                    
 
        sig_lap = 1.
        test_score_lap_oi = compute_laplacian_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],X_OI[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],sig_lap=sig_lap)
        test_score_lap_rec = compute_laplacian_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],mod.x_rec_ssh[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],sig_lap=sig_lap)        
        
        print(test_score_lap_oi['mse'])
        print(test_score_lap_oi['var_lap'])
        print(test_score_lap_oi['r_square'])
        
        print(test_score_lap_rec['mse'])
        print(test_score_lap_rec['var_lap'])
        print(test_score_lap_rec['r_square'])

        saveRes = True #False#
        if saveRes == True :
            ssh_gt = X_test
            ssh_oi = X_OI
            ssh_obs = mod.x_rec_ssh_obs
            ssh_rec = mod.x_rec_ssh 
            sst_feat = mod.x_feat_sst
            
            #print(sst_feat.shape)
            filename_res = pathCheckPOint.replace('.ckpt','_res.nc')
            filename_res = filename_res.replace('SLANATL60_12am_NewSolver3_ChckPt-boost-swot/','/tmp/')
            
            print('.... save all gt/rec fields in nc file '+filename_res)
            save_NetCDF(filename_res,
                        ind_start=iiTest+int(dT/2),
                        ind_end=jjTest-int(dT/2),
                        ssh_gt = ssh_gt , 
                        ssh_oi = ssh_oi, 
                        ssh_obs = ssh_obs,
                        ssh_rec = ssh_rec,
                        sst_feat = sst_feat,
                        lat = latArea,
                        lon = lonArea)
            
        if flagSWOTData == True :
            print('... Period/exp: '+suffix_exp+' / SWOT+nadir data')
        else:
            print('... Period/exp: '+suffix_exp+' / nadir-only data')
                
        test_norm_rmse_ssh = np.sqrt( np.mean( X_test **2 ) )
        
        if scale_dwscaling > 0 :
            print('..... Downscaling factor for SST obs: %d'%scale_dwscaling)
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... NRMSE Val dataset (SSH) : OI = %.3f %% -- 4DVarNN = %.3f %%'%(1.-np.sqrt(val_mseOI['mse'])/val_norm_rmse_ssh,1.-np.sqrt(val_mseRec['mse'])/val_norm_rmse_ssh))
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        print('....... MSE Val dataset (ngSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseNGrad'],val_mseRec['mseNGrad'],100. * (1.-val_mseRec['mseNGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseNGrad']/val_mseOI['mseNGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... NRMSE Test dataset (SSH) : OI = %.3f %% -- 4DVarNN = %.3f %%'%(1.-np.sqrt(test_mseOI['mse'])/test_norm_rmse_ssh,1.-np.sqrt(test_mseRec['mse'])/test_norm_rmse_ssh))
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))
        print('....... MSE Test dataset (ngSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseNGrad'],test_mseRec['mseNGrad'],100. * (1.-test_mseRec['mseNGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseNGrad']/test_mseOI['mseNGrad'])))
        print('....... MSE Test dataset (Lap) : OI = %.2f %% /%.2f -- 4DVarNN = %.2f %% /%.2f '%(100. * (1.-test_score_lap_oi['mse']/test_score_lap_oi['var_lap']),test_score_lap_oi['r_square'],100. * (1.-test_score_lap_rec['mse']/test_score_lap_rec['var_lap']),test_score_lap_rec['r_square']))
        print('',flush=True)

        #import xarray as xr
        #dw_grad = 0
        X_test = X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad]
        X_OI = X_OI[:,dw_grad:X_OI.shape[1]-dw_grad,dw_grad:X_OI.shape[2]-dw_grad]
        mod.x_rec_ssh = mod.x_rec_ssh[:,dw_grad:mod.x_rec_ssh.shape[1]-dw_grad,dw_grad:mod.x_rec_ssh.shape[2]-dw_grad]
        
        ds_OI = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(X_OI,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)
        ds_test = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(X_test,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)
        ds_rec = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(mod.x_rec_ssh,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)

        boost_swot_score_rmse_duacs = rmse_based_scores(ds_OI,ds_test )
        boost_swot_score_rmse_rec = rmse_based_scores(ds_rec,ds_test )

        boost_swot_score_psd_duacs = psd_based_scores(ds_OI,ds_test )
        boost_swot_score_psd_rec = psd_based_scores(ds_rec,ds_test )
        
        #boost_swot_score_psd_rec = psd_based_scores({'sossheig':mod.x_rec_ssh}, {'sossheig':X_test})
        #boost_swot_score_rmse_rec = rmse_based_scores({'sossheig':mod.x_rec_ssh}, {'sossheig':X_test})

        print('.... Boost swot RMSE scores ......')
        print('.... OI mse scores       : mu = %f  --  sigma = %f '%(boost_swot_score_rmse_duacs[2],boost_swot_score_rmse_duacs[3]) )
        print('.... 4DVarNet mse scores : mu = %f  --  sigma = %f '%(boost_swot_score_rmse_rec[2],boost_swot_score_rmse_rec[3]) )

        print('.... Boost swot PSD scores ......')
        print('.... OI mse scores       : lambda_x = %f  --  lambda_t = %f '%(boost_swot_score_psd_duacs[1],boost_swot_score_psd_duacs[2]) )
        print('.... 4DVarNet mse scores : lambda_x = %f  --  lambda_t = %f '%(boost_swot_score_psd_rec[1],boost_swot_score_psd_rec[2]) )
        
        #print(1. / (20. * torch.fft.rfftfreq(X_OI.shape[2])) )
        
    elif flagProcess == 4: ## SQG-based inversion
        
        # validation dataset
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_val_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        Y_val_sst = qSST[iiVal+int(dT/2):jjVal-int(dT/2),0:W,0:W]

        # test dataset
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_test_OI = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        Y_test_sst = qSST[iiTest+int(dT/2):jjTest-int(dT/2),0:W,0:W]

        ##########################################################
        ## Fractional Laplacian inverse filter
        def hanning2d(M, N):
            """
            A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for the 1d description
            """
        
            if N <= 1:
                return np.hanning(M)
            elif M <= 1:
                return np.hanning(N) # scalar unity; don't window if dims are too small
            else:
                return np.outer(np.hanning(M),np.hanning(N))

        def apply_gaussian_filter(img, sigma=1.):
            if len( img.shape ) == 3 :
                f_img = 1. * img
        
                for kk in range(0,img.shape[0]):
                    f_img[kk,:,:] = gaussian_filter(img[kk,:,:], sigma= sig_smoothing)
            else:
                f_img = gaussian_filter(img, sigma= sig_smoothing)
            return f_img
        
        
        def compute_sqg_filter(img):
            
            if len( img.shape ) == 3 :
                img_inv = 0. * img
                for n in range(0,img.shape[0]):
                    img_inv[n,:,:] = compute_sqg_filter( img[n,:,:].squeeze() )
                
            else:
                img = hanning2d(*img.shape) * img
                imgf = np.fft.fftshift(np.fft.fft2(img))
        
        
                FreqCompRows = np.fft.fftfreq(img.shape[0],d=1.)
                FreqCompCols = np.fft.fftfreq(img.shape[1],d=1.)
                FreqCompRows = np.fft.fftshift(FreqCompRows)
                FreqCompCols = np.fft.fftshift(FreqCompCols)
        
                S,D = np.meshgrid(FreqCompRows, FreqCompCols)
                w = np.sqrt( S**2 + D**2 )
                alpha = 1. + 0. * w
                alpha[ np.where( w > 0. ) ] = 1. / w[ np.where( w > 0. ) ]
        
                img_inv = np.real( np.fft.ifft2( np.fft.ifftshift( alpha * imgf ) ) )
        
            return img_inv


        def apply_inverse_sqg(sst,ssh,ssh_oi,niter_smoothing=100,niter_smoothing_ano=0,sigma=1.,alpha=0.):
        
            s_sst = 1. * sst
            for nn in range(0,niter_smoothing):
                s_sst = apply_gaussian_filter(s_sst, sigma= sig_smoothing)
        
            sst_ano = sst - s_sst
        
            if niter_smoothing_ano > 0 :
                for nn in range(0,niter_smoothing_ano):
                    sst_ano = apply_gaussian_filter(sst_ano, sigma= sig_smoothing)
        
            ### apply SGQ inverse filter
            sst_ano_sqg = compute_sqg_filter( sst_ano )
        
            if len(ssh.shape) == 3:
                hanning_window = hanning2d(*ssh.shape[1:])
                ssh_ano = ssh-ssh_oi
                for kk in range(0,ssh.shape[0]):
                        ssh_ano[kk,:,:] = ssh_ano[kk,:,:] * hanning_window        
            else:
                ssh_ano = hanning2d(*ssh.shape) * (ssh-ssh_oi)
                
            if alpha == 0. :
                alpha = np.mean( ssh_ano * sst_ano_sqg ) / np.mean( sst_ano_sqg * sst_ano_sqg )
            R2 = np.corrcoef(ssh_ano.ravel(), sst_ano_sqg.ravel())[0,1]
            print('.... R2 = %.2f'%R2)
        
            #print(alpha)
            #print(R2)
            ### reconstructed field
            ssh_sqg_rec = ssh_oi + alpha * sst_ano_sqg
        
            return ssh_sqg_rec,alpha


        ##########################################################
        ### high-pass filtering applied to SST
        niter_smoothing = 100
        sig_smoothing = 1.
        niter_smoothing_ano = 0
        dw_grad = 50
        
        print('..... Apply inverse SQG filter')
        if 1*0 :
            x_rec_ssh,alpha = apply_inverse_sqg(Y_test_sst,X_test,X_test_OI,niter_smoothing,niter_smoothing_ano,sig_smoothing) 
        else:
            x_rec_ssh,alpha_val = apply_inverse_sqg(Y_val_sst,X_val,X_val_OI,niter_smoothing,niter_smoothing_ano,sig_smoothing) 
            x_rec_ssh,alpha = apply_inverse_sqg(Y_test_sst,X_test,X_test_OI,niter_smoothing,niter_smoothing_ano,sig_smoothing,alpha_val) 
                
        #####################################################
        mask_earth_test = np.tile(mask_earth.reshape(1,mask_earth.shape[0],mask_earth.shape[1]),(X_test.shape[0],1,1))
        test_mseRec = compute_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],x_rec_ssh[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],mask_earth_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad])     
        test_mseOI  = compute_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],X_test_OI[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],mask_earth_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad])                    
 
        sig_lap = 1.
        test_score_lap_oi = compute_laplacian_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],X_test_OI[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],sig_lap=sig_lap)
        test_score_lap_rec = compute_laplacian_metrics(X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],x_rec_ssh[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad],sig_lap=sig_lap)        
        
        saveRes = False#True #
        if saveRes == True :
            ssh_gt = X_test
            ssh_oi = X_OI
            ssh_obs = mod.x_rec_ssh_obs
            ssh_rec = mod.x_rec_ssh 
            sst_feat = mod.x_feat_sst
            
            #print(sst_feat.shape)
            filename_res = '/tmp/temp_res_sqg.nc'
            
            print('.... save all gt/rec fields in nc file '+filename_res)
            save_NetCDF(filename_res,
                        ind_start=iiTest+int(dT/2),
                        ind_end=jjTest-int(dT/2),
                        ssh_gt = ssh_gt , 
                        ssh_oi = ssh_oi, 
                        ssh_obs = ssh_obs,
                        ssh_rec = ssh_rec,
                        sst_feat = sst_feat,
                        lat = latArea,
                        lon = lonArea)
            
        if flagSWOTData == True :
            print('... Period/exp: '+suffix_exp+' / SWOT+nadir data')
        else:
            print('... Period/exp: '+suffix_exp+' / nadir-only data')
                
        test_norm_rmse_ssh = np.sqrt( np.mean( X_test **2 ) )
        
        if scale_dwscaling > 0 :
            print('..... Downscaling factor for SST obs: %d'%scale_dwscaling)
        print(' ')
        print('....................................')
        #print('....... Validation dataset')
        #print('....... NRMSE Val dataset (SSH) : OI = %.3f %% -- 4DVarNN = %.3f %%'%(1.-np.sqrt(val_mseOI['mse'])/val_norm_rmse_ssh,1.-np.sqrt(val_mseRec['mse'])/val_norm_rmse_ssh))
        #print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        #print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        #print('....... MSE Val dataset (ngSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseNGrad'],val_mseRec['mseNGrad'],100. * (1.-val_mseRec['mseNGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseNGrad']/val_mseOI['mseNGrad'])))
        #print(' ')
        print('....... Test dataset')
        print('....... NRMSE Test dataset (SSH) : OI = %.3f %% -- 4DVarNN = %.3f %%'%(1.-np.sqrt(test_mseOI['mse'])/test_norm_rmse_ssh,1.-np.sqrt(test_mseRec['mse'])/test_norm_rmse_ssh))
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))
        print('....... MSE Test dataset (ngSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseNGrad'],test_mseRec['mseNGrad'],100. * (1.-test_mseRec['mseNGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseNGrad']/test_mseOI['mseNGrad'])))
        print('....... MSE Test dataset (Lap) : OI = %.2f %% /%.2f -- 4DVarNN = %.2f %% /%.2f '%(100. * (1.-test_score_lap_oi['mse']/test_score_lap_oi['var_lap']),test_score_lap_oi['r_square'],100. * (1.-test_score_lap_rec['mse']/test_score_lap_rec['var_lap']),test_score_lap_rec['r_square']))
        print('',flush=True)

        #import xarray as xr
        #dw_grad = 0
        
        X_test = X_test[:,dw_grad:X_test.shape[1]-dw_grad,dw_grad:X_test.shape[2]-dw_grad]
        X_OI = X_test_OI[:,dw_grad:X_test_OI.shape[1]-dw_grad,dw_grad:X_test_OI.shape[2]-dw_grad]
        x_rec_ssh = x_rec_ssh[:,dw_grad:x_rec_ssh.shape[1]-dw_grad,dw_grad:x_rec_ssh.shape[2]-dw_grad]
        
        ds_OI = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(X_OI,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)
        ds_test = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(X_test,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)
        ds_rec = xr.Dataset( {"sossheig": (["lat","lon", "time"], np.moveaxis(x_rec_ssh,0,-1))},
                        coords={"lon": 1.0 / 20. * np.arange(0, X_OI.shape[1]),
                                "lat": 1.0 / 20. * np.arange(0, X_OI.shape[2]),
                                "time": np.arange(0,X_OI.shape[0]),
                                },)

        boost_swot_score_rmse_duacs = rmse_based_scores(ds_OI,ds_test )
        boost_swot_score_rmse_rec = rmse_based_scores(ds_rec,ds_test )

        boost_swot_score_psd_duacs = psd_based_scores(ds_OI,ds_test )
        boost_swot_score_psd_rec = psd_based_scores(ds_rec,ds_test )
        
        #boost_swot_score_psd_rec = psd_based_scores({'sossheig':mod.x_rec_ssh}, {'sossheig':X_test})
        #boost_swot_score_rmse_rec = rmse_based_scores({'sossheig':mod.x_rec_ssh}, {'sossheig':X_test})

        print('.... Boost swot RMSE scores ......')
        print('.... OI mse scores       : mu = %f  --  sigma = %f '%(boost_swot_score_rmse_duacs[2],boost_swot_score_rmse_duacs[3]) )
        print('.... 4DVarNet mse scores : mu = %f  --  sigma = %f '%(boost_swot_score_rmse_rec[2],boost_swot_score_rmse_rec[3]) )

        print('.... Boost swot PSD scores ......')
        print('.... OI mse scores       : lambda_x = %f  --  lambda_t = %f '%(boost_swot_score_psd_duacs[1],boost_swot_score_psd_duacs[2]) )
        print('.... 4DVarNet mse scores : lambda_x = %f  --  lambda_t = %f '%(boost_swot_score_psd_rec[1],boost_swot_score_psd_rec[2]) )
