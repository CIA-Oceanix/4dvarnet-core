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
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from netCDF4 import Dataset

import os
import solver as NN_4DVar
from models import Decoder, Phi_r, Model_H, ModelLR, Gradient_img, LitModel

from sklearn.feature_extraction import image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.distributed as dist

import config
from omegaconf import OmegaConf

cfg = OmegaConf.create(config.params)

# Definiton of training, validation and test dataset
# from dayly indices over a one-year time series
iiTr1 = 0
jjTr1 = 50 - int(cfg.dT / 2)

iiTr2 = 130 + int(cfg.dT / 2)
jjTr2 = 365

iiVal = 60 - int(cfg.dT / 2)
jjVal = 80 + int(cfg.dT / 2)

iiTest = 90 - int(cfg.dT / 2)
jjTest = 110 + int(cfg.dT / 2)
 
############################################## Data generation ###############################################################
print('........ Random seed set to 100')
np.random.seed(100)
torch.manual_seed(100)

ncfile = Dataset(os.path.join(cfg.data_dir, 'NATL60-CJM165_NATL_ssh_y2013.1y.nc'),"r")

# select GF region
lon    = ncfile.variables['lon'][:]
lat    = ncfile.variables['lat'][:]
idLat  = np.where( (lat >= 33.) & (lat <= 43.) )[0]
idLon  = np.where( (lon >= -65.) & (lon <= -55.) )[0]
lon    = lon[idLon]
lat    = lat[idLat]

# dirSAVE = '/gpfswork/rech/yrf/uba22to/ResSLANATL60/'
os.makedirs(cfg.dir_save, exist_ok=True)

genSuffixObs = ''
ncfile = Dataset(os.path.join(cfg.data_dir, 'NATL60-CJM165_NATL_ssh_y2013.1y.nc'),"r")
qHR = ncfile.variables['ssh'][:,idLat,idLon]
ncfile.close()

if cfg.flagSWOTData == True:
    print('.... Use SWOT+4-nadir dataset')
    genFilename = 'resInterpSWOTSLAwOInoSST_' + str('%03d' % (cfg.W)) + 'x' + str('%03d' % (cfg.W)) + 'x' + str('%02d' % (cfg.dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(os.path.join(cfg.data_dir, 'ssh_NATL60_swot_4nadir.nc'), "r")
    qOI = ncfile.variables['ssh_mod'][:,idLat,idLon]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(os.path.join(cfg.data_dir, 'dataset_nadir_0d_swot.nc'),"r")
    qMask = ncfile.variables['ssh_mod'][:,idLat,idLon]
    qMask = 1.0 - qMask.mask.astype(float)
    ncfile.close()
else:
    print('.... Use SWOT+4-nadir dataset')
    genFilename = 'resInterpSWOTSLAwOInoSST_' + str('%03d' % (cfg.W)) + 'x' + str('%03d' % (cfg.W)) + 'x' + str(
        '%02d' % (cfg.dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(os.path.join(cfg.data_dir, 'ssh_NATL60_4nadir.nc'), "r")
    qOI = ncfile.variables['ssh_mod'][:,idLat,idLon]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(os.path.join(cfg.data_dir, 'dataset_nadir_0d.nc'),"r")
    qMask = ncfile.variables['ssh_mod'][:,idLat,idLon]
    qMask = 1.0 - qMask.mask.astype(float)
    ncfile.close()

print('----- MSE OI: %.3f'%np.mean((qOI-qHR)**2))
print()

## extraction of patches from the SSH field
#NoRndPatches = False  
#if ( Nbpatches == 1 ) & ( W == 200 ):
#NoRndPatches = True
print('... No random seed for the extraction of patches')

qHR   = qHR[:,0:200,0:200]
qOI   = qOI[:,0:200,0:200]
qMask = qMask[:,0:200,0:200]
    
def extract_SpaceTimePatches(q,i1,i2,W,dT,rnd1,rnd2,Nbpatches,D=1):
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
dataTraining1     = extract_SpaceTimePatches(qHR, iiTr1, jjTr1, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)
dataTrainingMask1 = extract_SpaceTimePatches(qMask, iiTr1, jjTr1, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches,cfg.dx)
dataTrainingOI1   = extract_SpaceTimePatches(qOI, iiTr1, jjTr1, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)

dataTraining2     = extract_SpaceTimePatches(qHR, iiTr2, jjTr2, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)
dataTrainingMask2 = extract_SpaceTimePatches(qMask, iiTr2, jjTr2, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)
dataTrainingOI2   = extract_SpaceTimePatches(qOI, iiTr2, jjTr2, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)

dataTraining      = np.concatenate((dataTraining1,dataTraining2),axis=0)
dataTrainingMask  = np.concatenate((dataTrainingMask1,dataTrainingMask2),axis=0)
dataTrainingOI    = np.concatenate((dataTrainingOI1,dataTrainingOI2),axis=0)

# test dataset
dataTest     = extract_SpaceTimePatches(qHR, iiTest, jjTest, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)
dataTestMask = extract_SpaceTimePatches(qMask, iiTest, jjTest, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)
dataTestOI   = extract_SpaceTimePatches(qOI, iiTest, jjTest, cfg.W, cfg.dT, cfg.rnd1, cfg.rnd2, cfg.Nbpatches, cfg.dx)

# validation dataset
dataVal     = extract_SpaceTimePatches(qHR,iiVal,jjVal,cfg.W,cfg.dT,cfg.rnd1,cfg.rnd2,cfg.Nbpatches, cfg.dx)
dataValMask = extract_SpaceTimePatches(qMask,iiVal,jjVal,cfg.W,cfg.dT,cfg.rnd1,cfg.rnd2,cfg.Nbpatches,cfg.dx)
dataValOI   = extract_SpaceTimePatches(qOI,iiVal,jjVal,cfg.W,cfg.dT,cfg.rnd1,cfg.rnd2,cfg.Nbpatches,cfg.dx)

meanTr     = np.mean(dataTraining)
x_train    = dataTraining - meanTr
stdTr      = np.sqrt( np.mean( x_train**2 ) )
x_train    = x_train / stdTr

x_trainOI   = (dataTrainingOI - meanTr) / stdTr
x_trainMask = dataTrainingMask

x_test     = (dataTest  - meanTr )
stdTt      = np.sqrt( np.mean( x_test**2 ) )
x_test     = x_test / stdTr
x_testOI   = (dataTestOI - meanTr) / stdTr
x_testMask  = dataTestMask


x_val     = (dataVal  - meanTr )
stdVal    = np.sqrt( np.mean( x_val**2 ) )
x_val     = x_val / stdTr
x_valOI   = (dataValOI - meanTr) / stdTr
x_valMask = dataValMask

print('----- MSE Tr OI: %.6f'%np.mean((dataTrainingOI[:,int(cfg.dT/2),:,:]-dataTraining[:,int(cfg.dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean((dataTestOI[:,int(cfg.dT/2),:,:]-dataTest[:,int(cfg.dT/2),:,:])**2))

print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

print('..... Masked points (Tr)) : %.3f'%(np.sum(x_trainMask)/(x_trainMask.shape[0]*x_trainMask.shape[1]*x_trainMask.shape[2]*x_trainMask.shape[3])))
print('..... Masked points (Tt)) : %.3f'%(np.sum(x_testMask)/(x_testMask.shape[0]*x_testMask.shape[1]*x_testMask.shape[2]*x_testMask.shape[3])) )

print('----- MSE Tr OI: %.6f'%np.mean(stdTr**2 * (x_trainOI[:,int(cfg.dT/2),:,:]-x_train[:,int(cfg.dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean(stdTr**2 * (x_testOI[:,int(cfg.dT/2),:,:]-x_test[:,int(cfg.dT/2),:,:])**2))

######################### data loaders
training_dataset   = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainMask),torch.Tensor(x_train)) # create your datset
val_dataset        = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valMask),torch.Tensor(x_val)) # create your datset
test_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testMask),torch.Tensor(x_test))  # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True),
}            

var_Tr    = np.var( x_train )
var_Tt    = np.var( x_test )
var_Val   = np.var( x_val )

#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')
shapeData      = cfg.shapeData #np.array(x_train.shape[1:])
# shapeData_test = np.array(x_test.shape[1:])
# shapeData[0]  += shapeData[0]
# shapeData_test[0] += shapeData_test[0]

gradient_img = Gradient_img()

#model_LR = ModelLR()


# loss weghing wrt time
w_ = np.zeros(cfg.dT)
w_[int(cfg.dT / 2)] = 1.
wLoss = torch.Tensor(w_)

# recompute the MSE for OI on training dataset
# to define weighing parameters in the training

if cfg.betaX is None or cfg.betagX is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss_GOI = 0.
    running_loss_OI = 0.
    num_loss = 0
    
    gradient_img = gradient_img.to(device)
    wLoss        = wLoss.to(device)

    for targets_OI, inputs_Mask, targets_GT in dataloaders['train']:
        targets_OI     = targets_OI.to(device)
        inputs_Mask    = inputs_Mask.to(device)
        targets_GT     = targets_GT.to(device)
        
        # gradient norm field
        g_targets_GT = gradient_img(targets_GT)
        loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, wLoss)
        loss_GOI = NN_4DVar.compute_WeightedLoss(gradient_img(targets_OI) - g_targets_GT, wLoss)
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


def save_netcdf(saved_path1, x_test_rec, lon, lat):
    '''
    saved_path1: string 
    x_test_rec: 3d numpy array (4DVarNet-based predictions)
    lon: 1d numpy array 
    lat: 1d numpy array
    '''

    lon = np.arange(np.min(lon), np.max(lon)+1./(20./dwscale), 1./(20./dwscale))
    lat = np.arange(np.min(lat), np.max(lat), 1./(20./dwscale))

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    indN_Tt = np.concatenate([np.arange(60, 80)])
    time = [datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]

    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time), \
                   'ssh': (('time', 'lat', 'lon'), x_test_rec[:, int(dT / 2), :, :])}, \
        coords={'lon': lon, 'lat': lat, 'time': indN_Tt})
    xrdata.time.attrs['units'] = 'days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')

def nrmse(ref,pred):
    '''
    ref: Ground Truth fields
    pred: interpolated fields
    '''
    return np.sqrt(np.nanmean(((ref-np.nanmean(ref))-(pred-np.nanmean(pred)))**2))/np.nanstd(ref)

def nrmse_scores(gt,oi,pred,resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''
    # Compute daily nRMSE scores
    nrmse_oi=[]
    nrmse_pred=[]
    for i in range(len(oi)):
        nrmse_oi.append(nrmse(gt[i],oi[i]))
        nrmse_pred.append(nrmse(gt[i],pred[i]))
    tab_scores = np.zeros((2,3))
    tab_scores[0,0] = np.nanmean(nrmse_oi)
    tab_scores[0,1] = np.percentile(nrmse_oi,5)
    tab_scores[0,2] = np.percentile(nrmse_oi,95)
    tab_scores[1,0] = np.nanmean(nrmse_pred)
    tab_scores[1,1] = np.percentile(nrmse_pred,5)
    tab_scores[1,2] = np.percentile(nrmse_pred,95)
    np.savetxt(fname=resfile,X=tab_scores,fmt='%2.2f')
    return tab_scores


def compute_metrics(X_test,X_rec):
    # MSE
    mse = np.mean( (X_test - X_rec)**2 )

    # MSE for gradient
    gX_rec = np.gradient(X_rec,axis=[1,2])
    gX_rec = np.sqrt(gX_rec[0]**2 +  gX_rec[1]**2)
    
    gX_test = np.gradient(X_test,axis=[1,2])
    gX_test = np.sqrt(gX_test[0]**2 +  gX_test[1]**2)
    
    gmse = np.mean( (gX_test - gX_rec)**2 )
    ng   = np.mean( (gX_rec)**2 )
    
    return {'mse':mse,'mseGrad': gmse,'meanGrad': ng}


w_loss = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)


if __name__ == '__main__':
    
    flagProcess = 0
    
    if flagProcess == 0: ## training model from scratch
    
        loadTrainedModel = False#True
        if loadTrainedModel == True :             
            
            pathCheckPOint = "./SLANATL60_ChckPt/modelSLAInterpGF-Exp3-epoch=36-val_loss=0.05.ckpt"  
            print('.... load pre-trained model :'+pathCheckPOint)
            mod = LitModel.load_from_checkpoint(pathCheckPOint)            
            mod.hparams.n_grad          = 10
            mod.hparams.iter_update     = [0, 25, 40, 100, 150, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [10, 10, 15, 15, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        else:
            mod = LitModel(hparam=cfg, w_loss=w_loss, var_Val=var_Val, var_Tr=var_Tr, var_Tt=var_Tt)

        print(mod.hparams)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= cfg.dir_save,
                                              filename='modelSLAInterpGF-Exp3-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 2 }

        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs,callbacks=[checkpoint_callback])
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
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
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad'])))
    
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
        mod.model.n_grad = 10
    
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


        pathCheckPOint = "./SLANATL60_ChckPt/modelSLAInterpGF-Exp3-epoch=36-val_loss=0.05.ckpt"
        
        mod = LitModel.load_from_checkpoint(pathCheckPOint)            
        mod.model.n_grad = 10
        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                
        val_mseRec = compute_metrics(X_val,mod.x_rec)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
                
        test_mseRec = compute_metrics(X_test,mod.x_rec)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        
        saveRes = False
        if saveRes == True :
            save_NetCDF('results/test.nc', mod.x_rec)

        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad'])))
