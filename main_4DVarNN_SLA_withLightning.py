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
sys.path.append('../4dvarnet-core')
import solver as NN_4DVar

from sklearn.feature_extraction import image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import math
#import torch.distributed as dist

flagRandomSeed = 0

## NN architectures and optimization parameters
batch_size      = 10#16#4#4#8#12#8#256#
DimAE           = 50#10#10#50
dimGradSolver   = 100 # dimension of the hidden state of the LSTM cell
rateDropout     = 0.25 # dropout rate 

# data generation
sigNoise        = 0. ## additive noise standard deviation
flagSWOTData    = True #False ## use SWOT data or not
dT              = 5 ## Time window of each space-time patch
W               = 200 ## width/height of each space-time patch
dx              = 1   ## subsampling step if > 1
Nbpatches       = 1#10#10#25 ## number of patches extracted from each time-step 
rnd1            = 0 ## random seed for patch extraction (space sam)
rnd2            = 100 ## random seed for patch extraction
dwscale         = 1

W = int(W/dx)

# 
UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
InterpFlag         = False # True => force reconstructed field to observed data after each gradient-based update

# Definiton of training, validation and test dataset
# from dayly indices over a one-year time series
iiTr1 = 0
jjTr1 = 50 - int(dT / 2)

iiTr2 = 130 + int(dT / 2)
jjTr2 = 365

iiVal = 60 - int(dT / 2)
jjVal = 80 + int(dT / 2)

iiTest = 90 - int(dT / 2)
jjTest = 110 + int(dT / 2)
 
############################################## Data generation ###############################################################
if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)
    torch.manual_seed(100)

dirSAVE = '/gpfswork/rech/yrf/ueh53pd/ResSLANATL60/'
genSuffixObs = ''

dirREF = "/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/"
if os.path.isdir(dirREF) == True :            
    dirDATA = dirREF
else:
    dirDATA = '/users/local/DATA/DataNATL60/GULFSTREAM/'

ncfile = Dataset(dirDATA+"ref/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc","r")
qHR    = ncfile.variables['ssh'][:]
ncfile.close()


if flagSWOTData == True :
    print('.... Use SWOT+4-nadir dataset')
    genFilename  = 'resInterpSLAwSWOT_Exp3_NewSolver_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"oi/ssh_NATL60_swot_4nadir.nc","r")
    qOI    = ncfile.variables['ssh_mod'][:]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc","r")
    
    qMask   = ncfile.variables['ssh_mod'][:]
    qMask   = 1.0-qMask.mask.astype(float)
    ncfile.close()

else:
    genFilename  = 'resInterp4NadirSLAwOInoSST_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
    print('.... Use 4-nadir dataset')
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"oi/ssh_NATL60_4nadir.nc","r")
    qOI    = ncfile.variables['ssh_mod'][:]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"data/gridded_data_swot_wocorr/dataset_nadir_0d.nc","r")
    
    qMask   = ncfile.variables['ssh_mod'][:]
    qMask   = 1.0-qMask.mask.astype(float)
    ncfile.close()

print('----- MSE OI: %.3f'%np.mean((qOI-qHR)**2))
print()

## extraction of patches from the SSH field
NoRndPatches = False  
if ( Nbpatches == 1 ) & ( W == 200 ):
    NoRndPatches = True
    print('... No random seed for the extraction of patches')
    
    qHR   = qHR[:,0:200,0:200]
    qOI   = qOI[:,0:200,0:200]
    qMask = qMask[:,0:200,0:200]
    
def extract_SpaceTimePatches(q,i1,i2,W,dT,rnd1,rnd2,D=1):
    dataTraining  = image.extract_patches_2d(np.moveaxis(q[i1:i2,::D,::D], 0, -1),(W,W),max_patches=Nbpatches,random_state=rnd1)
    dataTraining  = np.moveaxis(dataTraining, -1, 1)
    dataTraining  = dataTraining.reshape((Nbpatches,dataTraining.shape[1],W*W)) 
    
    if NoRndPatches == True :
        for ii in range(0,dataTraining.shape[1]-dT+1):
            if ii == 0:                
                temp = dataTraining[:,ii:ii+dT,:].reshape((1,dT,Nbpatches,W*W))
            else:
                temp = np.concatenate((temp,dataTraining[:,ii:ii+dT,:].reshape((1,dT,Nbpatches,W*W))),axis=0)

        dataTraining = np.moveaxis(temp, 1, 2)
    else:
        #dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),max_patches=dataTraining.shape[1]-dT+1,random_state=rnd2)
        dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),max_patches=None)
        #dataTraining  = dataTraining.reshape((dT,W*W,Nbpatches*dataTraining.shape[-1]))
    dataTraining  = dataTraining.reshape((dataTraining.shape[0],dataTraining.shape[1],dT,W,W)) 
    dataTraining  = np.moveaxis(dataTraining, 0, -1)
    dataTraining  = np.moveaxis(dataTraining, 0, -1)
    dataTraining  = dataTraining.reshape((dT,W,W,dataTraining.shape[3]*dataTraining.shape[4])) 
    dataTraining  = np.moveaxis(dataTraining, -1, 0)
    return dataTraining     

# training dataset
dtOI = 0
iiTr1 = 0
jjTr1 = 50 - int(dT / 2)

iiTr2 = 130 + int(dT / 2)
jjTr2 = 365

iiVal = 60 - int(dT / 2)
jjVal = 80 + int(dT / 2)
#iiVal = 90 - int(dT / 2)
#jjVal = 110 + int(dT / 2)

iiTest = 90 - int(dT / 2)
jjTest = 110 + int(dT / 2)

# training dataset
dataTraining1     = extract_SpaceTimePatches(qHR,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTrainingMask1 = extract_SpaceTimePatches(qMask,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTrainingOI1   = extract_SpaceTimePatches(qOI,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)

dataTraining2     = extract_SpaceTimePatches(qHR,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTrainingMask2 = extract_SpaceTimePatches(qMask,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTrainingOI2  = extract_SpaceTimePatches(qOI,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)

dataTraining      = np.concatenate((dataTraining1,dataTraining2),axis=0)
dataTrainingMask  = np.concatenate((dataTrainingMask1,dataTrainingMask2),axis=0)
dataTrainingOI    = np.concatenate((dataTrainingOI1,dataTrainingOI2),axis=0)

# test dataset
dataTest     = extract_SpaceTimePatches(qHR,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestMask = extract_SpaceTimePatches(qMask,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestOI   = extract_SpaceTimePatches(qOI,iiTest,jjTest,W,dT,rnd1,rnd2,dx)

# validation dataset
dataVal     = extract_SpaceTimePatches(qHR,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValMask = extract_SpaceTimePatches(qMask,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValOI   = extract_SpaceTimePatches(qOI,iiVal,jjVal,W,dT,rnd1,rnd2,dx)

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

print('----- MSE Tr OI: %.6f'%np.mean((dataTrainingOI[:,int(dT/2),:,:]-dataTraining[:,int(dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean((dataTestOI[:,int(dT/2),:,:]-dataTest[:,int(dT/2),:,:])**2))

print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

print('..... Masked points (Tr)) : %.3f'%(np.sum(x_trainMask)/(x_trainMask.shape[0]*x_trainMask.shape[1]*x_trainMask.shape[2]*x_trainMask.shape[3])))
print('..... Masked points (Tt)) : %.3f'%(np.sum(x_testMask)/(x_testMask.shape[0]*x_testMask.shape[1]*x_testMask.shape[2]*x_testMask.shape[3])) )

print('----- MSE Tr OI: %.6f'%np.mean(stdTr**2 * (x_trainOI[:,int(dT/2),:,:]-x_train[:,int(dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean(stdTr**2 * (x_testOI[:,int(dT/2),:,:]-x_test[:,int(dT/2),:,:])**2))

######################### data loaders
training_dataset   = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainMask),torch.Tensor(x_train)) # create your datset
val_dataset        = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valMask),torch.Tensor(x_val)) # create your datset
test_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testMask),torch.Tensor(x_test))  # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
}            

var_Tr    = np.var( x_train )
var_Tt    = np.var( x_test )
var_Val   = np.var( x_val )


#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')

shapeData = np.array(x_train.shape[1:])
shapeData_test = np.array(x_test.shape[1:])
shapeData[0] += shapeData[0]
shapeData_test[0] += shapeData_test[0]

dW  = 3
dW2 = 1
sS  = int(4/dx)
nbBlocks = 1
rateDr   = 0. * rateDropout

class BiLinUnit(torch.nn.Module):
    def __init__(self,dimIn,dim,dropout=0.):
        super(BiLinUnit, self).__init__()
        
        self.conv1  = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False)
        self.conv2  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.conv3  = torch.nn.Conv2d(2*dim, dimIn, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
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
              layers.append( BiLinUnit(dimAE,dimAE,dropout) )
          return torch.nn.Sequential(*layers)
      
    def forward(self, xinp):
        
        ## LR comlponent
        xLR = self.NNLR( self.pool1(xinp) )
        xLR = self.dropout(xLR)
        xLR = self.convTr( xLR ) 
        
        # HR component
        xHR = self.NNHR( xinp )
        
        return xLR + xHR
  
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
  
    def forward(self, x):
        return torch.mul(1.,x)


class Phi_r(torch.nn.Module):
    def __init__(self):
        super(Phi_r, self).__init__()
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

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout


#model_H = Model_H()


class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, im):

        if im.size(1) == 1:
            G_x = self.convGx(im)
            G_y = self.convGy(im)
            G = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2))
        else:

            for kk in range(0, im.size(1)):
                G_x = self.convGx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                G_y = self.convGy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                G_x = G_x.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                G_y = G_y.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                nG = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2))

                if kk == 0:
                    G = nG.view(-1, 1, im.size(1) - 2, im.size(2) - 2)
                else:
                    G = torch.cat((G, nG.view(-1, 1, im.size(1) - 2, im.size(2) - 2)), dim=1)
        return G


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
betaX  = None#42.20436766972647
betagX = None#77.99700321505073
if betaX is None or betagX is None:
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

def save_NetCDF(saved_path1, x_test_rec):
    extent = [-65., -55., 30., 40.]
    indLat = 200
    indLon = 200

    lon = np.arange(extent[0], extent[1], 1 / (20 / dwscale))
    lat = np.arange(extent[2], extent[3], 1 / (20 / dwscale))
    indLat = int(indLat / dwscale)
    indLon = int(indLon / dwscale)
    lon = lon[:indLon]
    lat = lat[:indLat]

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

############################################Lightning Module#######################################################################
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #self.m_Grad       = 
        self.IterUpdate   = [0, 25, 50, 75, 100, 150, 800]  # [0,2,4,6,9,15]
        self.NbGradIter   = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.lrUpdate     = [1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                         Model_H(), 
                                                         NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, dimGradSolver, rateDropout), 
                                                         None, None, shapeData, self.NbGradIter[0])
        self.model_LR     = ModelLR()
        self.gradient_img = Gradient_img()
        
        self.wLoss        = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)

        self.automatic_optimization = True#False#
        
        self.x_rec   = None # variable to store output of test method
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        if 1*1 :
            #optimizer   = optim.Adam(self.model.parameters(), lr = self.lrUpdate[0])
            optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.lrUpdate[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.lrUpdate[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.lrUpdate[0]},
                                    ], lr=0.)
        elif 1*0:
            optimizer   = optim.RMSprop([{'params': self.model.model_Grad.parameters(), 'lr': self.lrUpdate[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.lrUpdate[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.lrUpdate[0]},
                                    ], lr=0.)

        else:
            optimizer   = optim.ASGD([{'params': self.model.model_Grad.parameters(), 'lr': self.lrUpdate[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.lrUpdate[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.lrUpdate[0]},
                                    ], lr=0.)
        #optPhi     = optim.Adam(self.model.phi_r.parameters(),lr=self.lrUpdate[0])
        #optGrad    = optim.Adam(self.model.model_Grad.parameters(),lr=self.lrUpdate[0])
        #optVarCost = optim.Adam(self.model.model_VarCost.parameters(),lr=self.lrUpdate[0])


        return optimizer

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,):
        # skip the first 500 steps
        if (self.current_epoch in self.IterUpdate) & (self.current_epoch > 0):
            indx             = self.IterUpdate.index(self.current_epoch)
            
            lrCurrent = self.lrUpdate[indx]
            lr = np.array([lrCurrent,lrCurrent,0.5*lrCurrent,0.])            
            mm = 0
            for pg in optimizer.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
    
        # update params
        optimizer.step(closure=optimizer_closure)
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
        if (self.current_epoch in self.IterUpdate) & (self.current_epoch > 0) & ( batch_idx == 0 ):
            indx             = self.IterUpdate.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.NbGradIter[indx],self.lrUpdate[indx]))
            
            self.model.NGrad = self.NbGradIter[indx]
            
            for pg in opt.param_groups:
                pg['lr'] = self.lrUpdate[indx]# * self.hparams.learning_rate
                    
        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')

        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if 1:#(batch_idx + 1) % 2 == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')

        self.log("val_mse", metrics['mse'] / var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log("val_mse", metrics['mse'] / var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'preds': out.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    #def validation_epoch_end(self, training_step_outputs):
    #    # do something with all training_step outputs
    #    #print('.... Validation dataset')
    #    1.
        
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        x_test_rec = stdTr * x_test_rec + meanTr
        
        self.x_rec = x_test_rec[:,int(dT/2),:,:]

        return [{'mse':meanTr,'preds': meanTr}]

    def compute_loss(self, batch, phase):

        targets_OI, inputs_Mask, targets_GT = batch

        new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask ), dim=1)
        inputs_init    = torch.cat((targets_OI, inputs_Mask * (targets_GT - targets_OI)), dim=1)
        inputs_missing = torch.cat((targets_OI, inputs_Mask * (targets_GT - targets_OI)), dim=1)

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, new_masks)

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR   = outputs[:, 0:dT, :, :]
            outputs      = outputsSLR + outputs[:, dT:, :, :]

            # reconstruction losses
            g_outputs  = self.gradient_img(outputs)
            loss_All   = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.wLoss)
            loss_GAll  = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.wLoss)

            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.wLoss)
            loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.wLoss)

            # projection losses
            loss_AE     = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT         = torch.cat((targets_GT, outputsSLR - targets_GT), dim=1)
            #yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR      = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.wLoss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR      = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.wLoss)

            # total loss
            loss     = alpha_MSE * (betaX * loss_All + betagX * loss_GAll) + 0.5 * alpha_Proj * (loss_AE + loss_AE_GT)
            loss    += alpha_LR * loss_LR + alpha_SR * loss_SR
            
            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT,wLoss)
            mse       = loss_All.detach()
            mseGrad   = loss_GAll.detach()  
            metrics   = dict([('mse',mse),('mseGrad',mseGrad),('meanGrad',mean_GAll),('mseOI',loss_OI.detach()),('mseGOI',loss_GOI.detach())])
            #print(mse.cpu().detach().numpy())
            
        return loss,outputs, metrics

if __name__ == '__main__':
    
    flagProcess = 5
    
    if flagProcess == 0: ## training model from scratch
    
        mod = LitModel()
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= dirSAVE,
                                              filename='modelSLAInterpGF-Exp3-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        ncfile = Dataset("results/test.nc","r")
        X_rec    = ncfile.variables['ssh'][:]
        ncfile.close()
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
    
        #mseRec = np.mean( (X_val - X_rec)**2 )
        mseRec = np.mean( (X_val - mod.x_rec)**2 )
        mseOI  = np.mean( ( X_val -  X_OI)**2 )
        print('....... MSE Val dataset: OI = %.3e -- 4DVarNN = %.3e'%(mseOI,mseRec))
        print('....... Normalized MSE Test dataset: OI = %.3e -- 4DVarNN = %.3e'%(mseOI / stdTr**2,mseRec/ stdTr**2))
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        ncfile = Dataset("results/test.nc","r")
        X_rec  = ncfile.variables['ssh'][:]
        ncfile.close()
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
    
        mseRec = np.mean( (X_test - X_rec)**2 )
        mseOI  = np.mean( (X_test - X_OI)**2 )
        print('....... MSE Test dataset: OI = %.3e -- 4DVarNN = %.3e'%(mseOI,mseRec))
        print('....... Normalized MSE Test dataset: OI = %.3e -- 4DVarNN = %.3e'%(mseOI / stdTr**2,mseRec/ stdTr**2))
    
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
    
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
    elif flagProcess == 5: ## test trained model

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
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad'])))

    