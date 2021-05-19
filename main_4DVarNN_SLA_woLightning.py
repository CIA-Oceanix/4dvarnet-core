#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020

@author: rfablet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:33:30 2020

@author: rfablet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:09:32 2020

@author: rfablet
"""

import argparse
import numpy as np
import xarray as xr

#import matplotlib.pyplot as plt 
#import os
#import tensorflow.keras as keras
import datetime

import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from netCDF4 import Dataset
#import scipy
#from scipy.integrate import solve_ivp
#from sklearn.feature_extraction import image

# specific torch module 
#import dinAE_solver_torch as dinAE
#import torch_4DVarNN_withObsModel_v3 as NN_4DVar

import os
import sys
sys.path.append('../4dvarnet-core')
import solver as NN_4DVar
#import os
#os.chdir('../4dvarnet-core')
#import solver as NN_4DVar

#import torch4DVarNN_solver as NN_4DVar
#import torchvision.datasets as datasets
#from sklearn import decomposition
from sklearn.feature_extraction import image
#############################################################################################
##
## Implementation of a NN pytorch framework for the identification and resolution 
## of 4DVar assimilation model applied to Lorenz-96 dynamics (40-dimensional state)
## 
## Data generation
##      flagProcess == 0: load data and extract data spatio-temporal patches
##
## Bulding NN architectures
##      flagProcess == 2: Generation of the NN architectures for operator Phi and observation operator
##        flagAEType == 0: dense auto-encoder
##        flagAEType == 1:  two-scale GENN with DimAE-dimensional latent states
##        flagAEType == 2:  two-scale U-Net with a bilinear layer
##
## Learning schemes
##      flagProcess == 4: Supervised learning of 4DVarNN model
##
## Evaluation of trained model
##      flagProcess == 5: Apply a trained model for the evaluation of different performance metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    flagProcess    = [0,2,5]#Sequence fo processes to be run
    
    flagRandomSeed = 0
    flagSaveModel  = 1
       
    ## NN architectures and optimization parameters
    batch_size      = 2#16#4#4#8#12#8#256#
    flagAEType      = 2 # 0: AE, 1-2: GENN, 3: PCA
    DimAE           = 50#10#10#50
    GradType        = 1 # Gradient computation (1: L2 norm)
    OptimType       = 2 # 2: LSTM-based descent, no other option yet
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
    
    W = int(W/dx)

    # 
    UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    InterpFlag         = False # True => force reconstructed field to observed data after each gradient-based update
     
    import math
    def extract_image_patches(x, kernel, stride=1, dilation=1):
        # Do TF 'SAME' Padding
        b,c,h,w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
        
        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0,4,5,1,2,3).contiguous()
        
        return patches.view(b,-1,patches.shape[-2], patches.shape[-1])
    
    ## Multi-GPU model
    class MyDataParallel(torch.nn.DataParallel):
        """
        Allow nn.DataParallel to call model's attributes.
        """
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)
    
    def model_to_MultiGPU_basic(mod):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('... Number of GPUs: %d'%torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model             = MyDataParallel(mod)
        model.to(device)
        return model   
    
    def model_to_MultiGPU(mod):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('... Number of GPUs: %d'%torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model      = MyDataParallel(mod)
                model.phi_r   = MyDataParallel(mod.phi_r)
                model.model_Grad   = MyDataParallel(mod.model_Grad)
                model.model_H      = MyDataParallel(mod.model_H)
                #model.model_Grad.compute_Grad = MyDataParallel(mod.model_Grad.compute_Grad)
                #model.model_Grad.compute_Grad.H = MyDataParallel(mod.model_Grad.compute_Grad.H)
        model.to(device)
        model.model_H.to(device)
        model.phi_r.to(device)
        model.model_Grad.to(device)
        return model  

    for kk in range(0,len(flagProcess)):
        
        #################################################################
        ## data generation including noise sampling and missing data gaps
        if flagProcess[kk] == 0:  
            if flagRandomSeed == 0:
                print('........ Random seed set to 100')
                np.random.seed(100)
                torch.manual_seed(100)

            dirSAVE      = './ResSLANATL60/'
            genSuffixObs = ''
            dirREF = "/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/"
            if os.path.isdir(dirREF) == True :            
                dirDATA = dirREF
            else:
                dirDATA = '/users/local/DATA/DataNATL60/GULFSTREAM/'


            #ncfile = Dataset(dirDATA+"/gpfswork/rech/yrf/commun/DataNATL60/dataSLA_NATL60GulfStream.nc","r")
            ncfile = Dataset(dirDATA+"ref/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc","r")
            qHR    = ncfile.variables['ssh'][:]
            #qLR    = ncfile.variables['qLR'][:]
            #dt     = ncfile.variables['dt'][:]
            
            ncfile.close()

            if flagSWOTData == True :
                print('.... Use SWOT+4-nadir dataset')
                genFilename  = 'resInterpSLAwSWOT_Exp3_NewSolver_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
                # OI data using a noise-free OSSE (ssh_mod variable)
                #ncfile = Dataset("dirDATA+/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/oi/ssh_NATL60_swot_4nadir.nc","r")
                ncfile = Dataset(dirDATA+"oi/ssh_NATL60_swot_4nadir.nc","r")
                qOI    = ncfile.variables['ssh_mod'][:]
                #qLR    = ncfile.variables['qLR'][:]
                #dt     = ncfile.variables['dt'][:]            
                ncfile.close()
    
                # OI data using a noise-free OSSE (ssh_mod variable)
                #/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/data/gridded_data_swot_wocorr/
                ncfile = Dataset(dirDATA+"data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc","r")
                
                qMask   = ncfile.variables['ssh_mod'][:]
                qMask   = 1.0-qMask.mask.astype(float)
                #qLR    = ncfile.variables['qLR'][:]
                #dt     = ncfile.variables['dt'][:]            
                ncfile.close()

            else:
                genFilename  = 'resInterp4NadirSLAwOInoSST_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
                print('.... Use 4-nadir dataset')
                # OI data using a noise-free OSSE (ssh_mod variable)
                # = Dataset("./data/DataNATL60/dataSLA_NATL60GulfStream_OI.nc","r")
                #ncfile = Dataset("/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/oi/ssh_NATL60_4nadir.nc","r")
                ncfile = Dataset(dirDATA+"oi/ssh_NATL60_4nadir.nc","r")
                qOI    = ncfile.variables['ssh_mod'][:]
                #qLR    = ncfile.variables['qLR'][:]
                #dt     = ncfile.variables['dt'][:]            
                ncfile.close()
    
                # OI data using a noise-free OSSE (ssh_mod variable)
                #ncfile = Dataset("./data/DataNATL60/dataSLA_NATL60GulfStream_Obs.nc","r")
                #ncfile = Dataset("/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/data/gridded_data_swot_wocorr/dataset_nadir_0d.nc","r")
                ncfile = Dataset(dirDATA+"data/gridded_data_swot_wocorr/dataset_nadir_0d.nc","r")
                
                qMask   = ncfile.variables['ssh_mod'][:]
                qMask   = 1.0-qMask.mask.astype(float)
                #qLR    = ncfile.variables['qLR'][:]
                #dt     = ncfile.variables['dt'][:]            
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
                    #dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),dataTraining.shape[1]-dT+1,random_state=rnd2)
                    dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),None)
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

        ###############################################################
        ## Initial interpolation
        elif flagProcess[kk] == 1:        
            print('........ No initialization')
 
        ###############################################################
        ## AE architecture
        elif flagProcess[kk] == 2:        
            print('........ Define AE architecture')
            
            # freeze all ode parameters            
            #shapeData    = np.ones(3).astype(int)
            shapeData  =  np.array(x_train.shape[1:])            
            
            # two-scale states (x,dx)
            shapeData[0] += shapeData[0]

            if flagAEType == 0: ## MLP-AE
              genSuffixModel = '_AEMLP%02d'%DimAE  
            
              class Encoder(torch.nn.Module):
                  def __init__(self):
                      super(Encoder, self).__init__()
                      self.sLR  = 16
                      self.pool  = torch.nn.AvgPool2d((self.sLR,self.sLR))
                      self.fc1 = torch.nn.Linear(int(shapeData[0]*(shapeData[1]*shapeData[2])/(self.sLR*self.sLR)),6*DimAE)
                      self.fc2 = torch.nn.Linear(6*DimAE,2*DimAE)
                      self.fc3 = torch.nn.Linear(2*DimAE,DimAE)
            
                  def forward(self, x):
                      #x = self.fc1( torch.nn.Flatten(x) )
                      x = self.pool(x)
                      x = self.fc1( x.view(-1,int(shapeData[0]*shapeData[1]*shapeData[2]/(self.sLR*self.sLR))) )
                      x = self.fc2( F.relu(x) )
                      x = self.fc3( F.relu(x) )
                      return x

              class Decoder(torch.nn.Module):
                  def __init__(self):
                      super(Decoder, self).__init__()
                      self.sLR = 16
                      self.fc1 = torch.nn.Linear(DimAE,10*DimAE)
                      self.fc2 = torch.nn.Linear(10*DimAE,20*DimAE)
                      self.fc3 = torch.nn.Linear(20*DimAE,int(shapeData[0]*(shapeData[1]*shapeData[2])/(self.sLR*self.sLR)))
            
                      self.convTr1 = torch.nn.ConvTranspose2d(shapeData[0],shapeData[0],(4,4),stride=(4,4),bias=False)          
                      self.convTr2 = torch.nn.ConvTranspose2d(shapeData[0],shapeData[0],(int(self.sLR/4),int(self.sLR/4)),stride=(int(self.sLR/4),int(self.sLR/4)),bias=False)          

                  def forward(self, x):
                      x = self.fc1( x )
                      x = self.fc2( F.relu(x) )
                      x = self.fc3( F.relu(x) )
                      x = x.view(-1,shapeData[0],int(shapeData[1]/self.sLR),int(shapeData[2]/self.sLR))
                      x = self.convTr1( self.convTr1(x) )
                      
                      return x
            elif flagAEType == 1: ## Conv model with no use of the central point
              dW = 5
              sS = int(4/dx)
              rateDr   = 0. * rateDropout
              genSuffixModel = '_GENN_%d_%02d_%02d_%02d'%(flagAEType,DimAE,dW,int(100*rateDr))
              class Encoder(torch.nn.Module):
                  def __init__(self,shape0,DimAE,rateDropout):
                      super(Encoder, self).__init__()
                      self.pool1  = torch.nn.AvgPool2d(sS)
                      self.conv1  = torch.nn.Conv2d(shape0,2*DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
                      self.conv2  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                      
                      self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                      self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                      self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                      self.conv3  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                      #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)
            
                      self.conv2Tr = torch.nn.ConvTranspose2d(DimAE,shape0,(sS,sS),stride=(sS,sS),bias=False)          
                      #self.conv5 = torch.nn.Conv1d(2*shapeData[0]*DimAE,2*shapeData[0]*DimAE,3,padding=1,bias=False)
                      #self.conv6 = torch.nn.Conv1d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)
                      #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)
            
                      self.convHR1  = torch.nn.Conv2d(shape0,2*DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
                      self.convHR2  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                      
                      self.convHR21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                      self.convHR22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                      self.convHR23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                      self.convHR3  = torch.nn.Conv2d(2*DimAE,shape0,(1,1),padding=0,bias=False)

                  def forward(self, xinp):
                      #x = self.fc1( torch.nn.Flatten(x) )
                      #x = self.pool1( xinp )
                      x = self.pool1( xinp )
                      x = self.conv1( x )
                      x = self.conv2( F.relu(x) )
                      x = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
                      x = self.conv3( x )
                      x = self.conv2Tr( x )
                      
                      xHR = self.convHR1( xinp )
                      xHR = self.convHR2( F.relu(xHR) )
                      xHR = torch.cat((self.convHR21(xHR), self.convHR22(xHR) * self.convHR23(xHR)),dim=1)
                      xHR = self.convHR3( xHR )
                      
                      x   = x + xHR #torch.add(x,1.,xHR)
                      
                      #x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
                      return x
            
              class Decoder(torch.nn.Module):
                  def __init__(self):
                      super(Decoder, self).__init__()
            
                  def forward(self, x):
                      return torch.mul(1.,x)
                

            elif flagAEType == 2: ## Conv model with no use of the central point
              dW  = 3
              dW2 = 1
              sS  = int(4/dx)
              nbBlocks = 1
              rateDr   = 0. * rateDropout
              genSuffixModel = '_GENN_%d_%02d_%02d_%02d_%02d_%02d'%(flagAEType,DimAE,dW,dW2,nbBlocks,int(100*rateDropout))

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
                    
                  def __make_ResNet(self,dimAE,Nb_RU=2,dropout=0.):
                        layers = []
                        for kk in range(0,Nb_RU):
                            layers.append( ResLayer(dimAE,dropout) )
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

            elif flagAEType == 3: ## Conv model with no use of the central point
              dW  = 3
              dW2 = 1
              sS  = int(4/dx)
              nbBlocks = 2
              rateDr   = 0. * rateDropout
              genSuffixModel = '_GENN_%d_%02d_%02d_%02d_%02d_%02d'%(flagAEType,DimAE,dW,dW2,nbBlocks,int(100*rateDropout))

              class ResLayer(torch.nn.Module):
                  def __init__(self,dimAE,dropout=0.):
                      super(ResLayer, self).__init__()
                      self.RU = self.__make_RU(dimAE,dropout=0.)
                      
                  def __make_RU(self,DimAE,dropout=0.):
                        layers = []
                
                        layers.append(torch.nn.Conv2d(DimAE, 2*DimAE, (2*dW2+1,2*dW2+1), padding=dW2,bias=False))
                        #layers.append( torch.nn.BatchNorm2d(2*DimAE) )
                        layers.append(torch.nn.Dropout(dropout))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Conv2d(2*DimAE, DimAE, (2*dW2+1,2*dW2+1), padding=dW2,bias=False))
                        layers.append(torch.nn.Dropout(dropout))
                
                        return torch.nn.Sequential(*layers)

                  def forward(self,x):
                      
                      return x + self.RU(x)

              class BiLinNN(torch.nn.Module):
                  def __init__(self,dimIn,dim,NbRU=1,dropout=0.):
                      super(BiLinNN, self).__init__()
                      
                      self.conv1  = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False)
                      self.conv2  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.conv3  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.dropout  = torch.nn.Dropout(dropout)
                      
                      self.NbRU = NbRU
                      if self.NbRU > 0:
                          self.ResNet  = self.__make_ResNet(dim,NbRU,dropout)
                    
                  def __make_ResNet(self,dimAE,Nb_RU=2,dropout=0.):
                        layers = []
                        for kk in range(0,Nb_RU):
                            layers.append( ResLayer(dimAE,dropout) )
                        return torch.nn.Sequential(*layers)

                  def forward(self,xin):
                      
                      x = self.conv1(xin)
                      x = self.dropout(x)
                      x = self.conv2( F.relu(x) )
                      x = self.dropout(x)
                      
                      if self.NbRU > 0:
                          x = self.ResNet(x)
                      
                      x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)),dim=1)
                      x = self.dropout(x)
                      x = self.conv3( x )
                      
                      return x
                 
              class Encoder(torch.nn.Module):
                  def __init__(self,dimInp,dimAE,rateDropout=0.):
                      super(Encoder, self).__init__()

                      self.NbBlocks  = nbBlocks
                      self.DimAE     = dimAE

                      self.pool1   = torch.nn.AvgPool2d(sS)
                      self.convTr  = torch.nn.ConvTranspose2d(self.DimAE,dimInp,(sS,sS),stride=(sS,sS),bias=False)          

                      self.NNLR     = BiLinNN(dimInp,dimAE,self.NbBlocks,rateDropout) 
                      self.NNHR     = BiLinNN(dimInp,dimAE,self.NbBlocks,rateDropout)                      
                      self.dropout  = torch.nn.Dropout(rateDropout)
                                        
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
            elif flagAEType == 4: ## Conv model with no use of the central point
              dW       = 3
              dW2      = 1
              sS       = int(4/dx)
              nbBlocks = 0
              rateDr   = 0. * rateDropout
              genSuffixModel = '_GENN_%d_%02d_%02d_%02d_%02d_%02d'%(flagAEType,DimAE,dW,dW2,nbBlocks,int(100*rateDropout))

              class BiLinUnit(torch.nn.Module):
                  def __init__(self,dimIn,dim,dropout=0.):
                      super(BiLinUnit, self).__init__()
                      
                      self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.dropout = torch.nn.Dropout(dropout)
                      self.conv3   = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      
                  def forward(self,xin):
                      
                      x = torch.cat((self.bilin0(xin), self.bilin1(xin) * self.bilin2(xin)),dim=1)
                      x = self.dropout(x)
                      x = self.conv3( x )
                      
                      return x

              class FeatNN(torch.nn.Module):
                  def __init__(self,dimIn,dim,dropout=0.):
                      super(FeatNN, self).__init__()
                      
                      self.conv1   = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False)
                      self.conv2   = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
                      self.dropout = torch.nn.Dropout(dropout)
                      
                  def forward(self,xin):
                      
                      x = self.conv1(xin)
                      x = self.dropout(x)
                      x = self.conv2( F.relu(x) )
                      
                      return x

              class ResLayer(torch.nn.Module):
                  def __init__(self,dimAE,dropout=0.):
                      super(ResLayer, self).__init__()
                      self.RU = self.__make_RU(dimAE,dropout=0.)
                      
                  def __make_RU(self,DimAE,dropout=0.):
                        layers = []
                
                        layers.append(torch.nn.Conv2d(DimAE, 2*DimAE, (2*dW2+1,2*dW2+1), padding=dW2,bias=False))
                        #layers.append( torch.nn.BatchNorm2d(2*DimAE) )
                        layers.append(torch.nn.Dropout(dropout))
                        layers.append(torch.nn.ReLU())
                        layers.append(torch.nn.Conv2d(2*DimAE, DimAE, (2*dW2+1,2*dW2+1), padding=dW2,bias=False))
                        layers.append(torch.nn.Dropout(dropout))
                
                        return torch.nn.Sequential(*layers)

                  def forward(self,x):
                      
                      return x + self.RU(x)
                  
              class Encoder(torch.nn.Module):
                  def __init__(self,dimInp,dimAE,dropout=0.):
                      super(Encoder, self).__init__()

                      self.NbBlocks  = nbBlocks
                      self.DimAE     = dimAE
                      self.pool1     = torch.nn.AvgPool2d(sS)
                      self.convTr    = torch.nn.ConvTranspose2d(self.DimAE,dimInp,(sS,sS),stride=(sS,sS),bias=False)          

                      if self.NbBlocks > 0:
                          self.ResNetLR     = self.__make_ResNet(self.DimAE,self.NbBlocks,dropout)
                          self.ResNetHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,dropout)                      

                      self.FeatNNLR     = FeatNN(dimInp,self.DimAE,dropout)#self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)
                      self.FeatNNHR     = FeatNN(dimInp,self.DimAE,dropout)                      
                      
                      self.dropout      = torch.nn.Dropout(rateDropout)
                      
                      self.BilinNNLR    = self.__make_BilinNN(self.DimAE,self.DimAE,dropout)
                      self.BilinNNHR    = self.__make_BilinNN(self.DimAE,self.DimAE,dropout)  
                      
                      #self.BNLR         = torch.nn.BatchNorm2d(DimAE) 
                      #self.BNHR         = torch.nn.BatchNorm2d(DimAE) 
                   
                  def __make_BilinNN(self,dimInp,dimAE,dropout=0.): 
                        layers = []
                        layers.append( BiLinUnit(dimInp,dimAE,dropout) )
                        return torch.nn.Sequential(*layers)
                    
                  def __make_ResNet(self,dimAE,Nb_RU=1,dropout=0.):
                        layers = []
                        for kk in range(0,Nb_RU):
                            layers.append( ResLayer(dimAE,dropout) )
                        return torch.nn.Sequential(*layers)
                    
                  def forward(self, xinp):
                      
                      ## LR component
                      xLR = self.FeatNNLR( self.pool1(xinp) )
                      #xLR = self.BNLR( xLR )
                      xLR = self.dropout(xLR)
                      if self.NbBlocks > 0:
                          xLR = self.ResNetLR( xLR )                      
                          xLR = self.dropout(xLR)
                      xLR = self.BilinNNLR( xLR )                      
                      xLR = self.dropout(xLR)
                      xLR = self.convTr( xLR ) 
                      
                      # HR component
                      xHR = self.FeatNNHR( xinp )
                      #xLR = self.BNHR( xLR )
                      xHR = self.dropout( xHR )
                      if self.NbBlocks > 0:
                          xHR = self.ResNetHR( xHR )                      
                          xHR = self.dropout( xHR )
                      xHR = self.BilinNNHR( xHR )                      
                      
                      return xLR + xHR
              class Decoder(torch.nn.Module):
                  def __init__(self):
                      super(Decoder, self).__init__()
            
                  def forward(self, x):
                      return 1. * x

            class Model_Phi(torch.nn.Module):
                def __init__(self):
                    super(Model_Phi, self).__init__()
                    self.encoder = Encoder(shapeData[0],DimAE,rateDr)
                    self.decoder = Decoder()
            
                def forward(self, x):
                    x = self.encoder( x )
                    x = self.decoder( x )
                    return x
            model_phi           = Model_Phi()
            print('AE Model type: ' + genSuffixModel)
            print(model_phi)
            print('Number of trainable parameters = %d'%(sum(p.numel() for p in model_phi.parameters() if p.requires_grad)))


            class Model_H(torch.nn.Module):
                def __init__(self):
                    super(Model_H, self).__init__()
                    self.DimObs = 1
                    self.dimObsChannel = np.array([shapeData[0]])
                    
                def forward(self, x,y,mask):
                    dyout  = (x - y) * mask
                    return dyout

            model_H = Model_H()

            ## NN modle for gradient
            class Gradient_img(torch.nn.Module):
                def __init__(self):
                    super(Gradient_img, self).__init__()
            
                    a = np.array([[1., 0., -1.],[2.,0.,-2.],[1.,0.,-1.]])
                    self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
                    self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
            
                    b = np.array([[1., 2., 1.],[0.,0.,0.],[-1.,-2.,-1.]])
                    self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
                    self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
            
                def forward(self , im ):
                    if im.size(1) == 1:
                        G_x = self.convGx( im )
                        G_y = self.convGy( im )
                        G   = torch.sqrt(torch.pow(0.5*G_x,2)+ torch.pow(0.5*G_y,2))
                    else:
                        #G = torch.Tensor(np.zeros((im.size(0),shapeData[0],shapeData[1]-2,shapeData[2]-2)))
                        #G = G.to(device)
            
                        for kk in range(0,im.size(1)):
                            G_x = self.convGx( im[:,kk,:,:].view(-1,1,shapeData[1],shapeData[2]) )
                            G_y = self.convGy( im[:,kk,:,:].view(-1,1,shapeData[1],shapeData[2]) )
                            
                            G_x = G_x.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                            G_y = G_y.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                            nG  = torch.sqrt(torch.pow(0.5*G_x,2)+ torch.pow(0.5*G_y,2))
                            
                            if kk == 0:
                                G = nG.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                            else:
                                G = torch.cat((G,nG.view(-1,1,shapeData[1]-2,shapeData[2]-2)),dim=1)
                    return G
            gradient_img = Gradient_img()
            

            class ModelLR(torch.nn.Module):
                def __init__(self):
                    super(ModelLR, self).__init__()
            
                    self.pool  = torch.nn.AvgPool2d((16,16))
            
                def forward(self , im ):
                
                    return self.pool(im)
            model_LR = ModelLR()
                            
        ###############################################################
        ## Given a trained AE, train an assimilation solver
        elif flagProcess[kk] == 4:

            print('genSuffixModel:'+genSuffixModel)    
            
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
            print(shapeData.shape)

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )
                         
            alphaMSE         = 0.1
            alphaProj        = 0.5
            #alpha4DVar       = np.array([0.1,1.])            
            alpha_SR         = 0.5
            alpha_LR         = 0.5#1e4
            
            num_epochs     = 200
            SaveStep       = 20

            IterUpdate     = [0,25,50,75,100,125,800]#[0,2,4,6,9,15]
            #NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
            NbGradIter     = [5,5,10,10,15,15,20,20,20]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
            lrUpdate       = [1e-3,1e-4,1e-4,1e-5,1e-4,1e-5,1e-5,1e-6,1e-7]
            
            comptUpdate = 4
            iterInit    = 95#498
            
             # NiterProjection,NiterGrad: global variables
            NBGradCurrent   = NbGradIter[comptUpdate-1]
            lrCurrent       = lrUpdate[comptUpdate-1]
            
            # Define models
            m_Grad          = NN_4DVar.model_GradUpdateLSTM(shapeData,UsePriodicBoundary,dimGradSolver,rateDropout) 
            print(m_Grad)                
            if GradType == 1 :
                m_normObs   = NN_4DVar.Model_WeightedL2Norm()
                m_normPrior = NN_4DVar.Model_WeightedL2Norm()
            elif GradType == 2 :
                m_normObs   = NN_4DVar.Model_WeightedL1Norm()
                m_normPrior = NN_4DVar.Model_WeightedL2Norm()
            elif GradType == 3 :
                m_normObs   = NN_4DVar.Model_WeightedLorenzNorm()
                m_normPrior = NN_4DVar.Model_WeightedL2Norm()

            model           = NN_4DVar.Solver_Grad_4DVarNN(model_phi,model_H, m_Grad, m_normObs, m_normPrior, shapeData,NBGradCurrent)
            modelSave       = NN_4DVar.Solver_Grad_4DVarNN(model_phi,model_H, m_Grad, m_normObs, m_normPrior, shapeData,NBGradCurrent) 

            model           = model.to(device)
            gradient_img    = gradient_img.to(device)
            
            print('4DVar model: Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            print('----- MSE xTr OI: %.3e'%(np.mean((x_trainOI-x_train)**2)/var_Tr))
            print('----- MSE xTt OI: %.3e'%(np.mean((x_testOI-x_test)**2)/var_Tt))

            flagLoadModel   = 1
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_128x128x05_GENN_1_10_05_SZeros_HRObs_OIObs_MS_Grad_03_02_10_50_modelPHI_iter060.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_128x128x05_GENN_1_10_05_SZeros_HRObs_OIObs_MS_Grad_01_02_05_50_modelPHI_iter020.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_1_10_05_SZeros_HRObs_OIObs_MS_Grad_01_02_10_50_modelPHI_iter080.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_1_10_05_HRObs_OIObs_MS_Grad_01_02_10_50_modelPHI_iter080.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_200_modelPHI_iter080.mod'
            
            if flagLoadModel == 1:
                print('.... load AE+Grad+Sampling model: '+fileAEModelInit)
                model.phi_r.load_state_dict(torch.load(fileAEModelInit))
                model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelGrad_iter')))
                model.model_VarCost.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelVarCost_iter')))
                
            # multi-GPU model
            flag_UseMultiGPU = False
            if ( torch.cuda.device_count() > 1 ) & ( flag_UseMultiGPU == True ):

                modelMultiGPU          = model_to_MultiGPU(model)
                modelMultiGPU_LR       = model_to_MultiGPU_basic(model_LR)
            else:
                modelMultiGPU = model
                modelMultiGPU_LR       = model_LR

            # optimization setting: freeze or not the AE
            lambda_LRAE = 1.
            optimizer   = optim.Adam([{'params': modelMultiGPU.model_Grad.parameters(), 'lr': lrCurrent},
                                      {'params': modelMultiGPU.model_VarCost.parameters(), 'lr': lrCurrent},
                                    {'params': modelMultiGPU.phi_r.parameters(), 'lr': lambda_LRAE*lrCurrent},
                                    ], lr=lrCurrent)
            
            # Create training/test data pytorch tensors and associated  
            # list of tensors (xx[n][x] to access the nth sample for the xth field)
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainMask),torch.Tensor(x_train)) # create your datset
            #test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testMask),torch.Tensor(x_test)) # create your datset
            val_dataset          = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valMask),torch.Tensor(x_val)) # create your datset
            
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            }            
            dataset_sizes = {'train': len(training_dataset), 'val': len(val_dataset)}
        
            # training function for dinAE
            since = time.time()
        
            
            # Suffix for file naming
            genSuffixModelBase = genSuffixModel
            
            genSuffixModel = genSuffixModelBase+genSuffixObs
            if lambda_LRAE == 0. :
                genSuffixModel = genSuffixModel+'_NoFTrAE'
            
            genSuffixModel = genSuffixModel+'_HRObs'                
            genSuffixModel = genSuffixModel+'_OIObs'
            genSuffixModel = genSuffixModel+'_MS'

            genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))+'_'+str('%02d'%(dimGradSolver))

            print('...... Suffix trained models: '+genSuffixModel)
    
            best_model_wts         = copy.deepcopy(model.state_dict())
            best_loss = 1e10
    
            # loss weghing wrt time
            w_            = np.zeros(dT,)
            w_[int(dT/2)] = 1.
            #w_            = (1./dT)*np.ones(dT,)
            wLoss         = torch.Tensor(w_)
            wLoss         = wLoss.to(device)

            betaX  = 1.0
            betagX = 1.0
            # compute the mean loss for OI
            for phase in ['train']:        
                model.eval()
                    
                running_loss_GOI     = 0.
                running_loss_OI      = 0.
                num_loss = 0
                
                for targets_OI,inputs_Mask,targets_GT in dataloaders[phase]:
                    targets_OI     = targets_OI.to(device)
                    inputs_Mask    = inputs_Mask.to(device)
                    targets_GT     = targets_GT.to(device)
            
                                                
                    # gradient norm field
                    g_targets_GT   = gradient_img( targets_GT )
    
                    loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT-targets_OI,wLoss)
                    loss_GOI   = NN_4DVar.compute_WeightedLoss(gradient_img( targets_OI )-g_targets_GT,wLoss)

                            
                    running_loss_GOI         += loss_GOI.item() * targets_GT.size(0)
                    running_loss_OI          += loss_OI.item() * targets_GT.size(0)
           
                    num_loss                 += targets_GT.size(0)
              
                epoch_loss_GOI    = running_loss_GOI  / num_loss
                epoch_loss_OI     = running_loss_OI / num_loss
                   
                betaX  = 1. / epoch_loss_OI
                betagX = 1. / epoch_loss_GOI
                
                print(".... MSE(Tr) OI %.3f -- MSE(Tr) gOI %.3f "%(epoch_loss_OI,epoch_loss_GOI))
                        
            for epoch in range(iterInit,num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
            
                if ( epoch == IterUpdate[comptUpdate] ) & ( epoch > 0 ):
                    # update GradFP parameters
                    #NBProjCurrent = NbProjection[comptUpdate]
                    NBGradCurrent = NbGradIter[comptUpdate]
                    lrCurrent     = lrUpdate[comptUpdate]
                    
                    #if( (NBProjCurrent != NbProjection[comptUpdate-1]) | (NBGradCurrent != NbGradIter[comptUpdate-1]) ):
                    if( (NBGradCurrent != NbGradIter[comptUpdate-1]) ):
                        print("..... ")
                        print("..... ")
    
                        print('..... Update model architecture')
                        print("..... ")
                        model.NGrad = NbGradIter[comptUpdate]
                        print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d"%(modelMultiGPU.NGrad))
                                              
                        mm  = 0
                        lr = np.array([lrCurrent,lrCurrent,lambda_LRAE*lrCurrent,lambda_LRAE*lrCurrent])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr[mm]
                            mm += 1
                         
                        # Suffix for file naming
                        genSuffixModel = genSuffixModelBase+genSuffixObs
                        
                        genSuffixModel = genSuffixModel+'_HRObs'                        
                        genSuffixModel = genSuffixModel+'_OIObs'
                        genSuffixModel = genSuffixModel+'_MS'
            
                            
                        #genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
                        genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))+'_'+str('%02d'%(dimGradSolver))

                    else:
                        # update optimizer learning rate
                        print('..... Update learning rate')
                        mm = 0
                        lr = np.array([lrCurrent,lrCurrent,lambda_LRAE*lrCurrent,lambda_LRAE*lrCurrent])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr[mm]
                            mm += 1

                    # update counter
                    if comptUpdate < len(IterUpdate)-1:
                        comptUpdate += 1

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:        
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()
            
                    running_loss         = 0.0
                    running_loss_All     = 0.
                    running_loss_GAll    = 0.
                    running_loss_All1    = 0.
                    running_loss_GAll1   = 0.
                    running_loss_R       = 0.
                    running_loss_I       = 0.
                    running_loss_AE      = 0.
                    running_loss_AE_GT   = 0.
                    running_loss_Sampling  = 0.
                    running_loss_Sampling2 = 0.
                    num_loss             = 0
                    running_mean_GAll    = 0.
                    
                    running_loss_GOI     = 0.
                    running_loss_OI      = 0.
                    
                    running_loss_LR = 0.
                    running_loss_SR = 0.
            
                    # Iterate over data.
                    for targets_OI,inputs_Mask,targets_GT in dataloaders[phase]:
                        targets_OI     = targets_OI.to(device)
                        inputs_Mask    = inputs_Mask.to(device)
                        targets_GT     = targets_GT.to(device)
                        #print(inputs.size(0))
                
                                                    
                        # zero the parameter gradients
                        optimizer.zero_grad()
            
                        # use low-resolution
                        targets_GTLR = modelMultiGPU_LR(targets_OI)
                        #targets_GTSR = modelMultiGPU_SRLin(targets_OI)
                        
                        # sampling mask
                        # O:dT indices refer to the large-scale component with associated OI observation
                        new_masks             = inputs_Mask.repeat(1,2,1,1)
                        new_masks[:,0:dT,:,:] = 1.              
                            
                        inputs_init    = torch.cat((targets_OI , inputs_Mask * (targets_GT-targets_OI)),dim=1)
                        inputs_missing = torch.cat((targets_OI , inputs_Mask * (targets_GT-targets_OI)),dim=1)

                        # gradient norm field
                        g_targets_GT   = gradient_img( targets_GT )
    
                        # need to evaluate grad/backward during the evaluation and training phase for model_AE
                        with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                            inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                            outputs,hidden_new,cell_new,normgrad = modelMultiGPU(inputs_init,inputs_missing,new_masks)
            
           
                            outputsSLRHR = outputs
                            outputsSLR   = outputs[:,0:dT,:,:]
                            outputs      = outputsSLR + outputs[:,dT:,:,:]

                            # losses
                            g_outputs   = gradient_img( outputs )

                            loss_All    = NN_4DVar.compute_WeightedLoss((outputs-targets_GT) , wLoss)
                            loss_GAll   = NN_4DVar.compute_WeightedLoss(g_outputs-g_targets_GT,wLoss)
                            
                            #print('... %f'%loss_All.cpu().detach().numpy())

                            loss_All1    = NN_4DVar.compute_WeightedLoss(outputs-targets_OI,wLoss)
                            loss_GAll1   = NN_4DVar.compute_WeightedLoss(g_outputs-gradient_img( targets_OI ),wLoss)

                            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT-targets_OI,wLoss)
                            loss_GOI   = NN_4DVar.compute_WeightedLoss(gradient_img( targets_OI )-g_targets_GT,wLoss)

                            mean_GAll   = NN_4DVar.compute_WeightedLoss(g_targets_GT,wLoss)
                            
                            loss_AE    = torch.mean((modelMultiGPU.phi_r(outputsSLRHR) - outputsSLRHR)**2 ) 
                            yGT        = torch.cat((targets_GT,targets_GT-outputsSLR),dim=1)
                            #yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
                            loss_AE_GT = torch.mean((modelMultiGPU.phi_r(yGT) - yGT)**2 ) 
            
                            # training loss
                            loss        = alphaMSE * (betaX * loss_All + betagX * loss_GAll) + 0.5 * alphaProj * ( loss_AE + loss_AE_GT )

                            loss_SR    = NN_4DVar.compute_WeightedLoss(outputsSLR-targets_OI,wLoss)
                            loss_LR    = NN_4DVar.compute_WeightedLoss(modelMultiGPU_LR(outputs)-targets_GTLR,wLoss)
                               
                            loss      += alpha_LR * loss_LR + alpha_SR * loss_SR

                            # backward + optimize only if in training phase
                            if( phase == 'train' ):
                                loss.backward()
                                optimizer.step()
                        
                            # statistics
                            running_loss             += loss.item() * inputs_missing.size(0)
                            running_loss_All         += loss_All.item() * inputs_missing.size(0)
                            running_loss_GAll        += loss_GAll.item() * inputs_missing.size(0)
                            running_loss_All1        += loss_All1.item() * inputs_missing.size(0)
                            running_loss_GAll1       += loss_GAll1.item() * inputs_missing.size(0)
                            running_loss_AE          += loss_AE.item() * inputs_missing.size(0)
                            running_loss_AE_GT       += loss_AE_GT.item() * inputs_missing.size(0)
                            
                            running_loss_GOI         += loss_GOI.item() * inputs_missing.size(0)
                            running_loss_OI          += loss_OI.item() * inputs_missing.size(0)

                            running_loss_LR      += loss_LR.item() * inputs_missing.size(0)
                            running_loss_SR      += loss_SR.item() * inputs_missing.size(0)

                            running_mean_GAll        += mean_GAll.item() * inputs_missing.size(0)
            
                            num_loss                 += inputs_missing.size(0)
              
                    epoch_loss        = running_loss / num_loss
                    epoch_loss_All    = running_loss_All / num_loss
                    epoch_loss_All1   = running_loss_All1 / num_loss
                    epoch_loss_AE     = running_loss_AE / num_loss
                    epoch_loss_AE_GT  = running_loss_AE_GT / num_loss
            
                    epoch_loss_GAll   = running_loss_GAll / running_mean_GAll
                    epoch_loss_GAll1  = running_loss_GAll1 / running_mean_GAll
                    epoch_loss_GOI    = running_loss_GOI  / running_mean_GAll
                    epoch_loss_OI     = running_loss_OI / num_loss
                    
                    epoch_loss_LR      = running_loss_LR / num_loss
                    epoch_loss_SR      = running_loss_SR / num_loss

                    print('%f -- %f'%(epoch_loss_All,epoch_loss_GAll))
                    
                    if phase == 'val' :
                        epoch_loss_All  = epoch_loss_All #/ var_Tt
                        epoch_loss_All1 = epoch_loss_All1 / var_Tt
                        epoch_loss_AE   = epoch_loss_AE / var_Tt
                        epoch_loss_AE_GT  = epoch_loss_AE_GT / var_Tt
            
                        epoch_loss_OI     = epoch_loss_OI / var_Tt
                        epoch_loss_LR      = epoch_loss_LR / var_Tt
                        epoch_loss_SR      = epoch_loss_SR / var_Tt
                    elif phase == 'train' :
                        epoch_loss_All  = epoch_loss_All #/ var_Tr
                        epoch_loss_All1 = epoch_loss_All1 / var_Tr
                        epoch_loss_AE   = epoch_loss_AE / var_Tr
                        epoch_loss_AE_GT  = epoch_loss_AE_GT / var_Tr
                        
                        epoch_loss_OI     = epoch_loss_OI / var_Tr
                        epoch_loss_LR      = epoch_loss_LR / var_Tr
                        epoch_loss_SR      = epoch_loss_SR / var_Tr
            
                    print(' Loss OI: {:.4e} -- Loss GOI: {:.4e}'.format(epoch_loss_OI,epoch_loss_GOI),flush=True)
                                       
                    print('{} Loss: {:.4e} NLossAll: {:.4e} NLossLR: {:.4e} NLossSLR: {:.4e} NLossGAll: {:.4e} NLossAE: {:.4e}/{:.4e} '.format(
                            phase, epoch_loss,epoch_loss_All,epoch_loss_LR,epoch_loss_SR,epoch_loss_GAll,epoch_loss_AE,epoch_loss_AE_GT),flush=True)
                    
                    # deep copy the model
                    if phase == 'val' and epoch_loss < best_loss:
                        best_loss              = epoch_loss
                        best_model_wts         = copy.deepcopy(model.state_dict())
            
                        print('..')
                        print('.. Weighing factors (observation model)')
                        print( modelMultiGPU.model_VarCost.WObs.size() )
                        for jj in range(0,modelMultiGPU.model_H.DimObs):
                            print(modelMultiGPU.model_VarCost.WObs[jj,0:modelMultiGPU.model_H.dimObsChannel[jj]].detach().cpu().numpy())
                        print('.. Weighing factors (prior model)')
                        print(modelMultiGPU.model_VarCost.WReg.detach().cpu().numpy())
                        print('.. Epsilon values (obs model)')
                        print(modelMultiGPU.model_VarCost.epsObs.detach().cpu().numpy())
                        print('.. Epsilon values (prior model)')
                        print(modelMultiGPU.model_VarCost.epsReg.detach().cpu().numpy())
                        print('.. Lambda values (obs model)')
                        print(modelMultiGPU.model_VarCost.alphaObs.detach().cpu().numpy())
                        print('.. Lambda value (prior model)')
                        print(modelMultiGPU.model_VarCost.alphaReg.detach().cpu().numpy())
                        
                # Save model
                if ( flagSaveModel == 1 )  & ( ( np.mod(epoch,SaveStep) == 0  ) | ( epoch == num_epochs - 1) )  :                          
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelPHI_iter%03d'%(epoch)+'.mod'
                    
                    modelSave.load_state_dict(best_model_wts)
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(modelSave.phi_r.state_dict(), fileMod)
                    #torch.save(model.model_AE.state_dict(), fileMod)
    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelGrad_iter%03d'%(epoch)+'.mod'
                    print('.................. Gradient model '+fileMod)
                    torch.save(modelSave.model_Grad.state_dict(), fileMod)
                    #torch.save(model.model_Grad.state_dict(), fileMod)

                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelVarCost_iter%03d'%(epoch)+'.mod'
                    print('.................. VarCost model '+fileMod)
                    torch.save(modelSave.model_VarCost.state_dict(), fileMod)
                    #torch.save(model.model_Grad.state_dict(), fileMod)

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))

            # Save model
            if ( flagSaveModel == 1 )  & ( ( np.mod(epoch,SaveStep) == 0  ) | ( epoch == num_epochs - 1) )  :                          
                 fileMod = dirSAVE+genFilename+genSuffixModel+'_modelPHI_iter%03d'%(epoch)+'.mod'
                 
                 modelSave.load_state_dict(best_model_wts)
                 print('.................. Auto-Encoder '+fileMod)
                 torch.save(modelSave.phi_r.state_dict(), fileMod)
                 #torch.save(model.model_AE.state_dict(), fileMod)
 
                 fileMod = dirSAVE+genFilename+genSuffixModel+'_modelGrad_iter%03d'%(epoch)+'.mod'
                 print('.................. Gradient model '+fileMod)
                 torch.save(modelSave.model_Grad.state_dict(), fileMod)
                 #torch.save(model.model_Grad.state_dict(), fileMod)

                 fileMod = dirSAVE+genFilename+genSuffixModel+'_modelVarCost_iter%03d'%(epoch)+'.mod'
                 print('.................. VarCost model '+fileMod)
                 torch.save(modelSave.model_VarCost.state_dict(), fileMod)
 
        ###############################################################
        ## Load a trained model and save results for 
        ## test data
        elif flagProcess[kk] == 5:

            
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
            print(shapeData.shape)

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )
            var_Val   = np.var( x_val )
                         
            #alpha            = np.array([0.1,0.5])#np.array([1.,0.1])
            alphaMSE         = 0.1
            alphaProj        = 0.5
            alpha_SR         = 0.5
            alpha_LR         = 0.5#1e4
            
            # NiterProjection,NiterGrad: global variables
            # bug for NiterProjection = 0
            #model_AE_GradFP = Model_AE_GradFP(model_AE2,shapeData,NiterProjection,NiterGrad,GradType,OptimType)
            NBGradCurrent   = 10
            
            flagSaveTestResults = True

            #model           = NN_4DVar.Model_4DVarNN_Grad(model_AE,model_H,shapeData,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary,dimGradSolver,rateDropout)        
            m_Grad          = NN_4DVar.model_GradUpdateLSTM(shapeData,UsePriodicBoundary,dimGradSolver,rateDropout)                 
            if GradType == 1 :
                m_normObs   = NN_4DVar.Model_WeightedL2Norm()
                m_normPrior = NN_4DVar.Model_WeightedL2Norm()
            elif GradType == 2 :
                m_normObs   = NN_4DVar.Model_WeightedL1Norm()
                m_normPrior = NN_4DVar.Model_WeightedL2Norm()
            elif GradType == 3 :
                m_normObs   = NN_4DVar.Model_WeightedLorenzNorm()
                m_normPrior = NN_4DVar.Model_WeightedL2Norm()
            model           = NN_4DVar.Solver_Grad_4DVarNN(model_phi,model_H, m_Grad, m_normObs, m_normPrior, shapeData,NBGradCurrent)


            model           = model.to(device)
            gradient_img    = gradient_img.to(device)
            
            print('4DVar model: Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            print('----- MSE xTr OI: %.3e'%(np.mean((x_trainOI-x_train)**2)/var_Tr))
            print('----- MSE xTt OI: %.3e'%(np.mean((x_testOI-x_test)**2)/var_Tt))
                                    
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_128x128x05_GENN_1_10_05_SZeros_HRObs_OIObs_MS_Grad_03_02_10_50_modelPHI_iter060.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_128x128x05_GENN_1_10_05_SZeros_HRObs_OIObs_MS_Grad_01_02_05_50_modelPHI_iter020.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_1_10_05_SZeros_HRObs_OIObs_MS_Grad_01_02_10_50_modelPHI_iter080.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_1_10_05_HRObs_OIObs_MS_Grad_01_02_05_50_modelPHI_iter040.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_1_10_05_00_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter120.mod'
            
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_05_01_00_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter120.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_03_01_01_00_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter120.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter140.mod'
            #fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_03_01_01_05_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter120.mod'
            
            #fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_03_01_02_00_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter180.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_3_10_03_01_02_25_HRObs_OIObs_MS_Grad_01_02_05_50_modelPHI_iter040.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_03_01_01_10_HRObs_OIObs_MS_Grad_01_02_15_50_modelPHI_iter140.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_10_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_100_modelPHI_iter080.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_20_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_100_modelPHI_iter080.mod'
            fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_100_modelPHI_iter080.mod'
            #fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_15_100_modelPHI_iter120.mod'
            #fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_20_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_15_100_modelPHI_iter199.mod'
            #fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_200_modelPHI_iter080.mod'
            
            print('.... load AE+Grad+Sampling model: '+fileAEModelInit)
            model.phi_r.load_state_dict(torch.load(fileAEModelInit))
            model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelGrad_iter')))
            model.model_VarCost.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelVarCost_iter')))
            
                
            # multi-GPU model
            flag_UseMultiGPU = False
            if ( torch.cuda.device_count() > 1 ) & ( flag_UseMultiGPU == True ):
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                modelMultiGPU          = model_to_MultiGPU(model)
                modelMultiGPU_LR       = model_to_MultiGPU_basic(model_LR)
                #modelMultiGPU_Sampling = model_S
            else:
                modelMultiGPU = model
                modelMultiGPU_LR       = model_LR

 
            # Create training/test data pytorch tensors and associated  
            # list of tensors (xx[n][x] to access the nth sample for the xth field)
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testMask),torch.Tensor(x_test)) # create your datset
            val_dataset          = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valMask),torch.Tensor(x_val)) # create your datset
            
            dataloaders = {
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            
            dataset_sizes = {'val': len(val_dataset), 'test': len(test_dataset)}
        
            # training function for dinAE
            since = time.time()
        
    
            # loss weghing wrt time
            w_            = np.zeros(dT,)
            w_[int(dT/2)] = 1.
            #w_            = (1./dT)*np.ones(dT,)
            wLoss         = torch.Tensor(w_)
            wLoss         = wLoss.to(device)

            betaX = 1.0
            betagX = 1.0
            # compute the mean loss for OI
            for phase in ['val','test']:        
                #model.train()
                    
                running_loss_GOI     = 0.
                running_loss_OI      = 0.
                num_loss = 0
                
                for targets_OI,inputs_Mask,targets_GT in dataloaders[phase]:
                    targets_OI     = targets_OI.to(device)
                    inputs_Mask    = inputs_Mask.to(device)
                    targets_GT     = targets_GT.to(device)
            
                                                
                    # gradient norm field
                    g_targets_GT   = gradient_img( targets_GT )
    

                    loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT-targets_OI,wLoss)
                    loss_GOI   = NN_4DVar.compute_WeightedLoss(gradient_img( targets_OI )-g_targets_GT,wLoss)

                            
                    running_loss_GOI         += loss_GOI.item() * targets_GT.size(0)
                    running_loss_OI          += loss_OI.item() * targets_GT.size(0)
           
                    num_loss                 += targets_GT.size(0)
              
                epoch_loss_GOI    = running_loss_GOI  / num_loss
                epoch_loss_OI     = running_loss_OI / num_loss
                   
                betaX  = 1. / epoch_loss_OI
                betagX = 1. / epoch_loss_GOI
                
                if phase == 'val' :
                    print(".... MSE(validation) OI %.3f -- MSE(validation) gOI %.3f "%(epoch_loss_OI,epoch_loss_GOI))
                if phase == 'test' :
                    print(".... MSE(test) OI %.3f -- MSE(test) gOI %.3f "%(epoch_loss_OI,epoch_loss_GOI))
                        

            comptUpdate = 1
            iterInit    = 0#498

            if 1*1 :
                
                # Each epoch has a training and validation phase
                #for phase in ['train','val']:        
                #for phase in ['val','test']:        
                for phase in ['val','test']:        
                    model.eval()
                    
                    com = '....  Process '+phase+' data'
                    print(com,flush=True)
            
                    running_loss         = 0.0
                    running_loss_All     = 0.
                    running_loss_GAll    = 0.
                    running_loss_All1    = 0.
                    running_loss_GAll1   = 0.
                    running_loss_R       = 0.
                    running_loss_I       = 0.
                    running_loss_AE      = 0.
                    running_loss_AE_GT   = 0.
                    num_loss             = 0
                    running_mean_GAll    = 0.
                    
                    running_loss_GOI     = 0.
                    running_loss_OI      = 0.
                    
                    running_loss_LR = 0.
                    running_loss_SR = 0.
            
                    x_test_rec = []
                    x_test_OI  = []
                    x_test_GT  = []
                    x_test_S   = []
                    x_test_M1  = []
                    x_test_M2  = []

                    # Iterate over data.
                    for targets_OI,inputs_Mask,targets_GT in dataloaders[phase]:
                        targets_OI     = targets_OI.to(device)
                        inputs_Mask    = inputs_Mask.to(device)
                        targets_GT     = targets_GT.to(device)
                                                                
                        # use low-resolution
                        targets_GTLR = modelMultiGPU_LR(targets_OI)
                        
                        # sampling mask
                        # O:dT indices refer to the large-scale component with associated OI observation
                        new_masks = inputs_Mask.repeat(1,2,1,1)
                        new_masks[:,0:dT,:,:] = 1.              
                            
                        inputs_init    = torch.cat((targets_OI , inputs_Mask * (targets_GT-targets_OI)),dim=1)
                        inputs_missing = torch.cat((targets_OI , inputs_Mask * (targets_GT-targets_OI)),dim=1)

                        # gradient norm field
                        g_targets_GT   = gradient_img( targets_GT )
    
                        # need to evaluate grad/backward during the evaluation and training phase for model_AE
                        with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                            inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                            outputs,hidden_new,cell_new,normgrad = modelMultiGPU(inputs_init,inputs_missing,new_masks)
            
           
                            outputsSLRHR = outputs
                            outputsSLR   = outputs[:,0:dT,:,:].view(-1,dT,shapeData[1],shapeData[2])
                            outputs      = outputsSLR + outputs[:,dT:,:,:].view(-1,dT,shapeData[1],shapeData[2])

                            # losses
                            g_outputs   = gradient_img( outputs )
                            
                            loss_All    = NN_4DVar.compute_WeightedLoss((outputs-targets_GT) , wLoss)
                            loss_GAll   = NN_4DVar.compute_WeightedLoss(g_outputs-g_targets_GT,wLoss)

                            loss_All1    = NN_4DVar.compute_WeightedLoss(outputs-targets_OI,wLoss)
                            loss_GAll1   = NN_4DVar.compute_WeightedLoss(g_outputs-gradient_img( targets_OI ),wLoss)

                            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT-targets_OI,wLoss)
                            loss_GOI   = NN_4DVar.compute_WeightedLoss(gradient_img( targets_OI )-g_targets_GT,wLoss)

                            mean_GAll   = NN_4DVar.compute_WeightedLoss(g_targets_GT,wLoss)
                            
                            loss_AE    = torch.mean((modelMultiGPU.phi_r(outputsSLRHR) - outputsSLRHR)**2 ) 
                            yGT        = torch.cat((targets_GT,outputsSLR-targets_GT),dim=1)
                            loss_AE_GT = torch.mean((modelMultiGPU.phi_r(yGT) - yGT)**2 ) 
                                #loss_AE_GT = compute_WeightedLoss(modelMultiGPU.model_AE(yGT),yGT,wLoss)
                       
                            # training loss
                            #loss        = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
                            loss        = alphaMSE * (betaX * loss_All + betagX * loss_GAll) + 0.5 * alphaProj * ( loss_AE + loss_AE_GT )
                            loss_SR    = NN_4DVar.compute_WeightedLoss(outputsSLR-targets_OI,wLoss)
                            loss_LR    = NN_4DVar.compute_WeightedLoss(modelMultiGPU_LR(outputs)-targets_GTLR,wLoss)
                                
                            loss    += alpha_LR * loss_LR + alpha_SR * loss_SR

                            loss_All    = NN_4DVar.compute_WeightedLoss((outputs-targets_GT) , wLoss)
                            loss_GAll   = NN_4DVar.compute_WeightedLoss(g_outputs-g_targets_GT,wLoss)
            
                            # statistics
                            running_loss             += loss.item() * inputs_missing.size(0)
                            running_loss_All         += loss_All.item() * inputs_missing.size(0)
                            running_loss_GAll        += loss_GAll.item() * inputs_missing.size(0)
                            running_loss_All1        += loss_All1.item() * inputs_missing.size(0)
                            running_loss_GAll1       += loss_GAll1.item() * inputs_missing.size(0)
                            running_loss_AE          += loss_AE.item() * inputs_missing.size(0)
                            running_loss_AE_GT       += loss_AE_GT.item() * inputs_missing.size(0)
                            
                            running_loss_GOI         += loss_GOI.item() * inputs_missing.size(0)
                            running_loss_OI          += loss_OI.item() * inputs_missing.size(0)

                            running_loss_LR      += loss_LR.item() * inputs_missing.size(0)
                            running_loss_SR      += loss_SR.item() * inputs_missing.size(0)

                            running_mean_GAll        += mean_GAll.item() * inputs_missing.size(0)
            
                            num_loss                 += inputs_missing.size(0)
              
                        # store resuls
                        if ( phase == 'test' ) & (flagSaveTestResults == True) :
                            if( len(x_test_rec) == 0 ):
                                x_test_rec = np.copy(outputs.cpu().detach().numpy())
                                x_test_OI  = np.copy(targets_OI.cpu().detach().numpy())
                                x_test_GT  = np.copy(targets_GT.cpu().detach().numpy())

                                x_test_rec = stdTr * x_test_rec + meanTr
                                x_test_OI  = stdTr * x_test_OI + meanTr
                                x_test_GT  = stdTr * x_test_GT + meanTr

                                x_test_M1  = np.copy(inputs_Mask.cpu().detach().numpy())
                            else:
                                x_test_rec = np.concatenate((x_test_rec, stdTr * outputs.cpu().detach().numpy() + meanTr),axis=0)
                                x_test_OI  = np.concatenate((x_test_OI, stdTr * targets_OI.cpu().detach().numpy() + meanTr),axis=0)
                                x_test_GT  = np.concatenate((x_test_GT, stdTr * targets_GT.cpu().detach().numpy() + meanTr),axis=0)
                                
                                x_test_M1  = np.concatenate((x_test_M1,inputs_Mask.cpu().detach().numpy()),axis=0)

                    epoch_loss       = running_loss / num_loss
                    epoch_loss_All    = running_loss_All / num_loss
                    epoch_loss_All1   = running_loss_All1 / num_loss
                    epoch_loss_AE     = running_loss_AE / num_loss
                    epoch_loss_AE_GT  = running_loss_AE_GT / num_loss
            
                    epoch_loss_GAll   = running_loss_GAll / running_mean_GAll
                    epoch_loss_GAll1  = running_loss_GAll1 / running_mean_GAll
                    epoch_loss_GOI    = running_loss_GOI  / running_mean_GAll
                    epoch_loss_OI     = running_loss_OI / num_loss
                    
                    epoch_loss_LR      = running_loss_LR / num_loss
                    epoch_loss_SR      = running_loss_SR / num_loss

                    if phase == 'test' :
                        epoch_loss_All  = epoch_loss_All / var_Tt
                        epoch_loss_All1 = epoch_loss_All1 / var_Tt
                        epoch_loss_AE   = epoch_loss_AE / var_Tt
                        epoch_loss_AE_GT  = epoch_loss_AE_GT / var_Tt
            
                        epoch_loss_OI     = epoch_loss_OI / var_Tt
                        epoch_loss_LR      = epoch_loss_LR / var_Tt
                        epoch_loss_SR      = epoch_loss_SR / var_Tt
                    elif phase == 'val' :
                        epoch_loss_All  = epoch_loss_All / var_Val
                        epoch_loss_All1 = epoch_loss_All1 / var_Val
                        epoch_loss_AE   = epoch_loss_AE / var_Val
                        epoch_loss_AE_GT  = epoch_loss_AE_GT / var_Val
                        
                        epoch_loss_OI     = epoch_loss_OI / var_Val
                        epoch_loss_LR      = epoch_loss_LR / var_Val
                        epoch_loss_SR      = epoch_loss_SR / var_Val            

                    if phase == 'val' :
                        print('----------- Validation dataset ---------------------')
                    elif phase == 'test':
                        print('----------- Test dataset ---------------------')
                    print(' Loss OI: {:.4e} -- Loss GOI: {:.4e}'.format(epoch_loss_OI,epoch_loss_GOI),flush=True)
                    #print('{} Loss: {:.4e} NLossAll: {:.4e} NLossR: {:.4e} NLossI: {:.4e} NLossAE: {:.4e} LossSampling: {:.4e} '.format(
                    #    phase, epoch_loss,epoch_loss_All,epoch_loss_R,epoch_loss_I,epoch_loss_AE,epoch_loss_Sampling),flush=True)
                                       
                    print('{} Loss: {:.4e} NLossAll: {:.4e} NLossLR: {:.4e} NLossSLR: {:.4e} NLossGAll: {:.4e} NLossAE: {:.4e}/{:.4e} '.format(
                            phase, epoch_loss,epoch_loss_All,epoch_loss_LR,epoch_loss_SR,epoch_loss_GAll,epoch_loss_AE,epoch_loss_AE_GT),flush=True)

                # Save netcdf filere
                if ( flagSaveTestResults == True ) & ( phase == 'test' ):
                    filesave = fileAEModelInit.replace('.mod','_ResTestData.nc')
                    
                    domain = 'GULFSTREAM'
                    dwscale = 1
                    
                    def save_NetCDF(saved_path1,x_test_rec,x_test_OI,x_test_GT,x_test_M2):     
                    
                        if domain=="OSMOSIS":
                            extent     = [-19.5,-11.5,45.,55.]
                            indLat     = 200
                            indLon     = 160
                        elif domain=='GULFSTREAM':
                            extent     = [-65.,-55.,33.,43.]
                            indLat     = 200
                            indLon     = 200
                        else:
                            extent=[-65.,-55.,30.,40.]
                            indLat     = 200
                            indLon     = 200
                    
                        lon = np.arange(extent[0],extent[1],1/(20/dwscale))
                        lat = np.arange(extent[2],extent[3],1/(20/dwscale))
                        indLat     = int(indLat/dwscale)
                        indLon     = int(indLon/dwscale)
                        lon = lon[:indLon]
                        lat = lat[:indLat]
                        #extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
                    
                        mesh_lat, mesh_lon = np.meshgrid(lat, lon)
                        mesh_lat = mesh_lat.T
                        mesh_lon = mesh_lon.T
                                                            
                        indN_Tt = np.concatenate([np.arange(60,80)])
                        time    = [ datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01",'%Y-%m-%d')\
                                              + datetime.timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]
                        # interpolation
                        xrdata = xr.Dataset(\
                                    data_vars={'longitude': (('lat','lon'),mesh_lon),\
                                               'latitude' : (('lat','lon'),mesh_lat),\
                                               'Time'     : (('time'),time),\
                                               'ssh'  : (('time','lat','lon'),x_test_rec[:,int(dT/2),:,:]),\
                                               'sshOI'  : (('time','lat','lon'),x_test_OI[:,int(dT/2),:,:]),\
                                               'sshGT'  : (('time','lat','lon'),x_test_GT[:,int(dT/2),:,:]),\
                                               'mask'   : (('time','lat','lon'),x_test_M1[:,int(dT/2),:,:])},\
                                    coords={'lon': lon,'lat': lat,'time': indN_Tt})
                        xrdata.time.attrs['units']='days since 2012-10-01 00:00:00'
                        xrdata.to_netcdf(path=saved_path1, mode='w')

                    print('.... Save results for test dataset:  '+filesave)
                    save_NetCDF(filesave,x_test_rec,x_test_OI,x_test_GT,x_test_M1)
                    
                    print(" mseOI = %f"%np.mean((x_test_GT[:,int(dT/2),:,:]-x_test_OI[:,int(dT/2),:,:])**2))
                    print(" mseNN = %f"%np.mean((x_test_GT[:,int(dT/2),:,:]-x_test_rec[:,int(dT/2),:,:])**2))

                    #print(" verif = %f"%np.mean((x_test_GT[:,int(dT/2),:,:] - qHR[90:110,:,:])**2))

                    #print('----- MSE Tr OI: %.6f'%np.mean(stdTr**2 * (x_trainOI-x_train)**2))
                    #print('----- MSE Tt OI: %.6f'%np.mean(stdTr**2 * (x_testOI-x_test)**2))

                