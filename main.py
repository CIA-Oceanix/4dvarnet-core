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
import xarray as xr
import warnings 
import argparse
import numpy as np
import datetime

import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from netCDF4 import Dataset

import solver as NN_4DVar
import torchvision.datasets as datasets
from sklearn import decomposition
from sklearn.feature_extraction import image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


#############################################################################################
##
## Implementation of a NN pytorch framework for the identification and resolution 
## of 4DVar assimilation model applied to Lorenz-96 dynamics (40-dimensional state)
## 
## L96 Data generation
##      flagProcess == 0: Simulation of L96 time series, inc. noise and sampling scheme of observed states
##      flagProcess == 1: Initial interpolation of the observed time series using a linear interpolation
##
## Bulding NN architectures
##      flagProcess == 2: Generation of the NN architecture for the dynamical prior given flagAEType variable
##        flagAEType == 0: CNN-based implementation of L96 ODE (RK4 or Euler schemes)
##        flagAEType == 1: single-scale GENN with DimAE-dimensional latent states
##        flagAEType == 2: two-scale GENN with DimAE-dimensional latent states
##
## Learning schemes
##      flagProcess == 3: Supervised training of the dynamical prior
##      flagProcess == 4: Joint supervised training of the dynamical prior and 4DVar solver 
##      (See torch_4DVarNN_L96 for the different types of 4DVar Solver)
##      flagProcess == 5: Joint training of the dynamical prior and 4DVar solver using test data (noisy/irregularly-sampled)
##      flagProcess == 6: Joint training of the dynamical prior and 4DVar solver using 4 + 5
##
## Interpolation of L96 data
##      flagProcess == 10: Apply a trained model for the evaluation of different performance metrics
##      flagProcess == 11: Iterative application of a trained model from linear interpolation initialization (cf above)
##      flagProcess == 12: Fixed-step gradient-based solver (autograd) of the 4DVAR Assimilation given the trained priod

#if __name__ == '__main__':

parser = argparse.ArgumentParser()

flagProcess    = [0,2,4]#Sequence fo processes to be run

flagRandomSeed = 0
flagSaveModel  = 1
   
batch_size  = 12 #16#4#4#8#12#8#256#8
flagAEType  = 1 # 0: AE, 1-2: GENN, 3: PCA
DimAE       = 10#10#10#50

UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
InterpFlag         = False

flagUseObsData    = True # use along-track data or not
flagUseOI         = True
flagMultiScale    = True
sigNoise          = 0.#0.25#
flagSampOperator  = 0 # 0: SZeros, 1: SAll, 2: SColRow, 3xxx 

flagSWOTData      = True#False
flagNoUseObsCDay  = False

GradType        = 1 # Gradient computation (0: subgradient, 1: AD Gradient+L2 norm, 2: AD Gradient+L1 norm
OptimType       = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent

dT              = 5
W               = 128 #128
dx              = 1
dimGradSolver   = 50
Nbpatches       = 10 #10#25
rnd1            = 0
rnd2            = 100
rateDropout     = 0.25

W = int(W/dx)
dwscale = 1

import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
for kk in range(0,len(flagProcess)):
    
    #################################################################
    ## data generation including noise sampling and missing data gaps
    if flagProcess[kk] == 0:  
        if flagRandomSeed == 0:
            print('........ Random seed set to 100')
            np.random.seed(100)
            torch.manual_seed(100)

        dirSAVE      = '/gpfswork/rech/yrf/ueh53pd/ResSLANATL60/'
        genSuffixObs = ''
        ncfile = Dataset("/gpfswork/rech/yrf/commun/DataNATL60/dataSLA_NATL60GulfStream.nc","r")
        qHR    = ncfile.variables['ssh'][:]
        #qLR    = ncfile.variables['qLR'][:]
        #dt     = ncfile.variables['dt'][:]
        
        ncfile.close()

        if flagSWOTData == True :
            print('.... Use SWOT+4-nadir dataset')
            genFilename  = 'resInterpSWOTSLAwOInoSST_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
            # OI data using a noise-free OSSE (ssh_mod variable)
            ncfile = Dataset("/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/oi/ssh_NATL60_swot_4nadir.nc","r")
            qOI    = ncfile.variables['ssh_mod'][:]
            #qLR    = ncfile.variables['qLR'][:]
            #dt     = ncfile.variables['dt'][:]            
            ncfile.close()

            # OI data using a noise-free OSSE (ssh_mod variable)
            ncfile = Dataset("/gpfswork/rech/yrf/uba22to/DATA/GULFSTREAM/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc","r")
            
            qMask   = ncfile.variables['ssh_mod'][:]
            qMask   = 1.0-qMask.mask.astype(float)
            #qLR    = ncfile.variables['qLR'][:]
            #dt     = ncfile.variables['dt'][:]            
            ncfile.close()

        print('----- MSE OI: %.3f'%np.mean((qOI-qHR)**2))
        print()
        
        ## extraction of patches from the SSH field

         
        NoRndPatches = False            
        if ( Nbpatches == 1 ):
            NoRndPatches = True
            print('... No random seed for the extraction of patches')
            
            qHR   = qHR[:,0:200,0:200]
            qOI   = qOI[:,0:200,0:200]
            qMask = qMask[:,0:200,0:200]
            
        def extract_SpaceTimePatches(q,i1,i2,Wid,dT,rnd1,rnd2,D=1):
            dataTraining  = image.extract_patches_2d(np.moveaxis(q[i1:i2,::D,::D], 0, -1),(Wid,Wid),Nbpatches,random_state=rnd1)
            dataTraining  = np.moveaxis(dataTraining, -1, 1)
            dataTraining  = dataTraining.reshape((Nbpatches,dataTraining.shape[1],Wid*Wid)) 
            
            if NoRndPatches == True :
                for ii in range(0,dataTraining.shape[1]-dT+1):
                    if ii == 0:                
                        temp = dataTraining[:,ii:ii+dT,:].reshape((1,dT,Nbpatches,Wid*Wid))
                    else:
                        temp = np.concatenate((temp,dataTraining[:,ii:ii+dT,:].reshape((1,dT,Nbpatches,Wid*Wid))),axis=0)
        
                dataTraining = np.moveaxis(temp, 1, 2)
            else:
                dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),dataTraining.shape[1]-dT+1,random_state=rnd2)
                #dataTraining  = dataTraining.reshape((dT,W*W,Nbpatches*dataTraining.shape[-1]))
            dataTraining  = dataTraining.reshape((dataTraining.shape[0],dataTraining.shape[1],dT,Wid,Wid)) 
            dataTraining  = np.moveaxis(dataTraining, 0, -1)
            dataTraining  = np.moveaxis(dataTraining, 0, -1)
            dataTraining  = dataTraining.reshape((dT,Wid,Wid,dataTraining.shape[3]*dataTraining.shape[4])) 
            dataTraining  = np.moveaxis(dataTraining, -1, 0)
            return dataTraining     

        
        
        dtOI = 10
        ii1 = 10 
        jj1 = 40
        
        ii2 = 105
        jj2 = 365
        
        ii3 = 60-int(dT/2)
        jj3 = 80+int(dT/2)
        
        indN_Tt = np.arange(ii3+int(dT/2),jj3-+int(dT/2))

        dataTraining1     = extract_SpaceTimePatches(qHR,ii1,jj1,W,dT,rnd1,rnd2,dx)
        dataTrainingMask1 = extract_SpaceTimePatches(qMask,ii1,jj1,W,dT,rnd1,rnd2,dx)
        dataTrainingOI11  = extract_SpaceTimePatches(qOI,ii1-dtOI,jj1-dtOI,W,dT,rnd1,rnd2,dx)
        dataTrainingOI1   = extract_SpaceTimePatches(qOI,ii1,jj1,W,dT,rnd1,rnd2,dx)

        dataTraining2     = extract_SpaceTimePatches(qHR,ii2,jj2,W,dT,rnd1,rnd2,dx)
        dataTrainingMask2 = extract_SpaceTimePatches(qMask,ii2,jj2,W,dT,rnd1,rnd2,dx)
        dataTrainingOI21   = extract_SpaceTimePatches(qOI,ii2-dtOI,jj2-dtOI,W,dT,rnd1,rnd2,dx)
        dataTrainingOI2  = extract_SpaceTimePatches(qOI,ii2,jj2,W,dT,rnd1,rnd2,dx)

        dataTraining      = np.concatenate((dataTraining1,dataTraining2),axis=0)
        dataTrainingMask  = np.concatenate((dataTrainingMask1,dataTrainingMask2),axis=0)
        dataTrainingOI    = np.concatenate((dataTrainingOI1,dataTrainingOI2),axis=0)
        dataTrainingOI1   = np.concatenate((dataTrainingOI11,dataTrainingOI21),axis=0)

        dataTest     = extract_SpaceTimePatches(qHR,ii3,jj3,W,dT,rnd1,rnd2,dx)
        dataTestMask = extract_SpaceTimePatches(qMask,ii3,jj3,W,dT,rnd1,rnd2,dx)
        dataTestOI   = extract_SpaceTimePatches(qOI,ii3,jj3,W,dT,rnd1,rnd2,dx)
        dataTestOI1  = extract_SpaceTimePatches(qOI,ii3-dtOI,jj3-dtOI,W,dT,rnd1,rnd2,dx)

        if 1*0:
            for nn in range(0,20):
                print( np.mean( (dataTestOI[nn,int(dT/2),:,:].squeeze()-qOI[60+nn,0:200,0:200].squeeze())**2) )
                print( np.mean( (dataTest[nn,int(dT/2),:,:].squeeze()-qHR[60+nn,0:200,0:200].squeeze())**2) )
                print( np.mean( (dataTestMask[nn,int(dT/2),:,:].squeeze()-qMask[60+nn,0:200,0:200].squeeze())**2) )


        meanTr     = np.mean(dataTraining)
        x_train    = dataTraining - meanTr
        x_test     = dataTest  - meanTr
        # scale wrt std
        stdTr      = np.sqrt( np.mean( x_train**2 ) )
        x_train    = x_train / stdTr
        x_test     = x_test  / stdTr
        stdTt      = np.sqrt( np.mean( x_test**2 ) )
        
        x_trainOI  = (dataTrainingOI - meanTr) / stdTr
        x_trainOI1 = (dataTrainingOI1 - meanTr) / stdTr
        x_testOI   = (dataTestOI - meanTr) / stdTr
        x_testOI1  = (dataTestOI1 - meanTr) / stdTr
        x_trainMask = dataTrainingMask
        x_testMask  = dataTestMask

        training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainOI1),torch.Tensor(x_trainMask),torch.Tensor(x_train)) # create your datset
        val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testOI1),torch.Tensor(x_testMask),torch.Tensor(x_test)) # create your datset
        
        train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        

        NoRndPatches = True
        Nbpatches = 1
        qHR   = qHR[:,0:200,0:200]
        qOI   = qOI[:,0:200,0:200]
        qMask = qMask[:,0:200,0:200]

        dataTest     = extract_SpaceTimePatches(qHR,ii3,jj3,200,dT,rnd1,rnd2,dx)
        dataTestMask = extract_SpaceTimePatches(qMask,ii3,jj3,200,dT,rnd1,rnd2,dx)
        dataTestOI   = extract_SpaceTimePatches(qOI,ii3,jj3,200,dT,rnd1,rnd2,dx)
        dataTestOI1  = extract_SpaceTimePatches(qOI,ii3-dtOI,jj3-dtOI,200,dT,rnd1,rnd2,dx)

        x_test     = dataTest  - meanTr
        x_test     = x_test  / stdTr
        x_testOI   = (dataTestOI - meanTr) / stdTr
        x_testOI1  = (dataTestOI1 - meanTr) / stdTr
        x_testMask  = dataTestMask
        
        test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testOI1),torch.Tensor(x_testMask),torch.Tensor(x_test)) # create your datset

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

       
    ###############################################################
    elif flagProcess[kk] == 2:        
        print('........ Define AE architecture')
        
        # freeze all ode parameters            
        #shapeData    = np.ones(3).astype(int)
        shapeData  =  np.array(x_train.shape[1:])            
        shapeData_test = np.array(x_test.shape[1:]) 
        if flagMultiScale == True :
            shapeData[0] += shapeData[0]
            shapeData_test[0] += shapeData_test[0]

        dW = 5
        sS = int(4/dx)
        genSuffixModel = '_GENN_%d_%02d_%02d'%(flagAEType,DimAE,dW)
        class Encoder(torch.nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.pool1  = torch.nn.AvgPool2d(sS)
                self.conv1  = torch.nn.Conv2d(shapeData[0],2*DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
                self.conv2  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)

                self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                self.conv3  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)

                self.conv2Tr = torch.nn.ConvTranspose2d(DimAE,shapeData[0],(sS,sS),stride=(sS,sS),bias=False)          

                self.convHR1  = torch.nn.Conv2d(shapeData[0],2*DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
                self.convHR2  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)

                self.convHR21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                self.convHR22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                self.convHR23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                self.convHR3  = torch.nn.Conv2d(2*DimAE,shapeData[0],(1,1),padding=0,bias=False)

            def forward(self, xinp):
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

                x   = torch.add(x,1.,xHR)

                return x

        class Decoder(torch.nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()

            def forward(self, x):
                return torch.mul(1.,x)

        class Phi_r(torch.nn.Module):
            def __init__(self):
                super(Phi_r, self).__init__()
                self.encoder = Encoder()
                self.decoder = Decoder()
        
            def forward(self, x):
                x = self.encoder( x )
                x = self.decoder( x )
                return x

        phi_r           = Phi_r()

        print('AE Model type: '+genSuffixModel)
        print(phi_r)
        print('Number of trainable parameters = %d'%(sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))


        class Model_H(torch.nn.Module):
            def __init__(self):
                super(Model_H, self).__init__()
                self.DimObs = 1
                self.dimObsChannel = np.array([shapeData[0]])
                
            def forward(self, x,y,mask):
                dyout  = (x - y) * mask
                return dyout

        model_H = Model_H()

        class Gradient_img(torch.nn.Module):
            def __init__(self):
                super(Gradient_img, self).__init__()
        
                a = np.array([[1., 0., -1.],[2.,0.,-2.],[1.,0.,-1.]])
                self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
                self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
        
                b = np.array([[1., 2., 1.],[0.,0.,0.],[-1.,-2.,-1.]])
                self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
                self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
        
            def forward(self , im, phase):
                if phase == 'test':
                    shp_data= shapeData_test
                else:
                    shp_data= shapeData

                if im.size(1) == 1:
                    G_x = self.convGx( im )
                    G_y = self.convGy( im )
                    G   = torch.sqrt(torch.pow(0.5*G_x,2)+ torch.pow(0.5*G_y,2))
                else:
                    
                    for kk in range(0,im.size(1)):
                        G_x = self.convGx( im[:,kk,:,:].view(-1,1,shp_data[1],shp_data[2]) )
                        G_y = self.convGy( im[:,kk,:,:].view(-1,1,shp_data[1],shp_data[2]) )
                        
                        G_x = G_x.view(-1,1,shp_data[1]-2,shp_data[2]-2)
                        G_y = G_y.view(-1,1,shp_data[1]-2,shp_data[2]-2)
                        nG  = torch.sqrt(torch.pow(0.5*G_x,2)+ torch.pow(0.5*G_y,2))
                        
                        if kk == 0:
                            G = nG.view(-1,1,shp_data[1]-2,shp_data[2]-2)
                        else:
                            G = torch.cat((G,nG.view(-1,1,shp_data[1]-2,shp_data[2]-2)),dim=1)
                return G
        gradient_img = Gradient_img()

        if flagMultiScale == True :
            class ModelLR(torch.nn.Module):
                def __init__(self):
                    super(ModelLR, self).__init__()
            
                    self.pool  = torch.nn.AvgPool2d((16,16))
            
                def forward(self , im ):
                
                    return self.pool(im)
            model_LR = ModelLR()
            
            class ModelSRLin(torch.nn.Module):
                def __init__(self):
                    super(ModelSRLin, self).__init__()
            
                    self.convTr1 = torch.nn.ConvTranspose2d(int(shapeData[0]/2),int(shapeData[0]/2),(4,4),stride=(4,4),bias=False)          
                    self.convTr2 = torch.nn.ConvTranspose2d(int(shapeData[0]/2),int(shapeData[0]/2),(4,4),stride=(4,4),bias=False)          
            
                def forward(self , im ):
                    out = self.convTr1( im )
                    return self.convTr2( out )

            model_SRLin = ModelSRLin()

        lr_Sampling = 0.
        class Model_Sampling(torch.nn.Module):
            def __init__(self):
                super(Model_Sampling, self).__init__()
                self.DimObs = 1
                self.W      = torch.nn.Parameter(torch.Tensor(np.zeros((1,shapeData[0],shapeData[1],shapeData[2]))))
            def forward(self , y, phase ):           
                if phase == 'test':
                    shp_data= shapeData_test
                    wght     = torch.Tensor(np.zeros((1,shp_data[0],shp_data[1],shp_data[2])))
                    wght = wght.to(device)
                    yout1 = 0. * wght + 0. * y
                    yout2 = 1. * yout1
                else:
                    yout1 = 0. * self.W + 0. * y
                    yout2 = 1. * yout1

                return [yout1,yout2]
        genSuffixModel = genSuffixModel + '_SZeros'
#elif flagProcess[kk] == 4:
lr_Sampling = 1.

print('genSuffixModel:'+genSuffixModel)    

# mean-squared error loss

alpha            = np.array([0.1,0.5])#np.array([1.,0.1])
alpha_L1Sampling = 0.5
thr_L1Sampling   = 1.0 #0.05#0.025 # 0.05 ()
alpha_SR         = 0.5
alpha_LR         = 0.5#1e4

num_epochs     =200
SaveStep       =20

IterUpdate     = [0,25,50,100,500,600,800]#[0,2,4,6,9,15]
NbGradIter     = [5,5,10,10,15,15,20,20,20]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
lrUpdate       = [1e-3,1e-4,1e-4,1e-5,1e-4,1e-5,1e-5,1e-6,1e-7]

NBGradCurrent   = NbGradIter[0]
lrCurrent       = lrUpdate[0]


# optimization setting: freeze or not the AE
lambda_LRAE = 0.5

# training function for dinAE
since = time.time()

alpha_Grad = alpha[0]
#alpha_FP   = 1. - alpha[0]
alpha_AE   = alpha[1]

# Suffix for file naming
genSuffixModelBase = genSuffixModel

genSuffixModel = genSuffixModelBase+genSuffixObs
if lambda_LRAE == 0. :
    genSuffixModel = genSuffixModel+'_NoFTrAE'

if flagUseObsData == True :
    genSuffixModel = genSuffixModel+'_HRObs'
    
if flagUseOI == True :
    genSuffixModel = genSuffixModel+'_OIObs'

if flagMultiScale == True :
    genSuffixModel = genSuffixModel+'_MS'

if flagNoUseObsCDay == True :
    genSuffixModel = genSuffixModel+'_UnSup'
    
#genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))+'_'+str('%02d'%(dimGradSolver))

print('...... Suffix trained models: '+genSuffixModel)

best_loss = 1e10

# loss weghing wrt time
w_            = np.zeros(dT,)
w_[int(dT/2)] = 1.
wLoss         = torch.Tensor(w_)


betaX = 1.0
betagX = 1.0
# compute the mean loss for OI
for phase in ['train']:        
        
    running_loss_GOI     = 0.
    running_loss_OI      = 0.
    num_loss = 0
    
    for targets_OI,targets_OI1,inputs_Mask,targets_GT in train_dataloader:

        # gradient norm field
        g_targets_GT   = gradient_img( targets_GT, phase )


        loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT-targets_OI,wLoss)
        loss_GOI   = NN_4DVar.compute_WeightedLoss(gradient_img( targets_OI, phase )-g_targets_GT,wLoss)

                
        running_loss_GOI         += loss_GOI.item() * targets_GT.size(0)
        running_loss_OI          += loss_OI.item() * targets_GT.size(0)

        num_loss                 += targets_GT.size(0)
  
    epoch_loss_GOI    = running_loss_GOI  / num_loss
    epoch_loss_OI     = running_loss_OI / num_loss
       
    betaX  = 1. / epoch_loss_OI
    betagX = 1. / epoch_loss_GOI
    
    print(".... MSE(Tr) OI %.3f -- MSE(Tr) gOI %.3f "%(epoch_loss_OI,epoch_loss_GOI))
wLoss         = wLoss.to(device)      
comptUpdate = 1
iterInit    = 0#498

def save_NetCDF(saved_path1,x_test_rec):     

    extent=[-65.,-55.,30.,40.]
    indLat     = 200
    indLon     = 200

    lon = np.arange(extent[0],extent[1],1/(20/dwscale))
    lat = np.arange(extent[2],extent[3],1/(20/dwscale))
    indLat     = int(indLat/dwscale)
    indLon     = int(indLon/dwscale)
    lon = lon[:indLon]
    lat = lat[:indLat]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T
                                        
    indN_Tt = np.concatenate([np.arange(60,80)])
    time    = [ datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01",'%Y-%m-%d')\
                          + datetime.timedelta(days=np.float64(i)),"%Y-%m-%d") for i in indN_Tt ]
        
    xrdata = xr.Dataset(\
                data_vars={'longitude': (('lat','lon'),mesh_lon),\
                           'latitude' : (('lat','lon'),mesh_lat),\
                           'Time'     : (('time'),time),\
                           'ssh'  : (('time','lat','lon'),x_test_rec[:,int(dT/2),:,:])},\
                coords={'lon': lon,'lat': lat,'time': indN_Tt})
    xrdata.time.attrs['units']='days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.m_Grad 	    = NN_4DVar.model_GradUpdate2(shapeData,GradType,UsePriodicBoundary,model_H.DimObs,model_H.dimObsChannel,dimGradSolver,rateDropout)     
        self.model          = NN_4DVar.Solver_Grad_4DVarNN(phi_r,model_H, self.m_Grad, shapeData,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary,dimGradSolver,rateDropout)
        self.modelSave      = NN_4DVar.Solver_Grad_4DVarNN(phi_r,model_H, self.m_Grad, shapeData,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary,dimGradSolver,rateDropout) 
        self.IterUpdate     = [0,25,50,100,500,600,800]#[0,2,4,6,9,15]
        self.NbGradIter     = [5,5,10,10,15,15,20,20,20]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.lrUpdate       = [1e-3,1e-4,1e-4,1e-5,1e-4,1e-5,1e-5,1e-6,1e-7]
        self.model_S     = Model_Sampling()
        self.modelSave_S = Model_Sampling()
        self.model_LR = model_LR
        self.gradient_img = gradient_img

    def forward():
        return 1

    def configure_optimizers(self):
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters()},
                        {'params': self.model.phi_r.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent},
                        {'params': self.model.phi_r.decoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                        ], lr=lrCurrent)

        optimizer_Sampling =  optim.Adam(self.model_S.parameters(),lr= lr_Sampling * lrCurrent)

        scheduler1 = MultiStepLR(optimizer, milestones=[0,25,50,100,500,600,800], gamma=0.1)
        scheduler2 = MultiStepLR(optimizer_Sampling, milestones=[0,25,50,100,500,600,800], gamma=0.05)

        return [optimizer,optimizer_Sampling],[scheduler1,scheduler2]

    def training_step(self,train_batch,batch_idx,optimizer_idx):
        if (self.current_epoch in self.IterUpdate) & (self.current_epoch > 0) :
            indx = self.IterUpdate.index(self.current_epoch)
            self.model.NGrad = self.NbGradIter[indx]
        loss,out = self.compute_loss(train_batch,phase='train')
        return loss

    def validation_step(self,val_batch,batch_idx):
        loss,out = self.compute_loss(val_batch,phase='val')
        return loss
    
    def test_step(self,test_batch,batch_idx):
        loss,out = self.compute_loss(test_batch,phase='test')
        return {'preds': out.detach().cpu()}
    
    def test_epoch_end(self, outputs):
    	x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
    	x_test_rec = stdTr * x_test_rec + meanTr
    	path_save = '/gpfswork/rech/yrf/ueh53pd/4DVARR_Lightning/results/test.nc'
    	save_NetCDF(saved_path1= path_save,x_test_rec = x_test_rec)

    
    def compute_loss(self,batch,phase):
        running_loss         = 0.0
        
        num_loss             = 0
    
        targets_OI,targets_OI1,inputs_Mask,targets_GT = batch
            
        # use low-resolution
        if flagMultiScale == True :
            targets_GTLR = self.model_LR(targets_OI)
            
            # sampling mask
            new_masks = self.model_S(torch.cat((targets_OI1,targets_OI1),dim=1),phase)                                

            if flagUseObsData == True :
                inputs_Mask2 = inputs_Mask.repeat(1,2,1,1)
                new_masks[0] = inputs_Mask2 + (1.0 - inputs_Mask2) * new_masks[0]
                new_masks[1] = inputs_Mask2 + (1.0 - inputs_Mask2) * new_masks[1]
                          
            # init
            if flagUseOI == True :
                new_masks[0][:,0:dT,:,:] = 1.0 + 0. * new_masks[0][:,0:dT,:,:]
                new_masks[1][:,0:dT,:,:] = 1.0 + 0. * new_masks[1][:,0:dT,:,:]
           
            idxSampMat     = int(1)
            mask_t         = 1. - torch.nn.functional.threshold( 1.0 - new_masks[idxSampMat] , 0.9 , 0.)
               
            mask_t = mask_t[:,dT:,:,:]

            if flagUseOI == True :
                inputs_init    = torch.cat((targets_OI , mask_t * (targets_GT-targets_OI)),dim=1)
                inputs_missing = torch.cat((targets_OI , mask_t * (targets_GT-targets_OI)),dim=1)                                

        # gradient norm field
        g_targets_GT   = self.gradient_img( targets_GT,phase )

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True): 
        #with torch.set_grad_enabled(phase == 'train'):
            inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
            if self.model.OptimType == 1:
                outputs,grad_new,normgrad = self.model(inputs_init,inputs_missing,new_masks[idxSampMat])
                
            elif self.model.OptimType == 2:
                outputs,hidden_new,cell_new,normgrad = self.model(inputs_init,inputs_missing,new_masks[idxSampMat])

            else:                               
                outputs,normgrad = self.model(inputs_init,inputs_missing,new_masks[idxSampMat])
            
            if (phase == 'val') or (phase == 'test'):
            	outputs = outputs.detach()
            if phase == 'test':
                shp_data= shapeData_test
            else:
                shp_data= shapeData
            
            if flagMultiScale == True :
                outputsSLRHR = outputs
                outputsSLR   = outputs[:,0:dT,:,:].view(-1,dT,shp_data[1],shp_data[2])
                outputs      = outputsSLR + outputs[:,dT:,:,:].view(-1,dT,shp_data[1],shp_data[2])

            # losses
            g_outputs   = self.gradient_img( outputs ,phase )

            loss_All    = NN_4DVar.compute_WeightedLoss((outputs-targets_GT) , wLoss)
            loss_GAll   = NN_4DVar.compute_WeightedLoss(g_outputs-g_targets_GT,wLoss)

            loss_All1    = NN_4DVar.compute_WeightedLoss(outputs-targets_OI,wLoss)
            loss_GAll1   = NN_4DVar.compute_WeightedLoss(g_outputs-self.gradient_img( targets_OI,phase),wLoss)

            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT-targets_OI,wLoss)
            loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img( targets_OI,phase )-g_targets_GT,wLoss)

            mean_GAll   = NN_4DVar.compute_WeightedLoss(g_targets_GT,wLoss)
            
            loss_AE    = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR)**2 ) 
            yGT        = torch.cat((targets_GT,outputsSLR-targets_GT),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT)**2 ) 

            ## L1 vs. L0 cost for the sampling operator
            if flagMultiScale == True :
                loss_Sampling = torch.mean( new_masks[idxSampMat][:,dT:,:,:] )
                loss_Sampling2 = torch.mean(1. - torch.nn.functional.threshold( 1. - new_masks[idxSampMat][:,dT:,:,:] , 0.9 , 0. ))                            
            
            loss_Sampling = torch.nn.functional.relu( loss_Sampling - thr_L1Sampling )

            # training loss
            loss        = alpha_Grad * (betaX * loss_All + betagX * loss_GAll) + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
            loss       += alpha_L1Sampling * loss_Sampling

            if flagMultiScale == True :                                
                loss_SR    = NN_4DVar.compute_WeightedLoss(outputsSLR-targets_OI,wLoss)
                loss_LR    = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs)-targets_GTLR,wLoss)
                loss    += alpha_LR * loss_LR + alpha_SR * loss_SR

            loss_All    = NN_4DVar.compute_WeightedLoss((outputs-targets_GT) , wLoss)
            loss_GAll   = NN_4DVar.compute_WeightedLoss(g_outputs-g_targets_GT,wLoss)

            # statistics
            running_loss             += loss.item() * inputs_missing.size(0)
            num_loss                 += inputs_missing.size(0)
  
        epoch_loss       = running_loss / num_loss
        return loss, outputs 

mod = LitModel()
#checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='/gpfswork/rech/yrf/ueh53pd/4DVARR_Lightning/results', save_top_k = 3)
# training
mod.load_from_checkpoint("/gpfswork/rech/yrf/ueh53pd/4DVARR_Lightning/lightning_logs/version_0/checkpoints/epoch=9-step=2349.ckpt")
trainer = pl.Trainer(gpus=1,max_epochs=10)
#trainer.fit(mod, train_dataloader, val_dataloader)
trainer.test(mod,test_dataloaders=test_dataloader)

