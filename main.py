import os
import glob
from pathlib import Path
import argparse
import numpy as np
import datetime
import copy

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy
from scipy.integrate import solve_ivp


import os
import solver as NN_4DVar

from sklearn.feature_extraction import image




import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.distributed as dist
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt



if __name__ == '__main__':  
    if not os.path.exists("Checkpoints"):
        os.mkdir("Checkpoints")
        os.mkdir("Checkpoints/rec")
        os.mkdir("Checkpoints/rec+AE")
        os.mkdir("Checkpoints/rec+adv")
        os.mkdir("Checkpoints/rec+AE+adv")
    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")
        os.mkdir("lightning_logs/rec")
        os.mkdir("lightning_logs/rec+AE")
        os.mkdir("lightning_logs/rec+adv")
        os.mkdir("lightning_logs/rec+AE+adv")
    if not os.path.exists("Plots"):
        os.mkdir("Plots")
        os.mkdir("Plots/rec")
        os.mkdir("Plots/rec+AE")
        os.mkdir("Plots/rec+adv")
        os.mkdir("Plots/rec+AE+adv")
    
    
    #torch.manual_seed(100)
    np.random.seed(100)
    # NN architectures and optimization parameters
    batch_size      = 8#4#4#8#12#8#256#
    DimAE           = 50#10#10#50
    dW               = 5


    flagProcess = [0,1,2,3,4]
    #flagProcess = [0,1,2,3,5] #test

    flagRandomSeed = 0
    UsePriodicBoundary = True # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    InterpFlag         = False # True => force reconstructed field to observed data after each gradient-based update  
    
    betaDT    = 4.
    alpha          = np.array([1.,0.1,0.])
    alpha4DVar     = np.array([0.01,1.0,.0])

    log_rec = True
    log_AE  = True
    log_adv = False

    if log_rec==False:
        alpha[0] = 0.
    if log_AE== False:
        alpha[1] = 0.
    if log_adv== False:
        alpha[2] = 0.


    alpha_rec  = alpha[0]  
    alpha_AE   = alpha[1]
    alpha_adv  = alpha[2]
    
    LSTM_stoch_init = False #if True initialize LSTM initial state and hidden state N(0, sigma) 
    LSTM_stoch_update = False #if True add stochastic term to LSTM update N(0, sigma) 
    grad_stoch_update = False #if True add stochastic term to grad update N(0, sigma) at solver_step
    
    #apply only if stochastic generator
    stoch_test     = False #Test each input on different stochastic outputs by stochastic model (stochastic gradient/stochastic LSTM state) 
    stoch_outputs  = False #Test each input on the average of different stochastic outputs (outputs + stochastic * output)
    n_stoch        = 10 #number of stoch outputs
    
    LoadModel = False #resume training or start from epoch 0
    
    for kk in range(0,len(flagProcess)):
        
        ###############################################################
        ## data generation including noise sampling and missing data gaps
        if flagProcess[kk] == 0:        
            print('........ Data generation')
            if flagRandomSeed == 0:
                print('........ Random seed set to 100')
                np.random.seed(100)
                
            def AnDA_Lorenz_96(S,t,F,J):
                """ Lorenz-96 dynamical model. """
                x = np.zeros(J);
                x[0] = (S[1]-S[J-2])*S[J-1]-S[0];
                x[1] = (S[2]-S[J-1])*S[0]-S[1];
                x[J-1] = (S[0]-S[J-3])*S[J-2]-S[J-1];
                for j in range(2,J-1):
                    x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j];
                dS = x.T + F;
                return dS
            
            
            class GD:
                model = 'Lorenz_96'
                class parameters:
                    F = 8
                    J = 40
                dt_integration = 0.05 # integration time
                dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
                dt_obs = 4 # number of integration times between consecutive observations (for yo)
                var_obs = np.random.permutation(parameters.J)[0:20] # indices of the observed variables
                nb_loop_train = 10**4 # size of the catalog
                nb_loop_test = 10 # size of the true state and noisy observations
                sigma2_catalog = 0   # variance of the model error to generate the catalog   
                sigma2_obs = 2 # variance of the observation error to generate observations
            
            class time_series:
              values = 0.
              time   = 0.

            # 5 time steps (to be in the attractor space)
            GD = GD()    
            x0 = GD.parameters.F*np.ones(GD.parameters.J);
            x0[np.int(np.around(GD.parameters.J/2))] = x0[np.int(np.around(GD.parameters.J/2))] + 0.01;
            #S = odeint(AnDA_Lorenz_96,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
            S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_96(y,t,GD.parameters.F,GD.parameters.J),t_span=[0.,5+0.000001],y0=x0,first_step=GD.dt_integration,t_eval=np.arange(0,5+0.000001,GD.dt_integration),method='RK45')
            print(S.y.shape)
            x0 = S.y[:,-1]
            
            # generate true state (xt)
            #S = odeint(AnDA_Lorenz_96,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));       
            tt = np.arange(GD.dt_integration,GD.nb_loop_train*GD.dt_integration+0.000001,GD.dt_integration)
            S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_96(y,t,GD.parameters.F,GD.parameters.J),t_span=[GD.dt_integration,GD.nb_loop_train*GD.dt_integration+0.000001],y0=x0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
            S = S.y.transpose()

              
            ####################################################
            ## Generation of training and test dataset
            ## Extraction of time series of dT time steps  
            NbTraining = 2000#2000
            NbTest     = 256#256#500
            val_steps_per_epoch = NbTest // batch_size
            time_step  = 1
            dT         = 201
            sigNoise   = np.sqrt(2)
            rateMissingData = 0.75#0.95
            
            xt = time_series()
            xt.values = S
            xt.time   = tt
            # extract subsequences
            dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:7000:time_step,:],(dT,GD.parameters.J),NbTraining)
            dataTestNoNaN     = image.extract_patches_2d(xt.values[7500::time_step,:],(dT,GD.parameters.J),NbTest)
            
            # create missing data
            flagTypeMissData = 1
            if flagTypeMissData == 0:
                indRand         = np.random.permutation(dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2])
                indRand         = indRand[0:int(rateMissingData*len(indRand))]
                dataTraining    = np.copy(dataTrainingNoNaN).reshape((dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2],1))
                dataTraining[indRand] = float('nan')
                dataTraining    = np.reshape(dataTraining,(dataTrainingNoNaN.shape[0],dataTrainingNoNaN.shape[1],dataTrainingNoNaN.shape[2]))
                
                indRand         = np.random.permutation(dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2])
                indRand         = indRand[0:int(rateMissingData*len(indRand))]
                dataTest        = np.copy(dataTestNoNaN).reshape((dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2],1))
                dataTest[indRand] = float('nan')
                dataTest          = np.reshape(dataTest,(dataTestNoNaN.shape[0],dataTestNoNaN.shape[1],dataTestNoNaN.shape[2]))

                genSuffixObs    = '_ObsRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
            else:
                time_step_obs   = int(1./(1.-rateMissingData))
                dataTraining    = np.zeros((dataTrainingNoNaN.shape))
                dataTraining[:] = float('nan')
                dataTest        = np.zeros((dataTestNoNaN.shape))
                dataTest[:]     = float('nan')
                
                if 1*0:
                    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
                    dataTest[:,::time_step_obs,:]     = dataTestNoNaN[:,::time_step_obs,:]
                    
                    genSuffixObs    = '_ObsSub_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
                else:
                    for nn in range(0,dataTraining.shape[1],time_step_obs):
                        indrand = np.random.permutation(dataTraining.shape[2])[0:int(0.5*dataTraining.shape[2])]
                        dataTraining[:,nn,indrand] = dataTrainingNoNaN[:,nn,indrand]

                    for nn in range(0,dataTraining.shape[1],time_step_obs):
                        indrand = np.random.permutation(dataTraining.shape[2])[0:int(0.5*dataTraining.shape[2])]
                        dataTest[:,nn,indrand] = dataTestNoNaN[:,nn,indrand]

                    genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
                print('... Data type: '+genSuffixObs)
                #for nn in range(0,dataTraining.shape[1],time_step_obs):
                #    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
                
                #dataTest    = np.zeros((dataTestNoNaN.shape))
                #dataTest[:] = float('nan')
                #dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]

                
            # set to NaN patch boundaries
            if 1*0:
                dataTraining[:,0:10,:] =  float('nan')
                dataTest[:,0:10,:]     =  float('nan')
                dataTraining[:,dT-10:dT,:] =  float('nan')
                dataTest[:,dT-10:dT,:]     =  float('nan')
            
            # mask for NaN
            maskTraining = (dataTraining == dataTraining).astype('float')
            maskTest     = ( dataTest    ==  dataTest   ).astype('float')
            
            dataTraining = np.nan_to_num(dataTraining)
            dataTest     = np.nan_to_num(dataTest)
            
            # Permutation to have channel as #1 component
            dataTraining      = np.moveaxis(dataTraining,-1,1)
            maskTraining      = np.moveaxis(maskTraining,-1,1)
            dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
            
            dataTest      = np.moveaxis(dataTest,-1,1)
            maskTest      = np.moveaxis(maskTest,-1,1)
            dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
            
            # set to NaN patch boundaries
            #dataTraining[:,0:5,:] =  dataTrainingNoNaN[:,0:5,:]
            #dataTest[:,0:5,:]     =  dataTestNoNaN[:,0:5,:]
            
            ############################################
            ## raw data
            X_train         = dataTrainingNoNaN
            X_train_missing = dataTraining
            mask_train      = maskTraining
            
            X_test         = dataTestNoNaN
            X_test_missing = dataTest
            mask_test      = maskTest
            
            ############################################
            ## normalized data
            meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 
            
            x_train_missing = X_train_missing - meanTr
            x_test_missing  = X_test_missing - meanTr
            
            # scale wrt std
            stdTr           = np.sqrt( np.mean( X_train_missing**2 ) / np.mean(mask_train) )
            x_train_missing = x_train_missing / stdTr
            x_test_missing  = x_test_missing / stdTr
            
            x_train = (X_train - meanTr) / stdTr
            x_test  = (X_test - meanTr) / stdTr
            
            
            # Generate noisy observsation
            X_train_obs = X_train_missing + sigNoise * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
            X_test_obs  = X_test_missing  + sigNoise * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])
            
            x_train_obs = (X_train_obs - meanTr) / stdTr
            x_test_obs  = (X_test_obs - meanTr) / stdTr

            print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
            print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
            
        ###############################################################
        ## Initial interpolation
        elif flagProcess[kk] == 1:        
            print('........ Initialize interpolated states')

            # Initialization for interpolation
            flagInit = 1
            
            if flagInit == 0: 
              X_train_Init = mask_train * X_train_obs + (1. - mask_train) * (np.zeros(X_train_missing.shape) + meanTr)
              X_test_Init  = mask_test * X_test_obs + (1. - mask_test) * (np.zeros(X_test_missing.shape) + meanTr)
            else:
              X_train_Init = np.zeros(X_train.shape)
              for ii in range(0,X_train.shape[0]):
                # Initial linear interpolation for each component
                XInit = np.zeros((X_train.shape[1],X_train.shape[2]))
            
                for kk in range(0,mask_train.shape[1]):
                  indt  = np.where( mask_train[ii,kk,:] == 1.0 )[0]
                  indt_ = np.where( mask_train[ii,kk,:] == 0.0 )[0]
            
                  if len(indt) > 1:
                    indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                    indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                    fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,kk,indt])
                    XInit[kk,indt]  = X_train_obs[ii,kk,indt]
                    XInit[kk,indt_] = fkk(indt_)
                  else:
                    XInit = XInit + meanTr
            
                X_train_Init[ii,:,:] = XInit
            
              X_test_Init = np.zeros(X_test.shape)
              for ii in range(0,X_test.shape[0]):
                # Initial linear interpolation for each component
                XInit = np.zeros((X_test.shape[1],X_test.shape[2]))
            
                for kk in range(0,X_test.shape[1]):
                  indt  = np.where( mask_test[ii,kk,:] == 1.0 )[0]
                  indt_ = np.where( mask_test[ii,kk,:] == 0.0 )[0]
            
                  if len(indt) > 1:
                    indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                    indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                    fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,kk,indt])
                    XInit[kk,indt]  = X_test_obs[ii,kk,indt]
                    XInit[kk,indt_] = fkk(indt_)
                  else:
                    XInit = XInit + meanTr
            
                X_test_Init[ii,:,:] = XInit
                  #plt.figure()
                  #plt.figure()
                  #plt.plot(YObs[0:200,1],'r.')
                  #plt.plot(XGT[0:200,1],'b-')
                  #plt.plot(XInit[0:200,1],'k-')
                        
            x_train_Init = ( X_train_Init - meanTr ) / stdTr
            x_test_Init = ( X_test_Init - meanTr ) / stdTr


            # reshape to dT-1 for time dimension
            dT = dT-1
            X_train_obs        = X_train_obs[:,:,0:dT]
            X_train            = X_train[:,:,0:dT]
            X_train_missing    = X_train_missing[:,:,0:dT]
            mask_train         = mask_train[:,:,0:dT]
            
            x_train_obs        = x_train_obs[:,:,0:dT]
            x_train            = x_train[:,:,0:dT]
            x_train_Init       = x_train_Init[:,:,0:dT]
            X_train_Init       = X_train_Init[:,:,0:dT]

            X_test_obs        = X_test_obs[:,:,0:dT]
            X_test            = X_test[:,:,0:dT]
            X_test_missing    = X_test_missing[:,:,0:dT]
            mask_test         = mask_test[:,:,0:dT]

            x_test_obs        = x_test_obs[:,:,0:dT]
            x_test            = x_test[:,:,0:dT]
            x_test_Init       = x_test_Init[:,:,0:dT]
            X_test_Init       = X_test_Init[:,:,0:dT]

            print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
            print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
            train_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            val_dataset          = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset

        elif flagProcess[kk]==2:
            shapeData       = x_train.shape


            class Encoder(torch.nn.Module):
                def __init__(self):
                    super(Encoder, self).__init__()
                    self.pool1  = torch.nn.AvgPool2d((1,4))
                    self.conv11 = NN_4DVar.ConstrainedConv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
                    self.conv12 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv3  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                            
                    self.convTr = torch.nn.ConvTranspose2d(DimAE,DimAE,(1,4),stride=(1,4),bias=False)          
                    #self.conv5 = torch.nn.Conv1d(8*shapeData[0]*DimAE,16*shapeData[0]*DimAE,3,padding=1,bias=False)
                    #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)
                    self.conv11_1 = NN_4DVar.ConstrainedConv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
                    self.conv12_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv21_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv22_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv23_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
                    self.conv3_1  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                        
                    self.convF    = torch.nn.Conv2d(DimAE,1,1,padding=0,bias=False)
                def forward(self, xin):
                    x_1 = self.pool1(xin)
                    x_1 = torch.cat((x_1[:,:,x_1.size(2)-dW:,:],x_1,x_1[:,:,0:dW,:]),dim=2)
                    #x_1 = x_1.view(-1,1,xin.size(1)+2*dW,xin.size(2))
                    x   = self.conv11( x_1 )
                    x   = self.conv12( F.relu(x) )
                    x   = x[:,:,dW:xin.size(2)+dW,:]
                    x   = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
                    x   = self.conv3( x )
                    x   = self.convTr( x )
                                  
                    x_2 = torch.cat((xin[:,:,xin.size(2)-dW:,:],xin,xin[:,:,0:dW,:]),dim=2)
                    dx  = self.conv11_1( x_2 )
                    dx  = self.conv12_1( F.relu(dx) )
                    dx   = dx[:,:,dW:xin.size(2)+dW,:]
                    dx   = torch.cat((self.conv21_1(dx), self.conv22_1(dx) * self.conv23_1(dx)),dim=1)
                    dx   = self.conv3_1( dx )
                                  
                    x    = self.convF( x + dx )
                    #x = self.conv4( F.relu(x) )
                    x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
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
                    x = self.encoder(x)
                    x = self.decoder(x)
                    return x
            phi_r = Phi_r()
            
            
            class Discriminator(torch.nn.Module):
                def __init__(self, input_channels=1, hidden_dim=8):
                    super (Discriminator, self).__init__()
                    self.disc = torch.nn.Sequential(
                    self.disc_block(input_channels, hidden_dim),
                    self.disc_block(hidden_dim, hidden_dim*2),
                    #disc_block(hidden_dim*2, hidden_dim*4)
                    self.disc_block(hidden_dim*2, 1, final_layer=True))
                    self.fc1 = torch.nn.Sequential(torch.nn.Linear(1*69,1),
                                                  torch.nn.Sigmoid())
                    #self.fc2 = torch.nn.Linear(32, 1)
                def disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
                    if not final_layer:
                        return torch.nn.Sequential(
                                        torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                                        torch.nn.BatchNorm2d(output_channels),
                                        torch.nn.LeakyReLU(0.2, inplace=True)
                                    )
                    else:
                        return torch.nn.Sequential(
                                        torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride)
                                    )

                def forward(self, x):
                    x= self.disc(x)
                    x= torch.flatten(x, 1)
                    #une couche linear suffit + activation
                    x= self.fc1(x)
                    #x= self.fc2(x)
                    return(x)

            model_disc = Discriminator()

          
            print(phi_r)
            print('Number of trainable parameters = %d' % (sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))

            print(model_disc)
            print('Number of trainable parameters = %d' % (sum(p.numel() for p in model_disc.parameters() if p.requires_grad)))
            
            
            class Model_H(torch.nn.Module):
                def __init__(self):
                    super(Model_H, self).__init__()
                    self.DimObs = 1
                    self.dimObsChannel = np.array([shapeData[0]])

                def forward(self, x, y, mask):
                    dyout = (x - y) * mask
                    return dyout

            model_H = Model_H()
            
            class Gradient_img(torch.nn.Module):
                def __init__(self):
                    super(Gradient_img, self).__init__()
            
                    a = np.array([[0., 0., 0.],[1.,0.,-1.],[0.,0.,-0.]])
                    self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
                    self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
            
                    b = np.array([[0., 1., 0.],[0.,0.,0.],[0.,-1.,-0.]])
                    self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
                    self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
            
                def forward(self , im ):
                    if im.size(1) == 1:
                        Gx = self.convGx( im )
                        Gy = self.convGy( im )
                        G   = torch.sqrt(torch.pow(0.5*Gx,2)+ torch.pow(0.5*Gy,2))
                    else:
                        #G = torch.Tensor(np.zeros((im.size(0),shapeData[0],shapeData[1]-2,shapeData[2]-2)))
                        #G = G.to(device)
            
                        for kk in range(0,im.size(1)):
                            nG_x = self.convGx( im[:,kk,:,:].view(-1,1,shapeData[1],shapeData[2]) )
                            nG_y = self.convGy( im[:,kk,:,:].view(-1,1,shapeData[1],shapeData[2]) )
                            
                            nG_x = nG_x.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                            nG_y = nG_y.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                            nG  = torch.sqrt(torch.pow(0.5*nG_x,2)+ torch.pow(0.5*nG_y,2))
                            
                            if kk == 0:
                                Gx = nG_x.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                                Gy = nG_y.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                                G  = nG.view(-1,1,shapeData[1]-2,shapeData[2]-2)
                            else:
                                G = torch.cat((G,nG.view(-1,1,shapeData[1]-2,shapeData[2]-2)),dim=1)
                                Gx = torch.cat((Gx,nG_x.view(-1,1,shapeData[1]-2,shapeData[2]-2)),dim=1)
                                Gy = torch.cat((Gy,nG_y.view(-1,1,shapeData[1]-2,shapeData[2]-2)),dim=1)
                    return Gx,Gy,G
            
            shapeData       = [1, shapeData[1], shapeData[2]]
            

        elif flagProcess[kk]==3:
            gen_losses = []
            d= {"rec": log_rec, "AE":log_AE, "adv": log_adv}
            for k in d.keys():
                if d[k]==True:
                    gen_losses.append(k)
            loss_comp="+".join(gen_losses) 

            pathCheckPoint  = "Checkpoints/" + loss_comp + "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv)
            pathlogger = "lightning_logs/" + loss_comp
            class HParam:
                def __init__(self):
                    self.iter_update     = []#[0,2,4,6,9,15]
                    self.NbProjection    = []#[0,0,0,0,0,0]#[5,5,5,5,5]##
                    self.nb_grad_update  = []#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
                    self.lr_update       = []
                    self.lr_updated      = []
                    self.dropout         = 0.1
                    self.indx            = 0
                    self.n_grad          = 1
                    self.dim_grad_solver = 1
                    self.batch_size      = batch_size
                    self.lrCurrent       = 0.
                    self.lrCurrentd      = 0.
                    self.shapeData       = []
                    self.best_loss       = 0.
                    self.optimal_loss    = 0.
                    self.val_step        = 0
                    self.tst_step        = 0
                    self.best_epoch      = 0
                    self.log_adv         = log_adv
                    self.log_AE          = log_AE
                    self.log_rec         = log_rec
                    self.alpha_adv       = alpha_adv
                    self.alpha_AE        = alpha_AE
                    self.alpha_rec       = alpha_rec
                    self.alpha4DVar      = alpha4DVar
                    self.UsePriodicBoundary= UsePriodicBoundary
                    self.shapeData       = shapeData
                    self.stoch_test      = stoch_test
                    self.best_generator_wts = {}
                    self.stoch_test      = stoch_test 
                    self.stoch_outputs   = stoch_outputs
                    self.n_stoch         = n_stoch
                    self.LSTM_stoch_init = LSTM_stoch_init
                    self.LSTM_stoch_update = LSTM_stoch_update
                    self.grad_stoch_update = grad_stoch_update


            class LiteModel(pl.LightningModule):
                def __init__(self,conf=HParam(),*args, **kwargs):
                    super(LiteModel, self).__init__()
                    self.hparams.iter_update    = [0,4,50,100,150,200,250]#[0,4,50,100,200,250,400]
                    self.hparams.NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
                    self.hparams.nb_grad_update = [1,5,10,15,20,20,20]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
                    self.hparams.lr_update      = [1e-3,1e-4,1e-4,1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]
                    self.hparams.lr_updated     = [1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,1e-5,1e-6,1e-7]
                    self.hparams.dropout        = 0.1
                    self.hparams.dim_grad_solver= 200
                    self.hparams.indx           = 0
                    self.hparams.batch_size     = batch_size
                    self.hparams.n_grad         = self.hparams.nb_grad_update[self.hparams.indx]
                    self.hparams.lrCurrent      = self.hparams.lr_update[self.hparams.indx]
                    self.hparams.lrCurrentd     = self.hparams.lr_updated[self.hparams.indx]
                    self.hparams.shapeData      = shapeData
                    self.hparams.best_loss      = 10000.
                    self.hparams.optimal_loss   = 10000.      
                    self.hparams.val_step       = 0
                    self.hparams.tst_step       = 0
                    self.hparams.best_epoch     = 0
                    self.hparams.log_adv        = log_adv
                    self.hparams.log_AE         = log_AE
                    self.hparams.log_rec        = log_rec
                    self.hparams.alpha_adv      = alpha_adv
                    self.hparams.alpha_AE       = alpha_AE
                    self.hparams.alpha_rec      = alpha_rec
                    self.hparams.alpha4DVar     = alpha4DVar
                    self.hparams.UsePriodicBoundary= UsePriodicBoundary
                    self.hparams.shapeData      = shapeData
                    self.hparams.stoch_test     = stoch_test
                    self.hparams.stoch_outputs  = stoch_outputs
                    self.hparams.n_stoch        = n_stoch
                    self.hparams.LSTM_stoch_init= LSTM_stoch_init
                    self.hparams.LSTM_stoch_update = LSTM_stoch_update
                    self.hparams.grad_stoch_update = grad_stoch_update
                    self.test_outputs           = torch.zeros(( NbTest, 1, 40, dT))
                    self.test_inits             = torch.zeros(( NbTest, 1, 40, dT))
                    self.test_targets           = torch.zeros(( NbTest, 1, 40, dT))
                    self.test_outputs_stoch     = torch.zeros((self.hparams.n_stoch, NbTest, 1, 40, dT))
                    self.generator              = self.init_generator()
                    self.discriminator          = self.init_discriminator()
                    self.gradient_img           = self.init_gradient_img()
                    self.optimal_generator      = self.init_generator()
                    self.best_generator_wts     = {}
                    self.save_hyperparameters()                 

                def train_dataloader(self):
                    # Data loader
                    train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
                    return train_loader

                def val_dataloader(self):
                    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
                    return val_loader
                
                def test_dataloader(self):
                    test_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
                    return test_loader

                def init_generator(self):
                    generator = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                                Model_H(), 
                                                           NN_4DVar.model_GradUpdateLSTM(self.hparams.shapeData, self.hparams.UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, self.hparams.LSTM_stoch_init, self.hparams.LSTM_stoch_update), None, None, self.hparams.shapeData, self.hparams.n_grad, self.hparams.grad_stoch_update)
                    return generator

                
                def init_gradient_img(self):
                    gradient_img = Gradient_img()
                    return gradient_img
                    
                def init_discriminator(self):
                    discriminator = Discriminator()
                    return discriminator

                def configure_optimizers(self):
                    optimizer = optim.Adam([{'params': self.generator.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.generator.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.generator.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0.)
                    if self.hparams.log_adv==True:
                        disc_optimizer = optim.Adam([{'params': self.discriminator.parameters(), 'lr': self.hparams.lr_update[0]}], lr=0.)
                        return([disc_optimizer, optimizer])
                    return (optimizer)
                            
                def adversarial_loss(self, y_hat, y):
                    return F.binary_cross_entropy(y_hat, y, reduction='sum')
                                                   
                def forward(self, x):
                    outputs = self.generator.phi_r(x)
                    return outputs

                def on_epoch_start(self):
                    self.hparams.val_step = 0
                    if ((self.current_epoch> 0) & ( np.mod(self.current_epoch,10)==0)):
                        print("Optimal model epoch %d - %d: epoch = %d   loss = %.4f"%(self.current_epoch-10, self.current_epoch-1, self.hparams.best_epoch, self.hparams.best_loss))
                        self.hparams.best_epoch = self.current_epoch
                        self.hparams.best_loss  = 10000.
  
                    
                    self.generator.n_grad   = self.hparams.n_grad
                    print("Epoch %d"%(self.current_epoch))
                    if self.hparams.log_adv:              
                        [disc_optimizer, opt] = self.optimizers()
                    else:
                        opt = self.optimizers()
                    if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                        self.hparams.indx  = self.hparams.iter_update.index(self.current_epoch)
                        print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparams.nb_grad_update[self.hparams.indx],self.hparams.lr_update[self.hparams.indx]))
                                 
                        self.hparams.n_grad = self.hparams.nb_grad_update[self.hparams.indx]
                        self.generator.n_grad  = self.hparams.n_grad

                        self.generator.load_state_dict(self.optimal_generator.state_dict())
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        self.generator.to(device) 
                             
                        mm = 0
                        self.hparams.lrCurrent  = self.hparams.lr_update[self.hparams.indx]
                        self.hparams.lrCurrentd = self.hparams.lr_updated[self.hparams.indx]
                        lr = np.array([self.hparams.lrCurrent,self.hparams.lrCurrent,0.5*self.hparams.lrCurrent,0.]) 
                        lrd = np.array([self.hparams.lrCurrentd,0.])       
                        
                        if self.hparams.log_adv: 
                            dd=0 
                            for pg in self.optimizers()[1].param_groups:
                                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                                mm += 1
                            for pg in self.optimizers()[0].param_groups:
                                pg['lr'] = lrd[dd]# * self.hparams.learning_rate
                        else:
                            for pg in self.optimizers().param_groups:
                                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                                mm += 1  
               
                def compute_metrics(self,inputs_init, outputs, targets_GT, masks, inputs_missing, idx, phase):
                    #calculate metrics (discriminator or/and generator metrics)
                    output = {}
                    loss= 0.
                    if idx==0:
                        preds = self.discriminator (outputs)
                        real_preds = self.discriminator (targets_GT)
                        d_loss = self.adversarial_loss(preds, torch.zeros_like(preds))
                        d_loss+= self.adversarial_loss(real_preds, torch.ones_like(real_preds))
                        d_loss = torch.mul(d_loss, 1/(2*preds.size(0)))
                        self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                      {phase+'_d_loss': d_loss}, 
                                                                  global_step=self.global_step)
                            
                        output = {'loss': d_loss}
                        return (output) 
                    if  idx==1:
                         rec_loss= torch.mean((outputs - targets_GT)**2 )
                            
                         r_loss  = torch.sum((outputs - targets_GT)**2 * masks )
                         r_loss  = torch.mul(1.0 / torch.sum(masks),r_loss)
                            
                         I_loss  = torch.sum((outputs - targets_GT)**2 * (1. - masks) )
                         I_loss  = torch.mul(1.0 / torch.sum(1. - masks),I_loss)  
                         
                         obs_loss    = torch.sum( (outputs - inputs_missing)**2 * masks )  / torch.sum( masks )
                         
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                      {phase+'_obs_loss': obs_loss},
                                                                      global_step=self.global_step)
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                      {phase+'_rec_loss': rec_loss},
                                                                      global_step=self.global_step)
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                      {phase+'_r_loss': r_loss},
                                                                      global_step=self.global_step)
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                      {phase+'_I_loss': I_loss},
                                                                      global_step=self.global_step)
                         output['rec_loss']= rec_loss
                         output['r_loss']= r_loss
                         output['I_loss']= I_loss   
                         output['obs_loss']= obs_loss                 
                         if self.hparams.log_rec==True:
                            loss+= self.hparams.alpha_rec * rec_loss

                         AE_loss=  torch.mean((self.generator.phi_r(outputs) - outputs)**2 )
                         AE_GT_loss=  torch.mean((self.generator.phi_r(targets_GT) - targets_GT)**2 )  
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                  {phase+'_AE_loss': AE_loss,
                                                                   phase+'_AE_GT_loss':AE_GT_loss},
                                                                  global_step=self.global_step)  
                         output['AE_loss']= AE_loss
                         output['AE_GT_loss']= AE_GT_loss
                         if self.hparams.log_AE==True:
                             loss+= self.hparams.alpha_AE * (0.5*AE_loss + 0.5*AE_GT_loss)


                            
                         if ((self.hparams.log_AE==True) & (self.hparams.log_rec==True)):
                             score = self.hparams.alpha4DVar[0] * obs_loss + self.hparams.alpha4DVar[1] * AE_loss 
                             self.logger.experiment.add_scalars(phase+' metrics', 
                                                                  {phase+'_4DVar_score': score},
                                                                  global_step=self.global_step)
                             output["4DVar_score"] = score
                        
                         if self.hparams.log_adv==True:
                             preds = self.discriminator (outputs)
                             adv_loss= self.adversarial_loss(preds, torch.ones_like(preds))
                             adv_loss = torch.mul(adv_loss, 1/preds.size(0))
                             loss+= self.hparams.alpha_adv * adv_loss
                            
                             self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                    {phase+'_generator_adv_loss': adv_loss},
                                                                  global_step=self.global_step)
                             output['generator_adv_loss']= adv_loss
                            
                             preds = (preds > 0.5).float()
                             accur = (preds == torch.zeros_like(preds) ).float().sum() / preds.size(0) 
                             output["accuracy"] = accur                                   
                             self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                    {phase+'_accuracy': accur}, 
                                                                  global_step=self.global_step)
                             output["accuracy"] = accur
                            
                         output["loss"]= loss
                        
                         #loss grad w.r.t time
                         g_targets_GT   = self.gradient_img( targets_GT )
                         g_outputs      = self.gradient_img( outputs )
                         DT_loss       = torch.mean( (torch.abs(  g_outputs[0] - g_targets_GT[0]))**betaDT  )
                         output["DT_loss"]= DT_loss
                                                   
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                  {phase+'_DT_loss': DT_loss},
                                                                  global_step=self.global_step)
                                                                  
                         self.logger.experiment.add_scalars(phase + ' metrics', 
                                                                  {phase+'_g_loss': loss},
                                                                  global_step=self.global_step)
     
                         return (output)
                    
                
                def summarize_metrics(self,step_outputs, idx , phase, print_metrics = True):
                    #calculate mean of metrics over all batches
                    return_dict = {}
                    if idx==0:
                        avg_d_loss = torch.stack([x['loss'] for x in step_outputs]).mean()
                        self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_d_loss': avg_d_loss}, 
                                                                  global_step=self.current_epoch)
                        if print_metrics :
                            print("Epoch " +  phase +" discriminator adversarial loss = %f"%(avg_d_loss))
                        return_dict["avg_d_loss"] = avg_d_loss
                    if idx==1:
                        avg_g_loss = 0.
                        
                        avg_rec_loss = torch.stack([x['rec_loss'] for x in step_outputs]).mean()* stdTr**2
                        self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_rec_loss': avg_rec_loss}, 
                                                                  global_step=self.current_epoch)
                        return_dict["avg_rec_loss"] = avg_rec_loss
                        if print_metrics :
                            print("Epoch " +  phase +" generator reconstruction loss = %f"%(avg_rec_loss))
                        avg_r_loss = torch.stack([x['r_loss'] for x in step_outputs]).mean()* stdTr**2
                        return_dict["avg_r_loss"] = avg_r_loss
                        self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_r_loss': avg_r_loss}, 
                                                                  global_step=self.current_epoch)
                        if print_metrics :
                            print("Epoch " +  phase +" generator r_loss = %f"%(avg_r_loss))
                            
                        avg_I_loss = torch.stack([x['I_loss'] for x in step_outputs]).mean()* stdTr**2
                        return_dict["avg_I_loss"] = avg_I_loss
                        self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_I_loss': avg_I_loss}, 
                                                                  global_step=self.current_epoch)
                        if print_metrics :
                            print("Epoch " +  phase +" generator I_loss = %f"%(avg_I_loss))
                        if self.hparams.log_rec==True:
                            avg_g_loss+=self.hparams.alpha_rec*avg_rec_loss
                            
                            
                        
                        avg_obs_loss = torch.stack([x['obs_loss'] for x in step_outputs]).mean()* stdTr**2
                        return_dict["avg_obs_loss"] = avg_obs_loss
                        self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                      {'Epoch_'+ phase +'_obs_loss': avg_obs_loss}, 
                                                                      global_step=self.current_epoch)
                        
                        if print_metrics :
                            print("Epoch " +phase+ " observation loss = %f"%(avg_obs_loss))
                        
                        avg_AE_loss  = torch.stack([x['AE_loss'] for x in step_outputs]).mean()* stdTr**2
                        avg_AE_GT_loss  = torch.stack([x['AE_GT_loss'] for x in step_outputs]).mean()* stdTr**2
                        return_dict["avg_AE_loss"] = avg_AE_loss
                        return_dict["avg_AE_GT_loss"] = avg_AE_GT_loss
                        self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_AE_loss': avg_AE_loss,
                                                                   'Epoch_'+phase+'_AE_GT_loss': avg_AE_GT_loss}, 
                                                                  global_step=self.current_epoch)
                        if print_metrics :
                            print("Epoch " +  phase +" generator AE loss = %f"%(avg_AE_loss))
                            print("Epoch " +  phase +" AE_GT loss = %f"%(avg_AE_GT_loss))  
                        if self.hparams.log_AE==True:
                            avg_g_loss += 0.5*self.hparams.alpha_AE*avg_AE_loss
                            avg_g_loss += 0.5*self.hparams.alpha_AE*avg_AE_GT_loss
                            
                            
                        if ((self.hparams.log_AE==True) & (self.hparams.log_rec==True)):
                            avg_score = self.hparams.alpha4DVar[0] * avg_obs_loss + self.hparams.alpha4DVar[1] * avg_AE_loss
                            return_dict["avg_score"] = avg_score
                            self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_4DVar_score': avg_score}, 
                                                                      global_step=self.current_epoch)
                            if print_metrics :
                                print("Epoch "+ phase +" 4DVar score = %f"%(avg_score))                 
                        
                        if self.hparams.log_adv==True:
                            avg_adv_loss  = torch.stack([x['generator_adv_loss'] for x in step_outputs]).mean()
                            return_dict["avg_adv_loss"] = avg_adv_loss
                            self.logger.experiment.add_scalars('Epoch '+ phase +' metrics', 
                                                                  {'Epoch_'+ phase +'_generator_adv_loss': avg_adv_loss},
                                                                  global_step=self.current_epoch)
                            if print_metrics :
                                print("Epoch " +  phase +" generator adversarial loss = %f"%(avg_adv_loss))
                            avg_g_loss += self.hparams.alpha_adv * avg_adv_loss
                            
                            avg_accur  = torch.stack([x['accuracy'] for x in step_outputs]).mean()
                            return_dict["avg_accur"] = avg_accur
                            self.logger.experiment.add_scalars('Epoch '+ phase + ' metrics', 
                                                                    {'Epoch_'+  phase+'_accuracy': avg_accur}, 
                                                                  global_step=self.current_epoch)
                            if print_metrics :
                                print("Epoch " +  phase +" discriminator accuracy = %f"%(avg_accur))
                                                                                                  
                        avg_DT_loss = torch.stack([x['DT_loss'] for x in step_outputs]).mean()* stdTr**2
                        return_dict["avg_DT_loss"] = avg_DT_loss
                        self.logger.experiment.add_scalars('Epoch ' +phase + ' metrics', 
                                                                  {'Epoch_'+ phase +'_DT_loss': avg_DT_loss}, 
                                                                  global_step=self.current_epoch)
                        if print_metrics :
                            print("Epoch " +phase+ " DT loss = %f"%(avg_DT_loss))
                        
                        self.logger.experiment.add_scalars('Epoch ' +phase + ' metrics', 
                                                                  {'Epoch_'+ phase +'_g_loss': avg_g_loss}, 
                                                                  global_step=self.current_epoch)
                        if print_metrics :
                            print("Epoch " +phase+ " generator loss = %f"%(avg_g_loss))
                        return_dict["avg_g_loss"] = avg_g_loss
                        if (phase == "Validation"):
                            self.log('avg_val_g_loss', avg_g_loss, on_epoch=True)
                            if avg_g_loss.item() < self.hparams.best_loss:
                                self.hparams.best_loss = avg_g_loss.item()
                                self.hparams.best_epoch= self.current_epoch
                                self.best_generator_wts = copy.deepcopy(self.generator.state_dict())
   
                            if avg_g_loss.item() < self.hparams.optimal_loss:
                                self.hparams.optimal_loss = avg_g_loss.item()
                                self.hparams.optimal_epoch= self.current_epoch
                                self.optimal_generator.load_state_dict(self.generator.state_dict())
                        print("\n")
                    return(return_dict)
                
                
                
                
                
                def training_step(self, batch, batch_idx, optimizer_idx=0):
                    inputs_init, inputs_missing, masks, targets_GT = batch
                    if ((self.global_step == 0) & (self.current_epoch ==0)):
                        torch.save(inputs_init, plotCheckPoint + "/inputs_init.pt")
                        torch.save(targets_GT, plotCheckPoint + "/targets_GT.pt")
                    #reshape tensors
                    inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                    inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                            
                    with torch.set_grad_enabled(True):
                        inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
                        outputs,hidden_new,cell_new,normgrad = self.generator(inputs_init,inputs_missing,masks)
                    hidden_new,cell_new,normgrad = hidden_new.detach(),cell_new.detach(),normgrad.detach()
                    idx = optimizer_idx
                    if self.hparams.log_adv==False:
                         idx = idx + 1
                    output = self.compute_metrics(inputs_init, outputs, targets_GT, masks, inputs_missing, idx, 'Train')     
                    return(output)
 
                    

                def training_epoch_end(self, training_step_outputs):
                    if self.hparams.log_adv:
                        self.summarize_metrics(training_step_outputs[0], 0, 'Train')                 
                        self.summarize_metrics(training_step_outputs[1], 1, 'Train')
                    else:
                        self.summarize_metrics(training_step_outputs, 1, 'Train') 

                    
                def validation_step(self, batch, batch_idx):
                    inputs_init, inputs_missing, masks, targets_GT = batch
                    #reshape tensors
                    inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                    inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                                
                    with torch.set_grad_enabled(True):
                        inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
                        outputs,hidden_new,cell_new,normgrad = self.generator(inputs_init,inputs_missing,masks)
                    outputs, hidden_new,cell_new,normgrad = outputs.detach(), hidden_new.detach(),cell_new.detach(),normgrad.detach()
                    output=list()
                    if self.hparams.log_adv:
                        output.append( self.compute_metrics(inputs_init, outputs, targets_GT, masks, inputs_missing,0, 'Validation'))
                    output.append(self.compute_metrics(inputs_init, outputs, targets_GT, masks, inputs_missing,1, 'Validation') )                 
                    
                        
                    self.hparams.val_step = self.hparams.val_step + 1
                    
                    if ((self.hparams.val_step == val_steps_per_epoch) & ((self.current_epoch+1) % 20 == 0) & (self.current_epoch >1)):
                        torch.save(outputs, plotCheckPoint + "/plots_epoch%d.pt"%(self.current_epoch))
                    return (output)

                def validation_epoch_end(self, validation_step_outputs):
                    if self.hparams.log_adv:
                        disc_outputs = [x[0] for x in validation_step_outputs]
                        gen_outputs  = [x[1] for x in validation_step_outputs]
                        
                        self.summarize_metrics(disc_outputs,0, 'Validation')                 
                        self.summarize_metrics(gen_outputs,1, 'Validation')
                        
                    else:
                        gen_outputs  = [x[0] for x in validation_step_outputs]
                        self.summarize_metrics(gen_outputs,1, 'Validation')

                    print("\n\n\n")
                        
                    if ((np.mod(self.current_epoch+1, 25) == 0) & (self.current_epoch>0)):
                        torch.save(self.best_generator_wts, pathCheckPoint + "/best_generator_epoch%d loss = %.4f.mod"%(self.hparams.best_epoch, self.hparams.best_loss))
                def on_test_start(self):
                    self.hparams.tst_step = 0
                     
                def test_step(self, batch, batch_idx):
                    inputs_init, inputs_missing, masks, targets_GT = batch
                    #reshape tensors
                    inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                    inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    
                    #calculcate metrics n_stoch times (n_stoch  outputs)
                    if self.hparams.stoch_test==True:
                        outputs = [None]*self.hparams.n_stoch  
                        if self.hparams.log_adv :
                            output = [[None,None]]*self.hparams.n_stoch 
                        else:
                            output = [[None]]*self.hparams.n_stoch 
                        for i in range(self.hparams.n_stoch ):
                            with torch.set_grad_enabled(True):
                                inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
                                outputs[i],hidden_new,cell_new,normgrad = self.generator(inputs_init,inputs_missing,masks)
                            outputs[i], hidden_new,cell_new,normgrad = outputs[i].detach(),  hidden_new.detach(),cell_new.detach(),normgrad.detach()            
                            if self.hparams.log_adv:
                                output[i][0] = self.compute_metrics(inputs_init, outputs[i], targets_GT, masks, inputs_missing,0, 'Test')
                                output[i][1] = self.compute_metrics(inputs_init, outputs[i], targets_GT, masks, inputs_missing,1, 'Test') 
                            else:
                                output[i][0] = self.compute_metrics(inputs_init, outputs[i], targets_GT, masks, inputs_missing,1, 'Test') 
                              
                            self.test_outputs_stoch[i, self.hparams.tst_step*batch_size : (self.hparams.tst_step +1)*batch_size, :, :, :] = outputs[i]
                        
                        self.test_inits[self.hparams.tst_step*batch_size : (self.hparams.tst_step +1)*batch_size, :, :, :] = inputs_init
                        self.test_targets[self.hparams.tst_step*batch_size : (self.hparams.tst_step +1)*batch_size, :, :, :] = targets_GT
                        self.hparams.tst_step = self.hparams.tst_step + 1
                                         
                        if (self.hparams.tst_step == val_steps_per_epoch):  
                            torch.save(self.test_outputs_stoch, plotCheckPoint + "/plots_stoch_outputs_test.pt")
                            torch.save(self.test_inits, plotCheckPoint + "/plots_init_test.pt")
                            torch.save(self.test_targets, plotCheckPoint + "/plots_targets_test.pt")  
                            
                    
                    else:         
                        #calculcate mean of n_stoch outputs                                     
                        if self.hparams.stoch_outputs==True :  #if applied stoch. model
                            all_outputs = [None] *self.hparams.n_stoch
                            outputs = torch.zeros(target_GT.size())
                            for i in range(self.hparams.n_stoch):
                                with torch.set_grad_enabled(True):
                                    inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
                                    all_outputs[i],hidden_new,cell_new,normgrad = self.generator(inputs_init,inputs_missing,masks)
                                all_outputs[i], hidden_new,cell_new,normgrad = all_outputs[i].detach(),  hidden_new.detach(),cell_new.detach(),normgrad.detach()
                                outputs += all_outputs[i]/self.hparams.n_stoch
                        else:
                            with torch.set_grad_enabled(True):
                                inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
                                outputs,hidden_new,cell_new,normgrad = self.generator(inputs_init,inputs_missing,masks)
                            outputs, hidden_new,cell_new,normgrad = outputs.detach(),  hidden_new.detach(),cell_new.detach(),normgrad.detach()
                        output=list()
                        if self.hparams.log_adv:
                            output.append( self.compute_metrics(inputs_init, outputs, targets_GT, masks, inputs_missing,0, 'Test'))
                        output.append(self.compute_metrics(inputs_init, outputs, targets_GT, masks, inputs_missing,1, 'Test') )    
                        self.test_outputs[self.hparams.tst_step*batch_size : (self.hparams.tst_step +1)*batch_size, :, :, :] = outputs
                        self.test_inits[self.hparams.tst_step*batch_size : (self.hparams.tst_step +1)*batch_size, :, :, :] = inputs_init
                        self.test_targets[self.hparams.tst_step*batch_size : (self.hparams.tst_step +1)*batch_size, :, :, :] = targets_GT
                        self.tst_step = self.hparams.tst_step + 1
    
                        if (self.hparams.tst_step == val_steps_per_epoch):
                            if self.hparams.stoch_outputs :
                                torch.save(test_outputs, plotCheckPoint + "/plots_outputs_test_stoch.pt")
                            else :
                                torch.save(self.test_outputs, plotCheckPoint + "/plots_outputs.pt")
                            torch.save(self.test_inits, plotCheckPoint + "/plots_init_test.pt")
                            torch.save(self.test_targets, plotCheckPoint + "/plots_targets_test.pt")
                    
                    return (output)

                def test_epoch_end(self, test_step_outputs):
                    if  self.hparams.stoch_test==False: 
                        if self.hparams.log_adv:
                            disc_outputs = [x[0] for x in test_step_outputs]
                            gen_outputs  = [x[1] for x in test_step_outputs]
                            
                            self.summarize_metrics(disc_outputs,0, 'Test')                 
                            self.summarize_metrics(gen_outputs,1, 'Test')
                            
                        else:
                            gen_outputs  = [x[0] for x in test_step_outputs]
                            self.summarize_metrics(gen_outputs,1, 'Test')
                    else:
                        if self.hparams.log_adv:
                            disc_outputs = []
                            gen_outputs  = []
                            dict_gen_i_min = self.summarize_metrics([x[0][1] for x in test_step_outputs]  ,1, 'Test', False)
                            dict_gen_i_max = self.summarize_metrics([x[0][1] for x in test_step_outputs]  ,1, 'Test', False) 
                            j_min=0  
                            j_max=0     
                            for i in range(self.hparams.n_stoch):
                                disc_outputs_i = [x[i][0] for x in test_step_outputs]
                                gen_outputs_i  = [x[i][1] for x in test_step_outputs]
                                disc_outputs += disc_outputs_i
                                gen_outputs  += gen_outputs_i
                                dict_disc_i = self.summarize_metrics(disc_outputs_i,0, 'Test')
                                dict_gen_i  = self.summarize_metrics(gen_outputs_i, 1, 'Test')
                                rec_i = dict_gen_i["avg_rec_loss"]
                                if rec_i.item() < dict_gen_i_min["avg_rec_loss"].item():
                                    dict_gen_i_min = copy.deepcopy(dict_gen_i)
                                    dict_disc_i_min = copy.deepcopy(dict_disc_i)
                                    j_min = i
                                    
                                if rec_i.item() > dict_gen_i_max["avg_rec_loss"].item():
                                    dict_gen_i_max = copy.deepcopy(dict_gen_i)
                                    dict_disc_i_max = copy.deepcopy(dict_disc_i)
                                    j_max = i
                            print("Best_performance (rec_loss):\n")
                            _ = self.summarize_metrics([x[j_min][0] for x in test_step_outputs],0, 'Test')
                            _ = self.summarize_metrics([x[j_min][1] for x in test_step_outputs],1, 'Test')
                            
                            print("Worst_performance (rec_loss):\n")
                            _ = self.summarize_metrics([x[j_max][0] for x in test_step_outputs],0, 'Test')
                            _ = self.summarize_metrics([x[j_max][1] for x in test_step_outputs],1, 'Test')
                            
                            print("Average_performance (rec_loss):\n")                                   
                            _ = self.summarize_metrics(disc_outputs,0, 'Test')                 
                            _ = self.summarize_metrics(gen_outputs,1, 'Test')
                            

                        else:
                            gen_outputs  = []
                            dict_gen_i_min = self.summarize_metrics([x[0][0] for x in test_step_outputs]  ,1, 'Test', False)
                            dict_gen_i_max = self.summarize_metrics([x[0][0] for x in test_step_outputs]  ,1, 'Test', False) 
                            j_min=0  
                            j_max=0             
                            for i in range(self.hparams.n_stoch):
                                gen_outputs_i  = [x[i][1] for x in test_step_outputs]
                                gen_outputs  += gen_outputs_i
                                dict_gen_i  = self.summarize_metrics(gen_outputs_i, 1, 'Test', False)
                                rec_i = dict_gen_i["avg_rec_loss"]
                                if rec_i < dict_gen_i_min["avg_rec_loss"]:
                                    dict_gen_i_min = copy.deepcopy(dict_gen_i)
                                    j_min = i
                                    
                                if rec_i > dict_gen_i_max["avg_rec_loss"]:
                                    dict_gen_i_max = copy.deepcopy(dict_gen_i)
                                    j_max = i
                            print("Best_performance (rec_loss):\n")
                            _ = self.summarize_metrics([x[j_min][0] for x in test_step_outputs],1, 'Test')
                            
                            print("Worst_performance (rec_loss):\n")
                            _ = self.summarize_metrics([x[j_max][0] for x in test_step_outputs],1, 'Test')
                            
                            print("Average_performance (rec_loss):\n")             
                            _ = self.summarize_metrics(gen_outputs,1, 'Test')


            plotCheckPoint  = "Plots/" + loss_comp + "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv)
            pathCheckPoint  = "Checkpoints/" + loss_comp + "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv)
            pathlogger = "lightning_logs/" + loss_comp
            logger = TensorBoardLogger(pathlogger, 
                                      name="alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv), version=0)


            filename = "optimal_adv_LiteModel"

        elif flagProcess[kk]==4:
            if LoadModel == True :                      
                model= LiteModel.load_from_checkpoint(pathCheckPoint + "/last.ckpt")
               
                checkpoint_callback = ModelCheckpoint(monitor='avg_val_g_loss',
                                                      dirpath= pathCheckPoint,
                                                      filename='optimal_adv_LiteModel-{epoch:02d}-{avg_val_g_loss:.4f}',
                                                      mode='min',
                                                      save_last=True)
                                                                           
                trainer = Trainer(min_epochs= 3, max_epochs =3, fast_dev_run=False, gpus='1', 
                                  resume_from_checkpoint=pathCheckPoint + "/" + "last.ckpt", callbacks=[checkpoint_callback], progress_bar_refresh_rate=-1, logger=logger)
                trainer.fit(model)
                model.generator.load_state_dict(model.optimal_generator.state_dict())
                trainer.test()
            else:
                if not os.path.exists(pathCheckPoint):
                    os.makedirs(pathCheckPoint)
                if not os.path.exists(plotCheckPoint):
                    os.makedirs(plotCheckPoint)
                if not os.path.exists(pathlogger+ "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv)):
                    os.makedirs(pathlogger+ "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv))
                                
                files = glob.glob(pathCheckPoint+"/*")
                for f in files:
                    os.remove(f)
                
                files = glob.glob(plotCheckPoint+"/*")
                for f in files:
                    os.remove(f)
                

                model = LiteModel()
                checkpoint_callback = ModelCheckpoint(monitor='avg_val_g_loss',
                                                      dirpath= pathCheckPoint,
                                                      filename='optimal_adv_LiteModel-{epoch:02d}-{avg_val_g_loss:.4f}',
                                                      mode='min',
                                                      save_last=True)
                trainer = Trainer(min_epochs=250, max_epochs=250, fast_dev_run=False, gpus='1', callbacks=[checkpoint_callback], logger=logger, progress_bar_refresh_rate=-1)
                trainer.fit(model)
                model.generator.load_state_dict(model.optimal_generator.state_dict())
                trainer.test()

        elif flagProcess[kk]==5:
            plotCheckPoint  = "Plots/" + loss_comp + "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv)
            pathCheckPoint  = "Checkpoints/" + loss_comp + "/alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv)
            pathlogger = "lightning_logs/" + loss_comp
            logger = TensorBoardLogger(pathlogger, 
                                      name="alpha_rec=%.3f  alpha_AE=%.3f  alpha_adv=%.3f"%(alpha_rec, alpha_AE, alpha_adv), version=0)


            filename = "optimal_adv_LiteModel"
            
            
            model= LiteModel.load_from_checkpoint(pathCheckPoint + "/last.ckpt")
               
            checkpoint_callback = ModelCheckpoint(monitor='avg_val_g_loss',
                                                      dirpath= pathCheckPoint,
                                                      filename='optimal_adv_LiteModel-{epoch:02d}-{avg_val_g_loss:.4f}',
                                                      mode='min',
                                                      save_last=True)
                                                                           
            trainer = Trainer(min_epochs= 250, max_epochs =250, fast_dev_run=False, gpus='1', 
                                  resume_from_checkpoint=pathCheckPoint + "/" + "last.ckpt", callbacks=[checkpoint_callback], progress_bar_refresh_rate=-1, logger=logger)
            #trainer.fit(model)
            model.generator.load_state_dict(model.optimal_generator.state_dict())
            trainer.test(model)
