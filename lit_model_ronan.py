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

import os
import sys
#sys.path.append('../4dvarnet-core-old20211112')
#sys.path.append('../4dvarnet-core')
import solver_ronan as NN_4DVar
# import solver as NN_4DVar

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



from omegaconf import OmegaConf
OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
cfg = get_cfg("xp_aug/xp_repro/quentin_repro")
dm = instantiate(cfg.datamodule)
if __name__=='__main__':
    dm.setup()
meanTr, stdTr= dm.norm_stats 
shapeData = cfg.params.shape_data
rateDr = cfg.params.dropout_phi_r
nbBlocks = cfg.params.nbBlocks
sS = cfg.params.sS
dW = cfg.params.dW
dW2 = cfg.params.dW2
suffix_exp = 'repro_q'
dirSAVE = 'results'
UsePriodicBoundary = False
VAL_RND_TRAINING = 0.
USE_RND_TRAINING = False#True#
if USE_RND_TRAINING == False :
    RND_TRAINING = False
else:
    RND_TRAINING = True

print('.... RND Training %s %f'%(USE_RND_TRAINING,VAL_RND_TRAINING))
flagProcess =  0

## NN architectures and optimization parameters
batch_size      = 2#16#4#4#8#12#8#256#
DimAE           = 50#10#10#50
dimGradSolver   = 150#150 # dimension of the hidden state of the LSTM cell
rateDropout     = 0.25 # dropout rate
True#
flag_aug_state = 1#1 #False#
dim_aug_state =  0
flag_obs_model = 0#8#1 # 0: no-SST, 1: SST
flag_augment_training_data = True #False#         'modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_01-dgrad150-epoch=20-val_loss=1.95.ckpt''modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_01-dgrad150-epoch=20-val_loss=1.95.ckpt'
flagPhiParam = 1#4#1
flagAutomOptim = False#True#
width_med_filt_spatial = 1
width_med_filt_temp = 1

scale_dwscaling = 0

# data generation
sigNoise        = 0. ## additive noise standard deviation
flagSWOTData    = True #False ## use SWOT data or not
dT              = 7 ## Time window of each space-time patch
W               = 200#200 ## width/height of each space-time patch
dx              = 1   ## subsampling step if > 1
Nbpatches       = 1#10#10#25 ## number of patches extracted from each time-step 
rnd1            = 0 ## random seed for patch extraction (space sam)
rnd2            = 100 ## random seed for patch extraction
dwscale         = 1
padding_mode = 'reflect'#'zeros' # 
dim_obs_sst = 3

W = int(W/dx)

flag_data_240times240 = False#True #

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

            r = r.to(xinp.device)
            
            return xLR + r*xHR
        else:
            return xLR + xHR
  
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
  
    def forward(self, x):
        return torch.mul(1.,x)



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
if __name__ =='__main__':
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
        self.n_grad          = 1
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
    def __init__(self,
            hparam=hparam,
            conf=flag_obs_model,
            mean_Tr=meanTr,
            val_Tr=stdTr**2,
            dimGradSolver=dimGradSolver,
            rateDropout=rateDropout,
            shapeData=shapeData,
            padding_mode=padding_mode,
            flagAutomOptim=flagAutomOptim,
            *args,
           **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        #print( '... %d'%conf.obs_model , flush = True)
        self.hparams.obs_model = conf

        # hyperparameters
        self.hparams.iter_update     = [0, 20, 50, 70, 100, 150, 800]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        self.hparams.k_batch         = 1
        
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        self.hparams.k_n_grad        = 1
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

        self.model_LR     = ModelLR()
        self.gradient_img = Gradient_img()
        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_rec_ssh        = None # variable to store output of test method
        self.x_rec_oi        = None # variable to store output of test method
        self.x_rec_gt        = None # variable to store output of test method
        self.x_rec_ssh_obs = None
        self.x_feat_sst = None
        self.curr = 0

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
        return {
                'preds_ssh': out[0].detach().cpu(),
                'obs_ssh': out[1].detach().cpu(),
                'feat_sst': out[6].detach().cpu()
        }
        #if self.hparams.obs_model > 0 : #( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
        ##    return {'preds_ssh': out[0].detach().cpu(),'obs_ssh': out[1].detach().cpu(),'feat_sst': out[6].detach().cpu()}
        #else:
        #    return {'preds_ssh': out[0].detach().cpu(),'obs_ssh': out[1].detach().cpu(),'feat_sst': out[6].detach().cpu()}
            
    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    def test_epoch_end(self, outputs):
        print('I am here')
        x_test_rec = torch.cat([chunk['preds_ssh'] for chunk in outputs]).numpy()
        x_test_rec = stdTr * x_test_rec + meanTr        
        self.x_rec_ssh = x_test_rec[:,int(dT/2),:,:]


        x_test_ssh_obs = torch.cat([chunk['obs_ssh'] for chunk in outputs]).numpy()
        x_test_ssh_obs[ x_test_ssh_obs == 0. ] = float('NaN')
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
            targets_OI, inputs_obs, inputs_Mask, targets_GT = batch
        else:# self.hparams.obs_model > 0 : #( self.hparams.obs_model== 1 ) | ( self.hparams.obs_model== 2 ) | ( self.hparams.obs_model== 3 ) | ( self.hparams.obs_model== 4 ) :
            targets_OI, inputs_obs, inputs_Mask, inputs_SST, targets_GT = batch
            mask_SST  = 1. + 0. * inputs_SST
        inputs_Mask = inputs_Mask.float()
        targets_OI = targets_OI.float()
        inputs_obs = inputs_obs.float()
        if flag_aug_state == 1 :
            if dim_aug_state == 0 :
                inputs_init_    = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , inputs_Mask * (inputs_obs - targets_OI)), dim=1)
                inputs_missing = torch.cat((targets_OI, inputs_Mask * (inputs_obs - targets_OI) , 0. * targets_OI), dim=1)
                new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask , 0. * inputs_Mask ), dim=1)
            else  :
                init_aug_state = 0. * torch.randn((inputs_obs.size(0),dim_aug_state,inputs_obs.size(2),inputs_obs.size(3))).to(self.device)
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
                init_aug_state = 0. * torch.randn((inputs_obs.size(0),dim_aug_state,inputs_obs.size(2),inputs_obs.size(3))).to(self.device)
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
    
                loss_All   = NN_4DVar.compute_WeightedLoss((outputs[:,:,dw:outputs.size(2)-dw,dw:outputs.size(3)-dw] - targets_GT[:,:,dw:outputs.size(2)-dw,dw:outputs.size(3)-dw]), self.w_loss)
    
                loss_GAll  = NN_4DVar.compute_WeightedLoss(gx_outputs[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] - gx_targets_GT[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw], self.w_loss)
                loss_GAll += NN_4DVar.compute_WeightedLoss(gy_outputs[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw] - gy_targets_GT[:,:,dw:gx_outputs.size(2)-dw,dw:gx_outputs.size(3)-dw], self.w_loss)
    
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
                err = (outputs[:,int(dT/2),:,:] - targets_GT[:,int(dT/2),:,:]).view(-1,1,targets_GT.size(2),targets_GT.size(3))
                
                #fft_err = torch.fft.rfft2(err,dim=(2,3),norm="ortho")
                #fft_gt = torch.fft.rfft2( targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=(2,3),norm="ortho")
                
                window = torch.hann_window(targets_GT.size(2)).repeat(targets_GT.size(0),1,targets_GT.size(3))
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(self.device)
                fft_err = torch.fft.rfft(window * err,dim=2,norm="ortho")
                fft_gt = torch.fft.rfft( window * targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=2,norm="ortho")
                #loss_fft1 = torch.mean(fft_err.abs(),dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs(),dim=3) )
                loss_fft1 = torch.mean(fft_err.abs()**2,dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs()**2,dim=3) )
    
                window = torch.hann_window(targets_GT.size(3)).repeat(targets_GT.size(0),targets_GT.size(2),1)
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(self.device)
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
                window = window.to(self.device)
                fft_err = torch.fft.rfft(window * err,dim=2,norm="ortho")
                fft_gt = torch.fft.rfft( window * targets_GT[:,int(dT/2),:,:].view(-1,1,targets_GT.size(2),targets_GT.size(3)) ,dim=2,norm="ortho")
                #loss_fft1 = torch.mean(fft_err.abs(),dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs(),dim=3) )
                loss_fft1 = torch.mean(fft_err.abs()**2,dim=3) / (0.*1e-15 + torch.mean(fft_gt.abs()**2,dim=3) )
    
                window = torch.hann_window(targets_GT.size(3)).repeat(targets_GT.size(0),targets_GT.size(2),1)
                window = window.view(-1,1,targets_GT.size(2),targets_GT.size(3))
                window = window.to(self.device)
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

def compute_metrics(X_test,X_rec):
    # MSE
    mse = np.mean( (X_test - X_rec)**2 )

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
    err['time'] = (err.time - err.time[0]) / numpy.timedelta64(1, 'D')
    
    # Rechunk SSH_true
    signal = ds_ref['sossheig'].chunk({"lat":1, 'time': ds_ref['time'].size, 'lon': ds_ref['lon'].size})
    # make time vector in days units
    signal['time'] = (signal.time - signal.time[0]) / numpy.timedelta64(1, 'D')

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
    
        loadTrainedModel = False# True# 
        mod = LitModel( conf = flag_obs_model )

        # mod = mod.load_from_checkpoint('results-repro_q/modelSLA-L2-GF-augdata%02dtrue-augstate-repro_q-dT07-igrad05_03-dgrad150-epoch=27-val_loss=2.46.ckpt') 
        # mod = mod.load_from_checkpoint('results-repro_q/modelSLA-L2-GF-augdata%02dtrue-augstate-repro_q-dT07-igrad05_03-dgrad150-epoch=23-val_loss=2.50.ckpt') 
        # mod = mod.load_from_checkpoint('results-repro_q/modelSLA-L2-GF-augdata%02dtrue-augstate-repro_q-dT07-igrad05_03-dgrad150-epoch=13-val_loss=2.47.ckpt') 

        mod.hparams.n_grad          = 5
        mod.hparams.k_n_grad        = 3
        mod.hparams.iter_update     = [0, 200, 200, 320, 380, 400, 800]  # [0,2,4,6,9,15]
        mod.hparams.nb_grad_update  = [0, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        mod.hparams.lr_update       = [1e-4, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]

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
        mod.hparams.dw_loss = 10

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
            filename_chkpt = filename_chkpt + 'augdata%02dtrue-'            
                       
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

        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=[0], max_epochs=10,   callbacks=[checkpoint_callback])
    
        ## training loop
        trainer.fit(mod, dm.train_dataloader(), dm.val_dataloader() )
        # assert False
        trainer.test(mod, test_dataloaders=dm.val_dataloader())
        X_val    = dm.val_ds.datasets[0].gt_ds.ds.ssh.isel(time=slice(dT//2, -dT//2 + 1)).transpose('time', 'lat', 'lon').values
        X_OI     = dm.val_ds.datasets[0].oi_ds.ds.ssh_mod.isel(time=slice(dT//2, -dT//2 + 1)).transpose('time', 'lat', 'lon').values 
        val_mseRec = compute_metrics(X_val, mod.x_rec_ssh)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        vds = (
                dm.val_ds.datasets[0]
                .gt_ds.ds.isel(time=slice(dT//2, -dT//2 + 1))
                .transpose('time', 'lat', 'lon')
                .assign(pred=( ('time', 'lat', 'lon'), mod.x_rec_ssh))
                .assign(oi=( ('time', 'lat', 'lon'), X_OI))
                .transpose('lon', 'lat', 'time', )
                # .assign(time=lambda d: (d.time - d.time.min()).astype('float'))
        )
        vpsd_bs = psd_based_scores(
                vds[['pred']].assign(sossheig= vds.pred), 
                vds[['ssh']].assign(sossheig= vds.ssh), 
        )
        vpsd_bs_oi = psd_based_scores(
                vds[['oi']].assign(sossheig= vds.oi), 
                vds[['ssh']].assign(sossheig= vds.ssh), 
        )
        print(vpsd_bs[1], vpsd_bs[2])
        print(vpsd_bs_oi[1], vpsd_bs_oi[2])

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dm.test_dataloader())
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        c = 10
        crop = dict(
            lat=slice(c, -c),
            lon=slice(c, -c),
        )
        X_OI     = dm.test_ds.datasets[0].oi_ds.ds.ssh_mod.isel(time=slice(dT//2, - dT//2 + 1)).transpose('time', 'lat', 'lon').values 
        X_test    = dm.test_ds.datasets[0].gt_ds.ds.ssh.isel(time=slice(dT//2, -dT//2 + 1)).transpose('time', 'lat', 'lon').values
        test_mseOI  = compute_metrics(X_test,X_OI)     
        test_mseRec = compute_metrics(X_test,mod.x_rec_ssh)     
        tds = (
                dm.test_ds.datasets[0]
                .gt_ds.ds.isel(time=slice(dT//2, -dT//2 + 1))
                .transpose('time', 'lat', 'lon')
                .assign(pred=( ('time', 'lat', 'lon'), mod.x_rec_ssh))
                .assign(oi=( ('time', 'lat', 'lon'), X_OI))
                .transpose('lon', 'lat', 'time', )
                .isel(**crop)
        )
        (tds - tds.ssh).drop('ssh').pipe(lambda d: np.sqrt((d**2).mean()))
        tpsd_bs = psd_based_scores(
                tds[['pred']].assign(sossheig= tds.pred), 
                tds[['ssh']].assign(sossheig= tds.ssh), 
        )
        tpsd_bs_oi = psd_based_scores(
                tds[['oi']].assign(sossheig= tds.oi), 
                tds[['ssh']].assign(sossheig= tds.ssh), 
        )
            
        
        print(tpsd_bs[1], tpsd_bs[2])
        print(tpsd_bs_oi[1], tpsd_bs_oi[2])
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))
    
