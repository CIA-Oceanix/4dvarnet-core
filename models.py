import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import solver as NN_4DVar
from metrics import save_netcdf, nrmse_scores, plot_nrmse, plot_snr, plot_maps, plot_ensemble
from omegaconf import OmegaConf
import einops
from scipy import stats
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLinUnit(torch.nn.Module):
    def __init__(self,dimIn,dim,dW,dW2,dropout=0.):
        super(BiLinUnit, self).__init__()
        self.conv1  = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False)
        self.conv2  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.conv3  = torch.nn.Conv2d(2*dim, dimIn, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

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
    def __init__(self,dimInp,dimAE,dW,dW2,sS,nbBlocks,rateDropout=0.):
        super(Encoder, self).__init__()

        self.NbBlocks  = nbBlocks
        self.DimAE     = dimAE
        #self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        #self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        self.pool1   = torch.nn.AvgPool2d(sS)
        self.convTr  = torch.nn.ConvTranspose2d(dimInp,dimInp,(sS,sS),stride=(sS,sS),bias=False)          

        #self.NNtLR    = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        #self.NNHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)                      
        self.NNLR     = self.__make_BilinNN(dimInp,self.DimAE,dW,dW2,self.NbBlocks,rateDropout)
        self.NNHR     = self.__make_BilinNN(dimInp,self.DimAE,dW,dW2,self.NbBlocks,rateDropout)                      
        self.dropout  = torch.nn.Dropout(rateDropout)
      
    def __make_BilinNN(self,dimInp,dimAE,dW,dW2,Nb_Blocks=2,dropout=0.): 
          layers = []
          layers.append( BiLinUnit(dimInp,dimAE,dW,dW2,dropout) )
          for kk in range(0,Nb_Blocks-1):
              layers.append( BiLinUnit(dimAE,dimAE,dW,dW2,dropout) )
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

class CorrelateNoise(torch.nn.Module):
    def __init__(self,shape_data, dim_cn):
        super(CorrelateNoise, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data,dim_cn,(3,3),padding=1,bias=False)
        self.conv2 = torch.nn.Conv2d(dim_cn,2*dim_cn,(3,3),padding=1,bias=False)
        self.conv3 = torch.nn.Conv2d(2*dim_cn,shape_data,(3,3),padding=1,bias=False)

    def forward(self, w):
        w = self.conv1(F.relu(w)).to(device)
        w = self.conv2(F.relu(w)).to(device)
        w = self.conv3(w).to(device)
        return w

class RegularizeVariance(torch.nn.Module):
    def __init__(self,shape_data, dim_rv):
        super(RegularizeVariance, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data,dim_rv,(3,3),padding=1,bias=False)
        self.conv2 = torch.nn.Conv2d(dim_rv,2*dim_rv,(3,3),padding=1,bias=False)
        self.conv3 = torch.nn.Conv2d(2*dim_rv,shape_data,(3,3),padding=1,bias=False)

    def forward(self, v):
        v = self.conv1(F.relu(v)).to(device)
        v = self.conv2(F.relu(v)).to(device)
        v = self.conv3(v).to(device)
        return v

class Phi_r(torch.nn.Module):
    def __init__(self, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr,stochastic):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(shapeData,DimAE,dW,dW2,sS,nbBlocks,rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shapeData,10)
        self.regularize_variance = RegularizeVariance(shapeData,10)        
        self.stochastic = stochastic

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.stochastic==True:
            W = torch.randn(x.shape).to(device)
            # g(W) = alpha(x)*h(W) 
            #gW = torch.mul(self.regularize_variance(x),self.correlate_noise(W))
            gW = self.correlate_noise(W)
            #print(stats.describe(gW.detach().cpu().numpy()))
            x = x + gW
        return x

class Model_H(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_H, self).__init__()
        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout


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

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)


############################################ Lightning Module #######################################################################
class LitModel(pl.LightningModule):
    def __init__(self, hparam, *args, **kwargs):
        super().__init__()
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)
        self.save_hyperparameters(hparams)
        self.var_Val = kwargs['var_Val']
        self.var_Tr  = kwargs['var_Tr']
        self.var_Tt  = kwargs['var_Tt']
 
        # create longitudes & latitudes coordinates
        self.xmin = kwargs['min_lon']
        self.xmax = kwargs['max_lon']
        self.ymin = kwargs['min_lat']
        self.ymax = kwargs['max_lat']
        self.lon = np.arange(self.xmin,self.xmax,.05)
        self.lat = np.arange(self.ymin,self.ymax,.05)
        self.ds_size_time = kwargs['ds_size_time']
        self.ds_size_lon = kwargs['ds_size_lon']
        self.ds_size_lat = kwargs['ds_size_lat']

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.hparams.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS, self.hparams.nbBlocks, self.hparams.dropout, self.hparams.stochastic),
            Model_H(self.hparams.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.hparams.shapeData, self.hparams.UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout),
                None, None, self.hparams.shapeData, self.hparams.n_grad)

        self.model_LR     = ModelLR()
        self.gradient_img = Gradient_img()
        # loss weghing wrt time

        self.w_loss       = torch.nn.Parameter(kwargs['w_loss'], requires_grad=False) # duplicate for automatic upload to gpu
        self.x_gt  = None # variable to store Ground Truth
        self.x_oi  = None # variable to store OI
        self.x_rec = None # variable to store output of test method

        self.automatic_optimization = self.hparams.automatic_optimization
        
    def forward(self):
        return 1

    def configure_optimizers(self):

        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                  {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                                ], lr=0.)


        return optimizer

    def on_epoch_start(self):
        # enfore acnd check some hyperparameters
        self.model.n_grad   = self.hparams.n_grad

    def on_train_epoch_start(self):
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
                    
        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        if loss is None:
            return loss
        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.hparams.automatic_optimization == False :
            opt = self.optimizers()
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val , on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, targets_GT = test_batch
        if self.hparams.stochastic == True :
            loss = []
            out = []
            metrics = []
            for i in range(self.hparams.size_ensemble):
                loss_, out_, metrics_ = self.compute_loss(test_batch, phase='test')
                if loss_ is not None:
                    loss.append(loss_)
                    metrics.append(metrics_)
                self.log('test_loss', np.nanmean([loss[i].detach().cpu() for i in range(len(loss))]))
                self.log("test_mse", np.nanmean([ metrics[i]['mse'].detach().cpu()/self.var_Tt for i in range(len(metrics)) ]),
                                      on_step=False, on_epoch=True, prog_bar=True)
                self.log("test_mseG", np.nanmean([ metrics[i]['mseGrad'].detach().cpu()/metrics[i]['meanGrad'].detach().cpu()  for i in range(len(metrics)) ]),
                                      on_step=False, on_epoch=True, prog_bar=True)
                out.append(out_)
            return {'gt' : targets_GT.detach().cpu(),
                'oi' : targets_OI.detach().cpu(),
                'preds' : torch.stack([out_.detach().cpu() for out_ in out],dim=-1)}
        else:
            loss, out, metrics = self.compute_loss(test_batch, phase='test')
            if loss is not None:
                self.log('test_loss', loss)
                self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True)
                self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True)
            return {'gt' : targets_GT.detach().cpu(),
                'oi' : targets_OI.detach().cpu(),
                'preds' : out.detach().cpu()}

    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        print(pred.shape)
        ds_size = {'time': self.ds_size_time,
                   'lon': self.ds_size_lon,
                   'lat': self.ds_size_lat,
                   }

        gt = einops.rearrange(gt,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
        oi = einops.rearrange(oi,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
        if self.hparams.stochastic == True :
            pred = einops.rearrange(pred,
                '(t_idx lat_idx lon_idx ens_idx) win_time win_lat win_lon win_ens -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (ens_idx win_ens)',
                ens_idx=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
        else:
            pred = einops.rearrange(pred,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )

        self.x_gt = gt[:,int(self.hparams.dT/2),:,:]
        self.x_oi = oi[:,int(self.hparams.dT/2),:,:]

        # display ensemble
        if self.hparams.stochastic == True :
            path_save0 = self.logger.log_dir+'/maps_ensemble.png'
            plot_ensemble(pred[0,int(self.hparams.dT/2),:,:,:],
                          self.lon,self.lat,path_save0)
            self.x_rec = np.nanmean(pred[:,int(self.hparams.dT/2),:,:,:],axis=-1)
            pred = np.nanmean(pred,axis=-1)
        else:
            self.x_rec = pred[:,int(self.hparams.dT/2),:,:]

        # display map
        path_save0 = self.logger.log_dir+'/maps.png'
        plot_maps(gt[0,int(self.hparams.dT/2),:,:],
                  oi[0,int(self.hparams.dT/2),:,:],
                  pred[0,int(self.hparams.dT/2),:,:],
                  self.lon,self.lat,path_save0)
        # save NetCDF
        path_save1 = self.logger.log_dir+'/test.nc'
        save_netcdf(saved_path1 = path_save1, pred = pred,
            lon = self.lon,lat = self.lat, index_test = np.arange(60, 77))
        # compute nRMSE
        path_save2 = self.logger.log_dir+'/nRMSE.txt'
        tab_scores = nrmse_scores(gt,oi,pred,path_save2)
        print('*** Display nRMSE scores ***')
        print(tab_scores)
        # plot nRMSE
        path_save3 = self.logger.log_dir+'/nRMSE.png'
        nrmse_fig = plot_nrmse(gt,oi,pred,path_save3,index_test = np.arange(60, 77))
        self.logger.experiment.add_figure('NRMSE', nrmse_fig, global_step=self.current_epoch)
        # plot SNR
        path_save4 = self.logger.log_dir+'/SNR.png'
        snr_fig = plot_snr(gt,oi,pred,path_save4)
        self.logger.experiment.add_figure('SNR', snr_fig, global_step=self.current_epoch)

    def compute_loss(self, batch, phase):

        targets_OI, inputs_Mask, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_GT),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                      ('mseGOI', 0.)])
            )
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
            outputsSLR   = outputs[:, 0:self.hparams.dT, :, :]
            outputs      = outputsSLR + outputs[:, self.hparams.dT:, :, :]

            # reconstruction losses
            g_outputs  = self.gradient_img(outputs)
            loss_All   = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.w_loss)
            loss_GAll  = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.w_loss)

            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE     = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT         = torch.cat((targets_GT, outputsSLR - targets_GT), dim=1)
            #yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR      = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR      = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.w_loss)

            # total loss
            loss     = self.hparams.alpha_MSE * (self.hparams.betaX * loss_All + self.hparams.betagX * loss_GAll) \
                 + 0.5 * self.hparams.alpha_Proj * (loss_AE + loss_AE_GT)
            loss    += self.hparams.alpha_LR * loss_LR + self.hparams.alpha_SR * loss_SR
            
            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT,self.w_loss)
            mse       = loss_All.detach()
            mseGrad   = loss_GAll.detach()  
            metrics   = dict([('mse',mse),('mseGrad',mseGrad),('meanGrad',mean_GAll),('mseOI',loss_OI.detach()),('mseGOI',loss_GOI.detach())])

        return loss,outputs, metrics


class Model_HwithSST(torch.nn.Module):
    def __init__(self,shapeData,dim=5):
        super(Model_HwithSST, self).__init__()

        self.DimObs        = 2
        self.dimObsChannel = np.array([shapeData,dim])

        self.conv11  = torch.nn.Conv2d(shapeData,self.dimObsChannel[1],(3,3),padding=1,bias=False)
                    
        self.conv21  = torch.nn.Conv2d(int(shapeData/2),self.dimObsChannel[1],(3,3),padding=1,bias=False)
        self.convM   = torch.nn.Conv2d(int(shapeData/2),self.dimObsChannel[1],(3,3),padding=1,bias=False)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]
        dyout1 = self.conv11(x) - self.conv21(y1)
        dyout1 = dyout1 * self.S( self.convM( mask[1] ) )                  
        
        return [dyout,dyout1]


class LitModelWithSST(LitModel):
    def __init__(self, hparam, *args, **kwargs):
        super().__init__(hparam, *args, **kwargs)

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.hparams.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS, self.hparams.nbBlocks, self.hparams.dropout),
            Model_HwithSST(self.hparams.shapeData[0],self.hparams.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.hparams.shapeData, self.hparams.UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout),
                None, None, self.hparams.shapeData, self.hparams.n_grad)

    def configure_optimizers(self):

        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                  {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                                ], lr=0.)


        return optimizer

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx             = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparams.nb_grad_update[indx],self.hparams.lr_update[indx]))
            
            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad   = self.hparams.n_grad
            
            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent,lrCurrent,lrCurrent,0.5*lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, targets_GT, sst_GT = test_batch
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True)
        return {'gt' : targets_GT.detach().cpu(),
                'oi' : targets_OI.detach().cpu(),
                'preds' : out.detach().cpu()}

    def compute_loss(self, batch, phase): ## to be updated

        targets_OI, inputs_Mask, targets_GT, sst_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_GT),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                      ('mseGOI', 0.)])
            )
        new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask ), dim=1)
        inputs_init    = torch.cat((targets_OI, inputs_Mask * (targets_GT - targets_OI)), dim=1)
        inputs_missing = torch.cat((targets_OI, inputs_Mask * (targets_GT - targets_OI)), dim=1)
        mask_SST       = 1. + 0. * sst_GT


        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)
        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, [inputs_missing,sst_GT], [new_masks,mask_SST])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR   = outputs[:, 0:self.hparams.dT, :, :]
            outputs      = outputsSLR + outputs[:, self.hparams.dT:, :, :]

            # reconstruction losses
            g_outputs  = self.gradient_img(outputs)
            loss_All   = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.w_loss)
            loss_GAll  = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.w_loss)

            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE     = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT         = torch.cat((targets_GT, outputsSLR - targets_GT), dim=1)
            #yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR      = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR      = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.w_loss)

            # total loss
            loss     = self.hparams.alpha_MSE * (self.hparams.betaX * loss_All + self.hparams.betagX * loss_GAll) \
                 + 0.5 * self.hparams.alpha_Proj * (loss_AE + loss_AE_GT)
            loss    += self.hparams.alpha_LR * loss_LR + self.hparams.alpha_SR * loss_SR
            
            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT,self.w_loss)
            mse       = loss_All.detach()
            mseGrad   = loss_GAll.detach()  
            metrics   = dict([('mse',mse),('mseGrad',mseGrad),('meanGrad',mean_GAll),('mseOI',loss_OI.detach()),('mseGOI',loss_GOI.detach())])

        return loss,outputs, metrics

