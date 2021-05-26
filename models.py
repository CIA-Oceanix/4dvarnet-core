import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import solver as NN_4DVar

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


class Phi_r(torch.nn.Module):
    def __init__(self, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(shapeData,DimAE,dW,dW2,sS,nbBlocks,rateDr)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
        # self.save_hyperparameters(conf)
        self.hparam  = hparam
        self.var_Val = kwargs['var_Val']
        self.var_Tr  = kwargs['var_Tr']
        self.var_Tt  = kwargs['var_Tt']

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.hparam.shapeData[0], self.hparam.DimAE, self.hparam.dW, self.hparam.dW2, self.hparam.sS, self.hparam.nbBlocks, self.hparam.dropout),
            Model_H(self.hparam.shapeData[0]), 
            NN_4DVar.model_GradUpdateLSTM(self.hparam.shapeData, self.hparam.UsePriodicBoundary, self.hparam.dim_grad_solver, self.hparam.dropout), 
                None, None, self.hparam.shapeData, self.hparam.n_grad)

        self.model_LR     = ModelLR()
        self.gradient_img = Gradient_img()
        # loss weghing wrt time

        self.w_loss       = torch.nn.Parameter(kwargs['w_loss'], requires_grad=False) # duplicate for automatic upload to gpu
        self.x_gt  = None # variable to store Ground Truth
        self.x_oi  = None # variable to store OI
        self.x_rec = None # variable to store output of test method

        self.automatic_optimization = self.hparam.automatic_optimization
        
    def forward(self):
        return 1

    def configure_optimizers(self):

        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparam.lr_update[0]},
                                  {'params': self.model.model_VarCost.parameters(), 'lr': self.hparam.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparam.lr_update[0]},
                                ], lr=0.)


        return optimizer

    def on_epoch_start(self):
        # enfore acnd check some hyperparameters
        self.model.n_grad   = self.hparam.n_grad

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparam.iter_update) & (self.current_epoch > 0):
            indx             = self.hparam.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparam.nb_grad_update[indx],self.hparam.lr_update[indx]))
            
            self.hparam.n_grad = self.hparam.nb_grad_update[indx]
            self.model.n_grad   = self.hparam.n_grad 
            
            mm = 0
            lrCurrent = self.hparam.lr_update[indx]
            lr = np.array([lrCurrent,lrCurrent,0.5*lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparam.learning_rate
                mm += 1
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')

        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.hparam.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparam.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')

        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, targets_GT = test_batch
        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'gt' : targets_GT.detach().cpu(),
                'oi' : targets_OI.detach().cpu(),
                'preds' : out.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()

        self.x_gt = gt[:,int(self.hparam.dT/2),:,:]
        self.x_oi = oi[:,int(self.hparam.dT/2),:,:]
        self.x_rec = x_test_rec[:,int(self.hparam.dT/2),:,:]

        # save NetCDF
        path_save1 = self.logger.log_dir+'/test.nc'
        save_netcdf(saved_path1 = path_save1, x_test_rec = x_test_rec,
            lon = lon,lat = lat)
        # compute nRMSE
        path_save2 = self.logger.log_dir+'/nRMSE.txt'
        tab_scores = nrmse_scores(gt,oi,x_test_rec,path_save2)
        print('*** Display nRMSE scores ***')
        print(tab_scores)

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
            outputsSLR   = outputs[:, 0:self.hparam.dT, :, :]
            outputs      = outputsSLR + outputs[:, self.hparam.dT:, :, :]

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
            loss     = self.hparam.alpha_MSE * (self.hparam.betaX * loss_All + self.hparam.betagX * loss_GAll) \
                 + 0.5 * self.hparam.alpha_Proj * (loss_AE + loss_AE_GT)
            loss    += self.hparam.alpha_LR * loss_LR + self.hparam.alpha_SR * loss_SR
            
            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT,self.w_loss)
            mse       = loss_All.detach()
            mseGrad   = loss_GAll.detach()  
            metrics   = dict([('mse',mse),('mseGrad',mseGrad),('meanGrad',mean_GAll),('mseOI',loss_OI.detach()),('mseGOI',loss_GOI.detach())])
            #print(mse.cpu().detach().numpy())
            
        return loss,outputs, metrics
