import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
import solver as NN_4DVar

from data_utils import save_NetCDF


class LitModel(pl.LightningModule):
    def __init__(self, phi_r, model_H, model_LR, gradient_img, kw):
        super().__init__()
        self.kw       = kw
        self.m_Grad   = NN_4DVar.model_GradUpdateLSTM(kw['shapeData'], kw['UsePriodicBoundary'], kw['dimGradSolver'], kw['rateDropout'])
        self.model    = NN_4DVar.Solver_Grad_4DVarNN(phi_r, model_H, self.m_Grad, None, None, kw['shapeData'], kw['NBGradCurrent'])
        self.model_S  = Model_Sampling(kw['shapeData'])
        self.model_LR = model_LR
        self.IterUpdate = [0, 25, 50, 100, 500, 600, 800]  # [0,2,4,6,9,15]
        self.NbGradIter = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.gradient_img = gradient_img
        # loss weghing wrt time
        self.wLoss = torch.nn.Parameter(kw['wLoss'], requires_grad=False)

    def forward(self):
        return 1

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
                {'params': self.model.model_Grad.parameters()},
                {'params': self.model.phi_r.encoder.parameters(), 'lr': self.kw['lambda_LRAE'] * self.kw['lrCurrent']},
                {'params': self.model.phi_r.decoder.parameters(), 'lr': self.kw['lambda_LRAE'] * self.kw['lrCurrent']}
            ], 
        lr=self.kw['lrCurrent']
        )

        optimizer_Sampling = optim.Adam(self.model_S.parameters(), lr=self.kw['lr_Sampling'] * self.kw['lrCurrent'])
        scheduler1 = MultiStepLR(optimizer, milestones=[0, 25, 50, 100, 500, 600, 800], gamma=0.1)
        scheduler2 = MultiStepLR(optimizer_Sampling, milestones=[0, 25, 50, 100, 500, 600, 800], gamma=0.05)

        return [optimizer, optimizer_Sampling], [scheduler1, scheduler2]

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        if (self.current_epoch in self.IterUpdate) & (self.current_epoch > 0):
            indx = self.IterUpdate.index(self.current_epoch)
            self.model.NGrad = self.NbGradIter[indx]
        loss, out = self.compute_loss(train_batch, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, out = self.compute_loss(val_batch, phase='val')
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out = self.compute_loss(test_batch, phase='test')
        return {'preds': out.detach().cpu()}

    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        x_test_rec = kw['stdTr'] * x_test_rec + kw['meanTr']
        path_save = Path(kw['path_save'])
        print("*********************************x_test_rec.shape *********************************** = ", x_test_rec.shape )
        path_save.parent.mkdir(exist_ok=True, parents=True)
        save_NetCDF(saved_path1=path_save, x_test_rec=x_test_rec)

    def compute_loss(self, batch, phase):
        targets_OI, targets_OI1, inputs_Mask, targets_GT = batch
        # use low-resolution
        if self.kw['flagMultiScale'] == True:
            targets_GTLR = self.model_LR(targets_OI)

            # sampling mask
            new_masks = self.model_S(torch.cat((targets_OI1, targets_OI1), dim=1), phase)

            if self.kw['flagUseObsData'] == True:
                inputs_Mask2 = inputs_Mask.repeat(1, 2, 1, 1)
                new_masks[0] = inputs_Mask2 + (1.0 - inputs_Mask2) * new_masks[0]
                new_masks[1] = inputs_Mask2 + (1.0 - inputs_Mask2) * new_masks[1]

            # init
            if self.kw['flagUseOI'] == True:
                new_masks[0][:, 0:self.kw['dT'], :, :] = 1.0 + 0. * new_masks[0][:, 0:self.kw['dT'], :, :]
                new_masks[1][:, 0:self.kw['dT'], :, :] = 1.0 + 0. * new_masks[1][:, 0:self.kw['dT'], :, :]

            idxSampMat = int(1)
            mask_t = 1. - torch.nn.functional.threshold(1.0 - new_masks[idxSampMat], 0.9, 0.)
            mask_t = mask_t[:, self.kw['dT']:, :, :]

            if self.kw['flagUseOI'] == True:
                inputs_init    = torch.cat((targets_OI, mask_t * (targets_GT - targets_OI)), dim=1)
                inputs_missing = torch.cat((targets_OI, mask_t * (targets_GT - targets_OI)), dim=1)

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT, phase)

        # need to evaluate grad/bacself.kward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, new_masks[idxSampMat])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
            if phase == 'test':
                shp_data = self.kw['shapeData_test']
            else:
                shp_data = self.kw['shapeData']

            if self.kw['flagMultiScale'] == True:
                outputsSLRHR = outputs
                outputsSLR   = outputs[:, 0:self.kw['dT'], :, :].view(-1, self.kw['dT'], shp_data[1], shp_data[2])
                outputs = outputsSLR + outputs[:, self.kw['dT']:, :, :].view(-1, self.kw['dT'], shp_data[1], shp_data[2])

            # losses
            g_outputs  = self.gradient_img(outputs, phase)
            loss_All   = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.wLoss)
            loss_GAll  = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.wLoss)

            loss_OI    = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.wLoss)
            loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI, phase) - g_targets_GT, self.wLoss)

            loss_AE    = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT        = torch.cat((targets_GT, outputsSLR - targets_GT), dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            ## L1 vs. L0 cost for the sampling operator
            if self.kw['flagMultiScale'] == True:
                loss_Sampling = torch.mean(new_masks[idxSampMat][:, self.kw['dT']:, :, :])
            #TODO: else ??
            loss_Sampling = torch.nn.functional.relu(loss_Sampling - self.kw['thr_L1Sampling'])

            # training loss
            loss = self.kw['alpha_Grad'] * (self.kw['betaX'] * loss_All + self.kw['betagX'] * loss_GAll) \
                + 0.5 * self.kw['alpha_AE'] * (loss_AE + loss_AE_GT)
            loss += self.kw['alpha_L1Sampling'] * loss_Sampling

            if self.kw['flagMultiScale'] == True:
                loss_SR = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.wLoss)
                loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.wLoss)
                loss += self.kw['alpha_LR'] * loss_LR + self.kw['alpha_SR'] * loss_SR

        return loss, outputs


class Encoder(torch.nn.Module):
    def __init__(self, dW, sS, shapeData, DimAE):
        super(Encoder, self).__init__()
        self.pool1    = torch.nn.AvgPool2d(sS)
        self.conv1    = torch.nn.Conv2d(shapeData[0], 2 * DimAE, (2 * dW + 1, 2 * dW + 1), padding=dW, bias=False)
        self.conv2    = torch.nn.Conv2d(2 * DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.conv21   = torch.nn.Conv2d(DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.conv22   = torch.nn.Conv2d(DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.conv23   = torch.nn.Conv2d(DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.conv3    = torch.nn.Conv2d(2 * DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.conv2Tr  = torch.nn.ConvTranspose2d(DimAE, shapeData[0], (sS, sS), stride=(sS, sS), bias=False)
        self.convHR1  = torch.nn.Conv2d(shapeData[0], 2 * DimAE, (2 * dW + 1, 2 * dW + 1), padding=dW, bias=False)
        self.convHR2  = torch.nn.Conv2d(2 * DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.convHR21 = torch.nn.Conv2d(DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.convHR22 = torch.nn.Conv2d(DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.convHR23 = torch.nn.Conv2d(DimAE, DimAE, (1, 1), padding=0, bias=False)
        self.convHR3  = torch.nn.Conv2d(2 * DimAE, shapeData[0], (1, 1), padding=0, bias=False)

    def forward(self, xinp):
        x   = self.pool1(xinp)
        x   = self.conv1(x)
        x   = self.conv2(F.relu(x))
        x   = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)), dim=1)
        x   = self.conv3(x)
        x   = self.conv2Tr(x)
        xHR = self.convHR1(xinp)
        xHR = self.convHR2(F.relu(xHR))
        xHR = torch.cat((self.convHR21(xHR), self.convHR22(xHR) * self.convHR23(xHR)), dim=1)
        xHR = self.convHR3(xHR)
        x = torch.add(x, xHR, alpha=1.)
        return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return torch.mul(1., x)


class Phi_r(torch.nn.Module):
    def __init__(self, dW, sS, shapeData, DimAE):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(dW, sS, shapeData, DimAE)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Model_H(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_H, self).__init__()
        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData[0]])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout


class ModelLR(torch.nn.Module):
        def __init__(self):
            super(ModelLR, self).__init__()
            self.pool = torch.nn.AvgPool2d((16, 16))

        def forward(self, im):
            return self.pool(im)


class Model_Sampling(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_Sampling, self).__init__()
        self.DimObs = 1
        self.W = torch.nn.Parameter(torch.Tensor(np.zeros((1, shapeData[0], shapeData[1], shapeData[2]))))

    def forward(self, y, phase):
        yout1 = 0. * y
        yout2 = 1. * yout1
        return [yout1, yout2]