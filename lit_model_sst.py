from models import *
from lit_model import LitModel

class LitModelWithSST(LitModel):

    def __init__(self, hparam, *args, **kwargs):
        LitModel.__init__(self, hparam, *args, **kwargs)
        # adapt main model with SST
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks, self.hparams.dropout_phi_r),
            Model_HwithSST(self.shapeData[0], self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_grad)

    def configure_optimizers(self):

        optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0.)

        return optimizer

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad = self.hparams.n_grad

            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent, lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                mm += 1

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask,  inputs_obs, targets_GT, sst_GT = test_batch
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return {'gt'    : (targets_GT.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs'   : (inputs_obs.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr}

    def compute_loss(self, batch, phase):  ## to be updated

        targets_OI, inputs_Mask,  inputs_obs, targets_GT, sst_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_GT),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                      ('mseGOI', 0.)])
            )
        new_masks = torch.cat((1. + 0. * inputs_Mask, inputs_Mask), dim=1)
        mask_SST = 1. + 0. * sst_GT
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        inputs_init = torch.cat((targets_OI, inputs_obs), dim=1)
        inputs_missing = torch.cat((targets_OI, inputs_obs), dim=1)

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)
        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, [inputs_missing, sst_GT],
                                                                 [new_masks, mask_SST])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR = outputs[:, 0:self.hparams.dT, :, :]
            outputs = outputsSLR + outputs[:, self.hparams.dT:, :, :]

            # reconstruction losses
            g_outputs = self.gradient_img(outputs)
            loss_All = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.w_loss)
            loss_GAll = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.w_loss)

            loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT = torch.cat((targets_GT_wo_nan, outputsSLR - targets_GT_wo_nan), dim=1)
            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.w_loss)

            # supervised loss
            if self.hparams.supervised==True:
                loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
                loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR

            # unsupervised loss
            else:
                # MSE
                mask = (targets_GT_wo_nan!=0.)
                iT = int(self.hparams.dT / 2)
                new_tensor = torch.masked_select(outputs[:,iT,:,:],mask[:,iT,:,:]) - torch.masked_select(targets_GT[:,iT,:,:],mask[:,iT,:,:])
                loss = NN_4DVar.compute_WeightedLoss(new_tensor, torch.tensor(1.))
                # GradMSE
                mask = (self.gradient_img(targets_GT_wo_nan)!=0.)
                iT = int(self.hparams.dT / 2)
                new_tensor = torch.masked_select(self.gradient_img(outputs)[:,iT,:,:],mask[:,iT,:,:]) - torch.masked_select(self.gradient_img(targets_GT)[:,iT,:,:],mask[:,iT,:,:])
                loss_Grad = NN_4DVar.compute_WeightedLoss(new_tensor, torch.tensor(1.))

                #loss = self.hparams.alpha_mse_ssh * loss + self.hparams.alpha_mse_gssh * loss_Grad + 0.5 * self.hparams.alpha_proj * loss_AE + self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
                loss = self.hparams.alpha_mse_ssh * loss  + 0.5 * self.hparams.alpha_proj * loss_AE + self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR

            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT, self.w_loss)
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            metrics = dict([('mse', mse), ('mseGrad', mseGrad), ('meanGrad', mean_GAll), ('mseOI', loss_OI.detach()),
                            ('mseGOI', loss_GOI.detach())])

        return loss, outputs, metrics
