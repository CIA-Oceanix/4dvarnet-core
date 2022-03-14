from models import *
import xarray as xr
from lit_model import LitModel
import solver_4DEnVar as NN_4DVar

class LitModel4DEnVar(LitModel):

    def __init__(self, hparam, *args, **kwargs):
        LitModel.__init__(self, hparam, *args, **kwargs)
        self.Ns = self.hparams.size_ensemble

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks, self.hparams.dropout_phi_r, self.hparams.stochastic),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.Ns,
                                          self.hparams.dim_grad_solver,
                                          self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_fourdenvar, self.hparams.n_grad, self.Ns)

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics    
        loss, out, members, weights, metrics = self.compute_loss(train_batch, phase='train')
        if loss is None:
            return loss
        # log step metric        
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        # initial grad value
        if self.hparams.automatic_optimization == False:
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
        loss, out, members, weights, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, inputs_obs, targets_GT = test_batch
        loss, out, ens, w, metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs'   : (inputs_obs.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'members' : (ens.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX),:]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'weights' : w.detach().cpu()
             }

    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        obs = torch.cat([chunk['obs'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        members = torch.cat([chunk['members'] for chunk in outputs]).numpy()
        weights = torch.cat([chunk['weights'] for chunk in outputs]).numpy()

        ds_size = {'time': self.ds_size_time,
                   'lon': self.ds_size_lon,
                   'lat': self.ds_size_lat,
                   }

        gt, obs, oi, pred = map(
            lambda t: einops.rearrange(
                t,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
            ),
            [gt, obs, oi, pred])

        members = einops.rearrange(
                members,
                '(t_idx lat_idx lon_idx ens_idx) win_time win_lat win_lon win_ens -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (ens_idx win_ens)',
                ens_idx=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'])

        self.x_gt = gt[:,int(self.hparams.dT/2),:,:]
        self.x_obs = obs[:, int(self.hparams.dT / 2), :, :]
        self.x_oi = oi[:,int(self.hparams.dT/2),:,:]
        self.x_rec = pred[:,int(self.hparams.dT/2),:,:]

        # display ensemble
        path_save0 = self.logger.log_dir+'/maps_ensemble.png'
        plot_ensemble(members[0,int(self.hparams.dT/2),:,:,:],
                          self.lon,self.lat,path_save0)
        # display map
        path_save1 = self.logger.log_dir+'/maps.png'
        plot_maps(self.x_gt[0],
                  self.x_obs[0],
                  self.x_oi[0],
                  self.x_rec[0],
                  self.lon,self.lat,path_save1)
        # compute nRMSE
        path_save2 = self.logger.log_dir+'/nRMSE.txt'
        tab_scores = nrmse_scores(self.x_gt,self.x_oi,self.x_rec,path_save2)
        print('*** Display nRMSE scores ***')
        print(tab_scores)
        # plot nRMSE
        path_save3 = self.logger.log_dir+'/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,self.x_oi,self.x_rec,path_save3,time=self.time['time_test'])
        self.logger.experiment.add_figure('NRMSE', nrmse_fig, global_step=self.current_epoch)
        # plot SNR
        path_save4 = self.logger.log_dir+'/SNR.png'
        snr_fig = plot_snr(self.x_gt,self.x_oi,self.x_rec,path_save4)
        self.logger.experiment.add_figure('SNR', snr_fig, global_step=self.current_epoch)
        # save NetCDF
        path_save1 = self.logger.log_dir+'/maps.nc'
        save_netcdf(saved_path1 = path_save1, gt=gt, oi=oi, pred=pred,
            lon = self.lon,lat = self.lat, time=self.time['time_test'])
        # maps score
        if self.hparams.supervised==True:
            path_save = self.logger.log_dir + '/maps_score.png'
            maps_score(path_save,xr.open_dataset(path_save1),lon=self.lon, lat=self.lat)

    def compute_loss(self, batch, phase):

        targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_GT),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.),
                      ('mseOI', 0.),('mseGOI', 0.)])
            )
        new_masks = torch.cat((1. + 0. * inputs_Mask, inputs_Mask), dim=1)
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        targets_Mask = torch.where(targets_GT!=0.)
        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)

        # need to initialize with size of the ensemble
        inputs_init = torch.cat((targets_OI, inputs_obs), dim=1)
        inputs_missing = torch.cat((targets_OI, inputs_obs), dim=1)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
            weights, ensemble, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, new_masks)

            # ensemble mean
            xbar = torch.mean(ensemble,dim=4,keepdim=True)
            # anomalies
            anom = torch.mul(ensemble-xbar, np.sqrt(1./self.Ns))
            # weighting of the anomalies
            anom = anom * weights.reshape(inputs_init.size()[0], 1, 1, 1, -1)
            # new optimal state
            outputs = torch.squeeze(xbar,dim=4)+torch.sum(anom,dim=4)

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR = outputs[:, 0:self.hparams.dT, :, :]
            outputs = outputsSLR + outputs[:, self.hparams.dT:, :, :]

            ensembleSLRHR = ensemble
            ensembleSLR = ensemble[:, 0:self.hparams.dT, :, :, :]
            ensemble = ensembleSLR + ensemble[:, self.hparams.dT:, :, :, :]

            # ensemble mean
            xbar = torch.mean(ensemble,dim=4,keepdim=False)
            xbarSLR = torch.mean(ensembleSLR,dim=4,keepdim=False)
            xbarSLRHR = torch.mean(ensembleSLRHR,dim=4,keepdim=False)
            # anomalies
            anom = torch.clone(ensemble)
            for i in range(self.hparams.size_ensemble):
                anom[:,:,:,:,i] = anom[:,:,:,:,i] - xbar
            anom = torch.square(anom)
            # variance: batch*time*lat*lon
            std = 1 + torch.std(torch.sum(torch.square(anom),dim=4)/self.hparams.size_ensemble)

            # log-likelihood
            loss_log = torch.sum( (xbar - targets_GT)**2/2*torch.square(std) \
                                  + torch.log(std) )
            loss_log = torch.mul(1.0/torch.numel(xbar),loss_log)

            # MSE
            loss_MSE = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.w_loss)
            g_xbar = self.gradient_img(xbar)
            g_targets_GT = self.gradient_img(targets_GT)
            loss_GMSE = NN_4DVar.compute_WeightedLoss(g_xbar - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = torch.mean((self.model.phi_r(xbarSLRHR) - xbarSLRHR) ** 2)
            yGT = torch.cat((targets_GT_wo_nan, xbarSLR - targets_GT_wo_nan), dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_WeightedLoss(xbarSLR - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(xbar) - targets_GTLR, self.w_loss)

            # supervised loss
            if self.hparams.supervised==True:
                loss = loss_log + .5*loss_MSE

            # unsupervised loss
            else:
                # MSE
                mask = (targets_GT_wo_nan!=0.)
                iT = int(self.hparams.dT / 2)
                std = torch.sqrt(torch.sum(torch.square(anom),dim=4))
                mean = torch.squeeze(xbar)
                loss = torch.sum(torch.log(torch.masked_select(std[:,iT,:,:],mask[:,iT,:,:])) + 
                                           (torch.masked_select((mean-targets_GT_wo_nan)[:,iT,:,:],mask[:,iT,:,:])/(2*torch.square(torch.masked_select(std[:,iT,:,:],mask[:,iT,:,:])))))
                loss += 0.5 * self.hparams.alpha_proj * loss_AE 
                loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR

            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT, self.w_loss)
            mse = loss_MSE.detach()
            mseGrad = loss_GMSE.detach()
            metrics = dict([('mse', mse), ('mseGrad', mseGrad),
                        ('meanGrad', mean_GAll)])

        return loss, outputs, ensemble, weights, metrics

