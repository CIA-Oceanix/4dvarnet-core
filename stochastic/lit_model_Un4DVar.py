from models import *
from lit_model import LitModel
import xarray as xr

class LitModelUn4DVar(LitModel):

    def __init__(self, hparam, *args, **kwargs):
        LitModel.__init__(self, hparam, *args, **kwargs)
        self.shapeData = [self.hparams.dT*2,self.Ny,self.Nx]

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks, self.hparams.dropout_phi_r, self.hparams.stochastic),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout, self.hparams.stochastic),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)


    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, inputs_obs, targets_GT = test_batch

        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs'   : (inputs_obs.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX),:]*np.sqrt(self.var_Tr)) + self.mean_Tr}

    def test_epoch_end(self, outputs):
        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        obs = torch.cat([chunk['obs'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        ds_size = {'time': self.ds_size_time,
                   'lon': self.ds_size_lon,
                   'lat': self.ds_size_lat,
                   }

        gt, obs, oi = map(
            lambda t: einops.rearrange(
                t,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
            ),
            [gt, obs, oi])

        pred = einops.rearrange(pred,
                '(t_idx lat_idx lon_idx nvar_idx) win_time win_lat win_lon win_nvar -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (nvar_idx win_nvar)',
                nvar_idx=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )

        self.x_gt = gt[:,int(self.hparams.dT/2),:,:]
        self.x_obs = obs[:, int(self.hparams.dT / 2), :, :]
        self.x_oi = oi[:,int(self.hparams.dT/2),:,:]

        self.x_rec = pred[:,int(self.hparams.dT/2),:,:,0]
        self.x_var_rec = pred[:,int(self.hparams.dT/2),:,:,1]
        pred = pred[:,:,:0]

        # display map (mean)
        path_save0 = self.logger.log_dir+'/maps.png'
        plot_maps(self.x_gt[0],
                  self.x_obs[0],
                  self.x_oi[0],
                  self.x_rec[0],
                  self.lon,self.lat,path_save0)

        # display map (var)
        path_save1 = self.logger.log_dir+'/maps_uc.png'
        plot_maps(self.x_gt[0],
                  self.x_obs[0],
                  self.x_oi[0],
                  self.x_var_rec[0],
                  self.lon,self.lat,path_save0)        

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
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                      ('mseGOI', 0.)])
            )
        new_masks = torch.cat((1. + 0. * inputs_Mask, inputs_Mask), dim=1)
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        targets_Mask = torch.where(targets_GT!=0.)
        inputs_init = torch.cat((targets_OI, inputs_obs), dim=1)
        inputs_missing = torch.cat((targets_OI, inputs_obs), dim=1)

        # generate outputs
        g_targets_GT = self.gradient_img(targets_GT)
        with torch.set_grad_enabled(True):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
            
            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, new_masks)

            print(outputs.size())
            
            SLR = outputs[:, 0:self.hparams.dT, :, :]
            mean = SLR + outputs[:, self.hparams.dT:(2*self.hparams.dT), :, :]

            SLR_cov = outputs[:, (2*self.hparams.dT):(3*self.hparams.dT), :, :]
            cov = SLR_cov + outputs[:, (3*self.hparams.dT):, :, :]

            outputs = torch.stack([mean,cov],dim=-1)
            outputs[:,:,:,:,1] = torch.abs(outputs[:,:,:,:,1])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                     
            loss_likelihood    = torch.sum(((outputs[:,:,:,:,0] - targets_GT)*outputs[:,:,:,:,1]*100)**2-2*torch.log(outputs[:,:,:,:,1]*100))/torch.size(new_masks)[0]
            loss_MSE = torch.mean((outputs[:,:,:,:,0] - targets_GT)**2)
            loss_GMSE = NN_4DVar.compute_WeightedLoss(self.gradient_img(outputs[:,:,:,:,0]) - g_targets_GT, self.w_loss)
            loss_All = torch.mean((outputs[:,:,:,:,0] - targets_GT)**2)
            loss_AE_mu     = torch.mean((self.model.phi_r(outputs)[:,:,:,:,0] - outputs[:,:,:,:,0])**2 )
            loss_AE_GT_mu  = torch.mean((self.model.phi_r(targets_GT) - targets_GT)**2 )
            loss_AE_cov     = torch.mean((self.model.phi_r(outputs)[:,:,:,:,1] - outputs[:,:,:,:,1])**2 )
            loss_AE_GT_cov = torch.mean((self.model.phi_r(new_target)[:,:,:,:,1] - new_target[:,:,:,:,1])**2 )
            
            # total loss
            loss        = 0.1 * self.hparams.alpha[0] * loss_likelihood + 0.5 * self.hparams.alpha[1] * ( loss_AE_mu + loss_AE_GT_mu )+0.5*loss_MSE+0.01*(loss_AE_cov+loss_AE_GT_cov)
            
            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT, self.w_loss)
            mse = loss_MSE.detach()
            mseGrad = loss_GMSE.detach()
            metrics = dict([('mse', mse), ('mseGrad', mseGrad),
                        ('meanGrad', mean_GAll)])

        return loss, outputs, metrics

