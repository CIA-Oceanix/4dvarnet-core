from models import *
from lit_model import LitModel

class LitModelStochastic(LitModel):
 
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        self.log('val_loss', np.nanmean([loss[i].detach().cpu() for i in range(len(loss))]))
        self.log("val_mse", np.nanmean([ metrics[i]['mse'].detach().cpu()/self.var_Tt for i in range(len(metrics)) ]),
                on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", np.nanmean([ metrics[i]['mseGrad'].detach().cpu()/metrics[i]['meanGrad'].detach().cpu()  for i in range(len(metrics)) ]),
                on_step=False, on_epoch=True, prog_bar=True)

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
 
        # loss is the minimal loss aong the members
        min_ = torch.min(torch.stack(loss))
        # maintain variance of the ensemble
        print(out.size)
        var = torch.var(out)
        print(var)

        return min_ + 1./var

    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        self.log('val_loss', np.nanmean([loss[i].detach().cpu() for i in range(len(loss))]))
        self.log("val_mse", np.nanmean([ metrics[i]['mse'].detach().cpu()/self.var_Tt for i in range(len(metrics)) ]),
            on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", np.nanmean([ metrics[i]['mseGrad'].detach().cpu()/metrics[i]['meanGrad'].detach().cpu()  for i in range(len(metrics)) ]),
            on_step=False, on_epoch=True, prog_bar=True)

        return torch.min(torch.stack(loss)).detach()


    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, input_obs, targets_GT = test_batch

        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', np.nanmean([loss[i].detach().cpu() for i in range(len(loss))]))
        self.log("test_mse", np.nanmean([ metrics[i]['mse'].detach().cpu()/self.var_Tt for i in range(len(metrics)) ]),
                     on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mseG", np.nanmean([ metrics[i]['mseGrad'].detach().cpu()/metrics[i]['meanGrad'].detach().cpu()  for i in range(len(metrics)) ]),
                     on_step=False, on_epoch=True, prog_bar=True)

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
                '(t_idx lat_idx lon_idx ens_idx) win_time win_lat win_lon win_ens -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (ens_idx win_ens)',
                ens_idx=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )

        self.x_gt = gt[:,int(self.hparams.dT/2),:,:]
        self.x_obs = obs[:, int(self.hparams.dT / 2), :, :]
        self.x_oi = oi[:,int(self.hparams.dT/2),:,:]

        # temporary fix
        time = np.arange(len(gt))

        # display ensemble
        path_save0 = self.logger.log_dir+'/maps_ensemble.png'
        plot_ensemble(pred[0,int(self.hparams.dT/2),:,:,:],
                          self.lon,self.lat,path_save0)
        self.x_rec = np.nanmean(pred[:,int(self.hparams.dT/2),:,:,:],axis=-1)
        pred = np.nanmean(pred,axis=-1)

        # display map
        path_save0 = self.logger.log_dir+'/maps.png'
        plot_maps(self.x_gt[0],
                  self.x_obs[0],
                  self.x_oi[0],
                  self.x_rec[0],
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

        targets_OI, inputs_Mask, input_obs, targets_GT = batch
        loss = []
        out = []
        metrics = []

        for i in range(self.hparams.size_ensemble):

            loss_, outputs_, metrics_ = super().compute_loss(batch, phase)
            if loss_ is not None:
                loss.append(loss_)
                metrics.append(metrics_)
            out.append(outputs_)

        outputs = torch.stack([out_.detach().cpu() for out_ in out],dim=-1)

        return loss, outputs, metrics

