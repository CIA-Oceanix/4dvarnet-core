from models import *

class LitModelStochastic(LitModel):
        
    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, targets_GT = test_batch
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

    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
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
        pred = einops.rearrange(pred,
                '(t_idx lat_idx lon_idx ens_idx) win_time win_lat win_lon win_ens -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (ens_idx win_ens)',
                ens_idx=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )

        self.x_gt = gt[:,int(self.hparams.dT/2),:,:]
        self.x_oi = oi[:,int(self.hparams.dT/2),:,:]

        # display ensemble
        path_save0 = self.logger.log_dir+'/maps_ensemble.png'
        plot_ensemble(pred[0,int(self.hparams.dT/2),:,:,:],
                          self.lon,self.lat,path_save0)
        self.x_rec = np.nanmean(pred[:,int(self.hparams.dT/2),:,:,:],axis=-1)
        pred = np.nanmean(pred,axis=-1)

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

