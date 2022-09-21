from oi.models import *
from oi.models_spde import Phi_r as Phi_r_spde
from oi.models_spde import sparse_eye
import oi.solver_spde as NN_4DVar_spde
from oi.scipy_sparse_tools import *
import xarray as xr

class LitModel(pl.LightningModule):
    def __init__(self, hparam, *args, **kwargs):
        super().__init__()
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)
        self.save_hyperparameters(hparams)
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']

        # create longitudes & latitudes coordinates
        self.xmin = kwargs['min_lon']
        self.xmax = kwargs['max_lon']
        self.ymin = kwargs['min_lat']
        self.ymax = kwargs['max_lat']
        self.resolution = kwargs['resolution']
        self.Nx = int(((self.xmax-self.xmin)/self.resolution)/self.hparams.resize_factor)
        self.Ny = int(((self.ymax-self.ymin)/self.resolution)/self.hparams.resize_factor)
        self.lon = np.linspace(self.xmin, self.xmax, self.Nx)
        self.lat = np.linspace(self.ymin, self.ymax, self.Ny)
        self.shapeData = [self.hparams.dT,self.Ny,self.Nx]
        self.ds_size_time = kwargs['ds_size_time']
        self.ds_size_lon = kwargs['ds_size_lon']
        self.ds_size_lat = kwargs['ds_size_lat']

        self.time = kwargs['time'] 
        self.dX = kwargs['dX']
        self.dY = kwargs['dY']
        self.swX = kwargs['swX']
        self.swY = kwargs['swY']
        self.lon_ext = (kwargs['coord_ext'])['lon_ext']
        self.lat_ext = (kwargs['coord_ext'])['lat_ext']

        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.mean_Val = kwargs['mean_Val']
        self.mean_Tr = kwargs['mean_Tr']
        self.mean_Tt = kwargs['mean_Tt']

        self.original_coords = kwargs['original_coords']
        self.padded_coords = kwargs['padded_coords']

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks, self.hparams.dropout_phi_r, self.hparams.stochastic),
            #Phi_r_UNet(self.shapeData[0], self.hparams.dropout_phi_r, self.hparams.stochastic,shrink_factor=1),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout, self.hparams.stochastic),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)

        # SPDE-based model
        self.model_spde = NN_4DVar_spde.Solver_Grad_4DVarNN(
                Phi_r_spde(self.shapeData,pow=hparam.pow,diff_only=True,
                      given_parameters=True,
                      nc=xr.open_dataset(hparam.files_cfg.spde_params_path)),
                Model_H(self.shapeData[0]),
                NN_4DVar_spde.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout, self.hparams.stochastic),
                None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)

        self.model_LR = ModelLR()
        self.gradient_img = Gradient_img()
        # loss weghing wrt time
        self.w_loss = torch.nn.Parameter(kwargs['w_loss'], requires_grad=False)  # duplicate for automatic upload to gpu
        self.loss_type = self.hparams.loss_type

        self.x_gt = None  # variable to store Ground Truth
        self.x_rec = None  # variable to store output of test method
        self.test_figs = {}

        self.automatic_optimization = self.hparams.automatic_optimization

    def forward(self):
        return 1

    def configure_optimizers(self):

        optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0.)

        return optimizer

    def on_epoch_start(self):
        # enfore acnd check some hyperparameters
        self.model.n_grad = self.hparams.n_grad

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
            lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                mm += 1

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        if loss is None:
            return loss
        # log step metric        
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        self.log("tr_mse", metrics['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=False)

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
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def test_step(self, test_batch, batch_idx):

        inputs_obs, inputs_mask, targets_gt = test_batch
        loss, out, metrics, cmp_loss = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)


        return {'gt'    : (targets_gt.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs'   : (inputs_obs.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'cmp_loss': cmp_loss}

    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs]).numpy()
        obs = torch.cat([chunk['obs'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs]).numpy()
        obs = np.where(obs==self.mean_Tr,np.nan,obs)
        pred = torch.cat([chunk['preds'][:, int(self.hparams.dT / 2), :, :] for chunk in outputs]).numpy()

        ds_size = {'time': self.ds_size_time,
                   'lon': self.ds_size_lon,
                   'lat': self.ds_size_lat,
                   }

        gt, obs, pred = map(
            lambda t: einops.rearrange(
                t,
                '(lat_idx lon_idx t_idx )  win_lat win_lon -> t_idx  (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
            ),
            [gt, obs, pred])

        cmp_loss = torch.cat([(chunk['cmp_loss']).detach().cpu() for chunk in outputs]).numpy()
        # save lossOI and lossMSE as netcdf
        path_save = self.logger.log_dir + '/loss.nc'
        save_loss(path_save,cmp_loss)

        # old way: keep only points of the original domain
        '''
        iX = np.where( (self.lon_ext>=self.xmin) & (self.lon_ext<=self.xmax) )[0]
        iY = np.where( (self.lat_ext>=self.ymin) & (self.lat_ext<=self.ymax) )[0]
        gt = (gt[:,iY,:])[:,:,iX]
        obs = (obs[:,iY,:])[:,:,iX]
        pred = (pred[:,iY,:])[:,:,iX]

        self.x_gt = gt
        self.x_obs = obs
        self.x_rec = pred
        '''

        # new way: construct output test xr dataset
        self.ds_test = xr.Dataset(
            data_vars={
                'gt': (('time', 'lat', 'lon'), gt),
                'pred': (('time', 'lat', 'lon'), pred),
                'obs': (('time', 'lat', 'lon'), obs),
            },
            coords={
                'time': self.time['time_test'],
                'lat': self.padded_coords['lat'],
                'lon': self.padded_coords['lon'],
            }
        ).sel(
            lat=slice(
                self.original_coords['lat'].values[0],
                self.original_coords['lat'].values[-1]
            ),
            lon=slice(
                self.original_coords['lon'].values[0],
                self.original_coords['lon'].values[-1]
            ),
        )

        # get underlying data array
        self.x_gt = self.ds_test.gt.values
        self.x_obs = self.ds_test.obs.values
        self.x_rec = self.ds_test.pred.values

        # display map
        path_save0 = self.logger.log_dir + '/maps.png'
        fig_maps = plot_maps(
                  self.x_gt[0],
                  self.x_obs[0],
                  self.x_rec[0],
                  self.ds_test.lon, self.ds_test.lat, path_save0,
                  cartopy=False,
                  supervised=self.hparams.supervised)
        path_save0 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps(
                  self.x_gt[0],
                  self.x_obs[0],
                  self.x_rec[0],
                  self.ds_test.lon, self.ds_test.lat, path_save0,
                  cartopy=False,
                  grad=True, supervised=self.hparams.supervised)
        # compute nRMSE
        path_save2 = self.logger.log_dir + '/nRMSE.txt'
        tab_scores = nrmse_scores(gt, pred, path_save2)
        path_save21 = self.logger.log_dir + '/MSE.txt'
        tab_scores = mse_scores(gt, pred, path_save21)
        # plot nRMSE
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt, self.x_rec, path_save3, time=self.time['time_test'])
        self.test_figs['nrmse'] = nrmse_fig
        # plot MSE
        path_save31 = self.logger.log_dir + '/MSE.png'
        mse_fig = plot_mse(self.x_gt, self.x_rec, path_save31, time=self.time['time_test'])
        self.test_figs['mse'] = mse_fig
        self.logger.experiment.add_figure('Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure('NRMSE', nrmse_fig, global_step=self.current_epoch)
        self.logger.experiment.add_figure('MSE', mse_fig, global_step=self.current_epoch)
        # plot SNR
        path_save4 = self.logger.log_dir + '/SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig
        self.logger.experiment.add_figure('SNR', snr_fig, global_step=self.current_epoch)
        # save NetCDF
        path_save1 = self.logger.log_dir + '/maps.nc'
        save_netcdf(saved_path1=path_save1, gt=self.x_gt, obs=self.x_obs, pred=self.x_rec,
                    lon=self.ds_test.lon, lat = self.ds_test.lat, time=self.time['time_test'],
                    time_units='days since 2012-10-01 00:00:00')

    def compute_loss(self, batch, phase):

        inputs_obs, inputs_mask, targets_gt = batch
        # handle patch with no observation
        if inputs_mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_gt),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.)])
            )
        targets_gt_wo_nan = targets_gt.where(~targets_gt.isnan(), torch.zeros_like(targets_gt))
        targets_mask = torch.where(targets_gt!=0.)
        inputs_init = inputs_obs
        inputs_missing = inputs_obs

        # gradient norm field
        g_targets_gt = self.gradient_img(targets_gt)
        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
            outputs, cmp_loss, hidden_new, cell_new, normgrad = self.model(targets_gt,inputs_init, inputs_missing, inputs_mask)
            # spde model to retrieve SPDE parameters and Q
            _, _, _, _, _, params = self.model_spde(targets_gt,inputs_init,inputs_missing,inputs_mask,estim_parameters=False)
            n_b = outputs.shape[0]
            n_x = outputs.shape[3]
            n_y = outputs.shape[2]
            nb_nodes = n_x*n_y
            n_t = outputs.shape[1]
            H = torch.reshape(params[2],(len(outputs),2,2,n_x*n_y))
            m = None
            kappa = 1./3
            tau = 1.

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            # reconstruction losses
            g_outputs = self.gradient_img(outputs)
            loss_All = NN_4DVar.compute_WeightedLoss((outputs - targets_gt), self.w_loss)
            #loss_All = torch.tensor([NN_4DVar.compute_WeightedLoss(outputs[i] - targets_gt[i],self.w_loss) for i in range(len(targets_gt))])
            #loss_All = torch.mean(loss_All)

            loss_GAll = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_gt, self.w_loss)
            # projection losses
            loss_AE = torch.mean((self.model.phi_r(outputs) - outputs) ** 2)
            #loss_AE = 0

            # supervised loss
            loss = self.hparams.alpha_mse_ssh * loss_All #+ self.hparams.alpha_mse_gssh * loss_GAll
            #loss += 0.5 * self.hparams.alpha_proj * loss_AE 

            # OI loss
            Q = self.model_spde.phi_r.operator_spde(kappa, m, H, tau, square_root=False)
            Q.requires_grad=True
            dy = self.model_spde.model_H(outputs,inputs_missing,inputs_mask)
            xtQx = list()
            dy_new=list()
            for i in range(n_b):
                # prior regularization
                xtQ = torch.sparse.mm(Q[i],
                                     torch.reshape(torch.permute(outputs[i],(0,2,1)),(n_t*n_x*n_y,1))
                                    )
                xtQx_ = torch.matmul(torch.reshape(torch.permute(outputs[i],(0,2,1)),(1,n_t*n_x*n_y)),
                                  xtQ
                                 )
                xtQx.append(xtQx_[0,0])
                # observation term
                id_obs = torch.where(torch.flatten(inputs_mask[i])!=0.)[0]
                dyi = torch.index_select(torch.flatten(dy[i]), 0, id_obs).type(torch.FloatTensor).to(device)
                nb_obs = len(dyi)
                inv_R = 1e3*sparse_eye(nb_obs).type(torch.FloatTensor).to(device)
                iRdy = torch.sparse.mm(inv_R,torch.reshape(dyi,(nb_obs,1)))
                dyTiRdy = torch.matmul(torch.reshape(dyi,(1,nb_obs)),iRdy)
                dy_new.append(dyTiRdy[0,0])
            dy = torch.stack(dy_new)
            dx = torch.stack(xtQx)
            loss_OI = torch.mean(dy+dx)
            #dy = self.model.model_H(outputs,inputs_missing,inputs_mask)
            #loss_OI = torch.mean(self.model_spde.model_VarCost(dx,dy,square_root=False))

            if self.loss_type=="mse_loss":
                loss = loss
            if self.loss_type=="oi_loss":
                loss = loss_OI

            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_gt, self.w_loss)
            mse = loss_All.detach()
            mse_grad = loss_GAll.detach()
            oi_score = loss_OI.detach()
            metrics = dict([('mse', mse), ('mseGrad', mse_grad), ('meanGrad', mean_GAll), ('oiScore',oi_score)])

        if phase!="test":
            return loss, outputs, metrics
        else:
            return loss, outputs, metrics, cmp_loss
