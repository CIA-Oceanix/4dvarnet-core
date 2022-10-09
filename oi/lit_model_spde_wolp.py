from oi.models_spde import *
from math import sqrt
from oi.models_spde import sparse_eye
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

        self.diff_only = self.hparams.diff_only

        # use OI or for metrics or not
        self.use_oi = self.hparams.oi if hasattr(self.hparams, 'oi') else False

        self.estim_parameters = self.hparams.estim_parameters
        self.spde_params_path = hparam.files_cfg.spde_params_path
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(self.shapeData,pow=hparam.pow,diff_only=self.diff_only,
                      given_parameters=not self.estim_parameters,
                      nc=xr.open_dataset(hparam.files_cfg.spde_params_path)),
                Model_H(self.shapeData[0]),
                NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
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

    def run_simulation(self,i,M,x,dx,dy,dt,n_init_run=10):
        nb_nodes = M.shape[0]
        tau = 1.
        regul = (tau*sqrt(dt))/(dx*dy)
        random = torch.randn(nb_nodes).to(device)
        val = cupy.fromDlpack(to_dlpack(M.coalesce().values().data))
        ind = cupy.fromDlpack(to_dlpack(M.coalesce().indices().data))
        M_ = cupy_sp_csc_matrix((val,ind),shape=(M.size()[0],M.size()[0]))
        # if i==0: start stability run
        if i==0:
            xi = torch.randn(nb_nodes).to(device)
            for i in range(n_init_run):
                RM = torch.mul(regul,random)+torch.flatten(xi)
                RM_ = cupy.fromDlpack(to_dlpack(RM))
                xi_ = cupy_spsolve(M_, RM_)
                xi = torch.flatten(from_dlpack(xi_.toDlpack())).to(device)
                #xi = torch.flatten(cupy_solve_sparse.apply(M,RM)).to(device)     
        else:
            RM = torch.mul(regul,random)+torch.flatten(x[i-1])
            RM_ = cupy.fromDlpack(to_dlpack(RM))
            xi_ = cupy_spsolve(M_, RM_)
            xi = torch.flatten(from_dlpack(xi_.toDlpack())).to(device)
            #xi = torch.flatten(cupy_solve_sparse.apply(M,RM)).to(device)
        xi.requires_grad = True
        xi = torch.flatten(xi)
        x.append(xi)
        return x

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
        loss, out, param, metrics = self.compute_loss(train_batch, phase='train')
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
        loss, out, param, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def test_step(self, test_batch, batch_idx):

        if not self.use_oi:
            inputs_obs, inputs_mask, targets_gt = test_batch[:3]
            targets_oi = None
        else:
            inputs_obs, inputs_mask, targets_gt, targets_oi = test_batch        

        loss, out, param, metrics, simu, cmp_loss = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_oi_score", metrics['oiScore'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_gt.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'obs'   : (inputs_obs.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'simulations' : (simu.detach().cpu()[:,:,(self.dY):(self.swY-self.dY),(self.dX):(self.swX-self.dX)]*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'params': param, 'cmp_loss': cmp_loss}

    def test_epoch_end(self, outputs):

        ds_size = {'time': self.ds_size_time,
                   'lon': self.ds_size_lon,
                   'lat': self.ds_size_lat,
                   }

        def reshape(np_arr):
            np_arr = einops.rearrange(np_arr,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
            return np_arr

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        obs = torch.cat([chunk['obs'] for chunk in outputs]).numpy()
        obs = np.where(obs==self.mean_Tr,np.nan,obs)
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        simu = torch.cat([chunk['simulations'] for chunk in outputs]).numpy()
        
        if self.diff_only==False:
            kappa = torch.cat([(chunk['params'])[0].detach().cpu() for chunk in outputs]).numpy()
            m = torch.cat([(chunk['params'])[1].detach().cpu() for chunk in outputs]).numpy()
            H = torch.cat([(chunk['params'])[2].detach().cpu() for chunk in outputs]).numpy()           
        else:
            kappa = 1./3
            m = None
            H = torch.cat([(chunk['params'])[2].detach().cpu() for chunk in outputs]).numpy()
        param = [kappa, m, H]
        
        cmp_loss = torch.cat([(chunk['cmp_loss']).detach().cpu() for chunk in outputs]).numpy()
        # save lossOI and lossMSE as netcdf
        path_save = self.logger.log_dir + '/loss.nc'
        save_loss(path_save,cmp_loss)

        gt = np.moveaxis(reshape(gt),1,3)
        obs = np.moveaxis(reshape(obs),1,3)
        pred = np.moveaxis(reshape(pred),1,3)
        simu = np.moveaxis(reshape(simu),1,3)

        # keep only points of the original domain
        iX = np.where( (self.lon_ext>=self.xmin) & (self.lon_ext<=self.xmax) )[0]
        iY = np.where( (self.lat_ext>=self.ymin) & (self.lat_ext<=self.ymax) )[0]
        gt = (gt[:,iY,:,:])[:,:,iX,:]
        obs = (obs[:,iY,:,:])[:,:,iX,:]
        pred = (pred[:,iY,:,:])[:,:,iX,:]

        self.x_gt = gt[:,:,:,int(self.hparams.dT / 2)]
        self.x_obs = obs[:,:,:,int(self.hparams.dT / 2)]
        self.x_rec = pred[:,:,:,int(self.hparams.dT / 2)]
        self.x_simu = simu[:,:,:,int(self.hparams.dT / 2)]

        # display map
        path_save0 = self.logger.log_dir + '/maps.png'
        fig_maps = plot_maps(
                  self.x_gt[0],
                  self.x_obs[0],
                  self.x_rec[0],
                  self.lon, self.lat, path_save0,
                  cartopy=False,
                  supervised=self.hparams.supervised)
        path_save0 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps(
                  self.x_gt[0],
                  self.x_obs[0],
                  self.x_rec[0],
                  self.lon, self.lat, path_save0,
                  cartopy=False,
                  grad=True, supervised=self.hparams.supervised)

        # compute simulation
        path_save0 = self.logger.log_dir + '/simu.png'
        fig_maps = plot_maps(
                  self.x_gt[0],
                  self.x_obs[0],
                  self.x_simu[0],
                  self.lon, self.lat, path_save0,
                  cartopy=False,
                  supervised=self.hparams.supervised)

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
        # reshape parameters
        if self.diff_only==False:
            kappa, m, H  = param
            kappa = np.transpose(kappa,(0,4,3,2,1))
            kappa = einops.rearrange(kappa,
                '(t_idx lat_idx lon_idx dim1) win_time win_lat win_lon win_dim1 -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (dim1 win_dim1)',
                dim1=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
            kappa = np.transpose(kappa,(0,4,2,3,1))
            # m
            m = np.transpose(m,(0,4,3,2,1))
            m = einops.rearrange(m,
                '(t_idx lat_idx lon_idx dim1) win_time win_lat win_lon win_dim1 -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (dim1 win_dim1)',
                dim1=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
            m = np.transpose(m,(0,4,2,3,1))
            # H
            H = np.transpose(H,(0,5,4,3,1,2))
            H = einops.rearrange(H,
                '(t_idx lat_idx lon_idx dim1 dim2) win_time win_lat win_lon win_dim1 win_dim2-> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (dim1 win_dim1) (dim2 win_dim2)',
                dim1=1,
                dim2=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
            H = np.transpose(H,(0,4,5,2,3,1))
            # save NetCDF
            path_save1 = self.logger.log_dir + '/maps.nc'
            save_netcdf(gt,obs,pred,pred,pred,pred,[kappa, m, H],
                    self.lon,self.lat,path_save1,
                    time=self.time['time_test'],
                    time_units='days since 2012-10-01 00:00:00')
        else:
            kappa, m, H  = param
            # reshape parameters H
            H = np.transpose(H,(0,4,3,1,2))
            H = np.expand_dims(H,axis=1)
            H = einops.rearrange(H,
                '(t_idx lat_idx lon_idx dim1 dim2) win_time win_lat win_lon win_dim1 win_dim2-> t_idx win_time (lat_idx win_lat) (lon_idx win_lon) (dim1 win_dim1) (dim2 win_dim2)',
                dim1=1,
                dim2=1,
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
                )
            H = np.squeeze(H)
            H = np.transpose(H,(0,3,4,1,2))
            # save NetCDF
            path_save1 = self.logger.log_dir + '/maps.nc'
            save_netcdf(gt,obs,pred,pred,pred,pred,[H],
                    self.lon,self.lat,path_save1,
                    time=self.time['time_test'],
                    time_units='days since 2012-10-01 00:00:00')

    def compute_loss(self, batch, phase):

        if not self.use_oi:
            inputs_obs, inputs_mask, targets_gt = batch[:3]
            targets_oi = None
        else:
            inputs_obs, inputs_mask, targets_gt, targets_oi = batch


        # handle patch with no observation
        if inputs_mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_gt),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.)])
            )
        new_mask = torch.cat((1. + 0. * inputs_mask, inputs_mask), dim=1)
        targets_gt_wo_nan = targets_gt.where(~targets_gt.isnan(), torch.zeros_like(targets_gt))
        targets_mask = torch.where(targets_gt!=0.)
        inputs_init = inputs_obs
        inputs_missing = inputs_obs

        # gradient norm field
        g_targets_gt = self.gradient_img(targets_gt)
        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
            outputs, cmp_loss, hidden_new, cell_new, normgrad, params = self.model(targets_gt,
                                                                         inputs_init,
                                                                         inputs_missing,
                                                                         inputs_mask, targets_oi,
                                                                         estim_parameters=False)
            #outputs, params = self.model(inputs_init, inputs_missing, inputs_mask,.33,None,None)
            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            n_b = outputs.shape[0]
            n_x = outputs.shape[3]
            n_y = outputs.shape[2]
            nb_nodes = n_x*n_y
            n_t = outputs.shape[1]
            dx = 1
            dy = 1
            dt = 1
            if self.diff_only==True:
                H = torch.reshape(params[2],(len(outputs),2,2,n_x*n_y))
                m = None
                kappa = 1./3
                tau = 1.
            else:
                kappa = torch.reshape(params[0],(len(outputs),1,n_x*n_y,n_t))
                m = torch.reshape(params[1],(len(outputs),2,n_x*n_y,n_t))
                H = torch.reshape(params[2],(len(outputs),2,2,n_x*n_y,n_t))
                tau = torch.reshape(params[3],(len(outputs),1,n_x*n_y,n_t))

            # run n_batches*simulation
            I = sparse_eye(n_x*n_y)

            # PHASE: TEST -> conditional simulation
            if phase=="test":
                # run n_batches*simulation
                I = sparse_eye(n_x*n_y)
                x_simu = []
                for ibatch in range(len(outputs)):
                    x_simu_ = []
                    for i in range(n_t):
                        if self.diff_only==True:
                            A = DiffOperator(n_x,n_y,dx,dy,None,H[ibatch],kappa)
                        else:
                            A = DiffOperator(n_x,n_y,dx,dy,m[ibatch,:,:,i],
                                                       H[ibatch,:,:,:,i],
                                                       kappa[ibatch,:,:,i])
                        M = I+pow_diff_operator(A,1)
                        x_simu_ = self.run_simulation(i,M,x_simu_,dx,dy,dt,n_init_run=10)
                    x_simu_ = torch.stack(x_simu_,dim=0)
                    x_simu_ = torch.reshape(x_simu_,outputs.shape[1:])
                    # x,y -> y,x
                    x_simu_ = torch.permute(x_simu_,(0,2,1))
                    x_simu.append(x_simu_)
                x_simu = torch.stack(x_simu,dim=0).to(device)
                # interpolate the simulation based on LSTM-solver
                idx = torch.where(inputs_mask==0.)
                inputs_init_simu = x_simu.clone()
                inputs_init_simu[idx] = 0.
                inputs_missing_simu = inputs_init_simu
                x_itrp_simu,_ ,_ ,_ ,_, _ = self.model(targets_gt, inputs_init_simu,
                                                    inputs_missing_simu, inputs_mask,
                                                    targets_oi,
                                                    estim_parameters=False)
                # conditional simulation
                x_simu_cond = outputs+(x_simu-x_itrp_simu)

            # reconstruction losses
            g_outputs = self.gradient_img(outputs)
            loss_All = NN_4DVar.compute_WeightedLoss((outputs - targets_gt), self.w_loss)
            loss_GAll = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_gt, self.w_loss)
            
            # supervised loss
            loss = self.hparams.alpha_mse_ssh * loss_All #+ self.hparams.alpha_mse_gssh * loss_GAll

            # supervised loss (vs OI)
            if self.use_oi:
                loss_mse_vs_oi = NN_4DVar.compute_WeightedLoss((outputs - targets_oi), self.w_loss)
                loss_mseoi = self.hparams.alpha_mse_ssh * loss_mse_vs_oi

            # OI loss
            Q = self.model.phi_r.operator_spde(kappa, m, H, tau, square_root=False)
            Q.requires_grad=True
            dy = self.model.model_H(outputs,inputs_missing,inputs_mask)
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
            loss_OI = torch.nanmean(dy+dx)
            #dy = self.model.model_H(outputs,inputs_missing,inputs_mask)
            #loss_OI = torch.nanmean(self.model_spde.model_VarCost(dx,dy,square_root=False))

            if self.loss_type=="mse_loss":
                loss = loss
            if self.loss_type=="mseoi_loss":
                loss = loss_mseoi
            if self.loss_type=="oi_loss":
                loss = loss_OI

            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_gt, self.w_loss)
            mse = loss_All.detach()
            mse_grad = loss_GAll.detach()
            oi_score = loss_OI.detach()
            metrics = dict([('mse', mse), ('mseGrad', mse_grad),
                            ('meanGrad', mean_GAll),
                            ('oiScore',oi_score)])
            if self.use_oi:
                mse_oi = loss_mseoi.detach()
                metrics['mseoi'] = mse_oi

        if phase!="test":
            return loss, outputs, params, metrics
        else:
            return loss, outputs, params, metrics, x_simu_cond, cmp_loss
