import einops
import torch.distributed as dist
import kornia
from hydra.utils import instantiate
import pandas as pd
from functools import reduce
from torch.nn.modules import loss
import xarray as xr
from pathlib import Path
from hydra.utils import call
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import solver as NN_4DVar
import metrics
from metrics import save_netcdf, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, get_psd_score
from models import Model_H, Model_HwithSST, Phi_r, ModelLR, Gradient_img



def get_4dvarnet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


def get_4dvarnet_sst(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                Model_HwithSST(hparams.shape_state[0], dT=hparams.dT),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


def get_ronan_4dvarnet(hparams):

    import solver_ronan
    import lit_model_ronan

    return solver_ronan.Solver_Grad_4DVarNN(
        lit_model_ronan.Phi_r(), 
        lit_model_ronan.Model_H(), 
        solver_ronan.model_GradUpdateLSTM(hparams.shape_data, hparams.UsePriodicBoundary,
            hparams.dim_grad_solver, hparams.dropout, padding_mode=hparams.padding_mode), 
        None, None, hparams.shape_data, hparams.n_grad, 0.*1e-20 ,k_step_grad = 1. / (hparams.n_grad * hparams.k_n_grad) )
    


def get_phi(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = Phi_r(hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2,
                    hparams.sS, hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic)
            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks):
            return self.phi(state), None, None, None

    return PhiPassThrough()


def get_constant_crop(patch_size, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_size[d] for d in dim_order], dtype='float32')
        print(patch_size, crop)
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        print(patch_weight.sum())
        return patch_weight



############################################ Lightning Module #######################################################################

class LitModelAugstate(pl.LightningModule):

    MODELS = {
            '4dvarnet': get_4dvarnet,
            'ronan_4dvarnet': get_ronan_4dvarnet,
            '4dvarnet_sst': get_4dvarnet_sst,
            'phi': get_phi,
             }

    def __init__(self,
                 hparam=None,
                 min_lon=None, max_lon=None,
                 min_lat=None, max_lat=None,
                 ds_size_time=None,
                 ds_size_lon=None,
                 ds_size_lat=None,
                 time=None,
                 dX = None, dY = None,
                 swX = None, swY = None,
                 coord_ext = None,
                 test_domain=None,
                 *args, **kwargs):
        super().__init__()
        hparam = {} if hparam is None else hparam
        hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)

        # self.save_hyperparameters({**hparams, **kwargs})
        self.save_hyperparameters({**hparams, **kwargs}, logger=False)
        self.latest_metrics = {}
        # TOTEST: set those parameters only if provided
        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt

        # create longitudes & latitudes coordinates
        self.test_domain=test_domain
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None

        self.patch_weight = torch.nn.Parameter(
                torch.from_numpy(call(self.hparams.patch_weight)), requires_grad=False)

        self.var_Val = self.hparams.var_Val
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt
        self.mean_Val = self.hparams.mean_Val
        self.mean_Tr = self.hparams.mean_Tr
        self.mean_Tt = self.hparams.mean_Tt

        # main model

        self.model_name = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        self.use_sst = self.hparams.sst if hasattr(self.hparams, 'sst') else False
        self.aug_state = self.hparams.aug_state if hasattr(self.hparams, 'aug_state') else False
        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.grad_crop = lambda t: t[...,1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
                self.grad_crop(2.*kornia.filters.spatial_gradient(t, normalized=True)), 2)
        # loss weghing wrt time

        # self._w_loss = torch.nn.Parameter(torch.Tensor(self.patch_weight), requires_grad=False)  # duplicate for automatic upload to gpu
        self.w_loss = torch.nn.Parameter(torch.Tensor([0,0,0,1,0,0,0]), requires_grad=False)  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.obs_inp = None
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.test_figs = {}

        self.tr_loss_hist = []
        self.automatic_optimization = self.hparams.automatic_optimization if hasattr(self.hparams, 'automatic_optimization') else False

        self.median_filter_width = self.hparams.median_filter_width if hasattr(self.hparams, 'median_filter_width') else 1

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out=None
        for _ in range(self.hparams.n_fourdvar_iter):
            _loss, out, state, _metrics = self.compute_loss(batch, phase=phase, state_init=state_init)
            state_init = [s.detach() for s in state]
            losses.append(_loss)
            metrics.append(_metrics)
        return losses, out, metrics

    def configure_optimizers(self):

        if self.model_name == '4dvarnet':
            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                ]
                , lr=0., weight_decay=self.hparams.weight_decay)

            return optimizer
        elif self.model_name == '4dvarnet_sst':

            optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                ], lr=0., weight_decay=self.hparams.weight_decay)

            return optimizer
        else:
            opt = optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': opt,
            'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=50,),
            'monitor': 'val_loss'
        }

    def on_epoch_start(self):
        self.model.n_grad = self.hparams.n_grad

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                    self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

                self.hparams.n_grad = self.hparams.nb_grad_update[indx]
                self.model.n_grad = self.hparams.n_grad
                print("ngrad iter", self.model.n_grad)
                mm = 0
                lrCurrent = self.hparams.lr_update[indx]
                lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
                for pg in opt.param_groups:
                    pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                    mm += 1

    def training_epoch_end(self, outputs):
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if len(best_ckpt_path) > 0:
            def should_reload_ckpt(losses):
                diffs = losses.diff()
                if losses.max() > (10 * losses.min()):
                    print("Reloading because of check", 1)
                    return True

                if diffs.max() > (100 * diffs.abs().median()):
                    print("Reloading because of check", 2)
                    return True

            if should_reload_ckpt(torch.stack([out['loss'] for out in outputs])):
                print('reloading', best_ckpt_path)
                ckpt = torch.load(best_ckpt_path)
                self.load_state_dict(ckpt['state_dict'])



    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics

        losses, _, metrics = self(train_batch, phase='train')
        if losses[-1] is None:
            print("None loss")
            return None
        # loss = torch.stack(losses).sum()
        loss = 2*torch.stack(losses).sum() - losses[0]

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        # log step metric
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True)
        # self.log("tr_min_nobs", train_batch[1].sum(dim=[1,2,3]).min().item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # self.log("tr_n_nobs", train_batch[1].sum().item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def diag_step(self, batch, batch_idx, log_pref='test'):
        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch
        losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'inp_obs'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr}

    def test_step(self, test_batch, batch_idx):
        return self.diag_step(test_batch, batch_idx, log_pref='test')

    def test_epoch_end(self, step_outputs):
        return self.diag_epoch_end(step_outputs, log_pref='test')

    def validation_step(self, batch, batch_idx):
        return self.diag_step(batch, batch_idx, log_pref='val')

    def validation_epoch_end(self, outputs):
        print(f'epoch end {self.global_rank} {len(outputs)}')
        if (self.current_epoch + 1) % self.hparams.val_diag_freq == 0:
            return self.diag_epoch_end(outputs, log_pref='val')

    def diag_epoch_end(self, outputs, log_pref='test'):
        data_path = Path(f'{self.logger.log_dir}/{log_pref}_data')
        data_path.mkdir(exist_ok=True, parents=True)
        print(len(outputs))
        torch.save(outputs, data_path / f'{self.global_rank}.t')
        if dist.is_initialized():
            dist.barrier()
        if self.global_rank > 0:
            print(f'Saved data for rank {self.global_rank}')
            return

        full_outputs = [torch.load(f) for f in sorted(data_path.glob('*'))]

        print(len(full_outputs))
        if log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise Exception('unknown phase')
        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i]
               for i in range(len(diag_ds))
            ]
        self.outputs = full_outputs

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield (
                                outputs[bc][b]['gt'][i],
                                outputs[bc][b]['oi'][i],
                                outputs[bc][b]['preds'][i],
                                outputs[bc][b]['inp_obs'][i],
                        )

        dses =[
                xr.Dataset( {
                    'gt': (('time', 'lat', 'lon'), x_gt),
                    'oi': (('time', 'lat', 'lon'), x_oi),
                    'pred': (('time', 'lat', 'lon'), x_rec),
                    'obs_inp': (('time', 'lat', 'lon'), obs_inp),
                }, coords=coords)
            for  (x_gt, x_oi, x_rec, obs_inp), coords
            in zip(iter_item(self.outputs), self.test_patch_coords)
        ]
        import time
        t0 = time.time()
        fin_ds = xr.merge([xr.zeros_like(ds[['time','lat', 'lon']]) for ds in dses])
        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
        )
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
            )

        for ds in dses:
            ds_nans = ds.assign(weight=xr.ones_like(ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)
            xr_weight = xr.DataArray(self.patch_weight.detach().cpu(), ds.coords, dims=ds.gt.dims)
            _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
            fin_ds = fin_ds + _ds


        self.test_xr_ds = (
            (fin_ds.drop('weight') / fin_ds.weight)
            .sel(instantiate(self.test_domain))
            .pipe(lambda ds: ds.sel(time=~(np.isnan(ds.gt).all('lat').all('lon'))))
        ).transpose('time', 'lat', 'lon')

        self.x_gt = self.test_xr_ds.gt.data
        self.obs_inp = self.test_xr_ds.obs_inp.data
        self.x_oi = self.test_xr_ds.oi.data
        self.x_rec = self.test_xr_ds.pred.data
        self.x_rec_ssh = self.x_rec

        self.test_coords = self.test_xr_ds.coords
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data

        Path(self.logger.log_dir).mkdir(exist_ok=True)
        # display map
        path_save0 = self.logger.log_dir + '/maps.png'
        t_idx = 3
        fig_maps = plot_maps(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save0)
        path_save01 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps(
                  self.x_gt[t_idx],
                self.obs_inp[t_idx],
                  self.x_oi[t_idx],
                  self.x_rec[t_idx],
                  self.test_lon, self.test_lat, path_save01, grad=True)
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure(f'{log_pref} Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure(f'{log_pref} Maps Grad', fig_maps_grad, global_step=self.current_epoch)

        # animate maps
        if self.hparams.animate == True:
            path_save0 = self.logger.log_dir + '/animation.mp4'
            animate_maps(self.x_gt,
                    self.x_oi,
                    self.x_rec,
                    self.lon, self.lat, path_save0)
            # save NetCDF
        path_save1 = self.logger.log_dir + f'/test.nc'
        # PENDING: replace hardcoded 60
        self.test_xr_ds.to_netcdf(path_save1)
        # save_netcdf(saved_path1=path_save1, pred=self.x_rec,
        #         lon=self.test_lon, lat=self.test_lat, time=self.test_dates, time_units=None)

        # compute nRMSE
        # np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)
        nrmse_fn = lambda pred, ref, gt: (
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - ds.mean())
                .pipe(lambda ds: ds - (self.test_xr_ds[gt].pipe(lambda da: da - da.mean())))
                .pipe(lambda ds: ds ** 2 / self.test_xr_ds[gt].std())
                .to_dataframe()
                .pipe(lambda ds: np.sqrt(ds.mean()))
                .to_frame()
                .rename(columns={0: 'nrmse'})
                .assign(nrmse_ratio=lambda df: df / df.loc[ref])
        )
        mse_fn = lambda pred, ref, gt: (
                self.test_xr_ds[[pred, ref]]
                .pipe(lambda ds: ds - self.test_xr_ds[gt])
                .pipe(lambda ds: ds ** 2)
                .to_dataframe()
                .pipe(lambda ds: ds.mean())
                .to_frame()
                .rename(columns={0: 'mse'})
                .assign(mse_ratio=lambda df: df / df.loc[ref])
        )

        nrmse_df = nrmse_fn('pred', 'oi', 'gt')
        mse_df = mse_fn('pred', 'oi', 'gt')
        nrmse_df.to_csv(self.logger.log_dir + '/nRMSE.txt')
        mse_df.to_csv(self.logger.log_dir + '/MSE.txt')

        # plot nRMSE
        # PENDING: replace hardcoded 60
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,  self.x_oi, self.x_rec, path_save3, time=self.test_dates)
        self.test_figs['nrmse'] = nrmse_fig
        self.logger.experiment.add_figure(f'{log_pref} NRMSE', nrmse_fig, global_step=self.current_epoch)
        # plot SNR
        path_save4 = self.logger.log_dir + '/SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_oi, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig

        self.logger.experiment.add_figure(f'{log_pref} SNR', snr_fig, global_step=self.current_epoch)
        psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        fig, spatial_res_model, spatial_res_oi = get_psd_score(self.test_xr_ds.gt, self.test_xr_ds.pred, self.test_xr_ds.oi, with_fig=True)
        self.test_figs['res'] = fig
        self.logger.experiment.add_figure(f'{log_pref} Spat. Resol', fig, global_step=self.current_epoch)
        psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        psd_fig = metrics.plot_psd_score(psd_ds)
        self.test_figs['psd'] = psd_fig
        psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)

        mdf = pd.concat([
            nrmse_df.rename(columns=lambda c: f'{log_pref}_{c}_glob').loc['pred'].T,
            mse_df.rename(columns=lambda c: f'{log_pref}_{c}_glob').loc['pred'].T,
        ])
        md = {
            f'{log_pref}_spatial_res': float(spatial_res_model),
            f'{log_pref}_spatial_res_imp': float(spatial_res_model / spatial_res_oi),
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
            **mdf.to_dict(),
        }
        self.latest_metrics.update(md)
        print(pd.DataFrame([md]).T.to_markdown())
        self.logger.log_metrics(
                {
                    f'{log_pref}_spatial_res': float(spatial_res_model),
                    f'{log_pref}_spatial_res_imp': float(spatial_res_model / spatial_res_oi),
                **md,
            },
        step=self.current_epoch)

    def teardown(self, stage='test'):

        self.logger.log_hyperparams(
                {**self.hparams},
                self.latest_metrics
    )

    def get_init_state(self, batch, state=(None,)):
        if state[0] is not None:
            return state[0]

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

        if self.aug_state:
            init_state = torch.cat((targets_OI,
                                    inputs_Mask * (inputs_obs - targets_OI),
                                    inputs_Mask * (inputs_obs - targets_OI),),
                                   dim=1)
        else:
            init_state = torch.cat((targets_OI,
                                    inputs_Mask * (inputs_obs - targets_OI)),
                                   dim=1)
        return init_state

    def loss_ae(self, state_out):
        return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def compute_loss(self, batch, phase, state_init=(None,)):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt = batch

        #targets_OI, inputs_Mask, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_GT),
                    torch.cat((torch.zeros_like(targets_GT),
                              torch.zeros_like(targets_GT),
                              torch.zeros_like(targets_GT)), dim=1),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                        ('mseOI', 0.),
                        ('mseGOI', 0.)])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)

        state = self.get_init_state(batch, state_init)

        #state = torch.cat((targets_OI, inputs_Mask * (targets_GT_wo_nan - targets_OI)), dim=1)

        obs = torch.cat( (targets_OI, inputs_Mask * (inputs_obs - targets_OI)) ,dim=1)
        new_masks = torch.cat( (torch.ones_like(inputs_Mask), inputs_Mask) , dim=1)

        if self.aug_state:
            obs = torch.cat( (obs, 0. * targets_OI,) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.zeros_like(inputs_Mask)), dim=1)

        if self.use_sst:
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]

        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            state = torch.autograd.Variable(state, requires_grad=True)
            # print(state.shape)
            # print(obs.shape)
            # print(new_masks.shape)
            outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR = outputs[:, 0:self.hparams.dT, :, :]
            if self.aug_state:
                outputs = outputsSLR + outputs[:, 2*self.hparams.dT:, :, :]
            else:
                outputs = outputsSLR + outputs[:, self.hparams.dT:2*self.hparams.dT, :, :]

            # median filter
            if self.median_filter_width > 1:
                outputs = kornia.filters.median_blur(outputs, (self.median_filter_width, self.median_filter_width))

            # reconstruction losses
            g_outputs_x, g_outputs_y = self.gradient_img(outputs)

            # PENDING: add loss term computed on obs (outputs swath - obs_target)
            # loss_All = NN_4DVar.compute_spatio_temp_weighted_loss((outputs - targets_GT), self.w_loss)
            # loss_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs - g_targets_GT, self.w_loss)
            # loss_OI = NN_4DVar.compute_spatio_temp_weighted_loss(targets_GT - targets_OI, self.w_loss)
            # loss_GOI = NN_4DVar.compute_spatio_temp_weighted_loss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            loss_All = NN_4DVar.compute_spatio_temp_weighted_loss((outputs - targets_GT), self.patch_weight)
            loss_GAll = (
                    NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_x - g_targets_GT_x, self.grad_crop(self.patch_weight))
                +    NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_y - g_targets_GT_y, self.grad_crop(self.patch_weight))
            )
            loss_OI = NN_4DVar.compute_spatio_temp_weighted_loss(targets_GT - targets_OI, self.patch_weight)
            g_OI_x, g_OI_y = self.gradient_img(targets_OI)
            loss_GOI = (
                NN_4DVar.compute_spatio_temp_weighted_loss(g_OI_x - g_targets_GT_x, self.grad_crop(self.patch_weight))
                + NN_4DVar.compute_spatio_temp_weighted_loss(g_OI_y - g_targets_GT_y, self.grad_crop(self.patch_weight))
            )
            # projection losses
            loss_AE = self.loss_ae(outputsSLRHR)

            if self.aug_state:
                yGT = torch.cat((targets_OI,
                                 targets_GT_wo_nan - outputsSLR,
                                 targets_GT_wo_nan - outputsSLR),
                                dim=1)
            else:
                yGT = torch.cat((targets_OI,
                                 targets_GT_wo_nan - outputsSLR),
                                dim=1)
            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            # loss_SR = NN_4DVar.compute_spatio_temp_weighted_loss(outputsSLR - targets_OI, self.w_loss)
            loss_SR = NN_4DVar.compute_spatio_temp_weighted_loss(outputsSLR - targets_OI, self.patch_weight)
            targets_GTLR = self.model_LR(targets_OI)
            # loss_LR = NN_4DVar.compute_spatio_temp_weighted_loss(self.model_LR(outputs) - targets_GTLR, self.model_LR(self.w_loss))
            loss_LR = NN_4DVar.compute_spatio_temp_weighted_loss(self.model_LR(outputs) - targets_GTLR, self.model_LR(self.patch_weight))

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR

            # metrics
            # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_targets_GT_x, g_targets_GT_y) , self.grad_crop(self.patch_weight))
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            metrics = dict([
                ('mse', mse),
                ('mseGrad', mseGrad),
                ('meanGrad', mean_GAll),
                ('mseOI', loss_OI.detach()),
                ('mseGOI', loss_GOI.detach())])

        return loss, outputs, [outputsSLRHR, hidden_new, cell_new, normgrad], metrics


if __name__ =='__main__':
    
    import hydra
    import importlib
    from hydra.utils import instantiate, get_class, call
    import hydra_main 
    import lit_model_augstate 
    
    importlib.reload(lit_model_augstate)
    importlib.reload(hydra_main)

    def get_cfg(xp_cfg, overrides=None):
        overrides = overrides if overrides is not None else []
        def get():
            cfg = hydra.compose(config_name='main', overrides=
                [
                    f'xp={xp_cfg}',
                    'file_paths=jz',
                    'entrypoint=train',
                ] + overrides
            )

            return cfg
        try:
            with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
                return get()
        except Exception as e:
            raise e
            return get()

    def get_model(xp_cfg, ckpt, dm=None, add_overrides=None):
        overrides = []
        if add_overrides is not None:
            overrides =  overrides + add_overrides
        cfg = get_cfg(xp_cfg, overrides)
        lit_mod_cls = get_class(cfg.lit_mod_cls)
        if dm is None:
            dm = instantiate(cfg.datamodule)
        runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
        mod = runner._get_model(ckpt)
        return mod

    def get_dm(xp_cfg, setup=True, add_overrides=None):
        overrides = []
        if add_overrides is not None:
            overrides = overrides + add_overrides
        cfg = get_cfg(xp_cfg, overrides)
        dm = instantiate(cfg.datamodule)
        if setup:
            dm.setup()
        return dm
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638612/checkpoints/modelCalSLAInterpGF-epoch=30-val_loss=1.5786.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=13-val_loss=1.4913.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=16-val_loss=1.4879.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=19-val_loss=1.4220.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=30-val_loss=1.4054.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=43-val_loss=1.4010.ckpt'

    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=35-val_loss=1.3944.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=59-val_loss=1.3850.ckpt'
    # cfg_n, ckpt = 'dT5_240', 'dashboard/xp_interp_dt5_240_h/lightning_logs/version_1638603/checkpoints/modelCalSLAInterpGF-epoch=80-val_loss=1.3976.ckpt'

    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1638604/checkpoints/modelCalSLAInterpGF-epoch=18-val_loss=1.4573.ckpt'
    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1638604/checkpoints/modelCalSLAInterpGF-epoch=29-val_loss=1.4368.ckpt'

    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1638604/checkpoints/modelCalSLAInterpGF-epoch=40-val_loss=1.3377.ckpt'

    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1638604/checkpoints/modelCalSLAInterpGF-epoch=60-val_loss=1.3481.ckpt'
    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1638604/checkpoints/modelCalSLAInterpGF-epoch=72-val_loss=1.3462.ckpt'

    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1714109/checkpoints/modelCalSLAInterpGF-epoch=07-val_loss=1.3174.ckpt'
    # cfg_n, ckpt = 'full_core', 'dashboard/xp_interp_dt7_240_h/lightning_logs/version_1714109/checkpoints/modelCalSLAInterpGF-epoch=16-val_loss=1.3098.ckpt'


    # cfg_n, ckpt = 'quentin_repro_w_hugo_lit_240', 'dashboard/xp_interp_dt7_240_r/lightning_logs/version_1638608/checkpoints/modelCalSLAInterpGF-epoch=98-val_loss=1.4799.ckpt'
    # cfg_n, ckpt = 'quentin_repro_w_hugo_lit_240', 'dashboard/xp_interp_dt7_240_r/lightning_logs/version_1638608/checkpoints/modelCalSLAInterpGF-epoch=74-val_loss=1.5486.ckpt'
    # cfg_n, ckpt = 'quentin_repro_w_hugo_lit_240', 'dashboard/xp_interp_dt7_240_r/lightning_logs/version_1638608/checkpoints/modelCalSLAInterpGF-epoch=85-val_loss=1.5409.ckpt'

    # cfg_n, ckpt = 'quentin_repro_w_hugo_lit_240', 'modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=42-val_loss=1.28.ckpt'

    # cfg_n, ckpt = 'full_core', 'modelSLA-L2-GF-augdata01-augstate-boost-swot-dT07-igrad05_03-dgrad150-epoch=42-val_loss=1.28.ckpt'
    # cfg_n, ckpt = 'dT9', 'dashboard/xp_interp_dt9_240_h/lightning_logs/version_1715728/checkpoints/modelCalSLAInterpGF-epoch=23-val_loss=1.4065.ckpt'
    # cfg_n, ckpt = 'dT5_240_sst', None
    cfg_n, ckpt = 'full_core_hanning_t_grad', 'dashboard/xp_interp_dt7_240_h_hanning/lightning_logs/version_1733027/checkpoints/modelCalSLAInterpGF-epoch=04-val_loss=0.2668.ckpt'
    dm = get_dm(f"xp_aug/xp_repro/{cfg_n}", setup=False,
            add_overrides=[
                # 'params.files_cfg.obs_mask_path=/gpfsssd/scratch/rech/yrf/ual82ir/sla-data-registry/CalData/cal_data_new_errs.nc',
                # 'params.files_cfg.obs_mask_path=/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d.nc',
                # 'params.files_cfg.obs_mask_var=four_nadirs'
                # 'params.files_cfg.obs_mask_var=swot_nadirs_no_noise'
            ]


    )
    mod = get_model(
            f"xp_aug/xp_repro/{cfg_n}",
            ckpt,
            dm=dm)
    mod.hparams
    cfg = get_cfg(f"xp_aug/xp_repro/{cfg_n}")
    # cfg = get_cfg("xp_aug/xp_repro/quentin_repro")
    print(OmegaConf.to_yaml(cfg))
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner.test(ckpt)
    mod.test_figs['psd']
