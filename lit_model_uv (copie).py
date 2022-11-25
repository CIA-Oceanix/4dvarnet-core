"""
Model UV

Authors: R. Fablet, ...
"""

from pathlib import Path

import hydra
from hydra.utils import call, instantiate
import kornia
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import pytorch_lightning as pl
from scipy.ndimage import convolve1d, gaussian_filter, sobel
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr

from metrics import (
    animate_maps, compute_laplacian_metrics, compute_metrics, get_psd_score,
    plot_maps, plot_nrmse, plot_psd_score, plot_snr, psd_based_scores,
    rmse_based_scores, save_netcdf_uv,
)
from models import (
    Model_HwithSST_nolin_tanh, ModelLR, Model_H, Model_HwithSST,
    Model_HwithSSTBN, Model_HwithSSTBNAtt_nolin_tanh,
    Model_HwithSSTBN_nolin_tanh, Model_HwithSSTBNandAtt, Phi_r,
)
import solver as NN_4DVar


def _compute_grad(axis, u, alpha_d=1., sigma=0., _filter='diff-non-centered'):
    if sigma > 0.:
        u = gaussian_filter(u, sigma=sigma)

    if _filter == 'sobel':
        return alpha_d * sobel(u, axis=axis)
    elif _filter == 'diff-non-centered':
        return alpha_d * convolve1d(u, weights=[.3, .4, -.7], axis=axis)
    else:
        raise ValueError(f'Invalid argument: _filter={_filter}')


def compute_coriolis_force(lat, flag_mean_coriolis=False):
    omega = 7.2921e-5  # angular speed (rad/s)
    f = 2 * omega * np.sin(lat)

    if flag_mean_coriolis:
        f = np.mean(f) * np.ones((f.shape))

    return f


def compute_div_curl_strain_with_lat_lon(u, v, lat, lon, sigma=1.):
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    # coriolis / lat/lon scaling
    grid_lat = lat.reshape((1, u.shape[1], 1))
    grid_lat = np.tile(grid_lat, (v.shape[0], 1, v.shape[2]))
    grid_lon = lon.reshape((1, 1, v.shape[2]))
    grid_lon = np.tile(grid_lon, (v.shape[0], v.shape[1], 1))

    dx_from_dlon, dy_from_dlat = compute_dx_dy_dlat_dlon(
        grid_lat, grid_lon, dlat, dlon,
    )

    du_dx = compute_gradx(u, sigma=sigma)
    dv_dy = compute_grady(v, sigma=sigma)

    du_dy = compute_grady(u, sigma=sigma)
    dv_dx = compute_gradx(v, sigma=sigma)

    du_dx = du_dx / dx_from_dlon
    dv_dx = dv_dx / dx_from_dlon

    du_dy = du_dy / dy_from_dlat
    dv_dy = dv_dy / dy_from_dlat

    strain = np.sqrt((dv_dx + du_dy)**2 + (du_dx - dv_dy)**2)

    div = du_dx + dv_dy
    curl =  du_dy - dv_dx

    return div, curl, strain


def compute_dx_dy_dlat_dlon(lat, lon, dlat, dlon):
    def compute_c(lat, lon, dlat, dlon):
        a = np.sin(dlat / 2)**2 + np.cos(lat)**2 * np.sin(dlon/2)**2

        return 2 * 6.371e6 * np.arctan2(np.sqrt(a), np.sqrt(1. - a))

    dy_from_dlat = compute_c(lat, lon, dlat, 0.)
    dx_from_dlon = compute_c(lat, lon, 0., dlon)

    return dx_from_dlon, dy_from_dlat


def compute_gradx(u, alpha_dx=1., sigma=0., _filter='diff-non-centered'):
    return _compute_grad(
        axis=2, u=u, alpha_d=alpha_dx, sigma=sigma, _filter=_filter,
    )


def compute_grady(u, alpha_dy=1., sigma=0., _filter='diff-non-centered'):
    return _compute_grad(
        axis=1, u=u, alpha_d=alpha_dy, sigma=sigma, _filter=_filter,
    )


def compute_uv_geo_with_coriolis(
    ssh, lat, lon, sigma=0.5, alpha_uv_geo=1., flag_mean_coriolis=False,
):
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    # coriolis / lat/lon scaling
    grid_lat = lat.reshape((1, ssh.shape[1], 1))
    grid_lat = np.tile(grid_lat, (ssh.shape[0], 1, ssh.shape[2]))
    grid_lon = lon.reshape((1, 1, ssh.shape[2]))
    grid_lon = np.tile(grid_lon, (ssh.shape[0], ssh.shape[1], 1))

    f_c = compute_coriolis_force(
        grid_lat, flag_mean_coriolis=flag_mean_coriolis,
    )
    dx_from_dlon, dy_from_dlat = compute_dx_dy_dlat_dlon(
        grid_lat, grid_lon, dlat, dlon,
    )

    # (u, v) MSE
    ssh = gaussian_filter(ssh, sigma=sigma)
    dssh_dx = compute_gradx(ssh)
    dssh_dy = compute_grady(ssh)

    dssh_dx = dssh_dx / dx_from_dlon
    dssh_dy = dssh_dy / dy_from_dlat

    dssh_dy = (1. / f_c) * dssh_dy
    dssh_dx = (1. / f_c) * dssh_dx

    u_geo = -1. * dssh_dy
    v_geo = 1. * dssh_dx

    u_geo = alpha_uv_geo * u_geo
    v_geo = alpha_uv_geo * v_geo

    return u_geo, v_geo


def get_4dvarnet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
        Phi_r(
            hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2,
            hparams.sS, hparams.nbBlocks, hparams.dropout_phi_r,
            hparams.stochastic, hparams.phi_param,
        ),
        Model_H(hparams.shape_state[0]),
        NN_4DVar.model_GradUpdateLSTM(
            hparams.shape_state, hparams.UsePriodicBoundary,
            hparams.dim_grad_solver, hparams.dropout, hparams.asymptotic_term,
        ),
        hparams.norm_obs, hparams.norm_prior, hparams.shape_state,
        hparams.n_grad * hparams.n_fourdvar_iter,
    )


def get_4dvarnet_sst(hparams):
    print(f'...... Set model {hparams.use_sst_obs}', flush=True)
    if not hparams.use_sst_obs:
        return get_4dvarnet(hparams)

    if hparams.sst_model == 'linear-bn':
        _sst_model = Model_HwithSSTBN(
            hparams.shape_state[0], dT=hparams.dT,
            dim=hparams.dim_obs_sst_feat,
        )
    elif hparams.sst_model == 'nolinear-tanh-bn':
        res_wrt_geo_vel = hparams.residual_wrt_geo_velocities
        print(
            f'...... residual_wrt_geo_velocities = {res_wrt_geo_vel}',
            flush=True,
        )

        if res_wrt_geo_vel in (3, 4):
            _sst_model = Model_HwithSSTBN_nolin_tanh_withlatlon(
                hparams.shape_state[0], dT=hparams.dT,
                dim=hparams.dim_obs_sst_feat,
            )
        else:
            _sst_model = Model_HwithSSTBN_nolin_tanh(
                hparams.shape_state[0], dT=hparams.dT,
                dim=hparams.dim_obs_sst_feat,
            )
    elif hparams.sst_model == 'nolinear-tanh':
        _sst_model = Model_HwithSST_nolin_tanh(
            hparams.shape_state[0], dT=hparams.dT,
            dim=hparams.dim_obs_sst_feat,
        )
    elif hparams.sst_model == 'linear':
        _sst_model = Model_HwithSST(
            hparams.shape_state[0], dT=hparams.dT,
            dim=hparams.dim_obs_sst_feat,
        )
    elif hparams.sst_model == 'linear-bn-att':
        _sst_model = Model_HwithSSTBNandAtt(
            hparams.shape_state[0], dT=hparams.dT, dim=hparams.dim_obs_sst_feat,
        ),
    elif hparams.sst_model == 'nolinear-tanh-bn-att':
        _sst_model = Model_HwithSSTBNAtt_nolin_tanh(
            hparams.shape_state[0], dT=hparams.dT,
            dim=hparams.dim_obs_sst_feat,
        ),
    else:
        raise ValueError(
            f'Invalid argument: hparams.sst_model={hparams.sst_model}',
        )

    return NN_4DVar.Solver_Grad_4DVarNN(  # Final return
        Phi_r(
            hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2,
            hparams.sS, hparams.nbBlocks, hparams.dropout_phi_r,
            hparams.stochastic, hparams.phi_param,
        ),
        _sst_model,
        NN_4DVar.model_GradUpdateLSTM(
            hparams.shape_state, hparams.UsePriodicBoundary,
            hparams.dim_grad_solver, hparams.dropout, hparams.asymptotic_term,
        ),
        hparams.norm_obs, hparams.norm_prior, hparams.shape_state,
        hparams.n_grad * hparams.n_fourdvar_iter,
    )


def get_constant_crop(patch_size, crop, dim_order=('time', 'lat', 'lon')):
    patch_weight = np.zeros(
        [patch_size[d] for d in dim_order], dtype='float32',
    )

    mask = []
    for d in dim_order:
        if crop.get(d, 0) > 0:
            mask.append(slice(crop[d], -crop[d]))
        else:
            mask.append(slice(None, None))
    mask = tuple(mask)

    patch_weight[mask] = 1.

    return patch_weight


def get_cropped_hanning_mask(patch_size, crop, **kwargs):
    pw = get_constant_crop(patch_size, crop)
    t_msk = kornia.filters.get_hanning_kernel1d(patch_size['time'])
    patch_weight = t_msk[:, None, None] * pw

    return patch_weight.cpu().numpy()


def get_phi(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = Phi_r(
                hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2,
                hparams.sS, hparams.nbBlocks, hparams.dropout_phi_r,
                hparams.stochastic, hparams.phi_param,
            )
            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks, *internal_state):
            return self.phi(state), None, None, None

    return PhiPassThrough()


class LitModelUV(pl.LightningModule):
    MODELS = {
        '4dvarnet': get_4dvarnet,
        '4dvarnet_sst': get_4dvarnet_sst,
        'phi': get_phi,
    }

    def __init__(
        self,
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
        *args, **kwargs,
    ):
        super().__init__()

        hparams = hparam or {}
        if not isinstance(hparams, dict):
            hparams = OmegaConf.to_container(hparams, resolve=True)

        self.save_hyperparameters({**hparams, **kwargs}, logger=False)
        self.latest_metrics = {}

        # Version of `build_test_xr_ds` to use (1 or 2, the latter being
        # the default value)
        self.version_build_test_xr_ds = self.hparams.get(
            'version_build_test_xr_ds', 2
        )

        # TOTEST: set those parameters only if provided
        self.var_Val = self.hparams.var_Val  # Unused in the current file
        self.var_Tr = self.hparams.var_Tr
        self.var_Tt = self.hparams.var_Tt
        self.var_tr_uv = self.hparams.var_tr_uv

        self.alpha_dx = 1.
        self.alpha_dy = 1.

        # create longitudes & latitudes coordinates
        self.test_domain=test_domain
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None

        self.patch_weight = None
        self.patch_weight_train = torch.nn.Parameter(
            torch.from_numpy(call(self.hparams.patch_weight)),
            requires_grad=False,
        )
        self.patch_weight_diag = torch.nn.Parameter(
            torch.from_numpy(call(self.hparams.patch_weight)),
            requires_grad=False,
        )

        self.mean_Val = self.hparams.mean_Val
        self.mean_Tr = self.hparams.mean_Tr
        self.mean_Tt = self.hparams.mean_Tt

        # main model
        self.model_name = self.hparams.get('model', '4dvarnet')
        self.use_sst = self.hparams.get('sst', False)
        self.use_sst_obs = self.hparams.get('use_sst_obs', False)
        self.use_sst_state = self.hparams.get('use_sst_state', False)
        self.aug_state = self.hparams.get('aug_state', False)
        self.save_rec_netcdf = self.hparams.get('save_rec_netcdf', False)
        self.sig_filter_laplacian = self.hparams.get('sig_filter_laplacian', .5)
        self.scale_dwscaling_sst = self.hparams.get('scale_dwscaling_sst', 1.)
        self.sig_filter_div = self.hparams.get('sig_filter_div', 1.)
        self.sig_filter_div_diag = self.hparams.get(
            'sig_filter_div_diag', self.hparams.sig_filter_div,
        )
        self.hparams.alpha_mse_strain = self.hparams.get('alpha_mse_strain', 0.)

        self.type_div_train_loss = self.hparams.get('type_div_train_loss', 1)

        self.scale_dwscaling = self.hparams.get('scale_dwscaling', 1.)
        if self.scale_dwscaling > 1.:
            _w = torch.from_numpy(call(self.hparams.patch_weight))
            _w = torch.nn.functional.avg_pool2d(
                _w.view(1, -1, _w.size(1), _w.size(2)),
                (int(self.scale_dwscaling), int(self.scale_dwscaling)),
            )
            self.patch_weight_train = torch.nn.Parameter(
                _w.view(-1, _w.size(2), _w.size(3)), requires_grad=False,
            )

            _w = torch.from_numpy(call(self.hparams.patch_weight))
            self.patch_weight_diag = torch.nn.Parameter(_w, requires_grad=False)

        self.residual_wrt_geo_velocities = self.hparams.get(
            'residual_wrt_geo_velocities', 0,
        )

        self.use_lat_lon_in_obs_model = self.hparams.get(
            'use_lat_lon_in_obs_model', False,
        )
        if self.residual_wrt_geo_velocities in (3, 4):
            self.use_lat_lon_in_obs_model = True

        self.learning_sampling_uv = self.hparams.get(
            'learning_sampling_uv',
            'no_sammpling_learning',  # NOTE Typo?
        )
        self.nb_feat_sampling_operator = self.hparams.get(
            'nb_feat_sampling_operator', -1.
        )
        if self.nb_feat_sampling_operator > 0:
            if self.hparams.sampling_model == 'sampling-from-sst':
                self.model_sampling_uv = ModelSamplingFromSST(
                    self.hparams.dT, self.nb_feat_sampling_operator,
                )
            else:
                print('..... something is not expected with the sampling model')
        else:
            self.model_sampling_uv = None

        if self.hparams.k_n_grad == 0:
            self.hparams.n_fourdvar_iter = 1

        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.grad_crop = lambda t: t[..., 1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
            self.grad_crop(2.*kornia.filters.spatial_gradient(
                t, normalized=True,
            )), 2
        )

        if self.residual_wrt_geo_velocities in (3, 4):
            self.model.model_H.aug_state = self.hparams.aug_state
            self.model.model_H.var_tr_uv = self.var_tr_uv  # Unused in solver

        self.model.model_Grad.asymptotic_term = True

        b = self.hparams.get('apha_grad_descent_step', 0.)
        self.model.model_Grad.b = torch.nn.Parameter(
            torch.Tensor([b]), requires_grad=False,
        )

        self.hparams.learn_fsgd_param = self.hparams.get(
            'learn_fsgd_param', False,
        )

        if self.hparams.learn_fsgd_param == True :
            self.model.model_Grad.set_fsgd_param_trainable()

        self.compute_derivativeswith_lon_lat = (
            TorchComputeDerivativesWithLonLat(dT=self.hparams.dT)
        )

        self.w_loss = torch.nn.Parameter(
            torch.Tensor([0, 0, 0, 1, 0, 0, 0]),
            requires_grad=False,
        )  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.obs_inp = None
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.x_feat = None  # variable to store output of test method
        self.test_figs = {}

        self.tr_loss_hist = []
        self.automatic_optimization = self.hparams.get(
            'automatic_optimization', False,
        )

        self.median_filter_width = self.hparams.get('median_filter_width', 1)

        print(
            f'..... div. computation (sigma): {self.sig_filter_div}'
            + f' -- {self.sig_filter_div_diag}'
        )
        print(f'..  Div loss type : {self.type_div_train_loss}')

    def compute_div(self, u, v):
        # siletring
        f_u = kornia.filters.gaussian_blur2d(
            u, (5, 5), (self.sig_filter_div, self.sig_filter_div), border_type='reflect',
        )
        f_v = kornia.filters.gaussian_blur2d(
            v, (5, 5), (self.sig_filter_div, self.sig_filter_div), border_type='reflect',
        )

        # gradients
        du_dx, du_dy = self.gradient_img(f_u)
        dv_dx, dv_dy = self.gradient_img(f_v)

        # scaling
        du_dx = self.alpha_dx * dv_dx
        dv_dy = self.alpha_dy * dv_dy

        return du_dx + dv_dy

    def update_filename_chkpt(self, filename_chkpt):
        old_suffix = '-{epoch:02d}-{val_loss:.4f}'

        suffix_chkpt = (
            f'-{self.hparams.phi_param}_{self.hparams.DimAE:03d}-augdata'
        )

        if self.scale_dwscaling > 1.0:
            suffix_chkpt += f'-dws{int(self.scale_dwscaling):02d}'

        if self.scale_dwscaling_sst > 1.:
            suffix_chkpt += f'-dws-sst{int(self.scale_dwscaling_sst):02d}'

        if self.model_sampling_uv is not None:
            suffix_chkpt += (
                f'-sampling_sst_{self.hparams.nb_feat_sampling_operator}'
                + f'_{int(100*self.hparams.thr_l1_sampling_uv):03d}'
            )

        if self.hparams.n_grad > 0:
            if self.hparams.aug_state :
                suffix_chkpt += f'-augstate-dT{self.hparams.dT:02d}'

            if self.use_sst_state:
                suffix_chkpt += f'-mmstate-augstate-dT{self.hparams.dT:02d}'

            if self.use_sst_obs:
                suffix_chkpt += (
                    f'-sstobs-{self.hparams.sst_model}'
                    + f'_{self.hparams.dim_obs_sst_feat:02d}'
                )

            if self.residual_wrt_geo_velocities > 0:
                suffix_chkpt += f'-wgeo{self.residual_wrt_geo_velocities}'
            elif self.type_div_train_loss == 1:
                suffix_chkpt += '-nowgeo'

            if (
                self.hparams.alpha_mse_strain == 0.
                or self.hparams.alpha_mse_div == 0.
            ):
                if (
                    self.hparams.alpha_mse_strain == 0.
                    and self.hparams.alpha_mse_div == 0.
                ):
                    suffix_chkpt += '-nodivstrain'
                elif self.hparams.alpha_mse_strain == 0.:
                    suffix_chkpt += '-nostrain'
                else:
                    suffix_chkpt += '-nodiv'

            suffix_chkpt += (
                f'-grad_{self.hparams.n_grad:02d}'
                + f'_{self.hparams.k_n_grad:02d}'
                + f'_{self.hparams.dim_grad_solver:03d}'
            )
            if self.model.model_Grad.asymptotic_term:
                suffix_chkpt += '+fsgd'
            if self.hparams.learn_fsgd_param:
                suffix_chkpt += 'train'
        else:
            if self.use_sst and self.use_sst_state:
                suffix_chkpt += '-DirectInv-wSST'
            else:
                suffix_chkpt += '-DirectInv'
            suffix_chkpt += f'-dT{self.hparams.dT:02d}'

            if (
                self.hparams.alpha_mse_strain == 0.
                or self.hparams.alpha_mse_div == 0.
            ):
                if (
                    self.hparams.alpha_mse_strain == 0.
                    and self.hparams.alpha_mse_div == 0.
                ):
                    suffix_chkpt += '-nodivstrain'
                elif self.hparams.alpha_mse_strain == 0.:
                    suffix_chkpt += '-nostrain'
                else :
                    suffix_chkpt += '-nodiv'
        suffix_chkpt += old_suffix

        return filename_chkpt.replace(old_suffix, suffix_chkpt)

    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out = None

        for _k in range(self.hparams.n_fourdvar_iter):
            if self.model.model_Grad.asymptotic_term:
                self.model.model_Grad.iter = 0
                self.model.model_Grad.iter = 1. * _k * self.model.n_grad

            if phase == 'test' and self.use_sst:
                _loss, out, state, _metrics, sst_feat = self.compute_loss(
                    batch, phase=phase, state_init=state_init,
                )
            else:
                _loss, out, state, _metrics = self.compute_loss(
                    batch, phase=phase, state_init=state_init,
                )

            if self.hparams.n_grad > 0:
                state_init = [None if s is None else s.detach() for s in state]
            losses.append(_loss)
            metrics.append(_metrics)

        if phase == 'test' and self.use_sst:
            return losses, out, metrics, sst_feat
        else:
            return losses, out, metrics

    def configure_optimizers(self):
        # Optimiser
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)

        # Parameters to be given to the optimiser
        lr = self.hparams.lr_update[0]
        params_and_lr = [
            {'params': self.model.model_Grad.parameters(), 'lr': lr},
            {'params': self.model.model_VarCost.parameters(), 'lr': lr},
            {'params': self.model.model_H.parameters(), 'lr': lr},
            {'params': self.model.phi_r.parameters(), 'lr': .5*lr},
        ]

        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            if self.model_sampling_uv:
                params_and_lr.append(
                    {'params': self.model_sampling_uv.parameters(), 'lr': .5*lr}
                )

            return opt(params_and_lr)
        else:
            opt = optim.Adam(self.parameters(), lr=1e-4)

            return {
                'optimizer': opt,
                'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    opt, verbose=True, patience=50,
                ),
                'monitor': 'val_loss',
            }

    def on_epoch_start(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.n_grad = self.hparams.n_grad

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (
                (self.current_epoch in self.hparams.iter_update)
                and (self.current_epoch > 0)
            ):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print(
                    '... Update Iterations number/learning '
                    + f'rate #{self.current_epoch}: '
                    + f'NGrad = {self.hparams.nb_grad_update[indx]} -- '
                    + f'lr = {self.hparams.lr_update[indx]}'
                )

                self.hparams.n_grad = self.hparams.nb_grad_update[indx]
                self.model.n_grad = self.hparams.n_grad

                mm = 0
                lrCurrent = self.hparams.lr_update[indx]
                lr = np.array([lrCurrent, lrCurrent, .5*lrCurrent, 0.])
                for pg in opt.param_groups:
                    pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                    mm += 1

        self.patch_weight = self.patch_weight_train

    def training_epoch_end(self, outputs):
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if best_ckpt_path:
            def should_reload_ckpt(losses):
                diffs = losses.diff()
                if losses.max() > (10 * losses.min()):
                    print("Reloading because of check", 1)
                    return True

                if diffs.max() > (100 * diffs.abs().median()):
                    print("Reloading because of check", 2)
                    return True
                return False

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

        self.log(
            "tr_loss", loss, on_step=True, on_epoch=False, prog_bar=True,
            logger=True,
        )
        self.log(
            "tr_mse", metrics[-1]['mse'] / self.var_Tr, on_step=False,
            on_epoch=True, prog_bar=True,
        )
        self.log(
            "tr_mse_uv", metrics[-1]['mse_uv'] , on_step=False, on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "tr_l1_samp", metrics[-1]['l1_samp'] , on_step=False,
            on_epoch=True, prog_bar=True,
        )

        return loss

    def diag_step(self, batch, batch_idx, log_pref='test'):
        if not self.use_sst:
            (
                targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt,
                lat, lon, ais_dat 
            ) = batch

            losses, out, metrics = self(batch, phase='test')
        else:
            #targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt = batch
            (
                targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt,
                v_gt, lat, lon, ais_dat
            ) = batch

            #losses, out, metrics = self(batch, phase='test')
            losses, out, metrics, sst_feat = self(batch, phase='test')

        loss = losses[-1]
        if loss:
            self.log(f'{log_pref}_loss', loss)
            self.log(
                f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt,
                on_step=False, on_epoch=True, prog_bar=True,
            )
            self.log(
                f'{log_pref}_mse_uv', metrics[-1]["mse_uv"], on_step=False,
                on_epoch=True, prog_bar=True,
            )
            self.log(
                f'{log_pref}_l1_samp', metrics[-1]["l1_samp"], on_step=False,
                on_epoch=True, prog_bar=True,
            )
            self.log(
                f'{log_pref}_l0_samp', metrics[-1]["l0_samp"], on_step=False,
                on_epoch=True, prog_bar=True,
            )

        out_pred = out[0]
        out_u = out[1]
        out_v = out[2]

        aux_input = (  # Auxiliary value for computing `obs_inp` param
            inputs_obs.detach()
            .where(inputs_Mask, torch.full_like(inputs_obs, np.nan))
            .cpu()
        )

        params = {
            'gt': targets_GT.detach().cpu()*np.sqrt(self.var_Tr) + self.mean_Tr,
            'oi': targets_OI.detach().cpu()*np.sqrt(self.var_Tr) + self.mean_Tr,
            'u_gt': u_gt.detach().cpu() * np.sqrt(self.var_tr_uv),
            'v_gt': v_gt.detach().cpu() * np.sqrt(self.var_tr_uv),
            'obs_inp': aux_input*np.sqrt(self.var_Tr) + self.mean_Tr,
            'pred': out_pred.detach().cpu()*np.sqrt(self.var_Tr) + self.mean_Tr,
            'pred_u': out_u.detach().cpu() * np.sqrt(self.var_tr_uv),
            'pred_v': out_v.detach().cpu() * np.sqrt(self.var_tr_uv),
        }
        if self.use_sst :
            params['sst_feat'] = sst_feat.detach().cpu()

        return params

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

    def gather_outputs(self, outputs, log_pref):
        data_path = Path(f'{self.logger.log_dir}/{log_pref}_data')
        data_path.mkdir(exist_ok=True, parents=True)

        torch.save(outputs, data_path / f'{self.global_rank}.t')

        if dist.is_initialized():
            dist.barrier()

        if self.global_rank == 0:
            return [torch.load(f) for f in sorted(data_path.glob('*'))]

    def build_test_xr_ds(self, outputs, diag_ds, version, use_sst):
        if version not in (1, 2):
            raise ValueError(f'Invalid argument: version={version}')

        outputs_keys = list(outputs[0][0].keys())
        _outputs_keys = outputs_keys[:-1] if use_sst else outputs_keys

        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i] for i in range(len(diag_ds))
            ]

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield tuple(
                            [outputs[bc][b][k][i] for k in _outputs_keys]
                        )

        dses = []
        for xs, coords in zip(iter_item(outputs), self.test_patch_coords):
            data_vars = dict()
            for k, x_k in zip(outputs_keys, xs):
                data_vars[k] = (('time', 'lat', 'lon'), x_k)

            dses.append(xr.Dataset(data_vars, coords=coords))

        fin_ds = xr.merge(
            [xr.zeros_like(ds[['time','lat', 'lon']]) for ds in dses]
        )
        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
        )
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
            )

        if version == 1:
            # set the weight to a binadry (time window center + spatial
            # bounding box)
            print(".... Set weight matrix to binary mask for final outputs")
            w = np.zeros_like(self.patch_weight.detach().cpu().numpy())
            w[int(self.hparams.dT/2), :, :] = 1.
            w = w * self.patch_weight.detach().cpu().numpy()

        for ds in dses:
            ds_nans = (
                ds.assign(weight=xr.ones_like(ds.gt))
                .isnull()
                .broadcast_like(fin_ds)
                .fillna(0.)
            )

            if version == 1 and use_sst:
                xr_weight = xr.DataArray(w, ds.coords, dims=ds.gt.dims)
            else:
                xr_weight = xr.DataArray(
                    self.patch_weight.detach().cpu(), ds.coords, dims=ds.gt.dims,
                )

            _ds = (
                ds.pipe(lambda dds: dds * xr_weight)
                .assign(weight=xr_weight)
                .broadcast_like(fin_ds)
                .fillna(0.)
                .where(ds_nans==0, np.nan)
            )
            fin_ds = fin_ds + _ds

        if version == 2:
            return (
                (fin_ds.drop('weight') / fin_ds.weight)
                .sel(instantiate(self.test_domain))
                .isel(time=slice(self.hparams.dT//2, -self.hparams.dT//2 + 1))
            ).transpose('time', 'lat', 'lon')
        else:
            return (
                (fin_ds.drop('weight') / fin_ds.weight)
                .sel(instantiate(self.test_domain))
                .pipe(lambda ds: ds.sel(time=~(
                    np.isnan(ds.gt).all('lat').all('lon')
                )))
            ).transpose('time', 'lat', 'lon')

    def nrmse_fn(self, pred, ref, gt):
        centering = lambda da: da - da.mean()

        return (
            self.test_xr_ds[[pred, ref]]
            .pipe(lambda ds: ds - ds.mean())
            .pipe(lambda ds: ds - (self.test_xr_ds[gt].pipe(centering)))
            .pipe(lambda ds: ds ** 2 / self.test_xr_ds[gt].std())
            .to_dataframe()
            .pipe(lambda ds: np.sqrt(ds.mean()))
            .to_frame()
            .rename(columns={0: 'nrmse'})
            .assign(nrmse_ratio=lambda df: df / df.loc[ref])
        )

    def mse_fn(self, pred, ref, gt):
        return (
            self.test_xr_ds[[pred, ref]]
            .pipe(lambda ds: ds - self.test_xr_ds[gt])
            .pipe(lambda ds: ds ** 2)
            .to_dataframe()
            .pipe(lambda ds: ds.mean())
            .to_frame()
            .rename(columns={0: 'mse'})
            .assign(mse_ratio=lambda df: df / df.loc[ref])
        )

    def sla_uv_diag(self, t_idx=3, log_pref='test'):
        path_save0 = self.logger.log_dir + '/maps.png'
        t_idx = 3
        fig_maps = plot_maps(
            self.x_gt[t_idx],
            self.obs_inp[t_idx],
            self.x_oi[t_idx],
            self.x_rec[t_idx],
            self.test_lon, self.test_lat, path_save0,
        )
        path_save01 = self.logger.log_dir + '/maps_Grad.png'
        fig_maps_grad = plot_maps(
            self.x_gt[t_idx],
            self.obs_inp[t_idx],
            self.x_oi[t_idx],
            self.x_rec[t_idx],
            self.test_lon, self.test_lat, path_save01, grad=True,
        )
        self.test_figs['maps'] = fig_maps
        self.test_figs['maps_grad'] = fig_maps_grad
        self.logger.experiment.add_figure(
            f'{log_pref} Maps', fig_maps, global_step=self.current_epoch,
        )
        self.logger.experiment.add_figure(
            f'{log_pref} Maps Grad', fig_maps_grad,
            global_step=self.current_epoch,
        )

        # animate maps
        if self.hparams.animate:
            path_save0 = self.logger.log_dir + '/animation.mp4'
            animate_maps(
                self.x_gt, self.obs_inp, self.x_oi, self.x_rec, self.lon,
                self.lat, path_save0,
            )

        nrmse_df = self.nrmse_fn('pred', 'oi', 'gt')
        mse_df = self.mse_fn('pred', 'oi', 'gt')
        nrmse_df.to_csv(self.logger.log_dir + '/nRMSE.txt')
        mse_df.to_csv(self.logger.log_dir + '/MSE.txt')

        # plot nRMSE
        # PENDING: replace hardcoded 60
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(
            self.x_gt,  self.x_oi, self.x_rec,
            path_save3, time=self.test_dates,
        )
        self.test_figs['nrmse'] = nrmse_fig
        self.logger.experiment.add_figure(
            f'{log_pref} NRMSE', nrmse_fig, global_step=self.current_epoch,
        )
        # plot SNR
        path_save4 = self.logger.log_dir + '/SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_oi, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig

        self.logger.experiment.add_figure(
            f'{log_pref} SNR', snr_fig, global_step=self.current_epoch,
        )
        psd_ds, lamb_x, lamb_t = psd_based_scores(
            self.test_xr_ds.pred, self.test_xr_ds.gt,
        )
        fig, spatial_res_model, spatial_res_oi = get_psd_score(
            self.test_xr_ds.gt, self.test_xr_ds.pred, self.test_xr_ds.oi,
            with_fig=True,
        )

        self.test_figs['res'] = fig
        self.logger.experiment.add_figure(
            f'{log_pref} Spat. Resol', fig, global_step=self.current_epoch,
        )
        psd_fig = plot_psd_score(psd_ds)
        self.test_figs['psd'] = psd_fig
        self.logger.experiment.add_figure(
            f'{log_pref} PSD', psd_fig, global_step=self.current_epoch,
        )
        _, _, mu, sig = rmse_based_scores(
            self.test_xr_ds.pred, self.test_xr_ds.gt,
        )

        mdf = pd.concat([
            nrmse_df.rename(
                columns=lambda c: f'{log_pref}_{c}_glob'
            ).loc['pred'].T,
            mse_df.rename(
                columns=lambda c: f'{log_pref}_{c}_glob'
            ).loc['pred'].T,
        ])

        mse_metrics_pred = compute_metrics(
            self.test_xr_ds.gt, self.test_xr_ds.pred,
        )
        mse_metrics_oi = compute_metrics(
            self.test_xr_ds.gt, self.test_xr_ds.oi,
        )

        var_mse_pred_vs_oi = (
            100. * (1. - mse_metrics_pred['mse'] / mse_metrics_oi['mse'])
        )
        var_mse_grad_pred_vs_oi = (
            100.*(1. - mse_metrics_pred['mseGrad'] / mse_metrics_oi['mseGrad'])
        )

        mse_metrics_lap_oi = compute_laplacian_metrics(
            self.test_xr_ds.gt, self.test_xr_ds.oi,
            sig_lap=self.sig_filter_laplacian,
        )
        mse_metrics_lap_pred = compute_laplacian_metrics(
            self.test_xr_ds.gt, self.test_xr_ds.pred,
            sig_lap=self.sig_filter_laplacian,
        )

        mse_metrics_pred = compute_metrics(
            self.test_xr_ds.gt, self.test_xr_ds.pred,
        )

        var_mse_pred_lap = 100. * (1. - mse_metrics_lap_pred['mse'] / mse_metrics_lap_pred['var_lap'])
        var_mse_oi_lap = 100. * (1. - mse_metrics_lap_oi['mse'] / mse_metrics_lap_pred['var_lap'])

        # MSE (U,V) fields
        ## compute div/curl/strain metrics
        def compute_var_exp(x, y, dw=0):
            if dw == 0:
                mse = np.nanmean((x-y)**2)
                var = np.nanvar(x)
            else:
                _x = x[:, dw:x.shape[1]-dw, dw:x.shape[2]-dw]
                _y = y[:, dw:x.shape[1]-dw, dw:x.shape[2]-dw]
                mse = np.nanmean((_x-_y)**2)
                var = np.nanvar(_x)

            return 100. * (1. - mse / var)

        def compute_metrics_SSC(u_gt, v_gt, u, v, dw=2):
            if dw == 0 :
                mse_uv = np.nanmean((u_gt - u) ** 2 + (v_gt - v) ** 2 )
                var_uv = np.nanmean((u_gt) ** 2 + (v_gt) ** 2 )
            else:
                _u = u[:, dw:u.shape[1]-dw, dw:u.shape[2]-dw]
                _v = v[:, dw:u.shape[1]-dw, dw:u.shape[2]-dw]
                _u_gt = u_gt[:, dw:u.shape[1]-dw, dw:u.shape[2]-dw]
                _v_gt = v_gt[:, dw:u.shape[1]-dw, dw:u.shape[2]-dw]

                mse_uv = np.nanmean((_u_gt - _u)**2 + (_v_gt - _v)**2)
                var_uv = np.nanmean((_u_gt)**2 + (_v_gt)**2)
            var_mse_uv = 100. * (1. - mse_uv / var_uv)

            _, lamb_x_u, lamb_t_u = psd_based_scores(u, u_gt)
            _, lamb_x_v, lamb_t_v = psd_based_scores(v, v_gt)

            return var_mse_uv, lamb_x_u, lamb_t_u, lamb_x_v, lamb_t_v

        # Metrics for SSC fields
        alpha_uv_geo = 9.81
        lat_rad = np.radians(self.test_lat)
        lon_rad = np.radians(self.test_lon)

        u_geo_gt,v_geo_gt = compute_uv_geo_with_coriolis(
            self.test_xr_ds.gt, lat_rad, lon_rad, alpha_uv_geo=alpha_uv_geo,
            sigma=0.,
        )
        u_geo_oi,v_geo_oi = compute_uv_geo_with_coriolis(
            self.test_xr_ds.oi, lat_rad, lon_rad, alpha_uv_geo=alpha_uv_geo,
            sigma=0.,
        )
        u_geo_rec,v_geo_rec = compute_uv_geo_with_coriolis(
            self.test_xr_ds.pred, lat_rad, lon_rad, alpha_uv_geo=alpha_uv_geo,
            sigma=0.,
        )

        dw_diag = 3
        print('\n\n...... SSH-derived SSC metrics for true SSH')
        (
            var_mse_uv_ssh_gt, lamb_x_u_ssh_gt, lamb_t_u_ssh_gt,
            lamb_x_v_ssh_gt, lamb_t_v_ssh_gt
        ) = compute_metrics_SSC(
            self.test_xr_ds.u_gt, self.test_xr_ds.v_gt, u_geo_gt, v_geo_gt,
            dw=dw_diag,
        )
        print('\n\n...... SSH-derived SSC metrics for DUACS SSH')
        (
            var_mse_uv_ssh_oi, lamb_x_u_ssh_oi, lamb_t_u_ssh_oi,
            lamb_x_v_ssh_oi, lamb_t_v_ssh_oi
        ) = compute_metrics_SSC(
            self.test_xr_ds.u_gt, self.test_xr_ds.v_gt, u_geo_oi, v_geo_oi,
            dw=dw_diag,
        )
        print('\n\n...... SSH-derived SSC metrics for 4dVarNet SSH')
        (
            var_mse_uv_ssh_rec, lamb_x_u_ssh_rec, lamb_t_u_ssh_rec,
            lamb_x_v_ssh_rec, lamb_t_v_ssh_rec
        ) = compute_metrics_SSC(
            self.test_xr_ds.u_gt, self.test_xr_ds.v_gt, u_geo_rec, v_geo_rec,
            dw=dw_diag,
        )
        print('\n\n...... SSH-derived SSC metrics for 4dVarNet SSC')
        var_mse_uv, lamb_x_u, lamb_t_u, lamb_x_v, lamb_t_v = compute_metrics_SSC(
            self.test_xr_ds.u_gt, self.test_xr_ds.v_gt, self.test_xr_ds.pred_u,
            self.test_xr_ds.pred_v, dw=dw_diag,
        )

        print('.....')
        print('.....')
        print('..... Computation of div/curl/strain metrics')
        sig_div_curl = self.sig_filter_div_diag

        flag_heat_equation = False

        if flag_heat_equation:
            iter_heat = 5
            lam = self.sig_filter_div_diag
            t_compute_div_curl_strain_with_lat_lon = TorchComputeDerivativesWithLonLat()
            def heat_equation(u, iter, lam):
                t_u = torch.Tensor(u).view(-1, 1, u.shape[1], u.shape[2])
                f_u = t_compute_div_curl_strain_with_lat_lon.heat_equation(
                    t_u, iter=iter, lam=lam,
                )

                return f_u.detach().numpy().squeeze()

            f_u = heat_equation(
                self.test_xr_ds.u_gt.data, iter=iter_heat, lam=lam,
            )
            f_v = heat_equation(
                self.test_xr_ds.v_gt.data, iter=iter_heat, lam=lam,
            )
            div_gt, curl_gt, strain_gt = compute_div_curl_strain_with_lat_lon(
                f_u, f_v, lat_rad, lon_rad, sigma=0.,
            )

            f_u = heat_equation(
                self.test_xr_ds.pred_u.data, iter=iter_heat, lam=lam,
            )
            f_v = heat_equation(
                self.test_xr_ds.pred_v.data, iter=iter_heat, lam=lam,
            )
            div_uv_rec, curl_uv_rec, strain_uv_rec = compute_div_curl_strain_with_lat_lon(
                f_u, f_v, lat_rad, lon_rad, sigma=0.,
            )
        else :
            div_gt,curl_gt,strain_gt = compute_div_curl_strain_with_lat_lon(
                self.test_xr_ds.u_gt, self.test_xr_ds.v_gt, lat_rad, lon_rad,
                sigma=sig_div_curl,
            )
            div_uv_rec, curl_uv_rec, strain_uv_rec = compute_div_curl_strain_with_lat_lon(
                self.test_xr_ds.pred_u, self.test_xr_ds.pred_v, lat_rad, lon_rad, sigma=sig_div_curl,
            )

        var_mse_div = compute_var_exp(div_gt, div_uv_rec, dw=dw_diag)
        var_mse_curl = compute_var_exp(curl_gt, curl_uv_rec, dw=dw_diag)
        var_mse_strain = compute_var_exp(strain_gt, strain_uv_rec, dw=dw_diag)

        if sig_div_curl > 0.:
            f_ssh_gt = gaussian_filter(self.test_xr_ds.gt, sigma=sig_div_curl)
            f_ssh_oi = gaussian_filter(
                self.test_xr_ds.oi, sigma=4.*sig_div_curl,
            )
            f_ssh_rec = gaussian_filter(
                self.test_xr_ds.pred, sigma=sig_div_curl,
            )
        else:
            f_ssh_gt = self.test_xr_ds.gt
            f_ssh_oi = self.test_xr_ds.oi
            f_ssh_rec = self.test_xr_ds.pred

        f_u_geo_gt, f_v_geo_gt = compute_uv_geo_with_coriolis(
            f_ssh_gt, lat_rad, lon_rad, alpha_uv_geo=alpha_uv_geo, sigma=0.,
        )
        f_u_geo_oi, f_v_geo_oi = compute_uv_geo_with_coriolis(
            f_ssh_oi, lat_rad, lon_rad, alpha_uv_geo=alpha_uv_geo, sigma=0.,
        )
        f_u_geo_rec, f_v_geo_rec = compute_uv_geo_with_coriolis(
            f_ssh_rec, lat_rad, lon_rad, alpha_uv_geo=alpha_uv_geo, sigma=0.,
        )

        div_geo_gt, curl_geo_gt, strain_geo_gt = compute_div_curl_strain_with_lat_lon(
            f_u_geo_gt, f_v_geo_gt, lat_rad, lon_rad, sigma=0.,
        )
        div_geo_oi, curl_geo_oi, strain_geo_oi = compute_div_curl_strain_with_lat_lon(
            f_u_geo_oi, f_v_geo_oi, lat_rad, lon_rad, sigma=0.,
        )
        div_geo_rec,curl_geo_rec,strain_geo_rec = compute_div_curl_strain_with_lat_lon(
            f_u_geo_rec, f_v_geo_rec, lat_rad, lon_rad, sigma=0.,
        )

        var_mse_div_ssh_gt = compute_var_exp(div_gt, div_geo_gt, dw=dw_diag)
        var_mse_curl_ssh_gt = compute_var_exp(curl_gt, curl_geo_gt, dw=dw_diag)
        var_mse_strain_ssh_gt = compute_var_exp(
            strain_gt, strain_geo_gt, dw=dw_diag,
        )

        var_mse_div_ssh_oi = compute_var_exp(div_gt, div_geo_oi, dw=dw_diag)
        var_mse_curl_ssh_oi = compute_var_exp(curl_gt, curl_geo_oi, dw=dw_diag)
        var_mse_strain_ssh_oi = compute_var_exp(
            strain_gt, strain_geo_oi, dw=dw_diag,
        )

        var_mse_div_ssh_rec = compute_var_exp(div_gt, div_geo_rec, dw=dw_diag)
        var_mse_curl_ssh_rec = compute_var_exp(
            curl_gt, curl_geo_rec, dw=dw_diag,
        )
        var_mse_strain_ssh_rec = compute_var_exp(
            strain_gt, strain_geo_rec, dw = dw_diag,
        )

        md = {
            f'{log_pref}_spatial_res': float(spatial_res_model),
            f'{log_pref}_spatial_res_imp': (
                float(spatial_res_model / spatial_res_oi)
            ),
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_lambda_x_u': lamb_x_u,
            f'{log_pref}_lambda_t_u': lamb_t_u,
            f'{log_pref}_lambda_x_v': lamb_x_v,
            f'{log_pref}_lambda_t_v': lamb_t_v,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
            f'{log_pref}_var_mse_vs_oi': float(var_mse_pred_vs_oi),
            f'{log_pref}_var_mse_grad_vs_oi': float(var_mse_grad_pred_vs_oi),
            f'{log_pref}_var_mse_lap_pred': float(var_mse_pred_lap),
            f'{log_pref}_var_mse_lap_oi': float(var_mse_oi_lap),
            f'{log_pref}_var_mse_uv_gt': float(var_mse_uv_ssh_gt),
            f'{log_pref}_var_mse_uv_oi': float(var_mse_uv_ssh_oi),
            f'{log_pref}_var_mse_uv_pred': float(var_mse_uv_ssh_rec),
            f'{log_pref}_var_mse_uv': float(var_mse_uv),
            f'{log_pref}_var_mse_div_ssh_gt': float(var_mse_div_ssh_gt),
            f'{log_pref}_var_mse_div_oi': float(var_mse_div_ssh_oi),
            f'{log_pref}_var_mse_div_pred': float(var_mse_div_ssh_rec),
            f'{log_pref}_var_mse_div': float(var_mse_div),
            f'{log_pref}_var_mse_strain_ssh_gt': float(var_mse_strain_ssh_gt),
            f'{log_pref}_var_mse_strain_oi': float(var_mse_strain_ssh_oi),
            f'{log_pref}_var_mse_strain_pred': float(var_mse_strain_ssh_rec),
            f'{log_pref}_var_mse_strain': float(var_mse_strain),
            f'{log_pref}_var_mse_curl_ssh_gt': float(var_mse_curl_ssh_gt),
            f'{log_pref}_var_mse_curl_oi': float(var_mse_curl_ssh_oi),
            f'{log_pref}_var_mse_curl_pred': float(var_mse_curl_ssh_rec),
            f'{log_pref}_var_mse_curl': float(var_mse_curl),
        } | mdf.to_dict()
        print(pd.DataFrame([md]).T.to_markdown())

        return md

    def diag_epoch_end(self, outputs, log_pref='test'):
        full_outputs = self.gather_outputs(outputs, log_pref=log_pref)
        if full_outputs is None:
            print("full_outputs is None on", self.global_rank)
            return

        if log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise ValueError(f'Invalid argument: log_pref={log_pref}')

        self.test_xr_ds = self.build_test_xr_ds(
            full_outputs,
            diag_ds=diag_ds,
            version=self.version_build_test_xr_ds,
            use_sst=self.use_sst,
        )

        self.x_gt = self.test_xr_ds.gt.data#[2:42,:,:]
        self.obs_inp = self.test_xr_ds.obs_inp.data#[2:42,:,:]
        self.x_oi = self.test_xr_ds.oi.data#[2:42,:,:]
        self.x_rec = self.test_xr_ds.pred.data#[2:42,:,:]

        self.u_gt = self.test_xr_ds.u_gt.data
        self.v_gt = self.test_xr_ds.v_gt.data
        self.u_rec = self.test_xr_ds.pred_u.data#[2:42,:,:]
        self.v_rec = self.test_xr_ds.pred_v.data#[2:42,:,:]

        self.x_rec_ssh = self.x_rec

        def extract_seq(key, dw=None):
            seq = torch.cat([chunk[key] for chunk in outputs]).numpy()

            if not dw:
                # Determine dw such that:
                #     seq's (lat, lon) == u_gt's (lat, lon)
                # Here, we suppose lat == lon so we take only lon,
                # whence the index -1.
                dw = abs(seq.shape[-1] - self.u_gt.shape[-1]) // 2

            seq = seq[:, :, dw:seq.shape[2]-dw, dw:seq.shape[2]-dw]
            return seq

        self.test_coords = self.test_xr_ds.coords

        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data

        md = self.sla_uv_diag(t_idx=3, log_pref=log_pref)

        self.latest_metrics.update(md)
        self.logger.log_metrics(md, step=self.current_epoch)

        if self.scale_dwscaling_sst > 1.:
            print(
                f'.... Using downscaled SST by {self.scale_dwscaling_sst:.1f}'
            )
        print(f'..... Log directory: {self.logger.log_dir}')

        if self.save_rec_netcdf:
            path_save1 = self.hparams.path_save_netcdf.replace(
                '.ckpt', '_res_4dvarnet_all.nc',
            )
            path_save1 = f'{self.logger.log_dir}/{path_save1}'
            if not self.use_sst:
                # self.x_sst_feat_ssh = extract_seq('sst_feat')
                # self.x_sst_feat_ssh = self.x_sst_feat_ssh[:self.x_rec_ssh.shape[0], :, :, :]
                # sst_feat = self.x_sst_feat_ssh[:, 0, :, :].reshape(
                #     self.x_sst_feat_ssh.shape[0],
                #     1,
                #     self.x_sst_feat_ssh.shape[2],
                #     self.x_sst_feat_ssh.shape[3]
                # )
                sst_feat = None
            else:
                self.x_sst_feat_ssh = extract_seq('sst_feat')

                self.x_gt = extract_seq('gt')
                self.x_gt = self.x_gt[:, int(self.hparams.dT/2), :, :]

                self.obs_inp = extract_seq('obs_inp')
                self.obs_inp = self.obs_inp[:, int(self.hparams.dT/2), :, :]

                self.x_oi = extract_seq('oi')
                self.x_oi = self.x_oi[:, int(self.hparams.dT/2), :, :]

                self.x_rec = extract_seq('pred')
                self.x_rec = self.x_rec[:, int(self.hparams.dT/2), :, :]
                self.x_rec_ssh = self.x_rec

                sst_feat = self.x_sst_feat_ssh

            print(
                f'... Save nc file with all results: {path_save1}',
                f'    gt\t\t{self.x_gt.shape}',
                f'    obs\t\t{self.obs_inp.shape}',
                f'    oi\t\t{self.x_oi.shape}',
                f'    pred\t{self.x_rec_ssh.shape}',
                f'    u_gt\t{self.u_gt.shape}',
                f'    v_gt\t{self.v_gt.shape}',
                f'    u_pred\t{self.u_rec.shape}',
                f'    v_pred\t{self.v_rec.shape}',
                f'    lon\t\t{self.test_lon.shape}',
                f'    lat\t\t{self.test_lat.shape}',
                f'    time\t{self.test_dates.shape}',
                sep='\n'
            )
            if sst_feat is not None:
                print(f'    sst_feat\t{self.x_sst_feat_ssh.shape}')

            save_netcdf_uv(
                saved_path1=path_save1,
                gt=self.x_gt,
                obs=self.obs_inp,
                oi=self.x_oi,
                pred=self.x_rec_ssh,
                u_gt=self.u_gt,
                v_gt=self.v_gt,
                u_pred=self.u_rec,
                v_pred=self.v_rec,
                sst_feat=sst_feat,
                lon=self.test_lon,
                lat=self.test_lat,
                time=self.test_dates,
            )

    def teardown(self, stage='test'):
        self.logger.log_hyperparams(
            {**self.hparams}, self.latest_metrics
        )

    def get_init_state(self, batch, state=(None,), mask_sampling=None):
        if state[0] is not None:
            return state[0]

        (
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt,
            lat, lon, gx, gy,ais_dat
        ) = batch

        if mask_sampling:
            init_u = mask_sampling * u_gt
            init_v = mask_sampling * v_gt
        else:
            init_u = torch.zeros_like(targets_GT)
            init_v = torch.zeros_like(targets_GT)

        if self.aug_state :

            print("ligne 1427,",targets_OI.size(),inputs_obs.size(), init_u.size(), init_v.size(), inputs_Mask.size())
            targets_OI = targets_OI.permute(0,3,1,2)
            init_u = init_u.permute(0,3,1,2)
            init_v = init_v.permute(0,3,1,2)
            #inputs_obs = inputs_obs.permute(0,2,3,1)
            #inputs_Mask = inputs_Mask.permute(0,2,3,1)
            init_state = torch.cat(
                (
                    targets_OI,
                    inputs_Mask * (inputs_obs - targets_OI),
                    inputs_Mask * (inputs_obs - targets_OI),
                    init_u,
                    init_v,
                ),
                dim=1,
            )
        else:
            init_state = torch.cat(
                (
                    targets_OI,
                    inputs_Mask * (inputs_obs - targets_OI),
                    init_u,
                    init_v,
                ),
                dim=1,
            )

        if self.use_sst_state :
            init_state = torch.cat(
                (init_state, sst_gt,),
                dim=1,
            )
        return init_state

    def loss_ae(self, state_out):
        return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def sla_loss(self, gt, out):
        print(out.size(),gt.size(),"ligne 1460")
        out = out.permute(0,3,1,2)
        g_outputs_x, g_outputs_y = self.gradient_img(out)
        g_gt_x, g_gt_y = self.gradient_img(gt)
	
	
	
        #print("ligne 1463", out.size(),gt.size())
        #out = out.permute(0,3,1,2)
        loss = NN_4DVar.compute_spatio_temp_weighted_loss(
            (out - gt), self.patch_weight,
        )
        #print(g_outputs_x.size(),g_gt_x.size())
        loss_grad = (
            NN_4DVar.compute_spatio_temp_weighted_loss(
                g_outputs_x - g_gt_x, self.grad_crop(self.patch_weight),
            )
            + NN_4DVar.compute_spatio_temp_weighted_loss(
                g_outputs_y - g_gt_y, self.grad_crop(self.patch_weight),
            )
        )

        return loss, loss_grad

    def compute_uv_from_ssh(self, ssh, lat_rad, lon_rad, sigma=0.):
        ssh = np.sqrt(self.var_Tr) * ssh + self.mean_Tr
        u_geo, v_geo = self.compute_derivativeswith_lon_lat.compute_geo_velocities(
            ssh, lat_rad, lon_rad, sigma=0.,
        )

        return u_geo / np.sqrt(self.var_tr_uv), v_geo / np.sqrt(self.var_tr_uv)

    def compute_geo_factor(self, outputs, lat_rad, lon_rad, sigma=0.):
        return self.compute_derivativeswith_lon_lat.compute_geo_factor(
            outputs, lat_rad, lon_rad,sigma=0.,
        )

    def compute_div_curl_strain(self, u, v, lat_rad, lon_rad, sigma=0.):
        if sigma > 0:
            u = self.compute_derivativeswith_lon_lat.heat_equation_all_channels(
                u, iter=5, lam=self.sig_filter_div_diag,
            )
            v = self.compute_derivativeswith_lon_lat.heat_equation_all_channels(
                v, iter=5, lam=self.sig_filter_div_diag,
            )

        div_gt, curl_gt, strain_gt = (
            self.compute_derivativeswith_lon_lat.compute_div_curl_strain(
                u, v, lat_rad, lon_rad, sigma=self.sig_filter_div_diag,
            )
        )

        return div_gt, curl_gt, strain_gt

    def div_loss(self, gt, out):
        if self.type_div_train_loss == 0:
            return NN_4DVar.compute_spatio_temp_weighted_loss(
                (out - gt), self.patch_weight[:, 1:-1, 1:-1],
            )
        else:
            return NN_4DVar.compute_spatio_temp_weighted_loss(
                1.e4 * (out - gt), self.patch_weight,
            )

    def strain_loss(self, gt, out):
        return NN_4DVar.compute_spatio_temp_weighted_loss(
            1.e4 * (out - gt ), self.patch_weight,
        )

    def uv_loss(self, gt, out):
        #print("ligne 1530",gt[0].size(),out[0].size())
        out[0] = out[0].permute(0,3,1,2)
        out[1] = out[1].permute(0,3,1,2)
        loss = NN_4DVar.compute_spatio_temp_weighted_loss(
            (out[0] - gt[0]), self.patch_weight,
        )
        loss += NN_4DVar.compute_spatio_temp_weighted_loss(
            (out[1] - gt[1]), self.patch_weight,
        )

        return loss

    def reg_loss(self, y_gt, oi, out, out_lr, out_lrhr):
        l_ae = self.loss_ae(out_lrhr)
        l_ae_gt = self.loss_ae(y_gt)
        l_sr = NN_4DVar.compute_spatio_temp_weighted_loss(
            out_lr - oi, self.patch_weight,
        )

        gt_lr = self.model_LR(oi)
        out_lr_bis = self.model_LR(out)
        l_lr = NN_4DVar.compute_spatio_temp_weighted_loss(
            out_lr_bis - gt_lr, self.model_LR(self.patch_weight),
        )

        return l_ae, l_ae_gt, l_sr, l_lr

    def dwn_sample_batch(self, batch, scale = 1.):
        if scale > 1.:
            if not self.use_sst:
                (
                    targets_OI, inputs_Mask, inputs_obs,
                    targets_GT, u_gt, v_gt, lat, lon
                ) = batch
            else:
                (
                    targets_OI, inputs_Mask, inputs_obs,
                    targets_GT, sst_gt, u_gt, v_gt, lat, lon
                ) = batch

                scale_dwscaling = int(self.scale_dwscaling)
                targets_OI = torch.nn.functional.avg_pool2d(
                    targets_OI, (scale_dwscaling, scale_dwscaling),
                )
                targets_GT = torch.nn.functional.avg_pool2d(
                    targets_GT, (scale_dwscaling, scale_dwscaling),
                )
                u_gt = torch.nn.functional.avg_pool2d(
                    u_gt, (scale_dwscaling, scale_dwscaling),
                )
                v_gt = torch.nn.functional.avg_pool2d(
                    v_gt, (scale_dwscaling, scale_dwscaling),
                )
                if self.use_sst:
                    sst_gt = torch.nn.functional.avg_pool2d(
                        sst_gt, (scale_dwscaling, scale_dwscaling),
                    )

                targets_GT = targets_GT.detach()
                sst_gt = sst_gt.detach()
                u_gt = u_gt.detach()
                v_gt = v_gt.detach()

                inputs_Mask = inputs_Mask.detach()
                inputs_obs = inputs_obs.detach()

                inputs_Mask = torch.nn.functional.avg_pool2d(
                    inputs_Mask.float(), (scale_dwscaling, scale_dwscaling),
                )
                inputs_obs = torch.nn.functional.avg_pool2d(
                    inputs_obs, (scale_dwscaling, scale_dwscaling),
                )

                inputs_obs  = inputs_obs / (inputs_Mask + 1e-7)
                inputs_Mask = (inputs_Mask > 0.).float()

                lat = torch.nn.functional.avg_pool1d(
                    lat.view(-1, 1, lat.size(1)), scale_dwscaling,
                )
                lon = torch.nn.functional.avg_pool1d(
                    lon.view(-1, 1, lon.size(1)), scale_dwscaling,
                )

                lat = lat.view(-1, lat.size(2))
                lon = lon.view(-1, lon.size(2))

            if not self.use_sst:
                return (
                    targets_OI, inputs_Mask, inputs_obs,
                    targets_GT, u_gt, v_gt, lat, lon
                )
            else:
                return (
                    targets_OI, inputs_Mask, inputs_obs,
                    targets_GT, sst_gt, u_gt, v_gt, lat, lon
                )
        else:
            return batch

    def pre_process_batch(self, batch):
        if self.scale_dwscaling > 1.0:
            _batch = self.dwn_sample_batch(batch, scale=self.scale_dwscaling)
        else:
            _batch = batch

        if not self.use_sst:
            (
                targets_OI, inputs_Mask, inputs_obs,
                targets_GT, u_gt, v_gt, lat, lon, ais_dat
            ) = _batch
        else:
            (
                targets_OI, inputs_Mask, inputs_obs,
                targets_GT, sst_gt, u_gt, v_gt, lat, lon, ais_dat
            ) = _batch

        if self.scale_dwscaling_sst > 1:
            scale_dwscaling_sst = int(self.scale_dwscaling_sst)
            sst_gt = torch.nn.functional.avg_pool2d(
                sst_gt, (scale_dwscaling_sst, scale_dwscaling_sst),
            )
            sst_gt = torch.nn.functional.interpolate(
                sst_gt, scale_factor=self.scale_dwscaling_sst, mode='bicubic',
            )

        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        u_gt_wo_nan = u_gt.where(~u_gt.isnan(), torch.zeros_like(u_gt))
        v_gt_wo_nan = v_gt.where(~v_gt.isnan(), torch.zeros_like(u_gt))

        if not self.use_sst:
            sst_gt = None

        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT_wo_nan)

        # lat/lon in radians
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        if self.use_lat_lon_in_obs_model:
            self.model.model_H.lat_rad = lat_rad
            self.model.model_H.lon_rad = lon_rad

        return (
            targets_OI, inputs_Mask, inputs_obs, targets_GT_wo_nan, sst_gt,
            u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad, g_targets_GT_x,
            g_targets_GT_y,ais_dat
        )

    def get_obs_and_mask(
        self, targets_OI, inputs_Mask, inputs_obs, sst_gt, u_gt_wo_nan,
        v_gt_wo_nan,
    ):
        if self.model_sampling_uv is not None:
            w_sampling_uv = self.model_sampling_uv(sst_gt)
            w_sampling_uv = w_sampling_uv[1]

            mask_sampling_uv = 1. - torch.nn.functional.threshold(
                1.0 - w_sampling_uv, 0.9, 0.
            )
            
            print("ligne 1678,",targets_OI.size(),inputs_obs.size(), u_gt_wo_nan.size(), inputs_Mask.size())
            #inputs_obs = inputs_obs.(0,2,3,1)
            #inputs_Mask = inputs_Mask.permute(0,2,3,1)
            obs = torch.cat(
                (
                    targets_OI,
                    inputs_Mask * (inputs_obs - targets_OI),
                    u_gt_wo_nan,
                    v_gt_wo_nan,
                ), dim=1,
            )
        else:
            print("ligne 1690,",targets_OI.size(),inputs_obs.size(), u_gt_wo_nan.size(), inputs_Mask.size())
            targets_OI = targets_OI.permute(0,3,1,2)
            #inputs_obs = inputs_obs.permute(0,2,3,1)
            #inputs_Mask = inputs_Mask.permute(0,2,3,1)
            mask_sampling_uv = torch.zeros_like(u_gt_wo_nan).permute(0,3,1,2)
            w_sampling_uv = None
            obs = torch.cat(
                (
                    targets_OI,
                    inputs_Mask * (inputs_obs - targets_OI),
                    0. * targets_OI,
                    0. * targets_OI,
                ), dim=1,
            )

        new_masks = torch.cat(
            (
                torch.ones_like(inputs_Mask),
                inputs_Mask,
                mask_sampling_uv,
                mask_sampling_uv,
            ), dim=1,
        )

        if self.aug_state:
            obs = torch.cat((obs, 0. * targets_OI,), dim=1)
            new_masks = torch.cat(
                (new_masks, torch.zeros_like(inputs_Mask)), dim=1,
            )

        if self.use_sst_state:
            obs = torch.cat((obs, sst_gt,), dim=1)
            new_masks = torch.cat(
                (new_masks, torch.ones_like(inputs_Mask)), dim=1,
            )

        if self.use_sst_obs:
            new_masks = [new_masks, torch.ones_like(sst_gt)]
            obs = [obs, sst_gt]

        return obs, new_masks, w_sampling_uv, mask_sampling_uv

    def run_model(
        self, state, obs, new_masks, state_init, lat_rad, lon_rad, phase
    ):
        state = torch.autograd.Variable(state, requires_grad=True)

        outputs, hidden_new, cell_new, normgrad = self.model(
            state, obs, new_masks, *state_init[1:],
        )

        if phase in ('val', 'test'):
            outputs = outputs.detach()

        dT = self.hparams.dT
        outputsSLRHR = outputs
        outputsSLR = outputs[:, 0:dT, :, :]
        if self.aug_state:
            outputs = outputsSLR + outputsSLRHR[:, 2*dT:3*dT, :, :]
            outputs_u = outputsSLRHR[:, 3*dT:4*dT, :, :]
            outputs_v = outputsSLRHR[:, 4*dT:5*dT, :, :]
        else:
            outputs = outputsSLR + outputsSLRHR[:, dT:2*dT, :, :]
            outputs_u = outputsSLRHR[:, 2*dT:3*dT, :, :]
            outputs_v = outputsSLRHR[:, 3*dT:4*dT, :, :]

        # denormalize ssh
        u_geo_rec, v_geo_rec = self.compute_uv_from_ssh(
            outputs, lat_rad, lon_rad, sigma=0.,
        )

        # U, V prediction
        if self.residual_wrt_geo_velocities in (1, 3):
            alpha_uv_geo = 0.05
            outputs_u = alpha_uv_geo * outputs_u + u_geo_rec
            outputs_v = alpha_uv_geo * outputs_v + v_geo_rec
        elif self.residual_wrt_geo_velocities in (2, 4):
            u_geo_factor, v_geo_factor = self.compute_geo_factor(
                outputs, lat_rad, lon_rad, sigma=0.,
            )

            alpha_uv_geo = 0.05
            outputs_u = alpha_uv_geo * u_geo_factor * outputs_u
            outputs_v = alpha_uv_geo * v_geo_factor * outputs_v

        return (
            outputs, outputs_u, outputs_v, outputsSLRHR,
            outputsSLR, hidden_new, cell_new, normgrad
        )

    def compute_reg_loss(
        self, targets_OI, targets_GT_wo_nan, sst_gt, u_gt_wo_nan, v_gt_wo_nan,
        outputs, outputsSLR, outputsSLRHR, phase,
    ):
        if phase in ('val', 'test'):
            self.patch_weight = self.patch_weight_train
	
        print("ligne 1790", targets_OI.size(), targets_GT_wo_nan.size(), outputsSLR.size())
        if outputsSLR is not None:
            targets_OI = targets_OI.permute(0,3,1,2)
            targets_GT_wo_nan = targets_GT_wo_nan.permute(0,3,1,2)
            yGT = torch.cat(
                (
                    targets_OI,
                    targets_GT_wo_nan - outputsSLR,
                ), dim=1,
            )

            if self.aug_state:
                yGT = torch.cat((yGT, targets_GT_wo_nan - outputsSLR), dim=1)
            print("ligne 1803",yGT.size(),u_gt_wo_nan.size())
            u_gt_wo_nan = u_gt_wo_nan.permute(0,3,1,2)
            v_gt_wo_nan = v_gt_wo_nan.permute(0,3,1,2)
            
            yGT = torch.cat((yGT, u_gt_wo_nan, v_gt_wo_nan), dim=1)

            if self.use_sst_state :
                yGT = torch.cat((yGT, sst_gt), dim=1)

            loss_AE, loss_AE_GT, loss_SR, loss_LR =  self.reg_loss(
                yGT, targets_OI, outputs, outputsSLR, outputsSLRHR
            )
        else:
           loss_AE = 0.
           loss_AE_GT = 0.
           loss_SR = 0.
           loss_LR = 0.

        return loss_AE, loss_AE_GT, loss_SR, loss_LR

    def reinterpolate_outputs(self, outputs, outputs_u, outputs_v, batch):
        if not self.use_sst:
            (
                targets_OI, inputs_Mask, inputs_obs,
                targets_GT, u_gt, v_gt, lat, lon
            ) = batch
            sst_gt = None
        else:
            (
                targets_OI, inputs_Mask, inputs_obs,
                targets_GT, sst_gt, u_gt, v_gt, lat, lon
            ) = batch

        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        outputs = torch.nn.functional.interpolate(
            outputs, scale_factor=self.scale_dwscaling, mode='bicubic',
        )
        outputs_u = torch.nn.functional.interpolate(
            outputs_u, scale_factor=self.scale_dwscaling, mode='bicubic',
        )
        outputs_v = torch.nn.functional.interpolate(
            outputs_v, scale_factor=self.scale_dwscaling, mode='bicubic',
        )

        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        u_gt_wo_nan = u_gt.where(~u_gt.isnan(), torch.zeros_like(u_gt))
        v_gt_wo_nan = v_gt.where(~v_gt.isnan(), torch.zeros_like(u_gt))

        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)

        self.patch_weight = self.patch_weight_diag

        return (
            targets_OI, targets_GT_wo_nan, sst_gt, u_gt_wo_nan, v_gt_wo_nan,
            lat_rad, lon_rad, outputs, outputs_u, outputs_v, g_targets_GT_x,
            g_targets_GT_y
        )

    def compute_rec_loss(
        self, targets_GT_wo_nan, u_gt_wo_nan, v_gt_wo_nan, outputs, outputs_u,
        outputs_v, lat_rad, lon_rad, phase,
    ):
        flag_display_loss = False

        # median filter
        if self.median_filter_width > 1:
            outputs = kornia.filters.median_blur(
                outputs, (self.median_filter_width, self.median_filter_width),
            )

        # MSE loss for ssh and (u, v) components
        loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
        loss_uv = self.uv_loss(
            [outputs_u, outputs_v], [u_gt_wo_nan, v_gt_wo_nan],
        )

        # MSE for SSH-derived (u,v) fields
        # denormalize ssh
        u_geo_rec, v_geo_rec = self.compute_uv_from_ssh(
            outputs, lat_rad, lon_rad, sigma=0.,
        )
        u_geo_gt, v_geo_gt = self.compute_uv_from_ssh(
            targets_GT_wo_nan, lat_rad, lon_rad, sigma=0.,
        )

        loss_uv_geo = self.uv_loss([u_geo_rec, v_geo_rec], [u_geo_gt, v_geo_gt])

        if self.type_div_train_loss == 0:
            div_rec = self.compute_div(outputs_u, outputs_v)
            div_gt = self.compute_div(u_gt_wo_nan, v_gt_wo_nan)

            loss_div = self.div_loss(div_rec, div_gt)
            loss_strain = 0.
            if flag_display_loss:
                print(f'\n..  loss div = {self.hparams.alpha_mse_div*loss_div}')
        else:
            if phase in ('val', 'test'):
                div_gt, curl_gt, strain_gt = self.compute_div_curl_strain(
                    u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad,
                    sigma=self.sig_filter_div_diag,
                )
                div_rec, curl_rec, strain_rec = self.compute_div_curl_strain(
                    outputs_u, outputs_v, lat_rad, lon_rad,
                    sigma=self.sig_filter_div_diag,
                )

            else:
                div_gt, curl_gt, strain_gt = self.compute_div_curl_strain(
                    u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad,
                    sigma=self.sig_filter_div,
                )
                div_rec, curl_rec, strain_rec = self.compute_div_curl_strain(
                    outputs_u, outputs_v, lat_rad, lon_rad,
                    sigma=self.sig_filter_div,
                )

            loss_div = self.div_loss(div_rec, div_gt)
            loss_strain = self.strain_loss(strain_rec, strain_gt)

            if flag_display_loss:
                print(
                    f'\n..  loss div = {self.hparams.alpha_mse_div*loss_div}',
                    f'..  loss strain = {self.hparams.alpha_mse_strain*loss_strain}',
                    sep='\n',
                )

        if flag_display_loss:
            print(
                f'..  loss ssh = {self.hparams.alpha_mse_ssh * loss_All}',
                f'..  loss gssh = {self.hparams.alpha_mse_gssh * loss_GAll}',
                f'..  loss uv = {self.hparams.alpha_mse_uv * loss_uv}',
                sep='\n',
            )

        return loss_All, loss_GAll, loss_uv, loss_uv_geo, loss_div, loss_strain

    def compute_loss(self, batch, phase, state_init=(None,)):
        flag_display_loss = False

        _batch = self.pre_process_batch(batch)
        (
            targets_OI, inputs_Mask, inputs_obs, targets_GT_wo_nan,
            sst_gt, u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad,
            g_targets_GT_x, g_targets_GT_y, ais_dat
        ) = _batch

        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_OI),
                torch.cat(
                    (
                        torch.zeros_like(targets_OI),
                        torch.zeros_like(targets_OI),
                        torch.zeros_like(targets_OI),
                    ), dim=1,
                ),
                dict([
                    ('mse', 0.),
                    ('mseGrad', 0.),
                    ('meanGrad', 1.),
                    ('mseOI', 0.),
                    ('mse_uv', 0.),
                    ('mseGOI', 0.),
                    ('l0_samp', 0.),
                    ('l1_samp', 0.),
                ])
            )

        # intial state
        state = self.get_init_state(_batch, state_init)

        # obs and mask data
        obs, new_masks, w_sampling_uv, mask_sampling_uv = self.get_obs_and_mask(
            targets_OI, inputs_Mask, inputs_obs, sst_gt, u_gt_wo_nan, v_gt_wo_nan
        )

        # run forward_model
        with torch.set_grad_enabled(True):
            if self.hparams.n_grad > 0:
                (
                    outputs, outputs_u, outputs_v, outputsSLRHR, outputsSLR,
                    hidden_new, cell_new, normgrad
                ) = self.run_model(
                    state, obs, new_masks, state_init, lat_rad, lon_rad, phase,
                )
            else:
                dT = self.hparams.dT
                outputs = self.model.phi_r(obs)

                outputs_u = outputs[:, 1*dT:2*dT, :, :]
                outputs_v = outputs[:, 2*dT:3*dT, :, :]
                outputs = outputs[:, :dT, :, :]

                outputsSLR = None
                outputsSLRHR = None
                hidden_new = None
                cell_new = None
                normgrad = 0.

            # projection losses
            loss_AE, loss_AE_GT, loss_SR, loss_LR = self.compute_reg_loss(
                targets_OI, targets_GT_wo_nan, sst_gt, u_gt_wo_nan,
                v_gt_wo_nan, outputs, outputsSLR, outputsSLRHR, phase,
            )

            # re-interpolate at full-resolution field during test/val epoch
            if (phase in ('val', 'test')) and (self.scale_dwscaling > 1.0):
                _t = self.reinterpolate_outputs(
                    outputs, outputs_u, outputs_v, batch,
                )
                (
                    targets_OI, targets_GT_wo_nan, sst_gt, u_gt_wo_nan,
                    v_gt_wo_nan, lat_rad, lon_rad, outputs, outputs_u,
                    outputs_v, g_targets_GT_x, g_targets_GT_y
                ) = _t

            # reconstruction losses
            (
                loss_All, loss_GAll, loss_uv, loss_uv_geo, loss_div, loss_strain
            ) = self.compute_rec_loss(
                targets_GT_wo_nan, u_gt_wo_nan, v_gt_wo_nan, outputs, outputs_u,
                outputs_v, lat_rad, lon_rad, phase,
            )

            loss_OI, loss_GOI = self.sla_loss(targets_OI, targets_GT_wo_nan)

            if self.model_sampling_uv is not None:
                dT = self.hparams.dT
                _val = float(dT / (dT - int(dT/2)))

                loss_l1_sampling_uv = _val * torch.mean(w_sampling_uv)
                loss_l1_sampling_uv = torch.nn.functional.relu(
                    loss_l1_sampling_uv - self.hparams.thr_l1_sampling_uv
                )
                loss_l0_sampling_uv = _val * torch.mean(mask_sampling_uv)

            ######################################
            # Computation of total loss
            # reconstruction loss
            loss = self.hparams.alpha_mse_ssh * loss_All
            loss += self.hparams.alpha_mse_gssh * loss_GAll
            loss += self.hparams.alpha_mse_uv * loss_uv
            loss += self.hparams.alpha_mse_uv_geo * loss_uv_geo

            if self.hparams.alpha_mse_div > 0.:
                loss += self.hparams.alpha_mse_div * loss_div
            if self.hparams.alpha_mse_strain > 0.:
                loss += self.hparams.alpha_mse_strain * loss_strain

            # regularization loss
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss += self.hparams.alpha_lr*loss_LR + self.hparams.alpha_sr*loss_SR

            # sampling loss
            if self.model_sampling_uv is not None:
                loss += self.hparams.alpha_sampling_uv * loss_l1_sampling_uv

            if flag_display_loss :
                print('..  loss = %e' %loss )

            ######################################
            # metrics
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(
                torch.hypot(g_targets_GT_x, g_targets_GT_y),
                self.grad_crop(self.patch_weight),
            )
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            mse_uv = loss_uv.detach()

            mse_div = loss_div.detach()
            if self.model_sampling_uv is not None:
                l1_samp = loss_l1_sampling_uv.detach()
                l0_samp = loss_l0_sampling_uv.detach()
            else:
                l0_samp = 0. * mse
                l1_samp = 0. * mse

            metrics = dict([
                ('mse', mse),
                ('mse_uv', mse_uv),
                ('mse_div', mse_div),
                ('mseGrad', mseGrad),
                ('meanGrad', mean_GAll),
                ('mseOI', loss_OI.detach()),
                ('mseGOI', loss_GOI.detach()),
                ('l0_samp', l0_samp),
                ('l1_samp', l1_samp),
            ])

        if phase in ('val', 'test') and self.use_sst:
            _sub_sst = sst_gt[:, int(self.hparams.dT/2), :, :]
            out_feat = _sub_sst.view(-1, 1, sst_gt.size(2),sst_gt.size(3))

            if self.use_sst_obs:
                out_feat = torch.cat(
                    (out_feat, self.model.model_H.extract_sst_feature(sst_gt)),
                    dim = 1,
                )
                ssh_feat = self.model.model_H.extract_state_feature(outputsSLRHR)

                if self.scale_dwscaling > 1:
                    ssh_feat = torch.nn.functional.interpolate(
                        ssh_feat,
                        scale_factor=self.scale_dwscaling,
                        mode='bicubic',
                    )
                out_feat = torch.cat((out_feat, ssh_feat), dim=1)

            if self.model_sampling_uv is not None:
                out_feat = torch.cat((out_feat, w_sampling_uv), dim=1)

            return (
                loss,
                [outputs, outputs_u, outputs_v],
                [outputsSLRHR, hidden_new, cell_new, normgrad],
                metrics, out_feat
            )
        else:
            return (
                loss,
                [outputs, outputs_u, outputs_v],
                [outputsSLRHR, hidden_new, cell_new, normgrad],
                metrics
            )


class LitModelCycleLR(LitModelUV):
    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.CyclicLR(
                opt, **self.hparams.cycle_lr_kwargs,
            ),
            'monitor': 'val_loss',
        }

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            if (
                (self.current_epoch in self.hparams.iter_update)
                and (self.current_epoch > 0)
            ):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print(
                    f'... Update Iterations number #{self.current_epoch}: '
                    + f'NGrad = {self.hparams.nb_grad_update[indx]} -- '
                )

                self.hparams.n_grad = self.hparams.nb_grad_update[indx]
                self.model.n_grad = self.hparams.n_grad
                print("ngrad iter", self.model.n_grad)


class ModelSamplingFromSST(torch.nn.Module):
    def __init__(self, dT, nb_feat=10, thr=.1):
        super().__init__()
        self.dT = dT
        self.Tr = torch.nn.Threshold(thr, 0.)
        self.S = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(int(dT/2), nb_feat, (3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(
            nb_feat, dT-int(dT/2), (1, 1), padding=0, bias=True,
        )

    def forward(self, y):
        yconv = self.conv2(F.relu(self.conv1(y[:, :int(self.dT/2), :, :])))

        yout1 = self.S(yconv)
        yout1 = torch.cat(
            (torch.zeros_like(y[:, :int(self.dT/2), :, :]), yout1), dim=1,
        )
        yout2 = self.Tr(yout1)

        return [yout1, yout2]


class Model_HwithSSTBN_nolin_tanh_withlatlon(torch.nn.Module):
    def __init__(
        self, shape_data, dT=5, dim=5, width_kernel=3, padding_mode='reflect',
        type_wgeo=3,
    ):
        super().__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        self.w_kernel = width_kernel

        self.bn_feat = torch.nn.BatchNorm2d(
            self.dim_obs_channel[1], track_running_stats=False,
        )

        self.var_tr_uv = 1.
        self.dT = dT
        self.aug_state = False
        self.type_w_geo = type_wgeo

        self.type_geo_obs = 0
        if self.type_geo_obs == 1:
            dim_state = 4*self.dT
        elif self.type_geo_obs == 2:
            dim_state = 6*self.dT
        else:
            dim_state = 10*self.dT

        self.convx11 = torch.nn.Conv2d(
            dim_state, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,
            padding_mode=padding_mode,
        )
        self.convx12 = torch.nn.Conv2d(
            2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3),
            padding=1, bias=False, padding_mode=padding_mode,
        )
        self.convx21 = torch.nn.Conv2d(
            self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3),
            padding=1, bias=False, padding_mode=padding_mode,
        )
        self.convx22 = torch.nn.Conv2d(
            2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3),
            padding=1, bias=False, padding_mode=padding_mode,
        )

        self.convy11 = torch.nn.Conv2d(
            dT, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,
            padding_mode=padding_mode,
        )
        self.convy12 = torch.nn.Conv2d(
            2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3),
            padding=1, bias=False, padding_mode=padding_mode,
        )
        self.convy21 = torch.nn.Conv2d(
            self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3),
            padding=1, bias=False, padding_mode=padding_mode,
        )
        self.convy22 = torch.nn.Conv2d(
            2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3),
            padding=1, bias=False, padding_mode=padding_mode,
        )

        self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3),
        padding=1, bias=True, padding_mode=padding_mode,
        )
        self.sigmoid = torch.nn.Sigmoid()

        self.lat_rad = None
        self.lon_rad = None
        self.compute_derivativeswith_lon_lat = (
            TorchComputeDerivativesWithLonLat(dT=dT)
        )
        self.aug_state = False

    def compute_geo_factor(self, outputs, lat_rad, lon_rad, sigma=0.):
        return self.compute_derivativeswith_lon_lat.compute_geo_factor(
            outputs, lat_rad, lon_rad, sigma=0.,
        )

    def extract_sst_feature(self, y1):
        y1 = self.convy12(torch.tanh(self.convy11(y1)))
        y_feat = self.bn_feat(
            self.convy22(torch.tanh(self.convy21(torch.tanh(y1))))
        )

        return y_feat

    def extract_state_feature(self, x):
        if self.aug_state:
            xbar_ssh = x[:, :self.dT, :, :]
            dx_ssh = x[:, 2*self.dT:3*self.dT, :, :]
            x_u = x[:, 3*self.dT:4*self.dT, :, :]
            x_v = x[:, 4*self.dT:5*self.dT, :, :]
        else:
            xbar_ssh = x[:, :self.dT, :, :]
            dx_ssh = x[:, self.dT:2*self.dT, :, :]
            x_u = x[:, 2*self.dT:3*self.dT, :, :]
            x_v = x[:, 3*self.dT:4*self.dT, :, :]

        # geoostrophic factor
        u_geo_factor, v_geo_factor = self.compute_geo_factor(
            xbar_ssh, self.lat_rad, self.lon_rad,sigma=0.,
        )

        # latent component
        if self.type_geo_obs == 1:
            x_ssh_u = u_geo_factor * (xbar_ssh + dx_ssh)
            x_ssh_v = v_geo_factor * (xbar_ssh + dx_ssh)

            x_u_u = u_geo_factor * x_u
            x_v_v = v_geo_factor * x_v

            x_all = torch.cat((x_ssh_u,x_ssh_v,x_u_u,x_v_v), dim=1)
        elif self.type_geo_obs == 2:
            xbar_ssh_u = u_geo_factor * xbar_ssh
            xbar_ssh_v = v_geo_factor * xbar_ssh
            dx_ssh_u = u_geo_factor * dx_ssh
            dx_ssh_v = v_geo_factor * dx_ssh

            x_u_u = u_geo_factor * x_u
            x_v_v = v_geo_factor * x_v

            x_all = torch.cat(
                (xbar_ssh_u, xbar_ssh_v, dx_ssh_u, dx_ssh_v, x_u_u, x_v_v),
                dim=1,
            )
        else:
            xbar_ssh_u = u_geo_factor * xbar_ssh
            xbar_ssh_v = v_geo_factor * xbar_ssh

            dx_ssh_u = u_geo_factor * dx_ssh
            dx_ssh_v = v_geo_factor * dx_ssh
            x_u_u = u_geo_factor * x_u
            x_v_v = v_geo_factor * x_v

            x_all = torch.cat(
                (
                    xbar_ssh, xbar_ssh_u, xbar_ssh_v, dx_ssh, dx_ssh_u,
                    dx_ssh_v, x_u, x_u_u, x_v, x_v_v,
                ),
                dim=1,
            )

        x_feat = self.convx11(x_all)
        x_feat = torch.tanh(x_feat)
        x_feat = self.convx12(x_feat)
        x_feat = torch.tanh(x_feat)
        x_feat = self.convx21(x_feat)
        x_feat = torch.tanh(x_feat)
        x_feat = self.convx22(x_feat)
        x_feat = self.bn_feat(x_feat)

        return x_feat

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]

        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]


class TorchComputeDerivativesWithLonLat(torch.nn.Module):
    def __init__(self, dT=7, _filter='diff-non-centered'):
        super().__init__()

        # Initialise convGx and convGy parameters according to the filter
        if _filter == 'sobel':
            a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
            b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        elif _filter == 'diff-non-centered':
            a = np.array([[0., 0., 0.], [-.7, .4, .3], [0., 0., 0.]])
            b = np.transpose(a)
        elif _filter == 'diff':
            a = np.array([[0., 0., 0.], [0., 1., -1.], [0., 0., 0.]])
            b = np.array([[0., 0.3, 0.], [0., 1., 0.], [0., -1., 0.]])
        else:
            raise ValueError(f'Invalid argument: _filter={_filter}')

        self.convGx = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False,
            padding_mode='reflect',
        )
        self.convGy = torch.nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1, bias=False,
            padding_mode='reflect',
        )

        with torch.no_grad():
            self.convGx.weight = torch.nn.Parameter(
                torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                requires_grad=False,
            )
            self.convGy.weight = torch.nn.Parameter(
                torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),
                requires_grad=False,
            )

        # Initialise heat_filter and heat_filter_all_channels parameters
        a = np.array([[0., .25, 0.], [.25, 0., .25], [0., .25, 0.]])
        self.heat_filter = torch.nn.Conv2d(
            dT, dT, kernel_size=3, padding=1, bias=False,
            padding_mode='reflect',
        )
        with torch.no_grad():
            self.heat_filter.weight = torch.nn.Parameter(
                torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                requires_grad=False,
            )

        a = np.array([[0., .25, 0.], [.25, 0., .25], [0., .25, 0.]])
        self.heat_filter_all_channels = torch.nn.Conv2d(
            dT, dT, kernel_size=3, groups=dT, padding=1, bias=False,
            padding_mode='reflect',
        )
        with torch.no_grad():
            a = np.tile(a, (dT, 1, 1, 1))
            self.heat_filter_all_channels.weight = torch.nn.Parameter(
                torch.from_numpy(a).float(), requires_grad=False,
            )

        self.eps = 1e-10

    def compute_c(self, lat, lon, dlat, dlon):
        a = (
            torch.sin(dlat/2.)**2 + torch.cos(lat)**2 * torch.sin(dlon/2.)**2
        )

        c = 2. * 6.371e6 * torch.atan2(
            torch.sqrt(a + self.eps), torch.sqrt(1. - a + self.eps),
        )

        return c

    def compute_dx_dy_dlat_dlon(self, lat, lon, dlat, dlon):
        dy_from_dlat = self.compute_c(lat, lon, dlat, 0.*dlon)
        dx_from_dlon = self.compute_c(lat, lon, 0.*dlat, dlon)

        return dx_from_dlon, dy_from_dlat

    def _compute_grad(self, convG, u, sigma):
        if sigma > 0.:
            u = kornia.filters.gaussian_blur2d(
                u, (5, 5), (sigma, sigma), border_type='reflect',
            )

        grad = convG(u.view(-1, 1, u.size(2), u.size(3)))
        grad = grad.view(-1, u.size(1), u.size(2), u.size(3))

        return grad

    def compute_gradx(self, u, sigma=0.):
        return self._compute_grad(self.convGx, u, sigma)

    def compute_grady(self, u, sigma=0.):
        return self._compute_grad(self.convGy, u, sigma)

    def compute_gradxy(self, u, sigma=0.):
        if sigma > 0.:
            u = kornia.filters.gaussian_blur2d(
                u, (3, 3), (sigma, sigma), border_type='reflect',
            )

        G_x  = self.convGx(u[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)))
        G_y  = self.convGy(u[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)))

        for kk in range(1, u.size(1)):
            _G_x = self.convGx(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))
            _G_y = self.convGy(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))

            G_x = torch.cat((G_x, _G_x), dim = 1)
            G_y = torch.cat((G_y, _G_y), dim = 1)

        return G_x, G_y

    def compute_coriolis_force(self, lat, flag_mean_coriolis=False):
        omega = 7.2921e-5  # angular speed (rad/s)
        f = 2 * omega * torch.sin(lat)

        if flag_mean_coriolis:
            f = torch.mean(f) * torch.ones((f.size()))

        return f

    def compute_geo_velocities(
        self, ssh, lat, lon, sigma=0., alpha_uv_geo=9.81,
        flag_mean_coriolis=False,
    ):
        print("ligne 2490", ssh.size(),lat.size(),lon.size())
        #ssh = ssh.permute(0,3,1,2)
        dlat = lat[0, 1] - lat[0, 0]
        dlon = lon[0, 1] - lon[0, 0]

        # coriolis / lat/lon scaling
        grid_lat = lat.view(ssh.size(0), 1, ssh.size(2), 1)
        grid_lat = grid_lat.repeat(1, ssh.size(1), 1, ssh.size(3))
        grid_lon = lon.view(ssh.size(0), 1, 1, ssh.size(3))
        grid_lon = grid_lon.repeat(1, ssh.size(1), ssh.size(2), 1)

        dx_from_dlon, dy_from_dlat = self.compute_dx_dy_dlat_dlon(
            grid_lat, grid_lon, dlat, dlon,
        )
        f_c = self.compute_coriolis_force(
            grid_lat, flag_mean_coriolis=flag_mean_coriolis,
        )

        dssh_dx, dssh_dy = self.compute_gradxy(ssh, sigma=sigma)

        dssh_dx = dssh_dx / dx_from_dlon
        dssh_dy = dssh_dy / dy_from_dlat

        dssh_dy = (1. / f_c) * dssh_dy
        dssh_dx = (1. / f_c) * dssh_dx

        u_geo = -1. * dssh_dy
        v_geo = 1. * dssh_dx

        u_geo = alpha_uv_geo * u_geo
        v_geo = alpha_uv_geo * v_geo

        return u_geo, v_geo

    def heat_equation_one_channel(self, ssh, mask=None, iter=5, lam=0.2):
        out = torch.clone(ssh)
        for _ in range(iter):
            if mask is not None :
                _d = out - mask*self.heat_filter(out)
            else:
                _d = out - self.heat_filter(out)
            out -= lam*_d
        return out

    def heat_equation_all_channels(self, ssh, mask=None, iter=5, lam=0.2):
        out = 1. * ssh
        for _ in range(iter):
            if mask is not None :
                _d = out - mask*self.heat_filter_all_channels(out)
            else:
                _d = out - self.heat_filter_all_channels(out)
            out = out - lam*_d
        return out

    def heat_equation(self, u, mask=None, iter=5, lam=0.2):
        if mask:
            out = self.heat_equation_one_channel(
                u[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)),
                mask[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)),
                iter=iter, lam=lam,
            )
        else:
            out = self.heat_equation_one_channel(
                u[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)),
                iter=iter, lam=lam,
            )

        for k in range(1, u.size(1)):
            if mask:
                mask_view = mask[:, k, :, :].view(-1, 1, u.size(2), u.size(3))

                _out = self.heat_equation_one_channel(
                    u[:, k, :, :].view(-1, 1, u.size(2), mask_view, u.size(3)),
                    iter=iter, lam=lam,
                )
            else:
                _out = self.heat_equation_one_channel(
                    u[:, k, :, :].view(-1, 1, u.size(2), u.size(3)),
                    iter=iter, lam=lam,
                )

            out = torch.cat((out, _out), dim=1)

        return out

    def compute_geo_factor(
        self, ssh, lat, lon, sigma=0., alpha_uv_geo=9.81,
        flag_mean_coriolis=False,
    ):
        dlat = lat[0, 1] - lat[0, 0]
        dlon = lon[0, 1] - lon[0, 0]

        # coriolis / lat/lon scaling
        grid_lat = lat.view(ssh.size(0), 1, ssh.size(2), 1)
        grid_lat = grid_lat.repeat(1, ssh.size(1), 1, ssh.size(3))
        grid_lon = lon.view(ssh.size(0), 1, 1, ssh.size(3))
        grid_lon = grid_lon.repeat(1, ssh.size(1), ssh.size(2), 1)

        dx_from_dlon, dy_from_dlat = self.compute_dx_dy_dlat_dlon(
            grid_lat, grid_lon, dlat, dlon,
        )
        f_c = self.compute_coriolis_force(
            grid_lat, flag_mean_coriolis=flag_mean_coriolis,
        )

        dssh_dx = alpha_uv_geo / dx_from_dlon
        dssh_dy = alpha_uv_geo / dy_from_dlat

        dssh_dy = (1. / f_c) * dssh_dy
        dssh_dx = (1. / f_c) * dssh_dx

        factor_u_geo = -1. * dssh_dy
        factor_v_geo =  1. * dssh_dx

        return factor_u_geo, factor_v_geo

    def compute_div_curl_strain(self, u, v, lat, lon, sigma=0.):
        dlat = lat[0, 1] - lat[0, 0]
        dlon = lon[0, 1] - lon[0, 0]

        # coriolis / lat/lon scaling
        grid_lat = lat.view(u.size(0), 1, u.size(2), 1)
        grid_lat = grid_lat.repeat(1, u.size(1), 1, u.size(3))
        grid_lon = lon.view(u.size(0), 1, 1, u.size(3))
        grid_lon = grid_lon.repeat(1, u.size(1), u.size(2), 1)

        dx_from_dlon, dy_from_dlat = self.compute_dx_dy_dlat_dlon(
            grid_lat, grid_lon, dlat, dlon,
        )

        du_dx, du_dy = self.compute_gradxy(u, sigma=sigma)
        dv_dx, dv_dy = self.compute_gradxy(v, sigma=sigma)

        du_dx = du_dx / dx_from_dlon
        dv_dx = dv_dx / dx_from_dlon

        du_dy = du_dy / dy_from_dlat
        dv_dy = dv_dy / dy_from_dlat

        strain = torch.sqrt((dv_dx + du_dy)**2 + (du_dx - dv_dy)**2 + self.eps)

        div = du_dx + dv_dy
        curl =  du_dy - dv_dx

        return div,curl,strain

    def forward(self):
        return 1.
