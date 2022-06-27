import einops
import hydra
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
from metrics import save_netcdf,save_netcdf_with_obs,save_netcdf_uv, nrmse, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, get_psd_score
from models import Model_H, Model_HwithSST, Model_HwithSSTBN,Phi_r, ModelLR, Gradient_img, Model_HwithSSTBN_nolin_tanh, Model_HwithSST_nolin_tanh, Model_HwithSSTBNandAtt
from models import Model_HwithSSTBNAtt_nolin_tanh


from scipy import ndimage
from scipy.ndimage import gaussian_filter

def compute_coriolis_force(lat):
    omega = 7.2921e-5 # rad/s
    f = 2 * omega * np.sin(np.radians(lat))
    
    return f

def compute_uv_geo_with_coriolis(ssh,lat,lon,sigma=0.5,alpha_uv_geo = 1.):

    # coriolis / lat/lon scaling
    grid_lat = lat.reshape( (1,ssh.shape[1],1))
    grid_lat = np.tile( grid_lat , (ssh.shape[0],1,ssh.shape[2]) )
    grid_lon = lat.reshape( (1,1,ssh.shape[2]))
    grid_lon = np.tile( grid_lon , (ssh.shape[0],ssh.shape[1],1) )

    grid_lat = np.radians(grid_lat) 
    grid_lon = np.radians(grid_lon) 
        
    f_c = compute_coriolis_force(grid_lat)
    dy_from_dlat , dx_from_dlon = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,np.radians(1./20),np.radians(1./20) )     

    # (u,v) MSE
    ssh = gaussian_filter(ssh, sigma=sigma)
    dssh_dx = compute_gradx( ssh )
    dssh_dy = compute_grady( ssh )

    if 1*1 :
        dssh_dx = dssh_dx / dx_from_dlon 
        dssh_dy = dssh_dy / dy_from_dlat  

    if 1*1 :
         dssh_dy = ( 1. / f_c ) * dssh_dy
         dssh_dx = ( 1. / f_c  )* dssh_dx

    u_geo = -1. * dssh_dy
    v_geo = 1. * dssh_dx

    u_geo = alpha_uv_geo * u_geo
    v_geo = alpha_uv_geo * v_geo

    return u_geo,v_geo


def compute_dx_dy_dlat_dlon(lat,lon,dlat,dlon):
    
    def compute_c(lat,lon,dlat,dlon):
        a = np.sin(dlat / 2)**2 + np.cos(lat) ** 2 * np.sin(dlon / 2)**2
        return 2 * 6.371e6 * np.arctan2( np.sqrt(a), np.sqrt(1. - a))        
        #return 1. * np.arctan2( np.sqrt(a), np.sqrt(1. - a))        


    dx_from_dlat =  compute_c(lat,lon,dlat,0.)
    dy_from_dlon =  compute_c(lat,lon,0.,dlon)
    
    return dx_from_dlat , dy_from_dlon


def compute_gradx( u, alpha_dx = 1., sigma = 0. , _filter='diff-non-centered'):
    if sigma > 0. :
        u = gaussian_filter(u, sigma=sigma)
    if _filter == 'sobel' :
        return alpha_dx * ndimage.sobel(u,axis=2)
    elif _filter == 'diff-non-centered' :
        return alpha_dx * ndimage.convolve1d(u,weights=[0.3,0.4,-0.7],axis=2)

def compute_grady( u, alpha_dy= 1., sigma = 0., _filter='diff-non-centered' ):
    
    if sigma > 0. :
        u = gaussian_filter(u, sigma=sigma)
        
    if _filter == 'sobel' :
         return alpha_dy * ndimage.sobel(u,axis=1)
    elif _filter == 'diff-non-centered' :
        return alpha_dy * ndimage.convolve1d(u,weights=[0.3,0.4,-0.7],axis=1)
   
def compute_div(u,v,sigma=1.0,alpha_dx=1.,alpha_dy=1.):
    
    print('..... compute_div (sigma): %f'%sigma)

    du_dx = compute_gradx( u , alpha_dx = alpha_dx , sigma = sigma )
    dv_dy = compute_grady( v , alpha_dy = alpha_dy , sigma = sigma )
    
    return du_dx + dv_dy
    
def compute_curl(u,v,sigma=1.0,alpha_dx=1.,alpha_dy=1.):
    du_dy = compute_grady( u , alpha_dy = alpha_dy , sigma = sigma )
    dv_dx = compute_gradx( v , alpha_dx = alpha_dx , sigma = sigma )
    
    return du_dy - dv_dx

def compute_strain(u,v,sigma=1.0,alpha_dx=1.,alpha_dy=1.):
    du_dx = compute_gradx( u , alpha_dx = alpha_dx , sigma = sigma )
    dv_dy = compute_grady( v , alpha_dy = alpha_dy , sigma = sigma )

    du_dy = compute_grady( u , alpha_dy = alpha_dy , sigma = sigma )
    dv_dx = compute_gradx( v , alpha_dx = alpha_dx , sigma = sigma )

    return np.sqrt( ( dv_dx + du_dy ) **2 +  (du_dx - dv_dy) **2 )

def get_4dvarnet(hparams):
    return NN_4DVar.Solver_Grad_4DVarNN(
                Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                Model_H(hparams.shape_state[0]),
                NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)


def get_4dvarnet_sst(hparams):
    if hparams.use_sst_obs : 
        if hparams.sst_model == 'linear-bn' :
            return NN_4DVar.Solver_Grad_4DVarNN(
                        Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                            hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                        Model_HwithSSTBN(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                        NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                            hparams.dim_grad_solver, hparams.dropout),
                        hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
        elif hparams.sst_model == 'nolinear-tanh-bn' :
            return NN_4DVar.Solver_Grad_4DVarNN(
                        Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                            hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                        Model_HwithSSTBN_nolin_tanh(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                        NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                            hparams.dim_grad_solver, hparams.dropout),
                        hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
        elif hparams.sst_model == 'nolinear-tanh' :
            return NN_4DVar.Solver_Grad_4DVarNN(
                        Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                            hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                        Model_HwithSST_nolin_tanh(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                        NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                            hparams.dim_grad_solver, hparams.dropout),
                        hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
        elif hparams.sst_model == 'linear':
            return NN_4DVar.Solver_Grad_4DVarNN(
                            Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                            Model_HwithSST(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                            NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                                hparams.dim_grad_solver, hparams.dropout),
                            hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
        elif hparams.sst_model == 'linear-bn-att':
            return NN_4DVar.Solver_Grad_4DVarNN(
                            Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                            Model_HwithSSTBNandAtt(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                            NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                                hparams.dim_grad_solver, hparams.dropout),
                            hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
        elif hparams.sst_model == 'nolinear-tanh-bn-att':
            return NN_4DVar.Solver_Grad_4DVarNN(
                            Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                            Model_HwithSSTBNAtt_nolin_tanh(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                            NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                                hparams.dim_grad_solver, hparams.dropout),
                            hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)

    else:
       return get_4dvarnet(hparams)


class ModelSamplingFromSST(torch.nn.Module):
    def __init__(self,dT,nb_feat=10,thr=0.1):
        super(ModelSamplingFromSST, self).__init__()
        self.dT     = dT
        self.Tr     = torch.nn.Threshold(thr, 0.)
        self.S      = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)
        self.conv1  = torch.nn.Conv2d(int(dT/2),nb_feat,(3,3),padding=1)
        self.conv2  = torch.nn.Conv2d(nb_feat,dT-int(dT/2),(1,1),padding=0,bias=True)

    def forward(self , y ):
        yconv = self.conv2( F.relu( self.conv1( y[:,:int(self.dT/2),:,:] ) ) )

        yout1 = self.S( yconv )
        
        yout1 = torch.cat( (torch.zeros_like(y[:,:int(self.dT/2),:,:]),yout1) , dim=1)
        yout2 = self.Tr( yout1 )
        
        return [yout1,yout2]
       
def get_phi(hparams):
    class PhiPassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = Phi_r(hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2,
                    hparams.sS, hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param)
            self.phi_r = torch.nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks, *internal_state):
            return self.phi(state), None, None, None

    return PhiPassThrough()


def get_constant_crop(patch_size, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_size[d] for d in dim_order], dtype='float32')
        #print(patch_size, crop)
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        #print(patch_weight.sum())
        return patch_weight

def get_hanning_mask(patch_size, **kwargs):
        
    t_msk =kornia.filters.get_hanning_kernel1d(patch_size['time'])
    s_msk = kornia.filters.get_hanning_kernel2d((patch_size['lat'], patch_size['lon']))

    patch_weight = t_msk[:, None, None] * s_msk[None, :, :]
    return patch_weight.cpu().numpy()

def get_cropped_hanning_mask(patch_size, crop, **kwargs):
    pw = get_constant_crop(patch_size, crop)
        
    t_msk =kornia.filters.get_hanning_kernel1d(patch_size['time'])

    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()


class Div_uv(torch.nn.Module):
    def __init__(self,_filter='diff-non-centered'):
        super(Div_uv, self).__init__()

        if _filter == 'sobel':
            a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

            b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif _filter == 'diff-non-centered':
            a = np.array([[0., 0., 0.], [0.3, 0.4, -0.7], [0., 0., 0.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            b = np.array([[0., 0.3, 0.], [0., 0.4, 0.], [0., -0.7, 0.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif filter == 'diff':
            a = np.array([[0., 0., 0.], [0., 1., -1.], [0., 0., 0.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            b = np.array([[0., 0.3, 0.], [0., 1., 0.], [0., -1., 0.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
    def forward(self, u,v):
        
        if u.size(1) == 1:
            G_x = self.convGx(u)
            G_y = self.convGy(v)
            
            div_ = G_x + G_y 
            div = div_.view(-1, 1, u.size(1) - 2, u.size(2) - 2)
        else:

            for kk in range(0, u.size(1)):
                G_x = self.convGx(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))
                G_y = self.convGy(v[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))

                G_x = G_x.view(-1, 1, u.size(2) - 2, u.size(2) - 2)
                G_y = G_y.view(-1, 1, u.size(2) - 2, u.size(2) - 2)

                div_ = G_x + G_y 
                if kk == 0:
                    div = div_.view(-1, 1, u.size(2) - 2, u.size(2) - 2)
                else:
                    div = torch.cat((div, div_.view(-1, 1, u.size(2) - 2, u.size(3) - 2)), dim=1)
        
        return div

class Div_uv_with_lat_lon_scaling(torch.nn.Module):
    def __init__(self,_filter='diff-non-centered'):
        super(Div_uv, self).__init__()

        if _filter == 'sobel':
            a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

            b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif _filter == 'diff-non-centered':
            a = np.array([[0., 0., 0.], [0.3, 0.4, -0.7], [0., 0., 0.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            b = np.array([[0., 0.3, 0.], [0., 0.4, 0.], [0., -0.7, 0.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif filter == 'diff':
            a = np.array([[0., 0., 0.], [0., 1., -1.], [0., 0., 0.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            b = np.array([[0., 0.3, 0.], [0., 1., 0.], [0., -1., 0.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
            self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            
     
    def compute_c(self,lat,lon,dlat,dlon):
        a = torch.sin(dlat / 2)**2 + torch.cos(lat) ** 2 * torch.sin(dlon / 2)**2
        return 2 * 6.371e6 * np.atan2( torch.sqrt(a), torch.sqrt(1. - a))        
        
    def compute_dlat_dlon_scaling(self,lat,lon,dlat,dlon):
 
        dy_from_dlat =  self.compute_c(lat,lon,dlat,0.)
        dx_from_dlon =  self.compute_c(lat,lon,0.,dlon)
    
        return dx_from_dlon, dy_from_dlat
        
    def forward(self, u,v,lat,lon,dlat,dlon):
        
        dx_from_dlon, dy_from_dlat = self.compute_dlat_dlon_scaling(lat,lon,dlat,dlon)
                
        if u.size(1) == 1:
            G_x = self.convGx(u) 
            G_y = self.convGy(v) 
            
            div_ = G_x * dx_from_dlon + G_y * dy_from_dlat
            div = div_.view(-1, 1, u.size(1) - 2, u.size(2) - 2)
        else:

            for kk in range(0, u.size(1)):
                G_x = self.convGx(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))
                G_y = self.convGy(v[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))

                G_x = G_x.view(-1, 1, u.size(2) - 2, u.size(2) - 2)
                G_y = G_y.view(-1, 1, u.size(2) - 2, u.size(2) - 2)

                div_ = G_x * dx_from_dlon + G_y * dy_from_dlat
                if kk == 0:
                    div = div_.view(-1, 1, u.size(2) - 2, u.size(2) - 2)
                else:
                    div = torch.cat((div, div_.view(-1, 1, u.size(2) - 2, u.size(3) - 2)), dim=1)        
                    
        return div

############################################ Lightning Module #######################################################################

class LitModelUV(pl.LightningModule):

    MODELS = {
            '4dvarnet': get_4dvarnet,
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
        self.var_tr_uv = self.hparams.var_tr_uv       
        self.alpha_dx,self.alpha_dy,self.alpha_uv_geo = self.hparams.scaling_ssh_uv

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
        self.var_tr_uv = self.hparams.var_tr_uv
        self.var_Tt = self.hparams.var_Tt
        self.mean_Val = self.hparams.mean_Val
        self.mean_Tr = self.hparams.mean_Tr
        self.mean_Tt = self.hparams.mean_Tt

        # main model
        self.model_name = self.hparams.model if hasattr(self.hparams, 'model') else '4dvarnet'
        self.use_sst = self.hparams.sst if hasattr(self.hparams, 'sst') else False
        self.use_sst_obs = self.hparams.use_sst_obs if hasattr(self.hparams, 'use_sst_obs') else False
        self.use_sst_state = self.hparams.use_sst_state if hasattr(self.hparams, 'use_sst_state') else False
        self.aug_state = self.hparams.aug_state if hasattr(self.hparams, 'aug_state') else False
        self.save_rec_netcdf = self.hparams.save_rec_netcdf if hasattr(self.hparams, 'save_rec_netcdf') else './'
        self.sig_filter_laplacian = self.hparams.sig_filter_laplacian if hasattr(self.hparams, 'sig_filter_laplacian') else 0.5
        self.scale_dwscaling_sst = self.hparams.scale_dwscaling_sst if hasattr(self.hparams, 'scale_dwscaling_sst') else 1.0
        self.sig_filter_div = self.hparams.sig_filter_div if hasattr(self.hparams, 'sig_filter_div') else 1.0
        self.sig_filter_div_diag = self.hparams.sig_filter_div_diag if hasattr(self.hparams, 'sig_filter_div_diag') else self.hparams.sig_filter_div

        self.learning_sampling_uv = self.hparams.learning_sampling_uv if hasattr(self.hparams, 'learning_sampling_uv') else 'no_sammpling_learning'
        self.nb_feat_sampling_operator = self.hparams.nb_feat_sampling_operator if hasattr(self.hparams, 'nb_feat_sampling_operator') else -1.
        if self.nb_feat_sampling_operator > 0 :
            if self.hparams.sampling_model == 'sampling-from-sst':
                self.model_sampling_uv = ModelSamplingFromSST(self.hparams.dT,self.nb_feat_sampling_operator)
            else:
                print('..... something is not expected with the sampling model')
        else:
            self.model_sampling_uv = None
        
            
        if self.hparams.k_n_grad == 0 :
            self.hparams.n_fourdvar_iter = 1

        self.model = self.create_model()
        self.model_LR = ModelLR()
        self.grad_crop = lambda t: t[...,1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
                self.grad_crop(2.*kornia.filters.spatial_gradient(t, normalized=True)), 2)
        
        self.flag_compute_div_with_lat_scaling = True
        #if self.flag_compute_div_with_lat_scaling :
        #    self.div_field = Div_uv_with_lat_lon_scaling()
        #else:
        #    self.div_field = Div_uv()

        # loss weghing wrt time

        # self._w_loss = torch.nn.Parameter(torch.Tensor(self.patch_weight), requires_grad=False)  # duplicate for automatic upload to gpu
        self.w_loss = torch.nn.Parameter(torch.Tensor([0,0,0,1,0,0,0]), requires_grad=False)  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.obs_inp = None
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.x_feat = None  # variable to store output of test method
        self.test_figs = {}

        self.tr_loss_hist = []
        self.automatic_optimization = self.hparams.automatic_optimization if hasattr(self.hparams, 'automatic_optimization') else False

        self.median_filter_width = self.hparams.median_filter_width if hasattr(self.hparams, 'median_filter_width') else 1

        print('..... div. computation (sigma): %f -- %f'%(self.sig_filter_div,self.sig_filter_div_diag))


    def compute_div(self,u,v):
        # siletring
        f_u = kornia.filters.gaussian_blur2d(u, (5,5), (self.sig_filter_div,self.sig_filter_div), border_type='reflect')
        f_v = kornia.filters.gaussian_blur2d(v, (5,5), (self.sig_filter_div,self.sig_filter_div), border_type='reflect')
        
        # gradients
        du_dx, du_dy = self.gradient_img(f_u)
        dv_dx, dv_dy = self.gradient_img(f_v)
                       
        # scaling 
        du_dx = self.alpha_dx * dv_dx
        dv_dy = self.alpha_dy * dv_dy
        
        return du_dx + dv_dy  

    def compute_c(self,lat,lon,dlat,dlon):
        a = torch.sin(dlat / 2)**2 + torch.cos(lat) ** 2 * torch.sin(dlon / 2)**2
        return 2 * 6.371e6 * torch.atan2( torch.sqrt(a), torch.sqrt(1. - a))        
        
    def compute_dlat_dlon_scaling(self,lat,lon,dlat,dlon):
 
        dy_from_dlat =  self.compute_c(lat,lon,dlat,0. * dlon)
        dx_from_dlon =  self.compute_c(lat,lon,0. * dlat ,dlon)
    
        return dx_from_dlon, dy_from_dlat


    def compute_dlatlon2dxdy_scaling(self,lat,lon,res_latlon,dT):
        
        # coriolis / lat/lon scaling
        grid_lat = lat.view(lat.size(0),1,1,-1)
        grid_lat = grid_lat.repeat(1,dT,lon.size(1),1)
        grid_lon = lon.view(lon.size(0),1,-1,1)
        grid_lon = grid_lat.repeat(1,dT,1,lat.size(1))
            
        dx_from_dlon, dy_from_dlat  = self.compute_dlat_dlon_scaling(grid_lat,grid_lon,res_latlon,res_latlon )    
        
        self.alpha_dx = dx_from_dlon / torch.mean( dy_from_dlat ) 
        self.alpha_dy = dy_from_dlat / torch.mean( dy_from_dlat )   
        
        self.alpha_dx = self.alpha_dx[:,:,1:-1,1:-1]
        self.alpha_dy = self.alpha_dy[:,:,1:-1,1:-1]
        
       
    def update_filename_chkpt(self,filename_chkpt):
        
        old_suffix = '-{epoch:02d}-{val_loss:.4f}'

        suffix_chkpt = '-'+self.hparams.phi_param+'_%03d-augdata'%self.hparams.DimAE
        
        if self.model_sampling_uv is not None:
            suffix_chkpt = suffix_chkpt+'-sampling_sst_%d_%03d'%(self.hparams.nb_feat_sampling_operator,int(100*self.hparams.thr_l1_sampling_uv))
        
        if self.hparams.n_grad > 0 :
            
            if self.hparams.aug_state :
                suffix_chkpt = suffix_chkpt+'-augstate-dT%02d'%(self.hparams.dT)
            if self.use_sst_state :
                suffix_chkpt = suffix_chkpt+'-mmstate-augstate-dT%02d'%(self.hparams.dT)
            
            if self.use_sst_obs :
                suffix_chkpt = suffix_chkpt+'-sstobs-'+self.hparams.sst_model+'_%02d'%(self.hparams.dim_obs_sst_feat)
            
            suffix_chkpt = suffix_chkpt+'-grad_%02d_%02d_%03d'%(self.hparams.n_grad,self.hparams.k_n_grad,self.hparams.dim_grad_solver)
        else:
            if ( self.use_sst ) & ( self.use_sst_state ) :
                suffix_chkpt = suffix_chkpt+'-DirectInv-wSST'
            else:
                suffix_chkpt = suffix_chkpt+'-DirectInv'
            suffix_chkpt = suffix_chkpt+'-dT%02d'%(self.hparams.dT)
                
        suffix_chkpt = suffix_chkpt+old_suffix
        
        return filename_chkpt.replace(old_suffix,suffix_chkpt)
    
    def create_model(self):
        return self.MODELS[self.model_name](self.hparams)

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init = [None]
        out=None
        
        for _ in range(self.hparams.n_fourdvar_iter):
            if ( phase == 'test' ) & ( self.use_sst ):
                _loss, out, state, _metrics,sst_feat = self.compute_loss(batch, phase=phase, state_init=state_init)
            else:
                _loss, out, state, _metrics = self.compute_loss(batch, phase=phase, state_init=state_init)
            
            if self.hparams.n_grad > 0 :
                state_init = [None if s is None else s.detach() for s in state]
            losses.append(_loss)
            metrics.append(_metrics)
            
        if ( phase == 'test' ) & ( self.use_sst ):
            return losses, out, metrics, sst_feat
        else:    
            return losses, out, metrics

    def configure_optimizers(self):
        opt = torch.optim.Adam
        if hasattr(self.hparams, 'opt'):
            opt = lambda p: hydra.utils.call(self.hparams.opt, p)
        if self.model_name == '4dvarnet':
            if self.model_sampling_uv is not None :
                optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    {'params': self.model_sampling_uv.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    ])
            else:
                optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    ])
            return optimizer
        elif self.model_name == '4dvarnet_sst':
            if self.model_sampling_uv is not None :
                optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    {'params': self.model_sampling_uv.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    ])
            else:
                optimizer = opt([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                                    ])

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
        self.log("tr_mse_uv", metrics[-1]['mse_uv'] , on_step=False, on_epoch=True, prog_bar=True)
        #self.log("tr_l0_samp", metrics[-1]['l0_samp'] , on_step=False, on_epoch=True, prog_bar=True)
        self.log("tr_l1_samp", metrics[-1]['l1_samp'] , on_step=False, on_epoch=True, prog_bar=True)
        #self.log("tr_mseG", metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def diag_step(self, batch, batch_idx, log_pref='test'):
        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt = batch
        else:
            #targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt = batch
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = batch
           
        if ( self.use_sst ) :
          #losses, out, metrics = self(batch, phase='test')
          losses, out, metrics, sst_feat = self(batch, phase='test')
        else:
            losses, out, metrics = self(batch, phase='test')
        loss = losses[-1]
        if loss is not None:
            self.log(f'{log_pref}_loss', loss)
            self.log(f'{log_pref}_mse', metrics[-1]["mse"] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_mse_uv', metrics[-1]["mse_uv"] , on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_l1_samp', metrics[-1]["l1_samp"] , on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'{log_pref}_l0_samp', metrics[-1]["l0_samp"] , on_step=False, on_epoch=True, prog_bar=True)
            #self.log(f'{log_pref}_mseG', metrics[-1]['mseGrad'] / metrics[-1]['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        if not self.use_sst :
            return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'u_gt'    : (u_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'v_gt'    : (v_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred' : (out[0].detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred_u' : (out[1].detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'pred_v' : (out[2].detach().cpu() * np.sqrt(self.var_tr_uv)) }
        else:
            return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'u_gt'    : (u_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'v_gt'    : (v_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred' : (out[0].detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred_u' : (out[1].detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'pred_v' : (out[2].detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'sst_feat' : sst_feat.detach().cpu()}

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
        #print(len(outputs))
        torch.save(outputs, data_path / f'{self.global_rank}.t')

        if dist.is_initialized():
            dist.barrier()

        if self.global_rank == 0:
            return [torch.load(f) for f in sorted(data_path.glob('*'))]

    def build_test_xr_ds(self, outputs, diag_ds):

        outputs_keys = list(outputs[0][0].keys())
        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i]
               for i in range(len(diag_ds))
            ]

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield tuple(
                                [outputs[bc][b][k][i] for k in outputs_keys]
                        )
        
        dses =[
                xr.Dataset( {
                    k: (('time', 'lat', 'lon'), x_k) for k, x_k in zip(outputs_keys, xs)
                }, coords=coords)
            for  xs, coords
            in zip(iter_item(outputs), self.test_patch_coords)
        ]

        fin_ds = xr.merge([xr.zeros_like(ds[['time','lat', 'lon']]) for ds in dses])
        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
        )
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
            )

        # set the weight to a binadry (time window center  + spatial bounding box)
        if True :#False: #
            print(".... Set weight matrix to binary mask for final outputs")
            w = np.zeros_like( self.patch_weight.detach().cpu().numpy() )
            w[int(self.hparams.dT/2),:,:] = 1.
            w = w * self.patch_weight.detach().cpu().numpy()
            print('..... weight mask ')
            print(w[0:7,30,30])
        else:
            w = self.patch_weight.detach().cpu().numpy()
            print('..... weight mask ')
            print(w[0:7,30,30])

        for ds in dses:
            ds_nans = ds.assign(weight=xr.ones_like(ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)
            xr_weight = xr.DataArray(self.patch_weight.detach().cpu(), ds.coords, dims=ds.gt.dims)
            _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
            fin_ds = fin_ds + _ds


        return (
            (fin_ds.drop('weight') / fin_ds.weight)
            .sel(instantiate(self.test_domain))
            .pipe(lambda ds: ds.sel(time=~(np.isnan(ds.gt).all('lat').all('lon'))))
        ).transpose('time', 'lat', 'lon')


    def build_test_xr_ds_sst(self, outputs, diag_ds):

        outputs_keys = list(outputs[0][0].keys())
        
        with diag_ds.get_coords():
            self.test_patch_coords = [
               diag_ds[i]
               for i in range(len(diag_ds))
            ]

        def iter_item(outputs):
            n_batch_chunk = len(outputs)
            n_batch = len(outputs[0])
            for b in range(n_batch):
                bs = outputs[0][b]['gt'].shape[0]
                for i in range(bs):
                    for bc in range(n_batch_chunk):
                        yield tuple(
                                [outputs[bc][b][k][i] for k in outputs_keys[:-1]]
                        )
        
        dses =[
                xr.Dataset( {
                    k: (('time', 'lat', 'lon'), x_k) for k, x_k in zip(outputs_keys, xs)
                }, coords=coords)
            for  xs, coords
            in zip(iter_item(outputs), self.test_patch_coords)
        ]

        fin_ds = xr.merge([xr.zeros_like(ds[['time','lat', 'lon']]) for ds in dses])

        fin_ds = fin_ds.assign(
            {'weight': (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
        )
        
        for v in dses[0]:
            fin_ds = fin_ds.assign(
                {v: (fin_ds.dims, np.zeros(list(fin_ds.dims.values()))) }
            )

        # set the weight to a binadry (time window center  + spatial bounding box)
        if True :
            print(".... Set weight matrix to binary mask for final outputs")
            w = np.zeros_like( self.patch_weight.detach().cpu().numpy() )
            w[int(self.hparams.dT/2),:,:] = 1.
            w = w * self.patch_weight.detach().cpu().numpy()
        else:
            w = self.patch_weight.detach().cpu().numpy()
            print('..... weight mask ')
            print(w[0:7,30,30])
            
        for ds in dses:
            ds_nans = ds.assign(weight=xr.ones_like(ds.gt)).isnull().broadcast_like(fin_ds).fillna(0.)            
            #xr_weight = xr.DataArray(self.patch_weight.detach().cpu(), ds.coords, dims=ds.gt.dims)
            xr_weight = xr.DataArray(w, ds.coords, dims=ds.gt.dims)
            #print(xr_weight) 
            _ds = ds.pipe(lambda dds: dds * xr_weight).assign(weight=xr_weight).broadcast_like(fin_ds).fillna(0.).where(ds_nans==0, np.nan)
            fin_ds = fin_ds + _ds

        #fin_ds.weight.data = ( fin_ds.weight.data > 1. ).astype(float)

        return (
            (fin_ds.drop('weight') / fin_ds.weight)
            .sel(instantiate(self.test_domain))
            .pipe(lambda ds: ds.sel(time=~(np.isnan(ds.gt).all('lat').all('lon'))))
        ).transpose('time', 'lat', 'lon')

    def nrmse_fn(self, pred, ref, gt):
        return (
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

    def mse_fn(self, pred, ref, gt):
            return(
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
            animate_maps(self.x_gt, self.obs_inp, self.x_oi, self.x_rec, self.lon, self.lat, path_save0)
            
            # save NetCDF
        # PENDING: replace hardcoded 60
        # save_netcdf(saved_path1=path_save1, pred=self.x_rec,
        #         lon=self.test_lon, lat=self.test_lat, time=self.test_dates, time_units=None)

        # compute nRMSE
        # np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)

        nrmse_df = self.nrmse_fn('pred', 'oi', 'gt')
        mse_df = self.mse_fn('pred', 'oi', 'gt')
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
        #else:
        #    psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred[2:42,:,:], self.test_xr_ds.gt[2:42,:,:])
        #    fig, spatial_res_model, spatial_res_oi = get_psd_score(self.test_xr_ds.gt[2:42,:,:], self.test_xr_ds.pred[2:42,:,:], self.test_xr_ds.oi[2:42,:,:], with_fig=True)
        
        self.test_figs['res'] = fig
        self.logger.experiment.add_figure(f'{log_pref} Spat. Resol', fig, global_step=self.current_epoch)
        #psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        psd_fig = metrics.plot_psd_score(psd_ds)
        self.test_figs['psd'] = psd_fig
        #psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)
        self.logger.experiment.add_figure(f'{log_pref} PSD', psd_fig, global_step=self.current_epoch)
        _, _, mu, sig = metrics.rmse_based_scores(self.test_xr_ds.pred, self.test_xr_ds.gt)

        mdf = pd.concat([
            nrmse_df.rename(columns=lambda c: f'{log_pref}_{c}_glob').loc['pred'].T,
            mse_df.rename(columns=lambda c: f'{log_pref}_{c}_glob').loc['pred'].T,
        ])
        
        mse_metrics_pred = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.pred)
        mse_metrics_oi = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.oi)
        
        var_mse_pred_vs_oi = 100. * ( 1. - mse_metrics_pred['mse'] / mse_metrics_oi['mse'] )
        var_mse_grad_pred_vs_oi = 100. * ( 1. - mse_metrics_pred['mseGrad'] / mse_metrics_oi['mseGrad'] )
        
        mse_metrics_lap_oi = metrics.compute_laplacian_metrics(self.test_xr_ds.gt,self.test_xr_ds.oi,sig_lap=self.sig_filter_laplacian)
        mse_metrics_lap_pred = metrics.compute_laplacian_metrics(self.test_xr_ds.gt,self.test_xr_ds.pred,sig_lap=self.sig_filter_laplacian)        

        mse_metrics_pred = metrics.compute_metrics(self.test_xr_ds.gt, self.test_xr_ds.pred)

        #dw_lap = 20
        
        #mse_metrics_lap_oi = metrics.compute_laplacian_metrics(self.test_xr_ds.gt[:,dw_lap:self.test_xr_ds.gt.shape[1]-dw_lap,dw_lap:self.test_xr_ds.gt.shape[2]-dw_lap],self.test_xr_ds.oi[:,dw_lap:self.test_xr_ds.gt.shape[1]-dw_lap,dw_lap:self.test_xr_ds.gt.shape[2]-dw_lap],sig_lap=sig_lap)
        #mse_metrics_lap_pred = metrics.compute_laplacian_metrics(self.test_xr_ds.gt[:,dw_lap:self.test_xr_ds.gt.shape[1]-dw_lap,dw_lap:self.test_xr_ds.gt.shape[2]-dw_lap],self.test_xr_ds.pred[:,dw_lap:self.test_xr_ds.gt.shape[1]-dw_lap,dw_lap:self.test_xr_ds.gt.shape[2]-dw_lap],sig_lap=sig_lap)        

        var_mse_pred_lap = 100. * (1. - mse_metrics_lap_pred['mse'] / mse_metrics_lap_pred['var_lap'] )
        var_mse_oi_lap = 100. * (1. - mse_metrics_lap_oi['mse'] / mse_metrics_lap_pred['var_lap'] )

        # MSE (U,V) fields
        mse_uv = np.nanmean((self.test_xr_ds.u_gt - self.test_xr_ds.pred_u) ** 2 + (self.test_xr_ds.v_gt - self.test_xr_ds.pred_v) ** 2 )
        var_uv = np.nanmean((self.test_xr_ds.u_gt) ** 2 + (self.test_xr_ds.v_gt) ** 2 )
        var_mse_uv = 100. * ( 1. - mse_uv / var_uv )

        psd_ds_u, lamb_x_u, lamb_t_u = metrics.psd_based_scores(self.test_xr_ds.pred_u, self.test_xr_ds.u_gt)
        psd_ds_v, lamb_x_v, lamb_t_v = metrics.psd_based_scores(self.test_xr_ds.pred_v, self.test_xr_ds.v_gt)

        # MSE Div of (U,V) fields        
        #div_rec = self.div_field(self.test_xr_ds.pred_u,self.test_xr_ds.pred_v)
        #div_gt = self.div_field(self.test_xr_ds.v_gt,self.test_xr_ds.u_gt)


        def compute_div_curl_strain_metrics(u_gt,v_gt,u_rec,v_rec,sig_div=0.5,alpha_dx=1.,alpha_dy=1.,flag_compute_strain = False):
            
            div_uv_gt = compute_div(u_gt,v_gt,sigma=sig_div,alpha_dx=alpha_dx,alpha_dy=alpha_dy)
            div_uv_rec = compute_div(u_rec,v_rec,sigma=sig_div,alpha_dx=alpha_dx,alpha_dy=alpha_dy)
                    
            curl_uv_gt = compute_curl(u_gt,v_gt,sigma=sig_div,alpha_dx=alpha_dx,alpha_dy=alpha_dy)
            curl_uv_rec = compute_curl(u_rec,v_rec,sigma=sig_div,alpha_dx=alpha_dx,alpha_dy=alpha_dy)
            
            mse_div = np.nanmean( (div_uv_gt - div_uv_rec)**2 )
            nmse_div = mse_div / np.nanmean( (div_uv_gt )**2 )

            mse_curl = np.nanmean( (curl_uv_gt - curl_uv_rec)**2 )
            nmse_curl = mse_curl / np.nanmean( (curl_uv_gt )**2 )

            if flag_compute_strain :
                strain_uv_gt = compute_strain(u_gt,v_gt,sigma=sig_div,alpha_dx=alpha_dx,alpha_dy=alpha_dy)
                strain_uv_rec = compute_strain(u_rec,v_rec,sigma=sig_div,alpha_dx=alpha_dx,alpha_dy=alpha_dy)

                mse_strain = np.nanmean( (strain_uv_gt - strain_uv_rec)**2 )
                nmse_strain = mse_curl / np.nanmean( ( strain_uv_gt )**2 )
                
                return mse_div,nmse_div,mse_curl,nmse_curl,mse_strain,nmse_strain
            else:
                return mse_div,nmse_div,mse_curl,nmse_curl
         
        def compute_mse_uv_geo(u_gt,v_gt,ssh,sigma=0.5,alpha_dx=1.,alpha_dy=1.,alpha_uv_geo = 0.):
            
            # (u,v) MSE
            dssh_dx = compute_gradx( ssh , alpha_dx = alpha_dx , sigma = sigma )
            dssh_dy = compute_grady( ssh , alpha_dy = alpha_dy , sigma = sigma )
            
            u_geo = -1. * dssh_dy
            v_geo = 1.  * dssh_dx
            
            if np.max( np.abs(alpha_uv_geo) ) == 0. :
                alpha_uv_geo = float( np.mean( u_gt * u_geo + v_gt * v_geo) / np.mean( u_geo**2 + v_geo**2 ) )

            u_geo = alpha_uv_geo * u_geo
            v_geo = alpha_uv_geo * v_geo
            
            mse_uv_geo = np.nanmean( (u_geo - u_gt)**2 + (v_geo - v_gt)**2 )
            nmse_uv_geo = mse_uv_geo / np.nanmean( (u_gt)**2 + (v_gt)**2 )

            # div/curl mse (with filtering)
            dssh_dx = compute_gradx( ssh , alpha_dx = alpha_dx , sigma = sigma )
            dssh_dy = compute_grady( ssh , alpha_dy = alpha_dy , sigma = sigma )
            u_geo = -1. * dssh_dy
            v_geo = 1.  * dssh_dx
            u_geo = alpha_uv_geo * u_geo
            v_geo = alpha_uv_geo * v_geo

            psd_ds_ugeo, lamb_x_ugeo, lamb_t_ugeo = metrics.psd_based_scores(u_geo,self.test_xr_ds.u_gt)
            psd_ds_vgeo, lamb_x_vgeo, lamb_t_vgeo = metrics.psd_based_scores(v_geo,self.test_xr_ds.v_gt)
            
            print('......... lambda ugeo =  %.3f   / %.3f '%(lamb_x_ugeo,lamb_t_ugeo))
            print('......... lambda vgeo =  %.3f   / %.3f '%(lamb_x_vgeo,lamb_t_vgeo))
            
            if sigma > 0. :
                u_gt = gaussian_filter(u_gt, sigma=sigma)
                v_gt = gaussian_filter(v_gt, sigma=sigma)
            
            mse_stat =  compute_div_curl_strain_metrics(u_gt,v_gt,u_geo,v_geo,sig_div=0, 
                                                        alpha_dx = alpha_dx , alpha_dy = alpha_dy,
                                                        flag_compute_strain=True) 
            mse_div_geo, nmse_div_geo, mse_curl_geo, nmse_curl_geo, mse_strain_geo, nmse_strain_geo    = mse_stat   


            return mse_uv_geo, nmse_uv_geo, mse_div_geo, nmse_div_geo, mse_curl_geo, nmse_curl_geo, mse_strain_geo, nmse_strain_geo

        def compute_mse_uv_geo_with_coriolis(u_gt,v_gt,ssh,lat,lon,sigma=0.5,alpha_uv_geo = 1.):
            
            ssh = gaussian_filter(ssh, sigma=sigma)        
            u_geo,v_geo = compute_uv_geo_with_coriolis(ssh,lat,lon,sigma=0.,alpha_uv_geo = alpha_uv_geo)
            
            mse_uv_geo = np.nanmean( (u_geo - u_gt)**2 + (v_geo - v_gt)**2 )
            nmse_uv_geo = mse_uv_geo / np.nanmean( (u_gt)**2 + (v_gt)**2 )
 
            psd_ds_ugeo, lamb_x_ugeo, lamb_t_ugeo = metrics.psd_based_scores(u_geo,self.test_xr_ds.u_gt)
            psd_ds_vgeo, lamb_x_vgeo, lamb_t_vgeo = metrics.psd_based_scores(v_geo,self.test_xr_ds.v_gt)
            
            print('......... lambda ugeo =  %.3f   / %.3f '%(lamb_x_ugeo,lamb_t_ugeo))
            print('......... lambda vgeo =  %.3f   / %.3f '%(lamb_x_vgeo,lamb_t_vgeo))
            
            if sigma > 0. :
                u_gt = gaussian_filter(u_gt, sigma=sigma)
                v_gt = gaussian_filter(v_gt, sigma=sigma)
            
            mse_stat =  compute_div_curl_strain_metrics(u_gt,v_gt,u_geo,v_geo,lat,lon,sig_div=0, 
                                                        flag_compute_strain=True) 
            mse_div_geo, nmse_div_geo, mse_curl_geo, nmse_curl_geo, mse_strain_geo, nmse_strain_geo    = mse_stat   


            return mse_uv_geo, nmse_uv_geo, mse_div_geo, nmse_div_geo, mse_curl_geo, nmse_curl_geo, mse_strain_geo, nmse_strain_geo

        # compute (dx,dy) scaling for the computation of the derivative
        def compute_dxy_scaling(u,v,ssh,sigma=1.):
            
            u = gaussian_filter(u, sigma=sigma)
            v = gaussian_filter(v, sigma=sigma)
            ssh = gaussian_filter(ssh, sigma=sigma)
            
            dssh_dx = compute_gradx(ssh)
            dssh_dy = compute_grady(ssh)
             
            corr_x_u = float( np.mean( u * dssh_dx) / np.sqrt( np.mean( dssh_dx**2 ) * np.mean( u**2 ) ) )
            corr_x_v = float( np.mean( v * dssh_dx) / np.sqrt( np.mean( dssh_dx**2 ) * np.mean( v**2 ) ) )
            
            corr_y_u = float( np.mean( u * dssh_dy) / np.sqrt( np.mean( dssh_dy**2 ) * np.mean( u**2 ) ) )
            corr_y_v = float( np.mean( v * dssh_dy) / np.sqrt( np.mean( dssh_dy**2 ) * np.mean( v**2 ) ) )

            print('.... alpha = % f -- % f -- %f -- %f '%(corr_x_u,corr_x_v,corr_y_u,corr_y_v)  )
            alpha_dy_u = float( np.mean( -1. * dssh_dy * u) / np.mean( dssh_dy**2 ) )
            alpha_dx_v = float( np.mean(  1. * dssh_dx * v) / np.mean( dssh_dx**2 ) )

            print('... R**2: %f -- %f'%(corr_y_u,corr_x_v))
            print('.... alpha: %f -- %f '%(alpha_dx_v,alpha_dy_u)  )
            
            return 1.,alpha_dy_u/alpha_dx_v,alpha_dx_v

        def compute_dxy_scaling_with_coriolis(u,v,ssh,lat,lon,sigma=1.):
        
            u = gaussian_filter(u, sigma=sigma)
            v = gaussian_filter(v, sigma=sigma)
            ssh = gaussian_filter(ssh, sigma=sigma)
        
            u_geo,v_geo = compute_uv_geo_with_coriolis(ssh,lat,lon,sigma=0.,alpha_uv_geo = 1.)
        
            dssh_dy = -1. * u_geo
            dssh_dx = 1. * v_geo
                        
            corr_x_u = float( np.mean( u * dssh_dx) / np.sqrt( np.mean( dssh_dx**2 ) * np.mean( u**2 ) ) )
            corr_x_v = float( np.mean( v * dssh_dx) / np.sqrt( np.mean( dssh_dx**2 ) * np.mean( v**2 ) ) )
        
            corr_y_u = float( np.mean( u * dssh_dy) / np.sqrt( np.mean( dssh_dy**2 ) * np.mean( u**2 ) ) )
            corr_y_v = float( np.mean( v * dssh_dy) / np.sqrt( np.mean( dssh_dy**2 ) * np.mean( v**2 ) ) )
        
            print('.... corr = % f -- % f -- %f -- %f '%(corr_x_u,corr_x_v,corr_y_u,corr_y_v)  )
            alpha_dy_u = float( np.mean( -1. * dssh_dy * u) / np.mean( dssh_dy**2 ) )
            alpha_dx_v = float( np.mean(  1. * dssh_dx * v) / np.mean( dssh_dx**2 ) )
        
            alpha_uv_geo = float( np.mean( -1. * dssh_dy * u + 1. * dssh_dx * v ) / np.mean( dssh_dy**2 + dssh_dx**2) )
        
            print('... R**2: %f -- %f'%(corr_y_u,corr_x_v))
            print('.... alpha: %f -- %f -- %f'%(alpha_dx_v,alpha_dy_u,alpha_uv_geo)  )
        
            return alpha_uv_geo
        
        if 1*0 :
            alpha_dx, alpha_dy, alpha_uv_geo = compute_dxy_scaling(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,self.test_xr_ds.gt,sigma=4.0)
    
            print('.. Scaling [Training DS] : %f -- %f -- %f '%(self.alpha_dx,self.alpha_dy,self.alpha_uv_geo))
            print('.. Scaling [Test DS]     : %f -- %f -- %f '%(alpha_dx,alpha_dy,alpha_uv_geo))
    
            flag_use_uv_geo_scaling_training_ds = False#  True # 
            
            if flag_use_uv_geo_scaling_training_ds :
                alpha_dx = self.alpha_dx
                alpha_dy = self.alpha_dy
                alpha_uv_geo = self.alpha_uv_geo
        else:                
            alpha_uv_geo =  compute_dxy_scaling_with_coriolis(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,self.test_xr_ds.gt,self.test_lat,self.test_lon,sigma=4.)
        
            print('.. Scaling [Test DS]     : %f '%(alpha_uv_geo))

            # coriolis / lat/lon scaling
            def compute_dxdy_dlatdlon_fc(lat,lon,shape):                
                grid_lat = lat.reshape( (1,shape[1],1))
                grid_lat = np.tile( grid_lat , (shape[0],1,shape[2]) )
                grid_lon = lon.reshape( (1,1,shape[2]))
                grid_lon = np.tile( grid_lon , (shape[0],shape[1],1) )
        
                grid_lat = np.radians(grid_lat) 
                grid_lon = np.radians(grid_lon) 
                    
                dy_from_dlat , dx_from_dlon = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,np.radians(1./20),np.radians(1./20) ) 
                f_c = compute_coriolis_force(grid_lat)            
                
                return dx_from_dlon, dy_from_dlat , f_c 
            
            dx_from_dlon, dy_from_dlat , f_c =  compute_dxdy_dlatdlon_fc( self.test_lat, self.test_lon, self.test_xr_ds.gt.shape )
            alpha_dx = 1. / ( dx_from_dlon )
            alpha_dy = 1. / ( dy_from_dlat )
                        
            alpha_uv_geo = alpha_uv_geo / f_c
    
        sig_div = self.sig_filter_div_diag
        print('..... div. computation (sigma): %f -- %f'%(self.sig_filter_div,self.sig_filter_div_diag))
        mse_stat = compute_mse_uv_geo(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,
                                      self.test_xr_ds.gt,sigma=sig_div,
                                      alpha_dx=alpha_dx,alpha_dy=alpha_dy,
                                      alpha_uv_geo = alpha_uv_geo)
        mse_uv_ssh_gt,nmse_uv_ssh_gt,mse_div_ssh_gt, nmse_div_ssh_gt, mse_curl_ssh_gt, nmse_curl_ssh_gt,  mse_strain_ssh_gt, nmse_strain_ssh_gt = mse_stat
        
        var_mse_uv_ssh_gt = 100. * (1. - nmse_uv_ssh_gt )
        var_mse_div_ssh_gt = 100. * (1. - nmse_div_ssh_gt )
        var_mse_curl_ssh_gt = 100. * (1. - nmse_curl_ssh_gt )
        var_mse_strain_ssh_gt = 100. * (1. - nmse_strain_ssh_gt )
    
        mse_stat = compute_mse_uv_geo(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,
                                      self.test_xr_ds.oi,sigma=sig_div,
                                      alpha_dx=alpha_dx,alpha_dy=alpha_dy,
                                      alpha_uv_geo = alpha_uv_geo)
        mse_uv_oi,nmse_uv_oi,mse_div_oi, nmse_div_oi, mse_curl_oi, nmse_curl_oi, mse_strain_oi, nmse_strain_oi = mse_stat
        
        var_mse_uv_oi = 100. * (1. - nmse_uv_oi )
        var_mse_div_oi = 100. * (1. - nmse_div_oi )
        var_mse_curl_oi = 100. * (1. - nmse_curl_oi )
        var_mse_strain_oi = 100. * (1. - nmse_strain_oi )

        mse_stat = compute_mse_uv_geo(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,
                                      self.test_xr_ds.pred,sigma=sig_div,
                                      alpha_dx=alpha_dx,alpha_dy=alpha_dy,
                                      alpha_uv_geo = alpha_uv_geo)
        mse_uv_pred,nmse_uv_pred,mse_div_pred, nmse_div_pred, mse_curl_pred, nmse_curl_pred, mse_strain_pred, nmse_strain_pred = mse_stat
        var_mse_uv_pred     = 100. * (1. - nmse_uv_pred )
        var_mse_div_pred    = 100. * (1. - nmse_div_pred )
        var_mse_curl_pred   = 100. * (1. - nmse_curl_pred )
        var_mse_strain_pred = 100. * (1. - nmse_strain_pred )
                        
        stat_mse_div_curl_strain =  compute_div_curl_strain_metrics(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,
                                                                    self.test_xr_ds.pred_u,self.test_xr_ds.pred_v,
                                                                    sig_div=sig_div,
                                                                    alpha_dx=alpha_dx,alpha_dy=alpha_dy,
                                                                    flag_compute_strain = True)

        mse_div, nmse_div, mse_curl, nmse_curl, mse_strain, nmse_strain =  stat_mse_div_curl_strain

        var_mse_div = 100. * (1. - nmse_div )
        var_mse_curl = 100. * (1. - nmse_curl )
        var_mse_strain = 100. * (1. - nmse_strain )
        
        md = {
            f'{log_pref}_spatial_res': float(spatial_res_model),
            f'{log_pref}_spatial_res_imp': float(spatial_res_model / spatial_res_oi),
            f'{log_pref}_lambda_x': lamb_x,
            f'{log_pref}_lambda_t': lamb_t,
            f'{log_pref}_lambda_x_u': lamb_x_u,
            f'{log_pref}_lambda_t_u': lamb_t_u,
            f'{log_pref}_lambda_x_v': lamb_x_v,
            f'{log_pref}_lambda_t_v': lamb_t_v,
            f'{log_pref}_mu': mu,
            f'{log_pref}_sigma': sig,
            **mdf.to_dict(),
            f'{log_pref}_var_mse_vs_oi': float(var_mse_pred_vs_oi),
            f'{log_pref}_var_mse_grad_vs_oi': float(var_mse_grad_pred_vs_oi),
            f'{log_pref}_var_mse_lap_pred': float(var_mse_pred_lap),
            f'{log_pref}_var_mse_lap_oi': float(var_mse_oi_lap),
            f'{log_pref}_var_mse_uv_gt': float(var_mse_uv_ssh_gt),
            f'{log_pref}_var_mse_uv_oi': float(var_mse_uv_oi),
            f'{log_pref}_var_mse_uv_pred': float(var_mse_uv_pred),
            f'{log_pref}_var_mse_uv': float(var_mse_uv),
            f'{log_pref}_var_mse_div_ssh_gt': float(var_mse_div_ssh_gt),            
            f'{log_pref}_var_mse_div_oi': float(var_mse_div_oi),            
            f'{log_pref}_var_mse_div_pred': float(var_mse_div_pred),            
            f'{log_pref}_var_mse_div': float(var_mse_div),            
            f'{log_pref}_var_mse_strain_ssh_gt': float(var_mse_strain_ssh_gt),            
            f'{log_pref}_var_mse_strain_oi': float(var_mse_strain_oi),            
            f'{log_pref}_var_mse_strain_pred': float(var_mse_strain_pred),            
            f'{log_pref}_var_mse_strain': float(var_mse_strain),            
            f'{log_pref}_var_mse_curl_ssh_gt': float(var_mse_curl_ssh_gt),            
            f'{log_pref}_var_mse_curl_oi': float(var_mse_curl_oi),            
            f'{log_pref}_var_mse_curl_pred': float(var_mse_curl_pred),            
            f'{log_pref}_var_mse_curl': float(var_mse_curl),            
        }
        print(pd.DataFrame([md]).T.to_markdown())
        return md

    def diag_epoch_end(self, outputs, log_pref='test'):
        
        full_outputs = self.gather_outputs(outputs, log_pref=log_pref)
        
        if full_outputs is None:
            print("full_outputs is None on ", self.global_rank)
            return
        if log_pref == 'test':
            diag_ds = self.trainer.test_dataloaders[0].dataset.datasets[0]
        elif log_pref == 'val':
            diag_ds = self.trainer.val_dataloaders[0].dataset.datasets[0]
        else:
            raise Exception('unknown phase')

        if not self.use_sst :
            self.test_xr_ds = self.build_test_xr_ds(full_outputs, diag_ds=diag_ds)
        else:
            self.test_xr_ds = self.build_test_xr_ds_sst(full_outputs, diag_ds=diag_ds)

        #print(self.test_xr_ds.gt.data.shape)

        self.x_gt = self.test_xr_ds.gt.data#[2:42,:,:]
        self.obs_inp = self.test_xr_ds.obs_inp.data#[2:42,:,:]
        self.x_oi = self.test_xr_ds.oi.data#[2:42,:,:]
        self.x_rec = self.test_xr_ds.pred.data#[2:42,:,:]
        
        self.u_gt = self.test_xr_ds.u_gt.data
        self.v_gt = self.test_xr_ds.v_gt.data
        self.u_rec = self.test_xr_ds.pred_u.data#[2:42,:,:]
        self.v_rec = self.test_xr_ds.pred_v.data#[2:42,:,:]

        self.x_rec_ssh = self.x_rec

        def extract_seq(out,key,dw=20):
            seq = torch.cat([chunk[key] for chunk in outputs]).numpy()
            seq = seq[:,:,dw:seq.shape[2]-dw,dw:seq.shape[2]-dw]
            
            return seq
        
        self.x_sst_feat_ssh = extract_seq(outputs,'sst_feat',dw=20)

        
        #print('..... Shape evaluated tensors: %dx%dx%d'%(self.x_gt.shape[0],self.x_gt.shape[1],self.x_gt.shape[2]))
        
        self.test_coords = self.test_xr_ds.coords
        
        self.test_lat = self.test_coords['lat'].data
        self.test_lon = self.test_coords['lon'].data
        self.test_dates = self.test_coords['time'].data#[2:42]

        md = self.sla_uv_diag(t_idx=3, log_pref=log_pref)
        
        self.latest_metrics.update(md)
        self.logger.log_metrics(md, step=self.current_epoch)

        if self.scale_dwscaling_sst > 1. :
            print('.... Using downscaled SST by %.1f'%self.scale_dwscaling_sst)
        print('..... Log directory: '+self.logger.log_dir)
        
        if self.save_rec_netcdf == True :
            #path_save1 = self.logger.log_dir + f'/test_res_all.nc'
            path_save1 = self.hparams.path_save_netcdf.replace('.ckpt','_res_4dvarnet_all.nc')
            if not self.use_sst :
                print('... Save nc file with all results : '+path_save1)
                save_netcdf_with_obs(saved_path1=path_save1, gt=self.x_gt, obs = self.obs_inp , oi= self.x_oi, pred=self.x_rec_ssh,
                         lon=self.test_lon, lat=self.test_lat, time=self.test_dates)#, time_units=None)
            else:
                if 1*1 : 
                    if 1*1 :
                        def extract_seq(out,key,dw=20):
                            seq = torch.cat([chunk[key] for chunk in outputs]).numpy()
                            seq = seq[:,:,dw:seq.shape[2]-dw,dw:seq.shape[2]-dw]
                            
                            return seq
                        
                        self.x_sst_feat_ssh = extract_seq(outputs,'sst_feat',dw=20)
            
                        self.x_gt = extract_seq(outputs,'gt',dw=20)
                        self.x_gt = self.x_gt[:,int(self.hparams.dT/2),:,:]
            
                        self.obs_inp = extract_seq(outputs,'obs_inp',dw=20)
                        self.obs_inp = self.obs_inp[:,int(self.hparams.dT/2),:,:]
            
                        self.x_oi = extract_seq(outputs,'oi',dw=20)
                        self.x_oi = self.x_oi[:,int(self.hparams.dT/2),:,:]
            
                        self.x_rec = extract_seq(outputs,'pred',dw=20)
                        self.x_rec = self.x_rec[:,int(self.hparams.dT/2),:,:]
                        self.x_rec_ssh = self.x_rec
                        
                    else:
                        self.x_gt = self.x_gt[2:42,:,:]
                        self.obs_inp = self.obs_inp[2:42,:,:]
                        self.x_oi = self.x_oi[2:42,:,:]
                        self.x_rec = self.x_rec[2:42,:,:]
                        self.x_rec_ssh = self.x_rec[2:42,:,:]

                print('... Save nc file with all results : '+path_save1)
                print( self.x_rec.shape )
                print( self.v_rec.shape )
                print( self.u_gt.shape )
                save_netcdf_uv(saved_path1=path_save1, 
                                        gt=self.x_gt, obs = self.obs_inp , oi= self.x_oi, pred=self.x_rec_ssh, 
                                        u_gt=self.u_gt, v_gt=self.v_gt, 
                                        u_pred=self.u_rec, v_pred=self.v_rec,
                                        sst_feat=self.x_sst_feat_ssh,
                                        lon=self.test_lon, lat=self.test_lat, time=self.test_dates)#, time_units=None)


    def teardown(self, stage='test'):

        self.logger.log_hyperparams(
                {**self.hparams},
                self.latest_metrics
    )

    def get_init_state(self, batch, state=(None,),mask_sampling = None):
        if state[0] is not None:
            return state[0]

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt = batch
        else:
            #targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt = batch
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = batch
        
        if mask_sampling is not None :
            init_u = mask_sampling * u_gt
            init_v = mask_sampling * v_gt
        else:
            init_u = torch.zeros_like(targets_GT)
            init_v = torch.zeros_like(targets_GT)
            
        if self.aug_state :
            init_state = torch.cat((targets_OI,
                                    inputs_Mask * (inputs_obs - targets_OI),
                                    inputs_Mask * (inputs_obs - targets_OI),
                                    init_u,init_v),
                                   dim=1)
        else:
            init_state = torch.cat((targets_OI,
                                    inputs_Mask * (inputs_obs - targets_OI),
                                    init_u,init_v),
                                   dim=1)

        if self.use_sst_state :
            init_state = torch.cat((init_state,
                                    sst_gt,),
                                   dim=1)
        return init_state

    def loss_ae(self, state_out):
        return torch.mean((self.model.phi_r(state_out) - state_out) ** 2)

    def sla_loss(self, gt, out):
        g_outputs_x, g_outputs_y = self.gradient_img(out)
        g_gt_x, g_gt_y = self.gradient_img(gt)

        loss = NN_4DVar.compute_spatio_temp_weighted_loss((out - gt), self.patch_weight)
        loss_grad = (
                NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_x - g_gt_x, self.grad_crop(self.patch_weight))
            +    NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_y - g_gt_y, self.grad_crop(self.patch_weight))
        )

        return loss, loss_grad

    def div_loss(self, gt, out):

        return NN_4DVar.compute_spatio_temp_weighted_loss( (out - gt), self.patch_weight[:,1:-1,1:-1])

    def uv_loss(self, gt, out):

        loss = NN_4DVar.compute_spatio_temp_weighted_loss((out[0] - gt[0]), self.patch_weight)
        loss = loss + NN_4DVar.compute_spatio_temp_weighted_loss((out[1] - gt[1]), self.patch_weight)

        return loss

    def reg_loss(self, y_gt, oi, out, out_lr, out_lrhr):
        l_ae = self.loss_ae(out_lrhr)
        l_ae_gt = self.loss_ae(y_gt)
        l_sr = NN_4DVar.compute_spatio_temp_weighted_loss(out_lr - oi, self.patch_weight)

        gt_lr = self.model_LR(oi)
        out_lr_bis = self.model_LR(out)
        l_lr = NN_4DVar.compute_spatio_temp_weighted_loss(out_lr_bis - gt_lr, self.model_LR(self.patch_weight))

        return l_ae, l_ae_gt, l_sr, l_lr

    def compute_loss(self, batch, phase, state_init=(None,)):

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = batch

        if self.scale_dwscaling_sst > 1 :
            sst_gt = torch.nn.functional.avg_pool2d(sst_gt, (int(self.scale_dwscaling_sst),int(self.scale_dwscaling_sst)))
            sst_gt = torch.nn.functional.interpolate(sst_gt, scale_factor=self.scale_dwscaling_sst, mode='bicubic')
            
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
                        ('mse_uv', 0.),
                        ('mseGOI', 0.),
                        ('l0_samp', 0.),
                        ('l1_samp', 0.)])
                    )
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        u_gt_wo_nan = u_gt.where(~u_gt.isnan(), torch.zeros_like(u_gt) )
        v_gt_wo_nan = v_gt.where(~v_gt.isnan(), torch.zeros_like(u_gt) )
                
        if self.model_sampling_uv is not None :
            w_sampling_uv = self.model_sampling_uv( sst_gt )
            w_sampling_uv = w_sampling_uv[1]
            
            #mask_sampling_uv = torch.bernoulli( w_sampling_uv )
            mask_sampling_uv = 1. - torch.nn.functional.threshold( 1.0 - w_sampling_uv , 0.9 , 0.)
            obs = torch.cat( (targets_OI, inputs_Mask * (inputs_obs - targets_OI) , u_gt_wo_nan , v_gt_wo_nan ) ,dim=1)
            
            #print('%f '%( float( self.hparams.dT / (self.hparams.dT - int(self.hparams.dT/2))) * torch.mean(w_sampling_uv)) )
        else:
            mask_sampling_uv = torch.zeros_like(u_gt) 
            obs = torch.cat( (targets_OI, inputs_Mask * (inputs_obs - targets_OI), 0. * targets_OI ,  0. * targets_OI ) ,dim=1)
            
        new_masks = torch.cat( (torch.ones_like(inputs_Mask), inputs_Mask, mask_sampling_uv, mask_sampling_uv) , dim=1)

        state = self.get_init_state(batch, state_init)

        if self.aug_state :
            obs = torch.cat( (obs, 0. * targets_OI,) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.zeros_like(inputs_Mask)), dim=1)
        
        if self.use_sst_state :
            obs = torch.cat( (obs,sst_gt,) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.ones_like(inputs_Mask)), dim=1)

        if self.use_sst_obs :
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]

        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT)

        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            
            if self.hparams.n_grad > 0 :                
                state = torch.autograd.Variable(state, requires_grad=True)
                outputs, hidden_new, cell_new, normgrad = self.model(state, obs, new_masks, *state_init[1:])
    
                if (phase == 'val') or (phase == 'test'):
                    outputs = outputs.detach()
    
                outputsSLRHR = outputs
                outputsSLR = outputs[:, 0:self.hparams.dT, :, :]
                if self.aug_state :
                    outputs = outputsSLR + outputsSLRHR[:, 2*self.hparams.dT:3*self.hparams.dT, :, :]
                    outputs_u = outputsSLRHR[:, 3*self.hparams.dT:4*self.hparams.dT, :, :]
                    outputs_v = outputsSLRHR[:, 4*self.hparams.dT:5*self.hparams.dT, :, :]
                else:
                    outputs = outputsSLR + outputsSLRHR[:, self.hparams.dT:2*self.hparams.dT, :, :]
                    outputs_u = outputsSLRHR[:, 2*self.hparams.dT:3*self.hparams.dT, :, :]
                    outputs_v = outputsSLRHR[:, 3*self.hparams.dT:4*self.hparams.dT, :, :]

                # compute divergence for current field    
                # set dx/dy scaling from (lat,lon) position
                if self.flag_compute_div_with_lat_scaling :
                    dlat = lat[0,1]-lat[0,0]
                    dlon = lon[0,1]-lon[0,0]
                    print('dlat,dlon = %f -- %f'%( dlat.detach().cpu().numpy(),dlon.detach().cpu().numpy() ))
                    
                    self.compute_dlatlon2dxdy_scaling(lat,lon,dlat,outputs_u.size(1))
                    
                    #print( self.alpha_dy, flush = True )
                    
                    print(torch.mean(self.alpha_dx[0,0,0,:]) )
                    print(torch.min(self.alpha_dx[0,0,0,:]),flush=True )
                    print(torch.max(self.alpha_dx[0,0,0,:]),flush=True )
                    
                div_rec =  self.compute_div(outputs_u,outputs_v)
                div_gt =  self.compute_div(u_gt_wo_nan,v_gt_wo_nan)
  
                # median filter
                if self.median_filter_width > 1:
                    outputs = kornia.filters.median_blur(outputs, (self.median_filter_width, self.median_filter_width))
    
                # reconstruction losses
                # projection losses
    
                yGT = torch.cat((targets_OI,
                                 targets_GT_wo_nan - outputsSLR),
                                dim=1)
                if self.aug_state :
                    yGT = torch.cat((yGT, targets_GT_wo_nan - outputsSLR), dim=1)
                
                yGT = torch.cat((yGT, u_gt_wo_nan, v_gt_wo_nan), dim=1)
                           
                if self.use_sst_state :
                    yGT = torch.cat((yGT,sst_gt), dim=1)
    
                loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
                loss_uv = self.uv_loss( [outputs_u,outputs_v], [u_gt_wo_nan,v_gt_wo_nan])
                
                loss_div = self.div_loss( div_rec , div_gt ) 

                #print('.....')
                #print( loss_All )
                #print( loss_GAll )
                #print( loss_uv )
                #print( loss_div )

                loss_OI, loss_GOI = self.sla_loss(targets_OI, targets_GT_wo_nan)
                loss_AE, loss_AE_GT, loss_SR, loss_LR =  self.reg_loss(
                    yGT, targets_OI, outputs, outputsSLR, outputsSLRHR
                )
                #print('  %f'%torch.mean( w_sampling_uv ))
                
                if self.model_sampling_uv is not None :
                    loss_l1_sampling_uv = float( self.hparams.dT / (self.hparams.dT - int(self.hparams.dT/2))) *  torch.mean( w_sampling_uv )
                    loss_l1_sampling_uv = torch.nn.functional.relu( loss_l1_sampling_uv - self.hparams.thr_l1_sampling_uv )
                    loss_l0_sampling_uv = float( self.hparams.dT / (self.hparams.dT - int(self.hparams.dT/2))) * torch.mean( mask_sampling_uv ) 


                # total loss
                loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                loss += self.hparams.alpha_mse_uv * loss_uv + self.hparams.alpha_mse_div * loss_div
                loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
                loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
                if self.model_sampling_uv is not None :
                    loss += self.hparams.alpha_sampling_uv * loss_l1_sampling_uv
            else:                
                outputs = self.model.phi_r(obs)
                outputs = outputs[:, 0:self.hparams.dT, :, :] + outputs[:, self.hparams.dT:2*self.hparams.dT, :, :]
                outputs_u = outputs[:, 2*self.hparams.dT:3*self.hparams.dT, :, :]
                outputs_v = outputs[:, 3*self.hparams.dT:4*self.hparams.dT, :, :]

                # compute divergence for current field   
                flag_compute_div_with_lat = True
                if flag_compute_div_with_lat :
                    div_rec = self.div_field_with_lat_scaling(outputs_u,outputs_v,lat)
                    div_gt = self.div_div_field_with_lat_scalingfield(u_gt,v_gt,lat)
                else:
                    div_rec = self.div_field(outputs_u,outputs_v)
                    div_gt = self.div_field(u_gt,v_gt)
 
                loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
                loss_uv = self.uv_loss( [outputs_u,outputs_v], [u_gt_wo_nan,v_gt_wo_nan])                
                loss_div = self.div_loss( div_rec , div_gt ) 
                loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll
                loss += self.hparams.alpha_mse_uv * loss_uv + self.hparams.alpha_mse_div * loss_div
                loss_OI, loss_GOI = self.sla_loss(targets_OI, targets_GT_wo_nan)
                
                outputsSLRHR = None #0. * outputs
                hidden_new = None #0. * outputs
                cell_new = None # . * outputs
                normgrad = 0. 
            # metrics
            # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_targets_GT_x, g_targets_GT_y) , self.grad_crop(self.patch_weight))
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            mse_uv = loss_uv.detach()
            mse_div = loss_div.detach()
            if self.model_sampling_uv is not None :
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
                ('l1_samp', l1_samp)])

        if ( (phase == 'val') or (phase == 'test') ) & ( self.use_sst == True ) :
            out_feat = sst_gt[:,int(self.hparams.dT/2),:,:].view(-1,1,sst_gt.size(2),sst_gt.size(3))
            
            if self.use_sst_obs :
                #sst_feat = self.model.model_H.conv21( inputs_SST )
                out_feat = torch.cat( (out_feat,self.model.model_H.extract_sst_feature( sst_gt )) , dim = 1 )
                ssh_feat = self.model.model_H.extract_state_feature( outputsSLRHR )
                out_feat = torch.cat( (out_feat,ssh_feat) , dim=1)
                
            if self.model_sampling_uv is not None :
                out_feat = torch.cat( (out_feat,w_sampling_uv) , dim=1)
                    
            return loss, [outputs,outputs_u,outputs_v], [outputsSLRHR, hidden_new, cell_new, normgrad], metrics, out_feat
            
        else:
            return loss, [outputs,outputs_u,outputs_v], [outputsSLRHR, hidden_new, cell_new, normgrad], metrics

class LitModelCycleLR(LitModelUV):
    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        return {
            'optimizer': opt,
        'lr_scheduler': torch.optim.lr_scheduler.CyclicLR(
            opt, **self.hparams.cycle_lr_kwargs),
        'monitor': 'val_loss'
    }

    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print('... Update Iterations number #%d: NGrad = %d -- ' % (
                    self.current_epoch, self.hparams.nb_grad_update[indx]))

                self.hparams.n_grad = self.hparams.nb_grad_update[indx]
                self.model.n_grad = self.hparams.n_grad
                print("ngrad iter", self.model.n_grad)

if __name__ =='__main__':

    import hydra
    import importlib
    from hydra.utils import instantiate, get_class, call
    import hydra_main
    import lit_model_augstate

    importlib.reload(lit_model_augstate)
    importlib.reload(hydra_main)
    from utils import get_cfg, get_dm, get_model
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
    import hydra_config
    # cfg_n, ckpt = 'qxp16_aug2_dp240_swot_w_oi_map_no_sst_ng5x3cas_l2_dp025_00', 'archive_dash/qxp16_aug2_dp240_swot_w_oi_map_no_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=170-val_loss=0.0164.ckpt'
    cfg_n, ckpt = 'full_core', 'results/xpmultigpus/xphack4g_augx4/version_0/checkpoints/modelCalSLAInterpGF-epoch=26-val_loss=1.4156.ckpt'
    # cfg_n, ckpt = 'full_core', 'archive_dash/xp_interp_dt7_240_h/version_5/checkpoints/modelCalSLAInterpGF-epoch=47-val_loss=1.3773.ckpt'

    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=181-val_loss=0.0101.ckpt'
    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=191-val_loss=0.0101.ckpt'
    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=192-val_loss=0.0102.ckpt'
    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=192-val_loss=0.0102.ckpt'

    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=187-val_loss=0.0105.ckpt'
    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=198-val_loss=0.0103.ckpt'
    # cfg_n, ckpt = 'qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00', 'results/xp17/qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00/version_0/checkpoints/cal-epoch=194-val_loss=0.0106.ckpt'

    # cfg_n, ckpt = 'full_core_sst', 'archive_dash/xp_interp_dt7_240_h_sst/version_0/checkpoints/modelCalSLAInterpGF-epoch=114-val_loss=0.6698.ckpt'
    # cfg_n = f"xp_aug/xp_repro/{cfg_n}"
    # cfg_n, ckpt = 'full_core_hanning_sst', 'results/xpnew/hanning_sst/version_1/checkpoints/modelCalSLAInterpGF-epoch=95-val_loss=0.3419.ckpt'
    # cfg_n, ckpt = 'full_core_hanning_sst', 'results/xpnew/hanning_sst/version_1/checkpoints/modelCalSLAInterpGF-epoch=92-val_loss=0.3393.ckpt'
    # cfg_n, ckpt = 'full_core_hanning_sst', 'results/xpnew/hanning_sst/version_1/checkpoints/modelCalSLAInterpGF-epoch=99-val_loss=0.3438.ckpt'
    # cfg_n, ckpt = 'full_core_sst_fft', 'results/xpnew/sst_fft/version_0/checkpoints/modelCalSLAInterpGF-epoch=59-val_loss=2.0084.ckpt'
    # cfg_n, ckpt = 'full_core_sst_fft', 'results/xpnew/sst_fft/version_0/checkpoints/modelCalSLAInterpGF-epoch=92-val_loss=2.0447.ckpt'
    # cfg_n, ckpt = 'cycle_lr_sst', 'results/xpnew/xp_cycle_lr_sst/version_1/checkpoints/modelCalSLAInterpGF-epoch=103-val_loss=0.9093.ckpt'
    # cfg_n, ckpt = 'full_core_hanning', 'results/xpnew/hanning/version_0/checkpoints/modelCalSLAInterpGF-epoch=91-val_loss=0.5461.ckpt'
    # cfg_n, ckpt = 'full_core_hanning', 'results/xpnew/hanning/version_2/checkpoints/modelCalSLAInterpGF-epoch=88-val_loss=0.5654.ckpt'
    # cfg_n, ckpt = 'full_core_hanning', 'results/xpnew/hanning/version_2/checkpoints/modelCalSLAInterpGF-epoch=129-val_loss=0.5692.ckpt'
    # cfg_n, ckpt = 'full_core_hanning', 'results/xpmultigpus/xphack4g_hannAdamW/version_0/checkpoints/modelCalSLAInterpGF-epoch=129-val_loss=0.5664.ckpt'
    # cfg_n, ckpt = 'full_core_hanning', 'results/xpmultigpus/xphack4g_daugx3hann/version_1/checkpoints/modelCalSLAInterpGF-epoch=32-val_loss=0.5734.ckpt'
    # cfg_n, ckpt = 'full_core_hanning', 'results/xpmultigpus/xphack4g_daugx3hann/version_1/checkpoints/modelCalSLAInterpGF-epoch=34-val_loss=0.5793.ckpt'
    cfg_n, ckpt = 'full_core_hanning', 'results/xpmultigpus/xphack4g_daugx3hann/version_1/checkpoints/modelCalSLAInterpGF-epoch=43-val_loss=0.5658.ckpt'
    # cfg_n, ckpt = 'full_core_hanning_t_grad', 'results/xpnew/hanning_grad/version_0/checkpoints/modelCalSLAInterpGF-epoch=102-val_loss=3.7019.ckpt'
    cfg_n = f"xp_aug/xp_repro/{cfg_n}"
    dm = get_dm(cfg_n, setup=False,
            add_overrides=[
                # 'params.files_cfg.obs_mask_path=/gpfsssd/scratch/rech/yrf/ual82ir/sla-data-registry/CalData/cal_data_new_errs.nc',
                # 'params.files_cfg.obs_mask_path=/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d.nc',
                # 'params.files_cfg.obs_mask_var=four_nadirs'
                # 'params.files_cfg.obs_mask_var=swot_nadirs_no_noise'
            ]


    )
    mod = get_model(
            cfg_n,
            ckpt,
            dm=dm)
    mod.hparams
    cfg = get_cfg(cfg_n)
    # cfg = get_cfg("xp_aug/xp_repro/quentin_repro")
    print(OmegaConf.to_yaml(cfg))
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    
    mod = _mod or self._get_model(ckpt_path=ckpt)

    trainer = pl.Trainer(num_nodes=1, gpus=1, accelerator=None, **trainer_kwargs)
    trainer.test(mod, dataloaders=self.dataloaders[dataloader])

    
    runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner.test(ckpt)
    mod.test_figs['psd']
