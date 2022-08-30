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

def compute_coriolis_force(lat,flag_mean_coriolis=False):
    omega = 7.2921e-5 # rad/s
    f = 2 * omega * np.sin(lat)
    
    if flag_mean_coriolis == True :
        f = np.mean(f) * np.ones((f.shape)) 
    
    return f

def compute_uv_geo_with_coriolis(ssh,lat,lon,sigma=0.5,alpha_uv_geo = 1.,flag_mean_coriolis=False):

    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    # coriolis / lat/lon scaling
    grid_lat = lat.reshape( (1,ssh.shape[1],1))
    grid_lat = np.tile( grid_lat , (ssh.shape[0],1,ssh.shape[2]) )
    grid_lon = lon.reshape( (1,1,ssh.shape[2]))
    grid_lon = np.tile( grid_lon , (ssh.shape[0],ssh.shape[1],1) )
    
    f_c = compute_coriolis_force(grid_lat,flag_mean_coriolis=flag_mean_coriolis)
    dx_from_dlon , dy_from_dlat = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     

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
        
    dy_from_dlat =  compute_c(lat,lon,dlat,0.)
    dx_from_dlon =  compute_c(lat,lon,0.,dlon)
    
    return dx_from_dlon , dy_from_dlat

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


def compute_div_curl_strain_with_lat_lon(u,v,lat,lon,sigma=1.0):
    dlat = lat[1] - lat[0]
    dlon = lon[1] - lon[0]

    # coriolis / lat/lon scaling
    grid_lat = lat.reshape( (1,u.shape[1],1))
    grid_lat = np.tile( grid_lat , (v.shape[0],1,v.shape[2]) )
    grid_lon = lon.reshape( (1,1,v.shape[2]))
    grid_lon = np.tile( grid_lon , (v.shape[0],v.shape[1],1) )
    
    dx_from_dlon , dy_from_dlat = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     

    du_dx = compute_gradx( u , sigma = sigma )
    dv_dy = compute_grady( v , sigma = sigma )

    du_dy = compute_grady( u , sigma = sigma )
    dv_dx = compute_gradx( v , sigma = sigma )

    du_dx = du_dx / dx_from_dlon 
    dv_dx = dv_dx / dx_from_dlon 
        
    du_dy = du_dy / dy_from_dlat  
    dv_dy = dv_dy / dy_from_dlat  

    strain = np.sqrt( ( dv_dx + du_dy ) **2 + (du_dx - dv_dy) **2 )

    div = du_dx + dv_dy
    curl =  du_dy - dv_dx

    return div,curl,strain#,dx_from_dlon , dy_from_dlat
    #return div,curl,strain, np.abs(du_dx) , np.abs(du_dy)

class Torch_compute_derivatives_with_lon_lat(torch.nn.Module):
    def __init__(self,dT=7,_filter='diff-non-centered'):
        super(Torch_compute_derivatives_with_lon_lat, self).__init__()

        if _filter == 'sobel':
            a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='reflect')
            with torch.no_grad():
                self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

            b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='reflect')
            with torch.no_grad():
                self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif _filter == 'diff-non-centered':
            #a = np.array([[0., 0., 0.], [0.3, 0.4, -0.7], [0., 0., 0.]])
            a = np.array([[0., 0., 0.], [-0.7, 0.4, 0.3], [0., 0., 0.]])

            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='reflect')
            with torch.no_grad():
                self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            #b = np.array([[0., 0.3, 0.], [0., 0.4, 0.], [0., -0.7, 0.]])
            b = np.transpose(a)

            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='reflect')
            with torch.no_grad():
                self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        elif filter == 'diff':
            a = np.array([[0., 0., 0.], [0., 1., -1.], [0., 0., 0.]])
            self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='reflect')
            with torch.no_grad():
                self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            b = np.array([[0., 0.3, 0.], [0., 1., 0.], [0., -1., 0.]])
            self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False,padding_mode='reflect')
            with torch.no_grad():
                self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
        
        a = np.array([[0., 0.25, 0.], [0.25, 0., 0.25], [0., 0.25, 0.]])
        self.heat_filter = torch.nn.Conv2d(dT, dT, kernel_size=3, padding=1, bias=False,padding_mode='reflect')
        with torch.no_grad():
            self.heat_filter.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

        a = np.array([[0., 0.25, 0.], [0.25, 0., 0.25], [0., 0.25, 0.]])
        self.heat_filter_all_channels = torch.nn.Conv2d(dT, dT, kernel_size=3, groups=dT, padding=1, bias=False,padding_mode='reflect')
        with torch.no_grad():
            a = np.tile(a,(dT,1,1,1))
            self.heat_filter_all_channels.weight = torch.nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)

        self.eps = 1e-10#torch.Tensor([1.*1e-10])
    
    def compute_c(self,lat,lon,dlat,dlon):
        
        a = torch.sin(dlat / 2. )**2 + torch.cos(lat) ** 2 * torch.sin( dlon / 2. )**2

        c = 2. * 6.371e6 * torch.atan2( torch.sqrt(a + self.eps), torch.sqrt(1. - a + self.eps ))       
        #c = c.type(torch.cuda.FloatTensor)
        
        return c        
        #return 2. * 6.371 * torch.atan2( torch.sqrt(a + self.eps), torch.sqrt(1. - a + self.eps ))        

    def compute_dx_dy_dlat_dlon(self,lat,lon,dlat,dlon):
        
        dy_from_dlat =  self.compute_c(lat,lon,dlat,0.*dlon)
        dx_from_dlon =  self.compute_c(lat,lon,0.*dlat,dlon)
    
        return dx_from_dlon , dy_from_dlat
            
    def compute_gradx(self,u,sigma=0.):
               
        if sigma > 0. :
            u = kornia.filters.gaussian_blur2d(u, (5,5), (sigma,sigma), border_type='reflect')
        
        G_x = self.convGx(u.view(-1, 1, u.size(2), u.size(3)))
        G_x = G_x.view(-1,u.size(1), u.size(2), u.size(3))

        return G_x

    def compute_grady(self,u,sigma=0.):
        if sigma > 0. :
            u = kornia.filters.gaussian_blur2d(u, (5,5), (sigma,sigma), border_type='reflect')
        
        G_y = self.convGy(u.view(-1, 1, u.size(2), u.size(3)))
        G_y = G_y.view(-1,u.size(1), u.size(2), u.size(3))

        return G_y

    def compute_gradxy(self,u,sigma=0.):
               
        if sigma > 0. :
            u = kornia.filters.gaussian_blur2d(u, (3,3), (sigma,sigma), border_type='reflect')

        G_x  = self.convGx( u[:,0,:,:].view(-1,1,u.size(2), u.size(3)) )
        G_y  = self.convGy( u[:,0,:,:].view(-1,1,u.size(2), u.size(3)) )
        
        for kk in range(1,u.size(1)):
            _G_x  = self.convGx( u[:,kk,:,:].view(-1,1,u.size(2), u.size(3)) )
            _G_y  = self.convGy( u[:,kk,:,:].view(-1,1,u.size(2), u.size(3)) )
                
            G_x  = torch.cat( (G_x,_G_x) , dim = 1 )
            G_y  = torch.cat( (G_y,_G_y) , dim = 1 )
            
        return G_x,G_y
    
    def compute_coriolis_force(self,lat,flag_mean_coriolis=False):
        omega = 7.2921e-5 # rad/s
        f = 2 * omega * torch.sin(lat)
        
        if flag_mean_coriolis == True :
            f = torch.mean(f) * torch.ones((f.size())) 
        
        #f = f.type(torch.cuda.FloatTensor)

        return f
        
    def compute_geo_velocities(self,ssh,lat,lon,sigma=0.,alpha_uv_geo=9.81,flag_mean_coriolis=False):

        dlat = lat[0,1]-lat[0,0]
        dlon = lon[0,1]-lon[0,0]
        
        # coriolis / lat/lon scaling
        grid_lat = lat.view(ssh.size(0),1,ssh.size(2),1)
        grid_lat = grid_lat.repeat(1,ssh.size(1),1,ssh.size(3))
        grid_lon = lon.view(ssh.size(0),1,1,ssh.size(3))
        grid_lon = grid_lon.repeat(1,ssh.size(1),ssh.size(2),1)
        
        dx_from_dlon , dy_from_dlat = self.compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     
        f_c = self.compute_coriolis_force(grid_lat,flag_mean_coriolis=flag_mean_coriolis)
      
        dssh_dx , dssh_dy = self.compute_gradxy( ssh , sigma=sigma )

        #print(' dssh_dy %f '%torch.mean( torch.abs(dssh_dy)).detach().cpu().numpy() )
        #print(' dssh_dx %f '%torch.mean( torch.abs(dssh_dx)).detach().cpu().numpy() )

        dssh_dx = dssh_dx / dx_from_dlon 
        dssh_dy = dssh_dy / dy_from_dlat  

        dssh_dy = ( 1. / f_c ) * dssh_dy
        dssh_dx = ( 1. / f_c  )* dssh_dx

        u_geo = -1. * dssh_dy
        v_geo = 1. * dssh_dx

        u_geo = alpha_uv_geo * u_geo
        v_geo = alpha_uv_geo * v_geo
        
    
        return u_geo,v_geo
        
    def heat_equation_one_channel(self,ssh,mask=None,iter=5,lam=0.2):
        out = torch.clone( ssh )
        for kk in range(0,iter):
            if mask is not None :
                _d = out - mask * self.heat_filter(out)
            else:
                _d = out - self.heat_filter(out) 
            out -= lam * _d 
        return out
        
    def heat_equation_all_channels(self,ssh,mask=None,iter=5,lam=0.2):
        out = 1. * ssh
        for kk in range(0,iter):
            if mask is not None :
                _d = out - mask * self.heat_filter_all_channels(out)
            else:
                _d = out - self.heat_filter_all_channels(out) 
            out = out - lam * _d 
        return out

    def heat_equation(self,u,mask=None,iter=5,lam=0.2):
        
        if mask is not None :
            out = self.heat_equation_one_channel(u[:,0,:,:].view(-1,1,u.size(2), u.size(3)),mask[:,0,:,:].view(-1,1,u.size(2), u.size(3)),iter=iter,lam=lam)            
        else:
            out = self.heat_equation_one_channel(u[:,0,:,:].view(-1,1,u.size(2),u.size(3)), iter=iter,lam=lam)
        
        for kk in range(1,u.size(1)):
            if mask is not None :
                _out = self.heat_equation_one_channel(u[:,kk,:,:].view(-1,1,u.size(2),mask[:,kk,:,:].view(-1,1,u.size(2), u.size(3)), u.size(3)),iter=iter,lam=lam)
            else:
                _out = self.heat_equation_one_channel(u[:,kk,:,:].view(-1,1,u.size(2), u.size(3)),iter=iter,lam=lam)
                 
            out  = torch.cat( (out,_out) , dim = 1 )

        return out

    def compute_geo_factor(self,ssh,lat,lon,sigma=0.,alpha_uv_geo=9.81,flag_mean_coriolis=False):

        dlat = lat[0,1]-lat[0,0]
        dlon = lon[0,1]-lon[0,0]
        
        # coriolis / lat/lon scaling
        grid_lat = lat.view(ssh.size(0),1,ssh.size(2),1)
        grid_lat = grid_lat.repeat(1,ssh.size(1),1,ssh.size(3))
        grid_lon = lon.view(ssh.size(0),1,1,ssh.size(3))
        grid_lon = grid_lon.repeat(1,ssh.size(1),ssh.size(2),1)
        
        dx_from_dlon , dy_from_dlat = self.compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     
        f_c = self.compute_coriolis_force(grid_lat,flag_mean_coriolis=flag_mean_coriolis)

        dssh_dx = alpha_uv_geo / dx_from_dlon 
        dssh_dy = alpha_uv_geo / dy_from_dlat  

        dssh_dy = ( 1. / f_c ) * dssh_dy
        dssh_dx = ( 1. / f_c  )* dssh_dx

        factor_u_geo = -1. * dssh_dy
        factor_v_geo =  1. * dssh_dx
            
        return factor_u_geo , factor_v_geo

    def compute_div_curl_strain(self,u,v,lat,lon,sigma=0.):
        
        dlat = lat[0,1]-lat[0,0]
        dlon = lon[0,1]-lon[0,0]
        
        # coriolis / lat/lon scaling
        grid_lat = lat.view(u.size(0),1,u.size(2),1)
        grid_lat = grid_lat.repeat(1,u.size(1),1,u.size(3))
        grid_lon = lon.view(u.size(0),1,1,u.size(3))
        grid_lon = grid_lon.repeat(1,u.size(1),u.size(2),1)
        
        dx_from_dlon , dy_from_dlat = self.compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)

        du_dx , du_dy = self.compute_gradxy( u , sigma=sigma )
        dv_dx , dv_dy = self.compute_gradxy( v , sigma=sigma )
        
        du_dx = du_dx / dx_from_dlon 
        dv_dx = dv_dx / dx_from_dlon 

        du_dy = du_dy / dy_from_dlat  
        dv_dy = dv_dy / dy_from_dlat  

        strain = torch.sqrt( ( dv_dx + du_dy ) **2 + (du_dx - dv_dy) **2 + self.eps )

        div = du_dx + dv_dy
        curl =  du_dy - dv_dx

        return div,curl,strain#,dx_from_dlon,dy_from_dlat
        #return div,curl,strain, torch.abs(du_dx) , torch.abs(du_dy)
    
    def forward(self):
        return 1.

if 1*0 :
    def compute_div_with_lat_lon(u,v,lat,lon,sigma=1.0):
        dlat = lat[1] - lat[0]
        dlon = lon[1] - lon[0]
    
        # coriolis / lat/lon scaling
        grid_lat = lat.reshape( (1,u.shape[1],1))
        grid_lat = np.tile( grid_lat , (v.shape[0],1,v.shape[2]) )
        grid_lon = lon.reshape( (1,1,v.shape[2]))
        grid_lon = np.tile( grid_lon , (v.shape[0],v.shape[1],1) )
        
        dx_from_dlon , dy_from_dlat = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     
    
        du_dx = compute_gradx( u , sigma = sigma )
        dv_dy = compute_grady( v , sigma = sigma )
        
        du_dx = du_dx / dx_from_dlon 
        dv_dy = dv_dy / dy_from_dlat  
    
        return du_dx + dv_dy
    
    def compute_curl_with_lat_lon(u,v,lat,lon,sigma=1.0):
        
        dlat = lat[1] - lat[0]
        dlon = lon[1] - lon[0]
    
        # coriolis / lat/lon scaling
        grid_lat = lat.reshape( (1,u.shape[1],1))
        grid_lat = np.tile( grid_lat , (v.shape[0],1,v.shape[2]) )
        grid_lon = lon.reshape( (1,1,v.shape[2]))
        grid_lon = np.tile( grid_lon , (v.shape[0],v.shape[1],1) )
        
        dx_from_dlon , dy_from_dlat = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     
        
        du_dy = compute_grady( u , sigma = sigma )
        dv_dx = compute_gradx( v , sigma = sigma )
    
        dv_dx = dv_dx / dx_from_dlon 
        du_dy = du_dy / dy_from_dlat  
        
        return du_dy - dv_dx
    
    def compute_strain_with_lat_lon(u,v,lat,lon,sigma=1.0):
        dlat = lat[1] - lat[0]
        dlon = lon[1] - lon[0]
    
        # coriolis / lat/lon scaling
        grid_lat = lat.reshape( (1,u.shape[1],1))
        grid_lat = np.tile( grid_lat , (v.shape[0],1,v.shape[2]) )
        grid_lon = lon.reshape( (1,1,v.shape[2]))
        grid_lon = np.tile( grid_lon , (v.shape[0],v.shape[1],1) )
        
        dx_from_dlon , dy_from_dlat = compute_dx_dy_dlat_dlon(grid_lat,grid_lon,dlat,dlon)     
    
        du_dx = compute_gradx( u , sigma = sigma )
        dv_dy = compute_grady( v , sigma = sigma )
    
        du_dy = compute_grady( u , sigma = sigma )
        dv_dx = compute_gradx( v , sigma = sigma )
    
        du_dx = du_dx / dx_from_dlon 
        dv_dx = dv_dx / dx_from_dlon 
    
        du_dy = du_dy / dy_from_dlat  
        dv_dy = dv_dy / dy_from_dlat  
    
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
    print('...... Set mdoel %d'%hparams.use_sst_obs,flush=True)
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
            print('...... residual_wrt_geo_velocities = %d'%hparams.residual_wrt_geo_velocities,flush=True)
            if ( hparams.residual_wrt_geo_velocities == 3 ) or ( hparams.residual_wrt_geo_velocities == 4 ): 
                return NN_4DVar.Solver_Grad_4DVarNN(
                            Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                            Model_HwithSSTBN_nolin_tanh_withlatlon(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                            NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                                hparams.dim_grad_solver, hparams.dropout),
                            hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
            else:
                return NN_4DVar.Solver_Grad_4DVarNN(
                            Phi_r(hparams.shape_state[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                                hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic, hparams.phi_param),
                            Model_HwithSSTBN_nolin_tanh(hparams.shape_state[0], dT=hparams.dT,dim=hparams.dim_obs_sst_feat),
                            NN_4DVar.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                                hparams.dim_grad_solver, hparams.dropout),
                            hparams.norm_obs, hparams.norm_prior, hparams.shape_state, hparams.n_grad * hparams.n_fourdvar_iter)
            
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

def get_cropped_hanning_mask_ronan(patch_size, crop, **kwargs):
    _crop = dict([('time',int( (patch_size['time'] - crop['time']) / 2 )),('lat',crop['lat']),('lon',crop['lon'])])
    #print(crop)
    #print(patch_size)
    #print(_crop,flush=True)
    
    pw = get_constant_crop(patch_size, _crop)
        
    #t_msk =kornia.filters.get_hanning_kernel1d(patch_size['time'])
    t_msk1 = kornia.filters.get_hanning_kernel1d(crop['time'])
    t_msk  = np.zeros((patch_size['time'],))
    t_msk[_crop['time']:patch_size['time']-_crop['time']] = t_msk1.cpu().numpy()
    t_msk = torch.Tensor( t_msk )

    #print(t_msk1,flush=True) 
    #print()
    #print(t_msk1.size(),flush=True) 
    #print(t_msk,flush=True) 
    #print(t_msk.size(),flush=True) 
    
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
class Model_HwithSSTBN_nolin_tanh_withlatlon(torch.nn.Module):
    def __init__(self,shape_data, dT=5,dim=5,width_kernel=3,padding_mode='reflect',type_wgeo=3):
        super(Model_HwithSSTBN_nolin_tanh_withlatlon, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        #print('..... # im obs sst : %d'%dim)
        self.w_kernel = width_kernel

        self.bn_feat = torch.nn.BatchNorm2d(self.dim_obs_channel[1],track_running_stats=False)

        self.var_tr_uv = 1.
        self.dT = dT
        self.aug_state = False
        self.type_w_geo = type_wgeo
        
        self.type_geo_obs = 0
        if self.type_geo_obs == 1:
            dim_state = 4*self.dT#4*self.dT
        elif self.type_geo_obs == 2:
            dim_state = 6*self.dT#4*self.dT
        else:            
            dim_state = 10*self.dT#4*self.dT

        self.convx11 = torch.nn.Conv2d(dim_state, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        self.convy11 = torch.nn.Conv2d(dT, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)

        self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=True,padding_mode=padding_mode)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)
        
        self.lat_rad = None
        self.lon_rad = None
        self.compute_derivativeswith_lon_lat = Torch_compute_derivatives_with_lon_lat(dT=dT)
        self.aug_state = False
        
    def compute_geo_factor(self,outputs, lat_rad, lon_rad,sigma=0.):
        return self.compute_derivativeswith_lon_lat.compute_geo_factor(outputs, lat_rad, lon_rad,sigma=0.)

    def extract_sst_feature(self,y1):
        y1     = self.convy12( torch.tanh( self.convy11(y1) ) )
        y_feat = self.bn_feat( self.convy22( torch.tanh( self.convy21( torch.tanh(y1) ) ) ) )
       
        return y_feat
        
    def extract_state_feature(self,x):
        # ssh component
        #print('..... obs model with lat/lon')
        
        if self.aug_state :
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
        u_geo_factor, v_geo_factor = self.compute_geo_factor(xbar_ssh, self.lat_rad, self.lon_rad,sigma=0.) 

        # latent component
        if self.type_geo_obs == 1 :
            x_ssh_u = u_geo_factor * (xbar_ssh+dx_ssh)
            x_ssh_v = v_geo_factor * (xbar_ssh+dx_ssh)
            x_u_u = u_geo_factor * x_u
            #x_u_v = v_geo_factor * x_u
            #x_v_u = u_geo_factor * x_v
            x_v_v = v_geo_factor * x_v
            x_all = torch.cat((x_ssh_u,x_ssh_v,x_u_u,x_v_v),dim=1)
        elif self.type_geo_obs == 2 :
            xbar_ssh_u = u_geo_factor * xbar_ssh
            xbar_ssh_v = v_geo_factor * xbar_ssh

            dx_ssh_u = u_geo_factor * dx_ssh
            dx_ssh_v = v_geo_factor * dx_ssh
            x_u_u = u_geo_factor * x_u
            #x_u_v = v_geo_factor * x_u
            #x_v_u = u_geo_factor * x_v
            x_v_v = v_geo_factor * x_v
        
            x_all = torch.cat((xbar_ssh_u,xbar_ssh_v,dx_ssh_u,dx_ssh_v,x_u_u,x_v_v),dim=1)
            
        else:
            xbar_ssh_u = u_geo_factor * xbar_ssh
            xbar_ssh_v = v_geo_factor * xbar_ssh

            dx_ssh_u = u_geo_factor * dx_ssh
            dx_ssh_v = v_geo_factor * dx_ssh
            x_u_u = u_geo_factor * x_u
            #x_u_v = v_geo_factor * x_u
            #x_v_u = u_geo_factor * x_v
            x_v_v = v_geo_factor * x_v
        
            x_all = torch.cat((xbar_ssh,xbar_ssh_u,xbar_ssh_v,dx_ssh,dx_ssh_u,dx_ssh_v,x_u,x_u_u,x_v,x_v_v),dim=1)
        
        if 1*0 :
            if self.aug_state :
                if x.size(1) > 5*self.dT :
                    x_all = torch.cat((x_all,x[:, 5*self.dT:, :, :]),dim=1)
            else:
                if x.size(1) > 4*self.dT :
                    x_all = torch.cat((x_all,x[:, 4*self.dT:, :, :]),dim=1)
        
        #print('..... s_all_size')
        #print(x_all.size())
               
        x1     = self.convx12( torch.tanh( self.convx11( x_all ) ) )
        x_feat = self.bn_feat( self.convx22( torch.tanh( self.convx21( torch.tanh(x1) ) ) ) )
        
        return x_feat

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
                
        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]

class LitModelUV(pl.LightningModule):

    MODELS = {
            '4dvarnet': get_4dvarnet,
            '4dvarnet_sst': get_4dvarnet_sst,
            'phi': get_phi
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
        
        self.alpha_dx = 1.
        self.alpha_dy = 1.
        #self.alpha_uv_geo = 1.
        #self.alpha_dx,self.alpha_dy,self.alpha_uv_geo = self.hparams.scaling_ssh_uv

        # create longitudes & latitudes coordinates
        self.test_domain=test_domain
        self.test_coords = None
        self.test_ds_patch_size = None
        self.test_lon = None
        self.test_lat = None
        self.test_dates = None

        self.patch_weight = None

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
        self.hparams.alpha_mse_strain = self.hparams.alpha_mse_strain if hasattr(self.hparams, 'alpha_mse_strain') else 0.

        self.type_div_train_loss = self.hparams.type_div_train_loss if hasattr(self.hparams, 'type_div_train_loss') else 1
        
        self.scale_lr = self.hparams.scale_lr if hasattr(self.hparams, 'scale_lr') else 1.0

        _w = torch.from_numpy(call(self.hparams.patch_weight))
        _w =  torch.nn.functional.avg_pool2d(_w.view(1,-1,_w.size(1),_w.size(2)), (int(self.scale_lr),int(self.scale_lr)))
        self.patch_weight_lr = torch.nn.Parameter(_w.view(-1,_w.size(2),_w.size(3)), requires_grad=False)

        _dt = int( (self.hparams.dT-self.hparams.dT_hr_model)/2 )
        _w = torch.from_numpy(call(self.hparams.patch_weight))
        _w = _w[_dt:_dt+self.hparams.dT_hr_model,:,:]
        self.patch_weight_hr = torch.nn.Parameter(_w, requires_grad=False)

        self.residual_wrt_geo_velocities = self.hparams.residual_wrt_geo_velocities if hasattr(self.hparams, 'residual_wrt_geo_velocities') else 0
        #if self.residual_wrt_geo_velocities > 0 :
        #    self.type_div_train_loss = 1
        self.use_lat_lon_in_obs_model = self.hparams.use_lat_lon_in_obs_model if hasattr(self.hparams, 'use_lat_lon_in_obs_model') else False
        if ( self.residual_wrt_geo_velocities == 3 ) or ( self.residual_wrt_geo_velocities == 4 ):
            self.use_lat_lon_in_obs_model = True
           
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

        self.create_model()
        
        self.grad_crop = lambda t: t[...,1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
                self.grad_crop(2.*kornia.filters.spatial_gradient(t, normalized=True)), 2)
        
        if ( self.residual_wrt_geo_velocities == 3 ) or ( self.residual_wrt_geo_velocities == 4 ):
            self.model.model_H.aug_state = self.hparams.aug_state
            self.model.model_H.var_tr_uv = self.var_tr_uv        
        
        self.compute_derivativeswith_lon_lat_lr = Torch_compute_derivatives_with_lon_lat(dT=self.hparams.dT)
        self.compute_derivativeswith_lon_lat_hr = Torch_compute_derivatives_with_lon_lat(dT=self.hparams.dT_hr_model)

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
        print('..  Div loss type : %d'%self.type_div_train_loss)
        
    def compute_lr_state(self,x):
        return torch.nn.functional.avg_pool2d(x, (int(self.scale_lr),int(self.scale_lr)))
 
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
       
    def update_filename_chkpt(self,filename_chkpt):
        
        old_suffix = '-{epoch:02d}-{val_loss:.4f}'

        suffix_chkpt = '-'+self.hparams.phi_param+'_%03d-augdata'%self.hparams.DimAE
        
        if self.scale_lr > 1.0 :
            suffix_chkpt = suffix_chkpt+'-lr%02d'%int(self.scale_lr)

        if self.scale_dwscaling_sst > 1. :
            suffix_chkpt = suffix_chkpt+'-dws-sst%02d'%int(self.scale_dwscaling_sst)
            
        if self.model_sampling_uv is not None:
            suffix_chkpt = suffix_chkpt+'-sampling_sst_%d_%03d'%(self.hparams.nb_feat_sampling_operator,int(100*self.hparams.thr_l1_sampling_uv))
        
        if self.hparams.n_grad > 0 :
            
            if self.hparams.aug_state :
                suffix_chkpt = suffix_chkpt+'-augstate-dT%02d'%(self.hparams.dT)
            if self.use_sst_state :
                suffix_chkpt = suffix_chkpt+'-mmstate-augstate-dT%02d'%(self.hparams.dT)
            
            if self.use_sst_obs :
                suffix_chkpt = suffix_chkpt+'-sstobs-'+self.hparams.sst_model+'_%02d'%(self.hparams.dim_obs_sst_feat)
            
            if self.residual_wrt_geo_velocities > 0  :
                suffix_chkpt = suffix_chkpt+'-wgeo%d'%self.residual_wrt_geo_velocities
            elif self.type_div_train_loss == 1 :
                suffix_chkpt = suffix_chkpt+'-nowgeo'

            if ( self.hparams.alpha_mse_strain == 0. ) | ( self.hparams.alpha_mse_div == 0. ) :
                if ( self.hparams.alpha_mse_strain == 0. ) & ( self.hparams.alpha_mse_div == 0. ) :                    
                    suffix_chkpt = suffix_chkpt+'-nodivstrain'
                elif self.hparams.alpha_mse_strain == 0. :                    
                    suffix_chkpt = suffix_chkpt+'-nostrain'
                else :                    
                    suffix_chkpt = suffix_chkpt+'-nodiv'
                                            
            suffix_chkpt = suffix_chkpt+'-grad_%02d_%02d_%03d'%(self.hparams.n_grad,self.hparams.k_n_grad,self.hparams.dim_grad_solver)
        else:
            if ( self.use_sst ) & ( self.use_sst_state ) :
                suffix_chkpt = suffix_chkpt+'-DirectInv-wSST'
            else:
                suffix_chkpt = suffix_chkpt+'-DirectInv'
            suffix_chkpt = suffix_chkpt+'-dT%02d'%(self.hparams.dT)
                
            if ( self.hparams.alpha_mse_strain == 0. ) | ( self.hparams.alpha_mse_div == 0. ) :
                if ( self.hparams.alpha_mse_strain == 0. ) & ( self.hparams.alpha_mse_div == 0. ) :                    
                    suffix_chkpt = suffix_chkpt+'-nodivstrain'
                elif self.hparams.alpha_mse_strain == 0. :                    
                    suffix_chkpt = suffix_chkpt+'-nostrain'
                else :                    
                    suffix_chkpt = suffix_chkpt+'-nodiv'
        suffix_chkpt = suffix_chkpt+old_suffix
        
        return filename_chkpt.replace(old_suffix,suffix_chkpt)
    
    def create_model(self):
        print('...... Create 4dvarnet models')

        print('...... Set low-resolution model',flush=True)
        print('.... shape state lr : %dx%dx%d'%(self.hparams.shape_state_lr[0],self.hparams.shape_state_lr[1],self.hparams.shape_state_lr[2]) )
        self.hparams.shape_state_lr[1] = int( self.hparams.shape_state_lr[1] / self.hparams.scale_lr )
        self.hparams.shape_state_lr[2] = int( self.hparams.shape_state_lr[2] / self.hparams.scale_lr )
        self.hparams.shape_state = self.hparams.shape_state_lr
        self.model_4dvarnet_lr = get_4dvarnet_sst(self.hparams)
        
        print('...... Set fine-scale model',flush=True)
        self.hparams.shape_state_hr[0] = int( self.hparams.dT_hr_model *  self.hparams.shape_state_hr[0] / self.hparams.dT )
        self.hparams.shape_state = self.hparams.shape_state_hr
        dT = self.hparams.dT
        self.hparams.dT = self.hparams.dT_hr_model
        print('.... shape state hr : %dx%dx%d - dT = %d/%d'%(self.hparams.shape_state[0],self.hparams.shape_state[1],self.hparams.shape_state[2],self.hparams.dT,dT) )        
        self.model_4dvarnet_hr = get_4dvarnet_sst(self.hparams)
        self.hparams.state = None
        self.hparams.dT = dT

    def forward(self, batch, phase='test'):
        losses = []
        metrics = []
        state_init_lr = [None]
        state_init_hr = [None]
        out_lr = [None]
        out_hr = [None]
        
        out = None
        
        for _ in range(self.hparams.n_fourdvar_iter):
            if ( phase == 'test' ) & ( self.use_sst ):                
                # run low-resolution model
                _loss_lr, out_lr, state_lr, _metrics,sst_feat = self.compute_loss_lr(batch, phase=phase, out_hr=out_hr, state_init_lr=state_init_lr)
                                
                # run high-resolution model
                _loss, out_hr, state_hr, _metrics_lr,sst_feat_lr = self.compute_loss_hr(batch, phase=phase, out_lr=out_lr, state_init_hr=state_init_hr)
            else:
                # run low-resolution model
                _loss_lr, out, state_lr, _metrics,sst_feat = self.compute_loss_lr(batch, phase=phase, out_hr=out_hr, state_init_lr=state_init_lr)
 
                # run high-resolution model
                _loss, out, state_hr, _metrics_lr,sst_feat_lr = self.compute_loss_hr(batch, phase=phase, out_lr=out_lr,  state_init_hr=state_init_hr)
            
            if self.hparams.n_grad > 0 :
                state_init_lr = [None if s is None else s.detach() for s in state_lr]
                state_init_hr = [None if s is None else s.detach() for s in state_hr]

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
            
            if self.model_sampling_uv is not None :
                optimizer = opt([{'params': self.model_4dvarnet_lr.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_lr.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_lr.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_lr.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    {'params': self.model_sampling_uv.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    ])
            else:
                optimizer = opt([{'params': self.model_4dvarnet_lr.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_lr.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_lr.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_lr.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                    {'params': self.model_4dvarnet_hr.phi_r.parameters(), 'lr': 0.5 * self.hparams.lr_update[0]}
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_4dvarnet_lr.n_grad = self.hparams.n_grad
        self.model_4dvarnet_hr.n_grad = self.hparams.n_grad
        #self.compute_derivativeswith_lon_lat.to(device)
                
    def on_train_epoch_start(self):
        if self.model_name in ('4dvarnet', '4dvarnet_sst'):
            opt = self.optimizers()
            if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
                indx = self.hparams.iter_update.index(self.current_epoch)
                print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                    self.current_epoch, self.hparams.nb_grad_update[indx], self.hparams.lr_update[indx]))

                self.model_4dvarnet_lr.n_grad = self.hparams.n_grad
                self.model_4dvarnet_hr.n_grad = self.hparams.n_grad

                mm = 0
                lrCurrent = self.hparams.lr_update[indx]
                lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
                for pg in opt.param_groups:
                    pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                    mm += 1
        #self.patch_weight_lr = self.patch_weight_train 
        #self.patch_weight = self.patch_weight_train 

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

        out_pred = out[0]        
        out_u = out[1]
        out_v = out[2]
            
            
        if not self.use_sst :

            return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'u_gt'    : (u_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'v_gt'    : (v_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred' : (out_pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred_u' : (out_u.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'pred_v' : (out_v.detach().cpu() * np.sqrt(self.var_tr_uv)) }
        else:
            

            return {'gt'    : (targets_GT.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'oi'    : (targets_OI.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'u_gt'    : (u_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'v_gt'    : (v_gt.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'obs_inp'    : (inputs_obs.detach().where(inputs_Mask, torch.full_like(inputs_obs, np.nan)).cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred' : (out_pred.detach().cpu() * np.sqrt(self.var_Tr)) + self.mean_Tr,
                    'pred_u' : (out_u.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
                    'pred_v' : (out_v.detach().cpu() * np.sqrt(self.var_tr_uv)) ,
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
            print("\n.... Set weight matrix to binary mask for final outputs")
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
        ## compute div/curl/strain metrics
        def compute_var_exp(x,y,dw=0):
            if dw == 0 :
                mse = np.nanmean( (x-y)**2 )
                var = np.nanvar( x )
            else:
                _x  = x[:,dw:x.shape[1]-dw,dw:x.shape[2]-dw]
                _y  = y[:,dw:x.shape[1]-dw,dw:x.shape[2]-dw]
                mse = np.nanmean( (_x-_y)**2 )
                var = np.nanvar( _x )
                
            return 100. * ( 1. - mse / var )

        def compute_metrics_SSC(u_gt,v_gt,u,v,dw = 2):
            
            if dw == 0 :
                mse_uv = np.nanmean((u_gt - u) ** 2 + (v_gt - v) ** 2 )
                var_uv = np.nanmean((u_gt) ** 2 + (v_gt) ** 2 )
            else:
                _u    = u[:,dw:u.shape[1]-dw,dw:u.shape[2]-dw]
                _v    = v[:,dw:u.shape[1]-dw,dw:u.shape[2]-dw]
                _u_gt = u_gt[:,dw:u.shape[1]-dw,dw:u.shape[2]-dw]
                _v_gt = v_gt[:,dw:u.shape[1]-dw,dw:u.shape[2]-dw]
                
                mse_uv = np.nanmean((_u_gt - _u) ** 2 + (_v_gt - _v) ** 2 )
                var_uv = np.nanmean((_u_gt) ** 2 + (_v_gt) ** 2 )
            var_mse_uv = 100. * ( 1. - mse_uv / var_uv )
        
            psd_ds_u, lamb_x_u, lamb_t_u = metrics.psd_based_scores(u, u_gt)
            psd_ds_v, lamb_x_v, lamb_t_v = metrics.psd_based_scores(v, v_gt)
            
            return var_mse_uv, lamb_x_u, lamb_t_u, lamb_x_v, lamb_t_v

        # Metrics for SSC fields
        alpha_uv_geo = 9.81 
        lat_rad = np.radians(self.test_lat)
        lon_rad = np.radians(self.test_lon)
        
        if 1*0 :    
            lat_rad = lat_rad.reshape((1,1,lat_rad.hape[0],lat_rad.shape[1]))
            lon_rad = lon_rad.reshape((1,1,lon_rad.hape[0],lon_rad.shape[1]))
            lat_rad = lat_rad.repeat()
            
            
            t_compute_geo_velocities =  Torch_compute_derivatives_with_lon_lat()
            t_ssh = torch.Tensor(self.test_xr_ds.gt.data)#.view(-1,1,self.test_xr_ds.pred_u.shape[1],self.test_xr_ds.pred_u.shape[2])        
            t_ssh = t_ssh.view(-1,1,t_ssh.size(1),t_ssh.size(2))        
            t_lat_rad = torch.Tensor( lat_rad )
            t_lon_rad = torch.Tensor( lon_rad )
            
            t_u_geo_gt,t_v_geo_gt = t_compute_geo_velocities.compute_geo_velociites(t_ssh, t_lat_rad, t_lon_rad, sigma=0.)
            u_geo_gt = t_u_geo_gt.numpy().squeeze()
            v_geo_gt = t_v_geo_gt.numpy().squeeze()
        
        u_geo_gt,v_geo_gt = compute_uv_geo_with_coriolis(self.test_xr_ds.gt,lat_rad,lon_rad,alpha_uv_geo = alpha_uv_geo,sigma=0.)
        u_geo_oi,v_geo_oi = compute_uv_geo_with_coriolis(self.test_xr_ds.oi,lat_rad,lon_rad,alpha_uv_geo = alpha_uv_geo,sigma=0.)
        u_geo_rec,v_geo_rec = compute_uv_geo_with_coriolis(self.test_xr_ds.pred,lat_rad,lon_rad,alpha_uv_geo = alpha_uv_geo,sigma=0.)

        dw_diag = 3
        print('\n\n...... SSH-derived SSC metrics for true SSH')
        var_mse_uv_ssh_gt, lamb_x_u_ssh_gt, lamb_t_u_ssh_gt, lamb_x_v_ssh_gt, lamb_t_v_ssh_gt = compute_metrics_SSC( self.test_xr_ds.u_gt , self.test_xr_ds.v_gt , u_geo_gt , v_geo_gt  , dw = dw_diag)
        print('\n\n...... SSH-derived SSC metrics for DUACS SSH')
        var_mse_uv_ssh_oi, lamb_x_u_ssh_oi, lamb_t_u_ssh_oi, lamb_x_v_ssh_oi, lamb_t_v_ssh_oi = compute_metrics_SSC( self.test_xr_ds.u_gt , self.test_xr_ds.v_gt , u_geo_oi , v_geo_oi  , dw = dw_diag)
        print('\n\n...... SSH-derived SSC metrics for 4dVarNet SSH')
        var_mse_uv_ssh_rec, lamb_x_u_ssh_rec, lamb_t_u_ssh_rec, lamb_x_v_ssh_rec, lamb_t_v_ssh_rec = compute_metrics_SSC( self.test_xr_ds.u_gt , self.test_xr_ds.v_gt , u_geo_rec , v_geo_rec  , dw = dw_diag)
        print('\n\n...... SSH-derived SSC metrics for 4dVarNet SSC')
        var_mse_uv, lamb_x_u, lamb_t_u, lamb_x_v, lamb_t_v = compute_metrics_SSC( self.test_xr_ds.u_gt , self.test_xr_ds.v_gt , self.test_xr_ds.pred_u, self.test_xr_ds.pred_v  , dw = dw_diag)

        print('.....')
        print('.....')
        print('..... Computation of div/curl/strain metrics  ')
        sig_div_curl = self.sig_filter_div_diag

        flag_heat_equation = False
        
        if flag_heat_equation :
            iter_heat = 5
            lam = self.sig_filter_div_diag
            t_compute_div_curl_strain_with_lat_lon =  Torch_compute_derivatives_with_lon_lat()
            def heat_equation(u,iter,lam):
                t_u = torch.Tensor(u).view(-1,1,u.shape[1],u.shape[2]) 
                f_u = t_compute_div_curl_strain_with_lat_lon.heat_equation(t_u,iter=iter,lam=lam)
    
                return f_u.detach().numpy().squeeze()
            
            f_u = heat_equation(self.test_xr_ds.u_gt.data,iter=iter_heat,lam=lam)
            f_v = heat_equation(self.test_xr_ds.v_gt.data,iter=iter_heat,lam=lam)
            div_gt,curl_gt,strain_gt = compute_div_curl_strain_with_lat_lon(f_u,f_v,lat_rad,lon_rad,sigma=0.)        
        
            f_u = heat_equation(self.test_xr_ds.pred_u.data,iter=iter_heat,lam=lam)
            f_v = heat_equation(self.test_xr_ds.pred_v.data,iter=iter_heat,lam=lam)
            div_uv_rec,curl_uv_rec,strain_uv_rec = compute_div_curl_strain_with_lat_lon(f_u,f_v,lat_rad,lon_rad,sigma=0.)
        else :
            div_gt,curl_gt,strain_gt = compute_div_curl_strain_with_lat_lon(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,lat_rad,lon_rad,sigma=sig_div_curl)
            div_uv_rec,curl_uv_rec,strain_uv_rec = compute_div_curl_strain_with_lat_lon(self.test_xr_ds.pred_u,self.test_xr_ds.pred_v,lat_rad,lon_rad,sigma=sig_div_curl)
 
        var_mse_div = compute_var_exp( div_gt, div_uv_rec , dw = dw_diag)
        var_mse_curl = compute_var_exp( curl_gt, curl_uv_rec, dw = dw_diag)
        var_mse_strain = compute_var_exp( strain_gt, strain_uv_rec, dw = dw_diag)
        
        if 1*0 :
            t_u = torch.Tensor(self.test_xr_ds.pred_u.data)#.view(-1,1,self.test_xr_ds.pred_u.shape[1],self.test_xr_ds.pred_u.shape[2])
            t_v = torch.Tensor(self.test_xr_ds.pred_v.data)#.view(-1,1,self.test_xr_ds.pred_u.shape[1],self.test_xr_ds.pred_u.shape[2])
            
            t_u = t_u.view(-1,1,t_u.size(1),t_u.size(2))
            t_v = t_v.view(-1,1,t_v.size(1),t_v.size(2))
            
            t_lat_rad = torch.Tensor( lat_rad )
            t_lon_rad = torch.Tensor( lon_rad )
    
            t_compute_div_curl_strain_with_lat_lon =  Torch_compute_derivatives_with_lon_lat()
            t_div,t_curl,t_strain = t_compute_div_curl_strain_with_lat_lon.compute_div_curl_strain(t_u,t_v,t_lat_rad,t_lon_rad,sigma=sig_div_curl)
            
            div_uv_rec_ = t_div.numpy().squeeze()
            curl_uv_rec_ = t_curl.numpy().squeeze()
            strain_uv_rec_ = t_strain.numpy().squeeze()
                
            t_u = torch.Tensor(self.test_xr_ds.u_gt.data)#.view(-1,1,self.test_xr_ds.pred_u.shape[1],self.test_xr_ds.pred_u.shape[2])
            t_v = torch.Tensor(self.test_xr_ds.v_gt.data)#.view(-1,1,self.test_xr_ds.pred_u.shape[1],self.test_xr_ds.pred_u.shape[2])            
            t_u = t_u.view(-1,1,t_u.size(1),t_u.size(2))
            t_v = t_v.view(-1,1,t_v.size(1),t_v.size(2))

            t_compute_div_curl_strain_with_lat_lon =  Torch_compute_derivatives_with_lon_lat()
            t_div,t_curl,t_strain = t_compute_div_curl_strain_with_lat_lon.compute_div_curl_strain(t_u,t_v,t_lat_rad,t_lon_rad,sigma=sig_div_curl)
            
            div_gt_ = t_div.numpy().squeeze()
            curl_gt_ = t_curl.numpy().squeeze()
            strain_gt_ = t_strain.numpy().squeeze()

            var_mse_div_ = compute_var_exp( div_gt_, div_uv_rec_, dw = dw_diag)
            var_mse_curl_ = compute_var_exp( curl_gt_, curl_uv_rec_, dw = dw_diag)
            var_mse_strain_ = compute_var_exp( strain_gt_, strain_uv_rec_, dw = dw_diag)
    
            print('.... div %.2f -- %.2f -- %.2f'%(var_mse_div_,var_mse_div,compute_var_exp( div_uv_rec, div_uv_rec_)) )
            print('.... strain %.2f -- %.2f -- %.2f'%(var_mse_strain_,var_mse_strain,compute_var_exp( strain_uv_rec, strain_uv_rec_)))
            print('.... curl %.2f -- %.2f -- %.2f'%(var_mse_curl_,var_mse_curl,compute_var_exp( curl_uv_rec, curl_uv_rec_)))

        if sig_div_curl > 0. :
            f_ssh_gt = gaussian_filter(self.test_xr_ds.gt, sigma=sig_div_curl)
            f_ssh_oi = gaussian_filter(self.test_xr_ds.oi, sigma=4.*sig_div_curl)
            f_ssh_rec = gaussian_filter(self.test_xr_ds.pred, sigma=sig_div_curl)
            
        else:
            f_ssh_gt = self.test_xr_ds.gt
            f_ssh_oi = self.test_xr_ds.oi
            f_ssh_rec = self.test_xr_ds.pred
            
        f_u_geo_gt,f_v_geo_gt = compute_uv_geo_with_coriolis(f_ssh_gt,lat_rad,lon_rad,alpha_uv_geo = alpha_uv_geo,sigma=0.)
        f_u_geo_oi,f_v_geo_oi = compute_uv_geo_with_coriolis(f_ssh_oi,lat_rad,lon_rad,alpha_uv_geo = alpha_uv_geo,sigma=0.)
        f_u_geo_rec,f_v_geo_rec = compute_uv_geo_with_coriolis(f_ssh_rec,lat_rad,lon_rad,alpha_uv_geo = alpha_uv_geo,sigma=0.)

        div_geo_gt,curl_geo_gt,strain_geo_gt = compute_div_curl_strain_with_lat_lon(f_u_geo_gt,f_v_geo_gt,lat_rad,lon_rad,sigma=0.)
        div_geo_oi,curl_geo_oi,strain_geo_oi = compute_div_curl_strain_with_lat_lon(f_u_geo_oi,f_v_geo_oi,lat_rad,lon_rad,sigma=0.)
        div_geo_rec,curl_geo_rec,strain_geo_rec = compute_div_curl_strain_with_lat_lon(f_u_geo_rec,f_v_geo_rec,lat_rad,lon_rad,sigma=0.)

        var_mse_div_ssh_gt = compute_var_exp( div_gt, div_geo_gt , dw = dw_diag)
        var_mse_curl_ssh_gt = compute_var_exp( curl_gt, curl_geo_gt , dw = dw_diag)
        var_mse_strain_ssh_gt = compute_var_exp( strain_gt, strain_geo_gt , dw = dw_diag)

        var_mse_div_ssh_oi = compute_var_exp( div_gt, div_geo_oi , dw = dw_diag)
        var_mse_curl_ssh_oi = compute_var_exp( curl_gt, curl_geo_oi , dw = dw_diag)
        var_mse_strain_ssh_oi = compute_var_exp( strain_gt, strain_geo_oi , dw = dw_diag)

        var_mse_div_ssh_rec = compute_var_exp( div_gt, div_geo_rec , dw = dw_diag)
        var_mse_curl_ssh_rec = compute_var_exp( curl_gt, curl_geo_rec , dw = dw_diag)
        var_mse_strain_ssh_rec = compute_var_exp( strain_gt, strain_geo_rec , dw = dw_diag)

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

        #print(self.test_xr_ds.gt.data.shape,flush=True)

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
        
        #alpha_uv_geo = 9.81 
        #lat_rad = np.radians(self.test_lat)
        #lon_rad = np.radians(self.test_lon)
        
        #div_gt,curl_gt,strain_gt = compute_div_curl_strain_with_lat_lon(self.test_xr_ds.u_gt,self.test_xr_ds.v_gt,lat_rad,lon_rad,sigma=self.sig_filter_div_diag)
        #div_uv_rec,curl_uv_rec,strain_uv_rec = compute_div_curl_strain_with_lat_lon(self.test_xr_ds.pred_u,self.test_xr_ds.pred_v,lat_rad,lon_rad,sigma=self.sig_filter_div_diag)

        self.latest_metrics.update(md)
        self.logger.log_metrics(md, step=self.current_epoch)

        if self.scale_dwscaling_sst > 1. :
            print('.... Using downscaled SST by %.1f'%self.scale_dwscaling_sst)
        print('..... Log directory: '+self.logger.log_dir)
        
        if self.save_rec_netcdf == True :
            #path_save1 = self.logger.log_dir + f'/test_res_all.nc'
            path_save1 = self.hparams.path_save_netcdf.replace('.ckpt','_res_4dvarnet_all.nc')
            if True : #not self.use_sst :
                print('... Save nc file with all results : '+path_save1)
                #print(self.test_dates)
                save_netcdf_uv(saved_path1=path_save1, 
                                        gt=self.x_gt, obs = self.obs_inp , oi= self.x_oi, pred=self.x_rec_ssh, 
                                        u_gt=self.u_gt, v_gt=self.v_gt, 
                                        u_pred=self.u_rec, v_pred=self.v_rec,
                                        #curl_gt=curl_gt,strain_gt=strain_gt,
                                        #curl_pred=curl_uv_rec,strain_pred=strain_uv_rec,
                                        sst_feat=self.x_sst_feat_ssh[:,0,:,:].reshape(self.x_sst_feat_ssh.shape[0],1,self.x_sst_feat_ssh.shape[2],self.x_sst_feat_ssh.shape[3]),
                                        
                                        
                                        lon=self.test_lon, lat=self.test_lat, time=self.test_dates)#, time_units=None)
            else:
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
                        
                if 1*0:
                    self.x_gt = self.x_gt[2:42,:,:]
                    self.obs_inp = self.obs_inp[2:42,:,:]
                    self.x_oi = self.x_oi[2:42,:,:]
                    self.x_rec = self.x_rec[2:42,:,:]
                    self.x_rec_ssh = self.x_rec[2:42,:,:]

                print('... Save nc file with all results : '+path_save1)
                print( self.x_rec.shape )
                print( self.v_rec.shape )
                print( self.x_sst_feat_ssh.shape )
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

    def get_init_state_lr_from_hr(self, batch,out_hr=(None,),out_lr=(None,),mask_sampling = None):
        targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon, gx, gy = batch
        
        if out_hr[0] is not None:            
            # compute low-resolution lr state from hr state            
            init_ssh_ = torch.nn.functional.avg_pool2d(out_hr[0].detach(), (int(self.scale_lr),int(self.scale_lr)))
            init_u_ = torch.nn.functional.avg_pool2d(out_hr[1].detach(), (int(self.scale_lr),int(self.scale_lr)))
            init_v_ = torch.nn.functional.avg_pool2d(out_hr[2].detach(), (int(self.scale_lr),int(self.scale_lr)))
            
            init_ssh = out_lr[0]
            init_u = out_lr[1]
            init_v = out_lr[2]

            _dt = int( (self.hparams.dT-self.hparams.dT_hr_model)/2 )
            init_ssh[:,_dt:_dt+self.hparams.dT_hr_model,:,:] = init_ssh_
            init_u[:,_dt:_dt+self.hparams.dT_hr_model,:,:] = init_u_
            init_v[:,_dt:_dt+self.hparams.dT_hr_model,:,:] = init_v_   
            
            init_ssh = init_ssh.detach()
            init_u = init_u.detach()
            init_v = init_v.detach()
            
            print(' init state after one step ')
            print(init_ssh.size())
            print(init_u.size())
            print(init_v.size(),flush=True)
        else:              
            init_u = torch.zeros_like(targets_GT)
            init_v = torch.zeros_like(targets_GT)
            init_ssh = inputs_Mask * inputs_obs
        
        if mask_sampling is not None :
            init_u += mask_sampling * u_gt
            init_v += mask_sampling * v_gt

        if self.aug_state :
            init_state = torch.cat((init_ssh ,
                                    init_ssh ,
                                    init_u,init_v),
                                    dim=1)
        else:
            init_state = torch.cat((init_ssh ,
                                    init_u,init_v),
                                    dim=1)
        if self.use_sst_state :
            init_state = torch.cat((init_state,
                                    sst_gt,),
                                    dim=1)
        return init_state

    def get_init_state_hr_from_lr(self, batch, out_lr=(None,),state_hr=(None,),mask_sampling = None):
            
        if out_lr[0] is  None :
            print('...  THIS IS UNEXPECTED init state from hr with no hr state .....')
                        
        # re-interpolate lr state to hr grid 
        # for hr time window
        init_ssh_from_lr = torch.nn.functional.interpolate(out_lr[0].detach(), scale_factor=self.scale_lr, mode='bicubic')
        init_u_from_lr = torch.nn.functional.interpolate(out_lr[1].detach(), scale_factor=self.scale_lr, mode='bicubic')
        init_v_from_lr = torch.nn.functional.interpolate(out_lr[2].detach(), scale_factor=self.scale_lr, mode='bicubic')
                    
        _dt = int( (self.hparams.dT-self.hparams.dT_hr_model)/2 )
        init_ssh_from_lr = init_ssh_from_lr[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        init_u_from_lr = init_u_from_lr[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        init_v_from_lr = init_v_from_lr[:,_dt:_dt+self.hparams.dT_hr_model,:,:]

        
        targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon, gx, gy = batch

        if state_hr[0] is not None: 
            if self.aug_state :
                init_state = torch.cat((init_ssh_from_lr,
                                        state_hr[0][:,self.dT:2*self.dT,:,:],
                                        state_hr[0][:,2*self.dT:3*self.dT,:,:],
                                        init_u_from_lr,init_v_from_lr),
                                       dim=1)
            else:      
                init_state = torch.cat((init_ssh_from_lr,
                                        state_hr[0][:,self.dT:2*self.dT,:,:],
                                        init_u_from_lr,init_v_from_lr),
                                       dim=1)
            
            if self.use_sst_state :
                init_state = torch.cat((init_state,
                                        sst_gt,),
                                       dim=1)
            return init_state
            
        if self.aug_state :
            init_state = torch.cat((init_ssh_from_lr,
                                    inputs_Mask * (inputs_obs - init_ssh_from_lr),
                                    inputs_Mask * (inputs_obs - init_ssh_from_lr),
                                    init_u_from_lr,init_v_from_lr),
                                   dim=1)
        else:
            init_state = torch.cat((init_ssh_from_lr,
                                    inputs_Mask * (inputs_obs - targets_OI),
                                    init_u_from_lr,init_v_from_lr),
                                   dim=1)
 
        if self.use_sst_state :
            init_state = torch.cat((init_state,
                                    sst_gt,),
                                   dim=1)
        return init_state

    def loss_ae_lr(self, state_out):
        return torch.mean((self.model_4dvarnet_lr.phi_r(state_out) - state_out) ** 2)

    def loss_ae_hr(self, state_out):
        return torch.mean((self.model_4dvarnet_hr.phi_r(state_out) - state_out) ** 2)
    
    def sla_loss(self, gt, out):
        g_outputs_x, g_outputs_y = self.gradient_img(out)
        g_gt_x, g_gt_y = self.gradient_img(gt)

        loss = NN_4DVar.compute_spatio_temp_weighted_loss((out - gt), self.patch_weight)
        loss_grad = (
                NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_x - g_gt_x, self.grad_crop(self.patch_weight))
            +    NN_4DVar.compute_spatio_temp_weighted_loss(g_outputs_y - g_gt_y, self.grad_crop(self.patch_weight))
        )

        return loss, loss_grad
    
    def compute_uv_from_ssh(self,ssh, lat_rad, lon_rad,sigma=0.,flag_lr=False):
        ssh = np.sqrt(self.var_Tr) * ssh + self.mean_Tr
        if flag_lr == True :
            u_geo, v_geo = self.compute_derivativeswith_lon_lat_lr.compute_geo_velocities(ssh, lat_rad, lon_rad,sigma=0.)
        else:
            u_geo, v_geo = self.compute_derivativeswith_lon_lat_hr.compute_geo_velocities(ssh, lat_rad, lon_rad,sigma=0.)

        return u_geo / np.sqrt(self.var_tr_uv) , v_geo / np.sqrt(self.var_tr_uv)

    def compute_geo_factor(self,outputs, lat_rad, lon_rad,sigma=0.,flag_lr=False):
         if flag_lr == True :
            return self.compute_derivativeswith_lon_lat_lr.compute_geo_factor(outputs, lat_rad, lon_rad,sigma=0.)
         else:
            return self.compute_derivativeswith_lon_lat_hr.compute_geo_factor(outputs, lat_rad, lon_rad,sigma=0.)

    def compute_div_curl_strain(self,u,v,lat_rad, lon_rad , sigma =0.,flag_lr=False):
        if flag_lr == True :
            if sigma > 0:
                u = self.compute_derivativeswith_lon_lat_lr.heat_equation_all_channels(u,iter=5,lam=self.sig_filter_div_diag)
                v = self.compute_derivativeswith_lon_lat_lr.heat_equation_all_channels(v,iter=5,lam=self.sig_filter_div_diag)
            
            div_gt,curl_gt,strain_gt    = self.compute_derivativeswith_lon_lat_lr.compute_div_curl_strain(u, v, lat_rad, lon_rad , sigma = self.sig_filter_div_diag )
        else:
            if sigma > 0:
                u = self.compute_derivativeswith_lon_lat_hr.heat_equation_all_channels(u,iter=5,lam=self.sig_filter_div_diag)
                v = self.compute_derivativeswith_lon_lat_hr.heat_equation_all_channels(v,iter=5,lam=self.sig_filter_div_diag)
            
            div_gt,curl_gt,strain_gt    = self.compute_derivativeswith_lon_lat_hr.compute_div_curl_strain(u, v, lat_rad, lon_rad , sigma = self.sig_filter_div_diag )
    
        return div_gt,curl_gt,strain_gt 
    
    def div_loss(self, gt, out):
        if self.type_div_train_loss == 0 :
            return NN_4DVar.compute_spatio_temp_weighted_loss( (out - gt), self.patch_weight[:,1:-1,1:-1])
        else:
            return NN_4DVar.compute_spatio_temp_weighted_loss( 1.e4 * (out - gt ), self.patch_weight)
        
    def strain_loss(self, gt, out):
        return NN_4DVar.compute_spatio_temp_weighted_loss(1.e4 * (out - gt ), self.patch_weight)        
 
    def uv_loss(self, gt, out):
        loss = NN_4DVar.compute_spatio_temp_weighted_loss((out[0] - gt[0]), self.patch_weight)
        loss = loss + NN_4DVar.compute_spatio_temp_weighted_loss((out[1] - gt[1]), self.patch_weight)

        return loss

    def reg_loss_lr(self, y_gt, oi, out, out_lrhr):
        l_ae = self.loss_ae_lr(out_lrhr)
        l_ae_gt = self.loss_ae_lr(y_gt)
                
        return l_ae, l_ae_gt

    def reg_loss_hr(self, y_gt, out_lrhr):
        l_ae = self.loss_ae_hr(out_lrhr)
        l_ae_gt = self.loss_ae_hr(y_gt)

        return l_ae, l_ae_gt


    def dwn_sample_batch(self,batch,scale = 1. ):
        if scale > 1. :          
            if not self.use_sst:
                targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt, lat, lon = batch
            else:
                targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = batch
    
                targets_OI = torch.nn.functional.avg_pool2d(targets_OI, (int(self.scale_lr),int(self.scale_lr)))
                targets_GT = torch.nn.functional.avg_pool2d(targets_GT, (int(self.scale_lr),int(self.scale_lr)))
                u_gt = torch.nn.functional.avg_pool2d(u_gt, (int(self.scale_lr),int(self.scale_lr)))
                v_gt = torch.nn.functional.avg_pool2d(v_gt, (int(self.scale_lr),int(self.scale_lr)))
                if self.use_sst:
                    sst_gt = torch.nn.functional.avg_pool2d(sst_gt, (int(self.scale_lr),int(self.scale_lr)))
                
                targets_GT = targets_GT.detach()
                sst_gt = sst_gt.detach()
                u_gt = u_gt.detach()
                v_gt = v_gt.detach()
                
                
                inputs_Mask = inputs_Mask.detach()
                inputs_obs = inputs_obs.detach()
                
                inputs_Mask = torch.nn.functional.avg_pool2d(inputs_Mask.float(), (int(self.scale_lr),int(self.scale_lr)))
                inputs_obs  = torch.nn.functional.avg_pool2d(inputs_obs, (int(self.scale_lr),int(self.scale_lr)))
                
                inputs_obs  = inputs_obs / ( inputs_Mask + 1e-7 )
                inputs_Mask = (inputs_Mask > 0.).float()   
                
                lat = torch.nn.functional.avg_pool1d(lat.view(-1,1,lat.size(1)), (int(self.scale_lr)))
                lon = torch.nn.functional.avg_pool1d(lon.view(-1,1,lon.size(1)), (int(self.scale_lr)))
                
                lat = lat.view(-1,lat.size(2))
                lon = lon.view(-1,lon.size(2))
            
            
            if not self.use_sst:
                return targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt, lat, lon
            else:
                return targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon
        else:
            return batch


    def pre_process_batch_for_lr(self,batch):
        if self.scale_lr > 1.0 :
            _batch = self.dwn_sample_batch(batch,scale=self.scale_lr)
        else:
            _batch = batch
            
        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt, lat, lon = _batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = _batch

        if self.scale_dwscaling_sst > 1 :
            sst_gt = torch.nn.functional.avg_pool2d(sst_gt, (int(self.scale_dwscaling_sst),int(self.scale_dwscaling_sst)))
            sst_gt = torch.nn.functional.interpolate(sst_gt, scale_factor=self.scale_dwscaling_sst, mode='bicubic')
            
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        u_gt_wo_nan = u_gt.where(~u_gt.isnan(), torch.zeros_like(u_gt) )
        v_gt_wo_nan = v_gt.where(~v_gt.isnan(), torch.zeros_like(u_gt) )
        
        if not self.use_sst:
            sst_gt = None
            
        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT_wo_nan)

        # lat/lon in radians
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
            
        if self.use_lat_lon_in_obs_model  == True :
            self.model_4dvarnet_lr.model_H.lat_rad = lat_rad
            self.model_4dvarnet_lr.model_H.lon_rad = lon_rad

        return targets_OI, inputs_Mask, inputs_obs, targets_GT_wo_nan, sst_gt, u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad, g_targets_GT_x, g_targets_GT_y
    
    def get_obs_and_mask_lr(self,inputs_Mask,inputs_obs,sst_gt,u_gt_wo_nan,v_gt_wo_nan):
                
        if self.model_sampling_uv is not None :
            w_sampling_uv = self.model_sampling_uv( sst_gt )
            w_sampling_uv = w_sampling_uv[1]
            
            #mask_sampling_uv = torch.bernoulli( w_sampling_uv )
            mask_sampling_uv = 1. - torch.nn.functional.threshold( 1.0 - w_sampling_uv , 0.9 , 0.)
            obs = torch.cat( (inputs_Mask * inputs_obs , mask_sampling_uv * u_gt_wo_nan , mask_sampling_uv * v_gt_wo_nan ) ,dim=1)
            
        else:
            mask_sampling_uv = torch.zeros_like(inputs_obs)
            w_sampling_uv = None
            obs = torch.cat( (inputs_Mask * inputs_obs, torch.zeros_like(inputs_Mask) ,  torch.zeros_like(inputs_Mask) ) ,dim=1)
            
        new_masks = torch.cat( ( inputs_Mask, mask_sampling_uv, mask_sampling_uv) , dim=1)

        if self.aug_state :
            obs = torch.cat( (obs, torch.zeros_like(inputs_Mask),) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.zeros_like(inputs_Mask)), dim=1)
        
        if self.use_sst_state :
            obs = torch.cat( (obs,sst_gt,) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.ones_like(inputs_Mask)), dim=1)

        if self.use_sst_obs :
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]
        
        return obs,new_masks,w_sampling_uv,mask_sampling_uv

    def get_obs_and_mask_hr(self,init_ssh_lr,inputs_Mask,inputs_obs,sst_gt,u_gt_wo_nan,v_gt_wo_nan):
                
        _dt = int( (self.hparams.dT-self.hparams.dT_hr_model)/2 )
        init_ssh_from_lr = torch.nn.functional.interpolate(init_ssh_lr, scale_factor=self.scale_lr, mode='bicubic')
        init_ssh_from_lr = init_ssh_from_lr[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        
        if self.model_sampling_uv is not None :
            w_sampling_uv = self.model_sampling_uv( sst_gt )
            w_sampling_uv = w_sampling_uv[1]
            
            #mask_sampling_uv = torch.bernoulli( w_sampling_uv )
            mask_sampling_uv = 1. - torch.nn.functional.threshold( 1.0 - w_sampling_uv , 0.9 , 0.)
            obs = torch.cat( (init_ssh_from_lr , inputs_Mask * ( inputs_obs - init_ssh_from_lr ) , u_gt_wo_nan , v_gt_wo_nan ) ,dim=1)
            
            #print('%f '%( float( self.hparams.dT / (self.hparams.dT - int(self.hparams.dT/2))) * torch.mean(w_sampling_uv)) )
        else:
            mask_sampling_uv = torch.zeros_like(u_gt_wo_nan)
            w_sampling_uv = None
            obs = torch.cat( ( init_ssh_from_lr , inputs_Mask * ( inputs_obs - init_ssh_from_lr ) , torch.zeros_like(inputs_Mask) ,  torch.zeros_like(inputs_Mask) ) ,dim=1)
            
        new_masks = torch.cat( ( torch.ones_like(inputs_Mask) , inputs_Mask, mask_sampling_uv, mask_sampling_uv) , dim=1)

        if self.aug_state :
            obs = torch.cat( (obs, torch.zeros_like(inputs_Mask),) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.zeros_like(inputs_Mask)), dim=1)
        
        if self.use_sst_state :
            obs = torch.cat( (obs,sst_gt,) ,dim=1)
            new_masks = torch.cat( (new_masks, torch.ones_like(inputs_Mask)), dim=1)

        if self.use_sst_obs :
            new_masks = [ new_masks, torch.ones_like(sst_gt) ]
            obs = [ obs, sst_gt ]
        
        return obs,new_masks,w_sampling_uv,mask_sampling_uv

    def run_model_lr(self,state, obs, new_masks,state_init,lat_rad,lon_rad,phase):
        state = torch.autograd.Variable(state, requires_grad=True)

        outputs, hidden_new, cell_new, normgrad = self.model_4dvarnet_lr(state, obs, new_masks, *state_init[1:])
 
        if (phase == 'val') or (phase == 'test'):
            outputs = outputs.detach()

        outputsSLRHR = outputs
        outputs = outputs[:, 0:self.hparams.dT, :, :]
        
        if self.aug_state :
            outputs_u = outputsSLRHR[:, 2*self.hparams.dT:3*self.hparams.dT, :, :]
            outputs_v = outputsSLRHR[:, 3*self.hparams.dT:4*self.hparams.dT, :, :]
        else:
            outputs_u = outputsSLRHR[:, 1*self.hparams.dT:2*self.hparams.dT, :, :]
            outputs_v = outputsSLRHR[:, 2*self.hparams.dT:3*self.hparams.dT, :, :]

        # U,V prediction
        if ( self.residual_wrt_geo_velocities == 1 ) or (self.residual_wrt_geo_velocities == 3):            
            # denormalize ssh
            u_geo_rec , v_geo_rec = self.compute_uv_from_ssh(outputs, lat_rad, lon_rad,sigma=0.) 

            alpha_uv_geo = 0.05
            outputs_u = alpha_uv_geo * outputs_u + u_geo_rec
            outputs_v = alpha_uv_geo * outputs_v + v_geo_rec
            
        elif ( self.residual_wrt_geo_velocities == 2 ) or ( self.residual_wrt_geo_velocities == 4 ): 
            # denormalize ssh
            u_geo_rec, v_geo_rec = self.compute_uv_from_ssh(outputs, lat_rad, lon_rad,sigma=0.) 
            #u_geo_gt, v_geo_gt = self.compute_uv_from_ssh(targets_GT_wo_nan, lat_rad, lon_rad,sigma=0.) 

            u_geo_factor, v_geo_factor = self.compute_geo_factor(outputs, lat_rad, lon_rad,sigma=0.) 

            alpha_uv_geo = 0.05
            outputs_u = alpha_uv_geo * u_geo_factor * outputs_u
            outputs_v = alpha_uv_geo * v_geo_factor * outputs_v

        
        return outputs, outputs_u, outputs_v, outputsSLRHR, hidden_new, cell_new, normgrad

    def run_model_hr(self,state, obs, new_masks,state_init,lat_rad,lon_rad,phase):
        state = torch.autograd.Variable(state, requires_grad=True)

        outputs, hidden_new, cell_new, normgrad = self.model_4dvarnet_hr(state, obs, new_masks, *state_init[1:])

        if (phase == 'val') or (phase == 'test'):
            outputs = outputs.detach()

        outputsSLRHR = outputs
        outputsSLR = outputs[:, 0:self.hparams.dT_hr_model, :, :]
        if self.aug_state :
            outputs = outputsSLR + outputsSLRHR[:, 2*self.hparams.dT_hr_model:3*self.hparams.dT_hr_model, :, :]
            outputs_u = outputsSLRHR[:, 3*self.hparams.dT_hr_model:4*self.hparams.dT_hr_model, :, :]
            outputs_v = outputsSLRHR[:, 4*self.hparams.dT_hr_model:5*self.hparams.dT_hr_model, :, :]
        else:
            outputs = outputsSLR + outputsSLRHR[:, self.hparams.dT_hr_model:2*self.hparams.dT_hr_model, :, :]
            outputs_u = outputsSLRHR[:, 2*self.hparams.dT_hr_model:3*self.hparams.dT_hr_model, :, :]
            outputs_v = outputsSLRHR[:, 3*self.hparams.dT_hr_model:4*self.hparams.dT_hr_model, :, :]

        # U,V prediction
        if ( self.residual_wrt_geo_velocities == 1 ) or (self.residual_wrt_geo_velocities == 3):            
            # denormalize ssh
            u_geo_rec , v_geo_rec = self.compute_uv_from_ssh(outputs, lat_rad, lon_rad,sigma=0.) 

            alpha_uv_geo = 0.05
            outputs_u = alpha_uv_geo * outputs_u + u_geo_rec
            outputs_v = alpha_uv_geo * outputs_v + v_geo_rec
            
        elif ( self.residual_wrt_geo_velocities == 2 ) or ( self.residual_wrt_geo_velocities == 4 ): 
            # denormalize ssh
            u_geo_rec, v_geo_rec = self.compute_uv_from_ssh(outputs, lat_rad, lon_rad,sigma=0.) 
            #u_geo_gt, v_geo_gt = self.compute_uv_from_ssh(targets_GT_wo_nan, lat_rad, lon_rad,sigma=0.) 

            u_geo_factor, v_geo_factor = self.compute_geo_factor(outputs, lat_rad, lon_rad,sigma=0.) 

            alpha_uv_geo = 0.05
            outputs_u = alpha_uv_geo * u_geo_factor * outputs_u
            outputs_v = alpha_uv_geo * v_geo_factor * outputs_v

        
        return outputs, outputs_u, outputs_v, outputsSLRHR, outputsSLR, hidden_new, cell_new, normgrad

    def compute_reg_loss_lr(self,targets_OI,targets_GT_wo_nan, u_gt_wo_nan, sst_gt, v_gt_wo_nan,outputs, outputsSLRHR,phase):
        
        if outputsSLRHR is not None :

            if self.aug_state :
                yGT = torch.cat((targets_GT_wo_nan, targets_GT_wo_nan ), dim=1)
            else :
                yGT = targets_GT_wo_nan

            
            yGT = torch.cat((yGT, u_gt_wo_nan, v_gt_wo_nan), dim=1)

            if self.use_sst_state :
                yGT = torch.cat((yGT,sst_gt), dim=1)
    
                                                
            loss_AE, loss_AE_GT =  self.reg_loss_lr(
                yGT, targets_OI, outputs, outputsSLRHR
            )
        else:
           loss_AE = 0. 
           loss_AE_GT = 0. 
        
        return loss_AE, loss_AE_GT

    def compute_reg_loss_hr(self,targets_lr,targets_GT_wo_nan, u_gt_wo_nan, sst_gt, v_gt_wo_nan, outputsSLRHR,phase):
        
        if outputsSLRHR is not None :            
            yGT = torch.cat((targets_lr,
                             targets_GT_wo_nan - targets_lr),
                            dim=1)
            if self.aug_state :
                yGT = torch.cat((yGT, targets_GT_wo_nan - targets_lr), dim=1)
                                                            
            yGT = torch.cat((yGT, u_gt_wo_nan, v_gt_wo_nan), dim=1)

            if self.use_sst_state :
                yGT = torch.cat((yGT,sst_gt), dim=1)
                                                     
            loss_AE, loss_AE_GT =  self.reg_loss_hr(yGT , outputsSLRHR)
        else:
           loss_AE = 0. 
           loss_AE_GT = 0. 
        
        return loss_AE, loss_AE_GT

    def reinterpolate_outputs(self,outputs,outputs_u,outputs_v,batch):
        
        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt, lat, lon = batch
            sst_gt = None
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = batch

        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        outputs = torch.nn.functional.interpolate(outputs, scale_factor=self.scale_lr, mode='bicubic')
        outputs_u = torch.nn.functional.interpolate(outputs_u, scale_factor=self.scale_lr, mode='bicubic')
        outputs_v = torch.nn.functional.interpolate(outputs_v, scale_factor=self.scale_lr, mode='bicubic')

        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        u_gt_wo_nan = u_gt.where(~u_gt.isnan(), torch.zeros_like(u_gt) )
        v_gt_wo_nan = v_gt.where(~v_gt.isnan(), torch.zeros_like(u_gt) )
        
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT_wo_nan)

        self.patch_weight = self.patch_weight_diag
        
        return targets_OI,targets_GT_wo_nan,sst_gt,u_gt_wo_nan,v_gt_wo_nan,lat_rad,lon_rad,outputs,outputs_u,outputs_v,g_targets_GT_x,g_targets_GT_y

    def compute_rec_loss(self,targets_GT_wo_nan,u_gt_wo_nan,v_gt_wo_nan,outputs,outputs_u,outputs_v,lat_rad,lon_rad,phase,flag_lr):
        flag_display_loss = False

        # median filter
        if self.median_filter_width > 1:
            outputs = kornia.filters.median_blur(outputs, (self.median_filter_width, self.median_filter_width))
                               
    
        # MSE loss for ssh and (u,v) components
        loss_All, loss_GAll = self.sla_loss(outputs, targets_GT_wo_nan)
        loss_uv = self.uv_loss( [outputs_u,outputs_v], [u_gt_wo_nan,v_gt_wo_nan])                

        # MSE for SSH-derived (u,v) fields
        # denormalize ssh
        u_geo_rec , v_geo_rec = self.compute_uv_from_ssh(outputs, lat_rad, lon_rad,sigma=0.) 
        u_geo_gt  , v_geo_gt  = self.compute_uv_from_ssh(targets_GT_wo_nan, lat_rad, lon_rad,sigma=0.) 
                                
        loss_uv_geo = self.uv_loss( [u_geo_rec,v_geo_rec], [u_geo_gt,v_geo_gt])
            #loss_GAll = ( self.hparams.alpha_mse_uv_geo / self.hparams.alpha_mse_gssh )  * loss_uv_geo
            #if flag_display_loss :
            #    print('..  loss uv geo = %e' % ( self.hparams.alpha_mse_uv_geo * loss_uv_geo ) )                     
            #    print('..  loss gssh = %e' % (self.hparams.alpha_mse_gssh * loss_GAll) )                     

        if self.type_div_train_loss == 0 :
            div_rec = self.compute_div(outputs_u,outputs_v)
            div_gt =  self.compute_div(u_gt_wo_nan,v_gt_wo_nan)
            
            loss_div = self.div_loss( div_rec , div_gt )
            loss_strain = 0.
            if flag_display_loss :
                print('\n..  loss div = %e' % (self.hparams.alpha_mse_div *loss_div) )                     
        else:
            if ( (phase == 'val') or (phase == 'test') ) :
                div_gt,curl_gt,strain_gt    = self.compute_div_curl_strain(u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad , sigma = self.sig_filter_div_diag, flag_lr = flag_lr )
                div_rec,curl_rec,strain_rec = self.compute_div_curl_strain(outputs_u, outputs_v, lat_rad, lon_rad , sigma = self.sig_filter_div_diag, flag_lr = flag_lr )
                                        
            else:
                div_gt,curl_gt,strain_gt    = self.compute_div_curl_strain(u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad , sigma = self.sig_filter_div, flag_lr = flag_lr )
                div_rec,curl_rec,strain_rec = self.compute_div_curl_strain(outputs_u, outputs_v, lat_rad, lon_rad , sigma = self.sig_filter_div, flag_lr = flag_lr )
                
            loss_div = self.div_loss( div_rec , div_gt )
            loss_strain = self.strain_loss( strain_rec , strain_gt )
            
            if flag_display_loss :
                print('\n..  loss div = %e' % (self.hparams.alpha_mse_div *loss_div) )                     
                print('..  loss strain = %e' % (self.hparams.alpha_mse_strain *loss_strain) )
                        

        if flag_display_loss :
            print('..  loss ssh = %e' % (self.hparams.alpha_mse_ssh * loss_All) )                     
            print('..  loss gssh = %e' % (self.hparams.alpha_mse_gssh * loss_GAll) )                     
            print('..  loss uv = %e' % (self.hparams.alpha_mse_uv * loss_uv) )                     


        return loss_All,loss_GAll,loss_uv,loss_uv_geo,loss_div,loss_strain 

    def compute_loss_lr(self, batch, phase, out_hr=(None,),state_init_lr=(None,)):

        ##############################################
        # run lr model
        _batch = self.pre_process_batch_for_lr(batch)
        self.patch_weight = self.patch_weight_lr
        
        targets_OI, inputs_Mask, inputs_obs, targets_GT_wo_nan, sst_gt, u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad, g_targets_GT_x, g_targets_GT_y = _batch

        
        #targets_OI, inputs_Mask, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_OI),
                    torch.cat((torch.zeros_like(targets_OI),
                              torch.zeros_like(targets_OI),
                              torch.zeros_like(targets_OI)), dim=1),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                        ('mseOI', 0.),
                        ('mse_uv', 0.),
                        ('mseGOI', 0.),
                        ('l0_samp', 0.),
                        ('l1_samp', 0.)])
                    )
         
        # intial state
        state = self.get_init_state_lr_from_hr(_batch, out_hr=out_hr)

        # obs and mask data
        obs,new_masks,w_sampling_uv,mask_sampling_uv = self.get_obs_and_mask_lr(inputs_Mask,inputs_obs,sst_gt,u_gt_wo_nan,v_gt_wo_nan)

        # run forward_model
        with torch.set_grad_enabled(True):
            flag_display_loss = False#True
            
            print(state[0].size(),flush=True)
            print(obs[0].size(),flush=True)
            
            if self.hparams.n_grad > 0 :                
                outputs, outputs_u, outputs_v, outputsSLRHR, hidden_new, cell_new, normgrad = self.run_model_lr(state, obs, new_masks,state_init_lr,
                                                                                                                         lat_rad,lon_rad,phase)
                
            else:
                outputs = self.model_4dvarnet_lr.phi_r(obs)
                                
                outputs_u = outputs[:, 1*self.hparams.dT:2*self.hparams.dT, :, :]
                outputs_v = outputs[:, 2*self.hparams.dT:3*self.hparams.dT, :, :]
                outputs = outputs[:, 0:self.hparams.dT, :, :]

                outputsSLRHR = None #0. * outputs
                hidden_new = None #0. * outputs
                cell_new = None # . * outputs
                normgrad = 0. 

            # projection losses
            loss_AE, loss_AE_GT = self.compute_reg_loss_lr(targets_OI,targets_GT_wo_nan, sst_gt,
                                                                           u_gt_wo_nan, v_gt_wo_nan,outputs, 
                                                                           outputsSLRHR, phase)
                                                              
            # reconstruction losses
            loss_All,loss_GAll,loss_uv,loss_uv_geo,loss_div,loss_strain = self.compute_rec_loss(targets_GT_wo_nan,u_gt_wo_nan,v_gt_wo_nan,
                                                                                                outputs,outputs_u,outputs_v,
                                                                                                lat_rad,lon_rad,phase, flag_lr = True)            
            
            loss_OI, loss_GOI = self.sla_loss(targets_OI, targets_GT_wo_nan)
            
            ######################################
            # Computation of total loss
            # reconstruction loss
            loss = self.hparams.alpha_mse_ssh * loss_All 
            loss += self.hparams.alpha_mse_gssh * loss_GAll
            loss += self.hparams.alpha_mse_uv * loss_uv
            loss += self.hparams.alpha_mse_uv_geo * loss_uv_geo
            
            if self.hparams.alpha_mse_div > 0. :
                loss += self.hparams.alpha_mse_div * loss_div
            if self.hparams.alpha_mse_strain > 0. :
                loss += self.hparams.alpha_mse_strain *loss_strain
               
            # regularization loss
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            
            if flag_display_loss :
                print('..  loss = %e' %loss )                     

            ######################################
            # metrics
            # mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(g_targets_GT, self.w_loss)
            mean_GAll = NN_4DVar.compute_spatio_temp_weighted_loss(
                    torch.hypot(g_targets_GT_x, g_targets_GT_y) , self.grad_crop(self.patch_weight))
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            mse_uv = loss_uv.detach()

            mse_div = loss_div.detach()
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
            out_feat = sst_gt[:,int(self.hparams.dT /2),:,:].view(-1,1,sst_gt.size(2),sst_gt.size(3))
            
            if self.use_sst_obs :
                #sst_feat = self.model.model_H.conv21( inputs_SST )
                out_feat = torch.cat( (out_feat,self.model_4dvarnet_lr.model_H.extract_sst_feature( sst_gt )) , dim = 1 )
                ssh_feat = self.model_4dvarnet_lr.model_H.extract_state_feature( outputsSLRHR )
                
                ssh_feat = torch.nn.functional.interpolate(ssh_feat, scale_factor=self.scale_lr, mode='bicubic')
                out_feat = torch.nn.functional.interpolate(out_feat, scale_factor=self.scale_lr, mode='bicubic')
                out_feat = torch.cat( (out_feat,ssh_feat) , dim=1)
                
            if self.model_sampling_uv is not None :
                out_feat = torch.cat( (out_feat,w_sampling_uv) , dim=1)
                    
            return loss, [outputs,outputs_u,outputs_v], [outputsSLRHR, hidden_new, cell_new, normgrad], metrics, out_feat
            
        else:
            return loss, [outputs,outputs_u,outputs_v], [outputsSLRHR, hidden_new, cell_new, normgrad], metrics


    def pre_process_batch_for_hr(self,batch):
        # time window for hr component
        _dt = int( (self.hparams.dT-self.hparams.dT_hr_model)/2 )
        

        if not self.use_sst:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, u_gt, v_gt, lat, lon = batch
        else:
            targets_OI, inputs_Mask, inputs_obs, targets_GT, sst_gt, u_gt, v_gt, lat, lon = batch
            sst_gt = sst_gt[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
    
        targets_OI = targets_OI[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        inputs_Mask = inputs_Mask[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        inputs_obs = inputs_obs[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        targets_GT = targets_GT[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        u_gt = u_gt[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
        v_gt = v_gt[:,_dt:_dt+self.hparams.dT_hr_model,:,:]
                    
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)
        u_gt_wo_nan = u_gt.where(~u_gt.isnan(), torch.zeros_like(u_gt) )
        v_gt_wo_nan = v_gt.where(~v_gt.isnan(), torch.zeros_like(u_gt) )
        
        if not self.use_sst:
            sst_gt = None
            
        # gradient norm field
        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT_wo_nan)

        # lat/lon in radians
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)
            
        if self.use_lat_lon_in_obs_model  == True :
            self.model_4dvarnet_hr.model_H.lat_rad = lat_rad
            self.model_4dvarnet_hr.model_H.lon_rad = lon_rad

        return targets_OI, inputs_Mask, inputs_obs, targets_GT_wo_nan, sst_gt, u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad, g_targets_GT_x, g_targets_GT_y

    def compute_loss_hr(self, batch, phase, out_lr, state_init_hr=(None,)):

        ##############################################
        # run lr model
        _batch = self.pre_process_batch_for_hr(batch)
        self.patch_weight = self.patch_weight_lr
        
        targets_OI, inputs_Mask, inputs_obs, targets_GT_wo_nan, sst_gt, u_gt_wo_nan, v_gt_wo_nan, lat_rad, lon_rad, g_targets_GT_x, g_targets_GT_y = _batch
            
        print( inputs_obs.size() )
        
        # low-resolution reference
        targets_lr = torch.nn.functional.avg_pool2d(targets_GT_wo_nan, (int(self.scale_lr),int(self.scale_lr)))        
        targets_lr = torch.nn.functional.interpolate(targets_lr, scale_factor=self.scale_lr, mode='bicubic')
 
        self.patch_weight = self.patch_weight_hr

        g_targets_GT_x, g_targets_GT_y = self.gradient_img(targets_GT_wo_nan)        
        
        #targets_OI, inputs_Mask, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                    None,
                    torch.zeros_like(targets_OI),
                    torch.cat((torch.zeros_like(targets_OI),
                              torch.zeros_like(targets_OI),
                              torch.zeros_like(targets_OI)), dim=1),
                    dict([('mse', 0.),
                        ('mseGrad', 0.),
                        ('meanGrad', 1.),
                        ('mseOI', 0.),
                        ('mse_uv', 0.),
                        ('mseGOI', 0.),
                        ('l0_samp', 0.),
                        ('l1_samp', 0.)])
                    )
         
        # intial state
        state = self.get_init_state_hr_from_lr(_batch, out_lr, state_init_hr )

        # obs and mask data
        obs,new_masks,w_sampling_uv,mask_sampling_uv = self.get_obs_and_mask_hr(out_lr[0].detach(),inputs_Mask,inputs_obs,sst_gt,u_gt_wo_nan,v_gt_wo_nan)


        print('.... get_obs_mask_state_hr ')
        print( obs[0].size() )
        print( state.size() )
        print( new_masks[0].size(),flush=True )
        print( obs[1].size() )
        print( new_masks[1].size(),flush=True )

        # run forward_model
        with torch.set_grad_enabled(True):
            flag_display_loss = False#True
            
            if self.hparams.n_grad > 0 :                
                outputs, outputs_u, outputs_v, outputsSLRHR, outputsSLR, hidden_new, cell_new, normgrad = self.run_model_hr(state, obs, new_masks,state_init_hr,
                                                                                                                            lat_rad,lon_rad,phase)
                        
            else:
                outputs = self.model_4dvarnet_hr.phi_r(obs)
                                                
                outputs_u = outputs[:, 1*self.hparams.dT_hr_model:2*self.hparams.dT_hr_model, :, :]
                outputs_v = outputs[:, 2*self.hparams.dT_hr_model:3*self.hparams.dT_hr_model, :, :]
                outputs = outputs[:, 0:self.hparams.dT_hr_model, :, :]

                outputsSLR = None
                outputsSLRHR = None #0. * outputs
                hidden_new = None #0. * outputs
                cell_new = None # . * outputs
                normgrad = 0. 


            print('...... self.compute_reg_loss_hr')
            print(targets_lr.size())
            print(targets_GT_wo_nan.size())
            print(sst_gt.size())
            print(u_gt_wo_nan.size())
            print(v_gt_wo_nan.size())
            print(outputsSLR.size())
            print(outputsSLRHR.size())
            
            # projection losses
            loss_AE, loss_AE_GT = self.compute_reg_loss_hr(targets_lr,targets_GT_wo_nan, sst_gt,
                                                                             u_gt_wo_nan, v_gt_wo_nan,outputsSLRHR,phase)
                                                              
            # reconstruction losses
            loss_All,loss_GAll,loss_uv,loss_uv_geo,loss_div,loss_strain = self.compute_rec_loss(targets_GT_wo_nan,u_gt_wo_nan,v_gt_wo_nan,
                                                                                    outputs,outputs_u,outputs_v,
                                                                                    lat_rad,lon_rad,phase, flag_lr = False)
                        
            loss_OI, loss_GOI = self.sla_loss(targets_lr, targets_GT_wo_nan)
            
            if self.model_sampling_uv is not None :
                loss_l1_sampling_uv = float( self.hparams.dT / (self.hparams.dT - int(self.hparams.dT/2))) *  torch.mean( w_sampling_uv )
                loss_l1_sampling_uv = torch.nn.functional.relu( loss_l1_sampling_uv - self.hparams.thr_l1_sampling_uv )
                loss_l0_sampling_uv = float( self.hparams.dT / (self.hparams.dT - int(self.hparams.dT/2))) * torch.mean( mask_sampling_uv ) 

            ######################################
            # Computation of total loss
            # reconstruction loss
            loss = self.hparams.alpha_mse_ssh * loss_All 
            loss += self.hparams.alpha_mse_gssh * loss_GAll
            loss += self.hparams.alpha_mse_uv * loss_uv
            loss += self.hparams.alpha_mse_uv_geo * loss_uv_geo
            
            if self.hparams.alpha_mse_div > 0. :
                loss += self.hparams.alpha_mse_div * loss_div
            if self.hparams.alpha_mse_strain > 0. :
                loss += self.hparams.alpha_mse_strain *loss_strain
               
            # regularization loss
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            #loss += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
            
            # sampling loss
            if self.model_sampling_uv is not None :
                loss += self.hparams.alpha_sampling_uv * loss_l1_sampling_uv

            if flag_display_loss :
                print('..  loss = %e' %loss )                     

            ######################################
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
            out_feat = sst_gt[:,int(self.hparams.dT_hr_model/2),:,:].view(-1,1,sst_gt.size(2),sst_gt.size(3))
            
            if self.use_sst_obs :
                #sst_feat = self.model.model_H.conv21( inputs_SST )
                out_feat = torch.cat( (out_feat,self.model_4dvarnet_hr.model_H.extract_sst_feature( sst_gt )) , dim = 1 )
                ssh_feat = self.model_4dvarnet_hr.model_H.extract_state_feature( outputsSLRHR )
                
                if self.scale_dwscaling_sst > 1 :
                    ssh_feat = torch.nn.functional.interpolate(ssh_feat, scale_factor=self.scale_dwscaling_sst, mode='bicubic')
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
