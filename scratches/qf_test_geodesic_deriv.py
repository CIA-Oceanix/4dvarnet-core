import metpy.calc as mpcalc
import kornia.filters
import kornia
import torch
import numpy as np
import xarray as xr
import pyinterp
import pyinterp.backends.xarray
import pyinterp.fill


ds = xr.open_dataset('../sla-data-registry/qdata/natl20.nc')
ds.ssh.isel(time=1).plot()

## Metpy lib
def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

vort = lambda da: mpcalc.vorticity(
        *mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())
).metpy.dequantify()

geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()
geo_uv = lambda da:mpcalc.geostrophic_wind(da)

ds.load().ssh.pipe(remove_nan).pipe(vort).isel(time=0).plot()
ds.load().ssh.pipe(remove_nan).pipe(geo_energy).isel(time=0).plot()


## Implem ronan
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

        elif _filter == 'diff':
            print('should be here')
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
        return kornia.filters.spatial_gradient(u, normalized=True, mode='sobel').unbind(2)       

        # if sigma > 0. :
        #     u = kornia.filters.gaussian_blur2d(u, (3,3), (sigma,sigma), border_type='reflect')

        # G_x  = self.convGx( u[:,0,:,:].view(-1,1,u.size(2), u.size(3)) )
        # G_y  = self.convGy( u[:,0,:,:].view(-1,1,u.size(2), u.size(3)) )
        
        # for kk in range(1,u.size(1)):
        #     _G_x  = self.convGx( u[:,kk,:,:].view(-1,1,u.size(2), u.size(3)) )
        #     _G_y  = self.convGy( u[:,kk,:,:].view(-1,1,u.size(2), u.size(3)) )
                
        #     G_x  = torch.cat( (G_x,_G_x) , dim = 1 )
        #     G_y  = torch.cat( (G_y,_G_y) , dim = 1 )
            
        # return G_x,G_y
    
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
        return kornia.filters.spatial_gradient(u, normalized=True, mode='diff').unbind(2)       

        # if sigma > 0.:
        #     u = kornia.filters.gaussian_blur2d(
        #         u, (3, 3), (sigma, sigma), border_type='reflect',
        #     )

        # G_x  = self.convGx(u[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)))
        # G_y  = self.convGy(u[:, 0, :, :].view(-1, 1, u.size(2), u.size(3)))

        # for kk in range(1, u.size(1)):
        #     _G_x = self.convGx(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))
        #     _G_y = self.convGy(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))

        #     G_x = torch.cat((G_x, _G_x), dim = 1)
        #     G_y = torch.cat((G_y, _G_y), dim = 1)

        # return G_x, G_y

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

input_da = ds.isel(time=slice(None, 1)).ssh.pipe(remove_nan).transpose('time', 'lat', 'lon').astype(np.float32)
item, lat, lon = map(
        torch.from_numpy,
        input_da.pipe(
            lambda da: [ da.values, da.lat.values, da.lon.values]
))
item, lat, lon = item.float(), lat.float(), lon.float()
lon[None].shape



mod = Torch_compute_derivatives_with_lon_lat(dT=1,_filter='sobel')
# mod = TorchComputeDerivativesWithLonLat(dT=1,_filter='diff')

u, v = mod.compute_geo_velocities(item[None], torch.deg2rad(lat[None]), torch.deg2rad(lon[None]))

my_u, my_v = geo_uv(input_da)
my_u, my_v = my_u.metpy.dequantify(), my_v.metpy.dequantify()

((my_u).transpose('time', 'lat', 'lon') - (u.squeeze(0).numpy())).values

new_ds = my_u.to_dataset(name='mine').assign(
        ronan=( ('time', 'lat', 'lon'), u.squeeze(0).numpy()),
)
new_ds.to_array().isel(lat=slice(None, 10)).plot.pcolormesh(col='variable', row='time', robust=True)
new_ds.pipe(lambda ds: ds.mine - ds.ronan).plot.pcolormesh(row='time', robust=True)

new_ds.map(np.isnan).sum()
(
    new_ds
    .pipe(
        lambda ds: ds.mine - ds.ronan
    ).pipe(np.square).mean(['lon', 'time'])
    .isel(lat=slice(1, -1))
).plot()


mod.convGx.weight.chans(scale=30)
mod.convGy.weight.chans(scale=30)
