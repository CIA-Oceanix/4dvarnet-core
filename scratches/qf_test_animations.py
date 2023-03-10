
import hydra
import importlib
from hydra.utils import instantiate, get_class, call
import hydra_main
import lit_model_augstate

importlib.reload(lit_model_augstate)
importlib.reload(hydra_main)
from utils import get_cfg, get_dm, get_model
from omegaconf import OmegaConf
# OmegaConf.register_new_resolver("mul", lambda x,y: int(x)*y, replace=True)
import hydra_config

cfg_n, ckpt = 'full_core', 'results/xpmultigpus/xphack4g_augx4/version_0/checkpoints/modelCalSLAInterpGF-epoch=26-val_loss=1.4156.ckpt'
cfg_n = f"baseline/{cfg_n}"
dm = get_dm(cfg_n, setup=False,
        add_overrides=[
            'file_paths=dgx_ifremer',
        ]


)
mod = get_model(
        cfg_n,
        ckpt,
        dm=dm)
cfg = get_cfg(cfg_n,
        overrides=[
            'file_paths=dgx_ifremer',
        ]
        )
print(OmegaConf.to_yaml(cfg))
lit_mod_cls = get_class(cfg.lit_mod_cls)
runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
mod = runner.test(ckpt)
mod.test_figs['maps']
import cartopy.crs as ccrs
import matplotlib.animation as mpa

fig = mod.test_xr_ds.isel(time=0).to_array().plot.pcolormesh(
        x='lon', y='lat', col='variable', col_wrap=2)
newfig = mod.test_xr_ds.isel(time=1).to_array().plot.pcolormesh(
        x='lon', y='lat', col='variable', col_wrap=2)


import holoviews as hv
from holoviews import opts

hv.extension('matplotlib')

import xarray as xr
from scipy import ndimage
import numpy as np
def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
    return np.hypot(dx_ac, dx_al)
hvds = hv.Dataset(mod.test_xr_ds.pipe(sobel).isel(time=slice(None, 6)))
images = hv.Layout([
        hvds
        .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
        .options(cmap='RdYlBu')
        for v in ['pred', 'gt', 'obs_inp']
        ]).cols(2).opts(sublabel_format="")
images

hv.output(images, holomap='mp4', fps=3)


plt.cm.get_cmap("RdYlBu")

# def get_grid_fig(to_plot_ds):
# clims = Clim(to_plot_ds)
# hv_layout = hv.Layout([
#     hv.Dataset(
#         to_plot_ds, ['lon', 'lat'], var
#     ).to(
#         hv.QuadMesh, kdims=['lon', 'lat']
#     ).relabel(
#         f'{var}'
#     ).options(
#         colorbar=True,
#         cmap='PiYG',
#         clim=clims[var],
#         aspect=2
#     )
#     for var in to_plot_ds
# ]).cols(2)


vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()

def anim(test_xr_ds, deriv=None,  dvars=['ssh']):
    if deriv is None:
        tpds = test_xr_ds
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdBu'

    if deriv == 'grad':
        tpds = test_xr_ds.map(geo_energy)
        clim = (0, tpds[dvars].to_array().max().item())
        cmap = 'viridis'
    
    if deriv == 'lap':
        tpds = test_xr_ds.map(vort)
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdGy'

    hvds = hv.Dataset(tpds)
    if len(dvars) == 1:
        return hvds.to(hv.QuadMesh, ['lon', 'lat'], dvars[0]).relabel(dvars[0]).options(cmap=cmap, clim=clim,)
    images = hv.Layout([
            hvds
            .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
            .options(
                cmap=cmap,
                clim=clim,
                colorbar=True,
            )
            for v in dvars
            ]).opts(sublabel_format="")
    return images



import pyinterp.fill
import pyinterp.backends.xarray
smaller_domain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}
# img_ssh = anim(
#         all_dses[k].sel(smaller_domain).pipe(remove_nan).to_dataset(name='ssh'),
#         deriv=None, dvars=['ssh'])
# img_grad = anim(
#         all_dses[k].sel(smaller_domain).pipe(remove_nan).to_dataset(name='geo'),
#         deriv='grad', dvars=['geo'])
# img_lap = anim(
#         all_dses[k].sel(smaller_domain).pipe(remove_nan).to_dataset(name='vort'),
#         deriv='lap', dvars=['vort'])
# hv.output(img_ssh + img_grad + img_lap, holomap='gif', fps=2, dpi=125)
duacs = xr.open_dataset('/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc').coarsen(lat=2,lon=2, boundary='trim').mean().sel(smaller_domain)
glorys = xr.open_dataset('../sla-data-registry/GLORYS/cmems_mod_glo_phy_my_0.083_P1D-m_1664351861719.nc')
glorys = xr.open_dataset('../sla-data-registry/GLORYS/cmems_mod_glo_phy_my_0.083_P1D-m_1674061226086.nc')
glo_da = glorys.zos.load().rename(latitude='lat', longitude='lon').pipe(remove_nan).sel(smaller_domain)
# fourdvarnet = xr.open_dataset('version_8/test.nc').rename(pred='4dvarnet')
# fourdvarnet = xr.open_dataset('/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-01-18/11-38-40/ose_ssh_rec.nc').rename(rec_ssh='4dvarnet').map(lambda da: ndimage.median_filter(da, (1, 3, 3)))
fourdvarnet = xr.open_dataset('/raid/localscratch/qfebvre/4dvarnet-starter/tmp/ensemble_oseens_ose_rec_ssh.nc').rename(rec_ssh_9='4dvarnet')#.map(lambda da: ndimage.median_filter(da, (1, 3, 3)))

to_plot_ds = fourdvarnet.assign(glorys=glo_da.interp(**fourdvarnet.coords)+0.31446309894037083, duacs=duacs.ssh.interp(**fourdvarnet.coords)).load().map(remove_nan).isel(time=slice(40, 100,1))
img_ssh = anim( to_plot_ds, deriv=None, dvars=['4dvarnet', 'glorys', 'duacs'])
img_grad = anim( to_plot_ds, deriv='grad', dvars=['4dvarnet', 'glorys', 'duacs'])
# to_plot_ds.map(geo_energy).map(lambda da: ndimage.median_filter(da, 5))
hv.output((img_ssh+
           img_grad).cols(3)
          ,holomap='gif', fps=4, dpi=125)
psd_ds = fourdvarnet.assign(glorys=glo_da.interp(**fourdvarnet.coords), duacs=duacs.ssh.interp(**fourdvarnet.coords)).load()[['glorys', 'duacs', '4dvarnet']].map(remove_nan).isel(time=slice(30, -30))

psd_ds.to_netcdf('../ose_glo_duacs_4dvar.nc')
name = 'PSD geostrop energy'
name = 'PSD SSH'
name = 'PSD vort'
( 
    psd_ds
    # .map(geo_energy)
    .map(vort)
    .map(psd_fn).mean('lat').mean('time')
    .assign(wavelength=lambda ds: 1/ds.freq_lon*111)
    .swap_dims(freq_lon='wavelength').to_array()
    .pipe(lambda ds: ds.isel(wavelength=(ds.wavelength>=10)))
    .pipe(lambda ds: ds.isel(wavelength=(ds.wavelength<=400)))
    .to_dataset(name=name)[name]
    .plot.line(hue='variable', yscale='log')
)


import pandas as pd
import metpy.calc as mpcalc
import pyinterp.fill
import pyinterp.backends.xarray
import xrft


vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()

def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

def reset_time(ds):
    ds = ds.copy()
    ds['time'] = ((pd.to_datetime(ds.time) - pd.to_datetime('2006-01-01')) /pd.to_timedelta('1D')) %366 //1
    ds = ds.sortby('time')
    return ds


def reset_latlon(ds, dx=0.05):
    ds = ds.copy()
    ds['lat'] = np.arange(34, 44, dx)
    ds['lon'] = np.arange(-65, -55, dx)
    return ds


enatl_wo = xr.open_dataset('/raid/localscratch/qfebvre/sla-data-registry/qdata/enatl_wo_tide.nc')
enatl_w = xr.open_dataset('/raid/localscratch/qfebvre/sla-data-registry/qdata/enatl_w_tide.nc')
natl = xr.open_dataset('/raid/localscratch/qfebvre/sla-data-registry/qdata/natl20.nc')
glorys = xr.open_dataset('/raid/localscratch/qfebvre/sla-data-registry/qdata/glo12_free.nc')
orca = xr.open_dataset('/raid/localscratch/qfebvre/sla-data-registry/qdata/orca25.nc')

psd_fn = lambda da: xrft.power_spectrum(
        da, dim='lon', scaling='density', real_dim='lon', truncate=True, window='hann')

# psd vort
psd_ds = xr.Dataset(
    dict(natl= psd_fn(natl.ssh.load().pipe(remove_nan).pipe(vort)).mean('lat').mean('time'),
         enatl_wo_tide=psd_fn(enatl_wo.ssh.load().pipe(remove_nan).pipe(vort)).mean('lat').mean('time'),
         enatl_w_tide=psd_fn(enatl_w.ssh.load().pipe(remove_nan).pipe(vort)).mean('lat').mean('time'),
         # glorys=psd_fn(glorys.ssh.load().pipe(remove_nan).pipe(vort)).mean('lat').mean('time'),
         # orca=psd_fn(orca.ssh.load().pipe(remove_nan).pipe(vort)).mean('lat').mean('time'),
 ))

(
        psd_ds
        .assign(wavelength=lambda ds: 1/ds.freq_lon*111)
        .swap_dims(freq_lon='wavelength')
        .to_array()
        # .pipe(lambda ds: ds.isel(wavelength=(ds>10**-11).any('variable')))
        .pipe(lambda ds: ds.isel(wavelength=(ds.wavelength>=50) & (ds.wavelength<=200)))
        .plot.line(hue='variable', yscale='log')
)

# psd geo energy
psd_ds = xr.Dataset(
    dict(natl= psd_fn(natl.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
         enatl_wo_tide=psd_fn(enatl_wo.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
         enatl_w_tide=psd_fn(enatl_w.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
         # glorys=psd_fn(glorys.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
         # orca=psd_fn(orca.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
 ))

(
        psd_ds
        .assign(wavelength=lambda ds: 1/ds.freq_lon*111)
        .swap_dims(freq_lon='wavelength')
        .to_array()
        # .pipe(lambda ds: ds.isel(wavelength=(ds>10**-11).any('variable')))
        .pipe(lambda ds: ds.isel(wavelength=(ds.wavelength>=50) & (ds.wavelength<=200)))
        .plot.line(hue='variable', yscale='log')
)

# psd ssh
psd_ds = xr.Dataset(
    dict(natl= psd_fn(natl.ssh.load().pipe(remove_nan)).mean('lat').mean('time'),
         enatl_wo_tide=psd_fn(enatl_wo.ssh.load().pipe(remove_nan)).mean('lat').mean('time'),
         enatl_w_tide=psd_fn(enatl_w.ssh.load().pipe(remove_nan)).mean('lat').mean('time'),
         glorys=psd_fn(glorys.ssh.load().pipe(remove_nan)).mean('lat').mean('time'),
         orca=psd_fn(orca.ssh.load().pipe(remove_nan)).mean('lat').mean('time'),
 ))

(
        psd_ds
        .assign(wavelength=lambda ds: 1/ds.freq_lon*111)
        .swap_dims(freq_lon='wavelength')
        .to_array()
        # .pipe(lambda ds: ds.isel(wavelength=(ds>10**-11).any('variable')))
        .pipe(lambda ds: ds.isel(wavelength=(ds.wavelength>=50) & (ds.wavelength<=200)))
        .plot.line(hue='variable', yscale='log')
)

def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')
    # has_conv, da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
    #     pyinterp.backends.xarray.Grid3D(da))
    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.loess(
        pyinterp.backends.xarray.Grid3D(da))
    return da

smaller_domain = {'lat': slice(41, 43), 'lon':slice(-59,-57)}
smaller_domain = {'lat': slice(32, 44), 'lon':slice(-66,-54)} 


samp.sossheig.pipe(np.isnan).isel(time_counter=0).plot()
samp.nav_lat.isel(x=600).plot()
samp.nav_lon.isel(y=600).plot()
enatl_w.ssh.pipe(remove_nan).pipe(np.isnan).sum()
enatl_w.ssh.sel(smaller_domain).pipe(np.isnan).sum()
enatl_w.ssh.sel(smaller_domain).isel(time=1).plot()
anim_ds = xr.Dataset(
    dict(natl= natl.ssh.load().pipe(reset_time).pipe(remove_nan).pipe(geo_energy),
         enatl_wo_tide=enatl_wo.ssh.load().pipe(reset_time).pipe(remove_nan).pipe(geo_energy),
         enatl_w_tide=enatl_w.ssh.load().pipe(reset_time).pipe(remove_nan).pipe(geo_energy),
         # glorys=psd_fn(glorys.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
         # orca=psd_fn(orca.ssh.load().pipe(remove_nan).pipe(geo_energy)).mean('lat').mean('time'),
 ))
smaller_domain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}
anim_ds.isel(time=slice(30, None, 90)).sel(smaller_domain).to_array().plot.pcolormesh(row='variable', col='time', robust=True)
anim_ds.mean('lat').mean('lon').to_array().plot.line(hue='variable')
list(anim_ds)
img = anim(anim_ds.isel(time=slice(0, 15)), dvars=list(anim_ds))
hv.output(img,holomap='gif', fps=4, dpi=125)
