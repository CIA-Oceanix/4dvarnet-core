import xarray as xr
import numpy as np
import logging
import matplotlib.pylab as plt
from scipy import interpolate
import hvplot.xarray
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle

def find_wavelength_05_crossing(filename):
    
    ds = xr.open_dataset(filename)
    y = 1./ds.wavenumber
    x = (1. - ds.psd_diff/ds.psd_ref)
    f = interpolate.interp1d(x, y)
    
    xnew = 0.5
    ynew = f(xnew)
    
    return ynew
    
    

def plot_psd_score(filename):
    
    ds = xr.open_dataset(filename)
    
    resolved_scale = find_wavelength_05_crossing(filename)
    
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    ax.invert_xaxis()
    plt.plot((1./ds.wavenumber), ds.psd_ref, label='reference', color='k')
    plt.plot((1./ds.wavenumber), ds.psd_study, label='reconstruction', color='lime')
    plt.xlabel('wavelength [km]')
    plt.ylabel('Power Spectral Density [m$^{2}$/cy/km]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid(which='both')
    
    ax = plt.subplot(122)
    ax.invert_xaxis()
    plt.plot((1./ds.wavenumber), (1. - ds.psd_diff/ds.psd_ref), color='k', lw=2)
    plt.xlabel('wavelength [km]')
    plt.ylabel('PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]')
    plt.xscale('log')
    plt.hlines(y=0.5, 
              xmin=np.ma.min(np.ma.masked_invalid(1./ds.wavenumber)), 
              xmax=np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
              color='r',
              lw=0.5,
              ls='--')
    plt.vlines(x=resolved_scale, ymin=0, ymax=1, lw=0.5, color='g')
    ax.fill_betweenx((1. - ds.psd_diff/ds.psd_ref), 
                     resolved_scale, 
                     np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
                     color='green',
                     alpha=0.3, 
                     label=f'resolved scales \n $\lambda$ > {int(resolved_scale)}km')
    plt.legend(loc='best')
    plt.grid(which='both')
    
    logging.info(' ')
    logging.info(f'  Minimum spatial scale resolved = {int(resolved_scale)}km')
    
    plt.show()
    
    return fig, resolved_scale
    
    
def plot_spatial_statistics(filename):
    
    ds = xr.open_dataset(filename, group='diff')
#     try:
#         figure = ds['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clabel='RMSE [m]', cmap='Reds', coastline=True)
#     except KeyError:
#         figure = ds['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clabel='RMSE [m]', cmap='Reds')

    figure = ds['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clabel='RMSE [m]', cmap='Reds')
        
    return figure


def plot_temporal_statistics(filename):
    
    ds1 = xr.open_dataset(filename, group='diff')
    ds2 = xr.open_dataset(filename, group='alongtrack')
    rmse_score = 1. - ds1['rms']/ds2['rms']
    
    rmse_score = rmse_score.dropna(dim='time').where(ds1['count'] > 10, drop=True)
    
    figure = rmse_score.hvplot.line(ylabel='RMSE SCORE', shared_axes=True, color='r') + ds1['count'].dropna(dim='time').hvplot.step(ylabel='#Obs.', shared_axes=True, color='grey')
    
    return figure.cols(1) 
    

def spectral_score_intercomparison(list_of_filename, list_of_label):
    
    plt.figure(figsize=(15, 6))
    ax = plt.subplot(121)
    ax.invert_xaxis()
    ds = xr.open_dataset(list_of_filename[0])
    plt.plot((1./ds.wavenumber), ds.psd_ref, label='reference', color='k')
    for cfilename, clabel in zip(list_of_filename, list_of_label):
        ds = xr.open_dataset(cfilename)
        plt.plot((1./ds.wavenumber), ds.psd_study, label=clabel)
    plt.xlabel('wavelength [km]')
    plt.ylabel('Power Spectral Density [m$^{2}$/cy/km]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid(which='both')
    plt.xticks([50, 100, 200, 500, 1000], ["50km", "100km", "200km", "500km", "1000km"])
    
    ax = plt.subplot(122)
    ax.invert_xaxis()
    for cfilename, clabel in zip(list_of_filename, list_of_label):
        ds = xr.open_dataset(cfilename)
        score = (1. - ds.psd_diff/ds.psd_ref)
        plt.plot(*score.isel(wavenumber=score>0).pipe(lambda da: (1/da.wavenumber, da)) , lw=2, label=clabel)
        # plt.plot((1./ds.wavenumber), (1. - ds.psd_diff/ds.psd_ref), lw=2, label=clabel)
    plt.xlabel('wavelength [km]')
    plt.ylabel('PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]')
    plt.xscale('log')
    plt.hlines(y=0.5, 
              xmin=np.ma.min(np.ma.masked_invalid(1./ds.wavenumber)), 
              xmax=np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
              color='r',
              lw=0.5,
              ls='--')
#     plt.vlines(x=resolved_scale, ymin=0, ymax=1, lw=0.5, color='g')
#     ax.fill_betweenx((1. - ds.psd_diff/ds.psd_ref), 
#                      resolved_scale, 
#                      np.ma.max(np.ma.masked_invalid(1./ds.wavenumber)),
#                      color='green',
#                      alpha=0.3, 
#                      label=f'resolved scales \n $\lambda$ > {int(resolved_scale)}km')
    plt.legend(loc='best')
    plt.grid(which='both')
    plt.xticks([50, 100, 200, 500, 1000], ["50km", "100km", "200km", "500km", "1000km"])
    
    plt.show()
    

def intercomparison_temporal_statistics(list_of_filename, list_of_label):
    
    ds_diff = xr.concat([xr.open_dataset(filename, group='diff') for filename in list_of_filename], dim='experiment')
    ds_diff['experiment'] = list_of_label
    ds_alongtrack = xr.concat([xr.open_dataset(filename, group='alongtrack') for filename in list_of_filename], dim='experiment')
    ds_alongtrack['experiment'] = list_of_label
    
    rmse_score = 1. - ds_diff['rms']/ds_alongtrack['rms']
    rmse_score = rmse_score.dropna(dim='time').where(ds_diff['count'] > 10, drop=True)
    
    figure = rmse_score.hvplot.line(x='time', y='rms', by='experiment', ylim=(0, 1), title='RMSE SCORE', shared_axes=True) + ds_diff['count'][0, :].dropna(dim='time').hvplot.step(ylabel='#Obs.', shared_axes=True, color='grey')
    
    return figure.cols(1)


def intercomparison_spatial_statistics(baseline_filename, list_of_filename, list_of_label):
      
    ds_baseline = xr.open_dataset(baseline_filename, group='diff')
    ds = xr.concat([xr.open_dataset(filename, group='diff') for filename in list_of_filename], dim='experiment')
    ds['experiment'] = list_of_label
    
    delta_rmse = 100*(ds - ds_baseline)/ds_baseline

    figure = delta_rmse['rmse'].hvplot.image(x='lon', y='lat', z='rmse', clim=(-20, 20), by='experiment', 
                                             subplots=True, projection=ccrs.PlateCarree(),
                                             clabel='[%]', cmap='coolwarm', coastline=True)
    
    return figure.cols(2)


def hvplot_demo_obs_nadir(list_of_dataset, central_date, delta_t):

    ds_concat_nadirs = xr.concat(list_of_dataset, dim='time')
    ds_concat_nadirs = ds_concat_nadirs.sortby(ds_concat_nadirs.time)
    ds_concat_nadirs = ds_concat_nadirs.assign_coords({'longitude': ds_concat_nadirs.longitude, 'latitude': ds_concat_nadirs.latitude})

    ds_concat_nadirs_selection = ds_concat_nadirs.sel(time=slice(central_date - delta_t, central_date + delta_t)).drop(
        'time')

    plot = ds_concat_nadirs_selection.hvplot.scatter(x='longitude', y='latitude', color='sla_unfiltered',
                                                     height=300, width=400, cmap = 'gist_stern', datashade=True) 

    return plot


def plot_demo_obs(list_of_dataset, central_date, delta_t):

    tmin = central_date - delta_t
    tmax = central_date + delta_t

    list_of_dataset_sel = []
    for ds in list_of_dataset:
        ds_sel = ds.sel(time=slice(tmin, tmax))
        if ds_sel.time.size > 0:
            list_of_dataset_sel.append(ds_sel)

    plt.figure(figsize=(10, 10))
    plt.subplot(111)
    for ds in list_of_dataset_sel:
        plt.scatter(ds.longitude % 360., ds.latitude, c=ds.sla_unfiltered, s=20, cmap='gist_stern')
    ax = plt.gca()
    ax.add_patch(Rectangle((295, 33), 10, 10, fill=None, alpha=1))
    plt.xlabel('longitude', fontweight='bold')
    plt.ylabel('latitude', fontweight='bold')
    plt.title(f'SLA @ altimeter track')
    plt.colorbar(orientation='horizontal')
    plt.text(298, 43.5,'STUDY AREA')
    plt.show()
