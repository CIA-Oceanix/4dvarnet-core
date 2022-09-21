print(f"Using current {__name__}")
import datetime
import einops
import numpy as np
import xarray as xr
import matplotlib
import os
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import shapely
from shapely import wkt
#import geopandas as gpd
import cartopy
from os.path import expanduser
#cartopy.config['pre_existing_data_dir'] = expanduser('/gpfswork/rech/yrf/uba22to/4dvarnet-core/shapefiles/natural_earth/physical')
#cartopy.config['data_dir'] = '/gpfswork/rech/yrf/uba22to/4dvarnet-core/shapefiles/natural_earth/physical'
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cv2
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import xrft

from spectral import *

def plot_snr(gt,pred,resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''

    dt = pred.shape[1]

    # Compute Signal-to-Noise ratio
    f, pf = avg_err_rapsd2dv1(pred,gt,4.,True)
    wf = 1./f
    snr_pred = [wf, pf]

    # plot Signal-to-Noise ratio
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(snr_pred[0],snr_pred[1],color='blue',linewidth=2,label='4DVarNet')
    ax.set_xlabel("Wavenumber", fontweight='bold')
    ax.set_ylabel("Signal-to-noise ratio", fontweight='bold')
    ax.set_xscale('log') ; ax.set_yscale('log')
    plt.legend(loc='best',prop=dict(size='small'),frameon=False)
    plt.xticks([50, 100, 200, 500, 1000], ["50km", "100km", "200km", "500km", "1000km"])
    ax.invert_xaxis()
    plt.grid(which='both', linestyle='--')
    plt.savefig(resfile) # save the figure
    fig = plt.gcf()
    plt.close()          # close the figure
    return fig

def plot_nrmse(gt, pred, resfile, time):
    '''
    gt: 3d numpy array (Ground Truth)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    time: 1d array-like of time corresponding to the experiment
    '''

    # Compute daily nRMSE scores
    nrmse_pred = []
    for i in range(len(gt)):
        nrmse_pred.append(nrmse(gt[i], pred[i]))

    # plot nRMSE time series
    plt.plot(range(len(pred)),nrmse_pred,color='blue',
                 linewidth=2,label='4DVarNet')

    # graphical options
    plt.ylabel('nRMSE')
    plt.xlabel('Time (days)')
    plt.xticks(range(0,len(gt)),time,rotation=45, ha='right')
    plt.margins(x=0)
    plt.grid(True,alpha=.3)
    plt.legend(loc='upper left',prop=dict(size='small'),frameon=False,bbox_to_anchor=(0,1.02,1,0.2),ncol=2,mode="expand")
    plt.savefig(resfile,bbox_inches="tight")    # save the figure
    fig = plt.gcf()
    plt.close()                                 # close the figure
    return  fig

def plot_mse(gt, pred, resfile, time):
    '''
    gt: 3d numpy array (Ground Truth)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    time: 1d array-like of time corresponding to the experiment
    '''

    # Compute daily nRMSE scores
    mse_pred = []
    grad_mse_pred = []
    for i in range(len(gt)):
        mse_pred.append(mse(gt[i], pred[i]))
        grad_mse_pred.append(mse(gradient(gt[i],2), gradient(pred[i],2)))
    print("mse_pred = ", np.nanmean(mse_pred))
    print("grad_mse_pred = ", np.nanmean(grad_mse_pred))

    # plot nRMSE time series
    plt.plot(range(len(pred)),mse_pred,color='blue',
                 linewidth=2,label='4DVarNet')

    # graphical options
    plt.ylabel('MSE')
    plt.xlabel('Time (days)')
    plt.xticks(range(0,len(gt)),time,rotation=45, ha='right')
    plt.margins(x=0)
    plt.grid(True,alpha=.3)
    plt.legend(loc='upper left',prop=dict(size='small'),frameon=False,bbox_to_anchor=(0,1.02,1,0.2),ncol=2,mode="expand")
    plt.savefig(resfile,bbox_inches="tight")    # save the figure
    fig = plt.gcf()
    plt.close()                                 # close the figure
    return  fig


def plot(ax, lon, lat, data, title, cmap, norm, extent=[-65, -55, 30, 40], gridded=True, colorbar=True, orientation="horizontal", cartopy=True):
    if cartopy==True:
        ax.set_extent(list(extent))
        if gridded:
            im=ax.pcolormesh(lon, lat, data, cmap=cmap, \
                          norm=norm, edgecolors='face', alpha=1, \
                          transform=ccrs.PlateCarree(central_longitude=0.0))
        else:
            im=ax.scatter(lon, lat, c=data, cmap=cmap, s=1, \
                       norm=norm, edgecolors='face', alpha=1, \
                       transform=ccrs.PlateCarree(central_longitude=0.0))
    else:
        if gridded:
            im=ax.pcolormesh(lon, lat, data, cmap=cmap, \
                          norm=norm, edgecolors='face', alpha=1)
        else:
            im=ax.scatter(lon, lat, c=data, cmap=cmap, s=1, \
                       norm=norm, edgecolors='face', alpha=1)

    #  im.set_clim(vmin,vmax)
    if colorbar==True:
        clb = plt.colorbar(im, orientation=orientation, extend='both', pad=0.1, ax=ax)
    ax.set_title(title, pad=10, fontsize = 15)
    if cartopy==True:
        ax.add_feature(cfeature.LAND.with_scale('10m'), zorder=100,
                   edgecolor='k', facecolor='white')
        gl = ax.gridlines(alpha=0.5, zorder=200)#,draw_labels=True)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl = ax.gridlines(alpha=0.5, zorder=200)#,draw_labels=True)
        gl.xlabels_bottom = False
        gl.ylabels_right = False
        gl.xlabel_style = {'fontsize': 10, 'rotation' : 45}
        gl.ylabel_style = {'fontsize': 10}

def gradient(img, order):
    """ calculate x, y gradient and magnitude """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = sobelx/8.0
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = sobely/8.0
    sobel_norm = np.sqrt(sobelx*sobelx+sobely*sobely)
    if (order==0):
        return sobelx
    elif (order==1):
        return sobely
    else:
        return sobel_norm

def plot_maps(gt,obs,pred,lon,lat,resfile,grad=False, 
                 crop=None, cartopy=True, orthographic=True,supervised=True):

    if crop is not None:
        ilon = np.where((lon>=crop[0]) & (lon<=crop[1]))[0]
        ilat = np.where((lat>=crop[2]) & (lat<=crop[3]))[0]
        gt = (gt[:,ilat,:])[:,:,ilon]
        obs = (obs[:,ilat,:])[:,:,ilon]
        pred = (pred[:,ilat,:])[:,:,ilon]
        lon = lon[ilon]
        lat = lat[ilat]
    extent = [np.min(lon)-1,np.max(lon)+1,np.min(lat)-1,np.max(lat)+1]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    if orthographic:
        #crs = ccrs.Orthographic(central_lon,central_lat)
        crs = ccrs.Orthographic(-30,45)
    else:
        crs = ccrs.PlateCarree(central_longitude=0.0)

    if grad:
        vmax = np.nanmax(np.abs(gradient(gt, 2)))
        vmin = 0
        cm = plt.cm.viridis
        norm = colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
    else:
        vmax = np.nanmax(np.abs(gt))
        vmin = -1.*vmax
        cm = plt.cm.coolwarm
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]

    fig = plt.figure(figsize=(15,9))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    if supervised:
        ax1 = fig.add_subplot(gs[0, :2], projection=crs)
        ax2 = fig.add_subplot(gs[0, 2:], projection=crs)
        ax3 = fig.add_subplot(gs[1, 1:3], projection=crs)
        if grad:
            plot(ax1, lon, lat, gradient(gt, 2), r"$\nabla_{GT}$", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
            plot(ax2, lon, lat, np.where(np.isnan(obs), np.nan, 0.), "OBS (mask)", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
            plot(ax3, lon, lat, gradient(pred, 2), r"$\nabla_{4DVarNet}$", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        else:
            plot(ax1, lon, lat, gt, 'GT', extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
            plot(ax2, lon, lat, obs, 'OBS', extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
            plot(ax3, lon, lat, pred, '4DVarNet', extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.01])
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)
    plt.savefig(resfile)    # save the figure
    fig = plt.gcf()
    plt.close()             # close the figure
    return fig


def save_netcdf(saved_path1, gt, obs, pred, lon, lat, time,
                time_units='days since 2012-10-01 00:00:00'):
    '''
    saved_path1: string 
    pred: 3d numpy array (4DVarNet-based predictions)
    lon: 1d numpy array 
    lat: 1d numpy array
    time: 1d array-like of time corresponding to the experiment
    '''

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    dt = pred.shape[1]
    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time), \
                   'GT': (('time', 'lat', 'lon'), gt),
                   'OBS': (('time', 'lat', 'lon'), obs),
                   '4DVarNet': (('time', 'lat', 'lon'), pred)}, \
        coords={'lon': lon, 'lat': lat, 'time': np.arange(len(pred))})
    xrdata.time.attrs['units'] = time_units
    xrdata.to_netcdf(path=saved_path1, mode='w')

def save_NetCDF2(gt,obs,oi,pred,var_f,var_a,param,lon,lat,
                 figName,time,time_units='days since 2012-10-01 00:00:00'):

    extent_=[np.min(lon),np.max(lon),np.min(lat),np.max(lat)]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    # interpolation
    if len(param)==1:
        xrdata = xr.Dataset(\
                data_vars={'longitude': (('lat','lon'),mesh_lon),
                           'latitude' : (('lat','lon'),mesh_lat),
                           'Time'     : (('time'),time),
                           'gt'  : (('time','lat','lon','daw'),gt),
                           'obs'  : (('time','lat','lon','daw'),obs),
                           'pred'  : (('time','lat','lon','daw'),pred),
                           'var_f'   : (('time','lat','lon','daw'),var_f),
                           'var_a'   : (('time','lat','lon','daw'),var_a),
                           'oi'  : (('time','lat','lon','daw'),oi),
                           'H11': (('lat','lon'),param[0][0,0,0,:,:]),
                           'H12': (('lat','lon'),param[0][0,0,1,:,:]),
                           'H22': (('lat','lon'),param[0][0,1,1,:,:])},
                coords={'lon': lon,'lat': lat,
                        'time': range(len(time)), 'daw': range(gt.shape[3])})
    else:
        xrdata = xr.Dataset(\
                data_vars={'longitude': (('lat','lon'),mesh_lon),
                           'latitude' : (('lat','lon'),mesh_lat),
                           'Time'     : (('time'),time),
                           'gt'  : (('time','lat','lon','daw'),gt),
                           'obs'  : (('time','lat','lon','daw'),obs),
                           'pred'  : (('time','lat','lon','daw'),pred),
                           'var_f'   : (('time','lat','lon','daw'),var_f),
                           'var_a'   : (('time','lat','lon','daw'),var_a),
                           'oi'  : (('time','lat','lon','daw'),oi),
                           'kappa': (('time','lat','lon','daw'),param[0][:,0,:,:,:]),
                           'm1': (('time','lat','lon','daw'),param[1][:,0,:,:,:]),
                           'm2': (('time','lat','lon','daw'),param[1][:,1,:,:,:]),
                           'H11': (('time','lat','lon','daw'),param[2][:,0,0,:,:,:]),
                           'H12': (('time','lat','lon','daw'),param[2][:,0,1,:,:,:]),
                           'H22': (('time','lat','lon','daw'),param[2][:,1,1,:,:,:])},
                coords={'lon': lon,'lat': lat,
                        'time': range(len(time)), 'daw': range(gt.shape[3])})
    xrdata.time.attrs['units']='days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=figName, mode='w')

def save_loss(saved_path, loss):
    '''
    saved_path: string 
    loss: 3d numpy array (patch*ngrad*2)
    '''

    xrdata = xr.Dataset( \
            data_vars={'loss_mse': (('patch','iter'), loss[:,:,0]),
                       'loss_oi': (('patch','iter'), loss[:,:,1])},
            coords={'patch': np.arange(loss.shape[0]), 
                    'iter': np.arange(loss.shape[1])})
    xrdata.to_netcdf(path=saved_path, mode='w')

def save_grads(saved_path, grads):
    '''
    saved_path: string 
    loss: 6d numpy array (patch*2*nt*nx*ny*ngrad)
    '''

    xrdata = xr.Dataset( \
            data_vars={'grad': (('patch','iter','t','y','x'), grads[:,0,:,:,:,:]),
                'lstm_grad': (('patch','iter','t','y','x'), grads[:,1,:,:,:,:])},
            coords={'patch': np.arange(grads.shape[0]),
                    't': np.arange(grads.shape[2]),
                    'y': np.arange(grads.shape[3]),
                    'x': np.arange(grads.shape[4]),
                    'iter': np.arange(grads.shape[5])})
    xrdata.to_netcdf(path=saved_path, mode='w')

def nrmse(ref, pred):
    '''
    ref: Ground Truth fields
    pred: interpolated fields
    '''
    return np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)


def nrmse_scores(gt, pred, resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''
    # Compute daily nRMSE scores
    nrmse_pred = []
    for i in range(len(gt)):
        nrmse_pred.append(nrmse(gt[i], pred[i]))
    tab_scores = np.zeros((1, 3))
    tab_scores[0, 0] = np.nanmean(nrmse_pred)
    tab_scores[0, 1] = np.percentile(nrmse_pred, 5)
    tab_scores[0, 2] = np.percentile(nrmse_pred, 95)
    np.savetxt(fname=resfile, X=tab_scores, fmt='%2.2f')
    return tab_scores

def mse(ref, pred):
    '''
    ref: Ground Truth fields
    pred: interpolated fields
    '''
    return np.nanmean(((ref-np.nanmean(ref))-(pred-np.nanmean(pred)))**2)


def mse_scores(gt, pred, resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''
    # Compute daily nRMSE scores
    mse_pred = []
    for i in range(len(gt)):
        mse_pred.append(mse(gt[i], pred[i]))
    tab_scores = np.zeros((1, 3))
    tab_scores[0, 0] = np.nanmean(mse_pred)
    tab_scores[0, 1] = np.percentile(mse_pred, 5)
    tab_scores[0, 2] = np.percentile(mse_pred, 95)
    np.savetxt(fname=resfile, X=tab_scores, fmt='%2.2f')

def compute_metrics(x_test, x_rec):
    # MSE
    mse = np.mean((x_test - x_rec) ** 2)

    # MSE for gradient
    gx_rec = np.gradient(x_rec, axis=[1, 2])
    gx_rec = np.sqrt(gx_rec[0] ** 2 + gx_rec[1] ** 2)

    gx_test = np.gradient(x_test, axis=[1, 2])
    gx_test = np.sqrt(gx_test[0] ** 2 + gx_test[1] ** 2)

    gmse = np.mean((gx_test - gx_rec) ** 2)
    ng = np.mean((gx_rec) ** 2)

    return {'mse': mse, 'mseGrad': gmse, 'meanGrad': ng}


def get_psd_score(x_t, x, ref, with_fig=False):
    def psd_score(da: xr.DataArray) -> xr.DataArray:
        err = x_t - da
        psd_x_t = (
            x_t.copy()
                .pipe(
                lambda _da: xrft.isotropic_power_spectrum(_da, dim=['lat', 'lon'], window='hann', detrend='linear'))
                .mean(['time'])
        ).compute()

        psd_err = (
            err.copy()
                .pipe(
                lambda _da: xrft.isotropic_power_spectrum(_da, dim=['lat', 'lon'], window='hann', detrend='linear'))
                .mean(['time'])
        ).compute()

        psd_score = 1 - psd_err / psd_x_t
        return psd_score

    ref_score = psd_score(ref)
    model_score = psd_score(x)

    ref_score = ref_score.where(model_score > 0, drop=True).compute()
    model_score = model_score.where(model_score > 0, drop=True).compute()

    psd_plot_data: xr.DataArray = xr.DataArray(
        einops.rearrange([model_score.data, ref_score.data], 'var wl -> var wl'),
        name='PSD score',
        dims=('var', 'wl'),
        coords={
            'wl': ('wl', 20 * 5 * 1 / model_score.freq_r, {'long_name': 'Wavelength', 'units': 'km'}),
            'var': ('var', ['model', 'OI'], {}),
        },
    )

    spatial_resolution_model = (
        xr.DataArray(
            psd_plot_data.wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='model').data}
        ).interp(psd=0.5)
    )

    spatial_resolution_ref = (
        xr.DataArray(
            psd_plot_data.wl,
            dims=['psd'],
            coords={'psd': psd_plot_data.sel(var='OI').data}
        ).interp(psd=0.5)
    )

    if not with_fig:
        return spatial_resolution_model, spatial_resolution_ref

    fig, ax = plt.subplots()
    psd_plot_data.plot.line(x='wl', ax=ax)

    # Plot vertical line there
    for i, (sr, var) in enumerate([(spatial_resolution_ref, 'OI'), (spatial_resolution_model, 'model')]):
        plt.axvline(sr, ymin=0, color='0.5', ls=':')
        plt.annotate(f"resolution {var}: {float(sr):.2f} km", (sr * 1.1, 0.1 * i))
        plt.axhline(0.5, xmin=0, color='k', ls='--')
        plt.ylim([0, 1])

    plt.close()
    return fig, spatial_resolution_model, spatial_resolution_ref
