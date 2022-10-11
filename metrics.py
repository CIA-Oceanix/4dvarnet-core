import logging
from matplotlib.ticker import ScalarFormatter
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

from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cv2
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import xrft

from spectral import *

def plot_snr(gt,oi,pred,resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''

    dt = pred.shape[1]

    # Compute Signal-to-Noise ratio
    f, pf = avg_err_rapsd2dv1(oi,gt,4.,True)
    wf = 1./f
    snr_oi = [wf, pf]
    f, pf = avg_err_rapsd2dv1(pred,gt,4.,True)
    wf = 1./f
    snr_pred = [wf, pf]

    # plot Signal-to-Noise ratio
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(snr_oi[0],snr_oi[1],color='red',linewidth=2,label='OI')
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


def plot_nrmse(gt, oi, pred, resfile, time):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    time: 1d array-like of time corresponding to the experiment
    '''

    # Compute daily nRMSE scores
    nrmse_oi = []
    nrmse_pred = []
    for i in range(len(oi)):
        nrmse_oi.append(nrmse(gt[i], oi[i]))
        nrmse_pred.append(nrmse(gt[i], pred[i]))

    # plot nRMSE time series
    plt.plot(range(len(oi)),nrmse_oi,color='red',
                 linewidth=2,label='OI')
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

def plot_mse(gt, oi, pred, resfile, time):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    time: 1d array-like of time corresponding to the experiment
    '''

    # Compute daily nRMSE scores
    mse_oi = []
    mse_pred = []
    grad_mse_oi = []
    grad_mse_pred = []
    for i in range(len(oi)):
        mse_oi.append(mse(gt[i], oi[i]))
        mse_pred.append(mse(gt[i], pred[i]))
        grad_mse_oi.append(mse(gradient(gt[i],2), gradient(oi[i],2)))
        grad_mse_pred.append(mse(gradient(gt[i],2), gradient(pred[i],2)))
    print("mse_oi = ", np.nanmean(mse_oi))
    print("mse_pred = ", np.nanmean(mse_pred))
    print("grad_mse_oi = ", np.nanmean(grad_mse_oi))
    print("grad_mse_pred = ", np.nanmean(grad_mse_pred))
    print("percentage_ssh = ", np.abs(np.nanmean(mse_oi)-np.nanmean(mse_pred))/np.nanmean(mse_oi))
    print("percentage_ssh_grad = ", np.abs(np.nanmean(grad_mse_oi)-np.nanmean(grad_mse_pred))/np.nanmean(grad_mse_oi))

    # plot nRMSE time series
    plt.plot(range(len(oi)),mse_oi,color='red',
                 linewidth=2,label='OI')
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

def plot(ax, lon, lat, data, title, cmap, norm, extent=[-65, -55, 30, 40], gridded=True, colorbar=True, orientation="horizontal"):
    ax.set_extent(list(extent))
    if gridded:
        im=ax.pcolormesh(lon, lat, data, cmap=cmap, \
                          norm=norm, edgecolors='face', alpha=1, \
                          transform=ccrs.PlateCarree(central_longitude=0.0))
    else:
        im=ax.scatter(lon, lat, c=data, cmap=cmap, s=1, \
                       norm=norm, edgecolors='face', alpha=1, \
                       transform=ccrs.PlateCarree(central_longitude=0.0))
    #  im.set_clim(vmin,vmax)
    if colorbar==True:
        clb = plt.colorbar(im, orientation=orientation, extend='both', pad=0.1, ax=ax)
    ax.set_title(title, pad=10, fontsize = 15)
    ax.add_feature(cfeature.LAND.with_scale('10m'), zorder=100,
                   edgecolor='k', facecolor='white')
    try:
        gl = ax.gridlines(alpha=0.5, zorder=200)#,draw_labels=True)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.bottom_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'fontsize': 10, 'rotation' : 45}
        gl.ylabel_style = {'fontsize': 10}
    except Exception as e:
        import traceback
        print(traceback.format_exc()) 
        

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

def plot_maps(gt,obs,oi,pred,lon,lat,resfile,grad=False,
                 crop=None, orthographic=True,supervised=True):

    if crop is not None:
        ilon = np.where((lon>=crop[0]) & (lon<=crop[1]))[0]
        ilat = np.where((lat>=crop[2]) & (lat<=crop[3]))[0]
        gt = (gt[:,ilat,:])[:,:,ilon]
        obs = (obs[:,ilat,:])[:,:,ilon]
        oi = (oi[:,ilat,:])[:,:,ilon]
        pred = (pred[:,ilat,:])[:,:,ilon]
        lon = lon[ilon]
        lat = lat[ilat]
    extent = [np.min(lon)-1,np.max(lon)+1,np.min(lat)-1,np.max(lat)+1]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    if orthographic:
        crs = ccrs.Orthographic(central_lon,central_lat)
        #crs = ccrs.Orthographic(-30,45)
    else:
        crs = ccrs.PlateCarree(central_longitude=central_lon)

    if grad:
        vmax = np.nanmax(np.abs(gradient(oi, 2)))
        vmin = 0
        cm = plt.cm.viridis
        norm = colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
    else:
        vmax = np.nanmax(np.abs(oi))
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
        ax3 = fig.add_subplot(gs[1, :2], projection=crs)
        ax4 = fig.add_subplot(gs[1, 2:], projection=crs)
        if grad:
            plot(ax1, lon, lat, gradient(gt, 2), r"$\nabla_{GT}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax2, lon, lat, np.where(np.isnan(obs), np.nan, 0.), "OBS (mask)", extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax3, lon, lat, gradient(oi, 2), r"$\nabla_{OI}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax4, lon, lat, gradient(pred, 2), r"$\nabla_{4DVarNet}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
        else:
            plot(ax1, lon, lat, gt, 'GT', extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax2, lon, lat, obs, 'OBS', extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax3, lon, lat, oi, 'OI', extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax4, lon, lat, pred, '4DVarNet', extent=extent, cmap=cm, norm=norm, colorbar=False)
    else:
        ax1 = fig.add_subplot(gs[0, 1:3], projection=crs)
        ax2 = fig.add_subplot(gs[1, :2], projection=crs)
        ax3 = fig.add_subplot(gs[1, 2:], projection=crs)
        if grad:
            plot(ax1, lon, lat, np.where(np.isnan(obs), np.nan, 0.), "OBS (mask)", extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax2, lon, lat, gradient(oi, 2), r"$\nabla_{OI}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax3, lon, lat, gradient(pred, 2), r"$\nabla_{4DVarNet}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
        else:
            plot(ax1, lon, lat, obs, 'OBS', extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax2, lon, lat, oi, 'OI', extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax3, lon, lat, pred, '4DVarNet', extent=extent, cmap=cm, norm=norm, colorbar=False)

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.01])
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)
    plt.savefig(resfile)    # save the figure
    fig = plt.gcf()
    plt.close()             # close the figure
    return fig

def animate_maps(gt, obs, oi, pred, lon, lat, resfile,
                 crop=None, orthographic=True, dw=4, grad=False, supervised=True):

    if dw>1:
        # decrease the resolution
        Nlon = len(lon)
        Nlat = len(lat)
        ilon = np.arange(0,Nlon,dw)
        ilat = np.arange(0,Nlat,dw)
        gt = (gt[:,ilat,:])[:,:,ilon]
        obs = (obs[:,ilat,:])[:,:,ilon]
        oi = (oi[:,ilat,:])[:,:,ilon]
        pred = (pred[:,ilat,:])[:,:,ilon]
        lon = lon[ilon]
        lat = lat[ilat]

    if crop is not None:
        ilon = np.where((lon>=crop[0]) & (lon<=crop[1]))[0]
        ilat = np.where((lat>=crop[2]) & (lat<=crop[3]))[0]
        gt = (gt[:,ilat,:])[:,:,ilon]
        obs = (obs[:,ilat,:])[:,:,ilon]
        oi = (oi[:,ilat,:])[:,:,ilon]
        pred = (pred[:,ilat,:])[:,:,ilon]
        lon = lon[ilon]
        lat = lat[ilat]
    extent = [np.min(lon)-1,np.max(lon)+1,np.min(lat)-1,np.max(lat)+1]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    if grad:
        vmax = np.nanmax([np.nanmax(np.abs(gradient(oi[i], 2))) for i in range(len(oi))])
        vmin = 0
        cm = plt.cm.viridis
        norm = colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
    else:
        vmax = np.nanmax(np.abs(oi))
        vmin = -1.*vmax
        cm = plt.cm.coolwarm
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    def animate(i):
        print(i)
        if supervised:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            if grad==False:
                plot(ax1, lon, lat, gt[i], 'GT', extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax2, lon, lat, obs[i], 'OBS', extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax3, lon, lat, oi[i], 'OI', extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax4, lon, lat, pred[i], '4DVarNet', extent=extent, cmap=cm, norm=norm, colorbar=False)
            else:
                plot(ax1, lon, lat, gradient(gt[i], 2), r"$\nabla_{GT}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax2, lon, lat, np.where(np.isnan(obs[i]), np.nan, 0.), "OBS (mask)", extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax3, lon, lat, gradient(oi[i], 2), r"$\nabla_{OI}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax4, lon, lat, gradient(pred[i], 2), r"$\nabla_{4DVarNet}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
        else:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            if grad==False:
                plot(ax1, lon, lat, obs[i], 'OBS', extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax2, lon, lat, oi[i], 'OI', extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax3, lon, lat, pred[i], '4DVarNet', extent=extent, cmap=cm, norm=norm, colorbar=False)
            else:
                #plot(ax1, lon, lat, gradient(obs[i], 2), r"$\nabla_{OBS}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax1, lon, lat, np.where(np.isnan(obs[i]), np.nan, 0.), "OBS (mask)", extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax2, lon, lat, gradient(oi[i], 2), r"$\nabla_{OI}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
                plot(ax3, lon, lat, gradient(pred[i], 2), r"$\nabla_{4DVarNet}$", extent=extent, cmap=cm, norm=norm, colorbar=False)

    fig = plt.figure(figsize=(15,9))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.1)
    if orthographic:
        crs = ccrs.Orthographic(central_lon,central_lat)
        #crs = ccrs.Orthographic(-30,45)
    else:
        crs = ccrs.PlateCarree(central_longitude=central_lon)
    if supervised:
        ax1 = fig.add_subplot(gs[0, :2], projection=crs)
        ax2 = fig.add_subplot(gs[0, 2:], projection=crs)
        ax3 = fig.add_subplot(gs[1, :2], projection=crs)
        ax4 = fig.add_subplot(gs[1, 2:], projection=crs)
    else:
        ax1 = fig.add_subplot(gs[0, 1:3], projection=crs)
        ax1.set_extent(extent)
        ax2 = fig.add_subplot(gs[1, :2], projection=crs)
        ax2.set_extent(extent)
        ax3 = fig.add_subplot(gs[1, 2:], projection=crs)
        ax3.set_extent(extent)

    plt.subplots_adjust(hspace=0.05)
    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)

    ani = animation.FuncAnimation(fig, animate, frames=len(gt), interval=200, repeat=False)
    writergif = animation.PillowWriter(fps=3)
    writer = animation.FFMpegWriter(fps=3)
    ani.save(resfile, writer = writer)
    plt.close()

def plot_ensemble(pred,lon,lat,resfile,crop=None,
                   orthographic=True):

    vmax = np.nanmax(np.abs(pred))
    vmin = -1.*vmax
    cm = plt.cm.coolwarm
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    grad_vmax = np.nanmax(np.abs(gradient(pred,2)))
    grad_vmin = 0
    cm_grad = plt.cm.viridis
    norm_grad = colors.PowerNorm(gamma=0.7, vmin=grad_vmin, vmax=grad_vmax)

    if crop is not None:
        ilon = np.where((lon>=crop[0]) & (lon<=crop[1]))[0]
        ilat = np.where((lat>=crop[2]) & (lat<=crop[3]))[0]
        pred = (pred[ilat,:,:])[:,ilon,:]
        lon = lon[ilon]
        lat = lat[ilat]
    extent = [np.min(lon)-1,np.max(lon)+1,np.min(lat)-1,np.max(lat)+1]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    if orthographic:
        crs = ccrs.Orthographic(central_lon,central_lat)
        #crs = ccrs.Orthographic(-30,45)
    else:
        crs = ccrs.PlateCarree(central_longitude=central_lon)

    n_members = pred.shape[-1]
    fig, ax = plt.subplots(2,n_members,figsize=(5*n_members,15),squeeze=False,
                          subplot_kw=dict(projection=crs))
    for i in range(n_members):
        plot(ax[0,i],lon,lat,pred[:,:,i],'M'+str(i),extent=extent,cmap=cm,norm=norm,colorbar=False)
        plot(ax[1,i],lon,lat,gradient(pred[:,:,i],2),r"$\nabla_{M"+str(i)+"}$",extent=extent,
                                        cmap=cm_grad,norm=norm_grad, colorbar=False)
    plt.savefig(resfile)       # save the figure
    plt.close()                # close the figure

def maps_score(resfile, ds, lon, lat):
    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T
    cor_map_oi = xr.corr(ds['GT'],ds['OI'], dim='Time')
    rmse_map_oi = (((ds['GT'] - ds['OI'])**2).mean(dim=('Time')))**0.5
    cor_map_pred = xr.corr(ds['GT'],ds['4DVarNet'], dim='Time')
    rmse_map_pred = (((ds['GT'] - ds['4DVarNet'])**2).mean(dim=('Time')))**0.5

    lat = ds.latitude.values
    lon = ds.longitude.values
    extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(221,projection=ccrs.PlateCarree(central_longitude=central_lon))
    ax2 = fig.add_subplot(222,projection=ccrs.PlateCarree(central_longitude=central_lon))
    ax3 = fig.add_subplot(223,projection=ccrs.PlateCarree(central_longitude=central_lon))
    ax4 = fig.add_subplot(224,projection=ccrs.PlateCarree(central_longitude=central_lon))

    vmax = cor_map_oi.max()
    vmin = cor_map_oi.min()
    cmap_cor = plt.cm.coolwarm
    norm_cor = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    plot(ax1,lon,lat,cor_map_oi.values,'Correlation OI',extent=extent,cmap=cmap_cor,norm=norm_cor)
    plot(ax2,lon,lat,cor_map_pred.values,'Correlation 4DVarNet',extent=extent,cmap=cmap_cor,norm=norm_cor)
    vmax = rmse_map_oi.max()
    vmin = 0.
    cmap_rmse = plt.cm.viridis
    norm_rmse = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    plot(ax3,lon,lat,rmse_map_oi.values,'RMSE OI',extent=extent,cmap=cmap_rmse,norm=norm_rmse)
    plot(ax4,lon,lat,rmse_map_pred.values,'RMSE 4DVarNet',extent=extent,cmap=cmap_rmse,norm=norm_rmse)

    plt.savefig(resfile)
    fig = plt.gcf()
    plt.close()
    return fig

def save_netcdf(saved_path1, gt, oi, pred, lon, lat, time,
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
                   'OI': (('time', 'lat', 'lon'), oi),
                   '4DVarNet': (('time', 'lat', 'lon'), pred)}, \
        coords={'lon': lon, 'lat': lat, 'time': np.arange(len(pred))})
    xrdata.time.attrs['units'] = time_units
    xrdata.to_netcdf(path=saved_path1, mode='w')

def save_netcdf_with_sst(saved_path1, gt, obs, oi, pred, sst_feat, lon, lat, time,
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

    time = np.arange(gt.shape[0])
    dt = pred.shape[1]
    
    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time), \
                   'ssh_gt': (('time', 'lat', 'lon'), gt),
                   'ssh_obs': (('time', 'lat', 'lon'), obs),
                   'ssh_oi': (('time', 'lat', 'lon'), oi),
                   'ssh_rec': (('time', 'lat', 'lon'), pred),
                   'sst_feat': (('time', 'feat', 'lat', 'lon'), sst_feat)}, \
        coords={'lon': lon, 'lat': lat, 'time': np.arange(len(pred)),'feat':np.arange(sst_feat.shape[1])})

    #xrdata = xr.Dataset( \
    #    data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
    #               'latitude': (('lat', 'lon'), mesh_lat), \
                   #'Time': (('time'), time), \
                   #'ssh_gt': (('time', 'lat', 'lon'), gt), \
                   #'ssh_oi': (('time', 'lat', 'lon'), oi), \
                   #'ssh_obs': (('time', 'lat', 'lon'), obs), \
    #               'ssh_rec': (('time', 'lat', 'lon'), pred)})#, \
    #              'sst_feat': (('time', 'feat', 'lat', 'lon'), sst_feat)})#, \
    #    coords={'lon': lon, 'lat': lat, 'time': time,'feat':np.arange(sst_feat.shape[1])})
    
    #xrdata.time.attrs['units'] = time_units
    xrdata.to_netcdf(path=saved_path1, mode='w')
    print('... file saved',flush=True)

def save_netcdf_uv(saved_path1, gt, u_gt, v_gt, obs, oi, pred, lon, lat, time, 
                   u_pred=None, v_pred=None, sst_feat=None,
                   curl_gt = None, strain_gt = None,
                   curl_pred = None, strain_pred = None,
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

    #time = np.arange(gt.shape[0])
    #dt = pred.shape[1]
    #print(time)
    
    if  sst_feat is None :
        
        xrdata = xr.Dataset( \
            data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                       'latitude': (('lat', 'lon'), mesh_lat), \
                       'Time': (('time'), time), \
                       'ssh_gt': (('time', 'lat', 'lon'), gt),
                       'u_gt': (('time', 'lat', 'lon'), u_gt),
                       'v_gt': (('time', 'lat', 'lon'), v_gt),
                       'ssh_obs': (('time', 'lat', 'lon'), obs),
                       'ssh_oi': (('time', 'lat', 'lon'), oi),
                       'ssh_rec': (('time', 'lat', 'lon'), pred),
                       'u_rec': (('time', 'lat', 'lon'), u_pred),
                       'v_rec': (('time', 'lat', 'lon'), v_pred)}, \
            coords={'lon': lon, 'lat': lat, 'time': time})
    else:        
        xrdata = xr.Dataset( \
                data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                           'latitude': (('lat', 'lon'), mesh_lat), \
                           'Time': (('time'), time), \
                           'ssh_gt': (('time', 'lat', 'lon'), gt),
                           'u_gt': (('time', 'lat', 'lon'), u_gt),
                           'v_gt': (('time', 'lat', 'lon'), v_gt),
                           'ssh_obs': (('time', 'lat', 'lon'), obs),
                           'ssh_oi': (('time', 'lat', 'lon'), oi),
                           'ssh_rec': (('time', 'lat', 'lon'), pred),
                           'u_rec': (('time', 'lat', 'lon'), u_pred),
                           'v_rec': (('time', 'lat', 'lon'), v_pred),
                           'sst_feat': (('time', 'feat', 'lat', 'lon'), sst_feat)}, \
                coords={'lon': lon, 'lat': lat, 'time': time,'feat':np.arange(sst_feat.shape[1])})

        xrdata['sst_feat'] = (['time', 'feat', 'lat', 'lon'],  sst_feat)        
        
    if curl_gt is not None :
        xrdata["curl_gt"]=(['time', 'lat', 'lon'],  curl_gt)

    if strain_gt is not None :
        xrdata["strain_gt"]=(['time', 'lat', 'lon'],  strain_gt)

    if curl_pred is not None :
        xrdata["curl_rec"]=(['time', 'lat', 'lon'],  curl_pred)

    if strain_pred is not None :
        xrdata["strain_rec"]=(['time', 'lat', 'lon'],  strain_pred)

    #xrdata.time.attrs['units'] = time_units
    xrdata.to_netcdf(path=saved_path1, mode='w')
    print('... file saved',flush=True)
def save_netcdf_with_obs(saved_path1, gt, obs, oi, pred, lon, lat, time,
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

    time = np.arange(gt.shape[0])
    #dt = pred.shape[1]
    #delta_days = np.zeros(len(time),)
    #ii = 0
    #for _time in time:        
    #    delta = datetime.date(_time[:24],'%Y-%m-%dT%H:%M:%S.%f').date()  - datetime.date(2012, 10, 1)
    #    delta_days[ii] = delta.days
    #    ii += 1
    
    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time), \
                   'ssh_gt': (('time', 'lat', 'lon'), gt),
                   'ssh_obs': (('time', 'lat', 'lon'), obs),
                   'ssh_oi': (('time', 'lat', 'lon'), oi),
                   'ssh_rec': (('time', 'lat', 'lon'), pred)}, \
        coords={'lon': lon, 'lat': lat, 'time': time})

    
    #print(time)
    #xrdata.time.attrs['units'] = time_units
    xrdata.to_netcdf(path=saved_path1, mode='w')
    print('... file saved',flush=True)

def nrmse(ref, pred):
    '''
    ref: Ground Truth fields
    pred: interpolated fields
    '''
    return np.sqrt(np.nanmean(((ref - np.nanmean(ref)) - (pred - np.nanmean(pred))) ** 2)) / np.nanstd(ref)


def nrmse_scores(gt, oi, pred, resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''
    # Compute daily nRMSE scores
    nrmse_oi = []
    nrmse_pred = []
    for i in range(len(oi)):
        nrmse_oi.append(nrmse(gt[i], oi[i]))
        nrmse_pred.append(nrmse(gt[i], pred[i]))
    tab_scores = np.zeros((2, 3))
    tab_scores[0, 0] = np.nanmean(nrmse_oi)
    tab_scores[0, 1] = np.percentile(nrmse_oi, 5)
    tab_scores[0, 2] = np.percentile(nrmse_oi, 95)
    tab_scores[1, 0] = np.nanmean(nrmse_pred)
    tab_scores[1, 1] = np.percentile(nrmse_pred, 5)
    tab_scores[1, 2] = np.percentile(nrmse_pred, 95)
    np.savetxt(fname=resfile, X=tab_scores, fmt='%2.2f')
    return tab_scores

def mse(ref, pred):
    '''
    ref: Ground Truth fields
    pred: interpolated fields
    '''
    return np.nanmean(((ref-np.nanmean(ref))-(pred-np.nanmean(pred)))**2)


def mse_scores(gt, oi, pred, resfile):
    '''
    gt: 3d numpy array (Ground Truth)
    oi: 3d numpy array (OI)
    pred: 3d numpy array (4DVarNet-based predictions)
    resfile: string
    '''
    # Compute daily nRMSE scores
    mse_oi = []
    mse_pred = []
    for i in range(len(oi)):
        mse_oi.append(mse(gt[i], oi[i]))
        mse_pred.append(mse(gt[i], pred[i]))
    tab_scores = np.zeros((2, 3))
    tab_scores[0, 0] = np.nanmean(mse_oi)
    tab_scores[0, 1] = np.percentile(mse_oi, 5)
    tab_scores[0, 2] = np.percentile(mse_oi, 95)
    tab_scores[1, 0] = np.nanmean(mse_pred)
    tab_scores[1, 1] = np.percentile(mse_pred, 5)
    tab_scores[1, 2] = np.percentile(mse_pred, 95)
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

from scipy.ndimage import gaussian_filter

def compute_laplacian(x):
    
    if len( x.shape ) == 2 :
        lap = x[1:-1,1:-1] 
        lap = lap - 0.25 * x[1:-1,0:x.shape[1]-2]
        lap = lap - 0.25 * x[1:-1,2:x.shape[1]]
        lap = lap - 0.25 * x[0:x.shape[0]-2,1:-1]
        lap = lap - 0.25 * x[2:x.shape[0],1:-1]
    else:
        lap = x[:,1:-1,1:-1] 
        lap = lap - 0.25 * x[:,1:-1,0:x.shape[2]-2]
        lap = lap - 0.25 * x[:,1:-1,2:x.shape[2]]
        lap = lap - 0.25 * x[:,0:x.shape[1]-2,1:-1]
        lap = lap - 0.25 * x[:,2:x.shape[1],1:-1]

    
    return lap

def compute_laplacian_metrics(x_ref,x,sig_lap=1):

    lap_ref = compute_laplacian( gaussian_filter(x_ref, sigma=sig_lap))
    lap_rec = compute_laplacian( gaussian_filter(x, sigma=sig_lap))

    mse_lap = np.mean((lap_ref-lap_rec)**2)
    var_lap = np.var(lap_ref)
    
    R2 = np.corrcoef(lap_ref.ravel(), lap_rec.ravel())[0,1]
    
    return {'mse':mse_lap,'var_lap': var_lap,'r_square': R2}


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

    #print(ref.shape)
    #print(x.shape)

    ref_score = psd_score(ref)
    model_score = psd_score(x)

    ref_score = ref_score.where(model_score > 0, drop=True).compute()
    model_score = model_score.where(model_score > 0, drop=True).compute()

    psd_plot_data: xr.DataArray = xr.DataArray(
        einops.rearrange([model_score.data, ref_score.data], 'var wl -> var wl'),
        name='PSD score',
        dims=('var', 'wl'),
        coords={
            'wl': ('wl', 20 * 5 * 1 / model_score.freq_r.data, {'long_name': 'Wavelength', 'units': 'km'}),
            'var': ('var', ['model', 'OI'], {}),
        },
    )

    try:
        #print('here')
        #print(psd_plot_data.wl.data)
        spatial_resolution_model = (
            xr.DataArray(
                psd_plot_data.wl.data,
                dims=['psd'],
                coords={'psd': psd_plot_data.sel(var='model').data}
            ).interp(psd=0.5)
        ).data
    except (KeyError, ValueError) as e:
        spatial_resolution_model = -1

    try:
        spatial_resolution_ref = (
            xr.DataArray(
                psd_plot_data.wl.data,
                dims=['psd'],
                coords={'psd': psd_plot_data.sel(var='OI').data}
            ).interp(psd=0.5)
        ).data
    except (KeyError, ValueError) as e:
        spatial_resolution_ref = -1

    if not with_fig:
        return spatial_resolution_model, spatial_resolution_ref

    fig, ax = plt.subplots()
    if spatial_resolution_model == -1:
        plt.close()
        return fig, spatial_resolution_model, spatial_resolution_ref

    psd_plot_data.plot.line(x='wl', ax=ax)

    # Plot vertical line there
    for i, (sr, var) in enumerate([(spatial_resolution_ref, 'OI'), (spatial_resolution_model, 'model')]):
        plt.axvline(sr, ymin=0, color='0.5', ls=':')
        plt.annotate(f"resolution {var}: {float(sr):.2f} km", (sr * 1.1, 0.1 * i))
        plt.axhline(0.5, xmin=0, color='k', ls='--')
        plt.ylim([0, 1])

    plt.close()
    return fig, spatial_resolution_model, spatial_resolution_ref


def rmse_based_scores(da_rec, da_ref):
    # boost swot rmse score
    #logging.info('     Compute RMSE-based scores...')

    # RMSE(t) based score
    rmse_t = 1.0 - (((da_rec - da_ref)**2).mean(dim=('lon', 'lat')))**0.5/(((da_ref)**2).mean(dim=('lon', 'lat')))**0.5
    # RMSE(x, y) based score
    # rmse_xy = 1.0 - (((da_rec - da_ref)**2).mean(dim=('time')))**0.5/(((da_ref)**2).mean(dim=('time')))**0.5
    rmse_xy = (((da_rec - da_ref)**2).mean(dim=('time')))**0.5

    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')

    # Temporal stability of the error
    reconstruction_error_stability_metric = rmse_t.std().values

    # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
    leaderboard_rmse = 1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (
        ((da_ref) ** 2).mean()) ** 0.5

    logging.info('          => Leaderboard SSH RMSE score = %s', np.round(leaderboard_rmse.values, 2))
    logging.info('          Error variability = %s (temporal stability of the mapping error)', np.round(reconstruction_error_stability_metric, 2))

    return rmse_t, rmse_xy, np.round(leaderboard_rmse.values, 5), np.round(reconstruction_error_stability_metric, 5)


def psd_based_scores(da_rec, da_ref):
    # boost-swot-psd-score
    #logging.info('     Compute PSD-based scores...')

    # Compute error = SSH_reconstruction - SSH_true
    err = (da_rec - da_ref)
    err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
    # make time vector in days units
    err['time'] = (err.time - err.time[0]) / np.timedelta64(1, 'D')

    # Rechunk SSH_true
    signal = da_ref.chunk({"lat":1, 'time': da_ref['time'].size, 'lon': da_ref['lon'].size})
    # make time vector in days units
    signal['time'] = (signal.time - signal.time[0]) / np.timedelta64(1, 'D')

    # Compute PSD_err and PSD_signal
    psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='constant', window=True).compute()
    psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='constant', window=True).compute()

    # Averaged over latitude
    mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
    mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)

    # return PSD-based score
    psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

    # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score

    level = [0.5]
    cs = plt.contour(1./psd_based_score.freq_lon.values,1./psd_based_score.freq_time.values, psd_based_score, level)
        
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)

    logging.info('          => Leaderboard Spectral score = %s (degree lon)',
                 np.round(shortest_spatial_wavelength_resolved, 2))
    logging.info('          => shortest temporal wavelength resolved = %s (days)',
                 np.round(shortest_temporal_wavelength_resolved, 2))
    psd_da = (1.0 - mean_psd_err/mean_psd_signal)
    psd_da.name = 'psd_score'
    return psd_da.to_dataset(), np.round(shortest_spatial_wavelength_resolved, 3), np.round(shortest_temporal_wavelength_resolved, 3)


def plot_psd_score(ds):
    fig, ax = plt.subplots()
    #ax.invert_yaxis()
    #ax.invert_xaxis()
    c1 = plt.contourf(1./(ds['freq_lon']), 1./ds['freq_time'], ds['psd_score'],
                      levels=np.arange(0,1.1, 0.1), cmap='RdYlGn', extend='both')
    cbar = plt.colorbar(pad=0.01)
    plt.xlabel('spatial wavelenght (degree_lon)', fontweight='bold', fontsize=20)
    plt.ylabel('temporal wavelenght (days)', fontweight='bold', fontsize=20)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.grid(linestyle='--', lw=1, color='w')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('PSD-based score', fontweight='bold', fontsize=20)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    c2 = plt.contour(1./(ds['freq_lon']), 1./ds['freq_time'], ds['psd_score'], levels=[0.5], linewidths=2, colors='k')
    cbar.add_lines(c2)

    bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
    ax.annotate('Resolved scales',
            xy=(1.15, 0.8),
            xycoords='axes fraction',
            xytext=(1.15, 0.55),
            bbox=bbox_props,
            arrowprops=
                dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='center')

    ax.annotate('UN-resolved scales',
            xy=(1.15, 0.2),
            xycoords='axes fraction',
            xytext=(1.15, 0.45),
            bbox=bbox_props,
            arrowprops=
                dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='center')
    plt.close()
    return fig
