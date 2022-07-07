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
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import cv2
import xrft
import logging
from dask.diagnostics import ProgressBar
from matplotlib.ticker import ScalarFormatter
import hvplot
import hvplot.xarray
from scipy import interpolate

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
        clb = plt.colorbar(im, orientation=orientation, pad=0.1, ax=ax)
        clb.set_label(title,fontsize = 25)
        clb.ax.tick_params(labelsize = 25)

    ax.add_feature(cfeature.LAND.with_scale('10m'), zorder=100,
                   edgecolor='k', facecolor='white')
    gl = ax.gridlines(alpha=0.5, zorder=200,draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_bottom = False
    gl.ylabels_right = False
    gl.xlabel_style = {'fontsize': 10, 'rotation' : 45}
    gl.ylabel_style = {'fontsize': 10}

def plot_maps(ds,lon,lat,grad, 
              orthographic, methods, figsize):

    extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])
 
    fig = plt.figure(figsize=figsize)

    if orthographic:
        #crs = ccrs.Orthographic(central_lon,central_lat)
        crs = ccrs.Orthographic(-30,45)
    else:
        crs = ccrs.PlateCarree(central_longitude=0.0)

    if grad:
        vmax = np.nanmax(np.abs(gradient(ds[1].values, 2)))
        vmin = 0
        cm = plt.cm.viridis
        norm = colors.PowerNorm(gamma=0.7, vmin=vmin, vmax=vmax)
    else:
        vmax = np.nanmax(np.abs(ds[1]))
        vmin = -1.*vmax
        cm = plt.cm.coolwarm
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
       
    vmax_raw = np.nanmax(np.abs(ds[1]))
    vmin_raw = -1.*vmax
    cm_raw = plt.cm.coolwarm
    norm_raw = colors.Normalize(vmin=vmin_raw, vmax=vmax_raw)

    obs = ds[0]
    obs = np.where(np.isnan(obs),np.nan,obs)

    nr = int(np.ceil((len(ds)-1)/2))+1
    nc = 4    
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.5)
   
    ax = fig.add_subplot(gs[0,1:3], projection=crs)
    plot(ax,lon,lat,ds[0].values,'Obs',extent=extent, cmap=cm_raw, norm=norm_raw, colorbar=False)
    
    for i in np.arange(1,len(ds)):
        ir = int(np.floor((i-1)/2.)+1)
        ic = np.mod(i-1,2)
        ax = fig.add_subplot(gs[ir,(ic*2):(ic*2+2)], projection=crs)
        if grad:
            plot(ax,lon,lat,gradient(ds[i].values,2),r"$\nabla_{"+methods[i]+"$",extent=extent,cmap=cm,norm=norm,colorbar=False)
        else:
            plot(ax,lon,lat,ds[i].values,methods[i], extent=extent, cmap=cm, norm=norm, colorbar=False)
            
    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    cbar_ax.tick_params(labelsize=20) 
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)
    plt.show()    # save the figure
    fig = plt.gcf()
    return fig

def animate_maps(oi, pred, lon, lat,
                 crop=None, orthographic=True, grad=False):


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
        ax1.clear()
        ax2.clear()
        if grad==False:
            plot(ax1, lon, lat, oi[i], 'OI', extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax2, lon, lat, pred[i], '4DVarNet', extent=extent, cmap=cm, norm=norm, colorbar=False)
        else:
            plot(ax1, lon, lat, gradient(oi[i], 2), r"$\nabla_{OI}$", extent=extent, cmap=cm, norm=norm, colorbar=False)
            plot(ax2, lon, lat, gradient(pred[i], 2), r"$\nabla_{4DVarNet}$", extent=extent, cmap=cm, norm=norm, colorbar=False)

    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.5)
    if orthographic:
        #crs = ccrs.Orthographic(central_lon,central_lat)
        crs = ccrs.Orthographic(-30,45)
    else:
        crs = ccrs.PlateCarree(central_longitude=central_lon)
    ax1 = fig.add_subplot(gs[0, 0], projection=crs)
    ax1.set_extent(extent)
    ax2 = fig.add_subplot(gs[0, 1], projection=crs)
    ax2.set_extent(extent)

    #plt.subplots_adjust(hspace=0.05)
    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)

    ani = animation.FuncAnimation(fig, animate, frames=len(oi), interval=200, repeat=False)
    writergif = animation.PillowWriter(fps=10)
    writer = animation.FFMpegWriter(fps=10)
    ani.save(resfile, writer = writer)
    plt.close()

def maps_score(ds, lon, lat, methods, figsize):
    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]

    fig = plt.figure(figsize=figsize)
    crs = ccrs.Orthographic(-30,45)
    nr = int(np.ceil(len(ds)/3))
    nc = int(np.min((3,len(ds))))    
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.5)

    vmax = max([ds[i].max() for i in range(len(ds)) ])
    vmin = 0.
    cmap_rmse = plt.cm.viridis
    norm_rmse = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(len(ds)):
        ir = int(np.floor(i/3.))
        ic = np.mod(i,3)
        ax = fig.add_subplot(gs[ir,ic], projection=crs)
        ax.set_title(methods[i],fontsize=25)
        plot(ax,lon,lat,ds[i].values,'',extent=extent,cmap=cmap_rmse,norm=norm_rmse,colorbar=False)
    plt.subplots_adjust(hspace=0.05)
    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    cbar_ax.tick_params(labelsize=20) 
    sm = plt.cm.ScalarMappable(cmap=cmap_rmse, norm=norm_rmse)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)

    plt.show()
    fig = plt.gcf()
    return fig
    

def plot_temporal_statistics(filenames, methods, colors):
    
    ds1 = [xr.open_dataset(file, group='diff') for file in filenames]
    ds2 = [xr.open_dataset(file, group='alongtrack') for file in filenames] 
    rmse_score = [ 1. - ds1[i]['rms']/ds2[i]['rms'] for i in range(len(filenames)) ]
    rmse_score = [ rmse_score[i].dropna(dim='time').where(ds1[i]['count'] > 10, drop=True) for i in range(len(filenames)) ]
  
    rmse_score=xr.merge([rmse_score[i].to_dataset().rename({"rms":"rms_"+methods[i]}) for i in range(len(filenames)) ])
    print(rmse_score)
    plot1 = rmse_score.hvplot.line(x='time',y=['rms_'+methods[i] for  i in range(len(filenames)) ],ylabel='RMSE SCORE', shared_axes=True, color=colors[:len(filenames)])
    plot1.opts(legend_position='bottom',legend_cols=False)
    plot2 = ds1[0]['count'].dropna(dim='time').hvplot.step(ylabel='#Obs.', shared_axes=True, color='grey')
    figure = (plot1+plot2).cols(1) 
    hvplot.show(figure)
    
def find_wavelength_05_crossing(filename):
    
    ds = xr.open_dataset(filename)
    y = 1./ds.wavenumber
    x = (1. - ds.psd_diff/ds.psd_ref)
    f = interpolate.interp1d(x, y)
    
    xnew = 0.5
    ynew = f(xnew)
    
    return ynew
    
  
def plot_psd_score_intercomparison(filenames, methods, colors):
    
    ds = [xr.open_dataset(file) for file in filenames]
    ds = [dds.isel(wavenumber=(1. - dds.psd_diff/dds.psd_ref)>0) for dds in ds]
    resolved_scales = [find_wavelength_05_crossing(file) for file in filenames]
        
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    ax.invert_xaxis()
    plt.plot((1./ds[0].wavenumber), ds[0].psd_ref, label='reference', color='k')
    for i in range(len(filenames)):
        plt.plot((1./ds[i].wavenumber), ds[i].psd_study, label='reconstruction_'+methods[i], color=colors[i])
    plt.xlabel('wavelength [km]')
    plt.ylabel('Power Spectral Density [m$^{2}$/cy/km]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.grid(which='both')
    
    ax = plt.subplot(122)
    ax.invert_xaxis()    
    for i in range(len(filenames)):
        # score = (1. - ds[i].psd_diff/ds[i].psd_ref)
        # plt.plot(*score.isel(wavenumber=score>0).pipe(lambda da: (1/da.wavenumber, da)) , color=colors[i], lw=2)
        plt.plot((1./ds[i].wavenumber), (1. - ds[i].psd_diff/ds[i].psd_ref), color=colors[i], lw=2)
    plt.xlabel('wavelength [km]')
    plt.ylabel('PSD Score [1. - PSD$_{err}$/PSD$_{ref}$]')
    plt.xscale('log')
    plt.hlines(y=0.5, 
              xmin=np.ma.min(np.ma.masked_invalid(1./ds[0].wavenumber)), 
              xmax=np.ma.max(np.ma.masked_invalid(1./ds[0].wavenumber)),
              color='r',
              lw=0.5,
              ls='--')
    imax = np.argmin(resolved_scales)
    plt.vlines(x=resolved_scales[i], ymin=0, ymax=1, lw=0.5, color=colors[imax])
    ax.fill_betweenx((1. - ds[imax].psd_diff/ds[imax].psd_ref), 
                     resolved_scales[imax], 
                     np.ma.max(np.ma.masked_invalid(1./ds[imax].wavenumber)),
                     color=colors[imax],
                     alpha=0.3, 
                     label=f'Best resolved scales \n $\lambda$ > {int(resolved_scales[imax])}km')
    plt.legend(loc='best')
    plt.grid(which='both')
    
    logging.info(' ')
    logging.info(f'  Minimum spatial scale resolved = {int(resolved_scales[imax])}km')
    
    plt.show()
    
    return resolved_scales
    

