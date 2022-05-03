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
import cv2
import xrft
import logging
from dask.diagnostics import ProgressBar
from matplotlib.ticker import ScalarFormatter


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

def rmse_based_scores(ds,var_pred):
    
    logging.info('     Compute RMSE-based scores...')
    
    # RMSE(t) based score
    rmse_t = 1.0 - (((ds[var_pred] - ds['gt'])**2).mean(dim=('lon', 'lat')))**0.5/(((ds['gt'])**2).mean(dim=('lon', 'lat')))**0.5
    # RMSE(x, y) based score
    rmse_xy = (((ds[var_pred] - ds['gt'])**2).mean(dim=('time')))**0.5
    
    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')

    # Temporal stability of the error
    reconstruction_error_stability_metric = rmse_t.std().values

    # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
    leaderboard_rmse = 1.0 - (((ds[var_pred] - ds['gt']) ** 2).mean()) ** 0.5 / (
        ((ds['gt']) ** 2).mean()) ** 0.5

    logging.info('          => Leaderboard RMSE score = %s', np.round(leaderboard_rmse.values, 2))
    logging.info('          Error variability = %s (temporal stability of the mapping error)', np.round(reconstruction_error_stability_metric, 2))
    
    return rmse_t, rmse_xy, np.round(leaderboard_rmse.values, 2), np.round(reconstruction_error_stability_metric, 2)

def psd_based_scores(ds,var_pred):
    
    logging.info('     Compute PSD-based scores...')
    
    with ProgressBar():
        
        # Compute error = SSH_reconstruction - SSH_true
        err = (ds[var_pred] - ds['gt'])
        err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
        # make time vector in days units 
        err['time'] = (err.time - err.time[0]) / np.timedelta64(1, 'D')
        
        # Rechunk SSH_true
        signal = ds['gt'].chunk({"lat":1, 'time': ds['time'].size, 'lon': ds['lon'].size})
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

        logging.info('          => Leaderboard Spectral score = %s (degree (lon))',
                     np.round(shortest_spatial_wavelength_resolved, 2))
        logging.info('          => shortest temporal wavelength resolved = %s (time (days))',
                     np.round(shortest_temporal_wavelength_resolved, 2))

        return (1.0 - mean_psd_err/mean_psd_signal), np.round(shortest_spatial_wavelength_resolved, 2), np.round(shortest_temporal_wavelength_resolved, 2)

def plot_psd_score(ds_psd):
        
    try:
        nb_experiment = len(ds_psd.experiment)
    except:
        nb_experiment = 1

    fig, ax0 =  plt.subplots(1, 2+(nb_experiment-2), sharey=True, figsize=(20+(10*(nb_experiment-2)), 5), squeeze=True)
    #plt.subplots_adjust(right=0.1, left=0.09)
    for exp in range(nb_experiment):
        try:
            ctitle = ds_psd.experiment.values[exp]
        except:
            ctitle = ''

        if nb_experiment > 1:
            ax = ax0[exp]
            data = (ds_psd.isel(experiment=exp).values)
        else:
            ax = ax0
            data = (ds_psd.values)
        ax.invert_yaxis()
        ax.invert_xaxis()
        c1 = ax.contourf(1./(ds_psd.freq_lon), 1./ds_psd.freq_time, data,
                          levels=np.arange(0,1.1, 0.1), cmap='inferno', extend='both')
        ax.set_xlabel('spatial wavelength (degree lon)', fontweight='bold', fontsize=18)
        ax.set_ylabel('temporal wavelength (time days)', fontweight='bold', fontsize=18)
        #plt.xscale('log')
        ax.set_yscale('log')
        ax.grid(linestyle='--', lw=1, color='w')
        ax.tick_params(axis='both', labelsize=18)
        ax.set_title(f'{ctitle}', fontweight='bold', fontsize=18)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        c2 = ax.contour(1./(ds_psd.freq_lon), 1./ds_psd.freq_time, data, levels=[0.5], linewidths=2, colors='k')
        
        cbar = fig.colorbar(c1, ax=ax, pad=0.01)
        cbar.add_lines(c2)

    bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
    if nb_experiment > 1:
        ax0[-1].annotate('Resolved scales',
                    xy=(1.2, 0.8),
                    xycoords='axes fraction',
                    xytext=(1.2, 0.55),
                    bbox=bbox_props,
                    arrowprops=
                        dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left',
                        verticalalignment='center')

        ax0[-1].annotate('UN-resolved scales',
                    xy=(1.2, 0.2),
                    xycoords='axes fraction',
                    xytext=(1.2, 0.45),
                    bbox=bbox_props,
                    arrowprops=
                    dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left',
                        verticalalignment='center')
    else:
        ax.annotate('Resolved scales',
                    xy=(1.2, 0.8),
                    xycoords='axes fraction',
                    xytext=(1.2, 0.55),
                    bbox=bbox_props,
                    arrowprops=
                        dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left',
                        verticalalignment='center')

        ax.annotate('UN-resolved scales',
                    xy=(1.2, 0.2),
                    xycoords='axes fraction',
                    xytext=(1.2, 0.45),
                    bbox=bbox_props,
                    arrowprops=
                    dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left',
                        verticalalignment='center')

    plt.show()    
    plt.close()

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


def plot_maps(gt,obs,preds,labels,lon,lat,new_method,grad=False, 
              orthographic=True,cartopy=True, figsize=(20,20)):

    extent = [np.min(lon)-1,np.max(lon)+1,np.min(lat)-1,np.max(lat)+1]
    central_lon = np.mean(extent[:2])
    central_lat = np.mean(extent[2:])

    if orthographic:
        crs = ccrs.Orthographic(central_lon,central_lat)
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

    obs = np.where(np.isnan(gt),np.nan,obs)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.1,hspace=0.)

    ax1 = fig.add_subplot(gs[0, :2], projection=crs)
    ax2 = fig.add_subplot(gs[0, 2:], projection=crs)
    ax3 = fig.add_subplot(gs[1, :2], projection=crs)
    ax4 = fig.add_subplot(gs[1, 2:], projection=crs)
    if grad:
        plot(ax1, lon, lat, gradient(gt, 2), r"$\nabla_{GT}$", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        plot(ax2, lon, lat, gradient(obs, 2), r"$\nabla_{OBS}$", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        plot(ax3, lon, lat, gradient(preds[0], 2), r"$\nabla_{Baseline OI}$", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        plot(ax4, lon, lat, gradient(preds[1], 2), r"$\nabla_{"+new_method+"}$", extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
    else:
        plot(ax1, lon, lat, gt, 'GT', extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        plot(ax2, lon, lat, obs, 'OBS', extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        plot(ax3, lon, lat, preds[0], 'Baseline OI', extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)
        plot(ax4, lon, lat, preds[1], new_method, extent=extent, cmap=cm, norm=norm, colorbar=False, cartopy=cartopy)

    # Colorbar
    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    cbar_ax.tick_params(labelsize=20) 
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)
    plt.show()    
    fig = plt.gcf()
    plt.close()             # close the figure
    return fig

def TS(list_data,labels_data,colors,symbols,lstyle,lwidth,lday,gradient=False):

    N   = len(list_data[0])
    GT  = list_data[0]
    ## Compute spatial coverage  
    if any("Obs"==s for s in labels_data):
        spatial_coverage = []
        id_obs = np.where(labels_data=="Obs")[0][0]
        OBS = list_data[id_obs]
        for j in range(N):
            spatial_coverage.append(100*len(np.argwhere(np.isfinite(OBS[j].flatten())))/len(OBS[j].flatten()))
        
    # Compute nRMSE time series
    nRMSE = []
    id1=np.where(["GT" not in l for l in labels_data ])[0]
    id2=np.where(["Obs" not in l for l in labels_data ])[0]
    id_plot=np.intersect1d(id1,id2)
    for i in range(len(labels_data[id_plot])):
        nRMSE_=[]
        meth_i=list_data[id_plot[i]]
        for j in range(N):
            if gradient == False:
                nRMSE_.append((np.sqrt(np.nanmean(((GT[j]-np.nanmean(GT[j]))-(meth_i[j]-np.nanmean(meth_i[j])))**2)))/np.nanstd(GT[j]))
            else:
                nRMSE_.append((np.sqrt(np.nanmean(((Gradient(GT[j],2)-np.nanmean(Gradient(GT[j],2)))-(Gradient(meth_i[j],2)-np.nanmean(Gradient(meth_i[j],2))))**2)))/np.nanstd(Gradient(GT[j],2)))
        nRMSE.append(nRMSE_)
  
    # plot nRMSE time series
    for i in range(len(labels_data[id_plot])):
        if gradient == False:
            plt.plot(range(N),nRMSE[i],linestyle=lstyle[id_plot[i]],color=colors[id_plot[i]],linewidth=lwidth[id_plot[i]],label=labels_data[id_plot[i]])
        else:
            plt.plot(range(N),nRMSE[i],linestyle=lstyle[id_plot[i]],color=colors[id_plot[i]],linewidth=lwidth[id_plot[i]],label=r"$\nabla_{"+str.split(labels_data[id_plot[i]])[0]+"}$ "+str.split(labels_data[id_plot[i]])[1])


    # graphical options
    plt.ylim(0,np.ceil(np.max(nRMSE)*100)/100)
    plt.ylabel('nRMSE')
    plt.xlabel('Time (days)')
    plt.xticks([0,5,10,15,20],\
           [lday[0],lday[5],lday[10],lday[15],lday[20]],\
           rotation=45, ha='right')
    plt.margins(x=0)
    plt.grid(True,alpha=.3)
    plt.legend(loc='upper left',prop=dict(size='small'),frameon=False,bbox_to_anchor=(0,1.02,1,0.2),ncol=2,mode="expand")
    # second axis with spatial coverage
    axes2 = plt.twinx()
    width=0.75
    p1 = axes2.bar(range(N), spatial_coverage, width,color='r',alpha=0.25)
    axes2.set_ylim(0, 100)
    axes2.set_ylabel('Spatial Coverage (%)')
    axes2.margins(x=0)
    plt.show()    
    plt.close()                                 # close the figure


