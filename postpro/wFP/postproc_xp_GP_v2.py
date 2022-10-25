import numpy as np
import xarray as xr
import matplotlib
import os
import sys

matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

root="/gpfswork/rech/yrf/uba22to/4dvarnet-core"

def find_open_ncdf(path):
    path1 = root+"/dashboard/"+path+"/lightning_logs"
    latest_subdir = max([os.path.join(path1,d) for d in os.listdir(path1)], key=os.path.getmtime)
    print(latest_subdir)
    data = xr.open_dataset(latest_subdir+"/maps.nc")
    return data
    
def find_open_ncdf_loss(path):
    path1 = root + "/dashboard/"+path+"/lightning_logs"
    latest_subdir = max([os.path.join(path1,d) for d in os.listdir(path1)], key=os.path.getmtime)
    print(latest_subdir)
    data = xr.open_dataset(latest_subdir+"/loss.nc")
    return data
    
def find_open_ncdf_grads(path):
    path1 = root + "/dashboard/"+path+"/lightning_logs"
    latest_subdir = max([os.path.join(path1,d) for d in os.listdir(path1)], key=os.path.getmtime)
    print(latest_subdir)
    data = xr.open_dataset(latest_subdir+"/grads.nc")
    return data

def plot(ax, lon, lat, data, title, cmap, norm, extent=[-65, -55, 30, 40], colorbar=True, orientation="horizontal", extend="both",fs=12):
    im=ax.pcolormesh(lon, lat, data, cmap=cmap,
                          norm=norm, edgecolors='face', alpha=1)
    if colorbar==True:
        clb = plt.colorbar(im, orientation=orientation, pad=0.05, ax=ax,extend=extend)
        clb.ax.tick_params(labelsize = 18)
    ax.set_title(title,fontsize = fs)
 
gp_dataset = sys.argv[1]

# plots
for ipatch in range(19):   
    gt = find_open_ncdf("oi_gp_ref_"+gp_dataset).gt.isel(time=ipatch,daw=2)
    # REF
    oi_analytical = find_open_ncdf("oi_gp_ref_"+gp_dataset).pred.isel(time=ipatch,daw=2)
    oi_gd = find_open_ncdf("oi_gp_ref_gd_"+gp_dataset).pred.isel(time=ipatch,daw=2)
    # UNET
    unet_mse_loss = find_open_ncdf("oi_gp_unet_mse_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    unet_oi_loss = find_open_ncdf("oi_gp_unet_oi_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    unet_mseoi_loss = find_open_ncdf("oi_gp_unet_mseoi_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    # FP
    #fourdvarnet_fp_phi_cov = find_open_ncdf("oi_gp_spde_fp_wolp_"+gp_dataset).pred.isel(time=ipatch,daw=2)
    fourdvarnet_fp_phi_unet_mse_loss = find_open_ncdf("oi_gp_4dvarnet_fp_mse_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    fourdvarnet_fp_phi_unet_oi_loss = find_open_ncdf("oi_gp_4dvarnet_fp_oi_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    fourdvarnet_fp_phi_unet_mseoi_loss = find_open_ncdf("oi_gp_4dvarnet_fp_mseoi_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    # GradLSTM
    fourdvarnet_lstm_phi_cov_mse_loss = find_open_ncdf("oi_gp_spde_wolp_mse_loss_"+gp_dataset).pred.isel(time=ipatch,daw=2)
    fourdvarnet_lstm_phi_cov_oi_loss = find_open_ncdf("oi_gp_spde_wolp_oi_loss_"+gp_dataset).pred.isel(time=ipatch,daw=2)
    fourdvarnet_lstm_phi_cov_mseoi_loss = find_open_ncdf("oi_gp_spde_wolp_mseoi_loss_"+gp_dataset).pred.isel(time=ipatch,daw=2)
    fourdvarnet_lstm_phi_unet_mse_loss = find_open_ncdf("oi_gp_4dvarnet_mse_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    fourdvarnet_lstm_phi_unet_oi_loss = find_open_ncdf("oi_gp_4dvarnet_oi_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)
    fourdvarnet_lstm_phi_unet_mseoi_loss = find_open_ncdf("oi_gp_4dvarnet_mseoi_loss_"+gp_dataset)['4DVarNet'].isel(time=ipatch)

    fig = plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(7,4)
    gs.update(wspace=0.5,hspace=0.5)
    ax1 = fig.add_subplot(gs[:3, 1:3])
    # No params
    ax2 = fig.add_subplot(gs[3, 0])
    ax3 = fig.add_subplot(gs[3, 1])
    ax4 = fig.add_subplot(gs[3, 2])
    ax5 = fig.add_subplot(gs[3, 3])
    # MSE loss
    ax6 = fig.add_subplot(gs[4, 0])
    ax7 = fig.add_subplot(gs[4, 1])
    ax8 = fig.add_subplot(gs[4, 2])
    ax9 = fig.add_subplot(gs[4, 3])
    # MSE OI loss
    ax10 = fig.add_subplot(gs[5, 0])
    ax11 = fig.add_subplot(gs[5, 1])
    ax12 = fig.add_subplot(gs[5, 2])
    ax13 = fig.add_subplot(gs[5, 3])
    # OI loss
    ax14 = fig.add_subplot(gs[6, 0])
    ax15 = fig.add_subplot(gs[6, 1])
    ax16 = fig.add_subplot(gs[6, 2])
    ax17 = fig.add_subplot(gs[6, 3])

    extent = [np.min(gt.lon.values),np.max(gt.lon.values),np.min(gt.lat.values),np.max(gt.lat.values)]
    vmax = -2
    vmin = 2
    cmap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    im1 = plot(ax1,gt.lon,gt.lat,gt.values,'Ground Truth',extent=extent,cmap=cmap,norm=norm,colorbar=True,fs=18)
    # No parameters
    im2 = plot(ax2,gt.lon,gt.lat,oi_analytical.values,'OI (analytical)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im3 = plot(ax3,gt.lon,gt.lat,oi_gd.values,'OI (gradient descent) N=10000',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    #im4 = plot(ax4,gt.lon,gt.lat,fourdvarnet_fp_phi_cov.values,'4DVarNet-FP5-Phi-cov',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    ax4.set_visible(False)
    ax5.set_visible(False)
    # MSE loss
    im6 = plot(ax6,gt.lon,gt.lat,unet_mse_loss.values,'UNet (mse loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im7 = plot(ax7,gt.lon,gt.lat,fourdvarnet_fp_phi_unet_mse_loss.values,'4DVarNet-FP5-Phi-UNet (mse loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im8 = plot(ax8,gt.lon,gt.lat,fourdvarnet_lstm_phi_cov_mse_loss.values,'4DVarNet-LSTM-Phi-cov (mse loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im9 = plot(ax9,gt.lon,gt.lat,fourdvarnet_lstm_phi_unet_mse_loss.values,'4DVarNet-LSTM-Phi-UNet (mse loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    # MSE OI loss
    im10 = plot(ax10,gt.lon,gt.lat,unet_mseoi_loss.values,'UNet (mse oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im11 = plot(ax11,gt.lon,gt.lat,fourdvarnet_fp_phi_unet_mseoi_loss.values,'4DVarNet-FP5-Phi-UNet (mse oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im12 = plot(ax12,gt.lon,gt.lat,fourdvarnet_lstm_phi_cov_mseoi_loss.values,'4DVarNet-LSTM-Phi-cov (mse oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im13 = plot(ax13,gt.lon,gt.lat,fourdvarnet_lstm_phi_unet_mseoi_loss.values,'4DVarNet-LSTM-Phi-UNet (mse oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    # OI loss
    im14 = plot(ax14,gt.lon,gt.lat,unet_oi_loss.values,'UNet (OI loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im15 = plot(ax15,gt.lon,gt.lat,fourdvarnet_fp_phi_unet_oi_loss.values,'4DVarNet-FP5-Phi-UNet (oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im16 = plot(ax16,gt.lon,gt.lat,fourdvarnet_lstm_phi_cov_oi_loss.values,'4DVarNet-LSTM-Phi-cov (oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im17 = plot(ax17,gt.lon,gt.lat,fourdvarnet_lstm_phi_unet_oi_loss.values,'4DVarNet-LSTM-Phi-UNet (oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    plt.savefig('4DVarNet_xp_GP_'+gp_dataset+'_'+str(ipatch)+'_v2.png')
    fig = plt.gcf()
    plt.close() 

    # plot MSE loss vs OI loss 
    oi_loss_oi = find_open_ncdf_loss("oi_gp_ref_"+gp_dataset).loss_oi.isel(patch=ipatch)
    oi_loss_mse = find_open_ncdf_loss("oi_gp_ref_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # OI (gradient descent)
    oi_gd_loss_oi = find_open_ncdf_loss("oi_gp_ref_gd_"+gp_dataset).loss_oi.isel(patch=ipatch)
    oi_gd_loss_mse = find_open_ncdf_loss("oi_gp_ref_gd_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # UNet (mse_loss)
    unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_unet_mse_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_unet_mse_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet-FP (mse_loss)
    fourdvarnet_fp_phi_unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_fp_mse_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_fp_phi_unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_fp_mse_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet (mse_loss)
    fourdvarnet_lstm_phi_unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_mse_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_mse_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet (mse_loss)
    fourdvarnet_lstm_phi_cov_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_spde_wolp_mse_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_cov_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_spde_wolp_mse_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # UNet (mseoi_loss)
    unet_mseoi_loss_loss_oi = find_open_ncdf_loss("oi_gp_unet_mseoi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    unet_mseoi_loss_loss_mse = find_open_ncdf_loss("oi_gp_unet_mseoi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet-FP (mseoi_loss)
    fourdvarnet_fp_phi_unet_mseoi_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_fp_mseoi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_fp_phi_unet_mseoi_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_fp_mseoi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet (mseoi_loss)
    fourdvarnet_lstm_phi_unet_mseoi_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_mseoi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_unet_mseoi_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_mseoi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet (mseoi_loss)
    fourdvarnet_lstm_phi_cov_mseoi_loss_loss_oi = find_open_ncdf_loss("oi_gp_spde_wolp_mseoi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_cov_mseoi_loss_loss_mse = find_open_ncdf_loss("oi_gp_spde_wolp_mseoi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # UNet (oi_loss)
    unet_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_unet_oi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    unet_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_unet_oi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet-FP (oi_loss)
    fourdvarnet_fp_phi_unet_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_fp_oi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_fp_phi_unet_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_fp_oi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet (oi_loss)
    fourdvarnet_lstm_phi_unet_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_oi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_unet_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_oi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)
    # 4DVarNet (oi_loss)
    fourdvarnet_lstm_phi_cov_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_spde_wolp_oi_loss_"+gp_dataset).loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_cov_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_spde_wolp_oi_loss_"+gp_dataset).loss_mse.isel(patch=ipatch)

    fig = plt.figure(figsize=(10,7))
    min_oi = min(np.concatenate((oi_gd_loss_oi.values,
                 fourdvarnet_lstm_phi_unet_mse_loss_loss_oi.values,
                 fourdvarnet_lstm_phi_cov_oi_loss_loss_oi.values)))
    p1 = plt.scatter(min_oi, oi_loss_mse, c='darkgoldenrod',edgecolors='black', label='OI (analytical)', marker="*", s=300, zorder=3)
    #p1 = plt.scatter(oi_loss_oi, oi_loss_mse, c='darkgoldenrod',edgecolors='black', label='OI (analytical)', marker="*", s=300, zorder=3)
    p2 = plt.plot(oi_gd_loss_oi, oi_gd_loss_mse, 's--', c='seagreen', alpha=0.5, label='OI (gradient descent)')
    l = plt.plot(0,0,color="w",label="")

    p3 = plt.plot(unet_mse_loss_loss_oi, unet_mse_loss_loss_mse, '-o', c='slategray', alpha=0.5, label='UNet (mse loss)')
    p4 = plt.plot(unet_oi_loss_loss_oi, unet_oi_loss_loss_mse, 's--', c='slategray', alpha=0.5, label='UNet (oi loss)')
    p5 = plt.plot(unet_oi_loss_loss_oi, unet_oi_loss_loss_mse, '-^', c='slategray', alpha=0.5, label='UNet (mseoi loss)')
    
    p6 = plt.plot(fourdvarnet_fp_phi_unet_mse_loss_loss_oi, fourdvarnet_fp_phi_unet_mse_loss_loss_mse, '-o', c='darkorange', alpha=0.5,label='Prior: UNet / Solver: FP10 (mse loss)')
    p7 = plt.plot(fourdvarnet_fp_phi_unet_oi_loss_loss_oi, fourdvarnet_fp_phi_unet_oi_loss_loss_mse, 's--', c='darkorange', alpha=0.5,label='Prior: UNet / Solver: FP10 (oi loss)')
    p8 = plt.plot(fourdvarnet_fp_phi_unet_mseoi_loss_loss_oi, fourdvarnet_fp_phi_unet_mseoi_loss_loss_mse, '-^', c='darkorange', alpha=0.5,label='Prior: UNet / Solver: FP10 (mseoi loss)')
    
    p9 = plt.plot(fourdvarnet_lstm_phi_unet_mse_loss_loss_oi, fourdvarnet_lstm_phi_unet_mse_loss_loss_mse, '-o', c='darkslateblue', alpha=0.5,label='Prior: UNet / Solver: LSTM (20 updates) (mse loss)')
    p10 = plt.plot(fourdvarnet_lstm_phi_unet_oi_loss_loss_oi, fourdvarnet_lstm_phi_unet_oi_loss_loss_mse, 's--', c='darkslateblue', alpha=0.5,label='Prior: UNet / Solver: LSTM (100 updates) (oi loss)')
    p11 = plt.plot(fourdvarnet_lstm_phi_unet_mseoi_loss_loss_oi, fourdvarnet_lstm_phi_unet_mseoi_loss_loss_mse, '-^', c='darkslateblue', alpha=0.5,label='Prior: UNet / Solver: LSTM (100 updates) (mseoi loss)')
    
    p12 = plt.plot(fourdvarnet_lstm_phi_cov_mse_loss_loss_oi, fourdvarnet_lstm_phi_cov_mse_loss_loss_mse, '-o', c='firebrick', alpha=0.5,label='Prior: Covariance / Solver: LSTM (20 updates) (mse loss)')
    p13 = plt.plot(fourdvarnet_lstm_phi_cov_oi_loss_loss_oi, fourdvarnet_lstm_phi_cov_oi_loss_loss_mse, 's--', c='firebrick', alpha=0.5,label='Prior: Covariance / Solver: LSTM (100 updates) (oi loss)')
    p14 = plt.plot(fourdvarnet_lstm_phi_cov_mseoi_loss_loss_oi, fourdvarnet_lstm_phi_cov_mseoi_loss_loss_mse, '-^', c='firebrick', alpha=0.5,label='Prior: Covariance / Solver: LSTM (100 updates) (mseoi loss)')
    
    ymin = min(np.concatenate((oi_loss_mse.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_mse.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_mse.values)))
    ymax = max(np.concatenate((oi_loss_mse.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_mse.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_mse.values)))
    xmin = min(np.concatenate((oi_loss_oi.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_oi.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_oi.values)))
    xmax = max(np.concatenate((oi_loss_oi.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_oi.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_oi.values)))
    plt.xlim(xmin-(xmax-xmin)/10,xmax+(xmax-xmin)/10)
    plt.ylim(ymin-(ymax-ymin)/10,ymax+(ymax-ymin)/10)
    plt.xlabel('OI variational cost')
    plt.ylabel('MSE vs true state')
    plt.vlines(min_oi,ymin=0,ymax=10000,linestyle='solid',color='black')
    #plt.vlines(oi_loss_oi,ymin=0,ymax=10000,linestyle='solid',color='black')
    plt.grid()
    plt.gca().invert_xaxis()
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(1,l[0])
    labels.insert(1,'')
    order = [14,2,5,8,11, 0,3,6,9,12, 1,4,7,10,13]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
              loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, prop={'size':6})
    plt.savefig('4DVarNet_xp_GP_'+gp_dataset+'_loss_v2_'+str(ipatch)+'.png', bbox_inches='tight')
    fig = plt.gcf()
    plt.close() 

    # plot OI loss vs iterations
    fig = plt.figure(figsize=(10,7))
    p1 = plt.scatter(np.arange(len(oi_loss_mse)), min_oi, c='darkgoldenrod',edgecolors='black', label='OI (analytical)', marker="*", s=300, zorder=3)
    #p1 = plt.scatter(np.arange(len(oi_loss_mse)), oi_loss_oi, c='darkgoldenrod',edgecolors='black', label='OI (analytical)', marker="*", s=300, zorder=3)
    p2 = plt.plot(np.arange(len(oi_gd_loss_mse)), oi_gd_loss_oi, 's--', c='seagreen', alpha=0.5, label='OI (gradient descent)')
    l = plt.plot(0,0,color="w",label="")

    p3 = plt.plot(np.arange(len(unet_mse_loss_loss_mse)), unet_mse_loss_loss_oi, '-o', c='slategray', alpha=0.5, label='UNet (mse loss)')
    p4 = plt.plot(np.arange(len(unet_oi_loss_loss_mse)), unet_oi_loss_loss_oi, 's--', c='slategray', alpha=0.5, label='UNet (oi loss)')
    p5 = plt.plot(np.arange(len(unet_mseoi_loss_loss_mse)), unet_mseoi_loss_loss_oi, '-^', c='slategray', alpha=0.5, label='UNet (oi loss)')
    
    p6 = plt.plot(np.arange(len(fourdvarnet_fp_phi_unet_mse_loss_loss_oi)), fourdvarnet_fp_phi_unet_mse_loss_loss_oi, '-o', c='darkorange', alpha=0.5,label='Prior: UNet / Solver: FP10 (mse loss)')
    p7 = plt.plot(np.arange(len(fourdvarnet_fp_phi_unet_oi_loss_loss_oi)), fourdvarnet_fp_phi_unet_oi_loss_loss_oi, 's--', c='darkorange', alpha=0.5,label='Prior: UNet / Solver: FP10 (oi loss)')
    p8 = plt.plot(np.arange(len(fourdvarnet_fp_phi_unet_mseoi_loss_loss_oi)), fourdvarnet_fp_phi_unet_mseoi_loss_loss_oi, '-^', c='darkorange', alpha=0.5,label='Prior: UNet / Solver: FP10 (oi loss)')
    
    p9 = plt.plot(np.arange(len(fourdvarnet_lstm_phi_unet_mse_loss_loss_oi)), fourdvarnet_lstm_phi_unet_mse_loss_loss_oi, '-o', c='darkslateblue', alpha=0.5,label='Prior: UNet / Solver: LSTM (20 updates) (mse loss)')
    p10 = plt.plot(np.arange(len(fourdvarnet_lstm_phi_unet_oi_loss_loss_oi)), fourdvarnet_lstm_phi_unet_oi_loss_loss_oi, 's--', c='darkslateblue', alpha=0.5,label='Prior: UNet / Solver: LSTM (100 updates) (oi loss)')
    p11 = plt.plot(np.arange(len(fourdvarnet_lstm_phi_unet_mseoi_loss_loss_oi)), fourdvarnet_lstm_phi_unet_mseoi_loss_loss_oi, '-^', c='darkslateblue', alpha=0.5,label='Prior: UNet / Solver: LSTM (100 updates) (oi loss)')
    
    p12 = plt.plot(np.arange(len(fourdvarnet_lstm_phi_cov_mse_loss_loss_oi)), fourdvarnet_lstm_phi_cov_mse_loss_loss_oi, '-o', c='firebrick', alpha=0.5,label='Prior: Covariance / Solver: LSTM (20 updates) (mse loss)')
    p13 = plt.plot(np.arange(len(fourdvarnet_lstm_phi_cov_oi_loss_loss_oi)), fourdvarnet_lstm_phi_cov_oi_loss_loss_oi, 's--', c='firebrick', alpha=0.5,label='Prior: Covariance / Solver: LSTM (100 updates) (oi loss)')
    p14 = plt.plot(np.arange(len(fourdvarnet_lstm_phi_cov_mseoi_loss_loss_oi)), fourdvarnet_lstm_phi_cov_mseoi_loss_loss_oi, '-^', c='firebrick', alpha=0.5,label='Prior: Covariance / Solver: LSTM (100 updates) (oi loss)') 
    
    plt.hlines(min_oi,xmin=0,xmax=10000,linestyle='solid',color='black')
    #plt.hlines(oi_loss_oi,xmin=0,xmax=10000,linestyle='solid',color='black')    
    plt.xlim(0,10000)
    plt.ylim(xmin-(xmax-xmin)/10,xmax+(xmax-xmin)/10)
    plt.xlabel('Number of solver iterations')
    plt.ylabel('OI variational cost')
    plt.xscale('symlog')
    plt.grid()
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.insert(1,l[0])
    labels.insert(1,'')
    print(handles)
    print(labels)
    order = [14,2,5,8,11, 0,3,6,9,12, 1,4,7,10,13]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
              loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, prop={'size':6})
    plt.savefig('4DVarNet_xp_GP_'+gp_dataset+'_oi_vs_iter_v2_'+str(ipatch)+'.png', bbox_inches='tight')
    fig = plt.gcf()
    plt.close()

    # Grad vs GradLSTM
    grad = find_open_ncdf_grads("oi_gp_4dvarnet_mse_loss_test_grad_lstm_"+gp_dataset).grad.isel(patch=ipatch)
    lstm_grad = find_open_ncdf_grads("oi_gp_4dvarnet_mse_loss_test_grad_lstm_"+gp_dataset).lstm_grad.isel(patch=ipatch)
    fig, ax = plt.subplots(figsize=(20,20))
    ymin = min(np.concatenate((grad.mean(dim=["t","x","y"]),lstm_grad.mean(dim=["t","x","y"]))))
    ymax = max(np.concatenate((grad.mean(dim=["t","x","y"]),lstm_grad.mean(dim=["t","x","y"]))))
    ax2 = ax.twinx()
    ax.plot(np.arange(grad.shape[3]),grad.mean(dim=["t","x","y"]), '-o', c='green', alpha=0.5)
    ax2.plot(np.arange(grad.shape[3]),lstm_grad.mean(dim=["t","x","y"]),'-o', c='red', alpha=0.5)
    ax.set_ylim(ymin,ymax)
    ax2.set_ylim(ymin,ymax)
    plt.grid()
    plt.savefig('4DVarNet_xp_GP_'+gp_dataset+'_grads_'+str(ipatch)+'.png')
    fig = plt.gcf()
    plt.close()

