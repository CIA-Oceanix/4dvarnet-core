import numpy as np
import xarray as xr
import matplotlib
import os
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec

def find_open_ncdf(path):
    path1 = "/users/local/m19beauc/4dvarnet-core/dashboard/"+path+"/lightning_logs"
    latest_subdir = max([os.path.join(path1,d) for d in os.listdir(path1)], key=os.path.getmtime)
    print(latest_subdir)
    data = xr.open_dataset(latest_subdir+"/maps.nc")
    return data
    
    
def find_open_ncdf_loss(path):
    path1 = "/users/local/m19beauc/4dvarnet-core/dashboard/"+path+"/lightning_logs"
    latest_subdir = max([os.path.join(path1,d) for d in os.listdir(path1)], key=os.path.getmtime)
    print(latest_subdir)
    data = xr.open_dataset(latest_subdir+"/loss.nc")
    return data
    
def find_open_ncdf_grads(path):
    path1 = "/users/local/m19beauc/4dvarnet-core/dashboard/"+path+"/lightning_logs"
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
    
# plots
for ipatch in range(19):   
    gt = find_open_ncdf("oi_gp_ref").gt.isel(time=ipatch,daw=2)
    # REF
    oi_analytical = find_open_ncdf("oi_gp_ref").pred.isel(time=ipatch,daw=2)
    oi_gd = find_open_ncdf("oi_gp_ref_gd").pred.isel(time=ipatch,daw=2)
    # UNET
    unet_mse_loss = find_open_ncdf("oi_gp_unet_mse_loss")['4DVarNet'].isel(time=ipatch)
    unet_oi_loss = find_open_ncdf("oi_gp_unet_oi_loss")['4DVarNet'].isel(time=ipatch)
    # FP
    fourdvarnet_fp_phi_cov = find_open_ncdf("oi_gp_spde_fp_wolp").pred.isel(time=ipatch,daw=2)
    fourdvarnet_fp_phi_unet_mse_loss = find_open_ncdf("oi_gp_4dvarnet_fp_mse_loss")['4DVarNet'].isel(time=ipatch)
    fourdvarnet_fp_phi_unet_oi_loss = find_open_ncdf("oi_gp_4dvarnet_fp_oi_loss")['4DVarNet'].isel(time=ipatch)
    # GradLSTM
    fourdvarnet_lstm_phi_cov_mse_loss = find_open_ncdf("oi_gp_spde_wolp_mse_loss").pred.isel(time=ipatch,daw=2)
    fourdvarnet_lstm_phi_cov_oi_loss = find_open_ncdf("oi_gp_spde_wolp_oi_loss").pred.isel(time=ipatch,daw=2)
    fourdvarnet_lstm_phi_unet_mse_loss = find_open_ncdf("oi_gp_4dvarnet_mse_loss")['4DVarNet'].isel(time=ipatch)
    fourdvarnet_lstm_phi_unet_oi_loss = find_open_ncdf("oi_gp_4dvarnet_oi_loss")['4DVarNet'].isel(time=ipatch)

    fig = plt.figure(figsize=(20,20))
    gs = gridspec.GridSpec(6,4)
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
    # OI loss
    ax10 = fig.add_subplot(gs[5, 0])
    ax11 = fig.add_subplot(gs[5, 1])
    ax12 = fig.add_subplot(gs[5, 2])
    ax13 = fig.add_subplot(gs[5, 3])

    extent = [np.min(gt.lon.values),np.max(gt.lon.values),np.min(gt.lat.values),np.max(gt.lat.values)]
    vmax = -1.5
    vmin = 1.5
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
    # OI loss
    im10 = plot(ax10,gt.lon,gt.lat,unet_oi_loss.values,'UNet (OI loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im11 = plot(ax11,gt.lon,gt.lat,fourdvarnet_fp_phi_unet_oi_loss.values,'4DVarNet-FP5-Phi-UNet (oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im12 = plot(ax12,gt.lon,gt.lat,fourdvarnet_lstm_phi_cov_oi_loss.values,'4DVarNet-LSTM-Phi-cov (oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    im13 = plot(ax13,gt.lon,gt.lat,fourdvarnet_lstm_phi_unet_oi_loss.values,'4DVarNet-LSTM-Phi-UNet (oi loss)',extent=extent,cmap=cmap,norm=norm,colorbar=False)
    plt.savefig('4DVarNet_xp_GP_'+str(ipatch)+'.png')
    fig = plt.gcf()
    plt.close() 

# OI variational cost vs MSE_loss
for ipatch in range(19):
    oi_loss_oi = find_open_ncdf_loss("oi_gp_ref").loss_oi.isel(patch=ipatch)
    oi_loss_mse = find_open_ncdf_loss("oi_gp_ref").loss_mse.isel(patch=ipatch)
    # OI (gradient descent)
    oi_gd_loss_oi = find_open_ncdf_loss("oi_gp_ref_gd").loss_oi.isel(patch=ipatch)
    oi_gd_loss_mse = find_open_ncdf_loss("oi_gp_ref_gd").loss_mse.isel(patch=ipatch)
    # UNet (mse_loss)
    unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_unet_mse_loss").loss_oi.isel(patch=ipatch)
    unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_unet_mse_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet-FP (mse_loss)
    fourdvarnet_fp_phi_unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_fp_mse_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_fp_phi_unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_fp_mse_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet (mse_loss)
    fourdvarnet_lstm_phi_unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_mse_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_mse_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet (mse_loss)
    fourdvarnet_lstm_phil_unet_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_wphi_linear_mse_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phil_unet_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_wphi_linear_mse_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet (mse_loss)
    fourdvarnet_lstm_phi_cov_mse_loss_loss_oi = find_open_ncdf_loss("oi_gp_spde_wolp_mse_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_cov_mse_loss_loss_mse = find_open_ncdf_loss("oi_gp_spde_wolp_mse_loss").loss_mse.isel(patch=ipatch)
    # UNet (oi_loss)
    unet_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_unet_oi_loss").loss_oi.isel(patch=ipatch)
    unet_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_unet_oi_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet-FP (oi_loss)
    fourdvarnet_fp_phi_unet_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_fp_oi_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_fp_phi_unet_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_fp_oi_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet (oi_loss)
    fourdvarnet_lstm_phi_unet_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_4dvarnet_oi_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_unet_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_4dvarnet_oi_loss").loss_mse.isel(patch=ipatch)
    # 4DVarNet (oi_loss)
    fourdvarnet_lstm_phi_cov_oi_loss_loss_oi = find_open_ncdf_loss("oi_gp_spde_wolp_oi_loss").loss_oi.isel(patch=ipatch)
    fourdvarnet_lstm_phi_cov_oi_loss_loss_mse = find_open_ncdf_loss("oi_gp_spde_wolp_oi_loss").loss_mse.isel(patch=ipatch)

    fig = plt.figure(figsize=(10,7))
    plt.scatter(oi_loss_mse, oi_loss_oi, c='red', alpha=0.5, label='OI (analytical)', marker="*", s=300)
    plt.plot(oi_gd_loss_mse, oi_gd_loss_oi, '-o', c='green', alpha=0.5, label='OI (gradient descent)')
    plt.plot(unet_mse_loss_loss_mse, unet_mse_loss_loss_oi, '-o', c='blue', alpha=0.5, label='UNet (mse loss)')
    plt.plot(fourdvarnet_fp_phi_unet_mse_loss_loss_mse, fourdvarnet_fp_phi_unet_mse_loss_loss_oi, '-o', c='orange', alpha=0.5,label='4DVarNet-FP (mse loss)')
    plt.plot(fourdvarnet_lstm_phi_unet_mse_loss_loss_mse, fourdvarnet_lstm_phi_unet_mse_loss_loss_oi, '-o', c='yellow', alpha=0.5,label='4DVarNet-LSTM (mse loss)')
    plt.plot(fourdvarnet_lstm_phi_cov_mse_loss_loss_mse, fourdvarnet_lstm_phi_cov_mse_loss_loss_oi, '-o', c='purple', alpha=0.5,label='SPDE prior-LSTM (mse loss)')
    plt.plot(fourdvarnet_lstm_phil_unet_mse_loss_loss_mse, fourdvarnet_lstm_phil_unet_mse_loss_loss_oi, '-o', c='black', alpha=0.5,label='4DVarNet-LSTM with phi linear (mse loss)')
    plt.plot(unet_oi_loss_loss_mse, unet_oi_loss_loss_oi, 's--', c='blue', alpha=0.5, label='UNet (oi loss)')
    plt.plot(fourdvarnet_fp_phi_unet_oi_loss_loss_mse, fourdvarnet_fp_phi_unet_oi_loss_loss_oi, 's--', c='orange', alpha=0.5,label='4DVarNet-FP (oi loss)')
    plt.plot(fourdvarnet_lstm_phi_unet_oi_loss_loss_mse, fourdvarnet_lstm_phi_unet_oi_loss_loss_oi, 's--', c='yellow', alpha=0.5,label='4DVarNet-LSTM (oi loss)')
    plt.plot(fourdvarnet_lstm_phi_cov_oi_loss_loss_mse, fourdvarnet_lstm_phi_cov_oi_loss_loss_oi, 's--', c='purple', alpha=0.5,label='SPDE prior-LSTM (oi loss)')
    xmin = min(np.concatenate((oi_loss_mse.values,
           oi_gd_loss_mse.values,
           unet_mse_loss_loss_mse.values,
           fourdvarnet_fp_phi_unet_mse_loss_loss_mse.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_mse.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_mse.values)))
    xmax = max(np.concatenate((oi_loss_mse.values,
           oi_gd_loss_mse.values,
           unet_mse_loss_loss_mse.values,
           fourdvarnet_fp_phi_unet_mse_loss_loss_mse.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_mse.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_mse.values)))
    ymin = min(np.concatenate((oi_loss_oi.values,
           oi_gd_loss_oi.values,
           unet_mse_loss_loss_oi.values,
           fourdvarnet_fp_phi_unet_mse_loss_loss_oi.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_oi.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_oi.values)))
    ymax = max(np.concatenate((oi_loss_oi.values,
           oi_gd_loss_oi.values,
           unet_mse_loss_loss_oi.values,
           fourdvarnet_fp_phi_unet_mse_loss_loss_oi.values,
           fourdvarnet_lstm_phi_unet_mse_loss_loss_oi.values,
           fourdvarnet_lstm_phi_cov_mse_loss_loss_oi.values)))
    ymax = 3e5
    plt.xlim(xmin-(xmax-xmin)/10,xmax+(xmax-xmin)/10)
    plt.ylim(ymin-(ymax-ymin)/10,ymax+(ymax-ymin)/10)
    plt.xlabel('MSE loss')
    plt.ylabel('OI loss')
    #plt.yscale('log',base=10)
    plt.grid()
    plt.gca().invert_xaxis()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.savefig('4DVarNet_xp_GP_loss_'+str(ipatch)+'.png', bbox_inches='tight')
    fig = plt.gcf()
    plt.close() 

'''
# Grad vs GradLSTM
grad = find_open_ncdf_grads("oi_gp_4dvarnet_mse_loss_test_grad_lstm").loss_oi.isel(patch=1)
lstm_grad = find_open_ncdf_grads("oi_gp_4dvarnet_mse_loss_test_grad_lstm").loss_mse.isel(patch=1)
fig, ax = plt.subplots(figsize=(20,20))
ax2 = ax.twinx()
ax.plot(np.arange(grad.shape[0]),grad.mean(dim=["t","x","y"]), '-o', c='green', alpha=0.5)
ax2.plot(np.arange(grad.shape[0]),lstm_grad.mean(dim=["t","x","y"]),'-o', c='red', alpha=0.5)
plt.savefig('4DVarNet_xp_GP_grads.png')
fig = plt.gcf()
plt.close() 
'''
