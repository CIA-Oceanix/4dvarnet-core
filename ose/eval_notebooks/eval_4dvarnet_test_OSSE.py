# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: 4dvarnet
#     language: python
#     name: 4dvarnet
# ---

# %% [markdown]
# # 4DVarNet-OSE Evaluation Notebook (OSSE)

# %% [markdown]
# # 1. Import tools and change directory

# %%
import os
import sys
# sys.path.append('/raid/localscratch/qfebvre/4dvarnet-core/ose/eval_notebooks')

import logging
import pandas as pd

import traceback
from src.mod_inout import *
from src.mod_interp import *
from src.mod_stats import *
from src.mod_spectral import *
from src.mod_plot import *
from src.utils import *

# %%
# os.chdir("/gpfswork/rech/yrf/uba22to/4dvarnet-core")
# print(os.getcwd())

# %% [markdown]
# # 2. Test the code 

# %%
# ! HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python hydra_main.py xp=mbeaucha/xp_ose/osse_nadswot_gf_wsst file_paths=jz_ose entrypoint=test entrypoint.ckpt_path="/gpfsstore/rech/yrf/uba22to/4DVarNet_ckpt_GF_4nadirs1swot_wsst.ckpt"

# %% [markdown]
# # 3. Retrieve and import the results

# %%
import glob
from datetime import datetime
import numpy as np
from pathlib import Path

# dirs = glob.glob("/raid/localscratch/qfebvre/4dvarnet-core/dashboard/ose_gf_wsst/lightning_logs/version_*")
# file = max(dirs, key=os.path.getctime) + "/test.nc"

# data_4dvarnet = mod.test_xr_ds.isel(time=slice(10, -10))
# file = 'tmp/test08.nc'

# Path(file).unlink(missing_ok=True)
# data_4dvarnet.to_netcdf(file)
# xr.open_dataset(file)
#time = [ np.datetime_as_string(date, unit='D') for date in data_4dvarnet.time.values ]

# import OI baseline
# file_oi = "/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc"
# file_oi = cfg.params.files_cfg.oi_path
# data_baseline = xr.open_dataset(file_oi)
# data_baseline = data_baseline.sel(time=slice(time[0],time[-1]),
#                                   lat=slice(33,43),
#                                   lon=slice(-65.0,-55))

# # data_4dvarnet.interp_like(data_baseline, kwargs=dict(fill_value="extrapolate")).pred.isel(time=1).plot()
# # data_4dvarnet.interp_like(data_baseline,method='nearest' ).obs_inp.isel(time=1).plot()
# # rename variables
# data_baseline = data_baseline.update({
#                      'obs_inp':(('time','lat','lon'),data_4dvarnet.gt
#                          .interp_like(data_baseline, method='nearest').values),
#                      'pred':(('time','lat','lon'),data_baseline.ssh.values)})
# data_4dvarnet = data_4dvarnet.interp_like(data_baseline, method='nearest').update({
#                      'obs_inp':(('time','lat','lon'),data_4dvarnet.obs_inp
#                          .interp_like(data_baseline, method='nearest').values),
#                      'pred':(('time','lat','lon'),data_4dvarnet.pred
#                          .interp_like(data_baseline, kwargs=dict(fill_value="extrapolate")).values)})

def metrics_ose(data_src, alontrack_independent_dataset):
    try:
        Path('spa_stat_4dvarnet.nc').unlink(missing_ok=True)
        Path('spectrum_4dvarnet.nc').unlink(missing_ok=True)
        Path('TS_4dvarnet.nc').unlink(missing_ok=True)
        Path('spa_stat_OI.nc').unlink(missing_ok=True)
        Path('spectrum_oi.nc').unlink(missing_ok=True)
        Path('TS_OI.nc').unlink(missing_ok=True)
        print('Computing metrics...')
        data_baseline = data_src.copy().update({
                             'obs_inp':(('time','lat','lon'),data_src.gt.values),
                             'pred':(('time','lat','lon'),data_src.oi.values)})
        data_4dvarnet = data_src.copy().update({
                             'obs_inp':(('time','lat','lon'),data_src.obs_inp.values),
                             'pred':(('time','lat','lon'),data_src.pred.values)})
        time = data_4dvarnet.time.values
        # %% [markdown]
        # # 4. XP parameters

        # %%
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # study area
        #lon_min+180)%360-180
        lon_min = 295.
        lon_max = 305.
        lat_min = 33.
        lat_max = 43.
        is_circle = False
        time_min = '2017-01-05'
        time_max = '2017-12-26'

        # Outputs
        bin_lat_step = 1.
        bin_lon_step = 1.
        bin_time_step = '1D'

        # Spectral parameter
        # C2 parameter
        delta_t = 0.9434  # s
        velocity = 6.77   # km/s
        delta_x = velocity * delta_t
        lenght_scale = 1000 # km

        # %% [markdown]
        # # 5. Import Cryosat-2 dataset

        # %%
        # get independent along-track
        # alontrack_independent_dataset= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc'
        # alontrack_independent_dataset= cfg.params.files_cfg.test_track
        # Read along-track
        ds_alongtrack = read_l3_dataset(alontrack_independent_dataset,
                                                   lon_min=lon_min,
                                                   lon_max=lon_max,
                                                   lat_min=lat_min,
                                                   lat_max=lat_max,
                                                   time_min=time_min,
                                                   time_max=time_max)


        # %%
        def define_ssh_var(data,var,id):
          # data = xr.open_dataset(file)
          if 'obs_inp' in list(data.keys()):
              data = data.update({'obs':(('time','lat','lon'),data['obs_inp'].values)})
          data = data.update({'ssh':(('time','lat','lon'),data[var].values)})
          data = data.assign_coords(time=[np.datetime64(day) for day in data.time.values])
          newfile = Path('tmp') / 'test.nc'.replace(".nc","_method_"+id+".nc")
          Path(newfile).unlink(missing_ok=True)
          data.to_netcdf(newfile)
          return str(newfile)

        file_oi = define_ssh_var(data_src.copy(),'oi',"DUACS")
        print(xr.open_dataset(file_oi))
        file_4dvarnet = define_ssh_var(data_src.copy(),'pred',"4DVarNet")
        print(xr.open_dataset(file_4dvarnet))

        methods = ['OI', '4DVarNet']
        colors = ['k','red']

        # %%
        time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_oi_interp = interp_on_alongtrack(
            file_oi, ds_alongtrack, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
            lat_max=lat_max, time_min=time_min, time_max=time_max, is_circle=is_circle
        )


        time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_4dvarnet_interp = interp_on_alongtrack(
                file_4dvarnet, ds_alongtrack, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min,
                lat_max=lat_max, time_min=time_min, time_max=time_max, is_circle=is_circle)


        # %% [markdown]
        # # 6.Compute spatial and temporal scores

        # %%
        # Compute spatial and temporal statistics
        leaderboard_nrmse_oi, leaderboard_nrmse_std_oi = compute_stats(
            time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_oi_interp,
            bin_lon_step, bin_lat_step, bin_time_step, output_filename='spa_stat_OI.nc', output_filename_timeseries='TS_OI.nc')
        
        # Compute spatial and temporal statistics
        leaderboard_nrmse_4dvarnet, leaderboard_nrmse_std_4dvarnet = compute_stats(
            time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_4dvarnet_interp, bin_lon_step,
            bin_lat_step, bin_time_step, output_filename='spa_stat_4dvarnet.nc', output_filename_timeseries='TS_4dvarnet.nc')

        # %% [markdown]
        # # 7. Display spatial scores

        # %%
        filenames = ['spa_stat_OI.nc','spa_stat_4DVarNet.nc']  
        methods2 = ['OI', '4dvarnet']
        ds = [ xr.open_dataset('spa_stat_'+methods2[i]+'.nc',group='diff').rmse.sel(lon=slice(lon_min,lon_max),
                                                                             lat=slice(lat_min,lat_max)) for i in range(len(filenames))]
        maps_score(ds,ds[0].lon, ds[0].lat, methods, figsize=(20,10*len(methods)/2))

        # %% [markdown]
        # # 8. Display temporal scores

        # %%
        # %matplotlib inline
        # temporal scores        
        # fig_ts = plot_temporal_statistics(['TS_OI.nc', 'TS_4dvarnet.nc'],
                                 # methods,colors)

        # %% [markdown]
        # # 9. Compute and display spectral scores

        # %%
        # Compute spectral scores
        compute_spectral_scores(
            time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack,
            ssh_oi_interp, lenght_scale, delta_x, delta_t, 'spectrum_oi.nc')
        compute_spectral_scores(
            time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack,
            ssh_4dvarnet_interp, lenght_scale, delta_x, delta_t, 'spectrum_4dvarnet.nc')

        # Compute spectral scores
        fig_psd_oi, leaderboard_psds_score_oi = plot_psd_score('spectrum_oi.nc')
        fig_psd, leaderboard_psds_score_4dvarnet = plot_psd_score('spectrum_4dvarnet.nc')

        fig_inter_psd, resolved_scales = plot_psd_score_intercomparison(['spectrum_oi.nc', 'spectrum_4dvarnet.nc'], 
                                       methods,colors)

        # %%
        # Print leaderboard
        data = np.array([['OI','4DVarNet'],
                 [leaderboard_nrmse_oi,leaderboard_nrmse_4dvarnet],
                 [leaderboard_nrmse_std_oi,leaderboard_nrmse_std_4dvarnet],             
                 [int(leaderboard_psds_score_oi),int(leaderboard_psds_score_4dvarnet)]]).T
        Leaderboard = pd.DataFrame(data,
                                   columns=['Method',
                                            "µ(RMSE) ",
                                            "σ(RMSE)",
                                            'λx (km)'])
        print("Summary of the leaderboard metrics:")
        print(Leaderboard.to_markdown())

        # %% [markdown]
        # # 10. Display SSH and GradSSH fields

        # %%
        itime = 20

        filenames = [file_oi,file_oi,file_4dvarnet]
        vars = ['obs']
        vars.extend(methods)
        vars2 = ['gt','oi','pred']
        ds = [ xr.open_dataset(filenames[i])[vars2[i]].isel(time=itime) for i in range(len(filenames))] 
        fig_map = plot_maps(ds,ds[0].lon,ds[0].lat,
                  grad=False,orthographic=True, methods=vars, figsize=(20,10+10*len(methods)/2))

        # %%
        fig_grad = plot_maps(ds[1:],ds[0].lon,ds[0].lat,
                  grad=True,orthographic=False, methods=vars, figsize=(20,10+10*len(methods)/2))

        # %%
        # Clean
        figs = dict(
                fig_map=fig_map,
                fig_grad=fig_grad,
                fig_inter_psd=fig_inter_psd,
                fig_psd=fig_psd,
                fig_psd_oi=fig_psd_oi,
                # fig_ts=fig_ts,
        )
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__=='__main__':
    file = str(Path('/gpfsssd/scratch/rech/yrf/ual82ir/4dvarnet-core/lightning_logs/version_1626207') / "test.nc")
    alontrack_independent_dataset    = '/gpfsssd/scratch/rech/yrf/ual82ir/sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc'
    data_4dvarnet = xr.open_dataset(file)
    locals().update(metrics_ose(data_4dvarnet, alontrack_independent_dataset))
