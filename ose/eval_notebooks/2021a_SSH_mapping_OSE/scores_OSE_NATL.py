import os
import sys
sys.path.append('..')
import logging
import pandas as pd

from src.mod_inout import *
from src.mod_interp import *
from src.mod_stats import *
from src.mod_spectral import *
from src.mod_plot import *
from src.utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# study area
#lon_min+180)%360-180
lon_min = 295.
lon_max = 305.
lat_min = 33.
lat_max = 43.
is_circle = False
time_min = '2017-01-01'
time_max = '2017-12-31'

# Outputs
bin_lat_step = 1.
bin_lon_step = 1.
bin_time_step = '1D'
output_directory = 'results_NATL_NATL'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

os.chdir(output_directory)

# Spectral parameter
# C2 parameter
delta_t = 0.9434  # s
velocity = 6.77   # km/s
delta_x = velocity * delta_t
lenght_scale = 1000 # km

# independent along-track
alontrack_independent_dataset= '../inputs/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc'
# Read along-track
ds_alongtrack = read_l3_dataset(alontrack_independent_dataset, 
                                           lon_min=lon_min, 
                                           lon_max=lon_max, 
                                           lat_min=lat_min, 
                                           lat_max=lat_max, 
                                           time_min=time_min, 
                                           time_max=time_max)
ds_alongtrack

file_oi = '../inputs/maps_NATL_NATL.nc'
file_4dvarnet = '../inputs/maps_NATL_NATL.nc'
def define_ssh_var(file,var,id):
  data = xr.open_dataset(file_oi)
  data = data.update({'ssh':(('time','lat','lon'),data[var].values)})
  data = data.assign_coords(time=[np.datetime64(day) for day in data.time.values])
  newfile = file.replace(".nc","_method_"+id+".nc")
  data.to_netcdf(newfile)
  return newfile

file_oi = define_ssh_var(file_oi,'oi',"DUACS")
file_4dvarnet = define_ssh_var(file_4dvarnet,'pred',"4DVarNet")
methods = ['OI', '4DVarNet'] 
colors = ['k','red','seagreen','steelblue','mediumorchid','darkorange','darkblue']

# plot maps

filenames = [file_oi,file_oi,file_4dvarnet] 
vars = ['obs']
vars.extend(methods)                 
vars2 = ['gt','oi','pred']
ds = [ xr.open_dataset(filenames[i])[vars2[i]].isel(time=0) for i in range(len(filenames))]  
plot_maps("xp_BOOST_SWOT_Train_NATL_Test_GF_maps.png",ds,ds[0].lon,ds[0].lat,
          grad=False,orthographic=True, methods=vars, figsize=(20,10+10*len(methods)/2))
plot_maps("xp_BOOST_SWOT_Train_NATL_Test_GF_maps_grad.png",ds,ds[0].lon,ds[0].lat,
          grad=True,orthographic=True, methods=vars, figsize=(20,10+10*len(methods)/2))
          
# animate maps
ds = [ xr.open_dataset(filenames[i])[vars2[i]].isel(time=slice(0,365)) for i in range(len(filenames))]  
animate_maps("xp_BOOST_SWOT_Train_GF_Test_GF_maps.mp4",ds[1], ds[2],
             ds[0].lon,ds[0].lat,
             crop=None, orthographic=True, grad=False)
animate_maps("xp_BOOST_SWOT_Train_GF_Test_GF_maps_grad.mp4",ds[1].values, ds[2].values,
             ds[0].lon,ds[0].lat,
             crop=None, orthographic=True, grad=True)
                                                                          
# Interpolate maps onto alongtrack dataset for each dataset
time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_oi_interp = interp_on_alongtrack(file_oi, 
                                                                                                      ds_alongtrack,
                                                                                                       lon_min=lon_min, 
                                                                                                       lon_max=lon_max, 
                                                                                                       lat_min=lat_min, 
                                                                                                       lat_max=lat_max, 
                                                                                                       time_min=time_min, 
                                                                                                       time_max=time_max,
                                                                                                       is_circle=is_circle)
                                                                                                      
time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_4dvarnet_interp = interp_on_alongtrack(file_4dvarnet, 
                                                                                                      ds_alongtrack,
                                                                                                       lon_min=lon_min, 
                                                                                                       lon_max=lon_max, 
                                                                                                       lat_min=lat_min, 
                                                                                                       lat_max=lat_max, 
                                                                                                       time_min=time_min, 
                                                                                                       time_max=time_max,
                                                                                                       is_circle=is_circle)
                                                                                                  
# Compute spatial and temporal statistics
leaderboard_nrmse_oi, leaderboard_nrmse_std_oi = compute_stats(time_alongtrack, 
                                                         lat_alongtrack, 
                                                         lon_alongtrack, 
                                                         ssh_alongtrack, 
                                                         ssh_oi_interp, 
                                                         bin_lon_step,
                                                         bin_lat_step, 
                                                         bin_time_step,
                                                         output_filename='spa_stat_OI.nc',
                                                         output_filename_timeseries='TS_OI.nc')           
                                                        	                                                  
leaderboard_nrmse_4dvarnet, leaderboard_nrmse_std_4dvarnet = compute_stats(time_alongtrack, 
                                                         lat_alongtrack, 
                                                         lon_alongtrack, 
                                                         ssh_alongtrack, 
                                                         ssh_4dvarnet_interp, 
                                                         bin_lon_step,
                                                         bin_lat_step, 
                                                         bin_time_step,
                                                         output_filename='spa_stat_4DVarNet.nc',
                                                         output_filename_timeseries='TS_4DVarNet.nc') 
                                                         
# spatial_scores   
filenames = ['spa_stat_OI.nc','spa_stat_4DVarNet.nc']                   
ds = [ xr.open_dataset('spa_stat_'+methods[i]+'.nc',group='diff').rmse.sel(lon=slice(lon_min,lon_max),
                                                                     lat=slice(lat_min,lat_max)) for i in range(len(filenames))]  
maps_score('4DVarNet_xp_BOOST_SWOT_Train_NATL_Test_GF_scores_spatial.png', ds,
            ds[0].lon, ds[0].lat, figsize=(20,10*len(methods)/2))

# temporal scores        
plot_temporal_statistics('4DVarNet_xp_BOOST_SWOT_Train_NATL_Test_GF_scores_temporal.png', 
                         ['TS_OI.nc', 'TS_4DVarNet.nc'],
                         methods,colors)

# Compute spectral scores
compute_spectral_scores(time_alongtrack, 
                        lat_alongtrack, 
                        lon_alongtrack, 
                        ssh_alongtrack, 
                        ssh_oi_interp, 
                        lenght_scale,
                        delta_x,
                        delta_t,
                        'spectrum_oi.nc')   
compute_spectral_scores(time_alongtrack, 
                        lat_alongtrack, 
                        lon_alongtrack, 
                        ssh_alongtrack, 
                        ssh_4dvarnet_interp, 
                        lenght_scale,
                        delta_x,
                        delta_t,
                        'spectrum_4dvarnet.nc')     
                                                           
leaderboard_psds_score_oi = plot_psd_score('spectrum_oi.nc')  
leaderboard_psds_score_4dvarnet = plot_psd_score('spectrum_4dvarnet.nc')  

plot_psd_score_intercomparison('4DVarNet_xp_BOOST_SWOT_Train_NATL_Test_GF_scores_spectrum.png', 
                               ['spectrum_oi.nc', 'spectrum_4dvarnet.nc'], methods,colors)

                 
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
Leaderboard
print(Leaderboard.to_markdown())  
os.remove(file_oi)
os.remove(file_4dvarnet)                        

