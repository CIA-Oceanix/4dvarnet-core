import pyinterp
import numpy as np
import netCDF4
import logging 

from src.mod_write import *

def compute_stats(time_alongtrack, 
                  lat_alongtrack, 
                  lon_alongtrack, 
                  ssh_alongtrack, 
                  ssh_map_interp, 
                  bin_lon_step,
                  bin_lat_step, 
                  bin_time_step,
                  output_filename,
                  output_filename_timeseries):

    ncfile = netCDF4.Dataset(output_filename,'w')

    binning = pyinterp.Binning2D(
        pyinterp.Axis(np.arange(0, 360, bin_lon_step), is_circle=True),
        pyinterp.Axis(np.arange(-90, 90 + bin_lat_step, bin_lat_step)))

    # binning alongtrack
    binning.push(lon_alongtrack, lat_alongtrack, ssh_alongtrack, simple=True)
    write_stat(ncfile, 'alongtrack', binning)
    binning.clear()

    # binning map interp
    binning.push(lon_alongtrack, lat_alongtrack, ssh_map_interp, simple=True)
    write_stat(ncfile, 'maps', binning)
    binning.clear()

    # binning diff sla-msla
    binning.push(lon_alongtrack, lat_alongtrack, ssh_alongtrack - ssh_map_interp, simple=True)
    write_stat(ncfile, 'diff', binning)
    binning.clear()

    # add rmse
    diff2 = (ssh_alongtrack - ssh_map_interp)**2
    binning.push(lon_alongtrack, lat_alongtrack, diff2, simple=True)
    var = ncfile.groups['diff'].createVariable('rmse', binning.variable('mean').dtype, ('lat','lon'), zlib=True)
    var[:, :] = np.sqrt(binning.variable('mean')).T  
    
    ncfile.close()
    
    logging.info(f'  Results saved in: {output_filename}')

    # write time series statistics
    leaderboard_nrmse, leaderboard_nrmse_std = write_timeserie_stat(ssh_alongtrack, 
                                                                    ssh_map_interp, 
                                                                    time_alongtrack, 
                                                                    bin_time_step, 
                                                                    output_filename_timeseries)
    
    return leaderboard_nrmse, leaderboard_nrmse_std