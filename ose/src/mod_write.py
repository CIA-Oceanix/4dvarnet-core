import xarray as xr
import numpy as np
import logging


def write_stat(nc, group_name, binning):
    
    grp = nc.createGroup(group_name)
    grp.createDimension('lon', len(binning.x))
    grp.createDimension('lat', len(binning.y))
    
    longitude = grp.createVariable('lon', 'f4', 'lon', zlib=True)
    longitude[:] = binning.x
    latitude = grp.createVariable('lat', 'f4', 'lat', zlib=True)
    latitude[:] = binning.y
    
    stats = ['min', 'max', 'sum', 'sum_of_weights', 'variance', 'mean', 'count', 'kurtosis', 'skewness']
    for variable in stats:
        
        var = grp.createVariable(variable, binning.variable(variable).dtype, ('lat','lon'), zlib=True)
        var[:, :] = binning.variable(variable).T 

        
def write_timeserie_stat(ssh_alongtrack, ssh_map_interp, time_vector, freq, output_filename):
    
    
    diff = ssh_alongtrack - ssh_map_interp
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(diff, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    rmse = np.copy(vrms)
    
    # save stat to dataset
    ds = xr.Dataset(
        {
            "mean": (("time"), vmean.values),
            "min": (("time"), vminimum.values),
            "max": (("time"), vmaximum.values),
            "count": (("time"), vcount.values),
            "variance": (("time"), vvariance.values),
            "median": (("time"), vmedian.values),
            "rms": (("time"), vrms.values),            
        },
        {"time": vmean['time']},
    )
    
    ds.to_netcdf(output_filename, group='diff')
    
    
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_alongtrack, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    rms_alongtrack = np.copy(vrms)
    
    # save stat to dataset
    ds = xr.Dataset(
        {
            "mean": (("time"), vmean.values),
            "min": (("time"), vminimum.values),
            "max": (("time"), vmaximum.values),
            "count": (("time"), vcount.values),
            "variance": (("time"), vvariance.values),
            "median": (("time"), vmedian.values),
            "rms": (("time"), vrms.values),            
        },
        {"time": vmean['time']},
    )
    
    ds.to_netcdf(output_filename, group='alongtrack', mode='a')
    
    
    # convert data vector and time vector into xarray.Dataarray
    da = xr.DataArray(ssh_map_interp, coords=[time_vector], dims="time")
    
    # resample 
    da_resample = da.resample(time=freq)
    
    # compute stats
    vmean = da_resample.mean()
    vminimum = da_resample.min()
    vmaximum = da_resample.max()
    vcount = da_resample.count()
    vvariance = da_resample.var()
    vmedian = da_resample.median()
    vrms = np.sqrt(np.square(da).resample(time=freq).mean())
    
    # save stat to dataset
    ds = xr.Dataset(
        {
            "mean": (("time"), vmean.values),
            "min": (("time"), vminimum.values),
            "max": (("time"), vmaximum.values),
            "count": (("time"), vcount.values),
            "variance": (("time"), vvariance.values),
            "median": (("time"), vmedian.values),
            "rms": (("time"), vrms.values),            
        },
        {"time": vmean['time']},
    )
    
    ds.to_netcdf(output_filename, group='maps', mode='a')
    
    logging.info(' ')
    logging.info(f'  Results saved in: {output_filename}')
    
    rmse_score = 1. - rmse/rms_alongtrack
    # mask score if nb obs < nb_min_obs
    nb_min_obs = 10
    rmse_score = np.ma.masked_where(vcount.values < nb_min_obs, rmse_score)
    
    mean_rmse = np.ma.mean(np.ma.masked_invalid(rmse_score))
    std_rmse = np.ma.std(np.ma.masked_invalid(rmse_score))
    
    logging.info(' ')
    logging.info(f'  MEAN RMSE Score = {mean_rmse}')
    logging.info(' ')
    logging.info(f'  STD RMSE Score = {std_rmse}')
    
    return mean_rmse, std_rmse