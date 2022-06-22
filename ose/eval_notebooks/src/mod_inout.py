import xarray as xr
import numpy as np
#import pandas as pd
import pyinterp
import logging
import requests as rq

def read_l3_dataset(file,
                    lon_min=0., 
                    lon_max=360., 
                    lat_min=-90, 
                    lat_max=90., 
                    time_min='1900-10-01', 
                    time_max='2100-01-01'):
    
    ds = xr.open_dataset(file)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)
    ds = ds.where((ds["longitude"] >= lon_min%360.) & (ds["longitude"] <= lon_max%360.), drop=True)
    
    return ds


def read_l3_dataset_from_aviso(url_dataset, 
                               my_aviso_session,
                               lon_min=0., 
                               lon_max=360., 
                               lat_min=-90, 
                               lat_max=90., 
                               time_min='1900-10-01', 
                               time_max='2100-01-01'):
    
    # disable logger for library
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pydap").setLevel(logging.WARNING)

    #store = xr.backends.PydapDataStore.open(url_dataset, session=my_aviso_session)
    store = xr.backends.NetCDF4DataStore.open(url_dataset)
    ds = xr.open_dataset(store)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["latitude"] >= lat_min) & (ds["latitude"] <= lat_max), drop=True)
    ds = ds.where((ds["longitude"] >= lon_min%360.) & (ds["longitude"] <= lon_max%360.), drop=True)
    
    return ds


def read_l4_dataset(list_of_file, 
                    lon_min=0., 
                    lon_max=360., 
                    lat_min=-90, 
                    lat_max=90., 
                    time_min='1900-10-01', 
                    time_max='2100-01-01', 
                    is_circle=True):
    
    
    ds = xr.open_mfdataset(list_of_file, concat_dim ='time', combine='nested', parallel=True)
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["lon"]%360. >= lon_min) & (ds["lon"]%360. <= lon_max), drop=True)
    ds = ds.where((ds["lat"] >= lat_min) & (ds["lat"] <= lat_max), drop=True)
    
    x_axis = pyinterp.Axis(ds["lon"][:]%360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["lat"][:])
    z_axis = pyinterp.TemporalAxis(ds["time"].values)
    
    var = ds['ssh'][:]
    var = var.transpose('lon', 'lat', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass
    
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid


def read_l4_dataset_from_aviso(url_dataset, 
                               my_aviso_session, 
                               lon_min=0., 
                               lon_max=360., 
                               lat_min=-90, 
                               lat_max=90., 
                               time_min='1900-10-01', 
                               time_max='2100-01-01', 
                               is_circle=True):
    
    # disable logger for library
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pydap").setLevel(logging.WARNING)

    #store = xr.backends.PydapDataStore.open(url_dataset, session=my_aviso_session)
    store = xr.backends.NetCDF4DataStore.open(url_dataset)
    
    ds = xr.open_dataset(store, chunks={'time': '100MB'})
    ds = ds.sel(time=slice(time_min, time_max), drop=True)
    ds = ds.where((ds["lon"]%360. >= lon_min) & (ds["lon"]%360. <= lon_max), drop=True)
    ds = ds.where((ds["lat"] >= lat_min) & (ds["lat"] <= lat_max), drop=True)
    
    x_axis = pyinterp.Axis(ds["lon"][:]%360., is_circle=is_circle)
    y_axis = pyinterp.Axis(ds["lat"][:])
    z_axis = pyinterp.TemporalAxis(ds["time"][:])
    
    var = ds['ssh'][:]
    var = var.transpose('lon', 'lat', 'time')

    # The undefined values must be set to nan.
    try:
        var[var.mask] = float("nan")
    except AttributeError:
        pass
    
    grid = pyinterp.Grid3D(x_axis, y_axis, z_axis, var.data)
    
    del ds
    
    return x_axis, y_axis, z_axis, grid


def read_obs(input_file, oi_grid, oi_param, coarsening):
    
    logging.info('     Reading observations...')
    
    def preprocess(ds):
        return ds.coarsen(coarsening, boundary="trim").mean()
    
    ds_obs = xr.open_mfdataset(input_file, combine='nested', concat_dim='time', parallel=True, preprocess=preprocess) #.sortby('time')
    #ds_obs = ds_obs.coarsen(coarsening, boundary="trim").mean().sortby('time')
    ds_obs = ds_obs.sortby('time')
    
    lon_min = oi_grid.lon.min().values
    lon_max = oi_grid.lon.max().values
    lat_min = oi_grid.lat.min().values
    lat_max = oi_grid.lat.max().values
    time_min = oi_grid.time.min().values
    time_max = oi_grid.time.max().values
    
    ds_obs = ds_obs.sel(time=slice(time_min - np.timedelta64(int(2*oi_param.Lt.values), 'D'), 
                                   time_max + np.timedelta64(int(2*oi_param.Lt.values), 'D')), drop=True)
    
    # correct lon if domain is between [-180:180]
    if lon_min < 0:
        ds_obs['lon'] = xr.where(ds_obs['longitude'] >= 180., ds_obs['longitude']-360., ds_obs['longitude'])
        
    ds_obs = ds_obs.where((ds_obs['longitude'] >= lon_min - oi_param.Lx.values) & 
                          (ds_obs['longitude'] <= lon_max + oi_param.Lx.values) &
                          (ds_obs['latitude'] >= lat_min - oi_param.Ly.values) &
                          (ds_obs['latitude'] <= lat_max + oi_param.Ly.values) , drop=True)
    
    vtime = (ds_obs['time'].values - time_min) / np.timedelta64(1, 'D')
    ds_obs = ds_obs.assign_coords({'time': vtime})
    
    ds_obs = ds_obs.dropna(dim='time')
    
    return ds_obs


def read_obs_from_aviso(input_file, my_aviso_session, oi_grid, oi_param, coarsening):
    
    logging.info('     Reading observations from aviso...')
    
    # disable logger for library
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("pydap").setLevel(logging.WARNING)
    
    def preprocess(ds):
        return ds.coarsen(coarsening, boundary="trim").mean()
    
    list_of_dataset = []
    for url_dataset in input_file:
        store = xr.backends.PydapDataStore.open(url_dataset, session=my_aviso_session)
        ds_store = xr.open_dataset(store)
        list_of_dataset.append(ds_store)
    
    ds_obs = xr.concat(list_of_dataset, dim='time')
    #ds_obs = xr.open_mfdataset(input_file, combine='nested', concat_dim='time', parallel=True, preprocess=preprocess) #.sortby('time')
    #ds_obs = ds_obs.coarsen(coarsening, boundary="trim").mean().sortby('time')
    ds_obs = ds_obs.coarsen(coarsening, boundary="trim").mean()
    ds_obs = ds_obs.sortby('time')
    
    lon_min = oi_grid.lon.min().values
    lon_max = oi_grid.lon.max().values
    lat_min = oi_grid.lat.min().values
    lat_max = oi_grid.lat.max().values
    time_min = oi_grid.time.min().values
    time_max = oi_grid.time.max().values
    
    ds_obs = ds_obs.sel(time=slice(time_min - np.timedelta64(int(2*oi_param.Lt.values), 'D'), 
                                   time_max + np.timedelta64(int(2*oi_param.Lt.values), 'D')), drop=True)
    
    # correct lon if domain is between [-180:180]
    if lon_min < 0:
        ds_obs['longitude'] = xr.where(ds_obs['longitude'] >= 180., ds_obs['longitude']-360., ds_obs['longitude'])
        
    ds_obs = ds_obs.where((ds_obs['longitude'] >= lon_min - oi_param.Lx.values) & 
                          (ds_obs['longitude'] <= lon_max + oi_param.Lx.values) &
                          (ds_obs['latitude'] >= lat_min - oi_param.Ly.values) &
                          (ds_obs['latitude'] <= lat_max + oi_param.Ly.values) , drop=True)
    
    vtime = (ds_obs['time'].values - time_min) / np.timedelta64(1, 'D')
    ds_obs = ds_obs.assign_coords({'time': vtime})
    
    ds_obs = ds_obs.dropna(dim='time')
    
    return ds_obs


def reformate_oi_output(ds_oi, url_ds_mdt, my_aviso_session):
    ds_oi = ds_oi.drop(['gtime', 'ng', 'glon2', 'glat2', 'fglon', 'fglat', 'nobs'])
    ds_oi = ds_oi.rename({'gssh': 'sla'})
    
    
    store_ds_mdt = xr.backends.PydapDataStore.open(url_ds_mdt, session=my_aviso_session)
    mdt = xr.open_dataset(store_ds_mdt)
    
    # mdt = xr.open_dataset(mdt_file)
    mdt_interp = mdt.interp(lon=ds_oi.lon, lat=ds_oi.lat)
    
    ds_oi['ssh'] = ds_oi['sla'] + mdt_interp['mdt']
    
    return ds_oi
