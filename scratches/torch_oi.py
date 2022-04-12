import xarray as xr
import pandas as pd
import torch
import numpy as np
import traceback
import warnings
import time
import sys
import os
warnings.filterwarnings('ignore')
import numpy
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def read_obs(input_file, oi_grid, oi_param, simu_start_date, coarsening):
    
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
        ds_obs['lon'] = xr.where(ds_obs['lon'] >= 180., ds_obs['lon']-360., ds_obs['lon'])
        
    ds_obs = ds_obs.where((ds_obs['lon'] >= lon_min - oi_param.Lx.values) & 
                          (ds_obs['lon'] <= lon_max + oi_param.Lx.values) &
                          (ds_obs['lat'] >= lat_min - oi_param.Ly.values) &
                          (ds_obs['lat'] <= lat_max + oi_param.Ly.values) , drop=True)
    
    vtime = (ds_obs['time'].values - np.datetime64(simu_start_date)) / np.timedelta64(1, 'D')
    ds_obs = ds_obs.assign_coords({'time': vtime})
    
    return ds_obs

def oi_grid(glon, glat, gtime, simu_start_date):
    """
    
    """
    
    logging.info('     Set OI grid...')

    nx = len(glon)
    ny = len(glat)
    nt = len(gtime)

    # define & initialize ssh array
    gssh = np.empty((nt, ny, nx))
    nobs = np.empty(nt)

    # Make 2D grid
    glon2, glat2 = np.meshgrid(glon, glat)
    fglon = glon2.flatten()
    fglat = glat2.flatten()

    ng = len(fglat) # number of grid points
    vtime = (gtime - np.datetime64(simu_start_date)) / np.timedelta64(1, 'D')
    
    
    ds_oi_grid = xr.Dataset({'gssh' : (('time', 'lat', 'lon'), gssh), 
                             'glon2' : (('lat', 'lon'), glon2),
                             'glat2' : (('lat', 'lon'), glat2),
                             'fglon' : (('ng'), fglon),
                             'fglat' : (('ng'), fglat),
                             'nobs' : (('time'), nobs)},
                              coords={'gtime': (vtime).astype(np.float),
                                      'time': gtime,
                                      'lat': glat, 
                                      'lon': glon,
                                      'ng': np.arange(ng)})

    return ds_oi_grid
    

def oi_param(Lx, Ly, Lt, noise):
    
    logging.info('     Set OI params...')
    
    ds_oi_param = xr.Dataset({'Lx' : Lx,
                              'Ly' : Ly,
                              'Lt' : Lt,
                              'noise' : noise})
    
    return ds_oi_param
    

def oi_core(it, ds_oi_grid, ds_oi_param, ds_obs):
    
    ind1 = np.where((np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[it]) < 2.*ds_oi_param.Lt.values))[0]
    nobs = len(ind1)
    print('Processing time-step : ', it, '/', len(ds_oi_grid.gtime.values) - 1, '      nobs = ', nobs, end="\r")
    
    BHt = np.empty((len(ds_oi_grid.ng), nobs))
    HBHt = np.empty((nobs, nobs))
    
    obs_lon = ds_obs.lon.values[ind1]
    print(obs_lon.shape)
    obs_lat = ds_obs.lat.values[ind1]
    print(obs_lat.shape)
    obs_time = ds_obs.time.values[ind1]
    print(obs_time.shape)
    
    fglon = ds_oi_grid.fglon.values

    print(fglon.shape)
    fglat = ds_oi_grid.fglat.values
    print(fglat.shape)
    ftime = ds_oi_grid.gtime.values[it]
    print(ftime.shape)
    
    for iobs in range(nobs):
        # print(iobs)
        
        BHt[:,iobs] = np.exp(-((ftime - obs_time[iobs])/ds_oi_param.Lt.values)**2 - 
                                ((fglon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 - 
                                ((fglat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)
        
        HBHt[:,iobs] = np.exp(-((obs_time - obs_time[iobs])/ds_oi_param.Lt.values)**2 -
                                 ((obs_lon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 -
                                 ((obs_lat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)
    
    del obs_lon, obs_lat, obs_time

    R = np.diag(np.full((nobs), ds_oi_param.noise.values**2))

    Coo = HBHt + R
    print(f'{R.shape, HBHt.shape, Coo.shape, BHt.shape=}')
    Mi = np.linalg.inv(Coo)
    print( ds_obs.ssh_model.values.shape )
    print( ind1)
    print(ds_obs.ssh_model.values[ind1].shape)
    sol = np.dot(np.dot(BHt, Mi), ds_obs.ssh_model.values[ind1])

    ds_oi_grid.gssh[it, :, :] = sol.reshape(ds_oi_grid.lat.size, ds_oi_grid.lon.size)
    ds_oi_grid.nobs[it] = nobs


def my_oi(it, ds_oi_grid, ds_oi_param, ds_obs):
    
    ind1 = np.nonzero(np.abs(ds_obs.time.values - ds_oi_grid.gtime.values[it]) < 2.*ds_oi_param.Lt.item())
    ds_obs =  ds_obs.isel(time=ind1[0])
    obs = np.flatnonzero(np.isfinite(ds_obs.ssh_model.values))
    nobs = len(obs)
    print('Processing time-step : ', it, '/', len(ds_oi_grid.gtime.values) - 1, '      nobs = ', nobs, end="\r")

    BHt = np.empty((len(ds_oi_grid.ng), nobs))
    HBHt = np.empty((nobs, nobs))
    
    obs_lon = np.ravel(ds_obs.lon.broadcast_like(ds_obs.ssh_model).values)[obs]
    obs_lat = np.ravel(ds_obs.lat.broadcast_like(ds_obs.ssh_model).values)[obs]
    obs_time = np.ravel(ds_obs.time.broadcast_like(ds_obs.ssh_model).values)[obs]

    fglon = ds_oi_grid.fglon.values

    fglat = ds_oi_grid.fglat.values
    ftime = ds_oi_grid.gtime.values[it]
    
    
    # for iobs in range(nobs):
    #     # print(iobs)
        
    #     BHt[:,iobs] = np.exp(-((ftime - obs_time[iobs])/ds_oi_param.Lt.values)**2 - 
    #                             ((fglon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 - 
    #                             ((fglat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)
    #     HBHt[:,iobs] = np.exp(-((obs_time - obs_time[iobs])/ds_oi_param.Lt.values)**2 -
    #                              ((obs_lon - obs_lon[iobs])/ds_oi_param.Lx.values)**2 -
    #                              ((obs_lat - obs_lat[iobs])/ds_oi_param.Ly.values)**2)
  
    BHt = np.exp(
            - ((ftime - obs_time[None, :])/ds_oi_param.Lt.values)**2
            - ((fglon[:, None] - obs_lon[None, :])/ds_oi_param.Lx.values)**2
            - ((fglat[:, None] - obs_lat[None, :])/ds_oi_param.Ly.values)**2
        )
    HBHt = np.exp(-((obs_time[:, None] - obs_time[None, :])/ds_oi_param.Lt.values)**2 -
                             ((obs_lon[:, None] - obs_lon[None, :])/ds_oi_param.Lx.values)**2 -
                             ((obs_lat[:, None] - obs_lat[None, :])/ds_oi_param.Ly.values)**2)

    del obs_lon, obs_lat, obs_time

    R = np.diag(np.full((nobs), ds_oi_param.noise.values**2))

    Coo = HBHt + R
    Mi = np.linalg.inv(Coo)
    sol = np.dot(np.dot(BHt, Mi), np.ravel(ds_obs.ssh_model.values)[obs])

    ds_oi_grid.gssh[it, :, :] = sol.reshape(ds_oi_grid.lat.size, ds_oi_grid.lon.size)
    ds_oi_grid.nobs[it] = nobs
    return locals()


def prepare_oi_batch(
        obs_values, obs_time, obs_lon, obs_lat,
        c_time, c_lon, c_lat,
        ps={'time':1, 'lon': 2.5, 'lat': 2.5},
        Lt=7.,
        Lx=1.,
        Ly=1.,
    ):
    grid_time, grid_lon, grid_lat = (x.flatten() for x in torch.meshgrid(c_time, c_lon, c_lat))
    def gen_patch():
        c_time_start = c_time.min()
        while c_time_start <= c_time.max():
            c_lon_start = c_lon.min()
            while c_lon_start <= c_lon.max():
                c_lat_start = c_lat.min()
                while c_lat_start <= c_lat.max():
                    yield (
                            (c_time_start, c_time_start + ps['time']),
                            (c_lon_start, c_lon_start + ps['lon']),
                            (c_lat_start, c_lat_start + ps['lat']),
                    )
                    c_lat_start += ps['lat']
                c_lon_start += ps['lon']
            c_time_start += ps['time']

    for bounds in gen_patch():
        (ts, te), (los, loe), (las, lae) = bounds
        print(f'{(ts.item(), los.item(), las.item())=}')
        msk_grid = (grid_time.ge(ts) & grid_time.lt(te)) & (grid_lon.ge(los) & grid_lon.lt(loe)) & (grid_lat.ge(las) & grid_lat.lt(lae))
        msk_obs = (
                (obs_time.ge(ts - 2 * Lt) & obs_time.lt(te + 2 * Lt))
                & (obs_lon.ge(los - 2 * Lx) & obs_lon.lt(loe +  2* Lx))
                & (obs_lat.ge(las - 2 * Ly) & obs_lat.lt(lae + 2 * Ly))
        )

        yield (
            obs_values[msk_obs],
            obs_time[msk_obs],
            obs_lat[msk_obs],
            obs_lon[msk_obs],
            grid_time[msk_grid],
            grid_lat[msk_grid],
            grid_lon[msk_grid],
        )

def torch_oi(
        obs_values,
        obs_time,
        obs_lon,
        obs_lat,
        grid_time,
        grid_lon,
        grid_lat,
        Lt=7.,
        Lx=1.,
        Ly=1.,
        ps={'time': 1, 'lat': 12, 'lon': 12},
        noise=0.05,
    ):
    BHt = torch.exp(
            - ((grid_time[:, None] - obs_time[None, :]) / Lt)**2
            - ((grid_lon[:, None] - obs_lon[None, :]) / Lx)**2
            - ((grid_lat[:, None] - obs_lat[None, :]) / Ly)**2
        )
    HBHt = torch.exp(-((obs_time[:, None] - obs_time[None, :]) / Lt)**2 -
                             ((obs_lon[:, None] - obs_lon[None, :]) / Lx)**2 -
                             ((obs_lat[:, None] - obs_lat[None, :]) / Ly)**2)

    nobs = len(obs_time)
    # print(f'{nobs=}')
    # print(f'{obs_time=}')
    # print(f'{grid_time=}')
    # print(f'{HBHt=}')
    # print(f'{BHt=}')
    R = torch.diag(torch.full((nobs,), noise**2, device=grid_time.device))

    
    Coo = HBHt + R
    print(f'{Coo.isnan().sum()=}')
    Mi = torch.linalg.inv(Coo)
    # print(f'{Mi=}')
    Iw = torch.mm(BHt, Mi).float()
    # print(Iw)
    # print(obs_values)
    sol = torch.mv(Iw, obs_values)
    # print(sol)
    return sol

def main():
    try:
        device = 'cuda:0'
        # device = 'cpu'
        lon_min = -65.                                           # domain min longitude
        lon_max = -55.                                           # domain max longitude
        lat_min = 33.                                            # domain min latitude
        lat_max = 43.                                            # domain max latitude
        simu_start_date = np.datetime64('2012-10-01')                # domain min time
        dt = np.timedelta64(1, 'D')                           # temporal grid step
        time_min = np.datetime64('2012-10-22')
        time_max = np.datetime64('2012-12-02')
        dx = 0.2                                                 # zonal grid spatial step (in degree)
        dy = 0.2                                                 # meridional grid spatial step (in degree)


        glon = torch.arange(lon_min, lon_max + dx, dx).to(device)           # output OI longitude grid
        glat = torch.arange(lat_min, lat_max + dy, dy).to(device)           # output OI latitude grid
        gtime = torch.arange(
            (time_min - simu_start_date) / dt,
            (time_max - simu_start_date) / dt, 1
        ).to(device)        # output OI time grid

        # OI parameters
        Lx = 1.                                                  # Zonal decorrelation scale (in degree)
        Ly = 1.                                                  # Meridional decorrelation scale (in degree)
        Lt = 7.                                                  # Temporal decorrelation scale (in days)
        noise = 0.05                                             # Noise level (5%)

        path_oi_4nadir = '../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_4nadir.nc'
        path_ref_daily = '../sla-data-registry/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc'
        duacs_ds = xr.open_dataset(path_oi_4nadir)
        ref_daily_ds = xr.open_mfdataset(path_ref_daily)
        ref_daily_ds.time.attrs['units'] = 'seconds since 2012-10-01'
        ref_daily_ds = xr.decode_cf(ref_daily_ds)


        inputs = '../sla-data-registry/CalData/cal_data_new_errs.nc'


        ds_obs = (
                xr.open_dataset(inputs)
                .sel(lat=slice(lat_min - 2*Ly, lat_max + 2*Ly))
                .sel(lon=slice(lon_min - 2*Lx, lon_max + 2*Lx))
                .sel(time=slice(time_min - 2*Lt*dt, time_max + 2*Lt * dt))
        )

        full_outs = {}
        for obs_var in [
                # 'swot_nadirs_new_errors_w_wet_tropo',
                # 'four_nadirs',
                # 'five_nadirs',
                'swot_nadirs_old_errors',
                'swot_nadirs_no_noise',
                'swot_nadirs_new_errors_no_wet_tropo',
                ]:
        # obs_var = 'four_nadirs' 
            obs = np.flatnonzero(np.isfinite(ds_obs[obs_var].values))

            obs_values = torch.from_numpy(np.ravel(ds_obs[obs_var].values)[obs]).to(device)
            obs_lon = torch.from_numpy(np.ravel(ds_obs.lon.broadcast_like(ds_obs[obs_var]).values)[obs]).to(device)
            obs_lat = torch.from_numpy(np.ravel(ds_obs.lat.broadcast_like(ds_obs[obs_var]).values)[obs]).to(device)
            obs_time = torch.from_numpy((np.ravel(ds_obs.time.broadcast_like(ds_obs[obs_var]).values)[obs] - simu_start_date) / dt).float().to(device)

            outputs = []
            t0 = time.time()
            with torch.no_grad():
                for batch in prepare_oi_batch(
                        obs_values, obs_time, obs_lon, obs_lat,
                        gtime, glon, glat,
                        {'time': 1, 'lon': 1, 'lat': 1},
                        Lt, Lx, Ly):
                    print(f'{len(batch[0])=}')
                    # if 'cuda' in device:
                    #     print(torch.cuda.memory_summary(device))
                    #     print(torch.cuda.list_gpu_processes(device))
                    torch.cuda.empty_cache()
                    sol = torch_oi(
                        *batch,
                        Lt, Lx, Ly, noise
                    )
                    outputs.append((*(x.detach().cpu() for x in batch), sol.cpu()))
                    # break
            print(time.time() - t0)
            dfs = []
            for chunk in outputs:
                (*_, grid_time, grid_lat, grid_lon, grid_sol) = chunk
                dfs.append(
                    pd.DataFrame(
                        {
                            'ssh': grid_sol.numpy(),
                            'time': grid_time.numpy() * dt + simu_start_date,
                            'lat': grid_lat.numpy(),
                            'lon': grid_lon.numpy(),
                        }
                    )
                )

            out_ds = pd.concat(dfs).set_index(['time', 'lat', 'lon']).pipe(xr.Dataset.from_dataframe)
            ref_daily = ref_daily_ds.interp(out_ds[['time', 'lat', 'lon']].coords)
            duacs = duacs_ds.interp(out_ds[['time', 'lat', 'lon']].coords)

            duacs_mse = (duacs.ssh_mod - ref_daily.ssh).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
            mse = (out_ds.ssh - ref_daily.ssh).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
            full_outs[obs_var] = (out_ds, mse, mse/duacs_mse)

            print([(k, v[1], v[2]) for k,v in full_outs.items()])
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        print('Am I here')
        return locals()


def main_np():
    try:
        # OI Grid
        lon_min = -65.                                           # domain min longitude
        lon_max = -55.                                           # domain max longitude
        lat_min = 33.                                            # domain min latitude
        lat_max = 43.                                            # domain max latitude
        time_min = np.datetime64('2012-10-22')                # domain min time
        time_max = np.datetime64('2012-12-02')                # domain max time
        dx = 0.2                                                 # zonal grid spatial step (in degree)
        dy = 0.2                                                 # meridional grid spatial step (in degree)
        dt = np.timedelta64(1, 'D')                           # temporal grid step

        simu_start_date = '2012-10-01T00:00:00'                  # Nature run initial date

        glon = np.arange(lon_min, lon_max + dx, dx)           # output OI longitude grid
        glat = np.arange(lat_min, lat_max + dy, dy)           # output OI latitude grid
        gtime = np.arange(time_min, time_max + dt, dt)        # output OI time grid

        # OI parameters
        Lx = 1.                                                  # Zonal decorrelation scale (in degree)
        Ly = 1.                                                  # Meridional decorrelation scale (in degree)
        Lt = 7.                                                  # Temporal decorrelation scale (in days)
        noise = 0.05                                             # Noise level (5%)

        inputs = ['../sla-data-registry/NATL60/NATL/data_new/dataset_nadir_0d.nc']

        ds = xr.open_dataset(inputs[0])
        ds
        ds_oi1_param = oi_param(Lx, Ly, Lt, noise)
        ds_oi1_grid = oi_grid(glon, glat, gtime, simu_start_date)
        # Read input obs + discard a bit...
        coarsening = {'time': 1}
        ds_oi1_obs = read_obs(inputs, ds_oi1_grid, ds_oi1_param, simu_start_date, coarsening).rename({'ssh_mod': 'ssh_model'})
        for it in range(len(gtime)):
            oi_out = my_oi(it, ds_oi1_grid, ds_oi1_param, ds_oi1_obs)
            break
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        print('Am I here')
        return locals()

def scratch():
        full_outs = {}
        for obs_var in [
                'swot_nadirs_new_errors_w_wet_tropo',
                'four_nadirs',
                'five_nadirs',
                'swot_nadirs_old_errors',
                'swot_nadirs_no_noise',
                'swot_nadirs_new_errors_no_wet_tropo',
                ]:
            full_outs[obs_var] = xr.open_dataset(f'{obs_var}_oi.nc')


    list(full_outs.keys())
    date = '2012-10-27'
    from scipy import ndimage
    def sobel_grid(da):
        dlat = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lat')))) / 5
        dlon = da.pipe(lambda da:  da.groupby('time').apply(lambda da: ndimage.sobel(da, da.dims.index('lon')))) / 5
        return np.hypot(dlat, dlon)
    full_outs['four_nadirs'].ssh.sel(time=date).plot(figsize=(10,7))

    (full_outs['four_nadirs'].ssh.sel(time=date) - ref_daily.ssh.sel(time=date).compute().values).plot(figsize=(10,7))
    (full_outs['swot_nadirs_old_errors'].ssh.sel(time=date) - ref_daily.ssh.sel(time=date).compute().values).plot(figsize=(10,7))
    ref_daily.ssh.sel(time=date).plot(figsize=(10,7))
    ref_daily.ssh.sel(time=slice(date, date)).pipe(sobel_grid).sel(time=date).plot(figsize=(10,7))
    ref_daily_ds.ssh.sel(time=slice(date, date), **ds_oi_grid.pipe(lambda d: {'lat':slice(d.lat.min(), d.lat.max()), 'lon':slice(d.lon.min(), d.lon.max())})).pipe(sobel_grid).sel(time=date).plot(figsize=(10,7))
    ref_daily_ds.ssh.sel(time=slice(date, date), **ds_oi_grid.pipe(lambda d: {'lat':slice(d.lat.min(), d.lat.max()), 'lon':slice(d.lon.min(), d.lon.max())})).sel(time=date).plot(figsize=(10,7))
    full_outs['swot_nadirs_old_errors'].ssh.sel(time=date).plot(figsize=(10,7))
    full_outs['four_nadirs'].ssh.pipe(sobel_grid).sel(time=date).plot(figsize=(10,7))
    full_outs['swot_nadirs_old_errors'].ssh.pipe(sobel_grid).sel(time=date).plot(figsize=(10,7))

    inputs = '../sla-data-registry/CalData/cal_data_new_errs.nc'
    ds_obs = (
            xr.open_dataset(inputs)
            .sel(lat=slice(lat_min - 2*Ly, lat_max + 2*Ly))
            .sel(lon=slice(lon_min - 2*Lx, lon_max + 2*Lx))
            .sel(time=slice(time_min - 2*Lt*dt, time_max + 2*Lt * dt))
    )

    ds_obs['four_nadirs'].isel(time=19).plot(figsize=(10,7))
    ds_obs['swot_nadirs_old_errors'].isel(time=19).plot(figsize=(10,7))
    ds_obs['swot_nadirs_old_errors'].isel(time=19).plot(figsize=(10,7))
    path_oi_4nadir = '../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_4nadir.nc'

    (
            ds_obs
            .pipe(lambda ds: ds.swot_nadirs_old_errors - ds.swot_no_noise)
            .pipe(lambda da: da.where(np.isfinite(ds_obs.swot_no_noise.values), np.full_like(da,np.nan)))
            .sel(time=date).plot(figsize=(10,7))
    )
    for k, v in full_outs.items():
        print(k, v[1].item(), v[2].item())

    s = ...

    simu_start_date = s['simu_start_date']
    dt = s['dt']
    outs = s['outputs']
    outs[0]

    full_outs = s['full_outs']
    for k, v in full_outs.items():

        print('dumping', k)
        v[0].to_netcdf(f'{k}_oi.nc')

    dfs = []
    for chunk in outs:
        (*_, grid_time, grid_lat, grid_lon, grid_sol) = chunk
        dfs.append(
            pd.DataFrame(
                {
                    'ssh': grid_sol.numpy(),
                    'time': grid_time.numpy() * dt + simu_start_date,
                    'lat': grid_lat.numpy(),
                    'lon': grid_lon.numpy(),
                }
            )
        )

    out_ds = pd.concat(dfs).set_index(['time', 'lat', 'lon']).pipe(xr.Dataset.from_dataframe)
    out_ds.isel(time=0).ssh.plot()

    oi_ds = xr.open_dataset(path_oi_4nadir)
    ref_inst_ds = xr.open_dataset(path_ref_inst)
    ref_inst_ds.time.attrs['units'] = 'seconds since 2012-10-01'
    ref_inst_ds= xr.decode_cf(ref_inst_ds)
    ref_daily_ds = xr.open_mfdataset(path_ref_daily)
    ref_daily_ds.time.attrs['units'] = 'seconds since 2012-10-01'
    ref_daily_ds= xr.decode_cf(ref_daily_ds)

    duacs = oi_ds.interp(out_ds[['time', 'lat', 'lon']].coords)
    ref_daily = ref_daily_ds.interp(out_ds[['time', 'lat', 'lon']].coords)
    ref_inst = ref_inst_ds.interp(out_ds[['time', 'lat', 'lon']].coords)

    duacs.isel(time=0).ssh_mod.plot()
    ref_daily.isel(time=0).ssh.plot()

    (out_ds.ssh - ref_daily.ssh).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
    (duacs.ssh_mod - ref_daily.ssh).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()

    (out_ds.ssh - ref_inst.ssh).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
    (duacs.ssh_mod - ref_inst.ssh).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()

    s['ds']
    ds_oi1_param = s['ds_oi1_param']
    ds_oi1_grid = s['ds_oi1_grid']
    ds_oi1_obs = s['ds_oi1_obs']
    np.ravel(ds_oi1_obs.ssh_model)
    ds_oi1_obs = ds_oi1_obs.load() 
    

    init_grid = ds_oi1_grid.isel(gtime=1)
    obstimes = np.nonzero(np.abs(ds_oi1_obs.time.values - init_grid.gtime.item()) < 2.*ds_oi1_param.Lt.item())
    
    path_oi_4nadir = '../sla-data-registry/NATL60/NATL/oi/ssh_NATL60_4nadir.nc'
    path_ref_inst = '../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc'
    path_ref_daily = '../sla-data-registry/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc'
    
    oi_ds = xr.open_dataset(path_oi_4nadir)
    ref_inst_ds = xr.open_dataset(path_ref_inst)
    ref_inst_ds.time.attrs['units'] = 'seconds since 2012-10-01'
    ref_inst_ds= xr.decode_cf(ref_inst_ds)
    ref_daily_ds = xr.open_mfdataset(path_ref_daily)
    ref_daily_ds.time.attrs['units'] = 'seconds since 2012-10-01'
    ref_daily_ds= xr.decode_cf(ref_daily_ds)
    duacs = oi_ds.interp(ds_oi1_grid[['time', 'lat', 'lon']].coords).isel(time=0).ssh_mod
    ref_daily = ref_daily_ds.interp(ds_oi_grid[['time', 'lat', 'lon']].coords)
    new_oi = ds_oi1_grid.isel(time=0).gssh
    new_oi.plot()
    (new_oi - duacs).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
    inst_ref = ref_inst_ds.interp(ds_oi1_grid[['time', 'lat', 'lon']].coords).isel(time=0).ssh
    daily_ref = ref_daily_ds.interp(ds_oi1_grid[['time', 'lat', 'lon']].coords).isel(time=0).ssh
    (inst_ref - daily_ref).plot()
    (new_oi - inst_ref).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
    (duacs - inst_ref).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()

    (new_oi - daily_ref).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()
    (duacs - daily_ref).pipe(lambda da: np.sqrt(np.mean(da**2))).compute()

    obs = np.flatnonzero(np.isfinite(ds_oi1_obs.isel(time=obstimes[0]).ssh_model.values))
    len(obs)
    dse_oi1_obs
