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
        device = 'cuda:5'
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

        ds_obs['five_nadirs'].isel(time=0).plot()
        full_outs = {}
        for obs_var in [
                # 'swot_nadirs_new_errors_w_wet_tropo',
                # 'four_nadirs',
                'five_nadirs',
                # 'swot_nadirs_old_errors',
                # 'swot_nadirs_no_noise',
                # 'swot_nadirs_new_errors_no_wet_tropo',
                ]:
            # break
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
                        {'time': 1, 'lon': 12, 'lat': 12},
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

if __name__ =='__main__':
    locals().update(main())
