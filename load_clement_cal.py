import numpy as np
import importlib
import xarray as xr
import glob
import matplotlib.pyplot as plt

t_slice = slice(None, 300, None)
err1d_files = sorted(glob.glob('../clement_cal_all_err/err_gt/MYSWOT_1D*'))
err2d_files = sorted(glob.glob('../clement_cal_all_err/err_gt/MYSWOT_2D*'))
ds_errs = xr.open_mfdataset(err_files).isel(time=t_slice).load()

xac = ds_errs.xac.broadcast_like(ds_errs.karin_noise)
roll = ds_errs['roll'].broadcast_like(ds_errs.karin_noise) * xac * 1e-3

phase1 = ds_errs['phase1'].broadcast_like(ds_errs.karin_noise) * xac
phase2 = ds_errs['phase2'].broadcast_like(ds_errs.karin_noise) * xac
phase =  xr.apply_ufunc(lambda da: np.where(xac>0, da, phase2), phase1)* 1e-3

timing1 = ds_errs['timing1'].broadcast_like(ds_errs.karin_noise)
timing2 = ds_errs['timing2'].broadcast_like(ds_errs.karin_noise)
timing =  xr.apply_ufunc(lambda da: np.where(xac>0, da, timing2), timing1)

bd = ds_errs['bd'].broadcast_like(ds_errs.karin_noise) * xac**2

karin = ds_errs['karin_noise']

roll.T.plot(figsize=(10,3))
phase.T.plot(figsize=(10,3))
timing.T.plot(figsize=(10,3))
bd.T.plot(figsize=(10,3))
karin.T.plot(figsize=(10,3))

total_err = roll + phase + timing + bd + karin
total_err.T.plot(figsize=(10,3))

# Estimation

err_est_files =  sorted(glob.glob('../clement_cal_all_err/err_est/*'))
ds_err_est = xr.open_mfdataset(err_est_files[:2]).isel(time=t_slice).load()


lin = ds_err_est['lin'].broadcast_like(ds_errs.karin_noise) * xac
alin = ds_err_est['alin'].broadcast_like(ds_errs.karin_noise) * np.abs(xac)
linear =  (lin + alin) * 1e-3

const = ds_err_est['cst'].broadcast_like(ds_errs.karin_noise)
aconst = ds_err_est['acst'].broadcast_like(ds_errs.karin_noise) * np.sign(xac)
constant =  (const + aconst)

quad = ds_err_est['quad'].broadcast_like(ds_errs.karin_noise) * xac**2

linear.T.plot(figsize=(10,3))
constant.T.plot(figsize=(10,3))
quad.T.plot(figsize=(10,3))
total_est = linear + constant + quad
total_est.T.plot(figsize=(10,3))

(total_est - total_err).T.plot(figsize=(10,3))


# %% Convert to zarr
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
# %%
from tqdm import tqdm

if __name__ =='__main__':
    # %% store SWOT
    store = zarr.DirectoryStore('../clement_cal_all_err/zarr/err')
    xr.Dataset().to_zarr(store, mode="w", consolidated=True)

    with xr.set_options(file_cache_maxsize=512):
        files_per_chunk = 32
        err1d_files = sorted(glob.glob('../clement_cal_all_err/err_gt/MYSWOT_1D*'))
        err2d_files = sorted(glob.glob('../clement_cal_all_err/err_gt/MYSWOT_2D*'))
        err_est_files =  sorted(glob.glob('../clement_cal_all_err/err_est/*'))
        print(len(err2d_files))
        print(len(err1d_files))
        print(len(err_est_files))
        for chunk in tqdm(range(len(err1d_files) // files_per_chunk + 1)):
            with xr.open_mfdataset(
                    err1d_files[chunk * files_per_chunk: (chunk + 1) * files_per_chunk] +
                    err2d_files[chunk * files_per_chunk: (chunk + 1) * files_per_chunk],
                    chunks={"time": 1000000},
                    parallel=True,
                    compat='override',
                    coords=['time'],
            ) as ds_errs:
                ds_errs.time.attrs['units'] = 'days since 2012-10-01'
                ds_errs = xr.decode_cf(ds_errs).load()
                xac = ds_errs.xac.broadcast_like(ds_errs.karin_noise)
                roll = ds_errs['roll'].broadcast_like(ds_errs.karin_noise) * xac * 1e-3

                phase1 = ds_errs['phase1'].broadcast_like(ds_errs.karin_noise) * xac
                phase2 = ds_errs['phase2'].broadcast_like(ds_errs.karin_noise) * xac
                phase =  xr.apply_ufunc(lambda da: np.where(xac>0, da, phase2), phase1)* 1e-3

                timing1 = ds_errs['timing1'].broadcast_like(ds_errs.karin_noise)
                timing2 = ds_errs['timing2'].broadcast_like(ds_errs.karin_noise)
                timing =  xr.apply_ufunc(lambda da: np.where(xac>0, da, timing2), timing1)

                bd = ds_errs['bd'].broadcast_like(ds_errs.karin_noise) * xac**2

                karin = ds_errs['karin_noise']

                with xr.open_mfdataset(
                        err_est_files[chunk * files_per_chunk: (chunk + 1) * files_per_chunk],
                        chunks={"time": 1000000},
                        parallel=True,
                        compat='override',
                        coords=['time'],
                ) as ds_err_est:
                
                    ds_err_est.time.attrs['units'] = 'days since 2012-10-01'
                    ds_err_est = xr.decode_cf(ds_err_est).load()
                    lin = ds_err_est['lin'].broadcast_like(ds_errs.karin_noise) * xac
                    alin = ds_err_est['alin'].broadcast_like(ds_errs.karin_noise) * np.abs(xac)
                    linear =  (lin + alin) * 1e-3

                    const = ds_err_est['cst'].broadcast_like(ds_errs.karin_noise)
                    aconst = ds_err_est['acst'].broadcast_like(ds_errs.karin_noise) * np.sign(xac)
                    constant =  (const + aconst)

                    quad = ds_err_est['quad'].broadcast_like(ds_errs.karin_noise) * xac**2
                    total_err = xr.Dataset(
                            {
                                'roll': roll,
                                'phase': phase,
                                'timing': timing,
                                'bd': bd,
                                'karin': karin,
                                'linear': linear,
                                'constant': constant,
                                'quad': quad,
                            }
                    )
                
                for day, day_ds in total_err.resample(time='1D'):
                    dt = pd.to_datetime(day)
                    group = f"{dt.year}/{dt.month}"
                    print(group)
                    day_ds.map(lambda da: da.astype('float32', casting='same_kind')).to_zarr(store=store, consolidated=True,
                                                                                             group=group, append_dim="time")



# %%
groups = [f"{dt.year}/{dt.month}" for dt in pd.date_range("2012-10-01", "2013-4-01", freq='MS')]
for g in tqdm(groups):
    ds = xr.open_zarr(zarr.DirectoryStore('../clement_cal_all_err/zarr/err'), group=g)
    encodings = {
        v: {'chunks':  (100000, 52)} for v in ds.variables
    }
    ds.chunk(100000).to_zarr(zarr.DirectoryStore('../clement_cal_all_err/zarr/errfin'), group=g, consolidated=True, mode="w", encoding=encodings)

ds = xr.open_zarr(zarr.DirectoryStore('../clement_cal_all_err/zarr/errfin'), group='2013/1')
(
        ds.pipe(lambda ds: ds.isel(time=~np.isnan(ds.time)))
        .isel(time=slice(None, 10000)).load()
)
np.isnan(ds.time).sum()
err = (
    ds.assign(
        total_err=lambda ds: ds['roll'] + ds.phase + ds.karin + ds.bd + ds.timing,
        total_est=lambda ds: ds.linear + ds.quad + ds.constant,
        ).assign(
            err_cal = lambda ds: ds.total_err - ds.total_est
            ).compute()
        # .pipe(lambda ds: (ds.err_cal**2).sum()).compute()
)
# %%
