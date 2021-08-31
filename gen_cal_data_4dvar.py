# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ### Generate data cal
# - obs 4nadir
# - obs swot
# - roll swot
#

# %%
import sys
sys.path.append('../research-quentin')

# %%
import pandas as pd
import pandas as pd
import numpy as np
import xarray as xr

import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.commons.coords_to_dim import coords_to_dim, reindex
from src.data_processing.get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice

dates = pd.date_range("2012-10-01", "2013-10-01", freq='1D')



# %%
slice_args = {
    "time_min": dates[0],
    "time_max": dates[1],
}
natl = get_natl_slice('../research-quentin/data/raw/NATL60_regular_grid/1_10/natl60CH_H.nc', **slice_args)

natl['lon'] =  (natl['lon'] + 180)  % 360 - 180
natl = natl.sortby('lon')

reindex(natl, ('time', 'lat', 'lon')).isel(time=0).H.plot()



# %%
res = dict(
    time= np.timedelta64(1, 'D') / np.timedelta64(1, 'ns'),
    lat=1/20,
    lon=1/20,
)

full_lon_dim = np.arange(
    float(natl.lon.min()) / res['lon'] // 1 * res['lon'],
    float(natl.lon.max()) / res['lon'] // 1 * res['lon'],
    res['lon'],
)

full_lat_dim = np.arange(
    float(natl.lat.min()) / res['lat'] // 1 * res['lat'],
    float(natl.lat.max()) / res['lat'] // 1 * res['lat'],
    res['lat'],
)

def grid_ds(ds, resolutions=res):
    return (
        ds.assign(
            **{
                f'grid_{coord}': ds[coord].astype(float) / resolutions[coord] // 1 *
                                 resolutions[coord]
                for coord in resolutions.keys()
            }
        ).to_dataframe()
            .reset_index()
            .groupby([f'grid_{coord}' for coord in resolutions.keys()])
            .mean(numeric_only=True)
            .pipe(lambda df: xr.Dataset.from_dataframe(df))
    )


# %%
from pathlib import Path

import functools as ft
Path('obs').mkdir(exist_ok=True)

# %%
gen_files = {
     # 'nad' : {         
     #    'nadirs': ['ssh_model'],
    # },
     # 'nad_swot' : {         
     #    'swot': ['ssh_model'],
     #    'nadirs': ['ssh_model'],
    # },
    'nad_swot_roll_phase_bd_timing' : {         
        'swot': ['ssh_model', 'roll_err', 'phase_err', 'bd_err', 'timing_err'],
        'nadirs': ['ssh_model'],
    },
    # 'nad_swot_roll' : {         
    #     'swot': ['ssh_model', 'roll_err'],
    #     'nadirs': ['ssh_model'],
    # },
    # 'swot_roll' : {         
    #     'swot': ['ssh_model', 'roll_err'],
    # },
    # 'swot' : {         
    #     'swot': ['ssh_model'],
    # }
}
dates = pd.date_range("2012-10-01", "2013-10-01", freq='1D')

for dt_start, dt_end in zip(dates[:-1], dates[1:]):
    print(dt_start, dt_end, end='\r')
    slice_args = {
        "time_min": dt_start,
        "time_max": dt_end,
        "lat_min": float((natl.lat % 360).min()),
        "lat_max": float((natl.lat % 360).max()),
        "lon_min": float((natl.lon % 360).min()) ,
        "lon_max": float((natl.lon % 360).max()),
    }

    raw_item = {
        'nadirs': xr.concat([get_nadir_slice(f'../research-quentin/data/zarr/nadir/{name}', **slice_args) for name in
                             ['swot', 'en', 'tpn', 'g2', 'j1']], dim='time'),
        'swot': get_swot_slice(f'../research-quentin/data/zarr/swot', **slice_args),
    }
    
    grideds = []
    for tgt_var in gen_files: 
        grideds.append(
            pd.concat(
                [
                    raw_item[src]
                    .assign({tgt_var: lambda ds, src_vars=src_vars: ft.reduce(lambda dsv, dsw: dsv + dsw, [ds[_] for _ in src_vars])})
                    .to_dataframe()
                    .reset_index()
                    [['time', 'lat', 'lon', tgt_var]]
                    .dropna(subset=[tgt_var])
                for src, src_vars in gen_files[tgt_var].items()
                ], axis=0, ignore_index=True,
            )
            .pipe(xr.Dataset.from_dataframe)
            .pipe(
                lambda ds: grid_ds(
                    ds.pipe(lambda ds: ds.update(dict(lon=(ds.lon+ 180) % 360 - 180)))
                    .pipe(lambda ds: ds.isel(index=ds.lon>=natl.lon.min()))
                    .pipe(lambda ds: ds.isel(index=ds.lon<=natl.lon.max()))
                )
            ).drop(('lat', 'lon', 'index'))
            .rename({
                    'grid_lat': 'lat',
                    'grid_lon': 'lon',
                    'grid_time': 'time',
            }).assign_coords({
                    'time': [dt_start + (dt_end - dt_start) / 2],
            }).reindex(
                {
                    'lat': full_lat_dim,
                    'lon': full_lon_dim,
                },
            method='nearest', tolerance=10**-10
            )
        )
    
    xr.merge(grideds).to_netcdf(f'obs/{dt_start.date()}.nc')


# %%
xr.open_mfdataset('obs/*').to_netcdf('full_cal_obs.nc')

raise Exception('something else')
# %%
xr.open_dataset('full_cal_obs.nc').swot.isel(time=0).plot()

# %% [markdown]
# # TO dump :
# **obs :**
#  - nadirs
#  - nadirs + SWOT
#  - nadirs + SWOT + roll
#
# **cov :**
#  - SWOT
#  - SWOT + roll

# %%
foo = (
    pd.concat(
        [
            raw_item[src]
            .assign({tgt_var: lambda ds, src_vars=src_vars: ft.reduce(lambda dsv, dsw: dsv + dsw, [ds[_] for _ in src_vars])})
            .to_dataframe()
            .reset_index()
            [['time', 'lat', 'lon', tgt_var]]
            .dropna(subset=[tgt_var])
        for src, src_vars in list(gen_files[tgt_var].items())[1:]
        ], axis=0, ignore_index=True,
    )
    .pipe(xr.Dataset.from_dataframe)
#     .pipe(
#         lambda ds: grid_ds(
#             ds.pipe(lambda ds: ds.update(dict(lon=(ds.lon+ 180) % 360 - 180)))
#             .pipe(lambda ds: ds.isel(index=ds.lon>=natl.lon.min()))
#             .pipe(lambda ds: ds.isel(index=ds.lon<=natl.lon.max()))
#         )
#     ).drop(('lat', 'lon', 'index'))
#     .rename({
#             'grid_lat': 'lat',
#             'grid_lon': 'lon',
#             'grid_time': 'time',
#     }).assign_coords({
#             'time': [dt_start + (dt_end - dt_start) / 2],
#     }).reindex(
#         {
#             'lat': full_lat_dim,
#             'lon': full_lon_dim,
#         },
#     method='nearest', tolerance=10**-10
#     )
)

# %%
slice_args

# %%
pd.to_timedelta(1, unit='nanoseconds')

# %%
# get_nadir_slice?

# %%
xr.concat([get_nadir_slice(f'../research-quentin/data/zarr/nadir/{name}', **slice_args) for name in
                             ['swot', 'en', 'tpn', 'g2', 'j1']], dim='time').time.max()

# %%
np.unique(pd.to_datetime(foo.time.astype(float).values/ resolutions['time'] // 1 * resolutions['time']))

# %%
resolutions=res
(
    foo
        .pipe(
        lambda ds: ds.assign(
            **{
                f'grid_{coord}': ds[coord].astype(float) / resolutions[coord] // 1 *
                                 resolutions[coord]
                for coord in resolutions.keys()
            }
        )
#             .to_dataframe()
#             .reset_index()
#             .groupby([f'grid_{coord}' for coord in resolutions.keys()])
#             .mean(numeric_only=True)
#             .pipe(lambda df: xr.Dataset.from_dataframe(df))
    )
    
)

# %%
raw_item['nadirs'].time

# %%
pd.to_datetime(foo.grid_time)

# %%
np.sum(~np.isnan(grideds[0].nad_swot))

# %%
list(gen_files[tgt_var].items())[1:]

# %%
obs = xr.merge(grideds)

# %%
obs.swot_roll.plot()

# %%
obs.map(lambda da: da.astype('float32', casting='same_kind')).reindex(
                {
                    'lat': full_lat_dim,
                    'lon': full_lon_dim,
                }
            ).nad_swot_roll.plot()

# %%
obs.nad_swot.plot()

# %%
xr.merge(grideds)

# %%
grided.swot.plot()

# %%

# %%



grided = (
    pd.concat(
        [
            raw_item[src]
            .assign({tgt_var: lambda ds, src_vars=src_vars: ft.reduce(lambda dsv, dsw: dsv + dsw, [ds[_] for _ in src_vars])})
            .to_dataframe()
            .reset_index()
            [['time', 'lat', 'lon', tgt_var]]
            .dropna(subset=[tgt_var])
        for src, src_vars in gen_files[tgt_var].items()
        ], axis=0, ignore_index=True,
    )
    .pipe(xr.Dataset.from_dataframe)
    .pipe(
        lambda ds: grid_ds(
            ds.pipe(lambda ds: ds.update(dict(lon=(ds.lon+ 180) % 360 - 180)))
            .pipe(lambda ds: ds.isel(index=ds.lon>=natl.lon.min()))
            .pipe(lambda ds: ds.isel(index=ds.lon<=natl.lon.max()))
        )
    )
)

# %%
(    pd.concat(
        [
            raw_item[src]
            .assign({tgt_var: lambda ds, src_vars=src_vars: ft.reduce(lambda v, w: ds[v] + ds[w], src_vars)})
            .to_dataframe()
            .reset_index()
            [['time', 'lat', 'lon', tgt_var]]
            .dropna(subset=[tgt_var])
        for src, src_vars in gen_files[tgt_var].items()
        ], axis=0, ignore_index=True,
    ))

# %%
grided

# %%
grided[tgt_var].plot(figsize=(25,10))

# %%
raw_item['swot'].reset_index()

# %%
old = xr.open_dataset('/gpfsscratch/rech/nlu/commun/large/ssh_NATL60_swot_4nadir.nc')

# %%
grid_nadir.dims

# %%
full_nad_ds = xr.open_mfdataset('nadirs/*')

# %%
old

# %%
len(nadir_dses)

# %%
nad_ds = xr.concat(nadir_dses, dim='time')

# %%
nad_ds.to_netcdf(target_nadir_ds)

# %%
nad_ds2 = xr.open_dataset(target_nadir_ds)

# %%
nad_ds2

# %%
grid_nadir.ssh_model.isel(grid_time=0).plot(figsize=(15, 10))

# %%
natl.lon.max()

# %%

swot = (
    raw_item['swot']
    .pipe(lambda ds: ds.update(
        dict(
            lon=(ds.lon+ 180) % 360 - 180,
            lon_nadir=(ds.lon_nadir+ 180) % 360 - 180),
    ))
    .pipe(lambda ds: ds.isel(time=ds.lon_nadir>=natl.lon.min()))
    .pipe(lambda ds: ds.isel(time=ds.lon_nadir<=natl.lon.max()))
)
grid_swot = grid_ds(swot)
grid_swot['grid_time'] =  pd.to_datetime(grid_swot['grid_time'])

# %%
grid_swot.ssh_model.isel(grid_time=0).plot(figsize=(15, 10))

# %%
grid_swot.roll_err.isel(grid_time=0).plot(figsize=(15, 10))

# %%
old = xr.open_dataset('/users/local/DATA/OSSE/GULFSTREAM/data/gridded_data_swot_wocorr/dataset_nadir_0d.nc').ssh_mod.isel(time=0)
old.plot()

# %%
natl

# %%
natl_tpn = reindex(natl, ('time', 'lat', 'lon')).H.interp(lat=nadirs.lat, lon=nadirs.lon, time=nadirs.time)

# %%
natl_tpn

# %%
natl_tpn.plot()

# %%
grid_nadir = grid_ds(natl_tpn.to_dataset().rename({'H': 'ssh_model'}))
grid_nadir['grid_time'] =  pd.to_datetime(grid_nadir['grid_time'])

# %%
ref

# %%
ref =  (grid_nadir.isel(grid_time=0)
    .sel(grid_lon=slice(-65, -55))
    .sel(grid_lat=slice(33, 43))).ssh_model
ref.plot()

# %%
(ref - new.values).plot()

# %%
(ref - old.values).plot()

# %%
paths = sorted(Path(path).glob(glob))
raw_ds = xr.open_mfdataset(
                    paths
            )

# %%
raw_ds.time

# %%
raw_ds.ssh_model.isel(time=slice(0, None, 500)).plot()

# %%
en_ds = xr.open_dataset('../data/raw/output_SWOTsimulator/en/BOOST-SWOT_EN_nadir_c01.nc')
tpn_ds = xr.open_dataset('../data/raw/output_SWOTsimulator/tpn/BOOST-SWOT_TPN_nadir_c01.nc')
g2_ds = xr.open_dataset('../data/raw/output_SWOTsimulator/g2/BOOST-SWOT_G2_nadir_c01.nc')
j1_ds = xr.open_dataset('../data/raw/output_SWOTsimulator/j1/BOOST-SWOT_J1_nadir_c01.nc')

# %%
en_ds.time.attrs.update(units="seconds since 2012-10-01")
tpn_ds.time.attrs.update(units="seconds since 2012-10-01")
g2_ds.time.attrs.update(units="seconds since 2012-10-01")
j1_ds.time.attrs.update(units="seconds since 2012-10-01")

# %%
fig, axs = plt.subplots(2,2, figsize=(15,5))
en_ds.pipe(xr.decode_cf).ssh_model.plot(ax=axs[0,0])
tpn_ds.pipe(xr.decode_cf).ssh_model.plot(ax=axs[1,0])
g2_ds.pipe(xr.decode_cf).ssh_model.plot(ax=axs[0,1])
j1_ds.pipe(xr.decode_cf).ssh_model.plot(ax=axs[1,1])

# %%
pd.to_timedelta(3.736615, 'D')

# %%
get_timeshift = lambda ds : pd.to_timedelta(float(ds.time.isel(time=(np.where(~np.isnan(ds.model_index))[0][0]))), 's')

# %%
en_ds.time

# %%
print(f"{en_ds.pipe(get_timeshift)=}")
print(f"{tpn_ds.pipe(get_timeshift)=}")
print(f"{g2_ds.pipe(get_timeshift)=}")
print(f"{j1_ds.pipe(get_timeshift)=}")




# %%
timeshift

# %%
pd.to_timedelta(timeshift, 'D') / pd.to_timedelta(1, 's')

# %%
for path, glob, name, timeshift in [
#         ('data/raw/output_SWOTsimulator/swot_HD/', 'BOOST-SWOT_SWOT_nadir_c*.nc', 'swot', 0),
    ('../data/raw/output_SWOTsimulator/en/', 'BOOST-SWOT_EN_nadir_c*.nc', 'en', 22.10114),
    ('../data/raw/output_SWOTsimulator/tpn/', 'BOOST-SWOT_TPN_nadir_c*.nc', 'tpn', 3.731883),
    ('../data/raw/output_SWOTsimulator/g2/', 'BOOST-SWOT_G2_nadir_c*.nc', 'g2', 15.08489),
    ('../data/raw/output_SWOTsimulator/j1/', 'BOOST-SWOT_J1_nadir_c*.nc', 'j1', 3.736615),
]:
    paths = sorted(Path(path).glob(glob))

    store = zarr.DirectoryStore(f'../data/zarr/tmp_nadir/{name}')
    xr.Dataset().to_zarr(store, mode="w", consolidated=True)

    with xr.set_options(file_cache_maxsize=512):
        files_per_chunk = 32
        paths = sorted(paths)
        for chunk in tqdm(range(len(paths) // files_per_chunk + 1)):
            with xr.open_mfdataset(
                    paths[chunk * files_per_chunk: (chunk + 1) * files_per_chunk],
                    chunks={"time": 1000000},
                    parallel=True,
                    compat='override',
                    coords=['time'],
            ) as ds:
                ds['time'] = ds.time - (pd.to_timedelta(timeshift, 'D') / pd.to_timedelta(1, 's'))
                ds.time.attrs['units'] = 'seconds since 2012-10-01'
                ds["ncycle"] = ds.ncycle.compute().astype(int) / 10 ** 9 / 86400
                ds = xr.decode_cf(ds)
                
                
                for day, day_ds in ds.resample(time='1D'):
                    day_ds = day_ds.pipe(
                        lambda _ds: _ds.isel(time=(~np.isnan(_ds.ssh_model)) & (~np.isnan(_ds.model_index)))
                    ).load()
                    if day_ds.dims['time'] == 0:
                        continue
                    dt = pd.to_datetime(day)
                    group = f"{dt.year}/{dt.month}"
                    day_ds.map(lambda da: da.astype('float32', casting='same_kind')).to_zarr(store=store,
                                                                                             consolidated=True,
                                                                                             group=group,
                                                                                             append_dim="time")


# %%
nadir = get_nadir_slice(f'../data/zarr/tmp_nadir/en', **slice_args)
(
    nadir
    .pipe(lambda ds: ds.isel(time=~np.isnan(ds.model_index)))
    .pipe(lambda ds: ds.update(dict(lon=(ds.lon+ 180) % 360 - 180)))
    .pipe(lambda ds: ds.isel(time=ds.lon>=natl.lon.min()))
    .pipe(lambda ds: ds.isel(time=ds.lon<=natl.lon.max()))
)
grid_nadir = grid_ds(nadirs)
grid_nadir['grid_time'] =  pd.to_datetime(grid_nadir['grid_time'])
grid_nadir.ssh_model.isel(grid_time=0).plot(figsize=(15, 10))

# %%
# %%
groups = [f"{dt.year}/{dt.month}" for dt in pd.date_range("2012-10-01", "2013-09-30", freq='MS')]
for g in tqdm(groups):
    for name in tqdm(['en', 'tpn', 'g2', 'j1',]):
        ds = xr.open_zarr(zarr.DirectoryStore(f'../data/zarr/tmp_nadir/{name}'), group=g)
        encodings = {
            v: {'chunks':  (1000000,)} for v in ds.variables
        }
        ds.chunk(1000000).to_zarr(zarr.DirectoryStore(f'../data/zarr/nadir/{name}'), group=g, consolidated=True, mode="w", encoding=encodings)



# %%
