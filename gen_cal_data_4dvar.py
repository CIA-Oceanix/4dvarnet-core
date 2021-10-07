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
import zarr
import xarray as xr

import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from src.commons.coords_to_dim import coords_to_dim, reindex
from src.data_processing.get_slice import get_nadir_slice, get_swot_slice, get_oi_slice, get_natl_slice

dates = pd.date_range("2013-01-01", "2013-01-30", freq='1D')
# dates = pd.date_range("2012-10-01", "2013-10-01", freq='1D')


def compute_duacs_cal(swot_ds, oi_ds):
    return  (
        swot_ds
        .assign(uncal=lambda ds: ds.ssh_model + ds.err)
        .assign(
            oi=lambda ds: (
                ds.ssh_model.dims, 
                oi_ds
                .pipe(lambda ds: coords_to_dim(ds, ('lat', 'lon'), drop=('x', 'y')))
                .assign({f'clean_ssh': lambda _ds: (_ds.ssh.dims, np.where(_ds.ssh < 10, _ds.ssh, np.nan))})
                .clean_ssh
                .interp(
                    time=ds.time.broadcast_like(ds.ssh_model),
                    lat=ds.lat.broadcast_like(ds.ssh_model),
                    lon=ds.lon.broadcast_like(ds.ssh_model),
                ).data
            )
        ).pipe(lambda ds: ds.isel(time = ~np.isnan(ds.oi).any('nC')))
        .assign(res=lambda ds: ds.uncal - ds.oi)
        .assign(conti_chunks=lambda ds: (ds.time.dims, np.cumsum(np.diff(ds.time.values.astype(float), prepend=0) > pd.to_timedelta(0.5, 's') / pd.to_timedelta(1, 'ns'))))
        .pipe(
            lambda ds: ds.assign(
                op = (
                    ds.res.dims,
                    slope(ds).to_dataframe()
                    .groupby('conti_chunks')
                    .apply(
                        lambda df: df.assign(op_slope=lambda df: df.res.rolling(225, win_type="gaussian", center=True, min_periods=1).mean(std=75))
                    )
                    .reset_index(level=0, drop=True)
                    .pipe(xr.Dataset.from_dataframe)
                    .pipe(lambda _ds: _ds.op_slope * ds.x_ac)
                )
            )
        )
    )


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
Path('../obs').mkdir(exist_ok=True)

# %%
gen_files = {
     # 'nad' : {         
     #    'nadirs': ['ssh_model'],
    # },
     # 'nad_swot' : {         
     #    'swot': ['ssh_model'],
     #    'nadirs': ['ssh_model'],
    # },
    'nad_swot_clem_err' : {         
        'swot': ['ssh_model', 'total_err'],
        'nadirs': ['ssh_model'],
    },
    'clem_est' : {         
        'swot': ['total_est'],
    },
    # 'nad_swot_roll_phase_bd_timing_karin' : {         
    #     'swot': ['ssh_model', 'roll_err', 'phase_err', 'bd_err', 'timing_err', 'karin_err'],
    #     'nadirs': ['ssh_model'],
    # },
    # 'roll_phase_bd_timing_karin' : {         
    #     'swot': ['roll_err', 'phase_err', 'bd_err', 'timing_err', 'karin_err'],
    # },
    # 'roll' : {         
    #     'swot': ['roll_err'],
    # },
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
# dates = pd.date_range("2012-10-01", "2013-10-01", freq='1D')

clem_ds = (
    xr.open_zarr(zarr.DirectoryStore('../clement_cal_all_err/zarr/errfin'), group='2013/1')
    .pipe(lambda ds: ds.isel(time=~np.isnan(ds.time)))
)
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
        'oi': get_oi_slice('../research-quentin/data/raw/DUACS-OI_maps/ssh_model/ssh_sla_boost_NATL60_en_j1_tpn_g2.nc', **slice_args),
    }
    clem_slice = clem_ds.interp(time=raw_item['swot'].time, method='nearest',  ).load()

    raw_item['swot'] = raw_item['swot'].assign(
            total_err=clem_slice['roll'] + clem_slice.phase + clem_slice.karin + clem_slice.bd + clem_slice.timing,
            total_est=clem_slice.linear + clem_slice.quad + clem_slice.constant,
    )
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
    
    xr.merge(grideds).to_netcdf(f'../obs/{dt_start.date()}.nc')


# %%
xr.open_mfdataset('../obs/*').to_netcdf('full_cal_obs.nc')

raise Exception('something else')
# %%
xr.open_dataset('full_cal_obs.nc').nad_swot_clem_err.isel(time=0).plot()
xr.open_dataset('full_cal_obs.nc').clem_est.isel(time=0).plot()

