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
     'nad' : {         
        'nadirs': ['ssh_model'],
    },
     'nad_swot' : {         
        'swot': ['ssh_model'],
        'nadirs': ['ssh_model'],
    },
    'nad_swot_roll' : {         
        'swot': ['ssh_model', 'roll_err'],
        'nadirs': ['ssh_model'],
    },
    'swot_roll' : {         
        'swot': ['ssh_model', 'roll_err'],
    },
    'swot' : {         
        'swot': ['ssh_model'],
    }
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
 %%
