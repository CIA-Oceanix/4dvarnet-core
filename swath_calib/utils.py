import numpy as np
import torch
import scipy.ndimage as ndi
import pandas as pd
import zarr
import xarray as xr
from scipy import fftpack
import xrft

def to_swath_data(grid_ds):

        slice_args = dict(
                # time_min='2012-10-01', 
                time_min= pd.to_datetime(np.min(grid_ds['time']).values).date(),
                # time_max='2012-10-30',
                time_max= pd.to_datetime(np.max(grid_ds['time']).values).date(),
                lat_min=grid_ds['lat'].min().item(),
                lat_max=grid_ds['lat'].max().item(),
                lon_min=grid_ds['lon'].min().item() + 360,
                lon_max=grid_ds['lon'].max().item() + 360,
        )

        swath_data = get_swot_slice('../tgrs-data/sensor_zarr/zarr/new_swot', **slice_args, drop_vars=[
         'bd_err', 'karin_noise', 'karin_err', 'phase_err', 'roll_err','timing_err',
         'model_index',
         ])

        swathed_grid = grid_ds.interp(
            time=swath_data.time.broadcast_like(swath_data.ssh_model),
            lat=swath_data.lat.broadcast_like(swath_data.ssh_model),
            lon=swath_data.lon.broadcast_like(swath_data.ssh_model) - 360,
        )
        swath_data = (
            swath_data.assign({v: (swath_data.ssh_model.dims, swathed_grid[v].data)for v in swathed_grid} )
            
        )
        return swath_data

def get_swot_slice(path, drop_vars=('model_index',),
                   **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-09-30"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path), drop_variables=drop_vars, group=group,
                          decode_times=False,
                          consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/swot.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64().astype(np.float64)
            te = (dt_end - pd.to_datetime(reference_date)).to_timedelta64().astype(np.float64)
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir > slice_args.get('lon_min', -360)))).compute()
            )

    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        # print(
        #     f"no data found at {path} for {slice_args} {groups} {pd.date_range(start=dt_start.replace(day=1), end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )


def get_nadir_slice(path, **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path),
                          group=group, decode_times=False, consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/nadir.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64() / pd.to_timedelta(1, unit=units.strip())
            te = (dt_end - pd.to_datetime(reference_date)) / pd.to_timedelta(1, unit=units.strip())
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(lambda ds: ds.isel(time=(ds.lat > slice_args.get('lat_min', -360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon > slice_args.get('lon_min', -360)))).compute()
            )
    dses = [_ds for _ds in dses if _ds.dims['time']]
    if len(dses) == 0:
        # print(
        #     f"no data at {path} found for {slice_args} {groups} {pd.date_range(start=dt_start, end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        [xr.decode_cf(_ds) for _ds in dses if _ds.dims['time']],
        dim="time"
    )

def generate_cal_xrds(ds, lit_mod, trainer, var_name='cal'):
    # ds, lit_mod, trainer, var_name=cal_ds, cal_mod, trainer, 'cal'
    pred_dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    predictions = [p for pred in trainer.predict(lit_mod, pred_dl) for p in pred]
    with ds.get_coords():
        coords = [
           ds[i]
           for i in range(len(ds))
        ]

    return xr.concat(
            [
                coord.assign({var_name: lambda ds: (ds.ssh_model.dims, pred)})#+ coord.xb.values)})
                for pred, coord in zip(predictions, coords)
            ], dim='time'
    )

def to_grid(swath_data, tgt_grid):
    import pyinterp
    ds = swath_data[['pred', 'lat', 'lon', 'time']]
    binning = pyinterp.Binning2D(pyinterp.Axis(tgt_grid.lon.values), pyinterp.Axis(tgt_grid.lat.values))
    grid_day_dses = []

    for t in tgt_grid.time:
        binning.clear()
        tds = ds.isel(time=pd.to_datetime(ds.time.values).date == pd.to_datetime(t.values).date())

        values = np.ravel(tds.pred.values)
        lons = np.ravel(tds.lon.values) - 360
        lats = np.ravel(tds.lat.values)
        msk = np.isfinite(values)
        binning.push(lons[msk], lats[msk], values[msk])
        gridded =  (('time', 'lat', 'lon'), binning.variable('mean').T[None, ...])
        grid_day_dses.append(
           xr.Dataset(
               {'gridded': gridded},
               {'time': [t.values], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
            ).astype('float32', casting='same_kind')
        )
    tgt_ds = xr.concat(grid_day_dses, dim='time')
    return tgt_ds
