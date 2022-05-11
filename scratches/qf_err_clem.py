from dask.diagnostics import ProgressBar
from pathlib import Path
import zarr
from tqdm import tqdm
import traceback
import pandas as pd
import numpy as np
import time
import xarray as xr
from calibration import get_slice
import importlib
print('Done importing')

def sync_read(dates, swot_files, err_files):

    zarr_dir = Path('../sla-data-registry/sensor_zarr/zarr')
    store = zarr.DirectoryStore(zarr_dir / 'tmp_err_swot')
    xr.Dataset().to_zarr(store, mode="w", consolidated=True)
    ds_err = xr.open_dataset(err_files.pop(0))
    ds_swot = xr.open_dataset(swot_files.pop(0))
    swot_t0 = ds_swot.time.min().values
    err_t0 = ds_err.time.min().values
    prev_group =''
    for date in dates:
        print(date)
        while len(swot_files) > 0 and (ds_swot.time.max() - ds_swot.time.min()) < 30 * 3600 :
            ds_swot = xr.concat([ds_swot, xr.open_dataset(swot_files.pop(0))], dim='time')

        while len(err_files) > 0 and (ds_err.time.max() - ds_err.time.min()) < 30 * 3600 :
            ds_err = xr.concat([ds_err, xr.open_dataset(err_files.pop(0))], dim='num_lines')


        swot_chunk = (
                ds_swot.assign_coords(
                    dt=(('time',), pd.to_datetime('2012-10-01') + pd.to_timedelta(ds_swot.time, 's'))
                ).swap_dims({'time':'dt'})
                .sel(dt=str(date.date()))
        )

        err_chunk = (
            ds_err.assign_coords(
                dt=(('num_lines',), pd.to_datetime('2012-10-01') + pd.to_timedelta(ds_err.time - err_t0 + swot_t0, 's'))
            ).swap_dims({'num_lines':'dt'})
            .assign_coords(_x_ac=lambda ds: (('num_pixels',),ds.xac.isel(dt=0).values))
            .swap_dims({'num_pixels': '_x_ac'})
            .sel(_x_ac=np.concatenate([np.arange(-60, -8, 2), np.arange(10, 62, 2)]) * 1000, method='nearest')
            .sel(dt=str(date.date()))
        )
        
        add_vars = {
            v: (swot_chunk.ssh_model.dims, err_chunk[v].interp(dt=swot_chunk.dt, method='nearest').values)
            for v in ['karin_noise', 'wet_tropo_res', 'syst_error_uncalibrated']
        }
        chunk =  (
            swot_chunk
            .assign( **add_vars)
        )
        group = f"{date.year}/{date.month}"
        # print(day_ds.time)
        # xr.Dataset.to_zarr
        if prev_group != group:
            chunk.to_zarr(store=store, group=group, mode='a')
        else:
            chunk.to_zarr(store=store, group=group, append_dim="dt")
        prev_group = group
        ds_swot = ds_swot.drop_sel(time=swot_chunk.time)
        ds_err = ds_err.drop_isel(num_lines=ds_err.time <= err_chunk.time.max().values)

#


def compress_and_clean(src_folder, tgt_folder):
    src_store = zarr.DirectoryStore(src_folder)
    tgt_store = zarr.DirectoryStore(tgt_folder)
    xr.Dataset().to_zarr(tgt_store, mode="w", consolidated=True)
    groups = [f"{dt.year}/{dt.month}" for dt in pd.date_range("2012-10-01", "2013-10-01", freq='MS')]
    for g in tqdm(groups):
        ds = xr.open_zarr(src_store, group=g)
        new_ds = (
            ds
            .chunk({'dt': 1000000, 'nC': 52})
            .isel(dt=np.isfinite(ds.ssh_model).any('nC'))
            .astype('float32', casting='same_kind')
            .drop('time')
            .rename({'dt': 'time'})
        )
            

        new_ds.chunk({'time': 1000000, 'nC': 52}).to_zarr(tgt_store, group=g, consolidated=True, mode="w")


def main():
    try:

        zarr_dir = Path('../sla-data-registry/sensor_zarr/zarr')
        src_folder = zarr_dir / 'tmp_err_swot'
        tgt_folder = zarr_dir / 'new_swot'

        all_2d_files = sorted(list(Path('swot_errors').glob('*2DERROR*')))
        swot_files = sorted(list(Path('output_SWOTsimulator/swot_HD').glob('BOOST-SWOT_SWOT_c*_p*.nc')))
        all_2d_files = sorted(list(data_path.glob('*2DERROR*')))
        swot_files = swot_files
        err_files = all_2d_files
        dates = pd.date_range('2012-10-01', '2013-09-30')

        print("writing raw zarr")
        sync_read(dates=dates, swot_files=swot_files, err_files=err_files)

        print("writing raw zarr")
         compress_and_clean(src_folder, tgt_folder):
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        print('Am I here')
        return locals()
