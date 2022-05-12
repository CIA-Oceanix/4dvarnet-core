# %%
from time import time

import pandas as pd
import xarray as xr
import zarr
# from numcodecs import blosc
#
# blosc.set_nthreads(16)


# %%
def get_nadir_slice(path, **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    ts = (dt_start - pd.to_datetime("2012-10-01")).total_seconds()
    te = (dt_end - pd.to_datetime("2012-10-01")).total_seconds()

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
                    .pipe(xr.decode_cf)
                    .pipe(lambda ds: ds.isel(time=(ds.lat > slice_args.get('lat_min', 0))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon > slice_args.get('lon_min', 0)))).compute()
            )
    if len(dses) == 0:
        print(
            f"no data found for {slice_args} {groups} {pd.date_range(start=dt_start.replace(day=1), end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        dses,
        dim="time"
    )


def get_swot_slice(path, drop_vars=('model_index',),
                   **slice_args):
    dt_start = pd.to_datetime(slice_args.get('time_min', "2012-10-01"))
    dt_end = pd.to_datetime(slice_args.get('time_max', "2013-10-01"))
    groups = [f"{dt.year}/{dt.month}" for dt in
              pd.date_range(start=dt_start.date().replace(day=1), end=dt_end, freq='MS')]

    dses = []
    for group in groups:
        with xr.open_zarr(zarr.DirectoryStore(path), drop_variables=drop_vars, group=group,
                          decode_times=False,
                          consolidated=True,
                          synchronizer=zarr.ProcessSynchronizer(f'data/swot.sync')) as ds:
            units, reference_date = ds.time.attrs['units'].split('since')
            ts = (dt_start - pd.to_datetime(reference_date)).to_timedelta64().astype(float)
            te = (dt_end - pd.to_datetime(reference_date)).to_timedelta64().astype(float)
            dses.append(
                ds
                    .pipe(lambda ds: ds.isel(time=(ds.time < te) & (ds.time >= ts))).compute()
                    .pipe(xr.decode_cf)
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir > slice_args.get('lat_min', 0))))
                    .pipe(lambda ds: ds.isel(time=(ds.lat_nadir < slice_args.get('lat_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir < slice_args.get('lon_max', 360))))
                    .pipe(lambda ds: ds.isel(time=(ds.lon_nadir > slice_args.get('lon_min', 0)))).compute()
            )

    if len(dses) == 0:
        print(
            f"no data found for {slice_args} {groups} {pd.date_range(start=dt_start.replace(day=1), end=dt_end, freq='MS')}")
        return None
    return xr.concat(
        dses,
        dim="time"
    )


def get_oi_slice(path, **slice_args):
    with xr.open_dataset(path) as ds:
        return (
            ds
                .pipe(lambda ds: ds.sel(time=slice(slice_args.get('time_min', "2012-10-01"),
                                                   slice_args.get('time_max', "2013-10-01"))))
                .pipe(lambda ds: ds.isel(y=(ds.lat > slice_args.get('lat_min', 0))))
                .pipe(lambda ds: ds.isel(y=(ds.lat < slice_args.get('lat_max', 360))))
                .pipe(lambda ds: ds.isel(x=(ds.lon < slice_args.get('lon_max', 360))))
                .pipe(lambda ds: ds.isel(x=(ds.lon > slice_args.get('lon_min', 0))))
        ).compute()


def get_natl_slice(path, **slice_args):
    with xr.open_dataset(path) as ds:
        return (
            ds.pipe(lambda ds: ds.assign(time=ds.time.assign_attrs({'units': 'days since 2012-10-01'})))
                .pipe(xr.decode_cf)
                .pipe(lambda ds: ds.sel(time=slice(slice_args.get('time_min', "2012-10-01"),
                                                   slice_args.get('time_max', "2013-10-01"))))
                .pipe(lambda ds: ds.isel(y=(ds.lat > slice_args.get('lat_min', 0))))
                .pipe(lambda ds: ds.isel(y=(ds.lat < slice_args.get('lat_max', 360))))
                .pipe(lambda ds: ds.isel(x=(ds.lon < slice_args.get('lon_max', 360))))
                .pipe(lambda ds: ds.isel(x=(ds.lon > slice_args.get('lon_min', 0))))
        ).compute()


if __name__ == '__main__':
    slice_args = {
        "time_min": "2013-08-23",
        "time_max": "2013-09-03",
        "lat_min": 30,
        "lat_max": 40,
        "lon_min": 295,
        "lon_max": 305,
    }

    t0 = time()
    batch = {
        **{f'nadir_{name}': get_nadir_slice(f'data/zarr/nadir/{name}', **slice_args) for name in
           ['swot', 'en', 'tpn', 'g2', 'j1']},
        'swot': get_swot_slice(f'data/zarr/swot', **slice_args),
        'oi': get_oi_slice('data/raw/DUACS-OI_maps/ssh_model/ssh_sla_boost_NATL60_en_j1_tpn_g2.nc', **slice_args),
        'natl': get_natl_slice('data/raw/NATL60_regular_grid/1_10/natl60CH_H.nc', **slice_args),
    }
    print(time() - t0)
