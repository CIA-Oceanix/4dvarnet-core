import utils
from tqdm import tqdm
from pathlib import Path
import traceback
import xarray as xr
import xesmf as xe
import pyinterp
import numpy as np

base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    'file_paths={fp}'
]


def interp_unstruct_to_grid(src_da, tgt_da, algo='rbf'):
    def interp_day(src_day):
        mesh = pyinterp.RTree()
        mesh.packing(
                np.vstack([src_da.nav_lon.values.ravel(),
                          src_da.nav_lat.values.ravel()]).T,
                src_day.values.ravel()
        )
        glon, glat = tgt_da.pipe(lambda ds: np.meshgrid(ds.lon.values, ds.lat.values))
        rec = None
        if algo=='idw':
            idw, _ = mesh.inverse_distance_weighting(
                np.vstack((glon.ravel(), glat.ravel())).T,
                within=False,  # Extrapolation is forbidden
                k=11,  # We are looking for at most 11 neighbors
                radius=60000,
                num_threads=0)
            rec = idw.reshape(glon.shape)
        elif algo=='rbf':
            rbf, _ = mesh.radial_basis_function(
                np.vstack((glon.ravel(), glat.ravel())).T,
                within=False,  # Extrapolation is forbidden
                k=11,  # We are looking for at most 11 neighbors
                radius=60000,
                rbf='thin_plate',
                num_threads=0)
            rec = rbf.reshape(glon.shape)
        elif algo=='wf':
            wf, _ = mesh.window_function(
                np.vstack((glon.ravel(), glat.ravel())).T,
                within=False,  # Extrapolation is forbidden
                k=11,
                radius=60000,
                wf='parzen',
                num_threads=0)
            rec = wf.reshape(glon.shape)
        return xr.DataArray(
            rec,
            coords=dict(
                 lon=('lon', tgt_da.lon.values),
                 lat=('lat', tgt_da.lat.values)),
            dims=('lat', 'lon')
        )
    return src_da.groupby('time_counter').map(interp_day, shortcut=False)

def run1():
    try:
        len(list(Path('../sla-data-registry/enatl60').glob('*BLBT*')))
        len(list(Path('../sla-data-registry/enatl60').glob('*BLB0*')))
        tgt_grid = xr.open_dataset('../sla-data-registry/qdata/natl20.nc')

        blb0 = []
        for f in tqdm(list(Path('../sla-data-registry/enatl60').glob('*BLB0*'))):
            samp = xr.open_dataset(f)
            samp=samp.mean('time_counter', keepdims=True)
            regridded = interp_unstruct_to_grid(samp.sossheig, tgt_grid.ssh)
            blb0.append(regridded)

        bods = xr.concat(blb0, 'time_counter').rename(time_counter='time')

        blbt = []
        for f in tqdm(list(Path('../sla-data-registry/enatl60').glob('*BLBT*'))):
            samp = xr.open_dataset(f)
            samp=samp.mean('time_counter', keepdims=True)
            regridded = interp_unstruct_to_grid(samp.sossheig, tgt_grid.ssh)
            blbt.append(regridded)

        btds = xr.concat(blbt, 'time_counter').rename(time_counter='time')

        
        # ds_out = xe.util.grid_2d(-70, -50, 0.05, 30, 50, 0.05)

        # regridder = xe.Regridder(
        #         samp.assign_coords(lon=samp.nav_lon, lat=samp.nav_lat),
        #         ds_out, "bilinear")

        # regridder.

    except Exception as e:

        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    locals().update(main())
