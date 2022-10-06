import utils
from scipy import ndimage
import pyinterp 
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
import traceback
import swath_calib.utils


base_cfg = 'qfebvre/xp_oi'
fp = 'dgx_ifremer'
overrides = [
    'file_paths={fp}'
]

def compute_geo_area(ds):
    dy0 = 1852*60*np.abs(ds['lat'][2].data - ds['lat'][1].data)
    # ** note: dy0 is a constant over the globe ** 
    dx0 = 1852*60*np.cos(np.deg2rad(ds['lat'].data))*(ds['lon'][2].data - ds['lon'][1].data) 
    area = dy0*np.tile(dx0, (len(ds['lon'].data),1))
    area = np.transpose(area)
    dA = xr.DataArray(
        data=area,
        dims=["lat", "lon"],
        coords=dict(
        lon=(["lon"], ds['lon'].data), lat=(["lat"], ds['lat'].data),),
    )
    return dA

def run1():
    try:
        cfg = utils.get_cfg(base_cfg)
        # dm = utils.get_dm(base_cfg, add_overrides=overrides)
        # raw_natl = xr.open_dataset('../sla-data-registry/raw/NATL60_regular_grid/1_10/natl60CH_H.nc')
        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        # ref_obs = xr.open_dataset('../sla-data-registry/NATL60/NATL/data_new/dataset_nadir_0d.nc')
        area = compute_geo_area(natl)
        area.assign_coords(lon=np.where(area.lon > 180, area.lon %-360, area.lon)).sel(lat=slice(34, 46), lon=slice(-66, -54)).plot()
        nadirs = {
            nad: swath_calib.utils.get_nadir_slice(
                f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
                lat_min=31,
                lat_max=45,
                lon_min=293,
                lon_max=307,
            )
            for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}


        natl.assign(area=(area.dims, area.data))
        # natl.coarsen(lat=
        natl.sel(lat=slice(31, 45), lon=slice(-67, -53)).isel(time=1).ssh.plot()
        pd.to_timedelta((natl.time.max() - natl.time.min()).item(), unit='s')
        current_dt = pd.to_timedelta('1D')
        current_dx = 1/20
        # tgt_dt = pd.to_timedelta('1D')
        tgt_dt = pd.to_timedelta('14D')
        dses_obs = {}
        dses_ref = {}
        tgt_dx = 1/4
        for tgt_dx, _tgt_dt in [(1, '14D'), (1, '1D'), (1/2, '1D'), (1/2, '7D'),(1/4, '1D'),]:
            tgt_dt = pd.to_timedelta(_tgt_dt) 
            red_ds = (
                natl
                .sel(lat=slice(31, 45), lon=slice(-67, -53))
                .isel(time=slice(None, -1))
                .coarsen(
                    # time=14,
                    # lat=5,
                    # lon=5,
                    time=int(tgt_dt / current_dt),
                    lat=int(tgt_dx / current_dx),
                    lon=int(tgt_dx / current_dx),
                )
                .mean()
            )
            red_ds.isel(time=1).ssh.plot()
            
            up_ds = red_ds.interp(
                natl.sel(lat=slice(31, 45), lon=slice(-67, -53))
                .isel(time=slice(None, -1)).coords,
                kwargs={"fill_value": "extrapolate"}
            ).assign(time=lambda ds: pd.to_datetime('2012-10-01') + pd.to_timedelta(ds.time, 's'))

            nad_samp = nadirs['tpn'][['lat', 'lon', 'ssh_model']]

            binning = pyinterp.Binning2D(pyinterp.Axis(up_ds.lon.values), pyinterp.Axis(up_ds.lat.values))
            grid_day_dses = []
            for t in tqdm(up_ds.time):
                binning.clear()
                for nad in nadirs:
                    try:
                        nad_chunk=nadirs[nad].sel(time=str(pd.to_datetime(t.item()).date()))
                    except KeyError as e:
                        print(e)
                    if nad_chunk.dims['time'] == 0:
                        continue
                    nad_red = (
                        up_ds.interp(
                            time=t.broadcast_like(nad_chunk.ssh_model),
                            lat=nad_chunk.lat.broadcast_like(nad_chunk.ssh_model),
                            lon=nad_chunk.lon.broadcast_like(nad_chunk.ssh_model) - 360,
                        )
                    )
                    binning.push(nad_red.lon.values, nad_red.lat.values, nad_red.ssh.values)

                grid_day_dses.append(xr.Dataset(
                       data_vars={'ssh': (('time', 'lat', 'lon'), binning.variable('mean').T[None, ...])},
                       coords={'time': [t.values], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
                    ).astype('float32', casting='same_kind')
                )
            
            new_obs_ds = xr.concat(grid_day_dses, dim='time')
            dses_obs[f'ssh_{str(np.round(tgt_dx, 3)).replace(".", "")}_{_tgt_dt}'] = new_obs_ds.sel(lat=slice(32, 44), lon=slice(-66, -54))
            dses_ref[f'ssh_{str(np.round(tgt_dx, 3)).replace(".","")}_{_tgt_dt}'] = up_ds.sel(lat=slice(32, 44), lon=slice(-66, -54))
        (nad_red.ssh -nad_chunk.ssh_model) .reset_index('time').plot()
        nad_chunk.ssh_model.reset_index('time').plot()
        Path('../sla-data-registry/natl60_degraded').mkdir(exist_ok=True)
        xr.Dataset({k: v.ssh for k, v in dses_obs.items()}).to_netcdf('../sla-data-registry/natl60_degraded/obs220921.nc')
        xr.Dataset({k: v.ssh for k, v in dses_ref.items()}).to_netcdf('../sla-data-registry/natl60_degraded/ref220921.nc')

        # ref_obs.isel(time=31).sel(lat=slice(32, 44), lon=slice(-66, -54)).ssh_mod.plot()

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def anim_res():
    try:
        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        ds = xr.open_dataset('../sla-data-registry/natl60_degraded/ref220914.nc')
        ds = ds.assign(ref= (natl.dims, natl.ssh.sel(lat=slice(32, 44), lon=slice(-66, -54)).isel(time=slice(None, -1)).data))
        
        def anim(test_xr_ds, deriv=None,  dvars=['ssh_1_14D', 'ssh_1_1D', 'ssh_025_1D', 'ref']):
            def sobel(da):
                dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
                dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
                return np.hypot(dx_ac, dx_al)

            if deriv is None:
                tpds = test_xr_ds.isel(time=slice(None, 20, 1))
                clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
                cmap='RdBu'


            if deriv == 'grad':
                tpds = test_xr_ds.pipe(sobel).isel(time=slice(50, 150, 2))
                clim = (0, tpds[dvars].to_array().max().item())
                cmap = 'viridis'
            
            if deriv == 'lap':
                tpds = test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1)).isel(time=slice(None, 10, 1))
                clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
                cmap='RdGy'
            hvds = hv.Dataset(tpds)
            # hvds = hv.Dataset(mod.test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=0.1)).isel(time=slice(None, 10, 1)))
            # hvds = hv.Dataset(mod.test_xr_ds.map(ndimage.laplace).isel(time=slice(None, 10, 1)))
            images = hv.Layout([
                    hvds
                    .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
                    .options(
                        cmap=cmap,
                        clim=clim,
                    )
                    for v in dvars
                    ]).cols(2).opts(sublabel_format="")
            return images

        # img = anim(ds, dvars=['ssh_1_14D', 'ssh_1_1D', 'ref'])
        # hv.output(img, holomap='gif', fps=4, dpi=125)
        
        img = anim(fmtted_orca.to_dataset(name='ssh'), dvars=['ssh'])
        img[0]
        hv.output(img, holomap='gif', fps=4, dpi=125)
        
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def sample_nadirs_from_simu(nadirs, simu):
            binning = pyinterp.Binning2D(pyinterp.Axis(simu.lon.values), pyinterp.Axis(simu.lat.values))
            grid_day_dses = []
            for t in tqdm(simu.time):
                binning.clear()
                for nad in nadirs:
                    try:
                        nad_chunk=nadirs[nad].sel(time=str(pd.to_datetime(t.item()).date()))
                    except KeyError as e:
                        print(e)
                    if nad_chunk.dims['time'] == 0:
                        continue
                    nad_red = (
                        simu.interp(
                            time=t.broadcast_like(nad_chunk.ssh_model),
                            lat=nad_chunk.lat.broadcast_like(nad_chunk.ssh_model),
                            lon=nad_chunk.lon.broadcast_like(nad_chunk.ssh_model) - 360,
                        )
                    )
                    binning.push(nad_red.lon.values, nad_red.lat.values, nad_red.values)

                grid_day_dses.append(xr.Dataset(
                       data_vars={'ssh': (('time', 'lat', 'lon'), binning.variable('mean').T[None, ...])},
                       coords={'time': [t.values], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
                    ).astype('float32', casting='same_kind')
                )

            return xr.concat(grid_day_dses, dim='time')

def fmt_glorys12_dc2021():
    try:
        glorys = xr.open_dataset('../sla-data-registry/GLORYS/cmems_mod_glo_phy_my_0.083_P1D-m_1664351861719.nc')
        glorys = glorys.rename(latitude='lat', longitude='lon')
        fmtted_glorys = (
            glorys
            .pipe(lambda ds:
            ds.zos
            .interp(
                lat=natl.lat,
                lon=natl.lon,
            ))
        )

        fmtted_glorys.to_netcdf('../sla-data-registry/GLORYS/preprocessed_glorys_dc_2021.nc')
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def fmt_glorys12():
    try:
        glorys = xr.open_dataset('../sla-data-registry/GLORYS/cmems_mod_glo_phy_my_0.083_P1D-m_1663617411603.nc')

        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        natl['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(natl.time, 's')
        natl = natl.sel(lat=slice(32, 44), lon=slice(-66, -54))

        

        glorys = glorys.rename(latitude='lat', longitude='lon')
        glorys.lat
        fmtted_glorys = (
            glorys
            .pipe(lambda ds:
            ds.zos
            .interp(
                lat=natl.lat,
                lon=natl.lon,
            ))
        ).sel(
            time=slice(str(pd.to_datetime(natl.time.min().values).date()),
            str(pd.to_datetime(natl.time.max().values).date()))
        )

        nadirs = {
            nad: swath_calib.utils.get_nadir_slice(
                f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
                lat_min=31,
                lat_max=45,
                lon_min=293,
                lon_max=307,
            )
            for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}

         
        fmtted_glorys.isel(time=25).plot()
        new_obs_ds = ( sample_nadirs_from_simu(nadirs, fmtted_glorys))
        new_obs_ds.isel(time=30).ssh.plot()

        # xr.merge([new_obs_ds.rename(ssh='five_nadirs'), fmtted_glorys.to_dataset(name='ssh')]).to_netcdf('../sla-data-registry/GLORYS/preprocessed_glorys220921.nc')
        ref_ds = fmtted_glorys.isel(time=slice(None, -1))
        obs_ds = new_obs_ds.sel(ref_ds.coords, tolerance=0.01, method='nearest').assign_coords(ref_ds.coords)

        xr.merge([obs_ds.rename(ssh='five_nadirs'), ref_ds.to_dataset(name='ssh')]).to_netcdf('../sla-data-registry/GLORYS/preprocessed_glorys220928.nc')
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def fmt_orca25():
    try:
        orca = xr.open_dataset('../sla-data-registry/eORCA025/eORCA025_North_Atlantic_sossheig_y2013.nc')
        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        natl['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(natl.time, 's')
        natl = natl.sel(lat=slice(32, 44), lon=slice(-66, -54))

        orca1d = orca.resample(time_counter='1D').mean()
        
        orca1d['lat'] = orca1d.nav_lat.mean('x')
        orca1d['lon'] = orca1d.nav_lon.mean('y')
        orca1d['time'] = orca1d.time_counter + pd.to_timedelta('12H')
        orca1d.lat
        natl.lon

        fmtted_orca = (
            orca1d
            .swap_dims(time_counter='time', y='lat', x='lon')
            .pipe(lambda ds:
            ds.sossheig
            .interp(
                lat=natl.lat,
                lon=natl.lon,
            ))
        )


        nadirs = {
            nad: swath_calib.utils.get_nadir_slice(
                f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
                lat_min=31,
                lat_max=45,
                lon_min=293,
                lon_max=307,
            )
            for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}

         
        simu_offset= natl.time.min().values -fmtted_orca.time.min().values
        fmtted_orca.assign_coords(time=fmtted_orca.time + simu_offset).isel(time=25).plot()
        new_obs_ds = (
            sample_nadirs_from_simu(nadirs, fmtted_orca.assign_coords(time=fmtted_orca.time + simu_offset))
            .assign_coords(time=lambda ds: ds.time - simu_offset)
        )
        new_obs_ds.isel(time=30).ssh.plot()

        new_obs_ds.time
        fmtted_orca.time
        xr.merge([new_obs_ds.rename(ssh='five_nadirs'), fmtted_orca.to_dataset(name='ssh')]).to_netcdf('../sla-data-registry/eORCA025/preprocessed_eorca220921.nc')

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
