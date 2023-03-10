import utils
import metpy.calc as mpcalc
import metrics
import holoviews as hv
import panel as pn
pn.pane.Matplotlib.tight = True
try:
    hv.extension('matplotlib')
except:
    pass
finally:
    pass
import scipy.ndimage as ndi
import scipy.interpolate
import matplotlib.pyplot as plt
import xrft
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

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da) /2
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da) /2
    return np.hypot(dx_ac, dx_al)

vort = lambda da: mpcalc.vorticity(*mpcalc.geostrophic_wind(da.assign_attrs(units='m').metpy.quantify())).metpy.dequantify()
geo_energy = lambda da:np.hypot(*mpcalc.geostrophic_wind(da)).metpy.dequantify()

def anim(test_xr_ds, deriv=None,  dvars=['ssh_1_14D', 'ssh_1_1D', 'ssh_025_1D', 'ref']):
    if deriv is None:
        tpds = test_xr_ds.isel(time=slice(None, 20, 2))
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdBu'

    if deriv == 'grad':
        # tpds = test_xr_ds.pipe(sobel).isel(time=slice(50, 150, 2))
        tpds = test_xr_ds.map(geo_energy).isel(time=slice(None, 20, 2))
        clim = (0, tpds[dvars].to_array().max().item())
        cmap = 'viridis'
    
    if deriv == 'lap':
        # tpds = test_xr_ds.map(lambda da: ndimage.gaussian_laplace(da, sigma=1)).isel(time=slice(None, 10, 1))
        tpds = test_xr_ds.map(vort).isel(time=slice(None, 20, 2))
        clim = tpds[dvars].to_array().pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
        cmap='RdGy'

    hvds = hv.Dataset(tpds)
    if len(dvars) == 1:
        return hvds.to(hv.QuadMesh, ['lon', 'lat'], dvars[0]).relabel(dvars[0]).options(cmap=cmap, clim=clim,)
    images = hv.Layout([
            hvds
            .to(hv.QuadMesh, ['lon', 'lat'], v).relabel(v)
            .options(
                cmap=cmap,
                clim=clim,
                colorbar=True,
            )
            for v in dvars
            ]).opts(sublabel_format="")
    return images

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
        dses_obs = {}
        dses_ref = {}
        tgt_dx = 1/4
        for tgt_dx, _tgt_dt in [(1, '1D'), (1/2, '1D'), (1/4, '1D'),]:
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

def degrade(da, sig, subsamp, interp):
    def interp(src_da, tgt_da, kind='linear'):
        def interp_day(src_day):
            interpolator = scipy.interpolate.interp2d(
                    src_day.lon.values,
                    src_day.lat.values,
                    src_day.transpose('lon', 'lat').values,
                    # kind='cubic'
                    kind=kind
            )
            return xr.DataArray(
                 interpolator(tgt_da.lon.values,
                              tgt_da.lat.values),
                coords=dict(
                     lon=('lon', tgt_da.lon.values),
                     lat=('lat', tgt_da.lat.values)),
                dims=('lon', 'lat')
            )
        return src_da.groupby('time').map(interp_day, shortcut=False)

    coars = lambda da, sig, samp: (
            xr.apply_ufunc(lambda dd:ndi.gaussian_filter(dd, sig), da)
            .isel(lat=slice(None, None, samp), lon=slice(None, None,samp))
    )
    cda=coars(da, sig, subsamp)
    if subsamp > 1:
        return interp(cda, da, kind='interp')
    return cda

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

def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

def fix_time(da):
    da = da.copy()
    da['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(da.time, 's')
    return da
        
def sample_nadirs_from_simu(nadirs, simu, nadirs_start_time=pd.to_datetime('2012-10-01'), nb_day_nadirs=365):
    binning = pyinterp.Binning2D(pyinterp.Axis(simu.lon.values), pyinterp.Axis(simu.lat.values))
    grid_day_dses = []
    for i in tqdm(range(len(simu.time))):
        binning.clear()
        for nad in nadirs:
            try:
                nad_chunk=nadirs[nad].sel(time=str((nadirs_start_time + pd.to_timedelta(f'{i%365}D')).date()))
            except KeyError as e:
                print(e)
                continue
            if nad_chunk.dims['time'] == 0:
                continue
            nad_red = (
                simu.isel(time=i).drop('time').interp(
                    # time=t.broadcast_like(nad_chunk.ssh_model),
                    lat=nad_chunk.lat.broadcast_like(nad_chunk.ssh_model),
                    lon=nad_chunk.lon.broadcast_like(nad_chunk.ssh_model) - 360,
                )
            )
            binning.push(nad_red.lon.values, nad_red.lat.values, nad_red.values)

        grid_day_dses.append(xr.Dataset(
               data_vars={'ssh': (('time', 'lat', 'lon'), binning.variable('mean').T[None, ...])},
               coords={'time': [simu.time[i].values], 'lat': np.array(binning.y), 'lon': np.array(binning.x)}
            ).astype('float32', casting='same_kind')
        )

    return xr.concat(grid_day_dses, dim='time')

def run2():
    try:

        bigger_domain = {'lat': slice(31, 45), 'lon':slice(-67,-53)}
        domain = {'lat': slice(32, 44), 'lon':slice(-66,-54)}
        smaller_domain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}

        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc').pipe(fix_time)

        

        natl_dses = {
            'natl20': natl.sel(bigger_domain).ssh.load(),
            # 'natl20_g1': degrade(natl.sel(bigger_domain).ssh, sig=1, subsamp=1, interp="cubic").load(),
            # 'natl20_g3':degrade(natl.sel(bigger_domain).ssh, sig=3, subsamp=1, interp="cubic").load(),
            # 'natl20_g5':degrade(natl.sel(bigger_domain).ssh, sig=5, subsamp=1, interp="cubic").load(),
            # 'natl20_g8':degrade(natl.sel(bigger_domain).ssh, sig=8, subsamp=1, interp="cubic").load(),
        }

        compo_natl_dses = {
                # 'natl20_g1_90': natl_dses['natl20']*0.1 + natl_dses['natl20_g1']*0.9,
                # 'natl20_g3_90': natl_dses['natl20']*0.1 + natl_dses['natl20_g3']*0.9,
                # 'natl20_g5_90': natl_dses['natl20']*0.1 + natl_dses['natl20_g5']*0.9,
                # 'natl20_g8_90': natl_dses['natl20']*0.1 + natl_dses['natl20_g8']*0.9,
        }

        orca = xr.open_dataset('../sla-data-registry/eORCA025/eORCA025_North_Atlantic_sossheig_y2013.nc').load()
        
        pp_orca = interp_unstruct_to_grid(
            orca.resample(time_counter='1D').mean().sossheig, natl.ssh.sel(bigger_domain), algo='rbf'
        ).rename(time_counter='time')

        files_rea = Path('../sla-data-registry/GLORYS/GLORYS12V1').glob('*.nc')
        glo_rea = xr.open_mfdataset(files_rea).load()
        pp_glo_rea = interp_unstruct_to_grid(
            glo_rea.sossheig, natl.sel(bigger_domain).ssh
        ).rename(time_counter='time')

        files_free = Path('../sla-data-registry/GLORYS/GLORYS12V1-FREE').glob('*.nc')
        glo_free = xr.open_mfdataset(files_free).load()
        pp_glo_free = interp_unstruct_to_grid(
            glo_free.sossheig, natl.sel(bigger_domain).ssh
        ).rename(time_counter='time')

        others_dses = {
            'glo12_rea': pp_glo_rea,
            'glo12_free': pp_glo_free,
            'orca25': pp_orca,
        }

        all_dses = {**natl_dses, **others_dses, **compo_natl_dses}
        nadirs = {
            nad: swath_calib.utils.get_nadir_slice(
                f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
                lat_min=31,
                lat_max=45,
                lon_min=293,
                lon_max=307,
            )
            for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}
        
        obses_ds = {}
        for k, ds in all_dses.items():
            print(k)
            obses_ds[k] = sample_nadirs_from_simu(nadirs, ds)

        
        for  k in all_dses:
            print(k, np.nanmax(obses_ds[k].ssh.values - all_dses[k].values))


        for k in all_dses:
            ssh_da = all_dses[k].sel(domain)
            obs_da = obses_ds[k].ssh.sel(domain)
            ds = xr.Dataset(dict(ssh=(ssh_da.dims, ssh_da.values), nadir_obs=(obs_da.dims, obs_da.values)), coords=ssh_da.coords)
            print(k, ds.dims)
            ds.to_netcdf(f'../sla-data-registry/qdata/{k}.nc')



        # Analysis

        ## SSH psd
        psd_fn = lambda da: xrft.isotropic_power_spectrum(
                da, dim=('lat', 'lon'), truncate=True, window='hann')

        psds_ds = xr.Dataset(
                {k: psd_fn(ds.sel(smaller_domain).pipe(remove_nan)).mean('time')
                 for k, ds in all_dses.items()}
        ).ffill('freq_r')
        psds_ds.to_array().plot.line( x='freq_r', hue='variable', xscale='log', yscale='log', figsize=(12,8))



        ## Geostrophic energy psd
        psds_ds = xr.Dataset(
                {k: psd_fn(ds.sel(smaller_domain).pipe(remove_nan).pipe(geo_energy)).mean('time')
                 for k, ds in all_dses.items()}
        ).ffill('freq_r')
        psds_ds.to_array().plot.line( x='freq_r', hue='variable', xscale='log', yscale='log', figsize=(12,8))

        ## Vorticity energy psd
        psds_ds = xr.Dataset(
                {k: psd_fn(ds.sel(smaller_domain).pipe(remove_nan).pipe(vort)).mean('time')
                 for k, ds in all_dses.items()}
        ).ffill('freq_r')
        psds_ds.to_array().plot.line( x='freq_r', hue='variable', xscale='log', yscale='log', figsize=(12,8))

        weighted_scale = (
            psds_ds.sum() / (psds_ds * psds_ds.freq_r).sum('freq_r')
        ) 
        print((100*weighted_scale).to_array().to_dataframe(name='scale (km)').to_markdown())

        res_deg = []
        ref = natl_dses['natl20'].sel(smaller_domain).pipe(remove_nan)
        for k, da in {**natl_dses, **compo_natl_dses}.items():
            if k == 'natl20':
                continue
            da = da.sel(smaller_domain).pipe(remove_nan)
            spatial_res_model, _ = metrics.get_psd_score(ref, da, da, with_fig=False)
            _, _, mu, _ = metrics.rmse_based_scores(ref, da)
            _, lamb_x, lamb_t = metrics.psd_based_scores(ref, da)
            res_deg.append(
                   dict(
                       ds=k,
                       res=spatial_res_model,
                       mu=mu,
                       lamb_x=lamb_x,
                       lamb_t=lamb_t,
                   ) 
            )
        res_deg = pd.DataFrame(res_deg)
        print(res_deg.to_markdown())


        # Anim
        print(list(all_dses))
        k = 'natl20'
        img_ssh = anim(
                obses_ds[k].sel(smaller_domain),
                deriv=None, dvars=['ssh'])
        # img_ssh = anim(
        #         all_dses[k].sel(smaller_domain).pipe(remove_nan).to_dataset(name='ssh'),
        #         deriv=None, dvars=['ssh'])
        img_grad = anim(
                all_dses[k].sel(smaller_domain).pipe(remove_nan).to_dataset(name='geo'),
                deriv='grad', dvars=['geo'])
        img_lap = anim(
                all_dses[k].sel(smaller_domain).pipe(remove_nan).to_dataset(name='vort'),
                deriv='lap', dvars=['vort'])
        hv.output(img_ssh + img_grad + img_lap, holomap='gif', fps=2, dpi=125)
        


    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def find_good_down_sampling():
    try:

        def interpolate_na_2D(ds, max_value=100.):
            return (
                    ds.where(np.abs(ds) < max_value, np.nan)
                    .to_dataframe()
                    .interpolate(method='linear')
                    .pipe(xr.Dataset.from_dataframe)
        )

        def remove_nan(da):
            da['lon'] = da.lon.assign_attrs(units='degrees_east')
            da['lat'] = da.lat.assign_attrs(units='degrees_north')

            da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
                pyinterp.backends.xarray.Grid3D(da))[1]
            return da


        psd_fn = lambda da: xrft.isotropic_power_spectrum(
                da, dim=('lat', 'lon'), truncate=True, window='flattop')

        domain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}

        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        natl['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(natl.time, 's')
        # glorys = xr.open_dataset('../sla-data-registry/GLORYS/preprocessed_glorys220921.nc')
        glorys = xr.open_dataset('../sla-data-registry/GLORYS/cmems_mod_glo_phy_my_0.083_P1D-m_1664351861719.nc')
        glo_da = glorys.zos.load().rename(latitude='lat', longitude='lon').pipe(remove_nan)

        dom_da = natl.sel(domain).ssh.load().pipe(remove_nan)


        def interp(src_da, tgt_da, kind='linear'):
            def interp_day(src_day):
                interpolator = scipy.interpolate.interp2d(
                        src_day.lon.values,
                        src_day.lat.values,
                        src_day.transpose('lon', 'lat').values,
                        # kind='cubic'
                        kind=kind
                )
                # return tgt_da.sel(time=src_day.time)
                return xr.DataArray(
                     interpolator(tgt_da.lon.values,
                                  tgt_da.lat.values),
                    coords=dict(
                         lon=('lon', tgt_da.lon.values),
                         lat=('lat', tgt_da.lat.values)),
                    dims=('lon', 'lat')
                )
            return src_da.groupby('time').map(interp_day, shortcut=False)

        coars = lambda da, sig, samp: (
                xr.apply_ufunc(lambda dd:ndi.gaussian_filter(dd, sig), da)
                .isel(lat=slice(None, None, samp), lon=slice(None, None,samp))
        )


        # in_ds = interp(coars(dom_da, 5, 2), dom_da, 'cubic')
        res_deg = []
        for sig in range(1, 13, 4):
            for sub_samp in [1, 4, 10]:
                print(sig, sub_samp)
                in_ds = interp(coars(dom_da, sig, sub_samp), dom_da, 'cubic')
                spatial_res_model, spatial_res_oi = metrics.get_psd_score(dom_da, in_ds, in_ds, with_fig=False)
                _, _, mu, _ = metrics.rmse_based_scores(dom_da, in_ds)
                psd_ds, lamb_x, lamb_t = metrics.psd_based_scores(dom_da, in_ds)
                res_deg.append(
                       dict(
                           sig=sig,
                           sub_samp=sub_samp,
                           res=spatial_res_model,
                           mu=mu,
                           lamb_x=lamb_x,
                           lamb_t=lamb_t,
                           psd=psd_fn(in_ds).mean('time'),
                           psd_grad=psd_fn(in_ds.pipe(sobel)).mean('time'),
                           psd_lap=psd_fn(xr.apply_ufunc(ndi.laplace, in_ds)).mean('time'),
                       ) 
                )
        res_deg = pd.DataFrame(res_deg)
        res_deg = res_deg.assign(xp=res_deg.sig.map(str) + '_' +res_deg.sub_samp.map(str))

        psds_ds = xr.Dataset(
            res_deg.set_index('xp')
            .loc[lambda df: df.sub_samp==1]
            .psd.to_dict()
        ).ffill('freq_r')
        psds_ds.to_array().plot.line(
                x='freq_r', hue='variable',
                xscale='log', yscale='log',
                figsize=(10,6)
        )

          

        in_ds = interp(coars(dom_da, 9, 1), dom_da, 'cubic')
        (dom_da - in_ds).pipe(sobel).isel(time=1).plot()
        (in_ds).pipe(sobel).isel(time=1).plot()

        img = anim(
                xr.merge(
                    [dom_da.to_dataset(name='ssh'),
                     in_ds.to_dataset(name='down')]) ,
                    deriv='grad', dvars=['ssh', 'down'])
        img = anim(
                    (0.1*dom_da + 0.9*in_ds).to_dataset(name='down'),
                    deriv='grad', dvars=['down'])
        hv.output(img, holomap='gif', fps=4, dpi=125)

        (dom_da).pipe(sobel).isel(time=1).plot()

        
        fig

        psds = {
                'base20': psd_fn(dom_da).mean('time'),
                'base12': psd_fn(glo_da).mean('time'),
                'down9':psd_fn(in_ds).mean('time'),
                'down9_1':psd_fn((0.1*dom_da + 0.9*in_ds)).mean('time'),
                'down9_3':psd_fn((0.3*dom_da + 0.7*in_ds)).mean('time'),
                'down9_5':psd_fn((0.5*dom_da + 0.5*in_ds)).mean('time'),
                # 'reint_lin_12': psd_fn(interp(glo_da, dom_da, 'linear')).mean('time'),
                # 'reint_cub_12': psd_fn(interp(glo_da, dom_da, 'cubic')).mean('time'),
                # 'reint_quint_12': psd_fn(interp(glo_da, dom_da, 'quintic')).mean('time'),
                # 'c5': psd_fn(in_ds).mean('time'),
        }

        psds_ds = xr.Dataset(psds).ffill('freq_r')
        psds_ds.to_array().plot.line(
                x='freq_r', hue='variable',
                xscale='log', yscale='log',
                figsize=(10,6)
        )

        # in_ds.isel(time=0).plot()
        
        deltas = mpcalc.lat_lon_grid_deltas(latitude = dom_da.lat, longitude=dom_da.lon)
        gradx = mpcalc.gradient(dom_da.values, axes=[2], deltas=deltas[0])
        grady = mpcalc.gradient(dom_da.values, axes=[1], deltas=deltas[1].T)

        grady = mpcalc.gradient(dom_da.values, axes=[1], deltas=deltas[1].T)
        grad_ds = xr.DataArray( np.sqrt(gradx[0].to_tuple()[0]**2 + grady[0].to_tuple()[0]**2), dom_da.coords)
        grad_ds.isel(time=0).plot()
        dom_da.pipe(sobel).isel(time=0).plot()
       
        mpcalc.geostrophic_wind(dom_da)
        dgradx, dgrady = mpcalc.gradient(dom_da, axes=['lon', 'lat'])
        ddgradx, ddgrady = mpcalc.gradient(dom_da.rename(lat='latitude', lon='longitude'), axes=['longitude', 'latitude'])
        
        dgradx - ddgradx.rename(longitude='lon', latitude='lat')
        dgrad_ds = xr.DataArray( np.sqrt(dgradx[0].to_tuple()[0]**2 + dgrady[0].to_tuple()[0]**2), dom_da.coords)
        (grady[0]-dgrady).isel(time=1).plot()
        (dgrady).isel(time=1).plot()
        dgrady.metpy.dequantify()
        grad_ds = xr.DataArray( np.sqrt(gradx[0].to_tuple()[0]**2 + grady[0].to_tuple()[0]**2), dom_da.coords)
        len(mpcalc.gradient(dom_da))
        dom_da.assign_attrs(unit='m').metpy.quantify()
        grad_ds.data
        psds = dict(
            psd_geo = psd_fn(np.sqrt(geoy**2 + geox**2).metpy.dequantify()).mean('time'),
            grad_psd_grid = psd_fn(dom_da.pipe(sobel)).mean('time')
        )

        tpds = xr.Dataset({
            'down': in_ds,
            'natl': dom_da,
        })
        mpcalc.coriolis_parameter(dom_da.lat)

        geoy, geox = mpcalc.geostrophic_wind(dom_da)
        vort = mpcalc.vorticity(*mpcalc.geostrophic_wind(dom_da.assign_attrs(units='m').metpy.quantify()))

        vort.metpy.dequantify().isel(time=0).plot()

        xrft.isotropic_power_spectrum(
                vort.metpy.dequantify(), dim=['lat', 'lon'], window='hann'
        ).mean('time').plot.line(
                x='freq_r', hue='variable',
                xscale='log', yscale='log',
                figsize=(10,6)
        )

        (dom_da.pipe(sobel)).isel(time=0).plot(figsize=(12,10))
        (geoy**2 + geox**2).isel(time=0).plot(figsize=(12,10))
        img = anim(tpds, dvars=['down', 'natl'], deriv='grad')
        mr = hv.output(img, holomap='gif', fps=4, dpi=50, backend='matplotlib')

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()



def preprocess_glo():
    try:
        
        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        obs = xr.open_dataset('../sla-data-registry/NATL60/NATL/data_new/dataset_nadir_0d.nc')
        files = Path('../sla-data-registry/GLORYS/GLORYS12V1').glob('*.nc')
        dsa = xr.open_mfdataset(files).load()

        files = Path('../sla-data-registry/GLORYS/GLORYS12V1-FREE').glob('*.nc')
        dsf = xr.open_mfdataset(files).load()

        psd_fn = lambda da: xrft.isotropic_power_spectrum(
                da, dim=('lat', 'lon'), truncate=True, window='hann')

        domain = {'lat': slice(32, 44), 'lon':slice(-66,-54)}
        subdomain = {'lat': slice(33, 43), 'lon':slice(-65,-55)}

        def interpolate_na_2D(ds, max_value=100.):
            return (
                    ds.where(np.abs(ds) < max_value, np.nan)
                    .to_dataframe(name='ssh')
                    .interpolate(method='linear')
                    .pipe(xr.Dataset.from_dataframe).ssh
        )

        def remove_nan(da):
            da['lon'] = da.lon.assign_attrs(units='degrees_east')
            da['lat'] = da.lat.assign_attrs(units='degrees_north')

            da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
                pyinterp.backends.xarray.Grid3D(da))[1]
            return da

        dsf_pp = interp_unstruct_to_grid(dsf.sossheig, natl.sel(domain).ssh)
        dsa_pp = interp_unstruct_to_grid(dsa.sossheig, natl.sel(domain).ssh)


        dsf_no_na = dsf_pp.rename(time_counter='time').sel(subdomain).pipe(remove_nan)
        dsa_no_na = dsa_pp.rename(time_counter='time').sel(subdomain).pipe(remove_nan)

        dsf_no_na.isel(time=1).pipe(sobel).plot()
        dsa_no_na.isel(time=1).pipe(sobel).plot()
        dsa.sossheig.isel(
                time_counter=1,
                x=slice(60,190),
                y=slice(60,190)).pipe(sobel).plot()


        psdf = psd_fn(dsf_no_na).mean('time')
        psda = psd_fn(dsa_no_na).mean('time')
        psds_ds = xr.Dataset(
                dict(free= psdf, assim=psda)
                ).ffill('freq_r')
        psds_ds.to_array().plot.line(
                x='freq_r', hue='variable',
                xscale='log', yscale='log',
                figsize=(10,6)
        )

        import pyinterp.backends.xarray
        import pyinterp.fill
        interpolator = pyinterp.backends.xarray.Grid3D(dsf.sossheig)
        
        da_nat = natl.ssh.sel(domain).load()
        



        da_nat.pipe(remove_nan).isel(time=0).plot()

        psd_fn(orca)


    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def anim_res():
    try:
        ds = xr.open_dataset('../sla-data-registry/natl60_degraded/ref220914.nc')
        ds = ds.assign(ref= (natl.dims, natl.ssh.sel(lat=slice(32, 44), lon=slice(-66, -54)).isel(time=slice(None, -1)).data))
        
        # img = anim(ds, dvars=['ssh_1_14D', 'ssh_1_1D', 'ref'])
        # hv.output(img, holomap='gif', fps=4, dpi=125)
        
        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        domain = {'lat': slice(32, 44), 'lon':slice(-66,-54)}
        img = anim(natl.sel(domain), dvars=['ssh'])
        hv.output(img, holomap='gif', fps=4, dpi=125)
        hv.save(img, 'anim.gif', fps=4, dpi=125)
        
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


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

def refmt_enatl():
    try:

        ref = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        ref['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(ref.time, 's')
        ref = ref.sel(lat=slice(32, 44), lon=slice(-66, -54))

        raw_natl = xr.open_dataset('../sla-data-registry/raw/NATL60_regular_grid/1_10/natl60CH_H.nc')
        raw_natl['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(raw_natl.time, 's')
        raw_natl['lon'] =  raw_natl['lon'] - 360
        raw_natl = (
                raw_natl.swap_dims(x='lon', y='lat')
                .pipe(lambda ds: ds.isel(lat=  (44>=ds.lat) &  (ds.lat>=32) ))
                .pipe(lambda ds: ds.isel(lon=  (-66>=ds.lon) &  (ds.lon>=-44) ))
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

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()
        
def fmt_orca25():
    try:
        # orca = xr.open_dataset('../sla-data-registry/eORCA025/eORCA025_North_Atlantic_sossheig_y2013.nc')
        orca = xr.open_dataset('../sla-data-registry/eORCA025_North_Atlantic_sossheig_y2013_fixed.nc')
        orca.nav_lat
        natl = xr.open_dataset('../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc')
        natl['time'] =  pd.to_datetime('2012-10-01') + pd.to_timedelta(natl.time, 's')
        natl = natl.sel(lat=slice(31, 45), lon=slice(-67, -53))

        

        import xesmf as xe
        ds_out = xe.util.grid_2d(-67, -53, 0.05, 31, 45, 0.05)
        reggridder = xe.Regridder(orca, ds_out, "bilinear", unmapped_to_nan=True)
        out = reggridder(orca)

        out['time'] = out.time_counter + pd.to_timedelta('11H') + pd.to_timedelta('30min')
        fmtted_orca = (
            out
            .assign(lat=lambda ds: ds.lat.isel(x=0) -0.025, lon=lambda ds: ds.lon.isel(y=0) -0.025, )
            .swap_dims(time_counter='time', y='lat', x='lon')
        )


        nadirs = {
            nad: swath_calib.utils.get_nadir_slice(
                f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
                lat_min=31,
                lat_max=45,
                lon_min=293,
                lon_max=308,
            )
            for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}

         
        simu_offset= pd.to_timedelta(natl.time.min().values -fmtted_orca.time.min().values)
        # fmtted_orca.assign_coords(time=fmtted_orca.time + simu_offset).isel(time=25).plot()
        fmtted_orca.isel(time=0).sossheig.plot()
        mean_day = fmtted_orca.resample(time='1D').mean()
        mean_day.sel(lat=toto.lat, lon=toto.lon, method='nearest').isel(time=0).sossheig.plot()
        mean_day.sossheig.isel(time=1).plot()
        snap_day = fmtted_orca.sel(time=mean_day.time, method='nearest')
        new_obs_ds = (
            sample_nadirs_from_simu(nadirs, snap_day.assign_coords(time=pd.to_datetime(snap_day.time) + simu_offset).sossheig)
            .assign_coords(time=lambda ds: ds.time - simu_offset)
        )
        new_obs_ds.isel(time=30).sel(lat=toto.lat, lon=toto.lon, method='nearest').ssh.plot()
        mean_day.isel(time=30).sel(lat=toto.lat, lon=toto.lon, method='nearest').sossheig.plot()

        new_obs_ds.time
        fmtted_orca.time
        xr.merge([new_obs_ds.rename(ssh='five_nadirs'), fmtted_orca.to_dataset(name='ssh')]).to_netcdf('../sla-data-registry/eORCA025/preprocessed_eorca230303.nc')
        ds = xr.merge([new_obs_ds.rename(ssh='nadir_obs').interp(lat=toto.lat, lon=toto.lon, method='nearest').interp(time=mean_day.time, method='nearest'),
                        mean_day.rename(sossheig='ssh').interp(lat=toto.lat, lon=toto.lon, method='nearest'),])
        ds.isel(time=slice(0, 3)).to_array().plot(col='time', row='variable')
        ds.to_netcdf('../sla-data-registry/qdata/orca25.nc')
        toto = xr.open_dataset('../sla-data-registry/qdata/orca25.nc')
        toto.close() 
        toto
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def fmt_enatl():
    try:
        xr.open_dataset('../sla-data-registry/DAC_ERA_INTERIM_GF.nc')
        # 2009-11-14 -> 2010-03-21
        tgt_grid = xr.open_dataset('../sla-data-registry/qdata/natl20.nc')

        (tgt_grid.ssh - tgt_grid.nadir_obs).pipe(np.abs).pipe(np.nanmean)


        def proc(samp):
            ssamp = samp - samp.mean('x').mean('y') + samp.mean()
            pp_samp = ( ssamp
             .assign(lat=lambda ds: ds.nav_lat.isel(x=600), lon=lambda ds: ds.nav_lon.isel(y=600))
             .coarsen(x=3, y=3, boundary='trim').mean()
             .swap_dims(x='lon', y='lat').sossheig
             .interp(lon=tgt_grid.lon, lat=tgt_grid.lat)
            )

            return pp_samp.mean('time_counter'), pp_samp.sel(time_counter=samp.time_counter.mean(), method='nearest')

        nadirs = {
            nad: swath_calib.utils.get_nadir_slice(
                f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
                lat_min=31,
                lat_max=45,
                lon_min=293,
                lon_max=307,
            )
            for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}

        blb0_mean = []
        blb0_snapshot = []
        for f in tqdm(sorted(list(Path('../sla-data-registry/enatl60').glob('*BLB0*')))):
            samp = xr.open_dataset(f)
            mean, snap = proc(samp)
            blb0_mean.append(mean)
            blb0_snapshot.append(snap)
        
        bods = xr.concat(blb0_mean, 'time_counter').rename(time_counter='time').drop('nav_lon').drop('nav_lat')
        bods_snap = xr.concat(blb0_snapshot, 'time_counter').rename(time_counter='time').sortby('time')
        bods_obs_ds = (sample_nadirs_from_simu(nadirs, bods_snap)).ssh

        ds = xr.Dataset(dict(ssh=(bods.dims, bods.values), nadir_obs=(bods_obs_ds.dims, bods_obs_ds.values)), coords=bods.coords).load()
        ds_wo_tide = ds.sortby('time')
        ds_wo_tide['time']= pd.to_datetime('2009-07-01') + ds_wo_tide['time']*pd.to_timedelta('1D')
        ds_wo_tide.to_netcdf(f'../sla-data-registry/qdata/enatl_wo_tide.nc')

        blbt_snapshot = []
        blbt_mean = []
        for f in tqdm(sorted(list(Path('../sla-data-registry/enatl60').glob('*BLBT*')))):
            samp = xr.open_dataset(f)
            samp = samp - samp.mean('x').mean('y')
            mean, snap = proc(samp)
            blbt_mean.append(mean)
            blbt_snapshot.append(snap)
        

        btds = xr.concat(blbt_mean, 'time_counter').rename(time_counter='time').drop('nav_lon').drop('nav_lat')
        btds_snap = xr.concat(blbt_snapshot, 'time_counter').rename(time_counter='time').sortby('time')
        btds_obs_ds = ( sample_nadirs_from_simu(nadirs, btds_snap)).ssh


        ds = xr.Dataset(dict(ssh=(btds.dims, btds.values), nadir_obs=(btds_obs_ds.dims, btds_obs_ds.values)), coords=btds.coords).load()
        ds_w_tide = ds.sortby('time')
        ds_w_tide['time']= pd.to_datetime('2009-07-01') + ds_w_tide['time']*pd.to_timedelta('1D')
        ds_w_tide.to_netcdf(f'../sla-data-registry/qdata/enatl_w_tide.nc')

        
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()





def new_enatl_fmt():
    import xesmf as xe
    from tqdm import tqdm
    from pathlib import Path
    import xarray as xr
    import matplotlib.pyplot as plt
    import numpy as np

    tgt_grid = xr.open_dataset('../sla-data-registry/qdata/natl20.nc')
    nadirs = {
        nad: swath_calib.utils.get_nadir_slice(
            f'../sla-data-registry/sensor_zarr/zarr/nadir/{nad}',
            lat_min=31,
            lat_max=45,
            lon_min=293,
            lon_max=307,
        )
        for nad in ['tpn', 'en', 'j1', 'g2', 'swot']}


    dac = xr.open_dataset('../sla-data-registry/DAC_ERA_INTERIM_GF.nc')

    ds_out = xe.util.grid_2d(-67, -53, 0.05, 31, 45, 0.05)
    dac.nav_lon.attrs=samp.nav_lon.attrs
    dac.nav_lat.attrs=samp.nav_lat.attrs
    dac = dac.rename(nav_lat='lat', nav_lon='lon')


    reggridder = xe.Regridder(dac, ds_out, "bilinear")
    dac_out = reggridder(dac, keep_attrs=True)
    domain = {'lat': slice(32, 44), 'lon':slice(-66,-54)}
    pp_dac = (
        dac_out.assign(
            lat=lambda ds: ds.lat.isel(x=0) - 0.025,
            lon=lambda ds: ds.lon.isel(y=0) - 0.025,
        ).swap_dims(x='lon', y='lat').sel(domain)
    )
    pp_dac.DAC_ERA_INTERIM.isel(time_counter=slice(0,81,9)).plot(col="time_counter", col_wrap=3)

    def proc(samp, regridder=None):
        # ssamp = samp - samp.mean('x').mean('y') + samp.mean()
        ds = samp.rename(nav_lat='lat', nav_lon='lon').load().isel(
                x=(samp.nav_lon.isel(y=600)>-67) & (samp.nav_lon.isel(y=600)<-53),
                y=(samp.nav_lat.isel(x=600)>32) & (samp.nav_lat.isel(x=600)<46),
        )
        ds[['lat', 'lon']] = ds[['lat', 'lon']].to_dataframe().assign(
            lat=lambda df: df.lat.map(lambda l: np.nan if l==0 else l),
            lon=lambda df: df.lon.map(lambda l: np.nan if l==0 else l)
        ).interpolate().to_xarray()
        if regridder is None:
            reggridder = xe.Regridder(ds, ds_out, "bilinear")
        dr_out = reggridder(ds, keep_attrs=True)
        domain = {'lat': slice(32, 44), 'lon':slice(-66,-54)}
        pp_samp = (
            dr_out.assign(
                lat=lambda ds: ds.lat.isel(x=0) - 0.025,
                lon=lambda ds: ds.lon.isel(y=0) - 0.025,
            ).swap_dims(x='lon', y='lat').sel(domain)
        )
        pp_samp =  pp_samp + pp_dac.sel(time_counter=pp_samp.time_counter).DAC_ERA_INTERIM
        pp_samp['lat'] = tgt_grid.lat
        pp_samp['lon'] = tgt_grid.lon
        pp_samp = xr.where(tgt_grid.ssh.isel(time=0).pipe(np.isfinite), pp_samp, xr.full_like(pp_samp, np.nan))
        pp_samp = pp_samp - pp_samp.mean(('lat', 'lon'))
        snap = pp_samp.sel(time_counter=samp.time_counter.mean(), method='nearest')
        mean = pp_samp.mean('time_counter').assign(time_counter=snap.time_counter)
        return regridder, (mean, snap)

    out = proc(samp)
    samp.mean("time_counter").std()
    mean.sossheig.std()

    def build_enatl(g="BLB0"):
        regridder = None
        blb0_mean = []
        blb0_snapshot = []
        for i,f in enumerate(tqdm(sorted(list(Path('../sla-data-registry/enatl60').glob(f'*{g}*'))))):
            samp = xr.open_dataset(f)
            regridder, (mean, snap) = proc(samp, regridder)
            blb0_mean.append(mean)
            blb0_snapshot.append(snap)
            # if i > 1:
            #     break

        bods = xr.concat(blb0_mean, 'time_counter').drop('time').rename(time_counter='time').drop('time_centered').sortby('time').sossheig
        bods_snap = xr.concat(blb0_snapshot, 'time_counter').drop('time').rename(time_counter='time').drop('time_centered').sortby('time').sossheig
        bods_obs_ds = (sample_nadirs_from_simu(nadirs, bods_snap)).ssh
        return bods, bods_snap, bods_obs_ds

    bods, bods_snap, bods_obs_ds = build_natl('BLB0')
    ds = xr.Dataset(dict(ssh=(bods.dims, bods.values), nadir_obs=(bods_obs_ds.dims, bods_obs_ds.values)), coords=bods.coords).load()
    ds_wo_tide = ds.sortby('time')
    # xr.open_dataset(f'../sla-data-registry/qdata/enatl_wo_tide.nc').std()
    ds_wo_tide.to_netcdf(f'../sla-data-registry/qdata/enatl_wo_tide.nc')


    btds, btds_snap, btds_obs_ds = build_enatl('BLBT')
    ds = xr.Dataset(dict(ssh=(btds.dims, btds.values), nadir_obs=(btds_obs_ds.dims, btds_obs_ds.values)), coords=btds.coords).load()
    ds_w_tide = ds.sortby('time')
    ds_w_tide.to_netcdf(f'../sla-data-registry/qdata/enatl_w_tide.nc')

def new_enatl_w_tide_fmt():
    tgt_grid = xr.open_dataset('../sla-data-registry/qdata/natl20.nc')
    sla_5nad = xr.open_zarr('../sla-data-registry/enatl_preproc/SLA_SSH_5nadirs.zarr')
    sla_5nad.load()
    enatl = xr.open_zarr('../sla-data-registry/enatl_preproc/truth_SLA_SSH_NATL60.zarr')
    enatl.load()
    
    regridder = xe.Regridder(enatl, tgt_grid, "bilinear")
    enatl_out = regridder(enatl)
    regridder = xe.Regridder(sla_5nad, tgt_grid, "bilinear")
    obs_out = regridder(sla_5nad)
    enatl_obs = (sample_nadirs_from_simu(nadirs, enatl_out.ssh)).ssh
    ds = xr.Dataset(dict(ssh=(enatl_out.ssh.dims, enatl_out.ssh.values), nadir_obs=(enatl_obs.dims, enatl_obs.values)), coords=enatl_out.coords).load()
    ds.to_array().pipe(np.isnan).isel(time=slice(0, None, 45)).plot(col='variable', row='time')
    ds.to_netcdf(f'../sla-data-registry/qdata/enatl_w_tide.nc')


def fmt_duacs_emul():

    try:

        ose_oi_path= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc'
        ose_obs_mask_path= '/raid/localscratch/qfebvre/sla-data-registry/data_OSE/NATL/training/dataset_nadir_0d.nc'
        tgt_grid = xr.open_dataset('../sla-data-registry/qdata/natl20.nc')
        obs_ds = xr.open_dataset(ose_obs_mask_path)
        oi_ds = xr.open_dataset(ose_oi_path)

        obs = obs_ds.sel(lat=tgt_grid.lat, lon=tgt_grid.lon, method='nearest').ssh
        tgt = oi_ds.sel(lat=tgt_grid.lat, lon=tgt_grid.lon, method='nearest').ssh

        ds = xr.Dataset(dict(ssh=(tgt.dims, tgt.values), nadir_obs=(obs.dims, obs.values)), coords=tgt.coords)
        ds.to_netcdf(f'../sla-data-registry/qdata/duacs_emul_ose.nc')
        ds.close()
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def main():
    try:
        # fn = run1
        fn = fmt_enatl

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    locals().update(main())
