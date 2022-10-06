import xarray as xr
from pathlib import Path
import hydra_config
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xrft
from utils import get_cfg, get_dm, get_model
cfg_n = 'qxp26_no_sst_natl1_1D_aug1_ds2_dT29_8'


cfg = get_cfg(cfg_n, overrides=[ 'file_paths=jz',])

def interpolate_na_2D(ds, max_value=100.):
    return (
            ds.where(np.abs(ds) < max_value, np.nan)
            .to_dataframe()
            .interpolate(method='linear')
            .pipe(xr.Dataset.from_dataframe)
)


psd_fn = lambda da: xrft.isotropic_power_spectrum(
        da, dim=('lat', 'lon'), truncate=True, window='hann')
domain = {'lat': slice(35, 44), 'lon':slice(-65,-55)}
# ds = xr.open_dataset('../sla-data-registry/natl60_degraded/ref220914.nc')
# ds = ds.rename(latitude='lat', longitude='lon').sel(domain)
# ds.zos.isel(time=0).plot()
ds = xr.open_dataset(cfg.datamodule.gt_path)
ref = (
        xr.open_dataset(cfg.file_paths.natl_ssh_daily).ssh
        .isel(time=slice(None, -1))
        .assign_coords(time=ds.time)
        .sel(ds.coords)
)

def sobel(da):
    dx_ac = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -1), da)
    dx_al = xr.apply_ufunc(lambda _da: ndimage.sobel(_da, -2), da)
    return np.hypot(dx_ac, dx_al)



glorys=xr.open_dataset('../sla-data-registry/GLORYS/preprocessed_glorys220928.nc')
glorys=xr.open_dataset('../sla-data-registry/GLORYS/preprocessed_glorys220921.nc')

glorys.ssh.isel(time=0).plot()
(glorys.five_nadirs - glorys.ssh).isel(time=0).plot()

# glorys_fmt = glorys.coarsen(lat=2, lon=2).mean().zos.assign_coords(
#     time=pd.to_datetime(glorys.time).date
# ).sel(ref.coords, method='nearest', tolerance=0.01)

dsnona = ds.assign(ref=ref).assign(glorys=glorys.ssh).pipe(interpolate_na_2D)
# dsnona = xr.Dataset({p.parents[1].name: xr.open_dataset(p).pred for p in Path('test_logs').glob('**/test.nc')})
tpds = dsnona.map(lambda da: ndimage.laplace(da))#.isel(time=slice(None, 10, 1))
# tpds = dsnona.map(lambda da: ndimage.gaussian_laplace(da, sigma=1)).isel(time=slice(None, 10, 1))
# tpds = dsnona.pipe(sobel)# .isel(time=slice(None, 10, 1))
# tpds = dsnona



psds = {}
for v in tpds:
    psd = psd_fn(tpds[v])
    psds[v]=psd.mean('time')
    # weighted_scale = psd.sum() / (psd * 100/ (psd.freq_r)).sum(dim='freq_r')

    # print(v, 1/weighted_scale.mean('time'))

psds_ds = xr.Dataset(psds)
psds_ds.to_array().plot.line(
        x='freq_r',
        hue='variable',
        xscale='log',
        yscale='log',
        figsize=(10,6)
)

weighted_scale = (
    psds_ds.sum() / (psds_ds * psds_ds.freq_r).sum('freq_r')
) 
print((100*weighted_scale).to_array().to_dataframe(name='scale (km)').to_markdown())


(
    psd
    .to_dataset(name='psd')
    .mean('time')
    .assign(
        log_r=lambda ds:np.log(1/ds.freq_r),
        log_freq_r=lambda ds:np.log(ds.freq_r),
        log_psd=lambda ds: np.log(ds.psd)
    )
    .swap_dims(freq_r='log_freq_r')
    .log_psd.plot()
)
1/weighted_scale.mean('time')
weighted_scale.plot()
dsnona.isel(time=10).ssh_025_1D.plot()
dsnona.isel(time=10).ssh_1_1D.plot()
ft025_1D = xrft.isotropic_power_spectrum(dsnona.ssh_025_1D, dim=('lat', 'lon'), truncate=True)
ft1_1D = xrft.isotropic_power_spectrum(dsnona.ssh_1_1D, dim=('lat', 'lon'), truncate=True)
ft1_14D = xrft.isotropic_power_spectrum(dsnona.ssh_1_14D, dim=('lat', 'lon'), truncate=True)

plt.plot(np.log(1/ft025_1D.freq_r.values), np.log10(ft025_1D.mean('time').values))
plt.plot(np.log(1/ft1_1D.freq_r.values), np.log10(ft1_1D.mean('time').values))
plt.plot(np.log(1/ft1_14D.freq_r.values), np.log10(ft1_14D.mean('time').values))

