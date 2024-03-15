import xarray as xr

data = xr.open_dataset('dashboard/xp_BBP_dt7/lightning_logs/version_0/test.nc')
data = data[['pred','obs_inp']]
data.to_netcdf('dashboard/xp_BBP_dt7/lightning_logs/version_0/test0.nc')
