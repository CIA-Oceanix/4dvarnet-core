import sys
import xarray as xr
import shutil

data = xr.open_dataset('dashboard/xp_CHL_dt7/lightning_logs/version_0/test.nc')
data = data[['pred']].sel(time=data.time.dt.month==4)
data.to_netcdf('/DATASET/turbidity/CHL/full_med_interp/' + sys.argv[1] + '/04.nc')
