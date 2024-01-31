import sys
import xarray as xr
import shutil

data = xr.open_dataset('dashboard/xp_BBP_dt7_2/lightning_logs/version_0/test.nc')
data = data[['pred']].sel(time=data.time.dt.month==2)
data.to_netcdf('/DATASET/turbidity/BBP443/full_med_interp/' + sys.argv[1] + '/02.nc')
