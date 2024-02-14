import sys
import xarray as xr
import shutil

data = xr.open_dataset('dashboard/xp_DIATO_dt7_2/lightning_logs/version_0/test.nc')
data = data[['pred']].sel(time=data.time.dt.month==2)
data.to_netcdf('/DATASET/turbidity/DIATO/full_med_interp/' + sys.argv[1] + '/02.nc')
