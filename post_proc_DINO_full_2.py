import sys
import xarray as xr
import shutil

data = xr.open_dataset('dashboard/xp_DINO_dt7_2/lightning_logs/version_0/test.nc')
data = data[['pred']].sel(time=data.time.dt.month==10)
data.to_netcdf('/DATASET/turbidity/DINO/full_med_interp/' + sys.argv[1] + '/10.nc')
