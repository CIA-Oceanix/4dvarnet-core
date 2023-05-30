#!/bin/bash

concat_simu () { cat << HEREDOC
import xarray as xr
import numpy as np
N=${N}
data = xr.open_dataset('dashboard/oi_osse_spde/lightning_logs/version_'+str(N)+'/maps.nc')
for i in range(1,2):
    new_simu = xr.open_dataset('dashboard/oi_osse_spde/lightning_logs/version_'+str(N+i)+'/maps.nc')
    data = data.update({'simu':(('time','lat', 'lon', 'daw', 'nsimu'),
                                np.concatenate((data.simu.values,
                                                new_simu.simu.values),axis=4)),
		        'nsimu': np.arange(2*(i+1))})
nc = data
nc.to_netcdf('tmp_${start}.nc')
HEREDOC
}

ldate=($(seq 0 4 42 | xargs -I {} date -d "2012-10-22 {} days" +%Y-%m-%d) '2012-12-03')
len=${#ldate[@]}
# run all the dates
echo $len
for (( i=1; i<$len; i++ )); do
  start="${ldate[$((i-1))]}"
  end="${ldate[$i]}"
  end=$(date -d "$end + 3 days" '+%Y-%m-%d')
  echo "### RUN DATE ${start} to ${end} ###"
  # run all the simulations
  for (( j=0; j<2; j++ )); do
    echo "### RUN SIMU $((j*5)) to $(((j+1)*5)) ###"
    CUDA_VISIBLE_DEVICES=7 HYDRA_FULL_ERROR=1 python hydra_main.py xp=mbeaucha/xp_spde/ssh/oi_osse_spde file_paths=mee_a100_ssh entrypoint=test datamodule.test_slices.0._args_.0='"'$start'"' datamodule.test_slices.0._args_.1='"'$end'"' +entrypoint.ckpt_path=../../model_oi_osse_spde.ckpt
  done
  # merge NetCDF 
  N=`ls dashboard/oi_osse_spde/lightning_logs/ | tail -n 1 | cut -f2 -d'_'`
  N=$((N-1))
  concat_simu | python -
done
# merge NetCDF 
python <<HEREDOC
import xarray as xr
nc = xr.open_mfdataset('tmp*.nc',combine='nested',concat_dim='time')
nc.to_netcdf('/homes/m19beauc/maps_osse_4DvarNet_SPDE.nc') 
HEREDOC  

# cleaning
rm -rf tmp*.nc
cd dashboard
# clean before
for dir in `ls .` ; do
  if [ -d ${dir}/lightning_logs ] ; then
    echo ${dir}/lightning_logs
    cd ${dir}/lightning_logs
    ls -lt | tail -n +3 | awk '{print $9}' | xargs rm -r
    cd ../..
  fi
done
cd ..
