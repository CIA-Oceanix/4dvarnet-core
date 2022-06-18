import xarray as xr

# OBS
f1 = xr.open_dataset("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/dataset_nadir_0d.nc")
f2_nadswot = xr.open_dataset("/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc")
f2_nadswot = f2_nadswot[['mask', 'lag','flag','ssh_obs']]
f2_nadswot = f2_nadswot.rename({'ssh_obs':'ssh'})
f2_nad = xr.open_dataset("/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d.nc")
f2_nad = f2_nad[['mask', 'lag','flag','ssh_obs']]
f2_nad = f2_nad.rename({'ssh_obs':'ssh'})
f_nadswot = xr.combine_by_coords([f1,f2_nadswot])
f_nad = xr.combine_by_coords([f1,f2_nad])
f_nadswot.to_netcdf("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/data_OSE_OSSE_nadswot.nc")
f_nad.to_netcdf("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/data_OSE_OSSE_nad.nc")

# OI
f1 = xr.open_dataset("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc")
f2_nadswot = xr.open_dataset("/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc")
f2_nadswot = f2_nadswot[['ssh_obs']]
f2_nadswot = f2_nadswot.rename({'ssh_obs':'ssh'})
f_nadswot = xr.combine_by_coords([f1,f2_nadswot])
f_nadswot.to_netcdf("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/oi_OSE_OSSE_nadswot.nc")
f2_nad = xr.open_dataset("/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_4nadir.nc")
f2_nad = f2_nad[['ssh_obs']]
f2_nad = f2_nad.rename({'ssh_obs':'ssh'})
f_nad = xr.combine_by_coords([f1,f2_nad])
f_nad.to_netcdf("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/oi_OSE_OSSE_nad.nc")

# REF
f1 = xr.open_dataset("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/dataset_nadir_0d.nc")
f2 = xr.open_dataset("/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc")
f2.time.attrs["units"] = "seconds since 2012-09-30 12:00:00"
f2 = xr.decode_cf(f2)
f1 = f1[['ssh']]
f = xr.combine_by_coords([f1,f2])
f.to_netcdf("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_ssh_OSE_OSSE.nc")

# REF_SST
f1 = xr.open_dataset("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/sst_CMEMS.nc")
f2 = xr.open_dataset("/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_sst_y2013.1y.nc")
f2.time.attrs["units"] = "seconds since 2012-09-30 12:00:00"
f2 = xr.decode_cf(f2)
f1 = f1[['analysed_sst']]
f1 = f1.rename({'analysed_sst':'sst'})
# Kelvin to Celsius
newval = f1.sst.values-273.
f1 = f1.update({ 'sst' : (('time','lat','lon'),newval)})
f = xr.combine_by_coords([f1,f2])
f.to_netcdf("/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_sst_OSE_OSSE.nc")

