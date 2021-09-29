from . import params, dim_range, time_period

dim_range['lat'] = slice(33, 43)
dim_range['lon'] = slice(-65, -55)

time_period['train_slices'] = (slice("2012-10-01", "2013-09-01"),slice('2017-01-01', "2017-12-31"),)
time_period['test_slices'] = (slice('2017-01-01', "2017-12-31"),)
time_period['val_slices'] = (slice('2013-09-01', "2013-10-01"),slice('2017-03-01', "2017-04-01"),)

params['files_cfg'].update( 
    dict(
            oi_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/oi_OSE_OSSE.nc',
            oi_var='ssh',
            obs_mask_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/data_OSE_OSSE.nc',
            obs_mask_var='ssh',
            gt_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_ssh_OSE_OSSE.nc',
            gt_var='ssh',
            sst_path=None,
            sst_var=None,
    )
)
