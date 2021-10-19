from . import params, dim_range, time_period, def_time, time

dim_range['lat'] = slice(33, 43)
dim_range['lon'] = slice(-65, -55)

time_period['train_slices'] = (slice("2012-10-01", "2013-09-01"),slice('2016-12-30', "2018-01-02"),)
time_period['test_slices'] = (slice('2016-12-30', "2018-01-02"),)
time_period['val_slices'] = (slice('2013-09-01', "2013-10-01"),slice('2017-03-01', "2017-04-01"),)

params['dataloading']= 'with_sst'

params['files_cfg'].update( 
    dict(
            oi_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/oi_OSE_OSSE.nc',
            oi_var='ssh',
            obs_mask_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/data_OSE_OSSE.nc',
            obs_mask_var='ssh',
            #gt_path=None,
            #gt_var=None,
            gt_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_ssh_OSE_OSSE.nc',
            gt_var='ssh',
            sst_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ref_sst_OSE_OSSE.nc',
            sst_var='sst',
    )
)

new_time = def_time(params,time_period)
time['time_train'] = new_time['time_train']
time['time_val'] = new_time['time_val']
time['time_test'] = new_time['time_test']

params['supervised'] = False
params['alpha_mse_ssh'] = 10.
params['alpha_mse_gssh'] = 1.

