from . import params, dim_range, time_period

dim_range['lat'] = slice(33, 43)
dim_range['lon'] = slice(-65, -55)

time_period['train_slices'] = (slice('2016-12-30', "2018-01-02"),)
time_period['test_slices'] = (slice('2016-12-30', "2018-01-02"),)
time_period['val_slices'] = (slice('2017-01-01', "2017-01-25"),)

params['dataloading']= 'with_sst'
params['files_cfg'].update( 
    dict(
            oi_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc',
            oi_var='ssh',
            obs_mask_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/dataset_nadir_0d.nc',
            obs_mask_var='ssh',
            #gt_path=None,
            #gt_var=None,
            gt_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/dataset_nadir_0d.nc',
            gt_var='ssh',
            sst_path='/gpfsstore/rech/yrf/uba22to/data_OSE/NATL/training/sst_CMEMS.nc',
            sst_var='analysed_sst',
    )
)
params['supervised'] = False

