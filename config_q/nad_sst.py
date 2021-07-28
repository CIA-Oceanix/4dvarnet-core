from . import params

params['dataloading']= 'with_sst'
params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        obs_mask_var='nad',
        obs_target_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        obs_target_var='swot',
        sst_path='/gpfsscratch/rech/nlu/commun/large/NATL60-CJM165_NATL_sst_y2013.1y.nc',
        sst_var='sst'
    )
)
