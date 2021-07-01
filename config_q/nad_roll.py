from . import params

params['dataloading']= 'with_sst'
params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        obs_mask_var='nad',
        sst_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        sst_var='swot_roll',
    )
)
