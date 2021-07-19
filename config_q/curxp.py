from . import params
params['files_cfg'].update( 
    dict(
        obs_target_mask_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        obs_target_mask_var='swot',
        obs_mask_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        obs_mask_var='nad_swot_roll',
    )
)

