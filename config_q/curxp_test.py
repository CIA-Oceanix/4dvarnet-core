from . import params
params['files_cfg'].update( 
    dict(
        obs_target_path='/gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc',
        obs_target_var='swot',
        obs_mask_path='/gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc',
        obs_mask_var='nad_swot_roll',
    )
)

