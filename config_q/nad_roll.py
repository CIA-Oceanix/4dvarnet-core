from . import params

params['dataloading']= 'with_sst'
params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc',
        obs_mask_var='nad',
        sst_path='/gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc',
        sst_var='swot_roll',
    )
)
