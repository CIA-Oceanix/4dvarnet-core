from . import params

params['dataloading']= 'with_sst'
params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfswork/rech/yrf/commun/CalData/full_cal_obs.nc',
        obs_mask_var='nad_swot_roll',
        sst_path='/gpfsscratch/rech/nlu/commun/large/NATL60-CJM165_NATL_sst_y2013.1y.nc',
        sst_var='sst'
    )
)
