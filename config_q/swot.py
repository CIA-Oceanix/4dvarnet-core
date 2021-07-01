from . import params

params['files_cfg'].update( 
    dict(
        oi_path='/gpfsscratch/rech/nlu/commun/large/ssh_NATL60_swot_4nadir.nc',
        oi_var='ssh_mod',
        obs_mask_path='/gpfsscratch/rech/yrf/ual82ir/4dvarnet-core/full_cal_obs.nc',
        obs_mask_var='nad_swot',
    )
)

