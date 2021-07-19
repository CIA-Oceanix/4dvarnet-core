from . import params

params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
    )
)

params['stochastic'] = True

