from . import params

params['dataloading']= 'with_sst'
params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfsscratch/rech/nlu/commun/large/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
        sst_path='/gpfsscratch/rech/nlu/commun/large/NATL60-CJM165_NATL_sst_y2013.1y.nc',
        sst_var='sst'
    )
)
