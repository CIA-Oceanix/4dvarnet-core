from . import params, slice_win, strides

params['files_cfg'].update( 
    dict(
        obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
    )
)

params['stochastic'] = True
params['resize_factor'] = 2
params['sS'] = 4

slice_win['lon'] = int(slice_win['lon']/params['resize_factor'])
slice_win['lat'] = int(slice_win['lat']/params['resize_factor'])
strides['lon'] = int(strides['lon']/params['resize_factor'])
strides['lat'] = int(strides['lat']/params['resize_factor'])


