from . import *
import copy
params = copy.deepcopy(params)

params['model']= 'passthrough'
params['model']= 'vit'

params['files_cfg'].update( 
    dict(
        oi_path='/home/q20febvr/DATA/ssh_NATL60_swot_4nadir.nc',
        # oi_path='/home/q20febvr/DATA/ssh_NATL60_4nadir.nc',
        oi_var='ssh_mod',
        obs_target_path='/home/q20febvr/DATA/dataset_nadir_0d_swot.nc',
        obs_target_var='ssh_obs',
        obs_mask_path='/home/q20febvr/DATA/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
        gt_path='/home/q20febvr/DATA/NATL60-CJM165_GULFSTREAM_ssh_y2013.1y.nc',
        gt_var='ssh',

    )
)
params['swot_anom_wrt'] = 'low_res'
# params['swot_anom_wrt'] = 'high_res'
params['anom_swath_init'] = 'zeros'
# params['anom_swath_init'] = 'obs'
params['loss_glob'] = 1
params['loss_loc'] = 1
params['loss_proj'] = 1
params['loss_low_res'] = 1
#
params['alpha_loc_mse_ssh'] = 20.
params['alpha_loc_mse_gssh']  = 200.

