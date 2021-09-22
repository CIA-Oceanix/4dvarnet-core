from .. import * 
from ..swot import params
import copy
params = copy.deepcopy(params)

params['model'] = 'vit'
params['vit_mask'] = True
params['drop_out_rate'] = 0.2
params['drop_out_attn'] = 0.2
params['out_c'] = 1000
params['depth'] = 6
params['num_heads'] = 10

params['swot_anom_wrt'] = 'low_res'
# params['swot_anom_wrt'] = 'high_res'
params['anom_swath_init'] = 'zeros'
# params['anom_swath_init'] = 'obs'
params['loss_glob'] = 1
params['loss_loc'] = 0
params['loss_proj'] = 0
params['loss_low_res'] = 0
#
params.update({
    'alpha_proj'      : 0.,
    'alpha_sr'        : 0.,
    'alpha_lr'        : 0.,  
    'alpha_mse_ssh'   : 10.,
    'alpha_mse_gssh'  : 5.,

    'alpha_loc_mse_ssh'   : 0.,
    'alpha_loc_mse_gssh'  : 0.,
})
