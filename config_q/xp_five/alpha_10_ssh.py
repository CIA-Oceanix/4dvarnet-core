from .. import * 
from ..all_errs import params
import copy
params = copy.deepcopy(params)

params['model'] = '4dvarnet'
params['swot_anom_wrt'] = 'low_res'
# params['swot_anom_wrt'] = 'high_res'
params['anom_swath_init'] = 'zeros'
# params['anom_swath_init'] = 'obs'
params['loss_glob'] = 1
params['loss_loc'] = 1
params['loss_proj'] = 1
params['loss_low_res'] = 1
#
params['alpha_loc_mse_ssh'] = 100.
params['alpha_loc_mse_gssh']  = 20.
