from .. import *
from ..all_errs import params
params['swot_anom_wrt'] = 'low_res'
# params['swot_anom_wrt'] = 'high_res'
params['anom_swath_init'] = 'zeros'
# params['anom_swath_init'] = 'obs'
params['loss_glob'] = 1
params['loss_loc'] = 0
params['loss_proj'] = 1
params['loss_low_res'] = 1
# params['state'] = 'global'
# params['state'] = 'local'
# params['state'] = 'glob_loc'
