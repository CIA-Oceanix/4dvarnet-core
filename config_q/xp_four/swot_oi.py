from .. import *
from ..swot import params
import copy
params = copy.deepcopy(params)
params['files_cfg']['oi_path'] = '/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc'
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
