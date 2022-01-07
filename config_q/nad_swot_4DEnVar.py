from . import params

params['dataloading']= '4DEnVar'

params['files_cfg'].update(
    dict(
        obs_mask_path='/users/local/m19beauc/DATA/OSSE/data/dataset_nadir_0d_swot.nc',
        obs_mask_var='ssh_mod',
    )
)

params['stochastic'] = True
params['size_ensemble'] = 5


