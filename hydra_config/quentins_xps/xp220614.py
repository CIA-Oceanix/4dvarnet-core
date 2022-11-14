from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

BASE_DEFAULTS = [
  '/xp/baseline/full_core_hanning',
  '/splits/dc_boost_swot@datamodule',
]
XP=21
# norm

cs.store(name='11', node={
    'patch_weight': {
        '_target_': 'lit_model_augstate.get_constant_crop',
        'patch_size': '${datamodule.slice_win}',
        'crop': { 'time': 3, 'lat': 20, 'lon': 20, }
    },
    'dT': 11,
}, group='custom', package='params')

for  defaults in product(
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            # ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
        ],
        [
            ('sst', '/xp/qfebvre/sst/with_sst.yaml'),
            ('no_sst', '/xp/qfebvre/sst/without_sst.yaml'),
        ],
        [
            ('11', '/custom/11'),
        ],
    ):
    labels, defaults = zip(*defaults)
    defaults_xp_name = '_'.join(labels)

    xp_name = f'qxp{XP}_' + defaults_xp_name 
    cfg = {
            'xp_name': xp_name,
            'defaults': BASE_DEFAULTS + list(defaults) + ['_self_'],
        }
    cs.store(name=xp_name, node=cfg, group='xp', package='_global_')
    # print(cs.list('xp'))


if __name__== '__main__':
	for xp  in cs.list('xp'):
		print(xp)

