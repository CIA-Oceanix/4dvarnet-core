from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

BASE_DEFAULTS = [
  '/xp/xp_aug/xp_repro/full_core_hanning',
  '/splits/dc_boost_swot@datamodule',
]
XP=20
# norm
for  defaults in product(
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            # ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
        ],
        [
            ('sst', '/xp/qfebvre/sst/with_sst.yaml'),
            ('no_sst', '/xp/qfebvre/sst/without_sst.yaml'),
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

