from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

BASE_DEFAULTS = [
  '/xp/qfebvre/base_xp',
  '/splits/dc_boost_swot@datamodule',
]
XP=12
for  defaults in product(
        [
            ('aug1', '/xp/qfebvre/aug/with_aug_state.yaml'),
            ('aug2','/xp/qfebvre/aug/with_double_aug_state.yaml'),
            # ('aug0','/xp/qfebvre/aug/without_aug_state.yaml.yam'),
        ],
        [
            ('dp240','/xp/qfebvre/dl/d240_p240x5_s240x1.yaml'),
            # '/xp/qfebvre/dl/d200_p200x5_s200x1.yaml'
        ],
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
        ],
        [
            ('cal', '/xp/qfebvre/loss/calmap.yaml'),
            ('map', '/xp/qfebvre/loss/map.yaml'),
            # '/xp/qfebvre/loss/calmap_grad.yaml',
            # '/xp/qfebvre/loss/map_grad.yaml',
        ],
        [
            ('sst', '/xp/qfebvre/sst/with_sst.yaml'),
            ('no_sst', '/xp/qfebvre/sst/without_sst.yaml'),
        ],
        [
            ('ng5x3cas', '/xp/qfebvre/train_strat/const_lr_ngrad_5_3cas.yaml'),
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
    for xp in cs.list('xp'):
        print(xp)

