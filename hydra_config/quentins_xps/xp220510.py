from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

BASE_DEFAULTS = [
  '/xp/qfebvre/base_xp',
  '/splits/dc_boost_swot@datamodule',
]
XP=16
# norm
cs.store(name='l1', node={'norm_obs': 'l1', 'norm_prior': 'l1'}, group='norm', package='params')
cs.store(name='l2', node={'norm_obs': 'l2', 'norm_prior': 'l2'}, group='norm', package='params')
norm = {
        # 'l1': '/norm/l1',
        'l2': '/norm/l2'
        }

# dropout
cs.store(name='01_01', node={'dropout': 0.1, 'dropout_phi_r': 0.1}, group='dropout', package='params')
cs.store(name='025_00', node={'dropout': 0.25, 'dropout_phi_r': 0.}, group='dropout', package='params')
dropout = {
        # 'dp01_01': '/dropout/01_01',
        'dp025_00': '/dropout/025_00'
}

cs.store(name='yes', node={'aug_train_data': True}, group='dataaug', package='params')
cs.store(name='no', node={'aug_train_data': True}, group='dataaug', package='params')
dataaug = {
        'no_dataaug': '/dataaug/no',
        'dataaug': '/dataaug/yes'
}
for  defaults in product(
        [
            # ('aug1', '/xp/qfebvre/aug/with_aug_state.yaml'),
            ('aug2','/xp/qfebvre/aug/with_double_aug_state.yaml'),
            # ('aug0','/xp/qfebvre/aug/without_aug_state.yaml.yam'),
        ],
        [
            ('dp240','/xp/qfebvre/dl/d240_p240x5_s240x1.yaml'),
            # '/xp/qfebvre/dl/d200_p200x5_s200x1.yaml'
        ],
        [
            # ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
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
        norm.items(),
        dropout.items(),
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


XP=17

for  defaults in product(
        [
            # ('aug1', '/xp/qfebvre/aug/with_aug_state.yaml'),
            ('aug2','/xp/qfebvre/aug/with_double_aug_state.yaml'),
            # ('aug0','/xp/qfebvre/aug/without_aug_state.yaml.yam'),
        ],
        [
            ('dp240','/xp/qfebvre/dl/d240_p240x5_s240x1.yaml'),
            # '/xp/qfebvre/dl/d200_p200x5_s200x1.yaml'
        ],
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            # ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
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
        norm.items(),
        dropout.items(),
    ):
    labels, defaults = zip(*defaults)
    defaults_xp_name = '_'.join(labels)

    xp_name = f'qxp{XP}_' + defaults_xp_name
    cfg = {
            'xp_name': xp_name,
            'defaults': BASE_DEFAULTS + list(defaults) + ['_self_'],
        }
    cs.store(name=xp_name, node=cfg, group='xp', package='_global_')

XP=17

for  defaults in product(
        [
            # ('aug1', '/xp/qfebvre/aug/with_aug_state.yaml'),
            ('aug2','/xp/qfebvre/aug/with_double_aug_state.yaml'),
            # ('aug0','/xp/qfebvre/aug/without_aug_state.yaml.yam'),
        ],
        [
            ('dp240','/xp/qfebvre/dl/d240_p240x5_s240x1.yaml'),
            # '/xp/qfebvre/dl/d200_p200x5_s200x1.yaml'
        ],
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            # ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
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
        norm.items(),
        dropout.items(),
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

XP=18

for  defaults in product(
        [
            ('aug1', '/xp/qfebvre/aug/with_aug_state'),
            # ('aug2','/xp/qfebvre/aug/with_double_aug_state.yaml'),
            ('aug0','/xp/qfebvre/aug/without_aug_state'),
        ],
        [
            ('dp240','/xp/qfebvre/dl/d240_p240x5_s240x1.yaml'),
            # '/xp/qfebvre/dl/d200_p200x5_s200x1.yaml'
        ],
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            # ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            # ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
        ],
        [
            # ('cal', '/xp/qfebvre/loss/calmap.yaml'),
            ('map', '/xp/qfebvre/loss/map.yaml'),
            # '/xp/qfebvre/loss/calmap_grad.yaml',
            # ('map_grad', '/xp/qfebvre/loss/map_grad.yaml'),
        ],
        [
            ('sst', '/xp/qfebvre/sst/with_sst.yaml'),
            ('no_sst', '/xp/qfebvre/sst/without_sst.yaml'),
        ],
        [
            ('ng5x3cas', '/xp/qfebvre/train_strat/const_lr_ngrad_5_3cas.yaml'),
        ],
        norm.items(),
        dropout.items(),
        dataaug.items(),
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

XP=19

for  defaults in product(
        [
            # ('aug1', '/xp/qfebvre/aug/with_aug_state'),
            ('aug2','/xp/qfebvre/aug/with_double_aug_state.yaml'),
            # ('aug0','/xp/qfebvre/aug/without_aug_state'),
        ],
        [
            ('dp240','/xp/qfebvre/dl/d240_p240x5_s240x1.yaml'),
            # '/xp/qfebvre/dl/d200_p200x5_s200x1.yaml'
        ],
        [
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            # ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
        ],
        [
            ('cal', '/xp/qfebvre/loss/calmap.yaml'),
            ('map', '/xp/qfebvre/loss/map.yaml'),
            # '/xp/qfebvre/loss/calmap_grad.yaml',
            # ('map_grad', '/xp/qfebvre/loss/map_grad.yaml'),
        ],
        [
            ('sst', '/xp/qfebvre/sst/with_sst.yaml'),
            ('no_sst', '/xp/qfebvre/sst/without_sst.yaml'),
        ],
        [
            ('ng5x3cas', '/xp/qfebvre/train_strat/const_lr_ngrad_5_3cas.yaml'),
        ],
        norm.items(),
        dropout.items(),
        dataaug.items(),
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

