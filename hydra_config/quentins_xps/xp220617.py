from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

BASE_DEFAULTS = [
  '/xp/qfebvre/xp_oi',
  '/splits/dc_boost_swot@datamodule',
]
XP=22

cs.store(name='0', node={'aug_train_data': False}, group='aug_data', package='datamodule')
cs.store(name='1', node={'aug_train_data': True}, group='aug_data', package='datamodule')
cs.store(name='2', node={'aug_train_data': 2}, group='aug_data', package='datamodule')
cs.store(name='3', node={'aug_train_data': 3}, group='aug_data', package='datamodule')
cs.store(name='8', node={'aug_train_data': 8}, group='aug_data', package='datamodule')
aug = {
    'aug0': '/aug_data/0',
    'aug1': '/aug_data/1',
    'aug2': '/aug_data/2',
    'aug3': '/aug_data/3',
    'aug5': '/aug_data/8',
}

cs.store(name='2', node={'resize_factor': 2}, group='down_samp', package='datamodule')
cs.store(name='1', node={'resize_factor': 1}, group='down_samp', package='datamodule')
resize = {
    'ds2': '/down_samp/2',
    'ds1': '/down_samp/1',
}

# cs.store(name='11', node={
#     'patch_weight': {
#         '_target_': 'lit_model_augstate.get_constant_crop',
#         'patch_size': '${datamodule.slice_win}',
#         'crop': { 'time': 3, 'lat': 20, 'lon': 20, }
#     },
#     'dT': 11,
# }, group='custom', package='params')

for  defaults in product(
        [
            # ('swot', '/xp/qfebvre/ds/clean_swot_oi_no_swot.yaml'),
            ('5nad', '/xp/qfebvre/ds/five_nadirs.yaml'),
            ('swot_w_oi', '/xp/qfebvre/ds/clean_swot_oi_w_swot.yaml'),
            # ('roll', '/xp/qfebvre/ds/new_noisy_swot_roll.yaml'),
            # ('syst', '/xp/qfebvre/ds/new_noisy_swot_syst_errs.yaml'),
            # ('noisy', '/xp/qfebvre/ds/new_noisy_swot.yaml'),
        ],
        aug.items(),
        resize.items(),
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

