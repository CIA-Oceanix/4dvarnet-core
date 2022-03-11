from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()


BASE_DEFAULTS = [
    '/xp/qfebvre/base_xp220301',
    '/splits/dc_boost_swot@datamodule',
    '/xp/qfebvre/dl/d240_p240x5_s240x1.yaml',
    '/xp/qfebvre/train_strat/const_lr_ngrad_5_3cas.yaml',
]

# yaml defaults
losses = { 'calmap': '/xp/qfebvre/loss/calmap.yaml', 'map': '/xp/qfebvre/loss/map.yaml', }
sst = { 'yes': '/xp/qfebvre/sst/with_sst.yaml', 'no': '/xp/qfebvre/sst/without_sst.yaml', }
obs = { '5nad': '/xp/qfebvre/ds/five_nadirs.yaml', 'errs': '/xp/qfebvre/ds/new_noisy_swot.yaml', }
loc_estim = {'false': '/xp/qfebvre/aug/expl_err_no_loc', 'true': '/xp/qfebvre/aug/expl_err_loc'}

# xp defaults
# err scaling
cs.store(name='03', node={'train_error_scaling': False, 'init_err_scaling':0.3}, group='err_scaling', package='params')
cs.store(name='05', node={'train_error_scaling': False, 'init_err_scaling':0.5}, group='err_scaling', package='params')
cs.store(name='07', node={'train_error_scaling': False, 'init_err_scaling':0.7}, group='err_scaling', package='params')
cs.store(name='09', node={'train_error_scaling': False, 'init_err_scaling':0.9}, group='err_scaling', package='params')
cs.store(name='1', node={'train_error_scaling': False, 'init_err_scaling':1.}, group='err_scaling', package='params')
cs.store(name='13', node={'train_error_scaling': False, 'init_err_scaling':1.3}, group='err_scaling', package='params')
cs.store(name='0', node={'train_error_scaling': False, 'init_err_scaling':0.}, group='err_scaling', package='params')
cs.store(name='m03', node={'train_error_scaling': False, 'init_err_scaling':-0.3}, group='err_scaling', package='params')
cs.store(name='m05', node={'train_error_scaling': False, 'init_err_scaling':-0.5}, group='err_scaling', package='params')
cs.store(name='m07', node={'train_error_scaling': False, 'init_err_scaling':-0.7}, group='err_scaling', package='params')
cs.store(name='m09', node={'train_error_scaling': False, 'init_err_scaling':-0.9}, group='err_scaling', package='params')
cs.store(name='m1', node={'train_error_scaling': False, 'init_err_scaling':-1.}, group='err_scaling', package='params')
cs.store(name='m13', node={'train_error_scaling': False, 'init_err_scaling':-1.3}, group='err_scaling', package='params')
err_scaling = {
    '0': '/err_scaling/0',
    'm03': '/err_scaling/m03',
    'm05': '/err_scaling/m05',
    'm07': '/err_scaling/m07',
    'm09': '/err_scaling/m09',
    'm1': '/err_scaling/m1',
    'm13': '/err_scaling/m13',
    '03': '/err_scaling/03',
    '05': '/err_scaling/05',
    '07': '/err_scaling/07',
    '09': '/err_scaling/09',
    '1': '/err_scaling/1',
    '13': '/err_scaling/13',
}

# err loss
cs.store(name='false', node={'loss_err': False}, group='loss_err', package='params')
cs.store(name='true', node={'loss_err': True}, group='loss_err', package='params')
err_loss = {'true': '/loss_err/true', 'false': '/loss_err/false'}

# err prior
cs.store(name='same', node={'err_prior': 'same'}, group='err_prior', package='params')
cs.store(name='none', node={'err_prior': 'none'}, group='err_prior', package='params')
cs.store(name='diff', node={'err_prior': 'diff'}, group='err_prior', package='params')
err_prior = {'same': '/err_prior/same', 'none': '/err_prior/none', 'diff': '/err_prior/diff'}

# loc estim


# Xps test alpha
XPS5 = {
        **{f'errs_{loss}_es{scaling}_sst{use_sst}': [
            *BASE_DEFAULTS, losses[loss], sst[use_sst], obs['errs'], err_scaling[scaling], err_loss['true'], err_prior['same'], loc_estim['false']
        ] for loss, scaling, use_sst in [
                ('map', '0', 'no'),
                ('map', 'm05', 'no'),
                ('map', 'm1', 'no'),
                ('calmap', '0', 'no'),
                ('calmap', 'm05', 'no'),
                ('calmap', 'm1', 'no'),
                ('map', '0', 'yes'),
                ('map', 'm05', 'yes'),
                ('map', 'm1', 'yes'),
                ('calmap', '0', 'yes'),
                ('calmap', 'm05', 'yes'),
                ('calmap', 'm1', 'yes'),
        ]},
}

for xp_name, xp_defaults in XPS5.items():
    cs.store(f'qxp5_{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

# Xps best swot estim with 5nad sst
XPS6 = {
        **{f'5nad_sst_{loss}_ep{ep}': [
            *BASE_DEFAULTS, losses[loss], sst['yes'], obs['5nad'], err_scaling['0'], err_loss['false'], err_prior[ep], loc_estim['false']
        ] for loss, ep in [
                ('map', 'same'),
                ('map', 'none'),
                ('calmap', 'same'),
                ('calmap', 'none'),
        ]},
}

for xp_name, xp_defaults in XPS6.items():
    cs.store(f'qxp6_{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

XPS7 = {
        **{f'qxp7_errs_{loss}_es{scaling}_sst{use_sst}_ep{ep}': [
            *BASE_DEFAULTS, losses[loss], sst[use_sst], obs['errs'], err_scaling[scaling], err_loss['true'], err_prior[ep], loc_estim['false']
        ] for loss, scaling, use_sst, ep in [
                ('map', '0', 'no', 'same'),
                ('map', 'm03', 'no', 'same'),
                ('map', 'm05', 'no', 'same'),
                ('map', 'm07', 'no', 'same'),
                ('map', 'm09', 'no', 'same'),
                ('map', 'm1', 'no', 'same'),
                ('map', 'm13', 'no', 'same'),
                ('map', '0', 'yes', 'same'),
                ('map', 'm03', 'yes', 'same'),
                ('map', 'm05', 'yes', 'same'),
                ('map', 'm07', 'yes', 'same'),
                ('map', 'm09', 'yes', 'same'),
                ('map', 'm1', 'yes', 'same'),
                ('map', 'm13', 'yes', 'same'),
        ]},
}

for xp_name, xp_defaults in XPS7.items():
    cs.store(f'{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

# Xps best swot estim with 5nad sst

# norm
cs.store(name='l1', node={'norm_obs': 'l1', 'norm_prior': 'l1'}, group='norm', package='params')
cs.store(name='l2', node={'norm_obs': 'l2', 'norm_prior': 'l2'}, group='norm', package='params')
norm = {'l1': '/norm/l1', 'l2': '/norm/l2'}

# dropout
cs.store(name='01_01', node={'dropout': 0.1, 'dropout_phi_r': 0.1}, group='dropout', package='params')
cs.store(name='025_00', node={'dropout': 0.25, 'dropout_phi_r': 0.}, group='dropout', package='params')
dropout = {'01_01': '/dropout/01_01', '025_00': '/dropout/025_00'}

XPS8 = {
        **{f'qxp8_5nad_sst_{n}_dp{dp}': [
            *BASE_DEFAULTS, losses['map'], sst['yes'], obs['5nad'], err_scaling['0'], err_loss['false'], err_prior['same'], loc_estim['false'], norm[n], dropout[dp]
        ] for n, dp in [
                ('l1', '01_01'),
                ('l1', '025_00'),
                ('l2', '01_01'),
                ('l2', '025_00'),
        ]},
}

for xp_name, xp_defaults in XPS8.items():
    cs.store(f'{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

XPS9 = {
        **{f'qxp9_errs_{loss}_es{scaling}_sst{use_sst}_ep{ep}': [
            *BASE_DEFAULTS, losses[loss], sst[use_sst], obs['errs'], err_scaling[scaling], err_loss['true'], err_prior[ep], loc_estim['false']
        ] for loss, scaling, use_sst, ep in [
                ('calmap', '0', 'no', 'same'),
                ('calmap', 'm03', 'no', 'same'),
                ('calmap', 'm05', 'no', 'same'),
                ('calmap', 'm07', 'no', 'same'),
                ('calmap', 'm09', 'no', 'same'),
                ('calmap', 'm1', 'no', 'same'),
                ('calmap', 'm13', 'no', 'same'),
                ('calmap', '0', 'yes', 'same'),
                ('calmap', 'm03', 'yes', 'same'),
                ('calmap', 'm05', 'yes', 'same'),
                ('calmap', 'm07', 'yes', 'same'),
                ('calmap', 'm09', 'yes', 'same'),
                ('calmap', 'm1', 'yes', 'same'),
                ('calmap', 'm13', 'yes', 'same'),
                ('map', '0', 'no', 'diff'),
                ('map', 'm03', 'no', 'diff'),
                ('map', 'm05', 'no', 'diff'),
                ('map', 'm07', 'no', 'diff'),
                ('map', 'm09', 'no', 'diff'),
                ('map', 'm1', 'no', 'diff'),
                ('map', 'm13', 'no', 'diff'),
                ('map', '0', 'yes', 'diff'),
                ('map', 'm03', 'yes', 'diff'),
                ('map', 'm05', 'yes', 'diff'),
                ('map', 'm07', 'yes', 'diff'),
                ('map', 'm09', 'yes', 'diff'),
                ('map', 'm1', 'yes', 'diff'),
                ('map', 'm13', 'yes', 'diff'),
        ]},
}

for xp_name, xp_defaults in XPS9.items():
    cs.store(f'{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

XPS10 = {
        **{f'qxp10_errs_{loss}_es{scaling}_sst{use_sst}_ep{ep}': [
            *BASE_DEFAULTS, losses[loss], sst[use_sst], obs['errs'], err_scaling[scaling], err_loss['true'], err_prior[ep], loc_estim['false']
        ] for loss, scaling, use_sst, ep in [
                ('map', '03', 'no', 'same'),
                ('map', '05', 'no', 'same'),
                ('map', '07', 'no', 'same'),
                ('map', '09', 'no', 'same'),
                ('map', '1', 'no', 'same'),
                ('map', '13', 'no', 'same'),
                ('map', '03', 'yes', 'same'),
                ('map', '05', 'yes', 'same'),
                ('map', '07', 'yes', 'same'),
                ('map', '09', 'yes', 'same'),
                ('map', '1', 'yes', 'same'),
                ('map', '13', 'yes', 'same'),
        ]},
}

for xp_name, xp_defaults in XPS10.items():
    cs.store(f'{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

xpn= 11
for xp_name, xp_defaults in XPS9.items():
    cs.store(f'{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

XPS = {
        **{f'qxp{xpn}_errs_{loss}_es{scaling}_sst{use_sst}_ep{ep}': [
            *BASE_DEFAULTS, losses[loss], sst[use_sst], obs['errs'], err_scaling[scaling], err_loss['true'], err_prior[ep], loc_estim['false']
        ] for loss, scaling, use_sst, ep in [
                ('map', '03', 'no', 'diff'),
                ('map', '05', 'no', 'diff'),
                ('map', '07', 'no', 'diff'),
                ('map', '09', 'no', 'diff'),
                ('map', '1', 'no', 'diff'),
                ('map', '13', 'no', 'diff'),
                ('map', '03', 'yes', 'diff'),
                ('map', '05', 'yes', 'diff'),
                ('map', '07', 'yes', 'diff'),
                ('map', '09', 'yes', 'diff'),
                ('map', '1', 'yes', 'diff'),
                ('map', '13', 'yes', 'diff'),
                ('map', '03', 'no', 'none'),
                ('map', '05', 'no', 'none'),
                ('map', '07', 'no', 'none'),
                ('map', '09', 'no', 'none'),
                ('map', '1', 'no', 'none'),
                ('map', '13', 'no', 'none'),
                ('map', '03', 'yes', 'none'),
                ('map', '05', 'yes', 'none'),
                ('map', '07', 'yes', 'none'),
                ('map', '09', 'yes', 'none'),
                ('map', '1', 'yes', 'none'),
                ('map', '13', 'yes', 'none'),
        ]},
}

for xp_name, xp_defaults in XPS.items():
    cs.store(f'{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

if __name__== '__main__':
    for xp in cs.list('xp'):
        print(xp)
