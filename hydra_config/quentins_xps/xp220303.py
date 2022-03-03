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
cs.store(name='0', node={'train_error_scaling': False, 'init_err_scaling':0.}, group='err_scaling', package='params')
cs.store(name='m05', node={'train_error_scaling': False, 'init_err_scaling':-0.5}, group='err_scaling', package='params')
cs.store(name='m1', node={'train_error_scaling': False, 'init_err_scaling':-1.}, group='err_scaling', package='params')
err_scaling = {'0': '/err_scaling/0', 'm05': '/err_scaling/m05', 'm1': '/err_scaling/m1'}

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
if __name__== '__main__':
    for xp in cs.list('xp'):
        print(xp)
