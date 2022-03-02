from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()


BASE_DEFAULTS = [
    '/xp/qfebvre/base_xp220301',
    '/splits/dc_boost_swot@datamodule',
    '/xp/qfebvre/dl/d240_p240x5_s240x1.yaml',
    '/xp/qfebvre/ds/new_noisy_swot.yaml',
    '/xp/qfebvre/train_strat/const_lr_ngrad_5_3cas.yaml',
]

# yaml defaults
losses = { 'calmap': '/xp/qfebvre/loss/calmap.yaml', 'map': '/xp/qfebvre/loss/map.yaml', }
sst = { 'yes': '/xp/qfebvre/sst/with_sst.yaml', 'no': '/xp/qfebvre/sst/without_sst.yaml', }
obs = { '5nad': '/xp/qfebvre/ds/five_nadirs.yaml', 'errs': '/xp/qfebvre/ds/new_noisy_swot.yaml', }
loc_estim = {'false': '/xp/qfebvre/aug/expl_err_no_loc', 'true': '/xp/qfebvre/aug/expl_err_loc'}

# xp defaults
# err scaling
cs.store(name='false', node={'train_error_scaling': False}, group='err_scaling', package='params')
cs.store(name='true', node={'train_error_scaling': True}, group='err_scaling', package='params')
err_scaling = {'true': '/err_scaling/true', 'false': '/err_scaling/false'}

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



XPS = {
        '5nad_base': [ *BASE_DEFAULTS, losses['map'], sst['no'], obs['5nad'], err_scaling['false'], err_loss['false'], err_prior['none'], loc_estim['false']],
        'errs_base': [ *BASE_DEFAULTS, losses['map'], sst['no'], obs['errs'], err_scaling['false'], err_loss['false'], err_prior['none'], loc_estim['false']],
        **{f'errs_{loss}_es{scaling}_el{err_l}_ep{err_p}_loc{loc}': [
            *BASE_DEFAULTS, losses[loss], sst['no'], obs['errs'], err_scaling[scaling], err_loss[err_l], err_prior[err_p], loc_estim[loc]
        ] for loss, scaling, err_l, err_p, loc in [
                ('map', 'false', 'true', 'none', 'false'),
                ('map', 'true', 'true', 'none', 'false'),
                ('map', 'true', 'false', 'none', 'false'),
                ('map', 'true', 'true', 'same', 'false'),
                ('map', 'true', 'true', 'diff', 'false'),
                ('calmap', 'true', 'true', 'same', 'true'),
        ]},
        '5nad_base_sst': [ *BASE_DEFAULTS, losses['map'], sst['yes'], obs['5nad'], err_scaling['false'], err_loss['false'], err_prior['none'], loc_estim['false']],
        'errs_base_sst': [ *BASE_DEFAULTS, losses['map'], sst['yes'], obs['errs'], err_scaling['false'], err_loss['false'], err_prior['none'], loc_estim['false']],
        **{f'sst_errs_{loss}_es{scaling}_el{err_l}_ep{err_p}_loc{loc}': [
            *BASE_DEFAULTS, losses[loss], sst['yes'], obs['errs'], err_scaling[scaling], err_loss[err_l], err_prior[err_p], loc_estim[loc]
        ] for loss, scaling, err_l, err_p, loc in [
                ('map', 'false', 'true', 'none', 'false'),
                ('map', 'true', 'true', 'none', 'false'),
                ('map', 'true', 'false', 'none', 'false'),
                ('map', 'true', 'true', 'same', 'false'),
                ('map', 'true', 'true', 'diff', 'false'),
                ('calmap', 'true', 'true', 'same', 'true'),
        ]},
}

for xp_name, xp_defaults in XPS.items():
    cs.store(f'qxp4_{xp_name}', node={'xp_name': xp_name, 'defaults': xp_defaults + ['_self_']}, package='_global_', group='xp')

if __name__== '__main__':
    for xp in cs.list('xp'):
        print(xp)
