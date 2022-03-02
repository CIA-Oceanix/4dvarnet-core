from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from itertools import product

cs = ConfigStore.instance()

"""
Test explicit error term in state
- with / without sst
- with / without err_scaling learning
- with / without err_loss
- with / without local_estimation
- err_prior none, same, diff


xps:
- baseline 5nad:  
    -- map
    -- 5nad
    -- no scaling err
    -- no loc_estim
    -- none err prior
    -- no err loss

- baseline noisy_swot:  
    -- map
    -- noisy_swot /!\
    -- no err loss
    -- no scaling err
    -- no loc_estim
    -- none err prior

- noisy_swot  error loss:  
    -- map
    -- noisy_swot
    -- err loss /!\
    -- no scaling err
    -- no loc_estim
    -- none err prior

- noisy_swot  scaling error:  
    -- map
    -- noisy_swot
    -- no err loss
    -- scaling err /!\
    -- no loc_estim
    -- none err prior

- noisy_swot same_prior error :  
    -- map
    -- noisy_swot
    -- no err loss 
    -- no scaling err
    -- no loc_estim
    -- same err prior/!\

- noisy_swot diff prior error :  
    -- map
    -- noisy_swot
    -- no err loss 
    -- no scaling err
    -- no loc_estim
    -- same err prior/!\

- noisy_swot local_estim :  
    -- calmap
    -- noisy_swot
    -- no err loss 
    -- no scaling err
    -- loc_estim /!\
    -- same err prior

- noisy_swot all no local_estim :  
    -- calmap
    -- noisy_swot
    -- err loss 
    -- scaling err
    -- no loc_estim /!\
    -- diff err prior
"""

BASE_DEFAULTS = [
    '/xp/qfebvre/base_xp',
    '/splits/dc_boost_swot@datamodule',
    '/xp/qfebvre/dl/d240_p240x5_s240x1.yaml',
    '/xp/qfebvre/ds/new_noisy_swot.yaml',
    '/xp/qfebvre/train_strat/const_lr_ngrad_5_3cas.yaml',
]

# yaml defaults
losses = {
        'calmap': '/xp/qfebvre/loss/calmap.yaml',
         'map': '/xp/qfebvre/loss/map.yaml',
}
sst = {
        'yes': '/xp/qfebvre/sst/with_sst.yaml',
        'no': '/xp/qfebvre/sst/without_sst.yaml',
}
obs = {
        '5nad': '/xp/qfebvre/ds/five_nadirs.yaml',
        'errs': '/xp/qfebvre/ds/new_noisy_swot.yaml',
}

# xp defaults

# err scaling
cs.store(name='false', node={'train_err_scaling': False}, group='err_scaling', package='params')
cs.store(name='true', node={'train_err_scaling': True}, group='err_scaling', package='params')
err_scaling = {'true': 'err_scaling=true', 'false': 'err_scaling=false'}

# err loss
cs.store(name='false', node={'loss_err': False}, group='loss_err', package='params')
cs.store(name='true', node={'loss_err': True}, group='loss_err', package='params')
err_loss = {'true': 'loss_err=true', 'false': 'loss_err=false'}

# err prior
cs.store(name='same', node={'err_prior': 'same'}, group='err_prior', package='params')
cs.store(name='none', node={'err_prior': 'none'}, group='err_prior', package='params')
cs.store(name='diff', node={'err_prior': 'diff'}, group='err_prior', package='params')
err_prior = {'same': 'err_prior=same', 'none': 'err_prior=none', 'diff': 'err_prior=diff'}

# loc estim
cs.store(name='false', node={'loc_estim': False}, group='loc_estim', package='params')
cs.store(name='true', node={'loc_estim': True}, group='loc_estim', package='params')
loc_estim = {'false': 'loc_estim=false', 'true': 'loc_estim=true'}



XPS = {
        '5nad_base': [ *BASE_DEFAULTS, losses['map'], sst['no'], obs['5nad'], err_scaling['false'], err_loss['false'], err_prior['none'], loc_estim['false']],
        'errs_base': [ *BASE_DEFAULTS, losses['map'], sst['no'], obs['errs'], err_scaling['false'], err_loss['false'], err_prior['none'], loc_estim['false']],
        **{f'errs_{loss=}_{scaling=}_{err_l=}_{err_p=}_{loc=}': [
            *BASE_DEFAULTS, losses[loss], sst['no'], obs['errs'], err_scaling[scaling], err_loss[err_l], err_prior[err_p], loc_estim[loc]
        ] for loss, scaling, err_l, err_p, loc in [
                ('map', 'false', 'true', 'none', 'false'),
                ('map', 'true', 'true', 'none', 'false'),
                ('map', 'true', 'false', 'none', 'false'),
                ('map', 'true', 'true', 'none', 'false'),
        ]},
}
