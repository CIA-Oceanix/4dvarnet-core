from omegaconf import OmegaConf
import re
from pathlib import Path
import hydra_config.quentins_xps

common_cfg = OmegaConf.create(dict(
    net_cfg=dict(
        nhidden = 128,
        depth = 3,
        kernel_size = 3,
        num_repeat = 1,
        residual = True,
        norm_type = 'lrn',
        act_type = 'silu',
        mix = True,
        mix_residual = False,
        mix_act_type = 'none',
        mix_norm_type = 'none',
    ),
    lit_cfg=dict(
        lr_init=2e-3,
        wd=1e-2,
        loss_w={
            'tot':(5., 3., 3.),
            'rec':(0., 0., 0.,)
        },
    )
))

def bst_ckpt(dirpath, glob='version_*/checkpoints/*', ckpt_fmt='.+val_loss=(.+)\.ckpt'):
    print([Path(dirpath)])
    return min(Path(dirpath).glob(glob), key=lambda p: float(re.match(ckpt_fmt, str(p)).group(1)))



cfgs = {}
for xp, cfgn in (
    list(zip(
    [20]*100,
    [
        'qxp20_5nad_no_sst',
        'qxp20_5nad_sst',
    ]))
):
    cfgs[cfgn] =  OmegaConf.create(dict(
        fourdvar_cfg=cfgn,
        fourdvar_mod_ckpt=str(bst_ckpt(f'../4dvarnet-core/results/xp{xp}/{cfgn}')),
        cal_mod_ckpt=None,
        swath_ds_cfg=dict(
            sigmas_gt=(0,),
            gt_var='gt',
            ref_var='pred',
            xb_var='pred',
        ),
    ))



size_overrides_cfg = dict(
    pp10x2=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*2 for i in range(10)]),
        sigmas_xb=(0,*[(i+1)*2 for i in range(10)]),
    ))),
    pp20x2=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*2 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*2 for i in range(20)]),
    ))),
    pp40x2=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*2 for i in range(40)]),
        sigmas_xb=(0,*[(i+1)*2 for i in range(40)]),
    ))),
    pp10x5=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*5 for i in range(10)]),
        sigmas_xb=(0,*[(i+1)*5 for i in range(10)]),
    ))),
    pp10x8=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(10)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(10)]),
    ))),
    pp20x5=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*5 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*5 for i in range(20)]),
    ))),
    pp20x8=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
    ))),
    pp40x8=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,*[(i+1)*8 for i in range(40)]),
        sigmas_xb=(0,*[(i+1)*8 for i in range(40)]),
    ))),
    ppcustom=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0, 2, 4, 8, 12, 17, 22, 28, 34, 41, 48, 56, 64, 73, 82, 92, 102, 112, 122, 132, 142),
        sigmas_xb=(0, 2, 4, 8, 12, 17, 22, 28, 34, 41, 48, 56, 64, 73, 82, 92, 102, 112, 122, 132, 142),
    ))),
    no_pp=OmegaConf.create(dict(swath_ds_cfg=dict(
        sigmas_obs=(0,),
        sigmas_xb=(0,),
    ))),
)
overrides_cfg = dict(
    direct_obs=OmegaConf.create(
        dict(swath_ds_cfg=dict(
            sigmas_xb=tuple(),
            gt_var='ssh_model',
            ref_var='pred',
            xb_var='zeros',
        ))),
    duacs_base=OmegaConf.create(
        dict(
            swath_ds_cfg=dict(
            xb_var='oi',
        ))),
    non_residual=OmegaConf.create(dict(net_cfg=dict(residual=False,))),
    no_norm=OmegaConf.create(dict(net_cfg=dict(norm_type='none',))),
    no_mix=OmegaConf.create(dict(net_cfg=dict(mix='False',))),
    relu_act=OmegaConf.create(dict(net_cfg=dict(act_type='relu',))),
)

def register_configs_tgrs():
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from pathlib import Path
    from itertools import product

    cs = ConfigStore.instance()
    xpns = []
    basic_overrides = [overrides_cfg[o] for o in ['no_norm', 'no_mix', 'relu_act']]
    for xp_name, cfg in cfgs.items():
        for train_with_ff in [False]: #True, False]:
            xpns.append(f'ff{train_with_ff}_swath_calib_{xp_name}')
            cs.store(name=xpns[-1], node=OmegaConf.merge(
                common_cfg,
                cfg,
                size_overrides_cfg['pp10x8'],
                *basic_overrides,
                {'train_with_ff': train_with_ff}
            ), group='xp', package='_global_')
    return xpns

def register_configs():
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from pathlib import Path
    from itertools import product

    cs = ConfigStore.instance()
    xpns = []
    basic_overrides = [overrides_cfg[o] for o in ['no_norm', 'no_mix', 'relu_act']]
    for xp_name, cfg in cfgs.items():
        xpns.append(f'swath_calib_{xp_name}')
        cs.store(name=xpns[-1], node=OmegaConf.merge(
            common_cfg,
            cfg,
            size_overrides_cfg['pp10x8'],
            *basic_overrides,
        ), group='xp', package='_global_')
    return xpns
