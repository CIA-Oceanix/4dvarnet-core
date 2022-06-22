from omegaconf import OmegaConf
import re
from pathlib import Path

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
    return min(Path(dirpath).glob(glob), key=lambda p: float(re.match(ckpt_fmt, str(p)).group(1)))



cfgs = {}
for xp, cfgn in (
        # list(zip( [17]*100,
    # [
        # 'qxp17_aug2_dp240_swot_map_no_sst_ng5x3cas_l2_dp025_00',
        # 'qxp17_aug2_dp240_swot_cal_no_sst_ng5x3cas_l2_dp025_00',
        # 'qxp17_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00',
        # 'qxp17_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00',
    # ])) +
    # list(zip(
    # [19]*100,
    # [
    #     'qxp19_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_swot_map_no_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_swot_cal_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_5nad_map_no_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_5nad_cal_no_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_swot_map_no_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_5nad_cal_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_5nad_cal_no_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_swot_map_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_5nad_map_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_5nad_map_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_5nad_cal_sst_ng5x3cas_l2_dp025_00_dataaug',
    #     'qxp19_aug2_dp240_swot_cal_no_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_5nad_map_no_sst_ng5x3cas_l2_dp025_00_no_dataaug',
    #     'qxp19_aug2_dp240_swot_cal_no_sst_ng5x3cas_l2_dp025_00_dataaug',
    # ])) +
    # list(zip(
    # [20]*100,
    # [
    #     'qxp20_5nad_no_sst',
    #     # 'qxp20_swot_no_sst',
    #     # 'qxp20_swot_sst',
    #     # 'qxp20_5nad_sst',
    #     # 'qxp20_noisy_sst',
    #     # 'qxp20_noisy_no_sst',
# ]))
    # + 
    list(zip(
    [21]*100,
    [
        'qxp21_5nad_no_sst_11',
        # 'qxp20_swot_no_sst',
        # 'qxp20_swot_sst',
        # 'qxp20_5nad_sst',
        # 'qxp20_noisy_sst',
        # 'qxp20_noisy_no_sst',
]))
):
    cfgs[cfgn] =  OmegaConf.create(dict(
        fourdvar_cfg=cfgn,
        fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp{xp}/{cfgn}')),
        cal_mod_ckpt=None,
        swath_ds_cfg=dict(
            sigmas_gt=(0,),
            gt_var='gt',
            ref_var='pred',
            # gt_var='gt_res',
            # ref_var='ref_res',
            xb_var='pred',
        ),
    ))


# oi_res_cfg = OmegaConf.create(dict(
#     fourdvar_cfg='qxp2_aug2_dp240_5nad_map_sst_ng5x3cas_w1.0',
#     pred_var='pred',
#     fourdvar_mod_ckpt=str(bst_ckpt('results/xpfeb_train/newaug_5nad_1.0_map_sst')),
#     cal_mod_ckpt=str(next(Path(f'lightning_logs/baseline_duacs4nad_res/checkpoints').glob('epoch*.ckpt'), None)),
#     swath_ds_cfg=dict(
#         sigmas_obs=(0,*[(i+1)*8 for i in range(20)]),
#         sigmas_xb=(0,*[(i+1)*8 for i in range(20)]),
#         sigmas_gt=(0,),
#         gt_var='gt_res',
#         ref_var='ref_res',
#         xb_var='oi',
#     ),
# ))

# xp_no_sst = 'qxp12_aug1_dp240_5nad_map_no_sst_ng5x3cas_l1_dp025_00'
# no_sst1_cfg = OmegaConf.create(dict(
#     fourdvar_cfg=xp_no_sst,
#     pred_var=xp_no_sst,
#     fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp12/{xp_no_sst}')),
#     cal_mod_ckpt=str(next(Path(f'lightning_logs/1_{xp_no_sst}/version_0/checkpoints').glob('epoch*.ckpt'), None)),
#     swath_ds_cfg=dict(
#         sigmas_gt=(0,),
#         gt_var='gt_res',
#         ref_var='ref_res',
#         xb_var='pred',
#     ),
# ))

# xp_sst = 'qxp12_aug1_dp240_5nad_map_sst_ng5x3cas_l1_dp025_00'
# sst1_cfg = OmegaConf.create(dict(
#     fourdvar_cfg=xp_sst,
#     pred_var=xp_sst,
#     fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp12/{xp_sst}')),
#     cal_mod_ckpt=str(next(Path(f'lightning_logs/1_{xp_sst}/version_0/checkpoints').glob('epoch*.ckpt'), None)),
#     swath_ds_cfg=dict(
#         sigmas_gt=(0,),
#         gt_var='gt_res',
#         ref_var='ref_res',
#         xb_var='pred',
#     ),
# ))

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
