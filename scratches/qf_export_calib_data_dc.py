import utils
import hydra_config
import re
import pickle
from pathlib import Path
import numpy as np
import hashlib
import traceback

import swath_calib.configs
import swath_calib.utils
import swath_calib.models
import swath_calib.dataset
import swath_calib.versioning_cb
import swath_calib.report

from omegaconf import OmegaConf

base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    'file_paths={fp}'
]

def bst_ckpt(dirpath, glob='version_*/checkpoints/*', ckpt_fmt='.+val_loss=(.+)\.ckpt'):
    return min(Path(dirpath).glob(glob), key=lambda p: float(re.match(ckpt_fmt, str(p)).group(1)))

cfg = OmegaConf.create(dict(
    net_cfg=dict(
        nhidden = 128,
        depth = 3,
        kernel_size = (3, 3),
        num_repeat = 1,
        residual = True,
        norm_type = 'none',
        act_type = 'relu',
        mix = True,
        mix_residual = False,
        mix_act_type = 'none',
        mix_norm_type = 'none',
        apply_per_side = False,
    ),
    lit_cfg=dict(
        lr_init=3e-3,
        wd=1e-3,
        loss_w={
            'tot':(5., 1., 1.),
            'rec':(0., 0., 0.,)
        },
    ),
    fourdvar_cfg='qxp20_5nad_no_sst',
    fourdvar_mod_ckpt=str(bst_ckpt(f'results/xp20/qxp20_5nad_no_sst')),
    cal_mod_ckpt=None,
    swath_ds_cfg=dict(
        sigmas_gt=(0,),
        sigmas_obs=(0,*[(i+1)*4 for i in range(20)]),
        sigmas_xb=(0,*[(i+1)*4 for i in range(20)]),
        gt_var='gt',
        ref_var='pred',
        # gt_var='gt_res',
        # ref_var='ref_res',
        xb_var='pred',
    ),
))

from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
xpns = []
for xp, override in [
    ('base', dict()),
]:
    name = f'tgrs_{xp}'
    xpns.append(name)
    cs.store(name=xpns[-1], node=OmegaConf.merge(
        cfg,
        override, 
    ), group='xp', package='_global_')
    print(OmegaConf.merge(
                cfg,
                override, 
    ).swath_ds_cfg.sigmas_obs)

# vxp = 7
# results = []
# for xpn in xpns:
#     trained_cfg_w_metrics =None
#     print(xpn)
#     locals().update(full_from_scratch(vxp, xpn))
#     results.append(trained_cfg_w_metrics)

def run1():
    try:
        cfgn = 'tgrs_base'
        cfg = utils.get_cfg(f'{cfgn}')

        overrides = [
            '+datamodule.dl_kwargs.shuffle=False',
            f'file_paths={fp}',
            'params.files_cfg.obs_mask_path=${file_paths.new_noisy_swot}',
            'params.files_cfg.obs_mask_var=five_nadirs',
            'datamodule.aug_train_data=false',
        ]
       

        # Generate grided product on swath
        cfg_4dvar = utils.get_cfg(cfg.fourdvar_cfg, overrides=overrides)

        cfg_hash = hashlib.md5(OmegaConf.to_yaml(cfg_4dvar).encode()).hexdigest()
        saved_data_path = Path('tmp') / '231e96a055a4e6017fa0ea19a13f66fb.pk'
        print(saved_data_path)
        rms = lambda da: np.sqrt(np.mean(da**2))
        if saved_data_path.exists():
            print('loading  ', str(saved_data_path))
            with open(saved_data_path, 'rb') as f:
                swath_data = pickle.load(f)


        for stage in ['test', 'val', 'train']:
            print(
            f''' \n\n ############################# {cfgn}  {stage} ##########################################
            {rms(swath_data[stage].ssh_model - swath_data[stage].oi).item()=:.2e} m
            {rms(swath_data[stage].ssh_model - swath_data[stage].pred).item()=:.2e} m
            {rms(swath_data[stage].ssh_model - swath_data[stage].gt).item()=:.2e} m
            ''')


        swath_data = swath_data['test']
        swath_data = swath_data.assign(contiguous_chunk=lambda _df: (_df.x_al.diff('time').pipe(np.abs) > 3).cumsum())
        min_timestep = 500
        chunks = list(
                swath_data.groupby('contiguous_chunk').count()
                .isel(nC=0).pipe(lambda ds: ds.isel(contiguous_chunk=ds[list(ds)[0]] > min_timestep))
                .contiguous_chunk.values
        )
        len(chunks)

        evaluation_data = (
            swath_data.isel(time=swath_data.contiguous_chunk.isin(chunks))
            .assign(
                swot_with_error=lambda ds: ds.ssh_model + ds.syst_error_uncalibrated + ds.wet_tropo_res,
                swot_without_error=lambda ds: ds.ssh_model,
            )[['lat',
             'lat_nadir',
             'lon',
             'lon_nadir',
             'x_ac',
             'x_al',
             'swot_without_error',
             'swot_with_error',
            'contiguous_chunk',
           ]]
        )

        add_inter_sw = lambda ds:(
                    ds
                .assign_coords(x_ac=lambda ds: ('nC', ds.x_ac.isel(time=0).data))
                .swap_dims(nC='x_ac')
                .reindex(x_ac=np.arange(-60, 62, 2), fill_value=np.nan)
        )
        evaluation_data.isel(time=evaluation_data.contiguous_chunk==chunks[0]).pipe(add_inter_sw).swot_with_error.T.plot(figsize=(15,3))
        evaluation_data.isel(time=evaluation_data.contiguous_chunk==chunks[0]).pipe(add_inter_sw).swot_without_error.T.plot(figsize=(15,3))

        evaluation_data.to_netcdf('eval_swot_corr_err_calib.nc')

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def main():
    try:
        fn = run1

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    main()
