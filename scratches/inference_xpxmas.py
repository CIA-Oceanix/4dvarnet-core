import hydra
import torch
import matplotlib.pyplot as plt
import re
from hydra.utils import instantiate, get_class, call
from hydra_main import FourDVarNetHydraRunner
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import traceback
from IPython.display import display, Markdown, Latex, HTML

s = """
## TODO
Dev
- [x] load any model
- [x] load any datamodule
- [x] training with err testing without err 
- [x] training with inf diff from training (3 blocks vs 15 grad) -> pas de diff
- [x] implement new datamodule with noise filter
"""
display(Markdown(s))

base_xmas = 'xp=xmasxp/xp_xmas/base'
base_feb_sst = 'xp=xmasxp/xp_feb/base_sst'
base_feb_no_sst = 'xp=xmasxp/xp_feb/base_no_sst'
base_feb_cal = 'xp=xmasxp/xp_feb/base_cal'

or_cal = f'+xp/xmasxp/xp_xmas/cal=cal'
or_map = f'+xp/xmasxp/xp_xmas/cal=map'

or_fivenad = f'+xp/xmasxp/xp_xmas/obs=5nad'
or_old_errs = f'+xp/xmasxp/xp_xmas/obs=old_errs'
or_no_err = f'+xp/xmasxp/xp_xmas/obs=no_err'
or_errs = f'+xp/xmasxp/xp_xmas/obs=errs'

or_no_sst = f'+xp/xmasxp/xp_xmas/sst=no_sst'
or_sst = f'+xp/xmasxp/xp_xmas/sst=sst'

or_inc_ngrad = f'++xp/xmasxp/xp_xmas/train_strat=dec_lr_ngrad5_15_nocas'
or_dec_lr = f'++xp/xmasxp/xp_xmas/train_strat=dec_lr_ngrad_5_3cas'

def bst_ckpt(dirpath, glob='*'):
    return min(Path(dirpath).glob('version_*/checkpoints/*'), key=lambda p: float(re.match('.+val_loss=(.+)\.ckpt', str(p)).group(1)))

models = {
    '5nad': {'ckpt': bst_ckpt('xpxmas_train/5nad_map_no_sst'), 'base': base_xmas, 'overrides': [or_fivenad, or_map, or_no_sst]},
    '5nad_cal': {'ckpt': bst_ckpt('xpxmas_train/5nad_cal_no_sst_inc_ngrad'), 'base': base_xmas, 'overrides': [or_fivenad, or_cal, or_no_sst, or_inc_ngrad]},
    '5nad_sst': {'ckpt': bst_ckpt('xpxmas_train/5nad_map_sst_inc_ngrad'), 'base': base_xmas, 'overrides': [or_fivenad, or_map, or_sst]},
    'no_err': {'ckpt': bst_ckpt('xpxmas_train/no_err_map_no_sst'), 'base': base_xmas, 'overrides': [or_errs, or_map, or_no_sst]},
    'no_err_cal': {'ckpt': bst_ckpt('xpxmas_train/no_err_cal_no_sst_inc_ngrad'), 'base': base_xmas, 'overrides': [or_errs, or_cal, or_no_sst, or_inc_ngrad]},
    'no_err_sst': {'ckpt': bst_ckpt('xpxmas_train/no_err_map_sst_inc_ngrad'), 'base': base_xmas, 'overrides': [or_errs, or_map, or_sst, or_inc_ngrad]},
    'old_errs': {'ckpt': bst_ckpt('xpxmas_train/old_errs_map_no_sst'), 'base': base_xmas, 'overrides': [or_old_errs, or_map, or_no_sst]},
    'old_errs_cal': {'ckpt': bst_ckpt('xpxmas_train/old_errs_cal_no_sst'), 'base': base_xmas, 'overrides': [or_old_errs, or_cal, or_no_sst]},
    'old_errs_sst': {'ckpt': bst_ckpt('xpxmas_train/old_errs_map_sst_dec_lr'), 'base': base_xmas, 'overrides': [or_old_errs, or_map, or_sst, or_dec_lr]},
    'errs': {'ckpt': bst_ckpt('xpxmas_train/errs_map_no_sst_inc_ngrad'), 'base': base_xmas, 'overrides': [or_errs, or_map, or_no_sst, or_inc_ngrad]},
    'errs_sst': {'ckpt': bst_ckpt('xpxmas_train/errs_map_sst_inc_ngrad'), 'base': base_xmas, 'overrides': [or_errs, or_map, or_sst, or_inc_ngrad]},
    'errs_0.001_cal_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.001_cal_no_sst'), 'base': base_feb_cal, 'overrides': ['datamodule.item_prepro.w=0.001']},
    'errs_0.001_map_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.001_map_no_sst'), 'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.001']},
    'errs_0.001_map_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.001_map_sst'), 'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.001']},
    'errs_0.01_cal_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.01_cal_no_sst'), 'base': base_feb_cal, 'overrides': ['datamodule.item_prepro.w=0.01']},
    'errs_0.01_map_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.01_map_no_sst'), 'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.01']},
    'errs_0.01_map_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.01_map_sst'), 'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.01']},
    'errs_0.1_cal_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.1_cal_no_sst'), 'base': base_feb_cal, 'overrides': ['datamodule.item_prepro.w=0.1']},
    'errs_0.1_map_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.1_map_no_sst'), 'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.1']},
    'errs_0.1_map_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.1_map_sst'), 'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.1']},
    'errs_0.33_cal_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.33_cal_no_sst'), 'base': base_feb_cal, 'overrides': ['datamodule.item_prepro.w=0.33']},
    'errs_0.33_map_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.33_map_no_sst'), 'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.33']},
    'errs_0.33_map_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.33_map_sst'), 'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.33']},
    'errs_0.66_cal_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.66_cal_no_sst'), 'base': base_feb_cal, 'overrides': ['datamodule.item_prepro.w=0.66']},
    'errs_0.66_map_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.66_map_no_sst'), 'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.66']},
    'errs_0.66_map_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_0.66_map_sst'), 'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.66']},
    'errs_1.0_cal_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_1.0_cal_no_sst'), 'base': base_feb_cal, 'overrides': ['datamodule.item_prepro.w=1.0']},
    'errs_1.0_map_no_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_1.0_map_no_sst'), 'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=1.0']},
    'errs_1.0_map_sst': {'ckpt': bst_ckpt('xpfeb_train/errs_1.0_map_sst'), 'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=1.0']},

}

dms = {
    '5nad': {'overrides': [or_no_sst, or_fivenad], 'base': base_xmas},
    '5nad_sst': {'overrides': [or_sst, or_fivenad], 'base': base_xmas},
    'no_err': {'overrides': [or_no_sst, or_no_err], 'base': base_xmas},
    'no_err_sst': {'overrides': [or_sst, or_no_err], 'base': base_xmas},
    'old_err': {'overrides': [or_no_sst, or_old_errs], 'base': base_xmas},
    'old_err_sst': {'overrides': [or_sst, or_old_errs], 'base': base_xmas},
    'errs': {'overrides': [or_no_sst, or_errs], 'base': base_xmas},
    'errs_sst': {'overrides': [or_sst, or_errs], 'base': base_xmas},
    'errs_0.0': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.0']},
    'errs_0.0_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.0']},
    'errs_0.001': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.001']},
    'errs_0.001_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.001']},
    'errs_0.01': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.01']},
    'errs_0.01_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.01']},
    'errs_0.1': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.1']},
    'errs_0.1_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.1']},
    'errs_0.33': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.33']},
    'errs_0.33_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.33']},
    'errs_0.66': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=0.66']},
    'errs_0.66_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=0.66']},
    'errs_1.0': {'base': base_feb_no_sst, 'overrides': ['datamodule.item_prepro.w=1.0']},
    'errs_1.0_sst': {'base': base_feb_sst, 'overrides': ['datamodule.item_prepro.w=1.0']},
}

def get_cfg(base, overrides):
    print(overrides)
    with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
        cfg = hydra.compose(config_name='main', overrides=
            [
                base,
                'file_paths=dgx_ifremer',
                'entrypoint=train',
            ] + overrides
        )

    return cfg

def get_model(model_name, dm=None, add_overrides=None):
    overrides = models[model_name]['overrides']
    base = models[model_name]['base']
    print(overrides)
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    print(overrides)
    ckpt = models[model_name]['ckpt']
    cfg = get_cfg(base, overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(dm_name, setup=True, add_overrides=None):
    overrides = dms[dm_name]['overrides']
    base = dms[dm_name]['base']
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(base, overrides)
    dm = instantiate(cfg.datamodule)
    if setup:
        dm.setup()
    return dm

# for mn in models:
#     print(mn)
#     get_model(mn)

# for dmn in dms:
#     print(dmn)
#     get_dm(dmn)

import importlib
import calibration.dataset
import hydra_main
importlib.reload(calibration.dataset)
importlib.reload(hydra_main)
import json

def main():
    try:
        # fn = test_models_on_diff_noises
        # fn = data_stats
        fn = plot_results

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def foo():

        prepro_cfg={
                '_target_': 'calibration.dataset.weight_error',
                'w': 0.1,
        }
        or_str = str(prepro_cfg).replace(" ", "").replace("'","")
        dm = get_dm('errs', add_overrides=[f'+datamodule.item_prepro={or_str}'])
        batch = next(iter(dm.val_dataloader()))
        obs = batch[2]
        im = plt.imshow(obs[1,3,...])
        plt.colorbar(im)

def plot_results():
    noise_level_df = pd.read_csv('noise_levels.csv')
    print(noise_level_df.to_markdown())
    df = pd.read_csv('sst_mods_metrics3.csv')
    # df = pd.read_csv('no_sst_metrics3.csv').loc[lambda _df: ~_df.model.str.contains('cal') | _df.model.str.contains('5nad')]
    

    metrics = [
        'test_spatial_res_imp',
        'test_nrmse_ratio_glob',
        'test_mse_ratio_glob',
        'test_nrmse_swath',
        'test_mse_swath'
    ]
    for m in metrics:
        (
                df[['dm', 'model', m]]
                .pivot(columns='model', values=m, index='dm')
                .rename({'errs_0.0_sst': 'errs_0.000_sst', 'errs_0.0': 'errs_0.000'})
                .sort_index()
                .plot(title=m, figsize=(10, 5), logy=True)
        )


    return locals()

def data_stats():
    noise_level = []
    for dm_name in [
        'errs_0.0',
        'errs_0.001',
        'errs_0.01',
        'errs_0.1',
        'errs_0.33',
        'errs_0.66',
        'errs_1.0',
        '5nad',
    ]:
        dm = get_dm(dm_name)
        dm.setup()
        it = iter(dm.val_dataloader())
        next(it)
        oi, mask, obs, gt, obs_gt  = next(it)
        nanobs = obs.where(mask, torch.full_like(obs, float('nan'))) 
        print(dm_name, torch.nanmean((nanobs - obs_gt)**2).sqrt())
        noise_level.append({'dm': dm_name, 'rmse': torch.nanmean((nanobs - obs_gt)**2).sqrt().item()})
        # img = plt.imshow((nanobs-obs_gt)[1,3,...])
        # plt.colorbar(img)
        # plt.show()

        # img = plt.imshow(obs.where(mask, torch.full_like(obs,float('nan')))[1,3,...])
        # plt.colorbar(img)
        # plt.show()

    noise_level_df = pd.DataFrame(noise_level)
    noise_level_df.to_csv('noise_levels.csv')
    return locals()
                
def test_models_on_diff_noises():

        metrics_sst = []
        for model_name in [
                'errs_0.001_map_sst',
                'errs_0.01_map_sst',
                'errs_0.1_map_sst',
                'errs_0.33_map_sst',
                'errs_0.66_map_sst',
                'errs_1.0_map_sst',
                '5nad_sst',
                'no_err_sst',
                ]:
            for dm_name in [
                'errs_0.0_sst',
                'errs_0.001_sst',
                'errs_0.01_sst',
                'errs_0.1_sst',
                'errs_0.33_sst',
                'errs_0.66_sst',
                'errs_1.0_sst',
                '5nad_sst'
                ]:
                dm = get_dm(dm_name, setup=False)
                mod = get_model(model_name, dm)
                trainer = pl.Trainer(gpus=[3])
                trainer.test(mod, dataloaders=dm.test_dataloader())
                metrics_sst.append(
                    {
                        'model': model_name,
                        'dm': dm_name,
                        'metrics': mod.latest_metrics,
                    }
                )

        df_sst = pd.DataFrame(
            metrics_sst
        ).pipe(lambda _df:_df[['model', 'dm']].assign(**pd.DataFrame(list(_df.metrics)).to_dict(orient='list')))
        df_sst.to_csv('sst_mods_metrics3.csv')

        metrics = []
        for model_name in [
                'errs_0.001_cal_no_sst',
                'errs_0.001_map_no_sst',
                'errs_0.01_cal_no_sst',
                'errs_0.01_map_no_sst',
                'errs_0.1_cal_no_sst',
                'errs_0.1_map_no_sst',
                'errs_0.33_cal_no_sst',
                'errs_0.33_map_no_sst',
                'errs_0.66_cal_no_sst',
                'errs_0.66_map_no_sst',
                'errs_1.0_cal_no_sst',
                'errs_1.0_map_no_sst',
                '5nad',
                '5nad_cal',
                'no_err',
                'no_err_cal',
                ]:
            for dm_name in [
                'errs_0.0',
                'errs_0.001',
                'errs_0.01',
                'errs_0.1',
                'errs_0.33',
                'errs_0.66',
                'errs_1.0',
                '5nad'
                 ]:
                dm = get_dm(dm_name, setup=False)
                mod = get_model(model_name, dm)
                trainer = pl.Trainer(gpus=[3])
                trainer.test(mod, dataloaders=dm.test_dataloader())
                metrics.append(
                    {
                        'model': model_name,
                        'dm': dm_name,
                        'metrics': mod.latest_metrics,
                    }
                )

        df = pd.DataFrame(
            metrics
        ).pipe(lambda _df:_df[['model', 'dm']].assign(**pd.DataFrame(list(_df.metrics)).to_dict(orient='list')))

        df.to_csv('no_sst_metrics3.csv')

         

        return locals()
