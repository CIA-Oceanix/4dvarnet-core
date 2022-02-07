import hydra
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
- [ ] implement new datamodule with noise filter
- [ ] implement spectral filter
"""
display(Markdown(s))

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
    '5nad': {'ckpt': bst_ckpt('xpxmas_train/5nad_map_no_sst'), 'overrides': [or_fivenad, or_map, or_no_sst]},
    '5nad_cal': {'ckpt': bst_ckpt('xpxmas_train/5nad_cal_no_sst_inc_ngrad'), 'overrides': [or_fivenad, or_cal, or_no_sst, or_inc_ngrad]},
    '5nad_sst': {'ckpt': bst_ckpt('xpxmas_train/5nad_map_sst_inc_ngrad'), 'overrides': [or_fivenad, or_map, or_sst]},
    'no_err': {'ckpt': bst_ckpt('xpxmas_train/no_err_map_no_sst'), 'overrides': [or_errs, or_map, or_no_sst]},
    'no_err_cal': {'ckpt': bst_ckpt('xpxmas_train/no_err_cal_no_sst_inc_ngrad'), 'overrides': [or_errs, or_cal, or_no_sst, or_inc_ngrad]},
    'no_err_sst': {'ckpt': bst_ckpt('xpxmas_train/no_err_map_sst_inc_ngrad'), 'overrides': [or_errs, or_map, or_sst, or_inc_ngrad]},
    'old_errs': {'ckpt': bst_ckpt('xpxmas_train/old_errs_map_no_sst'), 'overrides': [or_old_errs, or_map, or_no_sst]},
    'old_errs_cal': {'ckpt': bst_ckpt('xpxmas_train/old_errs_cal_no_sst'), 'overrides': [or_old_errs, or_cal, or_no_sst]},
    'old_errs_sst': {'ckpt': bst_ckpt('xpxmas_train/old_errs_map_sst_dec_lr'), 'overrides': [or_old_errs, or_map, or_sst, or_dec_lr]},
    'errs': {'ckpt': bst_ckpt('xpxmas_train/errs_map_no_sst_inc_ngrad'), 'overrides': [or_errs, or_map, or_no_sst, or_inc_ngrad]},
    'errs_cal': {'ckpt': bst_ckpt('xpxmas_train/errs_cal_no_sst'), 'overrides': [or_errs, or_cal, or_no_sst]},
    'errs_sst': {'ckpt': bst_ckpt('xpxmas_train/errs_map_sst_inc_ngrad'), 'overrides': [or_errs, or_map, or_sst, or_inc_ngrad]},
}

dms = {
    '5nad': [or_no_sst, or_fivenad],
    '5nad_sst': [or_sst, or_fivenad],
    'no_err': [or_no_sst, or_no_err],
    'no_err_sst': [or_sst, or_no_err],
    'old_err': [or_no_sst, or_old_errs],
    'old_err_sst': [or_sst, or_old_errs],
    'errs': [or_no_sst, or_errs],
    'errs_sst': [or_sst, or_errs],
}

def get_cfg(overrides):
    print(overrides)
    with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp=xmasxp/xp_xmas/base',
                'file_paths=dgx_ifremer',
                'entrypoint=train',
            ] + overrides
        )

    return cfg

def get_model(model_name, dm=None, add_overrides=None):
    overrides = models[model_name]['overrides']
    print(overrides)
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    print(overrides)
    ckpt = models[model_name]['ckpt']
    cfg = get_cfg(overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(dm_name, setup=True, add_overrides=None):
    overrides = dms[dm_name]
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(overrides)
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

        prepro_cfg={
                '_target_': 'calibration.dataset.weight_error',
                'w': 0.,
        }
        or_str = str(prepro_cfg).replace(" ", "").replace("'","")
        dm = get_dm('errs', add_overrides=[f'+datamodule.item_prepro={or_str}'])
        batch = next(iter(dm.val_dataloader()))
        obs = batch[2]
        im = plt.imshow(obs[1,3,...])
        plt.colorbar(im)

    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

def test_models_on_diff_noises():

        metrics = []
        for model_name in [
                '5nad',
                '5nad_cal',
                'errs',
                'errs_cal',
                'old_errs',
                'old_errs_cal',
                'no_err',
                'no_err_cal',
                ]:
            for dm_name in ['no_err', '5nad', 'old_err', 'errs']:
                dm = get_dm(dm_name, setup=False)
                mod = get_model(model_name, dm)
                trainer = pl.Trainer(gpus=[2])
                trainer.test(mod, dataloaders=dm.val_dataloader())
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

        metrics_sst = []
        for model_name in [
                '5nad_sst',
                'old_errs_sst',
                'errs_sst',
                'no_err_sst',
                ]:
            for dm_name in ['no_err_sst', '5nad_sst', 'old_err_sst', 'errs_sst']:
                dm = get_dm(dm_name, setup=False)
                mod = get_model(model_name, dm)
                trainer = pl.Trainer(gpus=[2])
                trainer.test(mod, dataloaders=dm.val_dataloader())
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

         
        dm = get_dm('5nad', setup=False)
        mod = get_model('5nad', dm)
        mod_ngrad = get_model('5nad', dm)
        mod_ngrad.model.n_grad = 30
        mod_ngrad.hparams.n_fourdvar_iter = 2
        trainer = pl.Trainer(gpus=[2])
        trainer.test(mod, dataloaders=dm.val_dataloader())
        trainer.test(mod_ngrad, dataloaders=dm.val_dataloader())
        mod.latest_metrics

if False:
    df_sst = s['df_sst']
    df_sst.to_csv('sst_mods_metrics.csv')
    metrics_sst = s['metrics_sst']
    pd.options.plotting.backend = 'holoviews'
    from hvplot import parallel_coordinates
    parallel_coordinates(df_sst, class_column='model')

    df = s['df']
    df.to_csv('no_sst_metrics.csv')
    metrics = s['metrics']
    pd.options.plotting.backend = 'holoviews'
    from hvplot import parallel_coordinates
    parallel_coordinates(df_sst, class_column='model')
