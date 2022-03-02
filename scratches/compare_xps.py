import hydra
import torch
import re
import matplotlib.pyplot as plt
import re
from hydra.utils import instantiate, get_class, call
from hydra.core.config_store import ConfigStore
from hydra_main import FourDVarNetHydraRunner
import pytorch_lightning as pl
import pandas as pd
from pathlib import Path
import traceback
import hydra_config
from IPython.display import display, Markdown, Latex, HTML

s = """
## TODO
Dev
- [ ]  list xps
- [ ] from xp get dir and ckpt
"""
display(Markdown(s))


def main():
    try:
        fn = test_models_on_diff_noises
        # fn = data_stats
        # fn = plot_results

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

cs = ConfigStore.instance()


def bst_ckpt(dirpath, glob='*'):
    return min(Path(dirpath).glob('version_*/checkpoints/*'), key=lambda p: float(re.match('.+val_loss=(.+)\.ckpt', str(p)).group(1)))


def get_cfg(xp_cfg, overrides=None):
    overrides = overrides if overrides is not None else []
    with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp_cfg}',
                'file_paths=dgx_ifremer',
                'entrypoint=train',
            ] + overrides
        )

    return cfg

def get_model(xp_cfg, ckpt, dm=None, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides =  overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    if dm is None:
        dm = instantiate(cfg.datamodule)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls)
    mod = runner._get_model(ckpt)
    return mod

def get_dm(xp_cfg, setup=True, add_overrides=None):
    overrides = []
    if add_overrides is not None:
        overrides = overrides + add_overrides
    cfg = get_cfg(xp_cfg, overrides)
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
    try:
        metrics = []
        xps = cs.list('xp')
        for cfg in xps:
            if '5nad' in cfg: continue
            print(cfg)
            m = re.match(r'qxp(?P<xp_number>\d+)_(?P<xp_name>.+)\.yaml', cfg)
            xp_n = m.group('xp_number')
            xp_name = m.group('xp_name')
            

            # inference_configs:
            xp_dir = Path('xp_roll')
            data = {
                '5nad': '+xp/qfebvre/ds=five_nadirs',
                'roll': '+xp/qfebvre/ds=new_noisy_swot_roll',
                'syst': '+xp/qfebvre/ds=new_noisy_swot_syst_errs',
            }

            ws = {
                'w0.000': 'datamodule.item_prepro.w=0.000',
                'w0.001': 'datamodule.item_prepro.w=0.001',
                'w0.01': 'datamodule.item_prepro.w=0.01',
                'w0.1': 'datamodule.item_prepro.w=0.1',
                'w1.': 'datamodule.item_prepro.w=1',
            }

            ckpt = bst_ckpt(xp_dir / xp_name)

            
            for obs, obs_or in data.items():
                for w, w_or in ws.items():
                    print(obs, w) 
                    if (obs == '5nad') and not (w=='w1.'): continue

                    xp_cfg = get_cfg(cfg)
                    dm = get_dm(cfg, setup=False, add_overrides=[obs_or, w_or])
                    mod = get_model(cfg, ckpt, dm, add_overrides=[obs_or, w_or])
                    trainer = pl.Trainer(gpus=[3])
                    trainer.test(mod, dataloaders=dm.test_dataloader())
                    metrics.append(
                        {
                            'model': xp_name,
                            'xp_n': xp_n,
                            'obs_train': xp_cfg.datamodule.obs_mask_var,
                            'obs_test': obs + w,
                            'sst': xp_cfg.params.sst,
                            'cal': xp_cfg.params.loss_loc,
                            'aug': xp_cfg.params.aug_state,
                            'metrics': mod.latest_metrics,
                        }
                    )

                df = pd.DataFrame(
                    metrics
                ).pipe(
                        lambda _df:_df.assign(**pd.DataFrame(list(_df.metrics)).to_dict(orient='list'))
                ).drop('metrics', axis=1)
                df.to_csv('metrics_roll2.csv')
    except Exception as e:

        print(traceback.format_exc()) 

    finally:     

        return locals()
