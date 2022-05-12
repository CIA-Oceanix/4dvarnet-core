import torch.nn as nn
import traceback
import sys
import math
import functools
import torch
import kornia
import xarray as xr
from functools import reduce
import numpy as np
import pandas as pd
import torch
import hydra
from hydra.utils import get_class, instantiate, call
import hydra_config
from hydra_main import FourDVarNetHydraRunner
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import calibration
import hydra_main
import main
import einops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import metrics


# import importlib
# importlib.reload(calibration.lit_cal_model)
# importlib.reload(main)
# importlib.reload(metrics)
# importlib.reload(hydra_main)

def test(xp, ckpt):
    print(xp, ckpt)
    with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp}',
                'file_paths=dgx_ifremer',
                'logger.save_dir=scratches_logs',
                'logger.name=' + xp.split('/')[-1],
                'entrypoint=test',
                f'entrypoint.ckpt_path=' + CKPT.replace("=", r"\="),
            ]
        )



    pl.seed_everything(seed=cfg.get('seed', None))
    dm = instantiate(cfg.datamodule)
    if cfg.get('callbacks') is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks=[]

    if cfg.get('logger') is not None:
        logger = instantiate(cfg.logger)
    else:
        logger=True

    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks, logger=logger)
    mod = runner._get_model(CKPT)

    # psd_scores = metrics.psd_based_scores(mod.test_xr_ds.oi, mod.test_xr_ds.gt)
    # rmse_scores = metrics.rmse_based_scores(mod.test_xr_ds.oi, mod.test_xr_ds.gt)


    def fn(trainer, pl_module):
        print(pl_module.logger.log_dir)
        print(pl_module.local_rank)
        print(pl_module.global_rank)

    trainer = pl.Trainer(
            gpus=1, #[4,5],
            logger=logger,
            strategy='ddp',
            # callbacks=[pl.callbacks.LambdaCallback(on_test_end=fn)]
    )
    trainer.test(mod, datamodule=dm)
    test_xrds = xr.open_dataset(Path(trainer.logger.log_dir) / 'test.nc')

    # psd_scores = metrics.psd_based_scores(test_xrds.pred, test_xrds.gt)
    # print(psd_scores)
    # rmse_scores = metrics.rmse_based_scores(test_xrds.pred, test_xrds.gt)
    # print(rmse_scores)

# xp = 'xmasxp/xp_boost_swot_ref_l1'
# CKPT = 'dashboard/xp_boost_swot_ref_l1/checkpoints/modelCalSLAInterpGF-epoch=52-val_loss=0.0020.ckpt'
# test(xp, CKPT)

# xp = 'xmasxp/xp_boost_swot_ref_l2'
# CKPT = 'dashboard/xp_boost_swot_ref_l2/version_2/checkpoints/modelCalSLAInterpGF-epoch=97-val_loss=0.0021.ckpt'
# test(xp, CKPT)


# xp = 'xmasxp/xp_boost_swot_ref_cascade'
# CKPT = 'dashboard/xp_boost_swot_cascade/version_0/checkpoints/modelCalSLAInterpGF-epoch=135-val_loss=0.0020.ckpt'
# test(xp, CKPT)

# xp = 'xmasxp/xp_boost_swot_ref_l2'
# CKPT = 'dashboard/xp_boost_swot_ref_l2/version_2/checkpoints/modelCalSLAInterpGF-epoch=97-val_loss=0.0021.ckpt'
# test(xp, CKPT)

# xp = 'xmasxp/xp_boost_swot_ref_aug_state'
# CKPT = 'dashboard/xp_boost_swot_aug/version_0/checkpoints/modelCalSLAInterpGF-epoch=70-val_loss=0.0019.ckpt'
# test(xp, CKPT)

# xp = 'xmasxp/xp_boost_swot_ref_240'
# CKPT ='dashboard/xp_boost_swot_240/version_0/checkpoints/modelCalSLAInterpGF-epoch=73-val_loss=0.0015.ckpt'
# test(xp, CKPT)

# xp = 'xmasxp/full'
# CKPT ='dashboard/full/version_5/checkpoints/modelcalslainterpgf-epoch=52-val_loss=0.0015.ckpt'
# test(xp, CKPT)

xp = 'xmasxp/multi_patch'
CKPT='dashboard/multi_patch/version_8/checkpoints/modelCalSLAInterpGF-epoch=17-val_loss=0.0014.ckpt'
test(xp, CKPT)
