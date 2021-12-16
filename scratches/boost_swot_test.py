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


import importlib
importlib.reload(calibration.lit_cal_model)
importlib.reload(main)
importlib.reload(metrics)
importlib.reload(hydra_main)

xp = 'xmasxp/base_xp'
CKPT = 'dashboard/version_13/checkpoints/modelCalSLAInterpGF-epoch=52-val_loss=0.0020.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'file_paths=dgx_ifremer',
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
mod = call(cfg.entrypoint, self=runner)


psd_scores = metrics.psd_based_scores(mod.test_xr_ds.pred, mod.test_xr_ds.gt)
rmse_scores = metrics.rmse_based_scores(mod.test_xr_ds.pred, mod.test_xr_ds.gt)

psd_scores = metrics.psd_based_scores(mod.test_xr_ds.oi, mod.test_xr_ds.gt)
rmse_scores = metrics.rmse_based_scores(mod.test_xr_ds.oi, mod.test_xr_ds.gt)
