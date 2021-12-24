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
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import calibration
import hydra_main
import einops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from hydra_main import FourDVarNetHydraRunner
import torchvision.transforms as T


xp = 'xmasxp/xp_debug_instab/repro_insta'
# CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
CKPT = 'dashboard/repro_insta/version_3/checkpoints/epoch=101-step=3059.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'file_paths=dgx_ifremer',
            'entrypoint=train',
            'logger.name=instab'
        ]
    )
    if cfg.get('callbacks') is not None:
        callbacks = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks]
    else:
        callbacks=[]

    if cfg.get('logger') is not None:
        logger = instantiate(cfg.logger)
    else:
        logger=True
    
    dm = instantiate(cfg.datamodule)
    lit_mod_cls = get_class(cfg.lit_mod_cls)
    runner = FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks + [pl.callbacks.RichProgressBar()], logger=logger)
    trainer = pl.Trainer(gpus=1, callbacks=[pl.callbacks.RichProgressBar()] , logger=logger, limit_train_batches=5, limit_val_batches=5)
    mod = runner._get_model(CKPT)
    trainer.fit(mod, datamodule=dm)

    # mod = runner.train(CKPT, )
# loss degradation

class LoadStableCheckpoint(pl.Callback):
    def __init__(self, model_ckpt_cb, threshold=1.5):
        prev_val_loss = None
        prev_state_dict = None
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        print(f'{dir(trainer)}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(f'{outputs=}
