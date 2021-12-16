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
import torchvision.transforms as T


xp = 'sla_gf_hal'
# CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'file_paths=dgx_ifremer',
            'entrypoint=train',
        ]
    )


class DebugCb(pl.Callback):
    def __init__(self, threshold=1.5):
        prev_val_loss = None
        prev_state_dict = None
    
    def on_train_batch_start():
        ...

    def on_validation_batch_start():
        ...
