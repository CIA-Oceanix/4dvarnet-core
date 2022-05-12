import xarray as xr
import numpy as np
import sparse
import pandas as pd
import torch
import hydra
from hydra.utils import get_class, instantiate, call
import hydra_config
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import calibration


xp = 'sla_gf_hal'
# CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'entrypoint=train',
            'file_paths=jz',
            'datamodule.strides.lat=40',
            'datamodule.strides.lon=40',
            'datamodule.dim_range.lat._args_=[32.,44.]',
            'datamodule.dim_range.lon._args_=[-66., -54.]',
        ]
    )
import importlib
import new_dataloading
import calibration.lit_cal_model
import calibration.dataset
importlib.reload(new_dataloading)
importlib.reload(calibration.lit_cal_model)
importlib.reload(calibration.dataset)

dm = instantiate(cfg.datamodule)
dm.setup()


torch.save(batch, 'sample_batch_4dvarnet.torch')
import pickle
with open('sample_batch_4dvarnet.p', 'w')
