import xarray as xr
from functools import reduce
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
import hydra_main


xp = 'calmap_patch'
# CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'entrypoint=train',
        ]
    )
import importlib
import new_dataloading
import calibration.lit_cal_model
import models
import calibration.dataset
import solver
importlib.reload(new_dataloading)
importlib.reload(solver)
importlib.reload(models)
importlib.reload(calibration.lit_cal_model)
importlib.reload(calibration.dataset)


# dm = instantiate(cfg.datamodule)
callbacks = None
lit_mod_cls = get_class(cfg.lit_mod_cls)
runner = hydra_main.FourDVarNetHydraRunner(cfg.params, dm, lit_mod_cls, callbacks=callbacks)

lit_mod = runner._get_model()
# lit_mod = get_class(cfg.lit_mod_cls).load_from_checkpoint(CKPT, hparam=cfg.params)

trainer = pl.Trainer(gpus=1)
trainer.fit(lit_mod, datamodule=dm)
lit_mod.w_loss.shape
test_dl = dm.test_dataloader()
import inspect

test_ds = test_dl.dataset.datasets[0]
trainer.test(lit_mod, test_dataloaders=test_dl)

