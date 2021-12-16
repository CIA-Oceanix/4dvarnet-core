# %% Hard reload
"""
import sys
del sys.modules['main'], sys.modules['models'], sys.modules['new_dataloading']
"""
# %%
"""
:command! -nargs=0 PasteToPyPane !tmux paste-buffer -t py.0
nnoremap <leader>e :PasteToPyPane<cr>
"""
# %% [md]
"""
I want to compute metrics on the swath from the models 4Dvarnet with roll
"""

# %% Load nad_sst model
import pytorch_lightning as pl
# from models import LitModel, LitModelWithSST
import models
# from new_dataloading import FourDVarNetDataModule
import new_dataloading
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import torch
import xarray as xr
from collections import defaultdict
# from main import FourDVarNetRunner
import main

config_pkg = 'q.nad_sst'
runner = main.FourDVarNetRunner(config=config_pkg)

# ckpt_path="../4dvarnet-core/dashboard/train/nad_sst/checkpoints/modelSLAInterpGF-Exp3-epoch=22-val_loss=0.05.ckpt"
ckpt_path=None
mod = runner._get_model(ckpt_path=ckpt_path)

config = __import__("config_q.nad_sst")


cfg = OmegaConf.create(config.params)




# %% Generate maps

trainer = pl.Trainer(gpus=1)
trainer.test(mod, test_dataloaders=runner.dataloaders['test'])


# %%
# %matplotlib inline

# %%

mod.test_figs['maps']

# %%
mod.test_figs['nrmse']

# %%
grid

# %%
raw_item['natl']

# %% Misc try outs

import inspect

def src(obj):
    print(inspect.getsource(obj))

src(runner.dataloaders['test'].dataset.datasets[0].__getitem__)
src(runner.__class__)

%matplotlib inline

%load_ext autoreload
%autoreload 2

