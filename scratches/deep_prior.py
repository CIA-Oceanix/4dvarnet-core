import torch.nn as nn
import traceback
import sys
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


xp = 'james/map_aug'
# CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'file_paths=dgx_ifremer',
            'entrypoint=train',
            'datamodule.dl_kwargs.num_workers=4',
            'datamodule.dl_kwargs.batch_size=4',
        ]
    )


class GeoFieldClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])        
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), norm_transform])
        self.fine_tune_layers = nn.ModuleDict(
            {'inp': nn.Conv2d(5, 3, 3, padding=1),
            'out': nn.Linear(1000, 30)},
        )

        self.model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=False)

    def forward(self, x):
        # x = self.fine_tune_layers['inp'](x)
        x = self.transform(x)
        x = self.model(x)
        # x = self.fine_tune_layers['out'](x)
        return x

def main():
    try:
        torch.use_deterministic_algorithms(False)
        device='cuda:0'
        dm = instantiate(cfg.datamodule)
        dm.setup()
        batch = next(iter(dm.val_dataloader()))

        oi, mask, obs, gt, obs_gt = batch
        oi, mask, obs, gt, obs_gt = oi.to(device), mask.to(device), obs.to(device), gt.to(device), obs_gt.to(device) 
        inp_tgt = gt.where(~gt.isnan(), torch.zeros_like(gt))[:, :3, ...]
        state_init = torch.rand_like(inp_tgt).requires_grad_(True)
        mod = GeoFieldClassifier().to(device)
        out_tgt = mod(inp_tgt)
        
        opt = torch.optim.Adam((state_init,), lr=1**-4)#, weight_decay=0.001)
        states_acc = []
        print(states_acc)
        state = state_init
        for _ in range(500):
            opt.zero_grad()
            out = mod(state)
            loss = (out - out_tgt).abs().sum()
            loss.backward(inputs=state)
            print(loss)
            opt.step()
            states_acc.append(state.detach().cpu())
            
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        print('Am I here')
        return locals()


def scratch():
    s = ...
    list(s.keys())
    states = s['states_acc']
    inp_tgt = s['inp_tgt']
    plt.imshow(inp_tgt.detach().cpu()[3,2,...])
    states[0].shape
    for state in states[::100]:
        plt.imshow(state[3, 2, ...].cpu().numpy())
        plt.show()

