# %% [md]
"""
inp: obs - OI
out: gt - oi
model: Unet
"""

# %%

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import kornia.contrib
import einops
from einops.layers.torch import Rearrange

def get_passthrough(hparams):
    class PassThrough(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi_r = nn.Identity()

        def forward(self, state, obs, masks):

            low_res,  anom_glob_init, anom_swath_init = torch.split(state, split_size_or_sections=hparams.dT, dim=1)
            oi, sat_obs = torch.split(obs, split_size_or_sections=hparams.dT, dim=1)
            oi_mask, obs_mask = torch.split(masks, split_size_or_sections=hparams.dT, dim=1)

            # anom = torch.where(obs_mask, sat_obs - oi, torch.zeros_like(sat_obs))   
            # anom_glob = anom
            # anom_swath = anom

            anom_glob = torch.zeros_like(sat_obs)
            anom_swath = torch.zeros_like(sat_obs)

            outputs = torch.cat([oi, anom_glob, anom_swath], dim=1)
            return outputs, None, None, None 

    return PassThrough()

def get_vit(hparams):

    class Vit(nn.Module):
        def __init__(self):
            super().__init__()
            self.hparams = hparams
            self.patch_size = 20
            self.use_mask = self.hparams.vit_mask if hasattr(self.hparams, 'vit_mask') else True
            self.anom_scaling = self.hparams.anom_scaling if hasattr(self.hparams, 'anom_scaling') else 1.
            self.drop_out_rate = self.hparams.drop_out_rate if hasattr(self.hparams, 'drop_out_rate') else 0.
            self.drop_out_attn = self.hparams.drop_out_attn if hasattr(self.hparams, 'drop_out_attn') else 0.
            out_c = self.hparams.out_c if hasattr(self.hparams, 'out_c') else 1000
            depth = self.hparams.depth if hasattr(self.hparams, 'depth') else 6
            num_heads = self.hparams.num_heads if hasattr(self.hparams, 'num_heads') else 10
            if self.use_mask:
                in_ch = hparams.shape_obs[0] * 2
            else:
                in_ch = hparams.shape_obs[0]
            self.n_patch= hparams.W // self.patch_size
            self.vit =  kornia.contrib.VisionTransformer(
                num_heads=num_heads,
                depth=depth,
                image_size=hparams.W,
                patch_size=self.patch_size,
                in_channels=in_ch,
                embed_dim=out_c,
                dropout_attn=self.drop_out_attn,
                dropout_rate=self.drop_out_rate,
            )
            rec_mod_name = self.hparams.rec_mod if hasattr(self.hparams, 'rec_mod') else "default"
            if rec_mod_name == "default":
                self.rec_mod = nn.Sequential(
                    nn.Linear(out_c, self.patch_size **2),
                    Rearrange(
                        'b (ix iy) (h w) -> b () (ix h) (iy w)',
                        iy=self.n_patch,
                        ix=self.n_patch,
                        h=self.hparams.W // self.n_patch,
                        w=self.hparams.W // self.n_patch,
                    ),
                    nn.Conv2d(1, 4, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(4, 5, 1, padding=0),
                )
            elif rec_mod_name == "deconv":
                self.rec_mod = nn.Sequential(
                    Rearrange(
                        'b (ix iy) e -> b e ix iy ',
                        iy=self.n_patch,
                        ix=self.n_patch,
                    ),
                    nn.Conv2d(1, 4, 3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(4, 4, self.patch_size, stride=self.patch_size),
                    nn.Conv2d(4, 5, 1, padding=0),
                )
            elif rec_mod_name == "moreconv":
                self.rec_mod = nn.Sequential(
                    nn.Linear(out_c, self.patch_size **2),
                    Rearrange(
                        'b (ix iy) (h w) -> b () (ix h) (iy w)',
                        iy=self.n_patch,
                        ix=self.n_patch,
                        h=self.hparams.W // self.n_patch,
                        w=self.hparams.W // self.n_patch,
                    ),
                    nn.Conv2d(1, 8, 7, padding=3),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 5, 1, padding=0),
                )
            else:
                raise Exception("wrong rec mod " + rec_mod_name)
            self.phi_r = nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks):
            low_res,  anom_glob_init, anom_swath_init = torch.split(state, split_size_or_sections=hparams.dT, dim=1)
            oi, sat_obs = torch.split(obs, split_size_or_sections=hparams.dT, dim=1)
            oi_mask, obs_mask = torch.split(masks, split_size_or_sections=hparams.dT, dim=1)
            if self.use_mask:
                in_vit = einops.rearrange([obs, masks], 'l b t h w -> b (l t) h w')
            else: 
                in_vit = obs

            out_vit = self.vit(in_vit)

            reshaped_out_vit = self.rec_mod(out_vit[:, 1:, :])

            anom_obs = torch.where(obs_mask, sat_obs - oi, torch.zeros_like(sat_obs))  
            anom_glob = (reshaped_out_vit + anom_obs)  / self.anom_scaling
            anom_swath = (reshaped_out_vit  + anom_obs)/ self.anom_scaling

            outputs = torch.cat([low_res, anom_glob, anom_swath], dim=1)
            return outputs, None, None, None 
    return Vit()

if __name__ == '__main__':
    import main
    import importlib
    importlib.reload(main)
    import models
    importlib.reload(models)
    
    # import config_q.local
    # import xarray as xr
    # import pandas as pd
    # params = config_q.local.params 
    # path=params['files_cfg']['gt_path']
    # _ds = xr.open_dataset(path, decode_times=False)
    # if decode:
    #     _ds.time.attrs["units"] = "seconds since 2012-10-01"
    #     _ds['time'] = pd.to_datetime(_ds.time)
    # print('filtering')
    # self.ds = _ds.sel(**(params['dim_range'] or {}))
    runner = main.FourDVarNetRunner(config='q.xp_seven.direct_phi')
    runner.train(fast_dev_run=True)
    mod = runner.test()
    # mod = runner._get_model()
    print('Hello World')

