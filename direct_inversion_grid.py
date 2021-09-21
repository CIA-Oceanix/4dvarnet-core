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
            self.hparams =hparams
            self.patch_size = 20
            self.vit =  kornia.contrib.VisionTransformer(
                image_size=hparams.W,
                patch_size=self.patch_size,
                in_channels=hparams.shape_obs[0] * 2,
                embed_dim=400,
            )
            self.n_patch= hparams.W // self.patch_size
            self.phi_r = nn.Identity()
            self.n_grad = 0

        def forward(self, state, obs, masks):
            low_res,  anom_glob_init, anom_swath_init = torch.split(state, split_size_or_sections=hparams.dT, dim=1)
            oi, sat_obs = torch.split(obs, split_size_or_sections=hparams.dT, dim=1)
            oi_mask, obs_mask = torch.split(masks, split_size_or_sections=hparams.dT, dim=1)
            in_vit = einops.rearrange([obs, masks], 'l b t h w -> b (l t) h w')
            out_vit = self.vit(in_vit)

            reshaped_out_vit = einops.repeat(
                out_vit[:, 1:, :],
                'b (ix iy) (h w) -> b t (ix h) (iy w)',
                iy=self.n_patch,
                ix=self.n_patch,
                t=self.hparams.dT,
                h=self.hparams.W // self.n_patch,
                w=self.hparams.W // self.n_patch,
            )

            print(reshaped_out_vit.shape)

            anom_glob = reshaped_out_vit
            anom_swath = reshaped_out_vit

            outputs = torch.cat([low_res, anom_glob, anom_swath], dim=1)
            return outputs, None, None, None 
    return Vit()
if __name__ == '__main__':
    import main
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
    runner = main.FourDVarNetRunner(config='q.local')
    # runner.train(fast_dev_run=True)
    mod = runner.test()
    # mod = runner._get_model()
    print('Hello World')

