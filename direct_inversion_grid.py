# %% [md]
"""
inp: obs - OI
out: gt - oi
model: Unet
"""

# %%

from models import LitModel
import pytorch_lightning as pl
import torch
import torch.nn as nn




class LitDirectInv(LitModel):
    def create_model():
        class PassThrough(nn.Module):
            def forward(state, obs, new_masks):
                low_res,  anom_glob_init, anom_swath_init = torch.split(state, split_size_or_sections=self.hparams.dT, dim=1)
                oi, sat_obs = torch.split(state, split_size_or_sections=self.hparams.dT, dim=1)
                oi_mask, obs_mask = torch.split(obs, split_size_or_sections=self.hparams.dT, dim=1)

                anom_glob = sat_obs
                anom_swath = sat_obs

                outputs = torch.cat([low_res, anom_glob, anom_swath], dim=1)
                return outputs, hidden_new, cell_new, normgrad 

        return PassThrough()
