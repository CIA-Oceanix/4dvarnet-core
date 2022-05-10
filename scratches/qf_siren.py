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
            'datamodule.dl_kwargs.num_workers=4',
            'datamodule.dl_kwargs.batch_size=4',
        ]
    )

def exists(val):
    return val is not None


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1., w0_initial=30., use_bias=True,
                 final_activation=None):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias,
                                activation=final_activation)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)





class LitSirenAE(pl.LightningModule):
    def __init__(self, ss, d_state=128, d_pos=32):
        super().__init__()
        self.d_state = d_state
        # self.d_pos = d_pos
        self.d_pos = 3
        # self.t_pos_embd = nn.Embedding(ss['t'], d_pos)
        # self.x_pos_embd = nn.Embedding(ss['x'], d_pos)
        # self.y_pos_embd = nn.Embedding(ss['y'], d_pos)
        self.ss = ss
        # self.pos_embd = torch.nn.Parameter(torch.rand(ss['t'] * ss['x'] * ss['y'], d_pos))
        self.pos_embd = torch.nn.Parameter(
            F.normalize(einops.rearrange(
                torch.stack(
                    torch.broadcast_tensors(
                        torch.arange(ss['t'], dtype=torch.float32)[:, None, None],
                        torch.arange(ss['x'], dtype=torch.float32)[None, :, None],
                        torch.arange(ss['y'], dtype=torch.float32)[None, None, :],
                    )
                ), 't x y d -> (t x y) d'
            ), dim=0), requires_grad=False
        )

        d_inp = d_state + d_pos
        self.decoder = SirenNet(d_inp, dim_hidden=128, dim_out=1, num_layers=4)
        # self.encoder = self.decoder
        self.encoder = SirenNet(self.d_pos + 1, dim_hidden=128, dim_out=d_state, num_layers=4)

    # def encode(self, obs, mask, state_enc_init=None, n_iter=None, opt=None, detach=True):
    #     ss = einops.parse_shape(obs, 'b t x y')
    #     if state_enc_init is None:
    #         state_enc = torch.zeros((ss['b'], self.d_state), device=self.device, requires_grad=True)
    #     else: 
    #         state_enc = state_enc_init

    #     full_embed = (
    #             self.t_pos_embd(torch.arange(ss['t'], device=self.device))[:, None, None]
    #             + self.x_pos_embd(torch.arange(ss['x'], device=self.device))[None, :, None]
    #             + self.y_pos_embd(torch.arange(ss['y'], device=self.device))[None, None, :]
    #     ).detach()
    #     grid_inp = torch.cat(torch.broadcast_tensors(full_embed[None, ...], state_enc[:, None, None, None, :]), dim=-1)
        
    #     if opt is None:
    #         opt = torch.optim.Adam((state_enc,) + tuple(self.t_pos_embd.parameters()) + tuple(self.x_pos_embd.parameters()) + tuple(self.y_pos_embd.parameters()), lr=0.001)

    #     n_iter = n_iter if n_iter is not None else 20
    #     for _ in range(n_iter):
    #         out = self.encoder(grid_inp).squeeze()
    #         opt.zero_grad()
    #         loss = F.mse_loss(out[mask], obs[mask])
    #         loss.backward()
    #         opt.step()
    #     if n_iter > 0:
    #         self.log('enc_loss', loss, prog_bar=True)
    #     return state_enc.detach() 

    def encode(self, obs, mask, state_enc_init=None, n_iter=None, opt=None, detach=True):
        ss = einops.parse_shape(obs, 'b t x y')


        nz = einops.rearrange(mask, 'bs t x y -> bs (t x y)').nonzero(as_tuple=True)
        embd = self.pos_embd[nz[1], :]
        # nz =mask.nonzero(as_tuple=True)
        # embd = self.t_pos_embd(nz[1]) + self.x_pos_embd(nz[2]) + self.y_pos_embd(nz[3])


        # print(embd.shape)
        # print(obs[mask].shape)
        # print(obs[mask][None, :].shape)
        # print(obs[mask][:, None].shape)
        inp = torch.cat((obs[mask][:, None], embd), dim=1)

        out = self.encoder(inp)
        M = torch.zeros(nz[0].max()+1, len(out), device=embd.device)
        
        M[nz[0], torch.arange(len(out))] =1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        state_out = torch.mm(M, out)
        return state_out

    def decode(self, state_dec):
        full_embed = einops.rearrange(self.pos_embd, '(t x y) de -> t x y de', t=self.ss['t'], x=self.ss['x'], y=self.ss['y'])
        # full_embed = (
        #         self.t_pos_embd(torch.arange(self.t_pos_embd.num_embeddings, device=self.device))[:, None, None]
        #         + self.x_pos_embd(torch.arange(self.x_pos_embd.num_embeddings, device=self.device))[None, :, None]
        #         + self.y_pos_embd(torch.arange(self.y_pos_embd.num_embeddings, device=self.device))[None, None, :]
        # )

        bcsted_state = einops.repeat(state_dec, 'bs ds -> bs t x y ds', **einops.parse_shape(full_embed, 't x y _'))
        bcsted_embd = einops.repeat(full_embed, 't x y de -> bs t x y de', **einops.parse_shape(state_dec, 'bs _'))
        grid_inp = torch.cat((bcsted_embd, bcsted_state), dim=-1)
        return self.decoder(grid_inp).squeeze()

    def forward(self, obs, mask):
        state_dec = self.encode(obs, mask)
        return self.decode(state_dec)


    def training_step(self, batch, batch_idx):
        oi, mask, obs, gt = batch
        state_dec = self.encode(obs, mask, n_iter=10)
        # state_dec = state_dec / torch.std(state_dec)
        out = self.decode(state_dec).squeeze()
        loss = F.mse_loss(out, gt)
        self.log('decloss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
        
def main():
    try:

        device='cuda:0'
        dm = instantiate(cfg.datamodule)
        dm.setup()
        batch = next(iter(dm.val_dataloader()))
        oi, mask, obs, gt = batch
        ss = einops.parse_shape(obs, 'b t x y')

        trainer = pl.Trainer(
            gpus=[1],
            callbacks=[pl.callbacks.RichProgressBar()],
        )
        lit_mod = LitSirenAE(ss)
        trainer.fit(lit_mod, datamodule=dm)

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        print('Am I here')
        return locals()


def scratch():
    s = ...
    list(s.keys())
    lit_mod = s['lit_mod']
    oi, mask, obs, gt = lit_mod.transfer_batch_to_device(s['batch'], lit_mod.device, 0)
    lit_mod.encode(obs, mask)

    plt.imshow(lit_mod.decode(lit_mod.encode(obs, mask))[0,2,...].detach().cpu())
   
    nz = mask.nonzero(as_tuple=True)
    lit_mod.t_pos_embd(nz[1]).shape
    embd = lit_mod.t_pos_embd(nz[1]) + lit_mod.x_pos_embd(nz[2]) + lit_mod.y_pos_embd(nz[3])
    d_state=128
    state_dec = torch.zeros(obs.size(0), d_state)

    M = torch.zeros(nz[0].max()+1, len(embd), device=embd.device)
    
    M[nz[0], torch.arange(len(embd))] =1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    torch.mm(M, embd).shape

    M

    M.sum(1)

    nz[0]

    M.shape

    M[labels, torch.arange(4)] = 1
    M = torch.nn.functional.normalize(M, p=1, dim=1)
    torch.mm(M, samples)
    ss = s['ss']
    ss = s['ss']
    device = s['device']
    state_enc = s['state_enc']
    t_pos_embd = s['t_pos_embd']
    x_pos_embd = s['x_pos_embd']
    y_pos_embd = s['y_pos_embd']
    s['loss']
    out = s['out']
    out.squeeze().shape
    full_embed = (
            t_pos_embd(torch.arange(ss['t']))[:, None, None]
            + x_pos_embd(torch.arange(ss['x']))[None, :, None]
            + y_pos_embd(torch.arange(ss['y']))[None, None, :]
    ).to(device)
    torch.cat(torch.broadcast_tensors(full_embed[None, ...], state_enc[:, None, None, None, :]), dim=-1).shape
    full_embed.shape
    mask.nonzero().shape
