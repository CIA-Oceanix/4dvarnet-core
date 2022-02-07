import torch
import math
import traceback
from pathlib import Path
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from einops import rearrange
 
def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

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
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren network

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                print(x.shape)
                print(mod.shape)
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)

# wrapper

class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out

class LitSirenAE(pl.LightningModule):
    def __init__(self, net, state_dim=512, img_dim=240):
        super().__init__()
        self.net = net
        self.state_dim = state_dim
        self.mod = SirenWrapper(
            self.net,
            latent_dim = self.state_dim,
            image_width = img_dim,
            image_height = img_dim
        )


    def encode(self, obs, obs_mask, n_iter=50):
        with torch.enable_grad():
            state = torch.zeros(obs.shape[0], self.state_dim).normal_(0, 1e-2).requires_grad_() 
            opt = torch.optim.Adam((state,))
            print(state.shape)
            for _ in range(n_iter):
                out = self.mod(latent=state)
                loss = F.mse_loss(out[obs_mask], obs[obs_mask])
                opt.zero_grad()
                loss.backward()
                opt.step()

        return state.detach()


    def forward(self, obs, mask):
        state_dec = self.encode(obs, mask)
        return self.mod(latent =state_dec)


    def training_step(self, batch, batch_idx):
        oi, mask, obs, gt, *other = batch
        print(obs.shape)
        print(gt.shape)
        state_dec = self.encode(obs, mask, n_iter=10)
        print(state_dec.shape)
        out = self.mod(latent = state_dec)
        loss = F.mse_loss(out, gt)
        self.log('decloss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


def main():
    try:
        xp = 'xmasxp/xp_siren/base'
        with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
            cfg = hydra.compose(config_name='main', overrides=
                [
                    f'xp={xp}',
                    'file_paths=srv5',
                    'entrypoint=train',
                ]
            )


        dm = instantiate(cfg.datamodule)
        dm.setup()



        net = SirenNet(
            dim_in = 2,                        # input dimension, ex. 2d coor
            dim_hidden = 256,                  # hidden dimension
            dim_out = 5,                       # output dimension, ex. rgb value
            num_layers = 5,                    # number of layers
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        lit_mod  = LitSirenAE(net, state_dim=512, img_dim=240)

        dl = dm.val_dataloader()
        batch = next(iter(dl))

        batch[0].shape
        print('hello')
        loss = lit_mod.training_step(batch, 0)
        print(loss)
        trainer = pl.Trainer(gpus=0)

    except Exception as e:
        print('Am I here')
        print(traceback.format_exc()) 
    finally:
        print('I have to be here')
        return locals()

