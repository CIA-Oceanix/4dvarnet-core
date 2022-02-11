import torch
import kornia
import matplotlib.pyplot as plt
import math
import traceback
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch import nn
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from einops import rearrange, repeat
 
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
        self.dim_in = dim_in
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
                x *= rearrange(mod, 'b d -> b () d')

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
    def __init__(self, net, dims, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNet), 'SirenWrapper must receive a Siren network'

        self.net = net
        assert len(dims) == net.dim_in

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )
        
        self.dims = dims

        tensors = [torch.linspace(-1, 1, steps = d) for d in dims.values()]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, '... c-> (...) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = repeat(self.grid.clone().detach().requires_grad_(), '... -> b ...', b=latent.shape[0]) 
        out = self.net(coords, mods)
        out = rearrange(out, f'b ({" ".join(self.dims.keys())}) c -> b {" ".join(self.dims.keys())} c',**self.dims)

        if exists(img):
            return F.mse_loss(img, out)

        return out

class LitSirenAE(pl.LightningModule):
    def __init__(self, net, state_dim=512, dims={}):
        super().__init__()
        self.net = net
        self.state_dim = state_dim
        self.mod = SirenWrapper(
            self.net,
            latent_dim = self.state_dim,
            dims=dims,
        )
        self.state_init = nn.Parameter(torch.randn(self.state_dim))
        self.solver_mod = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )
        self.norm_grad = None


    def encode(self, obs, obs_mask):
        # return self.state
        with torch.enable_grad():
            # state = torch.zeros(obs.shape[0], self.state_dim, device=self.device).normal_(0, 1e-2).requires_grad_() 
            state = repeat(self.state_init, 'd -> b d', b=obs.shape[0])

            out = self.mod(latent=state)
            loss = F.mse_loss(out[obs_mask], obs[obs_mask])
            state_grad = torch.autograd.grad(loss, state, create_graph=True)[0]
            norm_grad = self.norm_grad or torch.sqrt( torch.mean( state_grad**2 + 0.))
            state_out = self.state_init + self.solver_mod(state_grad / norm_grad) 

        # return state.detach()
        return state_out

    def get_batch_obs(self, batch):
        oi, mask, obs, *other = batch
        mod_obs = torch.stack((oi, torch.where(mask, obs-oi, obs)), dim=-1)
        mod_mask = torch.stack((oi.isfinite(), mask), dim=-1)
        return mod_obs, mod_mask

    def format_output(self, mod_out):
        return mod_out.sum(-1)

    def forward(self, batch):
        obs, mask = self.get_batch_obs(batch)
        state_dec = self.encode(obs, mask)
        out = self.mod(latent=state_dec)

        return self.format_output(out)

    def process_batch(self, batch, phase='val'):

        oi, _, _, gt, *other = batch
        obs, mask = self.get_batch_obs(batch)
        state_dec = self.encode(obs, mask)
        mod_out = self.mod(latent = state_dec)
        out = self.format_output(mod_out)

        err = torch.where(gt.isfinite(), out - gt, torch.zeros_like(gt))
        loss_rec = (err**2).mean()
        self.log(f'{phase}_loss_rec', loss_rec)

        err_obs = torch.where(mask, mod_out - obs, torch.zeros_like(obs))
        loss_obs = (err_obs**2).mean()
        self.log(f'{phase}_err_obs', loss_obs)
        with torch.no_grad():
            err_oi = torch.where(gt.isfinite(), oi - gt, torch.zeros_like(gt))
            loss_oi = (err_oi**2).mean()
            self.log(f'{phase}_imp_wrt_oi', loss_rec / loss_oi)

                   
        loss = loss_rec + loss_obs
        self.log(f'{phase}_loss', loss)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.process_batch(batch, phase='train')


    def validation_step(self, batch, batch_idx):
        return self.process_batch(batch, phase='val')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=2e-3)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, factor=0.8, min_lr=1e-5),
            'monitor': 'val_loss'
        }


def main():
    try:
        xp = 'xmasxp/xp_feb/base_siren'
        with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
            cfg = hydra.compose(config_name='main', overrides=
                [
                    f'xp={xp}',
                    'file_paths=dgx_ifremer',
                    'entrypoint=train',
                ]
            )


        dm = instantiate(cfg.datamodule, dl_kwargs=dict(batch_size=4, num_workers=0, pin_memory=False))



        net = SirenNet(
            dim_in = 3,                        # input dimension, ex. 2d coor
            dim_hidden = 256,                 # hidden dimension
            dim_out = 2,                       # output dimension, ex. rgb value
            num_layers = 4,                    # number of layers
            w0_initial = 30.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        lit_mod  = LitSirenAE(net, state_dim=32, dims={'time': 5, 'lat': 240, 'lon': 240, })

        logger = instantiate(cfg.logger)
        trainer = pl.Trainer(
            gpus=[2],
            logger=logger,
            callbacks=[
                callbacks.LearningRateMonitor(),
                callbacks.RichProgressBar(),
                callbacks.RichModelSummary(max_depth=3),
                callbacks.GradientAccumulationScheduler({1: 1, 10: 5, 30: 10, 60: 20}),
            ],
            max_epochs=301,
            # overfit_batches=10,
        )
        dm.setup()
        trainer.fit(
            lit_mod,
            # train_dataloader=dm.train_dataloader(),
            train_dataloader=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader()
        )
        print(logger.save_dir)
        dl = dm.train_dataloader()
        trainer = pl.Trainer(
            gpus=[2],
        )
        predictions = trainer.predict(lit_mod, dataloaders=dl)
        oi, _, _, gt, *others = next(iter(dl))
        img = plt.imshow(kornia.filters.sobel(oi)[0,2,...])
        plt.colorbar(img)
        plt.show()
        img = plt.imshow(kornia.filters.sobel(gt)[0,2,...])
        plt.colorbar(img)
        plt.show()
        img = plt.imshow(kornia.filters.sobel(predictions[0])[0,2,...])
        plt.colorbar(img)
        plt.show()
        img = plt.imshow(predictions[0][0,2,...] - gt[0,2,...])
        plt.colorbar(img)
        plt.show()
        
        img = plt.imshow(oi[0,2,...] - gt[0,2,...])
        plt.colorbar(img)
        plt.show()
    except Exception as e:
        # print('Am I here')
        print(traceback.format_exc()) 
    finally:
        print('I have to be here')
        return locals()

