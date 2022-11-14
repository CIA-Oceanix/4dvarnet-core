import utils
import functools as ft
import pl_bolts.models.autoencoders
import pl_bolts.models.autoencoders.components as aecomp
import pl_bolts.models.gans.dcgan.components as gancomp
import einops
import tqdm
from tqdm import tqdm
from einops.layers.torch import Rearrange, Reduce
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcb
import matplotlib.pyplot as plt
import pyinterp
import pyinterp.backends.xarray
import pyinterp.fill
import numpy as np
import xarray as xr
import traceback
import hydra_config
import sys
from omegaconf import OmegaConf
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('scratches')
from qf_base_xr_to_torch import XrDataset, XrConcatDataset
from collections import namedtuple, OrderedDict
base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    f'file_paths={fp}'
]


def remove_nan(da):
    da['lon'] = da.lon.assign_attrs(units='degrees_east')
    da['lat'] = da.lat.assign_attrs(units='degrees_north')

    da.transpose('lon', 'lat', 'time')[:,:] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da))[1]
    return da

TrainingItem = namedtuple('TrainingItem', ['ssh', 'nadir_obs', 'shuffled_nadir_obs'])


def get_constant_crop(patch_size, crop, dim_order=['time', 'lat', 'lon']):
        patch_weight = np.zeros([patch_size[d] for d in dim_order], dtype='float32')
        mask = tuple(
                slice(crop[d], -crop[d]) if crop.get(d, 0)>0 else slice(None, None)
                for d in dim_order
        )
        patch_weight[mask] = 1.
        return patch_weight

def build_dataloaders(path, patch_dims, strides, train_period, val_period, ds=None):
    inp_ds = xr.open_dataset(path)
    perm = np.random.permutation(inp_ds.dims['time'])

    downsampling = ds or dict()
    xr.Dataset.coarsen
    _pre_fns = [
            lambda ds: ds.coarsen(downsampling, boundary='trim').mean(),
    ]
    _post_fns = [
        TrainingItem._make
    ]
    pre_fn = ft.partial(ft.reduce,lambda i, f: f(i), _pre_fns)
    post_fn = ft.partial(ft.reduce,lambda i, f: f(i), _post_fns)
    new_ds = inp_ds.assign(
        shuffled_nadir_obs=(
            inp_ds.ssh.dims, 
            np.where(np.isfinite(inp_ds.nadir_obs.values[perm]), inp_ds.ssh.values, np.nan)
        )).assign(dict(
            ssh=lambda ds: remove_nan(ds.ssh),
            # obs_mask=lambda ds: xr.apply_ufunc(np.isfinite, ds.nadir_obs),
            # nadir_obs=lambda ds: xr.apply_ufunc(np.nan_to_num, ds.nadir_obs),
            # shuffled_nadir_obs=lambda ds: xr.apply_ufunc(np.nan_to_num, ds.shuffled_nadir_obs),
    ))[[*TrainingItem._fields]]
     
    train_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').pipe(pre_fn).to_array().sel(time=train_period),
        patch_dims=patch_dims, strides=strides,
        # prepro_fn=pre_fn,
        postpro_fn=post_fn,
    )
    val_ds = XrDataset(
        new_ds.transpose('time', 'lat', 'lon').pipe(pre_fn).to_array().sel(time=val_period),
        patch_dims=patch_dims, strides=strides,
        # prepro_fn=pre_fn,
        postpro_fn=post_fn,
    )
    print(f'{len(train_ds)=}, {len(val_ds)=}')
    return torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True, num_workers=1), \
        torch.utils.data.DataLoader(val_ds, batch_size=8, num_workers=1)

def off_diagonal(x):
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class LitAE(pl.LightningModule):
            def __init__(self, enc, dec, proj, rec_weight, rec_init):
                super().__init__()
                self.enc = enc
                self.dec = dec
                self.proj = proj
                self.rec_weight = nn.Parameter(torch.from_numpy(rec_weight), requires_grad=False)
                self.rec_init = nn.Parameter(torch.from_numpy(rec_init), requires_grad=False)
                # self.renorm = nn.LazyBatchNorm2d(affine=False)
                self.num_features = 512

            @staticmethod
            def weighted_mse(err, weight):
                err_w = (err * weight[None, ...])
                non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.
                err_num = err.isfinite() & ~non_zeros
                if err_num.sum() == 0:
                    return torch.scalar_tensor(0., device=err_num.device)
                loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
                return loss

            def vic_reg(self, embed1, embed2):
                """
                from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
                """
                batch_size = embed1.size(0)
                x = self.proj(embed1)
                y = self.proj(embed2)

                repr_loss = F.mse_loss(x, y)

                x = x - x.mean(dim=0)
                y = y - y.mean(dim=0)

                std_x = torch.sqrt(x.var(dim=0) + 0.0001)
                std_y = torch.sqrt(y.var(dim=0) + 0.0001)
                std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

                cov_x = (x.T @ x) / (batch_size - 1)
                cov_y = (y.T @ y) / (batch_size - 1)
                cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
                    self.num_features
                ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

                return (25. * repr_loss + 25. * std_loss + 1. * cov_loss)
            
            def ssl_loss(self, inp1, inp2):
                embed1 = self.enc(inp1)
                embed2 = self.enc(inp2)
                return self.vic_reg(embed1, embed2)

            def rec_loss(self, inp, tgt):
                rec = self.dec(self.enc(inp)) 
                return self.weighted_mse(rec - tgt, self.rec_weight)

            def training_step(self, batch, batch_idx, optimizer_idx=None):
                return self.step(batch, 'tr', optimizer_idx)

            def validation_step(self, batch, batch_idx):
                return self.step(batch, 'val')

            def step(self, batch, phase='', opt_idx=None):
                loss = 0

                rec_init = einops.repeat(self.rec_init, 't x y -> b t x y', b=batch.ssh.size(0))

                mask_prop =0.3
                mask = (torch.rand_like(batch.ssh) > mask_prop).int()
                ae_loss = self.rec_loss((batch.ssh - rec_init) * mask, (batch.ssh - rec_init))
                self.log(f'{phase}_ae_loss', ae_loss, prog_bar=True)
                if (opt_idx==1) or (phase!='tr'):
                    loss += 10 * ae_loss

                mask_prop =0.8
                mask = (torch.rand_like(batch.ssh) > mask_prop).int()
                inv_loss = self.rec_loss((batch.ssh - rec_init) * mask, (batch.ssh - rec_init))
                self.log(f'{phase}_inv_loss', inv_loss, prog_bar=True)
                # if (opt_idx==1) or (phase!='tr'):
                #     loss += 10 * inv_loss


                mask_prop =0.7
                mask1 = (torch.rand_like(batch.ssh) > mask_prop).int()
                mask2 = (torch.rand_like(batch.ssh) > mask_prop).int()
                ssl_loss = self.ssl_loss(
                        (batch.ssh - rec_init).nan_to_num()*mask1,
                        (batch.ssh - rec_init).nan_to_num()*mask2
                )
                # ssl_loss = self.ssl_loss(batch.ssh, batch.nadir_obs)
                self.log(f'{phase}_ssl_loss', ssl_loss, prog_bar=True)
                if (opt_idx==0) or (phase!='tr'):
                    loss += 10 * ssl_loss

                self.log(f'{phase}_loss', loss)
                return loss

            def configure_optimizers(self):
                opt_enc = torch.optim.Adam(
                        [{'params': (*self.enc.parameters(), *self.proj.parameters()), 'lr':2e-3},
                        # {'params': [self.rec_init], 'lr': 1e-4}
                         ],
                        lr=1e-3)
                opt_dec = torch.optim.Adam([
                    {'params': self.dec.parameters(), 'lr': 2e-3},
                    {'params': self.enc.parameters(), 'lr': 1e-3},
                    # {'params': [self.rec_init], 'lr': 3e-3}
                    ], lr=1e-3)
                # sched_enc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc, T_max=50)
                sched_enc = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_enc, 'min', verbose=True)
                # sched_dec = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dec, T_max=50)
                sched_dec = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_dec, 'min', verbose=True)
                return (
                    {'optimizer': opt_enc,
                     'scheduler': sched_enc,
                     'frequency': 0},
                    {'optimizer': opt_dec, 
                     'scheduler': sched_dec,
                     'frequency': 1}
                )

            def test_step(self, batch, batch_idx):
                rec_init = einops.repeat(self.rec_init, 't x y -> b t x y', b=batch.ssh.size(0))
                return torch.stack(
                    [(batch.ssh - rec_init).cpu(),
                     self.dec(self.enc((batch.ssh- rec_init).nan_to_num())).cpu(),
                     self.dec(self.enc((batch.nadir_obs - rec_init).nan_to_num())).cpu()],
                    dim=1)


            def test_epoch_end(self, outputs):
                rec_da = (
                    self.trainer
                    .test_dataloaders[0].dataset
                    .reconstruct(outputs, self.rec_weight.cpu().numpy())
                )
                npa = rec_da.values
                lonidx = ~np.all(np.isnan(npa), axis=tuple([0, 1, 2]))
                latidx = ~np.all(np.isnan(npa), axis=tuple([0, 1, 3]))
                tidx = ~np.all(np.isnan(npa), axis=tuple([0, 2, 3]))
                self.test_data = xr.Dataset({
                    k: rec_da.isel(v0=i, time=tidx, lat=latidx, lon=lonidx)
                    for i,k  in enumerate(['ssh', 'rec_ssh', 'rec_nadir'])
                })




def run1():
    try:
        cfg = utils.get_cfg(base_cfg, overrides=overrides)
        # print(OmegaConf.to_yaml(cfg.file_paths))
        print(cfg.file_paths.data_registry_path)


        strides = dict(time=1, lat=240, lon=240)
        _patch_dims = dict(time=21, lat=240, lon=240)
        ds = dict(time=3, lat=12, lon=12)
        patch_dims = {k: _patch_dims[k]//ds.get(k,1) for k in _patch_dims}

        train_period = slice('2012-10-01', '2013-06-30')
        val_period = slice('2013-07-01', '2013-08-30')
        train_dl, val_dl = build_dataloaders(
            f'{cfg.file_paths.data_registry_path}/qdata/natl20.nc',
            patch_dims,
            strides,
            train_period,
            val_period,
            ds=ds
        )

        batch = next(iter(train_dl))
        print(batch._fields)

        


        # crop = dict(time=5, lat=10, lon=10)
        crop = dict()
        rec_weight = get_constant_crop(patch_dims, crop) 
        
        # import torchvision
        # feat = torchvision.models.feature_extraction.get_graph_node_names(mod)[0][-2]
        # feat_extr = torchvision.models.feature_extraction.create_feature_extractor(mod,[feat])
        # feat_extr(batch.ssh)[feat].shape
        # encoder = ...

        encoder = nn.Sequential(OrderedDict(dict(
            bn_in=nn.BatchNorm2d(patch_dims['time'], affine=False),
            conv_in=nn.Conv2d(patch_dims['time'], 3, 3, padding=1),
            backbone=aecomp.resnet18_encoder(first_conv=False, maxpool1=False),
            # bn=nn.LazyBatchNorm1d(affine=False),
        )))
        code = encoder(batch.ssh)
        latent_dim = code.size(1)
        code.size()

        
        class Interpolate(nn.Module):
            def __init__(self, size):
                super().__init__()
                self.size=size
                
            def forward(self, x):
                return F.interpolate(x, size=self.size)

        # decoder = ...
        decoder = nn.Sequential(OrderedDict(dict(
            backbone=aecomp.resnet18_decoder(latent_dim=latent_dim, input_height=patch_dims['lat'], first_conv=False, maxpool1=False),
            up_samp=Interpolate([patch_dims['lat'], patch_dims['lat']]),
            conv_out=nn.Conv2d(3, patch_dims['time'], 3, padding=1),
            # bn_out=nn.BatchNorm2d(patch_dims['time']),
        )))

        # decoder = nn.Sequential(OrderedDict(dict(
        #     res=Rearrange('b c -> b c () ()'),
        #     gan=gancomp.DCGANGenerator(latent_dim=latent_dim, feature_maps=128, image_channels=128),
        #     out_mod=nn.Sequential(
        #         nn.UpsamplingBilinear2d([128, 128]),
        #         nn.Conv2d(128, 128, 3, padding=1),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(),
        #         nn.Conv2d(128, 64, 3, padding=1),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(),
        #         nn.Conv2d(64, 32, 3, padding=1),
        #         nn.BatchNorm2d(32),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 32, 3, padding=1),
        #         nn.UpsamplingBilinear2d([patch_dims['lat'], patch_dims['lat']]),
        #         nn.BatchNorm2d(32),
        #         nn.Tanh(),
        #         nn.Conv2d(32, 32, 3, padding=1),
        #         nn.Tanh(),
        #         nn.Conv2d(32, patch_dims['time'], 3, padding=1),
        #     )
        # )))

        rec = decoder(code)
        rec.size()

        proj = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        proj(code)


        trainer = pl.Trainer(gpus=[1], logger=False,
            max_epochs=200,
            callbacks=[
                plcb.ModelCheckpoint(monitor='val_ae_loss', save_last=True),
                plcb.TQDMProgressBar(),
                # plcb.GradientAccumulationScheduler({10:2, 15:4, 25:8}),
                # plcb.StochasticWeightAveraging(),
                # plcb.RichProgressBar(),
                plcb.ModelSummary(max_depth=1),
            ]
        )
        rec_init = einops.repeat(
            train_dl.dataset.da.sel(variable='ssh').median('time').transpose('lat', 'lon').values.astype(np.float32),
            'x y -> t x y',
             t=patch_dims['time'])
        rec_init = np.zeros_like(rec_init)
        lit_mod = LitAE(enc=encoder, dec=decoder, proj=proj, rec_weight=rec_weight, rec_init=rec_init)
        trainer.fit(lit_mod, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # lit_mod.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
        trainer.test(lit_mod, dataloaders=[val_dl])
        lit_mod.test_data.to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(col='variable', row='time')

        # plt.imshow(lit_mod.rec_init.detach().cpu()[patch_dims['time']//2])

        trainer.test(lit_mod, dataloaders=[train_dl])
        lit_mod.test_data.to_array().isel(time=slice(0, 30, 10)).plot.pcolormesh(col='variable', row='time')

        lit_mod.test_data.pipe(lambda ds: ds.ssh - ds.rec_ssh).pipe(lambda da: np.sqrt(np.mean(da**2)))

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()



def main():
    try:
        fn = run1
        # fn = simple_models

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
    ...
    locals().update(main())
