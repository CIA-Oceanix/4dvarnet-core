import torch.nn as nn
import functools
import torch
import kornia
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
import einops
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T


xp = 'james/map'
# CKPT = 'archive_dash/finetune_calmap_gf_dec_lr/lightning_logs/version_2021874/checkpoints/modelCalSLAInterpGF-Exp3-epoch=49-val_loss=0.06.ckpt'
with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
    cfg = hydra.compose(config_name='main', overrides=
        [
            f'xp={xp}',
            'entrypoint=train',
            'datamodule.dl_kwargs.num_workers=4',
            'datamodule.dl_kwargs.batch_size=8',
        ]
    )


# encoder = kornia.contrib.vit_mobile.MobileViT(in_channels=10, patch_size=(10, 10))

class VitAE(nn.Module):
    def __init__(self, img_s=240, ps=10):
        super().__init__()
        self.ps = ps
        self.dd = 64
        self.ed = 256
        self.out_ch = 5
        self.img_s = img_s
        self.encoder=kornia.contrib.VisionTransformer(
                image_size=self.img_s,
                patch_size=self.ps,
                in_channels=10,
                embed_dim=self.ed,
                depth=4,
                num_heads=16,
                dropout_rate=0.1,
                dropout_attn=0.1,
                backbone=None,
        )
        self.enc_to_dec = nn.Linear(self.ed, self.dd)
        self.num_p = (self.img_s // self.ps)**2
        self.pos_embedding = nn.Embedding(self.num_p, self.dd)
        self.mask_enc = nn.Parameter(torch.randn(self.dd))
        self.decoder = nn.Transformer(
                d_model=self.dd,
                nhead=64,
                num_encoder_layers=0,
                num_decoder_layers=4,
                dim_feedforward=self.dd,
                dropout=0.1,
                batch_first=True,
        )
        self.to_pix = nn.Linear(self.dd, self.ps**2 * self.out_ch)
        self.out_filter =  nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2),
            nn.SiLU(),
            nn.Conv2d(self.out_ch, 1, 7, padding=3),
        )

    def forward(self, obs, mask):
        inp = einops.rearrange([obs, mask.float()], ' c2 b c ... -> b (c2 c) ...')
        enc = self.encoder(inp)
        mask_encs = einops.repeat(self.mask_enc, 'dd -> b np dd', b=inp.size(0), np=self.num_p) \
                + self.pos_embedding(torch.arange(self.num_p, device=obs.device))
        to_dec = torch.cat((mask_encs, self.enc_to_dec(enc)), dim=1)
        dec = self.decoder(self.enc_to_dec(enc), mask_encs)
        out = self.to_pix(dec)
        # print(out.shape)
        img_out = einops.rearrange(out, 'b (px py) (ph pw c) -> b c (px pw) (py ph)', px=self.img_s//self.ps, ph=self.ps, c=self.out_ch)
        # print(img_out.shape)
        return self.out_filter(img_out)


# %% Siren def
from torch import nn
import torch
import math
from einops import rearrange, repeat
import torch.nn.functional as F

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

class ViTSirenAE(nn.Module):
    def __init__(self, img_s=240, ps=10):
        super().__init__()
        self.ps = ps
        self.dd = 32
        self.ed = 384 #320 
        self.img_s = img_s
        self.inp_resize = T.transforms.Resize(256)
        self.encoder=kornia.contrib.vit_mobile.MobileViT(
                mode='xs',
                in_channels=5,
                patch_size=(2,2),
                dropout=0.1,
        )
        self.reduction_factor = img_s  // 8

        self.enc_to_dec = nn.Linear(self.ed, self.dd)
        self.rel_lat_pos_embedding = nn.Parameter(torch.rand(1, 1, 1, 1, self.reduction_factor, 1))
        self.rel_lon_pos_embedding = nn.Parameter(torch.rand(1, 1, 1, 1, 1, self.reduction_factor))
        self.up = nn.UpsamplingBilinear2d(scale_factor=self.reduction_factor) 
        # self.lat_pos_embedding = nn.Parameter(torch.rand(1, 1, self.img_s, 1))
        # self.lon_pos_embedding = nn.Parameter(torch.rand(1, self.img_s, 1, 1))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.img_s, self.img_s, 1))
        self.decoder = SirenNet(dim_in=self.dd, dim_hidden=256, dim_out=1, num_layers=8)

    def forward(self, inp):
        inp = self.inp_resize(inp)
        enc_tok = self.encoder(inp) # shape bs 320 8 8
        _dec_tok=enc_tok
        _dec_tok = self.up(_dec_tok)
        _dec_tok = einops.rearrange(_dec_tok, 'bs d x y -> bs x y d')
        _dec_tok = self.enc_to_dec(_dec_tok)
        # _dec_tok = einops.rearrange(_dec_tok, 'bs x y d-> bs x y d () ()')
        # _dec_tok =  _dec_tok + self.rel_lat_pos_embedding
        # _dec_tok =  _dec_tok - self.rel_lon_pos_embedding
        # print(_dec_tok.shape)
        # _dec_tok =  einops.rearrange(_dec_tok, 'bs x y d lat lon -> bs (x lon) (y lat) d')
        # print(_dec_tok.shape)
        # print(self.lat_pos_embedding.shape)
        # _dec_tok =  _dec_tok + self.lat_pos_embedding
        # print(_dec_tok.shape)
        # dec_tok =  _dec_tok - self.lon_pos_embedding
        dec_tok =  _dec_tok - self.pos_embedding
        _output = self.decoder(dec_tok)
        output = einops.rearrange(_output, 'bs px py () -> bs () px py')
        return output

class MultiScaleVitAE(nn.Module):
    def __init__(self, img_s=240, mod_cls=ViTSirenAE):
        super().__init__()
        self.mr_ratio = 5
        self.lr_ratio = 15

        self.hr_mod = mod_cls(img_s=img_s, ps=24)
        self.mr_mod = mod_cls(img_s=img_s // self.mr_ratio, ps=10)
        self.lr_mod = mod_cls(img_s=img_s // self.lr_ratio, ps=4)

        # self.up_lr = nn.Identity()
        # self.up_mr = nn.Identity()
        self.up_lr = nn.UpsamplingBilinear2d(scale_factor=self.lr_ratio)
        self.up_mr = nn.UpsamplingBilinear2d(scale_factor=self.mr_ratio)
        self.hr_weight = nn.Parameter(torch.ones(1, img_s, img_s) / 2)
        self.mr_weight = nn.Parameter(torch.ones(1, img_s, img_s) / 3)
        self.lr_weight = nn.Parameter(torch.ones(1, img_s, img_s) / 4)

    def forward(self, inp):
        # hr_inp = (obs, mask.float())
        hr_inp = inp
        mr_inp = einops.reduce(
                    hr_inp,
                    'b c (w dw) (h dh) -> b c w h',
                    dw=self.mr_ratio,
                    dh=self.mr_ratio,
                    reduction='mean'
        )
        
        lr_inp = einops.reduce(
                    hr_inp,
                    'b c (w dw) (h dh) -> b c w h',
                    dw=self.lr_ratio,
                    dh=self.lr_ratio,
                    reduction='mean'
                )
        

        out_hr = self.hr_mod(hr_inp)
        out_mr = self.up_mr( self.mr_mod(mr_inp))
        out_lr = self.up_lr( self.lr_mod(lr_inp))
        # print(out_lr.shape)
        # print(self.lr_weight.shape)
        out =  (
                out_lr * self.lr_weight 
                + out_mr * self.mr_weight 
                + out_hr * self.hr_weight 
        )

        return out


class GeoFieldClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])        
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), norm_transform])
        self.inp_layer = nn.Conv2d(1, 3, 3, padding=1)
        self.model = torchvision.models.mobilenet.mobilenet_v3_large(pretrained=False)
        self.out_layer = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.inp_layer(x)
        x = self.transform(x)
        x = self.model(x)
        x = self.out_layer(x)
        return torch.sigmoid(x) 


class DirectTraining(pl.LightningModule):
    def __init__(self, img_s=240):
        super().__init__()
        self.anom=True
        # self.gen = MultiScaleVitAE()
        self.gen = ViTSirenAE()
        # self.dis = GeoFieldClassifier() 
        self.test_crop = tuple([slice(20, -20), slice(20, -20)])
        self.automatic_optimization = True
        self.freq_gan = 5
        self.aug = nn.Sequential(
                # kornia.augmentation.RandomGaussianBlur(p=0.2, kernel_size=(11,11), sigma=(0.1, 0.1)),
                kornia.augmentation.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.5, 2)),
                kornia.augmentation.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.5, 2)),
                # kornia.augmentation.RandomCrop(p=0.2, size=(230, 230), padding=(5, 5)),
        )
    # def fp_solve(self, obs, mask, n_iter=5):
    #     x = obs
    #     clip_mask = torch.ones_like(x).bool()
    #     clip_mask[:, 2, ...] = mask[:, 2, ...]
    #     for _ in range(n_iter):
    #         pred = self(x, mask)
    #         if _ < n_iter -1:
    #             x = obs.where(clip_mask, einops.repeat(pred, 'b h w -> b c h w', c=mask.size(1)))

    #     return x


    
    def configure_optimizers(self):
        return (
            torch.optim.Adam(self.gen.parameters(), lr=1e-4, weight_decay=1e-4),
            # torch.optim.Adam(self.dis.parameters(), lr=5e-4, weight_decay=1e-5),
        )

    def training_step(self, batch, batch_idx):
        losses, pred, g_pred, tgt, g_tgt = self.process_batch(batch, phase='train')
        loss_g = (losses['mse'] + 2 * losses['mse_grad'] 
                + losses.get('aug_mse', 0.) + losses.get('aug_mse_grad', 0.)
                + losses.get('rm_mse', 0.) + losses.get('rm_mse_grad', 0.)
            )
        if (batch_idx == 0) and (self.current_epoch % 4 ==0):
            fig, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].imshow(pred.detach().cpu().numpy()[0, 0, ...])
            ax1[1].imshow((pred-tgt).detach().cpu().numpy()[0, 0, ...])
            ax2[0].imshow(g_pred.detach().cpu().numpy()[0, 0, ...])
            ax2[1].imshow((g_pred - g_tgt).detach().cpu().numpy()[0, 0, ...])
            self.logger.experiment.add_figure('Train Rec', fig, global_step=self.current_epoch)
        return loss_g

    def validation_step(self, batch, batch_idx):
        losses, pred, g_pred, tgt, g_tgt = self.process_batch(batch, phase='val')

        if (batch_idx == 0) and (self.current_epoch % 5 ==0):
            fig, (ax1, ax2) = plt.subplots(2, 2)
            ax1[0].imshow(pred.detach().cpu().numpy()[0, 0, ...])
            ax1[1].imshow((pred-tgt).detach().cpu().numpy()[0, 0, ...])
            ax2[0].imshow(g_pred.detach().cpu().numpy()[0, 0, ...])
            ax2[1].imshow((g_pred - g_tgt).detach().cpu().numpy()[0, 0, ...])
            self.logger.experiment.add_figure('Val Rec', fig, global_step=self.current_epoch)
        return losses['mse']

    def process_batch(self, batch, phase='val'):
        oi, mask, obs, gt, obs_gt = batch
        # inp = einops.rearrange([obs, mask.float()], ' c2 b c ... -> b (c2 c) ...')
        # inp = einops.rearrange([obs, oi], ' c2 b c ... -> b (c2 c) ...')
       
        if self.anom:
            gt = gt - oi
            inp = (obs - oi).where(mask, torch.zeros_like(obs))
            tgt = gt[:, 2:3, ...]
            oi = oi - oi
        else:
            inp = obs
            tgt = gt[:, 2:3, ...]

        mean, std = (
                einops.reduce(inp, 'b c h w -> c', reduction=red)
                for red in ('mean', torch.std)
        ) 
        # denorm = functools.partial(
        #         kornia.enhance.normalize(mean=mean, std=std)
        # )
        # print(oi.shape)
        # gen
        pred = self.gen(inp)

        test_pred = pred[:, :, self.test_crop[0], self.test_crop[1]]
        test_tgt = tgt[:, :, self.test_crop[0], self.test_crop[1]]
        nan_msk=~test_tgt.isnan()
        loss = F.mse_loss(test_pred[nan_msk], test_tgt[nan_msk])
        
        g_pred = kornia.filters.sobel(pred)[:, :, self.test_crop[0], self.test_crop[1]]
        g_tgt = kornia.filters.sobel(tgt)[:, :, self.test_crop[0], self.test_crop[1]]
        g_nan_msk=~g_tgt.isnan()
        g_loss = F.mse_loss(g_pred[g_nan_msk], g_tgt[g_nan_msk])


        # if phase =='train':
        #     aug_inp = self.aug(inp)
        # aug_pred = self.gen(inp)
        # test_aug_pred = aug_pred[:, :, self.test_crop[0], self.test_crop[1]]
        # g_aug_pred = kornia.filters.sobel(aug_pred)[:, :, self.test_crop[0], self.test_crop[1]]
        # aug_g_loss = F.mse_loss(g_aug_pred[g_nan_msk], g_tgt[g_nan_msk])
        # aug_loss = F.mse_loss(test_aug_pred[nan_msk], test_tgt[nan_msk])
        # g_aug_pred = kornia.filters.sobel(aug_pred)[:, :, self.test_crop[0], self.test_crop[1]]


        rm_mask = (torch.rand_like(gt) > 0.99) & (~gt.isnan())
        rm_obs = gt.where(rm_mask, torch.zeros_like(gt))
        rm_obs = self.aug(rm_obs)
        rm_pred = self.gen(rm_obs)
        rm_test_pred = rm_pred[:, :, self.test_crop[0], self.test_crop[1]]
        rm_loss = F.mse_loss(rm_test_pred[nan_msk], test_tgt[nan_msk])
        
        rm_g_pred = kornia.filters.sobel(rm_pred)[:, :, self.test_crop[0], self.test_crop[1]]
        rm_g_loss = F.mse_loss(rm_g_pred[g_nan_msk], g_tgt[g_nan_msk])

        p_oi = oi[:, 2:3, ...]
        test_oi = p_oi[:, :, self.test_crop[0], self.test_crop[1]]
        oi_mse =  F.mse_loss(test_oi[nan_msk], test_tgt[nan_msk])
        g_oi = kornia.filters.sobel(p_oi)[:, :, self.test_crop[0], self.test_crop[1]]
        g_oi_mse =  F.mse_loss(g_oi[g_nan_msk], g_tgt[g_nan_msk])
        # disc 
        # out_gt = self.dis(test_tgt)
        # out_pred = self.dis(test_pred)
        # loss_dis_pred = F.binary_cross_entropy(out_pred, torch.zeros_like(out_pred)) 
        # loss_gen_pred = F.binary_cross_entropy(out_pred, torch.ones_like(out_pred)) 
        # loss_dis_gt = F.binary_cross_entropy(out_gt, torch.ones_like(out_gt)) 

        losses = {
                'mse': loss,
                'mse_grad': g_loss,
                'rm_mse': rm_loss,
                'rm_grad_mse': rm_g_loss,
                # 'aug_mse': aug_loss,
                # 'aug_mse_grad': aug_g_loss,
                'oi_ratio': loss /oi_mse,
                'oi_grad_ratio': g_loss / g_oi_mse,
                # 'dis_pred': loss_dis_pred,
                # 'gen_pred': loss_gen_pred,
                # 'dis_gt': loss_dis_gt,
        }
        # print(losses)
        self.log_dict({phase + ' ' + k : v for k,v in losses.items()})
        return losses, pred, g_pred, tgt, g_tgt

lit_mod = DirectTraining()

dm = instantiate(cfg.datamodule)
# dm.setup()
# dl = dm.val_dataloader()
# batch = next(iter(dl))
# oi, mask, obs, gt, obs_gt =  lit_mod.transfer_batch_to_device(batch, lit_mod.device, 0)
# pred = lit_mod.fp_solve(obs, mask, n_iter=3)
# pred.isnan().sum()
# tgt = gt[:, 2, ...]
# nan_msk=~tgt.isnan()

# import matplotlib.pyplot as plt
# plt.imshow(pred.detach().cpu().numpy()[0])
# plt.imshow(tgt.detach().cpu().numpy()[0])
# plt.imshow(lit_mod.lr_weight.detach().cpu().numpy()[0])
# plt.imshow(lit_mod.mr_weight.detach().cpu().numpy()[0])
# plt.imshow(lit_mod.hr_weight.detach().cpu().numpy()[0])
# loss = F.mse_loss(pred[nan_msk], tgt[nan_msk])
# loss.backward()
# for n, p in lit_mod.named_parameters():
#     print(n, p.isnan().sum())
callbacks =[
    pl.callbacks.StochasticWeightAveraging(),
    # pl.callbacks.GradientAccumulationScheduler(scheduling={10: 2, 20: 4})
]
trainer = pl.Trainer(gpus=1, callbacks=callbacks, log_every_n_steps=1,progress_bar_refresh_rate=1)

# train_dl = dm.train_dataloader()
trainer.fit(lit_mod, datamodule=dm)

