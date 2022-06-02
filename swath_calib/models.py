import einops
from einops import rearrange
import kornia
from einops.layers.torch import Rearrange, Reduce
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_pad(h, w, kh, kw, s):
    # The total padding applied along the height and width is computed as:
    if (h % s[0] == 0):
        pad_along_height = max(kh - s[0], 0)
    else:
        pad_along_height = max(kh - (h % s[0]), 0)
    if w % s[1] == 0:
        pad_along_width = max(kw - s[1], 0)
    else:
        pad_along_width = max(kw - (w % s[1]), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return {'left': pad_left, 'right': pad_right, 'top': pad_top, 'bottom': pad_bottom}

    # %% models


class ConvSamePad(torch.nn.Module):
    def __init__(self, apply_per_side=True, *args, **kwargs):
        super().__init__()
        self.apply_per_side = apply_per_side
        self.conv = torch.nn.Conv2d(*args, **kwargs)

    def _forward(self, inp):
        inp_shape = einops.parse_shape(inp, 'bs inc w h')
        kernel_shape = einops.parse_shape(self.conv.weight, 'inc outc w h')
        same_pad = get_same_pad(
            inp_shape['h'],
            inp_shape['w'],
            kernel_shape['h'],
            kernel_shape['w'],
            self.conv.stride,
        )
        return self.conv(F.pad(inp, (same_pad['top'], same_pad['bottom'], same_pad['left'], same_pad['right']), mode='reflect'))

    def forward(self, x):
        if self.apply_per_side:
            sizes = einops.parse_shape(x, 'b c time nc')
            side_limit = sizes['nc'] // 2
            return einops.rearrange(
                [
                    self._forward(x[..., :side_limit]),
                    self._forward(x[..., side_limit:]),
                ],
                'sides b c time hnc -> b c time (hnc sides)'
            )

        return self._forward(x)

def build_net(
    in_channels,
    out_channels,
    nhidden = 128,
    depth = 3,
    kernel_size = 3,
    num_repeat = 1,
    residual = True,
    norm_type = 'none',
    act_type = 'relu',
    mix = False,
    mix_residual = False,
    mix_act_type = 'none',
    mix_norm_type = 'none',
):

    def norm(norm_type='bn', nh=nhidden):
        if norm_type=='none':
            return nn.Identity()
        elif norm_type=='bn':
            return nn.BatchNorm2d(num_features=nh)
        elif norm_type=='in':
            return nn.InstanceNorm2d(num_features=nh)
        elif norm_type=='lrn':
            return nn.LocalResponseNorm(size=5)
        else:
            assert False, 'Should not be here'

    def act(act_type='relu'):
        if act_type=='none':
            return nn.Identity()
        elif act_type=='relu':
            return nn.ReLU()
        elif act_type=='silu':
            return nn.SiLU()
        elif act_type=='gelu':
            return nn.GELU()
        else: 
            assert False, 'Should not be here'


    class ResidualBlock(nn.Module):
        def __init__(self, net,  res=True):
            super().__init__()
            self.net = net
            self.res = res

        def forward(self, x):
          if not self.res:
              return self.net(x)
          return x + self.net(x)

    def mixer(b=True, res=False):
         return ResidualBlock(
             nn.Sequential(
               Rearrange('b c t nC -> b nC t c'),
               ConvSamePad(in_channels=52, out_channels=52, kernel_size=1, apply_per_side=False),
               norm(mix_norm_type, nh=52),
               act(act_type=mix_act_type),
               Rearrange('b nC t c -> b c t nC'),
            ),
         res=res) if b else nn.Identity()

    
    inner_net = nn.Sequential(
        *[ nn.Sequential(
            ResidualBlock(
                nn.Sequential(
                    ConvSamePad(in_channels=nhidden,out_channels=nhidden, kernel_size=kernel_size),
                    norm(norm_type),
                    act(act_type=act_type),
                ), res=residual),
                mixer(mix, res=mix_residual),
        )
        for _ in range(depth) ],
    )
    net = nn.Sequential(
            ConvSamePad(in_channels=in_channels,out_channels=nhidden, kernel_size=1),
            norm(norm_type=norm_type),
            act(act_type=act_type),
            nn.Sequential(
                *[inner_net for _ in range(num_repeat)]
            ),
            ConvSamePad(in_channels=nhidden, out_channels=out_channels, kernel_size=1),
    )
    return net

class LitDirectCNN(pl.LightningModule):
            def __init__(
                    self,
                    net,
                    gt_var_stats,
                    lr_init=1e-3,
                    wd=1e-4,
                    loss_w={'tot':(.1, .1, .1), 'rec':(1., 1., 1.,)},
                    loss_budget_gt_vars=100,
                    f_th=0.01,
                    sig=1,
                    ff=False,
                ):
                super().__init__()
                self.net = net
                self.use_ff = ff
                self.ff = FourierFilter(f_th, sig)
                self.lr_init = lr_init
                self.wd = wd
                self.loss_budget_gt_vars = loss_budget_gt_vars
                self.loss_w = loss_w
                self.gt_means = nn.Parameter(torch.from_numpy(gt_var_stats[0])[None, :, None, None], requires_grad=False)
                self.gt_stds = nn.Parameter(torch.from_numpy(gt_var_stats[1])[None, :, None, None], requires_grad=False)
                self.save_hyperparameters()

            def forward(self, batch):
                x, *_ = batch 
                out = self.net(x)
                if self.use_ff:
                    out = self.ff(out)
                return out

            def loss(self, t1, t2):
                rmse = ((t1 -t2)**2).mean().sqrt()

                def sob(t):
                    if len(t.shape) == 4:
                        # return kornia.filters.sobel(rearrange(t, 'b d1 d2 c -> b c d1 d2'))
                        return kornia.filters.sobel(t)
                    elif len(t.shape) == 3:
                        return kornia.filters.sobel(rearrange(t, 'b d1 d2 -> b () d1 d2'))
                    else:
                        assert False, 'Should not be here'

                def lap(t):
                    if len(t.shape) == 4:
                        # return kornia.filters.laplacian(rearrange(t, 'b d1 d2 c -> b c d1 d2'), kernel_size=3)
                        return kornia.filters.laplacian(t, kernel_size=3)
                    elif len(t.shape) == 3:
                        return kornia.filters.laplacian(rearrange(t, 'b d1 d2 -> b () d1 d2'), kernel_size=3)
                    else:
                        assert False, 'Should not be here'

                rmse_grad = ((sob(t1) - sob(t2))**2).mean().sqrt()
                rmse_lap = ((lap(t1) - lap(t2))**2).mean().sqrt()

                return rmse, rmse_grad, rmse_lap

            def process_batch(self, batch, phase='val', ff=False):
                _, y, raw_gt, raw_ref = batch 
                out = self.forward(batch)
                losses = {}
                losses['err_tot'], losses['g_err_tot'], losses['l_err_tot'] = self.loss(out, y)

                rec_out = (out * self.gt_stds + self.gt_means).sum(dim=1)
                losses['err_rec'], losses['g_err_rec'], losses['l_err_rec'] = self.loss(rec_out, raw_gt)

                for ln, l in losses.items():
                    self.log(f'{phase}_{ln}', l)

                loss_ref, g_loss_ref, l_loss_ref= self.loss(raw_ref, raw_gt)
                self.log(f'{phase}_imp_mse', losses['err_rec'] / loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_grad_mse', losses['g_err_rec'] / g_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f'{phase}_imp_lap_mse', losses['l_err_rec'] / l_loss_ref, prog_bar=True, on_step=False, on_epoch=True)

                loss = (
                    self.loss_w['tot'][0] * losses['err_tot']
                    + self.loss_w['tot'][1] * losses['g_err_tot']
                    + self.loss_w['tot'][2] * losses['l_err_tot']
                    + self.loss_w['rec'][0] * losses['err_rec']
                    + self.loss_w['rec'][1] * losses['g_err_rec']
                    + self.loss_w['rec'][2] * losses['l_err_rec']
                )
                self.log(f'{phase}_loss', loss, prog_bar=False)
                return loss
                
            def training_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='train')

            def validation_step(self, batch, batch_idx):
                return self.process_batch(batch, phase='val')

            def predict_step(self, batch, batch_idx):
                out = self.forward(batch)
                # print(f'{out.isnan().sum()=}')

                rec_out = (out * self.gt_stds + self.gt_means).sum(dim=1)
                return rec_out.cpu().numpy()


            def configure_optimizers(self):
                # opt = torch.optim.AdamW(self.parameters(), lr=self.lr_init, weight_decay=self.wd)
                opt = torch.optim.Adam(
                        [{'params': self.parameters(), 'initial_lr': self.lr_init}],
                        lr=self.lr_init, weight_decay=self.wd)
                # opt = torch.optim.SGD(self.parameters(), lr=self.lr_init)
                return {
                    'optimizer': opt,
                    'lr_scheduler':
                    # torch.optim.lr_scheduler.ReduceLROnPlateau(
                    #     opt, verbose=True, factor=0.5, min_lr=1e-6, cooldown=5, patience=5,
                    # ),
                    # torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=5e-5, T_max=20),
                    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,
                    #     eta_min=5e-5, T_0=15, T_mult=2, last_epoch=-1),
                    torch.optim.lr_scheduler.CyclicLR(
                        opt, base_lr=5e-5, max_lr=5e-3,  step_size_up=15, step_size_down=25, cycle_momentum=False, mode='triangular2'),
                    'monitor': 'val_loss'
                }

class FourierFilter(torch.nn.Module):
    def __init__(self, f_th, sig):
        super().__init__()
        self.f_th = f_th
        self.sig = sig

    def forward(self, x):
        fft_out = torch.fft.rfft(x, dim=2)
        freqs = torch.fft.rfftfreq(x.size(2), 2).to(x.device)
        out_hf = torch.fft.irfft(fft_out.where(freqs[None, None,:,None] > self.f_th, torch.zeros_like(fft_out)), dim=2).real
        out_lf = torch.fft.irfft(fft_out.where(freqs[None, None,:,None] < self.f_th, torch.zeros_like(fft_out)), dim=2).real
        # ff_out = out_lf + out_hf
        ff_out = out_lf + kornia.filters.gaussian_blur2d(out_hf, kernel_size=(int(2*(2*self.sig))+1, 1), sigma=(self.sig, 0.001))
        # ff_out = out_lf + kornia.filters.median_blur(out_hf, kernel_size=(31, 1))
        diff = x.size(2) - ff_out.size(2)
        p_ff_out = F.pad(ff_out, [0, 0, 0, diff], mode='reflect')
        return p_ff_out
