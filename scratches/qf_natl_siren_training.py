import utils
import scipy.ndimage as ndi
import kornia
import numpy as np
import einops
import calibration.implicit_solver
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import matplotlib.pyplot as plt
import xarray as xr
import torchvision
import models


base_cfg = 'baseline/full_core'
fp = 'dgx_ifremer'
overrides = [
    f'file_paths={fp}',
    'datamodule.resize_factor=2',
    '+datamodule.dl_kwargs.drop_last=True',
    '+datamodule.dl_kwargs.shuffle=False',
]

class LitSiren(pl.LightningModule):
    def __init__(self, sh, nemb):
        super().__init__()
        self.save_hyperparameters()
        self.sh = sh
        self.embed_dim = 13
        embed_size = self.embed_dim * sh['t'] * sh['x'] * sh['y']
        self.embeddings = nn.Embedding(nemb, embed_size, max_norm=1., norm_type=np.inf)

        self.coords = nn.Parameter(torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, sh['t']),
            torch.linspace(-1, 1, sh['x']),
            torch.linspace(-1, 1, sh['y']),
        ), dim=-1), requires_grad=False)
        self.net = calibration.implicit_solver.SirenNet(
            dim_in=self.embed_dim +len(sh),
            dim_hidden=512,
            dim_out=1,
            num_layers=6
        )
        self.best_outs = np.inf, None
        self.g_loss_fn = lambda x1, x2: F.mse_loss(*map(kornia.filters.spatial_gradient, (x1, x2)))
        self.l_loss_fn = lambda x1, x2: F.mse_loss(*map(lambda t: kornia.filters.laplacian(t, 5), (x1, x2)))
        self.test_outs = None


    def forward(self, emb=None, batch_idx=None):
        if emb is None:
            assert batch_idx is not None
            emb = einops.rearrange(
                self.embeddings(torch.arange(batch_idx * 2, batch_idx *2 + 2, device=self.device)),
                'b (t x y d) -> b t x y d',
                d=self.embed_dim, **self.sh
            )
        batch_coords = einops.repeat(self.coords, '... -> b ...', b=emb.size(0))
        inp = torch.cat((batch_coords, emb), -1)
        out = self.net(inp).squeeze(-1)
        return out, emb

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.MultiStepLR(opt, [50, 100, 150], gamma=0.3)
        }
        # return torch.optim.AdamW(self.net.parameters(), lr=1e-3)

    def losses(self, out, gt):
        loss = F.mse_loss(out, gt)
        g_loss = self.g_loss_fn(out, gt)
        l_loss = self.l_loss_fn(out, gt)
        return loss, g_loss, l_loss

    def training_step(self, batch, batch_idx):
        ref, *_, gt = batch
        out, _ = self(None, batch_idx)
        # print(out.shape, gt.shape)
        loss, g_loss, l_loss = self.losses(out, gt)
        loss_ref, g_loss_ref, l_loss_ref = self.losses(ref, gt)
        if loss < self.best_outs[0]:
            self.best_outs = loss, out, gt, batch_idx, self.current_epoch
        self.log('imp', loss / loss_ref, prog_bar=True, on_step=False, on_epoch=True)
        self.log('gimp',  g_loss / g_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
        self.log('limp',  l_loss / l_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
        return loss  + g_loss # + l_loss

    def test_step(self, batch, batch_idx):
        ref, *_, gt = batch
        out, _ = self(None, batch_idx)
        return out.detach().cpu(), ref.detach().cpu(), gt.detach().cpu() 

    def test_epoch_end(self, outputs):
        out, ref, gt = map(torch.cat, zip(*outputs))
        self.test_outs = out, ref, gt

        l = F.mse_loss(out, gt).item()
        lref =  F.mse_loss(ref, gt).item()
        print(f'MSE {l:.2e}  {100 - l/lref*100:.1f} %')

        gl = self.g_loss_fn(out, gt).item()
        glref =  self.g_loss_fn(ref, gt).item()
        print(f'MSE grad {gl:.2e}  {100 - gl/glref*100:.1f} %')

        ll = self.l_loss_fn(out, gt).item()
        llref =  self.l_loss_fn(ref, gt).item()
        print(f'MSE lap {ll:.2e}  {100 - ll/llref*100:.1f} %')


def train_siren_natl():
    try:
        cfg = utils.get_cfg(base_cfg)
        dm = utils.get_dm(base_cfg, add_overrides=overrides)
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        oi, msk, obs, gt = batch
        sh = einops.parse_shape(gt, '... t x y')
        print(f'{len(dl.dataset)=}')
        print(f'{gt.shape=}')

        mod = LitSiren(sh=sh, nemb=len(dl.dataset))
        trainer = pl.Trainer(
            gpus=[7],
            max_epochs=200,
            logger=False,
            # accumulate_grad_batches={75:2,125:5,175:10}
        ) 
        trainer.fit(mod, dl)
        print(trainer.checkpoint_callback.best_model_path)
            


    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def visualize_trained_siren():
    try:
        ckpt = 'checkpoints/epoch=199-step=2800-v5.ckpt'
        mod = LitSiren.load_from_checkpoint(ckpt)
        dm = utils.get_dm(base_cfg, add_overrides=overrides)
        dl = dm.val_dataloader()
        trainer = pl.Trainer(gpus=[7], logger=False)
        trainer.test(mod,dataloaders=dl)

        out, ref, gt = mod.test_outs
        xrds = xr.Dataset({
            'pred': (('b', 't', 'x', 'y'), out),
            'ref': (('b', 't', 'x', 'y'), ref),
            'gt': (('b', 't', 'x', 'y'), gt),
        })

        1/0

        b, t= 5, 3
        sobel = lambda da: np.hypot(ndi.sobel(da, -1), ndi.sobel(da, -2))
        xrds.isel(b=b, t=t).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        xrds.isel(b=b, t=t).map(sobel).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        xrds.isel(b=b, t=t).map(ndi.laplace).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))

        # median = lambda da:  ndi.median_filter(da, 5)
        # mxrds = xrds.assign(pred=(xrds.pred.dims, median(xrds.pred)))
        # mxrds.isel(b=b, t=t).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        # mxrds.isel(b=b, t=t).map(sobel).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        # mxrds.isel(b=b, t=t).map(ndi.laplace).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
         
        gaussian = lambda da:  ndi.gaussian_filter(da, 0.75, truncate=2)
        gxrds = xrds.assign(pred=(xrds.pred.dims, gaussian(xrds.pred)))
        gxrds.isel(b=b, t=t).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        gxrds.isel(b=b, t=t).map(sobel).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        gxrds.isel(b=b, t=t).map(ndi.laplace).to_array().plot.pcolormesh('y', 'x', col='variable', col_wrap=3, figsize=(15, 5))
        
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

class LitDisc(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.classifier = torchvision.models.mobilenet_v3_small(num_classes=2)
        ckpt = 'checkpoints/epoch=199-step=2800-v5.ckpt'
        self.gen = LitSiren.load_from_checkpoint(ckpt)
        self.acc = torchmetrics.Accuracy(top_k=1)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    def forward(self, inp):

        # inp = kornia.filters.laplacian(inp, kernel_size=5)

        img = F.interpolate(inp, size=(240, 240))[:, 2:-2, 8:-8, 8:-8]
        img = self.normalize(img)
        return self.classifier(img)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.classifier.parameters(), lr=1e-4)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.MultiStepLR(opt, [50, 100, 150], gamma=0.3)
        }

    def losses(self, scores, tgt):
        if tgt == 1:
            tgt = einops.repeat(torch.tensor([1., 0.], device=self.device), '... -> b ...', b=scores.size(0))
            label = torch.tensor([0,0], device=self.device)
        elif tgt == 0:
            tgt = einops.repeat(torch.tensor([0., 1.], device=self.device), '... -> b ...', b=scores.size(0))
            label = torch.tensor([1,1], device=self.device)
        else:
            assert False

        l = F.cross_entropy(scores, tgt)
        acc = self.acc(scores, label)
        return l, acc

    def batch_outputs(self, batch, batch_idx):
        gen_item, embd = self.gen(None, batch_idx)
        oi, msk, obs, gt = batch
        true_score = self(gt) 
        lt, acc_true = self.losses(true_score, 1)
       
        false_score = self(gen_item) 
        lf, acc_false = self.losses(false_score, 0)
        return gen_item, lt, lf, acc_true, acc_false

    def training_step(self, batch, batch_idx):
        _, lt, lf, acc_true, acc_false = self.batch_outputs(batch, batch_idx)
        self.log('acc_true', acc_true, logger=False, on_epoch=True, prog_bar=True)
        self.log('acc_false', acc_false, logger=False, on_epoch=True, prog_bar=True)
        return lt + lf

def train_discriminator():
    try:
        cfg = utils.get_cfg(base_cfg)
        dm = utils.get_dm(base_cfg, add_overrides=overrides)
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        oi, msk, obs, gt = batch
        bs = gt.size(0)
        sh = einops.parse_shape(gt, '... t x y')
        

        img = F.interpolate(gt, size=(240, 240))[:, 2:-2, 8:-8, 8:-8]
        img.shape

        mod = LitDisc()
        mod = mod.to('cuda:7')
        mod.classifier(img.to(mod.device))
        

        trainer = pl.Trainer(
            gpus=[7],
            max_epochs=45,
            logger=False,
            accumulate_grad_batches={25:2,40:5,80:10}
        ) 
        trainer.fit(mod, dl)
        print(trainer.checkpoint_callback.best_model_path)

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()


def train_solver():
    try:
        pass
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

class LitGan(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.automatic_optimization = False

                ckpt_disc = 'checkpoints/epoch=46-step=476.ckpt'
                self.mod = LitDisc.load_from_checkpoint(ckpt_disc)
                self.siren = self.mod.gen.net
                self.disc = self.mod.classifier
                self.embeddings = self.mod.gen.embeddings

            def forward(self, **kwargs):
                return self.siren(**kwargs)

            def configure_optimizers(self):
                opt_siren = torch.optim.AdamW(self.siren.parameters(), lr=1e-3)
                opt_embd = torch.optim.AdamW(self.embeddings.parameters(), lr=1e-3)
                opt_disc = torch.optim.AdamW(self.disc.parameters(), lr=1e-3)
                sched_siren = torch.optim.lr_scheduler.MultiStepLR(opt_siren, [50, 100, 150], gamma=0.3)
                sched_embd = torch.optim.lr_scheduler.MultiStepLR(opt_embd, [50, 100, 150], gamma=0.3)
                sched_disc = torch.optim.lr_scheduler.MultiStepLR(opt_disc, [50, 100, 150], gamma=0.3)

                return (
                        {'optimizer':opt_siren, 'lr_scheduler':sched_siren, 'frequency': 10},
                        {'optimizer':opt_embd, 'lr_scheduler':sched_embd, 'frequency': 10},
                        {'optimizer':opt_disc, 'lr_scheduler':sched_disc, 'frequency': 10},
                )
                

            def training_step(self, batch, batch_idx):
                ref, msk, obs, gt = batch
                gen_item, lt, lf, acc_true, acc_false = self.mod.batch_outputs(batch, batch_idx)

                opt_siren, opt_embd, opt_disc = self.optimizers()

                loss, g_loss, l_loss = self.mod.gen.losses(out, gt)
                loss_ref, g_loss_ref, l_loss_ref = self.mod.gen.losses(ref, gt)

                self.mod.losses(self.mod(gen_item), 0)
                self.manual_backward(loss + g_loss)
                opt_siren.step()

                self.manual_backward(loss + g_loss)
                opt_siren.step()

                self.manual_backward(lt + lf)
                opt_disc.step()

                self.log('imp', loss / loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log('gimp',  g_loss / g_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log('limp',  l_loss / l_loss_ref, prog_bar=True, on_step=False, on_epoch=True)
                self.log('acc_true', acc_true, logger=False, on_epoch=True, prog_bar=True)
                self.log('acc_false', acc_false, logger=False, on_epoch=True, prog_bar=True)

def train_gan():
    try:
        

        dm = utils.get_dm(base_cfg, add_overrides=overrides)
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        oi, msk, obs, gt = batch
        bs = gt.size(0)
        sh = einops.parse_shape(gt, '... t x y')
        mod = LitGan()

        trainer = pl.Trainer(
            gpus=[7],
            max_epochs=200,
            logger=False,
            # accumulate_grad_batches={25:2,40:5,80:10}
        ) 
        trainer.fit(mod, dl)
        print(trainer.checkpoint_callback.best_model_path)

    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def fit_obs():
    try:
        ckpt_siren = 'checkpoints/epoch=199-step=2800-v5.ckpt'
        siren = LitSiren.load_from_checkpoint(ckpt_siren)

        ckpt_disc = 'checkpoints/epoch=46-step=476.ckpt'
        disc = LitDisc.load_from_checkpoint(ckpt_disc)
 

        dm = utils.get_dm(base_cfg, add_overrides=overrides)
        dl = dm.test_dataloader()
        siren=siren.to('cuda:7')
        disc=disc.to('cuda:7')
        batch = siren.transfer_batch_to_device(next(iter(dl)), siren.device, 0)
        ref, msk, obs, gt = batch
        lref =  F.mse_loss(ref, gt).item()
        bs = gt.size(0)
        sh = einops.parse_shape(gt, '... t x y')


        lr=1e-2
        embd = torch.rand(bs, *sh.values(), siren.embed_dim, device=siren.device, requires_grad=True)
        norm_grad = None
        for it in range(500):
            out, embd = siren(embd)
            loss = F.mse_loss(msk*out,msk*gt)
            tgt_true = einops.repeat(torch.tensor([1., 0.], device=disc.device), '... -> b ...', b=gt.size(0))
            dloss = F.cross_entropy(disc(out), tgt_true)
            wgrad = torch.autograd.grad(10000*loss + dloss, embd)[0]
            if norm_grad is None:
                norm_grad = (wgrad**2).mean().sqrt()
            embd = embd  - lr * wgrad/norm_grad
            if it %50 ==0:

                l = F.mse_loss(out, gt).item()
                print(f'MSE {l:.2e}  {100 - l/lref*100:.1f} %')
                print(f'obs {loss:.2e} ')
                print(f'disc {dloss:.2e} ')



        
    except Exception as e:
        print(traceback.format_exc()) 
    finally:
        return locals()

def main():
    try:
        # fn = train_siren_natl
        # fn = visualize_trained_siren
        # fn = train_discriminator
        fn = train_gan
        # fn = train_solver
        # fn = fit_obs

        locals().update(fn())
    except Exception as e:
        print('I am here')
        print(traceback.format_exc()) 
    finally:
        return locals()

if __name__ == '__main__':
        locals().update(main())
