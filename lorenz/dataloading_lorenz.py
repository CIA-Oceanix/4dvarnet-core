from logging import warn
import torch
import warnings
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import einops
from omegaconf import OmegaConf
from lorenz.lorenz import lorenz96, get_lorenz96_torch

config = OmegaConf.create(
    dict(
        data=dict(dim=40, npoint=10000, split=[500, 250, 250]),
        slice_cfg=dict(stride=25, time_window=200),
        dataloading=dict(
            train=dict(batch_size=1, shuffle=True),
            val=dict(batch_size=1, shuffle=False),
            test=dict(batch_size=1, shuffle=False),
        )
    )
)


class SlicerDataset(Dataset):
    def __init__(self, data, time_window, stride):
        super().__init__()
        self.data =  data
        print(self.data.shape)
        self.n_times = data.shape[0]
        self.stride = stride
        self.time_window = time_window
        if (self.n_times - self.time_window + 1) - len(self) > 0:
            warnings.warn(f" unused timesteps  ( strides: {stride} time_window: {time_window} shape: {self.data.shape} len ds: {len(self)}")
           
    def __len__(self):
        return (self.n_times - self.time_window + 1) // self.stride

    def __getitem__(self, item):
        return self.data[item * self.stride: item * self.stride + self.time_window]

class MaskerDataset(Dataset):
    def __init__(self, sl_ds, missing_rate=0.75):
        super().__init__()
        self.sl_ds = sl_ds
        self.missing_rate = missing_rate

    def __len__(self):
        return len(self.sl_ds)

    def __getitem__(self, item):
        item = self.sl_ds[item] 
        numel = item.size
        ind_nan = np.random.permutation(numel)
        idxs = np.unravel_index(ind_nan[:int(self.missing_rate * numel)], shape=item.shape)

        mask = np.ones_like(item)
        mask[idxs] = 0
        return item, mask

class LorenzDM(pl.LightningDataModule):
    def __init__(self, cfg=config):

        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self):
        # values, times = get_lorenz96_sim()
        values, times = get_lorenz96_torch()
        values, times = values.numpy(), times.numpy()
        train_data, val_data, test_data, _ = np.split(values, np.cumsum(self.cfg.data.split))
        self.train_ds = MaskerDataset(SlicerDataset(train_data, **self.cfg.slice_cfg))
        self.val_ds = MaskerDataset(SlicerDataset(val_data, **self.cfg.slice_cfg))
        self.test_ds = MaskerDataset(SlicerDataset(test_data, **self.cfg.slice_cfg))

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.cfg.dataloading.train)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.cfg.dataloading.val)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.cfg.dataloading.test)



if __name__ == '__main__':

    # torch_sim, t = get_lorenz96_torch(d=40, n=1000) 
    # torch_sim.shape
    # plt.imshow(torch_sim[-100:, :].numpy())

    # np_sim, t = get_lorenz96_sim() 
    # np_sim.shape
    # plt.imshow(np_sim[:100, :])
    # x0 = torch.from_numpy(np_sim[0])
    # plt.imshow(odeint(lorenz96, x0, torch.arange(0.05, 5.00001, 0.05), method='scipy_solver', options=dict(solver='RK45')).numpy())

    dm = LorenzDM()
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    item, mask = batch
    type(item)
    print(item.dtype)
    print(mask.dtype)
    obs = torch.where(mask.bool(), item, torch.full_like(item, np.nan))
    rec_state = odeint(lorenz96, item[:, 0, :], torch.arange(200) * 0.05)
    rec_state.shape
    print(weak_fourdvar_cost(item, obs))
    print(fourdvar_cost(item, obs))

    lr = 0.1 # 0.01
    state_hat = torch.zeros_like(obs)
    state_hat.requires_grad_(True)
    for it in range(1000):
        lr = 0.1 + 1 / (it // 3 + 1)
        var_cost = weak_fourdvar_cost(state_hat, obs)
        print(var_cost.item())
        grad = torch.autograd.grad((var_cost,), (state_hat,))[0]
        state_hat = state_hat - lr * grad
        state_hat.detach_().requires_grad_(True)

    plt.imshow(state_hat[0, ...].detach().numpy())
    plt.imshow(item[0, ...].detach().numpy())
    plt.imshow((state_hat - item)[0, ...].detach().numpy())
    # var_cost =  
    # first_item = item[0, ...]
    # first_mask = mask[0, ...]
    # first_mask.sum()
    # first_mask.numel()
    # plt.imshow(first_item)
    # obs = np.where(first_mask, first_item, np.nan)
    # plt.imshow(obs)
    
    # print(item.shape)
    
