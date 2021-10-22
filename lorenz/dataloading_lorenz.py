from logging import warn
import torch
import warnings
import numpy as np
import numpy.random
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import functools
import einops
from omegaconf import OmegaConf
from lorenz.lorenz import lorenz96, get_lorenz96_torch

config = OmegaConf.create(
    dict(
        data=dict(dim=40, split=[.50, .25, .25], npoint=4000, subsample=4, noise_sigma=1.4),
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

class ReducerDataset(Dataset):
    def __init__(self, fn, *dses):
        super().__init__()
        self.fn = fn
        self.dses =  dses
        assert len(dses) > 0, 'provide at least one dataset to reduce'
        assert all(len(_ds) == len(dses[0]) for _ds in self.dses), 'All ds must have the same length'

    def __len__(self):
        return len(self.dses[0])

    def __getitem__(self, item):
        return self.fn(ds[item] for ds in self.dses)

class LorenzDM(pl.LightningDataModule):
    def __init__(self, cfg=config):

        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self):
        # values, times = get_lorenz96_sim()
        values, times = get_lorenz96_torch(d=self.cfg.data.dim, n=self.cfg.data.npoint)
        values, times = values[::self.cfg.data.subsample, ...].numpy(), times[::self.cfg.data.subsample, ...].numpy()
        noise = np.random.randn(*values.shape).astype(np.float32) * self.cfg.data.noise_sigma
        train_data, val_data, test_data, _ = np.split(values, np.floor(np.cumsum(self.cfg.data.split) * len(times)).astype(np.int))
        train_times, val_times, test_times, _ = np.split(times, np.floor(np.cumsum(self.cfg.data.split) * len(times)).astype(np.int))
        train_noise, val_noise, test_noise, _ = np.split(noise, np.floor(np.cumsum(self.cfg.data.split) * len(times)).astype(np.int))

        self.train_ds, self.val_ds, self.test_ds = map(
            lambda data: ReducerDataset(
                lambda l: functools.reduce(lambda x, y: (*x, y), l),
                MaskerDataset(SlicerDataset(data[0], **self.cfg.slice_cfg)),
                SlicerDataset(data[1], **self.cfg.slice_cfg),
                SlicerDataset(data[2], **self.cfg.slice_cfg),
            ),
            [(train_data, train_noise, train_times), (val_data, val_noise, val_times), (test_data, test_noise, test_times)]
        )
            

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.cfg.dataloading.train)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.cfg.dataloading.val)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.cfg.dataloading.test)



if __name__ == '__main__':

    dm = LorenzDM()
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    item, mask, noise, times = batch
    type(item)
    print(times.shape)
    print(item.dtype)
    print(mask.dtype)
    obs = torch.where(mask.bool(), item + noise, torch.full_like(item, np.nan))

    plt.imshow(obs[0, ...].detach().numpy())
    plt.imshow(item[0, ...].detach().numpy())
    plt.imshow(mask[0, ...].detach().numpy())
    
