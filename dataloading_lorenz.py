from logging import warn
import torch
import warnings
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import einops
from omegaconf import OmegaConf

config = OmegaConf.create(
    dict(
        data=dict(dim=40, npoint=10000, split=[500, 250, 250]),
        slice_cfg=dict(stride=25, time_window=200),
        dataloading=dict(
            train=dict(batch_size=32, shuffle=True),
            val=dict(batch_size=32, shuffle=False),
            test=dict(batch_size=32, shuffle=False),
        )
    )
)
from torchdiffeq import odeint

def lorenz96(t, X, forcing=8):
    # advection
    # adv = torch.roll(X, -1, dims=-1) * (torch.roll(X, 1, dims=-1) - torch.roll(X, -2, dims=-1))
    adv = torch.roll(X, 1, dims=-1) * (torch.roll(X, -1, dims=-1) - torch.roll(X, 2, dims=-1))
    return adv - X + forcing

def get_lorenz96_torch(d=40, n=10**3, dt=0.05, forcing=8., warmup=5., eps=1e-6):
    # warmup to get to attractor space
    _x0 = torch.full((d,), forcing)
    _x0[d//2] +=  0.01
    out_warmup = odeint(lorenz96, _x0, torch.arange(0, warmup + eps, dt))

    x0 = out_warmup[-1]
    t = torch.arange(0, n * dt, dt)
    out = odeint(lorenz96, x0, t)
    return out, t

def get_lorenz96_sim():
    def AnDA_Lorenz_96(S,t,F,J):
        """ Lorenz-96 dynamical model. """
        x = np.zeros(J);
        x[0] = (S[1]-S[J-2])*S[J-1]-S[0];
        x[1] = (S[2]-S[J-1])*S[0]-S[1];
        x[J-1] = (S[0]-S[J-3])*S[J-2]-S[J-1];
        for j in range(2,J-1):
            x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j];
        dS = x.T + F;
        return dS


    class GdCls:
        model = 'Lorenz_96'

        class parameters:
            F = 8
            J = 40
        dt_integration = 0.05 # integration time
        dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
        dt_obs = 4 # number of integration times between consecutive observations (for yo)
        var_obs = np.random.permutation(parameters.J)[0:20] # indices of the observed variables
        nb_loop_train = 10**4 # size of the catalog
        nb_loop_test = 10 # size of the true state and noisy observations
        sigma2_catalog = 0   # variance of the model error to generate the catalog   
        sigma2_obs = 2 # variance of the observation error to generate observations

    class time_series:
      values = 0.
      time   = 0.

    # 5 time steps (to be in the attractor space)
    GD = GdCls()    
    x0 = GD.parameters.F*np.ones(GD.parameters.J);
    x0[np.int(np.around(GD.parameters.J/2))] = x0[np.int(np.around(GD.parameters.J/2))] + 0.01;

    #S = odeint(AnDA_Lorenz_96,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));
    S = solve_ivp(
        fun=lambda t,y: AnDA_Lorenz_96(y,t,GD.parameters.F,GD.parameters.J),
        t_span=[0.,5+1e-6],
        y0=x0,
        first_step=GD.dt_integration,
        t_eval=np.arange(0,5+1e-6,GD.dt_integration),
        method='RK45'
    )
    x0 = S.y[:,-1]

    # generate true state (xt)
    #S = odeint(AnDA_Lorenz_96,x0,np.arange(0.01,GD.nb_loop_test+0.000001,GD.dt_integration),args=(GD.parameters.F,GD.parameters.J));       
    tt = np.arange(GD.dt_integration, GD.nb_loop_train*GD.dt_integration+1e-6, GD.dt_integration)
    S = solve_ivp(
        fun=lambda t,y: AnDA_Lorenz_96(y, t, GD.parameters.F, GD.parameters.J),
        t_span=[GD.dt_integration, GD.nb_loop_train*GD.dt_integration+1e-6],
        y0=x0,
        first_step=GD.dt_integration,
        t_eval=tt,
        method='RK45'
    )
    S = S.y.transpose()

    return S, tt


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


def fourdvar_cost(state, obs, alpha_obs=1., alpha_prior=1., dt=0.05):
    shape = einops.parse_shape(state, 'b t d')
    obs_cost =  torch.sum(((obs.where(~obs.isnan(), torch.zeros_like(obs)) - state + eps ) * (~obs.isnan()).float())**2)
    obs_cost = torch.nansum((state - obs )**2)
    _rec_state = odeint(lorenz96, state[:, 0, :], torch.arange(shape['t']) * dt)
    rec_state = einops.rearrange(_rec_state, 't b d -> b t d')
    dyn_cost = torch.sum((state - rec_state )**2)
    return alpha_obs * obs_cost + alpha_prior * dyn_cost

def weak_fourdvar_cost(state, obs, alpha_obs=1., alpha_prior=1., dt=0.05, eps=1e-10):
    obs_cost =  torch.sum(((obs.where(~obs.isnan(), torch.zeros_like(obs)) - state + eps ) * (~obs.isnan()).float())**2)
    rec_state = odeint(lorenz96, state, torch.arange(2) * dt)
    dyn_cost = torch.sum((state[:, 1:, :] - rec_state[1, :, :-1, :] + eps)**2)
    return alpha_obs * obs_cost + alpha_prior * dyn_cost

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
        # lr = 0.1 + 1 / (it // 3 + 1)
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
    
