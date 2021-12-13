from torch import nn
from tqdm import tqdm
from lorenz.lorenz import lorenz96
import einops
from collections import OrderedDict
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


def sirenvar_cost1d(sirens, obs, times, dt_prior=0.2, alpha_obs=1., alpha_prior=1., eps=10**-6):
    shape = einops.parse_shape(obs, 'b t d')
    cost = 0.
    outs = []
    for siren, item_obs in zip(sirens, torch.split(obs, split_size_or_sections=1,dim=0)):
        b, t, d = (~item_obs.isnan()).nonzero(as_tuple=True)
        fwd_idx = (times - times.mean()) / times.std()
        # fwd_idx = (_fwd_idx - _fwd_idx.mean()) / _fwd_idx.std()
        out = siren(fwd_idx)
        obs_cost = torch.sum((out[b, t, d] - item_obs[b, t, d] + eps)**2)
        cost += alpha_obs * obs_cost
        if alpha_prior > 0.:
            # prior_idx = torch.arange(fwd_idx.min(), fwd_idx.max() + dt_prior, dt_prior, device=obs.device)[None, :, None]
            prior_idx = fwd_idx
            out_ad, dout_ad = torch.autograd.functional.jvp(siren, prior_idx, v=torch.ones_like(prior_idx), create_graph=True, strict=True)
            # dout_di = out_ad.diff(dim=1) / 0.05
            dout_eq = lorenz96(None, out_ad)
            # dyn_cost = torch.mean(((dout_eq[0, 1:, :1] + dout_eq[0, :-1, :1]) / 2 - dout_di[0, :, :1])**2)
            # dyn_cost = torch.mean(((dout_ad[0, 1:, :1] + dout_ad[0, :-1, :1]) / 2 - dout_di[0, :, :1])**2)
            # dyn_cost = torch.mean((dout_ad[0, :-1, ...] - dout_di + eps)**2)
            # dyn_cost = torch.mean(torch.sqrt((dout_ad - dout_eq + eps)[:, 50:-50,:]**2))
            dyn_cost = torch.mean(torch.sqrt((dout_ad - dout_eq+ eps)**2 ))
            cost +=  alpha_prior * dyn_cost
        outs.append(out)
    return cost, torch.cat(outs, dim=0) 

if __name__ == '__main__':
    from lorenz.dataloading_lorenz import LorenzDM
    import numpy as np
    from torchdiffeq import odeint
    dm = LorenzDM()
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    item, mask, noise, times = batch
    obs = torch.where(mask.bool(), item, torch.full_like(item, np.nan))
    noisy_obs = torch.where(mask.bool(), item + noise, torch.full_like(item, np.nan))
    times.shape

    # model = SirenNet(dim_in=2, dim_out=1)
    def model_factory(init=None):
        model = SirenNet(dim_in=1, dim_hidden=256, num_layers=12, dim_out=40)
        if init is None: 
            return model
        
        model_params = OrderedDict({pn: p.clone() for pn, p in init.named_parameters()})
        model.load_state_dict(model_params) 
        return model


    model_init = model_factory()
    device = 'cuda'
    current_obs = obs[:1].to(device)
    # current_obs = (noise + item)[:1].to(device)
    # current_obs = (item)[:1].to(device)
    t = times[..., None].to(device)
    tgt = item[:1].to(device)
    sirens = nn.ModuleList([model_factory(init=model_init) for _ in range(current_obs.shape[0])]).to(device)
    opt = torch.optim.SGD(sirens.parameters(), lr=1e-3)
    for it in range(10000):
        var_cost, out = sirenvar_cost1d(sirens, current_obs, t, dt_prior=0.05, alpha_obs=10.0, alpha_prior=1.0)
        print(it, var_cost.item(),  ((out - tgt)**2).mean().item())
        var_cost.backward(inputs=list(sirens.parameters()))
        opt.step()
        opt.zero_grad()
        # break

    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10))
    img = ax1.imshow(out[0].cpu().detach().numpy().T)
    plt.colorbar(img, ax=ax1)
    img = ax2.imshow(tgt[0].cpu().detach().numpy().T)
    plt.colorbar(img, ax=ax2)
    img = ax3.imshow((out[0] - tgt[0]).cpu().detach().numpy().T)
    plt.colorbar(img, ax=ax3)


    for siren, item_obs in zip(sirens, torch.split(current_obs, split_size_or_sections=1,dim=0)):
        break

    fig, ax = plt.subplots()
    ax.imshow(current_obs[0].detach().cpu().numpy().T)

    prior_idx = fwd_idx
    eps=1e-6
    out_ad, dout_ad = torch.autograd.functional.jvp(sirens[0], prior_idx, v=torch.ones_like(prior_idx) , create_graph=True, strict=True)

    dout_di = out_ad.where(item_obs.isnan(), item_obs).diff(dim=1) / 0.05
    dout_eq = lorenz96(None, out_ad.where(item_obs.isnan(), item_obs))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10))
    img = ax1.imshow(dout_ad[0].cpu().detach().numpy().T)
    plt.colorbar(img, ax=ax1)
    img = ax2.imshow((dout_eq[0].cpu().detach() - item[0]).numpy().T)
    plt.colorbar(img, ax=ax2)
    img = ax3.imshow(((dout_eq - dout_ad[0]).cpu().detach() - item[0]).numpy().T)
    plt.colorbar(img, ax=ax3)
    # img = ax2.imshow(dout_di[0].cpu().detach().numpy().T)
    # plt.colorbar(img, ax=ax2)
    # dyn_cost = torch.mean((dout_ad[0, :-1, ...] - dout_di + eps)**2)
    dyn_cost = torch.mean((dout_ad - dout_eq + eps)**2)

    dyn_cost = torch.mean((dout_eq[0, :-1, ...] - dout_di[0])**2)
    img = plt.imshow((dout_eq[0, :-1, ...] - dout_di[0]).T)
    plt.colorbar(img)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot((dout_eq[0, 1:, :1] + dout_eq[0, :-1, :1]) / 2)
    ax.plot(dout_di[0, :, :1], '+')

    fig, ax = plt.subplots()
    ax.plot((dout_ad[0, 1:, :1] + dout_ad[0, :-1, :1]) / 2 - dout_di[0, :, :1])

    dyn_cost = torch.mean(((dout_eq[0, 1:, :1] + dout_eq[0, :-1, :1]) / 2 - dout_di[0, :, :1])**2)

    plt.plot(dout_ad[0, :, :2].detach().cpu().numpy())
    for siren, item_obs in zip(sirens, torch.split(current_obs, split_size_or_sections=1,dim=0)):
        break
    prior_idx = fwd_idx
    _dout = torch.autograd.functional.jacobian(siren, prior_idx, create_graph=True)
    _dout = torch.autograd.functional.vjp(siren, prior_idx, create_graph=True)
    _dout = torch.autograd.functional.jvp(siren, prior_idx, create_graph=True)
    dout = _dout[0, torch.arange(shape['t']), :, 0, torch.arange(shape['t']),0][None, ...]
    dyn_cost = torch.mean((dout - lorenz96(torch.arange(2), out))**2)
        b, t, d = (~item_obs.isnan()).nonzero(as_tuple=True)
        fwd_idx = torch.arange(shape['t'], dtype=obs.dtype, device='cuda')[None,:, None]
        out = siren(fwd_idx)
        obs_cost = torch.sum((out[b, t, d] - item_obs[b, t, d])**2)
        prior_idx = fwd_idx
        _dout = torch.autograd.functional.jacobian(siren, prior_idx, create_graph=True)
        dout = _dout[0, torch.arange(shape['t']), :, 0, torch.arange(shape['t']),0][None, ...]
        torch.autograd.
