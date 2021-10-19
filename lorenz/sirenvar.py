from torch import nn
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


def sirenvar_cost2D(sirens, obs):
    for siren, item_obs in zip(sirens, torch.split(obs, split_size_or_sections=1,dim=0)):
        obs_idx = (~item_obs.isnan()).nonzero()
        obs_cost = torch.sum((siren(obs_idx) - item_obs[obs_idx])**2)
       
        prior_idx = torch.ones_like(obs_idx).nonzero()
        t, d = prior_idx.split(dim=0, split_size_or_sections=1)
        out = siren(prior_idx)
        # TODO check jacobian
        dout = torch.autograd.functional.jacobian(siren, prior_idx, create_graph=True)[..., 0,:]
        # TODO compute cost as   
        """
        siren(t, d) -> value
        df/dt(t, d) = f(t, d-1) - f(t, d+2)) * f(t, d+1) - f(t, d) + F
        """
        dyn_cost = dy 

def sirenvar_cost1d(sirens, obs, alpha_obs=1., alpha_prior=1., eps=10**-6):
    shape = einops.parse_shape(obs, 'b t d')
    cost = 0.
    for siren, item_obs in zip(sirens, torch.split(obs, split_size_or_sections=1,dim=0)):
        b, t, d = (~item_obs.isnan()).nonzero(as_tuple=True)
        _fwd_idx = torch.arange(shape['t'], dtype=obs.dtype, device='cuda')[None,:, None]
        fwd_idx = (_fwd_idx - _fwd_idx.mean()) / _fwd_idx.std()
        out = siren(fwd_idx)
        obs_cost = torch.mean((out[b, t, d] - item_obs[b, t, d] + eps)**2)
        cost += alpha_obs * obs_cost
        if alpha_prior > 0.:
            prior_idx = fwd_idx
            _dout = torch.autograd.functional.jacobian(siren, prior_idx, create_graph=True)
            dout = _dout[0, torch.arange(shape['t']), :, 0, torch.arange(shape['t']),0][None, ...]
            dyn_cost = torch.mean((dout - lorenz96(torch.arange(2), out))**2)
            cost +=  alpha_prior * dyn_cost
    return cost 

if __name__ == '__main__':
    from lorenz.dataloading_lorenz import LorenzDM
    import numpy as np
    from torchdiffeq import odeint
    dm = LorenzDM()
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    item, mask = batch
    obs = torch.where(mask.bool(), item, torch.full_like(item, np.nan))

    lr = 0.1 # 0.01

    # model = SirenNet(dim_in=2, dim_out=1)
    def model_factory(init=None):
        model = SirenNet(dim_in=1, dim_hidden=256, num_layers=12, dim_out=40)
        if init is None: 
            return model
        
        model_params = OrderedDict({pn: p.clone() for pn, p in init.named_parameters()})
        model.load_state_dict(model_params) 
        return model

    # foo = lorenz96(torch.arange(2), item_obs)
    # TODO: init a siren
    # Clone it as many time as batch size (use batch size of one) 
    # TODO invert using item as obs and alpha prior as 0 (direct fitting)
    # TODO invert using default values

    model_init = model_factory()
    model = model_factory(init=model_init)
    device = 'cuda'
    sirens = nn.ModuleList([model]).to(device)
    current_obs = item.to(device)
    opt = torch.optim.SGD(sirens.parameters(), lr=0.001)
    for it in range(10000):
        lr = 0.1 + 1 / (it // 3 + 1)
        var_cost = sirenvar_cost1d(sirens, current_obs, alpha_prior=0.)
        print(var_cost.item())
        var_cost.backward(inputs=list(sirens.parameters()))
        opt.step()
        opt.zero_grad()
        # break

    _fwd_idx = torch.arange(shape['t'], dtype=current_obs.dtype, device='cuda')[None,:, None]
    fwd_idx = (_fwd_idx - _fwd_idx.mean()) / _fwd_idx.std()
    out = sirens[0](fwd_idx)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7))
    ax1.imshow(out[0].cpu().detach().numpy().T)
    ax2.imshow(item[0].cpu().detach().numpy().T)
    shape = einops.parse_shape(obs, 'b t d')
    cost = 0.
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
