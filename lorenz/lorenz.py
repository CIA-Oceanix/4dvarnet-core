import torch
from torchdiffeq import odeint

def lorenz96(t, X, forcing=8):
    # advection
    adv = torch.roll(X, 1, dims=-1) * (torch.roll(X, -1, dims=-1) - torch.roll(X, 2, dims=-1))
    return adv - X + forcing

def get_lorenz96_torch(d=40, n=10**4, dt=0.05, forcing=8., warmup=5., eps=1e-6):
    # warmup to get to attractor space
    _x0 = torch.full((d,), forcing)
    _x0[d//2] +=  0.01
    out_warmup = odeint(lorenz96, _x0, torch.arange(0, warmup + eps, dt))

    x0 = out_warmup[-1]
    t = torch.arange(0, n * dt, dt)
    out = odeint(lorenz96, x0, t)
    return out, t
