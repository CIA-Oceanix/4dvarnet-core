import einops
import numpy as np
import torch
from torchdiffeq import odeint
from lorenz.lorenz import lorenz96

def weak_fourdvar_cost(state, obs, alpha_obs=1., alpha_prior=1., dt=0.05, eps=1e-10):
    # obs_cost =  torch.nansum((state - obs + eps )**2)
    obs_cost =  torch.sum(((obs.where(~obs.isnan(), torch.zeros_like(obs)) - state + eps ) * (~obs.isnan()).float())**2)
    rec_state = odeint(lorenz96, state, torch.arange(2) * 4*dt)
    dyn_cost = torch.sum((state[:, 1:, :] - rec_state[1, :, :-1, :] + eps)**2)
    return alpha_obs * obs_cost + alpha_prior * dyn_cost

if __name__ == '__main__':
    import importlib
    from lorenz.dataloading_lorenz import LorenzDM
    import lorenz.dataloading_lorenz
    importlib.reload(lorenz.dataloading_lorenz)
    dm = lorenz.dataloading_lorenz.LorenzDM()
    dm.setup()
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    item, mask, noise, times = batch
    obs = torch.where(mask.bool(), item, torch.full_like(item, np.nan))
    rec_state = odeint(lorenz96, item[:, 0, :], torch.arange(200) * 0.05)
    rec_state.shape
    print(weak_fourdvar_cost(item, obs))

    lr = 0.1 # 0.01
    state_hat = torch.zeros_like(obs)
    state_hat.requires_grad_(True)
    for it in range(10):
        lr = 0.1 + 1 / (it // 3 + 1)
        var_cost = weak_fourdvar_cost(state_hat, obs)
        print(var_cost.item())
        grad = torch.autograd.grad((var_cost,), (state_hat,))[0]
        state_hat = state_hat - lr * grad
        state_hat.detach_().requires_grad_(True)

    plt.imshow(state_hat[0, ...].detach().numpy())
    plt.imshow(item[0, ...].detach().numpy())
    plt.imshow((state_hat - item)[0, ...].detach().numpy())
