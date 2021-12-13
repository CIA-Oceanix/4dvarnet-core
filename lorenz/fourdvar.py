import einops
import numpy as np
import torch
from torchdiffeq import odeint
from lorenz.lorenz import lorenz96

=======
def weak_fourdvar_cost(state, obs, alpha_obs=1., alpha_prior=1., dt=0.20, dt_int=0.05, eps=1e-10):
    # obs_cost =  torch.nansum((state - obs + eps )**2)
    obs_cost =  torch.mean(
            torch.sqrt(
                ((obs.where(~obs.isnan(), torch.zeros_like(obs)) - state + eps ) * (~obs.isnan()).float())**2
                + eps
            )
    )
    rec_state = odeint(lorenz96, state, torch.arange(0, dt + dt_int, dt_int, device=obs.device))
    # state_with_obs = state.where(obs.isnan(), obs)
    dyn_cost = torch.mean(torch.sqrt((state[:, 1:, :] - rec_state[-1, :, :-1, :] + eps)**2 + eps))
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
    device = 'cuda'
    item, mask, noise, times = batch
    item, mask, noise, times = item.to(device), mask.to(device), noise.to(device), times.to(device)  
    type(item)
    print(item.dtype)
    print(mask.dtype)
    obs = torch.where(mask.bool(), item, torch.full_like(item, np.nan))
    noisy_obs = torch.where(mask.bool(), item + noise, torch.full_like(item, np.nan))
    print(weak_fourdvar_cost(item, noisy_obs))

    current_obs = noisy_obs
    state_hat = torch.zeros_like(current_obs)
    state_hat.requires_grad_(True)
    for it in range(10):
        lr = 0.1 + 1 / (it // 3 + 1)
        var_cost = weak_fourdvar_cost(state_hat, obs)
        print(var_cost.item())
    lr = 1.
    alpha_obs = 10**2
    alpha_prior = 10**2

    dt = times.diff().mean()
    dt_int = dt / 4
    for it in range(1000):
        var_cost = weak_fourdvar_cost(state_hat, current_obs, dt=dt, dt_int=dt_int, alpha_obs=alpha_obs, alpha_prior=alpha_prior)
        print(it, var_cost.detach().cpu().numpy().item(), ((state_hat - item)**2).mean().cpu().detach().numpy().item())
        grad = torch.autograd.grad((var_cost,), (state_hat,))[0]
        state_hat = state_hat - lr * grad
        state_hat.detach_().requires_grad_(True)

    plt.imshow(obs[0, ...].detach().cpu().numpy().T)
    plt.imshow(state_hat[0, ...].detach().cpu().numpy().T)
    plt.imshow(item[0, ...].detach().cpu().numpy().T)
    plt.imshow((state_hat - item)[0, ...].detach().cpu().numpy().T)
    ((state_hat - item)**2).mean()
