import solver
import einops
from torch import nn
import models
import torch
import torch.nn.functional as F
import math
import calibration.lit_cal_model


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


class SirenState(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.state_init = nn.Parameter(torch.rand(hparams.shape_state))

        self.fwd_fn = SirenNet(dim_in=hparams.shape_state[0], dim_hidden=64, dim_out=hparams.shape_data[0], num_layers=2)

    def forward(self, state):
        x = einops.rearrange(state, 'bs t lat lon -> bs lat lon t')
        x = self.fwd_fn(x)
        x = einops.rearrange(x, 'bs lat lon t -> bs t lat lon')
        return x


class ImplicitSolver_Grad_4DVarNN(solver.Solver_Grad_4DVarNN):
    def __init__(self, state_mod, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_mod = state_mod

    def var_cost(self, x, yobs, mask):
        s = x
        x = self.state_mod(s)
        dy = self.model_H(x,yobs,mask)
        dx = x - self.phi_r(x)
        
        loss = self.model_VarCost( dx , dy )
        
        var_cost_grad = torch.autograd.grad(loss, s, create_graph=True)[0]
        return loss, var_cost_grad

def get_4dvarsiren_sst(hparams):
    return ImplicitSolver_Grad_4DVarNN(
                SirenState(hparams=hparams),
                models.Phi_r(hparams.shape_data[0], hparams.DimAE, hparams.dW, hparams.dW2, hparams.sS,
                    hparams.nbBlocks, hparams.dropout_phi_r, hparams.stochastic),
                calibration.lit_cal_model.Model_H_SST_with_noisy_Swot(hparams.shape_data[0], hparams.shape_obs[0], hparams=hparams),
                solver.model_GradUpdateLSTM(hparams.shape_state, hparams.UsePriodicBoundary,
                    hparams.dim_grad_solver, hparams.dropout),
                hparams.norm_obs, hparams.norm_prior, hparams.shape_data, hparams.n_grad)

class ImpLitCalMod(calibration.lit_cal_model.LitCalModel):
    MODELS = {
            '4dsiren_sst': get_4dvarsiren_sst,
    }
    def get_init_state(self, batch, state):
        if state is not None:
            return state
        bs = batch[0].size(0)
        st_init = einops.repeat(self.model.state_mod.state_init, 't lat lon -> bs t lat lon', bs=bs)
        return st_init
        
    def get_outputs(self, batch, state_out):
        return super().get_outputs(batch, self.model.state_mod(state_out))

    def loss_ae(self, state_out):
        x_out = self.model.state_mod(state_out)
        return torch.mean((self.model.phi_r(x_out) - x_out) ** 2)
    def configure_optimizers(self):
        

        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': opt,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience=50,),
            'monitor': 'val_loss'
        }
