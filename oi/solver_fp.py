import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from oi.models import Model_H
import xarray as xr
from oi.models_spde import Phi_r as Phi_r_spde
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_eye(size, val = torch.tensor(1.0)):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size).to(device)
    if len(val.size())==0:
        values = (val.expand(size)).to(device)
    else:
        values = val.to(device)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])).to(device)


def compute_WeightedLoss(x2,w):
    #  fix normalizing factor ( Sum w = 1 != w~ bool index)
    if len(list(w.size()))>0:
        x2_msk = (x2 * w[None, :, None, None])[:, w>0, ...]
    else:
        x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    if x2_num.sum() == 0:
        return torch.scalar_tensor(0., device=x2_num.device)
    # loss2 = x2_msk[x2_num].sum()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    return loss2

#Fixed Point solver for 4dVarnet
class FP_Solver(nn.Module):

    def __init__(self, phi_r, mod_H, shape_data, n_iter_fp, stochastic=False,
                 spde_dataset_path="/users/local/m19beauc/SPDE_dc/python/SPDE_diffusion_dataset.nc"):

        super(FP_Solver, self).__init__()
        self.phi_r = phi_r
        spde = xr.open_dataset(spde_dataset_path)
        self.phi_r_spde = Phi_r_spde(shape_data, diff_only=True, square_root=False,
                                     given_parameters=True, nc=spde)

        self.model_H = mod_H

        with torch.no_grad():
            self.n_fp = int(n_iter_fp)

    def forward(self, gt, x, yobs, mask, *internal_state):
        return self.solve(gt, x, yobs, mask, *internal_state)
        
    def solve(self, gt, x_0, obs, mask):
        x_k = x_0
        cmp_loss = torch.ones(x_k.shape[0],self.n_fp+1,2)
        # more than 1 iteration of n_grad doesn't do anything
        for k in range(self.n_fp):
            x_k, cmp_loss_ = self.solver_step(gt, x_k, obs, mask)
            for batch in range(len(cmp_loss)):
                cmp_loss[batch,k,:] = cmp_loss_[batch,:]
        x_k = self.phi_r(x_k)
        # last iteration
        _, cmp_loss_ = self.var_cost(gt, x_k, obs, mask)
        for batch in range(len(cmp_loss)):
            cmp_loss[batch,self.n_fp,:] = cmp_loss_[batch,:]
        return x_k, cmp_loss

    def solver_step(self, gt, x_k, obs, mask):
        _, cmp_loss = self.var_cost(gt, x_k, obs, mask)
        #Get the observed values in the measured area
        y_obs = obs * mask
        #Get the interpolation for the empty area
        unmeasured_mask = torch.logical_not(mask)
        x_proj = self.phi_r(x_k) * unmeasured_mask
        #combine measured and interpolated data
        x_k_plus_1 = x_proj + y_obs

        return x_k_plus_1.type(torch.FloatTensor).to(device), cmp_loss

    def var_cost(self, gt, x, yobs, mask):
        dy = self.model_H(x,yobs,mask)
        # Jb
        dx = x - self.phi_r(x)
        # varcost
        # loss = self.model_VarCost( dx , dy)
        n_b, n_t, n_x, n_y = dy.shape
        # Jo
        dy_new=list()
        for i in range(n_b):
            id_obs = torch.where(torch.flatten(mask[i])!=0.)[0]
            dyi = torch.index_select(torch.flatten(dy[i]), 0, id_obs).type(torch.FloatTensor).to(device)
            nb_obs = len(dyi)
            inv_R = 1e3*sparse_eye(nb_obs).type(torch.FloatTensor).to(device)
            iRdy = torch.sparse.mm(inv_R,torch.reshape(dyi,(nb_obs,1)))
            dyTiRdy = torch.matmul(torch.reshape(dyi,(1,nb_obs)),iRdy)
            dy_new.append(dyTiRdy[0,0])
        dy = torch.stack(dy_new)
        # Jb_OI (xtQx)
        Q = self.phi_r_spde.Q
        xtQx = list()
        for i in range(n_b):
            # prior regularization
            xtQ = torch.sparse.mm(Q,
                                  torch.reshape(torch.permute(x[i],(0,2,1)),(n_t*n_x*n_y,1))
                                )
            xtQx_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),(1,n_t*n_x*n_y)),
                                 xtQ
                                )
            xtQx.append(xtQx_[0,0])
        dx = torch.stack(xtQx)
        # loss_OI
        loss_OI = dy + dx
        loss = torch.mean(loss_OI)

        # return loss OI (loss) and MSE
        loss_mse = torch.tensor([compute_WeightedLoss((gt[i] - x[i]),torch.Tensor(np.ones(5)).to(device)) for i in range(len(gt))])
        loss_mse = loss_mse.to(device)

        loss = torch.mean(loss_OI)

        return loss, torch.hstack([torch.reshape(loss_mse,(n_b,1)), torch.reshape(loss_OI,(n_b,1))])


