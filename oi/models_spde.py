import einops
import xarray as xr
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import oi.solver_spde as NN_4DVar
from oi.sqrtm import *
from oi.spde import *
from oi.scipy_sparse_tools import *
from oi.metrics import nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps
from oi.metrics import save_NetCDF2 as save_netcdf
from oi.kriging_tools import *

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

def sparse_repeat(size,n1,n2):
    """
    Returns a sparse zero-filled tensor repeating 
    a 2D size*size sparse zero-filled tensor
    n1 times along dim1 and n2 times along n2
    """
    return torch.sparse.FloatTensor(size*n1,size*n2).to(device)

class encode_param_CNN(torch.nn.Module):

    def __init__(self,shape_data,dim_ae):
        super(encode_param_CNN, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

        # Conv2D layers
        self.conv1   = torch.nn.Conv2d(self.n_t,dim_ae,(3,3),padding=1,bias=False)
        self.conv2   = torch.nn.Conv2d(dim_ae,2*dim_ae,(3,3),padding=1,bias=False)
        self.conv3   = torch.nn.Conv2d(2*dim_ae,6*self.n_t,(3,3),padding=1,bias=False)

    def forward(self, x):
        # input shape (b,t,y,x) --> output shape (b,6*t,y,x)
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))
        x = self.conv3(x)
        return x

class decode_param_CNN(torch.nn.Module):

    def __init__(self, shape_data):
        super(decode_param_CNN, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

    def forward(self, params):
        n_b = params.shape[0]
        # Compute Kappa value
        indices = torch.tensor(range(0,self.n_t)).to(device)
        kappa = torch.index_select(params, 1, indices)
        kappa = F.softplus(kappa,10)
        # Compute m advection vector
        indices = torch.tensor(range(self.n_t,2*self.n_t)).to(device)
        m1 = torch.index_select(params, 1, indices)
        indices = torch.tensor(range(2*self.n_t,3*self.n_t)).to(device)
        m2 = torch.index_select(params, 1, indices)
        m = torch.stack([m1,m2],dim=1)
        # Compute H diffusion tensor
        indices = torch.tensor(range(3*self.n_t,4*self.n_t)).to(device)
        gamma = torch.index_select(params, 1, indices)
        gamma = F.softplus(gamma,10)
        gamma = torch.reshape(gamma.permute(0,2,3,1),(n_b,self.n_t*self.n_y*self.n_x))
        indices = torch.tensor(range(4*self.n_t,5*self.n_t)).to(device)
        vx = torch.index_select(params, 1, indices)
        vx = torch.reshape(vx.permute(0,2,3,1),(n_b,self.n_t*self.n_y*self.n_x))
        indices = torch.tensor(range(5*self.n_t,6*self.n_t)).to(device)
        vy = torch.index_select(params, 1, indices)
        vy = torch.reshape(vy.permute(0,2,3,1),(n_b,self.n_t*self.n_y*self.n_x))
        vxy     = torch.stack([vx,vy],dim=2)
        vxyT    = torch.permute(vxy,(0,2,1))
        H       = torch.einsum('ij,bk->bijk',torch.eye(2).to(device),gamma)+\
                  torch.einsum('bki,bjk->bijk',vxy,vxyT)
        # reshape H
        kappa = torch.reshape(kappa,(n_b,1,self.n_y*self.n_x,self.n_t))
        m = torch.reshape(m,(n_b,2,self.n_y*self.n_x,self.n_t))
        H = torch.reshape(H,(n_b,2,2,self.n_y*self.n_x,self.n_t))
        #H = torch.permute(H,(0,1,2,5,3,4))

        # kappa has shape (b,t,y,x)
        # m has shape (b,2,t,y,x)
        # H has shape (b,2,2,t,y,x)
        return kappa.float(), m.float(), H.float()

class decode_param_CNN_diff(torch.nn.Module):

    def __init__(self, shape_data):
        super(decode_param_CNN_diff, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

    def forward(self, params):
        n_b = params.shape[0]

        # Compute H diffusion tensor
        gamma   = params[:,0,:,:]
        gamma   = torch.reshape(gamma,(n_b,self.n_x*self.n_y))
        gamma   = F.relu(gamma)+.001
        vx      = params[:,1,:,:]
        vx      = torch.reshape(vx,(n_b,self.n_x*self.n_y))
        vy      = params[:,2,:,:]
        vy      = torch.reshape(vy,(n_b,self.n_x*self.n_y))
        vxy     = torch.stack([vx,vy],dim=2)
        vxyT    = torch.permute(vxy,(0,2,1))
        H       = torch.einsum('ij,bk->bijk',torch.eye(2).to(device),gamma)+\
                  torch.einsum('bki,bjk->bijk',vxy,vxyT)
        # reshape
        H = torch.reshape(H,(n_b,2,2,self.n_x*self.n_y))

        # H has shape (n_b,2,2,Nx*Ny)
        return H.float()

class Prior_SPDE(torch.nn.Module):

    def __init__(self,shape_data, diff_only=False):
        super(Prior_SPDE, self).__init__()
        self.n_t, self.n_y, self.n_x = shape_data
        self.dx, self.dy, self.dt = [1,1,1]
        self.nb_nodes = self.n_x*self.n_y
        self.Id = sparse_eye(self.nb_nodes)
        self.Id2 = sparse_eye(self.nb_nodes*self.n_t)
        self.pow = 1
        self.alpha   = 2*self.pow
        self.nu      = int(self.alpha -1)
        self.diff_only=diff_only

    def create_Q(self,kappa,m,H):

        n_b = H.shape[0]
        Q = list()
        for batch in range(n_b):
            M = list()   # linear model evolution matrix
            tau = torch.empty(self.n_t,self.nb_nodes).to(device)           # regularization term
            for k in range(self.n_t):
                if self.diff_only==False:
                    A = DiffOperator(self.n_x,self.n_y,self.dx,self.dy,
                                         m[batch,:,:,k],H[batch,:,:,:,k],kappa[batch,:,:,k])
                    #A = DiffOperator_Isotropic(self.n_x,self.n_y,self.dx,self.dy,kappa)
                else:
                    A = DiffOperator(self.n_x,self.n_y,self.dx,self.dy,None,H[batch],kappa)
                    A=(1./2)*(A+A.t())
                    #A = DiffOperator_Diffusion(self.n_x,self.n_y,self.dx,self.dy,H[batch],kappa)
                    #A = DiffOperator_Isotropic(self.n_x,self.n_y,self.dx,self.dy,kappa)
                if self.pow>1:
                    B = pow_diff_operator(A.to(device),self.pow,sparse=True)
                else:
                    B = A
                M.append(self.Id+self.dt*B)
                if torch.is_tensor(kappa):
                    tau[k]    = regularize_variance(self.nu, torch.flatten(kappa[batch,:,:,k]))
                else:
                    tau[k]    = regularize_variance(self.nu, kappa)
            # Build the global precision matrix
            #QR = spspmm(sparse_eye(self.nb_nodes,1./tau[k]),(self.dx*self.dy)*self.Id)   # case of right-hand side white noise
            QR = self.Id
            # first line
            Qg = torch.hstack([spspmm(spspmm(M[0],M[0]),QR),
                                  -1.*spspmm(M[1],QR),
                                  sparse_repeat(self.nb_nodes,1,self.n_t-2)])
            # loop
            for i in np.arange(1,self.n_t-1):
                Qg_ = torch.hstack([sparse_repeat(self.nb_nodes,1,i-1),
                                  -1.*spspmm(M[i-1],QR),
                                  spspmm(spspmm(M[i],M[i])+self.Id,QR),
                                  -1.*spspmm(M[i+1],QR),
                                  sparse_repeat(self.nb_nodes,1,self.n_t-4-(i-2))])
                Qg  = torch.vstack([Qg,Qg_])
            # last line
            Qg_ = torch.hstack([sparse_repeat(self.nb_nodes,1,self.n_t-2),
                              -1.*spspmm(M[self.n_t-2],QR),
                              spspmm(spspmm(M[self.n_t-1],M[self.n_t-1]),QR)])
            Qg  = torch.vstack([Qg,Qg_])
            Q.append(Qg)
        Q = torch.stack(Q)
        return Q

    def square_root(self, mat):
        S = sqrtm(mat)
        return S

    def forward(self, kappa, m, H, square_root=False):  
        Q = self.create_Q(kappa, m, H)
        if square_root==True:
            S = list()
            for i in range(len(Q)):
                S.append(self.square_root(Q[i].to_dense()))
            op = list()
            for i in range(len(Q)):
                op.append(self.Id2 + S[i])
            op = torch.stack(op)
            return op
        else:
            return Q

class Phi_r(torch.nn.Module):

    def __init__(self, shape_data, diff_only=False, square_root=False):
        super().__init__()
        self.diff_only = diff_only
        self.encoder = encode_param_CNN(shape_data,10)
        if self.diff_only==False:
            self.decoder = decode_param_CNN(shape_data)
        else:
            self.decoder = decode_param_CNN_diff(shape_data)
        self.operator_spde = Prior_SPDE(shape_data,diff_only=self.diff_only)
        self.square_root = square_root

    def forward(self, x, kappa, m, H):
        x_new = list()
        n_b, n_t, n_y, n_x = x.shape
        # state space -> parameter space
        params = self.encoder(x)
        if self.diff_only==False:
            kappa, m, H = self.decoder(params)
        else:
            kappa = .33
            m = None
            H = self.decoder(params)
        # SPDE prior (sparse Q)
        Q = self.operator_spde(kappa, m, H, square_root=self.square_root)
        x_new = list()
        for i in range(n_b):
            x_ = torch.sparse.mm(Q[i],
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
        if self.diff_only==False:
            kappa = torch.reshape(kappa,(n_b,1,n_x,n_y,n_t))
            m = torch.reshape(m,(n_b,2,n_x,n_y,n_t))
            H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
            return x, [kappa, m, H]
        else:
            H = torch.reshape(H,(n_b,2,2,n_x,n_y))
            return x, [H]

class Phi_r2(torch.nn.Module):
    def __init__(self, shape_data, diff_only=False, square_root=False):
        super().__init__()
        self.diff_only = diff_only
        self.operator_spde = Prior_SPDE(shape_data,diff_only=self.diff_only)
        # SPDE diffusion parameters
        nb_nodes = np.prod(shape_data[1:])
        H = torch.empty((2,2,nb_nodes)).to(device)
        nc = xr.open_dataset("/users/local/m19beauc/deep_OI/toy_data/diffusion_dataset.nc")
        H11 = torch.reshape(torch.transpose(torch.Tensor(nc.H11.values),0,1),(nb_nodes,)).to(device)
        H12 = torch.reshape(torch.transpose(torch.Tensor(nc.H12.values),0,1),(nb_nodes,)).to(device)
        H22 = torch.reshape(torch.transpose(torch.Tensor(nc.H22.values),0,1),(nb_nodes,)).to(device)
        H[0,0,:] = H22
        H[0,1,:] = H12
        H[1,0,:] = H12
        H[1,1,:] = H11
        self.H = H
        self.square_root = square_root
        self.op = self.operator_spde(.33,None,torch.unsqueeze(H,dim=0),
                                     square_root=self.square_root)[0]

    def forward(self, x, kappa, m, H):
        x_new = list()
        n_b, n_t, n_y, n_x = x.shape
        for i in range(n_b):
            x_ = torch.sparse.mm(self.op,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
        if self.diff_only==False:
            kappa = torch.reshape(kappa,(n_b,1,n_x,n_y,n_t))
            m = torch.reshape(m,(n_b,2,n_x,n_y,n_t))
            H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
            return x, [kappa, m, H]
        else:
            with torch.no_grad():
                H = torch.stack(n_b*[self.H])
                H = torch.reshape(H,(n_b,2,2,n_x,n_y))
            return x, [H]

class Phi_r3(torch.nn.Module):
    def __init__(self, shape_data, diff_only=False):
        super().__init__()
        self.diff_only = diff_only
        self.operator_spde = Prior_SPDE(shape_data,diff_only=self.diff_only)
        # SPDE diffusion parameters
        self.nb_nodes = np.prod(shape_data[1:])
        self.n_t, self.n_x, self.n_y = shape_data
        H = torch.empty((2,2,self.nb_nodes)).to(device)
        nc = xr.open_dataset("/users/local/m19beauc/deep_OI/toy_data/diffusion_dataset.nc")
        H11 = torch.reshape(torch.transpose(torch.Tensor(nc.H11.values),0,1),(self.nb_nodes,)).to(device)
        H12 = torch.reshape(torch.transpose(torch.Tensor(nc.H12.values),0,1),(self.nb_nodes,)).to(device)
        H22 = torch.reshape(torch.transpose(torch.Tensor(nc.H22.values),0,1),(self.nb_nodes,)).to(device)
        H[0,0,:] = H22
        H[0,1,:] = H12
        H[1,0,:] = H12
        H[1,1,:] = H11
        self.H = H
        self.op = self.operator_spde(.33,None,torch.unsqueeze(H,dim=0),
                                     square_root=False)[0]

    # true OI solver 
    def forward(self, x, obs, mask, kappa, m, H):
        x_new = list()
        n_b, n_t, n_y, n_x = x.shape
        for i in range(n_b):
            # define sparse observation operator
            row = []
            col = []
            idx = 0
            for k in range(self.n_t):
                idD = torch.where(torch.flatten(torch.transpose(mask[i,k,:,:],0,1))!=0.)[0]
                if len(idD)>0:
                    row.extend( (idx + np.arange(0,len(idD))).tolist() )
                    col.extend( ((k*self.nb_nodes)+idD).tolist() )
                    idx = idx + len(idD)
            val = np.ones(len(row))
            nb_obs = len(torch.where(torch.flatten(mask[i])!=0.)[0])
            opH = torch.sparse.FloatTensor(torch.LongTensor([row,col]),
                                         torch.FloatTensor(val),
                                         torch.Size([nb_obs,self.n_t*self.nb_nodes]))
            inv_R = 1e3*sparse_eye(nb_obs)
            Qxx = self.op + spspmm(spspmm(torch.transpose(opH,0,1),inv_R),opH)
            Qxy = -1.*spspmm(torch.transpose(opH,0,1),inv_R)
            # Cholesky inverse of Qg+Ht*inv(R)*H for k=0..Nt
            Lxx = None
            # START SPDE-based conditional simulations)
            xa = SPDE_spatiotemporal_kriging(Qxx, Lxx, Qxy, obs[i], mask[i], sparse=True, torch_sparse_solve=False)
            x_new.append(xa)
        x = torch.stack(x_new)
        x = torch.permute(torch.reshape(x,(n_b,self.n_t,self.n_x,self.n_y)),(0,1,3,2))
        if self.diff_only==False:
            kappa = torch.reshape(kappa,(n_b,1,n_x,n_y,n_t))
            m = torch.reshape(m,(n_b,2,n_x,n_y,n_t))
            H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
            return x, [kappa, m, H]
        else:
            with torch.no_grad():
                H = torch.stack(n_b*[self.H])
                H = torch.reshape(H,(n_b,2,2,n_x,n_y))
            return x, [H]

class Model_H(torch.nn.Module):
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_data])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.conv_gx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv_gx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.conv_gy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv_gy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        self.eps=10**-3
        # self.eps=0.

    def forward(self, im):

        if im.size(1) == 1:
            g_x = self.conv_gx(im)
            g_y = self.conv_gy(im)
            g = torch.sqrt(torch.pow(0.5 * g_x, 2) + torch.pow(0.5 * g_y, 2) + self.eps)
        else:

            for kk in range(0, im.size(1)):
                g_x = self.conv_gx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                g_y = self.conv_gy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                g_x = g_x.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                g_y = g_y.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                ng = torch.sqrt(torch.pow(0.5 * g_x, 2) + torch.pow(0.5 * g_y, 2)+ self.eps)

                if kk == 0:
                    g = ng.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                else:
                    g = torch.cat((g, ng.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
        return g

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)

