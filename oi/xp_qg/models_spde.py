import einops
import xarray as xr
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import oi.xp_qg.solver_spde as NN_4DVar
from oi.sqrtm import *
from oi.spde import *
from oi.scipy_sparse_tools import *
from oi.xp_qg.metrics import nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, save_loss
from oi.xp_qg.metrics import save_NetCDF2 as save_netcdf
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
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv1   = torch.nn.Conv2d(self.n_t,dim_ae,(3,3),padding=1,bias=False)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv2   = torch.nn.Conv2d(dim_ae,2*dim_ae,(3,3),padding=1,bias=False)
        self.pool3 = torch.nn.MaxPool2d(4, 4)
        self.conv3   = torch.nn.Conv2d(2*dim_ae,7*self.n_t,(3,3),padding=1,bias=False)
        self.conv_t1 = torch.nn.ConvTranspose2d(7*self.n_t, 7*self.n_t, 5, stride=4)
        self.conv_t2 = torch.nn.ConvTranspose2d(7*self.n_t, 7*self.n_t, 2, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(7*self.n_t, 7*self.n_t, 2, stride=2)

    def forward(self, x):
        # input shape (b,t,y,x) --> output shape (b,7*t,y,x)
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv_t1(x))
        x = F.relu(self.conv_t2(x))
        x = self.conv_t3(x)
        return x

class encode_param_CNN_diff(torch.nn.Module):

    def __init__(self,shape_data,dim_ae):
        super(encode_param_CNN_diff, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

        # Conv2D layers
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv1   = torch.nn.Conv2d(self.n_t,dim_ae,(3,3),padding=1,bias=False)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv2   = torch.nn.Conv2d(dim_ae,2*dim_ae,(3,3),padding=1,bias=False)
        self.pool3 = torch.nn.MaxPool2d(4, 4)
        self.conv3   = torch.nn.Conv2d(2*dim_ae,3,(3,3),padding=1,bias=False)
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 3, 5, stride=4)
        self.conv_t2 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2)

    def forward(self, x):
        # input shape (b,t,y,x) --> output shape (b,7*t,y,x)
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv_t1(x))
        x = F.relu(self.conv_t2(x))
        x = self.conv_t3(x)
        return x

class encode_param_CNN_adv(torch.nn.Module):

    def __init__(self,shape_data,dim_ae):
        super(encode_param_CNN_adv, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

        # Conv2D layers
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv1   = torch.nn.Conv2d(self.n_t,dim_ae,(3,3),padding=1,bias=False)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv2   = torch.nn.Conv2d(dim_ae,2*dim_ae,(3,3),padding=1,bias=False)
        self.pool3 = torch.nn.MaxPool2d(4, 4)
        self.conv3   = torch.nn.Conv2d(2*dim_ae,3,(3,3),padding=1,bias=False)
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 2*self.n_t, 5, stride=4)
        self.conv_t2 = torch.nn.ConvTranspose2d(2*self.n_t, 2*self.n_t, 2, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(2*self.n_t, 2*self.n_t, 2, stride=2)

    def forward(self, x):
        # input shape (b,t,y,x) --> output shape (b,7*t,y,x)
        x = self.pool1(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv_t1(x))
        x = F.relu(self.conv_t2(x))
        x = self.conv_t3(x)
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
        # Compute tau value
        indices = torch.tensor(range(6*self.n_t,7*self.n_t)).to(device)
        tau = torch.index_select(params, 1, indices)
        tau = F.softplus(tau,10)

        # reshape 
        kappa = torch.reshape(kappa,(n_b,1,self.n_y*self.n_x,self.n_t))
        m = torch.reshape(m,(n_b,2,self.n_y*self.n_x,self.n_t))
        H = torch.reshape(H,(n_b,2,2,self.n_y*self.n_x,self.n_t))
        tau = torch.reshape(tau,(n_b,1,self.n_y*self.n_x,self.n_t))
        #H = torch.permute(H,(0,1,2,5,3,4))

        # kappa has shape (b,t,y,x)
        # m has shape (b,2,t,y,x)
        # H has shape (b,2,2,t,y,x)
        # tau has shape (b,t,y,x)
        return kappa.float(), m.float(), H.float(), tau.float()

class decode_param_CNN_diff(torch.nn.Module):

    def __init__(self, shape_data):
        super(decode_param_CNN_diff, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

    def forward(self, params):
        n_b = params.shape[0]

        # Compute H diffusion tensor
        gamma   = params[:,0,:,:]
        gamma   = torch.reshape(gamma,(n_b,self.n_x*self.n_y))
        gamma   = F.softplus(gamma,0.5)
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

class decode_param_CNN_adv(torch.nn.Module):

    def __init__(self, shape_data):
        super(decode_param_CNN_adv, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data

    def forward(self, params):
        n_b = params.shape[0]

        # Compute H diffusion tensor
        indices = torch.tensor(range(0,self.n_t)).to(device)
        u = torch.index_select(params, 1, indices)
        indices = torch.tensor(range(self.n_t,2*self.n_t)).to(device)
        v = torch.index_select(params, 1, indices)
        m = torch.stack([u,v],dim=1)

        # reshape
        m = torch.reshape(m,(n_b,2,self.n_x*self.n_y,self.n_t))

        # m has shape (n_b,2,Nx*Ny,Nt)
        return m.float()

class Prior_SPDE(torch.nn.Module):

    def __init__(self,shape_data, pow=1):
        super(Prior_SPDE, self).__init__()
        self.n_t, self.n_y, self.n_x = shape_data
        self.dx, self.dy, self.dt = [1,1,1]
        self.nb_nodes = self.n_x*self.n_y
        self.Id = sparse_eye(self.nb_nodes)
        self.Id2 = sparse_eye(self.nb_nodes*self.n_t)
        self.pow = pow
        self.alpha  = 2*self.pow
        self.nu = int(self.alpha -1)

    def create_Q(self,kappa,m,H,tau):

        n_b = m.shape[0]
        Q = list()
        tau = torch.transpose(torch.stack(self.nb_nodes*[torch.stack(self.n_t*[torch.tensor(tau)])]),0,1)
        for batch in range(n_b):
            inv_M = list()   # linear model evolution matrix
            B = list()
            for k in range(self.n_t-1):
                A = DiffOperator(self.n_x,self.n_y,self.dx,self.dy,m[batch,:,:,k+1],None,kappa)
                A=(1./2)*(A+A.t())
                if self.pow>1:
                    B_ = pow_diff_operator(A.to(device),self.pow,sparse=True)
                else:
                    B_ = A
                B.append(B_)
                inv_M.append(self.Id+self.dt*B_)

            # Build the global precision matrix
            Q0 = spspmm(sparse_eye(self.nb_nodes,1./((tau[0]**2)*self.dt)),
                        (self.dx*self.dy)*spspmm(B[0].t(),B[0]))
            # first line
            QR = spspmm(sparse_eye(self.nb_nodes,1./((tau[1]**2)*self.dt)),
                        (self.dx*self.dy)*self.Id) # case of right-hand side white noise
            QR = self.Id
            Q0 = spspmm(B[0].t(),B[0])
            Qg = torch.hstack([Q0+self.Id,
                               -1.*spspmm(inv_M[0],QR),
                               sparse_repeat(self.nb_nodes,1,self.n_t-2)])
            # loop
            for i in np.arange(1,self.n_t-1):
                QR1 = spspmm(sparse_eye(self.nb_nodes,1./((tau[i-1]**2)*self.dt)),
                        (self.dx*self.dy)*self.Id)
                QR2 = spspmm(sparse_eye(self.nb_nodes,1./((tau[i]**2)*self.dt)),
                        (self.dx*self.dy)*self.Id)
                QR1 = self.Id
                QR2 = self.Id
                Qg_ = torch.hstack([sparse_repeat(self.nb_nodes,1,i-1),
                                  -1.*spspmm(inv_M[i-1],QR1),
                                  spspmm(spspmm(inv_M[i-1],inv_M[i-1])+self.Id,QR1),
                                  -1.*spspmm(inv_M[i],QR2),
                                  sparse_repeat(self.nb_nodes,1,self.n_t-4-(i-2))])
                Qg  = torch.vstack([Qg,Qg_])
            # last line
            QR = spspmm(sparse_eye(self.nb_nodes,1./((tau[self.n_t-2]**2)*self.dt)),
                        (self.dx*self.dy)*self.Id)
            QR = self.Id
            Qg_ = torch.hstack([sparse_repeat(self.nb_nodes,1,self.n_t-2),
                              -1.*spspmm(inv_M[self.n_t-2],QR),
                              spspmm(spspmm(inv_M[self.n_t-2],inv_M[self.n_t-2]),QR)])
            Qg  = torch.vstack([Qg,Qg_])
            # enforce poistive definiteness
            Qg = (1./2)*(Qg+Qg.t()) + 5e-2*sparse_eye(self.n_t*self.nb_nodes)
            # add batch
            Q.append(Qg)
        Q = torch.stack(Q)
        return Q

    def square_root(self, mat):
        S = torch.transpose(cholesky_sparse.apply(mat.cpu()),0,1).to(device)
        return S

    def forward(self, kappa, m, H, tau, square_root=False):  
        Q = self.create_Q(kappa, m, H, tau)
        if square_root==True:
            op = list()
            for i in range(len(Q)):
                S = self.square_root(Q[i])
                print(S)
                #phi = torch.eye(self.nb_nodes*self.n_t).to(device)-S
                phi = (1.-S)
                #phi = sparse_eye(self.nb_nodes*self.n_t)-S
                op.append(phi)
            #op = torch.stack(op)
            return op
        else:
            return Q

class Phi_r(torch.nn.Module):

    def __init__(self, shape_data, pow=1, square_root=False):
        super().__init__()
        self.pow = pow
        self.encoder = encode_param_CNN_adv(shape_data,10)
        self.decoder = decode_param_CNN_adv(shape_data)
        self.operator_spde = Prior_SPDE(shape_data,pow=self.pow)
        self.square_root = square_root

    def forward(self, x):
        x_new = list()
        n_b, n_t, n_y, n_x = x.shape
        params = self.encoder(x)
        # state space -> parameter space
        kappa = .33
        tau = 1.
        H = None
        m = self.decoder(params)

        # SPDE prior (sparse Q)
        Q = self.operator_spde(kappa, m, H, tau, square_root=self.square_root)

        x_new = list()
        for i in range(n_b):
            Q_ = Q[i]
            if self.square_root==True:
                x_ = torch.matmul(Q_,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            else:
                x_ = torch.sparse.mm(Q_,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
        m = torch.reshape(m,(n_b,2,n_x,n_y,n_t))
        return x, [kappa,m,H,tau]

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

