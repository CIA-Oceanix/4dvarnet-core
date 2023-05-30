import einops
import xarray as xr
import numpy as np
#import functorch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import oi.xp_gp.solver_spde as NN_4DVar
from oi.sqrtm import *
from oi.spde import *
from oi.scipy_sparse_tools import *
from oi.xp_gp.metrics import nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, save_loss
from oi.xp_gp.metrics import save_NetCDF2 as save_netcdf
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

class Prior_SPDE(torch.nn.Module):

    def __init__(self,shape_data, pow=1, spde_type="diff"):

        super(Prior_SPDE, self).__init__()
        self.n_t, self.n_y, self.n_x = shape_data
        self.dx, self.dy, self.dt = [1,1,1]
        self.nb_nodes = self.n_x*self.n_y
        self.Id = sparse_eye(self.nb_nodes)
        self.Id2 = sparse_eye(self.nb_nodes*self.n_t)
        self.pow = pow
        self.alpha  = 2*self.pow
        self.nu = int(self.alpha -1)
        self.spde_type = spde_type

    def stack_indices(self,mat,ir,ic,row,col,val):

        row_ = mat.coalesce().indices()[0]
        col_ = mat.coalesce().indices()[1]
        val_ = mat.coalesce().values()
        row = torch.cat((row,row_+ir))
        col = torch.cat((col,col_+ic))
        val = torch.cat((val,val_))
        return row, col, val

    def create_Q(self,
                 kappa,
                 m,
                 H,
                 tau,
                 colored_noise=False,
                 store_block_diag=False):

        n_b = H.shape[0]

        # set regularization variance term to appropriate size
        if torch.is_tensor(tau):
            torch.full(tau.size(),1.)
            tau = torch.squeeze(tau,dim=1)
        else:
            tau = torch.stack(n_b*\
                              [torch.stack(self.nb_nodes*\
                                          [torch.stack(self.n_t*[torch.tensor(tau)])]\
                                          )]\
                              ).to(device)
 
        # initialize Qs (noise precision matrix)
        if colored_noise==False:
            Qs = sparse_eye(self.nb_nodes) 
        else:
            Qs = sparse_eye(self.nb_nodes) 
        
        # initialize outputs    
        Q = list()
        if store_block_diag==True:
            block_diag=list()
                 
        for batch in range(n_b):
            # Build model evolution and noise effects operator
            inv_M = list() # linear model evolution matrix (inverse)
            inv_S = list() # T*Tt with T the noise effect matrix (inverse) 
            for k in range(self.n_t):
                if self.spde_type=="adv_diff":
                    A = DiffOperator(self.n_x,self.n_y,self.dx,self.dy,
                                         m[batch,:,:,k],
                                         H[batch,:,:,:,k],
                                         kappa[batch,:,:,k])
                elif self.spde_type=="diff":
                    A = DiffOperator(self.n_x,self.n_y,self.dx,self.dy,
                                     None,
                                     H[batch],
                                     kappa)
                elif self.spde_type=="adv":
                    A = DiffOperator(self.n_x,self.n_y,self.dx,self.dy,
                                     m[batch],
                                     None,
                                     kappa)
                else:
                    A = DiffOperator_Isotropic(self.n_x,self.n_y,self.dx,self.dy,
                                     kappa)
                if self.pow>1:
                    B = pow_diff_operator(A.to(device),self.pow,sparse=True)
                else:
                    B = A

                # initialize Q0 = P0^-1 = cov(x0)
                if k==0:
                    Q0 = (self.dx*self.dy)*spspmm(B.t(),B)
                    Q0 = (1./2)*(Q0+Q0.t()) + 5e-2*sparse_eye(self.nb_nodes)
                else:
                    inverse_M = self.Id+self.dt*B
                    inverse_S = spspmm(spspmm(inverse_M.t(),Qs),inverse_M)
                    #inverse_S = (1./2)*(inverse_S+inverse_S.t()) + 5e-2*sparse_eye(self.nb_nodes)
                    inv_M.append(inverse_M)
                    inv_S.append(inverse_S)
 
            if store_block_diag==True:
                l = list(inv_S)
                l.insert(0,Q0)
                block_diag.append(l)
                    
            # Build the global precision matrix
            row = torch.tensor([]).to(device)
            col = torch.tensor([]).to(device) 
            val = torch.tensor([]).to(device)    
             
            # first line 
            inv_tau = sparse_eye(self.nb_nodes,1./(tau[batch,:,1]*np.sqrt(self.dt))) 
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qs),inv_tau) 
            row, col, val = self.stack_indices(Q0+self.Id, 
                                          0,0, 
                                          row,col,val) 
            row, col, val = self.stack_indices(-1.*spspmm(Qs_tilde,inv_M[0]), 
                                          0,self.nb_nodes, 
                                          row,col,val) 
            # loop 
            for i in np.arange(1,self.n_t-1): 
                inv_tau_1 = sparse_eye(self.nb_nodes,1./(tau[batch,:,i]*np.sqrt(self.dt))) 
                Qs_tilde1 = spspmm(spspmm(inv_tau_1.t(),Qs),inv_tau_1) 
                inv_tau_2 = sparse_eye(self.nb_nodes,1./(tau[batch,:,i+1]*np.sqrt(self.dt))) 
                Qs_tilde2 = spspmm(spspmm(inv_tau_2.t(),Qs),inv_tau_2) 
                row, col, val = self.stack_indices(-1.*spspmm(inv_M[i-1].t(),Qs_tilde1), 
                                              i*self.nb_nodes,(i-1)*self.nb_nodes, 
                                              row,col,val) 
                row, col, val = self.stack_indices(spspmm(spspmm(inv_M[i-1].t(),Qs_tilde1),inv_M[i-1])+Qs_tilde1, 
                                              i*self.nb_nodes,i*self.nb_nodes, 
                                              row,col,val) 
                row, col, val = self.stack_indices(-1.*spspmm(Qs_tilde2,inv_M[i]), 
                                              i*self.nb_nodes,(i+1)*self.nb_nodes, 
                                              row,col,val) 
            # last line 
            inv_tau = sparse_eye(self.nb_nodes,1./(tau[batch,:,self.n_t-1]*np.sqrt(self.dt))) 
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qs),inv_tau) 
            row, col, val = self.stack_indices(-1.*spspmm(inv_M[self.n_t-2].t(),Qs_tilde), 
                                          (self.n_t-1)*self.nb_nodes,(self.n_t-2)*self.nb_nodes, 
                                          row,col,val) 
            row, col, val = self.stack_indices(spspmm(spspmm(inv_M[self.n_t-2].t(),Qs_tilde),inv_M[self.n_t-2]), 
                                          (self.n_t-1)*self.nb_nodes,(self.n_t-1)*self.nb_nodes,row,col,val) 
             
            # create sparse tensor 
            index = torch.stack([row, col], dim=0) 
            value = val 
            Qg = torch.sparse.FloatTensor(index.long(), value,  
                                           torch.Size([self.n_t*self.nb_nodes, 
                                                       self.n_t*self.nb_nodes])).to(device) 
            # enforce positive definiteness
            #Qg = (1./2)*(Qg+Qg.t()) + 5e-2*sparse_eye(self.n_t*self.nb_nodes)
            # add batch
            Q.append(Qg)

        #Q = torch.stack(Q) # pb differentiating tensors

        if store_block_diag==True:
            # Q has size #batch*(nt*nbnodes)*(nt*nbnodes)
            # block_diag is list of size #batch
            return Q, block_diag
        else:
            return Q

    def square_root(self, mat):
        S = torch.transpose(cholesky_sparse.apply(mat.cpu()),0,1).to(device)
        return S

    def forward(self,
                kappa,
                m, 
                H, 
                tau, 
                square_root=False,
                store_block_diag=False):  

        if store_block_diag==False:
            Q = self.create_Q(kappa, m, H, tau,
                          store_block_diag=store_block_diag)
        else:
            Q, block_diag = self.create_Q(kappa, m, H, tau,
                            store_block_diag=store_block_diag)
        
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
            if store_block_diag==False:
                return op
        else:
            if store_block_diag==False:
                return Q
            else:
                return Q, block_diag

class Phi_r_no_estim(torch.nn.Module):

    def __init__(self, shape_data, pow=1, spde_type="diff",
                 square_root=False,
                 given_parameters=False, nc="spde_path.nc"):
        super().__init__()
        self.shape_data = shape_data
        self.spde_type = spde_type
        self.pow = pow
        self.operator_spde = Prior_SPDE(self.shape_data,pow=self.pow,spde_type=self.spde_type)
        self.square_root = square_root
        self.given_parameters = given_parameters
        self.nb_nodes = np.prod(shape_data[1:])
        # SPDE diffusion parameters
        H = torch.empty((2,2,self.nb_nodes),requires_grad=False).to(device)
        H11 = torch.reshape(torch.Tensor(nc.H11.values),(self.nb_nodes,)).to(device)
        H12 = torch.reshape(torch.Tensor(nc.H12.values),(self.nb_nodes,)).to(device)
        H22 = torch.reshape(torch.Tensor(nc.H22.values),(self.nb_nodes,)).to(device)
        H[0,0,:] = H22
        H[0,1,:] = H12
        H[1,0,:] = H12
        H[1,1,:] = H11
        self.params = [1./3, None, H]
        self.Q = self.operator_spde(.33,None,torch.unsqueeze(H,dim=0),1.,
                                        square_root=self.square_root)[0]

    def forward(self, x):
        n_b, n_t, n_y, n_x = x.shape
        kappa = .33
        m = None
        H = torch.stack(n_b*[self.params[2]])
        tau = 1.

        # SPDE prior (sparse Q)
        Q = torch.stack(n_b*[self.Q])
        x_new = list()
        for i in range(n_b):
            Q_ = self.Q
            if self.square_root==True:
                x_ = torch.matmul(Q_,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            else:
                x_ = sp_mm(Q_,torch.reshape(torch.permute(x[i],(0,2,1)),
                                            (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
        H = torch.stack(n_b*[self.params[2]])
        H = torch.reshape(H,(n_b,2,2,n_x,n_y))
        return x, [kappa,m,H,tau]

class Phi_r(torch.nn.Module):

    def __init__(self, shape_data, pow=1, spde_type="diff",
                 square_root=False,
                 given_parameters=False, nc="spde_path.nc"):
        super().__init__()
        self.shape_data = shape_data
        self.spde_type = spde_type
        self.pow = pow
        self.operator_spde = Prior_SPDE(self.shape_data,pow=self.pow,spde_type=self.spde_type)
        self.square_root = square_root
        self.given_parameters = given_parameters
        self.nb_nodes = np.prod(shape_data[1:])
        if self.given_parameters==True:
            # SPDE diffusion parameters
            H = torch.empty((2,2,self.nb_nodes),requires_grad=False).to(device)
            H11 = torch.reshape(torch.Tensor(nc.H11.values),(self.nb_nodes,)).to(device)
            H12 = torch.reshape(torch.Tensor(nc.H12.values),(self.nb_nodes,)).to(device)
            H22 = torch.reshape(torch.Tensor(nc.H22.values),(self.nb_nodes,)).to(device)
            H[0,0,:] = H22
            H[0,1,:] = H12
            H[1,0,:] = H12
            H[1,1,:] = H11
            self.params = [1./3, None, H]
            self.Q = self.operator_spde(.33,None,torch.unsqueeze(H,dim=0),1.,
                                        square_root=self.square_root)[0]

    def forward(self, state, estim_params=True):
        # augmented state [x,params], i.e. (#b,#t+#params,#y,#x)
        n_b, n_t, n_y, n_x = state.shape
        n_t = n_t-2
        x = state[:,:n_t,:,:]
        vx = state[:,n_t,:,:]
        vy = state[:,n_t+1,:,:]
        vx = torch.reshape(vx,(n_b,n_x*n_y))
        vy = torch.reshape(vy,(n_b,n_x*n_y))
        vxy = torch.stack([vx,vy],dim=2)
        vxyT = torch.permute(vxy,(0,2,1))
        gamma = 1*torch.ones(n_b).to(device)
        beta = 25*torch.ones(n_b).to(device)
        H  = torch.einsum('ij,bk->bijk',torch.eye(2).to(device),
                                        torch.unsqueeze(gamma,dim=1).expand(n_b,n_x*n_y))+\
             torch.einsum('b,bijk->bijk',beta,torch.einsum('bki,bjk->bijk',vxy,vxyT))
        # reshape
        params = torch.reshape(H,(n_b,2,2,n_x*n_y)).to(device)

        # state space -> parameter space
        if self.given_parameters==False:
            kappa = .33
            tau = 1.
            m = None
            H = torch.empty((n_b,2,2,self.nb_nodes),requires_grad=False).to(device)
            H[:,0,0,:] = torch.reshape(params[:,0,0,:],(n_b,self.nb_nodes))
            H[:,0,1,:] = torch.reshape(params[:,0,1,:],(n_b,self.nb_nodes))
            H[:,1,0,:] = torch.reshape(params[:,1,0,:],(n_b,self.nb_nodes))
            H[:,1,1,:] = torch.reshape(params[:,1,1,:],(n_b,self.nb_nodes))
        else:
            kappa = .33
            m = None
            H = torch.stack(n_b*[self.params[2]])
            tau = 1.

        # SPDE prior (sparse Q)
        if self.given_parameters==False:
            Q = self.operator_spde(kappa, m, H, tau, square_root=self.square_root)
        #else:
        #    Q = torch.stack(n_b*[self.Q])
        x_new = list()
        for i in range(n_b):
            if self.given_parameters==False:
                Q_ = Q[i]
            else:
                Q_ = self.Q
            if self.square_root==True:
                x_ = torch.matmul(Q_,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            else:
                x_ = sp_mm(Q_,torch.reshape(torch.permute(x[i],(0,2,1)),
                                            (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
        if self.given_parameters==False:
            H = torch.reshape(H,(n_b,2,2,n_x,n_y))
        else:
            H = torch.stack(n_b*[self.params[2]])
            H = torch.reshape(H,(n_b,2,2,n_x,n_y))
        return x, [kappa,m,H,tau]

class Phi_r_state_dependent(torch.nn.Module):

    def __init__(self, shape_data, pow=1, spde_type="diff",
                 square_root=False,
                 given_parameters=False, nc="spde_path.nc"):
        super().__init__()
        self.spde_type = spde_type
        self.pow = pow
        self.encoder = encode_param_CNN_diff_Fourier(shape_data,10,1)
        self.decoder = decode_param_CNN_diff_Fourier(shape_data,1)
        #self.encoder = encode_param_CNN_diff(shape_data,10)
        #self.decoder = decode_param_CNN_diff(shape_data)
        self.operator_spde = Prior_SPDE(shape_data,pow=self.pow,spde_type=self.spde_type)
        self.square_root = square_root
        self.given_parameters = given_parameters
        if self.given_parameters==True:
            # SPDE diffusion parameters
            self.nb_nodes = np.prod(shape_data[1:])
            H = torch.empty((2,2,self.nb_nodes),requires_grad=False).to(device)
            H11 = torch.reshape(torch.Tensor(nc.H11.values),(self.nb_nodes,)).to(device)
            H12 = torch.reshape(torch.Tensor(nc.H12.values),(self.nb_nodes,)).to(device)
            H22 = torch.reshape(torch.Tensor(nc.H22.values),(self.nb_nodes,)).to(device)
            H[0,0,:] = H22
            H[0,1,:] = H12
            H[1,0,:] = H12
            H[1,1,:] = H11

            self.params = [1./3, None, H] 
            self.Q = self.operator_spde(.33,None,torch.unsqueeze(H,dim=0),1.,
                                        square_root=self.square_root)[0]

    def forward(self, x, estim_params=True):
        x_new = list()
        n_b, n_t, n_y, n_x = x.shape
        # state space -> parameter space
        if self.given_parameters==False:
            params = self.encoder(x)
            kappa = .33
            tau = 1.
            m = None
            H = self.decoder(params)
            '''
            nc = xr.open_dataset("/users/local/m19beauc/4dvarnet-core/oi/xp_gp/data/SPDE_diffusion_dataset_kappa_033_pow_2_diff.nc")
            H11_gt = torch.transpose(torch.Tensor(nc.H11.values),0,1).to(device)
            H12_gt = torch.transpose(torch.Tensor(nc.H22.values),0,1).to(device)
            H22_gt = torch.transpose(torch.Tensor(nc.H12.values),0,1).to(device)
            H11_gt = (torch.unsqueeze(H11_gt,dim=0)).expand(n_b,100,100)
            H12_gt = (torch.unsqueeze(H12_gt,dim=0)).expand(n_b,100,100)
            H22_gt = (torch.unsqueeze(H22_gt,dim=0)).expand(n_b,100,100)
            H[:,0,0,:] = torch.reshape(H22_gt,(n_b,10000))
            H[:,0,1,:] = torch.reshape(H12_gt,(n_b,10000))
            H[:,1,0,:] = torch.reshape(H12_gt,(n_b,10000))
            H[:,1,1,:] = torch.reshape(H11_gt,(n_b,10000))
            '''
        else:
            kappa = .33
            m = None
            H = torch.stack(n_b*[self.params[2]])
            tau = 1.

        # SPDE prior (sparse Q)
        if self.given_parameters==False:
            Q = self.operator_spde(kappa, m, H, tau, square_root=self.square_root)
        #else:
        #    Q = torch.stack(n_b*[self.Q])
        x_new = list()
        for i in range(n_b):
            if self.given_parameters==False:
                Q_ = Q[i]
            else:
                Q_ = self.Q
            if self.square_root==True:
                x_ = torch.matmul(Q_,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            else:
                x_ = sp_mm(Q_,torch.reshape(torch.permute(x[i],(0,2,1)),
                                            (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
        if self.given_parameters==False:
            H = torch.reshape(H,(n_b,2,2,n_x,n_y))
        else:
            H = torch.stack(n_b*[self.params[2]])
            H = torch.reshape(H,(n_b,2,2,n_x,n_y))
        return x, [kappa,m,H,tau]

class Phi_r3(torch.nn.Module):
    def __init__(self, shape_data, pow=1, nc="spde_path.nc"):
        super().__init__()
        self.pow = pow
        self.operator_spde = Prior_SPDE(shape_data,pow=self.pow)
        # SPDE diffusion parameters
        self.nb_nodes = np.prod(shape_data[1:])
        self.n_t, self.n_x, self.n_y = shape_data
        H = torch.empty((2,2,self.nb_nodes),requires_grad=True).to(device)
        H11 = torch.reshape(torch.transpose(torch.Tensor(nc.H11.values),0,1),(self.nb_nodes,)).to(device)
        H12 = torch.reshape(torch.transpose(torch.Tensor(nc.H12.values),0,1),(self.nb_nodes,)).to(device)
        H22 = torch.reshape(torch.transpose(torch.Tensor(nc.H22.values),0,1),(self.nb_nodes,)).to(device)
        H[0,0,:] = H22
        H[0,1,:] = H12
        H[1,0,:] = H12
        H[1,1,:] = H11
        self.H = H
        self.op = self.operator_spde(.33,None,torch.unsqueeze(H,dim=0),
                                     1.,square_root=False)[0]

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
                                         torch.Size([nb_obs,self.n_t*self.nb_nodes])).to(device)
            inv_R = 1e3*sparse_eye(nb_obs)
            Qxx = self.op + spspmm(spspmm(torch.transpose(opH,0,1),inv_R),opH)
            Qxy = -1.*spspmm(torch.transpose(opH,0,1),inv_R)
            # Cholesky inverse of Qg+Ht*inv(R)*H for k=0..Nt
            Lxx = None
            Lxx = cholesky_sparse.apply(Qxx.cpu(),True)
            # START SPDE-based conditional simulations)
            #xa = SPDE_spatiotemporal_kriging(Qxx, Lxx, Qxy, obs[i], mask[i], sparse=True, torch_sparse_solve=False)
            xa = SPDE_spatiotemporal_kriging(Qxx, Lxx, Qxy.to_dense(), obs[i], mask[i], sparse=False, torch_sparse_solve=False)
            x_new.append(xa)
        x = torch.stack(x_new)
        x = torch.permute(torch.reshape(x,(n_b,self.n_t,self.n_x,self.n_y)),(0,1,3,2))
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

