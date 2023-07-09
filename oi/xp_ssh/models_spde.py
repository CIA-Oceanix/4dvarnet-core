import einops
import xarray as xr
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import oi.xp_ssh.solver_spde as NN_4DVar
from oi.sqrtm import *
from oi.spde import *
from oi.scipy_sparse_tools import *
from oi.xp_ssh.metrics import nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, save_loss
from oi.xp_ssh.metrics import save_NetCDF2 as save_netcdf
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

class encode_param_CNN_Fourier(torch.nn.Module):

    def __init__(self,shape_data,dim_ae,n_freq=1):
        super(encode_param_CNN_Fourier, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data
        self.n_freq = n_freq
        N = np.arange(1,self.n_freq+1)
        Z = np.arange(-1*(self.n_freq),(self.n_freq+1))
        # {0}*N
        zero = np.array([0])
        zero_N = np.array(np.meshgrid(0,N)).T.reshape(-1, 2)
        # N*Z
        N_Z = np.array(np.meshgrid(N,Z)).T.reshape(-1, 2)
        self.kl = np.vstack((zero_N,N_Z))
        self.n_params = self.n_t*((self.kl.shape[0]*12)+6+2) # + zero frequencies + gamma + beta

        # Conv2D layers
        self.pool1 = torch.nn.AvgPool2d(2, 2)
        self.conv1   = torch.nn.Conv2d(self.n_t,dim_ae,(3,3),padding=1,bias=False)
        self.pool2 = torch.nn.AvgPool2d(2, 2)
        self.conv2   = torch.nn.Conv2d(dim_ae,2*dim_ae,(3,3),padding=1,bias=False)
        self.pool3 = torch.nn.AvgPool2d(4, 4)
        self.conv3   = torch.nn.Conv2d(2*dim_ae,4*dim_ae,(3,3),padding=1,bias=False)
        self.fc1 = torch.nn.Linear(4*dim_ae*(self.n_y//16)*(self.n_x//16), 512)
        self.fc2 = torch.nn.Linear(512, self.n_params)

    def forward(self, x):
        # input shape (b,t,y,x) --> output shape (b,7*t,y,x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

class decode_param_CNN_Fourier(torch.nn.Module):

    def __init__(self, shape_data, n_freq=1):
        super(decode_param_CNN_Fourier, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data
        self.A = 100.
        self.B = 100.
        self.n_freq = n_freq
        N = np.arange(1,self.n_freq+1)
        Z = np.arange(-1*(self.n_freq),(self.n_freq+1))
        # {0}*N
        zero = np.array([0])
        zero_N = np.array(np.meshgrid(0,N)).T.reshape(-1, 2)
        # N*Z
        N_Z = np.array(np.meshgrid(N,Z)).T.reshape(-1, 2)
        self.kl = np.vstack((zero_N,N_Z))
        self.n_params = self.n_t*((self.kl.shape[0]*12)+6+2)

    def forward(self, params):
        n_b = params.shape[0]

        x = torch.arange(0,self.A).to(device)
        y = torch.arange(0,self.B).to(device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        gamma = params[:,0]
        beta = params[:,1]
        pi = torch.acos(torch.zeros(1)).item() * 2
            
        kappa_ = []
        m_ = []
        H_ = []
        tau_ = []

        for k in range(self.n_t):
            # init kappa, m1, m2, v_x, v_y and tau
            kappa = 0
            m1 = 0
            m2 = 0
            vx = 0
            vy = 0
            tau = 0
            for i in range(0,self.kl.shape[0]):
                start = k*((i*12)+8)+8
                k = self.kl[i,0]
                l = self.kl[i,1]
                kappa += torch.einsum('b,ij->bij',params[:,start],torch.cos(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                kappa += torch.einsum('b,ij->bij',params[:,start+1],torch.sin(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                m1 += torch.einsum('b,ij->bij',params[:,start+2],torch.cos(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                m1 += torch.einsum('b,ij->bij',params[:,start+3],torch.sin(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                m2 += torch.einsum('b,ij->bij',params[:,start+4],torch.cos(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                m2 += torch.einsum('b,ij->bij',params[:,start+5],torch.sin(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                vx += torch.einsum('b,ij->bij',params[:,start+6],torch.cos(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                vx += torch.einsum('b,ij->bij',params[:,start+7],torch.sin(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                vy += torch.einsum('b,ij->bij',params[:,start+8],torch.cos(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                vy += torch.einsum('b,ij->bij',params[:,start+9],torch.sin(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                tau += torch.einsum('b,ij->bij',params[:,start+10],torch.cos(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
                tau += torch.einsum('b,ij->bij',params[:,start+11],torch.sin(2*pi*((k*grid_x/self.A)+(l*grid_y/self.B))))
            for b in range(n_b):   
                kappa[b] += params[b,k*((i*12)+6+2)+2]
                m1[b] += params[b,k*((i*12)+6+2)+3]
                m2[b] += params[b,k*((i*12)+6+2)+4]
                vx[b] += params[b,k*((i*12)+6+2)+5]
                vy[b] += params[b,k*((i*12)+6+2)+6]
                tau[b] += params[b,k*((i*12)+6+2)+7]
            kappa = F.softplus(torch.reshape(kappa,(n_b,self.n_x*self.n_y)))
            m1 = torch.reshape(m1,(n_b,self.n_x*self.n_y))
            m2 = torch.reshape(m2,(n_b,self.n_x*self.n_y))
            vx = torch.reshape(vx,(n_b,self.n_x*self.n_y))
            vy = torch.reshape(vy,(n_b,self.n_x*self.n_y))
            tau = F.softplus(torch.reshape(tau,(n_b,self.n_x*self.n_y)))
            vxy = torch.stack([vx,vy],dim=2)
            vxyT = torch.permute(vxy,(0,2,1))
            gamma = F.softplus(params[:,k*((i*12)+6+2)])
            beta = F.softplus(params[:,k*((i*12)+6+2)+1])
            print(tau)
            H  = torch.einsum('ij,bk->bijk',torch.eye(2).to(device),
                                        torch.unsqueeze(gamma,dim=1).expand(n_b,self.n_x*self.n_y))+\
                 torch.einsum('b,bijk->bijk',beta,torch.einsum('bki,bjk->bijk',vxy,vxyT))
            m = torch.stack([m1,m2],dim=1)

            kappa_.append(kappa)
            m_.append(m)
            H_.append(H)
            tau_.append(tau)

        # stack the time dimension
        kappa = torch.stack(kappa_,dim=2)
        m = torch.stack(m_,dim=3)
        H = torch.stack(H_,dim=4)
        tau = torch.stack(tau_,dim=2)

        # reshape
        kappa = torch.reshape(kappa,(n_b,1,self.n_y*self.n_x,self.n_t))
        m = torch.reshape(m,(n_b,2,self.n_y*self.n_x,self.n_t))
        H = torch.reshape(H,(n_b,2,2,self.n_y*self.n_x,self.n_t))
        tau = torch.reshape(tau,(n_b,1,self.n_y*self.n_x,self.n_t))

        # kappa has shape (b,t,y,x)
        # m has shape (b,2,t,y,x)
        # H has shape (b,2,2,t,y,x)
        # tau has shape (b,t,y,x)
        return kappa.float(), m.float(), H.float(), tau.float()

class Prior_SPDE(torch.nn.Module):

    def __init__(self,shape_data, pow=1, spde_type="adv_diff"):

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
            Q_list = list()
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
                    inv_tau_0 = sparse_eye(self.nb_nodes,(1./tau[batch,:,0]*np.sqrt(self.dt)))
                    Qs_tilde0 = spspmm(spspmm(inv_tau_0.t(),
                                              Qs),
                                       inv_tau_0)
                    Q0 = (self.dx*self.dy)*spspmm(spspmm(B.t(),
                                                         Qs_tilde0),
                                                  B)
                    inv_M0 = self.Id+self.dt*B
                    inv_S0 = spspmm(spspmm(inv_M0.t(),
                                           Qs_tilde0),
                                    inv_M0)
                    Q0 = (1./2)*(Q0+Q0.t()) + 5e-2*sparse_eye(self.nb_nodes)
                else:
                    inv_tau_k = sparse_eye(self.nb_nodes,(1./tau[batch,:,k]*np.sqrt(self.dt)))
                    Qs_tilde = spspmm(spspmm(inv_tau_k.t(),
                                             Qs),
                                      inv_tau_k)
                    Q_k = (self.dx*self.dy)*spspmm(spspmm(B.t(),
                                                         Qs_tilde),
                                                  B)
                    inverse_M = self.Id+self.dt*B
                    inverse_S = spspmm(spspmm(inverse_M.t(),
                                              Qs_tilde),#Qs),
                                       inverse_M)
                    #inverse_S = (1./2)*(inverse_S+inverse_S.t()) + 5e-2*sparse_eye(self.nb_nodes)
                    inv_M.append(inverse_M)
                    inv_S.append(inverse_S)
                    Q_list.append(Q_k)
 
            if store_block_diag==True:
                l = list(inv_S)
                #l.insert(0,Q0)
                l.insert(0,inv_S0)
                #l = list(Q_list)
                #l.insert(0,Q0)

                block_diag.append(l)
                    
            # Build the global precision matrix
            row = torch.tensor([]).to(device)
            col = torch.tensor([]).to(device) 
            val = torch.tensor([]).to(device)    
             
            # first line 
            inv_tau = sparse_eye(self.nb_nodes,(1./tau[batch,:,1]*np.sqrt(self.dt))) 
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qs),inv_tau)
            # Q0+Qs_tilde,
            row, col, val = self.stack_indices(inv_S0+Qs_tilde,
                                          0,0, 
                                          row,col,val) 
            row, col, val = self.stack_indices(-1.*spspmm(Qs_tilde,inv_M[0]), 
                                          0,self.nb_nodes, 
                                          row,col,val) 
            # loop 
            for i in np.arange(1,self.n_t-1): 
                inv_tau_1 = sparse_eye(self.nb_nodes,(1./tau[batch,:,i]*np.sqrt(self.dt))) 
                Qs_tilde1 = spspmm(spspmm(inv_tau_1.t(),Qs),inv_tau_1) 
                inv_tau_2 = sparse_eye(self.nb_nodes,(1./tau[batch,:,i+1]*np.sqrt(self.dt))) 
                Qs_tilde2 = spspmm(spspmm(inv_tau_2.t(),Qs),inv_tau_2) 
                row, col, val = self.stack_indices(-1.*spspmm(inv_M[i-1].t(),Qs_tilde1), 
                                              i*self.nb_nodes,(i-1)*self.nb_nodes, 
                                              row,col,val) 
                row, col, val = self.stack_indices(spspmm(spspmm(inv_M[i-1].t(),Qs_tilde1),inv_M[i-1])+Qs_tilde2, 
                                              i*self.nb_nodes,i*self.nb_nodes, 
                                              row,col,val) 
                row, col, val = self.stack_indices(-1.*spspmm(Qs_tilde2,inv_M[i]), 
                                              i*self.nb_nodes,(i+1)*self.nb_nodes, 
                                              row,col,val) 
            # last line 
            inv_tau = sparse_eye(self.nb_nodes,(1./tau[batch,:,self.n_t-1]*np.sqrt(self.dt))) 
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
            '''
            # first line
            inv_tau = sparse_eye(self.nb_nodes,1./(tau[batch,:,1]*np.sqrt(self.dt)))
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qs),inv_tau)
            Qg = torch.hstack([Q0+self.Id,
                               -1.*spspmm(Qs_tilde,inv_M[0]),
                               sparse_repeat(self.nb_nodes,1,self.n_t-2)])
            # loop
            for i in np.arange(1,self.n_t-1):
                inv_tau_1 = sparse_eye(self.nb_nodes,1./(tau[batch,:,i]*np.sqrt(self.dt)))
                Qs_tilde1 = spspmm(spspmm(inv_tau_1.t(),Qs),inv_tau_1)
                inv_tau_2 = sparse_eye(self.nb_nodes,1./(tau[batch,:,i+1]*np.sqrt(self.dt)))
                Qs_tilde2 = spspmm(spspmm(inv_tau_2.t(),Qs),inv_tau_2)               
                Qg_ = torch.hstack([sparse_repeat(self.nb_nodes,1,i-1),
                                  -1.*spspmm(inv_M[i-1].t(),Qs_tilde1),
                                  spspmm(spspmm(inv_M[i-1].t(),Qs_tilde1),inv_M[i-1])+Qs_tilde1,
                                  -1.*spspmm(Qs_tilde2,inv_M[i]),
                                  sparse_repeat(self.nb_nodes,1,self.n_t-4-(i-2))])
                Qg  = torch.vstack([Qg,Qg_])
            # last line
            inv_tau = sparse_eye(self.nb_nodes,1./(tau[batch,:,self.n_t-1]*np.sqrt(self.dt)))
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qs),inv_tau)
            Qg_ = torch.hstack([sparse_repeat(self.nb_nodes,1,self.n_t-2),
                              -1.*spspmm(inv_M[self.n_t-2].t(),Qs_tilde),
                              spspmm(spspmm(inv_M[self.n_t-2].t(),Qs_tilde),inv_M[self.n_t-2])])
            Qg  = torch.vstack([Qg,Qg_])
            '''

            # enforce positive definiteness
            # Qg = (1./2)*(Qg+Qg.t()) + 5e-2*sparse_eye(self.n_t*self.nb_nodes)

            # add batch
            Q.append(Qg)

        # Q = torch.stack(Q) #pb differentiating tensor 

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
                return op, block_diag
        else:
            if store_block_diag==False:
                return Q
            else:
                return Q, block_diag

class Phi_r(torch.nn.Module):

    def __init__(self, shape_data, pow=1, spde_type="adv_diff",
                 square_root=False):
        super().__init__()
        self.spde_type = spde_type
        self.pow = pow
        #self.encoder = encode_param_CNN(shape_data,10)
        #self.decoder = decode_param_CNN(shape_data)
        self.encoder = encode_param_CNN_Fourier(shape_data,10)
        self.decoder = decode_param_CNN_Fourier(shape_data)
        self.operator_spde = Prior_SPDE(shape_data,pow=self.pow,spde_type=self.spde_type)
        self.square_root = square_root

    def forward(self, state, estim_params=True):
        # augmented state [x,params], i.e. (#b,#t+#params,#y,#x)
        n_b, n_t, n_y, n_x = state.shape
        n_t = n_t//9
        x = state[:,:n_t,:,:]
        kappa = state[:,n_t:2*n_t,:,:]
        tau = state[:,2*n_t:3*n_t,:,:]
        m1 = state[:,3*n_t:4*n_t,:,:]
        m2 = state[:,4*n_t:5*n_t,:,:]
        vx = state[:,5*n_t:6*n_t,:,:]
        vy = state[:,6*n_t:7*n_t,:,:]
        gamma = state[:,7*n_t:8*n_t,:,:]
        beta = state[:,8*n_t:9*n_t,:,:]
        H = []
        for k in range(n_t):
            vx_ = torch.reshape(vx[:,k,:,:],(n_b,n_x*n_y))
            vy_ = torch.reshape(vy[:,k,:,:],(n_b,n_x*n_y))
            vxy = torch.stack([vx_,vy_],dim=2)
            vxyT = torch.permute(vxy,(0,2,1))
            '''
            gamma = 1*torch.ones(n_b).to(device)
            beta = 1*torch.ones(n_b).to(device)
            H_  = torch.einsum('ij,bk->bijk',torch.eye(2).to(device),
                                        torch.unsqueeze(gamma,dim=1).expand(n_b,n_x*n_y))+\
             torch.einsum('b,bijk->bijk',beta,torch.einsum('bki,bjk->bijk',vxy,vxyT))
            '''
            gamma_ = torch.reshape(gamma[:,k,:,:],(n_b,n_x*n_y))
            beta_ = torch.reshape(beta[:,k,:,:],(n_b,n_x*n_y))
            H_ = torch.einsum('ij,bk->bijk',
                              torch.eye(2).to(device),
                              gamma_)+\
                 torch.einsum('bk,bijk->bijk',beta_,torch.einsum('bki,bjk->bijk',vxy,vxyT))
            H.append(H_)
        H = torch.stack(H,dim=4)
        m = torch.stack([m1,m2],dim=1)

        # parameters must have spatial coordinates as x/y (not y/x)
        '''
        kappa = torch.permute(kappa,(0,2,3,1))
        tau = torch.permute(tau,(0,2,3,1))
        m = torch.permute(m,(0,1,3,4,2))
        H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
        '''
        kappa = torch.permute(kappa,(0,3,2,1))
        tau = torch.permute(tau,(0,3,2,1))
        m = torch.permute(m,(0,1,4,3,2))
        H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
        H = torch.permute(H,(0,1,2,4,3,5))

        kappa = torch.reshape(kappa,(n_b,1,n_y*n_x,n_t))
        m = torch.reshape(m,(n_b,2,n_y*n_x,n_t))
        H = torch.reshape(H,(n_b,2,2,n_y*n_x,n_t))
        tau = torch.reshape(tau,(n_b,1,n_y*n_x,n_t))

        #m[:,0,:,:] = 0
        #m[:,1,:,:] = 0
        #H[:,0,0,:,:] = 1
        #H[:,0,1,:,:] = 0
        #H[:,1,0,:,:] = 0
        #H[:,1,1,:,:] = 1

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
                x_ = sp_mm(Q_,
                                 torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (n_t*n_x*n_y,1)))
            if self.square_root==False:
                x_ = torch.matmul(torch.reshape(torch.permute(x[i],(0,2,1)),
                                               (1,n_t*n_x*n_y)),x_)
                x_new.append(x_[0,0])
            else:
                x_new.append(torch.permute(torch.reshape(x_,(n_t,n_x,n_y)),(0,2,1)))
        x = torch.stack(x_new)
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

