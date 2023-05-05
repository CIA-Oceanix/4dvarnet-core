import numpy as np
import torch
from oi.scipy_sparse_tools import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_numpy_to_torch(A):
    rows, cols = A.nonzero()
    values = A.data
    indices = np.vstack((rows, cols))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    return torch.sparse.DoubleTensor(i, v, A.shape)

# cast torch_sparse COO matrix to scipy sparse csc matrix
def sparse_torch2scipy(A):
    B = scipy.sparse.csc_matrix((A.detach().cpu().coalesce().values().data.numpy(),
                         A.detach().cpu().coalesce().indices().data.numpy()),
                         shape=(A.size()[0],A.size()[0]))
    return B

def SPDE_spatiotemporal_kriging(Qxx, Lxx, Qxy, obs, mask, sparse=False, torch_sparse_solve=False):
    ''' Spatio-temporal Optimal Interpolation at time t_{1:Ldt}'''
    # Lxx = cholesky decomposition of Qxx
    # Qxy 

    Nt = len(obs)
    nbNodes = obs[0].shape[0]*obs[0].shape[1]
    DAW = np.arange(0,Nt)
    zD  = torch.tensor([]).to(device)
    for itime in range(Nt):
        idD = torch.where(torch.flatten(torch.transpose(mask[itime,:,:],0,1))!=0.)[0]
        if len(idD)>0:
            # Observations stack along DAW
            zD   = torch.cat((zD,torch.index_select(torch.flatten(torch.transpose(obs[itime,:,:],0,1)),0,idD)))

    RM  = torch.matmul(-1.*Qxy,zD)
    if sparse==False:
        if not isinstance(Lxx, tuple):
            RM = torch.unsqueeze(RM,1)
            z = torch.triangular_solve(RM,Lxx,upper=False)[0]
            z = torch.triangular_solve(z,torch.transpose(Lxx,0,1),upper=True)[0]
        else:
            Pxx_, Lxx, inv_Dxx = Lxx
            Lxx = Lxx.to_dense()
            RM = RM[Pxx_.long()]
            RM = torch.unsqueeze(RM,1)
            z = torch.triangular_solve(RM,Lxx,upper=False)[0]
            z = torch.mul(inv_Dxx,z[:,0])
            z = torch.unsqueeze(z,1)
            z = torch.triangular_solve(z,torch.transpose(Lxx,0,1),upper=True)[0]
            z = z[torch.argsort(Pxx_).long()]
    else:
        if torch_sparse_solve==False:
            #z = solve_sparse.apply(Lxx,RM)
            #z = solve_sparse.apply(torch.transpose(Lxx,0,1),z)
            z = cupy_solve_sparse.apply(Qxx,RM)
        else:
            A = torch.unsqueeze(Qxx,0).type(torch.sparse.DoubleTensor).coalesce()
            print(A.size())
            b = torch.unsqueeze(RM,0).type(torch.sparse.DoubleTensor)
            # Cholesky
            z = torch.squeeze(sparse_solve(A,torch.unsqueeze(b,2)),0)
            # CG
            #M = [None] * 1
            #M[0] = sparse_numpy_to_torch(diags(1. / sparse_torch2scipy(A[0]).diagonal(), format='csc'))
            #z = CG.apply(A,torch.unsqueeze(b,2),M)
    z = torch.reshape(z,(Nt,nbNodes)).to(device)
    return z
