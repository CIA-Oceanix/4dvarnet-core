import scipy
import sksparse
from scipy.sparse.linalg import spsolve 
import numpy as np
import cupy
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
from cupyx.scipy.sparse import csc_matrix as cupy_sp_csc_matrix
from torch_sparse.tensor import SparseTensor
from torch_sparse.convert import to_scipy
from torch_sparse.convert import from_torch_sparse
from torch_sparse.matmul import matmul as matmul2
from torch_sparse import spmm
from torch_sparse import spspmm as spsp_mm
from torch.autograd import Function
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch_sparse
import torch
from oi.chol_sp_diff import chol_rev
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sp_mm(A,B):
    indexA = A.coalesce().indices()
    valueA = A.coalesce().values()
    m = A.size()[0]
    n = A.size()[1]
    res = spmm(indexA, valueA, m, n, B).to(device)
    '''
    #v1
    m = B.size()[0]
    n = B.size()[1]
    k = A.size()[0]
    row = torch.arange(m)
    col = torch.full((m,), 0)
    index = torch.stack([row, col], dim=0).to(device)
    val = torch.flatten(B).to(device)
    B = torch.sparse.FloatTensor(index.long(), val, torch.Size([m,n])).to(device)
    res = torch.sparse.mm(A,B)
    val = res.coalesce().values()
    res = torch.unsqueeze(val,dim=1).to(device)
    '''
    '''
    v2
    res = torch.zeros((m,1)).to(device)
    for i in range(m):
        idx = torch.where(indexA[0]==i)
        res[i,0] = valueA[idx] @ B[indexA[1][idx],0]         
    '''
    #res = torch.sparse.mm(A,B)
    #res = res.to_sparse().coalesce()
    return res

def spspmm(A,B):
    """Matrix product of two sparse tensors. Both input sparse matrices need to
    be coalesced (use the :obj:`coalesced` attribute to force).
    Args:
        indexA (:class:`LongTensor`): The index tensor of first sparse matrix.
        valueA (:class:`Tensor`): The value tensor of first sparse matrix.
        indexB (:class:`LongTensor`): The index tensor of second sparse matrix.
        valueB (:class:`Tensor`): The value tensor of second sparse matrix.
        m (int): The first dimension of first corresponding dense matrix.
        k (int): The second dimension of first corresponding dense matrix and
            first dimension of second corresponding dense matrix.
        n (int): The second dimension of second corresponding dense matrix.
        coalesced (bool, optional): If set to :obj:`True`, will coalesce both
            input sparse matrices. (default: :obj:`False`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    res = torch.sparse.mm(A,B)
    res = res.coalesce()
    '''
    #v1
    indexA = A.coalesce().indices()
    valueA = A.coalesce().values()
    indexB = B.coalesce().indices()
    valueB = B.coalesce().values()
    m = A.size()[0]
    k = A.size()[1]
    n = B.size()[1]
    res = spsp_mm(indexA,valueA,indexB,valueB,m,k,n, coalesced=True)
    res = torch.sparse.FloatTensor(res[0],
                                   res[1],
                                   torch.Size([m,n])).to(device)
    res.requires_grad = True
    '''
    '''
    #v2
    coalesced =True
    indexA = A.coalesce().indices()
    valueA = A.coalesce().values()
    indexB = B.coalesce().indices()
    valueB = B.coalesce().values()
    m = A.size()[0]
    k = A.size()[1]
    n = B.size()[1]
    A2 = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B2 = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    C = matmul2(A2, B2)
    row, col, value = C.coo()
    indexC = torch.stack([row, col], dim=0)
    valueC = value
    res = torch.sparse.FloatTensor(indexC.long(), valueC, torch.Size([m,n])).to(device)
    res.requires_grad = True
    '''
    return res

# cast torch_sparse COO matrix to scipy sparse csc matrix
def sparse_torch2scipy(A):
    B = scipy.sparse.csc_matrix((A.coalesce().values(),
                         A.coalesce().indices()),
                         shape=(A.size()[0],A.size()[0]))
    return B

# solve Ax=b with A is sparse
class cupy_solve_sparse(Function):
    @staticmethod
    def forward(ctx, A, b):
        if A.ndim != 2 or (A.shape[0] != A.shape[1]):
            raise ValueError("A should be a square 2D matrix.")

        # cast torch_sparsematrix to scipy sparse csc matrix
        val = cupy.fromDlpack(to_dlpack(A.coalesce().values().data))
        ind = cupy.fromDlpack(to_dlpack(A.coalesce().indices().data))
        A_ = cupy_sp_csc_matrix((val,ind),shape=(A.size()[0],A.size()[0]))
        b_ = cupy.fromDlpack(to_dlpack(b))

        x_ = cupy_spsolve(A_, b_)
        x = from_dlpack(x_.toDlpack()).to(device)
        x.requires_grad=True
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad):
        xbar = grad
        A, b, x = ctx.saved_tensors
        gradb = cupy_solve_sparse.apply(A.t(), xbar)
        gradA = -gradb[:,None] * x[None,:]
        print(gradA)
        return gradA.to(device), gradb.to(device)

# provide Cholesky decomposition of sparse matrix
class cholesky_sparse(Function):

    @staticmethod
    def forward(ctx, A, perm=False):
        # cast torch_sparsematrix to scipy sparse csc matrix
        index, value = from_torch_sparse(A.coalesce())
        A_np = to_scipy(index, value, A.size()[0], A.size()[1])

        # Cholesky decomposition without permutation A=L'L
        if perm==True:
            chol = sksparse.cholmod.cholesky(A_np)
            P, LD = chol.P(), chol.L_D()
            L, D = LD

            P = torch.Tensor(P)
            Lcoo = L.tocoo()
            L = torch.sparse.FloatTensor(torch.LongTensor([Lcoo.row.tolist(),
                                                       Lcoo.col.tolist()]),
                                     torch.FloatTensor(Lcoo.data))
            Dcoo = D.tocoo()
            inv_D = torch.FloatTensor(1./Dcoo.data)

            P = torch.as_tensor(P, dtype=P.dtype).to(device)
            L = torch.as_tensor(L, dtype=L.dtype).to(device)
            inv_D = torch.as_tensor(inv_D, dtype=inv_D.dtype).to(device)
            ctx.save_for_backward(A, P, L, inv_D)
            return P, L, inv_D
        else:
            L = sksparse.cholmod.cholesky(A_np,ordering_method="natural").L()
            Lcoo = L.tocoo()
            # save input requires for backward (A here from Ax=b) and
            L = torch.sparse.FloatTensor(torch.LongTensor([Lcoo.row.tolist(),
                                                       Lcoo.col.tolist()]),
                                     torch.FloatTensor(Lcoo.data))
            L = torch.as_tensor(L, dtype=L.dtype).to(device)
            ctx.save_for_backward(A,L)
            return L

    @staticmethod
    def backward(ctx, grad):
        Lbar = grad.cpu()
        A, L = ctx.saved_tensors
        A, L = A.cpu(), L.cpu() 
        sp_L = scipy.sparse.csc_matrix((L.coalesce().values(),
                                L.coalesce().indices()),
                                shape=(L.shape[0],L.shape[0]))
        sp_Lbar = scipy.sparse.csc_matrix((Lbar.coalesce().values(),
                                Lbar.coalesce().indices()),
                                shape=(Lbar.shape[0],Lbar.shape[0]))
        # compute Abar = chol_rev(L,Lbar) (Lbar is input grad)
        gradA = chol_rev(sp_L,sp_Lbar)
        gradA = gradA.tocoo()
        gradA = torch.sparse.FloatTensor(torch.LongTensor([gradA.row.tolist(), gradA.col.tolist()]),
                                                torch.FloatTensor(gradA.data)).to(device)
        return gradA

