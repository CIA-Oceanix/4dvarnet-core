from scipy.sparse.linalg import spsolve 
import cupy
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
from cupyx.scipy.sparse import csc_matrix as cupy_sp_csc_matrix
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import matmul as matmul2
from torch.autograd import Function
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch_sparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """
    indexA = A.detach().cpu().coalesce().indices()
    valueA = A.detach().cpu().coalesce().values()
    indexB = B.detach().cpu().coalesce().indices()
    valueB = B.detach().cpu().coalesce().values()
    """
    indexA = A.coalesce().indices()
    valueA = A.coalesce().values()
    indexB = B.coalesce().indices()
    valueB = B.coalesce().values()
    m = A.size()[0]
    k = A.size()[1]
    n = B.size()[1]

    coalesced =True

    A2 = SparseTensor(row=indexA[0], col=indexA[1], value=valueA,
                     sparse_sizes=(m, k), is_sorted=not coalesced)
    B2 = SparseTensor(row=indexB[0], col=indexB[1], value=valueB,
                     sparse_sizes=(k, n), is_sorted=not coalesced)

    C = matmul2(A2, B2)
    row, col, value = C.coo()
    indexC = torch.stack([row, col], dim=0)
    valueC = value

    return torch.sparse.FloatTensor(indexC.long(), valueC, torch.Size([m,n])).to(device)

# cast torch_sparse COO matrix to scipy sparse csc matrix
def sparse_torch2scipy(A):
    B = scipy.sparse.csc_matrix((A.detach().cpu().coalesce().values().data.numpy(),
                         A.detach().cpu().coalesce().indices().data.numpy()),
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
        return gradA.to(device), gradb.to(device)

# provide Cholesky decomposition of sparse matrix
class cholesky_sparse(Function):
    @staticmethod
    def forward(ctx, A):

        # cast torch_sparsematrix to scipy sparse csc matrix
        A_np = sparse_torch2scipy(A)

        # Cholesky decomposition without permutation A=L'L
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
        Lbar = grad.detach().cpu()
        A, L = ctx.saved_tensor
        sp_L = scipy.sparse.csc_matrix((L.coalesce().values(),
                                L.coalesce().indices()),
                                shape=(L.shape[0],L.shape[0]))
        sp_Lbar = scipy.sparse.csc_matrix((Lbar.coalesce().values(),
                                Lbar.coalesce().indices()),
                                shape=(Lbar.shape[0],Lbar.shape[0]))
        # compute Abar = chol_rev(L,Lbar) (Lbar is input grad)
        gradA = chol_rev(sp_L,sp_Lbar)
        gradA = gradA.tocoo()
        gradA = torch.sparse.FloatTensor(torch.LongTensor([Gcoo.row.tolist(), Gcoo.col.tolist()]),
                                                torch.FloatTensor(Gcoo.data)).to(device)
        return gradA

