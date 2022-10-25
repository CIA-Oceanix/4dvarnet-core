"""Python code for pushing derivatives through the Cholesky decomposition

Implements forwards- and reverse-mode update rules from:
    Differentiation of the Cholesky decomposition,
    Iain Murray, February 2016.
    http://arxiv.org/abs/1602.07527
"""

from scipy.sparse.linalg import inv
from scipy.sparse import tril
import numpy as np

def _Phi(A):
    """Return lower-triangle of matrix and have the diagonal"""
    A = tril(A)
    A.setdiag(A.diagonal()*0.5)
    return A

def _chol_symbolic_rev(L, Lbar):
    """
    Reverse-mode differentiation through the Cholesky decomposition
    
    This version uses a short symbolic expression to return
    tril(Sigma_bar) where "_bar" means sensitivities in reverse-mode
    differentiation, and Sigma = L @ L.T.
    """
    P = _Phi(np.dot(L.T, Lbar))
    invL = inv(L)
    return _Phi(invL.T @ (P + P.T) @ invL)

def _level3partition(A, j, k):
    """Return views into A used by the blocked algorithms"""
    # Top left corner of diagonal block is [j,j]
    # Block size is NB = (k-j)
    R = A[j:k, :j]     # Row block                     /      \
    D = A[j:k, j:k]    # triangular block on Diagonal  |      |
    B = A[k:, :j]      # Big corner block              | R D  |
    C = A[k:, j:k]     # Column block                  \ B C  /
    return R, D, B, C

def _chol_blocked_rev(L, Abar, NB=256, inplace=False):
    """
    Reverse-mode differentiation through the Cholesky decomposition
    
    Obtain tril(Sigma_bar) from L_bar, where "_bar" means sensitivities
    in reverse-mode differentiation, and Sigma = L @ L.T.

    This version uses a blocked algorithm to update sensitivities Abar
    in place. tril(Abar) should start containing L_bar, and will end
    containing the tril(Sigma_bar). Take tril(Abar) at the end if
    triu(Abar,1) did not start out filled with zeros. Alternatively,
    (tril(Abar) + tril(Abar).T) will give the symmetric, redundant
    matrix of sensitivities.
    
    Unlike the unblocked routine, if the upper triangular part of Abar
    started with non-zero values, some of these will be overwritten.

    If inplace=False, a copy of Abar is modified instead of the
    original. The Abar that was modified is returned.
    """
    if not inplace:
        Abar = Abar.copy()
    for k in range(L.shape[0], -1, -NB):
        j = max(0, k - NB)
        R, D, B, C = _level3partition(L, j, k)
        Rbar, Dbar, Bbar, Cbar = _level3partition(Abar, j, k)
        Cbar[:] = Cbar @ inv(tril(D))
        Bbar -= Cbar @ R
        Dbar[:] = tril(Dbar) - tril( Cbar.T @ C )
        Dbar[:] = _chol_symbolic_rev(D, Dbar)
        Rbar -= (Cbar.T @ B) + ((Dbar + Dbar.T) @ R)
    return Abar

def chol_rev(L, Abar, NB=256, inplace=False):
    """
    Reverse-mode differentiation through the Cholesky decomposition
    
    Obtain Sigma_bar = df/dSigma from L_bar = df/dL, where Sigma = L @ L.T.

    The input tril(Abar) should contain L_bar. The lower-triangle of
    the answer will contain Sigma_bar. Take tril() of the answer if input
    triu(Abar,1) was not filled with zeros.
    
    If inplace=True the answer is placed into the lower triangle of Abar
    as well as being returned. If inplace=False, a copy of Abar is modified
    and returned, and the original array is not touched.

    The optimal block-size NB depends on the machine and size of L, but
    should not change the answer (beyond the usual round-off errors).
    If the Fortran version is compiled, any user-specified NB will be
    ignored, and the block size will be chosen by LAPACK.
    """
    return _chol_blocked_rev(L, Abar, NB, inplace)

