import torch
from scipy_sparse_tools import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pow_diff_operator(A,pow,sparse=False):
    B=A
    for i in range(pow-1):
        if sparse==False:
            B = torch.matmul(B,A)
        else:
            B = spspmm(B,A)
    return B

def DiffOperator_Isotropic(Nx, Ny, dx, dy, kappa):
    # kappa is a scalar
    nbNodes = Nx * Ny
    indices = torch.arange(0,nbNodes).to(device)
    ## Voisin à droite Points de la grille concernés : Retrait du bord droit
    index = torch.where(torch.fmod(indices+1,Nx) != 0)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1, (-1./(dx**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Voisin à gauche Points de la grille concernés : Retrait du bord gauche
    index = torch.where(torch.fmod(indices+1,Nx) != 1)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1, (-1./(dx**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Voisin du haut Points de la grille concernés : Retrait du bord haut
    index = torch.where((indices+1) <= (Ny-1)*Nx )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx, (-1./(dy**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Voisin du bas Points de la grille concernés : Retrait du bord bas
    index = torch.where((indices+1) >= (Nx+1) )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx, (-1/(dy**2))*torch.ones(len(indicesVoisins)).to(device) ))
    ## Point central
    k5 = torch.stack((indices.float(), indices.float(), (kappa**2 + 2 * (1./(dx**2) + 1./(dy**2)))*torch.ones(len(indices)).to(device) ))
    ## Tous les voisins
    k = torch.cat((k1, k2, k3, k4, k5),dim=1)
    return(torch.sparse.FloatTensor(k[0:2].long(), k[2], torch.Size([nbNodes,nbNodes])))

def SPDE_spatial_simulation(A, dx, dy):
    nbNodes = A.shape[0]
    random = torch.randn(nbNodes).to(device)
    RM  = torch.unsqueeze(torch.flatten((1/(dx*dy))*random),1)
    '''
    z = torch.triangular_solve(RM,L,upper=False)[0]
    z = torch.triangular_solve(z,torch.transpose(L,0,1),upper=True)[0]
    '''
    z = cupy_solve_sparse.apply(A.to(device),RM.to(device))
    z = torch.flatten(z).to(device)
    return z

