from oi.scipy_sparse_tools import *

def pow_diff_operator(A,pow,sparse=False):
    B=A
    for i in range(pow-1):
        if sparse==False:
            B = torch.matmul(B,A)
        else:
            B = spspmm(B,A)
    B=(1./2)*(B.t()+B)
    return B

def get_boundary_index(Nx,Ny):
    # bottom
    border = np.arange(0,Nx).tolist()
    # top
    border.extend(np.arange(Nx*(Ny-1),Nx*Ny).tolist())
    # left
    border.extend(np.arange(Nx,Nx*(Ny-1),Nx).tolist())
    # right
    border.extend(np.arange(2*Nx-1,Nx*(Ny-1),Nx).tolist())
    # sort border
    border.sort()
    border = np.array(border)
    return border
    #return torch.from_numpy(border).to(device)

def Gamma(tensor):
    return(torch.tensor(tensor,dtype=torch.float).lgamma().exp())

def regularize_variance(nu, kappa):
    # nu is an integer
    # kappa is a nbNodes-tensor
    d = 2
    pi = torch.acos(torch.zeros(1)).item() * 2
    return( ((Gamma(nu+d/2)*((4*pi)**(d/2))*(kappa**(2*nu))) / Gamma(nu) )**(1/2) )

def init_BS_basis(Nx,Ny,Nt,Nfbx,Nfby,Nfbt):
        # 3D B-splines basis functions
        bspl_basis_alongX = BSpline(domain_range=[-1,Nx+1], n_basis=Nfbx, order=3)
        bspl_basis_alongY = BSpline(domain_range=[-1,Ny+1], n_basis=Nfby, order=3)
        # time domain of definition: [t-Ntdt,t]
        bspl_basis_alongT = BSpline(domain_range=[-1*Nt,1], n_basis=Nfbt, order=3)
        # compute 3D B-splines basis functions
        grid = torch.reshape(torch.stack(torch.meshgrid([torch.arange(Nx),
                                                     torch.arange(Ny),
                                                     torch.arange(Nt)])),
                                              (3,Nx*Ny*Nt))
        bX=torch.from_numpy(bspl_basis_alongX(grid[0])[:,:,-1])
        bY=torch.from_numpy(bspl_basis_alongY(grid[1])[:,:,-1])
        bT=torch.from_numpy(bspl_basis_alongT(grid[2])[:,:,-1])
        bXY= torch.stack([ torch.einsum('i,j->ij',
                       bX[:,i],
                       bY[:,i]).reshape(bX.shape[0]*bY.shape[0]) for i in range(bX.shape[1]) ])
        bXY = torch.transpose(bXY,0,1)
        bXYT= torch.stack([ torch.einsum('i,j->ij',
                       bXY[:,i],
                       bT[:,i]).reshape(bXY.shape[0]*bT.shape[0]) for i in range(bXY.shape[1]) ])
        bXY = torch.transpose(bXY,0,1)
        return bXYT.to(device)

def DiffOperator(Nx, Ny, dx, dy, m, H, kappa):
# kappa is 1*(Ny*Nx)
# m is 2*(Ny*Nx)
# H is 2*2*(Ny*Nx)

    nbNodes = Nx * Ny 
    indices = torch.arange(0,nbNodes).to(device)
    ## Voisin à droite Points de la grille concernés : Retrait du bord droit
    index = torch.where(torch.fmod(indices+1,Nx) != 0)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2) + m[0, indicesVoisins]/dx))
    elif ( (m is None) and (H is not None) ):
        k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2)))
    elif ( (m is not None) and (H is None) ):
        k1 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + 1,
                          -1 * m[0, indicesVoisins]/dx))
    ## Voisin à gauche Points de la grille concernés : Retrait du bord gauche
    index = torch.where(torch.fmod(indices+1,Nx) != 1)[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2) - m[0, indicesVoisins]/dx))
    elif ( (m is None) and (H is not None) ):
        k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          -1 * (H[0, 0, indicesVoisins])/(dx**2)))
    elif ( (m is not None) and (H is None) ):
        k2 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - 1,
                          m[0, indicesVoisins]/dx))
    ## Voisin du haut Points de la grille concernés : Retrait du bord haut
    index = torch.where((indices+1) <= (Ny-1)*Nx )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                          -1 * (H[1, 1, indicesVoisins])/(dy**2) + m[1, indicesVoisins]/dy))
    elif ( (m is None) and (H is not None) ):
        k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                         -1 * (H[1, 1, indicesVoisins])/(dy**2)))
    elif ( (m is not None) and (H is None) ):
        k3 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx,
                          -1 * m[1, indicesVoisins]/dy))
    ## Voisin du bas Points de la grille concernés : Retrait du bord bas
    index = torch.where((indices+1) >= (Nx+1) )[0]
    indicesVoisins = torch.index_select(indices,0,index)
    if ( (m is not None) and (H is not None) ):
        k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx, 
                          -1 * (H[1, 1, indicesVoisins])/(dy**2) - m[1, indicesVoisins]/dy))
    elif ( (m is None) and (H is not None) ):
        k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx,
                          -1 * (H[1, 1, indicesVoisins])/(dy**2)))
    elif ( (m is not None) and (H is None) ):
        k4 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx, 
                          m[1, indicesVoisins]/dy))
    ## Point central
    if H is not None:
        if torch.is_tensor(kappa):
            k5 = torch.stack((indices.float(), indices.float(),
                             kappa[0,indices]**2 + 2 * (H[0, 0, indices]/(dx**2) + H[1, 1, indices]/(dy**2))))
        else:
            k5 = torch.stack((indices.float(), indices.float(),
                            kappa**2 + 2 * (H[0, 0, indices]/(dx**2) + H[1, 1, indices]/(dy**2))))
    else:
        if torch.is_tensor(kappa):
            k5 = torch.stack((indices.float(), indices.float(),
                              (kappa[0,indices]**2 + 2 * (1./(dx**2) + 1./(dy**2)))*torch.ones(len(indices)).to(device )))
        else:
            k5 = torch.stack((indices.float(), indices.float(), 
                              (kappa**2 + 2 * (1./(dx**2) + 1./(dy**2)))*torch.ones(len(indices)).to(device) ))
    if H is not None: 
        ## Voisin en haut à droite Points de la grille concernés : Retrait du bord haut et droit
        index = torch.where( (torch.fmod(indices+1,Nx) != 0) & ((indices+1)<= (Ny-1)*Nx) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k6 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx+1,
                         (H[0, 1, indicesVoisins] + H[1, 0, indicesVoisins])/(4*dx*dy)))
        ## Voisin en haut à gauche Points de la grille concernés : Retrait du bord haut et gauche
        index = torch.where( (torch.fmod(indices+1,Nx) != 1) & ((indices+1)<= (Ny-1)*Nx) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k7 = torch.stack((indicesVoisins.float(), indicesVoisins.float() + Nx-1, -1*(H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(4*dx*dy)))
        ## Voisin en bas à droite Points de la grille concernés : Retrait du bord bas et droit
        index = torch.where( (torch.fmod(indices+1,Nx) != 0) & ((indices+1)>=(Nx+1)) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k8 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx+1, -1*(H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(4*dx*dy)))
        ## Voisin en bas à gauche Points de la grille concernés : Retrait du bord bas et gauche
        index = torch.where( (torch.fmod(indices+1,Nx) != 1) & ((indices+1)>=(Nx+1)) )[0]
        indicesVoisins = torch.index_select(indices,0,index)
        k9 = torch.stack((indicesVoisins.float(), indicesVoisins.float() - Nx-1, (H[0, 1, indicesVoisins] + H[1, 0,indicesVoisins])/(4*dx*dy)))
        ## Tous les voisins
        k = torch.cat((k1, k2, k3, k4, k5, k6, k7, k8, k9),dim=1)
    else:
        ## Tous les voisins
        k = torch.cat((k1, k2, k3, k4, k5),dim=1)
    return(torch.sparse.FloatTensor(k[0:2].long(), k[2], torch.Size([nbNodes,nbNodes])))
    #return torch.sparse_coo_tensor(k[0:2].long(), k[2], torch.Size([nbNodes,nbNodes]))

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

