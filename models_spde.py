import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import solver as NN_4DVar
from metrics import save_netcdf, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, plot_ensemble, maps_score
import pac

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Prior_SPDE(shape_data,dim_ae):
    
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
            H = torch.reshape(H,(n_b,2,2,self.n_y,self.n_x,self.n_t))
            H = torch.permute(H,(0,1,2,5,3,4))

            # kappa has shape (b,t,y,x)
            # m has shape (b,2,t,y,x)
            # H has shape (b,2,2,t,y,x)
            return kappa.float(), m.float(), H.float()

    class SPDE_pac(torch.nn.Module):

        def __init__(self,shape_data):
            super(SPDE_pac, self).__init__()
            self.n_t, self.n_y, self.n_x = shape_data
            self.f = 3 # filter_size
            self.pac = pac.PacConv2d(1, 1, self.f, padding=1)
            self.conv1   = torch.nn.Conv2d(1,1,(3,3),padding=1,bias=False)

        def K(self,g_ch):
            f = self.f
            # create filter K to apply on the parameters
            # w has shape [g_ch,f,f]
            w = torch.zeros((g_ch,f,f))
            # kappa
            w[0,1,1] = 1
            # H11
            w[1,1,1] = 2
            # inv_H11
            w[2,1,0] = -1
            w[2,1,2] = -1
            # H22
            w[3,1,1] = 2
            # inv_H22
            w[4,0,1] = -1
            w[4,2,1] = -1
            # H12
            w[5,0,0] = -1./2
            w[5,2,2] = -1./2
            w[5,0,2] = 1./2
            w[5,2,0] = 1./2
            # m_1
            w[6,1,0] = -1
            w[6,1,2] = 1
            # m_2
            w[7,0,1] = 1
            w[7,2,1] = -1

            return w.to(device)

        def forward(self, x, kappa, m, H, stochastic=True):  
            # x has shape (b,t,y,x)
            b = kappa.shape[0]
            t, g_ch =  self.n_t, 8 # channel sizes of input/output and number of guidance channels (m,H,kappa) 
            f, h, w = 3, self.n_y, self.n_x # filter size, batch size, input height and width

            H11 = H[:,0,0,:,:,:]
            H22 = H[:,1,1,:,:,:]
            H12 = H[:,0,1,:,:,:]
            inv_H11 = 1./H11
            inv_H22 = 1./H22
            m1 = m[:,1,:,:,:]
            m2 = m[:,0,:,:,:]

            # here, guide contains the SPDE parameters (b,t,c,y,x) with c=8
            guide = torch.stack((kappa,H11,inv_H11,H22,inv_H22,H12,m1,m2),dim=2)   
      
            # now let compute the pre-defined kernel K
            # K has shape (b,t,c,f,f,y,x)
            w_A = torch.einsum('ijk,btipq->btjkpq', (self.K(g_ch), guide))

            # add ideself.n_tity kernel (see st-SPDE Eqs)
            w_I =  torch.tensor([[0., 0., 0.,],
	                         [0., 1., 0.,],
	                         [0., 0., 0.,]]).view(1, 1, 3, 3, 1, 1).repeat(b, self.n_t, 1, 1, self.n_y, self.n_x).to(device)
            w = w_I + w_A

            # encoding of x=(x_T,...,x0)            
            tau = 1/torch.std(x)
            dt = 1
            x_spde = torch.zeros((b,self.n_t,self.n_y,self.n_x)).to(device)
            for k in range(self.n_t-1):
                if stochastic==True:
                    z = tau*np.sqrt(dt)*torch.randn([b,1,self.n_y,self.n_x]).to(device)
                    #x_spde[:,k,:,:,] = torch.squeeze(self.conv1(x[:,k+1,:,:].unsqueeze(dim=1))+z)
                    x_spde[:,k,:,:] = torch.squeeze(self.pac(x[:,k+1,:,:].unsqueeze(dim=1),None,
                                                    w[:,k+1,:,:,:,:].unsqueeze(dim=1))+z)
                else:
                    x_spde[:,k,:,:] = torch.squeeze(self.pac(x[:,k+1,:,:].unsqueeze(dim=1), None,
                                    w[:,k+1,:,:,:,:].unsqueeze(dim=1)))
            return x_spde

    return encode_param_CNN(shape_data, dim_ae), decode_param_CNN(shape_data), SPDE_pac(shape_data)

class Phi_r(torch.nn.Module):
    def __init__(self, shape_data, dim_ae, stochastic=False):
        super().__init__()
        self.stochastic = stochastic
        self.encoder, self.decoder, self.spde_pac = Prior_SPDE(shape_data, dim_ae)

    def forward(self, x):
        params = self.encoder(x)
        kappa, m, H = self.decoder(params)
        x = self.spde_pac(x, kappa, m, H, self.stochastic)
        return x

class Model_H(torch.nn.Module):
    def __init__(self, shape_data):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shape_data])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class Model_HwithSST(torch.nn.Module):
    def __init__(self, shape_data, dim=5):
        super(Model_HwithSST, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])
        self.conv11 = torch.nn.Conv2d(shape_data, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv21 = torch.nn.Conv2d(int(shape_data / 2), self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv_m = torch.nn.Conv2d(int(shape_data / 2), self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
        dyout1 = self.conv11(x) - self.conv21(y1)
        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]


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

