import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
#from spde import *
#from scipy_sparse_tools import *
import solver as NN_4DVar
from metrics import save_netcdf, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, plot_ensemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLinUnit(torch.nn.Module):
    def __init__(self, dimIn, dimOut, dim, dW, dW2, dropout=0.):
        super(BiLinUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(dimIn, 2 * dim, (2 * dW + 1, 2 * dW + 1), padding=dW, bias=False)
        self.conv2 = torch.nn.Conv2d(2 * dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim, dimOut, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, xin):
        x = self.conv1(xin)
        x = self.dropout(x)
        x = self.conv2(F.relu(x))
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)), dim=1)
        x = self.dropout(x)
        x = self.conv3(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, dimInp, dimOut, dimAE, dW, dW2, sS, nbBlocks, rateDropout=0.):
        super(Encoder, self).__init__()

        self.NbBlocks = nbBlocks
        self.DimAE = dimAE
        # self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        # self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        self.pool1 = torch.nn.AvgPool2d(sS)
        self.convTr = torch.nn.ConvTranspose2d(dimOut, dimOut, (sS, sS), stride=(sS, sS), bias=False)

        # self.NNtLR    = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        # self.NNHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        self.NNLR = self.__make_BilinNN(dimInp, dimOut, self.DimAE, dW, dW2, self.NbBlocks, rateDropout)
        self.NNHR = self.__make_BilinNN(dimInp, dimOut, self.DimAE, dW, dW2, self.NbBlocks, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)

    def __make_BilinNN(self, dimInp, dimOut, dimAE, dW, dW2, Nb_Blocks=2, dropout=0.):
        layers = []
        layers.append(BiLinUnit(dimInp, dimOut, dimAE, dW, dW2, dropout))
        for kk in range(0, Nb_Blocks - 1):
            layers.append(BiLinUnit(dimAE, dimOut, dimAE, dW, dW2, dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, xinp):
        ## LR component
        xLR = self.NNLR(self.pool1(xinp))
        xLR = self.dropout(xLR)
        xLR = self.convTr(xLR)

        # HR component
        xHR = self.NNHR(xinp)

        return xLR + xHR

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return torch.mul(1., x)


class CorrelateNoise(torch.nn.Module):
    def __init__(self, shape_data, dim_cn):
        super(CorrelateNoise, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_cn, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_cn, 2 * dim_cn, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_cn, shape_data, (3, 3), padding=1, bias=False)

    def forward(self, w):
        w = self.conv1(F.relu(w)).to(device)
        w = self.conv2(F.relu(w)).to(device)
        w = self.conv3(w).to(device)
        return w


class RegularizeVariance(torch.nn.Module):
    def __init__(self, shape_data, dim_rv):
        super(RegularizeVariance, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_rv, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_rv, 2 * dim_rv, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_rv, shape_data, (3, 3), padding=1, bias=False)

    def forward(self, v):
        v = self.conv1(F.relu(v)).to(device)
        v = self.conv2(F.relu(v)).to(device)
        v = self.conv3(v).to(device)
        return v

class Phi_r(torch.nn.Module):
    def __init__(self, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr, stochastic=False):
        super(Phi_r, self).__init__()
        self.stochastic = stochastic
        if self.stochastic==True:
            self.encoder = Encoder(shapeData, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr)
        else:
            self.encoder = Encoder(shapeData, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shapeData, 10)
        self.regularize_variance = RegularizeVariance(shapeData, 10)
        self.Nx = 200
        self.Ny = 200
        #A = DiffOperator_Isotropic(self.Nx, self.Ny, dx=1, dy=1, kappa=1./3)
        #self.B = pow_diff_operator(A,2,sparse=True)

    def forward(self, x):
        white = True
        if self.stochastic == True:
            # pure white noise
            z = torch.randn([x.shape[0],x.shape[1],x.shape[2],x.shape[3]]).to(device)
            # correlated noise with regularization of the variance
            # z = torch.mul(self.regularize_variance(x),self.correlate_noise(z))
            # z = z/torch.std(z)
            # print(stats.describe(z.detach().cpu().numpy()))
            if white == False:
                print('toto')
                z = torch.stack( 
                         [ torch.stack([torch.reshape(SPDE_spatial_simulation(self.B, dx=1, dy=1),
                                       (self.Nx,self.Ny)) for j in range(x.shape[1])]) for i in range(x.shape[0])])
                print(z.size())
            x = self.encoder(x+z)
            #x = self.encoder(torch.cat([x,z],dim=1))
        else:
            x = self.encoder(x)
        x = self.decoder(x)
        return x

class Model_H(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shapeData])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class Model_HwithSST(torch.nn.Module):
    def __init__(self, shapeData, dim=5):
        super(Model_HwithSST, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shapeData, dim])
        self.conv11 = torch.nn.Conv2d(shapeData, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv21 = torch.nn.Conv2d(int(shapeData / 2), self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.convM = torch.nn.Conv2d(int(shapeData / 2), self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.S = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
        dyout1 = self.conv11(x) - self.conv21(y1)
        dyout1 = dyout1 * self.S(self.convM(mask[1]))

        return [dyout, dyout1]

class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)


        #self.eps=10**-6
        self.eps=0.

    def forward(self, im):

        if im.size(1) == 1:
            G_x = self.convGx(im)
            G_y = self.convGy(im)
            G = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2) + self.eps)
        else:

            for kk in range(0, im.size(1)):
                G_x = self.convGx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                G_y = self.convGy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                G_x = G_x.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                G_y = G_y.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                nG = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2)+ self.eps)

                if kk == 0:
                    G = nG.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                else:
                    G = torch.cat((G, nG.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
        return G

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)

