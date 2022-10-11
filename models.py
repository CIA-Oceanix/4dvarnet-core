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
import unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLinUnit(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim, dw, dw2, dropout=0.):
        super(BiLinUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(dim_in, 2 * dim, (2 * dw + 1, 2 * dw + 1), padding=dw, bias=False)
        self.conv2 = torch.nn.Conv2d(2 * dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim, dim_out, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
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
    def __init__(self, dim_inp, dim_out, dim_ae, dw, dw2, ss, nb_blocks, rateDropout=0.):
        super(Encoder, self).__init__()

        self.nb_blocks = nb_blocks
        self.dim_ae = dim_ae
        # self.conv1HR  = torch.nn.Conv2d(dim_inp,self.dim_ae,(2*dw+1,2*dw+1),padding=dw,bias=False)
        # self.conv1LR  = torch.nn.Conv2d(dim_inp,self.dim_ae,(2*dw+1,2*dw+1),padding=dw,bias=False)
        self.pool1 = torch.nn.AvgPool2d(ss)
        print(dim_inp, dim_out, dim_ae, dw, dw2, ss, nb_blocks, rateDropout)
        self.conv_tr = torch.nn.ConvTranspose2d(dim_out, dim_out, (ss, ss), stride=(ss, ss), bias=False)

        # self.nn_tlr    = self.__make_ResNet(self.dim_ae,self.nb_blocks,rateDropout)
        # self.nn_hr     = self.__make_ResNet(self.dim_ae,self.nb_blocks,rateDropout)
        self.nn_lr = self.__make_BilinNN(dim_inp, dim_out, self.dim_ae, dw, dw2, self.nb_blocks, rateDropout)
        self.nn_hr = self.__make_BilinNN(dim_inp, dim_out, self.dim_ae, dw, dw2, self.nb_blocks, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)

    def __make_BilinNN(self, dim_inp, dim_out, dim_ae, dw, dw2, nb_blocks=2, dropout=0.):
        layers = []
        layers.append(BiLinUnit(dim_inp, dim_out, dim_ae, dw, dw2, dropout))
        for kk in range(0, nb_blocks - 1):
            layers.append(BiLinUnit(dim_ae, dim_out, dim_ae, dw, dw2, dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, xinp):
        ## LR component
        x_lr = self.nn_lr(self.pool1(xinp))
        x_lr = self.dropout(x_lr)
        x_lr = self.conv_tr(x_lr)

        # HR component
        x_hr = self.nn_hr(xinp)

        return x_lr + x_hr

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
    def __init__(self, shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr, stochastic=False,phi_param='unet1'):
        super().__init__()
        print(shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr, stochastic)
        self.stochastic = stochastic
        
        if phi_param == 'unet-std' :
            self.encoder = unet.UNet4(shape_data,shape_data,False)            
        else :
            self.encoder = Encoder(shape_data, shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr)
            
        #self.encoder = Encoder(shape_data, 2*shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shape_data, 10)
        self.regularize_variance = RegularizeVariance(shape_data, 10)

    def forward(self, x):
        white = True
        if self.stochastic == True:
            # pure white noise
            z = torch.randn([x.shape[0],x.shape[1],x.shape[2],x.shape[3]]).to(device)        
            # correlated noise with regularization of the variance
            # z = torch.mul(self.regularize_variance(x),self.correlate_noise(z))
            z = z/torch.std(x)
            x = self.encoder(x+z)
        else:
            x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder_OI(torch.nn.Module):
    def __init__(self, dim_inp, dim_out, dim_ae, dw, dw2, ss, nb_blocks, rateDropout=0.):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.dim_ae = dim_ae
        self.nn = self.__make_BilinNN(dim_inp, dim_out, self.dim_ae, dw, dw2, self.nb_blocks, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)

    def __make_BilinNN(self, dim_inp, dim_out, dim_ae, dw, dw2, nb_blocks=2, dropout=0.):
        layers = []
        layers.append(BiLinUnit(dim_inp, dim_out, dim_ae, dw, dw2, dropout))
        for kk in range(0, nb_blocks - 1):
            layers.append(BiLinUnit(dim_ae, dim_out, dim_ae, dw, dw2, dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, xinp):
        # HR component
        x = self.nn(xinp)
        return x

class Phi_r_OI(torch.nn.Module):
    def __init__(self, shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr, stochastic=False,phi_param='unet1'):
        super().__init__()
        self.stochastic = stochastic
        self.encoder = Encoder_OI(shape_data, shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shape_data, 10)
        self.regularize_variance = RegularizeVariance(shape_data, 10)

    def forward(self, x):
        white = True
        if self.stochastic == True:
            # pure white noise
            z = torch.randn([x.shape[0],x.shape[1],x.shape[2],x.shape[3]]).to(device)
            z = z/torch.std(x)
            x = self.encoder(x+z)
        else:
            x = self.encoder(x)
        x = self.decoder(x)
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
    def __init__(self, shape_data, dT=5, dim=5):
        super(Model_HwithSST, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])
        self.conv11 = torch.nn.Conv2d(shape_data, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv21 = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def extract_sst_feature(self,y1):
        y_feat = self.conv21(y1)
       
        return y_feat
        
    def extract_state_feature(self,x):
        x_feat = self.conv11(x)
        
        return x_feat

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
        x_feat = self.extract_state_feature(x)
                
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))
        #dyout1 = self.conv11(x) - self.conv21(y1)
        #dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]
    
class Model_HwithSSTBN(torch.nn.Module):
    def __init__(self,shape_data, dT=5,dim=5,width_kernel=3,padding_mode='reflect'):
        super(Model_HwithSSTBN, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        self.w_kernel = width_kernel

        self.bn_feat = torch.nn.BatchNorm2d(self.dim_obs_channel[1],track_running_stats=False)

        self.conv11 = torch.nn.Conv2d(shape_data, self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.conv21 = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=True,padding_mode=padding_mode)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def extract_sst_feature(self,y1):
        y_feat = self.bn_feat( self.conv21(y1) )
       
        return y_feat
        
    def extract_state_feature(self,x):
        x_feat = self.bn_feat( self.conv11(x) )
        
        return x_feat

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
                
        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]

class Model_HwithSSTBNandAtt(torch.nn.Module):
    def __init__(self,shape_data, dT=5,dim=5,width_kernel=3,padding_mode='reflect'):
        super(Model_HwithSSTBNandAtt, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        self.w_kernel = width_kernel

        self.bn_feat = torch.nn.BatchNorm2d(self.dim_obs_channel[1],track_running_stats=False)

        self.conv11 = torch.nn.Conv2d(shape_data, self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.conv21 = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        #self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=True,padding_mode=padding_mode)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

        self.conv_m   = torch.nn.Conv2d(self.dim_obs_channel[1],self.dim_obs_channel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        
        self.lam_obs_sst  = torch.nn.Parameter(torch.Tensor(1. * np.ones((1,self.dim_obs_channel[1]))))
        self.thr_obs_sst  = torch.nn.Parameter(torch.Tensor(0.3 * np.ones((1,self.dim_obs_channel[1]))))

    def extract_sst_feature(self,y1):
        y_feat = self.bn_feat( self.conv21(y1) )
       
        return y_feat
        
    def extract_state_feature(self,x):
        x_feat = self.bn_feat( self.conv11(x) )
        
        return x_feat

    def compute_w(self,dyout1):
        for kk in range(0,self.dim_obs_channel[1]):
            wkk = ( self.lam_obs_sst[0,kk] * dyout1[:,kk,:,:] ) **2 - self.thr_obs_sst[0,kk]**2
            wkk = wkk.view(-1,1,dyout1.size(2),dyout1.size(3))
            
            if kk == 0 :
                w = 1. * wkk
            else:
                w = torch.cat( (w,wkk) , dim = 1)
                
        return self.sigmoid( self.conv_m( - F.relu( w ) ) )
        

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
                
        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        w = self.compute_w( dyout1 )
        
        dyout1 = dyout1 * w

        return [dyout, dyout1]

class Model_HwithSSTBN_nolin_tanh(torch.nn.Module):
    def __init__(self,shape_data, dT=5,dim=5,width_kernel=3,padding_mode='reflect'):
        super(Model_HwithSSTBN_nolin_tanh, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        #print('..... # im obs sst : %d'%dim)
        self.w_kernel = width_kernel

        self.bn_feat = torch.nn.BatchNorm2d(self.dim_obs_channel[1],track_running_stats=False)

        self.convx11 = torch.nn.Conv2d(shape_data, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        self.convy11 = torch.nn.Conv2d(dT, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=True,padding_mode=padding_mode)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def extract_sst_feature(self,y1):
        y1     = self.convy12( torch.tanh( self.convy11(y1) ) )
        y_feat = self.bn_feat( self.convy22( torch.tanh( self.convy21( torch.tanh(y1) ) ) ) )
       
        return y_feat
        
    def extract_state_feature(self,x):
        x1     = self.convx12( torch.tanh( self.convx11(x) ) )
        x_feat = self.bn_feat( self.convx22( torch.tanh( self.convx21( torch.tanh(x1) ) ) ) )
        
        return x_feat


    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
                
        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]



class Model_HwithSSTBNAtt_nolin_tanh(torch.nn.Module):
    def __init__(self,shape_data, dT=5,dim=5,width_kernel=3,padding_mode='reflect'):
        super(Model_HwithSSTBNAtt_nolin_tanh, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        #print('..... # im obs sst : %d'%dim)
        self.w_kernel = width_kernel

        self.bn_feat = torch.nn.BatchNorm2d(self.dim_obs_channel[1],track_running_stats=False)

        self.convx11 = torch.nn.Conv2d(shape_data, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        self.convy11 = torch.nn.Conv2d(dT, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        #self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=True,padding_mode=padding_mode)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

        self.conv_m   = torch.nn.Conv2d(self.dim_obs_channel[1],self.dim_obs_channel[1],(1,1),padding=0,bias=True,padding_mode=padding_mode)
        
        self.lam_obs_sst  = torch.nn.Parameter(torch.Tensor(1. * np.ones((1,self.dim_obs_channel[1]))))
        self.thr_obs_sst  = torch.nn.Parameter(torch.Tensor(0.3 * np.ones((1,self.dim_obs_channel[1]))))

    def extract_sst_feature(self,y1):
        y1     = self.convy12( torch.tanh( self.convy11(y1) ) )
        y_feat = self.bn_feat( self.convy22( torch.tanh( self.convy21( torch.tanh(y1) ) ) ) )
       
        return y_feat
        
    def extract_state_feature(self,x):
        x1     = self.convx12( torch.tanh( self.convx11(x) ) )
        x_feat = self.bn_feat( self.convx22( torch.tanh( self.convx21( torch.tanh(x1) ) ) ) )
        
        return x_feat

    def compute_w(self,dyout1):
        for kk in range(0,self.dim_obs_channel[1]):
            wkk = ( self.lam_obs_sst[0,kk] * dyout1[:,kk,:,:] ) **2 - self.thr_obs_sst[0,kk]**2
            wkk = wkk.view(-1,1,dyout1.size(2),dyout1.size(3))
            
            if kk == 0 :
                w = 1. * wkk
            else:
                w = torch.cat( (w,wkk) , dim = 1)
                
        return self.sigmoid( self.conv_m( - F.relu( w ) ) )

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
                
        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        #dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))
        w = self.compute_w( dyout1 )
        
        dyout1 = dyout1 * w

        return [dyout, dyout1]

class Model_HwithSST_nolin_tanh(torch.nn.Module):
    def __init__(self,shape_data, dT=5,dim=5,width_kernel=3,padding_mode='reflect'):
        super(Model_HwithSST_nolin_tanh, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])

        #print('..... # im obs sst : %d'%dim)
        self.w_kernel = width_kernel

        self.convx11 = torch.nn.Conv2d(shape_data, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convx22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        self.convy11 = torch.nn.Conv2d(dT, 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy12 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy21 = torch.nn.Conv2d(self.dim_obs_channel[1], 2*self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)
        self.convy22 = torch.nn.Conv2d(2*self.dim_obs_channel[1], self.dim_obs_channel[1], (3, 3), padding=1, bias=False,padding_mode=padding_mode)


        self.conv_m = torch.nn.Conv2d(dT, self.dim_obs_channel[1], (3, 3), padding=1, bias=True,padding_mode=padding_mode)
        self.sigmoid = torch.nn.Sigmoid()  # torch.nn.Softmax(dim=1)

    def extract_sst_feature(self,y1):
        y1     = self.convy12( torch.tanh( self.convy11(y1) ) )
        y_feat = self.convy22( torch.tanh( self.convy21( torch.tanh(y1) ) ) )
       
        return y_feat
        
    def extract_state_feature(self,x):
        x1     = self.convx12( torch.tanh( self.convx11(x) ) )
        x_feat = self.convx22( torch.tanh( self.convx21( torch.tanh(x1) ) ) )
        
        return x_feat

    def forward(self, x, y, mask):
        dyout = (x - y[0]) * mask[0]

        y1 = y[1] * mask[1]
                
        x_feat = self.extract_state_feature(x)
        y_feat = self.extract_sst_feature(y1)
        dyout1 = x_feat - y_feat

        dyout1 = dyout1 * self.sigmoid(self.conv_m(mask[1]))

        return [dyout, dyout1]
class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.conv_gx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_gx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.conv_gy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
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
        super().__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)

