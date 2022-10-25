import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
import oi.xp_gp.solver as NN_4DVar
from oi.xp_gp.metrics import save_netcdf, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, save_loss, save_grads

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
    def __init__(self, shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr, stochastic=False):
        super().__init__()
        self.stochastic = stochastic
        self.encoder = Encoder(shape_data, shape_data, DimAE, dw, dw2, ss, nb_blocks, rateDr)
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
    def __init__(self, shape_data, dim=5):
        super(Model_HwithSST, self).__init__()

        self.dim_obs = 2
        self.dim_obs_channel = np.array([shape_data, dim])
        self.conv11 = torch.nn.Conv2d(shape_data, self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv21 = torch.nn.Conv2d(int(shape_data / 2), self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.conv_m = torch.nn.Conv2d(int(shape_data / 2), self.dim_obs_channel[1], (3, 3), padding=1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

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


#UNet model classes below
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2, used for UNet implentation"""

    def __init__(self, in_channels, out_channels, mid_channels=None,rateDropout=0.,padding_mode='reflect',activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if activation == 'relu':
            self.double_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    torch.nn.BatchNorm2d(mid_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(rateDropout),
                    torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True)
                )
        elif activation == 'tanh' :
            self.double_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    torch.nn.BatchNorm2d(mid_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(rateDropout),
                    torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True) )
        elif activation == 'logsigmoid' :
            self.double_conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    torch.nn.BatchNorm2d(mid_channels),
                    torch.nn.LogSigmoid(inplace=True),
                    torch.nn.Dropout(rateDropout),
                    torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.LogSigmoid(inplace=True) )

    def forward(self, x):
        return self.double_conv(x)

class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv, used for encoder for UNet"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            #nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    '''classic UNet model for comparison. Taken from
    https://github.com/CIA-Oceanix/4dvarnet-forecast/blob/35f55997b40324b2b89eacbd73889962f9a9bd0f/unet.py#L229
    shrink factor reduces the number of conv layers, must be power of 2'''
    def __init__(self, n_channels, n_classes, dropout_rate, bilinear=False, shrink_factor = 2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.inc = DoubleConv(n_channels, 64//shrink_factor)
        self.down1 = Down(64//shrink_factor, 128//shrink_factor)
        self.down2 = Down(128//shrink_factor, 256//shrink_factor)
        self.down3 = Down(256//shrink_factor, 512//shrink_factor)
        factor = 2 if bilinear else 1

        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512//shrink_factor, 256//(shrink_factor * factor), bilinear)
        self.up3 = Up(256//shrink_factor, 128//(shrink_factor * factor), bilinear)
        self.up4 = Up(128//shrink_factor, 64//shrink_factor, bilinear)
        self.outc = OutConv(64//shrink_factor, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.dropout(self.down1(x1))
        x3 = self.dropout(self.down2(x2))
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)

        #x = self.up2(x4, x3)
        x = self.dropout(self.up3(x3, x2))
        x = self.dropout(self.up4(x, x1))
        out = self.outc(x)

        return out

class Phi_r_UNet(torch.nn.Module):
    '''This is a UNet phi_r replacement'''
    def __init__(self, shape_data, dropout_rate =0., stochastic=False, shrink_factor=2):
        super().__init__()
        self.stochastic = stochastic
        self.unet = UNet(shape_data, shape_data, dropout_rate, False, shrink_factor)
        self.correlate_noise = CorrelateNoise(shape_data, 10)
        self.regularize_variance = RegularizeVariance(shape_data, 10)

    def forward(self, x):
        white = True
        if self.stochastic == True:
            # pure white noise
            z = torch.randn([x.shape[0],x.shape[1],x.shape[2],x.shape[3]]).to(device)
            z = z/torch.std(x)
            return self.unet(x+z)
        else:
            return self.unet(x)

#Classes for linear Phi_r
#These classes replace the bilinear units of the normal Phi_rOI with linear layers
class Phi_r_OI_linear(torch.nn.Module):
    def __init__(self, shape_data, DimAE,  nb_blocks, stochastic=False):
        super().__init__()
        self.stochastic = stochastic
        self.encoder = Encoder_OI_linear(shape_data, shape_data, DimAE, nb_blocks)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shape_data[0], 10)
        self.regularize_variance = RegularizeVariance(shape_data[0], 10)

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

class Encoder_OI_linear(torch.nn.Module):
    def __init__(self, dim_inp, dim_out, dim_ae, nb_blocks):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.dim_ae = dim_ae
        self.dim_out = dim_out
        self.nn = self.__make_linear_blocks(dim_inp, dim_out, dim_ae, nb_blocks)
 


    def __make_linear_blocks(self, dim_inp, dim_out, dim_ae,  nb_blocks=2):
        total_size = dim_inp[0]*dim_inp[1]*dim_inp[2]
        layers = []
        layers.append(torch.nn.Linear(total_size,dim_ae))
        for kk in range(0, nb_blocks-1):
            layers.append(torch.nn.Linear(dim_ae, dim_ae))
        layers.append(torch.nn.Linear(dim_ae, total_size))
        return torch.nn.Sequential(*layers)

    def forward(self, xinp):
        batch_size = xinp.shape[0]
        x = torch.flatten(xinp, 1,-1)
        x = self.nn(x)
        x = torch.reshape(x,(batch_size, self.dim_out[0], self.dim_out[1], self.dim_out[2]))
        return x
