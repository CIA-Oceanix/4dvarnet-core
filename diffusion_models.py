#!/usr/bin/env python
import numpy as np
import torch
import torch.nn.functional as F
from models import Decoder


class GaussianFourierProjection(torch.nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization.
        # These weights are fixed during optimization and are not trainable.
        self.W = torch.nn.Parameter(torch.randn(embed_dim // 2) * scale,
                                    requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(torch.nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class BiLinUnit(torch.nn.Module):

    def __init__(self, dim_in, dim_out, dim, dw, dw2, dropout=0.):
        embed_dim = 512  # Valeur à tester
        super(BiLinUnit, self).__init__()
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            torch.nn.Linear(embed_dim, embed_dim))
        self.conv1 = torch.nn.Conv2d(dim_in,
                                     2 * dim, (2 * dw + 1, 2 * dw + 1),
                                     padding=dw,
                                     bias=False)
        self.dense1 = Dense(embed_dim, 2 * dim)
        self.conv2 = torch.nn.Conv2d(2 * dim,
                                     dim, (2 * dw2 + 1, 2 * dw2 + 1),
                                     padding=dw2,
                                     bias=False)
        self.dense2 = Dense(embed_dim, dim)
        self.conv3 = torch.nn.Conv2d(2 * dim,
                                     dim_out, (2 * dw2 + 1, 2 * dw2 + 1),
                                     padding=dw2,
                                     bias=False)
        self.dense3 = Dense(embed_dim, dim_out)
        self.bilin0 = torch.nn.Conv2d(dim,
                                      dim, (2 * dw2 + 1, 2 * dw2 + 1),
                                      padding=dw2,
                                      bias=False)
        self.bilin1 = torch.nn.Conv2d(dim,
                                      dim, (2 * dw2 + 1, 2 * dw2 + 1),
                                      padding=dw2,
                                      bias=False)
        self.bilin2 = torch.nn.Conv2d(dim,
                                      dim, (2 * dw2 + 1, 2 * dw2 + 1),
                                      padding=dw2,
                                      bias=False)
        self.dropout = torch.nn.Dropout(dropout)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, xin, k):
        embed = self.act(self.embed(k))
        x = self.conv1(xin) + self.dense1(embed)
        x = self.dropout(x)
        x = self.conv2(F.relu(x)) + self.dense2(embed)
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)), dim=1)
        x = self.dropout(x)
        x = self.conv3(x) + self.dense3(embed)
        # x = F.normalize(x)
        return x


class BiLinUnit2(torch.nn.Module):
    """ Test for a more complexe unet architecture """

    def __init__(self,
                 dim_in,
                 dim_out,
                 dim,
                 dw,
                 dw2,
                 dropout=0.,
                 embed_dim=512,
                 channels=[32, 64, 1228, 256]):
        super(BiLinUnit2, self).__init__()
        self.embed = torch.nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            torch.nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = torch.nn.Conv2d(dim_in,
                                     channels[0], (2 * dw + 1, 2 * dw + 1),
                                     padding=dw,
                                     bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.conv2 = torch.nn.Conv2d(channels[0],
                                     channels[1], (2 * dw + 1, 2 * dw + 1),
                                     padding=dw,
                                     bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.conv3 = torch.nn.Conv2d(channels[1],
                                     channels[2], (2 * dw + 1, 2 * dw + 1),
                                     padding=dw,
                                     bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.conv4 = torch.nn.Conv2d(channels[2],
                                     channels[3], (2 * dw + 1, 2 * dw + 1),
                                     padding=dw,
                                     bias=False)
        self.dense4 = Dense(embed_dim, channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = torch.nn.ConvTranspose2d(channels[3],
                                               channels[2],
                                               (2 * dw2 + 1, 2 * dw2 + 1),
                                               padding=dw2,
                                               bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tconv3 = torch.nn.ConvTranspose2d(channels[2] + channels[2],
                                               channels[1],
                                               (2 * dw2 + 1, 2 * dw2 + 1),
                                               padding=dw2,
                                               bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tconv2 = torch.nn.ConvTranspose2d(channels[1] + channels[1],
                                               channels[0],
                                               (2 * dw2 + 1, 2 * dw2 + 1),
                                               padding=dw2,
                                               bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tconv1 = torch.nn.ConvTranspose2d(channels[0] + channels[0],
                                               dim_out,
                                               (2 * dw2 + 1, 2 * dw2 + 1),
                                               padding=dw2,
                                               bias=False)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, k):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(k))
        # Encoding path
        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(h1)
        h1 = self.dropout(h1)
        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(h2)
        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(h3)
        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        # h = self.tconv3(h3)
        h += self.dense6(embed)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        # h = self.tconv2(h2)
        h += self.dense7(embed)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        return h


class Encoder_dm(torch.nn.Module):

    def __init__(self,
                 dim_inp,
                 dim_out,
                 dim_ae,
                 dw,
                 dw2,
                 ss,
                 nb_blocks,
                 rateDropout=0.):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.dim_ae = dim_ae
        # self.nn = self.__make_BilinNN(dim_inp, dim_out, self.dim_ae, dw, dw2,
        #                               self.nb_blocks, rateDropout)
        self.nn = BiLinUnit2(dim_inp, dim_out, dim_ae, dw, dw2, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)

    def forward(self, xinp, k):
        # HR component
        x = self.nn(xinp, k)
        return x


class Unet_dm(torch.nn.Module):
    """ Architecture of unet in diffusion model for the 4dvarnet """

    def __init__(self,
                 shape_data,
                 dimAE,
                 dw,
                 dw2,
                 ss,
                 nb_blocks,
                 rateDr,
                 stochastic=False):
        super().__init__()
        self.encoder = Encoder_dm(shape_data, shape_data, dimAE, dw, dw2, ss,
                                  nb_blocks, rateDr)
        self.decoder = Decoder()

    def forward(self, x, k):
        x = self.encoder(x, k)
        x = self.decoder(x)
        return x
