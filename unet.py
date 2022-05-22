#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:33:28 2022

@author: rfablet
"""
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

rateDropout = 0.2
padding_mode = 'reflect'

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect',activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if activation == 'relu':
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
        elif activation == 'tanh' :
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(mid_channels),
                    nn.Tanh(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(out_channels),
                    nn.Tanh(inplace=True) )
        elif activation == 'logsigmoid' :
            self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(mid_channels),
                    nn.LogSigmoid(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm2d(out_channels),
                    nn.LogSigmoid(inplace=True) )
        elif activation == 'bilin' :
            self.double_conv = DoubleConvBILIN(in_channels, mid_channels,padding_mode=padding_mode)

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_1D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect',activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        if activation == 'relu':
            self.double_conv = nn.Sequential(
                    nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm1d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
        elif activation == 'tanh' :
            self.double_conv = nn.Sequential(
                    nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm1d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(rateDropout),
                    nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True) )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvNoBN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(rateDropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvLeakyReLu(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(rateDropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvBILIN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,padding_mode='reflect'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv1  = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv21 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv22 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv23 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)
        self.conv24 = nn.Conv2d(2*mid_channels, out_channels, kernel_size=3, padding=1, bias=False,padding_mode=padding_mode)

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.relu( self.bn1(x1) )
        
        x11 = self.conv21(x1) 
        x12 = self.conv22(x1) 
        x13 = self.conv23(x1) 
        x1 = self.conv24( torch.cat((x11,x12*x13),dim=1) )
        
        x1 = self.bn2(x1)
        
        return x1

class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mid_channels=None,padding_mode='reflect',activation='relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels, mid_channels,padding_mode=padding_mode,activation='relu')
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down2Avg(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,mid_channels=None,padding_mode='reflect',activation='relu'):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels, mid_channels,padding_mode=padding_mode,activation='relu')
        )

    def forward(self, x):
        return self.avgpool_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_1D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,padding_mode='reflect',activation='relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            #nn.AvgPool2d(2),
            DoubleConv_1D(in_channels, out_channels,padding_mode=padding_mode,activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down4(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
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

class Up_1D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,padding_mode='reflect',activation='relu'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)#,padding_mode=padding_mode)
        self.conv = DoubleConv_1D(in_channels, out_channels,padding_mode=padding_mode,activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True,activation='relu'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels1 + in_channels2, out_channels,activation=activation)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
            self.conv = DoubleConv(2*in_channels2, out_channels,activation=activation)
            #self.conv = nn.Conv2d(2*in_channels2, out_channels, kernel_size=1, padding=0, bias=False)

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

class Up2Lin(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True,activation='relu'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels1 + in_channels2, out_channels,activation='relu')
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
            #self.conv = DoubleConv(2*in_channels2, out_channels)
            self.conv = nn.Conv2d(2*in_channels2, out_channels, kernel_size=1, padding=0, bias=False)

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


class Up4(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=4, stride=4)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 4, diffX - diffX // 4,
                        diffY // 4, diffY - diffY // 4])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class OutConv_1D(nn.Module):
    def __init__(self, in_channels, out_channels,padding_mode='reflect'):
        super(OutConv_1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1,padding_mode=padding_mode)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
    
class UNetAvg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetAvg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down2Avg(64, 128)
        self.down2 = Down2Avg(128, 256)
        self.down3 = Down2Avg(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
    
    
class UNet3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
    
class UNet4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet4_LeakyReLu(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4_LeakyReLu, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvLeakyReLu(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet4_NoBN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4_NoBN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvNoBN(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet4_Avg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet4_Avg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down2Avg(64, 128)
        self.down2 = Down2Avg(128, 256)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
class UNet5(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet5, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvBILIN(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet6(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet6, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConvBILIN(n_channels, 32)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
class UNet7(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet7, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
    
class UNet8(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet8, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet8_LeakyReLU(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet8_LeakyReLU, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvLeakyReLu(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet8_NoBN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet8_NoBN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvNoBN(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
    
class UNet8_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet8_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 50)
        self.down1 = Down2(50, 50, 50)
        self.down2 = Down2(50, 50, 50)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2(50, 50, 50, bilinear)
        self.up4 = Up2(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet8_3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet8_3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2(50, 50, 50)
        self.down2 = Down2(50, 50, 50)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2(50, 50, 50, bilinear)
        self.up4 = Up2(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2(50, 50, 100)
        self.down2 = Down2(50, 50, 100)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2Lin(50, 50, 50, bilinear)
        self.up4 = Up2Lin(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_Avg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_Avg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2Avg(50, 50, 100)
        self.down2 = Down2Avg(50, 50, 100)
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2Lin(50, 50, 50, bilinear)
        self.up4 = Up2Lin(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_AvgTanh(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_AvgTanh, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2Avg(50, 50, 100,activation='tanh')
        self.down2 = Down2Avg(50, 50, 100,activation='tanh')
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2Lin(50, 50, 50, bilinear)
        self.up4 = Up2Lin(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_AvgTanh2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_AvgTanh2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2Avg(50, 50, 100,activation='tanh')
        self.down2 = Down2Avg(50, 50, 100,activation='tanh')
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2(50, 50, 50, bilinear,activation='tanh')
        self.up4 = Up2(50, 50, 50, bilinear,activation='tanh')
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_AvgReLuBilin(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_AvgReLuBilin, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2Avg(50, 50, 100,activation='relu')
        self.down2 = Down2Avg(50, 50, 100,activation='relu')
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2(50, 50, 50, bilinear,activation='bilin')
        self.up4 = Up2(50, 50, 50, bilinear,activation='bilin')
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_Tanh(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_Tanh, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2(50, 50, 100,activation='tanh')
        self.down2 = Down2(50, 50, 100,activation='tanh')
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2Lin(50, 50, 50, bilinear)
        self.up4 = Up2Lin(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_LogSigmoid(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_LogSigmoid, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2(50, 50, 100,activation='logsigmoid')
        self.down2 = Down2(50, 50, 100,activation='logsigmoid')
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2Lin(50, 50, 50, bilinear)
        self.up4 = Up2Lin(50, 50, 50, bilinear)
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out

class UNet9_AvgReLuBilin2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet9_AvgReLuBilin2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #self.inc = DoubleConv(n_channels, 50)
        self.inc = nn.Conv2d(n_channels, 50-n_channels, kernel_size=(3,3), padding=1, bias=False,padding_mode=padding_mode)
        self.down1 = Down2Avg(50, 50, 100,activation='relu')
        self.down2 = Down2Avg(50, 50, 100,activation='relu')
        #self.down3 = Down(256, 512)
        #factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up2(50, 50, 50, bilinear,activation='bilin')
        self.up4 = Up2(50, 50, 50, bilinear,activation='bilin')
        #self.outc = OutConv(50, n_classes)
        self.outc = nn.Conv2d(50, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = torch.cat((x,x1),dim=1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out


class UNet_1D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv_1D(n_channels, 32)#,padding_mode=padding_mode,activation='relu')
        self.down1 = Down_1D(32, 64)
        self.down2 = Down_1D(64, 128)
        #self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        #self.down4 = Down(512, 1024 // factor)
        #self.up1 = Up(1024, 512 // factor, bilinear)
        #self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up_1D(128, 64 // factor, bilinear)
        self.up4 = Up_1D(64, 32, bilinear)
        self.outc = OutConv_1D(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #print(x3.shape)
        #print(x4.shape)
        #x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out
