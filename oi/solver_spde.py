#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sparse_eye(size, val = torch.tensor(1.0)):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size).to(device)
    if len(val.size())==0:
        values = (val.expand(size)).to(device)
    else:
        values = val.to(device)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])).to(device)

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

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3, stochastic=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        self.stochastic = stochastic
        #self.correlate_noise = CorrelateNoise(input_size, 10)
        #self.regularize_variance = RegularizeVariance(input_size, 10)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        """
        if self.stochastic == True:
            z = torch.randn(input_.shape).to(device)
            z = self.correlate_noise(z)
            z = (z-torch.mean(z))/torch.std(z)
            #z = torch.mul(self.regularize_variance(z),self.correlate_noise(z))
        """
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        """
        if self.stochastic == False:
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((torch.add(input_,z), prev_hidden), 1)
        """

        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class ConvLSTM1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv1d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

def compute_WeightedLoss(x2,w):
    #  fix normalizing factor ( Sum w = 1 != w~ bool index)
    if len(list(w.size()))>0:
        x2_msk = (x2 * w[None, :, None, None])[:, w>0, ...]
    else:
        x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    if x2_num.sum() == 0:
        return torch.scalar_tensor(0., device=x2_num.device)
    # loss2 = x2_msk[x2_num].sum()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    return loss2

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()
 
    def forward(self,x,w,eps=0.):
        loss_ = torch.nansum( x**2 , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
 
    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

class Model_WeightedLorenzNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedLorenzNorm, self).__init__()
 
    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.log( 1. + eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

class Model_WeightedGMcLNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
 
    def forward(self,x,w,eps):

        loss_ = torch.nansum( 1.0 - torch.exp( - eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

def compute_WeightedL2Norm1D(x2,w):
    loss_ = torch.nansum(x2**2 , dim = 2)
    loss_ = torch.nansum( loss_ , dim = 0)
    loss_ = torch.nansum( loss_ * w )
    loss_ = loss_ / (torch.sum(~torch.isnan(x2)) / x2.shape[1] )
    
    return loss_

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self,ShapeData,periodicBnd=False,DimLSTM=0,rateDropout=0.,stochastic=False):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.dim_state  = 5*self.shape[0]
            else :
                self.dim_state  = DimLSTM
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False

        self.convLayer     = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.dropout = torch.nn.Dropout(rateDropout)
        self.stochastic=stochastic

        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0],self.dim_state,3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0],self.dim_state,3,stochastic=self.stochastic)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.dim_state, self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.dim_state, self.shape[0], (1,1), padding=0,bias=False))

        return torch.nn.Sequential(*layers)
    def _make_LSTMGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0],self.dim_state,3))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(ConvLSTM2d(self.shape[0],self.dim_state,3,stochastic=self.stochastic))

        return torch.nn.Sequential(*layers)

    def forward(self,hidden,cell,grad,gradnorm=1.0):

        # compute gradient
        grad  = grad / gradnorm
        grad  = self.dropout( grad )

        if self.PeriodicBnd == True :
            dB     = 7
            #
            grad_  = torch.cat((grad[:,:,grad.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_,None)
            else:
                hidden_  = torch.cat((hidden[:,:,grad.size(2)-dB:,:],hidden,hidden[:,:,0:dB,:]),dim=2)
                cell_    = torch.cat((cell[:,:,grad.size(2)-dB:,:],cell,cell[:,:,0:dB,:]),dim=2)
                hidden_,cell_ = self.lstm(grad_,[hidden_,cell_])

            hidden = hidden_[:,:,dB:grad.size(2)+dB,:]
            cell   = cell_[:,:,dB:grad.size(2)+dB,:]
        else:
            if hidden is None:
                hidden,cell = self.lstm(grad,None)
            else:
                hidden,cell = self.lstm(grad,[hidden,cell])

        grad = self.dropout( hidden )
        grad = self.convLayer( grad )

        return grad,hidden,cell


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData,dim_obs=1,dim_obs_channel=0,dim_state=0):
        super(Model_Var_Cost, self).__init__()
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs        = dim_obs
        if dim_state > 0 :
            self.dim_state      = dim_state
        else:
            self.dim_state      = ShapeData[0]
            
        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.dim_obs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dim_obs_channel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,ShapeData[0]))))
            self.dim_obs_channel  = ShapeData[0] * np.ones((self.dim_obs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,np.max(self.dim_obs_channel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.dim_state,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi
        
    def forward(self, dx, dy, square_root=False):
        if square_root==True:
            loss = self.alphaReg**2 * self.normPrior(dx,self.WReg**2,self.epsReg)
        else:
            loss = self.alphaReg**2 * dx
        '''
        if self.dim_obs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.dim_obs):
                loss +=  (
                    self.alphaObs[kk]**2 
                    * self.normObs(
                        dy[kk],
                        self.WObs[kk,0:dy[kk].size(1)]**2,
                        self.epsObs[kk]
                    )
                )
        '''
        loss2 = torch.mean(torch.sum(dy**2))
        print(loss2)
        loss += loss2
        return loss

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters 
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    NORMS = {
            'l1': Model_WeightedL1Norm,
            'l2': Model_WeightedL2Norm,
    }
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, shape_data,n_iter_grad,
                 square_root=False):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r         = phi_r
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        else:
            m_NormObs = self.NORMS[m_NormObs]()
        if m_NormPhi == None:    
            m_NormPhi = Model_WeightedL2Norm()
        else:
            m_NormPhi = self.NORMS[m_NormPhi]()
        self.shape_data = shape_data
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.square_root = square_root
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, shape_data, mod_H.dim_obs, 
                                            mod_H.dim_obs_channel)

        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        
    def forward(self, gt, x, yobs, mask, oi, estim_parameters=True):
        return self.solve(
            gt=gt,
            x_0=x,
            obs=yobs,
            mask = mask,
            oi = oi,
            estim_parameters = estim_parameters)

    def solve(self, gt, x_0, obs, mask, oi, estim_parameters):
        x_k = torch.mul(x_0,1.) 
        hidden = None
        cell = None 
        normgrad_ = 0.
        x_k_plus_1 = None
        if oi is None:
            cmp_loss = torch.ones(x_k.shape[0],self.n_grad,2)
        else:
            cmp_loss = torch.ones(x_k.shape[0],self.n_grad,3)        
        for k in range(self.n_grad):
            x_k_plus_1, cmp_loss_, hidden, cell, normgrad_, params = self.solver_step(gt, x_k, obs, mask, oi,
                                                                           hidden, cell, normgrad_,
                                                                           estim_parameters)
            x_k = torch.mul(x_k_plus_1,1.)
            for batch in range(len(cmp_loss)):
                cmp_loss[batch,k,:] = cmp_loss_[batch,:]

        return x_k_plus_1, cmp_loss, hidden, cell, normgrad_, params

    def solver_step(self, gt,  x_k, obs, mask, oi, hidden, cell,normgrad = 0.,estim_parameters=True):
        _, cmp_loss, var_cost_grad, params = self.var_cost(gt, x_k, obs, mask, oi, estim_parameters)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, cmp_loss,  hidden, cell, normgrad_, params

    def var_cost(self, gt, x, yobs, mask, oi, estim_params):
        dy = self.model_H(x,yobs,mask)
        n_b, n_t, n_x, n_y = dy.shape
        dy_new=list()
        for i in range(n_b):
            id_obs = torch.where(torch.flatten(mask[i])!=0.)[0]
            dyi = torch.index_select(torch.flatten(dy[i]), 0, id_obs).type(torch.FloatTensor).to(device)
            nb_obs = len(dyi)
            inv_R = 1e3*sparse_eye(nb_obs).type(torch.FloatTensor).to(device)
            iRdy = torch.sparse.mm(inv_R,torch.reshape(dyi,(nb_obs,1)))
            dyTiRdy = torch.matmul(torch.reshape(dyi,(1,nb_obs)),iRdy)
            dy_new.append(dyTiRdy[0,0])
        dy = torch.stack(dy_new)
        if self.square_root==True:
            phi = self.phi_r(x,estim_params=estim_params)
            dx = x - phi[0]
            params = phi[1]
            loss = self.model_VarCost(dx,dy,square_root=self.square_root)
        else:
            phi = self.phi_r(x,estim_params=estim_params)
            dx = phi[0]
            params = phi[1]
            #loss = torch.mean(self.model_VarCost(dx,dy,square_root=self.square_root))
            #print(dx)
            #print(dy)
            loss_OI = dy + dx
            loss = torch.mean(loss_OI)
        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        # return loss OI (loss) and MSE
        loss_mse = torch.tensor([compute_WeightedLoss((gt[i] - x[i]),torch.Tensor(np.ones(5)).to(device)) for i in range(len(gt))])
        loss_mse = loss_mse.to(device)

        # if using OI
        if oi is not None:
            loss_mseoi = torch.tensor([compute_WeightedLoss((oi[i] - x[i]),torch.Tensor(np.ones(5)).to(device)) for i in range(len(oi))])
            loss_mseoi = loss_mseoi.to(device)
            return loss, torch.hstack([torch.reshape(loss_mse,(n_b,1)), torch.reshape(loss_OI,(n_b,1)), torch.reshape(loss_mseoi,(n_b,1))]), var_cost_grad, params
        else:
            return loss, torch.hstack([torch.reshape(loss_mse,(n_b,1)), torch.reshape(loss_OI,(n_b,1)), ]), var_cost_grad, params

