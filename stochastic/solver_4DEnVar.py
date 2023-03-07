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
import einops
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        stacked_inputs = stacked_inputs.reshape(batch_size,-1,1)
        gates = self.Gates(stacked_inputs)
        gates = torch.squeeze(gates,2)
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
    if len(list(w.size()))>0:
        x2_msk = (x2 * w[None, :, None, None])[:, w>0, ...]
    else:
        x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
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

class Model_L2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_L2Norm, self).__init__()

    def forward(self,x):
        loss_ = torch.nansum(x**2)
        return loss_

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self,ShapeData,DimLSTM=0,rateDropout=0.):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.dim_state  = 5*self.shape
            else :
                self.dim_state  = DimLSTM

        self.dropout = torch.nn.Dropout(rateDropout)

        self.convLayer = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.lstm = ConvLSTM1d(self.shape,self.dim_state,3)

    def _make_ConvGrad(self):
        layers = []
        layers.append(torch.nn.Conv1d(self.dim_state, self.shape, 1, padding=0,bias=False))
        return torch.nn.Sequential(*layers)

    def forward(self,hidden,cell,grad,gradnorm=1.0):

        # compute gradient
        grad  = grad / gradnorm
        grad  = self.dropout( grad )
        if hidden is None:
            hidden,cell = self.lstm(grad,None)
        else:
            hidden,cell = self.lstm(grad,[hidden,cell])
        grad = self.dropout( hidden )
        grad = self.convLayer( grad.reshape(grad.size()[0],-1,1) )
        grad = grad.squeeze(2)
        return grad,hidden,cell

# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormW, ShapeData,dim_obs=1,dim_obs_channel=0,dim_state=0, Ns=5):
        super(Model_Var_Cost, self).__init__()
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs        = dim_obs
        self.dim_ensemble = Ns
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
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.normObs   = m_NormObs
        self.normWeights = m_NormW
        
    def forward(self, dw, dy):

        loss = self.alphaReg**2 * self.normWeights(dw)
                
        if self.dim_obs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.dim_obs):
                loss +=  self.alphaObs[kk]**2 * self.normObs(dy[kk],self.WObs[kk,0:dy[kk].size(1)]**2,self.epsObs[kk])

        return loss

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters 
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    def __init__(self,phi_r,mod_H, m_Grad, m_NormObs, m_NormW, ShapeData,
                 n_iter_fourdenvar, n_iter_grad, Ns=10):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r = phi_r
        self.Ns = Ns

        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormW == None:    
            m_NormW = Model_L2Norm()
            
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormW, ShapeData,
                                            mod_H.dim_obs,mod_H.dim_obs_channel,
                                            Ns = self.Ns)

        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
            self.n_fourdenvar = int(n_iter_fourdenvar)
        
    def forward(self, x, yobs, mask):
        # w: weights of the members in the ensemble subspace
        # x: initial optimal state
        return self.solve(
            x_0=x,
            obs=yobs,
            mask = mask)

    def solve(self, x_0, obs, mask):
        batch_size = x_0.size()[0]
        # initial vector: unit vector of length Ns
        w_k = torch.nn.Parameter(torch.ones(batch_size,self.Ns).to(device), requires_grad=True)
        # initial ensemble: Ns replication of the initial state
        x_k = einops.repeat(x_0, 'b t h w -> b t h w k', k=self.Ns)

        for _ in range(self.n_fourdenvar):
        
            # ensemble mean
            xbar_k = torch.mean(x_k,dim=4,keepdim=True)
            # anomalies
            anom_k = torch.mul(x_k - xbar_k, np.sqrt(1./self.Ns))
            # weighting of the anomalies
            anom_k = anom_k * w_k.reshape(batch_size, 1, 1, 1, -1)
            # new optimal state
            x_k = torch.squeeze(xbar_k,dim=4)+torch.sum(anom_k,dim=4)

            # generate the ensemble from state_init
            x_k = torch.stack([self.phi_r(x_k) for _ in range(self.Ns)], dim=-1)

            hidden = None
            cell = None 
            normgrad_ = 0.
        
            # update the weights
            #for _ in range(self.n_grad):
            #    w_k, hidden, cell, normgrad_ = self.solver_step(w_k, x_k, obs, mask, hidden, cell, normgrad_)

        # return optimal weights and corresponding ensemble
        return w_k, x_k, hidden, cell, normgrad_

    def solver_step(self, w_k, x_k, obs, mask, hidden, cell,normgrad = 0.):
        var_cost, var_cost_grad = self.var_cost(w_k, x_k, obs, mask)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        w_k_plus_1 = w_k - grad
        return w_k_plus_1, hidden, cell, normgrad_

    def var_cost(self, w, x, yobs, mask):
        # In the 4DEnVar approach,
        # L(w) = l1||y-H(xbar-wXf)|| + l2||w||
        # x has dim Nb*Nt*Nx*Ny*Ns
        batch_size = x.size()[0]
        # ensemble mean
        xbar = torch.mean(x,dim=4,keepdim=True)
        # anomalies
        anom = torch.mul(x-xbar, np.sqrt(1./self.Ns))
        # weighting of the anomalies
        anom = anom * w.reshape(batch_size, 1, 1, 1, -1)
        # new optimal state
        xstar = torch.squeeze(xbar,dim=4)+torch.sum(anom,dim=4)
        # observation term
        dy = self.model_H(xstar,yobs,mask)
        loss = self.model_VarCost(w, dy)
        var_cost_grad = torch.autograd.grad(loss, inputs = [w, x], create_graph=True)[0]
        return loss, var_cost_grad
