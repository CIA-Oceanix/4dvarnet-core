import numpy as np
import torch
import einops
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

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
    loss_ = torch.sum(x2**2 , dim = 3)
    loss_ = torch.sum( loss_ , dim = 2)
    loss_ = torch.sum( loss_ , dim = 0)
    loss_ = torch.sum( loss_ * w )
    loss_ = loss_ / (x2.shape[0]*x2.shape[2]*x2.shape[2])
    
    return loss_



# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdate2(torch.nn.Module):
    def __init__(self,ShapeData,GradType,periodicBnd=False,DimObs=1,dimObsChannel=np.array([0]),DimLSTM=0,rateDropout=0.):
        super(model_GradUpdate2, self).__init__()

        with torch.no_grad():
            self.GradType  = GradType
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.DimState  = 5*self.shape[0]
            else :
                self.DimState  = DimLSTM
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
        #self.compute_Grad  = Compute_Grad(ShapeData,GradType,DimObs,dimObsChannel)
        self.convLayer     = self._make_ConvGrad()
        #self.bn1           = torch.nn.BatchNorm2d(self.shape[0])
        #self.lstm            = self._make_LSTMGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)
        
        self.dropout = torch.nn.Dropout(rateDropout)

        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0],self.DimState,3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0],self.DimState,3)

        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((DimObs,1))))
        #self.AlphaL2Obs    = torch.nn.Parameter(torch.Tensor(np.ones((DimObs,))))
        self.alphaAE     = torch.nn.Parameter(torch.Tensor([1.]))
        if dimObsChannel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((DimObs,ShapeData[0]))))
            self.dimObsChannel  = ShapeData[0] * np.ones((DimObs,))
        else:
            self.dimObsChannel   = dimObsChannel
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((DimObs,np.max(dimObsChannel)))))
        self.WAE         = torch.nn.Parameter(torch.Tensor(np.ones(ShapeData[0],)))
    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.DimState, self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data            
            layers.append(torch.nn.Conv2d(self.DimState, self.shape[0], (1,1), padding=0,bias=False))

        return torch.nn.Sequential(*layers)
    def _make_LSTMGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0],self.DimState,3))
        elif len(self.shape) == 3: ## 2D Data            
            layers.append(ConvLSTM2d(self.shape[0],self.DimState,3))

        return torch.nn.Sequential(*layers)
 
    def forward(self,hidden,cell,grad,gradnorm=1.0):

        # compute gradient
        #grad = self.compute_Grad(x,dy,xpred)
        
        #grad = grad /self.ScaleGrad
        #grad = grad / torch.sqrt( torch.mean( grad**2 ) )
        #grad = self.bn1(grad)
        grad  = grad / gradnorm
        #grad  = self.dropout( grad )
          
        if self.PeriodicBnd == True :
            dB     = 7
            #
            grad_  = torch.cat((grad[:,:,x.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_,None)
            else:
                hidden_  = torch.cat((hidden[:,:,x.size(2)-dB:,:],hidden,hidden[:,:,0:dB,:]),dim=2)
                cell_    = torch.cat((cell[:,:,x.size(2)-dB:,:],cell,cell[:,:,0:dB,:]),dim=2)
                hidden_,cell_ = self.lstm(grad_,[hidden_,cell_])
                
            hidden = hidden_[:,:,dB:x.size(2)+dB,:]
            cell   = cell_[:,:,dB:x.size(2)+dB,:]
        else:
            if hidden is None:
                hidden,cell = self.lstm(grad,None)
            else:
                hidden,cell = self.lstm(grad,[hidden,cell])

        grad = self.dropout( hidden )
        grad = self.convLayer( grad )
        #grad = self.convLayer( hidden )
        return grad,hidden,cell


class Solver_Grad_4DVarNN(nn.Module):
    def __init__(self ,phi_r,mod_H, m_Grad, ShapeData,NiterGrad,GradType,OptimType,InterpFlag=False,periodicBnd=False,DimGrad=0,rateDropout=0.):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r   = phi_r
        self.model_H    = mod_H
        self.model_Grad = m_Grad
        self.OptimType  = OptimType
        self.GradType   = GradType
        self.InterpFlag = InterpFlag
        self.periodicBnd = periodicBnd
        self.dimObsChannel = mod_H.dimObsChannel
        self.DimObs = mod_H.DimObs
        
        with torch.no_grad():
            self.OptimType = OptimType
            self.NGrad     = int(NiterGrad)

        

    def forward(self, x, yobs, mask):
        return self.solve(
            x_0=x,
            obs=yobs,
            mask = mask)

    def solve(self, x_0, obs, mask):
        x_k = torch.mul(x_0,1.) 
        hidden = None
        cell = None 
        for _ in range(self.NGrad):
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask,hidden, cell)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1, hidden, cell, normgrad_

    def solver_step(self, x_k, obs, mask, hidden, cell):
        var_cost, var_cost_grad= self.var_cost(x_k, obs, mask)
        normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 ) )
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.NGrad
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_

    def var_cost(self , x, yobs, mask):
        dy = self.model_H(x,yobs,mask)
        xpred = self.phi_r(x)
        loss = self.model_Grad.alphaAE**2 * compute_WeightedLoss(xpred-x,self.model_Grad.WAE**2)
        if self.DimObs == 1 :
            loss +=  self.model_Grad.alphaObs[0]**2 * compute_WeightedLoss(dy,self.model_Grad.WObs[0,:]**2)
        else:
            for kk in range(0,self.DimObs):
                loss +=  self.model_Grad.alphaObs[kk]**2 * compute_WeightedLoss(dy[kk],self.model_Grad.WObs[kk,0:dy[kk].size(1)]**2)
        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return loss, var_cost_grad