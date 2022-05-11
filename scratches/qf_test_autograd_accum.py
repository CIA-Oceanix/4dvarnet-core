import torch
from torch import nn
import torch.optim

mod = nn.Linear(4, 6)

inp1 = torch.rand(2, 10, 4, requires_grad=True)
inp2 = torch.rand(2, 10, 4, requires_grad=True)

opt = torch.optim.SGD(mod.parameters(), lr=0.)


# test_1
opt.zero_grad()
out1 = mod(inp1)
print(1, list(mod.parameters())[0].grad)
gout1 = torch.autograd.grad(out1.sum(), inp1, create_graph=True)
print(2, list(mod.parameters())[0].grad)

inp2 = inp1 - 0.1 * gout1[0]


out2 = mod(inp2)
print(1, list(mod.parameters())[0].grad)
gout2 = torch.autograd.grad(out2.sum(), inp2, create_graph=True)
print(2, list(mod.parameters())[0].grad)


