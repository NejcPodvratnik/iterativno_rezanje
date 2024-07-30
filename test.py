import torch
import numpy as np

x = torch.randn(2,3)
print(np.shape(x))
print(x)
x = x[x.nonzero(as_tuple=True)]
print(np.shape(x))
print(x)

all_alive_weights = torch.empty(0)
all_alive_weights = torch.cat((all_alive_weights,x), dim = 0)
all_alive_weights = torch.cat((all_alive_weights,all_alive_weights), dim = 0)
all_alive_weights = torch.cat((all_alive_weights,all_alive_weights), dim = 0)
all_alive_weights = torch.cat((all_alive_weights,all_alive_weights), dim = 0)
print(np.shape(all_alive_weights))
print(all_alive_weights)


