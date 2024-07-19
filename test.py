import torch

# Creating a sample tensor
tensor = torch.tensor([[0, 1, 0], 
                       [2, 0, 3], 
                       [0, 0, 4]])

# Extracting non-zero elements
alive = tensor[tensor.nonzero(as_tuple=True)]

print(alive)
