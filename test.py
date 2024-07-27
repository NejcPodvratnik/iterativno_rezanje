import torch

# Creating a tensor of integers
tensor_int = torch.tensor([1, 2, 3, 4])

# Converting to float using .float() method
tensor_float = tensor_int.float().numpy()

print(tensor_float)
print(tensor_float.dtype)
