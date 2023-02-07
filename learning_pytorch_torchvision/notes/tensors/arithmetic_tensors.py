import torch
import numpy as np

tensor = torch.ones(4,4)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor) #<-- creates randomly filled tensors of same dimensions as {tensor} to be filled with the output from {torch.matmul()}
print(y3)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor) #<-- creates randomly filled tensors of same dimensions as {tensor} to be filled with output from {torch.mul()}
print(z3)
torch.mul(tensor, tensor, out=z3)
print(z3)