import torch
import numpy as np

#The Method {tensor.shape} returns a tuple of the tensor dimensions. Trailing comma specifies it is tuple in python. In the functions below, it determines the dimensionality of the output vector.
shape = (2,3,) #<-- rows X columns
tensor = torch.rand(shape)
print(f"Tensor: \n {tensor} \n")

#Attribute getters are methods of tensor class.
print(f"Shape of tensor: {tensor.shape} \n")
print(f"Datatpye of tensor: {tensor.dtype} \n")
print(f"Device that tensor is stored on: {tensor.device} \n") #<-- important method for seeing if you've moved tensors to gpu or not (can't without CUDA/ROCm)