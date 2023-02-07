import torch
import numpy as np

#tensor created directly from data
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(f"Data is: \n {data} \n")
print(f"Data to Tensor is: \n {x_data} \n")

#tensor created from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(f"NumPy Array is: \n {np_array} \n")
print(f"Tensor from NumPy Array is: \n {x_np} \n")

#tensor created from another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor from data: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor from data: \n {x_rand} \n")