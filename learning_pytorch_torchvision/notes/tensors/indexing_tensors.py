import torch
import numpy as np

tensor = torch.ones(4,4)
print("First row: ",tensor[0])
print("First column: ", tensor[:, 0])
print("Last column: ", tensor[..., -1])

tensor[:,1] = 0 #<-- set each entry in 2nd column to 0
print(tensor)