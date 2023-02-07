import torch
import numpy as np

#NOTE: Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other



#tensor to numpy array
t = torch.ones(5)
n = t.numpy()
print(f"Original tensor and numpy array: \nt: {t}")
print(f"n: {n}\n")

#a change in the tensor reflects in the NumPy array
t.add_(1)
print(f"The change in the tensor reflects in the numpy array also: \nt: {t}")
print(f"n: {n}\n\n\n")



#numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(f"Original tensor and numpy array: \nn: {n}")
print(f"t: {t}\n")

#a change in the numpy array reflects in the tensor also
np.add(n, 1, out=n)
print(f"The change in the numpy array reflects in the tensor also: \nt: {t}")
print(f"n: {n}")

