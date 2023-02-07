import torch
import numpy as np

tensor = torch.ones(4,4)

#if you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can conert it to a python numerical value using item()

agg = tensor.sum()  #<--sums all the entries of tensor into an aggregation that is still a tensor datatype
agg_item = agg.item() #<-- turns single-element tensor into python item
print(agg_item, type(agg_item))