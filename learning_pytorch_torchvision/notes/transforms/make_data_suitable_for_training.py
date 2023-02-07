import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

#NOTE: this is about preprocessing data to make it suitable for training
#NOTE: All TorchVision datasets have two parameters {transform} to modify the features and {target_transform} to modify the labels
#NOTE: for training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. To make these transformations we use {ToTensor} and {Lambda}

#NOTE: In a {ONE-HOT ENCODED TENSOR}, each categorical value is represented as a binary vector. The vector has the same length as the number of categories,
#           and a single element is set to 1 to indicate the presence of a particular category, while the rest of the elements are set to 0.

#loading dataset from torchvision
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(), #<-- converts PIL image or a NumPy ndarray into a FloatTensor and scales the image's pixel intensity values in the range [0.,1.]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) #<-- turns labels into ONE-HOT ENCODED TENSOR so can use categories in the matrix operations
)