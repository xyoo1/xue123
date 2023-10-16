import torch

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
print(my_tensor)
print(my_tensor.device)
print(my_tensor.dtype)
print(my_tensor.shape)
print(my_tensor.requires_grad)
# Other common initialization methods
x = torch.empty(size=(3, 3))
y = torch.zeros((3, 3))
c = torch.rand((3, 3))
d = torch.ones((3, 3))
e = torch.eye(5, 5)  # I，eye
f = torch.arange(start=0, end=5, step=1)
g = torch.linspace(start=0.1, end=1, steps=10)
h = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
i = torch.empty(size=(1, 5)).uniform_(0, 1)
j = torch.diag(torch.ones(3))
# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())  # boolean True/False
print(tensor.short())  # int16
print(tensor.long())  # int64 (Important)
print(tensor.half())  # float16print(tensor.float(0)) #float32 (Important)print(tensor.double()) #float64
print(tensor.float(0))
print(tensor.double())
# Array to Tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
