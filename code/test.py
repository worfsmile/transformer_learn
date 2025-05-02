import torch

x = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32)
x1 = x.mean(dim=0)
x2 = x.mean(dim=1)
print(x)
print(x1)
print(x2)

