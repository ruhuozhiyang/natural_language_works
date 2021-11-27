import torch
from numpy import argmax

a = torch.tensor([1, 2, 3])
b = a.view(1, -1)
# print(b.max(dim=0, keepdim=True).values)
c = torch.tensor([2, 3, 4]).view(1, -1)
d = b + c
tags = torch.tensor([1, 2, 4, 5], dtype=torch.long)
result = torch.cat([torch.tensor([8], dtype=torch.long), tags])
print(result)
# print(d)
# print(d[0, argmax(d)])
#
# print(torch.log(torch.sum(torch.exp(d))))

# print(torch.zeros(1))
# e = torch.tensor([3], dtype=torch.long)
# f = torch.tensor([3, 4, 5])
# tags = torch.cat([e, f])
# print(tags)