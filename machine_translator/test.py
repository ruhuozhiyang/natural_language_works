import torch

pth_path = r'MT_Model.pth'
net = torch.load('MT_Model.pth')
print(net)
