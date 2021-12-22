import torch.nn as nn
import torch
import torch.nn.functional as f


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.att = nn.Linear(config.hidden_dim + config.dec_hid_dim, config.dec_hid_dim, bias=False)
        self.v = nn.Linear(config.dec_hid_dim, 1, bias=False)

    def forward(self, q, k):
        src_len = k.shape[1]
        q = q.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.att(torch.cat((q, k), dim=2)))
        attention = self.v(energy).squeeze(2)
        return f.softmax(attention, dim=1)
