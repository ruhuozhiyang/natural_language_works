import torch
import torch.nn as nn


class BiLstmEncoder(nn.Module):
    def __init__(self, config, emb_matrix):
        super(BiLstmEncoder, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.emb_dim = config.char_dim
        self.dropout = nn.Dropout(config.dropout)

        self.char_emb = nn.Embedding.from_pretrained(emb_matrix, freeze=True)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.f = nn.Linear(self.hidden_dim, config.dec_hid_dim)

    def forward(self, char_ids):
        embedding = self.dropout(self.char_emb(char_ids))
        outputs, (h_n, c_n) = self.lstm(embedding)
        f_h = torch.tanh(self.f(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)))
        return outputs, f_h
