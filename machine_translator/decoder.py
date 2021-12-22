import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, config, out_dim, attention, emb_matrix):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.out_dim = out_dim

        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=True)
        self.attention = attention
        self.lstm = nn.LSTM(config.hidden_dim + config.char_dim, config.dec_hid_dim)
        self.f_out = nn.Linear(config.hidden_dim + config.char_dim + config.dec_hid_dim, out_dim)

    def forward(self, decoder_input, q, k):
        decoder_input = decoder_input.unsqueeze(1)
        embedding = self.dropout(self.embedding(decoder_input))  # [batch_size, 1, emb_dim] 中文词向量

        a = self.attention(q, k).unsqueeze(1)  # [batch_size, 1, s_len] 得到的中文的注意力向量

        # 对于encoder各个词隐藏状态的加权求和.
        c = torch.bmm(a, k)  # [batch_size, 1, hidden_dim]

        rnn_input = torch.cat((embedding, c), dim=2).transpose(0, 1)
        q = q.unsqueeze(0)
        dec_output, (h_n, c_n) = self.lstm(rnn_input, (q, q))

        embedded = embedding.squeeze(1)
        dec_output = dec_output.squeeze(0)
        c = c.squeeze(1)
        prediction = self.f_out(torch.cat((dec_output, c, embedded), dim=1))
        return prediction
