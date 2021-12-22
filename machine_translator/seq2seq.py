import random

import torch.nn as nn
import torch


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, b_s, b_t, t_f_ratio):
        batch_size = b_s.shape[0]
        t_len = b_t.shape[1]
        t_vocab_size = self.decoder.out_dim

        outputs = torch.zeros(t_len, batch_size, t_vocab_size)

        k, q = self.encoder(b_s)
        b_t = b_t.transpose(0, 1)
        dec_input = b_t[0, :]

        for t in range(1, t_len):
            # 解码端起先的输入是中文文本的第一个值.
            # q是前一时刻的隐藏层状态. 解码端q起先是encoder的final hidden state.
            # k是encoder所有时刻的最后一层的hidden state.
            dec_output = self.decoder(dec_input, q, k)
            outputs[t] = dec_output
            teacher_force = random.random() < t_f_ratio
            top1 = dec_output.argmax(1)
            dec_input = b_t[t] if teacher_force else top1
        return outputs
