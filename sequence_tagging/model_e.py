import torch
import torch.nn as nn
from TorchCRF import CRF


class NER_LSTM_CRF(nn.Module):
    def __init__(self, config, word2id, tag2id, emb_matrix):
        super(NER_LSTM_CRF, self).__init__()

        self.hidden_dim = config.hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tag_set_size = len(tag2id)

        self.char_emb = nn.Embedding.from_pretrained(emb_matrix, freeze=False, padding_idx=7)
        self.emb_dim = config.char_dim
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_set_size)
        self.crf = CRF(self.tag_set_size)

    def forward(self, char_ids, mask=None):
        embedding = self.char_emb(char_ids)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return self.crf.viterbi_decode(outputs, mask)

    def log_likelihood(self, char_ids, tags_ids, mask=None):
        embedding = self.char_emb(char_ids)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        return - self.crf(outputs, tags_ids, mask)
