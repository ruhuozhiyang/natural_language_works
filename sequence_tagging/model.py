import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class NER_CRF_LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout, word2id, tag2id):
        super(NER_CRF_LSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id)
        self.tag_to_ix = tag2id
        self.tag_set_size = len(tag2id)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_set_size)
        self.transition = self.get_transition(self.tag_set_size)

    # 通过lstm获取序列特征
    def _get_lstm_features(self, batch_sentence):
        embeddings = self.word_embeds(batch_sentence.view(1, batch_sentence.size(0)))
        lstm_out, hidden = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 前向算法 计算配分函数
    def _forward_alg(self, lstm_feats):
        init_alphas = torch.full((1, self.tag_set_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas

        for feat in lstm_feats[0]:
            print(feat)
            alphas_t = []
            for next_tag in range(self.tag_set_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_set_size)
                trans_score = self.transition[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            print(forward_var)
        terminal_var = forward_var + self.transition[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, lstm_feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(lstm_feats):
            score = score + self.transition[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transition[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def forward(self, x):
        lstm_feats = self._get_lstm_features(x)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def viterbi_decode(self, features):  # features是序列的特征表示.
        backpointers = []
        init_vvars = torch.full((1, self.tag_set_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars
        for feat in features:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tag_set_size):
                next_tag_var = forward_var + self.transition[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transition[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def get_transition(self, dim):
        temp = nn.Parameter(torch.randn(dim, dim))
        temp.data[self.tag_to_ix[START_TAG], :] = -10000
        temp.data[:, self.tag_to_ix[STOP_TAG]] = -10000
        return temp


def log_sum_exp(status_matrix):
    max_score = status_matrix.max(dim=0, keepdim=True).values
    return (status_matrix - max_score).exp().sum(axis=0, keepdim=True).log() + max_score


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()