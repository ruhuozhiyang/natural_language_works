import torch
import torch.nn as nn

START_TAG = "<START>"
STOP_TAG = "<STOP>"
not_likelihood = -10000.


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


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
        self.crf = CRF()
        self.transition = self.get_transition(self.tag_set_size)
        # self.bert = BertModel()

    def _get_lstm_features(self, batch_sentence):
        embeddings = self.word_embeds(batch_sentence)
        lstm_out, hidden = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, lstm_feats, pad_index):
        batch_alpha = []
        for j, one_sentence in enumerate(lstm_feats):
            init_alphas = torch.full((1, self.tag_set_size), not_likelihood)
            init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
            forward_var = init_alphas
            for i, word_feat in enumerate(one_sentence):
                if i == pad_index[j]:
                    break
                alphas_t = []
                for next_tag in range(self.tag_set_size):
                    emit_score = word_feat[next_tag].view(1, -1).expand(1, self.tag_set_size)
                    trans_score = self.transition[next_tag].view(1, -1)
                    next_tag_var = forward_var + trans_score + emit_score
                    alphas_t.append(log_sum_exp(next_tag_var).view(1))
                forward_var = torch.cat(alphas_t).view(1, -1)
            terminal_var = forward_var + self.transition[self.tag_to_ix[STOP_TAG]]
            alpha = log_sum_exp(terminal_var)
            batch_alpha.append(alpha.view(1, -1))
        result = torch.cat(batch_alpha).view(1, -1)
        return torch.mean(result)

    def _score_sentence(self, lstm_feats, tags, ignore_index):
        batch_size = lstm_feats.size()[0]
        batch_scores = []
        for j, one_sentence in enumerate(lstm_feats):
            start_tensor = self.tag_to_ix[START_TAG] * torch.ones(batch_size, 1, dtype=torch.long)
            tags = torch.cat([start_tensor, tags], 1)
            score = torch.zeros(1)
            for i, word_feat in enumerate(one_sentence):
                if tags[j][i + 1] == ignore_index:
                    break
                score += self.transition[tags[j][i + 1], tags[j][i]] + word_feat[tags[j][i + 1]]
            score = score + self.transition[self.tag_to_ix[STOP_TAG], tags[j][-1]]
            batch_scores.append(score.view(1, -1))
        result = torch.cat(batch_scores).view(1, -1)
        return torch.mean(result)

    def _viterbi_decode(self, batch_features):  # features是序列的特征表示.
        path_scores = []
        best_paths = []
        for one_sentence in batch_features:
            init_vars = torch.full((1, self.tag_set_size), not_likelihood)
            init_vars[0][self.tag_to_ix[START_TAG]] = 0
            forward_var = init_vars
            back_pointers = []
            for word_feat in one_sentence:
                bp_trs_t = []
                viterbi_vars_t = []
                for next_tag in range(self.tag_set_size):
                    next_tag_var = forward_var + self.transition[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bp_trs_t.append(best_tag_id)
                    viterbi_vars_t.append(next_tag_var[0][best_tag_id].view(1))
                forward_var = (torch.cat(viterbi_vars_t) + word_feat).view(1, -1)
                back_pointers.append(bp_trs_t)
            terminal_var = forward_var + self.transition[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]
            best_path = [best_tag_id]
            for bp_trs_t in reversed(back_pointers):
                best_tag_id = bp_trs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]
            best_path.reverse()
            path_scores.append(path_score)
            best_paths.append(best_path)
        return torch.tensor(path_scores), best_paths

    def neg_log_likelihood(self, sentence, tags, pad_index, ignore_index):
        feats = self._get_lstm_features(sentence)
        gold_score = self._score_sentence(feats, tags, ignore_index)
        forward_score = self._forward_alg(feats, pad_index)
        return forward_score - gold_score

    def get_transition(self, dim):
        temp = nn.Parameter(torch.randn(dim, dim))
        temp.data[self.tag_to_ix[START_TAG], :] = not_likelihood
        temp.data[:, self.tag_to_ix[STOP_TAG]] = not_likelihood
        return temp

    def forward(self, x):
        lstm_feats = self._get_lstm_features(x)
        scores, tag_seqs = self._viterbi_decode(lstm_feats)
        return scores, tag_seqs
