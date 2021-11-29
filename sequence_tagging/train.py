import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils import build_vocab, build_dict, cal_max_length, Config, load_emb_matrix
from model_e import NER_LSTM_CRF
# from model import NER_CRF_LSTM  # 使用自己写的CRF时候使用
from torch.optim import Adam
from tqdm import tqdm


class NER_Dataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag_id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()
        label = open(label_file).readlines()
        self.sen_idx = []
        self.mask = []
        # self.pad_index = []  # 使用自己写的CRF时候使用
        self.corpus = []
        self.label = []
        self.word2id = word2id
        self.tag2id = tag_id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpus_.split()])
            self.label.append([tag_id[temp_label] for temp_label in label_.split()])
            self.mask.append([1] * len(corpus_.split()))
            if len(self.corpus[-1]) > max_length:
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                # self.pad_index.append(max_length)  # 使用自己写的CRF时候使用
            else:
                """
                padding处理.便于batch.
                """
                # self.pad_index.append(len(self.corpus[-1]))  # 使用自己写的CRF时候使用
                while len(self.corpus[-1]) < max_length:
                    self.corpus[-1].append(word2id['pad'])
                    self.label[-1].append(tag_id['PAD'])
                    self.mask[-1].append(0)

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.mask = torch.tensor(self.mask, dtype=torch.uint8)

    def __getitem__(self, item):
        # 使用自己写的CRF时候使用
        # return self.corpus[item], self.label[item], self.pad_index[item]
        return self.corpus[item], self.label[item], self.mask[item]

    def __len__(self):
        return len(self.label)


def val(val_config, model):
    model.eval()
    test_set = NER_Dataset(val_config.data_dir, 'test', word2id, tag2id, max_length)
    val_data_loader = DataLoader(test_set, batch_size=val_config.batch_size)
    predictions, labels = [], []
    for index, data in enumerate(val_data_loader):
        optimizer.zero_grad()
        # corpus, label, _ = data  # 使用自己写的CRF时候使用
        corpus, label, mask = data
        predict = model(corpus, mask)
        len_g = []
        for i in label:
            tmp = []
            for j in i:
                if j.item() < 7:
                    tmp.append(j.item())
            len_g.append(tmp)
        for ind, i in enumerate(predict):
            predictions.extend(i[:len(len_g[ind])])

        for ind, i in enumerate(label):
            labels.extend(i[:len(len_g[ind])])

    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    f1 = f1_score(labels, predictions, average='macro')
    report = classification_report(labels, predictions, zero_division=0)
    print('\n')
    print(report)
    model.train()
    return precision, recall, f1


def train(train_config, model, train_data_loader, train_optimizer):
    best_f1 = 0.0
    for epoch in range(train_config.epoch):
        with tqdm(train_data_loader) as t:
            t.set_description('Epoch {}/{}:'.format(epoch + 1, train_config.epoch))
            for index, data in enumerate(t):
                train_optimizer.zero_grad()
                # corpus, label, pad_index = data  # 使用自己写的CRF时候使用
                corpus, label, masks = data
                # loss = model.neg_log_likelihood(char_ids=corpus, tags_ids=label, pad_index,
                # ignore_index=7)  # 使用自己写的CRF时候使用
                loss = model.log_likelihood(char_ids=corpus, tags_ids=label, mask=masks)
                loss.mean().backward()
                train_optimizer.step()
                loss = loss.mean().item()
                t.set_postfix(loss=loss)
            precision, recall, f1 = val(train_config, model)
            if f1 > best_f1:
                torch.save(model, train_config.save_model)


if __name__ == '__main__':
    config = Config()
    word_dict = build_vocab(config.data_dir)
    word2id, tag2id = build_dict(word_dict)
    emb_matrix = load_emb_matrix(word2id)
    max_length = cal_max_length(config.data_dir)
    train_set = NER_Dataset(config.data_dir, 'train', word2id, tag2id, max_length)
    data_loader = DataLoader(train_set, batch_size=config.batch_size)
    # ner_crf_lstm = NER_CRF_LSTM(config.embedding_dim, config.hidden_dim, config.dropout,
    # word2id, tag2id)
    ner_crf_lstm = NER_LSTM_CRF(config, word2id, tag2id, emb_matrix)
    optimizer = Adam(ner_crf_lstm.parameters(), config.learning_rate)

    train(config, ner_crf_lstm, data_loader, optimizer)
