import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from utils import build_vocab, build_dict, cal_max_length, Config
from model import NER_CRF_LSTM
from torch.optim import Adam
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class NER_Dataset(Dataset):

    def __init__(self, data_dir, split, word2id, tag2id, max_length):
        file_dir = data_dir + split
        corpus_file = file_dir + '_corpus.txt'
        label_file = file_dir + '_label.txt'
        corpus = open(corpus_file).readlines()
        label = open(label_file).readlines()
        self.corpus = []
        self.label = []
        self.length = []
        self.word2id = word2id
        self.tag2id = tag2id
        for corpus_, label_ in zip(corpus, label):
            assert len(corpus_.split()) == len(label_.split())
            self.corpus.append([word2id[temp_word] if temp_word in word2id else word2id['unk']
                                for temp_word in corpus_.split()])
            self.label.append([tag2id[temp_label] for temp_label in label_.split()])
            self.length.append(len(corpus_.split()))
            if len(self.corpus[-1]) > max_length:
                self.corpus[-1] = self.corpus[-1][:max_length]
                self.label[-1] = self.label[-1][:max_length]
                self.length[-1] = max_length
            else:
                """
                padding处理.便于batch.
                """
                while len(self.corpus[-1]) < max_length:
                    self.corpus[-1].append(word2id['pad'])
                    self.label[-1].append(tag2id['PAD'])

        self.corpus = torch.Tensor(self.corpus).long()
        self.label = torch.Tensor(self.label).long()
        self.length = torch.Tensor(self.length).long()

    def __getitem__(self, item):
        return self.corpus[item], self.label[item], self.length[item]

    def __len__(self):
        return len(self.label)


def val(val_config, model):
    # ignore the pad label
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    test_set = NER_Dataset(val_config.data_dir, 'test', word2id, tag2id, max_length)
    val_data_loader = DataLoader(test_set, batch_size=val_config.batch_size)
    preds, labels = [], []
    for index, data in enumerate(val_data_loader):
        optimizer.zero_grad()
        corpus, label, length = data
        corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
        output = model(corpus)
        predict = torch.argmax(output, dim=-1)
        loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
        leng = []
        for i in label.cpu():
            tmp = []
            for j in i:
                if j.item() < 7:
                    tmp.append(j.item())
            leng.append(tmp)

        for ind, i in enumerate(predict.tolist()):
            preds.extend(i[:len(leng[ind])])

        for ind, i in enumerate(label.tolist()):
            labels.extend(i[:len(leng[ind])])

    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)
    model.train()
    return precision, recall, f1


def train(train_config, model, train_data_loader, train_optimizer):
    # ignore the pad label
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=7)
    best_f1 = 0.0
    for epoch in range(train_config.epoch):
        for index, data in enumerate(train_data_loader):
            train_optimizer.zero_grad()
            corpus, label, length = data
            # corpus, label, length = corpus.cuda(), label.cuda(), length.cuda()
            # output = model(corpus)
            # loss = loss_function(output.view(-1, output.size(-1)), label.view(-1))
            loss = model.neg_log_likelihood(corpus, label)
            loss.backward()
            train_optimizer.step()
            if index % 100 == 0:
                print('epoch: ', epoch, ' step:%04d,------------loss:%f' % (index, loss.item()))

        pre, rec, f1 = val(train_config, model)
        if f1 > best_f1:
            torch.save(model, train_config.save_model)


if __name__ == '__main__':
    config = Config()
    word_dict = build_vocab(config.data_dir)
    word2id, tag2id = build_dict(word_dict)
    max_length = cal_max_length(config.data_dir)
    train_set = NER_Dataset(config.data_dir, 'train', word2id, tag2id, max_length)
    data_loader = DataLoader(train_set, batch_size=config.batch_size)
    ner_crf_lstm = NER_CRF_LSTM(config.embedding_dim, config.hidden_dim,
                                config.dropout, word2id, tag2id)
    optimizer = Adam(ner_crf_lstm.parameters(), config.learning_rate)

    train(config, ner_crf_lstm, data_loader, optimizer)