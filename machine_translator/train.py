import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from bi_lstm_encoder import BiLstmEncoder
from decoder import Decoder
from attention import Attention
from seq2seq import Seq2Seq
from utils import Config, build_vocab, build_dict, max_zh_sen, max_en_sen
from word2vec import load_emb_matrix

word_dict_zh = 'data/word_dict_zh.npy'
word_dict_en = 'data/word_dict_en.npy'

word_dict_val_zh = 'data/word_dict_val_zh.npy'
word_dict_val_en = 'data/word_dict_val_en.npy'

word2id_g = {}
word2id_en_g = {}


class BestValidLoss:
    def __init__(self):
        self.best_valid_loss = float('inf')

    def get_best_loss(self):
        return self.best_valid_loss

    def set_best_loss(self, v):
        self.best_valid_loss = v


# 英——>中 翻译
class MT_Dataset(Dataset):
    def __init__(self, zh_dir, en_dir, word2id_zh, word2id_e):
        zh_lines = open(zh_dir).readlines()
        en_lines = open(en_dir).readlines()
        self.zh_lines = []
        self.en_lines = []
        for zh, en in zip(zh_lines, en_lines):
            self.en_lines.append([word2id_e[t_w] if t_w in word2id_e else word2id_e['<unk>']
                                  for t_w in en.split()])
            temp = zh.split()
            temp.append('<eos>')
            self.zh_lines.append([word2id_zh[t_w] if t_w in word2id_zh else word2id_zh['<unk>']
                                  for t_w in temp])
            if len(self.en_lines[-1]) > max_en_sen:
                self.en_lines[-1] = self.en_lines[-1][:max_en_sen]
            else:
                while len(self.en_lines[-1]) < max_en_sen:
                    self.en_lines[-1].append(word2id_e['<pad>'])

            if len(self.zh_lines[-1]) > max_zh_sen:
                self.zh_lines[-1] = self.zh_lines[-1][:max_zh_sen]
            else:
                while len(self.zh_lines[-1]) < max_zh_sen:
                    self.zh_lines[-1].append(word2id_zh['<pad>'])
        self.zh_lines = torch.Tensor(self.zh_lines).long()
        self.en_lines = torch.Tensor(self.en_lines).long()

    def __getitem__(self, item):
        return self.zh_lines[item], self.en_lines[item]

    def __len__(self):
        return len(self.en_lines)


def validate(v_config, model, criterion):
    val_set = MT_Dataset(config.val_zh_dir, config.val_en_dir, word2id_g, word2id_en_g)
    data_load = DataLoader(val_set, batch_size=v_config.batch_size)
    model.eval()
    for epoch in range(1):
        epoch_loss = 0
        with torch.no_grad():
            for index, data in enumerate(data_load):
                zh, en = data
                output = model(en, zh, 0)
                output_dim = output.shape[-1]
                output = output.view(-1, output_dim)
                zh = zh.view(-1)
                loss = criterion(output, zh)
                epoch_loss += loss.item()
        valid_loss = epoch_loss / len(data_loader)
        print('val_loss: {}'.format(epoch + 1, 10, valid_loss))
        if valid_loss < BestValidLoss().get_best_loss():
            BestValidLoss.set_best_loss(valid_loss)
            torch.save(model.state_dict(), v_config.save_model)


def train(t_config, train_loader, seq_seq, s_opt, ig_i):
    criterion = nn.CrossEntropyLoss(ignore_index=ig_i)
    seq_seq.train()
    for epoch in range(t_config.epoch):
        epoch_loss = 0.0
        with tqdm(train_loader) as t:
            t.set_description('Epoch {}/{}:'.format(epoch + 1, t_config.epoch))
            for index, data in enumerate(t):
                s_opt.zero_grad()
                zh, en = data
                prediction = seq_seq(en, zh, t_config.teacher_forcing_ratio)
                vac_dim = prediction.shape[-1]

                pre = prediction.view(-1, vac_dim)
                target = zh.view(-1)
                loss = criterion(pre, target)
                loss.backward()
                s_opt.step()
                epoch_loss += loss.item()
                t.set_postfix(loss=loss.item())
            print('Epoch {}/{}: train_loss: {}'.format(epoch + 1, t_config.epoch,
                                                       epoch_loss / len(train_loader)))
        validate(t_config, seq_seq, criterion)


if __name__ == '__main__':
    config = Config()
    word_dict = build_vocab(config.train_zh_dir, word_dict_zh)
    word2id = build_dict(word_dict, 'zh')
    word2id_g = word2id
    emb_matrix = load_emb_matrix(word2id, 'zh')

    word_dict_en = build_vocab(config.train_en_dir, word_dict_en)
    word2id_en = build_dict(word_dict_en, 'en')
    word2id_en_g = word2id_en
    emb_matrix_en = load_emb_matrix(word2id_en, 'en')

    b_l_e = BiLstmEncoder(config, emb_matrix_en)
    att = Attention(config)
    decoder = Decoder(config, len(word2id), att, emb_matrix)
    seq2seq = Seq2Seq(b_l_e, decoder)

    train_set = MT_Dataset(config.train_zh_dir, config.train_en_dir, word2id, word2id_en)
    data_loader = DataLoader(train_set, batch_size=config.batch_size)

    # 优化器用Adam
    opt = optim.Adam(seq2seq.parameters(), lr=config.learning_rate)

    train(config, data_loader, seq2seq, opt, word2id.get('<pad>'))
