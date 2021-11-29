import multiprocessing

import torch
import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

word_dict_name = 'word_dict.npy'


class Config:
    def __init__(self):
        self.learning_rate = 1e-2
        self.dropout = 0.9
        self.epoch = 5
        self.data_dir = 'data/'
        self.hidden_dim = 512
        self.save_model = 'NER_Model.pth'
        self.batch_size = 32
        self.char_dim = 100
        self.emb_file = 'data/emb_vec.txt'


def build_vocab(data_dir):
    """
    :param data_dir: the dir of train_corpus.txt
    :return: the word dict for training
    """
    if os.path.isfile(word_dict_name):
        word_dict = np.load(word_dict_name, allow_pickle=True).item()
        return word_dict
    else:
        word_dict = {}
        train_corpus = data_dir + 'train' + '_corpus.txt'
        lines = open(train_corpus).readlines()
        for line in lines:
            word_list = line.split()
            for word in word_list:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        word_dict = dict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
        np.save(word_dict_name, word_dict)
        word_dict = np.load(word_dict_name, allow_pickle=True).item()
        return word_dict


def build_dict(word_dict):
    """
    :param word_dict:
    :return: word2id and tag2id
    """
    tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3,
              'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'PAD': 7}
    word2id = {}
    for key in word_dict:
        word2id[key] = len(word2id)
    word2id['unk'] = len(word2id)
    word2id['pad'] = len(word2id)
    return word2id, tag2id


def cal_max_length(data_dir):
    file = data_dir + 'train' + '_corpus.txt'
    lines = open(file).readlines()
    max_len = 0
    for line in lines:
        if len(line.split()) > max_len:
            max_len = len(line.split())
    return max_len


def load_emb_matrix(vocab):
    emb_index = load_w2v(Config().emb_file)
    vocab_size = len(vocab)
    emb_matrix = np.zeros((vocab_size, Config().char_dim))
    for word, index in vocab.items():
        vector = emb_index.get(word)
        if vector is not None:
            emb_matrix[index] = vector
    emb_matrix = torch.FloatTensor(emb_matrix)
    return emb_matrix


def load_w2v(path):
    file = open(path, encoding="utf-8")
    emb_idx = {}
    for i, line in enumerate(file):
        value = line.split()
        char = value[0]
        emb = np.asarray(value[1:], dtype="float32")
        if len(emb) != Config().char_dim: continue
        emb_idx[char] = emb
    return emb_idx


def pre_train_emb():
    path = 'data/train_corpus.txt'
    model = Word2Vec(LineSentence(path), sg=1, vector_size=Config().char_dim, epochs=10,
                     window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.wv.save_word2vec_format(Config().emb_file, binary=False)


if __name__ == '__main__':
    pre_train_emb()
