import multiprocessing

import numpy as np
import torch
from gensim.models.word2vec import LineSentence

from utils import Config
from gensim.models import Word2Vec

config = Config()


def load_emb_matrix(vocab, f):
    emb_index = load_w2v(config.emb_file if f == 'zh' else config.emb_file_en)
    vocab_size = len(vocab)
    emb_matrix = np.zeros((vocab_size, config.char_dim))
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
        if len(emb) != config.char_dim:
            continue
        emb_idx[char] = emb
    return emb_idx


def pre_train_emb():
    # path = 'data/train.zh'
    path = 'data/train.en'
    model = Word2Vec(LineSentence(path), sg=1, vector_size=config.char_dim, epochs=10,
                     window=5, min_count=5, workers=multiprocessing.cpu_count())
    # model.wv.save_word2vec_format(config.emb_file, binary=False)
    model.wv.save_word2vec_format(config.emb_file_en, binary=False)


if __name__ == '__main__':
    pre_train_emb()
