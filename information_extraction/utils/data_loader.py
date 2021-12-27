import logging
import random
import numpy as np
import os
import torch
import utils.utils as utils


def load_embeddings(emb_path, emb_delimiter):
  with open(emb_path, 'r') as f:
    for line in f:
      values = line.strip().split(emb_delimiter)
      word = values[0]
      emb_vector = list(map(lambda emb: float(emb),
                            filter(lambda val: val and not val.isspace(), values[1:])))
      yield word, emb_vector


# 大小写映射.
def get_embedding_word(word, embedding_words):
  if word in embedding_words:
    return word
  elif word.lower() in embedding_words:
    return word.lower()
  return None


class DataLoader(object):
  def __init__(self, data_dir, embedding_file, word_emb_dim, max_len=100, pos_dis_limit=50,
               pad_word='<pad>', unk_word='<unk>', other_label='Other'):
    self.data_dir = data_dir
    self.embedding_file = embedding_file
    self.max_len = max_len
    self.word_emb_dim = word_emb_dim
    self.pos_clip = pos_dis_limit

    self.pad_word = pad_word
    self.unk_word = unk_word
    self.other_label = other_label

    self.word2idx = dict()
    self.label2idx = dict()

    self.embedding_vectors = list()
    self.unique_words = list()

    if pad_word is not None:
      self.pad_idx = len(self.word2idx)
      self.word2idx[pad_word] = self.pad_idx
      self.embedding_vectors.append(utils.generate_zero_vector(self.word_emb_dim))
    if unk_word is not None:
      self.unk_idx = len(self.word2idx)
      self.word2idx[unk_word] = self.unk_idx
      self.embedding_vectors.append(utils.generate_random_vector(self.word_emb_dim))

    vocab_path = os.path.join(self.data_dir, 'words.txt')
    with open(vocab_path, 'r') as f:
      for line in f:
        self.unique_words.append(line.strip())

    labels_path = os.path.join(data_dir, 'labels.txt')
    with open(labels_path, 'r') as f:
      for i, line in enumerate(f):
        self.label2idx[line.strip()] = i

    other_label_idx = self.label2idx[self.other_label]
    self.metric_labels = list(self.label2idx.values())
    self.metric_labels.remove(other_label_idx)

  def get_loaded_embedding_vectors(self):
    return torch.FloatTensor(np.asarray(self.embedding_vectors))

  def load_embeddings_and_unique_words(self, emb_delimiter=' '):
    embedding_words = [emb_word for emb_word, _ in
                       load_embeddings(self.embedding_file, emb_delimiter)]
    emb_word2unique_word = dict()
    for unique_word in self.unique_words:
      emb_word = get_embedding_word(unique_word, embedding_words)
      if emb_word is not None:
        if emb_word not in emb_word2unique_word:
          emb_word2unique_word[emb_word] = [unique_word]
        else:
          emb_word2unique_word[emb_word].append(unique_word)

    for emb_word, emb_vector in load_embeddings(self.embedding_file, emb_delimiter):
      if emb_word in emb_word2unique_word:
        for unique_word in emb_word2unique_word[emb_word]:
          self.word2idx[unique_word] = len(self.word2idx)
          self.embedding_vectors.append(emb_vector)
    logging.info('loaded vocabulary from embedding file and unique words successfully.')

  def load_sentences_labels(self, sentences_file, labels_file, d):
    sent_s, pos1s, pos2s, labels = list(), list(), list(), list()

    with open(sentences_file, 'r') as f:
      for i, line in enumerate(f):
        e1, e2, sent = line.strip().split('\t')
        words = sent.split(' ')
        e1 = e1.split(' ')[0] if ' ' in e1 else e1
        e2 = e2.split(' ')[0] if ' ' in e2 else e2
        try:
          e1_idx = words.index(e1)
        except IndexError:
          logging.info("{} does not exist in the words list".format(e1))
        try:
          e2_idx = words.index(e2)
        except IndexError:
          logging.info("{} does not exist in the words list".format(e2))

        # sent是以每个词的id保存一行句子. pos1是每个词相对e1的位置. pos2是每个词相对e2的位置.
        sent, pos1, pos2 = list(), list(), list()
        for idx, word in enumerate(words):
          emb_word = get_embedding_word(word, self.word2idx)
          if emb_word:
            sent.append(self.word2idx[word])
          else:
            sent.append(self.unk_idx)
          pos1.append(self.get_pos_feature(idx - e1_idx))
          pos2.append(self.get_pos_feature(idx - e2_idx))
        sent_s.append(sent)
        pos1s.append(pos1)
        pos2s.append(pos2)

    with open(labels_file, 'r') as f:
      for line in f:
        idx = self.label2idx[line.strip()]
        labels.append(idx)

    assert len(labels) == len(sent_s)

    d['data'] = {'sents': sent_s, 'pos1s': pos1s, 'pos2s': pos2s}
    d['labels'] = labels
    d['size'] = len(sent_s)

  def get_pos_feature(self, x):
    if x < -self.pos_clip:
      return 0
    elif -self.pos_clip <= x <= self.pos_clip:
      return x + self.pos_clip + 1
    else:
      return self.pos_clip * 2 + 2

  def load_data(self, data_type):
    data = dict()
    if data_type in ['train', 'test']:
      sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
      labels_file = os.path.join(self.data_dir, data_type, 'labels.txt')
      self.load_sentences_labels(sentences_file, labels_file, data)
    else:
      raise ValueError("data type not in ['train', 'test']")
    return data

  def data_iterator(self, data, batch_size, shuffle='False'):
    order = list(range(data['size']))
    if shuffle:
      random.seed(230)
      random.shuffle(order)

    for i in range((data['size']) // batch_size):
      batch_sents = [data['data']['sents'][idx] for idx in
                     order[i * batch_size:(i + 1) * batch_size]]
      batch_pos1s = [data['data']['pos1s'][idx] for idx in
                     order[i * batch_size:(i + 1) * batch_size]]
      batch_pos2s = [data['data']['pos2s'][idx] for idx in
                     order[i * batch_size:(i + 1) * batch_size]]
      batch_labels = [data['labels'][idx] for idx in
                      order[i * batch_size:(i + 1) * batch_size]]

      temp_len = max([len(s) for s in batch_sents])
      batch_max_len = temp_len if temp_len < self.max_len else self.max_len

      batch_data_sents = self.pad_idx * np.ones((batch_size, batch_max_len))
      batch_data_pos1s = (self.pos_clip * 2 + 2) * np.ones((batch_size, batch_max_len))
      batch_data_pos2s = (self.pos_clip * 2 + 2) * np.ones((batch_size, batch_max_len))
      for j in range(batch_size):
        cur_len = len(batch_sents[j])
        min_len = min(cur_len, batch_max_len)
        batch_data_sents[j][:min_len] = batch_sents[j][:min_len]
        batch_data_pos1s[j][:min_len] = batch_pos1s[j][:min_len]
        batch_data_pos2s[j][:min_len] = batch_pos2s[j][:min_len]

      batch_data = {
        'sents': torch.LongTensor(batch_data_sents),
        'pos1s': torch.LongTensor(batch_data_pos1s),
        'pos2s': torch.LongTensor(batch_data_pos2s)
      }

      batch_labels = torch.LongTensor(batch_labels)
      yield batch_data, batch_labels
