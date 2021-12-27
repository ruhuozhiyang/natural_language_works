import argparse
import logging
import os
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, type=int)
parser.add_argument('--min_count_tag', default=1, type=int)
parser.add_argument('--data_dir', default='data/SemEval2010_task8')


def save_to_txt(vocab, txt_path):
  with open(txt_path, 'w') as f:
    for token in vocab:
      f.write(token + '\n')


def update_vocab(txt_path, vocab):
  with open(txt_path) as f:
    for i, line in enumerate(f):
      line = line.strip()
      if line.endswith('...'):
        line = line.rstrip('...')
      word_seq = line.split('\t')[-1].split(' ')
      vocab.update(word_seq)
  return i + 1


def update_labels(txt_path, label):
  with open(txt_path) as f:
    for i, line in enumerate(f):
      line = line.strip()
      label.update([line])
  return i + 1


if __name__ == '__main__':
  args = parser.parse_args()

  words = Counter()
  size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
  size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)

  labels = Counter()
  size_train_tags = update_labels(os.path.join(args.data_dir, 'train/labels.txt'), labels)
  size_test_tags = update_labels(os.path.join(args.data_dir, 'test/labels.txt'), labels)

  assert size_train_sentences == size_train_tags
  assert size_test_sentences == size_test_tags

  words = sorted([tok for tok, count in words.items() if count >= args.min_count_word])
  labels = sorted([tok for tok, count in labels.items() if count >= args.min_count_tag])

  save_to_txt(words, os.path.join(args.data_dir, 'words.txt'))
  save_to_txt(labels, os.path.join(args.data_dir, 'labels.txt'))
  logging.info('build vocabulary successfully.')
