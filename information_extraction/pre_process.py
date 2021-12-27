import os
import re

pattern_repl = re.compile('(<e1>)|(</e1>)|(<e2>)|(</e2>)|(\'s)')
pattern_e1 = re.compile('<e1>(.*)</e1>')
pattern_e2 = re.compile('<e2>(.*)</e2>')
pattern_symbol = re.compile('^[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]|[!"#$%&\\\'()*+,-./:;<=>?@['
                            '\\]^_`{|}~]$')


def load_dataset(path_dataset):
  dataset = []
  with open(path_dataset) as f:
    piece = list()
    for line in f:
      line = line.strip()
      if line:
        piece.append(line)
      elif piece:
        sentence = piece[0].split('\t')[1].strip('"')
        e1 = delete_symbol(pattern_e1.findall(sentence)[0])
        e2 = delete_symbol(pattern_e2.findall(sentence)[0])
        new_sentence = list()
        for word in pattern_repl.sub('', sentence).split(' '):
          new_word = delete_symbol(word)
          if new_word:
            new_sentence.append(new_word)
        relation = piece[1]
        dataset.append(((e1, e2, ' '.join(new_sentence)), relation))
        piece = list()
  return dataset


def delete_symbol(text):
  if pattern_symbol.search(text):
    return pattern_symbol.sub('', text)
  return text


def save_dataset(dataset, save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
    for words, labels in dataset:
      file_sentences.write('{}\n'.format('\t'.join(words)))
      file_labels.write('{}\n'.format(labels))


def pro_link(path_dataset):
  count = 0
  temp = ['<e1>', '</e1>', '<e2>', '</e2>']
  mv = [-1, 5, -1, 5]
  with open(path_dataset) as f:
    for line in f:
      for index, i in enumerate(temp):
        if count % 4 == 0 and line.index(i) > 0 and line[line.index(i) + mv[index]] != ' ' \
          and line[line.index(i) + mv[index]] != '.' and line[line.index(i) + mv[index]] != ',' \
          and line[line.index(i) + mv[index]] != '"' and line[line.index(i) + mv[index]] != "'" \
          and line[line.index(i) + mv[index]] != ':' and line[line.index(i) + mv[index]] != ";":
          print(line)
          break
      count += 1


if __name__ == '__main__':
  path_train = 'data/SemEval2010_task8_data/TRAIN_FILE.TXT'
  path_test = 'data/SemEval2010_task8_data/TEST_FILE_FULL.TXT'
  msg = "{} or {} file not found.".format(path_train, path_test)
  assert os.path.isfile(path_train) and os.path.isfile(path_test), msg

  # pro_link(path_train)
  train_dataset = load_dataset(path_train)
  test_dataset = load_dataset(path_test)

  save_dataset(train_dataset, 'data/SemEval2010_task8/train')
  save_dataset(test_dataset, 'data/SemEval2010_task8/test')
