import numpy as np
import re
import os

train_zh_sc = 'data/en-zh/train_sc.zh'
train_zh_sc_t = 'data/en-zh/train_t.zh'


class Config:
    def __init__(self):
        self.learning_rate = 1e-2
        self.dropout = 0.9
        self.epoch = 5
        self.train_en_dir = 'data/en-zh/train.en'
        self.train_zh_dir = 'data/en-zh/train.zh'
        self.hidden_dim = 512
        self.save_model = 'NER_Model.pth'
        self.batch_size = 32
        self.char_dim = 100
        self.emb_file = 'data/emb_vec.txt'


def seg_zh(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars


# 将源中文文本按字分隔处理.
def seg_zh_wt():
    if os.path.isfile(train_zh_sc_t):
        print('已存在同名文件, 中文文本已分隔好')
        return
    sc_lines = open(train_zh_sc).readlines()
    w_content = open(train_zh_sc_t, 'w')
    for i in sc_lines:
        if i == '\n':
            w_content.write('<blank>')
        else:
            w_content.write(' '.join(seg_zh(i)))
        w_content.write('\n')
    w_content.close()


def clear_blank(data_dir, target_dir):
    if os.path.isfile(data_dir):
        w_content = open(target_dir, 'w')
        lines = open(data_dir, 'r').readlines()
        for i in lines:
            if i == '\n':
                i = i.strip('\n')
                continue
            if i == '<blank>\n':
                i = '\n'
            w_content.write(i)
        w_content.close()
    else:
        print('不存在文件: {}'.format(data_dir))


def data_info(path_dir):
    if os.path.isfile(path_dir):
        lines = open(path_dir).readlines()
        sen_len_l = [len(s.split()) for s in lines]
        rows = len(lines)
        d_ave = np.mean(sen_len_l)
        d_max = np.max(sen_len_l)
        print('the file path: {}'.format(path_dir))
        print('the number of the file rows: {}'.format(rows))
        print('the average length of rows: {}'.format(d_ave))
        print('the max length of rows: {}'.format(d_max))
    else:
        print('不存在文件: {}'.format(path_dir))


if __name__ == '__main__':
    seg_zh_wt()
    clear_blank(train_zh_sc_t, Config().train_zh_dir)
    data_info(Config().train_zh_dir)
    data_info(Config().train_en_dir)
