import numpy as np
import re
import os

train_zh_sc = 'data/en-zh/train_sc.zh'
train_zh_sc_t = 'data/train_t.zh'
train_zh_sc_t1 = 'data/train_t1.zh'

val_zh_sc = 'data/en-zh/valid.zh'
val_zh_sc_t = 'data/val_t.zh'
val_zh_sc_t1 = 'data/val_t1.zh'

train_en_sc = 'data/en-zh/train_sc.en'

val_en_sc = 'data/en-zh/valid.en'

max_zh_sen = 50
max_en_sen = 54
max_threshold = 50


class Config:
    def __init__(self):
        self.learning_rate = 1e-2  # 学习率.
        self.dropout = 0.9  # 防止过拟合.
        self.epoch = 5  # 训练轮次.
        self.train_en_dir = 'data/train.en'  # 英文训练语料路径.
        self.train_zh_dir = 'data/train.zh'  # 中文训练语料路径.
        self.val_en_dir = 'data/valid.en'  # 英文验证语料路径.
        self.val_zh_dir = 'data/valid.zh'  # 中文验证语料路径.
        self.hidden_dim = 128  # 编码端Bi-LSTM隐藏层的节点数(双层总数).
        self.dec_hid_dim = 6   # 解码端LSTM隐藏层的节点数.
        self.save_model = 'MT_Model.pth'  # 保存模型的名称.
        self.batch_size = 32  # 批量训练的个数.
        self.char_dim = 50  # 预训练的词向量的维度(中英文都是).
        self.emb_file = 'data/emb_vec.txt'  # 预训练的中文词向量的存储位置.
        self.emb_file_en = 'data/emb_vec_en.txt'  # 预训练的英文词向量的存储位置.
        self.teacher_forcing_ratio = 0.5  # 解码端使用正确值作为下一预测值输入的概率.


def build_vocab(data_dir, word_dict_name):
    if os.path.isfile(word_dict_name):
        word_dict = np.load(word_dict_name, allow_pickle=True).item()
        return word_dict
    else:
        word_dict = {}
        lines = open(data_dir).readlines()
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


def build_dict(word_dict, f):
    word2id = {}
    for key in word_dict:
        word2id[key] = len(word2id)
    word2id['<pad>'] = len(word2id)
    word2id['<unk>'] = len(word2id)
    if f == 'zh':
        word2id['<eos>'] = len(word2id)
    return word2id


def seg_zh(sent):
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip()) > 0]
    return chars


# 将源中文文本按字分隔处理.
def seg_zh_wt(zh_sc, zh_sc_t):
    if os.path.isfile(zh_sc_t):
        print('file: {} exists, the file has been split well'.format(zh_sc_t))
        return
    sc_lines = open(zh_sc).readlines()
    w_content = open(zh_sc_t, 'w')
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
        print('file: {} does not exist'.format(data_dir))


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
        print('file: {} does not exist'.format(path_dir))


def del_long_sen_zh(s, t):
    del_max_l = []
    f = 0
    if os.path.isfile(s):
        w_content = open(t, 'w')
        lines = open(s, 'r').readlines()
        for i in lines:
            f += 1
            if len(i.split()) > max_threshold:
                del_max_l.append(f)
                i = i.strip(i)
            w_content.write(i)
        w_content.close()
    else:
        print('file: {} does not exist'.format(s))
    return del_max_l


def del_long_sen_en(s, t, li):
    f = 0
    if os.path.isfile(s):
        w_content = open(t, 'w')
        lines = open(s, 'r').readlines()
        for i in lines:
            f += 1
            if f in li:
                i = i.strip(i)
            w_content.write(i)
        w_content.close()
    else:
        print('file: {} does not exist'.format(s))


if __name__ == '__main__':
    print('the following is the operation of handling the chinese file')
    # seg_zh_wt(train_zh_sc, train_zh_sc_t)
    seg_zh_wt(val_zh_sc, val_zh_sc_t)
    # clear_blank(train_zh_sc_t, train_zh_sc_t1)
    clear_blank(val_zh_sc_t, val_zh_sc_t1)
    # index_del = del_long_sen_zh(train_zh_sc_t1, Config().train_zh_dir)
    index_del = del_long_sen_zh(val_zh_sc_t1, Config().val_zh_dir)
    # del_long_sen_en(train_en_sc, Config().train_en_dir, index_del)
    del_long_sen_en(val_en_sc, Config().val_en_dir, index_del)
    # data_info(Config().train_zh_dir)
    # data_info(Config().train_en_dir)
    data_info(Config().val_zh_dir)
    data_info(Config().val_en_dir)
