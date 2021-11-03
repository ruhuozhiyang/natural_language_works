# Pytorch用torch.utils.data.Dataset构建数据集，想要构建自己的数据集，则需继承Dataset类.
import numpy as np
import torch
import pre_process as en
import pre_process_zh as zh
from torch.utils.data import Dataset

# 全局参数配置
en_data_dir = './data/en_num.txt'
zh_data_dir = './data/zh_num.txt'
# CONTEXT_SIZE = 3  # 上下文词个数

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 每次返回一行句子序列.
class MyDataSetRnn(Dataset):
    def __getitem__(self, index):
        index = self.step
        content_list = []
        target_list = []
        while self.test_sentence[index] != -1:
            content_list.append([self.test_sentence[index - j - 1] for j in range(self.context_size)])
            target_list.append(self.test_sentence[index])
            index += 1
        self.step = index + 1 + self.context_size
        content_list = torch.tensor(np.array(content_list))
        target_list = torch.tensor(np.array([target_list[-1]]))
        return content_list, target_list

    def __len__(self):
        return self.lines_num

    def __init__(self, load_flag, context_size):
        self.context_size = context_size
        self.step = self.context_size
        self.test_sentence = []
        f = open(en_data_dir if load_flag == 'en' else zh_data_dir)
        lines = f.readlines()
        self.lines_num = len(lines)
        for line in lines:
            for word in line.split():
                self.test_sentence.append(int(word))
            self.test_sentence.append(-1)
        self.vocab = en.get_vocab_list() if load_flag == 'en' else zh.get_vocab_list()

    def get_vocab(self):
        return len(self.vocab), self.vocab
