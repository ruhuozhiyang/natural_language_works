# Pytorch用torch.utils.data.Dataset构建数据集，想要构建自己的数据集，则需继承Dataset类.
import numpy as np
import torch
import pre_process as en
import pre_process_zh as zh
from torch.utils.data import Dataset

# 全局参数配置
en_data_dir = './data/en_num.txt'
zh_data_dir = './data/zh_num.txt'
CONTEXT_SIZE = 3  # 上下文词个数

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MyDataSet(Dataset):
    def __getitem__(self, index):
        index = index + self.step
        if self.test_sentence[index] == -1:
            context = [self.test_sentence[index + CONTEXT_SIZE - j]
                       for j in range(CONTEXT_SIZE)]
            target = self.test_sentence[index + 1 + CONTEXT_SIZE]
            self.step = self.step + CONTEXT_SIZE + 1
        else:
            context = [self.test_sentence[index - j - 1] for j in range(CONTEXT_SIZE)]
            target = self.test_sentence[index]
        context = torch.tensor(np.array(context))
        return context, target

    def __len__(self):
        return len(self.test_sentence)

    def __init__(self, load_flag):
        self.step = CONTEXT_SIZE
        self.test_sentence = []
        f = open(en_data_dir if load_flag == 'en' else zh_data_dir)
        lines = f.readlines()
        for line in lines:
            for word in line.split():
                self.test_sentence.append(int(word))
            self.test_sentence.append(-1)
        print(len(self.test_sentence))
        self.vocab = en.get_vocab_list() if load_flag == 'en' else zh.get_vocab_list()

    def get_vocab(self):
        return len(self.vocab), self.vocab
