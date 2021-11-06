# Pytorch用torch.utils.data.Dataset构建数据集，想要构建自己的数据集，则需继承Dataset类.
import numpy as np
import random
import torch
import pre_process_en as en
import pre_process_zh as zh
from torch.utils.data import Dataset

# 全局参数配置
en_data_dir = './data/en_num.txt'
zh_data_dir = './data/zh_num.txt'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MyDataSetCw(Dataset):
    def __getitem__(self, index):
        index = self.step
        wrong_list = []
        if self.test_sentence[index + self.context_size] == -1:
            self.step = index + 2 * self.context_size + 1
        index = self.step
        correct_list = [self.test_sentence[j] for j in range(index - self.context_size, index + self.context_size + 1)]
        # for item in self.vocab:
        #     temp = correct_list
        #     temp[self.context_size] = self.word2int[item]
        #     wrong_list.append(temp)
        word = random.choice(list(self.vocab))
        temp = correct_list.copy()
        temp[self.context_size] = self.word2int[word]
        wrong_list.append(temp)
        self.step = index + 1
        correct_sample = torch.tensor(np.array(correct_list))
        wrong_samples = torch.tensor(np.array(wrong_list))
        return correct_sample, wrong_samples

    def __len__(self):
        return
        # return self.iter_num

    def __init__(self, load_flag, context_size):
        self.iter_num = 0
        self.context_size = context_size
        self.step = self.context_size
        self.test_sentence = []
        f = open(en_data_dir if load_flag == 'en' else zh_data_dir)
        lines = f.readlines()
        for line in lines:
            self.iter_num += len(line) - 2*context_size
            for word in line.split():
                self.test_sentence.append(int(word))
            self.test_sentence.append(-1)
        self.vocab = en.get_vocab_list() if load_flag == 'en' else zh.get_vocab_list()
        self.word2int = en.get_word2int() if load_flag == 'en' else en.get_word2int()

    def get_vocab(self):
        return len(self.vocab), self.vocab
