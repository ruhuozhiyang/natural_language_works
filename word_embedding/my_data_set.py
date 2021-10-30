# Pytorch用torch.utils.data.Dataset构建数据集，想要构建自己的数据集，则需继承Dataset类.
import numpy as np
import torch
from torch.utils.data import Dataset

# 全局参数配置
en_data_dir = './data/en_num.txt'
zh_data_dir = './data/zh.txt'
CONTEXT_SIZE = 3  # 上下文词个数


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MyDataSet(Dataset):
    def __getitem__(self, index):
        context = [int(self.test_sentence[index - j - 1]) for j in range(CONTEXT_SIZE)]
        target = int(self.test_sentence[index])
        context = torch.tensor(np.array(context), dtype=torch.long)
        # target = torch.tensor(np.array(target), dtype=torch.long)
        return context, target

    def __len__(self):
        return len(self.test_sentence)

    def __init__(self, load_flag):
        with open(en_data_dir if load_flag == 'en' else zh_data_dir) as f:
            content = f.read()
            f.close()
        self.test_sentence = content.split()
        self.vocab = set(self.test_sentence)  # 所有词汇（去除重复的）
        # self.word2int = {word: i for i, word in enumerate(self.vocab)}  # 建立词典

    def get_vocab(self):
        return len(self.vocab), self.vocab
