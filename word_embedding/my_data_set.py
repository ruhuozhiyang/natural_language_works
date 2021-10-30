# Pytorch用torch.utils.data.Dataset构建数据集，想要构建自己的数据集，则需继承Dataset类.
from torch.utils.data import Dataset

# 全局参数配置
en_data_dir = './data/en.txt'
zh_data_dir = './data/zh.txt'
CONTEXT_SIZE = 3  # 上下文词个数


class MyDataSet(Dataset):
    def __getitem__(self, index):
        return [self.test_sentence[index - j - 1] for j in range(CONTEXT_SIZE)], self.test_sentence[index]

    def __len__(self):
        return len(self.test_sentence)

    def __init__(self, load_flag):
        with open(en_data_dir if load_flag == 'en' else zh_data_dir) as f:
            content = f.read()
            f.close()
        self.test_sentence = content.split()

    def build_dict(self):
        vocab = set(self.test_sentence)  # 所有词汇（去除重复的）
        word2int = {word: i for i, word in enumerate(vocab)}  # 建立词典
        return vocab, word2int
