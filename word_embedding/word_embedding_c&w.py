import pre_process_zh
import pre_process_en
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from my_data_set_cw import MyDataSetCw

en_vector_path = './result/cw/result_vector.txt'
zh_vector_path = './result/cw/result_vector_zh.txt'

epochs = 20
CONTEXT_SIZE = 3
EMBEDDING_DIM = 50
learning_rate = 0.001
flag = 'en'


class CWModel(nn.Module):
    # 数据集、打分函数、得到两个得分score、损失函数
    def __init__(self, embedding_dim, vocab_size):
        super(CWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(7*embedding_dim, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return out


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    @staticmethod
    def forward(inputs):
        result = max(0, 1 - inputs[0] + inputs[1])
        return result


train_data = MyDataSetCw(flag, CONTEXT_SIZE)
vocab_len, vocab = train_data.get_vocab()
loss_function = LossFunction()
model = CWModel(EMBEDDING_DIM, vocab_len)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 三层循环
for epoch in range(epochs):
    total_loss = 0
    model.train()

    train_data = MyDataSetCw(flag, CONTEXT_SIZE)

    with tqdm(train_data) as t:
        t.set_description('Epoch {}/{}:'.format(epoch + 1, epochs))
        for index, (correct_sample, wrong_samples) in enumerate(t):
            model.zero_grad()
            correct_score = model(correct_sample)
            for wrong_sample in wrong_samples:
                wrong_score = model(wrong_sample)
                loss = loss_function(torch.tensor([correct_score, wrong_score], requires_grad=True))
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item()
                t.set_postfix(loss=total_loss)

with open(en_vector_path if flag == 'en' else zh_vector_path, 'w') as file_object:
    word2int = pre_process_en.get_word2int() if flag == 'en' else pre_process_zh.get_word2int()
    for item in vocab:
        file_object.write(item)
        file_object.write(' ')
        file_object.write(str(model.embeddings.weight[word2int[item]].cpu().detach().numpy().tolist()))
        file_object.write('\n')
torch.save(model, './result/cw/result_model.model')