import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizer
from torch.utils.data import DataLoader

import pre_process
import pre_process_zh
from my_data_set_rnn import MyDataSetRnn
from tqdm import tqdm

# 超参数
CONTEXT_SIZE = 1  # 上下文词个数
EMBEDDING_DIM = 50  # 词向量维度
rnn_layers = 3
rnn_neurons = 128
per_batch_size = 50
epochs = 20

flag = 'zh'
en_vector_path = './result/result_vector.txt'
zh_vector_path = './result/result_vector_zh.txt'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_dataset = MyDataSetRnn(flag, CONTEXT_SIZE)
vocab_len, vocab = train_dataset.get_vocab()
# RNN是带有时序信息的。DNN全连接层Dense的输入维度是不能变化的。
# RNN得输入的是一个序列.图片就按列/行，看成长度为图像边长像素的序列。文本也得输入一段文字，才能看成序列。
train_loader = DataLoader(dataset=train_dataset, batch_size=per_batch_size)


class RNNLM2Gram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(RNNLM2Gram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.basic_rnn = nn.LSTM(context_size * embedding_dim, rnn_neurons, rnn_layers)
        self.FC = nn.Linear(rnn_neurons, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).permute(1, 0, 2, 3)
        embeds = embeds.view(embeds.size(0), embeds.size(1), -1)
        out, _ = self.basic_rnn(embeds)
        # print(out[-1, :, :])  为[batch_size, rnn_neurons]
        out = self.FC(out[-1, :, :])  # 取最后一个序列，并将其塞入Linear.
        probability = f.log_softmax(out, dim=1)
        return probability


loss_function = nn.NLLLoss()
model = RNNLM2Gram(vocab_len, EMBEDDING_DIM, CONTEXT_SIZE)
model.to(device)
optimizer = optimizer.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    total_loss = 0
    model.train()

    with tqdm(train_dataset) as t:
        t.set_description('Epoch {}/{}:'.format(epoch + 1, epochs))
        for index, (context_tensor, target_tensor) in enumerate(t):
            model.zero_grad()
            context_tensor = context_tensor.view(1, context_tensor.size(0), -1)
            context_tensor = context_tensor.to(device)
            target_tensor = target_tensor.to(device)
            result_prob = model(context_tensor)
            loss = loss_function(result_prob, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            t.set_postfix(loss=total_loss)

with open(en_vector_path if flag == 'en' else zh_vector_path, 'w') as file_object:
    word2int = pre_process.get_word2int() if flag == 'en' else pre_process_zh.get_word2int()
    for item in vocab:
        file_object.write(item)
        file_object.write(' ')
        # TypeError: can't convert cuda:0 device type tensor to numpy.
        # Use Tensor.cpu() to copy the tensor to host memory first.
        file_object.write(str(model.embeddings.weight[word2int(item)].cpu().detach().numpy().tolist()))
        file_object.write('\n')
torch.save(model, './result/result_model.model')
