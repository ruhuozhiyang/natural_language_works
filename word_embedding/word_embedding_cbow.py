import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optimizer
import numpy as np
import pre_process_en
import pre_process_zh
from my_data_set_cbow import MyDataSetCBow
from tqdm import tqdm

# 超参数
CONTEXT_SIZE = 3  # 上下文词个数
EMBEDDING_DIM = 50  # 词向量维度
epochs = 20

flag = 'en'
en_vector_path = './result/cbow/result_vector.txt'
zh_vector_path = './result/cbow/result_vector_zh.txt'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_dataset = MyDataSetCBow(flag, CONTEXT_SIZE)
vocab_len, vocab = train_dataset.get_vocab()


class CBowModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBowModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear2 = nn.Linear(6, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = np.mean(embeds.detach().numpy().tolist(), axis=0)
        out = torch.tensor(out)
        probability = f.log_softmax(self.linear2(out))
        return probability


loss_function = nn.NLLLoss()
model = CBowModel(vocab_len, EMBEDDING_DIM, CONTEXT_SIZE)
model.to(device)
optimizer = optimizer.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    total_loss = 0
    model.train()

    train_dataset = MyDataSetCBow(flag, CONTEXT_SIZE)

    with tqdm(train_dataset) as t:
        t.set_description('Epoch {}/{}:'.format(epoch + 1, epochs))
        for index, (context_tensor, target_tensor) in enumerate(t):
            # print(context_tensor)
            # print(target_tensor)
            model.zero_grad()
            context_tensor = context_tensor.to(device)
            target_tensor = target_tensor.to(device)
            result_prob = model(context_tensor)
            print(result_prob)
            loss = loss_function(torch.tensor([result_prob.numpy()]), target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            t.set_postfix(loss=total_loss)

with open(en_vector_path if flag == 'en' else zh_vector_path, 'w') as file_object:
    word2int = pre_process_en.get_word2int() if flag == 'en' else pre_process_zh.get_word2int()
    for item in vocab:
        file_object.write(item)
        file_object.write(' ')
        # TypeError: can't convert cuda:0 device type tensor to numpy.
        # Use Tensor.cpu() to copy the tensor to host memory first.
        file_object.write(str(model.embeddings.weight[word2int[item]].cpu().detach().numpy().tolist()))
        file_object.write('\n')
torch.save(model, './result/cbow/result_model.model')
