import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

import pre_process_en
import pre_process_zh
from my_data_set import MyDataSet
# from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参数
CONTEXT_SIZE = 3  # 上下文词个数
EMBEDDING_DIM = 50  # 词向量维度
# per_batch_size = 50
epochs = 20

flag = 'zh'
en_vector_path = './result/dnn/result_vector.txt'
zh_vector_path = './result/dnn/result_vector_zh.txt'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_dataset = MyDataSet(flag, CONTEXT_SIZE)
vocab_len, vocab = train_dataset.get_vocab()
# train_loader = DataLoader(dataset=train_dataset, batch_size=per_batch_size)


class DNNLMNGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(DNNLMNGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        probability = F.log_softmax(out, dim=1)
        return probability


loss_function = nn.NLLLoss()
model = DNNLMNGram(vocab_len, EMBEDDING_DIM, CONTEXT_SIZE)
model.to(device)
optimizer = optimizer.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    total_loss = 0
    model.train()

    with tqdm(train_dataset) as t:
        t.set_description('Epoch {}/{}:'.format(epoch + 1, epochs))
        for index, (context_tensor, target) in enumerate(t):
            model.zero_grad()
            context_tensor = context_tensor.to(device)  # 这行代码气死我了
            result_prob = model(context_tensor)
            target_tensor = torch.tensor([target])
            target_tensor = target_tensor.to(device)
            print(result_prob)
            print(target_tensor)
            loss = loss_function(result_prob, target_tensor)
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
        file_object.write(str(model.embeddings.weight[word2int(item)].cpu().detach().numpy().tolist()))
        file_object.write('\n')
torch.save(model, './result/dnn/result_model.model')
