import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import pre_process
from my_data_set import MyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm

# 超参数
CONTEXT_SIZE = 3  # 上下文词个数
EMBEDDING_DIM = 50  # 词向量维度
per_batch_size = 50
epochs = 20

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_dataset = MyDataSet('en')
vocab_len, vocab = train_dataset.get_vocab()
# train_loader = DataLoader(dataset=train_dataset, batch_size=per_batch_size)


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


loss_function = nn.NLLLoss()
model = NGramLanguageModeler(vocab_len, EMBEDDING_DIM, CONTEXT_SIZE)
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
            log_probs = model(context_tensor)
            target_tensor = torch.tensor([target], dtype=torch.long)
            target_tensor = target_tensor.to(device)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            t.set_postfix(loss=total_loss)

with open('./result/result_vector.txt', 'w') as file_object:
    int2word = pre_process.get_int2word()
    for item in vocab:
        file_object.write(int2word[int(item)])
        file_object.write(' ')
        # TypeError: can't convert cuda:0 device type tensor to numpy.
        # Use Tensor.cpu() to copy the tensor to host memory first.
        file_object.write(str(model.embeddings.weight[int(item)].cpu().detach().numpy().tolist()))
        file_object.write('\n')
torch.save(model, './result/result_model.model')
