import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from my_data_set import MyDataSet
from torch.utils.data import DataLoader
from tqdm import tqdm

CONTEXT_SIZE = 3  # 上下文词个数
EMBEDDING_DIM = 50  # 词向量维度
per_batch_size = 50

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

train_dataset = MyDataSet('en')
vocab, word2int = train_dataset.build_dict()
train_loader = DataLoader(dataset=train_dataset, batch_size=per_batch_size)


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
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
model.to(device)
optimizer = optimizer.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    model.train()

    with tqdm(train_dataset) as t:
        t.set_description('Epoch {}/10:'.format(epoch + 1))
        t.set_postfix(loss=total_loss)
        for index, (context, target) in enumerate(t):
            context_tensor = torch.LongTensor([word2int[w] for w in context])
            context_tensor.to(device)
            target_tensor = torch.LongTensor([word2int[target]])
            target_tensor.to(device)
            model.zero_grad()
            log_probs = model(context_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

# To get the embedding of a particular word, e.g. "beauty"
print(model.embeddings.weight[word2int["beauty"]])
