import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from tqdm import tqdm


CONTEXT_SIZE = 3  # 上下文词个数
EMBEDDING_DIM = 50  # 词向量维度

with open('./data/en.txt') as f:
    content = f.read()
    f.close()
test_sentence = content.split()
# we should tokenize the input, but we will ignore that for now
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)], test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
vocab = set(test_sentence)  # 所有词汇（去除重复的）
word2int = {word: i for i, word in enumerate(vocab)}  # 建立词典


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
optimizer = optimizer.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    with tqdm(ngrams) as t:
        for context, target in t:
            t.set_description('Epoch {}/10:'.format(epoch + 1))
            t.set_postfix(loss=total_loss)
            context_tensor = torch.tensor([word2int[w] for w in context], dtype=torch.long)
            target_tensor = torch.tensor([word2int[target]], dtype=torch.long)
            model.zero_grad()
            log_probs = model(context_tensor)
            loss = loss_function(log_probs, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

# To get the embedding of a particular word, e.g. "beauty"
print(model.embeddings.weight[word2int["beauty"]])
