# from gensim.models.word2vec import Word2Vec
#
# en_wiki_word2vec_model = Word2Vec.load('./result/model.model')
#
# test_words = ['中国']
# for i in range(len(test_words)):
#     res = en_wiki_word2vec_model.wv.most_similar(test_words[i])
#     print(test_words[i])
#     print(res)
import torch
from torch.nn.functional import log_softmax


inputs = torch.LongTensor([[2, 3], [2, 5]])

print(torch.mean(inputs, dim=0))