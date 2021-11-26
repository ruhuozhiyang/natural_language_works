# 以下代码是测试代码，测试使用word2vec生成的语言模型词向量.
from gensim.models.word2vec import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('./result/model.model')
test_words = ['中国']
for i in range(len(test_words)):
    res = en_wiki_word2vec_model.wv.most_similar(test_words[i])
    print(test_words[i])
    print(res)