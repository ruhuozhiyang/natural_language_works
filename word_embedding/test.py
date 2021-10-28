# from gensim.models.word2vec import Word2Vec
#
# en_wiki_word2vec_model = Word2Vec.load('./result/model.model')
#
# test_words = ['中国']
# for i in range(len(test_words)):
#     res = en_wiki_word2vec_model.wv.most_similar(test_words[i])
#     print(test_words[i])
#     print(res)


with open('./data/en.txt') as f:
    content = f.read()
    f.close()
print(content)
