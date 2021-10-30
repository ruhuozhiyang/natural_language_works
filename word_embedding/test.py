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
test_sentence = content.split()
vocab = set(test_sentence)  # 所有词汇（去除重复的）
word2int = {word: i for i, word in enumerate(vocab)}  # 建立词典
int2word = {char: ind for ind, char in word2int.items()}
sentence2int = [word2int[item] for item in test_sentence]
fileObject = open('./data/en_num.txt', 'w')
for item in sentence2int:
    fileObject.write(str(item))
    if int2word[item] == '.':
        fileObject.write('\n')
    else:
        fileObject.write(' ')
fileObject.close()

# from my_data_set import MyDataSet
# test = MyDataSet('en')
# print(len(test))
