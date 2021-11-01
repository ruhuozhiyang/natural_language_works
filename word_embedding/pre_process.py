# 训练词向量一般不需要去除停用词
# stop_words = {}


f = open('./data/en.txt')
lines = f.readlines()
for line in lines:
    print(line)
# test_sentence = content.split()
# vocab = set(test_sentence)  # 所有词汇（去除重复的）
# word2int = {word: i for i, word in enumerate(vocab)}  # 建立词典
# int2word = {char: ind for ind, char in word2int.items()}
# sentence2int = [word2int[item] for item in test_sentence]

# with open('./data/en_num.txt', 'w') as fileObject:
#     for item in sentence2int:
#         fileObject.write(str(item))
#         if int2word[item] == '.':
#             fileObject.write('\n')
#         else:
#             fileObject.write(' ')


# def get_int2word():
#     return int2word
