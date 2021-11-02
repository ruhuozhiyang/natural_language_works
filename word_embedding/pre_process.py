# 训练词向量一般不需要去除停用词
# stop_words = {}

old_en_data_path = './data/en.txt'
new_en_data_path = './data/en_num.txt'


def list_filter_number(li):
    return list(filter(lambda x: not str(x).isdigit(), li))


def write_to_txt(li):
    with open(new_en_data_path, 'w') as fileObject:
        for one_line in li:
            for (index, word) in enumerate(one_line):
                fileObject.write(str(word2int[word]))
                if index < len(one_line) - 1:
                    fileObject.write(' ')
            fileObject.write('\n')


sentences = []
vocab_list = set()  # 所有词汇（去除重复的）
f = open(old_en_data_path)
lines = f.readlines()
for line in lines:
    # 去除各种标签符号
    line = line.strip().replace(',', '').replace('.', '').replace('"', '').replace('?', '')
    line = line.split()
    line = [word.lower() for word in line]  # 小写化
    line = list_filter_number(line)  # 过滤数字
    sentences.append(line)
    vocab_temp = set(line)
    vocab_list = set.union(vocab_list, vocab_temp)
word2int = {word: i for i, word in enumerate(vocab_list)}  # 建立词典 单词->数字
int2word = {char: ind for ind, char in word2int.items()}  # 数字->单词
write_to_txt(sentences)


def get_int2word():
    return int2word


def get_vocab_list():
    return vocab_list
