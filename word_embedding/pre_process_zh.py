old_zh_data_path = './data/zh.txt'
new_zh_data_path = './data/zh_num.txt'

symbols_removed = ['、', '。', '!', '？', '?', ',', '”', '“', '‘', '’']


def list_filter_number(li):
    return list(filter(lambda x: not str(x).isdigit(), li))


def write_to_txt(li):
    with open(new_zh_data_path, 'w') as fileObject:
        for one_line in li:
            for (index, word) in enumerate(one_line):
                fileObject.write(str(word2int[word]))
                if index < len(one_line) - 1:
                    fileObject.write(' ')
            fileObject.write('\n')


sentences = []
vocab_list = set()  # 所有词汇（去除重复的）
f = open(old_zh_data_path)
lines = f.readlines()
for line in lines:
    # 去除各种标签符号
    line = line.strip()
    for symbol in symbols_removed:
        line = line.replace(symbol, '')
    line = line.split()
    line = list_filter_number(line)  # 过滤数字
    sentences.append(line)
    vocab_temp = set(line)
    vocab_list = set.union(vocab_list, vocab_temp)
word2int = {word: i for i, word in enumerate(vocab_list)}  # 建立词典 汉字词语->数字
int2word = {char: ind for ind, char in word2int.items()}  # 数字->汉字词语
write_to_txt(sentences)


def get_int2word():
    return int2word


def get_vocab_list():
    return vocab_list
