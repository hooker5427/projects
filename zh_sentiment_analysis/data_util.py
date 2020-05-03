import numpy as np
from tqdm import tqdm
import  tensorflow.keras as keras
from  tensorflow.keras.preprocessing import  sequence
from config import  *


def create_vocab(data_path, save_vocab_path):
    """
    根据分词数据集构建词典
    :param data_path: 数据集路径
    :param save_vocab_path:
    :return:
    """
    vocab_words = []
    with open(data_path, "r", encoding="utf8") as fo:
        for line in fo:
            sentence_words = line.strip().split("\t")[1].split()
            for word in sentence_words:
                if word not in vocab_words:
                    vocab_words.append(word)
    with open(save_vocab_path, "w", encoding="utf8") as fw:
        for word in vocab_words:
            fw.write(word + "\n")


def read_vocab(vocab_path, special_words):
    with open(vocab_path, "r", encoding="utf8") as fo:
        vocab_words = [word.strip() for word in fo]
    vocab_words = special_words + vocab_words
    idx2vocab = {idx: word for idx, word in enumerate(vocab_words)}
    vocab2idx = {word: idx for idx, word in enumerate(vocab_words)}
    return vocab_words, idx2vocab, vocab2idx


def read_dataset(data_path, vocab2idx):
    all_datas, all_labels = [], []
    with open(data_path, "r", encoding="utf8") as fo:
        lines = (line.strip() for line in fo)
        for line in tqdm(lines):
            label, sentence = line.split("\t")
            label = int(label)
            sentence = sentence.strip()
            sent2idx = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in sentence]
            all_datas.append(sent2idx)
            all_labels.append(label)
    return all_datas, all_labels


def load_stop_words(path):
    file = open(path, 'r', encoding='utf-8')
    stops = []
    for line in file.readlines():
        stops.append(line.strip())
    file.close()
    return stops


def text_to_idx(texts, max_length, word_to_index):
    X = []
    stop_words_path = './cn_stopwords.txt'
    stopwords = load_stop_words(stop_words_path)
    import jieba
    for text in texts:
        words_line = jieba.lcut(text)
        wlist = []
        for w in words_line:
            if w not in stopwords:
                wlist.append(w)
        ids = [word_to_index.get(w, word_to_index['<UNK>']) for w in wlist]
        if len(ids) >= max_length:
            ids = ids[:max_length]
        else:
            ids = ids + [word_to_index['<PAD>']] * (max_length - len(ids))
        X.append(ids)
    return np.array(X)


def train_test_split(all_datas, all_labels, test_size=0.2):
    count = len(all_labels)
    # 数据集划分比例
    rate1, rate2 = 0.8, 0.9  # train-0.8, test-0.1, dev-0.1
    # 数据的填充，不够长度左侧padding，大于长度右侧截断
    new_datas = sequence.pad_sequences(all_datas, maxlen=max_seq_length, padding='post', truncating='post')
    # 类别one-hot化
    new_labels = keras.utils.to_categorical(all_labels, n_class)
    # 根据比例划分训练集、测试集、验证集
    x_train, y_train = new_datas[:int(count * rate1)], new_labels[:int(count * rate1)]
    x_test, y_test = new_datas[int(count * rate1):int(count * rate2)], new_labels[int(count * rate1):int(count * rate2)]
    x_valid, y_valid = new_datas[int(count * rate2):], new_labels[int(count * rate2):]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def batch_generator(all_data, batch_size, shuffle=False):
    """
    :param all_data : all_data整个数据集
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 每次是否打乱顺序
    :return:
    """
    all_data = [np.array(d) for d in all_data]
    data_size = all_data[0].shape[0]
    if shuffle:
        p = np.random.permutation(data_size)
        all_data = [d[p] for d in all_data]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


def get_comments(keyword):
    path = './spider/jd_comment_' + keyword + ".txt"
    return open(path, 'r', encoding='utf-8').readlines()
