import codecs
import re
import jieba
from collections import Counter
import time
import numpy as np
import tensorflow.keras as keras
from gensim.models import word2vec

re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation


def read_file(filename):
    """
    read_file
    return label , content  use jieba lcut function
    """
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        for w in jieba.cut(blk):
                            if len(w) >= 2:
                                word.append(w)
                contents.append(word)
            except:
                pass
    return labels, contents


def build_vocab(filenames, vocab_dir, vocab_size=8000):
    all_data = []
    for filename in filenames:
        _, data_train = read_file(filename)
        for content in data_train:
            all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)
    with codecs.open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居',
                  '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


class Make_Sentences(object):
    def __init__(self, filenames):
        self.filenames = filenames

    def __iter__(self):
        for filename in self.filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for _, line in enumerate(f):
                    try:
                        line = line.strip()
                        line = line.split('\t')
                        assert len(line) == 2
                        blocks = re_han.split(line[1])
                        word = []
                        for blk in blocks:
                            if re_han.match(blk):
                                word.extend(jieba.lcut(blk))
                        yield word
                    except:
                        pass


def train_word2vec(filenames, vector_word_filename):
    t1 = time.time()
    sentences = Make_Sentences(filenames)
    model = word2vec.Word2Vec(sentences,
                              size=100,
                              window=5,
                              min_count=1,
                              workers=4)
    model.wv.save_word2vec_format(vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))




def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_training_word2vec_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    labels, contents = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id,
                                                       max_length,
                                                       padding='post',
                                                       truncating='post')
    y_pad = keras.utils.to_categorical(label_id)
    return x_pad, y_pad



