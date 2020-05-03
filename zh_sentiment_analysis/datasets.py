import numpy as np


def read_vocab(vocab_path, special_words):
    with  open(vocab_path, mode='r', encoding="utf-8") as file:
        tempwords = file.readlines()
        vocab_words = [word.strip().rstrip() for word in tempwords]
        vocab_words = special_words + vocab_words
        id2word = {idx: word for idx, word in enumerate(vocab_words)}
        word2id = {word: idx for idx, word in enumerate(vocab_words)}
        return id2word, word2id


def create_datasets(data_path, word2id):
    all_datasets, all_lables = [], []
    with open(data_path, mode='r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines()]
        for line in lines:
            label, sentence = line.split("\t")
            label = int(label)
            sentence = sentence.strip()
            all_datasets.append([word2id[word] if word in word2id else word2id["<unk>"] for word in sentence])
            all_lables.append(label)
    return all_datasets, all_lables


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


if __name__ == '__main__':
    vocab_path = "../data/vocab_words.txt"
    data_path = "../data/zh_sentiment_dataset_seg.txt"

    id2word, word2id = read_vocab(vocab_path, ['<pad>', "<unk>"])
    sample = ["我", "要", "去", "深圳", "<pad>"]
    indices = [word2id[w] for w in sample]
    print(indices)

    gen_setences = [id2word[i] for i in indices]
    print(gen_setences)
