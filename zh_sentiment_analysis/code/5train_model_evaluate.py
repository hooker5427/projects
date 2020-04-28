import tensorflow.keras as keras
from tensorflow.keras.preprocessing import sequence
from data_util import read_vocab, read_dataset
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as  plt
import pandas as pd

data_path = "../data/zh_sentiment_dataset_seg.txt"
vocab_path = "../data/vocab_words.txt"
special_words = ['<PAD>', '<UNK>']

# 加载词典和数据集
idx2vocab, vocab2idx = read_vocab(vocab_path, special_words)
all_datas, all_labels = read_dataset(data_path, vocab2idx)

vocab_size = len(vocab2idx)
max_seq_length = 100
embedding_size = 100
hidden_size = 128
layer_size = 2
n_class = 2

learning_rate = 0.001
batch_size = 64
epochs = 10

# --------------------padding and split data ------------------

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
x_val, y_val = new_datas[int(count * rate2):], new_labels[int(count * rate2):]


# -----------------train and test model------------------
def build_model(vocab_size=10000, embed_size=128, num_class=2):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embed_size, mask_zero=True, input_shape=[max_seq_length, ]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(num_class, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


model = build_model(vocab_size, embedding_size)

callbacks = []
es = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01, patience=3, verbose=0,
    restore_best_weights=True)

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=[es, ])

model.save('lstm_sentimant.h5')
model.to_yaml()


def plot_learning_curves(history):
    plt.figure(figsize=(8, 8))
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

model.evaluate(x_test)
