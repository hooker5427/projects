from tensorflow.keras.callbacks import EarlyStopping
from data_util import read_vocab, read_dataset ,train_test_split
import matplotlib.pyplot as  plt
import pandas as pd
import  tensorflow.keras as keras

from config import *

# 加载词典和数据集
vocab_words , idx2vocab, vocab2idx = read_vocab(vocab_path, special_words)
all_datas, all_labels = read_dataset(data_path, vocab2idx)

vocab_size = len(vocab2idx)

x_train, y_train, x_valid, y_valid, x_test, y_test = train_test_split(all_datas, all_labels)

from models import text_cnn

model = text_cnn(vocab_size, embedding_size)
keras.utils.plot_model( model  ,to_file='textcnn.png' , show_shapes= True )
callbacks = []
es = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01, patience=3, verbose=0,
    restore_best_weights=True)

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=(x_valid, y_valid),
                    callbacks=[es, ])

model.save('textcnn_sentimant.h5')


def plot_learning_curves(history):
    plt.figure(figsize=(8, 8))
    pd.DataFrame(history.history).plot(figsize=(8, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig("text_cnn_loss.png")
    plt.show()


plot_learning_curves(history)

model.evaluate(x_test)
