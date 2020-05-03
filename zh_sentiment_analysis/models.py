import tensorflow as tf
import tensorflow.keras as keras
from numpy.random import seed
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Embedding, Input, Bidirectional, LSTM
from layers.attention import  Attention
SEED = 17
seed(SEED)
tf.random.set_seed(SEED)
max_seq_length = 100


def build_GRU(vocab_size=10000, embed_size=128, num_class=2):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embed_size, mask_zero=True, input_shape=[max_seq_length, ]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(num_class, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def text_cnn(vocab_size=10000, embed_size=128, ):
    main_input = keras.layers.Input(shape=(max_seq_length), dtype='float64')
    embedder = keras.layers.Embedding(vocab_size, embed_size)

    # embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = keras.layers.Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = keras.layers.MaxPooling1D(pool_size=38)(cnn1)
    cnn2 = keras.layers.Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = keras.layers.MaxPooling1D(pool_size=37)(cnn2)
    cnn3 = keras.layers.Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = keras.layers.MaxPooling1D(pool_size=36)(cnn3)
    # 合并三个模型的输出向量
    cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = keras.layers.Flatten()(cnn)
    drop = keras.layers.Dropout(0.2)(flat)
    main_output = keras.layers.Dense(2, activation='softmax')(drop)
    model = keras.Model(inputs=main_input, outputs=main_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    return model


def rnn_attention_classifier(vocab_size=10000, embed_size=128, ):
    inputs = Input(shape=(max_seq_length,))
    output = Embedding(100000, 300)(inputs)
    output = Bidirectional(LSTM(150, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(output)

    output = Attention(max_seq_length)(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.25)(output)
    output = Dense(2, activation="sigmoid")(output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
