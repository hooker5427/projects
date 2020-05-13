import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Dense, Embedding, Input, Bidirectional, LSTM
from models.attention import  Attention
from config import *

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def rnn_attention_classifier(x_train , embedding_matrix  ):
    main_input = Input(shape=(max_length), dtype='float64')
    embedder = Embedding(vocab_size,
                         100,
                         input_length=x_train.shape[0],
                         weights=[embedding_matrix],
                         trainable=False)
    embed = embedder(main_input)
    output = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.25))(embed)

    output = Attention(max_length)(output)
    output = Dense(128, activation="relu")(output)
    output = Dropout(0.5)(output)
    output = Dense(2, activation="sigmoid")(output)
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss= loss_function ,
                  optimizer= optimizer ,
                  metrics=['accuracy']  )
    return model