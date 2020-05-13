import tensorflow as tf
import tensorflow.keras as keras
from config import *

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def build_GRU(x_train, embedding_matrix, vocab_size=8000, num_class=10):
    main_input = keras.layers.Input(shape=(max_length), dtype='float64')
    embedder = keras.layers.Embedding(vocab_size,
                                      100,
                                      input_length=x_train.shape[0],
                                      weights=[embedding_matrix],
                                      trainable=False)
    embed = embedder(main_input)

    gru1 = keras.layers.GRU(128, return_sequences=True)
    gru1_ = gru1(embed)

    gru2 = keras.layers.GRU(128 )
    gru2_ = gru2(gru1_)

    out_layer  = keras.layers.Dense(num_class )
    out = out_layer( gru2_)
    model = keras.Model( main_input ,out  )

    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    model.summary()

    return model
