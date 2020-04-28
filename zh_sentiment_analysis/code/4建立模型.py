import tensorflow as tf
import tensorflow.keras as keras
from numpy.random import seed

SEED = 17
seed(SEED)
tf.random.set_seed(SEED)


def build_model(vocab_size=10000, embed_size=128):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embed_size, mask_zero=True, input_shape=[None]),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.GRU(128),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
