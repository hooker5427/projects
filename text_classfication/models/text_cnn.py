import tensorflow as tf
from tensorflow.keras import  Model
from tensorflow.keras.layers import Input, Flatten, Dropout, Embedding, Conv1D, MaxPooling1D, Dense
from config import  *


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)




def build_text_cnn( x_train , embedding_matrix ):

    main_input = Input(shape=(max_length), dtype='float64')
    embedder = Embedding(vocab_size,
                         100,
                         input_length=x_train.shape[0],
                         weights=[embedding_matrix],
                         trainable=False)
    # embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
    embed = embedder(main_input)
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
    cnn1 = MaxPooling1D(pool_size=38)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=37)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=36)(cnn3)
    # 合并三个模型的输出向量
    cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3], axis=1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    main_output = Dense(10 )(drop)
    model = Model(inputs=main_input, outputs=main_output)
    model.compile(loss= loss_function ,
                  optimizer= optimizer ,
                  metrics=['accuracy']  )

    model.summary()

    return  model