import cv2
import  random
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from preprocess import preprocess, horizon_flip, origal, GaussianBlur

train_file_label_path = "./file_label.txt"

file_list = []
label_list = []
for x in open(train_file_label_path, 'r').readlines():
    file, label = x.split('\t')
    file_list.append(file)
    label_list.append(label)

sample_imgae = cv2.imread(file_list[0], cv2.IMREAD_COLOR)
width, height, channles = sample_imgae.shape
print(width, height, channles)

num_classes = 4


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = preprocess(img)
    algo = random.choice([origal, horizon_flip, GaussianBlur])
    image = algo(img)

    return image


def create_batch(file_list, label_list, shuffle, batch_size):
    while True:
        if shuffle:
            np.random.seed(4)
            np.random.shuffle(file_list)
            np.random.seed(4)
            np.random.shuffle(label_list)

        X_batch = []
        y_batch = []
        number_batchs = len(file_list) // batch_size
        for i in range(number_batchs):
            start = i * batch_size
            end = start + batch_size
            for path in file_list[start:end]:
                img = load_image(path)
                X_batch.append(img)
            for y in label_list[start: end]:
                y_batch.append(y)
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch).reshape(-1, 1)
            yield X_batch, y_batch
            X_batch, y_batch = [], []


def build_model():
    # 建立序贯模型
    model = Sequential()

    model.add(Conv2D(64, (1, 1), strides=1, padding='same', input_shape=(width, height, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.summary()

    return model


from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
y = lbe.fit_transform(label_list)

print(y)
# labels = keras.utils.to_categorical(label_list, num_classes=num_classes)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(file_list, y, test_size=0.2)

model = build_model()

model.compile(optimizer='adam', metrics=['accuracy'], loss='sparse_categorical_crossentropy')

epoch = 2
batch_size = 8

steps_per_epoch_train = len(x_train) // batch_size
steps_per_epoch_valid = len(x_valid) // batch_size

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='accuracy', patience=2, min_delta=0.01)

train_generator = create_batch(x_train, y_train, shuffle=True, batch_size=batch_size)
valid_generator = create_batch(x_valid, y_valid, shuffle=False, batch_size=batch_size)

history = model.fit_generator(train_generator,
                              epochs=epoch,
                              steps_per_epoch=steps_per_epoch_train,
                              callbacks=[es, ],
                              validation_data=valid_generator,
                              validation_steps=steps_per_epoch_valid,
                              )


def plot_learing_rate(history):
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.DataFrame(history.history).plot()
    plt.gca()
    plt.savefig('loss_acc.png')


plot_learing_rate(history)

model.save('model.h5')

