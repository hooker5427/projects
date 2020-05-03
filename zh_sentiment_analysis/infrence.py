import os
import numpy as np
import tensorflow.keras as  keras
from config import *
from data_util import text_to_idx


def get_response(texts, vocab2idx):
    # import time

    if not os.listdir(os.path.join(BASE_DIR, 'models')):
        print("先进行训练")
        return None, None
    else:
        model = keras.models.load_model(model_path)
        print("加载模型成功")
        # start = time.time()
        X = text_to_idx(texts, max_seq_length, vocab2idx)

        print(X)
        y_pred = model.predict(X)
        y_label = np.argmax(y_pred, axis=1)

        print(y_label)
        # print("每条评论耗时", (time.time() - start) / len(texts))

        return y_pred, y_label


if __name__ == '__main__':
    get_response(["这东西真的垃圾！", "好东西啊,下次再来买"])
