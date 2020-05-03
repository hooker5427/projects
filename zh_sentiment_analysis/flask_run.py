from flask import Flask, Response, json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_util import read_vocab
from config import *
import warnings
import tensorflow.keras as keras
from data_util import text_to_idx

warnings.filterwarnings('ignore')

from infrence import get_response
from data_util import get_comments

app = Flask(__name__)

# 加载词典和数据集
vocab_words, idx2vocab, vocab2idx = read_vocab(vocab_path, special_words)
vocab_size = len(vocab2idx)

print("词表信息， 加载成功！ ")

model = keras.models.load_model(model_path)
print("加载模型成功")


@app.route('/query/<keyword>', methods=['POST', 'GET'])
def query(keyword):
    content = get_comments(keyword)
    print("加载评论成功!")
    X = text_to_idx(content, max_seq_length, vocab2idx)
    print(X)
    y_pred = model.predict(X)
    y_label = np.argmax(y_pred, axis=1)

    postive_score = round(np.sum(y_label) / (len(y_label)), 2)
    negtiave_score = 1 - postive_score

    sns.countplot(y_label)
    plt.show()

    return Response(json.dumps({'postive_score': postive_score, "negtiave_score": negtiave_score, 'type': 'segment'}),
                    content_type='application/json')


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8888, debug=False)
