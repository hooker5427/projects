import numpy as np
import pandas as pd

data_path = "../data/zh_sentiment_dataset_seg.txt"

file = open(data_path, 'r', encoding='utf-8')
print(repr(file.readline()))

file.close()

# '1\t手机 质量 不错 ， 帮 亲戚 购买 的 。\n'
# \t  分割
