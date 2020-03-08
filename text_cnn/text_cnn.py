#!/usr/bin/env python
# coding: utf-8

# #### 导入所有需要的库

# In[1]:


import codecs
import numpy as np 
import random 
import string 
import  tensorflow.keras as keras 
import re 
from collections import Counter
import jieba
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import os 
train_file_path ="sample__train_0.2.txt"
valid_file_path ="sample__valid_0.2.txt"
test_file_path ="sample__test_0.2.txt" 
base_dir = os.path.curdir
file_list = [train_file_path ,valid_file_path  ,test_file_path] 
for i , filename in enumerate( file_list ) :
    file_list[i] = os.path.join( 
        os.path.abspath(base_dir) , filename) 
file_list


# In[3]:


def read_file(filename):
    """
    read_file 
    return label , content  use jieba lcut function 
    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation
    contents,labels=[],[]
    with codecs.open(filename,'r',encoding='utf-8') as f:
        for line in f:
            try:
                line=line.rstrip()
                assert len(line.split('\t'))==2
                label,content=line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        for w in jieba.cut(blk):
                            if len(w)>=2:
                                word.append(w)
                contents.append(word)
            except:
                pass
    return labels,contents


# In[4]:


def build_vocab(filenames,vocab_dir,vocab_size=8000):
    all_data = []
    for filename in filenames:
        _,data_train=read_file(filename)
        for content in data_train:
            all_data.extend(content)
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)
    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')


# In[ ]:


vocab_dir ="vocab.txt"
vocab_size = 8000
build_vocab(filenames= file_list,
            vocab_dir =vocab_dir,
            vocab_size=vocab_size)


# In[5]:


def read_vocab(vocab_dir):
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read()            .strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    categories = ['体育', '财经', '房产', '家居', 
                  '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return categories,cat_to_id


# In[6]:


vocab_dir ="vocab.txt"
words,word_to_id = read_vocab(vocab_dir)
categories,cat_to_id = read_category()


#  ### 训练词向量  ，利用word2vec
# 

# #### 1.生成semtences  , 必须是可迭代的对象

# In[7]:


re_han= re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)") # the method of cutting text by punctuation

class Make_Sentences(object):
    def __init__(self,filenames):
        self.filenames= filenames

    def __iter__(self):
        for filename in self.filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for _,line in enumerate(f):
                    try:
                        line=line.strip()
                        line=line.split('\t')
                        assert len(line)==2
                        blocks=re_han.split(line[1])
                        word=[]
                        for blk in blocks:
                            if re_han.match(blk):
                                word.extend(jieba.lcut(blk))
                        yield word
                    except:
                        pass
    


# In[ ]:


from gensim.models import word2vec
def train_word2vec(filenames ,vector_word_filename):
    import time
    t1 = time.time()
    sentences = Make_Sentences(filenames)
    model = word2vec.Word2Vec(sentences, 
                                size=100,
                                  window=5,
                               min_count=1,
                              workers=4)
    model.wv.save_word2vec_format(vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" %           (time.time() - t1))


# ####  或者加入停用词

# In[8]:


vocab_size = len(words)


# In[9]:


def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    with np.load(filename) as data:
        return data["embeddings"]


# In[10]:


vector_word_filename='vector_word.txt'  #vector_word trained by word2vec
train_word2vec(file_list ,vector_word_filename)
vector_word_npz='vector_word.npz'   # save vector_word to numpy file
# trans vector file to numpy file
if not os.path.exists(vector_word_npz):
    export_word2vec_vectors(word_to_id,
                            vector_word_filename,
                            vector_word_npz)
pre_trianing = get_training_word2vec_vectors(vector_word_npz)


# In[11]:


def process_file(filename,word_to_id,cat_to_id,max_length=600):
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad=keras.preprocessing.sequence.pad_sequences(data_id,
                                                     max_length,
                                                     padding='post', 
                                                     truncating='post')
    y_pad=keras.utils.to_categorical(label_id)
    return x_pad,y_pad

train_file_name = "sample__train_0.2.txt"
x_train,y_train = process_file(train_file_name ,
                               word_to_id,
                               cat_to_id,
                               max_length=600)


# In[12]:


valid_file_name = "sample__valid_0.2.txt"
x_valid,y_valid = process_file(valid_file_name ,
                               word_to_id,
                               cat_to_id,max_length=600)


# In[13]:


x_train.shape


# In[14]:


y_train.shape


# In[19]:


x_train


# In[15]:


y_train = y_train.astype("int")
y_train 


# In[16]:


train_embedding  = get_training_word2vec_vectors("vector_word.npz")


# In[17]:


train_embedding


# In[20]:


train_embedding.shape


# In[24]:


import tensorflow.keras as keras 
import tensorflow as tf 
from  tensorflow.keras import Sequential ,Model
from  tensorflow.keras.layers import Input ,Flatten ,Dropout ,                             Embedding ,Conv1D,MaxPooling1D ,Dense

max_length = 600 
model = Sequential()
main_input = Input(shape=( max_length), dtype='float64')
embedder = Embedding(vocab_size, 
                     100, 
                     input_length=x_train.shape[0],
                     weights=[train_embedding],
                     trainable=False)
#embedder = Embedding(len(vocab) + 1, 300, input_length=50, trainable=False)
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
main_output = Dense(10, activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# In[ ]:


history = model.fit(x_train,
          y_train,
          batch_size= 64 , 
          epochs=20 ,
         validation_data=(x_valid, y_valid))


# In[ ]:




