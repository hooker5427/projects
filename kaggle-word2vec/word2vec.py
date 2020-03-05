#!/usr/bin/env python
# coding: utf-8

# ## import所需库

# In[1]:


# bs4 nltk gensim
import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords


# In[3]:


# nltk.download()


# ### 用pandas读入训练数据

# In[2]:


df = pd.read_csv('../data/labeledTrainData.tsv', sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df.head()


# ### 对影评数据做预处理，大概有以下环节：
# 
# 1. 去掉html标签
# 1. 移除标点
# 1. 切分成词/token
# 1. 去掉停用词
# 1. 重组为新的句子

# In[3]:


df['review'][1000]


# In[4]:


#去掉HTML标签的数据
example = BeautifulSoup(df['review'][1000], 'html.parser').get_text()
example


# In[5]:


#去掉标点符号
example_letters = re.sub(r'[^a-zA-Z]', ' ', example)
example_letters


# In[6]:


words = example_letters.lower().split()
words


# In[7]:


#下载停用词和其他语料会用到
#nltk.download()


# In[8]:


#去停用词
stopwords = {}.fromkeys([ line.rstrip() for line in open('../stopwords.txt')])
words_nostop = [w for w in words if w not in stopwords]
words_nostop


# In[9]:


eng_stopwords = set(stopwords)

def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [w for w in words if w not in eng_stopwords]
    return ' '.join(words)


# In[10]:


df['review'][1000]


# In[11]:


clean_text(df['review'][1000])


# ### 清洗数据添加到dataframe里

# In[12]:


df['clean_review'] = df.review.apply(clean_text)
df.head()


# ### 抽取bag of words特征(用sklearn的CountVectorizer)

# In[13]:


vectorizer = CountVectorizer(max_features = 5000) 
train_data_features = vectorizer.fit_transform(df.clean_review).toarray()
train_data_features.shape


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data_features,df.sentiment,test_size = 0.2, random_state = 0)


# In[16]:


import matplotlib.pyplot as plt
import itertools
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### 训练分类器

# In[17]:


LR_model = LogisticRegression()
LR_model = LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

print("accuracy metric in the testing dataset: ", (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[18]:


df = pd.read_csv('../data/unlabeledTrainData.tsv', sep='\t', escapechar='\\')
print('Number of reviews: {}'.format(len(df)))
df.head()


# In[19]:


df['clean_review'] = df.review.apply(clean_text)
df.head()


# In[20]:


review_part = df['clean_review']
review_part.shape


# In[21]:


import warnings
warnings.filterwarnings("ignore")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def split_sentences(review):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = [clean_text(s) for s in raw_sentences if s]
    return sentences
sentences = sum(review_part.apply(split_sentences), [])
print('{} reviews -> {} sentences'.format(len(review_part), len(sentences)))


# In[22]:


sentences[0]


# In[23]:


sentences_list = []
for line in sentences:
    sentences_list.append(nltk.word_tokenize(line))


# -  sentences：可以是一个list
# -  sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
# -  size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
# -  window：表示当前词与预测词在一个句子中的最大距离是多少
# -  alpha: 是学习速率
# -  seed：用于随机数发生器。与初始化词向量有关。
# -  min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
# -  max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
# 
# -  workers参数控制训练的并行数。
# -  hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（defau·t），则negative sampling会被使用。
# -  negative: 如果>0,则会采用negativesamp·ing，用于设置多少个noise words
# -  iter： 迭代次数，默认为5

# In[24]:


# 设定词向量训练的参数
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
model_name = '{}features_{}minwords_{}context.model'.format(num_features, min_word_count, context)


# In[25]:


from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences_list, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model.save(os.path.join('..', 'models', model_name))


# In[26]:


print(model.doesnt_match(['man','woman','child','kitchen']))
#print(model.doesnt_match('france england germany berlin'.split())


# In[27]:


model.most_similar("boy")


# In[28]:


model.most_similar("bad")


# In[29]:


df = pd.read_csv('../data/labeledTrainData.tsv', sep='\t', escapechar='\\')
df.head()


# In[30]:


from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

def clean_text(text, remove_stopwords=False):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in eng_stopwords]
    return words

def to_review_vector(review):
    global word_vec
    
    review = clean_text(review, remove_stopwords=True)
    #print (review)
    #words = nltk.word_tokenize(review)
    word_vec = np.zeros((1,300))
    for word in review:
        #word_vec = np.zeros((1,300))
        if word in model:
            word_vec += np.array([model[word]])
    #print (word_vec.mean(axis = 0))
    return pd.Series(word_vec.mean(axis = 0))

train_data_features = df.review.apply(to_review_vector)
train_data_features.head()


# In[32]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data_features,df.sentiment,test_size = 0.2, random_state = 0)


# In[33]:


LR_model = LogisticRegression()
LR_model = LR_model.fit(X_train, y_train)
y_pred = LR_model.predict(X_test)
cnf_matrix = confusion_matrix(y_test,y_pred)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

print("accuracy metric in the testing dataset: ", (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()


# In[ ]:




