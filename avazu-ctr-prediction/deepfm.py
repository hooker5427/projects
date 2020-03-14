import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import  warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from  sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder ,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

np.random.seed(10)

train_data = pd.read_csv("F:\\rs\\L8\\ctr\\train.csv")
train_data.head()

train_data.drop('hour'  ,axis  =1 ,  inplace= True )
train_data.drop('id'  ,axis  =1 ,  inplace= True )

feature1   =  [ 'device_id' ,'device_ip' , 'device_model'] 
feature2 = ['site_id' ,'site_domain']
feature3 = ['C1' ,'banner_pos' ,'site_category' ,'app_id',
 'app_domain','app_category' , 'device_type','device_conn_type',
 'C14','C15','C16','C17','C18','C19','C21'] 

train_features =  feature1 + feature2 + feature3 
for  feature in train_features :
    encoder = LabelEncoder()
    train_data[feature] = encoder.fit_transform(train_data[feature] )
target = ['click']


from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat,get_feature_names


# 计算每个特征中的 不同特征值的个数
fixlen_feature_column1 = [SparseFeat( name = feature, 
                                    vocabulary_size = int(train_data[feature].nunique() * 0.01 ) ,
                                    embedding_dim=4, 
                                     use_hash= True  ) 
                          for feature in feature1]
fixlen_feature_column2 = [SparseFeat( name = feature, 
                                   vocabulary_size = int( train_data[feature].nunique() * 0.05) ,
                                    embedding_dim=4, 
                                     use_hash= True  ) 
                          for feature in feature2]

fixlen_feature_column3 = [SparseFeat( name = feature,
                         vocabulary_size =  train_data[feature].nunique(),
                                    embedding_dim=4, 
                                     use_hash= False  ) 
                          for feature in feature3]
fixlen_feature_columns = fixlen_feature_column1 + fixlen_feature_column2 +fixlen_feature_column3 
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# 将数据集切分成训练集和测试集
train, valid = train_test_split(train_data, test_size=0.2)
train_model_input = {name:train[name].values for name in feature_names}
valid_model_input = {name:valid[name].values for name in feature_names}


model = DeepFM(linear_feature_columns,dnn_feature_columns,
               task='binary' , 
               dnn_hidden_units=( 128 ,  256  ) ,
                l2_reg_linear= 0.01 , 
                l2_reg_embedding=0.01 ,
                init_std=0.0001,
                seed=1024,
                dnn_dropout=0.3,
                dnn_activation='selu',
                dnn_use_bn=True,)


import tensorflow.keras as keras
import tensorflow as tf 
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(opt , "binary_crossentropy",
              metrics=['accuracy'], )


history = model.fit(train_model_input, train[target].values,
                    batch_size=128, epochs=30, verbose=1, 
                   validation_data= (valid_model_input , valid[target].values) )

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)