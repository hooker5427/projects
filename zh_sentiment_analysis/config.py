# special token
special_words = ['<PAD>', '<UNK>']

# paths 
BASE_DIR = "../"
model_path = "../models/textcnn_sentimant.h5"
data_path = "../data/zh_sentiment_dataset_seg.txt"
vocab_path = "../data/vocab_words.txt"

#  train 
max_seq_length = 100
embedding_size = 100
hidden_size = 128
layer_size = 2
n_class = 2
learning_rate = 0.001
batch_size = 64
epochs = 10