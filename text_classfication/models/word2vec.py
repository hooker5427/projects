import os
from config import *
from utils.preprocessing import *
import warnings
warnings.filterwarnings("ignore")


base_dir = os.path.curdir
file_list = [train_file_path, valid_file_path, test_file_path]
for i, filename in enumerate(file_list):
    file_list[i] = os.path.join(
        os.path.abspath(base_dir), filename)

build_vocab(filenames=file_list, vocab_dir=vocab_dir, vocab_size=vocab_size)

words, word_to_id = read_vocab(vocab_dir)
categories, cat_to_id = read_category()

vocab_size = len(words)

vector_word_filename = 'vector_word.txt'  # vector_word trained by word2vec
if  not os.path.exists( vector_word_filename):
    train_word2vec(file_list, vector_word_filename)
vector_word_npz = 'vector_word.npz'  # save vector_word to numpy file
# trans vector file to numpy file
if not os.path.exists(vector_word_npz):
    export_word2vec_vectors(word_to_id,
                            vector_word_filename,
                            vector_word_npz)


if __name__ == '__main__':
    # load embedding
    pre_trianing = get_training_word2vec_vectors(vector_word_npz)
