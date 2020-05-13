import codecs
import random
def sample_file ( filename , type_str , thresold  ) :
    res = []
    with codecs.open( filename , 'r' ,encoding= "utf-8") as file  :
        while True :
            line = file.readline()
            if  random.random() < thresold :
                res.append(  line)
            else :
                pass
            if not line :
                break
    savefilename = "sample__%s_%s"%( type_str ,thresold) +".txt"
    with open (savefilename, 'w' ,encoding= 'utf-8') as file :
        file.write(  "".join(res))


if __name__ == '__main__':
    train_file_name = "F:\\tsinguhacnews\\cnews.train.txt"
    valid_file_name = "F:\\tsinguhacnews\\cnews.val.txt"
    test_file_name = "F:\\tsinguhacnews\\cnews.test.txt"

    sample_file(train_file_name, "train", 0.2)
    sample_file(valid_file_name, "valid", 0.2)
    sample_file(test_file_name,"test" ,  0.2)