import  os
import cv2

train_path ="../training"

file_list = []

for path1 in os.listdir(train_path):
    path1 = os.path.join( train_path , path1)
    for  path2 in os.listdir( path1 ) :
        path2 = os.path.join( path1 ,  path2 )
        for path in  os.listdir( path2 ) :
            path = os.path.join( path2 , path )
            path = path.replace("\\" ,'/')
            file_list.append( path )

with  open('file_label.txt' ,'w' ,encoding= 'utf-8')  as file :
    for path  in file_list :
        label  = path.split('/')[2]
        file.write( path + '\t' + label + '\n')


print ("load  image  ok! ")

sample_image_path = file_list[0]

sample_image = cv2.imread( sample_image_path , cv2.IMREAD_COLOR )
cv2.imshow( 'win1' , sample_image)
cv2.waitKey()

