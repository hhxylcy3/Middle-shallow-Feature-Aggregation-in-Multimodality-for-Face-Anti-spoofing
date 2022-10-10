import os
import glob
'''功能描述：将训练集中所有图片的路径和标签保存到记事本中'''

#dirpath = "E:\\学习\\face_anti\\(mlfa)ChaLearn_liveness_challenge-master\\ChaLearn_liveness_challenge-master\\mydata"
#print(os.listdir(dirpath))
file = open('.\\02.txt','w')
path = os.getcwd()
dirpath = path+".\\mydata\\fake_part\\fake_train\\*\\*_en*_s*\\*\\*.jpg"
fakepath= glob.glob(dirpath)   #获取当前更目录下的mydata文件夹下的所有文件夹下的名为*_train的文件夹下所有文件夹下的所有文件夹下的所有jpg图像
#print(fakepath)
realpath = glob.glob(path+".\\mydata\\real_part\\real_train\\*\\*\\*\\*.jpg")
path2 = fakepath + realpath     #将两个列表合并成一个列表
print(path2)
for x in path2:
    if 'real' in str(x):
        file.write(str(x)+' 1'+'\n') 
    else:
        file.write(str(x)+' 0'+'\n')
file.close()
