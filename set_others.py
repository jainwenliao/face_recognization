import sys
import os
import cv2
''' 将others文件夹下面的数据集划分为train，test和validation三个部分'''
#原图片的文件夹地址，以及要保存的灰度图片的地址
input_dir = r'E:/Deep learning/face recognization/database/others'
output_dir1 = r'E:/face_recongnition/test/others/'
output_dir2 = r'E:/face_recongnition/train/others/'
output_dir3 = r'E:/face_recongnition/validation/others/'
size = 64
#检测输出地址是否存在，如果不存在就创建一个
if not os.path.exists(output_dir1 and output_dir2 and output_dir3):
    os.makedirs(output_dir1 and output_dir2 and output_dir3)
'''
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)
if not os.path.exists(output_dir3):
    os.makedirs(output_dir3)
'''
#人脸分类器
classfier = cv2.CascadeClassifier(r'E:\opencvCascade\\haarcascade_frontalface_alt_tree.xml')

index = 1

imgs = os.listdir(input_dir)#获取路径下的所有文件

for img in imgs:
   if img.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = input_dir + '/' + img

            image_name = img
            #读取路径下的所有图片
            img = cv2.imread(img_path)
            #将图片灰度化
            gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
            # 人脸检测，调整图片的尺寸
            faceRects = classfier.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 3, minSize = (64, 64))
            if len(faceRects) > 0:            #大于0则检测到人脸                                   
                for faceRect in faceRects:    #单独框出每一张人脸
                    x, y, w, h = faceRect  
                    cv2.imshow('image',img)
                    #将当前帧保存为图像 
                    if index <= 100:
                        faces_path = '%s%s.jpg'%(output_dir1,index)           
                        image = gray[y - 10: y + h + 10, x - 10: x + w + 10]
                        image = cv2.resize(image,(size,size))
                        cv2.imwrite(faces_path, image)
                    elif index <= 1100: 
                        faces_path = '%s%s.jpg'%(output_dir2,index)           
                        image = gray[y - 10: y + h + 10, x - 10: x + w + 10]
                        image = cv2.resize(image,(size,size))
                        cv2.imwrite(faces_path, image)
                    elif index <= 1200:
                        faces_path = '%s%s.jpg'%(output_dir3,index)           
                        image = gray[y - 10: y + h + 10, x - 10: x + w + 10]
                        image = cv2.resize(image,(size,size))
                        cv2.imwrite(faces_path, image)

                    index += 1

            if index == 1200:
                break

cv2.destroyAllWindows()