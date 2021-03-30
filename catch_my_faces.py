import os
import sys
import cv2
from PIL import Image


def get_faces(window_name, camera_id, catch_pic_num, path):  

    size = 64
    cv2.namedWindow(window_name)#给窗口命名

    #视频来源,通过摄像通获取视频流，0表示电脑自带摄像头
    cap = cv2.VideoCapture(camera_id)

    #使用默认的人脸分类器
    classfier = cv2.CascadeClassifier(r'E:\opencvCascade\\haarcascade_frontalface_default.xml')

    #识别出人脸后画出边框的颜色

    color = (0, 255, 0)
    num = 0

    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:  
            break

        #将当前帧转换为灰度图像

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #人脸检测，1.2和2 分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (64, 64))
        
        if len(faceRects) > 0:            #大于0则检测到人脸                                   
            for faceRect in faceRects:    #单独框出每一张人脸
                x, y, w, h = faceRect        
            name = 'liaojianwen' + str(num)
            #将当前帧保存为图像
            img_name = '%s/%s.jpg'%(path,name)               
            image = gray[y - 10: y + h + 10, x - 10: x + w + 10]
            image = cv2.resize(image,(size,size))
            cv2.imwrite(img_name, image) 


            num += 1
            if num > (catch_pic_num): 
                 break

            #画出矩形
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

            #显示拍摄的图片数
            #字体设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, #图片
            'num: %d'%(num),#添加的文字
            (x+30, y+30), #左上角坐标
            font,#字体 
            1, #字体大小
            (255,0,255),#颜色
            4#字体粗细
            )

        #超过最大指定数，结束
        if num > (catch_pic_num): break



        #显示图像
        cv2.imshow(window_name,frame)
        c = cv2.waitKey(10)

        if c & 0xFF == ord('q'): 
            break 

    cap.release()
    cv2.destroyAllWindows()

#只有在该脚本下才能运行，import到其他脚本不被运行
if __name__ == '__main__':
    if len(sys.argv) != 1: 
        print('Usage: %s camera_idex \r\n'%(sys.argv[0]))
    else:  
        #get_faces('catch_faces',0,100, 'E:\\face_recongnition\\test\\liaojianwen')

        #get_faces('catch_faces',0,1000, 'E:\\face_recongnition\\train\\liaojianwen')

        get_faces('catch_faces',0,100, 'E:\\face_recongnition\\validation\\liaojianwen')

    