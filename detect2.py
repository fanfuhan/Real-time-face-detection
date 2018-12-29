#-*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np
import cv2

def openVideo(window_name):
    cv2.namedWindow(window_name)
    
    cap = cv2.VideoCapture(0)  #获取视频数据
    classifier=cv2.CascadeClassifier('D:\software\python\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')
    color=(0,255,0)
    num = 0
    while cap.isOpened():
        ok,frame = cap.read()  #ok表示返回的状态  frame存储着图像数据矩阵 mat类型的
        if not ok:
            break
        
        #图像灰度化
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #加载分类器 opencv自带
        faceRects = classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        
        if len(faceRects) > 0 :
            for faceRect in faceRects:
                x,y,w,h = faceRect
                
                #将得到的人脸图片保存  用作神经网络的训练数据
                image_name='traindata/%d.jpg' % num  #这里为每个捕捉到的图片进行命名，每个图片按数字递增命名。
                image=frame[y-5:y+h+5,x:x+w]  #将当前帧含人脸部分保存为图片
                cv2.imwrite(image_name,image)
                
                num+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2) #画出举行框
                #font = cv2.FONT_HERSHEY_SIMPLEX              #获取内置字体
                #cv2.putText(frame,('%d'%num),(x+30,y+30),font,1,(255,0,255),4) #调用函数，对人脸坐标位置，添加一个(x+30,y+30）的矩形框用于显示当前捕捉到了多少人脸图片

        
        cv2.imshow(window_name,frame) #将捕获的数据显示出来
        c = cv2.waitKey(30)
        if c & 0xff == ord('q'): #按q退出
            break
        
    cap.release()
    cv2.destroyWindow(window_name)
    
#主程序调用方法运行
if __name__ == '__main__':   
    openVideo('openvideo')
    



    