import cv2
import numpy as np
import svm

# 鼠标回调函数
# x, y 都是相对于窗口内的图像的位置

def draw_circle(event,x,y,flags,param):

    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,(x,y),2,(255,255,255),-1)

def DigitRecognition(img):
    predict = svm.CardPredictor()
    predict.train_svm()
    predict.predict_digit()


if __name__ == '__main__':
    # 创建一个黑色图像，并绑定窗口和鼠标回调函数
    img = np.zeros((100,100,3), np.uint8)
    cv2.namedWindow('image')
    # 设置鼠标事件回调
    cv2.setMouseCallback('image',draw_circle)

    while(True):
        cv2.imshow('image',img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

    cv2.imwrite("MousePaint03.png", img)
    DigitRecognition(img)