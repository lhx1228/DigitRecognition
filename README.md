# 手写数字识别（opencv+svm）

## 项目简介
[项目地址](https://github.com/lhx1228/DigitRecognition)

运行项目后，使用鼠标在写入数字，然后输入字符'n'，可以看到底部会输出对应识别出的数字；当输入字符'q'时，结束运行。

<video id="video" controls="" preload="none" poster="http://om2bks7xs.bkt.clouddn.com/2017-08-26-Markdown-Advance-Video.jpg">
<source id="mp4" src="https://www.leihx.top/image/Presentation.mov" type="video/mp4">
</video>

## 实现过程

### 获取训练数据

该项目使用的训练数据集为[mnist](http://yann.lecun.com/exdb/mnist/)数据集(我这里只使用了training set images)：

![](https://www.leihx.top/image/DigitRecognition_1.png)

当然在我的[github](https://github.com/lhx1228/DigitRecognition)中，已经包含了数据集（train.zip），可以直接使用。

在这个项目中，我将mnist数据集中的数据转化为了图片然后再进行训练，转化方法参考 -> [python 读取 MNIST 数据集，并解析为图片文件](https://xinancsd.github.io/MachineLearning/mnist_parser.html)

### 训练SVM模型

```
import os
import cv2
import numpy as np
import img_recognition

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF) #Radial basis function (RBF), a good choice in most cases.
        self.model.setType(cv2.ml.SVM_C_SVC) #n-class classification

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
        self.model.save("svm.dat")

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        #print(r)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    #def __del__(self):
    #    self.save_traindata()

    def train_svm(self):
        self.model = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("train"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = int(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    chars_label.append(root_int)

            chars_train = list(map(img_recognition.deskew, chars_train)) #对图片进行抗扭斜处理
            chars_train = img_recognition.preprocess_hog(chars_train)    #获得hog特征
            chars_label = np.array(chars_label)
            self.model.train(chars_train, chars_label)
    def predict_digit(self):
        chars_train = []
        digit_img = cv2.imread('MousePaint03.png')
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        chars_train.append(digit_img)
        chars_train = list(map(img_recognition.deskew, chars_train))
        chars_train = img_recognition.preprocess_hog(chars_train)
        ch = self.model.predict(chars_train)
        print(int(ch))


#if __name__ == '__main__':
#    predict = CardPredictor()
#    predict.train_svm()
#    predict.predict_digit()

```

遍历训练数据，将其存入列表chars\_train中，并对其进行抗扭斜处理、获取hog特征等操作。
而将训练数据标签存入chars\_label中， 然后使用SVM对其进行模型训练。

### 实现绘画板

```
import cv2
import numpy as np
import svm

# 鼠标回调函数
# x, y 都是相对于窗口内的图像的位置

def draw_circle(event,x,y,flags,param):

    if flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,(x,y),3,(255,255,255),-1)

def DigitRecognition(img):
    predict = svm.CardPredictor()
    predict.train_svm()
    predict.predict_digit()


if __name__ == '__main__':
    # 创建一个黑色图像，并绑定窗口和鼠标回调函数
    img = np.zeros((150,150,3), np.uint8)
    cv2.namedWindow('image')
    # 设置鼠标事件回调
    cv2.setMouseCallback('image',draw_circle)

    while(True):
        cv2.imshow('image',img)
        if cv2.waitKey(1) == ord('q'):  #按q时结束运行
            break
        if cv2.waitKey(1) == ord('n'):  #按n时，识别并刷新，以便进行下次识别
            cv2.imwrite("MousePaint03.png", img)
            DigitRecognition(img)
            img = np.zeros((150, 150, 3), np.uint8)
    cv2.destroyAllWindows()
```

我这里将绘画板里写下的数字保存为图片"digit.png"，在进行识别时，需要对图片进行一些预处理，包括调整尺寸、灰度化、抗扭斜处理等：

```
def predict_digit(self):
    chars_train = []
    digit_img = cv2.imread('digit.png')
    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)  #灰度化
    digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA) #调整尺寸
    chars_train.append(digit_img)
    chars_train = list(map(img_recognition.deskew, chars_train)) #抗扭斜处理
    chars_train = img_recognition.preprocess_hog(chars_train)
    ch = self.model.predict(chars_train)
    print(int(ch))
```