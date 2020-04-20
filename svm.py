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
        # 识别英文字母和数字
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
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(img_recognition.deskew, chars_train))
            chars_train = img_recognition.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            #print(chars_train.shape)
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
