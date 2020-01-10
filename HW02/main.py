from mnist_reader import load_mnist
import numpy as np
from math import log
import os.path
from MLE_gau import NB_gauss

class Binning(object):
    def __init__(self,borders=None):
        if borders is not None:
            self.borders=borders
            self.bins=borders.shape[0]
    def fit(self,X,bins=32):
        self.bins=bins
        self.borders=np.zeros(bins)
        data=np.sort(X.flatten())
        data_size=data.shape[0]
        for i in range(bins):
            self.borders[i]=data[int(data_size*i/bins)]
            #avoid extreme situations
            if i>0 and self.borders[i]<=self.borders[i-1]:
                for j in range(int(data_size*i/bins),data_size):
                    if data[j]>self.borders[i-1]:
                        self.borders[i]=data[j]
                        break
        print(self.borders)


    def value(self,X):
        for i in range(self.bins-1):
            if X>=self.borders[i] and X<self.borders[i+1]:
                return i
        return self.bins-1

class NB(object):
    def fit(self,table,bin_method):
        self.bin_method=bin_method
        self.pixel_num=28*28
        self.table=table
        self.label_ans=np.zeros((10,1))
        for labels in range(10):
            self.label_ans[labels]=np.sum(table[:, :, labels])/(28*28)
        self.data_size=np.sum(self.label_ans)

    def predict_img(self,img):
        log_likelihood=np.zeros(10)
        for labels in range(10):
            for pixel in range(self.pixel_num):
                log_likelihood[labels] += log(self.table[pixel, self.bin_method.value(img[pixel]), labels] + 1) - log(self.label_ans[labels] + 2)
            log_likelihood[labels] += log(self.label_ans[labels] / self.data_size)
        return log_likelihood,np.argmax(log_likelihood)

    def predict(self,test_x,test_y):
        correct=0
        for img_index in range(test_x.shape[0]):
            if img_index % 1000 == 0:
                print("Predicting : {}".format(img_index))
            p_lkh,p_l = self.predict_img(test_x[img_index, :])
            #print(p_lkh)
            if (p_l == test_y[img_index, 0]):
                correct += 1
        print("Sample likelihood:{0}".format(p_lkh))
        print("Accuracy: {0} %".format(correct * 100 / 10000))

if __name__ == "__main__":
    pixel_num=28*28
    train_x,train_y,test_x,test_y=load_mnist()

    mode=input("Mode choose:")
    if int(mode)==0:
        calculated_borders=np.array([0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,
           12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,
           24,   25,   26,   64,  143,  218,  252,  253])
        bin_to_32=Binning(calculated_borders)

        if os.path.exists('table_new.npy'):
            table=np.load('table_new.npy')
        else:
            table = np.zeros((pixel_num, 32, 10))
            for img_index in range(train_x.shape[0]):
                for pixel in range(pixel_num):
                    table[pixel, bin_to_32.value(train_x[img_index, pixel]), int(train_y[img_index, 0])] += 1
            np.save('table_new.npy', table)

        #bin_to_32.fit(train_x)
        NB_try=NB()
        NB_try.fit(table,bin_to_32)
        NB_try.predict(test_x,test_y)

    elif int(mode) ==1:
        NB_gauss(train_x,train_y,test_x,test_y)

