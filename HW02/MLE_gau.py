from mnist_reader import load_mnist
import numpy as np
from math import exp,sqrt,pi,log,pow

def LogGaussVal(x,mean,var2):
    #what if var==0?
    var2+=1e-9
    return (-0.5)*(x-mean)*(x-mean)/var2-(log(2*pi*var2))*(0.5)
"""
def GaussVal(x,mean,var2):
    #what if var==0?
    if var2==0:
        var2=0.001
    return exp((-0.5)*(x-mean)*(x-mean)/var2)/sqrt(2*pi*var2)
"""
def Gauss_predict(img,mean_var2,pixel_num=28*28,data_size=10000):
    log_likelihood=np.zeros(10)
    table = np.load('table_new.npy')
    for label_predict in range(10):
        label_ans = np.sum(table[:, :, label_predict])/(28*28)
        #p_label=np.sum(table[:, :, label_predict])/data_size
        for pixel in range(pixel_num):
            log_likelihood[label_predict]+=LogGaussVal(img[pixel],mean_var2[label_predict,pixel,0],mean_var2[label_predict,pixel,1])
        log_likelihood[label_predict] +=log(label_ans/data_size)
    return log_likelihood,np.argmax(log_likelihood)

def NB_gauss(train_x,train_y,test_x,test_y,pixel_num=28*28):
    #label,pixel,(mean,var)

    mean_var2=np.zeros((10,pixel_num,2))
    listoflist=[[] for i in range(10*pixel_num)]
    for index in range(train_x.shape[0]):
        for pixel in range(pixel_num):
            listoflist[int(train_y[index,0]*pixel_num+pixel)].append(train_x[index,pixel])
    print("Finished storage.")
    for label in range(10):
        for pixel in range(pixel_num):
            #get mean
            list_len=len(listoflist[int(label * pixel_num + pixel)])
            mean_var2[label,pixel,0]=sum(listoflist[int(label*pixel_num+pixel)])/list_len
            mean_var2[label,pixel,1]=sum((x-mean_var2[label,pixel,0])**2 for x in listoflist[int(label*pixel_num+pixel)])/list_len
    print("Finished mean,var calculation.")
    correct=0
    for img_index in range(test_x.shape[0]):
        if img_index%1000==0:
            print("Predicting {}".format(img_index))
        p_lkh,p_l=Gauss_predict(test_x[img_index,:],mean_var2,data_size=train_x.shape[0])
        #print(p_lkh)
        if(p_l==test_y[img_index,0]):
            correct+=1
    print("Sample likelihood:{0}".format(p_lkh))
    print("Accuracy: {0} %".format(correct*100/10000))
"""
def Gauss_predict(img,mean_var2,pixel_num=28*28,data_size=0):
    log_likelihood=np.zeros(10)
    table = np.load('table_new.npy')
    for label_predict in range(10):
        label_ans = np.sum(table[:, :, label_predict])/(28*28)
        #p_label=np.sum(table[:, :, label_predict])/data_size
        for pixel in range(pixel_num):
            log_likelihood[label_predict]+=LogGaussVal(img[pixel],mean_var2[label_predict,0],mean_var2[label_predict,1])
        log_likelihood[label_predict] +=log(label_ans/data_size)
    return log_likelihood,np.argmax(log_likelihood)

def NB_gauss(train_x,train_y,test_x,test_y,pixel_num=28*28):
    #label,pixel,(mean,var)

    mean_var2=np.zeros((10,2))
    listoflist=[[] for i in range(10)]
    for index in range(train_x.shape[0]):
        for pixel in range(pixel_num):
            listoflist[int(train_y[index,0])].append(train_x[index,pixel])
    print("Finished storage.")
    for label in range(10):
        #get mean, this partOK
        list_len=len(listoflist[label])
        mean_var2[label,0]=sum(listoflist[int(label)])/list_len
        mean_var2[label,1]=sum((x-mean_var2[label,0])**2 for x in listoflist[int(label)])/list_len
    correct=0
    for img_index in range(test_x.shape[0]):
        if img_index%1000==0:
            print("Predicting {}".format(img_index))
        p_lkh,p_l=Gauss_predict(test_x[img_index,:],mean_var2,data_size=train_x.shape[0])
        #print(p_lkh)
        if(p_l==test_y[img_index,0]):
            correct+=1
    print("Sample likelihood:{0}".format(p_lkh))
    print("Accuracy: {0} %".format(correct*100/10000))
"""
if __name__ == "__main__":
    train_x,train_y,test_x,test_y=load_mnist()
    NB_gauss(train_x,train_y,test_x,test_y)
