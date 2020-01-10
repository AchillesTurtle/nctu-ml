from mnist_reader import load_mnist
import numpy as np
from pathlib import Path
from math import log

pixel_num = 28 * 28
train_x, train_y, test_x, test_y = load_mnist()
binning_fn='60000_bin.npy'
print(train_x.shape,train_y.shape)
bin_file = Path(binning_fn)
if bin_file.is_file():
    print("Found binning file {}".format(binning_fn))
    train_x=np.load(binning_fn)
else:
    print("Binning data not found, start binning")
    for iy,ix in np.ndindex(train_x.shape):
        train_x[iy,ix]=0 if train_x[iy,ix]<126 else 1
    print("Save bin file")
    np.save(binning_fn,train_x)
pn=np.zeros((train_x.shape[1],10))
ln=np.zeros((train_x.shape[1],10))

for image,pixel in np.ndindex(train_x.shape):
    ln[pixel,int(train_y[image,0])]+=1
    if train_x[image,pixel]==1:
        pn[pixel,int(train_y[image,0])]+=1
    if image%5000==0 and pixel==0:
        print("Iterating through {} image".format(image))
pn=pn/ln
#ln doesn't need to be this big
ln=ln/train_x.shape[0]
#for i in range(pixel):
#    print(pn[i,:])
acc=0
predict=np.zeros(train_x.shape[0])
con_mat=np.zeros((10,10))
for image in range(train_x.shape[0]):
    if image%5000==0:
        print("Predicting {} image".format(image))
    prob=np.zeros(10)
    for pixel in range(train_x.shape[1]):
        for c in range(10):
            if train_x[image,pixel]==1:
                prob[c]=prob[c]+log(max(ln[pixel,c],1e-9))+log(max(pn[pixel,c],1e-9))
            else:
                prob[c]=prob[c]+log(max(ln[pixel,c],1e-9))+log(max((1-pn[pixel,c]),1e-9))

    predict[image]=np.argmax(prob)
    #print(predict[image],train_y[image,0])
    con_mat[int(train_y[image, 0]), int(predict[image])] += 1
    if predict[image]==train_y[image,0]:
        acc+=1


for x in range(con_mat.shape[0]):
    print(con_mat[x,:])

for i in range(10):
    print("Sensitivity of class{}: {}".format(i,con_mat[i,i]/np.sum(con_mat[i,:])))
    print("Specificity of class{}: {}".format(i,(np.sum(con_mat[:,i])-con_mat[i,i])/(x.shape[0]-np.sum(con_mat[i,:])-np.sum(con_mat[:,i])+con_mat[i,i])))