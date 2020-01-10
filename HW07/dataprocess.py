from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def readData(name,delimiter=','):
    #load csv file to numpy array
    print("Reading data {}".format(name))
    data=np.loadtxt(name,delimiter=delimiter)
    if data.ndim==2:
        return data[:,:]
    elif data.ndim==1:
        return data[:]

def loadFaces(totalImageCount=400):
    #load faces by running through directories
    #extra directories may cause failure
    faceFolderPath = Path('att_faces/')
    imageArray = np.zeros((totalImageCount, 112, 92))
    imageIndex = 0
    for subjectFolder in faceFolderPath.iterdir():
        if subjectFolder.is_dir():
            for imagePath in subjectFolder.iterdir():
                imageArray[imageIndex, :, :] = plt.imread(str(imagePath))
                imageIndex += 1
    return imageArray

if __name__=="__main__":
    #data=readData("X_train.csv")
    #print(data.shape)
    loadFaces()