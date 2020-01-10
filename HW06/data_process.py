import numpy as np
from ops import calculateMean
def loadData(filename):
    dataArray=np.loadtxt(filename,delimiter=',')
    return dataArray[:,:]

if __name__=="__main__":
    dataArray=loadData("circle.txt")
    print(dataArray)
    print(dataArray.shape)
    #print(calculateMean(dataArray))