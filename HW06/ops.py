import numpy as np
from math import exp
def calculateMean(dataArray):
    #dataSize,dataDimension=dataArray.shape
    print(dataArray.shape)
    resultMean=np.mean(dataArray,axis=0)
    return resultMean

def kernelRBF(aPoint,bPoint,gamma):
    dataDimension=aPoint.shape[0]
    euclideanSquareDistance=0
    for dimensionIndex in range(dataDimension):
        euclideanSquareDistance+=(aPoint[dimensionIndex]-bPoint[dimensionIndex])**2
    resultDistance=exp(euclideanSquareDistance*(-gamma))
    return resultDistance