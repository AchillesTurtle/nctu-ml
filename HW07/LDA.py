import numpy as np
from numpy import linalg as lng
import matplotlib.pyplot as plt
class LDA:
    def __init__(self,dataArray,classArray):
        self.dataArray=dataArray
        self.classArray = classArray
        self.realClasses,self.classDataCount=np.unique(self.classArray, return_counts=True)
        self.totalClassCount=self.realClasses.size
        self.classDict=dict(zip(self.realClasses,range(self.totalClassCount)))
        self.dataSize, self.dataDimension = self.dataArray.shape

    def project(self,projectDimension=2):
        self.projectDimension=projectDimension
        self.inClassScatter=np.zeros((self.dataDimension,self.dataDimension))
        self.betweenClassScatter=np.zeros((self.dataDimension, self.dataDimension))
        #calculate class means
        print("Calculating in class scatter...")
        inClassMean=np.zeros((self.dataDimension,self.totalClassCount))
        for dataIndex in range(self.dataSize):
            inClassMean[:,self.classDict[self.classArray[dataIndex]]]+=self.dataArray[dataIndex,:]
        for classIndex in range(self.totalClassCount):
            inClassMean[:,classIndex]/=self.classDataCount[classIndex]
        #calculate inclass scatter (sum over in class)
        for dataIndex in range(self.dataSize):
            diffToClassMean=(self.dataArray[dataIndex, :] - inClassMean[:, self.classDict[self.classArray[dataIndex]]])[:,np.newaxis]
            self.inClassScatter+=(diffToClassMean@np.transpose(diffToClassMean))

        #calculate between class scatter
        print("Calculate between class scatter..")
        allMean=np.average(inClassMean,axis=1,weights=self.classDataCount)[:,np.newaxis]
        for classIndex in range(self.totalClassCount):
            diffToTotalMean=inClassMean[:,classIndex]-allMean
            self.betweenClassScatter=self.classDataCount[classIndex]*(diffToTotalMean@np.transpose(diffToTotalMean))

        #calculate eigen vectors
        print("Calculate eigens...")
        inverseSwSb=lng.pinv(self.inClassScatter)@self.betweenClassScatter
        self.eigenValue, self.eigenVector = lng.eig(inverseSwSb)
        sortIndices = self.eigenValue.argsort()[::-1][:self.projectDimension]
        self.eigenValue = self.eigenValue[sortIndices]
        self.eigenVector = self.eigenVector[:, sortIndices]
        print(self.eigenVector)
        self.projectData = self.dataArray @ self.eigenVector

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dataX=self.projectData[:,0]
        dataY=self.projectData[:,1]
        scatter = ax.scatter(dataX, dataY, c=self.classArray, s=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(scatter)
        plt.show()

if __name__=="__main__":
    from dataprocess import readData
    trainData=readData("X_train.csv")
    dataLabel=readData("T_train.csv")
    ldaTest=LDA(trainData,dataLabel)
    ldaTest.project()
    ldaTest.plot()