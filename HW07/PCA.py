import numpy as np
from numpy import linalg as lng
import matplotlib.pyplot as plt
class PCA:
    def __init__(self,dataArray,classArray=None):
        #flattens array to 2-D
        if dataArray.ndim>2:
            dataArray=dataArray.reshape(dataArray.shape[0],-1)
        self.dataArray=dataArray
        self.classArray = classArray
        self.dataSize, self.dataDimension = self.dataArray.shape
    def project(self,projectDimension=2):
        #project process here
        self.projectDimension=projectDimension

        print("Calculating covariance matrix...")
        print(self.dataArray.shape)
        covarArray=np.cov(self.dataArray,rowvar=False)
        #covarArray=np.corrcoef(self.dataArray,rowvar=False)

        print("Solving Eigen Values...")
        self.eigenValue, self.eigenVector = lng.eig(covarArray)
        #reverse argsort() indexes (was sorted in ascending)
        sortIndices=self.eigenValue.argsort()[::-1][:]
        self.eigenValue=self.eigenValue[sortIndices]
        self.eigenVector = self.eigenVector[:, sortIndices]

        print("Projecting data...")
        #select eigenvectors with largest eigen values
        self.projectEigenVector=self.eigenVector[:,:self.projectDimension]
        self.projectData=self.dataArray@self.projectEigenVector
        print("Finished projecting!")

    def plot(self):
        #plt figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #draw corresponding first two eigenvectors values..
        dataX=self.projectData[:,0]
        dataY=self.projectData[:,1]
        scatter = ax.scatter(dataX, dataY, c=self.classArray, s=5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #classArray can be empty (it's unsupervised after all)
        if self.classArray is not None:
            plt.colorbar(scatter)
        plt.show()
if __name__=="__main__":
    from dataprocess import readData
    trainData=readData("X_train.csv")
    #dataLabel=readData("T_train.csv")
    dataLabel=readData("Spectral_linearRBF_labels.csv").astype(int)
    pcaTest=PCA(trainData,dataLabel)
    pcaTest.project()
    pcaTest.plot()
