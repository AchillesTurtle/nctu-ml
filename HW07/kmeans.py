import numpy as np
from numpy import linalg as lng
from ops import calculateMean,kernelRBF
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os
from scipy.spatial.distance import cdist,pdist,squareform
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel
import random
from scipy.linalg import sqrtm

class SpectralCluster:
    def __init__(self,dataArray,clusterCount=3,gamma=10,mode='ratio',kernel='linear',name="SpectralClustering",saveEigenImage=False):
        self.mode=mode
        self.dataArray=dataArray
        self.clusterCount=clusterCount
        self.saveEigenImage=saveEigenImage
        self.gamma=gamma
        self.name=name
        #to be more comprehensive
        self.dataSize,self.dataDimension=self.dataArray.shape
        self.graphMatrix = np.zeros((self.dataSize, self.dataSize))
        self.degreeMatrix = np.zeros((self.dataSize,self.dataSize))
        self.graphLaplacian = np.zeros((self.dataSize,self.dataSize))
        print("Initializing Graph Matrix...")

        #Help with scipy to speed up the calculation
        if kernel=='linear':
            self.graphMatrix=linear_kernel(self.dataArray)
        elif kernel=='RBF':
            self.graphMatrix = rbf_kernel(self.dataArray, gamma=gamma)
            #self.graphMatrix=np.maximum(self.graphMatrix-(1e-4),0)
        elif kernel=='linearRBF':
            self.graphMatrix=linear_kernel(self.dataArray)+rbf_kernel(self.dataArray, gamma=gamma)
        """
            for dataIndex in range(self.dataSize):
                
                #show progess
                processPercentage = int(dataIndex * 100 / self.dataSize)
                if processPercentage % 10 == 0:
                    print("progress: {} %".format(processPercentage))
                #calculate RBF
                for dataIndex2 in range(dataIndex, self.dataSize):
                    self.graphMatrix[dataIndex,dataIndex2] = kernelRBF(self.dataArray[dataIndex], self.dataArray[dataIndex2],
                                                                       gamma=gamma)
                    self.graphMatrix[dataIndex2,dataIndex] = self.graphMatrix[dataIndex,dataIndex2]
            """
        print('Calculating degree and graph laplacian...')
        flatDegree=np.sum(self.graphMatrix,axis=1)
        #calculate degree D
        for dataIndex in range(self.dataSize):
            self.degreeMatrix[dataIndex,dataIndex]=flatDegree[dataIndex]
        #calculate L= D-W
        if mode=='ratio':
            self.graphLaplacian=self.degreeMatrix-self.graphMatrix
        elif mode=='normalized':
            self.graphLaplacian = self.degreeMatrix - self.graphMatrix
            degreeMatrixInvSqrt=sqrtm(lng.inv(self.degreeMatrix))
            self.graphLaplacian=degreeMatrixInvSqrt@self.graphLaplacian@degreeMatrixInvSqrt
        else:
            print("unknown mode")
            return
        print('Calculating eigenvalues and eigenvectors')
        if mode=='ratio':
            self.eigenValue, self.eigenVector = lng.eig(self.graphLaplacian,self.degreeMatrix)
        else:
            self.eigenValue,self.eigenVector=lng.eig(self.graphLaplacian)
        #get first few eigenvalue and eigenvectors
        sortIndices=self.eigenValue.argsort()
        self.eigenValue=self.eigenValue[sortIndices[:self.clusterCount]]
        for index in range(self.eigenValue.shape[0]):
            if self.eigenValue[index]<1e-10:
                self.eigenValue[index]=0
        self.eigenVector=self.eigenVector[:,sortIndices[:self.clusterCount]]
        #self.eigenSpace=np.transpose(self.eigenVector)\
        if mode=='normalized':
            for eigenRow in range(self.eigenVector.shape[0]):
                normTerm=(np.square(self.eigenVector[eigenRow,:]))**0.5
                self.eigenVector[eigenRow,:]=self.eigenVector[eigenRow,:]/normTerm
        print('Start Kmean cluster...')
        self.kmeansSolver=KMeansCluster(dataArray=self.eigenVector,clusterCount=self.clusterCount,gamma=self.gamma,kernelFunction='Euclidean',name=self.name,saveImages=self.saveEigenImage)
        self.kmeansSolver.clusterProcess()
        #plot result

        self.kmeansSolver.dataArray=self.dataArray
        self.kmeansSolver.plotData(-1,forceSaveImage=True)
    def saveClassArray(self):
        filename=self.name+"_labels.csv"
        np.savetxt(filename, self.kmeansSolver.classArray, delimiter=" ")
        return

class KMeansCluster:
    def __init__(self,dataArray,clusterCount=3,gamma=80,kernelFunction='Euclidean',name="KMmeansCluster",saveImages=True):
        #needed storage
        self.dataArray=dataArray
        self.clusterCount=clusterCount
        self.kernelFunction=kernelFunction#NEEDSMODIFICATION

        #to be more comprehensive
        self.dataSize,self.dataDimension=self.dataArray.shape
        print(self.dataArray.shape)
        #needed
        #self.mean=np.random.rand(clusterCount,self.dataDimension)
        self.mean=self.meanInit()
        self.classArray=np.random.randint(clusterCount+9,size=(self.dataSize,1))
        self.END_DIFFERENCE=1e-3

        #for animation
        self.name=name
        self.saveImages=saveImages
        #Initializing functions here
        #self.testClassInit()
        if kernelFunction=='RBF':
            self.gramMatrix=np.zeros((self.dataSize,self.dataSize))
            print("Initializing Gram Matrix...")
            for dataIndex in range(self.dataSize):
                #show process
                processPercentage=int(dataIndex*100/self.dataSize)
                if processPercentage%10==0:
                    print("progress: {} %".format(processPercentage))
                for dataIndex2 in range(dataIndex,self.dataSize):
                    self.gramMatrix[dataIndex][dataIndex2]=kernelRBF(self.dataArray[dataIndex],self.dataArray[dataIndex2],gamma=gamma)
                    self.gramMatrix[dataIndex2][dataIndex]=self.gramMatrix[dataIndex][dataIndex2]
    def meanInit(self):
        randomIndex=np.random.randint(self.dataSize,size=self.clusterCount)
        return self.dataArray[randomIndex,:]
    def testClassInit(self):
        for dataIndex in range(self.dataSize):
            if self.classArray[dataIndex]>1:
                print('modify')
                self.classArray[dataIndex]=0
            #if self.dataArray[dataIndex,1]>0.5:
            #    self.classArray[dataIndex]=0
            #elif self.dataArray[dataIndex,1]<-0.25:
            #    self.classArray[dataIndex]=1
            #else:
            #    self.classArray[dataIndex]=random.randint(0,1)
    def distance(self, aPointIndex, meanPoint,clusterIndex):
        if self.kernelFunction=='Euclidean':
            resultDistance=np.sum(np.absolute(self.dataArray[aPointIndex,:]-meanPoint))
        elif self.kernelFunction=='RBF':
            selfDistance=self.gramMatrix[aPointIndex,aPointIndex]
            neighborDistanceSum=0
            neighborGramMatrixSum=0
            dataIndices=np.where(self.classArray == clusterIndex)[0]
            clusterDataCount=dataIndices.shape[0]
            for dataIndex in dataIndices:
                neighborDistanceSum+=self.gramMatrix[aPointIndex,dataIndex]
                for dataIndex2 in dataIndices:
                    neighborGramMatrixSum += self.gramMatrix[dataIndex, dataIndex2]
            resultDistance=selfDistance-2*neighborDistanceSum/max(clusterDataCount,1e-6)+neighborGramMatrixSum/(max(clusterDataCount,1e-6)**2)
        #print(resultDistance)
        return resultDistance
    def eStep(self):
        for dataIndex in range(self.dataSize):
            minDistance=-1
            clusterResult=-1
            for clusterIndex in range(self.clusterCount):
                distanceToMean=self.distance(dataIndex,self.mean[clusterIndex,:],clusterIndex=clusterIndex)
                #if there is a closer class, choose it
                if minDistance==-1 or distanceToMean < minDistance:
                    minDistance=distanceToMean
                    clusterResult=clusterIndex
            self.classArray[dataIndex]=clusterResult
        return
    def mStep(self):
        #save oldmean to determine end conditions in isEndCondition()
        self.oldMean=np.copy(self.mean)
        for clusterIndex in range(self.clusterCount):
            dataClusterIndices=np.where(self.classArray==clusterIndex)[0]
            self.mean[clusterIndex,:]=calculateMean(self.dataArray[dataClusterIndices,:])
    def isEndCondition(self):
        if np.sum(np.absolute(self.oldMean-self.mean)) < self.END_DIFFERENCE:
            return 1
        return 0
    def clusterProcess(self):
        stepIndex=0
        while(1):
            self.plotData(stepIndex)
            stepIndex+=1
            self.eStep()
            self.plotData(stepIndex)
            self.mStep()
            if self.kernelFunction=='Euclidean':
                stepIndex+=1
            if(self.isEndCondition()):
                break
        self.plotData(stepIndex)
    def plotData(self,imageIndex,forceSaveImage=False):
        if forceSaveImage==False and self.saveImages == False:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111)
        dataX=self.dataArray[:,0]
        dataY=self.dataArray[:,1]
        scatter = ax.scatter(dataX, dataY, c=self.classArray, s=20)
        for clusterIndex in range(self.clusterCount):
            ax.scatter(self.mean[clusterIndex,0], self.mean[clusterIndex,1], s=50, c='red', marker='+')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(scatter)
        imageName=self.name+'_'+str(imageIndex)+'.png'
        folderPath=Path('images/'+self.name)
        print(imageName)
        if not os.path.exists(str(folderPath)):
            os.mkdir(str(folderPath))
        fig.savefig(str(folderPath / imageName))

if __name__=="__main__":
    from dataprocess import readData
    trainData=readData("X_train.csv")
    dataLabel=readData("T_train.csv")
    #dataCircle=readData("circle.txt")
    #dataMoon=readData("moon.txt")
    mnistKmeans=SpectralCluster(dataArray=trainData,gamma=40,clusterCount=5,mode='normalized',kernel='linearRBF',name='Spectral_linearRBF',saveEigenImage=True)
    mnistKmeans.saveClassArray()
    #test=KMeansCluster(dataArray=dataCircle,gamma=40,clusterCount=4,kernelFunction='RBF',name="RBF_circle")
    #test.clusterProcess()
    #test2=KMeansCluster(dataArray=dataMoon,gamma=40,clusterCount=4,kernelFunction='RBF',name="RBF_moon")
    #test2.clusterProcess()

    #spectraltest=SpectralCluster(dataArray=dataMoon,gamma=40,clusterCount=2,name="Spec_moon_init",saveEigenImage=True)
    #spectraltest=SpectralCluster(dataArray=dataMoon,gamma=20,clusterCount=2,name="Spec_moon",saveEigenImage=True)